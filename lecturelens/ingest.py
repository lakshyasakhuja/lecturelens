"""Getting pixels and audio out of a lecture video with ffmpeg.

Three products, each sized for its consumer:

* **Audio** as 16 kHz mono PCM, the input format Whisper was trained on.
* **Detection frames**: tiny greyscale frames at 1 fps, piped straight into a numpy array.
  Slide-change detection only needs coarse structure, and area-averaged downscaling removes
  most cursor movement and compression noise before the detector ever sees it. Nothing is
  written to disk, and an hour of video is about 50 MB in memory.
* **Keyframe stills**: one full-resolution frame per slide, extracted by timestamp after the
  detector has decided which frames matter. OCR and the image encoder see these, so slide
  text is read at the video's native resolution rather than from a thumbnail.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class FfmpegError(RuntimeError):
    """ffmpeg or ffprobe is missing, or exited with an error."""


@dataclass(frozen=True)
class VideoInfo:
    duration_s: float
    width: int
    height: int
    fps: float
    has_audio: bool


def _binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise FfmpegError(
            f"{name} was not found on PATH. Install ffmpeg (macOS: `brew install ffmpeg`, "
            "Debian/Ubuntu: `apt install ffmpeg`) and try again."
        )
    return path


def _run(cmd: Sequence[str], *, capture_stdout: bool = False) -> bytes:
    proc = subprocess.run(
        list(cmd),
        stdout=subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", errors="replace").strip().splitlines()[-5:]
        raise FfmpegError(f"{Path(cmd[0]).name} failed ({proc.returncode}): " + " | ".join(tail))
    return proc.stdout if capture_stdout else b""


def _parse_rate(rate: str) -> float:
    """Parse ffprobe's ``num/den`` frame-rate strings."""
    try:
        num, _, den = rate.partition("/")
        value = float(num) / float(den) if den else float(num)
    except (ValueError, ZeroDivisionError):
        return 0.0
    return value if value == value else 0.0  # NaN guard


def probe(video: str | Path) -> VideoInfo:
    """Read duration, frame size and frame rate from a video file."""
    video = Path(video)
    if not video.is_file():
        raise FileNotFoundError(f"video not found: {video}")
    out = _run(
        [
            _binary("ffprobe"),
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(video),
        ],
        capture_stdout=True,
    )  # fmt: skip
    meta = json.loads(out)
    streams = meta.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video_stream is None:
        raise FfmpegError(f"{video.name} has no video stream")
    duration = float(meta.get("format", {}).get("duration") or video_stream.get("duration") or 0)
    return VideoInfo(
        duration_s=duration,
        width=int(video_stream.get("width", 0)),
        height=int(video_stream.get("height", 0)),
        fps=_parse_rate(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate", "")),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
    )


def extract_audio(video: str | Path, out_wav: str | Path, sample_rate: int = 16_000) -> Path:
    """Write the audio track as mono 16-bit PCM at ``sample_rate`` Hz."""
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_wav.with_suffix(".partial.wav")
    _run(
        [
            _binary("ffmpeg"), "-y", "-v", "error",
            "-i", str(video),
            "-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le",
            str(tmp),
        ]
    )  # fmt: skip
    tmp.replace(out_wav)  # atomic: a crashed run never leaves a half-written file behind
    return out_wav


def read_gray_frames(
    video: str | Path, fps: float = 1.0, size: tuple[int, int] = (160, 90)
) -> np.ndarray:
    """Decode the video to small greyscale frames, sampled at ``fps``.

    Returns a ``uint8`` array of shape ``(n_frames, height, width)``. Frame ``i`` shows the
    video at time ``i / fps`` seconds. ``size`` is ``(width, height)``; the aspect ratio is
    not preserved, which is harmless for change detection and keeps masks (given as
    fractions of the frame) valid for any source resolution.
    """
    width, height = size
    raw = _run(
        [
            _binary("ffmpeg"), "-v", "error", "-nostdin",
            "-i", str(video),
            "-an", "-sn",
            "-vf", f"fps={fps:g},scale={width}:{height}:flags=area,format=gray",
            "-f", "rawvideo", "-pix_fmt", "gray",
            "pipe:1",
        ],
        capture_stdout=True,
    )  # fmt: skip
    frame_bytes = width * height
    n_frames = len(raw) // frame_bytes
    if n_frames == 0:
        raise FfmpegError(f"no frames could be decoded from {Path(video).name}")
    return np.frombuffer(raw, dtype=np.uint8, count=n_frames * frame_bytes).reshape(
        n_frames, height, width
    )


def extract_still(
    video: str | Path,
    t: float,
    out_path: str | Path,
    max_width: int = 1920,
    quality: int = 2,
) -> Path:
    """Save the frame shown at time ``t`` (seconds) as a JPEG, at most ``max_width`` wide.

    ``-ss`` before ``-i`` seeks to the preceding keyframe and decodes forward, so the result
    is frame-accurate without decoding the whole file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            _binary("ffmpeg"), "-y", "-v", "error", "-nostdin",
            "-ss", f"{max(0.0, t):.3f}",
            "-i", str(video),
            "-frames:v", "1",
            "-vf", f"scale='min({max_width},iw)':-2",
            "-q:v", str(quality),
            str(out_path),
        ]
    )  # fmt: skip
    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise FfmpegError(f"no frame at t={t:.1f}s in {Path(video).name}")
    return out_path


def extract_stills(
    video: str | Path,
    times: Sequence[float],
    out_dir: str | Path,
    max_width: int = 1920,
    duration_s: float | None = None,
) -> dict[float, Path]:
    """Extract one still per timestamp, skipping any that already exist.

    Files are named by the timestamp in milliseconds, so a keyframe time maps to exactly one
    file and re-running the pipeline is idempotent. Returns ``{time: path}``.
    """
    out_dir = Path(out_dir)
    stills: dict[float, Path] = {}
    for t in sorted(set(times)):
        path = out_dir / f"{round(t * 1000):09d}.jpg"
        if not path.is_file():
            # A timestamp at or past the last frame yields nothing; step back just inside it.
            seek = t if duration_s is None else min(t, max(0.0, duration_s - 0.25))
            extract_still(video, seek, path, max_width=max_width)
        stills[t] = path
    return stills
