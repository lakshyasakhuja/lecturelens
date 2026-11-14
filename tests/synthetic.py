"""Synthetic lecture footage with exact ground truth, for testing without any real video.

Two levels:

* ``slide_image`` / ``hold`` / ``timeline`` build detection-resolution greyscale frames
  directly as numpy arrays. Fast enough for every slide-detector unit test.
* ``write_video`` encodes frames to a real mp4 with ffmpeg, for tests that exercise decoding.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

WIDTH, HEIGHT = 160, 90
BACKGROUND = 0.96
TEXT_INK = 0.76  # body text after downscaling: about 20% ink coverage
TITLE_INK = 0.60  # titles are larger and bolder


def slide_image(seed: int, lines: int = 6, picture: bool = False) -> np.ndarray:
    """A plausible slide at detection resolution: light background, title bar, text lines.

    Returns ``float32`` in [0, 1] of shape ``(HEIGHT, WIDTH)``. Text is drawn as grey dashes
    of random length, which is what slide text looks like after area-averaged downscaling to
    160x90: glyphs cover roughly a fifth of their line box, so a line of black text on white
    becomes a light grey smear rather than a black bar. Getting this density right matters,
    because it sets how large a text-only slide change is relative to the thresholds.
    """
    rng = np.random.default_rng(seed)
    img = np.full((HEIGHT, WIDTH), BACKGROUND, dtype=np.float32)
    img[6:11, 10 : 10 + int(rng.integers(50, 120))] = TITLE_INK
    for row in range(lines):
        add_text_line(img, row, rng)
    if picture:
        ys, xs = np.mgrid[0:36, 0:56]
        phase = rng.uniform(0, np.pi)
        texture = 0.5 + 0.35 * np.sin(xs / rng.uniform(3, 9) + phase) * np.cos(
            ys / rng.uniform(3, 9)
        )
        img[46:82, 96:152] = texture.astype(np.float32)
    return img


def add_text_line(img: np.ndarray, row: int, rng: np.random.Generator) -> None:
    """Draw one line of dash 'words' on text row ``row`` (in place)."""
    y = 20 + row * 9
    x = 12
    limit = int(rng.integers(70, WIDTH - 12))
    while x < limit:
        word = int(rng.integers(4, 14))
        img[y : y + 3, x : min(x + word, limit)] = TEXT_INK
        x += word + 3


def to_uint8(frames: np.ndarray) -> np.ndarray:
    return np.clip(np.round(frames * 255.0), 0, 255).astype(np.uint8)


def hold(image: np.ndarray, seconds: int) -> np.ndarray:
    """``seconds`` identical frames (1 fps)."""
    return np.repeat(image[None], seconds, axis=0)


def timeline(parts: list[np.ndarray]) -> np.ndarray:
    """Concatenate frame stacks into one float video."""
    return np.concatenate(parts, axis=0).astype(np.float32)


def add_sensor_noise(frames: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    """Independent Gaussian noise per pixel per frame."""
    rng = np.random.default_rng(seed)
    return np.clip(frames + rng.normal(0, sigma, frames.shape).astype(np.float32), 0, 1)


def add_cursor(frames: np.ndarray, seed: int = 0) -> np.ndarray:
    """A small dark cursor that jumps to a new position every frame."""
    rng = np.random.default_rng(seed)
    out = frames.copy()
    for frame in out:
        y = int(rng.integers(0, HEIGHT - 2))
        x = int(rng.integers(0, WIDTH - 2))
        frame[y : y + 2, x : x + 2] = 0.0
    return out


def add_webcam(frames: np.ndarray, seed: int = 0) -> tuple[np.ndarray, tuple]:
    """A picture-in-picture webcam in the bottom-right corner that changes every frame.

    Returns the frames and the mask ``[x0, y0, x1, y1]`` that covers the overlay.
    """
    rng = np.random.default_rng(seed)
    out = frames.copy()
    y0, x0 = int(HEIGHT * 0.70), int(WIDTH * 0.78)
    out[:, y0:, x0:] = rng.uniform(0.1, 0.9, (len(out), HEIGHT - y0, WIDTH - x0)).astype(np.float32)
    return out, (0.78, 0.70, 1.0, 1.0)


def crossfade(a: np.ndarray, b: np.ndarray, steps: int) -> np.ndarray:
    """``steps`` intermediate frames blending from ``a`` to ``b`` (exclusive of both ends)."""
    alphas = np.linspace(0, 1, steps + 2)[1:-1]
    return np.stack([(1 - alpha) * a + alpha * b for alpha in alphas]).astype(np.float32)


def churn(seconds: int, seed: int = 0) -> np.ndarray:
    """Continuously changing content, like an embedded video: a new picture every frame."""
    return np.stack([slide_image(seed * 1000 + i, picture=True) for i in range(seconds)])


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def write_video(
    path: Path,
    frames: np.ndarray,
    size: tuple[int, int] = (640, 360),
    audio: bool = True,
) -> Path:
    """Encode 1-fps greyscale ``uint8`` frames as an H.264 mp4, upscaled, with a tone track.

    Each input frame is held for one second and the output runs at 25 fps, like a real
    screen recording. Nearest-neighbour upscaling keeps slide edges sharp.
    """
    frames = to_uint8(frames) if frames.dtype != np.uint8 else frames
    n, height, width = frames.shape
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{width}x{height}", "-r", "1",
        "-i", "pipe:0",
    ]  # fmt: skip
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=330:sample_rate=44100:duration={n}"]
    cmd += [
        "-vf", f"scale={size[0]}:{size[1]}:flags=neighbor,fps=25",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-g", "50",
    ]  # fmt: skip
    if audio:
        cmd += ["-c:a", "aac", "-b:a", "64k", "-shortest"]
    cmd.append(str(path))
    subprocess.run(cmd, input=frames.tobytes(), check=True, capture_output=True)
    return path
