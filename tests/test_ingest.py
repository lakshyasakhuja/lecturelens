"""ffmpeg round-trips: encode a synthetic lecture, then check what we decode from it."""

from __future__ import annotations

import wave

import numpy as np
import pytest
import synthetic as syn
from PIL import Image

from slideseek import ingest
from slideseek.slides import detect_slides

pytestmark = pytest.mark.ffmpeg


def test_probe_reports_duration_size_and_audio(lecture_video):
    path, _, seconds = lecture_video
    info = ingest.probe(path)
    assert info.duration_s == pytest.approx(seconds, abs=0.5)
    assert (info.width, info.height) == (640, 360)
    assert info.fps == pytest.approx(25, abs=0.1)
    assert info.has_audio


def test_probe_missing_file():
    with pytest.raises(FileNotFoundError):
        ingest.probe("does/not/exist.mp4")


def test_extract_audio_is_16khz_mono_pcm(lecture_video, tmp_path):
    path, _, seconds = lecture_video
    wav_path = ingest.extract_audio(path, tmp_path / "audio" / "lec.wav")
    with wave.open(str(wav_path), "rb") as wav:
        assert wav.getframerate() == 16_000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getnframes() / wav.getframerate() == pytest.approx(seconds, abs=0.5)
    assert not list(wav_path.parent.glob("*.partial.wav"))


def test_extract_audio_can_cut_an_excerpt(lecture_video, tmp_path):
    path, _, _ = lecture_video
    clip = ingest.extract_audio(path, tmp_path / "clip.wav", start=10, duration=7.5)
    with wave.open(str(clip), "rb") as wav:
        assert wav.getnframes() / wav.getframerate() == pytest.approx(7.5, abs=0.05)


def test_read_audio_returns_normalised_mono_samples(lecture_video):
    path, _, seconds = lecture_video
    samples = ingest.read_audio(path)
    assert samples.dtype == np.float32 and samples.ndim == 1
    assert len(samples) / 16_000 == pytest.approx(seconds, abs=0.5)
    assert 0.05 < np.abs(samples).max() <= 1.0  # the tone is audible and not clipped


def test_gray_frames_are_one_per_second_and_time_aligned(lecture_video, lecture_frames):
    path, _, seconds = lecture_video
    source, _ = lecture_frames
    frames = ingest.read_gray_frames(path)
    assert frames.dtype == np.uint8
    assert frames.shape[1:] == (syn.HEIGHT, syn.WIDTH)
    assert abs(len(frames) - seconds) <= 1

    # Frame i must show the video at second i: compare against the frames we encoded.
    n = min(len(frames), len(source))
    error = np.abs(frames[:n].astype(np.float32) / 255.0 - source[:n]).mean(axis=(1, 2))
    assert error.max() < 0.02


def test_detector_recovers_cuts_from_an_encoded_video(lecture_video):
    path, cuts, _ = lecture_video
    detection = detect_slides(ingest.read_gray_frames(path))
    assert [round(t) for t in detection.transitions] == cuts


def test_extract_stills_are_named_by_time_and_show_the_right_slide(
    lecture_video, lecture_frames, tmp_path
):
    path, _, seconds = lecture_video
    source, _ = lecture_frames
    times = [5.0, 30.0, 60.0, float(seconds)]  # the last one is past the final frame
    stills = ingest.extract_stills(
        path, times, tmp_path / "keys", max_width=320, duration_s=seconds
    )

    assert sorted(stills) == times
    assert stills[5.0].name == "000005000.jpg"
    for t, still in stills.items():
        with Image.open(still) as img:
            assert img.width == 320
            small = np.asarray(img.convert("L").resize((syn.WIDTH, syn.HEIGHT)), dtype=np.float32)
        expected = source[min(int(t), len(source) - 1)]
        assert np.abs(small / 255.0 - expected).mean() < 0.03

    # Idempotent: a second call reuses the files rather than re-extracting.
    mtimes = {t: p.stat().st_mtime_ns for t, p in stills.items()}
    again = ingest.extract_stills(path, times, tmp_path / "keys", max_width=320, duration_s=seconds)
    assert {t: p.stat().st_mtime_ns for t, p in again.items()} == mtimes


def test_ffmpeg_failure_raises_a_readable_error(tmp_path):
    bogus = tmp_path / "not_a_video.mp4"
    bogus.write_bytes(b"this is not a video")
    with pytest.raises(ingest.FfmpegError):
        ingest.read_gray_frames(bogus)
