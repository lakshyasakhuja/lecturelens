from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make ``tests/synthetic.py`` importable as ``synthetic`` from every test module.
sys.path.insert(0, str(Path(__file__).parent))

import synthetic


def pytest_collection_modifyitems(config, items):
    """Skip ffmpeg tests when the binary is absent, and model tests unless asked for."""
    wants_models = "models" in (config.getoption("-m") or "")
    skip_ffmpeg = pytest.mark.skip(reason="ffmpeg is not installed")
    skip_models = pytest.mark.skip(reason="runs real models; use `pytest -m models`")
    for item in items:
        if "ffmpeg" in item.keywords and not synthetic.have_ffmpeg():
            item.add_marker(skip_ffmpeg)
        if "models" in item.keywords and not wants_models:
            item.add_marker(skip_models)


@pytest.fixture(scope="session")
def lecture_frames():
    """A four-slide lecture as detection frames, with its cut times: ``(frames, cuts)``."""
    durations = [20, 35, 12, 40]
    slides = [synthetic.slide_image(seed, picture=seed % 2 == 0) for seed in range(4)]
    frames = synthetic.timeline(
        [synthetic.hold(s, d) for s, d in zip(slides, durations, strict=True)]
    )
    cuts = [20, 55, 67]
    return frames, cuts


@pytest.fixture(scope="session")
def lecture_video(tmp_path_factory, lecture_frames):
    """The same lecture encoded as a real mp4 with an audio track."""
    frames, cuts = lecture_frames
    path = tmp_path_factory.mktemp("video") / "lec_test.mp4"
    synthetic.write_video(path, frames)
    return path, cuts, len(frames)
