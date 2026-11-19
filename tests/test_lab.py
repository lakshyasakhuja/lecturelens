"""The lab, and the security policy of the pages that add a lecture in the browser."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="the web extra is not installed")

from fakes import FakeImageEmbedder, FakePictureOcr, FakeTextEmbedder, fast_accounts, make_index
from fastapi.testclient import TestClient

from slideseek.search import Searcher
from slideseek.slides import SlideConfig, detect_slides
from slideseek.web.server import (
    CONTENT_SECURITY_POLICY,
    SPEECH_CONTENT_SECURITY_POLICY,
    create_app,
)


def make_client(lab_dir=None):
    index = make_index()
    searcher = Searcher(
        index,
        text_embedder=FakeTextEmbedder(),
        image_embedder=FakeImageEmbedder(),
        ocr=FakePictureOcr(),
    )
    app = create_app(index, searcher, warm_up=False, accounts=fast_accounts(), lab_dir=lab_dir)
    return TestClient(app)


def test_there_is_no_lab_unless_it_is_asked_for():
    client = make_client()
    assert client.get("/lab").status_code == 404
    assert client.post("/api/lab/files/run/x.json", content=b"{}").status_code == 404
    assert client.post("/api/lab/detect?count=1", content=b"\0" * 14400).status_code == 404


def test_the_lab_saves_under_its_own_directory_only(tmp_path):
    client = make_client(tmp_path / "lab")
    page = client.get("/lab")
    assert page.status_code == 200 and "Index a recording in the browser" in page.text
    assert client.get("/lab/lab.js").status_code == 200

    saved = client.post("/api/lab/files/run/stills/000012000.jpg", content=b"\xff\xd8")
    assert saved.status_code == 200
    assert (tmp_path / "lab" / "run" / "stills" / "000012000.jpg").read_bytes() == b"\xff\xd8"
    for bad in ("../escape.txt", "run/../../x", "Run/X.json", "a/b/c/d.json", ".hidden"):
        assert client.post(f"/api/lab/files/{bad}", content=b"x").status_code in (400, 404), bad
    assert not (tmp_path / "escape.txt").exists()


def test_the_lab_finds_the_slides_as_the_indexer_does(tmp_path, lecture_frames):
    import synthetic

    frames, cuts = lecture_frames
    frames = synthetic.to_uint8(frames)  # the page sends bytes, as ffmpeg's frames are
    client = make_client(tmp_path)
    reply = client.post(f"/api/lab/detect?count={len(frames)}", content=frames.tobytes())
    assert reply.status_code == 200
    found = reply.json()
    expected = detect_slides(frames, SlideConfig())
    assert [s["t_start"] for s in found["segments"]] == [s.t_start for s in expected.segments]
    assert len(found["keyframe_times"]) >= len(cuts) + 1
    wrong = client.post("/api/lab/detect?count=3", content=np.zeros(5, np.uint8).tobytes())
    assert wrong.status_code == 400


def test_only_the_pages_that_add_a_lecture_may_compile_webassembly_and_reach_the_hub(tmp_path):
    client = make_client(tmp_path)
    assert client.get("/").headers["content-security-policy"] == CONTENT_SECURITY_POLICY
    assert "wasm-unsafe-eval" not in CONTENT_SECURITY_POLICY
    for path in ("/lab", "/capture/speech-worker.js", "/capture/video.js"):
        policy = client.get(path).headers["content-security-policy"]
        assert policy == SPEECH_CONTENT_SECURITY_POLICY, path
    assert "script-src 'self' 'wasm-unsafe-eval'" in SPEECH_CONTENT_SECURITY_POLICY
    assert "connect-src 'self' https://huggingface.co https://*.hf.co" in (
        SPEECH_CONTENT_SECURITY_POLICY
    )
    assert "media-src 'self' blob:" in SPEECH_CONTENT_SECURITY_POLICY
    assert "frame-ancestors" in SPEECH_CONTENT_SECURITY_POLICY


def test_the_capture_scripts_load_nothing_from_elsewhere():
    from slideseek.web.server import STATIC_DIR

    for path in (STATIC_DIR / "capture").glob("*.js"):
        script = path.read_text(encoding="utf-8")
        assert "https://" not in script, path.name
        assert "innerHTML" not in script and "eval(" not in script, path.name
    worker = (STATIC_DIR / "capture" / "speech-worker.js").read_text(encoding="utf-8")
    assert 'from "../vendor/browser/transformers.min.js"' in worker
