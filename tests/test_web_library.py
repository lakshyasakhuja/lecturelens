"""Adding your own lectures over HTTP, and searching them with the public ones."""

from __future__ import annotations

import gzip
import io

import pytest

pytest.importorskip("fastapi", reason="the web extra is not installed")

import synthetic
from fakes import (
    ADA,
    FakeImageEmbedder,
    FakeIndexImageEmbedder,
    FakeIndexTextEmbedder,
    FakeOcr,
    FakePictureOcr,
    FakeTextEmbedder,
    fast_accounts,
    make_index,
    make_picture,
)
from fastapi.testclient import TestClient

from slideseek import library as lib
from slideseek.library import Library, Models
from slideseek.search import Searcher
from slideseek.web.server import create_app

GRACE = {**ADA, "username": "grace", "email": "grace@example.com", "name": "Grace Hopper"}
FILE = {"name": "lecture 4.mp4", "size": 147_000_000, "type": "video/mp4"}
WORDS = {
    "model": "whisper in a browser",
    "segments": [
        {"words": [{"w": "kernels", "s": 30.0, "e": 30.6}, {"w": "smooth", "s": 31, "e": 31.4}]}
    ],
}


def jpeg() -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 36), "white").save(buffer, "JPEG")
    return buffer.getvalue()


@pytest.fixture()
def site(tmp_path):
    index = make_index()
    searcher = Searcher(
        index,
        text_embedder=FakeTextEmbedder(),
        image_embedder=FakeImageEmbedder(),
        ocr=FakePictureOcr(),
    )
    library = Library(
        tmp_path / "libraries",
        lambda: Models(FakeIndexTextEmbedder(), FakeIndexImageEmbedder(), FakeOcr()),
        worker=False,
    )
    accounts = fast_accounts()
    app = create_app(index, searcher, warm_up=False, accounts=accounts, library=library)
    return app, library, accounts


def visitor(app, form=None):
    client = TestClient(app)
    if form:
        assert client.post("/api/account/signup", json=form).status_code == 201
    return client


@pytest.fixture()
def frames(lecture_frames):
    stack, _ = lecture_frames
    return gzip.compress(synthetic.to_uint8(stack).tobytes())


def add_lecture(client, library, frames, title="Gaussian processes"):
    made = client.post(
        "/api/library/lectures", json={"title": title, "duration_s": 107.0, "file": FILE}
    )
    assert made.status_code == 201, made.text
    lecture_id = made.json()["lecture"]["lecture_id"]
    sent = client.put(
        f"/api/library/lectures/{lecture_id}/frames",
        content=frames,
        headers={"content-type": "application/gzip"},
    )
    assert sent.status_code == 200, sent.text
    for t in sent.json()["lecture"]["keyframe_times"]:
        still = client.put(
            f"/api/library/lectures/{lecture_id}/stills/{round(t * 1000)}.jpg",
            content=jpeg(),
            headers={"content-type": "image/jpeg"},
        )
        assert still.status_code == 200, still.text
    words = client.put(f"/api/library/lectures/{lecture_id}/transcript", json=WORDS)
    assert words.status_code == 200, words.text
    library.run_pending()
    return lecture_id


def search(client, q, **params):
    reply = client.get("/api/search", params={"q": q, "k": 10, **params})
    assert reply.status_code == 200, reply.text
    return reply.json()["results"]


def test_your_own_lecture_is_searched_with_the_public_ones(site, frames):
    app, library, _ = site
    ada = visitor(app, ADA)
    lecture_id = add_lecture(ada, library, frames)

    listing = ada.get("/api/library")
    assert listing.headers["cache-control"] == "no-store"
    (mine,) = listing.json()["lectures"]
    assert mine["lecture_id"] == lecture_id and mine["stage"] == "ready"
    assert listing.json()["limits"]["lectures"] == lib.MAX_LECTURES

    hit = next(r for r in search(ada, "kernels smooth") if r["lecture_id"] == lecture_id)
    assert hit["own"] is True and hit["lecture_title"] == "Gaussian processes"
    assert hit["player"] == {"kind": "own", "lecture_id": lecture_id, "file": {"name": FILE["name"], "size": FILE["size"]}, "start": 20}  # fmt: skip
    assert hit["video_link"] == ""
    thumb = ada.get(hit["thumb"])
    assert hit["thumb"].startswith(f"/api/library/thumbs/{lecture_id}/")
    assert thumb.status_code == 200 and thumb.headers["cache-control"].startswith("private")

    # The scope switch: yours only, the public ones only, or both.
    assert {r["lecture_id"] for r in search(ada, "kernels dropout", scope="mine")} == {lecture_id}
    assert lecture_id not in {r["lecture_id"] for r in search(ada, "kernels", scope="public")}
    both = {r["lecture_id"] for r in search(ada, "kernels dropout")}
    assert {lecture_id, "lec01"} <= both
    assert search(ada, "kernels", scope="mine", lecture=["lec01"]) == []
    assert ada.get("/api/search", params={"q": "x", "scope": "theirs"}).status_code == 400

    # A picture is searched the same way.
    picture = make_picture("anything", words="Gaussian Processes slide")
    found = ada.post(
        "/api/search/picture",
        params={"scope": "mine"},
        content=picture,
        headers={"content-type": "image/png"},
    )
    assert found.status_code == 200, found.text
    assert {r["lecture_id"] for r in found.json()["results"]} == {lecture_id}


def test_nobody_else_sees_your_lectures(site, frames):
    app, library, _ = site
    ada, grace = visitor(app, ADA), visitor(app, GRACE)
    lecture_id = add_lecture(ada, library, frames)
    thumb = next(r["thumb"] for r in search(ada, "kernels") if r["own"])

    assert grace.get("/api/library").json()["lectures"] == []
    assert lecture_id not in {r["lecture_id"] for r in search(grace, "kernels smooth")}
    assert grace.get("/api/search", params={"q": "kernels", "lecture": lecture_id}).status_code == 400  # fmt: skip
    assert search(grace, "kernels", scope="mine") == []
    for method, path, body in [
        ("GET", f"/api/library/lectures/{lecture_id}", None),
        ("PATCH", f"/api/library/lectures/{lecture_id}", {"title": "Mine now"}),
        ("DELETE", f"/api/library/lectures/{lecture_id}", None),
        ("PUT", f"/api/library/lectures/{lecture_id}/transcript", WORDS),
        ("GET", thumb, None),
    ]:
        assert grace.request(method, path, json=body).status_code == 404, (method, path)
    assert ada.get(f"/api/library/lectures/{lecture_id}").status_code == 200


def test_every_library_route_needs_somebody_signed_in(site):
    app, _, _ = site
    stranger = visitor(app)
    lecture = "/api/library/lectures/own-0123456789ab"
    for method, path in [
        ("GET", "/api/library"),
        ("POST", "/api/library/lectures"),
        ("GET", lecture),
        ("PUT", f"{lecture}/frames"),
        ("PUT", f"{lecture}/stills/000001000.jpg"),
        ("PUT", f"{lecture}/transcript"),
        ("PATCH", lecture),
        ("DELETE", lecture),
        ("GET", "/api/library/thumbs/own-0123456789ab/000001000.jpg"),
    ]:
        assert stranger.request(method, path).status_code == 401, (method, path)


def test_other_sites_cannot_add_change_or_delete(site, frames):
    app, library, _ = site
    ada = visitor(app, ADA)
    lecture_id = add_lecture(ada, library, frames)
    evil = {"origin": "https://evil.example", "sec-fetch-site": "cross-site"}
    for method, path, body in [
        ("POST", "/api/library/lectures", {"title": "x", "duration_s": 100, "file": FILE}),
        ("PATCH", f"/api/library/lectures/{lecture_id}", {"title": "Hijacked"}),
        ("DELETE", f"/api/library/lectures/{lecture_id}", None),
    ]:
        reply = ada.request(method, path, json=body, headers=evil)
        assert reply.status_code == 403, (method, path)
    assert ada.get("/api/library").json()["lectures"][0]["title"] == "Gaussian processes"


def test_what_is_sent_is_checked(site, frames, monkeypatch):
    app, _, _ = site
    ada = visitor(app, ADA)
    bad = ada.post("/api/library/lectures", json={"title": "", "duration_s": 100, "file": FILE})
    assert bad.status_code == 400 and "title" in bad.json()["detail"]
    long = ada.post("/api/library/lectures", json={"title": "x", "duration_s": 9e4, "file": FILE})
    assert long.status_code == 413

    made = ada.post("/api/library/lectures", json={"title": "A", "duration_s": 107, "file": FILE})
    lecture_id = made.json()["lecture"]["lecture_id"]
    base = f"/api/library/lectures/{lecture_id}"
    assert ada.post("/api/library/lectures", json={"title": "B", "duration_s": 107, "file": FILE}).status_code == 409  # fmt: skip
    as_text = ada.put(f"{base}/frames", content=frames, headers={"content-type": "text/plain"})
    assert as_text.status_code == 415
    monkeypatch.setattr(lib, "MAX_FRAMES_UPLOAD_BYTES", 100)
    too_big = ada.put(f"{base}/frames", content=frames, headers={"content-type": "application/gzip"})  # fmt: skip
    assert too_big.status_code == 413
    monkeypatch.undo()
    sent = ada.put(f"{base}/frames", content=frames, headers={"content-type": "application/gzip"})
    t = sent.json()["lecture"]["keyframe_times"][0]
    still = f"{base}/stills/{round(t * 1000)}.jpg"
    assert ada.put(still, content=jpeg(), headers={"content-type": "image/png"}).status_code == 415
    assert ada.put(f"{base}/stills/12.png", content=jpeg(), headers={"content-type": "image/jpeg"}).status_code == 404  # fmt: skip
    assert ada.put(f"{base}/stills/1.jpg", content=jpeg(), headers={"content-type": "image/jpeg"}).status_code == 404  # fmt: skip
    garbled = ada.put(f"{base}/transcript", content=b"{", headers={"content-type": "application/json"})  # fmt: skip
    assert garbled.status_code == 400

    # A page that was reloaded is told what has arrived, so it sends only what is missing.
    ada.put(still, content=jpeg(), headers={"content-type": "image/jpeg"})
    progress = ada.get(base).json()["lecture"]
    assert progress["received"] == [round(t * 1000)] and progress["transcript"] is False
    assert progress["stage"] == "stills"


def test_renaming_and_deleting_your_lecture(site, frames):
    app, library, _ = site
    ada = visitor(app, ADA)
    lecture_id = add_lecture(ada, library, frames)
    renamed = ada.patch(f"/api/library/lectures/{lecture_id}", json={"title": "Kernels, week 4"})
    assert renamed.status_code == 200
    assert next(r for r in search(ada, "kernels") if r["own"])["lecture_title"] == "Kernels, week 4"
    assert ada.delete(f"/api/library/lectures/{lecture_id}").status_code == 204
    assert not any(r["own"] for r in search(ada, "kernels smooth"))
    assert ada.get("/api/library").json()["lectures"] == []


def test_deleting_your_account_deletes_your_lectures(site, frames, tmp_path):
    app, library, _ = site
    ada = visitor(app, ADA)
    add_lecture(ada, library, frames)
    assert any((tmp_path / "libraries").iterdir())
    reply = ada.request("DELETE", "/api/account", json={"password": ADA["password"]})
    assert reply.status_code in (200, 204), reply.text
    assert not any((tmp_path / "libraries").iterdir())


def test_an_admin_sees_how_much_each_account_keeps_but_not_what(site, frames):
    app, library, accounts = site
    ada = visitor(app, ADA)  # the first account is the default admin's
    accounts.set_default_admin("ada")
    add_lecture(ada, library, frames)
    visitor(app, GRACE)
    listed = {a["username"]: a for a in ada.get("/api/admin/accounts").json()["accounts"]}
    assert listed["ada"]["library"]["lectures"] == 1 and listed["ada"]["library"]["bytes"] > 0
    assert listed["grace"]["library"] == {"lectures": 0, "bytes": 0}
    assert "Gaussian" not in str(listed)


def test_without_a_library_there_are_no_library_routes(tmp_path):
    index = make_index()
    searcher = Searcher(index, text_embedder=FakeTextEmbedder(), image_embedder=FakeImageEmbedder())
    app = create_app(index, searcher, warm_up=False, accounts=fast_accounts())
    ada = visitor(app, ADA)
    assert ada.get("/api/library").status_code == 404
    assert search(ada, "dropout", scope="mine") == []
    assert search(ada, "dropout", scope="public")


def test_a_library_made_from_a_directory_shares_the_searchers_models(tmp_path):
    index = make_index()
    searcher = Searcher(index, text_embedder=FakeTextEmbedder(), image_embedder=FakeImageEmbedder())
    app = create_app(
        index, searcher, warm_up=False, accounts=fast_accounts(), library=tmp_path / "mine"
    )
    library = app.state.library
    try:
        assert isinstance(library, Library) and library.root == tmp_path / "mine"
        assert library.config.embed.text_model == searcher.embed_config.text_model
        assert library.config.embed.image_model == searcher.embed_config.image_model
        models = library.models
        assert models.text_embedder is searcher.text_embedder
        assert models.image_embedder is searcher.image_embedder
    finally:
        library.close()


def test_the_page_is_told_whether_it_may_offer_adding_a_lecture(site, tmp_path):
    app, _, _ = site
    assert visitor(app).get("/api/info").json()["own_lectures"] is True
    index = make_index()
    searcher = Searcher(index, text_embedder=FakeTextEmbedder(), image_embedder=FakeImageEmbedder())
    plain = create_app(index, searcher, warm_up=False, accounts=fast_accounts())
    assert visitor(plain).get("/api/info").json()["own_lectures"] is False


def test_the_page_may_play_a_file_from_this_computer_but_compiles_no_webassembly(site):
    from slideseek.web.server import CONTENT_SECURITY_POLICY

    app, _, _ = site
    page = visitor(app).get("/")
    assert "media-src 'self' blob:" in page.headers["content-security-policy"]
    assert "wasm-unsafe-eval" not in CONTENT_SECURITY_POLICY
    # The page and its scripts change together: a browser checks each is current.
    for path in ("/", "/app.js", "/styles.css", "/capture/adder.js", "/capture/speech-worker.js"):
        assert visitor(app).get(path).headers["cache-control"] == "no-cache", path


def test_the_page_offers_your_lectures_with_everything_its_script_needs():
    from slideseek.web.server import STATIC_DIR

    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    for hook in (
        'id="yours"',
        'id="adder"',
        'id="scope"',
        'data-scope="mine"',
        'id="recording-input"',
    ):
        assert hook in html, hook
    adder = (STATIC_DIR / "capture" / "adder.js").read_text(encoding="utf-8")
    worker = (STATIC_DIR / "capture" / "speech-worker.js").read_text(encoding="utf-8")
    # The pinned model, kept in the site's private files rather than the Cache API.
    assert "b3f77bf9a8c4d5ea3415827033d1ffea7955fd9a" in adder
    assert "env.useCustomCache = true" in worker and "getDirectory" in worker
