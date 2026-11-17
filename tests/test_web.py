"""The web API, on the hand-built index with stand-in encoders."""

from __future__ import annotations

import dataclasses
import json

import pytest

pytest.importorskip("fastapi", reason="the web extra is not installed")

from fakes import (
    ADA,
    FakeImageEmbedder,
    FakePictureOcr,
    FakeTextEmbedder,
    fast_accounts,
    make_index,
    make_picture,
)
from fastapi.testclient import TestClient

from slideseek.catalog import Lecture
from slideseek.search import Searcher
from slideseek.web.server import (
    MAX_QUERY_CHARS,
    create_app,
    load_benchmark,
    load_examples,
    youtube_id,
)

EXAMPLES = [{"query": "chain rule", "type": "keyword"}]


def sign_up(client, **changes):
    """Sign ``client`` up, and so in: its cookie jar keeps the session from then on."""
    reply = client.post("/api/account/signup", json={**ADA, **changes})
    assert reply.status_code == 201, reply.text
    return client


def make_client(tmp_path, *, with_thumbs=False, videos=False, benchmark=None, signed_in=True):
    index = make_index()
    for lecture_id, lecture in list(index.lectures.items()):
        index.lectures[lecture_id] = dataclasses.replace(
            lecture,
            course="Intro to ML",
            instructors="Ada Example",
            licence="CC BY-NC-SA 4.0",
            source_url=f"https://ocw.example.edu/{lecture_id}",
            video_link_template="https://www.youtube.com/watch?v=abcDEF12345&t={seconds}s",
        )
    index.moments["video_link"] = [
        index.lectures[row.lecture_id].link_at(row.t_start) for row in index.moments.itertuples()
    ]
    if with_thumbs:
        index.moments["thumb"] = [
            f"thumbs/{row.lecture_id}/{n:09d}.jpg"
            for n, row in enumerate(index.moments.itertuples())
        ]
        index.save(tmp_path / "index")
        for relative in index.moments["thumb"]:
            path = tmp_path / "index" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\xff\xd8\xff\xd9")  # a minimal JPEG
    videos_dir = None
    if videos:
        videos_dir = tmp_path / "videos"
        videos_dir.mkdir()
        (videos_dir / "lec02.mp4").write_bytes(b"not really a video")
        (videos_dir / "unrelated.mp4").write_bytes(b"x")
    searcher = Searcher(
        index,
        text_embedder=FakeTextEmbedder(),
        image_embedder=FakeImageEmbedder(),
        ocr=FakePictureOcr(),
    )
    app = create_app(
        index,
        searcher,
        videos_dir=videos_dir,
        examples=EXAMPLES,
        benchmark=benchmark,
        warm_up=False,
        accounts=fast_accounts(),
    )
    client = TestClient(app)
    return sign_up(client) if signed_in else client


@pytest.fixture()
def client(tmp_path):
    return make_client(tmp_path)


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["ready"] is True


def test_info_describes_the_library(client):
    body = client.get("/api/info").json()
    assert (body["n_lectures"], body["n_moments"], body["n_keyframes"]) == (2, 8, 7)
    assert body["examples"] == EXAMPLES
    assert body["benchmark"] is None
    first = body["lectures"][0]
    assert first == {
        "lecture_id": "lec01",
        "title": "Generalisation",
        "course": "Intro to ML",
        "instructors": "Ada Example",
        "licence": "CC BY-NC-SA 4.0",
        "source_url": "https://ocw.example.edu/lec01",
        "duration_s": 315.0,
        "n_moments": 5,
        "n_slides": 4,
    }
    assert body["total_duration_s"] == 315.0 + 260.0


def test_search_returns_explained_results(client):
    body = client.get("/api/search", params={"q": "  chain   rule "}).json()
    assert body["query"] == "chain rule"  # whitespace normalised
    assert body["count"] == len(body["results"]) >= 1
    assert body["took_ms"] >= 0
    assert body["warm"] is True  # the models were loaded before this query arrived
    top = body["results"][0]
    assert top["moment_id"] == "lec02:0006"
    assert top["lecture_title"] == "Neural networks"
    assert (top["timestamp"], top["end_timestamp"]) == ("1:20", "2:50")
    assert top["video_link"] == "https://www.youtube.com/watch?v=abcDEF12345&t=80s"
    assert top["player"] == {"kind": "youtube", "video_id": "abcDEF12345", "start": 80}
    assert "said" in top["badges"]
    assert top["signals"]["bm25"] == 1
    assert [top["snippet"][a:b] for a, b in top["highlights"]] == ["chain", "rule"]
    assert top["thumb"] is None  # this index has no thumbnails on disk
    json.dumps(body)  # everything is plain JSON


def test_search_limits_and_filters(client):
    assert client.get("/api/search", params={"q": "training", "k": 2}).json()["count"] <= 2
    filtered = client.get(
        "/api/search", params={"q": "the model", "k": 20, "lecture": "lec02"}
    ).json()
    assert filtered["results"] and all(r["lecture_id"] == "lec02" for r in filtered["results"])
    both = client.get(
        "/api/search", params=[("q", "the model"), ("lecture", "lec01"), ("lecture", "lec02")]
    )
    assert both.status_code == 200


@pytest.mark.parametrize(
    "params, status, message",
    [
        ({"q": "   "}, 400, "empty"),
        ({"q": "x" * (MAX_QUERY_CHARS + 1)}, 400, "longer than"),
        ({"q": "ok", "lecture": "lec99"}, 400, "Unknown lecture"),
        ({"q": "ok", "k": 0}, 422, None),
        ({"q": "ok", "k": 500}, 422, None),
        ({}, 422, None),
    ],
)
def test_search_rejects_bad_requests(client, params, status, message):
    response = client.get("/api/search", params=params)
    assert response.status_code == status
    if message:
        assert message in response.json()["detail"]


def test_no_match_is_an_empty_result_not_an_error(tmp_path):
    client = make_client(tmp_path)
    body = client.get("/api/search", params={"q": "zzzz qqqq", "lecture": "lec01"}).json()
    assert isinstance(body["results"], list)


def test_frontend_and_security_headers(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "Slideseek" in page.text
    policy = page.headers["content-security-policy"]
    assert "default-src 'self'" in policy
    assert "frame-src https://www.youtube-nocookie.com" in policy
    assert page.headers["x-content-type-options"] == "nosniff"
    assert (
        client.get("/api/search", params={"q": "x"}).headers["x-content-type-options"] == "nosniff"
    )


def test_unknown_paths(client):
    api = client.get("/api/nope")
    assert api.status_code == 404 and api.json() == {"detail": "Not found."}
    page = client.get("/some/page")
    assert page.status_code == 404 and "Slideseek" in page.text


def test_thumbnails_are_served_from_the_index_and_nothing_else(tmp_path):
    client = make_client(tmp_path, with_thumbs=True)
    top = client.get("/api/search", params={"q": "chain rule"}).json()["results"][0]
    assert top["thumb"] == "/thumbs/lec02/000000006.jpg"
    image = client.get(top["thumb"])
    assert image.status_code == 200
    assert image.headers["cache-control"] == "public, max-age=86400"
    # The index directory also holds the manifest and the vectors; those must not be reachable.
    assert client.get("/thumbs/../manifest.json").status_code == 404
    assert client.get("/thumbs/%2e%2e/manifest.json").status_code == 404
    assert client.get("/manifest.json").status_code == 404


def test_local_recordings_are_played_in_place(tmp_path):
    client = make_client(tmp_path, videos=True)
    results = client.get("/api/search", params={"q": "chain rule"}).json()["results"]
    assert results[0]["player"] == {"kind": "file", "src": "/videos/lec02.mp4", "start": 80}
    assert client.get("/videos/lec02.mp4").status_code == 200
    # A lecture without a local file still links out.
    other = client.get("/api/search", params={"q": "dropout", "lecture": "lec01"}).json()[
        "results"
    ][0]
    assert other["player"]["kind"] == "youtube"


def test_benchmark_is_shown_only_when_it_was_measured(tmp_path):
    assert load_benchmark(tmp_path / "missing.json") is None
    metrics = {
        "split": "test",
        "n_queries": 40,
        "tolerance_s": 15.0,
        "latency": {"p50_ms": 98.0, "p95_ms": 120.0, "device": "cpu", "n": 120},
        "systems": [
            {"number": 1, "label": "BM25, transcript only", "overall": {"r1": 0.4, "r5": 0.6, "mrr10": 0.5},
             "by_type": {"keyword": {"r5": 0.9}, "visual": {"r5": 0.1}}},
            {"number": None, "label": "an extra", "overall": {"r1": 0.5, "r5": 0.7, "mrr10": 0.6}, "by_type": {}},
        ],
    }  # fmt: skip
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(metrics))
    benchmark = load_benchmark(path)
    assert benchmark["n_queries"] == 40
    assert [s["label"] for s in benchmark["systems"]] == [
        "BM25, transcript only"
    ]  # table rows only
    assert benchmark["systems"][0]["by_type"] == {"keyword": 0.9, "visual": 0.1}

    client = make_client(tmp_path, benchmark=benchmark)
    assert client.get("/api/info").json()["benchmark"]["systems"][0]["r5"] == 0.6


def test_examples_file(tmp_path):
    path = tmp_path / "examples.json"
    path.write_text(
        json.dumps([{"query": " attention ", "type": "keyword"}, {"query": ""}, {"type": "x"}])
    )
    assert load_examples(path) == [{"query": "attention", "type": "keyword"}]
    assert load_examples(tmp_path / "none.json") == []
    assert load_examples(None) == []


def test_examples_for_lectures_that_are_not_in_the_index_are_dropped(tmp_path):
    path = tmp_path / "examples.json"
    path.write_text(
        json.dumps(
            [
                {"query": "in this index", "type": "keyword", "lecture_id": "lec01"},
                {
                    "query": "from another course",
                    "type": "keyword",
                    "lecture_id": "mit6-7960-lec04",
                },
                {"query": "no lecture named", "type": "visual"},
            ]
        )
    )
    lectures = make_index().lectures
    assert [e["query"] for e in load_examples(path, lectures)] == [
        "in this index",
        "no lecture named",
    ]
    # Without an index to check against, nothing is filtered.
    assert len(load_examples(path)) == 3


def test_shipped_examples_are_valid_and_name_catalogued_lectures():
    from pathlib import Path

    from slideseek.catalog import load_catalog

    catalog = load_catalog("config/lectures.csv")
    raw = json.loads(Path("config/examples.json").read_text(encoding="utf-8"))
    assert 4 <= len(raw) <= 8
    assert {item["type"] for item in raw} == {"keyword", "paraphrase", "spoken", "visual"}
    assert all(item["lecture_id"] in catalog for item in raw)
    assert len(load_examples("config/examples.json", catalog)) == len(raw)


@pytest.mark.parametrize(
    "template, expected",
    [
        ("https://www.youtube.com/watch?v=bxVkZ4M-hIE&t={seconds}s", "bxVkZ4M-hIE"),
        ("https://www.youtube.com/watch?list=PL1&v=Q1HOKrNeh2M&t={seconds}s", "Q1HOKrNeh2M"),
        ("https://youtu.be/EiO8BBa-xdc?t={seconds}", "EiO8BBa-xdc"),
        ("file:///lectures/lec01.mp4#t={seconds}", None),
        ("", None),
    ],
)
def test_youtube_id(template, expected):
    assert youtube_id(Lecture("x", "X", video_link_template=template)) == expected


# ---- the static frontend ---------------------------------------------------------------------


def test_frontend_obeys_its_own_content_security_policy():
    """The policy forbids inline scripts and inline styles, so the page must use neither."""
    import re

    from slideseek.web.server import STATIC_DIR

    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert not re.search(r"<script(?![^>]*\bsrc=)[^>]*>", html), "inline <script> would be blocked"
    assert "<style" not in html, "inline <style> would be blocked"
    assert not re.search(r"\sstyle=", html), "style attributes would be blocked"
    assert not re.search(r"\son[a-z]+=", html), "inline event handlers would be blocked"

    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert "innerHTML" not in script and "insertAdjacentHTML" not in script
    assert "eval(" not in script


def test_every_local_asset_the_page_references_exists(client):
    import re

    from slideseek.web.server import STATIC_DIR

    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assets = set(re.findall(r'(?:href|src)="(/[^"#]+)"', html))
    assert {"/styles.css", "/app.js", "/favicon.svg"} <= assets
    for asset in assets:
        response = client.get(asset)
        assert response.status_code == 200, asset
    assert "javascript" in client.get("/app.js").headers["content-type"]
    assert client.get("/styles.css").headers["content-type"].startswith("text/css")
    assert client.get("/favicon.svg").headers["content-type"].startswith("image/svg")


def test_page_has_the_hooks_the_script_fills_in():
    import re

    from slideseek.web.server import STATIC_DIR

    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    ids = set(re.findall(r'\sid="([^"]+)"', html))
    wanted = set(re.findall(r'\$\("([^"]+)"\)', script))
    assert wanted, "the script should look elements up by id"
    assert wanted <= ids, (
        f"ids used by the script but missing from the page: {sorted(wanted - ids)}"
    )


# ---- reranking, and running on somebody else's application -----------------------------------


class ReverseReranker:
    """Scores documents so that the first-stage order comes back reversed."""

    def __init__(self):
        self.calls = []

    def score(self, query, documents):
        self.calls.append((query, list(documents)))
        return [float(position) for position in range(len(documents))]


def make_app(reranker=None, **options):
    index = make_index()
    searcher = Searcher(
        index,
        text_embedder=FakeTextEmbedder(),
        image_embedder=FakeImageEmbedder(),
        reranker=reranker,
    )
    options.setdefault("accounts", fast_accounts())
    return create_app(index, searcher, warm_up=False, **options)


def test_the_json_api_does_not_offer_reranking_unless_the_host_does():
    assert TestClient(make_app()).get("/api/info").json()["rerank"] is False
    assert TestClient(make_app(rerank=True)).get("/api/info").json()["rerank"] is True
    client = sign_up(TestClient(make_app()))
    body = client.get("/api/search", params={"q": "gradient descent"}).json()
    assert body["reranked"] is False and "rerank_ms" not in body


def test_run_search_reranks_the_fused_candidates():
    reranker = ReverseReranker()
    run_search = make_app(reranker, rerank=True).state.run_search

    plain = run_search("gradient descent learning rate", k=20)
    reranked = run_search("gradient descent learning rate", k=20, rerank=True)

    assert len(reranker.calls) == 1, "only the reranked search calls the reranker"
    query, documents = reranker.calls[0]
    assert query == "gradient descent learning rate" and len(documents) >= 2
    assert all(document.strip() for document in documents)
    assert reranked["reranked"] is True and reranked["rerank_ms"] >= 0
    # The highest score went to the last candidate, so it now comes first.
    assert reranked["results"][0]["moment_id"] != plain["results"][0]["moment_id"]
    assert [r["rank"] for r in reranked["results"]] == list(range(1, reranked["count"] + 1))
    assert {r["moment_id"] for r in reranked["results"]} <= {
        r["moment_id"] for r in run_search("gradient descent learning rate", k=20)["results"]
    } | {r["moment_id"] for r in plain["results"]}


def test_run_search_refuses_what_the_api_refuses():
    from slideseek.web.server import BadRequest

    run_search = make_app().state.run_search
    for arguments in [("   ",), ("x" * (MAX_QUERY_CHARS + 1),), ("x", 0), ("x", 5, ["nope"])]:
        with pytest.raises(BadRequest):
            run_search(*arguments)


def test_routes_attach_to_an_existing_application_and_leave_room_for_its_own():
    from fastapi import FastAPI

    host = FastAPI()
    returned = make_app(app=host)
    assert returned is host

    # Added after ours, as Gradio's server adds its routes at launch.
    @host.get("/later")
    def later():
        return {"reached": True}

    client = sign_up(TestClient(host))
    assert "Slideseek" in client.get("/").text
    assert client.get("/api/search", params={"q": "chain rule"}).status_code == 200
    assert client.get("/later").json() == {"reached": True}


def test_frontend_files_are_served_one_by_one(client):
    from slideseek.web.server import STATIC_DIR

    vendored = client.get("/vendor/gradio-client.js")
    assert vendored.status_code == 200 and "javascript" in vendored.headers["content-type"]
    assert client.head("/styles.css").status_code == 200
    # Conditional requests still work, as they did when the directory was mounted whole.
    etag = client.get("/app.js").headers["etag"]
    assert client.get("/app.js", headers={"If-None-Match": etag}).status_code == 304
    assert client.get("/vendor/").status_code == 404
    for path in STATIC_DIR.rglob("*"):
        if path.is_file() and not path.name.startswith("."):
            assert client.get(f"/{path.relative_to(STATIC_DIR).as_posix()}").status_code == 200


def test_the_script_loads_the_gradio_client_from_this_origin_only():
    from slideseek.web.server import STATIC_DIR

    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert 'import("/vendor/gradio-client.js")' in script
    assert "cdn." not in script and "https://" not in script.replace(
        "https://www.youtube-nocookie.com", ""
    )
    vendored = (STATIC_DIR / "vendor" / "gradio-client.js").read_text(encoding="utf-8")
    assert vendored.startswith("/* Gradio JavaScript client, @gradio/client ")
    assert "eval(" not in vendored and "new Function" not in vendored


# ---- searching with a picture ------------------------------------------------------------------


def post_picture(client, data, content_type="image/png", **params):
    return client.post(
        "/api/search/picture", content=data, headers={"Content-Type": content_type}, params=params
    )


def test_a_picture_finds_the_moment_its_slide_was_shown(client):
    reply = post_picture(client, make_picture("plot two curves rising falling", "Overfitting"))
    assert reply.status_code == 200
    body = reply.json()
    assert body["kind"] == "picture" and body["read"] == "Overfitting"
    top = body["results"][0]
    assert top["moment_id"] == "lec01:0002"
    assert set(top["badges"]) == {"on slide", "visual"}
    assert top["signals"] == {"image": 1, "bm25_slide": 1}


def test_a_picture_search_can_be_limited_to_a_lecture(client):
    body = post_picture(
        client, make_picture("plot two curves", "Overfitting"), lecture="lec02"
    ).json()
    assert body["count"] >= 1 and {r["lecture_id"] for r in body["results"]} == {"lec02"}


def test_what_is_not_a_usable_picture_is_refused_clearly(client):
    assert post_picture(client, b"").status_code == 400
    assert "not a picture" in post_picture(client, b"plain text, not a png").json()["detail"]
    assert post_picture(client, make_picture("x"), content_type="text/plain").status_code == 415
    unknown = post_picture(client, make_picture("x"), lecture="nope")
    assert unknown.status_code == 400 and "Unknown lecture" in unknown.json()["detail"]
    assert post_picture(client, make_picture("x"), k=0).status_code == 422


def test_a_picture_over_the_size_limit_is_refused(client, monkeypatch):
    import slideseek.web.server as server

    monkeypatch.setattr(server, "MAX_PICTURE_BYTES", 1000)
    reply = post_picture(client, make_picture("x", size=(800, 600)) + b"\0" * 2000)
    assert reply.status_code == 413


def test_an_uploaded_photo_is_turned_upright_and_scaled_down():
    import io

    from PIL import Image

    from slideseek.web.server import PICTURE_SIDE, load_picture

    exif = Image.Exif()
    exif[0x0112] = 6  # stored sideways: rotate 90 degrees to show it upright
    buffer = io.BytesIO()
    Image.new("RGB", (3200, 1800), "white").save(buffer, "JPEG", exif=exif)
    picture = load_picture(buffer.getvalue())
    assert picture.mode == "RGB"
    assert picture.size[1] > picture.size[0], "the sideways photo is now portrait"
    assert max(picture.size) == PICTURE_SIDE


def test_the_page_offers_picture_search_and_may_show_the_picture(client):
    page = client.get("/")
    assert 'id="picture-input"' in page.text and 'accept="image/*"' in page.text
    assert "blob:" in page.headers["content-security-policy"]
    info = client.get("/api/info").json()
    assert info["pictures"]["read_text"] is True and info["pictures"]["benchmark"] is None
