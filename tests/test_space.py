"""The Hugging Face Space: the web app on Gradio's server, with reranking as a Gradio call."""

from __future__ import annotations

import re
import socket
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="the web extra is not installed")
pytest.importorskip("gradio", reason="the space extra is not installed")
pytest.importorskip("spaces", reason="the space extra is not installed")

from fakes import ADA, FakeImageEmbedder, FakeTextEmbedder, fast_accounts, make_index
from fastapi.testclient import TestClient

from slideseek.search import Searcher
from slideseek.web.space import (
    GpuReranker,
    create_space,
    launch,
    on_zero_gpu,
    rerank_enabled,
)

ROOT = Path(__file__).resolve().parent.parent


class StubCrossEncoder:
    """Stands in for ``Reranker``: one prompt per document, longest document scores highest."""

    def __init__(self):
        self.scored = []

    def prompts(self, query, documents):
        return [[len(query), len(document)] for document in documents]

    def score_prompts(self, prompts):
        self.scored.append([list(prompt) for prompt in prompts])
        return np.asarray([float(length) for _, length in prompts])


def make_space(reranker=None):
    index = make_index()
    searcher = Searcher(index, text_embedder=FakeTextEmbedder(), image_embedder=FakeImageEmbedder())
    return create_space(
        index,
        searcher,
        reranker=reranker or GpuReranker(StubCrossEncoder()),
        warm_up=False,
        accounts=fast_accounts(),
    )


def signed_in(server) -> TestClient:
    client = TestClient(server)
    assert client.post("/api/account/signup", json=ADA).status_code == 201
    return client


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_gpu_reranker_tokenises_here_and_scores_in_the_gpu_call():
    model = StubCrossEncoder()
    scores = GpuReranker(model).score("why", ["short", "a longer document"])
    assert scores.tolist() == [5.0, 17.0]
    assert model.scored == [[[3, 5], [3, 17]]], "the GPU call receives token ids, nothing else"


def test_not_on_zero_gpu_here(monkeypatch):
    monkeypatch.delenv("SPACES_ZERO_GPU", raising=False)
    assert on_zero_gpu() is False
    monkeypatch.setenv("SPACES_ZERO_GPU", "true")
    assert on_zero_gpu() is True


def test_reranking_is_off_unless_the_environment_asks(monkeypatch):
    monkeypatch.delenv("SLIDESEEK_RERANK", raising=False)
    assert rerank_enabled() is False
    monkeypatch.setenv("SLIDESEEK_RERANK", "0")
    assert rerank_enabled() is False
    monkeypatch.setenv("SLIDESEEK_RERANK", "1")
    assert rerank_enabled() is True


def test_a_space_without_reranking_loads_no_reranker_and_has_no_endpoint():
    from gradio_client import Client

    index = make_index()
    searcher = Searcher(index, text_embedder=FakeTextEmbedder(), image_embedder=FakeImageEmbedder())
    server = create_space(index, searcher, rerank=False, warm_up=False, accounts=fast_accounts())
    assert searcher.reranker is None, "nothing was loaded"
    _, url, _ = launch(server, prevent_thread_lock=True, server_port=free_port(), quiet=True)
    try:
        client = signed_in(server)
        assert "Slideseek" in client.get("/").text
        assert client.get("/api/info").json()["rerank"] is False
        assert client.get("/api/search", params={"q": "chain rule"}).json()["count"] >= 1
        with pytest.raises(Exception):  # noqa: B017 -- gradio_client has no one error type
            Client(url, verbose=False).predict(query="x", k=5, lectures=None, api_name="/rerank")
    finally:
        server.get_blocks().close()


def test_the_space_serves_the_same_page_and_api():
    server = make_space()
    assert TestClient(server).get("/api/search", params={"q": "chain rule"}).status_code == 401
    client = signed_in(server)
    assert "Slideseek" in client.get("/").text
    info = client.get("/api/info").json()
    assert info["rerank"] is True and info["n_moments"] > 0
    body = client.get("/api/search", params={"q": "chain rule"}).json()
    assert body["count"] >= 1 and body["reranked"] is False
    assert client.get("/api/nope").json() == {"detail": "Not found."}


def test_reranking_is_a_gradio_call():
    from gradio_client import Client

    server = make_space()
    _, url, _ = launch(server, prevent_thread_lock=True, server_port=free_port(), quiet=True)
    try:
        browser = signed_in(server)
        plain = browser.get("/api/search", params={"q": "gradient descent learning rate", "k": 20})

        # Reranking needs an account too: a call without the session cookie is refused.
        with pytest.raises(Exception, match="Sign in to search"):
            Client(url, verbose=False).predict(
                query="gradient descent", k=5, lectures=None, api_name="/rerank"
            )

        cookie = f"slideseek_session={browser.cookies['slideseek_session']}"
        client = Client(url, verbose=False, headers={"Cookie": cookie})
        reranked = client.predict(
            query="gradient descent learning rate", k=20, lectures=None, api_name="/rerank"
        )
        assert reranked["reranked"] is True and reranked["count"] == plain.json()["count"]
        assert {r["moment_id"] for r in reranked["results"]} == {
            r["moment_id"] for r in plain.json()["results"]
        }

        only = client.predict(query="descent", k=3, lectures=["lec02"], api_name="/rerank")
        assert {r["lecture_id"] for r in only["results"]} <= {"lec02"}

        with pytest.raises(Exception, match="The query is empty"):
            client.predict(query="   ", k=5, lectures=None, api_name="/rerank")
        with pytest.raises(Exception, match="Unknown lecture"):
            client.predict(query="x", k=5, lectures=["nope"], api_name="/rerank")

        # Gradio's own routes are reachable beside ours, and ours still win at "/".
        assert TestClient(server).get("/gradio_api/info").status_code == 200
        assert "Slideseek" in TestClient(server).get("/").text
    finally:
        server.get_blocks().close()


# ---- the files the Space is built from -------------------------------------------------------


def requirement_names(lines) -> set[str]:
    names = set()
    for line in lines:
        line = line.split("#")[0].strip()
        if line:
            names.add(re.split(r"[<>=!~\[ ]", line, maxsplit=1)[0].lower().replace("_", "-"))
    return names


def toml_list(text: str, key: str) -> list[str]:
    """The strings of ``key = [ ... ]`` in pyproject.toml (Python 3.10 has no TOML reader)."""
    block = re.search(rf"^{key} = \[(.*?)^\]", text, re.M | re.S).group(1)
    return re.findall(r'"([^"]+)"', block)


def test_space_requirements_cover_what_the_app_imports():
    """A Gradio Space installs requirements.txt and nothing else; it does not install us."""
    project = (ROOT / "pyproject.toml").read_text()
    wanted = requirement_names(toml_list(project, "dependencies"))
    wanted |= requirement_names(toml_list(project, "models"))
    wanted |= requirement_names(toml_list(project, "ocr"))
    wanted |= requirement_names(toml_list(project, "web"))
    wanted |= requirement_names(toml_list(project, "postgres"))
    # Command-line only; and FastAPI and its server, which come with Gradio.
    wanted -= {"typer", "fastapi", "uvicorn"}
    listed = requirement_names((ROOT / "deploy/space/requirements.txt").read_text().splitlines())
    assert wanted <= listed, f"missing from the Space's requirements: {sorted(wanted - listed)}"
    assert not listed & {"gradio", "spaces"}, "the Space's SDK provides these; do not pin them"


def test_space_metadata_matches_the_entry_point():
    card = (ROOT / "deploy/space/README.md").read_text()
    front = card.split("---")[1]
    assert re.search(r"^sdk: gradio$", front, re.M)
    assert re.search(r"^app_file: app\.py$", front, re.M)
    assert re.search(r"^python_version: ['\"]?3\.12", front, re.M)
    version = re.search(r"^sdk_version: ['\"]?([\d.]+)", front, re.M).group(1)
    pinned = re.search(r'"gradio>=([\d.]+)', (ROOT / "pyproject.toml").read_text()).group(1)
    assert tuple(map(int, version.split("."))) >= tuple(map(int, pinned.split(".")))
    entry = (ROOT / "deploy/space/app.py").read_text()
    code = [line for line in entry.splitlines() if line.startswith(("import ", "from "))]
    assert code[0].startswith("import spaces"), "spaces must be imported before anything else"
