"""The Hugging Face Space: the same web app on Gradio's server, with the reranker on a GPU.

Hugging Face lends free accounts a GPU only through ZeroGPU, and ZeroGPU runs only Gradio
apps. Gradio's server mode (``gradio.Server``) is a FastAPI application with Gradio's queue
attached, so the routes and the page from ``server.py`` are served by it unchanged, and one
more endpoint is added through Gradio:

    /gradio_api/call/rerank    the search, then the cross-encoder over the fused top 20

Searching needs an account here as everywhere; the endpoint checks the visitor's session
cookie like the JSON routes do. Everything a visitor needs to search runs on the Space's
CPU, exactly as it does anywhere else, and costs them nothing. Only the forward passes of the reranker run inside
``@spaces.GPU``, where ZeroGPU attaches a GPU for the length of the call and charges the
time to the visitor's daily allowance. The page calls that endpoint through Gradio's
JavaScript client, which is what carries the visitor's identity to ZeroGPU.

Outside ZeroGPU, ``@spaces.GPU`` does nothing and the reranker runs on whatever device it
is given, so the Space can be run and tested on a laptop.

A Space's own disk is wiped whenever it restarts, so accounts kept on it last only until
then. ``SLIDESEEK_DATABASE_URL`` names a database that outlives it (PostgreSQL); failing
that, a Space with persistent storage keeps them under ``/data``.

Reranking is off unless ``SLIDESEEK_RERANK=1``, because on ZeroGPU it does not work yet.
Measured on the deployed Space in November 2026 (torch 2.13.0, spaces 0.51.3): a GPU call
succeeds until the Qwen3 text encoder has run one forward pass on the CPU in this process,
and from then on every call fails in ZeroGPU's worker with "No CUDA GPUs are available".
Loading the encoder, and loading and running SigLIP, do not have that effect. Searching
needs that forward pass, so the two cannot share a process as things stand; see
``deploy/README.md``.

``spaces`` has to be imported before anything initialises CUDA, so the Space's ``app.py``
imports it first; the import here is second in line.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import gradio
import numpy as np
import spaces

from slideseek import __version__
from slideseek.accounts import AccountStore
from slideseek.index import Index
from slideseek.rerank import Reranker
from slideseek.search import Searcher
from slideseek.web.server import (
    DEFAULT_DATABASE_URL,
    SIGN_IN_FIRST,
    BadRequest,
    accounts_from_env,
    create_app,
    load_benchmark,
    load_examples,
    load_index_from_env,
    load_picture_benchmark,
)

log = logging.getLogger(__name__)

# The longest one reranking call may hold a GPU, in seconds. Twenty short documents take
# about a second; the rest is headroom for the first call, which also moves the model
# across. A shorter limit is kinder to visitors: ZeroGPU will not start a call that asks
# for more time than they have left today.
RERANK_GPU_SECONDS = 20


def on_zero_gpu() -> bool:
    """Is this process running on ZeroGPU hardware?"""
    return os.environ.get("SPACES_ZERO_GPU", "").lower() in {"1", "t", "true"}


def rerank_enabled() -> bool:
    """Does the environment ask for reranking? ``SLIDESEEK_RERANK=1`` turns it on."""
    return os.environ.get("SLIDESEEK_RERANK", "").lower() in {"1", "t", "true", "yes", "on"}


def default_database_url() -> str:
    """Where accounts go without ``SLIDESEEK_DATABASE_URL``: persistent storage if the Space
    has it (mounted at ``/data``), or else a file that the next restart will wipe."""
    data = Path("/data")
    if data.is_dir() and os.access(data, os.W_OK):
        return f"sqlite:///{data / 'slideseek-accounts.db'}"
    if not os.environ.get("SLIDESEEK_DATABASE_URL"):
        log.warning(
            "no SLIDESEEK_DATABASE_URL and no persistent storage: accounts will be lost "
            "when the Space restarts"
        )
    return DEFAULT_DATABASE_URL


def request_cookies(request: Any) -> dict[str, str]:
    """The cookies of the HTTP request behind a Gradio call, however Gradio holds them."""
    raw = getattr(request, "request", None)
    if raw is not None:
        return dict(raw.cookies)
    kwargs = getattr(request, "kwargs", None) or {}
    return dict(kwargs.get("cookies") or {})


@spaces.GPU(duration=1)
def _gpu_placeholder() -> None:
    """Never called. ZeroGPU will not start a Space that has no ``@spaces.GPU`` function,
    and with reranking off the app would otherwise have none."""


class GpuReranker:
    """What ``Searcher`` sees as its reranker: tokenise here, score inside ``@spaces.GPU``."""

    def __init__(self, reranker: Reranker, seconds: int = RERANK_GPU_SECONDS) -> None:
        self.reranker = reranker

        @spaces.GPU(duration=seconds)
        def forward(prompts: list[list[int]]) -> list[float]:
            return reranker.score_prompts(prompts).tolist()

        self._forward = forward

    def score(self, query: str, documents: Sequence[str]) -> np.ndarray:
        prompts = self.reranker.prompts(query, documents)
        return np.asarray(self._forward(prompts), dtype=np.float64)


def create_space(
    index: Index,
    searcher: Searcher | None = None,
    *,
    reranker: Any | None = None,
    rerank: bool = True,
    examples: Sequence[Mapping[str, str]] = (),
    benchmark: Mapping[str, Any] | None = None,
    picture_benchmark: Mapping[str, Any] | None = None,
    device: str = "cpu",
    warm_up: bool = True,
    accounts: AccountStore | None = None,
) -> gradio.Server:
    """The web app as a Gradio server. Call ``.launch()`` on the result to serve it.

    ``reranker`` is anything with ``score(query, documents)``. By default it is the
    cross-encoder, loaded now: ZeroGPU can only hand a model to a GPU later if the model
    was placed on ``cuda`` while the app was starting. With ``rerank=False`` no reranker is
    loaded, the page shows no switch and the endpoint does not exist.

    Every model is loaded here, before the server starts, rather than in the background as
    ``create_app`` does. ZeroGPU patches PyTorch for the thread that imported ``spaces``,
    so that thread is where models are meant to be loaded; and a Space is not shown to
    visitors until it has started, so nobody waits on it.
    """
    if on_zero_gpu():
        # There is no GPU outside a @spaces.GPU call, whatever torch.cuda.is_available says.
        device = "cpu"
    if rerank and reranker is None:
        model = Reranker(device="cuda" if on_zero_gpu() else device).load()
        reranker = GpuReranker(model)
    if searcher is None:
        searcher = Searcher(index, device=device)
    if rerank:
        searcher.reranker = reranker
    if warm_up:
        searcher.warm_up()
        searcher.warm_up_pictures()

    server = gradio.Server(
        title="Slideseek", version=__version__, docs_url=None, redoc_url=None, openapi_url=None
    )
    create_app(
        index,
        searcher,
        examples=examples,
        benchmark=benchmark,
        picture_benchmark=picture_benchmark,
        device=device,
        warm_up=False,
        app=server,
        rerank=rerank,
        accounts=accounts,
    )
    if not rerank:
        return server
    run_search, account_for = server.state.run_search, server.state.account_for

    # One call at a time: the queue is Gradio's, and so is the place a visitor waits in it.
    @server.api(name="rerank", concurrency_limit=1)
    def rerank(
        query: str,
        k: int = 5,
        lectures: list[str] | None = None,
        request: gradio.Request = None,
    ) -> dict:
        """Search, then rescore the fused top results with the cross-encoder."""
        if account_for(request_cookies(request)) is None:
            raise gradio.Error(SIGN_IN_FIRST)
        try:
            return run_search(query, int(k), lectures or None, rerank=True)
        except BadRequest as error:
            raise gradio.Error(str(error)) from error

    return server


def create_space_from_env() -> gradio.Server:
    """The Space, configured from the environment like ``create_app_from_env``.

    It reads the same variables: ``SLIDESEEK_INDEX``, ``SLIDESEEK_INDEX_REPO``,
    ``SLIDESEEK_EXAMPLES``, ``SLIDESEEK_BENCHMARK`` and ``SLIDESEEK_DEVICE`` (the
    device of the query encoders, ``cpu`` unless set), and one more: ``SLIDESEEK_RERANK``,
    which turns reranking on when it is ``1``; and ``SLIDESEEK_DATABASE_URL``, where the
    accounts are kept (see ``default_database_url`` for what happens without it).
    """
    from slideseek.logs import configure_logging

    configure_logging()
    # The reranker is tokenised in this process and scored in one forked from it.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    index = load_index_from_env()
    return create_space(
        index,
        rerank=rerank_enabled(),
        examples=load_examples(
            os.environ.get("SLIDESEEK_EXAMPLES", "config/examples.json"), index.lectures
        ),
        benchmark=load_benchmark(
            os.environ.get("SLIDESEEK_BENCHMARK", "results/test/metrics.json")
        ),
        picture_benchmark=load_picture_benchmark(
            os.environ.get("SLIDESEEK_PICTURE_BENCHMARK", "results/screenshot/metrics.json")
        ),
        device=os.environ.get("SLIDESEEK_DEVICE", "cpu"),
        accounts=accounts_from_env(default_database_url()),
    )


def launch(server: gradio.Server, **options: Any):
    """Serve the Space. No server-side rendering: the page is ours, not Gradio's."""
    options.setdefault("ssr_mode", False)
    options.setdefault("show_error", True)
    return server.launch(**options)
