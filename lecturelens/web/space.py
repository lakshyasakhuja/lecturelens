"""The Hugging Face Space: the same web app on Gradio's server, with the reranker on a GPU.

Hugging Face lends free accounts a GPU only through ZeroGPU, and ZeroGPU runs only Gradio
apps. Gradio's server mode (``gradio.Server``) is a FastAPI application with Gradio's queue
attached, so the routes and the page from ``server.py`` are served by it unchanged, and one
more endpoint is added through Gradio:

    /gradio_api/call/rerank    the search, then the cross-encoder over the fused top 20

Everything a visitor needs to search runs on the Space's CPU, exactly as it does anywhere
else, and costs them nothing. Only the forward passes of the reranker run inside
``@spaces.GPU``, where ZeroGPU attaches a GPU for the length of the call and charges the
time to the visitor's daily allowance. The page calls that endpoint through Gradio's
JavaScript client, which is what carries the visitor's identity to ZeroGPU.

Outside ZeroGPU, ``@spaces.GPU`` does nothing and the reranker runs on whatever device it
is given, so the Space can be run and tested on a laptop.

``spaces`` has to be imported before anything initialises CUDA, so the Space's ``app.py``
imports it first; the import here is second in line.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

import gradio
import numpy as np
import spaces

from lecturelens import __version__
from lecturelens.index import Index
from lecturelens.rerank import Reranker
from lecturelens.search import Searcher
from lecturelens.web.server import (
    BadRequest,
    create_app,
    load_benchmark,
    load_examples,
    load_index_from_env,
)

# The longest one reranking call may hold a GPU, in seconds. Twenty short documents take
# about a second; the rest is headroom for the first call, which also moves the model
# across. A shorter limit is kinder to visitors: ZeroGPU will not start a call that asks
# for more time than they have left today.
RERANK_GPU_SECONDS = 20


def on_zero_gpu() -> bool:
    """Is this process running on ZeroGPU hardware?"""
    return os.environ.get("SPACES_ZERO_GPU", "").lower() in {"1", "t", "true"}


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
    examples: Sequence[Mapping[str, str]] = (),
    benchmark: Mapping[str, Any] | None = None,
    device: str = "cpu",
    warm_up: bool = True,
) -> gradio.Server:
    """The web app as a Gradio server. Call ``.launch()`` on the result to serve it.

    ``reranker`` is anything with ``score(query, documents)``. By default it is the
    cross-encoder, loaded now: ZeroGPU can only hand a model to a GPU later if the model
    was placed on ``cuda`` while the app was starting.

    Every model is loaded here, before the server starts, rather than in the background as
    ``create_app`` does. ZeroGPU patches PyTorch for the thread that imported ``spaces``,
    so that thread is where models are meant to be loaded; and a Space is not shown to
    visitors until it has started, so nobody waits on it.
    """
    if on_zero_gpu():
        # There is no GPU outside a @spaces.GPU call, whatever torch.cuda.is_available says.
        device = "cpu"
    if reranker is None:
        model = Reranker(device="cuda" if on_zero_gpu() else device).load()
        reranker = GpuReranker(model)
    if searcher is None:
        searcher = Searcher(index, device=device)
    searcher.reranker = reranker
    if warm_up:
        searcher.warm_up()

    server = gradio.Server(
        title="LectureLens", version=__version__, docs_url=None, redoc_url=None, openapi_url=None
    )
    create_app(
        index,
        searcher,
        examples=examples,
        benchmark=benchmark,
        device=device,
        warm_up=False,
        app=server,
        rerank=True,
    )
    run_search = server.state.run_search

    # One call at a time: the queue is Gradio's, and so is the place a visitor waits in it.
    @server.api(name="rerank", concurrency_limit=1)
    def rerank(query: str, k: int = 5, lectures: list[str] | None = None) -> dict:
        """Search, then rescore the fused top results with the cross-encoder."""
        try:
            return run_search(query, int(k), lectures or None, rerank=True)
        except BadRequest as error:
            raise gradio.Error(str(error)) from error

    return server


def create_space_from_env() -> gradio.Server:
    """The Space, configured from the environment like ``create_app_from_env``.

    It reads the same variables: ``LECTURELENS_INDEX``, ``LECTURELENS_INDEX_REPO``,
    ``LECTURELENS_EXAMPLES``, ``LECTURELENS_BENCHMARK`` and ``LECTURELENS_DEVICE`` (the
    device of the query encoders, ``cpu`` unless set).
    """
    from lecturelens.logs import configure_logging

    configure_logging()
    # The reranker is tokenised in this process and scored in one forked from it.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    index = load_index_from_env()
    return create_space(
        index,
        examples=load_examples(
            os.environ.get("LECTURELENS_EXAMPLES", "config/examples.json"), index.lectures
        ),
        benchmark=load_benchmark(
            os.environ.get("LECTURELENS_BENCHMARK", "results/test/metrics.json")
        ),
        device=os.environ.get("LECTURELENS_DEVICE", "cpu"),
    )


def launch(server: gradio.Server, **options: Any):
    """Serve the Space. No server-side rendering: the page is ours, not Gradio's."""
    options.setdefault("ssr_mode", False)
    options.setdefault("show_error", True)
    return server.launch(**options)
