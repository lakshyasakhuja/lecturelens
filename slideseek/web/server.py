"""The web server: a small JSON API over the searcher, plus the static frontend.

    GET /api/health                 is the server up, and are the encoders loaded?
    GET /api/info                   what the index contains: lectures, counts, models, examples
    GET /api/search?q=...           the top moments for a query                 (signed in)
    POST /api/search/picture        the moments where the slide in a picture was shown  (signed in)
    GET /thumbs/<lecture>/<ms>.jpg  slide thumbnails, straight from the index directory
    /api/library...                 your own lectures, added from the browser: ``library.py``
    GET /                           the frontend (plain HTML, CSS and JavaScript, no build step)
    GET /admin                      the accounts page, for admins
    /api/account...                 signing up, in and out: see ``auth.py``
    /api/admin/...                  seeing and changing every account (admins): ``admin.py``

Searching needs an account. Without one, the page and what the library contains are still
shown, but a search is answered with 401 until the visitor signs in.

Beyond the loaded index, the state is the accounts, which live in a SQL database
(``accounts.py``), and, when the server is given one, the lectures people added themselves
(``slideseek.library``). Each account searches the public lectures and its own together:
``scope`` narrows that to ``mine`` or ``public``.

The server hosts no video: a result either links out to where the lecture already lives (a
YouTube id parsed from the lecture's link template), or, when run locally with a directory
of recordings, plays the local file, or, for somebody's own lecture, tells the page which
file on their computer to play it from.

Search runs on a CPU. One query at a time goes through the encoders, guarded by a lock: a
query takes about a tenth of a second, and serialising them is simpler and safer than
letting several threads share the models.

The routes can be attached to an application that already exists (``create_app(app=...)``),
which is how the Hugging Face Space runs them on Gradio's server: see ``space.py``.
"""

from __future__ import annotations

import io
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from slideseek import __version__
from slideseek.accounts import Account, AccountStore, redacted
from slideseek.catalog import VIDEO_EXTENSIONS, Lecture
from slideseek.index import Index
from slideseek.search import PICTURE_RANKERS, Result, Searcher
from slideseek.utils import format_timestamp, read_json
from slideseek.web.admin import add_admin_routes
from slideseek.web.auth import UNSAFE_METHODS, add_account_routes, same_origin

if TYPE_CHECKING:
    from slideseek.library import Library

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
MAX_QUERY_CHARS = 300
MAX_RESULTS = 20
# A picture query: what is accepted, and the size it is scaled down to before searching.
MAX_PICTURE_BYTES = 8 * 1024 * 1024
MAX_PICTURE_PIXELS = 40_000_000
PICTURE_SIDE = 1600
# Where accounts are kept when nothing says otherwise: a file beside where the server runs.
DEFAULT_DATABASE_URL = "sqlite:///accounts.db"
SIGN_IN_FIRST = "Sign in to search."
# Files a browser must check are current before using a copy it kept.
REVALIDATED = (".html", ".js", ".mjs", ".css", ".wasm")
# What a search covers: everything the visitor can search, only their own lectures, or only
# the public ones.
SCOPES = ("all", "mine", "public")
# Searchers over somebody's own lectures and the public ones, kept for the accounts that
# searched most recently.
OWN_SEARCHERS = 20


class BadRequest(ValueError):
    """The request cannot be answered as asked; the message says why, for the visitor."""


_YOUTUBE_ID = re.compile(r"(?:youtube\.com/watch\?(?:.*&)?v=|youtu\.be/)([A-Za-z0-9_-]{6,})")

# What the page may load. Everything is first-party except the video player and its poster.
CONTENT_SECURITY_POLICY = "; ".join(
    [
        "default-src 'self'",
        "script-src 'self'",
        "style-src 'self'",
        # blob: is the visitor's own picture, previewed before it is searched.
        "img-src 'self' data: blob: https://i.ytimg.com",
        # blob: is a recording on the visitor's own computer: read to add it as a lecture,
        # and played from there.
        "media-src 'self' blob:",
        "frame-src https://www.youtube-nocookie.com",
        "base-uri 'none'",
        "form-action 'self'",
        # Hugging Face shows a Space inside a frame on its own pages.
        "frame-ancestors 'self' https://huggingface.co https://*.hf.space",
    ]
)
# Adding a lecture in the browser (static/capture/): ONNX Runtime compiles WebAssembly and
# runs in a worker, and Whisper's weights come from the Hugging Face Hub, which hands the
# download on to storage under hf.co. A worker is governed by the policy it was served with,
# so only the worker's scripts get this one; the page that starts the worker does not.
SPEECH_PATHS = ("/capture/", "/vendor/browser/", "/lab/")
SPEECH_CONTENT_SECURITY_POLICY = (
    CONTENT_SECURITY_POLICY.replace("script-src 'self'", "script-src 'self' 'wasm-unsafe-eval'")
    .replace(
        "default-src 'self'",
        "default-src 'self'; worker-src 'self'; "
        "connect-src 'self' https://huggingface.co https://*.hf.co",
    )
)


def load_picture(data: bytes):
    """Decode an uploaded picture, upright and in RGB, no larger than ``PICTURE_SIDE``."""
    from PIL import Image, ImageOps, UnidentifiedImageError

    if not data:
        raise BadRequest("The picture is empty.")
    if len(data) > MAX_PICTURE_BYTES:
        raise BadRequest("The picture is larger than 8 MB.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            width, height = probe.size
            if width * height > MAX_PICTURE_PIXELS:
                raise BadRequest("The picture has too many pixels.")
            probe.verify()
        with Image.open(io.BytesIO(data)) as opened:
            # A phone photo is stored sideways with a note saying which way is up.
            picture = ImageOps.exif_transpose(opened).convert("RGB")
    except BadRequest:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        ValueError,
        Image.DecompressionBombError,
    ) as error:
        raise BadRequest("That file is not a picture this app can read.") from error
    picture.thumbnail((PICTURE_SIDE, PICTURE_SIDE))
    return picture


def youtube_id(lecture: Lecture) -> str | None:
    """The YouTube video id in a lecture's link template, if that is where it lives."""
    match = _YOUTUBE_ID.search(lecture.video_link_template or "")
    return match.group(1) if match else None


def find_local_videos(
    videos_dir: str | Path | None, lectures: Mapping[str, Lecture]
) -> dict[str, str]:
    """``{lecture_id: file name}`` for lectures whose recording is in ``videos_dir``."""
    if not videos_dir:
        return {}
    found = {}
    for path in sorted(Path(videos_dir).iterdir()):
        if path.suffix.lower() in VIDEO_EXTENSIONS and path.stem in lectures:
            found[path.stem] = path.name
    return found


def load_examples(
    path: str | Path | None, lectures: Mapping[str, Lecture] | None = None
) -> list[dict[str, str]]:
    """Example queries for the search box: ``[{"query": ..., "type": ...}, ...]``.

    An example may name the lecture that answers it (``"lecture_id"``). Given the index's
    lectures, examples for lectures that are not in it are dropped, so the questions
    written for the public demo never appear over somebody's own recordings.
    """
    if not path or not Path(path).is_file():
        return []
    examples = []
    for item in read_json(path):
        query = str(item.get("query", "")).strip()
        lecture_id = item.get("lecture_id")
        if not query or (lectures is not None and lecture_id and lecture_id not in lectures):
            continue
        examples.append({"query": query, "type": str(item.get("type", ""))})
    return examples


def load_benchmark(path: str | Path | None) -> dict[str, Any] | None:
    """The measured ablation, if the benchmark has been run. Never anything else.

    The page shows numbers only from ``metrics.json`` as written by ``eval/evaluate.py``.
    When that file does not exist, the page says the benchmark is pending.
    """
    if not path or not Path(path).is_file():
        return None
    metrics = read_json(path)
    return {
        "split": metrics.get("split"),
        "n_queries": metrics.get("n_queries"),
        "tolerance_s": metrics.get("tolerance_s"),
        "latency": metrics.get("latency"),
        "systems": [
            {
                "number": system.get("number"),
                "label": system.get("label"),
                "r1": system["overall"]["r1"],
                "r5": system["overall"]["r5"],
                "mrr10": system["overall"]["mrr10"],
                "by_type": {kind: block["r5"] for kind, block in system["by_type"].items()},
            }
            for system in metrics.get("systems", [])
            if system.get("number")
        ],
    }


def load_picture_benchmark(path: str | Path | None) -> dict[str, Any] | None:
    """The measured picture search, as written by ``eval/eval_screenshot.py``, if it exists."""
    if not path or not Path(path).is_file():
        return None
    metrics = read_json(path)
    return {key: metrics.get(key) for key in ("queries", "lectures", "methods")}


def create_app(
    index: Index,
    searcher: Searcher | None = None,
    *,
    videos_dir: str | Path | None = None,
    examples: Sequence[Mapping[str, str]] = (),
    benchmark: Mapping[str, Any] | None = None,
    picture_benchmark: Mapping[str, Any] | None = None,
    device: str = "cpu",
    warm_up: bool = True,
    app: FastAPI | None = None,
    rerank: bool = False,
    accounts: AccountStore | None = None,
    lab_dir: str | Path | None = None,
    library: Library | str | Path | None = None,
) -> FastAPI:
    """Build the application around a loaded index.

    ``app`` attaches the routes to an existing application instead of a new one. ``rerank``
    says that the host offers the cross-encoder as a second stage: ``/api/info`` reports it
    so the page can show the option, and ``app.state.run_search(..., rerank=True)`` runs it.
    The JSON API itself never reranks; the host decides how that step is reached.

    ``accounts`` is where the accounts are kept. Without one they are kept in memory and
    are gone when the server stops, which suits a test and nothing else.

    ``lab_dir`` turns on the lab (``lab.py``), which saves what it is sent into that
    directory. Only for a server on your own machine.

    ``library`` lets people add their own lectures (``slideseek.library``): a directory to
    keep them in, or a ``Library`` already made. Either way it shares ``searcher``'s models
    and lock, and is told when an account is deleted.
    """
    searcher = searcher or Searcher(index, device=device)
    if accounts is None:
        accounts = AccountStore("sqlite://")
        log.warning("accounts are kept in memory: they will be lost when the server stops")
    local_videos = find_local_videos(videos_dir, index.lectures)
    lock = threading.Lock()
    ready = threading.Event()
    if isinstance(library, (str, Path)):
        library = make_library(library, searcher, lock)
    elif library is not None:
        library.model_lock = lock

    def _warm() -> None:
        with lock:
            try:
                searcher.warm_up()
                searcher.warm_up_pictures()
            finally:
                ready.set()

    if warm_up:
        threading.Thread(target=_warm, name="warm-up", daemon=True).start()
    else:
        ready.set()

    if app is None:
        app = FastAPI(title="Slideseek", version=__version__, docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        # The session cookie is sent from other sites too (see auth.py), so anything that
        # changes something must come from this site's own page.
        if (
            request.method in UNSAFE_METHODS
            and request.url.path.startswith("/api/")
            and not same_origin(request)
        ):
            response = JSONResponse({"detail": "Requests from other sites are refused."}, 403)
        else:
            response = await call_next(request)
        response.headers.setdefault(
            "Content-Security-Policy",
            SPEECH_CONTENT_SECURITY_POLICY
            if request.url.path == "/lab" or request.url.path.startswith(SPEECH_PATHS)
            else CONTENT_SECURITY_POLICY,
        )
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        if request.url.path.startswith("/thumbs/"):
            response.headers.setdefault("Cache-Control", "public, max-age=86400")
        elif request.url.path == "/" or request.url.path.endswith(REVALIDATED):
            # The page, its scripts and the speech worker change together: a browser must
            # check each is current (a cheap 304 when it is), never mix an old one with new.
            response.headers.setdefault("Cache-Control", "no-cache")
        return response

    add_account_routes(app, accounts)
    add_admin_routes(app, accounts, library)
    if library is not None:
        from slideseek.library import owner_key

        accounts.on_account_deleted = lambda account: library.delete_owner(owner_key(account))
    own_searchers: OrderedDict[str, tuple[int, Searcher, dict[str, dict]]] = OrderedDict()

    def searcher_for(account: Account | None) -> tuple[Searcher, dict[str, dict]]:
        """The searcher for this account, and the files its own lectures came from.

        Call it holding ``lock``: a new searcher shares the models with ``searcher``.
        """
        if library is None or account is None:
            return searcher, {}
        owner = owner_key(account)
        version = library.version(owner)
        held = own_searchers.get(owner)
        if held is not None and held[0] == version:
            own_searchers.move_to_end(owner)
            return held[1], held[2]
        own = library.index(owner)
        if own is None:
            own_searchers.pop(owner, None)
            return searcher, {}
        combined = Searcher(
            Index.combine(index, own),
            config=searcher.config,
            text_embedder=searcher.text_embedder,
            image_embedder=searcher.image_embedder,
            reranker=searcher.reranker,
            device=device,
            ocr=searcher.ocr if searcher.can_read_pictures else None,
        )
        files = {
            item["lecture_id"]: item["file"]
            for item in library.lectures(owner)
            if item["stage"] == "ready"
        }
        own_searchers[owner] = (version, combined, files)
        while len(own_searchers) > OWN_SEARCHERS:
            own_searchers.popitem(last=False)
        return combined, files

    def scope_mask(active: Searcher, scope: str, lecture: Sequence[str] | None):
        """Which moments a search may return: ``scope`` and ``lecture`` together."""
        if scope not in SCOPES:
            raise BadRequest(f"Search scope is one of {', '.join(SCOPES)}.")
        wanted = set(lecture or active.index.lectures)
        if scope == "public":
            wanted &= set(index.lectures)
        elif scope == "mine":
            wanted -= set(index.lectures)
        if not wanted:
            return np.zeros(len(active.index), dtype=bool)  # nothing: no results
        if wanted >= set(active.index.lectures):
            return None
        return active.lecture_mask(wanted)

    def signed_in(request: Request) -> Account:
        account = app.state.account_for(request.cookies)
        if account is None:
            raise HTTPException(status_code=401, detail=SIGN_IN_FIRST)
        return account

    def player(
        lecture_id: str, start: float, own: Mapping[str, dict] | None = None
    ) -> dict[str, Any] | None:
        if own and lecture_id in own:
            # Somebody's own lecture: the page plays it from the file on their computer.
            return {
                "kind": "own",
                "lecture_id": lecture_id,
                "file": dict(own[lecture_id]),
                "start": int(start),
            }
        if lecture_id in local_videos:
            return {
                "kind": "file",
                "src": f"/videos/{local_videos[lecture_id]}",
                "start": int(start),
            }
        lecture = index.lectures.get(lecture_id)
        video_id = youtube_id(lecture) if lecture else None
        if video_id:
            return {"kind": "youtube", "video_id": video_id, "start": int(start)}
        return None

    def serialise(
        result: Result, active: Searcher | None = None, own: Mapping[str, dict] | None = None
    ) -> dict[str, Any]:
        relative = str((active or searcher).index.moments["thumb"].iat[result.row])
        if result.thumb is None:
            thumb = None
        elif own and result.lecture_id in own:
            thumb = f"/api/library/thumbs/{result.lecture_id}/{Path(relative).name}"
        else:
            thumb = f"/{relative}"
        return {
            "rank": result.rank,
            "moment_id": result.moment_id,
            "lecture_id": result.lecture_id,
            "lecture_title": result.lecture_title,
            "t_start": result.t_start,
            "t_end": result.t_end,
            "timestamp": result.timestamp,
            "end_timestamp": format_timestamp(result.t_end),
            "video_link": result.video_link if result.video_link.startswith("http") else "",
            "player": player(result.lecture_id, result.t_start, own),
            "thumb": thumb,
            "own": bool(own and result.lecture_id in own),
            "badges": list(result.badges),
            "signals": result.signals,
            "snippet": result.snippet,
            "highlights": [list(span) for span in result.highlights],
            "slide_text": " ".join(result.slide_text.split())[:280],
        }

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "ready": ready.is_set(), "version": __version__}

    @app.get("/api/info")
    def info() -> dict[str, Any]:
        moments = index.moments
        counts = moments.groupby("lecture_id").agg(
            n_moments=("moment_id", "size"),
            n_slides=("image_idx", "nunique"),
            length=("t_end", "max"),
        )
        lectures = []
        for lecture_id, lecture in index.lectures.items():
            row = counts.loc[lecture_id] if lecture_id in counts.index else None
            lectures.append(
                {
                    "lecture_id": lecture_id,
                    "title": lecture.title,
                    "course": lecture.course,
                    "instructors": lecture.instructors,
                    "licence": lecture.licence,
                    "source_url": lecture.source_url
                    if lecture.source_url.startswith("http")
                    else "",
                    "duration_s": float(
                        lecture.duration_s or (row["length"] if row is not None else 0)
                    ),
                    "n_moments": int(row["n_moments"]) if row is not None else 0,
                    "n_slides": int(row["n_slides"]) if row is not None else 0,
                }
            )
        manifest = index.manifest
        return {
            "name": "Slideseek",
            "version": __version__,
            "ready": ready.is_set(),
            "n_lectures": len(index.lectures),
            "n_moments": len(index),
            "n_keyframes": len(index.image),
            "total_duration_s": float(sum(item["duration_s"] for item in lectures)),
            "models": manifest.get("models", {}),
            "built": manifest.get("created_utc"),
            "lectures": lectures,
            "examples": list(examples),
            "benchmark": benchmark,
            "rerank": rerank,
            "pictures": {
                "read_text": searcher.can_read_pictures,
                "max_bytes": MAX_PICTURE_BYTES,
                "benchmark": picture_benchmark,
            },
            # Can people add their own lectures here? The page offers it only if so.
            "own_lectures": library is not None,
        }

    def run_search(
        q: str,
        k: int = 5,
        lecture: Sequence[str] | None = None,
        *,
        rerank: bool = False,
        account: Account | None = None,
        scope: str = "all",
    ) -> dict[str, Any]:
        """Answer one query for ``account``. Raises ``BadRequest`` for a query that cannot
        be answered."""
        query = " ".join(str(q).split())
        if not query:
            raise BadRequest("The query is empty.")
        if len(query) > MAX_QUERY_CHARS:
            raise BadRequest(f"The query is longer than {MAX_QUERY_CHARS} characters.")
        if not 1 <= k <= MAX_RESULTS:
            raise BadRequest(f"Ask for between 1 and {MAX_RESULTS} moments.")

        was_ready = ready.is_set()
        with lock:
            active, own = searcher_for(account)
            check_request(k, lecture, active)
            mask = scope_mask(active, scope, lecture)
            # Timed inside the lock: the figure is the search itself, not time spent queued
            # behind another query or behind the models loading.
            started = time.perf_counter()
            encoding = active.encode(query)
            order, scores, per_ranker = active.fuse(encoding, mask=mask)
            took_ms = (time.perf_counter() - started) * 1000.0
        payload: dict[str, Any] = {"query": query, "took_ms": round(took_ms, 1), "warm": was_ready}
        if rerank and order:
            # Outside the lock: the reranker may wait for a GPU, and plain searches from
            # other visitors should not wait with it.
            started = time.perf_counter()
            if active.reranker is None and searcher.reranker is not None:
                active.reranker = searcher.reranker
            order, scores = active.rerank(query, order, scores)
            if searcher.reranker is None:
                searcher.reranker = active.reranker  # one cross-encoder for every searcher
            payload["rerank_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
        results = active.results(order, scores, encoding, per_ranker, k)
        payload["reranked"] = "rerank_ms" in payload
        payload["count"] = len(results)
        payload["results"] = [serialise(result, active, own) for result in results]
        return payload

    app.state.run_search = run_search

    def check_request(k: int, lecture: Sequence[str] | None, active: Searcher) -> None:
        if not 1 <= k <= MAX_RESULTS:
            raise BadRequest(f"Ask for between 1 and {MAX_RESULTS} moments.")
        unknown = sorted(set(lecture or []) - set(active.index.lectures))
        if unknown:
            raise BadRequest(f"Unknown lecture: {', '.join(unknown)}")

    def run_picture_search(
        data: bytes,
        k: int = 5,
        lecture: Sequence[str] | None = None,
        *,
        account: Account | None = None,
        scope: str = "all",
    ) -> dict[str, Any]:
        """Answer a picture query. Raises ``BadRequest`` for a picture that cannot be used."""
        if not 1 <= k <= MAX_RESULTS:
            raise BadRequest(f"Ask for between 1 and {MAX_RESULTS} moments.")
        if scope not in SCOPES:
            raise BadRequest(f"Search scope is one of {', '.join(SCOPES)}.")
        picture = load_picture(data)
        was_ready = ready.is_set()
        with lock:
            active, own = searcher_for(account)
            check_request(k, lecture, active)
            mask = scope_mask(active, scope, lecture)
            started = time.perf_counter()
            encoding = active.encode_picture(picture, read_text=active.can_read_pictures)
            rankers = PICTURE_RANKERS if encoding.terms else ("image",)
            order, scores, per_ranker = active.fuse(encoding, rankers, mask=mask)
            took_ms = (time.perf_counter() - started) * 1000.0
        results = active.results(order, scores, encoding, per_ranker, k)
        return {
            "kind": "picture",
            "read": " ".join(encoding.query.split())[:200],
            "took_ms": round(took_ms, 1),
            "warm": was_ready,
            "reranked": False,
            "count": len(results),
            "results": [serialise(result, active, own) for result in results],
        }

    app.state.run_picture_search = run_picture_search

    @app.post("/api/search/picture")
    async def search_picture(
        request: Request,
        k: int = Query(5, ge=1, le=MAX_RESULTS, description="How many moments to return."),
        lecture: list[str] | None = Query(None, description="Restrict to these lecture ids."),
        scope: str = Query("all", description="all, mine (your own lectures) or public."),
        account: Account = Depends(signed_in),
    ) -> dict[str, Any]:
        """The request body is the picture itself: PNG, JPEG, WebP or GIF."""
        if not request.headers.get("content-type", "").startswith("image/"):
            raise HTTPException(status_code=415, detail="Send the picture as an image file.")
        size, chunks = 0, []
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_PICTURE_BYTES:
                raise HTTPException(status_code=413, detail="The picture is larger than 8 MB.")
            chunks.append(chunk)
        try:
            return await run_in_threadpool(
                lambda: run_picture_search(
                    b"".join(chunks), k, lecture, account=account, scope=scope
                )
            )
        except BadRequest as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/search")
    def search(
        q: str = Query(..., description="The question, in plain English."),
        k: int = Query(5, ge=1, le=MAX_RESULTS, description="How many moments to return."),
        lecture: list[str] | None = Query(None, description="Restrict to these lecture ids."),
        scope: str = Query("all", description="all, mine (your own lectures) or public."),
        account: Account = Depends(signed_in),
    ) -> dict[str, Any]:
        try:
            return run_search(q, k, lecture, account=account, scope=scope)
        except BadRequest as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.exception_handler(404)
    async def not_found(request: Request, exc: Exception):
        if request.url.path.startswith(("/api/", "/gradio_api/")):
            return JSONResponse({"detail": "Not found."}, status_code=404)
        return FileResponse(STATIC_DIR / "index.html", status_code=404)

    if index.root is not None and (index.root / "thumbs").is_dir():
        app.mount("/thumbs", StaticFiles(directory=index.root / "thumbs"), name="thumbs")
    if local_videos:
        app.mount("/videos", StaticFiles(directory=Path(videos_dir)), name="videos")
    if lab_dir is not None:
        from slideseek.web.lab import add_lab_routes

        add_lab_routes(app, lab_dir)
    if library is not None:
        from slideseek.web.library import add_library_routes

        add_library_routes(app, library, signed_in)
    app.state.library = library
    add_frontend(app)
    return app


def make_library(root: str | Path, searcher: Searcher, lock: threading.Lock) -> Library:
    """A library in ``root`` that embeds with ``searcher``'s models, under ``lock``, and reads
    slides with the indexer's OCR settings."""
    from slideseek.library import Library, Models
    from slideseek.pipeline import PipelineConfig

    def models() -> Models:
        from slideseek.ocr import OcrEngine

        with lock:  # loading them must not race the warm-up, which loads them too
            return Models(searcher.text_embedder, searcher.image_embedder, OcrEngine())

    return Library(root, models, lock=lock, config=PipelineConfig(embed=searcher.embed_config))


def add_frontend(app: FastAPI) -> None:
    """Serve the static frontend: one route per file, and the page itself at ``/``.

    A catch-all mount at ``/`` would be shorter, but it would also swallow every route added
    after it, and a host such as Gradio's server adds its own once the app is launched.
    """
    files = StaticFiles(directory=STATIC_DIR)

    def serve(relative: str):
        async def asset(request: Request):
            return await files.get_response(relative, request.scope)

        return asset

    for path in sorted(STATIC_DIR.rglob("*")):
        relative = path.relative_to(STATIC_DIR)
        if path.is_file() and not any(part.startswith(".") for part in relative.parts):
            app.add_api_route(
                f"/{relative.as_posix()}",
                serve(relative.as_posix()),
                methods=["GET", "HEAD"],
                include_in_schema=False,
            )
    app.add_api_route("/", serve("index.html"), methods=["GET", "HEAD"], include_in_schema=False)
    # The page asks whether the visitor is an admin; the API is what refuses everyone else.
    app.add_api_route(
        "/admin", serve("admin.html"), methods=["GET", "HEAD"], include_in_schema=False
    )


def load_index_from_env() -> Index:
    """The index named by ``SLIDESEEK_INDEX``, downloaded first if it is not there yet."""
    index_dir = Path(os.environ.get("SLIDESEEK_INDEX", "index"))
    repo = os.environ.get("SLIDESEEK_INDEX_REPO")
    if repo and not (index_dir / "manifest.json").is_file():
        from slideseek.publish import pull_index

        return pull_index(repo, index_dir)
    return Index.load(index_dir)


def accounts_token_from_env() -> str | None:
    """The token for the accounts' dataset: ``SLIDESEEK_ACCOUNTS_TOKEN``, else ``HF_TOKEN``."""
    for name in ("SLIDESEEK_ACCOUNTS_TOKEN", "HF_TOKEN"):
        token = os.environ.get(name, "").strip()
        if token:
            return token
    return None


def accounts_from_env(default: str = DEFAULT_DATABASE_URL) -> AccountStore:
    """The account store the environment asks for.

    ``SLIDESEEK_DATABASE_URL`` names the database; without it, ``default`` does. With
    ``SLIDESEEK_ACCOUNTS_REPO`` set and no database URL, the accounts are kept in the SQLite
    file ``default`` names, with a copy in that private Hugging Face dataset
    (``slideseek.backup``), written with the token from ``accounts_token_from_env``.

    ``SLIDESEEK_ADMIN`` names the default admin's username (``AccountStore.set_default_admin``).
    """
    url = os.environ.get("SLIDESEEK_DATABASE_URL", "").strip()
    repo = os.environ.get("SLIDESEEK_ACCOUNTS_REPO", "").strip()
    if repo and url:
        log.warning("SLIDESEEK_ACCOUNTS_REPO is ignored: SLIDESEEK_DATABASE_URL is set")
    if repo and not url:
        from slideseek.backup import backed_up_accounts

        store, _ = backed_up_accounts(repo, default, token=accounts_token_from_env())
        log.info(
            "accounts are kept in %s, with a copy in the dataset %s", redacted(store.url), repo
        )
    else:
        store = AccountStore(url or default)
        log.info("accounts are kept in %s", redacted(store.url))
    # After the copy is watched, so that making the default admin is copied too.
    store.set_default_admin(os.environ.get("SLIDESEEK_ADMIN"))
    return store


def libraries_from_env(default: str | Path = "libraries"):
    """Where the lectures people add themselves are kept, and the copy kept of them.

    ``SLIDESEEK_LIBRARIES`` names the directory; ``SLIDESEEK_LIBRARIES_REPO`` a private
    dataset to keep a copy in, restored into the directory now (with ``default`` as the
    directory if none is named). Returns ``(directory, LibraryBackup)``, either of them
    ``None``. Raises ``BackupError`` if there is a copy that cannot be restored.
    """
    folder = os.environ.get("SLIDESEEK_LIBRARIES", "").strip()
    repo = os.environ.get("SLIDESEEK_LIBRARIES_REPO", "").strip()
    if not folder and not repo:
        return None, None
    folder = Path(folder or default)
    if not repo:
        return folder, None
    from slideseek.backup import LibraryBackup

    copy = LibraryBackup(repo, folder, token=accounts_token_from_env())
    copy.restore()
    log.info("people's own lectures are kept in %s, with a copy in the dataset %s", folder, repo)
    return folder, copy


def create_app_from_env() -> FastAPI:
    """App factory for ``uvicorn slideseek.web.server:create_app_from_env --factory``.

    Configuration comes from the environment, which is how a container is configured:

    ``SLIDESEEK_INDEX``       index directory (default ``index``)
    ``SLIDESEEK_INDEX_REPO``  Hugging Face dataset to download the index from if the
                                directory has none
    ``SLIDESEEK_VIDEOS``      directory of local recordings to play instead of linking out
    ``SLIDESEEK_EXAMPLES``    example queries (default ``config/examples.json``)
    ``SLIDESEEK_BENCHMARK``   benchmark metrics (default ``results/test/metrics.json``)
    ``SLIDESEEK_PICTURE_BENCHMARK``  picture search metrics
                                (default ``results/screenshot/metrics.json``)
    ``SLIDESEEK_DEVICE``      device for the query encoders (default ``cpu``)
    ``SLIDESEEK_DATABASE_URL``  where accounts are kept: a SQLAlchemy URL such as
                                ``postgresql://...`` (default ``sqlite:///accounts.db``)
    ``SLIDESEEK_ACCOUNTS_REPO``  a private Hugging Face dataset to keep a copy of the
                                accounts' SQLite file in, for a disk that does not last
    ``SLIDESEEK_ACCOUNTS_TOKEN``  a token that can write to that dataset (or ``HF_TOKEN``)
    ``SLIDESEEK_ADMIN``       the username of the default admin, an account that exists
    ``SLIDESEEK_LIBRARIES``   a directory to keep the lectures people add themselves in;
                                without it (or the next), nobody can add any
    ``SLIDESEEK_LIBRARIES_REPO``  a private Hugging Face dataset to keep a copy of them in
    """
    from slideseek.logs import configure_logging

    configure_logging()
    index = load_index_from_env()
    folder, copy = libraries_from_env()
    app = create_app(
        index,
        videos_dir=os.environ.get("SLIDESEEK_VIDEOS") or None,
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
        accounts=accounts_from_env(),
        library=folder,
    )
    if copy is not None:
        copy.watch(app.state.library)
    return app
