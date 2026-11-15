"""The web server: a small JSON API over the searcher, plus the static frontend.

    GET /api/health                 is the server up, and are the encoders loaded?
    GET /api/info                   what the index contains: lectures, counts, models, examples
    GET /api/search?q=...           the top moments for a query
    GET /thumbs/<lecture>/<ms>.jpg  slide thumbnails, straight from the index directory
    GET /                           the frontend (plain HTML, CSS and JavaScript, no build step)

The server holds no state beyond the loaded index. It hosts no video: a result either links
out to where the lecture already lives (a YouTube id parsed from the lecture's link
template) or, when run locally with a directory of recordings, plays the local file.

Search runs on a CPU. One query at a time goes through the encoders, guarded by a lock: a
query takes about a tenth of a second, and serialising them is simpler and safer than
letting several threads share the models.
"""

from __future__ import annotations

import os
import re
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from lecturelens import __version__
from lecturelens.catalog import VIDEO_EXTENSIONS, Lecture
from lecturelens.index import Index
from lecturelens.search import Result, Searcher
from lecturelens.utils import format_timestamp, read_json

STATIC_DIR = Path(__file__).parent / "static"
MAX_QUERY_CHARS = 300
MAX_RESULTS = 20

_YOUTUBE_ID = re.compile(r"(?:youtube\.com/watch\?(?:.*&)?v=|youtu\.be/)([A-Za-z0-9_-]{6,})")

# What the page may load. Everything is first-party except the video player and its poster.
CONTENT_SECURITY_POLICY = "; ".join(
    [
        "default-src 'self'",
        "script-src 'self'",
        "style-src 'self'",
        "img-src 'self' data: https://i.ytimg.com",
        "media-src 'self'",
        "frame-src https://www.youtube-nocookie.com",
        "base-uri 'none'",
        "form-action 'self'",
        # Hugging Face shows a Space inside a frame on its own pages.
        "frame-ancestors 'self' https://huggingface.co https://*.hf.space",
    ]
)


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


def load_examples(path: str | Path | None) -> list[dict[str, str]]:
    """Example queries for the search box: ``[{"query": ..., "type": ...}, ...]``."""
    if not path or not Path(path).is_file():
        return []
    examples = []
    for item in read_json(path):
        query = str(item.get("query", "")).strip()
        if query:
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


def create_app(
    index: Index,
    searcher: Searcher | None = None,
    *,
    videos_dir: str | Path | None = None,
    examples: Sequence[Mapping[str, str]] = (),
    benchmark: Mapping[str, Any] | None = None,
    device: str = "cpu",
    warm_up: bool = True,
) -> FastAPI:
    """Build the application around a loaded index."""
    searcher = searcher or Searcher(index, device=device)
    local_videos = find_local_videos(videos_dir, index.lectures)
    lock = threading.Lock()
    ready = threading.Event()

    def _warm() -> None:
        with lock:
            try:
                searcher.warm_up()
            finally:
                ready.set()

    if warm_up:
        threading.Thread(target=_warm, name="warm-up", daemon=True).start()
    else:
        ready.set()

    app = FastAPI(title="LectureLens", version=__version__, docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("Content-Security-Policy", CONTENT_SECURITY_POLICY)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        if request.url.path.startswith("/thumbs/"):
            response.headers.setdefault("Cache-Control", "public, max-age=86400")
        return response

    def player(lecture_id: str, start: float) -> dict[str, Any] | None:
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

    def serialise(result: Result) -> dict[str, Any]:
        relative = str(index.moments["thumb"].iat[result.row])
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
            "player": player(result.lecture_id, result.t_start),
            "thumb": f"/{relative}" if result.thumb is not None else None,
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
            "name": "LectureLens",
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
        }

    @app.get("/api/search")
    def search(
        q: str = Query(..., description="The question, in plain English."),
        k: int = Query(5, ge=1, le=MAX_RESULTS, description="How many moments to return."),
        lecture: list[str] | None = Query(None, description="Restrict to these lecture ids."),
    ) -> dict[str, Any]:
        query = " ".join(q.split())
        if not query:
            raise HTTPException(status_code=400, detail="The query is empty.")
        if len(query) > MAX_QUERY_CHARS:
            raise HTTPException(
                status_code=400, detail=f"The query is longer than {MAX_QUERY_CHARS} characters."
            )
        unknown = sorted(set(lecture or []) - set(index.lectures))
        if unknown:
            raise HTTPException(status_code=400, detail=f"Unknown lecture: {', '.join(unknown)}")

        started = time.perf_counter()
        with lock:
            results = searcher.search(query, top_k=k, lectures=lecture or None)
        took_ms = (time.perf_counter() - started) * 1000.0
        return {
            "query": query,
            "took_ms": round(took_ms, 1),
            "count": len(results),
            "results": [serialise(result) for result in results],
        }

    @app.exception_handler(404)
    async def not_found(request: Request, exc: Exception):
        if request.url.path.startswith("/api/"):
            return JSONResponse({"detail": "Not found."}, status_code=404)
        return FileResponse(STATIC_DIR / "index.html", status_code=404)

    if index.root is not None and (index.root / "thumbs").is_dir():
        app.mount("/thumbs", StaticFiles(directory=index.root / "thumbs"), name="thumbs")
    if local_videos:
        app.mount("/videos", StaticFiles(directory=Path(videos_dir)), name="videos")
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="frontend")
    return app


def create_app_from_env() -> FastAPI:
    """App factory for ``uvicorn lecturelens.web.server:create_app_from_env --factory``.

    Configuration comes from the environment, which is how a container is configured:

    ``LECTURELENS_INDEX``       index directory (default ``index``)
    ``LECTURELENS_INDEX_REPO``  Hugging Face dataset to download the index from if the
                                directory has none
    ``LECTURELENS_VIDEOS``      directory of local recordings to play instead of linking out
    ``LECTURELENS_EXAMPLES``    example queries (default ``config/examples.json``)
    ``LECTURELENS_BENCHMARK``   benchmark metrics (default ``results/test/metrics.json``)
    ``LECTURELENS_DEVICE``      device for the query encoders (default ``cpu``)
    """
    from lecturelens.logs import configure_logging

    configure_logging()
    index_dir = Path(os.environ.get("LECTURELENS_INDEX", "index"))
    repo = os.environ.get("LECTURELENS_INDEX_REPO")
    if repo and not (index_dir / "manifest.json").is_file():
        from lecturelens.publish import pull_index

        index = pull_index(repo, index_dir)
    else:
        index = Index.load(index_dir)
    return create_app(
        index,
        videos_dir=os.environ.get("LECTURELENS_VIDEOS") or None,
        examples=load_examples(os.environ.get("LECTURELENS_EXAMPLES", "config/examples.json")),
        benchmark=load_benchmark(
            os.environ.get("LECTURELENS_BENCHMARK", "results/test/metrics.json")
        ),
        device=os.environ.get("LECTURELENS_DEVICE", "cpu"),
    )
