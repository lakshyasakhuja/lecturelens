"""The lab: a page for measuring the browser's part of adding a lecture, on this machine.

    GET  /lab                    the page (``web/lab/``)
    POST /api/lab/detect         find the slides in the frames the page sampled, as the
                                 indexer does, and return the keyframe times
    POST /api/lab/files/<path>   save what the page captured under the lab directory

``slideseek serve --lab DIR`` turns it on; nothing else does, and the Space never does. The
page samples frames, grabs stills and transcribes speech in the browser, and what it saves
lands in ``DIR/<run>/`` next to the Python pipeline's ``work/`` directories, for comparing
the two. It writes files on request, which is why it is for a server on your own machine.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from slideseek.moments import MomentConfig, plan_windows
from slideseek.slides import SlideConfig, detect_slides, load_slide_config

LAB_PAGE_DIR = Path(__file__).parent / "lab"
FRAME_WIDTH, FRAME_HEIGHT = SlideConfig().size
_PART = re.compile(r"[a-z0-9][a-z0-9._-]{0,80}")
MASKS = Path("config/masks.yaml")


def lab_path(root: Path, relative: str) -> Path:
    """Where a saved file goes: under ``root``, by plain names only."""
    parts = relative.split("/")
    if not 1 <= len(parts) <= 3 or not all(_PART.fullmatch(part) for part in parts):
        raise HTTPException(status_code=400, detail="Name files with a-z, 0-9, '.', '_' and '-'.")
    return root.joinpath(*parts)


def detect(frames: bytes, count: int, lecture: str | None = None) -> dict:
    """The slide segments and keyframe times for ``count`` frames of 160x90 greyscale."""
    size = FRAME_WIDTH * FRAME_HEIGHT
    if count < 1 or len(frames) != count * size:
        raise HTTPException(
            status_code=400, detail=f"Expected {count} frames of {size} bytes each."
        )
    stack = np.frombuffer(frames, dtype=np.uint8).reshape(count, FRAME_HEIGHT, FRAME_WIDTH)
    config = SlideConfig()
    if lecture and MASKS.is_file():
        config = load_slide_config(MASKS, lecture, config)
    detection = detect_slides(stack, config)
    windows = plan_windows(detection.segments, MomentConfig())
    return {
        "segments": [segment.to_dict() for segment in detection.segments],
        "keyframe_times": sorted({window.keyframe_t for window in windows}),
        "t_high": detection.t_high,
        "t_low": detection.t_low,
    }


def add_lab_routes(app: FastAPI, root: str | Path) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    def page(name: str):
        async def serve():
            return FileResponse(LAB_PAGE_DIR / name)

        return serve

    app.add_api_route("/lab", page("lab.html"), methods=["GET"], include_in_schema=False)
    for name in ("lab.js", "lab.css"):
        app.add_api_route(f"/lab/{name}", page(name), methods=["GET"], include_in_schema=False)

    @app.post("/api/lab/detect", include_in_schema=False)
    async def lab_detect(
        request: Request,
        count: int = Query(..., ge=1),
        lecture: str | None = Query(None),
    ) -> dict:
        return await run_in_threadpool(detect, await request.body(), count, lecture)

    @app.post("/api/lab/files/{relative:path}", include_in_schema=False)
    async def lab_save(relative: str, request: Request) -> dict:
        path = lab_path(root, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = await request.body()
        path.write_bytes(data)
        return {"saved": relative, "bytes": len(data)}
