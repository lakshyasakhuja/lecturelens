"""Adding your own lectures over HTTP (``slideseek.library``).

    GET    /api/library                                  your lectures, newest first, and the limits
    POST   /api/library/lectures                         a new lecture: {title, duration_s, file}
    GET    /api/library/lectures/{id}                    one lecture, with what has arrived so far
    PUT    /api/library/lectures/{id}/frames             the 160x90 grey frames, gzipped
    PUT    /api/library/lectures/{id}/stills/{ms}.jpg    the still at that time
    PUT    /api/library/lectures/{id}/transcript         the words: {model, segments}
    PATCH  /api/library/lectures/{id}                    rename it: {title}
    DELETE /api/library/lectures/{id}                    delete it
    GET    /api/library/thumbs/{id}/{ms}.jpg             a slide thumbnail of one of yours

Every route needs somebody signed in, and sees only that account's lectures: another
account's lecture is answered as if it did not exist. Nothing here is cached by anything
but the visitor's own browser.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict

from slideseek import library as lib
from slideseek.accounts import Account
from slideseek.library import Library, LibraryError, owner_key


class NewLecture(BaseModel):
    model_config = ConfigDict(extra="ignore", str_max_length=1000)
    title: str = ""
    duration_s: float = 0.0
    file: dict[str, Any] = {}


class Rename(BaseModel):
    model_config = ConfigDict(extra="ignore", str_max_length=1000)
    title: str = ""


def limits() -> dict[str, Any]:
    """What the page tells the visitor before they start."""
    return {
        "lectures": lib.MAX_LECTURES,
        "max_duration_s": lib.MAX_DURATION_S,
        "min_duration_s": lib.MIN_DURATION_S,
        "title_chars": lib.MAX_TITLE_CHARS,
    }


async def read_body(request: Request, limit: int, kinds: tuple[str, ...]) -> bytes:
    """The request body, if it is one of ``kinds`` and no larger than ``limit`` bytes."""
    kind = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if kind not in kinds:
        raise HTTPException(status_code=415, detail=f"Send this as {kinds[0]}.")
    size, chunks = 0, []
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise HTTPException(status_code=413, detail="That is larger than this app accepts.")
        chunks.append(chunk)
    return b"".join(chunks)


def private(body: Any, status: int = 200) -> JSONResponse:
    return JSONResponse(body, status, headers={"Cache-Control": "no-store"})


def add_library_routes(
    app: FastAPI, library: Library, signed_in: Callable[[Request], Account]
) -> None:
    """Add the library routes to ``app``. ``signed_in`` answers 401 to nobody signed in."""

    def owner(account: Account = Depends(signed_in)) -> str:
        return owner_key(account)

    async def call(function, *args) -> Any:
        try:
            return await run_in_threadpool(function, *args)
        except LibraryError as error:
            raise HTTPException(status_code=error.status, detail=str(error)) from error

    @app.get("/api/library")
    async def lectures(who: str = Depends(owner)) -> Response:
        found = await call(library.lectures, who)
        return private({"lectures": found, "limits": limits()})

    @app.post("/api/library/lectures")
    async def create(form: NewLecture, who: str = Depends(owner)) -> Response:
        def make():
            return library.create(who, title=form.title, duration_s=form.duration_s, file=form.file)

        return private({"lecture": await call(make)}, 201)

    @app.get("/api/library/lectures/{lecture_id}")
    async def one(lecture_id: str, who: str = Depends(owner)) -> Response:
        return private({"lecture": await call(library.progress, who, lecture_id)})

    @app.put("/api/library/lectures/{lecture_id}/frames")
    async def frames(lecture_id: str, request: Request, who: str = Depends(owner)) -> Response:
        await call(library.get, who, lecture_id)  # 404 before reading a large body
        data = await read_body(
            request, lib.MAX_FRAMES_UPLOAD_BYTES, ("application/gzip", "application/octet-stream")
        )
        return private({"lecture": await call(library.put_frames, who, lecture_id, data)})

    @app.put("/api/library/lectures/{lecture_id}/stills/{name}")
    async def still(
        lecture_id: str, name: str, request: Request, who: str = Depends(owner)
    ) -> Response:
        stem, dot, extension = name.partition(".")
        if not (stem.isdigit() and len(stem) <= 9 and dot and extension == "jpg"):
            raise HTTPException(status_code=404, detail="Not found.")
        data = await read_body(request, lib.MAX_STILL_BYTES, ("image/jpeg",))
        return private({"lecture": await call(library.put_still, who, lecture_id, int(stem), data)})

    @app.put("/api/library/lectures/{lecture_id}/transcript")
    async def transcript(lecture_id: str, request: Request, who: str = Depends(owner)) -> Response:
        import json

        await call(library.get, who, lecture_id)
        data = await read_body(request, lib.MAX_TRANSCRIPT_BYTES, ("application/json",))
        try:
            payload = json.loads(data)
        except ValueError as error:
            raise HTTPException(status_code=400, detail="The transcript is not JSON.") from error
        return private({"lecture": await call(library.put_transcript, who, lecture_id, payload)})

    @app.patch("/api/library/lectures/{lecture_id}")
    async def rename(lecture_id: str, form: Rename, who: str = Depends(owner)) -> Response:
        return private({"lecture": await call(library.rename, who, lecture_id, form.title)})

    @app.delete("/api/library/lectures/{lecture_id}")
    async def delete(lecture_id: str, who: str = Depends(owner)) -> Response:
        await call(library.delete, who, lecture_id)
        return Response(status_code=204)

    @app.get("/api/library/thumbs/{lecture_id}/{name}")
    async def thumb(lecture_id: str, name: str, who: str = Depends(owner)) -> Response:
        path = library.thumb(who, lecture_id, name)
        if path is None:
            raise HTTPException(status_code=404, detail="Not found.")
        # Only this browser may keep it: the slide is somebody's own.
        return FileResponse(path, headers={"Cache-Control": "private, max-age=86400"})
