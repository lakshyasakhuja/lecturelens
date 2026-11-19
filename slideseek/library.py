"""Private libraries: lectures people add themselves, from the browser.

The browser does the slow part on the visitor's own computer (``web/static/capture/``): it
samples small greyscale frames, grabs a still of each slide and transcribes the speech. It
sends those, and the video never leaves the computer. The server does the rest with the
indexing stages in ``pipeline``, in this order::

    frames      slide segments and windows, at once (seconds); the keyframe times go back
    stills      one per keyframe time; when the last arrives, OCR is queued
    transcript  the words; once OCR is done too, moments and embeddings are queued
    ready       the lecture has an index of its own, searched with the public one

On disk, one directory per account and one per lecture::

    libraries/<owner>/<lecture_id>/
      lecture.json   title, the file it came from, its length, what has arrived and been done
      work/          the pipeline's stage files: segments.json, keyframes/, ocr.json, ...
      index/         the lecture's own index, once it is ready (the stills are then deleted)

``<owner>`` is the account's id and the time it was made (``owner_key``), so an account
that is later given a reused id never sees an earlier account's lectures.

**One worker for the whole site.** OCR and the embeddings run on one background thread,
with the models the searcher already has. It uses them in small batches, each under the
search lock, so a search never waits long behind somebody's lecture.

Only the account that added a lecture can list it, search it or see its slides. An admin
sees how many lectures each account has and how much space they take (``usage``), never
what is in them.
"""

from __future__ import annotations

import io
import logging
import math
import re
import secrets
import shutil
import threading
import time
import zlib
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from slideseek import asr
from slideseek.catalog import Lecture
from slideseek.index import Index
from slideseek.moments import plan_windows
from slideseek.pipeline import (
    PipelineConfig,
    assemble,
    embed_moments,
    find_slides,
    keyframe_times,
    load_segments,
    moments_table,
    read_slides,
    slide_texts,
    still_name,
)
from slideseek.utils import read_json, write_json

log = logging.getLogger(__name__)

# ---- limits ------------------------------------------------------------------------------

MAX_LECTURES = 20  # per account
MIN_DURATION_S = 10
MAX_DURATION_S = 4 * 3600
MAX_TITLE_CHARS = 120
MAX_FILE_NAME_CHARS = 255
# The frames as sent, compressed with gzip. Slides compress well: 84 minutes of 160x90 grey
# frames are 72 MB raw and 14 MB compressed (docs/browser-indexing.md).
MAX_FRAMES_UPLOAD_BYTES = 96 * 1024 * 1024
MAX_STILL_BYTES = 3 * 1024 * 1024
MAX_STILL_PIXELS = 1920 * 1920
MAX_TRANSCRIPT_BYTES = 16 * 1024 * 1024
MAX_WORDS_PER_SECOND = 8  # nobody speaks faster than this; more is not a transcript
MAX_WORD_CHARS = 64
MAX_IN_PROGRESS = 12  # lectures being added at once, across the whole site
ABANDONED_AFTER_S = 2 * 86400  # a lecture still unfinished after this long is removed
EMBED_BATCH = 8  # moments embedded per turn of the search lock

FRAME_WIDTH, FRAME_HEIGHT = 160, 90
FRAME_BYTES = FRAME_WIDTH * FRAME_HEIGHT
LECTURE_ID = re.compile(r"own-[0-9a-f]{12}")
OWNER = re.compile(r"[0-9]+-[0-9]+")
STATE_FILE = "lecture.json"

# What ``Library.on_change`` is told.
LECTURE_READY = "lecture ready"
LECTURE_CHANGED = "lecture changed"  # renamed
LECTURE_DELETED = "lecture deleted"
OWNER_DELETED = "owner deleted"

# Where a lecture is, as the page is told: what it should send next, or that it is done.
#   frames    made, waiting for the frames
#   stills    waiting for stills
#   speech    waiting for the transcript
#   building  everything has arrived; the server is reading the slides and embedding
#   ready     searchable
#   failed    something went wrong; ``error`` says what
BROWSER_STAGES = ("frames", "stills", "speech")


class LibraryError(ValueError):
    """A request the library refuses. ``status`` is the HTTP status that says why."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class NoSuchLecture(LibraryError):
    def __init__(self) -> None:
        super().__init__("There is no such lecture in your library.", status=404)


def owner_key(account: Any) -> str:
    """Whose library: the account's id and the time it was made."""
    return f"{int(account.id)}-{int(account.created_at)}"


def expected_frames(duration_s: float) -> int:
    """How many frames, one a second, the browser samples from a recording this long."""
    return max(1, math.floor(duration_s))


# ---- the models, shared with the searcher -----------------------------------------------------


@dataclass
class Models:
    """What the worker uses: the searcher's two encoders, and an OCR engine for slides."""

    text_embedder: Any  # encode_documents(texts)
    image_embedder: Any  # encode_images(paths) and .calibration
    ocr: Any  # read(path) -> ocr.SlideText


class _Locked:
    """A model used a few items at a time, each batch under ``lock``."""

    def __init__(self, model: Any, lock: threading.Lock, batch: int = EMBED_BATCH) -> None:
        self.model, self.lock, self.batch = model, lock, batch

    @property
    def calibration(self):
        return self.model.calibration

    def _batched(self, method: str, items: Sequence) -> np.ndarray:
        call = getattr(self.model, method)
        if not len(items):
            with self.lock:
                return call(list(items))
        parts = []
        for start in range(0, len(items), self.batch):
            with self.lock:
                parts.append(call(list(items[start : start + self.batch])))
        return np.concatenate(parts)

    def encode_documents(self, texts: Sequence[str], progress: bool = False) -> np.ndarray:
        return self._batched("encode_documents", texts)

    def encode_images(self, paths: Sequence[Path]) -> np.ndarray:
        return self._batched("encode_images", paths)

    def read(self, path: Path):
        with self.lock:
            return self.model.read(path)


# ---- checking what the browser sends ------------------------------------------------------------


def clean_title(title: Any) -> str:
    text = " ".join(str(title or "").split())
    text = "".join(ch for ch in text if ch.isprintable())
    if not text:
        raise LibraryError("Give the lecture a title.")
    if len(text) > MAX_TITLE_CHARS:
        raise LibraryError(f"Keep the title under {MAX_TITLE_CHARS} characters.")
    return text


def clean_file(file: Any) -> dict[str, Any]:
    """The name and size of the recording on the visitor's computer, to find it again."""
    if not isinstance(file, Mapping):
        raise LibraryError("Say which file the lecture comes from.")
    name = "".join(ch for ch in str(file.get("name", "")).strip() if ch.isprintable())
    size = file.get("size")
    if not name or len(name) > MAX_FILE_NAME_CHARS:
        raise LibraryError("The file name is missing or too long.")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise LibraryError("The file size is missing.")
    kind = str(file.get("type", ""))[:100]
    return {"name": name, "size": size, "type": kind}


def clean_duration(duration: Any) -> float:
    try:
        seconds = float(duration)
    except (TypeError, ValueError):
        seconds = float("nan")
    if not math.isfinite(seconds) or seconds < MIN_DURATION_S:
        raise LibraryError(f"A lecture has to be at least {MIN_DURATION_S} seconds long.")
    if seconds > MAX_DURATION_S:
        raise LibraryError(
            f"Lectures can be up to {MAX_DURATION_S // 3600} hours long; this one is longer.",
            status=413,
        )
    return round(seconds, 3)


def gunzip(data: bytes, limit: int) -> bytes:
    """``data`` decompressed, refusing anything that grows past ``limit`` bytes."""
    inflater = zlib.decompressobj(wbits=31)  # gzip
    try:
        out = inflater.decompress(data, limit + 1)
        if len(out) > limit or inflater.unconsumed_tail:
            raise LibraryError("The frames are larger than this lecture's length allows.", 413)
        out += inflater.flush()
    except zlib.error as error:
        raise LibraryError("The frames are not compressed with gzip.") from error
    if not inflater.eof:
        raise LibraryError("The frames were cut short.")
    return out


def check_still(data: bytes) -> tuple[int, int]:
    """The size of a JPEG still, if it is one this library can use."""
    from PIL import Image, UnidentifiedImageError

    if not data:
        raise LibraryError("The still is empty.")
    if len(data) > MAX_STILL_BYTES:
        raise LibraryError("The still is larger than 3 MB.", status=413)
    try:
        with Image.open(io.BytesIO(data)) as probe:
            if probe.format != "JPEG":
                raise LibraryError("Send each still as a JPEG.")
            width, height = probe.size
            if width * height > MAX_STILL_PIXELS:
                raise LibraryError("The still has too many pixels.", status=413)
            probe.verify()
        with Image.open(io.BytesIO(data)) as opened:
            opened.load()
    except LibraryError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as error:
        raise LibraryError("That still is not a picture this app can read.") from error
    return width, height


def _seconds(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LibraryError("A time in the transcript is not a number.")
    seconds = float(value)
    if not math.isfinite(seconds):
        raise LibraryError("A time in the transcript is not a number.")
    return seconds


def clean_transcript(payload: Any, duration_s: float) -> tuple[list[asr.AsrSegment], str]:
    """The transcript's segments, checked and in time order, and the model that made it."""
    if not isinstance(payload, Mapping) or not isinstance(payload.get("segments"), list):
        raise LibraryError("Send the transcript as {model, segments}.")
    model = " ".join(str(payload.get("model", "")).split())[:120] or "unknown"
    limit = duration_s + 2.0
    budget = max(100, int(duration_s * MAX_WORDS_PER_SECOND))
    segments, count = [], 0
    for raw in payload["segments"]:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("words"), list):
            raise LibraryError("Each part of the transcript needs its words.")
        words = []
        for item in raw["words"]:
            if not isinstance(item, Mapping):
                raise LibraryError("Each word needs w, s and e.")
            text = " ".join(str(item.get("w", "")).split())
            start, end = _seconds(item.get("s")), _seconds(item.get("e"))
            prob = item.get("p", 1.0)
            prob = float(prob) if isinstance(prob, (int, float)) and math.isfinite(prob) else 1.0
            if not text or len(text) > MAX_WORD_CHARS:
                raise LibraryError("A word in the transcript is empty or too long.")
            if not 0 <= start <= end <= limit:
                raise LibraryError("A word in the transcript is outside the lecture's length.")
            words.append(asr.Word(text, start, end, min(max(prob, 0.0), 1.0)))
        count += len(words)
        if count > budget:
            raise LibraryError("The transcript has more words than the lecture's length allows.")
        if not words:
            continue
        words.sort(key=lambda w: (w.start, w.end))
        segments.append(
            asr.AsrSegment(
                words[0].start, words[-1].end, " ".join(w.text for w in words), tuple(words)
            )
        )
    segments.sort(key=lambda segment: (segment.start, segment.end))
    return segments, model


# ---- the library ---------------------------------------------------------------------------------


class Library:
    """Every account's own lectures, under ``root``.

    ``models`` is called the first time the worker needs the models; ``lock`` is the lock
    every use of them is made under (the searcher's). ``config`` must name the embedding
    models the public index was built with, so the two can be searched together.

    ``on_change``, if set, is called with ``(change, owner, lecture_id)`` after a lecture
    is ready, renamed or deleted, and ``(OWNER_DELETED, owner, None)`` when an account's
    whole library is.

    With ``worker=False`` nothing runs in the background; ``run_pending`` does the queued
    work in the calling thread (for tests).
    """

    def __init__(
        self,
        root: str | Path,
        models: Callable[[], Models],
        *,
        lock: threading.Lock | None = None,
        config: PipelineConfig | None = None,
        worker: bool = True,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = config or PipelineConfig()
        self.model_lock = lock or threading.Lock()
        self.on_change: Callable[[str, str, str | None], None] | None = None
        self._make_models = models
        self._models: Models | None = None
        self._clock = clock
        self._files = threading.RLock()  # every read and write of a library's files
        self._queue: OrderedDict[tuple[str, str], None] = OrderedDict()
        self._wake = threading.Condition()
        self._stopping = False
        self._versions: dict[str, int] = {}
        self._indexes: dict[str, tuple[int, Index | None]] = {}
        self._thread: threading.Thread | None = None
        self.recover()
        if worker:
            self._thread = threading.Thread(target=self._work, name="library", daemon=True)
            self._thread.start()

    def __repr__(self) -> str:
        return f"Library({str(self.root)!r})"

    # ---- where things are ----------------------------------------------------------------

    def _owner_dir(self, owner: str) -> Path:
        if not OWNER.fullmatch(owner):
            raise NoSuchLecture()
        return self.root / owner

    def _dir(self, owner: str, lecture_id: str) -> Path:
        if not LECTURE_ID.fullmatch(str(lecture_id)):
            raise NoSuchLecture()
        return self._owner_dir(owner) / lecture_id

    def _state(self, owner: str, lecture_id: str) -> dict[str, Any]:
        path = self._dir(owner, lecture_id) / STATE_FILE
        if not path.is_file():
            raise NoSuchLecture()
        return read_json(path)

    def _save(self, owner: str, state: dict[str, Any]) -> None:
        folder = self._dir(owner, state["lecture_id"])
        if folder.is_dir():  # not if it was deleted meanwhile
            state["updated"] = int(self._clock())
            write_json(folder / STATE_FILE, state)

    def _states(self, owner: str) -> list[dict[str, Any]]:
        folder = self._owner_dir(owner)
        if not folder.is_dir():
            return []
        found = []
        for path in sorted(folder.glob(f"own-*/{STATE_FILE}")):
            try:
                found.append(read_json(path))
            except (OSError, ValueError):
                log.warning("unreadable lecture state: %s", path)
        return found

    def owners(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and OWNER.fullmatch(p.name))

    @staticmethod
    def stage(state: Mapping[str, Any]) -> str:
        if state.get("error"):
            return "failed"
        if state.get("ready"):
            return "ready"
        if not state.get("frames"):
            return "frames"
        if len(state.get("stills", [])) < len(state.get("keyframe_times", [])):
            return "stills"
        if not state.get("transcript"):
            return "speech"
        return "building"

    def describe(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """A lecture as its owner is shown it."""
        return {
            "lecture_id": state["lecture_id"],
            "title": state["title"],
            "file": {"name": state["file"]["name"], "size": state["file"]["size"]},
            "duration_s": state["duration_s"],
            "created": state["created"],
            "stage": self.stage(state),
            "stills": [len(state.get("stills", [])), len(state.get("keyframe_times", []))],
            "error": state.get("error"),
            "n_moments": state.get("n_moments", 0),
            "n_slides": state.get("n_slides", 0),
            "bytes": state.get("bytes", 0),
        }

    @staticmethod
    def lecture(state: Mapping[str, Any]) -> Lecture:
        """The catalogue entry search results are labelled with."""
        return Lecture(
            state["lecture_id"],
            state["title"],
            course="Your lecture",
            duration_s=float(state["duration_s"]),
        )

    # ---- reading ---------------------------------------------------------------------------

    def lectures(self, owner: str) -> list[dict[str, Any]]:
        """The account's lectures, newest first."""
        with self._files:
            states = self._states(owner)
        states.sort(key=lambda s: (s["created"], s["lecture_id"]), reverse=True)
        return [self.describe(state) for state in states]

    def get(self, owner: str, lecture_id: str) -> dict[str, Any]:
        with self._files:
            return self.describe(self._state(owner, lecture_id))

    def file_of(self, owner: str, lecture_id: str) -> dict[str, Any]:
        """The name and size of the recording a lecture came from."""
        with self._files:
            state = self._state(owner, lecture_id)
        return {"name": state["file"]["name"], "size": state["file"]["size"]}

    def version(self, owner: str) -> int:
        """Goes up whenever what the account can search changes."""
        return self._versions.get(owner, 0)

    def _changed(self, owner: str) -> None:
        self._versions[owner] = self.version(owner) + 1

    def index(self, owner: str) -> Index | None:
        """Every ready lecture of the account as one index, or ``None`` if it has none."""
        version = self.version(owner)
        cached = self._indexes.get(owner)
        if cached is not None and cached[0] == version:
            return cached[1]
        with self._files:
            parts = []
            for state in self._states(owner):
                if not state.get("ready") or state.get("error"):
                    continue
                folder = self._dir(owner, state["lecture_id"]) / "index"
                try:
                    part = Index.load(folder)
                except Exception:
                    log.exception("the index of %s/%s cannot be read", owner, state["lecture_id"])
                    continue
                part.lectures = {state["lecture_id"]: self.lecture(state)}
                parts.append(part)
        index = Index.combine(*parts) if parts else None
        self._indexes[owner] = (version, index)
        return index

    def thumb(self, owner: str, lecture_id: str, name: str) -> Path | None:
        """One of the account's own slide thumbnails, if there is such a file."""
        if not re.fullmatch(r"[0-9]{9}\.jpg", name):
            return None
        try:
            path = self._dir(owner, lecture_id) / "index" / "thumbs" / lecture_id / name
        except NoSuchLecture:
            return None
        return path if path.is_file() else None

    def usage(self) -> dict[str, dict[str, int]]:
        """``{owner: {lectures, bytes}}``: how much each account keeps here, for admins."""
        found = {}
        with self._files:
            for owner in self.owners():
                states = self._states(owner)
                found[owner] = {
                    "lectures": len(states),
                    "bytes": sum(_size(self._dir(owner, s["lecture_id"])) for s in states),
                }
        return found

    # ---- adding a lecture -------------------------------------------------------------------

    def create(self, owner: str, *, title: Any, duration_s: Any, file: Any) -> dict[str, Any]:
        """Make a lecture, waiting for its frames. Returns how its owner is shown it."""
        title, file, duration = clean_title(title), clean_file(file), clean_duration(duration_s)
        self._owner_dir(owner)
        with self._files:
            self.expire()
            mine = self._states(owner)
            if len(mine) >= MAX_LECTURES:
                raise LibraryError(
                    f"Your library holds {MAX_LECTURES} lectures. Delete one to add another.",
                    status=409,
                )
            unfinished = [s for s in mine if self.stage(s) in BROWSER_STAGES]
            if unfinished:
                raise LibraryError(
                    f"Finish adding “{unfinished[0]['title']}” first, or delete it.", status=409
                )
            busy = sum(
                1
                for other in self.owners()
                for s in self._states(other)
                if self.stage(s) not in ("ready", "failed")
            )
            if busy >= MAX_IN_PROGRESS:
                raise LibraryError(
                    "Too many lectures are being added right now. Try again in a few minutes.",
                    status=503,
                )
            lecture_id = f"own-{secrets.token_hex(6)}"
            now = int(self._clock())
            state = {
                "lecture_id": lecture_id,
                "title": title,
                "file": file,
                "duration_s": duration,
                "created": now,
                "updated": now,
                "frames": False,
                "keyframe_times": [],
                "stills": [],
                "ocr": False,
                "transcript": False,
                "asr_model": None,
                "ready": False,
                "error": None,
            }
            (self._dir(owner, lecture_id) / "work").mkdir(parents=True)
            self._save(owner, state)
        log.info("library %s: added %s (%.0f s)", owner, lecture_id, duration)
        return self.describe(state)

    def put_frames(self, owner: str, lecture_id: str, data: bytes) -> dict[str, Any]:
        """The 160x90 grey frames, one a second, compressed with gzip. Finds the slides and
        returns the times a still is wanted at (``keyframe_times``)."""
        with self._files:
            state = self._state(owner, lecture_id)
        if state["frames"]:  # sent again, after a reload: the answer has not changed
            return {**self.describe(state), "keyframe_times": state["keyframe_times"]}
        if self.stage(state) == "failed":
            raise LibraryError("This lecture could not be added. Delete it and try again.", 409)
        if len(data) > MAX_FRAMES_UPLOAD_BYTES:
            raise LibraryError("The frames are larger than 96 MB.", status=413)
        count = expected_frames(state["duration_s"])
        raw = gunzip(data, count * FRAME_BYTES)
        if len(raw) != count * FRAME_BYTES:
            raise LibraryError(f"Expected {count} frames of {FRAME_WIDTH}x{FRAME_HEIGHT}.")
        frames = np.frombuffer(raw, dtype=np.uint8).reshape(count, FRAME_HEIGHT, FRAME_WIDTH)
        found = find_slides(
            frames, self.config.slides, lecture_id=lecture_id, duration_s=state["duration_s"]
        )
        with self._files:
            state = self._state(owner, lecture_id)
            work = self._dir(owner, lecture_id) / "work"
            write_json(work / "segments.json", found)
            windows = plan_windows(load_segments(work / "segments.json"), self.config.moments)
            state["frames"] = True
            state["keyframe_times"] = keyframe_times(windows)
            state["stills"] = []
            self._save(owner, state)
        return {**self.describe(state), "keyframe_times": state["keyframe_times"]}

    def put_still(self, owner: str, lecture_id: str, ms: int, data: bytes) -> dict[str, Any]:
        """The still at ``ms`` milliseconds, one of the keyframe times, as a JPEG."""
        check_still(data)
        with self._files:
            state = self._state(owner, lecture_id)
            wanted = {round(t * 1000): t for t in state["keyframe_times"]}
            if ms not in wanted:
                raise LibraryError("No still is wanted at that time.", status=404)
            if state["ocr"] or self.stage(state) in ("ready", "failed"):
                return self.describe(state)  # already read; a late copy changes nothing
            folder = self._dir(owner, lecture_id) / "work" / "keyframes"
            folder.mkdir(parents=True, exist_ok=True)
            partial = folder / f"{ms:09d}.partial"
            partial.write_bytes(data)
            partial.replace(folder / still_name(wanted[ms]))
            state["stills"] = sorted({*state["stills"], ms})
            self._save(owner, state)
        if self.stage(state) not in BROWSER_STAGES[:2]:
            self._enqueue(owner, lecture_id)
        return self.describe(state)

    def put_transcript(self, owner: str, lecture_id: str, payload: Any) -> dict[str, Any]:
        """The words, with their times: ``{model, segments: [{words: [{w, s, e, p}]}]}``."""
        with self._files:
            state = self._state(owner, lecture_id)
        segments, model = clean_transcript(payload, float(state["duration_s"]))
        with self._files:
            state = self._state(owner, lecture_id)
            if self.stage(state) in ("ready", "failed"):
                return self.describe(state)
            asr.save_segments(self._dir(owner, lecture_id) / "work" / "asr.jsonl", segments)
            state["transcript"] = True
            state["asr_model"] = model
            self._save(owner, state)
        self._enqueue(owner, lecture_id)
        return self.describe(state)

    # ---- changing and removing ----------------------------------------------------------------

    def rename(self, owner: str, lecture_id: str, title: Any) -> dict[str, Any]:
        title = clean_title(title)
        with self._files:
            state = self._state(owner, lecture_id)
            state["title"] = title
            self._save(owner, state)
            if state.get("ready"):
                self._changed(owner)
        if state.get("ready"):
            self._tell(LECTURE_CHANGED, owner, lecture_id)
        return self.describe(state)

    def delete(self, owner: str, lecture_id: str) -> None:
        with self._files:
            state = self._state(owner, lecture_id)
            shutil.rmtree(self._dir(owner, lecture_id))
            self._changed(owner)
        log.info("library %s: deleted %s", owner, lecture_id)
        if state.get("ready"):
            self._tell(LECTURE_DELETED, owner, lecture_id)

    def delete_owner(self, owner: str) -> None:
        """Remove everything an account added: for when the account is deleted."""
        with self._files:
            folder = self._owner_dir(owner)
            if folder.is_dir():
                shutil.rmtree(folder)
            self._changed(owner)
            self._indexes.pop(owner, None)
        log.info("library %s: removed with its account", owner)
        self._tell(OWNER_DELETED, owner, None)

    def expire(self) -> list[str]:
        """Remove lectures left unfinished for ``ABANDONED_AFTER_S``. Returns their ids."""
        removed = []
        with self._files:
            cutoff = self._clock() - ABANDONED_AFTER_S
            for owner in self.owners():
                for state in self._states(owner):
                    if not state.get("ready") and state.get("updated", 0) < cutoff:
                        shutil.rmtree(self._dir(owner, state["lecture_id"]))
                        removed.append(state["lecture_id"])
        if removed:
            log.info("library: removed %d unfinished lecture(s)", len(removed))
        return removed

    def _tell(self, change: str, owner: str, lecture_id: str | None) -> None:
        if self.on_change is None:
            return
        try:
            self.on_change(change, owner, lecture_id)
        except Exception:  # the change is made; whoever was told of it failed
            log.exception("library on_change failed after: %s", change)

    # ---- the worker ------------------------------------------------------------------------------

    @property
    def models(self) -> Models:
        if self._models is None:
            self._models = self._make_models()
        return self._models

    def recover(self) -> None:
        """Queue whatever was left half done when the server last stopped."""
        self.expire()
        with self._files:
            for owner in self.owners():
                for state in self._states(owner):
                    stage = self.stage(state)
                    if stage == "building" or (stage == "speech" and not state.get("ocr")):
                        self._enqueue(owner, state["lecture_id"])

    def _enqueue(self, owner: str, lecture_id: str) -> None:
        with self._wake:
            self._queue[(owner, lecture_id)] = None
            self._wake.notify()

    def pending(self) -> int:
        with self._wake:
            return len(self._queue)

    def run_pending(self) -> None:
        """Do every queued piece of work now, in this thread."""
        while True:
            with self._wake:
                if not self._queue:
                    return
                key, _ = self._queue.popitem(last=False)
            self._advance(*key)

    def _work(self) -> None:
        while True:
            with self._wake:
                while not self._queue and not self._stopping:
                    self._wake.wait()
                if self._stopping:
                    return
                key, _ = self._queue.popitem(last=False)
            self._advance(*key)

    def close(self, timeout: float = 5.0) -> None:
        with self._wake:
            self._stopping = True
            self._wake.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)

    def _advance(self, owner: str, lecture_id: str) -> None:
        """Do what can be done for this lecture: read its slides, then build its index."""
        try:
            with self._files:
                state = self._state(owner, lecture_id)
            if self.stage(state) in ("frames", "stills", "ready", "failed"):
                return
            if not state["ocr"]:
                self._read_slides(owner, state)
                with self._files:
                    state = self._state(owner, lecture_id)
            if state["ocr"] and state["transcript"] and not state["ready"]:
                self._build(owner, state)
        except NoSuchLecture:
            return  # deleted while it waited, or while it was worked on
        except Exception:
            log.exception("library %s: could not finish %s", owner, lecture_id)
            with self._files:
                try:
                    state = self._state(owner, lecture_id)
                except NoSuchLecture:
                    return
                state["error"] = "This lecture could not be finished. Delete it and add it again."
                self._save(owner, state)

    def _keyframes(self, owner: str, state: Mapping[str, Any]) -> dict[float, Path]:
        folder = self._dir(owner, state["lecture_id"]) / "work" / "keyframes"
        return {t: folder / still_name(t) for t in state["keyframe_times"]}

    def _read_slides(self, owner: str, state: dict[str, Any]) -> None:
        started = time.perf_counter()
        ocr = _Locked(self.models.ocr, self.model_lock)
        pages = read_slides(ocr, self._keyframes(owner, state))
        with self._files:
            state = self._state(owner, state["lecture_id"])
            write_json(self._dir(owner, state["lecture_id"]) / "work" / "ocr.json", pages)
            state["ocr"] = True
            self._save(owner, state)
        log.info(
            "library %s: read %d slides of %s in %.0f s",
            owner, len(pages), state["lecture_id"], time.perf_counter() - started,
        )  # fmt: skip

    def _build(self, owner: str, state: dict[str, Any]) -> None:
        started = time.perf_counter()
        lecture_id = state["lecture_id"]
        folder = self._dir(owner, lecture_id)
        work = folder / "work"
        with self._files:
            windows = plan_windows(load_segments(work / "segments.json"), self.config.moments)
            times = keyframe_times(windows)
            texts = slide_texts(read_json(work / "ocr.json"), times)
            words = list(asr.iter_words(asr.load_segments(work / "asr.jsonl")))
        frame = moments_table(self.lecture(state), windows, words, texts)
        keyframes = self._keyframes(owner, state)
        vectors = embed_moments(
            frame,
            [keyframes[t] for t in times],
            _Locked(self.models.text_embedder, self.model_lock),
            _Locked(self.models.image_embedder, self.model_lock),
        )
        with self._files:
            state = self._state(owner, lecture_id)
            frame.to_parquet(work / "moments.parquet", index=False)
            np.savez(work / "vectors.npz", **vectors)
            index = assemble(
                {lecture_id: self.lecture(state)},
                {lecture_id: work},
                folder / "index",
                self.config,
                asr_model=state.get("asr_model"),
            )
            shutil.rmtree(work / "keyframes", ignore_errors=True)  # the thumbnails stay
            state.update(
                ready=True,
                n_moments=len(index),
                n_slides=len(index.image),
                bytes=_size(folder),
            )
            self._save(owner, state)
            self._changed(owner)
        log.info(
            "library %s: %s is ready, %d moments (%.0f s)",
            owner, lecture_id, len(index), time.perf_counter() - started,
        )  # fmt: skip
        self._tell(LECTURE_READY, owner, lecture_id)


def _size(folder: Path) -> int:
    return sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())
