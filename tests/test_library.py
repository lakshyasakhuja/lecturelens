"""Private libraries: a lecture added from the browser's pieces, and everything refused."""

from __future__ import annotations

import gzip
import io
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import synthetic
from fakes import FakeIndexImageEmbedder, FakeIndexTextEmbedder, FakeOcr

from slideseek import library as lib
from slideseek.library import Library, LibraryError, Models, NoSuchLecture, owner_key

OWNER, OTHER = "1-1700000000", "2-1700000001"
FILE = {"name": "lecture 4.mp4", "size": 147_000_000, "type": "video/mp4"}


def fake_models():
    return Models(FakeIndexTextEmbedder(), FakeIndexImageEmbedder(), FakeOcr())


def make_library(root, **options):
    options.setdefault("worker", False)
    return Library(root, fake_models, **options)


def jpeg(size=(64, 36)) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", size, "white").save(buffer, "JPEG")
    return buffer.getvalue()


@pytest.fixture(scope="module")
def frames(lecture_frames):
    """The four-slide lecture of 107 s, as the browser sends it: bytes, gzipped."""
    stack, _ = lecture_frames
    return gzip.compress(synthetic.to_uint8(stack).tobytes())


WORDS = {
    "model": "whisper in a browser",
    "segments": [
        {"words": [{"w": "hello", "s": 1.0, "e": 1.5}, {"w": "everyone", "s": 1.5, "e": 2.0}]},
        {"words": [{"w": "kernels", "s": 30.0, "e": 30.6, "p": 0.9}]},
        {"words": [{"w": "goodbye", "s": 100.0, "e": 100.5}]},
    ],
}


def add(library, frames, owner=OWNER, title="Lecture 4", words=WORDS, finish=True):
    made = library.create(owner, title=title, duration_s=107.0, file=FILE)
    lecture_id = made["lecture_id"]
    sent = library.put_frames(owner, lecture_id, frames)
    for t in sent["keyframe_times"]:
        library.put_still(owner, lecture_id, round(t * 1000), jpeg())
    library.put_transcript(owner, lecture_id, words)
    if finish:
        library.run_pending()
    return lecture_id


# ---- the whole way through -------------------------------------------------------------------


def test_a_lecture_goes_from_pieces_to_searchable(tmp_path, frames, lecture_frames):
    _, cuts = lecture_frames
    library = make_library(tmp_path)
    told = []
    library.on_change = lambda *change: told.append(change)

    made = library.create(OWNER, title="  Lecture   4 ", duration_s=107.0, file=FILE)
    lecture_id = made["lecture_id"]
    assert lib.LECTURE_ID.fullmatch(lecture_id)
    assert made["stage"] == "frames" and made["title"] == "Lecture 4"

    sent = library.put_frames(OWNER, lecture_id, frames)
    assert sent["stage"] == "stills" and sent["stills"] == [0, 4]
    times = sent["keyframe_times"]
    assert len(times) == 4
    for n, t in enumerate(times, start=1):
        got = library.put_still(OWNER, lecture_id, round(t * 1000), jpeg())
        assert got["stills"] == [n, 4]
    assert got["stage"] == "speech"
    assert library.pending() == 1  # the slides can be read while the browser listens

    assert library.put_transcript(OWNER, lecture_id, WORDS)["stage"] == "building"
    library.run_pending()
    ready = library.get(OWNER, lecture_id)
    assert ready["stage"] == "ready" and ready["n_moments"] == 4 and ready["n_slides"] == 4
    assert ready["bytes"] > 0
    assert told == [(lib.LECTURE_READY, OWNER, lecture_id)]

    index = library.index(OWNER)
    assert index.moments["lecture_id"].unique().tolist() == [lecture_id]
    assert index.moments["t_start"].tolist() == [0.0, *map(float, cuts)]
    assert index.moments["transcript"].tolist() == ["hello everyone", "kernels", "", "goodbye"]
    assert index.moments["slide_text"].iat[0].startswith("Gaussian Processes slide at")
    assert index.manifest["models"]["asr"] == "whisper in a browser"
    assert index.lectures[lecture_id].title == "Lecture 4"
    assert all(index.thumb_path(row) is not None for row in range(len(index)))

    # The full-size stills are gone; the thumbnails are kept, and only the owner gets them.
    folder = tmp_path / OWNER / lecture_id
    assert not (folder / "work" / "keyframes").exists()
    name = index.thumb_path(0).name
    assert library.thumb(OWNER, lecture_id, name) == index.thumb_path(0)
    assert library.thumb(OTHER, lecture_id, name) is None
    assert library.thumb(OWNER, lecture_id, "../lecture.json") is None


def test_the_pieces_can_arrive_in_any_order(tmp_path, frames):
    library = make_library(tmp_path)
    lecture_id = library.create(OWNER, title="Talk", duration_s=107.0, file=FILE)["lecture_id"]
    library.put_transcript(OWNER, lecture_id, WORDS)  # before the frames, even
    times = library.put_frames(OWNER, lecture_id, frames)["keyframe_times"]
    # Frames sent again after a reload: the same times, and nothing is lost.
    assert library.put_frames(OWNER, lecture_id, frames)["keyframe_times"] == times
    for t in reversed(times):
        library.put_still(OWNER, lecture_id, round(t * 1000), jpeg())
    library.run_pending()
    assert library.get(OWNER, lecture_id)["stage"] == "ready"


def test_a_library_is_its_owners_alone(tmp_path, frames):
    library = make_library(tmp_path)
    lecture_id = add(library, frames)
    assert [item["lecture_id"] for item in library.lectures(OWNER)] == [lecture_id]
    assert library.lectures(OTHER) == []
    assert library.index(OTHER) is None
    for attempt in (
        lambda: library.get(OTHER, lecture_id),
        lambda: library.rename(OTHER, lecture_id, "Mine now"),
        lambda: library.delete(OTHER, lecture_id),
        lambda: library.put_transcript(OTHER, lecture_id, WORDS),
        lambda: library.get(OWNER, "../2-1700000001/own-000000000000"),
        lambda: library.get("../" + OWNER, lecture_id),
    ):
        with pytest.raises(NoSuchLecture):
            attempt()


def test_an_account_given_a_reused_id_starts_empty():
    first = SimpleNamespace(id=7, created_at=1_700_000_000)
    later = SimpleNamespace(id=7, created_at=1_800_000_000)
    assert owner_key(first) != owner_key(later)


# ---- what is refused ---------------------------------------------------------------------------


def refused(call, status=400, match=None):
    with pytest.raises(LibraryError, match=match) as caught:
        call()
    assert caught.value.status == status, caught.value
    return caught.value


def test_what_a_new_lecture_must_say(tmp_path):
    library = make_library(tmp_path)
    create = library.create
    refused(lambda: create(OWNER, title="  ", duration_s=100, file=FILE), match="title")
    refused(lambda: create(OWNER, title="x" * 121, duration_s=100, file=FILE), match="title")
    refused(lambda: create(OWNER, title="A", duration_s=5, file=FILE), match="at least")
    refused(lambda: create(OWNER, title="A", duration_s="soon", file=FILE), match="at least")
    refused(lambda: create(OWNER, title="A", duration_s=5 * 3600, file=FILE), 413, "4 hours")
    refused(lambda: create(OWNER, title="A", duration_s=100, file={"name": "a.mp4"}), match="size")
    refused(lambda: create(OWNER, title="A", duration_s=100, file="a.mp4"), match="file")
    assert library.lectures(OWNER) == []


def test_one_lecture_at_a_time_and_twenty_in_all(tmp_path, frames, monkeypatch):
    library = make_library(tmp_path)
    first = library.create(OWNER, title="First", duration_s=107.0, file=FILE)
    refused(
        lambda: library.create(OWNER, title="Second", duration_s=107.0, file=FILE), 409, "First"
    )
    library.delete(OWNER, first["lecture_id"])

    monkeypatch.setattr(lib, "MAX_LECTURES", 2)
    add(library, frames, title="One")
    add(library, frames, title="Two")
    refused(lambda: library.create(OWNER, title="Three", duration_s=107.0, file=FILE), 409, "2 lectures")  # fmt: skip


def test_only_so_many_lectures_are_added_at_once_across_the_site(tmp_path, monkeypatch):
    monkeypatch.setattr(lib, "MAX_IN_PROGRESS", 1)
    library = make_library(tmp_path)
    library.create(OWNER, title="Mine", duration_s=107.0, file=FILE)
    refused(lambda: library.create(OTHER, title="Theirs", duration_s=107.0, file=FILE), 503)


def test_frames_must_be_the_lectures_frames(tmp_path, frames):
    library = make_library(tmp_path)
    lecture_id = library.create(OWNER, title="A", duration_s=107.0, file=FILE)["lecture_id"]
    put = library.put_frames
    refused(lambda: put(OWNER, lecture_id, b"not gzip"), match="gzip")
    short = gzip.compress(b"\0" * (106 * 14400))
    refused(lambda: put(OWNER, lecture_id, short), match="107 frames")
    bomb = gzip.compress(b"\0" * (300 * 14400))
    refused(lambda: put(OWNER, lecture_id, bomb), 413)
    refused(lambda: put(OWNER, lecture_id, frames[:-20]), match="cut short")
    assert library.get(OWNER, lecture_id)["stage"] == "frames"


def test_stills_must_be_wanted_jpegs(tmp_path, frames):
    from PIL import Image

    library = make_library(tmp_path)
    lecture_id = library.create(OWNER, title="A", duration_s=107.0, file=FILE)["lecture_id"]
    refused(lambda: library.put_still(OWNER, lecture_id, 19000, jpeg()), 404)  # no frames yet
    times = library.put_frames(OWNER, lecture_id, frames)["keyframe_times"]
    ms = round(times[0] * 1000)
    png = io.BytesIO()
    Image.new("RGB", (8, 8)).save(png, "PNG")
    refused(lambda: library.put_still(OWNER, lecture_id, ms, png.getvalue()), match="JPEG")
    refused(lambda: library.put_still(OWNER, lecture_id, ms, b"\xff\xd8\xff garbage"))
    refused(lambda: library.put_still(OWNER, lecture_id, ms, b""), match="empty")
    refused(lambda: library.put_still(OWNER, lecture_id, ms + 1, jpeg()), 404)
    refused(lambda: library.put_still(OWNER, lecture_id, ms, jpeg((2000, 2000))), 413)
    assert library.get(OWNER, lecture_id)["stills"] == [0, 4]


def test_a_transcript_must_fit_the_lecture(tmp_path, monkeypatch):
    library = make_library(tmp_path)
    lecture_id = library.create(OWNER, title="A", duration_s=107.0, file=FILE)["lecture_id"]

    def words(*items):
        return {"model": "m", "segments": [{"words": list(items)}]}

    put = library.put_transcript
    refused(lambda: put(OWNER, lecture_id, {"words": []}), match="model, segments")
    refused(lambda: put(OWNER, lecture_id, words({"w": "late", "s": 200, "e": 201})), match="outside")  # fmt: skip
    refused(lambda: put(OWNER, lecture_id, words({"w": "back", "s": 5, "e": 4})), match="outside")
    refused(lambda: put(OWNER, lecture_id, words({"w": "nan", "s": float("nan"), "e": 1})), match="number")  # fmt: skip
    refused(lambda: put(OWNER, lecture_id, words({"w": "", "s": 1, "e": 2})), match="empty")
    refused(lambda: put(OWNER, lecture_id, words({"w": "x", "s": True, "e": 2})), match="number")
    monkeypatch.setattr(lib, "MAX_WORDS_PER_SECOND", 0)  # a budget of 100 words
    many = [{"w": "word", "s": 1, "e": 1.1}] * 101
    refused(lambda: put(OWNER, lecture_id, words(*many)), match="more words")

    # Words out of order are put in order; empty parts are dropped.
    put(OWNER, lecture_id, {"model": "m", "segments": [
        {"words": [{"w": "second", "s": 3, "e": 4}, {"w": "first", "s": 1, "e": 2}]},
        {"words": []},
    ]})  # fmt: skip
    from slideseek.asr import load_segments

    (segment,) = load_segments(tmp_path / OWNER / lecture_id / "work" / "asr.jsonl")
    assert [w.text for w in segment.words] == ["first", "second"]
    assert segment.text == "first second"


# ---- changing and removing ---------------------------------------------------------------------


def test_renaming_and_deleting(tmp_path, frames):
    library = make_library(tmp_path)
    told = []
    library.on_change = lambda *change: told.append(change)
    lecture_id = add(library, frames)
    version = library.version(OWNER)

    library.rename(OWNER, lecture_id, "Gaussian processes")
    assert library.version(OWNER) > version
    assert library.index(OWNER).lectures[lecture_id].title == "Gaussian processes"

    library.delete(OWNER, lecture_id)
    assert library.lectures(OWNER) == [] and library.index(OWNER) is None
    assert not (tmp_path / OWNER / lecture_id).exists()
    assert [change for change, _, _ in told] == [
        lib.LECTURE_READY, lib.LECTURE_CHANGED, lib.LECTURE_DELETED
    ]  # fmt: skip


def test_deleting_an_account_deletes_its_library(tmp_path, frames):
    from fakes import ADA, fast_accounts

    accounts = fast_accounts()
    library = make_library(tmp_path)
    accounts.on_account_deleted = lambda account: library.delete_owner(owner_key(account))
    ada = accounts.sign_up(**ADA)
    add(library, frames, owner=owner_key(ada))
    add(library, frames, owner=OTHER)
    assert accounts.delete_account(ada, ADA["password"])
    assert not (tmp_path / owner_key(ada)).exists()
    assert library.lectures(owner_key(ada)) == []
    assert len(library.lectures(OTHER)) == 1  # nobody else's


def test_unfinished_lectures_are_removed_after_two_days(tmp_path, frames):
    now = [1_800_000_000.0]
    library = make_library(tmp_path, clock=lambda: now[0])
    done = add(library, frames, title="Done")
    left = library.create(OWNER, title="Left", duration_s=107.0, file=FILE)["lecture_id"]
    now[0] += lib.ABANDONED_AFTER_S + 1
    assert library.expire() == [left]
    assert [item["lecture_id"] for item in library.lectures(OWNER)] == [done]


# ---- the worker ----------------------------------------------------------------------------------


def test_work_left_half_done_is_picked_up_after_a_restart(tmp_path, frames):
    first = make_library(tmp_path)
    lecture_id = add(first, frames, finish=False)
    assert first.get(OWNER, lecture_id)["stage"] == "building"

    again = make_library(tmp_path)  # the server started again
    assert again.pending() == 1
    again.run_pending()
    assert again.get(OWNER, lecture_id)["stage"] == "ready"


def test_a_lecture_that_cannot_be_finished_says_so(tmp_path, frames):
    class BrokenOcr:
        def read(self, path):
            raise RuntimeError("the OCR model is missing")

    library = Library(
        tmp_path,
        lambda: Models(FakeIndexTextEmbedder(), FakeIndexImageEmbedder(), BrokenOcr()),
        worker=False,
    )
    lecture_id = add(library, frames)
    failed = library.get(OWNER, lecture_id)
    assert failed["stage"] == "failed" and "Delete it" in failed["error"]
    assert library.put_transcript(OWNER, lecture_id, WORDS)["stage"] == "failed"
    assert library.index(OWNER) is None
    library.delete(OWNER, lecture_id)  # it can still be removed


def test_the_worker_runs_in_the_background(tmp_path, frames):
    library = make_library(tmp_path, worker=True)
    try:
        lecture_id = add(library, frames, finish=False)
        deadline = time.monotonic() + 20
        while library.get(OWNER, lecture_id)["stage"] != "ready":
            assert time.monotonic() < deadline, library.get(OWNER, lecture_id)
            time.sleep(0.05)
    finally:
        library.close()


def test_the_models_are_used_a_batch_at_a_time_under_the_search_lock():
    lock = threading.Lock()
    seen = []

    class Recorder:
        calibration = (1.0, 0.0)

        def encode_documents(self, texts, progress=False):
            seen.append((len(texts), lock.locked()))
            return np.zeros((len(texts), 3), np.float32)

    shared = lib._Locked(Recorder(), lock, batch=8)
    assert shared.encode_documents(["x"] * 20).shape == (20, 3)
    assert seen == [(8, True), (8, True), (4, True)]
    assert shared.encode_documents([]).shape == (0, 3)
    assert shared.calibration == (1.0, 0.0)
