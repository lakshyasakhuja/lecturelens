"""The indexing pipeline: stage order, caching, invalidation and assembly.

Decoding, slide detection, windowing and alignment are real. Only the four models are
replaced by deterministic stand-ins, so these tests need ffmpeg but no weights.
"""

from __future__ import annotations

import dataclasses
import shutil

import numpy as np
import pytest
from fakes import FakeIndexImageEmbedder, FakeIndexTextEmbedder, FakeOcr, FakeWhisper

from lecturelens.catalog import Lecture
from lecturelens.embed import EmbedConfig
from lecturelens.index import Index
from lecturelens.pipeline import Indexer, PipelineConfig, StageCache, fingerprint
from lecturelens.slides import SlideConfig
from lecturelens.utils import read_json


def make_indexer(work, config=None):
    indexer = Indexer(config or PipelineConfig(), work_dir=work)
    indexer._asr_model = FakeWhisper()
    indexer._ocr_engine = FakeOcr()
    indexer._text_embedder = FakeIndexTextEmbedder()
    indexer._image_embedder = FakeIndexImageEmbedder()
    return indexer


def calls(indexer):
    return (
        indexer._asr_model.calls,
        indexer._ocr_engine.calls,
        indexer._text_embedder.calls,
        indexer._image_embedder.calls,
    )


LECTURE = Lecture("lec_test", "Test lecture", video_link_template="https://v.example/?t={seconds}")


# ---- fingerprints and the stage cache ------------------------------------------------------


def test_fingerprint_is_stable_and_sensitive():
    assert fingerprint("a", {"x": 1, "y": 2}) == fingerprint("a", {"y": 2, "x": 1})
    assert fingerprint("a", {"x": 1}) != fingerprint("a", {"x": 2})
    assert fingerprint(dataclasses.asdict(SlideConfig())) != fingerprint(
        dataclasses.asdict(SlideConfig(min_stable_s=4))
    )


def test_stage_cache_persists_and_checks_outputs(tmp_path):
    output = tmp_path / "result.json"
    cache = StageCache(tmp_path)
    assert not cache.fresh("stage", "key1", output)

    output.write_text("{}")
    cache.done("stage", "key1")
    reopened = StageCache(tmp_path)
    assert reopened.fresh("stage", "key1", output)
    assert not reopened.fresh("stage", "key2", output)  # inputs changed
    output.unlink()
    assert not reopened.fresh("stage", "key1", output)  # output went missing


# ---- one lecture, end to end -----------------------------------------------------------------


@pytest.mark.ffmpeg
def test_build_produces_a_valid_searchable_index(lecture_video, tmp_path):
    video, cuts, seconds = lecture_video
    indexer = make_indexer(tmp_path / "work")
    indexer.build({"lec_test": LECTURE}, {"lec_test": video}, tmp_path / "index")

    index = Index.load(tmp_path / "index")
    moments = index.moments
    # Four slides, cut where the synthetic lecture cuts.
    assert moments["t_start"].tolist() == [0.0, *map(float, cuts)]
    assert moments["t_end"].iat[-1] == pytest.approx(seconds, abs=1)
    assert moments["moment_id"].tolist() == [f"lec_test:{i:04d}" for i in range(4)]

    # Speech is aligned to the slide it was spoken over: the first slide ends at 20 s.
    assert moments["transcript"].iat[0] == " ".join(
        f"second {t} of the lecture" for t in (0, 5, 10, 15)
    )
    assert moments["transcript"].iat[1].startswith("second 20 of the lecture")
    # Slide text comes from each moment's own keyframe, the last frame before the cut.
    assert moments["slide_text"].iat[0] == f"Gaussian Processes slide at {cuts[0] - 1}"
    assert moments["video_link"].tolist() == [f"https://v.example/?t={t}" for t in (0, *cuts)]

    assert index.text.shape == index.speech.shape == index.slide.shape == (4, 4096)
    assert index.image.shape == (4, 4096)
    assert index.manifest["image_calibration"] == {"scale": 10.0, "bias": -5.0}
    assert index.manifest["total_duration_s"] == pytest.approx(seconds, abs=1)
    for row in range(4):
        thumb = index.thumb_path(row)
        assert thumb is not None and thumb.stat().st_size > 0

    # Jargon from the slides reached the recogniser as a glossary prompt.
    assert indexer._asr_model.hotwords == "Glossary: Processes, Gaussian."


@pytest.mark.ffmpeg
def test_second_run_is_fully_cached(lecture_video, tmp_path):
    video, _, _ = lecture_video
    first = make_indexer(tmp_path / "work")
    first.build({"lec_test": LECTURE}, {"lec_test": video}, tmp_path / "index")
    assert calls(first) == (1, 4, 3, 1)  # one transcription, four slides, three text passes

    second = make_indexer(tmp_path / "work")
    second.build({"lec_test": LECTURE}, {"lec_test": video}, tmp_path / "index")
    assert calls(second) == (0, 0, 0, 0)
    assert second.timings == {"lec_test": {}}


@pytest.mark.ffmpeg
def test_a_changed_setting_reruns_only_the_stages_downstream_of_it(lecture_video, tmp_path):
    video, _, _ = lecture_video
    work, out = tmp_path / "work", tmp_path / "index"
    make_indexer(work).build({"lec_test": LECTURE}, {"lec_test": video}, out)

    # A different embedding model: only the vectors are recomputed.
    embed_changed = PipelineConfig(embed=EmbedConfig(text_model="another/model"))
    indexer = make_indexer(work, embed_changed)
    indexer.build({"lec_test": LECTURE}, {"lec_test": video}, out)
    assert calls(indexer) == (0, 0, 3, 1)

    # Jargon boost off: speech recognition reruns (its prompt changed), OCR does not.
    no_boost = dataclasses.replace(embed_changed, jargon_boost=False)
    indexer = make_indexer(work, no_boost)
    indexer.build({"lec_test": LECTURE}, {"lec_test": video}, out)
    assert calls(indexer) == (1, 0, 3, 1)
    assert indexer._asr_model.hotwords is None

    # A slide-detection setting: everything visual reruns, speech recognition does not.
    resegmented = dataclasses.replace(no_boost, slides=SlideConfig(min_stable_s=2.0))
    indexer = make_indexer(work, resegmented)
    indexer.build({"lec_test": LECTURE}, {"lec_test": video}, out)
    assert calls(indexer) == (0, 4, 3, 1)

    # --force redoes all of it.
    indexer = make_indexer(work, resegmented)
    indexer.build({"lec_test": LECTURE}, {"lec_test": video}, out, force=True)
    assert calls(indexer) == (1, 4, 3, 1)


@pytest.mark.ffmpeg
def test_adding_a_lecture_processes_only_the_new_one(lecture_video, tmp_path):
    video, _, _ = lecture_video
    second_video = tmp_path / "lec_two.mp4"
    shutil.copy(video, second_video)
    work, out = tmp_path / "work", tmp_path / "index"
    make_indexer(work).build({"lec_test": LECTURE}, {"lec_test": video}, out)

    lectures = {"lec_test": LECTURE, "lec_two": Lecture("lec_two", "Second lecture")}
    indexer = make_indexer(work)
    index = indexer.build(lectures, {"lec_test": video, "lec_two": second_video}, out)

    assert calls(indexer) == (1, 4, 3, 1)  # the first lecture came from its work directory
    assert index.moments["lecture_id"].tolist() == ["lec_test"] * 4 + ["lec_two"] * 4
    # Image rows are offset per lecture, so the second lecture points at its own keyframes.
    assert index.moments["image_idx"].tolist() == list(range(8))
    assert len(index.image) == 8
    assert set(index.lectures) == {"lec_test", "lec_two"}
    assert (out / "thumbs" / "lec_two").is_dir()
    assert index.moments["video_link"].iat[4] == ""  # no link template for the second one


@pytest.mark.ffmpeg
def test_masks_file_is_applied_per_lecture(lecture_video, tmp_path):
    video, _, _ = lecture_video
    masks = tmp_path / "masks.yaml"
    masks.write_text("lectures:\n  lec_test:\n    t_high: 0.9\n    t_low: 0.5\n", encoding="utf-8")
    indexer = make_indexer(tmp_path / "work", PipelineConfig(masks_path=masks))
    index = indexer.build({"lec_test": LECTURE}, {"lec_test": video}, tmp_path / "index")

    # With thresholds that high nothing counts as a slide change: one segment, windowed.
    stored = read_json(tmp_path / "work" / "lec_test" / "segments.json")
    assert len(stored["segments"]) == 1
    assert stored["t_high"] == 0.9
    assert len(index) == 4  # 107 s tiled by 45 s windows with at least 15 s overlap
    assert index.moments["image_idx"].nunique() == 1  # which all share the one keyframe
    assert np.allclose(index.image[0], index.image[index.moments["image_idx"].iat[3]])


def test_build_with_no_lectures_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="no lectures"):
        make_indexer(tmp_path).build({}, {}, tmp_path / "index")
