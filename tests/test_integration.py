"""The whole system on the synthetic demo lectures, with the real models.

    pytest -m models

Skipped by default: it downloads several gigabytes of weights on first use and takes a few
minutes. It is the regression test for everything the unit tests replace with stand-ins:
Whisper, PP-OCRv5, Qwen3-Embedding and SigLIP 2, wired together through the real pipeline.

The demo lectures are clean synthetic slides with a synthetic voice, so the thresholds
below are deliberately loose. They assert that the system works, not how well.
"""

from __future__ import annotations

import json
import shutil

import pytest

from lecturelens import metrics
from lecturelens.asr import iter_words, load_segments
from lecturelens.catalog import discover_videos, load_catalog, resolve_lectures
from lecturelens.demo import make_demo
from lecturelens.evaluation import ABLATION, Evaluator, load_queries
from lecturelens.index import Index
from lecturelens.pipeline import Indexer, PipelineConfig
from lecturelens.search import Searcher
from lecturelens.utils import read_json

pytestmark = [pytest.mark.models, pytest.mark.ffmpeg]


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    if not (shutil.which("say") or shutil.which("espeak-ng") or shutil.which("espeak")):
        pytest.skip("no text-to-speech engine to narrate the demo lectures")
    root = make_demo(tmp_path_factory.mktemp("demo"))
    videos = discover_videos(root / "videos")
    lectures = resolve_lectures(videos, load_catalog(root / "lectures.csv"))
    indexer = Indexer(PipelineConfig(), work_dir=root / "work")
    indexer.build(lectures, videos, root / "index")
    return root


@pytest.fixture(scope="module")
def index(demo):
    return Index.load(demo / "index")


def test_every_slide_transition_is_found_and_builds_are_not(demo):
    for lecture_id in ("demo01", "demo02"):
        truth = read_json(demo / "eval" / "transitions" / f"{lecture_id}.json")
        segments = read_json(demo / "work" / lecture_id / "segments.json")["segments"]
        predicted = [seg["t_start"] for seg in segments[1:]]
        score = metrics.match_transitions(predicted, truth["transitions"], tolerance=1.0)
        assert score.precision == 1.0, f"{lecture_id}: spurious cuts at {score.spurious}"
        assert score.recall == 1.0, f"{lecture_id}: missed cuts at {score.missed}"
        # Bullet reveals on the build slide must not have been reported as slide changes.
        for build in truth["builds"]:
            assert all(abs(t - build) > 1.0 for t in predicted)


def test_ocr_reads_the_slides(demo, index):
    found = total = 0
    for lecture_id in ("demo01", "demo02"):
        truth = read_json(demo / "truth" / f"{lecture_id}.json")
        rows = index.moments[index.moments["lecture_id"] == lecture_id]
        assert len(rows) == len(truth["slides"])
        for slide, slide_text in zip(truth["slides"], rows["slide_text"], strict=True):
            for line in slide["text"]:
                total += 1
                found += line.lower() in slide_text.lower()
    assert found / total >= 0.9, f"OCR recovered only {found} of {total} slide lines"


def test_transcripts_are_accurate_and_aligned_to_slides(demo, index):
    pooled = None
    for lecture_id in ("demo01", "demo02"):
        truth = read_json(demo / "truth" / f"{lecture_id}.json")
        words = list(iter_words(load_segments(demo / "work" / lecture_id / "asr.jsonl")))
        reference = " ".join(slide["narration"] for slide in truth["slides"])
        score = metrics.word_errors(reference, " ".join(w.text for w in words))
        pooled = score if pooled is None else pooled + score

        # Each slide's narration must have ended up in that slide's moment, not a neighbour's.
        rows = index.moments[index.moments["lecture_id"] == lecture_id]
        for slide, transcript in zip(truth["slides"], rows["transcript"], strict=True):
            per_slide = metrics.word_errors(slide["narration"], transcript)
            assert per_slide.wer <= 0.2, f"{lecture_id} slide {slide['index']}: {transcript!r}"
    assert pooled.wer <= 0.05


def test_the_index_is_complete(index):
    assert set(index.lectures) == {"demo01", "demo02"}
    assert index.text.shape[1] == 1024 and index.image.shape[1] == 768
    assert (abs((index.text**2).sum(axis=1) - 1) < 1e-2).all()  # unit vectors, after float16
    assert index.manifest["image_calibration"]["scale"] > 1
    for row in range(len(index)):
        assert index.thumb_path(row) is not None


def test_search_finds_each_kind_of_query(demo, index):
    searcher = Searcher(index, device="cpu")
    queries = load_queries(demo / "eval" / "queries.jsonl", index)
    evaluator = Evaluator(searcher, queries)
    hybrid = evaluator.run(next(s for s in ABLATION if s.key == "hybrid"), tolerance=2.0)
    summary = hybrid.summary(queries)
    assert summary["overall"]["r5"] >= 0.85
    assert summary["overall"]["mrr10"] >= 0.7

    # Each signal must be doing real work on the query type it exists for.
    image_only = evaluator.run(next(s for s in ABLATION if s.key == "image"), tolerance=2.0)
    assert image_only.summary(queries)["by_type"]["visual"]["r1"] >= 0.75
    speech_only = evaluator.run(next(s for s in ABLATION if s.key == "dense_speech"), tolerance=2.0)
    assert speech_only.summary(queries)["by_type"]["spoken"]["r5"] == 1.0


def test_results_are_explained(index):
    searcher = Searcher(index, device="cpu")
    top = searcher.search("the student who memorises last year's exam answers")[0]
    assert top.lecture_id == "demo01"
    assert "said" in top.badges
    assert top.highlights, "matched words should be highlighted in the snippet"
    assert top.video_link.endswith(f"#t={int(top.t_start)}")
    assert top.thumb is not None and top.thumb.is_file()

    # JSON-serialisable signals, as the CLI and the web app will emit them.
    json.dumps({"signals": top.signals, "badges": top.badges})
