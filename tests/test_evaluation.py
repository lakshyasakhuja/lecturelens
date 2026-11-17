"""The retrieval benchmark harness, on the hand-built index with stand-in encoders."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import synthetic as syn
from fakes import FakeImageEmbedder, FakeTextEmbedder, make_index

from slideseek.evaluation import (
    ABLATION,
    EXTRAS,
    Evaluator,
    Query,
    QueryError,
    System,
    ablation_markdown,
    failures_markdown,
    load_queries,
    log_test_run,
    tolerance_markdown,
    tune,
    tuned_system,
)
from slideseek.search import Searcher
from slideseek.slides import SlideDetector

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    """Import a script from eval/ as a module."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "eval" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


QUERIES = [
    # qid, query, type, lecture, ground truth, split
    ("q1", "chain rule", "keyword", "lec02", [[80, 170]], "dev"),
    ("q2", "dropout", "keyword", "lec01", [[270, 315]], "test"),
    (
        "q3",
        "data kept aside that the model is never fitted on",
        "paraphrase",
        "lec01",
        [[60, 150]],
        "dev",
    ),
    ("q4", "walking downhill in fog", "spoken", "lec02", [[170, 260]], "test"),
    ("q5", "diagram circles connected lines", "visual", "lec02", [[0, 80]], "dev"),
    ("q6", "plot two curves rising falling", "visual", "lec01", [[150, 240]], "test"),
    # Nothing in the index says this, in any modality: every system should miss it.
    ("q7", "eigenvalues of the hessian", "paraphrase", "lec02", [[400, 450]], "test"),
]


def write_queries(tmp_path, rows=QUERIES):
    path = tmp_path / "queries.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for qid, query, kind, lecture, gt, split in rows:
            handle.write(
                json.dumps(
                    {
                        "qid": qid,
                        "query": query,
                        "type": kind,
                        "lecture_id": lecture,
                        "gt": gt,
                        "split": split,
                    }
                )
                + "\n"
            )
    return path


@pytest.fixture()
def evaluator(tmp_path):
    index = make_index()
    searcher = Searcher(index, text_embedder=FakeTextEmbedder(), image_embedder=FakeImageEmbedder())
    queries = load_queries(write_queries(tmp_path))
    return Evaluator(searcher, queries)


# ---- loading and validating queries --------------------------------------------------------


def test_load_queries_parses_fields_and_timestamps(tmp_path):
    path = tmp_path / "queries.jsonl"
    path.write_text(
        '{"qid": "q017", "query": " why hold out data ", "type": "paraphrase", '
        '"lecture_id": "lec01", "gt": [["1:00", "2:30"], [200, 240]], "split": "test", "author": "friend"}\n'
        "\n",
        encoding="utf-8",
    )
    (query,) = load_queries(path, make_index())
    assert query == Query(
        "q017",
        "why hold out data",
        "paraphrase",
        "lec01",
        ((60.0, 150.0), (200.0, 240.0)),
        "test",
        "friend",
    )


@pytest.mark.parametrize(
    "line, message",
    [
        ("not json", "not valid JSON"),
        ('{"qid": "a", "query": "x"}', "missing field"),
        (
            '{"qid":"a","query":"x","type":"vibes","lecture_id":"lec01","gt":[[1,2]],"split":"dev"}',
            "type must be",
        ),
        (
            '{"qid":"a","query":"x","type":"keyword","lecture_id":"lec01","gt":[[1,2]],"split":"train"}',
            "split must be",
        ),
        (
            '{"qid":"a","query":" ","type":"keyword","lecture_id":"lec01","gt":[[1,2]],"split":"dev"}',
            "empty query",
        ),
        (
            '{"qid":"a","query":"x","type":"keyword","lecture_id":"lec01","gt":[[5,2]],"split":"dev"}',
            "start < end",
        ),
        (
            '{"qid":"a","query":"x","type":"keyword","lecture_id":"lec01","gt":[],"split":"dev"}',
            "start < end",
        ),
        (
            '{"qid":"a","query":"x","type":"keyword","lecture_id":"lec01","gt":[7],"split":"dev"}',
            "pairs",
        ),
        (
            '{"qid":"a","query":"x","type":"keyword","lecture_id":"lec99","gt":[[1,2]],"split":"dev"}',
            "not in the index",
        ),
        (
            '{"qid":"a","query":"x","type":"keyword","lecture_id":"lec01","gt":[[9000,9100]],"split":"dev"}',
            "after the end",
        ),
    ],
)
def test_load_queries_rejects_bad_input(tmp_path, line, message):
    path = tmp_path / "queries.jsonl"
    path.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(QueryError, match=message):
        load_queries(path, make_index())


def test_load_queries_rejects_duplicates_and_empty_files(tmp_path):
    row = (
        '{"qid":"a","query":"x","type":"keyword","lecture_id":"lec01","gt":[[1,2]],"split":"dev"}\n'
    )
    path = tmp_path / "queries.jsonl"
    path.write_text(row + row, encoding="utf-8")
    with pytest.raises(QueryError, match="duplicate qid"):
        load_queries(path)
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(QueryError, match="no queries"):
        load_queries(path)


# ---- running systems -----------------------------------------------------------------------


def test_each_signal_wins_the_query_type_it_should(evaluator):
    by_key = {system.key: evaluator.run(system) for system in (*ABLATION, *EXTRAS)}
    ranks = {
        key: dict(zip((q.qid for q in evaluator.queries), result.ranks, strict=True))
        for key, result in by_key.items()
    }

    # Keyword spoken aloud: found by transcript search.
    assert ranks["bm25_speech"]["q1"] == 1
    # A visual description shares no words with the transcript or slide: only the image
    # ranker finds it, and the hybrid inherits that.
    assert ranks["bm25"]["q5"] is None or ranks["bm25"]["q5"] > 1
    assert ranks["image"]["q5"] == 1
    assert ranks["hybrid"]["q5"] == 1
    # The unanswerable query is missed by everything.
    assert all(ranks[key]["q7"] is None for key in ranks)


def test_summary_reports_overall_and_per_type_metrics(evaluator):
    result = evaluator.run(next(s for s in ABLATION if s.key == "hybrid"))
    summary = result.summary(evaluator.queries)

    assert summary["number"] == 6
    assert summary["overall"]["n"] == 7
    assert summary["overall"]["r5"] == pytest.approx(6 / 7)
    low, high = summary["overall"]["r5_ci"]
    assert low <= 6 / 7 <= high
    assert {kind: block["n"] for kind, block in summary["by_type"].items()} == {
        "keyword": 2, "paraphrase": 2, "spoken": 1, "visual": 2,
    }  # fmt: skip
    assert summary["by_type"]["paraphrase"]["r5"] == 0.5  # q3 found, q7 unanswerable
    assert summary["by_type"]["visual"]["r5"] == 1.0
    assert summary["overall"]["median_landing_offset_s"] == 0.0


def test_results_are_deduplicated_like_the_served_system(evaluator):
    system = System("bm25", "bm25", ("bm25",))
    regularisation = Query("x", "regularisation", "keyword", "lec01", ((240.0, 315.0),), "dev")
    rows = evaluator.retrieve(system, regularisation)
    starts = [evaluator._starts[r] for r in rows]
    assert sorted(starts) == [240.0, 270.0]  # 30 s apart: distinct landing points, both kept


def test_queries_are_encoded_once_however_many_systems_run(evaluator):
    text, image = evaluator.searcher.text_embedder, evaluator.searcher.image_embedder
    for system in ABLATION:
        evaluator.run(system)
    assert text.calls == len(evaluator.queries)
    assert image.calls == len(evaluator.queries)


def test_rescore_under_a_different_tolerance(evaluator):
    # A system that returns the moment just before the answer: relevant only if the
    # tolerance reaches across the gap.
    late = Query("x", "welcome to the course", "keyword", "lec01", ((85.0, 140.0),), "dev")
    wide = Evaluator(evaluator.searcher, [late])
    result = wide.run(System("bm25", "bm25", ("bm25",)), tolerance=30)
    assert result.ranks == [1]  # moment [0, 60] is within 30 s of an answer starting at 85
    assert wide.rescore(result, tolerance=15) == [None]
    assert wide.rescore(result, tolerance=25) == [1]
    # The moment starts 85 s before the answer: an overlap, but not a landing.
    assert wide.rescore(result, tolerance=30, criterion="landing") == [None]


def test_relevant_rows_under_both_criteria(evaluator):
    q1 = evaluator.queries[0]  # answer: lec02, 80 s to 170 s, which is exactly row 6
    # By overlap, the neighbours count too: row 5 ends where the answer begins and row 7
    # begins where it ends, so both touch it.
    assert evaluator.relevant_rows(q1) == [5, 6, 7]
    # By landing, only a moment that starts in or just before the answer counts.
    assert evaluator.relevant_rows(q1, criterion="landing") == [6]
    assert evaluator.relevant_rows(evaluator.queries[-1]) == []  # the unanswerable one


# ---- tuning --------------------------------------------------------------------------------


def test_tune_prefers_the_plain_setting_on_a_tie_and_reports_dev_scores(evaluator):
    dev = Evaluator(evaluator.searcher, [q for q in evaluator.queries if q.split == "dev"])
    settings = tune(dev, weight_grid=(0.0, 1.0, 2.0), k_grid=(60,), floor_grid=(0.0,))

    assert settings["n_queries"] == 3
    assert settings["settings_tried"] == 9
    assert settings["weights"]["dense"] == 1.0
    assert settings["dev_r5"] == 1.0
    # Several settings reach the same score here; the tie goes to the untuned defaults.
    assert settings["weights"] == {"bm25": 1.0, "dense": 1.0, "image": 1.0}
    assert settings["rrf_k"] == 60

    system = tuned_system(settings)
    assert system.key == "hybrid_tuned"
    assert system.weights == settings["weights"]


def test_tune_finds_a_setting_that_matters():
    # Dev queries that only the image ranker can answer: tuning must keep its weight above 0.
    index = make_index()
    searcher = Searcher(index, text_embedder=FakeTextEmbedder(), image_embedder=FakeImageEmbedder())
    visual = [
        Query("v1", "diagram circles connected lines", "visual", "lec02", ((0.0, 80.0),), "dev"),
        Query("v2", "photo of a foggy valley", "visual", "lec02", ((170.0, 260.0),), "dev"),
    ]
    settings = tune(
        Evaluator(searcher, visual), weight_grid=(0.0, 1.0), k_grid=(60,), floor_grid=(0.0,)
    )
    assert settings["weights"]["image"] > 0
    assert settings["dev_mrr10"] == 1.0


# ---- reports -------------------------------------------------------------------------------


def test_ablation_table_has_a_row_per_system_and_a_column_per_type(evaluator):
    summaries = [evaluator.run(system).summary(evaluator.queries) for system in ABLATION]
    table = ablation_markdown(
        summaries, "test", 15.0, comparison=(0.25, 0.05, 0.45),
        latency={"p50_ms": 98.0, "p95_ms": 120.0, "n": 21, "device": "cpu"},
    )  # fmt: skip
    rows = [line for line in table.splitlines() if line.startswith("| ") and "System" not in line]
    assert len(rows) == len(ABLATION)
    assert rows[0].startswith("| 1 | BM25, transcript only |")
    assert rows[-1].startswith("| 6 | Hybrid RRF (2 + 4 + 5) |")
    for column in ("Keyword (n=2)", "Paraphrase (n=2)", "Spoken-only (n=1)", "Visual (n=2)"):
        assert column in table
    assert "+25.0 points" in table and "excludes zero" in table
    assert "p50 98 ms, p95 120 ms" in table
    assert "includes zero" in ablation_markdown(
        summaries, "test", 15.0, comparison=(0.1, -0.1, 0.3)
    )


def test_failure_report_lists_misses_with_a_diagnosis_starting_point(evaluator):
    hybrid = next(s for s in ABLATION if s.key == "hybrid")
    report = failures_markdown(evaluator, evaluator.run(hybrid), tolerance=15.0)
    assert "1 of 7 queries" in report
    assert "## q7: eigenvalues of the hessian" in report
    assert "No moment in the index overlaps the expected interval" in report
    assert "Returned instead:" in report
    assert "Cause: _to be filled in_" in report

    # A miss whose answer exists but was ranked too low shows where each ranker put it.
    bm25_only = System("bm25_speech", "BM25, transcript only", ("bm25_speech",))
    report = failures_markdown(evaluator, evaluator.run(bm25_only), tolerance=15.0)
    assert "## q5: diagram circles connected lines" in report
    assert "Where each ranker put the answer: bm25 not retrieved, dense" in report
    assert "image #1" in report


def test_tolerance_table():
    table = tolerance_markdown(
        [("Hybrid", {"overlap ±15 s": (0.6, 0.8, 0.7), "lands ≤15 s early": (0.5, 0.7, 0.6)})]
    )
    assert "| System | overlap ±15 s | lands ≤15 s early |" in table
    assert "| Hybrid | 60.0 / 80.0 / 70.0 | 50.0 / 70.0 / 60.0 |" in table


def test_test_runs_are_logged_append_only(tmp_path, evaluator):
    summary = evaluator.run(next(s for s in ABLATION if s.key == "hybrid")).summary(
        evaluator.queries
    )
    log = tmp_path / "results" / "test_runs.log"
    log_test_run(log, 7, summary)
    log_test_run(log, 7, summary)
    lines = log.read_text().splitlines()
    assert len(lines) == 2
    assert "commit " in lines[0] and "7 queries" in lines[0] and "R@5 85.7" in lines[0]


# ---- the evaluation scripts' own logic -------------------------------------------------------


def test_transition_annotations_accept_player_timestamps(tmp_path):
    script = load_script("eval_transitions")
    (tmp_path / "lec01.json").write_text(
        '{"lecture_id": "lec01", "transitions": ["1:35", 42, "0:07", 187.5]}', encoding="utf-8"
    )
    (tmp_path / "week2.json").write_text('{"transitions": [10]}', encoding="utf-8")
    assert script.load_annotations(tmp_path) == {"lec01": [7.0, 42.0, 95.0, 187.5], "week2": [10.0]}


def test_threshold_tuning_recovers_cuts_that_the_defaults_miss(lecture_frames):
    script = load_script("eval_transitions")
    frames, cuts = lecture_frames
    # Halve the contrast: the slide changes are now smaller than the default threshold.
    faint = 0.5 + (frames - 0.5) * 0.12
    detector = SlideDetector(syn.to_uint8(faint))
    assert [round(t) for t in detector.detect().transitions] != cuts

    tuned = script.tune_thresholds(detector, [float(c) for c in cuts], tolerance=2.0)
    assert [round(t) for t in detector.detect(**tuned).transitions] == cuts
    assert tuned["t_low"] < tuned["t_high"]


def test_asr_excerpts_and_hypothesis_window(tmp_path):
    from slideseek.asr import Word

    script = load_script("eval_asr")
    (tmp_path / "lec02_excerpt.txt").write_text("The corrected transcript.", encoding="utf-8")
    (tmp_path / "excerpts.jsonl").write_text(
        '{"lecture_id": "lec02", "t0": "12:00", "t1": "17:00", "reference": "lec02_excerpt.txt"}\n\n',
        encoding="utf-8",
    )
    (excerpt,) = script.load_excerpts(tmp_path)
    assert (excerpt["t0"], excerpt["t1"]) == (720.0, 1020.0)
    assert excerpt["reference"] == "The corrected transcript."

    words = [
        Word("before", 718.0, 719.0),
        Word("inside", 720.2, 720.6),
        Word("after", 1020.1, 1021.0),
    ]
    assert script.hypothesis(words, 720.0, 1020.0) == "inside"
    with pytest.raises(FileNotFoundError):
        script.load_excerpts(tmp_path / "missing")


def test_wer_report_pools_totals():
    from slideseek.metrics import word_errors

    script = load_script("eval_asr")
    rows = [
        {
            "lecture_id": "lec01",
            "t0": 0.0,
            "t1": 300.0,
            "scores": {"Indexed transcript": word_errors("a b c d", "a b c d")},
        },
        {
            "lecture_id": "lec02",
            "t0": 60.0,
            "t1": 360.0,
            "scores": {"Indexed transcript": word_errors("e f", "e")},
        },
    ]
    totals = {
        "Indexed transcript": rows[0]["scores"]["Indexed transcript"]
        + rows[1]["scores"]["Indexed transcript"]
    }
    report = script.render(rows, totals, "large-v3-turbo")
    assert "| lec02 | 1:00 to 6:00 | 2 | 50.00 |" in report
    assert "| **Total** | | 6 | **16.67** |" in report
    assert np.isclose(totals["Indexed transcript"].wer, 1 / 6)
