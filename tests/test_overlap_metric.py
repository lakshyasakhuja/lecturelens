"""Relevance, ranking metrics, transition matching and word error rate."""

from __future__ import annotations

import numpy as np
import pytest

from slideseek.metrics import (
    bootstrap_ci,
    first_relevant_rank,
    hits_at_k,
    is_relevant,
    landing_offset,
    lands,
    match_transitions,
    mrr_at_k,
    normalize_for_wer,
    paired_bootstrap,
    recall_at_k,
    word_errors,
)

GT = [(1312.0, 1420.0)]


# ---- relevance by overlap ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "span, expected",
    [
        ((1300, 1350), True),  # overlaps the start
        ((1350, 1400), True),  # inside
        ((1400, 1500), True),  # overlaps the end
        ((1000, 2000), True),  # contains it
        ((1250, 1297), True),  # ends 15 s before the answer: exactly at the tolerance
        ((1250, 1296.9), False),  # just outside it
        ((1435, 1500), True),  # starts 15 s after the answer ends
        ((1435.1, 1500), False),
        ((0, 100), False),
    ],
)
def test_overlap_with_tolerance(span, expected):
    assert is_relevant(span, GT, tolerance=15) is expected


def test_tolerance_widens_and_narrows_what_counts():
    span = (1250, 1290)  # ends 22 s before the answer starts
    assert not is_relevant(span, GT, tolerance=15)
    assert is_relevant(span, GT, tolerance=30)
    assert not is_relevant((1250, 1310), GT, tolerance=0)
    assert is_relevant((1250, 1312), GT, tolerance=0)  # touching counts


def test_any_of_several_ground_truth_intervals_counts():
    truth = [(100.0, 150.0), (900.0, 960.0)]
    assert is_relevant((910, 930), truth)
    assert is_relevant((60, 90), truth)  # within tolerance of the first
    assert not is_relevant((400, 500), truth)
    assert not is_relevant((400, 500), [])


def test_first_relevant_rank_requires_the_right_lecture():
    results = [
        ("lec02", 1350.0, 1400.0),  # right time, wrong lecture
        ("lec03", 10.0, 50.0),
        ("lec03", 1330.0, 1375.0),
        ("lec03", 1375.0, 1420.0),
    ]
    assert first_relevant_rank(results, "lec03", GT) == 3
    assert first_relevant_rank(results[:2], "lec03", GT) is None
    assert first_relevant_rank([], "lec03", GT) is None


@pytest.mark.parametrize(
    "span, expected",
    [
        ((1350, 1400), True),  # starts inside the answer
        ((1312, 1330), True),  # starts exactly with it
        ((1297, 1330), True),  # starts 15 s early: at the tolerance
        ((1296, 1330), False),  # 16 s early
        ((1230, 1320), False),  # a long moment that only ends inside the answer
        ((1420, 1500), False),  # starts as the answer ends
    ],
)
def test_landing_is_stricter_than_overlap(span, expected):
    assert lands(span, GT, tolerance=15) is expected
    assert is_relevant(span, GT, tolerance=15)  # every one of these overlaps


def test_first_relevant_rank_by_landing():
    results = [("lec03", 1230.0, 1320.0), ("lec03", 1320.0, 1365.0)]
    assert first_relevant_rank(results, "lec03", GT) == 1
    assert first_relevant_rank(results, "lec03", GT, criterion="landing") == 2


def test_landing_offset():
    assert landing_offset((1350, 1400), GT) == 0  # starts inside the answer
    assert landing_offset((1312, 1400), GT) == 0
    assert landing_offset((1250, 1330), GT) == 62  # a minute of lead-in before the answer
    assert landing_offset((1430, 1500), GT) == 10  # landed just after it ended
    assert landing_offset((500, 520), [(100.0, 150.0), (560.0, 600.0)]) == 60  # nearest answer


# ---- recall and MRR -------------------------------------------------------------------------


def test_recall_and_mrr_from_ranks():
    ranks = [1, 3, None, 7, 12]
    assert recall_at_k(ranks, 1) == pytest.approx(1 / 5)
    assert recall_at_k(ranks, 5) == pytest.approx(2 / 5)
    assert recall_at_k(ranks, 10) == pytest.approx(3 / 5)
    assert mrr_at_k(ranks, 10) == pytest.approx((1 + 1 / 3 + 0 + 1 / 7 + 0) / 5)
    assert hits_at_k(ranks, 5).tolist() == [1, 1, 0, 0, 0]


def test_metrics_at_the_extremes():
    assert recall_at_k([1, 1, 1], 1) == 1.0
    assert mrr_at_k([1, 1, 1]) == 1.0
    assert recall_at_k([None, None], 5) == 0.0
    assert mrr_at_k([None, None]) == 0.0
    assert np.isnan(recall_at_k([], 5))
    assert np.isnan(mrr_at_k([]))


# ---- bootstrap --------------------------------------------------------------------------------


def test_bootstrap_ci_brackets_the_mean_and_tightens_with_more_queries():
    rng = np.random.default_rng(0)
    small = (rng.random(20) < 0.7).astype(float)
    large = (rng.random(2000) < 0.7).astype(float)
    low_s, high_s = bootstrap_ci(small)
    low_l, high_l = bootstrap_ci(large)
    assert low_s <= small.mean() <= high_s
    assert low_l <= large.mean() <= high_l
    assert (high_l - low_l) < (high_s - low_s) / 3
    assert bootstrap_ci(small) == bootstrap_ci(small)  # seeded: reproducible


def test_bootstrap_ci_of_a_constant_and_of_nothing():
    assert bootstrap_ci([1.0] * 10) == (1.0, 1.0)
    assert all(np.isnan(v) for v in bootstrap_ci([]))


def test_paired_bootstrap_sees_a_consistent_difference_that_unpaired_intervals_hide():
    # System A beats B on every fourth query and ties elsewhere. Their separate confidence
    # intervals overlap heavily, but paired, A is never worse.
    b = np.array([1.0, 0.0] * 20)
    a = b.copy()
    a[1::4] = 1.0
    diff, low, _ = paired_bootstrap(a, b)
    assert diff == pytest.approx(0.25)
    assert low > 0
    assert bootstrap_ci(a)[0] < bootstrap_ci(b)[1]  # unpaired intervals overlap

    same = paired_bootstrap(b, b)
    assert same == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="same queries"):
        paired_bootstrap([1.0, 0.0], [1.0])


# ---- slide transitions ------------------------------------------------------------------------


def test_transition_matching_within_tolerance():
    score = match_transitions([10.4, 55.0, 67.9, 200.0], [10.0, 56.5, 70.0, 120.0], tolerance=2.0)
    assert score.matches == ((10.4, 10.0), (55.0, 56.5))
    assert score.spurious == (67.9, 200.0)  # 67.9 is 2.1 s from 70.0: just outside
    assert score.missed == (70.0, 120.0)
    assert (score.true_positives, score.false_positives, score.false_negatives) == (2, 2, 2)
    assert score.precision == score.recall == score.f1 == 0.5


def test_one_prediction_cannot_match_two_annotations():
    score = match_transitions([100.0], [99.0, 101.0], tolerance=2.0)
    assert score.true_positives == 1
    assert score.false_negatives == 1
    # ...and two predictions near one annotation are one hit and one false alarm.
    score = match_transitions([99.0, 101.0], [100.0], tolerance=2.0)
    assert (score.true_positives, score.false_positives) == (1, 1)


def test_matching_is_maximal_for_interleaved_times():
    # Pairing 10 with 11.5 first would strand 12; the sweep pairs (10, 9) and (12, 11.5).
    score = match_transitions([10.0, 12.0], [9.0, 11.5], tolerance=2.0)
    assert score.true_positives == 2


def test_transition_edge_cases():
    perfect = match_transitions([5, 10], [10, 5])
    assert (perfect.precision, perfect.recall, perfect.f1) == (1.0, 1.0, 1.0)

    nothing = match_transitions([], [])
    assert (nothing.precision, nothing.recall) == (1.0, 1.0)

    silent = match_transitions([], [10.0, 20.0])
    assert (silent.precision, silent.recall, silent.f1) == (1.0, 0.0, 0.0)

    noisy = match_transitions([10.0, 20.0], [])
    assert (noisy.precision, noisy.recall, noisy.f1) == (0.0, 1.0, 0.0)


# ---- word error rate --------------------------------------------------------------------------


def test_normalisation_removes_convention_not_content():
    assert normalize_for_wer("Held-out data, don't TOUCH it!") == "held out data don't touch it"
    assert normalize_for_wer("the 'quoted' word") == "the quoted word"
    assert normalize_for_wer("L2  penalty: λ‖w‖²") == "l2 penalty w"
    assert normalize_for_wer("isn’t") == "isn't"


def test_word_errors_counts_each_kind():
    score = word_errors("the quick brown fox jumps", "the quick red fox jumps high")
    assert (score.substitutions, score.deletions, score.insertions) == (1, 0, 1)
    assert score.errors == 2
    assert score.reference_words == 5
    assert score.wer == pytest.approx(0.4)

    dropped = word_errors("we hold out some data", "we hold data")
    assert (dropped.substitutions, dropped.deletions, dropped.insertions) == (0, 2, 0)


def test_identical_text_up_to_convention_has_zero_errors():
    assert word_errors("Held-out data. It's fine!", "held out data it's fine").wer == 0.0


def test_scores_pool_across_excerpts():
    a = word_errors("one two three four", "one two three four")  # 0 / 4
    b = word_errors("five six", "five")  # 1 / 2
    pooled = a + b
    assert pooled.wer == pytest.approx(1 / 6)  # not the mean of 0 and 0.5


def test_word_errors_agrees_with_jiwer():
    jiwer = pytest.importorskip("jiwer")
    rng = np.random.default_rng(3)
    vocabulary = "alpha beta gamma delta epsilon zeta eta theta".split()
    for _ in range(25):
        ref = " ".join(rng.choice(vocabulary, size=rng.integers(1, 30)))
        hyp = " ".join(rng.choice(vocabulary, size=rng.integers(1, 30)))
        assert word_errors(ref, hyp).wer == pytest.approx(jiwer.wer(ref, hyp))


def test_empty_reference_has_no_defined_rate():
    assert np.isnan(word_errors("", "anything").wer)
    assert word_errors("something", "").wer == 1.0
