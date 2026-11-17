"""Reciprocal Rank Fusion, deduplication and snippets: the pure functions of search."""

from __future__ import annotations

import pytest

from slideseek.search import dedupe, make_snippet, ranks_from_order, rrf
from slideseek.text import analyze

# ---- rrf -------------------------------------------------------------------------------


def test_ranks_from_order_starts_at_one():
    assert ranks_from_order([7, 3, 9]) == {7: 1, 3: 2, 9: 3}


def test_single_ranker_keeps_its_order_and_scores_are_reciprocal():
    fused = rrf({"a": ranks_from_order([5, 2, 8])}, k=60)
    assert [doc for doc, _ in fused] == [5, 2, 8]
    assert [score for _, score in fused] == pytest.approx([1 / 61, 1 / 62, 1 / 63])


def test_agreement_between_rankers_beats_a_single_top_rank():
    # Doc 1 is second for both rankers; docs 0 and 2 are each first for one and absent
    # from the other. Consensus should win.
    fused = rrf({"a": {0: 1, 1: 2}, "b": {2: 1, 1: 2}}, k=60)
    assert fused[0][0] == 1
    assert fused[0][1] == pytest.approx(2 / 62)


def test_a_document_missing_from_a_ranker_gets_nothing_from_it():
    fused = dict(rrf({"a": {0: 1}, "b": {1: 1, 0: 50}}, k=60))
    assert fused[0] == pytest.approx(1 / 61 + 1 / 110)
    assert fused[1] == pytest.approx(1 / 61)


def test_scores_depend_only_on_ranks_not_on_raw_scores():
    # RRF never sees scores, so rankers on wildly different scales fuse the same way.
    bm25_order = [3, 1, 2]  # imagine scores 41.2, 7.9, 0.3
    cosine_order = [3, 2, 1]  # imagine scores 0.71, 0.70, 0.69
    fused = rrf({"bm25": ranks_from_order(bm25_order), "dense": ranks_from_order(cosine_order)})
    assert fused[0][0] == 3
    assert fused[1][1] == pytest.approx(fused[2][1])  # 1 and 2 swap places: an exact tie


def test_smaller_k_rewards_top_ranks_more():
    rankings = {"a": {0: 1, 1: 3}, "b": {1: 2}}
    # Doc 0: one first place. Doc 1: a third and a second place.
    assert rrf(rankings, k=1)[0][0] == 1  # 1/4 + 1/3 > 1/2
    assert rrf(rankings, k=0)[0][0] == 0  # 1/1 > 1/3 + 1/2
    assert rrf(rankings, k=60)[0][0] == 1  # large k flattens ranks; breadth wins


def test_weights_scale_a_rankers_contribution():
    rankings = {"text": {0: 1, 1: 2}, "image": {1: 1, 0: 2}}
    assert rrf(rankings)[0][1] == pytest.approx(rrf(rankings)[1][1])  # symmetric: a tie
    assert rrf(rankings, weights={"image": 2.0})[0][0] == 1
    assert rrf(rankings, weights={"image": 0.5})[0][0] == 0
    # Weight zero removes the ranker entirely, including documents only it retrieved.
    only_text = rrf({"text": {0: 1}, "image": {9: 1}}, weights={"image": 0.0})
    assert [doc for doc, _ in only_text] == [0]


def test_ties_are_broken_deterministically():
    rankings = {"a": {4: 1, 2: 2}, "b": {2: 1, 4: 2}}
    assert [doc for doc, _ in rrf(rankings)] == [2, 4]  # equal score, equal best rank: by id
    assert rrf(rankings) == rrf(dict(reversed(list(rankings.items()))))


def test_shared_ranks_are_supported():
    # Three windows of one slide tie for first place in the image ranker.
    fused = dict(rrf({"image": {10: 1, 11: 1, 12: 1, 20: 2}}))
    assert fused[10] == fused[11] == fused[12] == pytest.approx(1 / 61)
    assert fused[20] == pytest.approx(1 / 62)


def test_empty_input():
    assert rrf({}) == []
    assert rrf({"a": {}}) == []


# ---- dedupe ------------------------------------------------------------------------------


def test_dedupe_collapses_nearby_hits_in_the_same_lecture():
    lectures = ["lec01", "lec01", "lec01", "lec02"]
    starts = [100.0, 112.0, 400.0, 105.0]
    # Row 1 starts 12 s after row 0 in the same lecture: the same landing point.
    assert dedupe([0, 1, 2, 3], lectures, starts, window_s=20) == [0, 2, 3]


def test_dedupe_keeps_the_better_ranked_of_two_neighbours():
    lectures = ["lec01", "lec01"]
    starts = [100.0, 112.0]
    assert dedupe([1, 0], lectures, starts) == [1]


def test_dedupe_boundary_and_limit():
    lectures = ["lec01"] * 4
    starts = [0.0, 20.0, 40.1, 300.0]
    # Exactly 20 s apart is still a duplicate; row 2 is 40.1 s from row 0, so it stays.
    assert dedupe([0, 1, 2, 3], lectures, starts, window_s=20) == [0, 2, 3]
    assert dedupe([0, 1, 2, 3], lectures, starts, window_s=20, limit=2) == [0, 2]
    assert dedupe([], lectures, starts) == []


def test_dedupe_compares_against_kept_results_only():
    # 0 and 1 are duplicates; 2 is within 20 s of the dropped 1 but not of the kept 0.
    lectures = ["lec01"] * 3
    starts = [0.0, 15.0, 30.0]
    assert dedupe([0, 1, 2], lectures, starts, window_s=20) == [0, 2]


# ---- snippets ----------------------------------------------------------------------------


def test_short_text_is_returned_whole_with_highlights():
    snippet, spans = make_snippet("We hold out a validation set.", analyze("validation sets"))
    assert snippet == "We hold out a validation set."
    assert [snippet[a:b] for a, b in spans] == ["validation", "set"]


def test_long_text_is_cut_around_the_matches():
    filler = "the lecturer talks about administrative details for a while. " * 12
    text = filler + "Dropout randomly removes units during training. " + filler
    snippet, spans = make_snippet(text, analyze("dropout units"), width=160)

    assert len(snippet) <= 160 + 4  # the width plus ellipses
    assert snippet.startswith("… ") and snippet.endswith(" …")
    assert [snippet[a:b] for a, b in spans] == ["Dropout", "units"]
    assert "  " not in snippet


def test_snippet_without_matches_is_the_opening_of_the_text():
    text = "Gradient descent follows the slope downhill. " * 20
    snippet, spans = make_snippet(text, analyze("unrelated query"), width=100)
    assert snippet.startswith("Gradient descent")
    assert snippet.endswith(" …")
    assert spans == ()


def test_snippet_of_empty_text():
    assert make_snippet("   ", ["x"]) == ("", ())


def test_snippet_prefers_the_window_with_the_most_matches():
    lone = "An early mention of loss in passing. " + "filler words go here. " * 15
    cluster = "Validation loss rises while training loss keeps falling. "
    snippet, spans = make_snippet(
        lone + cluster + "more filler. " * 10, analyze("validation training loss"), width=120
    )
    assert "Validation loss rises" in snippet
    assert len(spans) >= 4
