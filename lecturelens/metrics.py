"""Metrics, as pure functions.

Retrieval
---------
A query has one or more ground-truth intervals ``[t0, t1]`` in one lecture: the stretch of
video that answers it. A returned moment is **relevant** if it is in that lecture and its
own span overlaps a ground-truth interval widened by a tolerance on each side (15 s by
default). The tolerance says: a result counts if it lands the student close enough to see
the explanation, even if the moment boundary and the annotator's boundary differ a little.

Overlap is lenient in one way worth knowing about: a moment that merely *ends* where the
answer begins also overlaps it, however long that moment is. So a second, stricter
criterion is reported alongside: **landing**. A result lands if it starts inside the answer
or at most ``tolerance`` seconds before it, which is the literal version of "close enough
to see the explanation".

From the rank of the first relevant result:

* ``Recall@k``: share of queries with a relevant moment in the top ``k``. (With one answer
  per query this is the same as hit rate or success@k.)
* ``MRR@k``: mean of ``1 / rank``, counting 0 when nothing relevant is in the top ``k``.

Because the benchmark is small, every aggregate comes with a bootstrap confidence
interval, and systems are compared with a paired bootstrap over the same queries.

Slide transitions
-----------------
Predicted and annotated transition times are matched one to one within a tolerance
(2 s by default); precision, recall and F1 follow from the matches.

Speech recognition
------------------
Word error rate after a light normalisation that removes differences which are not
recognition errors (case, punctuation, hyphenation).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

Interval = tuple[float, float]

# --------------------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------------------


def is_relevant(span: Interval, truth: Sequence[Interval], tolerance: float = 15.0) -> bool:
    """Does ``span`` overlap any ground-truth interval widened by ``tolerance`` seconds?"""
    start, end = span
    return any(start <= t1 + tolerance and end >= t0 - tolerance for t0, t1 in truth)


def lands(span: Interval, truth: Sequence[Interval], tolerance: float = 15.0) -> bool:
    """Does ``span`` start inside a ground-truth interval, or at most ``tolerance`` s before it?"""
    start = span[0]
    return any(t0 - tolerance <= start < t1 for t0, t1 in truth)


CRITERIA = {"overlap": is_relevant, "landing": lands}


def first_relevant_rank(
    results: Sequence[tuple[str, float, float]],
    lecture_id: str,
    truth: Sequence[Interval],
    tolerance: float = 15.0,
    criterion: str = "overlap",
) -> int | None:
    """1-based rank of the first relevant result, or ``None``.

    ``results`` are ``(lecture_id, t_start, t_end)`` in ranked order. ``criterion`` is
    ``"overlap"`` (the headline definition) or ``"landing"`` (the stricter one).
    """
    relevant = CRITERIA[criterion]
    for rank, (result_lecture, start, end) in enumerate(results, start=1):
        if result_lecture == lecture_id and relevant((start, end), truth, tolerance):
            return rank
    return None


def hits_at_k(ranks: Sequence[int | None], k: int) -> np.ndarray:
    """Per query: 1.0 if a relevant result is in the top ``k``, else 0.0."""
    return np.array([1.0 if rank is not None and rank <= k else 0.0 for rank in ranks])


def reciprocal_ranks(ranks: Sequence[int | None], k: int) -> np.ndarray:
    """Per query: ``1 / rank`` if a relevant result is in the top ``k``, else 0.0."""
    return np.array([1.0 / rank if rank is not None and rank <= k else 0.0 for rank in ranks])


def recall_at_k(ranks: Sequence[int | None], k: int) -> float:
    return float(hits_at_k(ranks, k).mean()) if len(ranks) else float("nan")


def mrr_at_k(ranks: Sequence[int | None], k: int = 10) -> float:
    return float(reciprocal_ranks(ranks, k).mean()) if len(ranks) else float("nan")


def landing_offset(span: Interval, truth: Sequence[Interval]) -> float:
    """Seconds between where a result starts and the nearest answer.

    Zero means the result starts inside an answer. Otherwise it is how long the student
    watches before the answer begins, or, for a result that starts after the answer has
    ended, how far past it they landed. Overlap-based relevance is lenient towards long
    moments; this number shows what that leniency costs.
    """
    start = span[0]
    return min(max(0.0, t0 - start) if start <= t1 else start - t1 for t0, t1 in truth)


def bootstrap_ci(
    values: Sequence[float], level: float = 0.95, n_resamples: int = 5000, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of per-query ``values``."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(values), size=(n_resamples, len(values)))
    means = values[samples].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def paired_bootstrap(
    a: Sequence[float], b: Sequence[float], n_resamples: int = 5000, seed: int = 0
) -> tuple[float, float, float]:
    """Compare two systems on the same queries.

    Returns ``(mean difference a - b, CI low, CI high)`` at 95%. Queries are resampled
    together, so the comparison is paired: a query that is hard for both systems does not
    blur the difference between them. If the interval excludes 0, the difference is unlikely
    to be an accident of which queries happened to be in the set.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired comparison needs the same queries for both systems")
    if len(a) == 0:
        return float("nan"), float("nan"), float("nan")
    diff = a - b
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(diff), size=(n_resamples, len(diff)))
    means = diff[samples].mean(axis=1)
    return float(diff.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


# --------------------------------------------------------------------------------------
# Slide transitions
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TransitionScore:
    true_positives: int
    false_positives: int
    false_negatives: int
    matches: tuple[tuple[float, float], ...]  # (predicted, annotated) pairs
    spurious: tuple[float, ...]  # predictions with no annotation nearby
    missed: tuple[float, ...]  # annotations with no prediction nearby

    @property
    def precision(self) -> float:
        found = self.true_positives + self.false_positives
        return self.true_positives / found if found else 1.0

    @property
    def recall(self) -> float:
        actual = self.true_positives + self.false_negatives
        return self.true_positives / actual if actual else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def match_transitions(
    predicted: Sequence[float], annotated: Sequence[float], tolerance: float = 2.0
) -> TransitionScore:
    """Match predicted to annotated transition times, one to one, within ``tolerance``.

    Both lists are swept in time order and the two earliest unmatched times are paired
    whenever they are within the tolerance; otherwise the earlier of the two is left
    unmatched. On a line this greedy sweep finds a maximum matching, so a detector is never
    credited twice for one transition, nor denied a match it could have had.
    """
    predicted = sorted(float(t) for t in predicted)
    annotated = sorted(float(t) for t in annotated)
    matches, spurious, missed = [], [], []
    i = j = 0
    while i < len(predicted) and j < len(annotated):
        gap = predicted[i] - annotated[j]
        if abs(gap) <= tolerance:
            matches.append((predicted[i], annotated[j]))
            i += 1
            j += 1
        elif gap < 0:
            spurious.append(predicted[i])
            i += 1
        else:
            missed.append(annotated[j])
            j += 1
    spurious.extend(predicted[i:])
    missed.extend(annotated[j:])
    return TransitionScore(
        true_positives=len(matches),
        false_positives=len(spurious),
        false_negatives=len(missed),
        matches=tuple(matches),
        spurious=tuple(spurious),
        missed=tuple(missed),
    )


# --------------------------------------------------------------------------------------
# Speech recognition
# --------------------------------------------------------------------------------------

_DASHES = re.compile(r"[-‐-―]")
_NOT_WORD = re.compile(r"[^a-z0-9' ]+")
_EDGE_APOSTROPHE = re.compile(r"(?<![a-z0-9])'|'(?![a-z0-9])")


def normalize_for_wer(text: str) -> str:
    """Lower-case, drop punctuation and split hyphenated words.

    These are differences of transcription convention, not recognition errors: "Held-out
    data." and "held out data" should compare equal. Apostrophes inside words are kept, so
    "don't" stays one word. Spelling variants ("generalisation" / "generalization") and
    number formats ("five" / "5") are *not* normalised; write the reference the way the
    recogniser is expected to write.
    """
    text = _DASHES.sub(" ", text.lower().replace("’", "'"))
    text = _NOT_WORD.sub(" ", text)
    text = _EDGE_APOSTROPHE.sub(" ", text)
    return " ".join(text.split())


@dataclass(frozen=True)
class WerScore:
    errors: int  # substitutions + deletions + insertions
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int

    @property
    def wer(self) -> float:
        return self.errors / self.reference_words if self.reference_words else float("nan")

    def __add__(self, other: WerScore) -> WerScore:
        return WerScore(
            self.errors + other.errors,
            self.substitutions + other.substitutions,
            self.deletions + other.deletions,
            self.insertions + other.insertions,
            self.reference_words + other.reference_words,
        )


def word_errors(reference: str, hypothesis: str) -> WerScore:
    """Word-level edit distance between normalised reference and hypothesis.

    Scores add, so the error rate over several excerpts is pooled (total errors over total
    reference words) rather than an average of per-excerpt rates.
    """
    ref = normalize_for_wer(reference).split()
    hyp = normalize_for_wer(hypothesis).split()
    # Standard dynamic programme, tracking the operation counts along the best path.
    previous = [(j, 0, 0, j) for j in range(len(hyp) + 1)]  # (cost, sub, del, ins)
    for i in range(1, len(ref) + 1):
        current = [(i, 0, i, 0)]
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                current.append(previous[j - 1])
                continue
            sub, dele, ins = previous[j - 1], previous[j], current[j - 1]
            best = min(sub, dele, ins, key=lambda cell: cell[0])
            if best is sub:
                current.append((sub[0] + 1, sub[1] + 1, sub[2], sub[3]))
            elif best is dele:
                current.append((dele[0] + 1, dele[1], dele[2] + 1, dele[3]))
            else:
                current.append((ins[0] + 1, ins[1], ins[2], ins[3] + 1))
        previous = current
    cost, subs, dels, inss = previous[-1]
    return WerScore(cost, subs, dels, inss, len(ref))
