"""Retrieval evaluation: the ablation table, per-type breakdown and failure analysis.

The benchmark is a file of queries, each with ground-truth intervals, a type and a split::

    {"qid": "q017", "query": "why do we hold out data we never train on",
     "type": "paraphrase", "lecture_id": "lec03",
     "gt": [[1312, 1420]], "split": "test", "author": "me"}

Every system in the ablation is evaluated on the same queries, over the whole library (a
student does not say which lecture the answer is in), and through the same final steps as
the served system: fusion, then deduplication into distinct landing points.

**Dev and test.** Anything with a knob (fusion weights, ``k``, the image ranker's
abstention floor) is chosen on the dev split by ``tune``. The test split is for reporting.
Every test evaluation is appended to ``results/test_runs.log`` with the commit it ran at,
so "test was only looked at once" is a claim the repository can back up.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from slideseek import metrics
from slideseek.index import Index
from slideseek.search import QueryEncoding, Searcher, dedupe, rrf
from slideseek.utils import format_timestamp, parse_timestamp

QUERY_TYPES = ("keyword", "paraphrase", "spoken", "visual")
TYPE_LABELS = {
    "keyword": "Keyword",
    "paraphrase": "Paraphrase",
    "spoken": "Spoken-only",
    "visual": "Visual",
}
SPLITS = ("dev", "test")


class QueryError(ValueError):
    """The query file is malformed or does not match the index."""


@dataclass(frozen=True)
class Query:
    qid: str
    query: str
    type: str
    lecture_id: str
    gt: tuple[tuple[float, float], ...]
    split: str
    author: str = "me"


def load_queries(path: str | Path, index: Index | None = None) -> list[Query]:
    """Read and validate ``queries.jsonl``. Timestamps may be seconds or ``m:ss`` strings."""
    path = Path(path)
    queries: list[Query] = []
    seen: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        where = f"{path}:{line_no}"
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as err:
            raise QueryError(f"{where}: not valid JSON ({err.msg})") from err
        missing = [k for k in ("qid", "query", "type", "lecture_id", "gt", "split") if k not in raw]
        if missing:
            raise QueryError(f"{where}: missing field(s) {missing}")
        if raw["qid"] in seen:
            raise QueryError(f"{where}: duplicate qid {raw['qid']!r}")
        seen.add(raw["qid"])
        if raw["type"] not in QUERY_TYPES:
            raise QueryError(f"{where}: type must be one of {QUERY_TYPES}, got {raw['type']!r}")
        if raw["split"] not in SPLITS:
            raise QueryError(f"{where}: split must be one of {SPLITS}, got {raw['split']!r}")
        if not str(raw["query"]).strip():
            raise QueryError(f"{where}: empty query")
        try:
            gt = tuple((parse_timestamp(t0), parse_timestamp(t1)) for t0, t1 in raw["gt"])
        except (TypeError, ValueError) as err:
            raise QueryError(f"{where}: gt must be a list of [start, end] pairs") from err
        if not gt or any(t1 <= t0 or t0 < 0 for t0, t1 in gt):
            raise QueryError(f"{where}: each gt interval needs 0 <= start < end")
        if index is not None:
            if raw["lecture_id"] not in index.lectures:
                raise QueryError(
                    f"{where}: lecture {raw['lecture_id']!r} is not in the index "
                    f"(has: {', '.join(index.lectures)})"
                )
            rows = index.moments[index.moments["lecture_id"] == raw["lecture_id"]]
            length = float(rows["t_end"].max()) if len(rows) else 0.0
            if any(t0 > length for t0, _ in gt):
                raise QueryError(
                    f"{where}: gt starts after the end of {raw['lecture_id']} "
                    f"({format_timestamp(length)})"
                )
        queries.append(
            Query(
                qid=str(raw["qid"]),
                query=str(raw["query"]).strip(),
                type=raw["type"],
                lecture_id=raw["lecture_id"],
                gt=gt,
                split=raw["split"],
                author=str(raw.get("author", "me")),
            )
        )
    if not queries:
        raise QueryError(f"{path}: no queries")
    return queries


# --------------------------------------------------------------------------------------
# Systems
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class System:
    key: str
    label: str
    rankers: tuple[str, ...]
    number: int | None = None  # row in the headline ablation table
    weights: Mapping[str, float] = field(default_factory=dict)
    rrf_k: int | None = None  # None: the searcher's configured value
    image_min_prob: float | None = None
    rerank: bool = False


ABLATION = (
    System("bm25_speech", "BM25, transcript only", ("bm25_speech",), number=1),
    System("bm25", "BM25, transcript + slide OCR", ("bm25",), number=2),
    System("dense_speech", "Dense, transcript only", ("dense_speech",), number=3),
    System("dense", "Dense, transcript + slide text", ("dense",), number=4),
    System("image", "Image only (SigLIP 2)", ("image",), number=5),
    System("hybrid", "Hybrid RRF (2 + 4 + 5)", ("bm25", "dense", "image"), number=6),
)
RERANKED = System(
    "hybrid_rerank", "Hybrid + reranker", ("bm25", "dense", "image"), number=7, rerank=True
)
EXTRAS = (
    System("text_hybrid", "Text-only hybrid RRF (2 + 4)", ("bm25", "dense")),
    System(
        "dense_fields",
        "Dense, separate slide and speech vectors (RRF)",
        ("dense_speech", "dense_slide"),
    ),
)
BASELINE = "bm25_speech"  # the stand-in for ordinary keyword transcript search
HEADLINE = "hybrid"


def tuned_system(settings: Mapping[str, Any]) -> System:
    """The hybrid with the settings ``tune`` chose on the dev split."""
    return System(
        "hybrid_tuned",
        "Hybrid, weights tuned on dev",
        ("bm25", "dense", "image"),
        weights=dict(settings.get("weights", {})),
        rrf_k=settings.get("rrf_k"),
        image_min_prob=settings.get("image_min_prob"),
    )


@dataclass
class SystemResult:
    system: System
    ranks: list[int | None]  # rank of the first relevant result, per query
    offsets: list[float | None]  # landing offset of that result, per query
    top: list[list[int]]  # the rows returned, per query

    def summary(self, queries: Sequence[Query]) -> dict[str, Any]:
        """Overall and per-type metrics with confidence intervals."""

        def block(indices: list[int]) -> dict[str, Any]:
            ranks = [self.ranks[i] for i in indices]
            offsets = [self.offsets[i] for i in indices if self.offsets[i] is not None]
            return {
                "n": len(ranks),
                "r1": metrics.recall_at_k(ranks, 1),
                "r5": metrics.recall_at_k(ranks, 5),
                "mrr10": metrics.mrr_at_k(ranks, 10),
                "r5_ci": metrics.bootstrap_ci(metrics.hits_at_k(ranks, 5)),
                "mrr10_ci": metrics.bootstrap_ci(metrics.reciprocal_ranks(ranks, 10)),
                "median_landing_offset_s": float(np.median(offsets)) if offsets else None,
            }

        everything = list(range(len(queries)))
        return {
            "key": self.system.key,
            "label": self.system.label,
            "number": self.system.number,
            "overall": block(everything),
            "by_type": {
                kind: block([i for i in everything if queries[i].type == kind])
                for kind in QUERY_TYPES
            },
        }


# --------------------------------------------------------------------------------------
# The evaluator
# --------------------------------------------------------------------------------------


class Evaluator:
    """Runs systems over a fixed set of queries, encoding each query only once."""

    def __init__(self, searcher: Searcher, queries: Sequence[Query], depth: int = 10) -> None:
        self.searcher = searcher
        self.queries = list(queries)
        self.depth = depth  # how many deduplicated results are scored (MRR@10 needs 10)
        self._encodings: dict[str, QueryEncoding] = {}
        self._ranks: dict[tuple[str, str, float | None], dict[int, int]] = {}
        moments = searcher.index.moments
        self._lecture_ids = moments["lecture_id"].tolist()
        self._starts = moments["t_start"].to_numpy(dtype=np.float64)
        self._ends = moments["t_end"].to_numpy(dtype=np.float64)

    def encoding(self, query: Query) -> QueryEncoding:
        if query.qid not in self._encodings:
            # Encode for every ranker family up front; systems then reuse the vectors.
            self._encodings[query.qid] = self.searcher.encode(
                query.query, rankers=("bm25", "dense", "image")
            )
        return self._encodings[query.qid]

    def ranker_ranks(
        self, ranker: str, query: Query, image_min_prob: float | None = None
    ) -> dict[int, int]:
        floor = image_min_prob if ranker == "image" else None
        key = (query.qid, ranker, floor)
        if key not in self._ranks:
            self._ranks[key] = self.searcher.rank(
                ranker, self.encoding(query), image_min_prob=floor
            )
        return self._ranks[key]

    def retrieve(self, system: System, query: Query) -> list[int]:
        """The rows a system returns for a query, as served: fused, then deduplicated."""
        per_ranker = {
            name: self.ranker_ranks(name, query, system.image_min_prob) for name in system.rankers
        }
        config = self.searcher.config
        fused = rrf(per_ranker, k=system.rrf_k or config.rrf_k, weights=system.weights)
        order = [row for row, _ in fused]
        if system.rerank and order:
            order, _ = self.searcher.rerank(query.query, order, dict(fused))
        return dedupe(order, self._lecture_ids, self._starts, config.dedupe_window_s, self.depth)

    def run(self, system: System, tolerance: float = 15.0) -> SystemResult:
        ranks, offsets, top = [], [], []
        for query in self.queries:
            rows = self.retrieve(system, query)
            spans = [(self._lecture_ids[r], self._starts[r], self._ends[r]) for r in rows]
            rank = metrics.first_relevant_rank(spans, query.lecture_id, query.gt, tolerance)
            ranks.append(rank)
            offsets.append(
                None if rank is None else metrics.landing_offset(spans[rank - 1][1:], query.gt)
            )
            top.append(rows)
        return SystemResult(system, ranks, offsets, top)

    def rescore(
        self, result: SystemResult, tolerance: float, criterion: str = "overlap"
    ) -> list[int | None]:
        """First-relevant ranks of an existing result under another tolerance or criterion."""
        ranks = []
        for query, rows in zip(self.queries, result.top, strict=True):
            spans = [(self._lecture_ids[r], self._starts[r], self._ends[r]) for r in rows]
            ranks.append(
                metrics.first_relevant_rank(spans, query.lecture_id, query.gt, tolerance, criterion)
            )
        return ranks

    def relevant_rows(
        self, query: Query, tolerance: float = 15.0, criterion: str = "overlap"
    ) -> list[int]:
        """Every moment in the index that would count as an answer to ``query``."""
        relevant = metrics.CRITERIA[criterion]
        return [
            row
            for row in range(len(self._starts))
            if self._lecture_ids[row] == query.lecture_id
            and relevant((self._starts[row], self._ends[row]), query.gt, tolerance)
        ]


# --------------------------------------------------------------------------------------
# Tuning on the dev split
# --------------------------------------------------------------------------------------


def tune(
    evaluator: Evaluator,
    weight_grid: Sequence[float] = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0),
    k_grid: Sequence[int] = (10, 30, 60, 100),
    floor_grid: Sequence[float] = (0.0, 0.02, 0.05, 0.1),
    tolerance: float = 15.0,
) -> dict[str, Any]:
    """Grid-search the hybrid's fusion settings. Run this on dev queries only.

    RRF scores are unchanged when every weight is scaled together, so the dense ranker's
    weight is fixed at 1 and the other two vary. The objective is MRR@10, with Recall@5 and
    then closeness to the untuned defaults as tie-breakers, because on a few dozen queries
    many settings tie and the plainest one is the least likely to be an accident.
    """
    best, best_key = None, None
    tried = 0
    for k in k_grid:
        for floor in floor_grid:
            for w_bm25 in weight_grid:
                for w_image in weight_grid:
                    weights = {"bm25": w_bm25, "dense": 1.0, "image": w_image}
                    system = System(
                        "candidate", "candidate", ("bm25", "dense", "image"),
                        weights=weights, rrf_k=k, image_min_prob=floor,
                    )  # fmt: skip
                    ranks = evaluator.run(system, tolerance).ranks
                    tried += 1
                    plainness = -(abs(w_bm25 - 1) + abs(w_image - 1) + abs(k - 60) / 60 + floor)
                    key = (metrics.mrr_at_k(ranks, 10), metrics.recall_at_k(ranks, 5), plainness)
                    if best_key is None or key > best_key:
                        best_key = key
                        best = {"weights": weights, "rrf_k": k, "image_min_prob": floor}
    assert best is not None and best_key is not None
    return {
        **best,
        "dev_mrr10": best_key[0],
        "dev_r5": best_key[1],
        "n_queries": len(evaluator.queries),
        "settings_tried": tried,
    }


# --------------------------------------------------------------------------------------
# Latency
# --------------------------------------------------------------------------------------


def measure_latency(
    searcher: Searcher, queries: Sequence[Query], repeats: int = 3
) -> dict[str, float]:
    """End-to-end ``search`` time per query in milliseconds, encoders included."""
    searcher.warm_up()
    timings = []
    for _ in range(repeats):
        for query in queries:
            started = time.perf_counter()
            searcher.search(query.query)
            timings.append((time.perf_counter() - started) * 1000.0)
    return {
        "p50_ms": float(np.percentile(timings, 50)),
        "p95_ms": float(np.percentile(timings, 95)),
        "mean_ms": float(np.mean(timings)),
        "n": len(timings),
        "device": searcher.embed_config.device,
    }


# --------------------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------------------


def _pct(value: float | None) -> str:
    return "n/a" if value is None or value != value else f"{100 * value:.1f}"


def ablation_markdown(
    summaries: Sequence[Mapping[str, Any]],
    split: str,
    tolerance: float,
    comparison: tuple[float, float, float] | None = None,
    latency: Mapping[str, float] | None = None,
) -> str:
    """The headline table: one row per system, overall metrics and Recall@5 by query type."""
    counts = summaries[0]["by_type"]
    n = summaries[0]["overall"]["n"]
    header = ["#", "System", "R@1", "R@5", "R@5 95% CI", "MRR@10"] + [
        f"{TYPE_LABELS[kind]} (n={counts[kind]['n']})" for kind in QUERY_TYPES
    ]
    lines = [
        f"# Retrieval ablation: {split} split",
        "",
        f"{n} queries. A result is relevant if it overlaps a ground-truth interval within "
        f"±{tolerance:g} s. All numbers are percentages; the per-type columns are Recall@5.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * 2 + ["---:"] * (len(header) - 2)) + "|",
    ]
    for item in summaries:
        overall = item["overall"]
        low, high = overall["r5_ci"]
        cells = [
            str(item["number"] or ""),
            item["label"],
            _pct(overall["r1"]),
            _pct(overall["r5"]),
            f"{_pct(low)} to {_pct(high)}",
            _pct(overall["mrr10"]),
        ] + [_pct(item["by_type"][kind]["r5"]) for kind in QUERY_TYPES]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    if comparison is not None:
        diff, low, high = comparison
        verdict = "excludes" if low > 0 or high < 0 else "includes"
        lines += [
            f"Hybrid minus keyword transcript search (system 6 minus system 1), Recall@5, "
            f"paired bootstrap over the same queries: {100 * diff:+.1f} points "
            f"(95% CI {100 * low:+.1f} to {100 * high:+.1f}), which {verdict} zero.",
            "",
        ]
    if latency is not None:
        lines += [
            f"Query latency, hybrid, end to end on {latency['device']} including both query "
            f"encoders: p50 {latency['p50_ms']:.0f} ms, p95 {latency['p95_ms']:.0f} ms "
            f"({latency['n']} searches).",
            "",
        ]
    return "\n".join(lines)


def tolerance_markdown(rows: Sequence[tuple[str, Mapping[str, tuple[float, float, float]]]]) -> str:
    """How the numbers move when the definition of "relevant" is tightened or loosened.

    ``rows`` are ``(system label, {column label: (R@1, R@5, MRR@10)})``; the columns appear
    in the order of the first row's mapping.
    """
    columns = list(rows[0][1])
    lines = [
        "# Sensitivity to the definition of relevance",
        "",
        "Recall@1 / Recall@5 / MRR@10 (percent).",
        "",
        "- **overlap ±N s** (the headline uses ±15 s): the result's span overlaps the "
        "ground-truth interval widened by N seconds on each side.",
        "- **lands ≤N s early**: stricter. The result *starts* inside the ground-truth "
        "interval or at most N seconds before it. A long moment that only ends where the "
        "answer begins passes the overlap test but not this one.",
        "",
        "| System | " + " | ".join(columns) + " |",
        "|---|" + "|".join(["---:"] * len(columns)) + "|",
    ]
    for label, by_column in rows:
        cells = [" / ".join(_pct(v) for v in by_column[column]) for column in columns]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def failures_markdown(
    evaluator: Evaluator, result: SystemResult, tolerance: float, k: int = 5, limit: int = 12
) -> str:
    """Queries the system missed, with what it returned and where each ranker put the answer.

    The per-ranker ranks of the answer are the starting point for a diagnosis: no BM25 rank
    means the words were never recognised or read; a deep dense rank with a good BM25 rank
    means the fusion buried it; no relevant moment at all means segmentation or the
    annotation is off. The cause line is left for a human to fill in after looking.
    """
    index = evaluator.searcher.index
    moments = index.moments
    lines = [f"# Failure analysis: {result.system.label}", ""]
    misses = [
        (query, rows)
        for query, rows, rank in zip(evaluator.queries, result.top, result.ranks, strict=True)
        if rank is None or rank > k
    ]
    lines.append(
        f"{len(misses)} of {len(evaluator.queries)} queries have no relevant moment in the "
        f"top {k}." + (f" Showing the first {limit}." if len(misses) > limit else "")
    )
    for query, rows in misses[:limit]:
        gt = ", ".join(f"{format_timestamp(a)} to {format_timestamp(b)}" for a, b in query.gt)
        lecture = index.lectures[query.lecture_id]
        lines += [
            "",
            f"## {query.qid}: {query.query}",
            "",
            f"- Type: {TYPE_LABELS[query.type]}",
            f"- Expected: {lecture.title}, {gt}  ({lecture.link_at(query.gt[0][0]) or 'no link'})",
        ]
        answers = evaluator.relevant_rows(query, tolerance)
        if not answers:
            lines.append(
                "- No moment in the index overlaps the expected interval: check the annotation "
                "and the slide segmentation of this lecture."
            )
        else:
            best = {
                name: min(evaluator.ranker_ranks(name, query).get(row, 10**9) for row in answers)
                for name in ("bm25", "dense", "image")
            }
            placed = ", ".join(
                f"{name} {'#' + str(rank) if rank < 10**9 else 'not retrieved'}"
                for name, rank in best.items()
            )
            thumb = moments["thumb"].iat[answers[0]]
            lines.append(f"- Where each ranker put the answer: {placed}")
            if thumb:
                lines.append(f"- Expected keyframe: `{thumb}`")
        lines.append("- Returned instead:")
        for position, row in enumerate(rows[:3], start=1):
            title = index.lectures[moments["lecture_id"].iat[row]].title
            text = " ".join(str(moments["transcript"].iat[row]).split())[:140]
            lines.append(
                f"  {position}. {title} at {format_timestamp(moments['t_start'].iat[row])}: "
                f"{text or '(no speech)'}"
            )
        lines.append("- Cause: _to be filled in_")
    return "\n".join(lines) + "\n"


def plot_per_type(summaries: Sequence[Mapping[str, Any]], path: str | Path, split: str) -> Path:
    """Grouped bar chart: Recall@5 by query type for each system."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # A colour-blind-safe palette (Okabe-Ito), one colour per system.
    palette = [
        "#999999",
        "#E69F00",
        "#56B4E9",
        "#0072B2",
        "#CC79A7",
        "#009E73",
        "#D55E00",
        "#000000",
    ]
    groups = list(QUERY_TYPES)
    width = 0.8 / len(summaries)
    figure, axis = plt.subplots(figsize=(10, 4.6))
    for position, item in enumerate(summaries):
        values = [100 * (item["by_type"][kind]["r5"] or 0) for kind in groups]
        offsets = [g + (position - (len(summaries) - 1) / 2) * width for g in range(len(groups))]
        axis.bar(
            offsets, values, width=width * 0.92, label=item["label"], color=palette[position % 8]
        )
    axis.set_xticks(range(len(groups)))
    axis.set_xticklabels(
        [f"{TYPE_LABELS[k]}\n(n={summaries[0]['by_type'][k]['n']})" for k in groups]
    )
    axis.set_ylabel("Recall@5 (%)")
    axis.set_ylim(0, 105)
    axis.set_title(f"Recall@5 by query type ({split} split)")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", linewidth=0.5, alpha=0.5)
    axis.set_axisbelow(True)
    axis.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def git_commit() -> str:
    """The current commit, with ``+dirty`` if the working tree has uncommitted changes."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def log_test_run(path: str | Path, n_queries: int, headline: Mapping[str, Any]) -> None:
    """Append a line recording that the test split was evaluated, and what it showed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    overall = headline["overall"]
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"{stamp}  commit {git_commit()}  {n_queries} queries  "
            f"{headline['label']}: R@1 {_pct(overall['r1'])}  R@5 {_pct(overall['r5'])}  "
            f"MRR@10 {_pct(overall['mrr10'])}\n"
        )
