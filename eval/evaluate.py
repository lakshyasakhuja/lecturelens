"""Retrieval benchmark: the ablation table, per-type chart, tolerance sweep and failures.

    # 1. Tune fusion settings on the dev split (never on test)
    python eval/evaluate.py --split dev --tune

    # 2. Report on the test split, once, including the tuned system
    python eval/evaluate.py --split test --tuned results/dev/tuned.json --latency

Outputs go to ``results/<split>/``: ``ablation.md``, ``metrics.json``, ``per_type.png``,
``tolerance.md`` and ``failures.md``. Test runs are also appended to
``results/test_runs.log``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the scripts runnable from a checkout whether or not the package is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer
from rich.console import Console

from slideseek import metrics
from slideseek.evaluation import (
    ABLATION,
    BASELINE,
    EXTRAS,
    HEADLINE,
    RERANKED,
    Evaluator,
    QueryError,
    ablation_markdown,
    failures_markdown,
    load_queries,
    log_test_run,
    measure_latency,
    plot_per_type,
    tolerance_markdown,
    tune,
    tuned_system,
)
from slideseek.index import Index, InvalidIndex
from slideseek.logs import configure_logging
from slideseek.search import Searcher
from slideseek.utils import read_json, write_json

console = Console()
configure_logging(console)


def main(
    index_dir: Path = typer.Option(Path("index"), "--index", "-i", help="Index directory."),
    queries_path: Path = typer.Option(Path("eval/queries.jsonl"), "--queries", "-q"),
    split: str = typer.Option("dev", help="dev, test or all. Tune on dev; report on test."),
    out: Path = typer.Option(Path("results"), "--out", "-o", help="Results directory."),
    tolerance: float = typer.Option(15.0, help="Relevance tolerance in seconds."),
    do_tune: bool = typer.Option(False, "--tune", help="Grid-search fusion settings (dev only)."),
    tuned: Path | None = typer.Option(None, help="tuned.json from a dev run, to evaluate here."),
    rerank: bool = typer.Option(False, help="Also evaluate the hybrid with the cross-encoder."),
    latency: bool = typer.Option(False, help="Measure end-to-end query latency."),
    device: str = typer.Option("cpu", help="Device for the query encoders."),
) -> None:
    """Evaluate every system in the ablation on one split of the query set."""
    if split not in ("dev", "test", "all"):
        raise typer.BadParameter("split must be dev, test or all")
    if do_tune and split != "dev":
        raise typer.BadParameter("--tune only runs on the dev split; tuning on test is cheating")

    try:
        index = Index.load(index_dir)
        queries = load_queries(queries_path, index)
    except (InvalidIndex, QueryError, FileNotFoundError) as err:
        console.print(f"[red]error:[/red] {err}")
        raise typer.Exit(1) from err
    if split != "all":
        queries = [q for q in queries if q.split == split]
    if not queries:
        console.print(f"[red]error:[/red] no queries in the {split} split")
        raise typer.Exit(1)

    out_dir = out / split
    searcher = Searcher(index, device=device)
    evaluator = Evaluator(searcher, queries)
    console.print(f"{len(queries)} {split} queries over {len(index)} moments")

    systems = list(ABLATION)
    if rerank:
        systems.append(RERANKED)
    systems += list(EXTRAS)

    if do_tune:
        with console.status("tuning fusion settings on dev"):
            settings = tune(evaluator, tolerance=tolerance)
        write_json(out_dir / "tuned.json", settings)
        console.print(
            f"tuned on {settings['n_queries']} dev queries over {settings['settings_tried']} "
            f"settings: weights {settings['weights']}, k={settings['rrf_k']}, "
            f"image floor {settings['image_min_prob']} -> {out_dir / 'tuned.json'}"
        )
        tuned = out_dir / "tuned.json"
    if tuned is not None:
        systems.append(tuned_system(read_json(tuned)))

    results = {}
    for system in systems:
        with console.status(f"evaluating: {system.label}"):
            results[system.key] = evaluator.run(system, tolerance)
    summaries = [results[system.key].summary(queries) for system in systems]

    # Hybrid against the keyword-transcript baseline, paired over the same queries.
    comparison = metrics.paired_bootstrap(
        metrics.hits_at_k(results[HEADLINE].ranks, 5),
        metrics.hits_at_k(results[BASELINE].ranks, 5),
    )
    timing = measure_latency(searcher, queries) if latency else None

    out_dir.mkdir(parents=True, exist_ok=True)
    table = ablation_markdown(summaries, split, tolerance, comparison, timing)
    (out_dir / "ablation.md").write_text(table, encoding="utf-8")

    variants = [
        ("overlap ±5 s", 5.0, "overlap"),
        ("overlap ±15 s", 15.0, "overlap"),
        ("overlap ±30 s", 30.0, "overlap"),
        ("lands ≤15 s early", 15.0, "landing"),
        ("lands ≤30 s early", 30.0, "landing"),
    ]
    sweep = []
    for key in (BASELINE, HEADLINE):
        by_variant = {}
        for label, tol, criterion in variants:
            ranks = evaluator.rescore(results[key], tol, criterion)
            by_variant[label] = (
                metrics.recall_at_k(ranks, 1),
                metrics.recall_at_k(ranks, 5),
                metrics.mrr_at_k(ranks, 10),
            )
        sweep.append((results[key].system.label, by_variant))
    (out_dir / "tolerance.md").write_text(tolerance_markdown(sweep), encoding="utf-8")
    (out_dir / "failures.md").write_text(
        failures_markdown(evaluator, results[HEADLINE], tolerance), encoding="utf-8"
    )
    plot_per_type([s for s in summaries if s["number"]], out_dir / "per_type.png", split)
    write_json(
        out_dir / "metrics.json",
        {
            "split": split,
            "n_queries": len(queries),
            "tolerance_s": tolerance,
            "index": {k: index.manifest.get(k) for k in ("n_moments", "n_lectures", "models")},
            "systems": summaries,
            "hybrid_minus_baseline_r5": dict(
                zip(("diff", "ci_low", "ci_high"), comparison, strict=True)
            ),
            "latency": timing,
            "per_query": {
                system.key: dict(
                    zip((q.qid for q in queries), results[system.key].ranks, strict=True)
                )
                for system in systems
            },
        },
    )
    if split == "test":
        headline = next(s for s in summaries if s["key"] == HEADLINE)
        log_test_run(out / "test_runs.log", len(queries), headline)

    console.print()
    console.print(table)
    console.print(f"written to [bold]{out_dir}[/bold]")


if __name__ == "__main__":
    typer.run(main)
