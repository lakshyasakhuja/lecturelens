"""Slide-transition detection: precision and recall against hand annotations.

    python eval/eval_transitions.py --videos videos --tune-on lec01

Annotations live in ``eval/transitions/<lecture_id>.json``::

    {"lecture_id": "lec01", "transitions": ["0:42", "1:35", 187.5, ...]}

A transition is the moment a new slide appears. Bullet points or animation steps appearing
on the same slide are not transitions. Times may be seconds or ``m:ss`` as shown by a
video player; a prediction within the tolerance (2 s by default) of an annotation counts.

Compared detectors:

* the Slideseek detector, with default thresholds
* the same with thresholds tuned on one lecture (``--tune-on``); its scores on the *other*
  lectures are the ones to report, since the tuning lecture has been seen
* two ablations of it: plain mean absolute difference, and 1 - SSIM
* PySceneDetect's ``ContentDetector`` at its defaults, as an off-the-shelf baseline
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

# Make the scripts runnable from a checkout whether or not the package is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from typing import Any

import typer
from rich.console import Console

from slideseek import ingest
from slideseek.catalog import discover_videos
from slideseek.logs import configure_logging
from slideseek.metrics import TransitionScore, match_transitions
from slideseek.slides import SlideConfig, SlideDetector, load_slide_config
from slideseek.utils import parse_timestamp, read_json, write_json

console = Console()
configure_logging(console)

# Threshold grid for --tune-on. t_low is searched as a fraction of t_high so the two keep a
# sensible hysteresis band.
GRID = {
    "t_high": (0.002, 0.0025, 0.003, 0.0035, 0.004, 0.005, 0.006, 0.008, 0.012),
    "low_ratio": (0.3, 0.45, 0.6),
    "min_stable_s": (2.0, 3.0, 4.0),
    "one_sided_weight": (0.15, 0.25, 0.5, 1.0),
}


def load_annotations(directory: Path) -> dict[str, list[float]]:
    annotations = {}
    for path in sorted(directory.glob("*.json")):
        data = read_json(path)
        lecture_id = data.get("lecture_id", path.stem)
        annotations[lecture_id] = sorted(parse_timestamp(t) for t in data["transitions"])
    return annotations


def tune_thresholds(
    detector: SlideDetector, truth: list[float], tolerance: float
) -> dict[str, Any]:
    """Best settings on one lecture by F1; ties go to the setting nearest the defaults."""
    default_high, _ = SlideConfig().thresholds
    best, best_key = None, None
    for t_high, ratio, stable, weight in itertools.product(*GRID.values()):
        overrides = {
            "t_high": t_high,
            "t_low": round(t_high * ratio, 6),
            "min_stable_s": stable,
            "one_sided_weight": weight,
        }
        score = match_transitions(detector.detect(**overrides).transitions, truth, tolerance)
        closeness = -(
            abs(t_high - default_high) / default_high + abs(stable - 3.0) + abs(weight - 0.25)
        )
        key = (round(score.f1, 6), closeness)
        if best_key is None or key > best_key:
            best, best_key = overrides, key
    assert best is not None
    return best


def scenedetect_transitions(video: Path) -> list[float]:
    """Cut times from PySceneDetect's ContentDetector at default settings."""
    import logging

    from scenedetect import ContentDetector, detect

    logging.getLogger("pyscenedetect").setLevel(logging.WARNING)  # it resets its level on import
    scenes = detect(str(video), ContentDetector())
    # `seconds` replaced `get_seconds()` in PySceneDetect 0.7.
    return [
        float(start.seconds) if hasattr(start, "seconds") else start.get_seconds()
        for start, _ in scenes[1:]
    ]


def row(name: str, lecture_id: str, score: TransitionScore, note: str = "") -> dict[str, Any]:
    return {
        "detector": name,
        "lecture_id": lecture_id,
        "precision": score.precision,
        "recall": score.recall,
        "f1": score.f1,
        "tp": score.true_positives,
        "fp": score.false_positives,
        "fn": score.false_negatives,
        "spurious": list(score.spurious),
        "missed": list(score.missed),
        "note": note,
    }


def render(
    rows: list[dict[str, Any]], tolerance: float, tuned: dict | None, tune_on: str | None
) -> str:
    lines = [
        "# Slide transition detection",
        "",
        f"Precision and recall of detected slide transitions against hand annotations, "
        f"matched one to one within ±{tolerance:g} s.",
        "",
        "| Detector | Lecture | Precision | Recall | F1 | Found | Spurious | Missed |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in rows:
        lecture = item["lecture_id"] + (f" ({item['note']})" if item["note"] else "")
        lines.append(
            f"| {item['detector']} | {lecture} | {100 * item['precision']:.1f} | "
            f"{100 * item['recall']:.1f} | {100 * item['f1']:.1f} | {item['tp']} | "
            f"{item['fp']} | {item['fn']} |"
        )
    if tuned:
        settings = ", ".join(f"{k}={v:g}" for k, v in tuned.items())
        lines += [
            "",
            f"Tuned on `{tune_on}`: {settings}. Rows for that lecture are marked "
            '"tuning lecture" and are not a fair measure; report the others.',
        ]
    return "\n".join(lines) + "\n"


def main(
    videos: Path = typer.Option(..., help="Directory containing the annotated lecture videos."),
    annotations: Path = typer.Option(
        Path("eval/transitions"), help="Directory of annotation files."
    ),
    masks: Path | None = typer.Option(
        None, help="masks.yaml. Default: config/masks.yaml if present."
    ),
    tolerance: float = typer.Option(2.0, help="Matching tolerance in seconds."),
    tune_on: str | None = typer.Option(None, help="Lecture id to tune thresholds on."),
    baseline: bool = typer.Option(True, help="Also run PySceneDetect's ContentDetector."),
    out: Path = typer.Option(Path("results"), "--out", "-o"),
) -> None:
    """Score the slide detector, its ablations and PySceneDetect on annotated lectures."""
    masks = masks or (Path("config/masks.yaml") if Path("config/masks.yaml").is_file() else None)
    truth = load_annotations(annotations)
    if not truth:
        console.print(f"[red]error:[/red] no annotation files in {annotations}")
        raise typer.Exit(1)
    found = discover_videos(videos)
    missing = sorted(set(truth) - set(found))
    if missing:
        console.print(f"[red]error:[/red] no video in {videos} for: {', '.join(missing)}")
        raise typer.Exit(1)
    if tune_on and tune_on not in truth:
        console.print(f"[red]error:[/red] --tune-on {tune_on}: no annotation for that lecture")
        raise typer.Exit(1)

    detectors: dict[str, SlideDetector] = {}
    variants: dict[str, dict[str, SlideDetector]] = {"ssim": {}}
    for lecture_id in truth:
        with console.status(f"decoding {lecture_id}"):
            config = load_slide_config(masks, lecture_id)
            frames = ingest.read_gray_frames(found[lecture_id], fps=config.fps, size=config.size)
            detectors[lecture_id] = SlideDetector(frames, config)
            # SSIM changes the distance itself, so it needs its own preprocessing pass.
            ssim_config = SlideConfig(metric="ssim", masks=config.masks, camera=config.camera)
            variants["ssim"][lecture_id] = SlideDetector(frames, ssim_config)

    tuned = None
    if tune_on:
        with console.status(f"tuning on {tune_on}"):
            tuned = tune_thresholds(detectors[tune_on], truth[tune_on], tolerance)

    rows = []
    for lecture_id, annotated in truth.items():
        detector = detectors[lecture_id]

        def score(transitions, annotated=annotated):
            return match_transitions(transitions, annotated, tolerance)

        rows.append(row("Slideseek", lecture_id, score(detector.detect().transitions)))
        if tuned:
            note = "tuning lecture" if lecture_id == tune_on else "held out"
            rows.append(
                row(
                    "Slideseek, tuned",
                    lecture_id,
                    score(detector.detect(**tuned).transitions),
                    note,
                )
            )
        rows.append(
            row(
                "  ablation: plain mean absolute difference",
                lecture_id,
                score(detector.detect(one_sided_weight=1.0).transitions),
            )
        )
        rows.append(
            row(
                "  ablation: 1 - SSIM",
                lecture_id,
                score(variants["ssim"][lecture_id].detect().transitions),
            )
        )
        if baseline:
            with console.status(f"PySceneDetect on {lecture_id}"):
                rows.append(
                    row(
                        "PySceneDetect ContentDetector (defaults)",
                        lecture_id,
                        score(scenedetect_transitions(found[lecture_id])),
                    )
                )

    report = render(rows, tolerance, tuned, tune_on)
    out.mkdir(parents=True, exist_ok=True)
    (out / "transitions.md").write_text(report, encoding="utf-8")
    write_json(out / "transitions.json", {"tolerance_s": tolerance, "tuned": tuned, "rows": rows})
    console.print(report)
    console.print(f"written to [bold]{out / 'transitions.md'}[/bold]")


if __name__ == "__main__":
    typer.run(main)
