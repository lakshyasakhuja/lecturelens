"""Speech recognition: word error rate on hand-corrected excerpts.

    python eval/eval_asr.py --work work                    # score the indexed transcripts
    python eval/eval_asr.py --work work --compare-jargon   # and measure the jargon boost

Excerpts are listed in ``eval/asr_ref/excerpts.jsonl``, one per line::

    {"lecture_id": "lec02", "t0": "12:00", "t1": "17:00", "reference": "lec02_excerpt.txt"}

``reference`` is a text file next to it holding the corrected transcript of that stretch.
The hypothesis is every recognised word whose midpoint falls inside ``[t0, t1)``.

``--compare-jargon`` re-transcribes each excerpt twice from the lecture's audio, once with
no prompt and once with the glossary built from that lecture's slide OCR, and reports both.
This is the measurement behind the claim that slide text helps Whisper spell technical
terms; it needs the speech model, so it is slower than the default mode.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Make the scripts runnable from a checkout whether or not the package is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from typing import Any

import typer
from rich.console import Console

from lecturelens import asr, ingest
from lecturelens.logs import configure_logging
from lecturelens.metrics import WerScore, word_errors
from lecturelens.ocr import SlideText
from lecturelens.utils import format_timestamp, parse_timestamp, read_json, write_json

console = Console()
configure_logging(console)
PADDING_S = 5.0  # context on each side of an excerpt when it is re-transcribed


def load_excerpts(refs: Path) -> list[dict[str, Any]]:
    manifest = refs / "excerpts.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"{manifest} not found")
    excerpts = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        excerpts.append(
            {
                "lecture_id": item["lecture_id"],
                "t0": parse_timestamp(item["t0"]),
                "t1": parse_timestamp(item["t1"]),
                "reference": (refs / item["reference"]).read_text(encoding="utf-8"),
            }
        )
    return excerpts


def hypothesis(words: list[asr.Word], t0: float, t1: float) -> str:
    return " ".join(w.text for w in asr.words_between(words, t0, t1))


def retranscribe(work: Path, excerpt: dict[str, Any], model, hotwords: str | None) -> str:
    """Transcribe one excerpt from the lecture's audio, with a little context either side."""
    audio = work / excerpt["lecture_id"] / "audio.wav"
    start = max(0.0, excerpt["t0"] - PADDING_S)
    length = excerpt["t1"] + PADDING_S - start
    with tempfile.TemporaryDirectory() as tmp:
        clip = ingest.extract_audio(audio, Path(tmp) / "clip.wav", start=start, duration=length)
        segments = asr.transcribe(clip, model=model, hotwords=hotwords)
    words = [
        asr.Word(w.text, w.start + start, w.end + start, w.prob) for w in asr.iter_words(segments)
    ]
    return hypothesis(words, excerpt["t0"], excerpt["t1"])


def lecture_hotwords(work: Path, lecture_id: str) -> str:
    pages = read_json(work / lecture_id / "ocr.json")
    return asr.build_hotwords(SlideText.from_dict(page).text for page in pages.values())


def render(rows: list[dict[str, Any]], totals: dict[str, WerScore], model: str) -> str:
    columns = list(totals)
    lines = [
        "# Speech recognition word error rate",
        "",
        f"Model: `{model}`. Word error rate (%) after lower-casing and removing punctuation "
        "and hyphenation. The total is pooled: all errors over all reference words.",
        "",
        "| Lecture | Excerpt | Reference words | " + " | ".join(columns) + " |",
        "|---|---|---:|" + "|".join(["---:"] * len(columns)) + "|",
    ]
    for item in rows:
        cells = [f"{100 * item['scores'][c].wer:.2f}" for c in columns]
        lines.append(
            f"| {item['lecture_id']} | {format_timestamp(item['t0'])} to "
            f"{format_timestamp(item['t1'])} | {item['scores'][columns[0]].reference_words} | "
            + " | ".join(cells)
            + " |"
        )
    total_words = next(iter(totals.values())).reference_words
    lines.append(
        f"| **Total** | | {total_words} | "
        + " | ".join(f"**{100 * totals[c].wer:.2f}**" for c in columns)
        + " |"
    )
    lines.append("")
    for name, score in totals.items():
        lines.append(
            f"{name}: {score.substitutions} substitutions, {score.deletions} deletions, "
            f"{score.insertions} insertions."
        )
    return "\n".join(lines) + "\n"


def main(
    work: Path = typer.Option(Path("work"), help="Work directory written by `lecturelens index`."),
    refs: Path = typer.Option(
        Path("eval/asr_ref"), help="Directory with excerpts.jsonl and references."
    ),
    compare_jargon: bool = typer.Option(
        False, help="Re-transcribe each excerpt with and without the slide glossary."
    ),
    asr_model: str = typer.Option("large-v3-turbo", help="Model for --compare-jargon."),
    out: Path = typer.Option(Path("results"), "--out", "-o"),
) -> None:
    """Word error rate of the indexed transcripts against hand-corrected excerpts."""
    try:
        excerpts = load_excerpts(refs)
    except FileNotFoundError as err:
        console.print(f"[red]error:[/red] {err}")
        raise typer.Exit(1) from err
    if not excerpts:
        console.print(f"[red]error:[/red] {refs / 'excerpts.jsonl'} lists no excerpts")
        raise typer.Exit(1)

    model = asr.load_model(asr.AsrConfig(model=asr_model)) if compare_jargon else None
    rows = []
    for excerpt in excerpts:
        lecture_id = excerpt["lecture_id"]
        transcript = work / lecture_id / "asr.jsonl"
        if not transcript.is_file():
            console.print(f"[red]error:[/red] {transcript} not found; index {lecture_id} first")
            raise typer.Exit(1)
        words = list(asr.iter_words(asr.load_segments(transcript)))
        scores = {
            "Indexed transcript": word_errors(
                excerpt["reference"], hypothesis(words, excerpt["t0"], excerpt["t1"])
            )
        }
        if compare_jargon:
            with console.status(f"re-transcribing {lecture_id}"):
                plain = retranscribe(work, excerpt, model, None)
                boosted = retranscribe(work, excerpt, model, lecture_hotwords(work, lecture_id))
            scores["No prompt"] = word_errors(excerpt["reference"], plain)
            scores["Slide glossary prompt"] = word_errors(excerpt["reference"], boosted)
        rows.append({**{k: excerpt[k] for k in ("lecture_id", "t0", "t1")}, "scores": scores})

    totals: dict[str, WerScore] = {}
    for item in rows:
        for name, score in item["scores"].items():
            totals[name] = totals[name] + score if name in totals else score

    indexed_with = read_json(work / excerpts[0]["lecture_id"] / "asr.meta.json").get(
        "model", asr_model
    )
    report = render(rows, totals, indexed_with)
    out.mkdir(parents=True, exist_ok=True)
    (out / "asr.md").write_text(report, encoding="utf-8")
    write_json(
        out / "asr.json",
        {
            "model": indexed_with,
            "total": {
                name: {"wer": s.wer, "errors": s.errors, "reference_words": s.reference_words}
                for name, s in totals.items()
            },
            "excerpts": [
                {
                    **{k: item[k] for k in ("lecture_id", "t0", "t1")},
                    **{name: s.wer for name, s in item["scores"].items()},
                }
                for item in rows
            ],
        },
    )
    console.print(report)
    console.print(f"written to [bold]{out / 'asr.md'}[/bold]")


if __name__ == "__main__":
    typer.run(main)
