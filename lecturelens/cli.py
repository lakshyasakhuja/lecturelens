"""Command-line interface.

    lecturelens demo   ./demo                 generate two synthetic lectures to try things on
    lecturelens index  ./videos               build or update the search index
    lecturelens search "why hold out data"    query it
    lecturelens info                          what is in the index
    lecturelens slides lec01.mp4              run slide detection alone, to tune masks

Indexing your own recordings with this CLI keeps everything on your machine: nothing is
uploaded, and the index links back to the local files.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from lecturelens.logs import configure_logging

app = typer.Typer(
    name="lecturelens",
    help="Semantic search for lecture recordings.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
console = Console()
log = logging.getLogger("lecturelens")


def setup_logging(verbose: bool = False) -> None:
    configure_logging(console, verbose)


BADGE_STYLE = {"said": "green", "on slide": "blue", "visual": "magenta"}


def fail(message: str) -> None:
    console.print(f"[red]error:[/red] {message}")
    raise typer.Exit(code=1)


def load_index(path: Path):
    from lecturelens.index import Index, IndexError_

    try:
        return Index.load(path)
    except IndexError_ as err:
        fail(str(err))


# --------------------------------------------------------------------------------------
# index
# --------------------------------------------------------------------------------------


@app.command()
def index(
    videos: Path = typer.Argument(..., help="Directory of lecture videos, one file per lecture."),
    out: Path = typer.Option(Path("index"), "--out", "-o", help="Where to write the index."),
    work: Path = typer.Option(Path("work"), help="Per-lecture intermediate results (the cache)."),
    catalog: Path | None = typer.Option(
        None,
        help="lectures.csv with titles, licences and video links. Default: config/lectures.csv if present.",
    ),
    masks: Path | None = typer.Option(
        None,
        help="masks.yaml with per-lecture slide-detection settings. Default: config/masks.yaml if present.",
    ),
    only: str | None = typer.Option(None, help="Comma-separated lecture ids to process."),
    asr_model: str = typer.Option("large-v3-turbo", help="faster-whisper model name."),
    device: str = typer.Option(
        "auto", help="auto, cuda, mps or cpu (encoders). ASR uses cuda or cpu."
    ),
    jargon_boost: bool = typer.Option(
        True, help="Hint speech recognition with terms read off the slides."
    ),
    force: bool = typer.Option(False, help="Ignore cached stages and redo everything."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Build or update the search index from a directory of lecture videos.

    Each video becomes one lecture, named after its file. Stages are cached per lecture, so
    re-running after adding a lecture, or after an interruption, only does what is missing.
    """
    setup_logging(verbose)
    from lecturelens.asr import AsrConfig
    from lecturelens.catalog import CatalogError, discover_videos, load_catalog, resolve_lectures
    from lecturelens.embed import EmbedConfig
    from lecturelens.ingest import FfmpegError
    from lecturelens.pipeline import Indexer, PipelineConfig

    catalog = catalog or (
        Path("config/lectures.csv") if Path("config/lectures.csv").is_file() else None
    )
    masks = masks or (Path("config/masks.yaml") if Path("config/masks.yaml").is_file() else None)
    try:
        found = discover_videos(videos)
        entries = load_catalog(catalog) if catalog else {}
    except (FileNotFoundError, CatalogError) as err:
        fail(str(err))
    if only:
        wanted = {item.strip() for item in only.split(",") if item.strip()}
        missing = wanted - set(found)
        if missing:
            fail(f"no video for lecture id(s): {', '.join(sorted(missing))}")
        found = {lecture_id: path for lecture_id, path in found.items() if lecture_id in wanted}
    if not found:
        fail(f"no video files found in {videos}")
    lectures = resolve_lectures(found, entries)
    uncatalogued = [lecture_id for lecture_id in found if lecture_id not in entries]
    if catalog and uncatalogued:
        log.info(
            "%d lecture(s) not in %s, using file names and local links: %s",
            len(uncatalogued), catalog, ", ".join(uncatalogued),
        )  # fmt: skip

    asr_device = device if device in ("cuda", "cpu") else "auto"
    config = PipelineConfig(
        asr=AsrConfig(model=asr_model, device=asr_device),
        embed=EmbedConfig(device=device),
        jargon_boost=jargon_boost,
        masks_path=masks,
    )
    try:
        built = Indexer(config, work_dir=work).build(lectures, found, out, force=force)
    except FfmpegError as err:
        fail(str(err))
    console.print(
        f"[green]Indexed[/green] {len(built.lectures)} lecture(s): {len(built)} moments, "
        f"{len(built.image)} keyframes -> [bold]{out}[/bold]"
    )


# --------------------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------------------


def _highlighted(snippet: str, spans) -> Text:
    text = Text(snippet)
    for start, end in spans:
        text.stylize("bold yellow", start, end)
    return text


@app.command()
def search(
    query: str = typer.Argument(..., help="What you are looking for, in plain English."),
    index_dir: Path = typer.Option(Path("index"), "--index", "-i", help="Index directory."),
    top_k: int = typer.Option(5, "--top-k", "-k", help="How many moments to return."),
    lecture: list[str] | None = typer.Option(
        None, "--lecture", "-l", help="Restrict to this lecture id (repeatable)."
    ),
    rankers: str = typer.Option(
        "bm25,dense,image", help="Comma-separated rankers to fuse: bm25, dense, image, ..."
    ),
    rerank: bool = typer.Option(False, help="Rescore the top results with the cross-encoder."),
    device: str = typer.Option("cpu", help="Device for the query encoders."),
    as_json: bool = typer.Option(False, "--json", help="Print results as JSON."),
) -> None:
    """Find the moments that answer a question."""
    setup_logging()
    from lecturelens.search import SearchConfig, Searcher

    loaded = load_index(index_dir)
    chosen = tuple(name.strip() for name in rankers.split(",") if name.strip())
    try:
        config = SearchConfig(rankers=chosen)
    except ValueError as err:
        fail(str(err))
    unknown = set(lecture or []) - set(loaded.lectures)
    if unknown:
        fail(
            f"unknown lecture id(s): {', '.join(sorted(unknown))}. Known: {', '.join(loaded.lectures)}"
        )

    searcher = Searcher(loaded, config, device=device)
    results = searcher.search(query, top_k=top_k, lectures=lecture, rerank=rerank)

    if as_json:
        payload = [
            {
                "rank": r.rank,
                "moment_id": r.moment_id,
                "lecture_id": r.lecture_id,
                "lecture_title": r.lecture_title,
                "t_start": r.t_start,
                "t_end": r.t_end,
                "timestamp": r.timestamp,
                "video_link": r.video_link,
                "score": r.score,
                "signals": r.signals,
                "badges": list(r.badges),
                "snippet": r.snippet,
                "thumb": str(r.thumb) if r.thumb else None,
            }
            for r in results
        ]
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    if not results:
        console.print("No matching moments.")
        return
    for r in results:
        header = Text.assemble(
            (f"{r.rank}. ", "bold"),
            (r.lecture_title, "bold"),
            ("  at  ", "dim"),
            (r.timestamp, "bold cyan"),
            "  ",
        )
        for badge in r.badges:
            header.append(f" {badge} ", style=f"reverse {BADGE_STYLE.get(badge, 'white')}")
            header.append(" ")
        console.print(header)
        if r.snippet:
            console.print(Text("   ") + _highlighted(r.snippet, r.highlights))
        signals = "  ".join(f"{name} #{rank}" for name, rank in r.signals.items())
        console.print(f"   [dim]{signals}[/dim]")
        if r.video_link:
            console.print(f"   [dim]{r.video_link}[/dim]", soft_wrap=True)
        console.print()


# --------------------------------------------------------------------------------------
# info
# --------------------------------------------------------------------------------------


@app.command()
def info(
    index_dir: Path = typer.Option(Path("index"), "--index", "-i", help="Index directory."),
) -> None:
    """Show what an index contains and how it was built."""
    from lecturelens.utils import format_timestamp

    loaded = load_index(index_dir)
    manifest = loaded.manifest
    console.print(
        f"[bold]{index_dir}[/bold]: {manifest['n_lectures']} lectures, "
        f"{manifest['n_moments']} moments, {manifest['n_keyframes']} keyframes, "
        f"{format_timestamp(manifest.get('total_duration_s', 0))} of video"
    )
    for role, name in manifest.get("models", {}).items():
        console.print(f"  {role:<6} {name}")
    console.print(f"  built  {manifest.get('created_utc', '?')}  (format {manifest.get('format')})")

    moments = loaded.moments
    for lecture_id, lecture in loaded.lectures.items():
        rows = moments[moments["lecture_id"] == lecture_id]
        spoken = int((rows["transcript"].str.len() > 0).sum())
        on_slide = int((rows["slide_text"].str.len() > 0).sum())
        console.print(f"\n[bold]{lecture_id}[/bold]  {lecture.title}")
        console.print(
            f"  {len(rows)} moments from {rows['image_idx'].nunique()} keyframes; "
            f"{spoken} with speech, {on_slide} with slide text",
            highlight=False,
        )
        if lecture.licence or lecture.source_url:
            console.print(
                f"  [dim]{' | '.join(filter(None, (lecture.licence, lecture.source_url)))}[/dim]",
                soft_wrap=True,
            )


# --------------------------------------------------------------------------------------
# slides
# --------------------------------------------------------------------------------------


@app.command()
def slides(
    video: Path = typer.Argument(..., help="A lecture video."),
    masks: Path | None = typer.Option(None, help="masks.yaml to apply."),
    lecture_id: str | None = typer.Option(
        None, help="Entry to use in masks.yaml. Default: file name."
    ),
    sheet: Path | None = typer.Option(
        None, help="Write a contact sheet of the detected keyframes to this JPEG."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the segments as JSON."),
) -> None:
    """Run slide detection on one video and show the segments.

    Use this to check masks and thresholds before indexing: the contact sheet shows one
    keyframe per detected slide, so a missed or spurious cut is visible at a glance.
    """
    from lecturelens import ingest
    from lecturelens.catalog import slugify
    from lecturelens.slides import detect_slides, load_slide_config
    from lecturelens.utils import format_timestamp

    try:
        config = load_slide_config(masks, lecture_id or slugify(video.stem))
        frames = ingest.read_gray_frames(video, fps=config.fps, size=config.size)
    except (ingest.FfmpegError, FileNotFoundError, ValueError) as err:
        fail(str(err))
    detection = detect_slides(frames, config)

    if as_json:
        typer.echo(
            json.dumps(
                {
                    "t_high": detection.t_high,
                    "t_low": detection.t_low,
                    "noise": detection.noise,
                    "segments": [seg.to_dict() for seg in detection.segments],
                },
                indent=2,
            )
        )
    else:
        console.print(
            f"{len(detection.segments)} segments  "
            f"(t_high {detection.t_high:.4f}, t_low {detection.t_low:.4f}, "
            f"noise floor {detection.noise:.4f}, {detection.merged} merged)"
        )
        table = Table(show_edge=False, pad_edge=False)
        for column in ("#", "start", "end", "length", "keyframe", "kind"):
            table.add_column(column, justify="right")
        for number, seg in enumerate(detection.segments, start=1):
            table.add_row(
                str(number),
                format_timestamp(seg.t_start),
                format_timestamp(seg.t_end),
                f"{seg.duration:.0f}s",
                format_timestamp(seg.keyframe_t),
                seg.kind,
            )
        console.print(table)

    if sheet:
        _contact_sheet(video, detection.segments, sheet)
        console.print(f"contact sheet -> [bold]{sheet}[/bold]")


def _contact_sheet(video: Path, segments, out: Path, columns: int = 5, width: int = 320) -> None:
    """A grid of keyframes, each labelled with its segment number and start time."""
    import tempfile

    from PIL import Image, ImageDraw, ImageFont

    from lecturelens import ingest
    from lecturelens.utils import format_timestamp

    with tempfile.TemporaryDirectory() as tmp:
        stills = ingest.extract_stills(
            video, [seg.keyframe_t for seg in segments], tmp, max_width=width
        )
        tiles = []
        for seg in segments:
            with Image.open(stills[seg.keyframe_t]) as still:
                tiles.append(still.convert("RGB"))
    height = max(tile.height for tile in tiles)
    label = 22
    rows = -(-len(tiles) // columns)
    canvas = Image.new("RGB", (columns * width, rows * (height + label)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=15)
    for i, (tile, seg) in enumerate(zip(tiles, segments, strict=True)):
        x, y = (i % columns) * width, (i // columns) * (height + label)
        canvas.paste(tile, (x, y + label))
        text = f"{i + 1}  {format_timestamp(seg.t_start)}" + (
            "  dynamic" if seg.kind == "dynamic" else ""
        )
        draw.text((x + 6, y + 2), text, fill=(20, 20, 20), font=font)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=88)


# --------------------------------------------------------------------------------------
# demo
# --------------------------------------------------------------------------------------


@app.command()
def demo(
    out: Path = typer.Argument(Path("demo"), help="Directory to create."),
) -> None:
    """Generate two short synthetic lectures, with ground truth, to try the pipeline on.

    Needs ffmpeg and a text-to-speech command (`say` on macOS, `espeak-ng` on Linux).
    """
    from lecturelens.demo import make_demo

    try:
        make_demo(out)
    except RuntimeError as err:
        fail(str(err))
    console.print(f"Demo lectures written to [bold]{out}[/bold]. Next:")
    console.print(
        f"  lecturelens index {out}/videos --catalog {out}/lectures.csv --out {out}/index --work {out}/work"
    )
    console.print(
        f'  lecturelens search "why do we hold out data we never train on" --index {out}/index'
    )


if __name__ == "__main__":
    app()
