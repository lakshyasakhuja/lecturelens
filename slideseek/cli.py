"""Command-line interface.

    slideseek demo   ./demo                 generate two synthetic lectures to try things on
    slideseek index  ./videos               build or update the search index
    slideseek search "why hold out data"    query it
    slideseek serve                         open the search page in a browser
    slideseek info                          what is in the index
    slideseek slides lec01.mp4              run slide detection alone, to tune masks
    slideseek push   user/my-index          publish an index as a Hugging Face dataset
    slideseek pull   user/my-index          fetch a published index

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

from slideseek.logs import configure_logging

app = typer.Typer(
    name="slideseek",
    help="Find the moment a slide was shown in a lecture recording, from a question or a picture.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
console = Console()
log = logging.getLogger("slideseek")


def setup_logging(verbose: bool = False) -> None:
    configure_logging(console, verbose)


BADGE_STYLE = {"said": "green", "on slide": "blue", "visual": "magenta"}


def fail(message: str) -> None:
    console.print(f"[red]error:[/red] {message}")
    raise typer.Exit(code=1)


def load_index(path: Path):
    from slideseek.index import Index, InvalidIndex

    try:
        return Index.load(path)
    except InvalidIndex as err:
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
    device: str = typer.Option("auto", help="Device for the encoders: auto, cuda, mps or cpu."),
    asr_device: str = typer.Option(
        "auto",
        help="Device for speech recognition: auto (cuda if present, else cpu), cuda, mps or cpu.",
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
    from slideseek.asr import AsrConfig
    from slideseek.catalog import CatalogError, discover_videos, load_catalog, resolve_lectures
    from slideseek.embed import EmbedConfig
    from slideseek.ingest import FfmpegError
    from slideseek.pipeline import Indexer, PipelineConfig

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
    query: str | None = typer.Argument(None, help="What you are looking for, in plain English."),
    picture: Path | None = typer.Option(
        None, "--picture", "-p", help="Search with a picture of a slide instead of a question."
    ),
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
    """Find the moments that answer a question, or where the slide in a picture was shown."""
    setup_logging()
    from slideseek.search import SearchConfig, Searcher

    if (query is None) == (picture is None):
        fail("give either a question or --picture, not both")
    if picture is not None and not picture.is_file():
        fail(f"picture not found: {picture}")
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
    if picture is not None:
        from PIL import Image, ImageOps

        with Image.open(picture) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        results = searcher.search_picture(image, top_k=top_k, lectures=lecture)
    else:
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
    from slideseek.utils import format_timestamp

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
    from slideseek import ingest
    from slideseek.catalog import slugify
    from slideseek.slides import detect_slides, load_slide_config
    from slideseek.utils import format_timestamp

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

    from slideseek import ingest
    from slideseek.utils import format_timestamp

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
# serve
# --------------------------------------------------------------------------------------


@app.command()
def serve(
    index_dir: Path = typer.Option(Path("index"), "--index", "-i", help="Index directory."),
    videos: Path | None = typer.Option(
        None, help="Directory of the recordings. Results then play the local file in the page."
    ),
    host: str = typer.Option(
        "127.0.0.1", help="Address to listen on. 0.0.0.0 exposes it on the network."
    ),
    port: int = typer.Option(7860, help="Port to listen on."),
    device: str = typer.Option("cpu", help="Device for the query encoders."),
    database: str = typer.Option(
        "sqlite:///accounts.db",
        envvar="SLIDESEEK_DATABASE_URL",
        help="Where accounts are kept, as a SQLAlchemy URL. Put a PostgreSQL URL with a "
        "password in the environment variable, not on the command line.",
    ),
    admin: str | None = typer.Option(
        None,
        envvar="SLIDESEEK_ADMIN",
        help="Username of the default admin: an account you have signed up for. It is made an "
        "admin, and no other admin can change or delete it.",
    ),
    lab: Path | None = typer.Option(
        None,
        help="Serve the lab at /lab, a page that measures adding a lecture in the browser, "
        "and save what it captures in this directory. On your own machine only.",
    ),
    library: Path | None = typer.Option(
        None,
        envvar="SLIDESEEK_LIBRARIES",
        help="Let people add their own lectures from the browser, and keep them in this "
        "directory. Run `slideseek vendor` first.",
    ),
) -> None:
    """Open the search page for an index in your browser.

    Searching needs an account: sign up on the page the first time. Without --videos,
    results link to wherever the catalogue says each lecture lives. With it, they play your
    local recordings, and nothing leaves your machine.
    """
    setup_logging()
    try:
        import uvicorn

        from slideseek.accounts import AccountStore, redacted
        from slideseek.web.server import (
            create_app,
            load_benchmark,
            load_examples,
            load_picture_benchmark,
        )
    except ImportError:
        fail("the web app needs extra packages: run `uv sync --extra web --extra models`")

    loaded = load_index(index_dir)
    if videos is not None and not videos.is_dir():
        fail(f"video directory not found: {videos}")
    accounts = AccountStore(database)
    if accounts.set_default_admin(admin):
        console.print(f"The default admin is [bold]{accounts.default_admin}[/bold]")
    elif admin:
        console.print(
            f"[yellow]No account is called {admin.strip().removeprefix('@')}, so there is no "
            "default admin. Sign up as it first, without --admin, then start again with it.[/yellow]"
        )
    web_app = create_app(
        loaded,
        videos_dir=videos,
        examples=load_examples("config/examples.json", loaded.lectures),
        benchmark=load_benchmark("results/test/metrics.json"),
        picture_benchmark=load_picture_benchmark("results/screenshot/metrics.json"),
        device=device,
        accounts=accounts,
        lab_dir=lab,
        library=library,
    )
    if lab is not None:
        if host not in ("127.0.0.1", "localhost", "::1"):
            fail("--lab saves what it is sent, so it only runs on 127.0.0.1")
        console.print(f"The lab is at [bold]http://{host}:{port}/lab[/bold], saving to {lab}")
    console.print(f"Accounts are kept in {redacted(accounts.url)}")
    if library is not None:
        console.print(f"Lectures people add themselves are kept in {library}")
    console.print(f"Slideseek is at [bold]http://{host}:{port}[/bold]  (Ctrl+C to stop)")
    uvicorn.run(web_app, host=host, port=port, log_level="warning", access_log=False)


# --------------------------------------------------------------------------------------
# push / pull
# --------------------------------------------------------------------------------------


@app.command()
def push(
    repo_id: str = typer.Argument(
        ..., help="Hugging Face dataset repository, e.g. user/slideseek-index."
    ),
    index_dir: Path = typer.Option(Path("index"), "--index", "-i", help="Index directory."),
    private: bool = typer.Option(False, help="Create the repository as private."),
) -> None:
    """Upload an index to a Hugging Face dataset repository, with an attribution card.

    Only publish an index of lectures whose licence allows sharing derivatives. Uses the
    token from `huggingface-cli login` or the HF_TOKEN environment variable.
    """
    setup_logging()
    from slideseek.publish import push_index

    load_index(index_dir)  # fail early, with a clear message, if there is nothing to push
    try:
        url = push_index(index_dir, repo_id, private=private)
    except Exception as err:  # the hub client raises many unrelated types
        fail(f"upload failed: {err}")
    console.print(f"[green]Pushed[/green] {index_dir} -> {url}")


@app.command()
def pull(
    repo_id: str = typer.Argument(..., help="Hugging Face dataset repository holding an index."),
    index_dir: Path = typer.Option(Path("index"), "--index", "-i", help="Where to put it."),
) -> None:
    """Download a published index, ready for `slideseek search`."""
    setup_logging()
    from slideseek.index import InvalidIndex
    from slideseek.publish import pull_index

    try:
        loaded = pull_index(repo_id, index_dir)
    except InvalidIndex as err:
        fail(str(err))
    except Exception as err:
        fail(f"download failed: {err}")
    console.print(
        f"[green]Pulled[/green] {repo_id} -> {index_dir}: "
        f"{len(loaded.lectures)} lectures, {len(loaded)} moments"
    )


@app.command()
def deploy(
    repo_id: str = typer.Argument(
        ..., help="Hugging Face Space to create or update, e.g. user/slideseek."
    ),
    index_repo: str = typer.Option(
        ..., "--index-repo", help="Dataset repository holding the index (see `slideseek push`)."
    ),
    private: bool = typer.Option(False, help="Create the Space as private."),
    hardware: str = typer.Option(
        "zero-a10g", help="Hardware for a new Space. The default is ZeroGPU, which is free."
    ),
    accounts_repo: str | None = typer.Option(
        None,
        "--accounts-repo",
        help="Private dataset to keep a copy of the accounts in, e.g. user/slideseek-accounts. "
        "Created, private, if missing. The Space also needs the secret "
        "SLIDESEEK_ACCOUNTS_TOKEN, set by hand.",
    ),
    admin: str | None = typer.Option(
        None,
        "--admin",
        help="Username of the default admin, an account already signed up for on the Space. "
        "It is made an admin at start-up, and no other admin can change or delete it.",
    ),
    libraries_repo: str | None = typer.Option(
        None,
        "--libraries-repo",
        help="Let people add their own lectures from the browser, and keep a copy of them in "
        "this private dataset, e.g. user/slideseek-libraries. Created, private, if missing; "
        "written with the same token as the accounts.",
    ),
) -> None:
    """Deploy the web app as a Hugging Face Space on ZeroGPU, serving a published index.

    Run it from the root of the repository, after `slideseek push`. It uploads the app's
    source and nothing else: no video and no index. Uses the token from `hf auth login` or
    the HF_TOKEN environment variable.
    """
    setup_logging()
    from slideseek.publish import push_space

    if libraries_repo:
        from slideseek.web.vendor import VendorError, fetch

        try:
            fetch()  # the page's speech recognition, checked by hash
        except (VendorError, OSError) as err:
            fail(str(err))
    try:
        url = push_space(
            repo_id,
            index_repo,
            private=private,
            hardware=hardware,
            accounts_repo=accounts_repo,
            admin=admin,
            libraries_repo=libraries_repo,
        )
    except (FileNotFoundError, ValueError) as err:
        fail(str(err))
    except Exception as err:  # the hub client raises many unrelated types
        fail(
            f"deployment failed: {err}\n"
            "A free account can host two ZeroGPU Spaces, once its email is verified and it "
            "is more than 30 days old."
        )
    console.print(f"[green]Deployed[/green] {url}")
    console.print("It builds for a few minutes, then loads the models; watch the Logs tab there.")


@app.command()
def vendor() -> None:
    """Fetch the scripts the page runs speech recognition with, at their pinned versions.

    They are checked against the npm registry's checksum and a hash recorded in
    slideseek/web/vendor.py, then served from the page's own origin.
    """
    setup_logging()
    from slideseek.web.vendor import VENDOR_DIR, VendorError, fetch

    try:
        written = fetch()
    except (VendorError, OSError) as err:
        fail(str(err))
    if written:
        console.print(f"[green]Fetched[/green] {', '.join(written)} -> {VENDOR_DIR}")
    else:
        console.print(f"Already in place: {VENDOR_DIR}")


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
    from slideseek.demo import make_demo

    try:
        make_demo(out)
    except RuntimeError as err:
        fail(str(err))
    console.print(f"Demo lectures written to [bold]{out}[/bold]. Next:")
    console.print(
        f"  slideseek index {out}/videos --catalog {out}/lectures.csv --out {out}/index --work {out}/work"
    )
    console.print(
        f'  slideseek search "why do we hold out data we never train on" --index {out}/index'
    )


if __name__ == "__main__":
    app()
