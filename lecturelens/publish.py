"""Moving an index between the machine that builds it and the machine that serves it.

Indexing runs once on a GPU notebook; search runs on a CPU host. The index travels between
them as a Hugging Face *dataset* repository, which keeps large binary files out of git and
lets the serving side fetch it at start-up.

An index is derived from other people's lectures, so it is published with a dataset card
that names every lecture, its source and its licence. Only index lectures you are allowed
to redistribute derivatives of; private recordings stay on your own machine.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from lecturelens.index import Index
from lecturelens.utils import format_timestamp

# Licence strings as people write them -> identifiers the Hub recognises.
_HUB_LICENCES = {
    "cc by-nc-sa 4.0": "cc-by-nc-sa-4.0",
    "cc-by-nc-sa-4.0": "cc-by-nc-sa-4.0",
    "cc by-nc-sa": "cc-by-nc-sa-4.0",
    "cc by-sa 4.0": "cc-by-sa-4.0",
    "cc by 4.0": "cc-by-4.0",
    "cc by-nc 4.0": "cc-by-nc-4.0",
    "cc0": "cc0-1.0",
}


def hub_licence(index: Index) -> str:
    """The Hub licence id shared by every lecture, or ``other`` if they differ or are unknown."""
    found = {(lecture.licence or "").strip().lower() for lecture in index.lectures.values()}
    if len(found) == 1:
        return _HUB_LICENCES.get(found.pop(), "other")
    return "other"


def dataset_card(
    index: Index, code_url: str = "https://github.com/lakshyasakhuja/lecturelens"
) -> str:
    """A README for the dataset repository: what the index is, and whose lectures it covers."""
    manifest = index.manifest
    models = manifest.get("models", {})
    lines = [
        "---",
        f"license: {hub_licence(index)}",
        "language:",
        "- en",
        "tags:",
        "- lecturelens",
        "- retrieval",
        "- multimodal",
        "pretty_name: LectureLens search index",
        "---",
        "",
        "# LectureLens search index",
        "",
        f"A prebuilt search index over {len(index.lectures)} lecture recording(s): "
        f"{len(index)} moments, {len(index.image)} slide keyframes, "
        f"{format_timestamp(manifest.get('total_duration_s', 0))} of video. "
        f"Built and read by [LectureLens]({code_url}).",
        "",
        "It contains **derived data only**: machine transcripts, slide text read by OCR, slide "
        "thumbnails and embeddings. It does not contain the lecture videos. Each result links "
        "to the original recording at its source.",
        "",
        "## Lectures and attribution",
        "",
        "| Lecture | Title | Course | Instructors | Licence | Source |",
        "|---|---|---|---|---|---|",
    ]
    for lecture in index.lectures.values():
        source = f"[link]({lecture.source_url})" if lecture.source_url else "-"
        lines.append(
            f"| `{lecture.lecture_id}` | {lecture.title} | {lecture.course or '-'} | "
            f"{lecture.instructors or '-'} | {lecture.licence or 'not stated'} | {source} |"
        )
    lines += [
        "",
        "The transcripts, slide text and thumbnails are derivatives of these lectures and are "
        "shared under the same licence as the lecture they come from, with credit to the "
        "original authors and for non-commercial use where that licence says so.",
        "",
        "## How it was built",
        "",
        f"- Speech recognition: `{models.get('asr', '?')}` (faster-whisper), word timestamps",
        f"- Slide text: {models.get('ocr', '?')}",
        f"- Text embeddings: `{models.get('text', '?')}`",
        f"- Image embeddings: `{models.get('image', '?')}`",
        f"- Built: {manifest.get('created_utc', '?')} with LectureLens "
        f"{manifest.get('lecturelens_version', '?')} (index format {manifest.get('format', '?')})",
        "",
        "## Files",
        "",
        "| File | Contents |",
        "|---|---|",
        "| `manifest.json` | models, counts, build settings |",
        "| `lectures.json` | the catalogue entries above |",
        "| `moments.parquet` | one row per moment: times, slide text, transcript, links |",
        "| `text.npy`, `speech.npy`, `slide.npy` | text embeddings, one row per moment |",
        "| `image.npy` | keyframe embeddings, one row per keyframe |",
        "| `thumbs/` | one thumbnail per keyframe |",
        "",
    ]
    return "\n".join(lines)


def push_index(
    index_dir: str | Path, repo_id: str, token: str | None = None, private: bool = False
) -> str:
    """Upload an index directory to a Hugging Face dataset repository. Returns its URL.

    The dataset card is regenerated from the index on every push, so attribution cannot
    fall out of date with the lectures actually included.
    """
    from huggingface_hub import HfApi

    index_dir = Path(index_dir)
    index = Index.load(index_dir)
    (index_dir / "README.md").write_text(dataset_card(index), encoding="utf-8")
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(index_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Index: {len(index.lectures)} lectures, {len(index)} moments",
        delete_patterns=["thumbs/**"],  # drop thumbnails of lectures no longer in the index
    )
    return f"https://huggingface.co/datasets/{repo_id}"


def pull_index(repo_id: str, local_dir: str | Path, token: str | None = None) -> Index:
    """Download an index from a Hugging Face dataset repository and load it."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=repo_id, repo_type="dataset", local_dir=str(local_dir), token=token
    )
    return Index.load(path)


# ---- the hosted demo ---------------------------------------------------------------------

# What a Hugging Face Space built with the Gradio SDK reads from the root of its repository.
SPACE_FILES = ("README.md", "app.py", "requirements.txt")
# ZeroGPU, by the name the Hub's API knows it under.
ZERO_GPU = "zero-a10g"


def stage_space(dest: str | Path, root: str | Path = ".") -> Path:
    """Assemble the files a Space runs from, in ``dest``. Returns ``dest``.

    A Gradio Space does not install this package; it runs ``app.py`` beside a copy of it.
    So the Space gets the package's source (with the page and its one example photo), the
    example questions, the measured benchmarks if there are any, and its own three files
    from ``deploy/space``. No video and no index: the index is fetched from its dataset
    repository when the Space starts.
    """
    dest, root = Path(dest), Path(root)
    source = root / "deploy" / "space"
    missing = [name for name in SPACE_FILES if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{source} has no {', '.join(missing)}; run this from the root of the repository"
        )
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        Path(__file__).parent,
        dest / "lecturelens",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".*"),
        dirs_exist_ok=True,
    )
    for name in SPACE_FILES:
        shutil.copy2(source / name, dest / name)
    for optional in (
        "config/examples.json",
        "results/test/metrics.json",
        "results/screenshot/metrics.json",
    ):
        if (root / optional).is_file():
            (dest / optional).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / optional, dest / optional)
    return dest


def push_space(
    repo_id: str,
    index_repo: str,
    root: str | Path = ".",
    token: str | None = None,
    private: bool = False,
    hardware: str = ZERO_GPU,
) -> str:
    """Create or update the Hugging Face Space that serves ``index_repo``. Returns its URL.

    The Space is created on ZeroGPU, which is the hardware a free account may use; an
    existing Space keeps the hardware it has. The index must already be published
    (``push_index``), and publicly: the Space is given its name, not a token to read it.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.repo_info(index_repo, repo_type="dataset")  # fail now, not when the Space starts
    api.create_repo(
        repo_id,
        repo_type="space",
        space_sdk="gradio",
        space_hardware=hardware,
        private=private,
        exist_ok=True,
    )
    api.add_space_variable(repo_id, "LECTURELENS_INDEX_REPO", index_repo)
    with tempfile.TemporaryDirectory() as staging:
        stage_space(staging, root)
        api.upload_folder(
            folder_path=staging,
            repo_id=repo_id,
            repo_type="space",
            commit_message="Deploy LectureLens",
            # Files removed from the package should not linger in the Space.
            delete_patterns=["lecturelens/**", "config/**", "results/**"],
        )
    return f"https://huggingface.co/spaces/{repo_id}"
