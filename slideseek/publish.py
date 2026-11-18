"""Moving an index between the machine that builds it and the machine that serves it.

Indexing runs once on a GPU notebook; search runs on a CPU host. The index travels between
them as a Hugging Face *dataset* repository, which keeps large binary files out of git and
lets the serving side fetch it at start-up.

An index is derived from other people's lectures, so it is published with a dataset card
that names every lecture, its source and its licence. Only index lectures you are allowed
to redistribute derivatives of; private recordings stay on your own machine.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

from slideseek.index import Index
from slideseek.utils import format_timestamp

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
    index: Index, code_url: str = "https://github.com/lakshyasakhuja/slideseek"
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
        "- slideseek",
        "- retrieval",
        "- multimodal",
        "pretty_name: Slideseek search index",
        "---",
        "",
        "# Slideseek search index",
        "",
        f"A prebuilt search index over {len(index.lectures)} lecture recording(s): "
        f"{len(index)} moments, {len(index.image)} slide keyframes, "
        f"{format_timestamp(manifest.get('total_duration_s', 0))} of video. "
        f"Built and read by [Slideseek]({code_url}).",
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
        f"- Built: {manifest.get('created_utc', '?')} with Slideseek "
        f"{manifest.get('slideseek_version') or manifest.get('lecturelens_version', '?')} (index format {manifest.get('format', '?')})",
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
# What a username may be: ``slideseek.accounts.USERNAME_PATTERN``, copied so that deploying
# does not need the web extra. A test keeps the two the same.
USERNAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{2,29}")


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
        dest / "slideseek",
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
    accounts_repo: str | None = None,
    admin: str | None = None,
) -> str:
    """Create or update the Hugging Face Space that serves ``index_repo``. Returns its URL.

    The Space is created on ZeroGPU, which is the hardware a free account may use; an
    existing Space keeps the hardware it has. The index must already be published
    (``push_index``), and publicly: the Space is given its name, not a token to read it.

    ``accounts_repo`` is the private dataset the Space keeps a copy of its accounts in
    (``slideseek.backup``). It is created, private, if it does not exist, and refused if it
    is public. The token the Space writes to it with is a secret, set on the Space by hand.

    ``admin`` is the username of the default admin (``SLIDESEEK_ADMIN``): an account that
    has been signed up for on the Space already.
    """
    from huggingface_hub import HfApi

    if admin is not None:
        admin = admin.strip().removeprefix("@")
        if not USERNAME.fullmatch(admin):
            raise ValueError(f"{admin!r} is not a username an account can have")
    api = HfApi(token=token)
    api.repo_info(index_repo, repo_type="dataset")  # fail now, not when the Space starts
    if accounts_repo:
        api.create_repo(accounts_repo, repo_type="dataset", private=True, exist_ok=True)
        if getattr(api.repo_info(accounts_repo, repo_type="dataset"), "private", None) is not True:
            raise ValueError(
                f"{accounts_repo} is public; the accounts are only ever copied to a private "
                "dataset. Make it private, or name another."
            )
    api.create_repo(
        repo_id,
        repo_type="space",
        space_sdk="gradio",
        space_hardware=hardware,
        private=private,
        exist_ok=True,
    )
    api.add_space_variable(repo_id, "SLIDESEEK_INDEX_REPO", index_repo)
    if accounts_repo:
        api.add_space_variable(repo_id, "SLIDESEEK_ACCOUNTS_REPO", accounts_repo)
    if admin:
        api.add_space_variable(repo_id, "SLIDESEEK_ADMIN", admin)
    with tempfile.TemporaryDirectory() as staging:
        stage_space(staging, root)
        api.upload_folder(
            folder_path=staging,
            repo_id=repo_id,
            repo_type="space",
            commit_message="Deploy Slideseek",
            # Files removed from the package should not linger in the Space.
            delete_patterns=["slideseek/**", "config/**", "results/**"],
        )
    return f"https://huggingface.co/spaces/{repo_id}"
