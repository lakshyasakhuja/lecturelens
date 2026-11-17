"""Publishing an index: the attribution card, and the push and pull wrappers."""

from __future__ import annotations

import dataclasses
import sys
import types
from pathlib import Path

import pytest
from fakes import make_index

from slideseek.publish import (
    dataset_card,
    hub_licence,
    pull_index,
    push_index,
    push_space,
    stage_space,
)


def licensed_index(root=None, licences=("CC BY-NC-SA 4.0", "CC BY-NC-SA 4.0")):
    index = make_index()
    for (lecture_id, lecture), licence in zip(list(index.lectures.items()), licences, strict=True):
        index.lectures[lecture_id] = dataclasses.replace(
            lecture,
            course="6.036 Introduction to Machine Learning",
            instructors="Prof. Ada Example",
            licence=licence,
            source_url=f"https://ocw.example.edu/{lecture_id}",
        )
    index.manifest.update(
        {
            "models": {"asr": "large-v3-turbo", "text": "Qwen/Qwen3-Embedding-0.6B"},
            "total_duration_s": 575,
        }
    )
    if root is not None:
        index.save(root)
    return index


def test_hub_licence_is_shared_licence_or_other():
    assert hub_licence(licensed_index()) == "cc-by-nc-sa-4.0"
    assert hub_licence(licensed_index(licences=("CC BY-NC-SA 4.0", "CC BY 4.0"))) == "other"
    assert (
        hub_licence(licensed_index(licences=("All rights reserved", "All rights reserved")))
        == "other"
    )
    assert hub_licence(make_index()) == "other"  # no licence recorded


def test_dataset_card_credits_every_lecture():
    card = dataset_card(licensed_index())
    assert card.startswith("---\nlicense: cc-by-nc-sa-4.0\n")
    assert "2 lecture recording(s): 8 moments, 7 slide keyframes, 9:35 of video" in card
    assert "It does not contain the lecture videos" in card
    assert (
        "| `lec01` | Generalisation | 6.036 Introduction to Machine Learning | "
        "Prof. Ada Example | CC BY-NC-SA 4.0 | [link](https://ocw.example.edu/lec01) |"
    ) in card
    assert "| `lec02` | Neural networks |" in card
    assert "`large-v3-turbo`" in card and "`Qwen/Qwen3-Embedding-0.6B`" in card


def test_dataset_card_says_so_when_a_licence_is_missing():
    card = dataset_card(make_index())
    assert "license: other" in card
    assert "| - | not stated | - |" in card


def _fake_hub(monkeypatch, calls):
    hub = types.ModuleType("huggingface_hub")

    class HfApi:
        def __init__(self, token=None):
            calls.append(("init", token))

        def create_repo(self, repo_id, **kwargs):
            calls.append(("create_repo", repo_id, kwargs))

        def upload_folder(self, **kwargs):
            # The folder may be temporary, so note what is in it while it exists.
            folder = Path(kwargs["folder_path"])
            kwargs["files"] = sorted(
                path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file()
            )
            calls.append(("upload_folder", kwargs))

        def repo_info(self, repo_id, **kwargs):
            calls.append(("repo_info", repo_id, kwargs))
            if repo_id.endswith("/missing"):
                raise RuntimeError("404: no such repository")

        def add_space_variable(self, repo_id, key, value):
            calls.append(("add_space_variable", repo_id, key, value))

    def snapshot_download(**kwargs):
        calls.append(("snapshot_download", kwargs))
        return kwargs["local_dir"]

    hub.HfApi = HfApi
    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)


def test_push_writes_the_card_and_uploads_as_a_dataset(tmp_path, monkeypatch):
    calls = []
    _fake_hub(monkeypatch, calls)
    licensed_index(tmp_path / "index")

    url = push_index(tmp_path / "index", "someone/lectures-index", token="t0k3n", private=True)

    assert url == "https://huggingface.co/datasets/someone/lectures-index"
    card = (tmp_path / "index" / "README.md").read_text()
    assert "## Lectures and attribution" in card
    assert calls[0] == ("init", "t0k3n")
    assert calls[1] == (
        "create_repo",
        "someone/lectures-index",
        {"repo_type": "dataset", "private": True, "exist_ok": True},
    )
    upload = calls[2][1]
    assert upload["repo_type"] == "dataset"
    assert upload["folder_path"] == str(tmp_path / "index")
    assert upload["commit_message"] == "Index: 2 lectures, 8 moments"


def test_pull_downloads_and_loads(tmp_path, monkeypatch):
    calls = []
    _fake_hub(monkeypatch, calls)
    licensed_index(tmp_path / "index")  # stands in for what the download would write

    index = pull_index("someone/lectures-index", tmp_path / "index")

    assert len(index) == 8
    assert calls[0][1]["repo_type"] == "dataset"
    assert calls[0][1]["repo_id"] == "someone/lectures-index"


# ---- the Space ---------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent


def test_staged_space_holds_the_app_and_nothing_it_should_not(tmp_path):
    staged = stage_space(tmp_path / "space", ROOT)

    files = {path.relative_to(staged).as_posix() for path in staged.rglob("*") if path.is_file()}
    assert {"README.md", "app.py", "requirements.txt", "config/examples.json"} <= files
    assert {
        "slideseek/__init__.py",
        "slideseek/web/space.py",
        "slideseek/web/static/index.html",
        "slideseek/web/static/vendor/gradio-client.js",
    } <= files
    assert (staged / "README.md").read_text().startswith("---\ntitle: Slideseek\n")
    assert "sdk: gradio" in (staged / "README.md").read_text()

    # No recordings, no index and no caches. The one binary file is the page's example
    # photo of a slide, which the Hub API uploads like any other file; it must stay small.
    sample = "slideseek/web/static/samples/slide-photo.jpg"
    assert sample in files and (staged / sample).stat().st_size < 100_000
    media = {".mp4", ".mkv", ".mov", ".webm", ".wav", ".npy", ".parquet", ".jpg", ".png", ".pyc"}
    assert not [name for name in files if Path(name).suffix.lower() in media and name != sample]
    assert not [name for name in files if "__pycache__" in name or "/." in f"/{name}"]
    for name in files - {sample}:
        (staged / name).read_text(encoding="utf-8")  # raises if a file is not text


def test_staging_needs_the_repository(tmp_path):
    with pytest.raises(FileNotFoundError, match="root of the repository"):
        stage_space(tmp_path / "space", tmp_path)


def test_push_space_creates_a_zerogpu_gradio_space_and_points_it_at_the_index(monkeypatch):
    calls = []
    _fake_hub(monkeypatch, calls)

    url = push_space("someone/slideseek", "someone/lectures-index", root=ROOT, token="t0k3n")

    assert url == "https://huggingface.co/spaces/someone/slideseek"
    names = [call[0] for call in calls]
    assert names == ["init", "repo_info", "create_repo", "add_space_variable", "upload_folder"]
    assert calls[1][1:] == ("someone/lectures-index", {"repo_type": "dataset"})
    assert calls[2][1:] == (
        "someone/slideseek",
        {
            "repo_type": "space",
            "space_sdk": "gradio",
            "space_hardware": "zero-a10g",
            "private": False,
            "exist_ok": True,
        },
    )
    assert calls[3][1:] == (
        "someone/slideseek",
        "SLIDESEEK_INDEX_REPO",
        "someone/lectures-index",
    )
    upload = calls[4][1]
    assert upload["repo_type"] == "space" and upload["repo_id"] == "someone/slideseek"
    assert {"app.py", "requirements.txt", "README.md", "slideseek/web/space.py"} <= set(
        upload["files"]
    )


def test_push_space_stops_before_creating_anything_if_the_index_is_not_published(monkeypatch):
    calls = []
    _fake_hub(monkeypatch, calls)
    with pytest.raises(RuntimeError, match="no such repository"):
        push_space("someone/slideseek", "someone/missing", root=ROOT)
    assert [call[0] for call in calls] == ["init", "repo_info"]
