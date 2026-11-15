"""Publishing an index: the attribution card, and the push and pull wrappers."""

from __future__ import annotations

import dataclasses
import sys
import types

from fakes import make_index

from lecturelens.publish import dataset_card, hub_licence, pull_index, push_index


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
            calls.append(("upload_folder", kwargs))

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
