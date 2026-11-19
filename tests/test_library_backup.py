"""The copy of people's own lectures in a private dataset: uploading, deleting, restoring."""

from __future__ import annotations

import fnmatch
import gzip
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("sqlalchemy", reason="the web extra is not installed")
pytest.importorskip("huggingface_hub", reason="huggingface_hub is not installed")

import httpx2
import synthetic
from fakes import FakeIndexImageEmbedder, FakeIndexTextEmbedder, FakeOcr
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

from slideseek.backup import BackupError, LibraryBackup
from slideseek.library import Library, Models

REPO = "someone/slideseek-libraries"
OWNER, OTHER = "1-1700000000", "2-1700000001"
FILE = {"name": "lecture.mp4", "size": 1000, "type": "video/mp4"}
LARGE = (".npy", ".parquet", ".jpg")


class FakeHub:
    """The calls of ``HfApi`` the library's copy makes, over one repository in memory."""

    def __init__(self, *, private=True, exists=True):
        self.private, self.exists = private, exists
        self.files: dict[str, bytes] = {}
        self.lfs: dict[str, int] = {}  # every large file ever stored, by hash
        self.commits: list[str] = ["initial commit"]
        self.calls: list[str] = []
        self.failures: dict[str, int] = {}

    def _call(self, name):
        self.calls.append(name)
        if self.failures.get(name, 0) > 0:
            self.failures[name] -= 1
            raise ConnectionError(f"{name}: the network is down")

    def _put(self, path, data):
        self.files[path] = data
        if path.endswith(LARGE):
            self.lfs[hashlib.sha256(data).hexdigest()] = len(data)

    def repo_info(self, repo_id, *, repo_type=None):
        self._call("repo_info")
        if not self.exists:
            request = httpx2.Request("GET", f"https://huggingface.co/api/datasets/{repo_id}")
            raise RepositoryNotFoundError(
                "Repository Not Found", response=httpx2.Response(404, request=request)
            )
        return SimpleNamespace(private=self.private)

    def create_repo(self, repo_id, *, repo_type=None, private=None, exist_ok=False):
        self._call("create_repo")
        self.exists, self.private = True, bool(private)

    def snapshot_download(self, *, repo_id, repo_type=None, local_dir=None):
        self._call("snapshot_download")
        for path, data in self.files.items():
            target = Path(local_dir) / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        return local_dir

    def upload_folder(self, *, repo_id, folder_path, path_in_repo, repo_type, allow_patterns, commit_message):  # fmt: skip
        self._call("upload_folder")
        folder = Path(folder_path)
        for path in folder.rglob("*"):
            relative = path.relative_to(folder).as_posix()
            if path.is_file() and any(fnmatch.fnmatch(relative, p) for p in allow_patterns):
                self._put(f"{path_in_repo}/{relative}", path.read_bytes())
        self.commits.append(commit_message)

    def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type, commit_message):
        self._call("upload_file")
        self._put(path_in_repo, Path(path_or_fileobj).read_bytes())
        self.commits.append(commit_message)

    def delete_folder(self, path_in_repo, repo_id, *, repo_type=None, commit_message=None):
        self._call("delete_folder")
        gone = [path for path in self.files if path.startswith(path_in_repo + "/")]
        if not gone:
            raise EntryNotFoundError(f"{path_in_repo} is not in {repo_id}")
        for path in gone:
            del self.files[path]
        self.commits.append(commit_message)

    def super_squash_history(self, repo_id, *, repo_type=None, commit_message=None):
        self._call("super_squash_history")
        self.commits = [commit_message]

    def list_repo_tree(self, repo_id, *, repo_type=None, recursive=False):
        self._call("list_repo_tree")
        for path, data in self.files.items():
            lfs = None
            if path.endswith(LARGE):
                lfs = SimpleNamespace(sha256=hashlib.sha256(data).hexdigest(), size=len(data))
            yield SimpleNamespace(path=path, size=len(data), lfs=lfs)

    def list_lfs_files(self, repo_id, *, repo_type=None):
        self._call("list_lfs_files")
        return [SimpleNamespace(file_oid=oid, size=size) for oid, size in self.lfs.items()]

    def permanently_delete_lfs_files(self, repo_id, lfs_files, *, repo_type=None):
        self._call("permanently_delete_lfs_files")
        for item in lfs_files:
            del self.lfs[item.file_oid]

    def lectures(self):
        return sorted({"/".join(path.split("/")[:2]) for path in self.files})


def jpeg():
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 36), "white").save(buffer, "JPEG")
    return buffer.getvalue()


def make_library(root):
    return Library(
        root,
        lambda: Models(FakeIndexTextEmbedder(), FakeIndexImageEmbedder(), FakeOcr()),
        worker=False,
    )


@pytest.fixture(scope="module")
def frames(lecture_frames):
    stack, _ = lecture_frames
    return gzip.compress(synthetic.to_uint8(stack).tobytes())


def add(library, frames, owner=OWNER, title="Lecture"):
    lecture_id = library.create(owner, title=title, duration_s=107.0, file=FILE)["lecture_id"]
    for t in library.put_frames(owner, lecture_id, frames)["keyframe_times"]:
        library.put_still(owner, lecture_id, round(t * 1000), jpeg())
    words = {"model": "m", "segments": [{"words": [{"w": "kernels", "s": 30, "e": 31}]}]}
    library.put_transcript(owner, lecture_id, words)
    library.run_pending()
    return lecture_id


@pytest.fixture()
def watched(tmp_path):
    hub = FakeHub()
    library = make_library(tmp_path / "libraries")
    copy = LibraryBackup(REPO, library.root, api=hub)
    library.on_change = copy.changed  # what watch() does, without the exit handlers
    yield hub, library, copy
    copy.close()


def test_a_finished_lecture_is_copied_and_nothing_else(watched, frames):
    hub, library, copy = watched
    lecture_id = add(library, frames)
    assert copy.flush()
    folder = f"{OWNER}/{lecture_id}"
    assert hub.lectures() == [folder]
    assert f"{folder}/lecture.json" in hub.files
    assert f"{folder}/index/manifest.json" in hub.files
    assert any(path.startswith(f"{folder}/index/thumbs/") for path in hub.files)
    assert not [path for path in hub.files if "/work/" in path]  # the stage files stay
    assert hub.commits[-1] == "Add a lecture" and "Lecture" not in " ".join(hub.commits)

    library.rename(OWNER, lecture_id, "Kernels")
    assert copy.flush()
    assert b'"Kernels"' in hub.files[f"{folder}/lecture.json"]
    assert hub.calls.count("upload_folder") == 1  # a rename sends lecture.json alone


def test_a_deleted_lecture_is_gone_from_the_history_and_the_storage(watched, frames):
    hub, library, copy = watched
    deleted = add(library, frames, title="Deleted")
    copy.flush()
    library.delete(OWNER, deleted)
    kept = add(library, frames, title="Kept")
    assert copy.flush()
    assert hub.lectures() == [f"{OWNER}/{kept}"]
    # Squashed after the deletion: no commit that held the deleted lecture is left.
    assert hub.commits[0] == "The lectures" and "Remove a lecture" not in hub.commits
    # Every large file left belongs to the lecture that is left.
    current = {hashlib.sha256(d).hexdigest() for p, d in hub.files.items() if p.endswith(LARGE)}
    assert current and set(hub.lfs) == current
    assert deleted not in str(sorted(hub.files))


def test_deleting_an_account_deletes_its_copy(watched, frames):
    hub, library, copy = watched
    add(library, frames)
    theirs = add(library, frames, owner=OTHER)
    copy.flush()
    library.delete_owner(OWNER)
    assert copy.flush()
    assert hub.lectures() == [f"{OTHER}/{theirs}"]
    assert hub.commits[0] == "The lectures"


def test_a_lecture_deleted_before_it_was_finished_has_nothing_to_delete(watched, frames):
    hub, library, copy = watched
    lecture_id = library.create(OWNER, title="Half", duration_s=107.0, file=FILE)["lecture_id"]
    library.delete(OWNER, lecture_id)
    assert not copy.pending  # nothing was copied, so nothing is to be deleted
    library.delete_owner(OTHER)  # nor of a library that was never copied
    assert copy.flush() and hub.files == {}


def test_a_failed_copy_is_tried_again(watched, frames):
    hub, library, copy = watched
    hub.failures["upload_folder"] = 1
    lecture_id = add(library, frames)
    assert not copy.flush()
    assert copy.pending and hub.files == {}
    assert copy.flush()
    assert hub.lectures() == [f"{OWNER}/{lecture_id}"]


def test_the_lectures_come_back_after_a_restart(tmp_path, watched, frames):
    hub, library, copy = watched
    lecture_id = add(library, frames)
    copy.flush()

    # A new disk: the copy is restored before the library opens, and searches as before.
    fresh = tmp_path / "after-restart"
    assert LibraryBackup(REPO, fresh, api=hub).restore() == 1
    again = make_library(fresh)
    (item,) = again.lectures(OWNER)
    assert item["lecture_id"] == lecture_id and item["stage"] == "ready"
    index = again.index(OWNER)
    assert len(index) == 4 and all(index.thumb_path(r) for r in range(4))

    # A lecture already on the disk is left as it is.
    assert LibraryBackup(REPO, fresh, api=hub).restore() == 0


def test_lectures_are_never_copied_to_a_public_dataset(tmp_path, frames):
    hub = FakeHub(private=False)
    library = make_library(tmp_path)
    copy = LibraryBackup(REPO, tmp_path, api=hub)
    with pytest.raises(BackupError, match="public"):
        copy.restore()
    library.on_change = copy.changed
    add(library, frames)
    assert not copy.flush()
    assert hub.files == {} and "upload_folder" not in hub.calls
    copy._closed = True  # stop the thread without one more try


def test_a_missing_dataset_is_created_private(tmp_path):
    hub = FakeHub(exists=False)
    assert LibraryBackup(REPO, tmp_path, api=hub).restore() == 0
    assert hub.exists and hub.private


def test_the_environment_names_the_directory_and_the_dataset(tmp_path, monkeypatch, frames):
    from slideseek.web import server

    monkeypatch.delenv("SLIDESEEK_LIBRARIES", raising=False)
    monkeypatch.delenv("SLIDESEEK_LIBRARIES_REPO", raising=False)
    assert server.libraries_from_env() == (None, None)

    monkeypatch.setenv("SLIDESEEK_LIBRARIES", str(tmp_path / "mine"))
    assert server.libraries_from_env() == (tmp_path / "mine", None)

    hub = FakeHub()
    monkeypatch.setattr("huggingface_hub.HfApi", lambda token=None: hub)
    monkeypatch.delenv("SLIDESEEK_LIBRARIES")
    monkeypatch.setenv("SLIDESEEK_LIBRARIES_REPO", REPO)
    folder, copy = server.libraries_from_env(tmp_path / "default")
    assert folder == tmp_path / "default" and copy.repo_id == REPO
    assert "snapshot_download" in hub.calls
