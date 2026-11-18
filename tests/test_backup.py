"""The copy of the accounts in a private dataset: restoring it, keeping it current, keeping it safe.

The Hub is played by ``FakeHub``, which keeps repositories in memory and offers the calls of
``huggingface_hub.HfApi`` that the backup makes.
"""

from __future__ import annotations

import hashlib
import logging
import os
import signal
import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("sqlalchemy", reason="the web extra is not installed")
pytest.importorskip("argon2", reason="the web extra is not installed")
pytest.importorskip("huggingface_hub", reason="huggingface_hub is not installed")

import httpx2
from fakes import ADA, fast_accounts
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError, RepositoryNotFoundError

import slideseek.backup as backup_module
from slideseek.accounts import ACCOUNT_DELETED, ACCOUNT_UPDATED
from slideseek.backup import (
    FILENAME,
    BackupError,
    DatasetBackup,
    backed_up_accounts,
    on_sigterm,
    sqlite_path,
)
from slideseek.web.server import accounts_from_env, accounts_token_from_env

REPO = "someone/slideseek-accounts"
GRACE = {**ADA, "username": "grace", "email": "grace@example.com", "name": "Grace Hopper"}


def hub_error(kind, status: int, message: str):
    request = httpx2.Request("GET", f"https://huggingface.co/api/datasets/{REPO}")
    return kind(message, response=httpx2.Response(status, request=request))


class FakeHub:
    """The calls of ``HfApi`` the backup makes, over repositories held in memory."""

    def __init__(self) -> None:
        self.repos: dict[str, dict] = {}
        self.calls: list[str] = []
        self.failures: dict[str, int] = {}  # call name -> how many more times it fails
        self.can_create = True
        self.upload_times: list[float] = []

    def add(self, repo_id=REPO, *, private=True, copy: bytes | None = None) -> dict:
        repo = {"private": private, "files": {}, "commits": ["initial commit"], "lfs": {}}
        self.repos[repo_id] = repo
        if copy is not None:
            self._store(repo, copy, "an earlier copy")
        return repo

    def _call(self, name: str) -> None:
        self.calls.append(name)
        if self.failures.get(name, 0) > 0:
            self.failures[name] -= 1
            raise ConnectionError(f"{name}: the network is down")

    def _repo(self, repo_id: str) -> dict:
        if repo_id not in self.repos:
            raise hub_error(RepositoryNotFoundError, 404, "Repository Not Found")
        return self.repos[repo_id]

    @staticmethod
    def _store(repo: dict, data: bytes, message: str) -> None:
        repo["files"][FILENAME] = data
        repo["commits"].append(message)
        repo["lfs"][hashlib.sha256(data).hexdigest()] = len(data)

    def repo_info(self, repo_id, *, repo_type=None):
        self._call("repo_info")
        assert repo_type == "dataset"
        return SimpleNamespace(private=self._repo(repo_id)["private"])

    def create_repo(self, repo_id, *, repo_type=None, private=None, exist_ok=False):
        self._call("create_repo")
        if not self.can_create:
            raise hub_error(HfHubHTTPError, 403, "This token cannot create repositories")
        if repo_id not in self.repos:
            self.add(repo_id, private=bool(private))

    def hf_hub_download(self, repo_id, filename, *, repo_type=None, local_dir=None):
        self._call("hf_hub_download")
        repo = self._repo(repo_id)
        if filename not in repo["files"]:
            raise EntryNotFoundError(f"{filename} is not in {repo_id}")
        path = Path(local_dir) / filename
        path.write_bytes(repo["files"][filename])
        return str(path)

    def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type, commit_message):
        self._call("upload_file")
        assert path_in_repo == FILENAME and repo_type == "dataset"
        self._store(self._repo(repo_id), Path(path_or_fileobj).read_bytes(), commit_message)
        self.upload_times.append(time.monotonic())

    def super_squash_history(self, repo_id, *, repo_type=None, commit_message=None):
        self._call("super_squash_history")
        self._repo(repo_id)["commits"] = [commit_message]

    def list_lfs_files(self, repo_id, *, repo_type=None):
        self._call("list_lfs_files")
        return [
            SimpleNamespace(file_oid=oid, size=size)
            for oid, size in self._repo(repo_id)["lfs"].items()
        ]

    def permanently_delete_lfs_files(self, repo_id, lfs_files, *, repo_type=None):
        self._call("permanently_delete_lfs_files")
        for item in lfs_files:
            del self._repo(repo_id)["lfs"][item.file_oid]

    # -- for the tests

    def copy(self, repo_id=REPO) -> bytes:
        return self.repos[repo_id]["files"][FILENAME]

    def usernames(self, tmp_path: Path, repo_id=REPO) -> list[str]:
        return usernames_in(self.copy(repo_id), tmp_path)


def usernames_in(data: bytes, tmp_path: Path) -> list[str]:
    path = tmp_path / f"inspect-{hashlib.sha256(data).hexdigest()[:8]}.db"
    path.write_bytes(data)
    with closing(sqlite3.connect(path)) as db:
        return [row[0] for row in db.execute("SELECT username FROM slideseek_users ORDER BY id")]


def database_bytes(tmp_path: Path, *people) -> bytes:
    """A database file holding an account for each of ``people``."""
    path = tmp_path / f"made-{len(list(tmp_path.iterdir()))}.db"
    store = fast_accounts(f"sqlite:///{path}")
    for person in people:
        store.sign_up(**person)
    store.engine.dispose()
    return path.read_bytes()


@pytest.fixture()
def hub():
    hub = FakeHub()
    hub.add()
    return hub


@pytest.fixture()
def opened():
    """Opens backed-up stores, and closes their backups after the test."""
    backups = []

    def open_store(hub, folder: Path, **options):
        folder.mkdir(parents=True, exist_ok=True)
        options.setdefault("delay", 3600)  # no upload unless the test asks for one
        backup = DatasetBackup(REPO, folder / "accounts.db", api=hub, **options)
        backups.append(backup)
        restored = backup.restore(wait=0)
        store = fast_accounts(f"sqlite:///{backup.path}")
        store.on_change = backup.changed
        return store, backup, restored

    yield open_store
    for backup in backups:
        backup.close(timeout=1)


# ---- restoring -----------------------------------------------------------------------------


def test_accounts_and_sessions_outlive_a_restart(hub, opened, tmp_path):
    store, backup, restored = opened(hub, tmp_path / "first")
    assert not restored  # a first start: the dataset has no copy yet
    account = store.sign_up(**ADA)
    token = store.start_session(account)
    assert backup.pending
    assert backup.flush() and not backup.pending
    assert hub.usernames(tmp_path) == ["ada"]

    # The Space restarts on a new, empty disk.
    again, _, restored = opened(hub, tmp_path / "second")
    assert restored
    assert again.sign_in("ada", ADA["password"]) == account
    assert again.account_for_token(token) == account  # nobody is signed out by a restart


def test_the_copy_holds_password_hashes_never_passwords(hub, opened, tmp_path):
    store, backup, _ = opened(hub, tmp_path)
    store.sign_up(**ADA)
    backup.flush()
    assert ADA["password"].encode() not in hub.copy()
    assert b"$argon2id$" in hub.copy()


def test_a_missing_dataset_is_created_private(opened, tmp_path):
    hub = FakeHub()
    _, _, restored = opened(hub, tmp_path)
    assert not restored
    assert hub.repos[REPO]["private"] is True


def test_a_dataset_that_cannot_be_created_stops_start_up(tmp_path):
    hub = FakeHub()
    hub.can_create = False
    backup = DatasetBackup(REPO, tmp_path / "accounts.db", api=hub)
    with pytest.raises(BackupError, match="Create it as a private dataset"):
        backup.restore(wait=0)


def test_a_public_dataset_is_never_read_from_or_written_to(hub, opened, tmp_path):
    hub.repos[REPO]["private"] = False
    backup = DatasetBackup(REPO, tmp_path / "accounts.db", api=hub)
    with pytest.raises(BackupError, match="is public"):
        backup.restore(wait=0)
    assert "hf_hub_download" not in hub.calls
    assert not backup.path.exists()

    # Made public while the server runs: the next upload is refused, and kept for later.
    hub.repos[REPO]["private"] = True
    store, backup, _ = opened(hub, tmp_path / "running")
    store.sign_up(**ADA)
    hub.repos[REPO]["private"] = False
    assert not backup.flush()
    assert FILENAME not in hub.repos[REPO]["files"]
    assert backup.pending


def test_start_up_fails_rather_than_start_empty_when_the_copy_cannot_be_fetched(hub, tmp_path):
    hub.add(copy=database_bytes(tmp_path, ADA))
    hub.failures["hf_hub_download"] = 99
    backup = DatasetBackup(REPO, tmp_path / "disk" / "accounts.db", api=hub)
    with pytest.raises(BackupError, match="not starting"):
        backup.restore(attempts=4, wait=0)
    assert hub.calls.count("hf_hub_download") == 4
    assert not backup.path.exists()


def test_a_download_that_fails_once_is_tried_again(hub, opened, tmp_path):
    hub.add(copy=database_bytes(tmp_path, ADA))
    hub.failures["hf_hub_download"] = 1
    store, _, restored = opened(hub, tmp_path / "disk")
    assert restored and store.sign_in("ada", ADA["password"]) is not None


def test_a_damaged_copy_stops_start_up_and_leaves_the_disk_alone(hub, tmp_path):
    hub.add(copy=b"this is not a database" * 100)
    path = tmp_path / "accounts.db"
    path.write_bytes(b"whatever was here")
    backup = DatasetBackup(REPO, path, api=hub)
    with pytest.raises(BackupError, match="not a SQLite database"):
        backup.restore(wait=0)
    assert path.read_bytes() == b"whatever was here"


def test_the_copy_replaces_what_is_on_disk_and_any_journal_beside_it(hub, opened, tmp_path):
    hub.add(copy=database_bytes(tmp_path, ADA))
    folder = tmp_path / "disk"
    folder.mkdir()
    (folder / "accounts.db").write_bytes(database_bytes(tmp_path, GRACE))
    (folder / "accounts.db-journal").write_bytes(b"left over from a crash")
    store, _, restored = opened(hub, folder)
    assert restored
    assert not (folder / "accounts.db-journal").exists()
    assert store.sign_in("ada", ADA["password"]) is not None
    assert store.sign_in("grace", GRACE["password"]) is None


def test_on_a_first_start_accounts_already_on_disk_are_uploaded(hub, opened, tmp_path):
    folder = tmp_path / "disk"
    folder.mkdir()
    (folder / "accounts.db").write_bytes(database_bytes(tmp_path, ADA))
    _, backup, restored = opened(hub, folder)
    assert not restored and backup.pending
    assert backup.flush()
    assert hub.usernames(tmp_path) == ["ada"]


# ---- keeping it current --------------------------------------------------------------------


def test_nothing_is_uploaded_until_something_changes(hub, opened, tmp_path):
    store, backup, _ = opened(hub, tmp_path)
    store.sign_in("nobody", "Analytical-Engine1")
    assert not backup.pending and backup.flush()
    assert "upload_file" not in hub.calls


def test_changes_in_a_burst_make_one_upload_and_uploads_are_spaced_out(hub, opened, tmp_path):
    store, backup, _ = opened(hub, tmp_path, delay=0.3, min_interval=0.8)
    ada = store.sign_up(**ADA)
    store.sign_up(**GRACE)
    tokens = [store.start_session(ada) for _ in range(3)]
    wait_until(lambda: backup.uploads == 1)
    time.sleep(0.4)
    assert backup.uploads == 1
    assert hub.usernames(tmp_path) == ["ada", "grace"]

    store.end_session(tokens[0])
    wait_until(lambda: backup.uploads == 2)
    first, second = hub.upload_times
    assert second - first >= 0.75


def test_a_failed_upload_is_tried_again_later(hub, opened, tmp_path, caplog):
    now = [0.0]
    store, backup, _ = opened(hub, tmp_path, clock=lambda: now[0], min_interval=120)
    store.sign_up(**ADA)
    hub.failures["upload_file"] = 1
    with caplog.at_level(logging.WARNING, logger="slideseek.backup"):
        assert not backup.flush()
    assert backup.pending
    assert "trying again in 240 s" in caplog.text  # the wait doubles after each failure
    assert backup.flush()
    assert hub.usernames(tmp_path) == ["ada"]


def test_closing_uploads_what_is_left_and_can_be_repeated(hub, opened, tmp_path):
    store, backup, _ = opened(hub, tmp_path)
    store.sign_up(**ADA)
    assert backup.close() and backup.close()
    assert backup.uploads == 1
    assert hub.usernames(tmp_path) == ["ada"]


# ---- forgetting deleted accounts -----------------------------------------------------------


def test_a_deleted_account_is_gone_from_the_history_and_from_earlier_copies(hub, opened, tmp_path):
    store, backup, _ = opened(hub, tmp_path)
    ada = store.sign_up(**ADA)
    backup.flush()
    store.sign_up(**GRACE)
    backup.flush()
    repo = hub.repos[REPO]
    assert len(repo["commits"]) == 3 and len(repo["lfs"]) == 2

    assert store.delete_account(ada, ADA["password"])
    assert backup.flush()
    assert repo["commits"] == ["The accounts"]
    assert list(repo["lfs"]) == [hashlib.sha256(hub.copy()).hexdigest()]
    assert hub.usernames(tmp_path) == ["grace"]


def test_the_history_is_squashed_every_so_often_anyway(hub, opened, tmp_path):
    store, backup, _ = opened(hub, tmp_path, squash_every=3)
    ada = store.sign_up(**ADA)
    for _ in range(2):
        store.start_session(ada)
        backup.flush()
    assert "super_squash_history" not in hub.calls
    store.start_session(ada)
    backup.flush()
    assert hub.calls.count("super_squash_history") == 1
    assert len(hub.repos[REPO]["commits"]) == 1


def test_a_failed_squash_is_tried_again_without_another_upload(hub, opened, tmp_path):
    store, backup, _ = opened(hub, tmp_path)
    ada = store.sign_up(**ADA)
    store.delete_account(ada, ADA["password"])
    hub.failures["super_squash_history"] = 1
    assert not backup.flush() and backup.pending
    assert backup.flush()
    assert hub.calls.count("upload_file") == 1
    assert hub.repos[REPO]["commits"] == ["The accounts"]


def test_earlier_copies_are_kept_if_the_hub_does_not_list_the_current_one(
    hub, opened, tmp_path, monkeypatch
):
    store, backup, _ = opened(hub, tmp_path)
    ada = store.sign_up(**ADA)
    backup.flush()
    monkeypatch.setattr(
        hub, "list_lfs_files", lambda *a, **k: [SimpleNamespace(file_oid="0" * 64, size=1)]
    )
    store.delete_account(ada, ADA["password"])
    assert backup.flush()
    assert "permanently_delete_lfs_files" not in hub.calls


def test_deletions_are_told_apart_by_the_name_of_the_change(hub, tmp_path):
    fast_accounts(f"sqlite:///{tmp_path / 'accounts.db'}").sign_up(**ADA)
    backup = DatasetBackup(REPO, tmp_path / "accounts.db", api=hub, delay=3600)
    backup.changed("session started")
    assert backup.close(timeout=1)
    assert "super_squash_history" not in hub.calls
    backup.changed(ACCOUNT_DELETED)
    assert backup.close(timeout=1)
    assert "super_squash_history" in hub.calls


def test_an_admins_change_squashes_too_so_the_old_details_go(hub, tmp_path):
    fast_accounts(f"sqlite:///{tmp_path / 'accounts.db'}").sign_up(**ADA)
    backup = DatasetBackup(REPO, tmp_path / "accounts.db", api=hub, delay=3600)
    backup.changed(ACCOUNT_UPDATED)
    assert backup.close(timeout=1)
    assert "super_squash_history" in hub.calls


def test_a_missing_database_is_not_uploaded_as_an_empty_one(hub, tmp_path):
    backup = DatasetBackup(REPO, tmp_path / "accounts.db", api=hub, delay=3600)
    backup.changed()
    assert not backup.close(timeout=1)
    assert not backup.path.exists() and FILENAME not in hub.repos[REPO]["files"]


# ---- stopping ------------------------------------------------------------------------------


def test_sigterm_runs_the_cleanup_and_then_what_it_used_to(monkeypatch):
    seen = []
    original = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda signum, frame: seen.append("as before"))
    try:
        assert on_sigterm(lambda: seen.append("cleanup"))
        os.kill(os.getpid(), signal.SIGTERM)
        wait_until(lambda: len(seen) == 2)
        assert seen == ["cleanup", "as before"]
    finally:
        signal.signal(signal.SIGTERM, original)


def test_sigterm_can_only_be_handled_from_the_main_thread():
    result = []
    thread = threading.Thread(target=lambda: result.append(on_sigterm(lambda: None)))
    thread.start()
    thread.join()
    assert result == [False]


# ---- configuration -------------------------------------------------------------------------


@pytest.fixture()
def quiet_exit(monkeypatch):
    """No atexit or signal handlers left behind by the store under test."""
    monkeypatch.setattr(backup_module.atexit, "register", lambda function: None)
    monkeypatch.setattr(backup_module, "on_sigterm", lambda cleanup: True)


def test_the_environment_turns_the_copy_on(hub, tmp_path, monkeypatch, quiet_exit):
    tokens = []

    def make_api(token=None):
        tokens.append(token)
        return hub

    monkeypatch.setattr("huggingface_hub.HfApi", make_api)
    monkeypatch.delenv("SLIDESEEK_DATABASE_URL", raising=False)
    monkeypatch.setenv("SLIDESEEK_ACCOUNTS_REPO", REPO)
    monkeypatch.setenv("SLIDESEEK_ACCOUNTS_TOKEN", "hf_test_token")
    store = accounts_from_env(f"sqlite:///{tmp_path / 'accounts.db'}")
    assert tokens == ["hf_test_token"]
    assert store.url == f"sqlite:///{tmp_path / 'accounts.db'}"
    backup = store.on_change.__self__
    store.sign_up(**ADA)
    assert backup.flush()
    assert hub.usernames(tmp_path) == ["ada"]
    backup.close(timeout=1)


def test_a_database_url_wins_over_the_dataset(hub, tmp_path, monkeypatch, caplog):
    monkeypatch.setattr("huggingface_hub.HfApi", lambda token=None: hub)
    monkeypatch.setenv("SLIDESEEK_DATABASE_URL", f"sqlite:///{tmp_path / 'elsewhere.db'}")
    monkeypatch.setenv("SLIDESEEK_ACCOUNTS_REPO", REPO)
    with caplog.at_level(logging.WARNING):
        store = accounts_from_env()
    assert "SLIDESEEK_ACCOUNTS_REPO is ignored" in caplog.text
    assert store.on_change is None and hub.calls == []


def test_the_token_comes_from_its_own_variable_or_from_hf_token(monkeypatch):
    monkeypatch.delenv("SLIDESEEK_ACCOUNTS_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert accounts_token_from_env() is None
    monkeypatch.setenv("HF_TOKEN", "hf_general")
    assert accounts_token_from_env() == "hf_general"
    monkeypatch.setenv("SLIDESEEK_ACCOUNTS_TOKEN", "hf_accounts_only")
    assert accounts_token_from_env() == "hf_accounts_only"


def test_backed_up_accounts_restores_before_it_opens(hub, tmp_path):
    hub.add(copy=database_bytes(tmp_path, ADA))
    store, backup = backed_up_accounts(
        REPO, f"sqlite:///{tmp_path / 'disk.db'}", api=hub, watch=False
    )
    assert store.count() == 1 and store.on_change == backup.changed
    backup.close(timeout=1)


def test_only_a_sqlite_file_can_be_copied():
    assert sqlite_path("sqlite:///accounts.db") == Path("accounts.db")
    assert sqlite_path("sqlite:////data/a.db") == Path("/data/a.db")
    for url in ("sqlite://", "sqlite:///:memory:", "postgresql+psycopg://u:p@h/db"):
        with pytest.raises(ValueError):
            sqlite_path(url)


def wait_until(condition, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("timed out")
        time.sleep(0.01)
