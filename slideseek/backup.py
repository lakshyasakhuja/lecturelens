"""A copy of the accounts in a private Hugging Face dataset, for a server whose disk is wiped.

A free Hugging Face Space loses its disk whenever it restarts: after a deploy, after it has
slept, after a crash. Persistent storage costs money, and a PostgreSQL server may be out of
reach, since a Space can only connect out on ports 80, 443 and 8080. So the accounts stay in
a SQLite file on the Space's disk, as they would anywhere else, and a copy of that file is
kept in a private dataset repository on the Hub, which is reached over HTTPS:

    at start-up      the copy is downloaded, before the database is opened
    after a change   a snapshot of the file is uploaded: ten seconds after the first change,
                     so that a burst of changes makes one commit, and never more often than
                     every two minutes, which keeps well inside the Hub's limit on commits
    on the way out   whatever changed since the last upload is uploaded

A restart therefore loses nothing; a crash loses at most the last two minutes.

Three rules keep the copy safe:

* Nothing is uploaded to a public repository. The dataset is checked before every upload.
* If the copy exists but cannot be downloaded, or is damaged, start-up fails. Starting with
  no accounts instead would upload an empty database over the real one at the first change.
* When an account is deleted, or an admin changes one, the dataset's history is squashed,
  and the earlier copies are deleted from the Hub's file storage, so the account, or what
  it said before, does not live on in an old commit. The history is also squashed every 25
  uploads, to keep it short.
"""

from __future__ import annotations

import atexit
import hashlib
import logging
import os
import shutil
import signal
import sqlite3
import tempfile
import threading
import time
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import Any

from sqlalchemy import make_url

from slideseek.accounts import ACCOUNT_DELETED, ACCOUNT_UPDATED, AccountStore

log = logging.getLogger(__name__)

# Changes that leave details in the earlier copies that should be gone: squash after them.
ERASING = frozenset({ACCOUNT_DELETED, ACCOUNT_UPDATED})

FILENAME = "accounts.db"
# Seconds from the first change to its upload, so that a burst of changes makes one commit.
DELAY = 10.0
# The least time between two uploads. The Hub limits commits per repository per hour.
MIN_INTERVAL = 120.0
# After a failed upload the wait doubles, up to this.
MAX_BACKOFF = 30 * 60.0
SQUASH_EVERY = 25
# Files SQLite may keep beside a database; a restored copy must not be mixed with them.
SIDE_FILES = ("-journal", "-wal", "-shm")


class BackupError(RuntimeError):
    """The copy of the accounts cannot be used as it has to be. The message says why."""


def _not_found_errors() -> tuple[type[Exception], ...]:
    from huggingface_hub.errors import EntryNotFoundError, RevisionNotFoundError

    return EntryNotFoundError, RevisionNotFoundError


def _repository_not_found() -> type[Exception]:
    from huggingface_hub.errors import RepositoryNotFoundError

    return RepositoryNotFoundError


def _read_only(path: Path) -> sqlite3.Connection:
    # Read-only, so that a missing file is an error rather than a new, empty database.
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def snapshot(source: Path, dest: Path) -> None:
    """A consistent copy of the SQLite database at ``source``, even while it is written to."""
    with closing(_read_only(source)) as db, closing(sqlite3.connect(dest)) as out:
        db.backup(out)


def check_database(path: Path) -> None:
    """Raise ``BackupError`` unless ``path`` is an undamaged SQLite database."""
    try:
        with closing(_read_only(path)) as db:
            result = db.execute("PRAGMA quick_check").fetchone()
    except sqlite3.DatabaseError as error:
        raise BackupError(f"the copy of the accounts is not a SQLite database: {error}") from error
    if result is None or result[0] != "ok":
        raise BackupError(f"the copy of the accounts is damaged: {result}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sqlite_path(url: str) -> Path:
    """The file a ``sqlite:///...`` URL names. Only a file can be copied to a dataset."""
    parsed = make_url(url)
    if not parsed.drivername.startswith("sqlite") or parsed.database in (None, "", ":memory:"):
        raise ValueError(f"a copy can only be kept of a SQLite file, not of {parsed.drivername}")
    return Path(parsed.database)


class DatasetBackup:
    """Keeps the SQLite database at ``path`` copied to the private dataset ``repo_id``.

    Call ``restore`` before the database is opened, give ``changed`` to the account store
    as its ``on_change`` (``watch`` does both that and ``close`` at exit), and call
    ``close`` on the way out. ``api`` is a ``huggingface_hub.HfApi``, made with ``token``
    unless one is given.
    """

    def __init__(
        self,
        repo_id: str,
        path: str | Path,
        *,
        token: str | None = None,
        api: Any | None = None,
        delay: float = DELAY,
        min_interval: float = MIN_INTERVAL,
        squash_every: int = SQUASH_EVERY,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if api is None:
            from huggingface_hub import HfApi

            api = HfApi(token=token)
        self.repo_id = repo_id
        self.path = Path(path)
        self.api = api
        self.delay, self.min_interval, self.squash_every = delay, min_interval, squash_every
        self._clock = clock
        # Re-entrant: ``close`` may run in a signal handler on a thread that holds it.
        self._lock = threading.Condition(threading.RLock())
        self._upload_lock = threading.RLock()
        self._dirty_since: float | None = None
        self._next_allowed = float("-inf")
        self._failures = 0
        self._deletions = 0  # accounts deleted or changed by an admin, all told
        self._squashed = 0  # of which this many are gone from the dataset's history
        self._since_squash = 0
        self._uploaded: str | None = None  # SHA-256 of the copy in the dataset
        self._thread: threading.Thread | None = None
        self._closed = False
        self.uploads = 0

    def __repr__(self) -> str:
        return f"DatasetBackup({self.repo_id!r}, {str(self.path)!r})"

    # -- the dataset

    def check_private(self) -> None:
        """Make sure the dataset exists and is private, creating it (private) if need be."""
        try:
            info = self.api.repo_info(self.repo_id, repo_type="dataset")
        except _repository_not_found() as missing:
            try:
                self.api.create_repo(self.repo_id, repo_type="dataset", private=True, exist_ok=True)
                info = self.api.repo_info(self.repo_id, repo_type="dataset")
            except Exception as error:
                raise BackupError(
                    f"cannot find or create the dataset {self.repo_id}. Create it as a private "
                    "dataset on huggingface.co, and give the token write access to it "
                    f"({error})"
                ) from missing
            log.info("created the private dataset %s for the accounts", self.repo_id)
        if getattr(info, "private", None) is not True:
            raise BackupError(
                f"the dataset {self.repo_id} is public. Accounts are never copied to a public "
                "dataset: make it private under its Settings on huggingface.co."
            )

    def restore(self, *, attempts: int = 4, wait: float = 2.0) -> bool:
        """Put the dataset's copy of the database at ``path``, replacing any file there.

        Returns ``False`` if the dataset has no copy yet, which is how a first start looks;
        a database already at ``path`` is then kept, and uploaded soon. Raises
        ``BackupError`` if there is a copy that cannot be fetched after ``attempts`` tries,
        or that is damaged.
        """
        not_found = _not_found_errors()
        last: Exception | None = None
        for attempt in range(attempts):
            if attempt:
                time.sleep(wait * 2 ** (attempt - 1))
            try:
                self.check_private()
                with tempfile.TemporaryDirectory() as folder:
                    try:
                        fetched = self.api.hf_hub_download(
                            self.repo_id, FILENAME, repo_type="dataset", local_dir=folder
                        )
                    except not_found:
                        return self._first_start()
                    self._put_in_place(Path(fetched))
                log.info("restored the accounts from the dataset %s", self.repo_id)
                return True
            except BackupError:
                raise
            except Exception as error:  # the hub client raises many unrelated types
                last = error
                log.warning(
                    "could not download the accounts from %s (try %d of %d): %s",
                    self.repo_id,
                    attempt + 1,
                    attempts,
                    error,
                )
        raise BackupError(
            f"could not download the accounts from {self.repo_id}, so not starting: without "
            f"them the first change would overwrite them with nothing. Last error: {last}"
        ) from last

    def _first_start(self) -> bool:
        if self.path.exists():
            log.info("the dataset %s has no accounts yet; uploading the ones here", self.repo_id)
            self.changed()
        else:
            log.info("the dataset %s has no accounts yet; starting with none", self.repo_id)
        return False

    def _put_in_place(self, fetched: Path) -> None:
        check_database(fetched)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        partial = self.path.with_name(self.path.name + ".restoring")
        shutil.copyfile(fetched, partial)
        for suffix in SIDE_FILES:
            self.path.with_name(self.path.name + suffix).unlink(missing_ok=True)
        os.replace(partial, self.path)
        self._uploaded = sha256(self.path)

    # -- changes

    def changed(self, change: str = "") -> None:
        """Note that the database has changed, so that it is uploaded soon. Meant to be an
        ``AccountStore.on_change``."""
        with self._lock:
            if self._dirty_since is None:
                self._dirty_since = self._clock()
            if change in ERASING:
                self._deletions += 1
            if self._closed:
                return  # ``close`` uploads what is left; no thread is started now
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="accounts-backup", daemon=True
                )
                self._thread.start()
            self._lock.notify_all()

    @property
    def pending(self) -> bool:
        """Has anything changed that is not in the dataset yet?"""
        with self._lock:
            return self._dirty_since is not None

    def _run(self) -> None:
        while True:
            with self._lock:
                while not self._closed and self._dirty_since is None:
                    self._lock.wait()
                if self._closed:
                    return
                due = max(self._dirty_since + self.delay, self._next_allowed)
                left = due - self._clock()
                if left > 0:
                    self._lock.wait(left)
                    continue
            self.flush()

    def flush(self) -> bool:
        """Upload now, if anything has changed. ``True`` once the dataset is up to date."""
        with self._upload_lock:
            with self._lock:
                if self._dirty_since is None:
                    return True
                deletions = self._deletions
                # Changes from here on are for the next upload.
                self._dirty_since = None
            try:
                self._upload(deletions)
            except Exception as error:  # the hub client raises many unrelated types
                with self._lock:
                    self._failures += 1
                    backoff = min(self.min_interval * 2**self._failures, MAX_BACKOFF)
                    self._next_allowed = self._clock() + backoff
                    if self._dirty_since is None:
                        self._dirty_since = self._clock()
                log.warning(
                    "could not copy the accounts to %s; trying again in %d s: %s",
                    self.repo_id,
                    backoff,
                    error,
                )
                return False
            with self._lock:
                self._failures = 0
                self._next_allowed = self._clock() + self.min_interval
            return True

    def _upload(self, deletions: int) -> None:
        self.check_private()
        with tempfile.TemporaryDirectory() as folder:
            copy = Path(folder) / FILENAME
            snapshot(self.path, copy)
            digest = sha256(copy)
            if digest != self._uploaded:
                self.api.upload_file(
                    path_or_fileobj=str(copy),
                    path_in_repo=FILENAME,
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    commit_message="Back up the accounts",
                )
                self._uploaded = digest
                self.uploads += 1
                self._since_squash += 1
        if deletions > self._squashed or self._since_squash >= self.squash_every:
            self.api.super_squash_history(
                self.repo_id, repo_type="dataset", commit_message="The accounts"
            )
            self._since_squash = 0
            self._squashed = deletions
            self._forget_earlier_copies(digest)
            log.info("squashed the history of %s", self.repo_id)

    def _forget_earlier_copies(self, current: str) -> None:
        """Delete the files of earlier copies from the Hub. Squashing removes the commits
        that pointed at them, but not the files themselves."""
        try:
            files = list(self.api.list_lfs_files(self.repo_id, repo_type="dataset"))
            if not any(item.file_oid == current for item in files):
                # Not the bookkeeping this expects: leave every file alone rather than
                # risk deleting the copy that the dataset points at now.
                log.warning("the Hub does not list the current copy; earlier ones were kept")
                return
            earlier = [item for item in files if item.file_oid != current]
            if earlier:
                self.api.permanently_delete_lfs_files(self.repo_id, earlier, repo_type="dataset")
                log.info("deleted %d earlier copies of the accounts", len(earlier))
        except Exception as error:  # the copy is up to date; this is tidying up
            log.warning("could not delete earlier copies of the accounts: %s", error)

    # -- stopping

    def close(self, timeout: float = 20.0) -> bool:
        """Upload whatever has not been uploaded, and stop. Safe to call more than once.
        ``True`` if the dataset is up to date."""
        with self._lock:
            self._closed = True
            self._lock.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
        done = self.flush()
        if not done:
            log.error("the last changes to the accounts were not copied to %s", self.repo_id)
        return done

    def watch(self, store: AccountStore) -> None:
        """Copy the database whenever ``store`` changes it, and once more at exit."""
        store.on_change = self.changed
        atexit.register(self.close)
        on_sigterm(self.close)


def on_sigterm(cleanup: Callable[[], Any]) -> bool:
    """Run ``cleanup`` when the process is asked to stop, then stop as it would have.

    A Space is stopped with SIGTERM, which by default ends Python without running ``atexit``
    handlers. Only the main thread can set a handler; elsewhere this does nothing and
    returns ``False``.
    """
    if threading.current_thread() is not threading.main_thread():
        return False
    previous = signal.getsignal(signal.SIGTERM)

    def stop(signum: int, frame: Any) -> None:
        try:
            cleanup()
        finally:
            signal.signal(signal.SIGTERM, previous if previous is not None else signal.SIG_DFL)
            if callable(previous):
                previous(signum, frame)
            else:
                os.kill(os.getpid(), signum)

    signal.signal(signal.SIGTERM, stop)
    return True


def backed_up_accounts(
    repo_id: str,
    url: str,
    *,
    token: str | None = None,
    api: Any | None = None,
    watch: bool = True,
    **store_options: Any,
) -> tuple[AccountStore, DatasetBackup]:
    """An account store in the SQLite file at ``url``, restored from and copied to the
    private dataset ``repo_id``. Raises ``BackupError`` if the copy cannot be restored."""
    backup = DatasetBackup(repo_id, sqlite_path(url), token=token, api=api)
    backup.restore()
    store = AccountStore(f"sqlite:///{backup.path}", **store_options)
    if watch:
        backup.watch(store)
    else:
        store.on_change = backup.changed
    return store, backup
