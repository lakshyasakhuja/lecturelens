"""Accounts: who may search, and how they prove it.

Every search needs an account. A visitor signs up once with their name, a username, an email
address, a phone number and a password; from then on a session cookie says who they are.

What is checked

    name       1 to 80 characters
    username   3 to 30 letters, numbers, dots, dashes or underscores, starting with a letter or
               a number. Unique, ignoring case: "Ada" and "ada" are the same username.
    email      a well-formed address (email-validator). Unique, ignoring case.
    phone      a valid number for its country (libphonenumber), kept in E.164 form: +61412345678
    password   8 to 128 characters, with an uppercase letter, a lowercase letter, a number and
               a special character (any punctuation or symbol)

How it is kept

    A password is hashed with Argon2id, the winner of the Password Hashing Competition and
    OWASP's first choice, using argon2-cffi's defaults (the second setting RFC 9106
    recommends: 64 MiB of memory, 3 passes, 4 lanes) and a random salt of its own. Only the
    hash is stored. The password is never written down, logged or sent back.

    A session is 256 random bits. The visitor's cookie holds the token and the database holds
    only its SHA-256, so a copy of the database cannot be used to sign in as anyone.

    Any SQL database that SQLAlchemy can reach will do: SQLite on a laptop, PostgreSQL where
    the accounts have to outlive the server. A Hugging Face Space's disk is wiped whenever
    it restarts, so the hosted demo keeps them in SQLite and a copy of the file in a private
    dataset (``slideseek.backup``), told of every change through ``AccountStore.on_change``.
"""

from __future__ import annotations

import hashlib
import logging
import re
import secrets
import threading
import time
import unicodedata
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import phonenumbers
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from email_validator import EmailNotValidError, validate_email
from sqlalchemy import (
    BigInteger,
    Column,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    delete,
    func,
    insert,
    make_url,
    or_,
    select,
    update,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import StaticPool

log = logging.getLogger(__name__)

MIN_USERNAME, MAX_USERNAME = 3, 30
MIN_PASSWORD, MAX_PASSWORD = 8, 128
MAX_NAME = 80
MAX_EMAIL = 254
SESSION_DAYS = 30
# How long a signed-in visitor is remembered in memory before the database is asked again.
SESSION_CACHE_SECONDS = 60

USERNAME_PATTERN = re.compile(
    rf"[A-Za-z0-9][A-Za-z0-9._-]{{{MIN_USERNAME - 1},{MAX_USERNAME - 1}}}"
)
# Names that would look official on a page that showed them.
RESERVED_USERNAMES = frozenset(
    {
        "account",
        "accounts",
        "admin",
        "administrator",
        "api",
        "help",
        "login",
        "logout",
        "moderator",
        "root",
        "security",
        "signin",
        "signup",
        "slideseek",
        "staff",
        "support",
        "system",
    }
)

# The password rules, in the order the page lists them. The page checks the same rules as
# the visitor types; this module is the one that decides.
PASSWORD_RULES = {
    "length": f"At least {MIN_PASSWORD} characters",
    "upper": "An uppercase letter",
    "lower": "A lowercase letter",
    "number": "A number",
    "special": "A special character, such as ! ? # or @",
}

USERNAME_RULE = (
    f"Use {MIN_USERNAME} to {MAX_USERNAME} letters, numbers, dots, dashes or underscores, "
    "starting with a letter or a number."
)
PASSWORD_RULE = (
    f"Use at least {MIN_PASSWORD} characters, with an uppercase letter, a lowercase letter, "
    "a number and a special character."
)

# What ``AccountStore.on_change`` is told, after the change has been committed.
ACCOUNT_CREATED = "account created"
ACCOUNT_DELETED = "account deleted"
PASSWORD_REHASHED = "password rehashed"
SESSION_STARTED = "session started"
SESSION_ENDED = "session ended"

metadata = MetaData()

users = Table(
    "slideseek_users",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("username", String(MAX_USERNAME), nullable=False),
    # Uniqueness ignores case, so it is enforced on a lower-cased copy.
    Column("username_key", String(MAX_USERNAME), nullable=False, unique=True),
    Column("name", String(MAX_NAME), nullable=False),
    Column("email", String(MAX_EMAIL), nullable=False),
    Column("email_key", String(MAX_EMAIL), nullable=False, unique=True),
    Column("phone", String(20), nullable=False),
    Column("password_hash", String(255), nullable=False),
    Column("created_at", BigInteger, nullable=False),
)

sessions = Table(
    "slideseek_sessions",
    metadata,
    Column("token_hash", String(64), primary_key=True),
    Column(
        "user_id",
        Integer,
        ForeignKey("slideseek_users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    Column("created_at", BigInteger, nullable=False),
    Column("expires_at", BigInteger, nullable=False, index=True),
)


class FieldError(ValueError):
    """One field of a form is not acceptable; the message says why, for the visitor."""


class AccountError(ValueError):
    """An account cannot be created as asked. ``errors`` says why, field by field.

    ``conflict`` is set when the only trouble is that the username or the email address
    already belongs to an account.
    """

    def __init__(self, errors: Mapping[str, str], *, conflict: bool = False) -> None:
        self.errors = dict(errors)
        self.conflict = conflict
        super().__init__(" ".join(self.errors.values()))


# ---- checking what a visitor typed ---------------------------------------------------------


def check_name(name: Any) -> str:
    name = " ".join(str(name or "").split())
    if not name:
        raise FieldError("Enter your name.")
    if len(name) > MAX_NAME:
        raise FieldError(f"Use at most {MAX_NAME} characters.")
    if any(unicodedata.category(char).startswith("C") for char in name):
        raise FieldError("Your name has characters in it that cannot be shown.")
    return name


def check_username(username: Any) -> str:
    username = str(username or "").strip()
    if not username:
        raise FieldError("Choose a username.")
    if not USERNAME_PATTERN.fullmatch(username):
        raise FieldError(USERNAME_RULE)
    if username.lower() in RESERVED_USERNAMES:
        raise FieldError("That username is reserved. Choose another.")
    return username


def check_email(email: Any) -> str:
    """The address in normalised form: the domain lower-cased, Unicode in one form."""
    email = str(email or "").strip()
    if not email:
        raise FieldError("Enter your email address.")
    if len(email) > MAX_EMAIL:
        raise FieldError("That email address is too long.")
    try:
        return validate_email(email, check_deliverability=False).normalized
    except EmailNotValidError as error:
        raise FieldError(str(error)) from error


def email_key(email: str) -> str:
    """What two addresses must share to count as the same: case is ignored throughout."""
    try:
        email = validate_email(email.strip(), check_deliverability=False).normalized
    except EmailNotValidError:
        email = email.strip()
    return email.lower()


def check_phone(number: Any, country: Any = "") -> str:
    """The number in E.164 form. ``country`` is a region code (``AU``) for a local number;
    a number written with its country code (``+61 ...``) needs none."""
    number = str(number or "").strip()
    region = str(country or "").strip().upper() or None
    if not number:
        raise FieldError("Enter your phone number.")
    if not number.startswith("+") and region not in phonenumbers.SUPPORTED_REGIONS:
        raise FieldError("Choose the country or region of your phone number.")
    try:
        parsed = phonenumbers.parse(number, None if number.startswith("+") else region)
    except phonenumbers.NumberParseException as error:
        raise FieldError("Enter a phone number, such as 412 345 678.") from error
    if not phonenumbers.is_valid_number(parsed):
        code = (
            f"+{phonenumbers.country_code_for_region(region)}"
            if region and not number.startswith("+")
            else "its country code"
        )
        raise FieldError(f"That is not a valid phone number with {code}.")
    return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)


def format_phone(e164: str) -> str:
    """``+61412345678`` as people write it: ``+61 412 345 678``."""
    try:
        parsed = phonenumbers.parse(e164, None)
    except phonenumbers.NumberParseException:
        return e164
    return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.INTERNATIONAL)


def unmet_password_rules(password: str) -> list[str]:
    """The names of the ``PASSWORD_RULES`` that ``password`` does not meet."""
    kinds = {unicodedata.category(char) for char in password}
    unmet = []
    if len(password) < MIN_PASSWORD:
        unmet.append("length")
    if "Lu" not in kinds:
        unmet.append("upper")
    if "Ll" not in kinds:
        unmet.append("lower")
    if "Nd" not in kinds:
        unmet.append("number")
    # Punctuation (P*) or a symbol (S*): ! ? # @ € + and the like. A space is neither.
    if not any(kind[0] in "PS" for kind in kinds):
        unmet.append("special")
    return unmet


def check_password(password: Any) -> str:
    password = password if isinstance(password, str) else ""
    if not password:
        raise FieldError("Choose a password.")
    if len(password) > MAX_PASSWORD:
        raise FieldError(f"Use at most {MAX_PASSWORD} characters.")
    if unmet_password_rules(password):
        raise FieldError(PASSWORD_RULE)
    return password


def phone_regions() -> list[dict[str, Any]]:
    """Every region a phone number can be from, with its calling code: ``AU`` is 61."""
    return [
        {"region": region, "code": phonenumbers.country_code_for_region(region)}
        for region in sorted(phonenumbers.SUPPORTED_REGIONS)
    ]


# ---- accounts and sessions -----------------------------------------------------------------


@dataclass(frozen=True)
class Account:
    id: int
    username: str
    name: str
    email: str
    phone: str  # E.164
    created_at: int  # Unix time

    def public(self) -> dict[str, Any]:
        """What the account's own holder is shown. Never the password hash."""
        return {
            "username": self.username,
            "name": self.name,
            "email": self.email,
            "phone": format_phone(self.phone),
            "created": datetime.fromtimestamp(self.created_at, timezone.utc).date().isoformat(),
        }


def _account(row: Any) -> Account:
    return Account(
        id=row.id,
        username=row.username,
        name=row.name,
        email=row.email,
        phone=row.phone,
        created_at=row.created_at,
    )


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def database_url(url: str) -> str:
    """A URL SQLAlchemy can open. Hosts hand out ``postgres://`` URLs; that means psycopg."""
    url = url.strip()
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix) :]
    return url


def is_in_memory(url: str) -> bool:
    return url in {"sqlite://", "sqlite:///:memory:"} or "mode=memory" in url


def redacted(url: str) -> str:
    """The URL with any password masked, for a log line."""
    return make_url(url).render_as_string(hide_password=True)


class AccountStore:
    """The accounts and their sessions, in the SQL database at ``url``.

    ``sqlite://`` keeps them in memory, for as long as the process runs; a path such as
    ``sqlite:///accounts.db`` keeps them in a file; a ``postgresql://`` URL keeps them in
    PostgreSQL. The tables are created if they are not there.

    ``on_change``, if set, is called with one of the names above (``ACCOUNT_CREATED`` and
    so on) after each change to the database has been committed.
    """

    on_change: Callable[[str], None] | None = None

    def __init__(
        self,
        url: str = "sqlite://",
        *,
        hasher: PasswordHasher | None = None,
        session_days: float = SESSION_DAYS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.url = database_url(url)
        self.persistent = not is_in_memory(self.url)
        if self.url.startswith("sqlite"):
            options: dict[str, Any] = {"connect_args": {"check_same_thread": False, "timeout": 15}}
            if not self.persistent:
                # One connection shared by every thread: each new one would be a new database.
                options["poolclass"] = StaticPool
        else:
            # A database that sleeps when idle drops its connections; test each before use.
            options = {"pool_pre_ping": True, "pool_recycle": 300, "pool_size": 5}
        self.engine = create_engine(self.url, **options)
        metadata.create_all(self.engine)
        self.hasher = hasher or PasswordHasher()
        self.session_seconds = int(session_days * 86400)
        self._clock = clock
        self._cache: dict[str, tuple[Account, float, int]] = {}
        self._cache_lock = threading.Lock()
        self._decoy: str | None = None

    def __repr__(self) -> str:
        return f"AccountStore({redacted(self.url)!r})"

    def _changed(self, change: str) -> None:
        if self.on_change is None:
            return
        try:
            self.on_change(change)
        except Exception:  # the change is made; whoever was told of it failed, not the change
            log.exception("on_change failed after: %s", change)

    # -- signing up

    def sign_up(
        self,
        *,
        name: Any,
        username: Any,
        email: Any,
        phone: Any,
        country: Any = "",
        password: Any,
    ) -> Account:
        """Create an account. Raises ``AccountError`` naming every field that needs fixing."""
        checks = {
            "name": lambda: check_name(name),
            "username": lambda: check_username(username),
            "email": lambda: check_email(email),
            "phone": lambda: check_phone(phone, country),
            "password": lambda: check_password(password),
        }
        values, errors = {}, {}
        for field, check in checks.items():
            try:
                values[field] = check()
            except FieldError as error:
                errors[field] = str(error)
        if errors:
            raise AccountError(errors)

        username_key, address_key = values["username"].lower(), email_key(values["email"])
        taken = self._taken(username_key, address_key)
        if taken:
            raise AccountError(taken, conflict=True)

        row = {
            "username": values["username"],
            "username_key": username_key,
            "name": values["name"],
            "email": values["email"],
            "email_key": address_key,
            "phone": values["phone"],
            "password_hash": self.hasher.hash(values["password"]),
            "created_at": int(self._clock()),
        }
        try:
            with self.engine.begin() as db:
                account_id = db.execute(insert(users).values(**row)).inserted_primary_key[0]
        except IntegrityError:
            # Someone else took the name or the address between the check and the insert.
            taken = self._taken(username_key, address_key)
            raise AccountError(
                taken or {"username": "That username is taken. Choose another."}, conflict=True
            ) from None
        log.info("account created: %s", values["username"])
        self._changed(ACCOUNT_CREATED)
        return Account(
            id=account_id,
            username=row["username"],
            name=row["name"],
            email=row["email"],
            phone=row["phone"],
            created_at=row["created_at"],
        )

    def _taken(self, username_key: str, address_key: str) -> dict[str, str]:
        with self.engine.connect() as db:
            rows = db.execute(
                select(users.c.username_key, users.c.email_key).where(
                    or_(users.c.username_key == username_key, users.c.email_key == address_key)
                )
            ).all()
        errors = {}
        if any(row.username_key == username_key for row in rows):
            errors["username"] = "That username is taken. Choose another."
        if any(row.email_key == address_key for row in rows):
            errors["email"] = "An account with this email address already exists. Sign in instead."
        return errors

    # -- signing in

    def sign_in(self, login: Any, password: Any) -> Account | None:
        """The account for a username or an email address and its password, or ``None``.

        A wrong password and an unknown account take the same time to say no, so how long
        the answer took gives away nothing.
        """
        login = str(login or "").strip()
        password = password if isinstance(password, str) else ""
        row = None
        if login and password and len(password) <= MAX_PASSWORD:
            column, key = (
                (users.c.email_key, email_key(login))
                if "@" in login
                else (users.c.username_key, login.lower())
            )
            with self.engine.connect() as db:
                row = db.execute(select(users).where(column == key)).first()
        if row is None:
            self._decoy_check(password)
            return None
        if not self._verify(row.password_hash, password):
            return None
        if self.hasher.check_needs_rehash(row.password_hash):
            # The hashing settings have been raised since this password was set.
            with self.engine.begin() as db:
                db.execute(
                    update(users)
                    .where(users.c.id == row.id)
                    .values(password_hash=self.hasher.hash(password))
                )
            self._changed(PASSWORD_REHASHED)
        return _account(row)

    def _verify(self, password_hash: str, password: str) -> bool:
        try:
            return self.hasher.verify(password_hash, password)
        except (VerifyMismatchError, VerificationError, InvalidHashError):
            return False

    def _decoy_check(self, password: str) -> None:
        if self._decoy is None:
            self._decoy = self.hasher.hash(secrets.token_urlsafe(16))
        self._verify(self._decoy, password[:MAX_PASSWORD] or "x")

    def start_session(self, account: Account) -> str:
        """A new session for ``account``. The token is returned once and never stored."""
        token = secrets.token_urlsafe(32)
        now = int(self._clock())
        with self.engine.begin() as db:
            db.execute(delete(sessions).where(sessions.c.expires_at <= now))
            db.execute(
                insert(sessions).values(
                    token_hash=_digest(token),
                    user_id=account.id,
                    created_at=now,
                    expires_at=now + self.session_seconds,
                )
            )
        self._changed(SESSION_STARTED)
        return token

    def account_for_token(self, token: str | None) -> Account | None:
        """Who a session token belongs to, or ``None`` if it is unknown or has expired."""
        if not token or len(token) > 128:
            return None
        key, now = _digest(token), self._clock()
        with self._cache_lock:
            cached = self._cache.get(key)
        if cached and cached[1] > time.monotonic() and cached[2] > now:
            return cached[0]
        with self.engine.connect() as db:
            row = db.execute(
                select(users, sessions.c.expires_at)
                .join(sessions, sessions.c.user_id == users.c.id)
                .where(sessions.c.token_hash == key)
            ).first()
        if row is None or row.expires_at <= now:
            return None
        account = _account(row)
        with self._cache_lock:
            if len(self._cache) > 10_000:
                self._cache.clear()
            self._cache[key] = (account, time.monotonic() + SESSION_CACHE_SECONDS, row.expires_at)
        return account

    def end_session(self, token: str | None) -> None:
        if not token or len(token) > 128:
            return
        key = _digest(token)
        with self._cache_lock:
            self._cache.pop(key, None)
        with self.engine.begin() as db:
            ended = db.execute(delete(sessions).where(sessions.c.token_hash == key)).rowcount
        if ended:
            self._changed(SESSION_ENDED)

    # -- leaving

    def delete_account(self, account: Account, password: Any) -> bool:
        """Delete ``account`` and end all its sessions, if ``password`` is its password."""
        password = password if isinstance(password, str) else ""
        with self.engine.connect() as db:
            row = db.execute(select(users).where(users.c.id == account.id)).first()
        if row is None or not password or not self._verify(row.password_hash, password):
            return False
        with self.engine.begin() as db:
            db.execute(delete(sessions).where(sessions.c.user_id == account.id))
            db.execute(delete(users).where(users.c.id == account.id))
        with self._cache_lock:
            for key in [key for key, (held, _, _) in self._cache.items() if held.id == account.id]:
                del self._cache[key]
        log.info("account deleted: %s", account.username)
        self._changed(ACCOUNT_DELETED)
        return True

    def count(self) -> int:
        with self.engine.connect() as db:
            return db.execute(select(func.count()).select_from(users)).scalar_one()


class Throttle:
    """At most ``limit`` events for one key in any ``window`` seconds, kept in memory.

    Used to slow down guessing: sign-in attempts per address and per account, sign-ups per
    address. Memory is bounded: keys whose events have all expired are dropped.
    """

    def __init__(
        self, limit: int, window: float, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.limit, self.window, self._clock = limit, window, clock
        self._events: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def _recent(self, key: str, now: float) -> deque[float]:
        events = self._events.get(key, deque())
        while events and events[0] <= now - self.window:
            events.popleft()
        return events

    def retry_after(self, key: str) -> int:
        """Seconds until another event is allowed for ``key``; 0 if it is allowed now."""
        now = self._clock()
        with self._lock:
            events = self._recent(key, now)
            if len(events) < self.limit:
                return 0
            return max(1, int(events[0] + self.window - now + 0.999))

    def record(self, key: str) -> None:
        now = self._clock()
        with self._lock:
            events = self._recent(key, now)
            events.append(now)
            self._events[key] = events
            if len(self._events) > 50_000:
                self._events = {
                    k: v for k, v in self._events.items() if v and v[-1] > now - self.window
                }

    def clear(self, key: str) -> None:
        with self._lock:
            self._events.pop(key, None)
