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

Admins

    An admin sees every account and can change any of them: their details, their password,
    whether they are an admin, signing them out everywhere and deleting them. An admin can
    also create accounts, admins included. Every one of those methods takes the admin who is
    asking, and checks again in the database that they are still one.

    The default admin is the account whose username ``set_default_admin`` names (the server
    reads it from ``SLIDESEEK_ADMIN``). It is made an admin at start-up, and it is protected:
    no other admin can change it, it cannot stop being an admin, be renamed or be deleted.
    While no account has that username, nobody can take it, so that nobody but its owner can
    become the default admin by signing up first.
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
    Boolean,
    Column,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    delete,
    false,
    func,
    insert,
    inspect,
    make_url,
    or_,
    select,
    text,
    true,
    update,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import StaticPool
from sqlalchemy.schema import CreateColumn

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
DEFAULT_ADMIN_STAYS = "The default admin account is always an admin, and cannot be deleted."
DEFAULT_ADMIN_NAME = "The default admin's username is set where the server runs, not here."

# What ``AccountStore.on_change`` is told, after the change has been committed.
ACCOUNT_CREATED = "account created"
ACCOUNT_UPDATED = "account updated"  # by an admin: details, password or admin rights
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
    # Added after the first release; ``AccountStore`` adds it to a table made without it.
    Column("is_admin", Boolean, nullable=False, server_default=false()),
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


class NotAllowed(PermissionError):
    """The one asking may not do this. The message says why, for them."""


class NoSuchAccount(LookupError):
    """No account has that id (any more)."""


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


def phone_parts(e164: str) -> tuple[str, str]:
    """``+61412345678`` as the sign-up form takes it: its region and the number as it is
    written there, ``("AU", "0412 345 678")``."""
    try:
        parsed = phonenumbers.parse(e164, None)
    except phonenumbers.NumberParseException:
        return "", e164
    region = phonenumbers.region_code_for_number(parsed) or ""
    if not region or region == "001":  # a number for no one country, such as a satellite
        return "", phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.INTERNATIONAL)
    return region, phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.NATIONAL)


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
    is_admin: bool = False

    def public(self) -> dict[str, Any]:
        """What the account's own holder is shown. Never the password hash."""
        return {
            "username": self.username,
            "name": self.name,
            "email": self.email,
            "phone": format_phone(self.phone),
            "created": datetime.fromtimestamp(self.created_at, timezone.utc).date().isoformat(),
            "admin": self.is_admin,
        }


@dataclass(frozen=True)
class AccountRecord:
    """An account as an admin sees it in the list: with how many sessions it has open."""

    account: Account
    sessions: int
    signed_in_at: int | None  # when the newest open session started


def _account(row: Any) -> Account:
    return Account(
        id=row.id,
        username=row.username,
        name=row.name,
        email=row.email,
        phone=row.phone,
        created_at=row.created_at,
        is_admin=bool(row.is_admin),
    )


def _like(query: str) -> str:
    """A ``LIKE`` pattern for text that contains ``query``, its wildcards taken literally."""
    escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


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

    ``default_admin`` is the lower-cased username of the default admin, or ``None``; set it
    with ``set_default_admin``.
    """

    on_change: Callable[[str], None] | None = None
    default_admin: str | None = None

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
        self._add_missing_columns()
        self.hasher = hasher or PasswordHasher()
        self.session_seconds = int(session_days * 86400)
        self._clock = clock
        self._cache: dict[str, tuple[Account, float, int]] = {}
        self._cache_lock = threading.Lock()
        self._decoy: str | None = None

    def __repr__(self) -> str:
        return f"AccountStore({redacted(self.url)!r})"

    def _add_missing_columns(self) -> None:
        """Add the columns a later version added to the table that an earlier one made.

        ``create_all`` makes missing tables but leaves existing ones alone, and the hosted
        demo's accounts were made before there were admins.
        """
        present = {column["name"] for column in inspect(self.engine).get_columns(users.name)}
        missing = [column for column in users.columns if column.name not in present]
        if not missing:
            return
        with self.engine.begin() as db:
            for column in missing:
                definition = CreateColumn(column).compile(dialect=self.engine.dialect)
                db.execute(text(f"ALTER TABLE {users.name} ADD COLUMN {definition}"))
        log.info("added to the accounts: %s", ", ".join(column.name for column in missing))

    def _changed(self, change: str) -> None:
        if self.on_change is None:
            return
        try:
            self.on_change(change)
        except Exception:  # the change is made; whoever was told of it failed, not the change
            log.exception("on_change failed after: %s", change)

    def _forget(self, account_id: int) -> None:
        """Drop what the session cache remembers of an account that has just changed."""
        with self._cache_lock:
            for key in [key for key, (held, _, _) in self._cache.items() if held.id == account_id]:
                del self._cache[key]

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
        fields = {"name": name, "username": username, "email": email, "password": password}
        return self._create({**fields, "phone": phone, "country": country}, is_admin=False)

    def _checked(self, fields: Mapping[str, Any]) -> dict[str, str]:
        """Check each of ``fields`` that is given: ``name``, ``username``, ``email``,
        ``phone`` (with ``country``) and ``password``. Raises ``AccountError`` naming every
        one that needs fixing."""
        checks = {
            "name": lambda: check_name(fields["name"]),
            "username": lambda: self._check_username(fields["username"]),
            "email": lambda: check_email(fields["email"]),
            "phone": lambda: check_phone(fields["phone"], fields.get("country", "")),
            "password": lambda: check_password(fields["password"]),
        }
        values, errors = {}, {}
        for field, check in checks.items():
            if field not in fields:
                continue
            try:
                values[field] = check()
            except FieldError as error:
                errors[field] = str(error)
        if errors:
            raise AccountError(errors)
        return values

    def _check_username(self, username: Any) -> str:
        username = check_username(username)
        if self.default_admin and username.lower() == self.default_admin:
            # Kept for the default admin, so that nobody else can take it while it is free.
            raise FieldError("That username is reserved. Choose another.")
        return username

    def _create(self, fields: Mapping[str, Any], *, is_admin: bool, own: bool = True) -> Account:
        values = self._checked(fields)
        username_key, address_key = values["username"].lower(), email_key(values["email"])
        taken = self._taken(username_key, address_key, own=own)
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
            "is_admin": is_admin,
        }
        try:
            with self.engine.begin() as db:
                account_id = db.execute(insert(users).values(**row)).inserted_primary_key[0]
        except IntegrityError:
            # Someone else took the name or the address between the check and the insert.
            taken = self._taken(username_key, address_key, own=own)
            raise AccountError(
                taken or {"username": "That username is taken. Choose another."}, conflict=True
            ) from None
        log.info("account created: %s%s", values["username"], " (an admin)" if is_admin else "")
        self._changed(ACCOUNT_CREATED)
        return Account(
            id=account_id,
            username=row["username"],
            name=row["name"],
            email=row["email"],
            phone=row["phone"],
            created_at=row["created_at"],
            is_admin=is_admin,
        )

    def _taken(
        self, username_key: str, address_key: str, *, own: bool = True, other_than: int = 0
    ) -> dict[str, str]:
        """Which of the username and the address another account has. ``own`` is for
        someone signing up for themselves; an admin is told it differently."""
        with self.engine.connect() as db:
            rows = db.execute(
                select(users.c.username_key, users.c.email_key).where(
                    or_(users.c.username_key == username_key, users.c.email_key == address_key),
                    users.c.id != other_than,
                )
            ).all()
        errors = {}
        if any(row.username_key == username_key for row in rows):
            errors["username"] = "That username is taken. Choose another."
        if any(row.email_key == address_key for row in rows):
            errors["email"] = (
                "An account with this email address already exists. Sign in instead."
                if own
                else "Another account has this email address."
            )
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
        """Delete ``account`` and end all its sessions, if ``password`` is its password.

        Raises ``NotAllowed`` for the default admin, which is never deleted.
        """
        if self.is_default_admin(account):
            raise NotAllowed(DEFAULT_ADMIN_STAYS)
        password = password if isinstance(password, str) else ""
        with self.engine.connect() as db:
            row = db.execute(select(users).where(users.c.id == account.id)).first()
        if row is None or not password or not self._verify(row.password_hash, password):
            return False
        self._delete(account)
        log.info("account deleted: %s", account.username)
        return True

    def _delete(self, account: Account) -> None:
        with self.engine.begin() as db:
            db.execute(delete(sessions).where(sessions.c.user_id == account.id))
            db.execute(delete(users).where(users.c.id == account.id))
        self._forget(account.id)
        self._changed(ACCOUNT_DELETED)

    def count(self) -> int:
        with self.engine.connect() as db:
            return db.execute(select(func.count()).select_from(users)).scalar_one()

    def count_admins(self) -> int:
        with self.engine.connect() as db:
            return db.execute(
                select(func.count()).select_from(users).where(users.c.is_admin == true())
            ).scalar_one()

    def get(self, account_id: int) -> Account | None:
        """The account with this id, as the database has it now."""
        with self.engine.connect() as db:
            row = db.execute(select(users).where(users.c.id == account_id)).first()
        return _account(row) if row else None

    # -- the default admin

    def is_default_admin(self, account: Account) -> bool:
        return self.default_admin is not None and account.username.lower() == self.default_admin

    def set_default_admin(self, username: str | None) -> Account | None:
        """Make the account called ``username`` the default admin, and an admin if it is not
        one yet. ``None`` or ``""`` means there is no default admin. While no account has the
        username, nobody can sign up with it (see the module's docstring)."""
        key = str(username or "").strip().removeprefix("@").lower()
        self.default_admin = key or None
        if not key:
            return None
        if not USERNAME_PATTERN.fullmatch(key):
            log.warning("the default admin %r is not a username that an account can have", key)
            return None
        with self.engine.connect() as db:
            row = db.execute(select(users).where(users.c.username_key == key)).first()
        if row is None:
            log.warning(
                "no account is called %s, so there is no default admin. The username is kept "
                "free for its owner: to claim it, start once without the setting, sign up as "
                "%s, and set it again",
                key,
                key,
            )
            return None
        account = _account(row)
        if account.is_admin:
            log.info("the default admin is %s", account.username)
            return account
        with self.engine.begin() as db:
            db.execute(update(users).where(users.c.id == account.id).values(is_admin=True))
        self._forget(account.id)
        log.info("made %s the default admin", account.username)
        self._changed(ACCOUNT_UPDATED)
        return self.get(account.id)

    # -- what admins can do

    def _admin(self, by: Account) -> Account:
        """``by`` as the database has it now, if it is still an admin."""
        actor = self.get(by.id)
        if actor is None or not actor.is_admin:
            raise NotAllowed("Only an admin can do that.")
        return actor

    def _target(self, account_id: int) -> Account:
        target = self.get(account_id)
        if target is None:
            raise NoSuchAccount("There is no such account. It may have just been deleted.")
        return target

    def _guard_default(self, actor: Account, target: Account) -> None:
        if self.is_default_admin(target) and actor.id != target.id:
            raise NotAllowed("Only the default admin can change the default admin's account.")

    def list_accounts(
        self, query: str = "", *, limit: int = 100, offset: int = 0
    ) -> tuple[list[AccountRecord], int]:
        """Accounts whose username, name or email address contains ``query``, newest first,
        and how many match in all. Only for an admin; the caller checks."""
        now = int(self._clock())
        open_sessions = (
            select(
                sessions.c.user_id,
                func.count().label("open"),
                func.max(sessions.c.created_at).label("newest"),
            )
            .where(sessions.c.expires_at > now)
            .group_by(sessions.c.user_id)
            .subquery()
        )
        condition = true()
        query = " ".join(str(query or "").split()).lower()[:MAX_EMAIL]
        if query:
            pattern = _like(query)
            condition = or_(
                users.c.username_key.like(pattern, escape="\\"),
                func.lower(users.c.name).like(pattern, escape="\\"),
                users.c.email_key.like(pattern, escape="\\"),
            )
        with self.engine.connect() as db:
            total = db.execute(
                select(func.count()).select_from(users).where(condition)
            ).scalar_one()
            rows = db.execute(
                select(users, open_sessions.c.open, open_sessions.c.newest)
                .outerjoin(open_sessions, open_sessions.c.user_id == users.c.id)
                .where(condition)
                .order_by(users.c.created_at.desc(), users.c.id.desc())
                .limit(max(1, min(limit, 500)))
                .offset(max(0, offset))
            ).all()
        records = [AccountRecord(_account(row), row.open or 0, row.newest) for row in rows]
        return records, total

    def create_account(self, by: Account, *, admin: bool = False, **fields: Any) -> Account:
        """Create an account for someone else: ``name``, ``username``, ``email``, ``phone``,
        ``country`` and ``password``, as for ``sign_up``. ``admin`` makes it an admin."""
        actor = self._admin(by)
        wanted = ("name", "username", "email", "phone", "country", "password")
        account = self._create(
            {field: fields.get(field, "") for field in wanted}, is_admin=bool(admin), own=False
        )
        log.info("%s created the account %s", actor.username, account.username)
        return account

    def update_account(
        self,
        by: Account,
        account_id: int,
        *,
        name: Any = None,
        username: Any = None,
        email: Any = None,
        phone: Any = None,
        country: Any = None,
        password: Any = None,
        admin: bool | None = None,
        keep_session: str | None = None,
    ) -> Account:
        """Change what is given of an account's details, its password and whether it is an
        admin. A new password signs the account out everywhere, except, when an admin changes
        their own, the session ``keep_session``."""
        actor = self._admin(by)
        target = self._target(account_id)
        self._guard_default(actor, target)
        default = self.is_default_admin(target)

        given = {"name": name, "email": email, "password": password}
        fields = {field: value for field, value in given.items() if value is not None}
        if phone is not None:
            fields.update(phone=phone, country=country or "")
        try:
            values, errors = self._checked(fields), {}
        except AccountError as error:
            values, errors = {}, dict(error.errors)
        wanted = None if username is None else str(username).strip()
        if wanted is not None and wanted != target.username:
            renamed = wanted.lower() != target.username.lower()
            try:
                if renamed and default:
                    raise FieldError(DEFAULT_ADMIN_NAME)
                # The same username written with other capitals is still the account's own.
                values["username"] = (self._check_username if renamed else check_username)(wanted)
            except FieldError as error:
                errors["username"] = str(error)
        if errors:
            raise AccountError(errors)

        changes: dict[str, Any] = {}
        if "name" in values and values["name"] != target.name:
            changes["name"] = values["name"]
        if "username" in values and values["username"] != target.username:
            changes["username"] = values["username"]
            changes["username_key"] = values["username"].lower()
        if "email" in values and values["email"] != target.email:
            changes["email"] = values["email"]
            changes["email_key"] = email_key(values["email"])
        if "phone" in values and values["phone"] != target.phone:
            changes["phone"] = values["phone"]
        if "password" in values:
            changes["password_hash"] = self.hasher.hash(values["password"])
        if admin is not None and bool(admin) != target.is_admin:
            if not admin and default:
                raise NotAllowed(DEFAULT_ADMIN_STAYS)
            if not admin and actor.id == target.id:
                raise NotAllowed("You cannot remove your own admin rights. Ask another admin.")
            changes["is_admin"] = bool(admin)
        if not changes:
            return target

        username_key = changes.get("username_key", target.username.lower())
        address_key = changes.get("email_key", email_key(target.email))
        if "username_key" in changes or "email_key" in changes:
            taken = self._taken(username_key, address_key, own=False, other_than=target.id)
            if taken:
                raise AccountError(taken, conflict=True)
        kept = keep_session if actor.id == target.id else None
        try:
            with self.engine.begin() as db:
                db.execute(update(users).where(users.c.id == target.id).values(**changes))
                if "password_hash" in changes:
                    db.execute(self._sessions_of(target.id, kept))
        except IntegrityError:
            taken = self._taken(username_key, address_key, own=False, other_than=target.id)
            raise AccountError(
                taken or {"username": "That username is taken. Choose another."}, conflict=True
            ) from None
        self._forget(target.id)
        # Which fields, never what they now say.
        named = ", ".join(sorted(key.removesuffix("_hash") for key in changes if "_key" not in key))
        log.info("%s changed the account %s: %s", actor.username, target.username, named)
        self._changed(ACCOUNT_UPDATED)
        return self._target(target.id)

    def _sessions_of(self, account_id: int, kept: str | None):
        """The statement that ends every session of an account, but ``kept``."""
        statement = delete(sessions).where(sessions.c.user_id == account_id)
        if kept:
            statement = statement.where(sessions.c.token_hash != _digest(kept))
        return statement

    def sign_out_everywhere(
        self, by: Account, account_id: int, *, keep_session: str | None = None
    ) -> int:
        """End every session of an account, and say how many there were. An admin signing
        themselves out everywhere keeps ``keep_session``, the one they are asking from."""
        actor = self._admin(by)
        target = self._target(account_id)
        self._guard_default(actor, target)
        kept = keep_session if actor.id == target.id else None
        with self.engine.begin() as db:
            ended = db.execute(self._sessions_of(target.id, kept)).rowcount
        self._forget(target.id)
        if ended:
            log.info("%s signed %s out everywhere", actor.username, target.username)
            self._changed(SESSION_ENDED)
        return ended

    def remove_account(self, by: Account, account_id: int) -> Account:
        """Delete someone else's account, with all its sessions. Not the default admin's,
        and not the admin's own: that is ``delete_account``, which asks for the password."""
        actor = self._admin(by)
        target = self._target(account_id)
        if self.is_default_admin(target):
            raise NotAllowed(DEFAULT_ADMIN_STAYS)
        if actor.id == target.id:
            raise NotAllowed("To delete your own account, use Delete account in your profile.")
        self._delete(target)
        log.info("%s deleted the account %s", actor.username, target.username)
        return target


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
