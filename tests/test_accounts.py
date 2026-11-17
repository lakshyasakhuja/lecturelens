"""Accounts: the sign-up rules, how passwords and sessions are kept, and throttling."""

from __future__ import annotations

import pytest

pytest.importorskip("sqlalchemy", reason="the web extra is not installed")
pytest.importorskip("argon2", reason="the web extra is not installed")

from argon2 import PasswordHasher
from fakes import ADA, fast_accounts
from sqlalchemy import select

from slideseek.accounts import (
    ACCOUNT_CREATED,
    ACCOUNT_DELETED,
    PASSWORD_REHASHED,
    PASSWORD_RULES,
    SESSION_ENDED,
    SESSION_STARTED,
    AccountError,
    FieldError,
    Throttle,
    check_email,
    check_password,
    check_phone,
    check_username,
    database_url,
    format_phone,
    phone_regions,
    redacted,
    sessions,
    unmet_password_rules,
    users,
)

GRACE = {**ADA, "username": "grace", "email": "grace@example.com", "name": "Grace Hopper"}


@pytest.fixture()
def store():
    return fast_accounts()


def stored(store, column):
    with store.engine.connect() as db:
        return [row[0] for row in db.execute(select(column)).all()]


# ---- signing up ----------------------------------------------------------------------------


def test_signing_up_keeps_what_was_asked_for(store):
    account = store.sign_up(**ADA)
    assert (account.username, account.name, account.email) == (
        "ada",
        "Ada Lovelace",
        "ada@example.com",
    )
    assert account.phone == "+61412345678"
    assert account.public() == {
        "username": "ada",
        "name": "Ada Lovelace",
        "email": "ada@example.com",
        "phone": "+61 412 345 678",
        "created": account.public()["created"],
    }
    assert "password" not in str(account.public())


def test_the_password_is_stored_only_as_a_salted_argon2id_hash(store):
    store.sign_up(**ADA)
    store.sign_up(**{**GRACE, "password": ADA["password"]})
    hashes = stored(store, users.c.password_hash)
    assert all(value.startswith("$argon2id$") for value in hashes)
    assert ADA["password"] not in " ".join(hashes)
    # The same password, salted twice, hashes differently.
    assert hashes[0] != hashes[1]


def test_the_default_hasher_is_argon2id_at_owasp_strength():
    hasher = PasswordHasher()
    assert hasher.memory_cost >= 19 * 1024 and hasher.time_cost >= 2
    assert hasher.hash("x").startswith("$argon2id$")


def test_a_username_can_belong_to_one_account_only_whatever_its_case(store):
    store.sign_up(**ADA)
    with pytest.raises(AccountError) as caught:
        store.sign_up(**{**GRACE, "username": "ADA"})
    assert caught.value.conflict
    assert caught.value.errors == {"username": "That username is taken. Choose another."}


def test_an_email_address_can_belong_to_one_account_only_whatever_its_case(store):
    store.sign_up(**ADA)
    with pytest.raises(AccountError) as caught:
        store.sign_up(**{**GRACE, "email": "  Ada@EXAMPLE.com "})
    assert caught.value.conflict and set(caught.value.errors) == {"email"}
    assert store.count() == 1


def test_both_conflicts_are_reported_together(store):
    store.sign_up(**ADA)
    with pytest.raises(AccountError) as caught:
        store.sign_up(**ADA)
    assert set(caught.value.errors) == {"username", "email"}


def test_a_race_for_the_same_username_is_caught_by_the_database(store, monkeypatch):
    store.sign_up(**ADA)
    # As if the other sign-up had not committed yet when this one checked.
    real, checks = store._taken, []

    def first_check_misses(*keys):
        checks.append(keys)
        return {} if len(checks) == 1 else real(*keys)

    monkeypatch.setattr(store, "_taken", first_check_misses)
    with pytest.raises(AccountError) as caught:
        store.sign_up(**{**GRACE, "username": "Ada"})
    assert caught.value.conflict and "username" in caught.value.errors
    assert store.count() == 1


def test_every_field_that_needs_fixing_is_named_at_once(store):
    with pytest.raises(AccountError) as caught:
        store.sign_up(
            name=" ", username="a", email="no-at-sign", phone="12", country="AU", password="short"
        )
    assert set(caught.value.errors) == {"name", "username", "email", "phone", "password"}
    assert not caught.value.conflict
    assert store.count() == 0


@pytest.mark.parametrize(
    "password, unmet",
    [
        ("Analytical-Engine1", []),
        ("Ab1!", ["length"]),
        ("analytical-engine1", ["upper"]),
        ("ANALYTICAL-ENGINE1", ["lower"]),
        ("Analytical-Engine", ["number"]),
        ("AnalyticalEngine1", ["special"]),
        ("Analytical Engine1", ["special"]),  # a space is not a special character
        ("ÉCOLE-école-1", []),  # letters beyond ASCII count
        ("", list(PASSWORD_RULES)),
    ],
)
def test_password_rules(password, unmet):
    assert unmet_password_rules(password) == unmet


def test_a_password_must_meet_every_rule_and_not_be_absurdly_long():
    assert check_password("Analytical-Engine1") == "Analytical-Engine1"
    for bad in ("", "password", "Password1", "Pass!1", None, 12345678):
        with pytest.raises(FieldError):
            check_password(bad)
    with pytest.raises(FieldError, match="at most 128"):
        check_password("Aa1!" * 40)


@pytest.mark.parametrize("username", ["ada", "Ada.L", "a_1", "x" * 30, "9lives", "j-doe"])
def test_usernames_that_are_allowed(username):
    assert check_username(f"  {username} ") == username


@pytest.mark.parametrize(
    "username", ["", "ab", "x" * 31, "_ada", ".ada", "ada lovelace", "ädä", "ada@home", "Admin"]
)
def test_usernames_that_are_not(username):
    with pytest.raises(FieldError):
        check_username(username)


def test_email_addresses_are_checked_and_normalised():
    assert check_email(" ada@Example.COM ") == "ada@example.com"
    for bad in ("", "ada", "ada@", "@example.com", "ada@@example.com", "ada@localhost"):
        with pytest.raises(FieldError):
            check_email(bad)


def test_phone_numbers_are_checked_for_their_country_and_kept_in_e164():
    assert check_phone("0412 345 678", "AU") == "+61412345678"
    assert check_phone("(202) 555-0143", "us") == "+12025550143"
    # A number written with its country code needs no country, and ignores a wrong one.
    assert check_phone("+44 7911 123456", "") == "+447911123456"
    assert check_phone("+44 7911 123456", "AU") == "+447911123456"
    with pytest.raises(FieldError, match="valid phone number with \\+61"):
        check_phone("12345", "AU")
    with pytest.raises(FieldError, match="country or region"):
        check_phone("0412 345 678", "")
    with pytest.raises(FieldError):
        check_phone("not a number", "AU")
    assert format_phone("+61412345678") == "+61 412 345 678"


def test_phone_regions_carry_their_calling_codes():
    regions = {item["region"]: item["code"] for item in phone_regions()}
    assert regions["AU"] == 61 and regions["US"] == 1 and regions["IN"] == 91
    assert len(regions) > 200


# ---- signing in and sessions ---------------------------------------------------------------


def test_signing_in_with_a_username_in_any_case_or_with_the_email_address(store):
    created = store.sign_up(**ADA)
    for login in ("ada", "ADA", " ada ", "ada@example.com", "ADA@example.com"):
        assert store.sign_in(login, ADA["password"]) == created


def test_a_wrong_password_or_an_unknown_account_is_refused(store):
    store.sign_up(**ADA)
    assert store.sign_in("ada", "Analytical-Engine2") is None
    assert store.sign_in("ada", ADA["password"].lower()) is None
    assert store.sign_in("nobody", ADA["password"]) is None
    assert store.sign_in("", ADA["password"]) is None
    assert store.sign_in("ada", "") is None
    assert store.sign_in("ada", None) is None


def test_a_password_hashed_with_older_settings_is_rehashed_on_sign_in():
    weak = fast_accounts()
    weak.sign_up(**ADA)
    before = stored(weak, users.c.password_hash)[0]
    weak.hasher = PasswordHasher(time_cost=2, memory_cost=16, parallelism=1)
    assert weak.sign_in("ada", ADA["password"]) is not None
    after = stored(weak, users.c.password_hash)[0]
    assert after != before and "t=2" in after
    assert weak.sign_in("ada", ADA["password"]) is not None


def test_a_session_names_its_account_until_it_ends(store):
    account = store.sign_up(**ADA)
    token = store.start_session(account)
    assert len(token) >= 43
    assert store.account_for_token(token) == account
    store.end_session(token)
    assert store.account_for_token(token) is None
    for nothing in (None, "", "made-up", "x" * 500):
        assert store.account_for_token(nothing) is None


def test_only_a_hash_of_the_session_token_is_stored(store):
    token = store.start_session(store.sign_up(**ADA))
    kept = stored(store, sessions.c.token_hash)
    assert len(kept) == 1 and token not in kept[0] and len(kept[0]) == 64


def test_sessions_expire():
    now = [1_000_000.0]
    store = fast_accounts(session_days=1, clock=lambda: now[0])
    token = store.start_session(store.sign_up(**ADA))
    now[0] += 86400 - 1
    assert store.account_for_token(token) is not None
    now[0] += 2
    assert store.account_for_token(token) is None
    # Expired sessions are cleared out when the next one starts.
    store.start_session(store.sign_in("ada", ADA["password"]))
    assert len(stored(store, sessions.c.token_hash)) == 1


def test_deleting_an_account_needs_its_password_and_ends_its_sessions(store):
    account = store.sign_up(**ADA)
    tokens = [store.start_session(account) for _ in range(2)]
    assert not store.delete_account(account, "Analytical-Engine2")
    assert store.account_for_token(tokens[0]) == account
    assert store.delete_account(account, ADA["password"])
    assert all(store.account_for_token(token) is None for token in tokens)
    assert store.sign_in("ada", ADA["password"]) is None
    assert stored(store, sessions.c.token_hash) == []
    # The username and the address are free again.
    store.sign_up(**ADA)


def test_every_change_is_announced_once_it_is_made(store):
    seen = []
    store.on_change = seen.append
    account = store.sign_up(**ADA)
    token = store.start_session(account)
    # What changes nothing is not announced.
    store.sign_in("ada", "Analytical-Engine2")
    store.account_for_token(token)
    with pytest.raises(AccountError):
        store.sign_up(**ADA)
    store.end_session(token)
    store.end_session(token)
    store.delete_account(account, "Analytical-Engine2")
    store.delete_account(account, ADA["password"])
    assert seen == [ACCOUNT_CREATED, SESSION_STARTED, SESSION_ENDED, ACCOUNT_DELETED]


def test_a_failing_listener_does_not_undo_or_fail_the_change(store):
    def broken(change):
        raise RuntimeError("the listener is broken")

    store.on_change = broken
    account = store.sign_up(**ADA)
    assert store.sign_in("ada", ADA["password"]) == account


def test_a_rehash_is_announced():
    weak = fast_accounts()
    weak.sign_up(**ADA)
    seen = []
    weak.on_change = seen.append
    weak.hasher = PasswordHasher(time_cost=2, memory_cost=16, parallelism=1)
    weak.sign_in("ada", ADA["password"])
    weak.sign_in("ada", ADA["password"])
    assert seen == [PASSWORD_REHASHED]


def test_accounts_in_a_file_outlive_the_store(tmp_path):
    url = f"sqlite:///{tmp_path / 'accounts.db'}"
    first = fast_accounts(url)
    token = first.start_session(first.sign_up(**ADA))
    first.engine.dispose()
    again = fast_accounts(url)
    assert again.persistent
    assert again.account_for_token(token).username == "ada"
    assert again.sign_in("ada@example.com", ADA["password"]) is not None
    assert not fast_accounts().persistent


def test_database_urls():
    assert database_url("postgres://u:p@host/db") == "postgresql+psycopg://u:p@host/db"
    assert database_url("postgresql://u:p@host/db?sslmode=require") == (
        "postgresql+psycopg://u:p@host/db?sslmode=require"
    )
    assert database_url("postgresql+psycopg://u:p@h/db") == "postgresql+psycopg://u:p@h/db"
    assert database_url("sqlite:///accounts.db") == "sqlite:///accounts.db"
    assert "secret" not in redacted("postgresql+psycopg://user:secret@host/db")
    assert "secret" not in repr(fast_accounts())


# ---- throttling ----------------------------------------------------------------------------


def test_a_throttle_allows_a_burst_then_says_how_long_to_wait():
    now = [0.0]
    throttle = Throttle(3, 60, clock=lambda: now[0])
    for _ in range(3):
        assert throttle.retry_after("ip") == 0
        throttle.record("ip")
    assert throttle.retry_after("ip") == 60
    assert throttle.retry_after("another") == 0
    now[0] = 30
    assert throttle.retry_after("ip") == 30
    now[0] = 60.5
    assert throttle.retry_after("ip") == 0
    throttle.record("ip")
    throttle.clear("ip")
    assert throttle.retry_after("ip") == 0
