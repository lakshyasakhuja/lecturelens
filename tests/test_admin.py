"""Admins: who they are, what they can do to other accounts, and what protects the default one."""

from __future__ import annotations

import sqlite3

import pytest

pytest.importorskip("sqlalchemy", reason="the web extra is not installed")
pytest.importorskip("argon2", reason="the web extra is not installed")

from fakes import ADA, fast_accounts

from slideseek.accounts import (
    ACCOUNT_CREATED,
    ACCOUNT_DELETED,
    ACCOUNT_UPDATED,
    SESSION_ENDED,
    AccountError,
    NoSuchAccount,
    NotAllowed,
)

GRACE = {**ADA, "username": "grace", "email": "grace@example.com", "name": "Grace Hopper"}
ALAN = {**ADA, "username": "alan", "email": "alan@example.com", "name": "Alan Turing"}
NEW_PASSWORD = "Bombe-Machine-1940"


@pytest.fixture()
def store():
    """Ada is the default admin; Grace and Alan are not admins."""
    store = fast_accounts()
    store.sign_up(**ADA)
    store.set_default_admin("ada")
    store.sign_up(**GRACE)
    store.sign_up(**ALAN)
    return store


def account(store, username):
    [record] = [r for r in store.list_accounts(username)[0] if r.account.username == username]
    return store.get(record.account.id)


def promote(store, username):
    ada = account(store, "ada")
    return store.update_account(ada, account(store, username).id, admin=True)


# ---- the column, and the default admin -----------------------------------------------------


def test_a_table_made_before_admins_gets_the_column_and_keeps_its_accounts(tmp_path):
    path = tmp_path / "accounts.db"
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE slideseek_users (id INTEGER PRIMARY KEY, username VARCHAR(30) NOT NULL,"
            " username_key VARCHAR(30) NOT NULL UNIQUE, name VARCHAR(80) NOT NULL,"
            " email VARCHAR(254) NOT NULL, email_key VARCHAR(254) NOT NULL UNIQUE,"
            " phone VARCHAR(20) NOT NULL, password_hash VARCHAR(255) NOT NULL,"
            " created_at BIGINT NOT NULL)"
        )
        db.execute(
            "INSERT INTO slideseek_users VALUES (1, 'ada', 'ada', 'Ada Lovelace',"
            " 'ada@example.com', 'ada@example.com', '+61412345678', 'not-a-hash', 0)"
        )
    db.close()
    store = fast_accounts(f"sqlite:///{path}")
    ada = store.get(1)
    assert (ada.username, ada.is_admin) == ("ada", False)
    assert store.set_default_admin("ada").is_admin
    assert fast_accounts(f"sqlite:///{path}").get(1).is_admin, "the column is there for good"


def test_the_default_admin_is_made_an_admin_once_and_said_so_once():
    store = fast_accounts()
    store.sign_up(**ADA)
    changes = []
    store.on_change = changes.append
    assert store.set_default_admin("@Ada").is_admin
    assert store.set_default_admin("ada").is_admin
    assert changes == [ACCOUNT_UPDATED]
    assert store.get(1).public()["admin"] is True


def test_signing_up_never_makes_an_admin(store):
    assert not account(store, "grace").is_admin
    assert store.count_admins() == 1


def test_nobody_can_take_the_default_admins_username_while_it_is_free():
    store = fast_accounts()
    assert store.set_default_admin("ada") is None
    with pytest.raises(AccountError) as caught:
        store.sign_up(**{**ADA, "username": "ADA"})
    assert caught.value.errors == {"username": "That username is reserved. Choose another."}
    store.set_default_admin(None)
    assert store.sign_up(**ADA).username == "ada"


def test_a_default_admin_that_cannot_be_a_username_makes_no_admin(store):
    assert store.set_default_admin("not a username!") is None


# ---- who may act ---------------------------------------------------------------------------


def test_only_an_admin_can_create_change_sign_out_or_delete(store):
    grace, alan = account(store, "grace"), account(store, "alan")
    for attempt in (
        lambda: store.create_account(grace, **{**ALAN, "username": "x1", "email": "x@x.org"}),
        lambda: store.update_account(grace, alan.id, name="Someone"),
        lambda: store.sign_out_everywhere(grace, alan.id),
        lambda: store.remove_account(grace, alan.id),
    ):
        with pytest.raises(NotAllowed):
            attempt()
    assert account(store, "alan").name == "Alan Turing"


def test_an_admin_whose_rights_were_removed_cannot_act_on_what_they_held(store):
    grace = promote(store, "grace")
    store.update_account(account(store, "ada"), grace.id, admin=False)
    with pytest.raises(NotAllowed):
        store.update_account(grace, account(store, "alan").id, name="Someone")


def test_removing_admin_rights_reaches_a_session_at_once(store):
    grace = promote(store, "grace")
    token = store.start_session(grace)
    assert store.account_for_token(token).is_admin  # now remembered in the cache
    store.update_account(account(store, "ada"), grace.id, admin=False)
    assert not store.account_for_token(token).is_admin


def test_an_unknown_account_is_said_to_be_missing(store):
    with pytest.raises(NoSuchAccount):
        store.update_account(account(store, "ada"), 999, name="Nobody")


# ---- creating ------------------------------------------------------------------------------


def test_an_admin_creates_accounts_and_admins(store):
    ada = account(store, "ada")
    changes = []
    store.on_change = changes.append
    plain = store.create_account(
        ada, **{**ALAN, "username": "edsger", "email": "edsger@example.com"}
    )
    admin = store.create_account(
        ada, admin=True, **{**ALAN, "username": "barbara", "email": "barbara@example.com"}
    )
    assert (plain.is_admin, admin.is_admin) == (False, True)
    assert changes == [ACCOUNT_CREATED, ACCOUNT_CREATED]
    assert store.count_admins() == 2
    assert store.sign_in("barbara", ALAN["password"]).is_admin


def test_a_new_admin_can_create_admins_too(store):
    grace = promote(store, "grace")
    made = store.create_account(
        grace, admin=True, **{**ALAN, "username": "barbara", "email": "barbara@example.com"}
    )
    assert made.is_admin


def test_creating_checks_every_field_as_signing_up_does(store):
    with pytest.raises(AccountError) as caught:
        store.create_account(
            account(store, "ada"), name="", username="a", email="x", phone="1", password="weak"
        )
    assert set(caught.value.errors) == {"name", "username", "email", "phone", "password"}


def test_an_admin_is_told_of_a_taken_address_in_their_own_words(store):
    with pytest.raises(AccountError) as caught:
        store.create_account(account(store, "ada"), **{**GRACE, "username": "grace2"})
    assert caught.value.conflict
    assert caught.value.errors == {"email": "Another account has this email address."}


# ---- changing ------------------------------------------------------------------------------


def test_an_admin_changes_someones_details(store):
    ada, grace = account(store, "ada"), account(store, "grace")
    changes = []
    store.on_change = changes.append
    changed = store.update_account(
        ada,
        grace.id,
        name="  Rear Admiral   Hopper ",
        username="amazing.grace",
        email="Grace@Navy.example",
        phone="+1 202 555 0143",
    )
    assert (changed.name, changed.username, changed.email, changed.phone) == (
        "Rear Admiral Hopper",
        "amazing.grace",
        "Grace@navy.example",
        "+12025550143",
    )
    assert changes == [ACCOUNT_UPDATED]
    assert store.sign_in("grace@navy.example", GRACE["password"]).id == grace.id
    assert store.sign_in("grace", GRACE["password"]) is None


def test_nothing_changed_is_nothing_written(store):
    grace = account(store, "grace")
    changes = []
    store.on_change = changes.append
    same = store.update_account(
        account(store, "ada"), grace.id, name=grace.name, email=grace.email, username="grace"
    )
    assert same == grace and changes == []


def test_a_change_is_checked_like_a_sign_up(store):
    ada, grace = account(store, "ada"), account(store, "grace")
    with pytest.raises(AccountError) as caught:
        store.update_account(ada, grace.id, name="", username="admin", email="nope")
    assert set(caught.value.errors) == {"name", "username", "email"}
    with pytest.raises(AccountError) as caught:
        store.update_account(ada, grace.id, username="Alan", email="ALAN@example.com")
    assert caught.value.conflict and set(caught.value.errors) == {"username", "email"}
    with pytest.raises(AccountError) as caught:
        store.update_account(ada, grace.id, username="ADA")
    assert caught.value.errors == {"username": "That username is reserved. Choose another."}


def test_capitals_alone_can_change_in_a_username(store):
    grace = account(store, "grace")
    assert store.update_account(account(store, "ada"), grace.id, username="Grace").username == (
        "Grace"
    )


def test_a_new_password_signs_the_account_out_everywhere(store):
    ada, grace = account(store, "ada"), account(store, "grace")
    tokens = [store.start_session(grace) for _ in range(2)]
    store.update_account(ada, grace.id, password=NEW_PASSWORD)
    assert all(store.account_for_token(token) is None for token in tokens)
    assert store.sign_in("grace", GRACE["password"]) is None
    assert store.sign_in("grace", NEW_PASSWORD).id == grace.id


def test_an_admin_changing_their_own_password_stays_signed_in_where_they_are(store):
    ada = account(store, "ada")
    here, elsewhere = store.start_session(ada), store.start_session(ada)
    store.update_account(ada, ada.id, password=NEW_PASSWORD, keep_session=here)
    assert store.account_for_token(here).id == ada.id
    assert store.account_for_token(elsewhere) is None


def test_a_new_password_must_meet_the_rules(store):
    with pytest.raises(AccountError) as caught:
        store.update_account(account(store, "ada"), account(store, "grace").id, password="weak")
    assert set(caught.value.errors) == {"password"}


def test_admin_rights_are_given_and_taken_away(store):
    ada = account(store, "ada")
    grace = promote(store, "grace")
    assert grace.is_admin and store.count_admins() == 2
    assert not store.update_account(ada, grace.id, admin=False).is_admin


def test_nobody_removes_their_own_admin_rights(store):
    grace = promote(store, "grace")
    with pytest.raises(NotAllowed, match="your own admin rights"):
        store.update_account(grace, grace.id, admin=False)


# ---- signing out and deleting --------------------------------------------------------------


def test_an_admin_signs_someone_out_everywhere(store):
    ada, grace = account(store, "ada"), account(store, "grace")
    tokens = [store.start_session(grace) for _ in range(3)]
    changes = []
    store.on_change = changes.append
    assert store.sign_out_everywhere(ada, grace.id) == 3
    assert all(store.account_for_token(token) is None for token in tokens)
    assert changes == [SESSION_ENDED]
    assert store.sign_out_everywhere(ada, grace.id) == 0
    assert changes == [SESSION_ENDED], "nothing ended, nothing to copy"


def test_an_admin_signing_themselves_out_everywhere_keeps_this_session(store):
    ada = account(store, "ada")
    here, elsewhere = store.start_session(ada), store.start_session(ada)
    assert store.sign_out_everywhere(ada, ada.id, keep_session=here) == 1
    assert store.account_for_token(here) and not store.account_for_token(elsewhere)


def test_an_admin_deletes_someones_account(store):
    ada, grace = account(store, "ada"), account(store, "grace")
    token = store.start_session(grace)
    changes = []
    store.on_change = changes.append
    assert store.remove_account(ada, grace.id).username == "grace"
    assert store.get(grace.id) is None and store.account_for_token(token) is None
    assert changes == [ACCOUNT_DELETED]


def test_an_admin_deletes_their_own_account_from_their_profile_not_the_list(store):
    grace = promote(store, "grace")
    with pytest.raises(NotAllowed, match="your profile"):
        store.remove_account(grace, grace.id)
    assert store.delete_account(grace, GRACE["password"])


# ---- the default admin is protected --------------------------------------------------------


def test_no_other_admin_can_touch_the_default_admin(store):
    grace, ada = promote(store, "grace"), account(store, "ada")
    for attempt in (
        lambda: store.update_account(grace, ada.id, name="Someone"),
        lambda: store.update_account(grace, ada.id, password=NEW_PASSWORD),
        lambda: store.update_account(grace, ada.id, admin=False),
        lambda: store.sign_out_everywhere(grace, ada.id),
        lambda: store.remove_account(grace, ada.id),
    ):
        with pytest.raises(NotAllowed):
            attempt()
    assert account(store, "ada") == ada


def test_the_default_admin_stays_an_admin_keeps_its_name_and_is_never_deleted(store):
    ada = account(store, "ada")
    with pytest.raises(NotAllowed, match="always an admin"):
        store.update_account(ada, ada.id, admin=False)
    with pytest.raises(AccountError) as caught:
        store.update_account(ada, ada.id, username="lovelace")
    assert set(caught.value.errors) == {"username"}
    with pytest.raises(NotAllowed):
        store.delete_account(ada, ADA["password"])
    assert store.get(ada.id).is_admin


def test_the_default_admin_changes_their_own_details(store):
    ada = account(store, "ada")
    changed = store.update_account(ada, ada.id, name="Augusta Ada King", username="Ada")
    assert (changed.name, changed.username) == ("Augusta Ada King", "Ada")


# ---- the list ------------------------------------------------------------------------------


def test_the_list_is_newest_first_with_open_sessions(store):
    grace = account(store, "grace")
    store.start_session(grace)
    store.start_session(grace)
    records, total = store.list_accounts()
    assert total == 3
    assert [r.account.username for r in records] == ["alan", "grace", "ada"]
    sessions = {r.account.username: r.sessions for r in records}
    assert sessions == {"alan": 0, "grace": 2, "ada": 0}
    assert next(r for r in records if r.account.username == "grace").signed_in_at is not None


def test_the_list_is_searched_by_username_name_or_email_address(store):
    def found(query):
        return [r.account.username for r in store.list_accounts(query)[0]]

    assert found("GRA") == ["grace"]
    assert found("turing") == ["alan"]
    assert found("ada@example") == ["ada"]
    assert found("example.com") == ["alan", "grace", "ada"]
    assert found("_") == [] and found("%") == [], "wildcards are taken literally"


def test_the_list_comes_in_pages(store):
    first, total = store.list_accounts(limit=2)
    rest, _ = store.list_accounts(limit=2, offset=2)
    assert total == 3 and len(first) == 2 and len(rest) == 1
