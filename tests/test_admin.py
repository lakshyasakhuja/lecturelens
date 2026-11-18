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


# ---- over HTTP -----------------------------------------------------------------------------

fastapi = pytest.importorskip("fastapi", reason="the web extra is not installed")

from fakes import FakeImageEmbedder, FakePictureOcr, FakeTextEmbedder, make_index  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from slideseek.accounts import check_phone  # noqa: E402
from slideseek.search import Searcher  # noqa: E402
from slideseek.web.auth import COOKIE as COOKIE_NAME  # noqa: E402
from slideseek.web.server import accounts_from_env, create_app  # noqa: E402


@pytest.fixture()
def app(store):
    index = make_index()
    searcher = Searcher(
        index,
        text_embedder=FakeTextEmbedder(),
        image_embedder=FakeImageEmbedder(),
        ocr=FakePictureOcr(),
    )
    return create_app(index, searcher, warm_up=False, accounts=store)


def visitor(app, login=None, password=None):
    """A browser of its own, signed in as ``login`` if one is given."""
    client = TestClient(app)
    if login:
        reply = client.post("/api/account/signin", json={"login": login, "password": password})
        assert reply.status_code == 200, reply.text
    return client


@pytest.fixture()
def ada(app):
    return visitor(app, "ada", ADA["password"])


def listed(client, **params):
    reply = client.get("/api/admin/accounts", params=params)
    assert reply.status_code == 200, reply.text
    return reply.json()


def account_id(client, username):
    return next(a["id"] for a in listed(client)["accounts"] if a["username"] == username)


def test_nobody_signed_in_and_anyone_but_an_admin_are_refused(app, store):
    routes = [
        ("GET", "/api/admin/accounts", None),
        ("POST", "/api/admin/accounts", {**ALAN, "username": "x1"}),
        ("PATCH", "/api/admin/accounts/1", {"name": "Someone"}),
        ("POST", "/api/admin/accounts/1/signout", None),
        ("DELETE", "/api/admin/accounts/1", None),
    ]
    stranger, grace = visitor(app), visitor(app, "grace", GRACE["password"])
    for method, path, body in routes:
        assert stranger.request(method, path, json=body).status_code == 401, (method, path)
        assert grace.request(method, path, json=body).status_code == 403, (method, path)
    assert store.get(1).name == "Ada Lovelace" and store.count() == 3


def test_an_admin_sees_every_account_but_never_a_password(ada, app):
    visitor(app, "grace", GRACE["password"])
    body = listed(ada)
    assert (body["total"], body["all"], body["admins"]) == (3, 3, 1)
    grace = next(a for a in body["accounts"] if a["username"] == "grace")
    assert grace == {
        "id": grace["id"],
        "username": "grace",
        "name": "Grace Hopper",
        "email": "grace@example.com",
        "phone": "+61 412 345 678",
        "country": "AU",
        "national": "0412 345 678",
        "created": grace["created"],
        "admin": False,
        "default": False,
        "you": False,
        "sessions": 1,
        "signed_in": grace["signed_in"],
    }
    me = next(a for a in body["accounts"] if a["username"] == "ada")
    assert me["default"] and me["you"] and me["admin"]
    text = ada.get("/api/admin/accounts").text
    assert "argon2" not in text and GRACE["password"] not in text
    assert ada.get("/api/admin/accounts").headers["cache-control"] == "no-store"
    assert [a["username"] for a in listed(ada, q="turing")["accounts"]] == ["alan"]


def test_the_phone_number_an_admin_is_shown_goes_back_into_the_form_unchanged(ada):
    for account in listed(ada)["accounts"]:
        assert check_phone(account["national"], account["country"]) == check_phone(account["phone"])


def test_an_admin_creates_an_account_and_an_admin(ada, app):
    reply = ada.post("/api/admin/accounts", json={**ALAN, "username": "edsger", "email": "e@x.org"})
    assert reply.status_code == 201 and reply.json()["account"]["admin"] is False
    reply = ada.post(
        "/api/admin/accounts",
        json={**ALAN, "username": "barbara", "email": "b@x.org", "admin": True},
    )
    assert reply.status_code == 201 and reply.json()["account"]["admin"] is True
    barbara = visitor(app, "barbara", ALAN["password"])
    assert barbara.get("/api/account").json()["account"]["admin"] is True
    assert listed(barbara)["admins"] == 2
    assert COOKIE_NAME not in reply.headers.get("set-cookie", ""), "the admin stays themselves"


def test_creating_names_each_field_to_fix(ada):
    reply = ada.post("/api/admin/accounts", json={"name": "", "username": "a b"})
    assert reply.status_code == 400
    assert set(reply.json()["errors"]) == {"name", "username", "email", "phone", "password"}
    reply = ada.post("/api/admin/accounts", json={**GRACE, "username": "grace2"})
    assert reply.status_code == 409
    assert reply.json()["errors"] == {"email": "Another account has this email address."}


def test_an_admin_changes_an_account(ada, app):
    grace = visitor(app, "grace", GRACE["password"])
    target = account_id(ada, "grace")
    reply = ada.patch(
        f"/api/admin/accounts/{target}",
        json={"name": "Grace Brewster Hopper", "country": "US", "phone": "(202) 555-0143"},
    )
    assert reply.status_code == 200, reply.text
    assert reply.json()["account"]["phone"] == "+1 202-555-0143"
    assert grace.get("/api/account").json()["account"]["name"] == "Grace Brewster Hopper"
    reply = ada.patch(f"/api/admin/accounts/{target}", json={"password": NEW_PASSWORD})
    assert reply.status_code == 200
    assert grace.get("/api/account").json() == {"account": None}, "signed out by a new password"
    visitor(app, "grace", NEW_PASSWORD)


def test_a_change_that_needs_fixing_or_is_not_allowed_is_refused(ada, app):
    target = account_id(ada, "grace")
    reply = ada.patch(f"/api/admin/accounts/{target}", json={"email": "nope", "password": "x"})
    assert reply.status_code == 400 and set(reply.json()["errors"]) == {"email", "password"}
    reply = ada.patch(f"/api/admin/accounts/{target}", json={"username": "alan"})
    assert reply.status_code == 409
    reply = ada.patch(f"/api/admin/accounts/{account_id(ada, 'ada')}", json={"admin": False})
    assert reply.status_code == 403 and "always an admin" in reply.json()["detail"]
    reply = ada.patch("/api/admin/accounts/999", json={"name": "Nobody"})
    assert reply.status_code == 404 and "no such account" in reply.json()["detail"]


def test_an_admin_changing_their_own_password_stays_signed_in(ada, app):
    elsewhere = visitor(app, "ada", ADA["password"])
    reply = ada.patch(
        f"/api/admin/accounts/{account_id(ada, 'ada')}", json={"password": NEW_PASSWORD}
    )
    assert reply.status_code == 200
    assert ada.get("/api/account").json()["account"]["username"] == "ada"
    assert elsewhere.get("/api/account").json() == {"account": None}


def test_another_admin_cannot_touch_the_default_admin(ada, app):
    ada.patch(f"/api/admin/accounts/{account_id(ada, 'grace')}", json={"admin": True})
    grace = visitor(app, "grace", GRACE["password"])
    me = account_id(ada, "ada")
    for method, path, body in (
        ("PATCH", f"/api/admin/accounts/{me}", {"password": NEW_PASSWORD}),
        ("POST", f"/api/admin/accounts/{me}/signout", None),
        ("DELETE", f"/api/admin/accounts/{me}", None),
    ):
        assert grace.request(method, path, json=body).status_code == 403, (method, path)
    assert ada.get("/api/account").json()["account"]["username"] == "ada"


def test_an_admin_whose_rights_are_taken_is_refused_on_the_next_request(ada, app):
    grace_id = account_id(ada, "grace")
    ada.patch(f"/api/admin/accounts/{grace_id}", json={"admin": True})
    grace = visitor(app, "grace", GRACE["password"])
    assert grace.get("/api/admin/accounts").status_code == 200
    ada.patch(f"/api/admin/accounts/{grace_id}", json={"admin": False})
    assert grace.get("/api/admin/accounts").status_code == 403
    assert grace.get("/api/account").json()["account"]["admin"] is False


def test_an_admin_signs_an_account_out_everywhere_and_deletes_it(ada, app, store):
    grace = visitor(app, "grace", GRACE["password"])
    target = account_id(ada, "grace")
    reply = ada.post(f"/api/admin/accounts/{target}/signout")
    assert reply.json() == {"ended": 1}
    assert grace.get("/api/account").json() == {"account": None}
    assert ada.delete(f"/api/admin/accounts/{target}").status_code == 204
    assert store.get(target) is None
    assert ada.delete(f"/api/admin/accounts/{target}").status_code == 404
    reply = ada.delete(f"/api/admin/accounts/{account_id(ada, 'ada')}")
    assert reply.status_code == 403


def test_the_default_admin_cannot_delete_their_account_from_the_profile(ada):
    reply = ada.request("DELETE", "/api/account", json={"password": ADA["password"]})
    assert reply.status_code == 403 and "cannot be deleted" in reply.json()["detail"]
    assert ada.get("/api/account").json()["account"]["username"] == "ada"


@pytest.mark.parametrize(
    "headers", [{"Sec-Fetch-Site": "cross-site"}, {"Origin": "https://evil.example"}]
)
def test_admin_requests_from_other_sites_are_refused(ada, headers):
    target = account_id(ada, "grace")
    for method, path, body in (
        ("POST", "/api/admin/accounts", {**ALAN, "username": "x1", "email": "x@x.org"}),
        ("PATCH", f"/api/admin/accounts/{target}", {"admin": True}),
        ("POST", f"/api/admin/accounts/{target}/signout", None),
        ("DELETE", f"/api/admin/accounts/{target}", None),
    ):
        assert ada.request(method, path, json=body, headers=headers).status_code == 403
    assert listed(ada)["admins"] == 1 and listed(ada)["all"] == 3


def test_the_environment_names_the_default_admin(monkeypatch, tmp_path):
    url = f"sqlite:///{tmp_path / 'accounts.db'}"
    fast_accounts(url).sign_up(**ADA)
    monkeypatch.setenv("SLIDESEEK_DATABASE_URL", url)
    monkeypatch.delenv("SLIDESEEK_ACCOUNTS_REPO", raising=False)
    monkeypatch.setenv("SLIDESEEK_ADMIN", "ada")
    store = accounts_from_env()
    assert store.default_admin == "ada" and store.get(1).is_admin
    monkeypatch.delenv("SLIDESEEK_ADMIN")
    assert accounts_from_env().default_admin is None


# ---- the page ------------------------------------------------------------------------------


def test_the_accounts_page_is_served_to_anyone_and_shows_nothing_by_itself(app):
    reply = TestClient(app).get("/admin")
    assert reply.status_code == 200 and "text/html" in reply.headers["content-type"]
    assert "Accounts." in reply.text
    assert "@example.com" not in reply.text, "the accounts come from the API, for admins only"
    assert TestClient(app).get("/admin.js").status_code == 200


def test_the_accounts_page_sends_what_the_api_takes_and_lists_the_same_rules():
    import re

    from slideseek.accounts import PASSWORD_RULES
    from slideseek.web.admin import Changes, NewAccount
    from slideseek.web.server import STATIC_DIR

    html = (STATIC_DIR / "admin.html").read_text(encoding="utf-8")
    script = (STATIC_DIR / "admin.js").read_text(encoding="utf-8")
    inputs = set(re.findall(r'id="admin-(\w+)"', html))
    assert set(NewAccount.model_fields) <= inputs
    assert set(Changes.model_fields) <= inputs
    assert re.findall(r'data-rule="(\w+)"', html) == list(PASSWORD_RULES)
    for rule in PASSWORD_RULES:
        assert f"{rule}: (text) =>" in script, f"the page does not check the {rule} rule"
    assert "document.cookie" not in script and "innerHTML" not in script
    # Someone else's details are not one's own: no browser should offer to fill them in.
    for field in ("name", "username", "email", "phone"):
        assert re.search(rf'id="admin-{field}"[^>]*autocomplete="off"', html), field


def test_the_search_page_offers_the_accounts_page_to_admins_only():
    from slideseek.web.server import STATIC_DIR

    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert 'id="admin-link" href="/admin" hidden' in html
    assert '$("admin-link").hidden = !account.admin;' in script
    assert "never\n          shown to anyone else" not in html, "admins can see the details now"
