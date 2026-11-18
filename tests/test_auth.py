"""Signing up and in over HTTP, and searching only when signed in."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="the web extra is not installed")

from fakes import (
    ADA,
    FakeImageEmbedder,
    FakePictureOcr,
    FakeTextEmbedder,
    fast_accounts,
    make_index,
)
from fakes import make_picture as picture
from fastapi.testclient import TestClient

from slideseek.search import Searcher
from slideseek.web.auth import COOKIE
from slideseek.web.server import create_app

GRACE = {**ADA, "username": "grace", "email": "grace@example.com", "name": "Grace Hopper"}


def make_client(base_url="http://testserver"):
    index = make_index()
    searcher = Searcher(
        index,
        text_embedder=FakeTextEmbedder(),
        image_embedder=FakeImageEmbedder(),
        ocr=FakePictureOcr(),
    )
    app = create_app(index, searcher, warm_up=False, accounts=fast_accounts())
    return TestClient(app, base_url=base_url)


@pytest.fixture()
def client():
    return make_client()


def search(client):
    return client.get("/api/search", params={"q": "chain rule"})


def search_picture(client):
    return client.post(
        "/api/search/picture",
        content=picture("plot", "Overfitting"),
        headers={"Content-Type": "image/png"},
    )


def session_cookie(response) -> str:
    [line] = [c for c in response.headers.get_list("set-cookie") if c.startswith(f"{COOKIE}=")]
    return line


# ---- what a visitor without an account can and cannot do ---------------------------------


def test_without_an_account_nothing_can_be_searched(client):
    assert client.get("/api/account").json() == {"account": None}
    for reply in (search(client), search_picture(client)):
        assert reply.status_code == 401
        assert reply.json() == {"detail": "Sign in to search."}


def test_without_an_account_the_page_and_the_library_are_still_shown(client):
    assert client.get("/").status_code == 200
    assert client.get("/api/info").json()["n_lectures"] == 2
    assert client.get("/api/health").json()["status"] == "ok"
    countries = client.get("/api/account/countries").json()
    assert {"region": "AU", "code": 61} in countries["countries"]
    assert list(countries["password_rules"]) == ["length", "upper", "lower", "number", "special"]


# ---- signing up ----------------------------------------------------------------------------


def test_signing_up_signs_in_and_unlocks_search(client):
    reply = client.post("/api/account/signup", json=ADA)
    assert reply.status_code == 201
    assert reply.json() == {
        "account": {
            "username": "ada",
            "name": "Ada Lovelace",
            "email": "ada@example.com",
            "phone": "+61 412 345 678",
            "created": reply.json()["account"]["created"],
            "admin": False,
        }
    }
    cookie = session_cookie(reply)
    for attribute in ("HttpOnly", "Path=/", "SameSite=Lax"):
        assert attribute in cookie
    assert "Secure" not in cookie, "plain HTTP on a laptop: a Secure cookie would never be sent"
    assert client.get("/api/account").json()["account"]["username"] == "ada"
    assert search(client).status_code == 200
    assert search_picture(client).status_code == 200


def test_the_password_never_comes_back(client):
    replies = [
        client.post("/api/account/signup", json=ADA),
        client.get("/api/account"),
        client.post("/api/account/signin", json={"login": "ada", "password": ADA["password"]}),
    ]
    for reply in replies:
        assert ADA["password"] not in reply.text and "argon2" not in reply.text


def test_every_field_that_needs_fixing_is_named(client):
    reply = client.post(
        "/api/account/signup",
        json={"name": "", "username": "a b", "email": "ada", "country": "AU", "phone": "1", "password": "password"},
    )  # fmt: skip
    assert reply.status_code == 400
    assert set(reply.json()["errors"]) == {"name", "username", "email", "phone", "password"}
    assert COOKIE not in reply.headers.get("set-cookie", "")
    assert client.get("/api/account").json() == {"account": None}


@pytest.mark.parametrize(
    "changes, field",
    [
        ({"email": "grace@example.com"}, "username"),
        ({"username": "Grace"}, "username"),
        ({"username": "ada2"}, "email"),
        ({"username": "ada2", "email": "GRACE@example.com"}, "email"),
    ],
)
def test_a_taken_username_or_email_address_is_refused(client, changes, field):
    assert client.post("/api/account/signup", json=GRACE).status_code == 201
    reply = make_client_sharing(client).post("/api/account/signup", json={**GRACE, **changes})
    assert reply.status_code == 409
    assert field in reply.json()["errors"]


def make_client_sharing(client) -> TestClient:
    """A second visitor on the same server: no cookies from the first."""
    return TestClient(client.app)


@pytest.mark.parametrize(
    "password",
    ["Short1!", "lowercase-only-1", "UPPERCASE-ONLY-1", "No-Numbers-Here", "NoSpecials123"],
)
def test_a_password_that_breaks_a_rule_is_refused(client, password):
    reply = client.post("/api/account/signup", json={**ADA, "password": password})
    assert reply.status_code == 400 and set(reply.json()["errors"]) == {"password"}


def test_a_phone_number_needs_its_country(client):
    missing = client.post("/api/account/signup", json={**ADA, "country": ""})
    assert missing.status_code == 400 and "phone" in missing.json()["errors"]
    written_in_full = {**ADA, "country": "", "phone": "+44 7911 123456"}
    reply = client.post("/api/account/signup", json=written_in_full)
    assert reply.status_code == 201
    assert reply.json()["account"]["phone"] == "+44 7911 123456"


def test_a_malformed_request_is_refused(client):
    assert client.post("/api/account/signup", content=b"not json").status_code == 422
    assert client.post("/api/account/signup", json={**ADA, "name": "x" * 5000}).status_code == 422


# ---- signing in and out --------------------------------------------------------------------


def test_signing_in_with_a_username_or_an_email_address(client):
    client.post("/api/account/signup", json=ADA)
    for login in ("ada", "Ada", "ada@example.com"):
        visitor = make_client_sharing(client)
        reply = visitor.post(
            "/api/account/signin", json={"login": login, "password": ADA["password"]}
        )
        assert reply.status_code == 200 and reply.json()["account"]["username"] == "ada"
        assert search(visitor).status_code == 200


def test_a_wrong_password_and_an_unknown_account_get_the_same_answer(client):
    client.post("/api/account/signup", json=ADA)
    visitor = make_client_sharing(client)
    wrong = visitor.post("/api/account/signin", json={"login": "ada", "password": "Wrong-pass1"})
    unknown = visitor.post("/api/account/signin", json={"login": "bob", "password": "Wrong-pass1"})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()
    assert search(visitor).status_code == 401


def test_each_sign_in_starts_a_new_session(client):
    first = client.post("/api/account/signup", json=ADA).cookies[COOKIE]
    second = client.post(
        "/api/account/signin", json={"login": "ada", "password": ADA["password"]}
    ).cookies[COOKIE]
    assert first != second


def test_signing_out_ends_the_session_for_good(client):
    client.post("/api/account/signup", json=ADA)
    token = client.cookies[COOKIE]
    reply = client.post("/api/account/signout")
    assert reply.status_code == 204
    assert "Max-Age=0" in session_cookie(reply)
    assert search(client).status_code == 401
    # The old token is dead even if somebody kept a copy of it.
    replay = make_client_sharing(client)
    replay.cookies.set(COOKIE, token)
    assert search(replay).status_code == 401
    assert replay.get("/api/account").json() == {"account": None}


def test_a_stale_cookie_is_cleared(client):
    client.cookies.set(COOKIE, "made-up")
    reply = client.get("/api/account")
    assert reply.json() == {"account": None}
    assert "Max-Age=0" in session_cookie(reply)


def test_guessing_passwords_is_slowed_down(client):
    client.post("/api/account/signup", json=ADA)
    visitor = make_client_sharing(client)
    for _ in range(10):
        reply = visitor.post("/api/account/signin", json={"login": "ada", "password": "Guess-123!"})
        assert reply.status_code == 401
    blocked = visitor.post(
        "/api/account/signin", json={"login": "ADA", "password": ADA["password"]}
    )
    assert blocked.status_code == 429
    assert int(blocked.headers["retry-after"]) > 0
    assert "Try again" in blocked.json()["detail"]
    # Another account is not held up by guesses at this one.
    client.post("/api/account/signout")
    make_client_sharing(client).post("/api/account/signup", json=GRACE)
    other = visitor.post(
        "/api/account/signin", json={"login": "grace", "password": GRACE["password"]}
    )
    assert other.status_code == 200


# ---- deleting an account -------------------------------------------------------------------


def test_deleting_an_account_needs_its_password(client):
    client.post("/api/account/signup", json=ADA)
    wrong = client.request("DELETE", "/api/account", json={"password": "Wrong-pass1"})
    assert wrong.status_code == 403 and "password" in wrong.json()["errors"]
    assert search(client).status_code == 200

    gone = client.request("DELETE", "/api/account", json={"password": ADA["password"]})
    assert gone.status_code == 204
    assert search(client).status_code == 401
    visitor = make_client_sharing(client)
    signin = visitor.post("/api/account/signin", json={"login": "ada", "password": ADA["password"]})
    assert signin.status_code == 401
    # The username and the address can be used again.
    assert visitor.post("/api/account/signup", json=ADA).status_code == 201


def test_deleting_needs_somebody_signed_in(client):
    reply = client.request("DELETE", "/api/account", json={"password": ADA["password"]})
    assert reply.status_code == 401


# ---- cookies over HTTPS, and requests from other sites -------------------------------------


def test_over_https_the_cookie_is_secure_and_works_inside_a_frame():
    for client, headers in (
        (make_client("https://testserver"), {}),
        (make_client(), {"X-Forwarded-Proto": "https"}),  # behind Hugging Face's proxy
    ):
        reply = client.post("/api/account/signup", json=ADA, headers=headers)
        cookie = session_cookie(reply)
        for attribute in ("Secure", "HttpOnly", "SameSite=None", "Partitioned"):
            assert attribute in cookie, (attribute, cookie)


@pytest.mark.parametrize(
    "headers",
    [
        {"Sec-Fetch-Site": "cross-site"},
        {"Sec-Fetch-Site": "same-site"},
        {"Origin": "https://evil.example"},
        {"Origin": "null"},
    ],
)
def test_requests_from_other_sites_are_refused(client, headers):
    client.post("/api/account/signup", json=ADA)
    for method, path, body in (
        ("POST", "/api/account/signin", {"login": "ada", "password": ADA["password"]}),
        ("POST", "/api/account/signout", None),
        ("DELETE", "/api/account", {"password": ADA["password"]}),
        ("POST", "/api/account/signup", GRACE),
    ):
        reply = client.request(method, path, json=body, headers=headers)
        assert reply.status_code == 403, (method, path)
    picture_search = client.post(
        "/api/search/picture",
        content=picture("plot"),
        headers={"Content-Type": "image/png", **headers},
    )
    assert picture_search.status_code == 403
    # Nothing happened: still signed in, and the account is still there.
    assert client.get("/api/account").json()["account"]["username"] == "ada"


@pytest.mark.parametrize(
    "headers",
    [{"Sec-Fetch-Site": "same-origin"}, {"Origin": "http://testserver"}, {}],
)
def test_requests_from_this_site_are_let_through(client, headers):
    reply = client.post("/api/account/signup", json=ADA, headers=headers)
    assert reply.status_code == 201


# ---- the page's forms ----------------------------------------------------------------------


def test_the_page_asks_for_everything_the_server_needs_and_lists_the_same_rules():
    import re

    from slideseek.accounts import PASSWORD_RULES
    from slideseek.web.server import STATIC_DIR

    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    # Each field says what it is, so browsers and password managers fill in the right thing.
    for field, autocomplete in [
        ("signup-name", "name"),
        ("signup-username", "username"),
        ("signup-email", "email"),
        ("signup-country", "country"),
        ("signup-phone", "tel-national"),
        ("signup-password", "new-password"),
        ("signin-login", "username"),
        ("signin-password", "current-password"),
        ("delete-password", "current-password"),
    ]:
        assert re.search(rf'id="{field}"[^>]*autocomplete="{autocomplete}"', html), field
    assert re.findall(r'data-rule="(\w+)"', html) == list(PASSWORD_RULES)
    for rule in PASSWORD_RULES:
        assert f"{rule}: (text) =>" in script, f"the page does not check the {rule} rule"
    # The session cookie is HttpOnly: nothing in the script may try to read or write it.
    assert "document.cookie" not in script
