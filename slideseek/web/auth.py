"""Signing up and signing in over HTTP, and telling who is asking.

    GET    /api/account            who is signed in: {"account": {...}}, or {"account": null}
    GET    /api/account/countries  the countries a phone number can be from, with their codes
    POST   /api/account/signup     create an account, and sign in to it
    POST   /api/account/signin     sign in with a username or an email address and a password
    POST   /api/account/signout    end this session
    DELETE /api/account            delete the account; the body repeats its password

The session travels in an HttpOnly cookie, out of reach of the page's own scripts. Over
HTTPS it is also ``SameSite=None; Partitioned``: Hugging Face shows a Space inside a frame on
huggingface.co, and a cookie any stricter than that is never sent from inside the frame.
That makes it the server's job to refuse requests made by other sites, which
``same_origin`` does for every request that changes something.

Guessing is slowed down in memory: failed sign-ins per address and per account, sign-ups
per address. The address is the first one in ``X-Forwarded-For`` when a proxy (Hugging Face's)
sets it.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict

from slideseek.accounts import (
    PASSWORD_RULES,
    Account,
    AccountError,
    AccountStore,
    NotAllowed,
    Throttle,
    phone_regions,
)

log = logging.getLogger(__name__)

COOKIE = "slideseek_session"
UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
WRONG_LOGIN = "That username, email address or password is not right."


class _Form(BaseModel):
    model_config = ConfigDict(extra="ignore", str_max_length=1000)


class SignUp(_Form):
    name: str = ""
    username: str = ""
    email: str = ""
    country: str = ""
    phone: str = ""
    password: str = ""


class SignIn(_Form):
    login: str = ""
    password: str = ""


class Confirm(_Form):
    password: str = ""


def session_token(cookies: Mapping[str, str] | None) -> str | None:
    return (cookies or {}).get(COOKIE) or None


def is_https(request: Request) -> bool:
    forwarded = request.headers.get("x-forwarded-proto", "").split(",")[0].strip()
    return (forwarded or request.url.scheme) == "https"


def client_address(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    return forwarded or (request.client.host if request.client else "unknown")


def same_origin(request: Request) -> bool:
    """Was this request made by this site's own page, or by no page at all?

    Browsers say where a request comes from in ``Sec-Fetch-Site``, and older ones in
    ``Origin``. A request with neither is not from a browser, so it carries no visitor's
    cookie and there is nobody to protect.
    """
    site = request.headers.get("sec-fetch-site")
    if site is not None:
        return site in {"same-origin", "none"}
    origin = request.headers.get("origin")
    if origin is None:
        return True
    hosts = {request.headers.get("host", ""), request.headers.get("x-forwarded-host", "")}
    return urlsplit(origin).netloc in hosts - {""}


def session_cookie(token: str, max_age: int, *, secure: bool) -> str:
    """The ``Set-Cookie`` value for a session.

    Written out by hand because Starlette writes ``Partitioned`` only on Python 3.14 and
    later. The token is URL-safe base64, so it needs no quoting.
    """
    parts = [f"{COOKIE}={token}", f"Max-Age={max_age}", "Path=/", "HttpOnly"]
    if max_age <= 0:
        parts.append("Expires=Thu, 01 Jan 1970 00:00:00 GMT")
    parts += ["Secure", "SameSite=None", "Partitioned"] if secure else ["SameSite=Lax"]
    return "; ".join(parts)


def set_session_cookie(response: Response, request: Request, token: str, max_age: int) -> None:
    response.headers.append("set-cookie", session_cookie(token, max_age, secure=is_https(request)))


def clear_session_cookie(response: Response, request: Request) -> None:
    # A cookie is removed by setting it again, expired, with the same attributes.
    set_session_cookie(response, request, "", 0)


def problem(status: int, detail: str, errors: Mapping[str, str] | None = None, **headers):
    body: dict[str, Any] = {"detail": detail}
    if errors:
        body["errors"] = dict(errors)
    return JSONResponse(body, status_code=status, headers=headers or None)


def too_many(seconds: int) -> JSONResponse:
    minutes = max(1, round(seconds / 60))
    wait = "a minute" if minutes == 1 else f"{minutes} minutes"
    return problem(429, f"Too many attempts. Try again in {wait}.", **{"Retry-After": str(seconds)})


def add_account_routes(
    app: FastAPI,
    store: AccountStore,
    *,
    sign_in_per_address: Throttle | None = None,
    failures_per_account: Throttle | None = None,
    sign_ups_per_address: Throttle | None = None,
) -> None:
    """Add the account routes to ``app``, and ``app.state.account_for(cookies)``."""
    per_address = sign_in_per_address or Throttle(30, 15 * 60)
    per_account = failures_per_account or Throttle(10, 15 * 60)
    sign_ups = sign_ups_per_address or Throttle(30, 60 * 60)

    def account_for(cookies: Mapping[str, str] | None) -> Account | None:
        return store.account_for_token(session_token(cookies))

    app.state.accounts = store
    app.state.account_for = account_for

    def signed_in(response: Response, request: Request, account: Account) -> Response:
        token = store.start_session(account)
        set_session_cookie(response, request, token, store.session_seconds)
        return response

    @app.get("/api/account")
    def who(request: Request) -> JSONResponse:
        account = account_for(request.cookies)
        response = JSONResponse({"account": account.public() if account else None})
        response.headers["Cache-Control"] = "no-store"
        if account is None and session_token(request.cookies):
            clear_session_cookie(response, request)
        return response

    @app.get("/api/account/countries")
    def countries() -> JSONResponse:
        body = {"countries": phone_regions(), "password_rules": PASSWORD_RULES}
        return JSONResponse(body, headers={"Cache-Control": "public, max-age=86400"})

    @app.post("/api/account/signup")
    def sign_up(form: SignUp, request: Request) -> Response:
        address = client_address(request)
        wait = sign_ups.retry_after(address)
        if wait:
            return too_many(wait)
        try:
            account = store.sign_up(**form.model_dump())
        except AccountError as error:
            if error.conflict:
                # Each conflict says an account exists: count them, to slow down harvesting.
                sign_ups.record(address)
                return problem(409, next(iter(error.errors.values())), error.errors)
            return problem(400, "Check the highlighted fields.", error.errors)
        sign_ups.record(address)
        body = JSONResponse({"account": account.public()}, status_code=201)
        return signed_in(body, request, account)

    @app.post("/api/account/signin")
    def sign_in(form: SignIn, request: Request) -> Response:
        address, login = client_address(request), form.login.strip().lower()
        wait = max(per_address.retry_after(address), per_account.retry_after(login))
        if wait:
            return too_many(wait)
        account = store.sign_in(form.login, form.password)
        if account is None:
            # Only failures count, so a lecture theatre behind one address can still sign in.
            per_address.record(address)
            if login:
                per_account.record(login)
            return problem(401, WRONG_LOGIN)
        per_account.clear(login)
        return signed_in(JSONResponse({"account": account.public()}), request, account)

    @app.post("/api/account/signout")
    def sign_out(request: Request) -> Response:
        store.end_session(session_token(request.cookies))
        response = Response(status_code=204)
        clear_session_cookie(response, request)
        return response

    @app.delete("/api/account")
    def delete_account(form: Confirm, request: Request) -> Response:
        account = account_for(request.cookies)
        if account is None:
            return problem(401, "Sign in first.")
        key = f"delete:{account.id}"
        wait = per_account.retry_after(key)
        if wait:
            return too_many(wait)
        try:
            deleted = store.delete_account(account, form.password)
        except NotAllowed as error:
            return problem(403, str(error))
        if not deleted:
            per_account.record(key)
            return problem(
                403, "That password is not right.", {"password": "That password is not right."}
            )
        response = Response(status_code=204)
        clear_session_cookie(response, request)
        return response
