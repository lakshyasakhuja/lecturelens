"""What an admin does over HTTP.

    GET    /api/admin/accounts                the accounts, newest first: ?q= searches the
                                              usernames, names and email addresses;
                                              ?offset= and ?limit= page through them
    POST   /api/admin/accounts                create an account, as at sign-up; "admin": true
                                              makes it an admin
    PATCH  /api/admin/accounts/{id}           change what is given: "name", "username",
                                              "email", "phone" with "country", "password",
                                              "admin"
    POST   /api/admin/accounts/{id}/signout   end every session of the account
    DELETE /api/admin/accounts/{id}           delete the account

Each answers 401 to nobody signed in and 403 to anyone who is not an admin. The store asks
the database again whether the one asking is an admin, and refuses what the default admin
is protected from (see ``slideseek.accounts``). An admin is shown every account's details,
never its password hash.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response

from slideseek.accounts import (
    Account,
    AccountError,
    AccountRecord,
    AccountStore,
    NoSuchAccount,
    NotAllowed,
    format_phone,
    phone_parts,
)
from slideseek.web.auth import SignUp, _Form, problem, session_token

log = logging.getLogger(__name__)

PAGE = 100


class NewAccount(SignUp):
    admin: bool = False


class Changes(_Form):
    name: str | None = None
    username: str | None = None
    email: str | None = None
    country: str | None = None
    phone: str | None = None
    password: str | None = None
    admin: bool | None = None


def _when(timestamp: int | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(timespec="seconds")


def admin_view(
    store: AccountStore, account: Account, actor: Account, record: AccountRecord | None = None
) -> dict[str, Any]:
    """An account as an admin is shown it."""
    region, national = phone_parts(account.phone)
    return {
        "id": account.id,
        "username": account.username,
        "name": account.name,
        "email": account.email,
        "phone": format_phone(account.phone),
        "country": region,
        "national": national,
        "created": datetime.fromtimestamp(account.created_at, timezone.utc).date().isoformat(),
        "admin": account.is_admin,
        "default": store.is_default_admin(account),
        "you": account.id == actor.id,
        "sessions": record.sessions if record else None,
        "signed_in": _when(record.signed_in_at) if record else None,
    }


def refused(error: Exception) -> JSONResponse:
    """The answer for what the store would not do."""
    if isinstance(error, AccountError):
        if error.conflict:
            return problem(409, next(iter(error.errors.values())), error.errors)
        return problem(400, "Check the highlighted fields.", error.errors)
    if isinstance(error, NoSuchAccount):
        return problem(404, str(error))
    return problem(403, str(error))


def add_admin_routes(app: FastAPI, store: AccountStore) -> None:
    """Add the admin routes to ``app``, whose ``state.account_for`` says who is asking."""

    def admin(request: Request) -> Account:
        account = app.state.account_for(request.cookies)
        if account is None:
            raise HTTPException(status_code=401, detail="Sign in first.")
        actor = store.get(account.id)  # not the cached copy: rights may have just been taken
        if actor is None or not actor.is_admin:
            raise HTTPException(status_code=403, detail="Only an admin can see this.")
        return actor

    def no_store(response: Response) -> Response:
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/admin/accounts")
    def accounts(
        q: str = Query("", max_length=254),
        offset: int = Query(0, ge=0),
        limit: int = Query(PAGE, ge=1, le=500),
        actor: Account = Depends(admin),
    ) -> Response:
        records, total = store.list_accounts(q, limit=limit, offset=offset)
        body = {
            "accounts": [admin_view(store, r.account, actor, r) for r in records],
            "total": total,
            "offset": offset,
            "all": store.count(),
            "admins": store.count_admins(),
        }
        return no_store(JSONResponse(body))

    @app.post("/api/admin/accounts")
    def create(form: NewAccount, actor: Account = Depends(admin)) -> Response:
        try:
            made = store.create_account(actor, **form.model_dump())
        except (AccountError, NotAllowed) as error:
            return refused(error)
        return no_store(JSONResponse({"account": admin_view(store, made, actor)}, 201))

    @app.patch("/api/admin/accounts/{account_id}")
    def change(
        account_id: int, form: Changes, request: Request, actor: Account = Depends(admin)
    ) -> Response:
        try:
            changed = store.update_account(
                actor,
                account_id,
                **form.model_dump(),
                keep_session=session_token(request.cookies),
            )
        except (AccountError, NotAllowed, NoSuchAccount) as error:
            return refused(error)
        return no_store(JSONResponse({"account": admin_view(store, changed, actor)}))

    @app.post("/api/admin/accounts/{account_id}/signout")
    def sign_out(account_id: int, request: Request, actor: Account = Depends(admin)) -> Response:
        try:
            ended = store.sign_out_everywhere(
                actor, account_id, keep_session=session_token(request.cookies)
            )
        except (NotAllowed, NoSuchAccount) as error:
            return refused(error)
        return no_store(JSONResponse({"ended": ended}))

    @app.delete("/api/admin/accounts/{account_id}")
    def remove(account_id: int, actor: Account = Depends(admin)) -> Response:
        try:
            store.remove_account(actor, account_id)
        except (NotAllowed, NoSuchAccount) as error:
            return refused(error)
        return Response(status_code=204)
