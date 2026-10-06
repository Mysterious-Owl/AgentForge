"""Public-deploy guards - what changes when localhost becomes a URL on your resume.

A shared link means strangers spend YOUR key and reach YOUR approval gate. Four guards,
all in-process (one instance, like the rest of the capstone's state):

  1. ADMIN TOKEN  -> /approve, /actions, /audit and /memory need `Authorization: Bearer
                     <token>`. Any visitor may ask the agent for an intro and watch it pause
                     at the gate; only the token holder (the student) decides it. Unset token
                     = open (localhost only).
  2. RATE LIMIT   -> POSTs per client per minute. 429 + Retry-After when exceeded.
  3. DAILY BUDGET -> total model spend per UTC day, across ALL callers. The per-request
                     ceiling (app/budget.py) bounds one call; this bounds the day. The check
                     runs before a call and the spend is recorded after it, so the day can
                     overshoot by at most one request - itself capped by the ceiling.
  4. BODY SIZE    -> a request body over `max_body_bytes` is a 413 before it is read, and a
                     body that declares no length is a 411. Render Free has 512 MB of RAM.
"""
from __future__ import annotations

import hmac
import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone

from fastapi import Header, HTTPException, Request
from fastapi.responses import JSONResponse

from app.budget import actual_cost
from app.config import get_settings

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_HITS: dict[str, deque[float]] = {}
_SPEND: dict[str, float] = {"day": "", "usd": 0.0}


def reset() -> None:
    """Forget every hit and all spend - used by the test suite between tests."""
    with _LOCK:
        _HITS.clear()
        _SPEND.update(day="", usd=0.0)


# ── 1. Admin token ───────────────────────────────────────────────────────────────

def require_admin(authorization: str | None = Header(default=None)) -> None:
    """FastAPI dependency: the caller must present the admin bearer token, if one is set."""
    token = get_settings().admin_token
    if not token:
        return                                   # localhost / the shoot / the test suite
    scheme, _, presented = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(presented, token):
        raise HTTPException(status_code=401, detail="admin token required",
                            headers={"WWW-Authenticate": "Bearer"})


# ── 2. Rate limit ────────────────────────────────────────────────────────────────

# Set by the CDN edge in front of the app (Cloudflare fronts Render), which overwrites any
# value the client sent - so these are the client IP as the edge saw it.
_EDGE_IP_HEADERS = ("true-client-ip", "cf-connecting-ip")


def client_key(request: Request) -> str:
    """Who is calling, for the rate limit.

    Behind a proxy the socket peer is the proxy, shared by every visitor, so with
    TRUST_FORWARDED_FOR on: an edge-set client-IP header first, else the FIRST
    X-Forwarded-For hop (the proxies append theirs after it - keying on the last hop would
    give everyone one shared limit). A client can forge the first hop to dodge its own limit,
    never to use up someone else's; the daily budget still caps what that can cost.
    Check the headers your own deploy receives before you rely on this.
    """
    if get_settings().trust_forwarded_for:
        for name in _EDGE_IP_HEADERS:
            if value := request.headers.get(name, "").strip():
                return value
        first = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        if first:
            return first
    return request.client.host if request.client else "unknown"


async def rate_limit(request: Request, call_next):
    """HTTP middleware: cap POSTs per client per minute. GETs (the page, the card) are free."""
    limit = get_settings().rate_limit_per_minute
    if request.method != "POST" or limit <= 0:
        return await call_next(request)
    key, now = client_key(request), time.monotonic()
    with _LOCK:
        hits = _HITS.setdefault(key, deque())
        while hits and now - hits[0] >= 60:
            hits.popleft()
        if len(hits) >= limit:
            retry = int(60 - (now - hits[0])) + 1
            logger.warning("rate limit hit: %s POSTs/min", limit)
            return JSONResponse(
                {"detail": {"error": "rate_limited", "limit_per_minute": limit,
                            "retry_after_s": retry}},
                status_code=429, headers={"Retry-After": str(retry)})
        hits.append(now)
    return await call_next(request)


# ── 3. Daily budget ──────────────────────────────────────────────────────────────

def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def spent_today() -> float:
    with _LOCK:
        return _SPEND["usd"] if _SPEND["day"] == _today() else 0.0


def record_spend(usd: float) -> None:
    with _LOCK:
        if _SPEND["day"] != _today():
            _SPEND.update(day=_today(), usd=0.0)
        _SPEND["usd"] = round(_SPEND["usd"] + usd, 6)


def require_budget() -> None:
    """FastAPI dependency: refuse a model call once today's budget is spent (429)."""
    cap, spent = get_settings().daily_budget_usd, spent_today()
    if spent >= cap:
        logger.warning("daily budget exhausted: $%.6f of $%.2f", spent, cap)
        raise HTTPException(status_code=429, detail={
            "error": "daily_budget_exhausted", "spent_usd": spent, "daily_budget_usd": cap})


def metered(generate):
    """Wrap a model call so what it actually cost is charged to today's budget - every call
    of the loop, tool turns included."""
    def _call(messages, model, tools, budget=None):
        turn = generate(messages, model, tools, budget)
        record_spend(actual_cost(turn.prompt_tokens, turn.completion_tokens, model))
        return turn
    return _call


# ── 4. Body size ─────────────────────────────────────────────────────────────────

_BODY_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


async def limit_body_size(request: Request, call_next):
    """HTTP middleware: refuse an oversized body BEFORE it is read (413), and a body that does
    not declare its length (411) - a chunked upload could not be measured up front."""
    if request.method in _BODY_METHODS:
        limit = get_settings().max_body_bytes
        length = request.headers.get("content-length")
        if length is None:
            if request.headers.get("transfer-encoding"):
                return JSONResponse({"detail": {"error": "length_required"}}, status_code=411)
        elif not length.isdigit() or int(length) > limit:
            logger.warning("refused a %s-byte body (limit %s)", length, limit)
            return JSONResponse(
                {"detail": {"error": "body_too_large", "limit_bytes": limit}}, status_code=413)
    return await call_next(request)
