"""Memory layer - session-scoped state, per-user keys, an audit log, and a hard delete.

The quietest, most dangerous failures live here (the red-team pass probes them on purpose):
cross-session bleed and cross-user reads. So the store is keyed FIRST by user, then by
session - user Y physically cannot reach user X's bucket, and session B cannot see session A.
Every answer and every proposed/executed action is written to the asking user's audit log.
`delete()` is a HARD delete: the value is genuinely removed (the audit entry is retained,
because that is the record you are required to keep).

In-process by default - a real deployment swaps this seam for Redis (see the architecture
note), but the scoping contract is identical and the tests need nothing extra.
"""
from __future__ import annotations

import logging
import threading
from collections import deque

from app.schemas import AuditEntry
from app.scrub import scrub

logger = logging.getLogger(__name__)

# user_id -> {"sessions": {session_id: {key: value}}, "audit": deque[AuditEntry]}
_STATE: dict[str, dict] = {}
# Bounded, because /ask and /a2a are public: each user keeps the newest
# AUDIT_MAX_PER_USER lines, and past MAX_USERS the least recently written user is dropped.
AUDIT_MAX_PER_USER = 500
MAX_USERS = 10_000
# FastAPI runs these sync handlers in a threadpool, so two requests can touch the store at
# once. One lock makes every read-modify-write here atomic (check-then-delete included).
_LOCK = threading.Lock()


def _bucket(user_id: str) -> dict:
    """The user's bucket, created on first WRITE (and moved to the newest). Lock held."""
    bucket = _STATE.pop(user_id, None)
    if bucket is None:
        bucket = {"sessions": {}, "audit": deque(maxlen=AUDIT_MAX_PER_USER)}
        if len(_STATE) >= MAX_USERS:
            del _STATE[next(iter(_STATE))]           # the least recently written user
    _STATE[user_id] = bucket
    return bucket


def _peek(user_id: str) -> dict:
    """The user's bucket for a READ - never creates one, so probing an id leaves no trace."""
    return _STATE.get(user_id, {"sessions": {}, "audit": []})


def remember(user_id: str, session_id: str, key: str, value: str) -> None:
    """Store a value scoped to (user, session)."""
    with _LOCK:
        _bucket(user_id)["sessions"].setdefault(session_id, {})[key] = value


def recall(user_id: str, session_id: str, key: str) -> str | None:
    """Read a value - ONLY from this user's own session. No cross-user, no cross-session."""
    with _LOCK:
        return _peek(user_id)["sessions"].get(session_id, {}).get(key)


def delete(user_id: str, session_id: str, key: str) -> bool:
    """Hard-delete one value. Returns True if something was removed. Audit is retained."""
    with _LOCK:
        session = _peek(user_id)["sessions"].get(session_id, {})
        existed = session.pop(key, None) is not None
    if existed:
        log_audit(AuditEntry(user_id=user_id, session_id=session_id,
                             kind="memory:deleted", detail=key))
    return existed


def log_audit(entry: AuditEntry) -> None:
    """Append one line to the asking user's audit log.

    The audit log is a side channel too - it is rendered back into the UI - and its details
    carry user text (a question, an action, a key). So every line is scrubbed HERE, the one
    place all of them pass through, and no route can forget to."""
    entry = entry.model_copy(update={"detail": scrub(entry.detail)})
    with _LOCK:
        _bucket(entry.user_id)["audit"].append(entry)


def audit_log(user_id: str) -> list[AuditEntry]:
    """The audit log for ONE user - never another user's."""
    with _LOCK:
        return list(_peek(user_id)["audit"])


def reset() -> None:
    """Test/ops helper - clear all memory."""
    with _LOCK:
        _STATE.clear()
