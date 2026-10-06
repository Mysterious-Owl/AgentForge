"""A2A 1.0 over JSON-RPC - the endpoint the Agent Card's `supportedInterfaces` points at.

The card is how another agent FINDS this one; this is how it TALKS to it. One method does
the work - `SendMessage`, answered with a `Message` (the agent replies directly; no task is
created, so there is nothing to poll). Every other core method still answers, with the
error the spec defines for an agent that does not create tasks, stream, or push:

  SendMessage                 -> {"message": {...}}  the grounded, routed answer
  ListTasks                   -> an empty page       (no tasks are ever created)
  GetTask / CancelTask        -> -32001 TaskNotFound
  SendStreamingMessage,
  SubscribeToTask             -> -32004 UnsupportedOperation  (capabilities.streaming=false)
  *PushNotificationConfig     -> -32003 PushNotificationNotSupported
  GetExtendedAgentCard        -> -32007 ExtendedAgentCardNotConfigured

The answer itself comes from the SAME path as POST /ask - routing, the agent loop, the cost
ceiling, the citation check and the audit line are not re-implemented here, only re-wrapped.
One SendMessage is one question: A2A callers carry their own history in their messages.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Callable

from fastapi import HTTPException
from pydantic import ValidationError

from app.schemas import AskRequest, AskResponse

logger = logging.getLogger(__name__)

A2A_USER = "a2a-client"          # audit-log owner for calls arriving over A2A

# JSON-RPC 2.0 + A2A 1.0 error codes (spec section 5.4).
PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL = (
    -32700, -32600, -32601, -32602, -32603)
TASK_NOT_FOUND, PUSH_NOT_SUPPORTED, UNSUPPORTED_OPERATION = -32001, -32003, -32004
CONTENT_TYPE_NOT_SUPPORTED, EXTENDED_CARD_NOT_CONFIGURED, VERSION_NOT_SUPPORTED = (
    -32005, -32007, -32009)

_FIXED_ERRORS = {
    "GetTask": (TASK_NOT_FOUND, "Task not found", "TASK_NOT_FOUND"),
    "CancelTask": (TASK_NOT_FOUND, "Task not found", "TASK_NOT_FOUND"),
    "SendStreamingMessage": (UNSUPPORTED_OPERATION, "Streaming is not supported",
                             "UNSUPPORTED_OPERATION"),
    "SubscribeToTask": (UNSUPPORTED_OPERATION, "Streaming is not supported",
                        "UNSUPPORTED_OPERATION"),
    "GetExtendedAgentCard": (EXTENDED_CARD_NOT_CONFIGURED, "No extended Agent Card",
                             "EXTENDED_AGENT_CARD_NOT_CONFIGURED"),
}
_PUSH_METHODS = {"CreateTaskPushNotificationConfig", "GetTaskPushNotificationConfig",
                 "ListTaskPushNotificationConfigs", "DeleteTaskPushNotificationConfig"}


class RpcError(Exception):
    def __init__(self, code: int, message: str, reason: str | None = None,
                 metadata: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code, self.message, self.reason, self.metadata = code, message, reason, metadata


def error_response(rpc_id: Any, err: RpcError) -> dict[str, Any]:
    body: dict[str, Any] = {"code": err.code, "message": err.message}
    if err.reason:
        body["data"] = [{"@type": "type.googleapis.com/google.rpc.ErrorInfo",
                         "reason": err.reason, "domain": "a2a-protocol.org",
                         "metadata": {k: str(v) for k, v in (err.metadata or {}).items()}}]
    return {"jsonrpc": "2.0", "id": rpc_id, "error": body}


def _question(params: Any) -> tuple[str, str | None]:
    """Pull the question text (and contextId) out of SendMessageRequest params."""
    message = params.get("message") if isinstance(params, dict) else None
    if not isinstance(message, dict):
        raise RpcError(INVALID_PARAMS, "params.message is required")
    if not message.get("messageId"):
        raise RpcError(INVALID_PARAMS, "message.messageId is required")
    parts = message.get("parts")
    if not isinstance(parts, list) or not parts:
        raise RpcError(INVALID_PARAMS, "message.parts must hold at least one part")
    if any(not isinstance(p, dict) or "text" not in p for p in parts):
        raise RpcError(CONTENT_TYPE_NOT_SUPPORTED, "only text parts are supported",
                       "CONTENT_TYPE_NOT_SUPPORTED", {"accepted": "text/plain"})
    return "\n".join(str(p["text"]) for p in parts).strip(), message.get("contextId")


def _http_to_rpc(exc: HTTPException) -> RpcError:
    """Map the /ask guard that fired onto the closest JSON-RPC error."""
    detail = exc.detail if isinstance(exc.detail, dict) else {"detail": exc.detail}
    reason = str(detail.get("error", "")).upper() or {
        503: "CONTEXT_PACK_UNAVAILABLE", 502: "MODEL_CALL_FAILED"}.get(exc.status_code, "")
    code = INVALID_PARAMS if exc.status_code == 413 else INTERNAL
    return RpcError(code, f"request refused ({exc.status_code})", reason or None, detail)


def _page_size(params: Any) -> int:
    """ListTasks pageSize: an integer from 1 up (default 50, at most 100)."""
    size = params.get("pageSize", 50) if isinstance(params, dict) else 50
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise RpcError(INVALID_PARAMS, "pageSize must be a positive integer")
    return min(size, 100)


def _send_message(params: Any, answer: Callable[[AskRequest], AskResponse]) -> dict[str, Any]:
    text, context_id = _question(params)
    context_id = context_id or uuid.uuid4().hex
    try:
        req = AskRequest(question=text, user_id=A2A_USER, session_id=context_id)
    except ValidationError as exc:
        raise RpcError(INVALID_PARAMS, exc.errors()[0]["msg"]) from None
    try:
        resp = answer(req)
    except HTTPException as exc:
        raise _http_to_rpc(exc) from None
    return {"message": {
        "messageId": uuid.uuid4().hex,
        "contextId": context_id,
        "role": "ROLE_AGENT",
        "parts": [{"text": resp.answer}],
        # The same numbers POST /ask returns - routing, the tool trace, the verified
        # citations and the cost stay visible over A2A too.
        "metadata": {"tier": resp.tier, "model": resp.model, "grounded": resp.grounded,
                     "skillMatched": resp.skill_matched, "costUsd": resp.cost_usd,
                     "savedUsd": resp.saved_usd,
                     "toolsCalled": [{"tool": s.tool, "args": s.args, "success": s.success}
                                     for s in resp.tools_called],
                     "citations": resp.citations,
                     "pendingActionId": resp.pending_action.id if resp.pending_action
                     else None},
    }}


def handle(payload: Any, version: str | None, protocol_version: str,
           answer: Callable[[AskRequest], AskResponse]) -> dict[str, Any]:
    """Dispatch one JSON-RPC request. Always returns a JSON-RPC response object."""
    if not isinstance(payload, dict) or payload.get("jsonrpc") != "2.0" \
            or not isinstance(payload.get("method"), str):
        return error_response(payload.get("id") if isinstance(payload, dict) else None,
                      RpcError(INVALID_REQUEST, "not a JSON-RPC 2.0 request"))
    rpc_id, method, params = payload.get("id"), payload["method"], payload.get("params", {})
    try:
        # The spec: the header is Major.Minor, and a MISSING one MUST be read as 0.3. This
        # agent speaks 1.0 only, so anything else - no header included - is refused.
        asked = (version or "").strip() or "0.3"
        if asked != protocol_version:
            raise RpcError(VERSION_NOT_SUPPORTED, f"A2A {asked} is not supported",
                           "VERSION_NOT_SUPPORTED", {"supported": protocol_version})
        if method == "SendMessage":
            result = _send_message(params, answer)
        elif method == "ListTasks":
            result = {"tasks": [], "totalSize": 0, "pageSize": _page_size(params),
                      "nextPageToken": ""}
        elif method in _FIXED_ERRORS:
            raise RpcError(*_FIXED_ERRORS[method])
        elif method in _PUSH_METHODS:
            raise RpcError(PUSH_NOT_SUPPORTED, "Push notifications are not supported",
                           "PUSH_NOTIFICATION_NOT_SUPPORTED")
        else:
            raise RpcError(METHOD_NOT_FOUND, f"method '{method}' not found")
    except RpcError as err:
        logger.info("a2a %s -> error %s", method, err.code)
        return error_response(rpc_id, err)
    return {"jsonrpc": "2.0", "id": rpc_id, "result": result}
