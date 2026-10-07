"""The public-URL contract: A2A 1.0 over JSON-RPC, and the guards a shared link needs.

Offline like the rest of the suite - the model is scripted at `app.llm.chat`.
"""
import logging
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import config, guard, llm, memory
from app.context import ContextPackError
from app.main import app
from tests.scripted import answer, calls, scripted

client = TestClient(app)
ANSWER = "AgentForge is an engineering-operations copilot [AGENTS.md · overview]."
# A cheap POST that never reaches a model - for the rate-limit tests.
CHEAP = {"jsonrpc": "2.0", "id": 1, "method": "ListTasks"}


def _stub(text=ANSWER, ptok=1200, ctok=120):
    """One tool call, then a cited answer."""
    return scripted(calls("get_architecture", section="overview", ptok=600, ctok=20),
                    answer(text, ptok, ctok))


def _rpc(method, params=None, rpc_id=1, headers=None):
    body = {"jsonrpc": "2.0", "id": rpc_id, "method": method, "params": params or {}}
    r = client.post("/a2a", json=body, headers={"A2A-Version": "1.0", **(headers or {})})
    assert r.status_code == 200          # JSON-RPC: the outcome is in the body, not the status
    return r.json()


def _send(text, **message):
    return {"message": {"messageId": "m-1", "role": "ROLE_USER",
                        "parts": [{"text": text}], **message}}


def _settings(monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    config.get_settings.cache_clear()


# ---------- A2A 1.0: SendMessage is /ask, wrapped ----------

def test_a2a_send_message_answers_like_ask(monkeypatch):
    monkeypatch.setattr(llm, "chat", _stub())
    out = _rpc("SendMessage", _send("What is AgentForge?", contextId="ctx-7"))
    msg = out["result"]["message"]
    assert out["id"] == 1 and msg["role"] == "ROLE_AGENT"
    assert msg["parts"] == [{"text": ANSWER}]
    assert msg["contextId"] == "ctx-7"                  # the conversation id round-trips
    assert msg["metadata"]["tier"] == "small"           # routing stays visible over A2A
    assert msg["metadata"]["costUsd"] > 0
    # ...and so does the agent's trace: what it called, and what it cited
    assert msg["metadata"]["toolsCalled"] == [
        {"tool": "get_architecture", "args": {"section": "overview"}, "success": True}]
    assert msg["metadata"]["citations"] == ["AGENTS.md · overview"]
    # Same audit trail as /ask - the A2A caller's context id is its session.
    assert memory.audit_log("a2a-client")[0].session_id == "ctx-7"


@pytest.mark.parametrize("params, code", [
    ({}, -32602),                                                         # no message
    ({"message": {"role": "ROLE_USER", "parts": [{"text": "x" * 20}]}}, -32602),  # no id
    ({"message": {"messageId": "m", "role": "ROLE_USER", "parts": []}}, -32602),   # no parts
    (_send("hi"), -32602),                                                # under min_length
    ({"message": {"messageId": "m", "role": "ROLE_USER",
                  "parts": [{"url": "https://x/y.png"}]}}, -32005),       # not a text part
])
def test_a2a_bad_send_message_is_a_jsonrpc_error(params, code):
    assert _rpc("SendMessage", params)["error"]["code"] == code


@pytest.mark.parametrize("raw, code", [
    (b"{not json", -32700),                                          # parse error
    (b"[" * 100_000, -32700),                                        # nested past the limit
    (b'{"method": "SendMessage"}', -32600),                          # not JSON-RPC 2.0
    (b'{"jsonrpc": "2.0", "id": 1, "method": "NoSuchMethod"}', -32601),
], ids=["not_json", "deeply_nested", "no_jsonrpc_field", "unknown_method"])
def test_a2a_transport_errors(raw, code):
    r = client.post("/a2a", content=raw, headers={"Content-Type": "application/json",
                                                  "A2A-Version": "1.0"})
    assert r.status_code == 200 and r.json()["error"]["code"] == code


@pytest.mark.parametrize("method, code", [
    ("GetTask", -32001), ("CancelTask", -32001),                # no task is ever created
    ("SendStreamingMessage", -32004), ("SubscribeToTask", -32004),   # streaming=false
    ("CreateTaskPushNotificationConfig", -32003),               # pushNotifications=false
    ("GetExtendedAgentCard", -32007),                           # extendedAgentCard=false
])
def test_a2a_core_methods_answer_with_the_spec_error(method, code):
    assert _rpc(method, {"id": "t-1"})["error"]["code"] == code


def test_a2a_list_tasks_is_an_empty_page():
    page = _rpc("ListTasks", {"pageSize": 10})["result"]
    assert page == {"tasks": [], "totalSize": 0, "pageSize": 10, "nextPageToken": ""}


def test_a2a_send_message_that_proposes_an_intro_returns_its_id(monkeypatch):
    monkeypatch.setattr(llm, "chat", scripted(
        calls("request_intro", name="Jane", contact="j@x.io", reason="hiring", message="hi"),
        answer("Your request is waiting for the student's approval.")))
    msg = _rpc("SendMessage", _send("Please have the student contact me at j@x.io"))
    action_id = msg["result"]["message"]["metadata"]["pendingActionId"]
    assert action_id and client.get("/actions").json()[0]["id"] == action_id
    assert client.get("/actions").json()[0]["status"] == "input-required"


def test_a2a_maps_the_ask_guards(monkeypatch):
    monkeypatch.setattr(llm, "chat", _stub())
    big = _rpc("SendMessage", _send("Describe your architecture. " * 45000))["error"]
    assert big["code"] == -32602
    assert big["data"][0]["reason"] == "COST_CEILING_EXCEEDED"

    def broken():
        raise ContextPackError("missing context-pack file: AGENTS.md")
    monkeypatch.setattr("app.main.get_context", broken)
    down = _rpc("SendMessage", _send("What is the cost ceiling?"))["error"]
    assert down["code"] == -32603
    assert down["data"][0]["reason"] == "CONTEXT_PACK_UNAVAILABLE"


# ---------- Admin token: anyone may propose, only the token holder decides ----------

def test_admin_token_guards_the_gate_and_private_state(monkeypatch):
    _settings(monkeypatch, ADMIN_TOKEN="s3cret-token")
    # Any visitor's question can make the agent propose an intro; that mutates nothing.
    monkeypatch.setattr(llm, "chat", scripted(
        calls("request_intro", name="Jane", contact="j@x.io", reason="hiring", message="hi"),
        answer("Your request is waiting for the student's approval.")))
    proposed = client.post("/ask", json={"question": "Please have the student contact me at "
                                                     "j@x.io"})
    assert proposed.status_code == 200                       # proposing needs no token
    decision = {"action_id": proposed.json()["pending_action"]["id"], "approve": True}
    assert client.post("/approve", json=decision).status_code == 401
    wrong = {"Authorization": "Bearer nope"}
    assert client.post("/approve", json=decision, headers=wrong).status_code == 401
    for method, path in (("GET", "/audit/anon"), ("GET", "/memory?key=k"),
                         ("DELETE", "/memory"), ("GET", "/actions")):
        assert client.request(method, path, json={"key": "k"}).status_code == 401
    good = {"Authorization": "Bearer s3cret-token"}
    executed = client.post("/approve", json=decision, headers=good)
    assert executed.status_code == 200 and executed.json()["status"] == "executed"


def _admin_routes():
    from app.guard import require_admin
    return {(m, r.path) for r in app.routes if hasattr(r, "dependant")
            for d in r.dependant.dependencies if d.call is require_admin for m in r.methods}


def test_every_owner_route_needs_the_token(monkeypatch):
    assert _admin_routes() == {("GET", "/actions"), ("POST", "/approve"),
                               ("GET", "/audit/{user_id:path}"), ("POST", "/memory"),
                               ("GET", "/memory"), ("DELETE", "/memory")}
    _settings(monkeypatch, ADMIN_TOKEN="s3cret-token")
    bodies = {"/approve": {"action_id": "a", "approve": True},
              "/memory": {"key": "k", "value": "v"}}
    for method, path in sorted(_admin_routes()):
        url = path.replace("{user_id:path}", "anon")
        kwargs = {"params": {"key": "k"}} if method == "GET" else {"json": bodies.get(path)}
        assert client.request(method, url, **kwargs).status_code == 401, (method, path)


def test_a_non_ascii_token_is_a_401_not_a_500(monkeypatch):
    _settings(monkeypatch, ADMIN_TOKEN="tok")
    raw = TestClient(app, raise_server_exceptions=False)
    r = raw.get("/actions", headers={"Authorization": "Bearer café".encode("latin-1")})
    assert r.status_code == 401


def test_render_refuses_to_boot_without_an_admin_token(monkeypatch):
    monkeypatch.setenv("RENDER", "true")
    with pytest.raises(ValueError, match="ADMIN_TOKEN"):
        config.Settings()
    monkeypatch.setenv("ADMIN_TOKEN", "s3cret-token")
    assert config.Settings().render is True


def test_render_url_becomes_the_card_interface(monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL")
    _settings(monkeypatch, RENDER_EXTERNAL_URL="https://portfolio.onrender.com")
    card = client.get("/.well-known/agent-card.json").json()
    assert card["supportedInterfaces"][0]["url"] == "https://portfolio.onrender.com/a2a"
    _settings(monkeypatch, AGENT_BASE_URL="https://me.example.com")   # explicit setting wins
    card = client.get("/.well-known/agent-card.json").json()
    assert card["supportedInterfaces"][0]["url"] == "https://me.example.com/a2a"
    _settings(monkeypatch, AGENT_BASE_URL="https://me.example.com/ ")  # a trailing slash is dropped
    card = client.get("/.well-known/agent-card.json").json()
    assert card["supportedInterfaces"][0]["url"] == "https://me.example.com/a2a"
    assert card["documentationUrl"] == "https://me.example.com/readme"


# ---------- Rate limit and daily budget ----------

def test_rate_limit_caps_posts_per_client(monkeypatch):
    _settings(monkeypatch, RATE_LIMIT_PER_MINUTE="3")
    assert [client.post("/a2a", json=CHEAP).status_code for _ in range(3)] == [200] * 3
    limited = client.post("/a2a", json=CHEAP)
    assert limited.status_code == 429 and int(limited.headers["Retry-After"]) > 0
    assert limited.json()["detail"]["error"] == "rate_limited"
    assert client.get("/health").status_code == 200          # reading is never limited


def test_daily_budget_stops_model_spend(monkeypatch):
    # 1200 in + 120 out on the small tier = $0.00039 an answer; a $0.0005 day affords two.
    _settings(monkeypatch, DAILY_BUDGET_USD="0.0005")
    monkeypatch.setattr(llm, "chat", scripted(answer(ANSWER, 1200, 120)))
    q = {"question": "What is the cost ceiling?"}
    assert [client.post("/ask", json=q).status_code for _ in range(2)] == [200, 200]
    assert guard.spent_today() == pytest.approx(0.00078)
    refused = client.post("/ask", json=q)
    assert refused.status_code == 429
    assert refused.json()["detail"]["error"] == "daily_budget_exhausted"
    over_a2a = _rpc("SendMessage", _send("What is the cost ceiling?"))["error"]
    assert over_a2a["data"][0]["reason"] == "DAILY_BUDGET_EXHAUSTED"   # no side door


def test_the_rate_limit_window_slides_after_a_minute(monkeypatch):
    _settings(monkeypatch, RATE_LIMIT_PER_MINUTE="1")
    clock = [1000.0]
    monkeypatch.setattr(guard, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    assert client.post("/a2a", json=CHEAP).status_code == 200
    clock[0] += 59
    assert client.post("/a2a", json=CHEAP).status_code == 429       # still inside the minute
    clock[0] += 1
    assert client.post("/a2a", json=CHEAP).status_code == 200       # the first hit has aged out


def test_the_daily_budget_rolls_over_at_midnight_utc(monkeypatch):
    _settings(monkeypatch, DAILY_BUDGET_USD="0.0005")
    monkeypatch.setattr(llm, "chat", scripted(answer(ANSWER, 1200, 120)))
    day = ["2026-10-07"]
    monkeypatch.setattr(guard, "_today", lambda: day[0])
    q = {"question": "What is the cost ceiling?"}
    assert [client.post("/ask", json=q).status_code for _ in range(3)] == [200, 200, 429]
    day[0] = "2026-10-08"
    assert guard.spent_today() == 0.0
    assert client.post("/ask", json=q).status_code == 200


# ---------- Startup wiring and the access log (the side channel a public URL adds) ----------

def test_lifespan_installs_the_scrubbers_on_uvicorn_handlers():
    from app.scrub import AccessLogScrubFilter, PIIScrubFilter

    access = logging.getLogger("uvicorn.access")
    handler = logging.StreamHandler()
    access.addHandler(handler)
    try:
        with TestClient(app):                     # enters lifespan, unlike a bare TestClient
            pass
        assert any(isinstance(f, AccessLogScrubFilter) for f in handler.filters)
        assert any(isinstance(f, PIIScrubFilter)
                   for h in logging.getLogger().handlers for f in h.filters)
    finally:
        access.removeHandler(handler)


def test_access_log_line_is_scrubbed():
    from uvicorn.logging import AccessFormatter

    from app.scrub import AccessLogScrubFilter

    record = logging.LogRecord(
        "uvicorn.access", logging.INFO, __file__, 1, '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:5000", "GET", "/memory?key=k&user_id=jane.doe%40example.com", "1.1", 200),
        None)
    AccessLogScrubFilter().filter(record)
    line = AccessFormatter('%(client_addr)s - "%(request_line)s" %(status_code)s',
                           use_colors=False).format(record)
    assert "[REDACTED_EMAIL]" in line
    assert "jane.doe" not in line and "%40" not in line
