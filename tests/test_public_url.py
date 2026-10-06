"""What a stranger on the shared URL can do - and what they cannot.

Body and field limits, store caps, who the rate limit counts, the headers a refused request
still carries, strict A2A version negotiation, an honest audit trail, and the UI on a
deployment that requires the admin token. Offline: the model is stubbed.
"""
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import config, llm, memory, tools
from app.main import app
from app.schemas import AuditEntry
from tests.scripted import answer, calls, scripted

client = TestClient(app)
ROOT = Path(__file__).resolve().parent.parent
# A cheap POST that never reaches a model - for the rate-limit tests.
CHEAP = {"jsonrpc": "2.0", "id": 1, "method": "ListTasks"}
INTRO = {"name": "Jane", "contact": "jane.doe@example.com", "reason": "hiring",
         "message": "We are hiring."}


def _intro_via_agent(monkeypatch, user_id, **overrides):
    """A visitor's question makes the model call request_intro - the real path to the gate."""
    monkeypatch.setattr(llm, "chat", scripted(calls("request_intro", **{**INTRO, **overrides}),
                                              answer("Waiting for the student's approval.")))
    body = client.post("/ask", json={"question": "Please have the student contact me",
                                     "user_id": user_id}).json()
    return body["pending_action"]


def _settings(monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    config.get_settings.cache_clear()


def _rpc(method, params=None, headers=None):
    body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
    return client.post("/a2a", json=body, headers=headers or {}).json()


# ── Size limits: a stranger cannot fill the free instance's 512 MB ──────────────────

def test_a_body_over_the_limit_is_refused_before_it_is_read(monkeypatch):
    monkeypatch.setattr(llm, "chat", lambda *a, **k: pytest.fail("model called"))
    r = client.post("/ask", json={"question": "x" * 2_200_000})
    assert r.status_code == 413
    assert r.json()["detail"]["error"] == "body_too_large"


def test_the_oversized_demo_still_reaches_the_cost_ceiling():
    # The UI's "Oversized input (413)" pill sends 1.26M characters: the lesson is that the
    # COST CEILING refuses it, so the body limit must sit above it.
    r = client.post("/ask", json={"question": "Summarise this: " + "word " * 252_000})
    assert r.status_code == 413
    assert r.json()["detail"]["error"] == "cost_ceiling_exceeded"


def test_a_body_without_a_declared_length_is_refused():
    r = client.post("/approve", content=iter([b'{"action_id": "a", "approve": true}']),
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 411


@pytest.mark.parametrize("path, body", [
    ("/approve", {"action_id": "a" * 129, "approve": True}),
    ("/approve", {"action_id": "a", "approve": True, "approver": "o" * 129}),
    ("/ask", {"question": "What is the cost ceiling?", "user_id": "u" * 129}),
    ("/memory", {"user_id": "alice", "session_id": "s1", "key": "k", "value": "v" * 4001}),
    ("/memory", {"user_id": "alice", "session_id": "s1", "key": "k" * 129, "value": "v"}),
    ("/ask", {"question": "What is the cost ceiling?", "session_id": "s" * 129}),
])
def test_over_long_fields_are_422(path, body):
    assert client.post(path, json=body).status_code == 422


def test_each_users_audit_log_is_capped():
    for n in range(memory.AUDIT_MAX_PER_USER + 25):
        memory.log_audit(AuditEntry(user_id="u", session_id="s", kind="ask", detail=str(n)))
    log = memory.audit_log("u")
    assert len(log) == memory.AUDIT_MAX_PER_USER
    assert log[-1].detail == str(memory.AUDIT_MAX_PER_USER + 24)      # the newest is kept


def test_open_actions_are_capped_and_decided_ones_make_room(monkeypatch):
    monkeypatch.setattr(tools, "MAX_ACTIONS", 3)
    first = tools.request_intro(dict(INTRO), "u", "s").data["action_id"]
    tools.request_intro(dict(INTRO), "u", "s")
    tools.request_intro(dict(INTRO), "u", "s")
    full = tools.request_intro(dict(INTRO), "u", "s")      # the model gets an envelope back
    assert full.success is False and "waiting for a decision" in full.error
    tools.decide(first, approve=False)               # a decided action can be evicted
    assert tools.request_intro(dict(INTRO), "u", "s").success is True
    assert tools.get_action(first) is None


# ── Who the rate limit counts ───────────────────────────────────────────────────────

def test_rate_limit_prefers_the_edge_set_client_ip_headers(monkeypatch):
    _settings(monkeypatch, RATE_LIMIT_PER_MINUTE="1", TRUST_FORWARDED_FOR="true")

    def post(**headers):
        return client.post("/a2a", json=CHEAP, headers=headers).status_code
    # Cloudflare (in front of Render) sets True-Client-IP / CF-Connecting-IP itself.
    assert post(**{"True-Client-IP": "1.1.1.1", "X-Forwarded-For": "9.9.9.9, 10.0.0.1"}) == 200
    assert post(**{"True-Client-IP": "1.1.1.1", "X-Forwarded-For": "8.8.8.8, 10.0.0.2"}) == 429
    assert post(**{"CF-Connecting-IP": "2.2.2.2", "X-Forwarded-For": "1.1.1.1, 10.0.0.1"}) == 200


def test_rate_limit_falls_back_to_the_first_forwarded_hop(monkeypatch):
    # Render's own statement: the first X-Forwarded-For entry is the client; the hops after
    # it are Render's proxies, shared by every visitor - keying on them makes one global limit.
    _settings(monkeypatch, RATE_LIMIT_PER_MINUTE="1", TRUST_FORWARDED_FOR="true")

    def post(xff):
        return client.post("/a2a", json=CHEAP, headers={"X-Forwarded-For": xff}).status_code
    assert post("1.1.1.1, 10.0.0.1, 10.0.0.2") == 200
    assert post("1.1.1.1, 10.0.0.1, 10.0.0.2") == 429      # the same visitor
    assert post("2.2.2.2, 10.0.0.1, 10.0.0.2") == 200      # a different visitor, same proxies


def test_forwarded_headers_are_ignored_unless_trusted(monkeypatch):
    _settings(monkeypatch, RATE_LIMIT_PER_MINUTE="1")
    assert client.post("/a2a", json=CHEAP, headers={"True-Client-IP": "1.1.1.1"}).status_code \
        == 200
    assert client.post("/a2a", json=CHEAP, headers={"True-Client-IP": "2.2.2.2"}).status_code \
        == 429                                     # not trusted: one socket peer, one bucket


def test_a_rate_limited_response_still_carries_cors_and_a2a_headers(monkeypatch):
    _settings(monkeypatch, RATE_LIMIT_PER_MINUTE="1")
    origin = {"Origin": "https://another-agent.example"}
    client.post("/a2a", json=CHEAP, headers=origin)
    r = client.post("/a2a", json=CHEAP, headers=origin)
    assert r.status_code == 429
    assert r.headers["A2A-Version"] == "1.0"
    assert r.headers["Access-Control-Allow-Origin"] == "*"
    assert r.headers["Retry-After"]


# ── A2A version negotiation, per the spec ───────────────────────────────────────────

@pytest.mark.parametrize("version", [None, "", "0.3", "1.05", "1.0x", "2.0", "1"])
def test_only_a2a_1_0_is_accepted(version):
    headers = {} if version is None else {"A2A-Version": version}
    err = _rpc("ListTasks", headers=headers)["error"]
    assert err["code"] == -32009
    assert err["data"][0]["metadata"]["supported"] == "1.0"


@pytest.mark.parametrize("version", ["1.0", " 1.0 "])
def test_a2a_1_0_is_accepted(version):
    assert "result" in _rpc("ListTasks", headers={"A2A-Version": version})


@pytest.mark.parametrize("size", ["lots", 0, -5, 1.5, True])
def test_list_tasks_rejects_a_bad_page_size(size):
    err = _rpc("ListTasks", {"pageSize": size}, headers={"A2A-Version": "1.0"})["error"]
    assert err["code"] == -32602


def test_list_tasks_keeps_a_valid_page_size():
    out = _rpc("ListTasks", {"pageSize": 10}, headers={"A2A-Version": "1.0"})["result"]
    assert out["pageSize"] == 10


# ── An honest audit trail ───────────────────────────────────────────────────────────

def test_a_decided_action_cannot_be_decided_again(monkeypatch):
    a = _intro_via_agent(monkeypatch, "u1")
    assert client.post("/approve", json={"action_id": a["id"], "approve": False}).status_code \
        == 200
    again = client.post("/approve", json={"action_id": a["id"], "approve": True})
    assert again.status_code == 409
    assert again.json()["detail"] == {"error": "already_decided", "status": "rejected"}
    kinds = [e["kind"] for e in client.get("/audit/u1").json()]
    assert kinds == ["ask", "intro:proposed", "intro:rejected"]       # no phantom decision
    assert tools.get_action(a["id"]).status == "rejected"


def test_every_audit_line_is_scrubbed(monkeypatch):
    a = _intro_via_agent(monkeypatch, "u2")          # its contact is jane.doe@example.com
    client.post("/approve", json={"action_id": a["id"], "approve": False,
                                  "approver": "ops@example.com"})
    client.post("/memory", json={"user_id": "u2", "session_id": "s", "key": "bob@example.com",
                                 "value": "x"})
    client.request("DELETE", "/memory", json={"user_id": "u2", "session_id": "s",
                                              "key": "bob@example.com"})
    details = " | ".join(e["detail"] for e in client.get("/audit/u2").json())
    assert "@example.com" not in details
    assert details.count("[REDACTED_EMAIL]") == 4


# ── Input hygiene ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("question", ["        ", "   hi        ", "\n\t\n\t\n\t\n\t"])
def test_whitespace_does_not_count_toward_a_question(question, monkeypatch):
    monkeypatch.setattr(llm, "chat", lambda *a, **k: pytest.fail("model called"))
    assert client.post("/ask", json={"question": question}).status_code == 422


def test_a_padded_question_is_answered_trimmed(monkeypatch):
    model = scripted(answer("OUT_OF_SCOPE ok", ptok=10, ctok=1))
    monkeypatch.setattr(llm, "chat", model)
    assert client.post("/ask", json={"question": "  What is the cost ceiling?  "}).status_code \
        == 200
    assert model.seen["messages"][0][-1]["content"] == "What is the cost ceiling?"


# ── The UI on a deployment that requires the admin token ────────────────────────────

def test_health_says_whether_the_admin_token_is_required(monkeypatch):
    assert client.get("/health").json()["admin_token_required"] is False
    _settings(monkeypatch, ADMIN_TOKEN="t0ken")
    assert client.get("/health").json()["admin_token_required"] is True


def test_the_ui_handles_an_owner_only_answer():
    page = (ROOT / "index.html").read_text(encoding="utf-8")
    audit = page[page.index("async function viewAudit"):]
    audit = audit[:audit.index("\n  }\n")]
    assert "r.ok" in audit                          # a 401 is not rendered as "undefined entries"
    assert "sessionStorage" in page                 # the owner's token stays in this tab only
    assert "localStorage" not in page
    assert re.search(r"Authorization['\"]?\s*:", page)   # sent on the owner-only calls
