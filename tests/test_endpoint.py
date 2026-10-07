"""Smoke + contract tests for all four layers - no real API calls, no network, no key.

The model is scripted at `app.llm.chat` (tests/scripted.py), so routing, the agent loop, the
approval gate, memory and the caps are all exercised offline in a few seconds.
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import config, llm, memory, scrub
from app.context import ContextPackError, get_context
from app.main import app
from app.scrub import CARD_REDACTION, PHONE_REDACTION
from tests.scripted import answer, calls, grounded_model, scripted

client = TestClient(app)
ROOT = Path(__file__).resolve().parent.parent

SMALL = "gpt-5.4-nano-2026-03-17"
FRONTIER = "gpt-5.4-mini-2026-03-17"
INTRO = {"name": "Jane", "company": "Acme", "contact": "jane@example.com", "reason": "hiring",
         "message": "We are hiring an AI engineer."}


def _settings(monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    config.get_settings.cache_clear()


# ---------- health and the card ----------

def test_health_reports_settings(monkeypatch):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert (body["small_model"], body["frontier_model"]) == (SMALL, FRONTIER)
    assert body["max_iterations"] == 8 and body["cost_ceiling_usd"] == 0.05
    # The price table is on /health too - a price you cannot see is a price nobody trusts.
    assert body["pricing_per_1m"]["frontier"]["input"] == 0.75
    assert body["pricing_per_1m"]["small"]["input"] == 0.20
    assert body["admin_token_required"] is False
    _settings(monkeypatch, ADMIN_TOKEN="t0ken")                 # lets the UI label owner calls
    assert client.get("/health").json()["admin_token_required"] is True


def test_agent_card_contract():
    card = client.get("/.well-known/agent-card.json").json()
    assert card["name"] and card["version"] == "1.0.0"          # the AGENT's own version
    # A2A 1.0: WHERE to call the agent is an interface, with the version THAT endpoint speaks.
    assert card["supportedInterfaces"] == [{"url": "http://localhost:8000/a2a",
                                            "protocolBinding": "JSONRPC",
                                            "protocolVersion": "1.0"}]
    assert "url" not in card and "protocolVersion" not in card
    # capabilities = protocol flags, and none is claimed that /a2a does not serve
    assert card["capabilities"] == {"streaming": False, "pushNotifications": False,
                                    "extendedAgentCard": False}
    assert {s["id"] for s in card["skills"]} == {"capstone_qa", "request_intro"}
    # input/output modes are MEDIA TYPES ("text/plain"), never a bare word or a schema
    assert card["defaultInputModes"] == card["defaultOutputModes"] == ["text/plain"]
    for skill in card["skills"]:
        assert skill["tags"] and skill["examples"]
        assert skill["inputModes"] == skill["outputModes"] == ["text/plain"], skill["id"]


def test_card_signatures_are_jws_objects():
    """A2A 1.0 card signatures: a LIST of detached-JWS objects {protected, signature}."""
    from pydantic import ValidationError

    from app.schemas import AgentCard

    iface = [{"url": "u"}]
    signed = AgentCard(name="n", description="d", supported_interfaces=iface,
                       signatures=[{"protected": "eyJhbGciOiJFUzI1NiJ9", "signature": "c2ln"}])
    assert signed.model_dump(by_alias=True)["signatures"][0]["protected"].startswith("eyJ")
    for bad in (["a-bare-string"], [{"issuer": "acme", "value": "eyJ..."}]):
        with pytest.raises(ValidationError):
            AgentCard(name="n", description="d", supported_interfaces=iface, signatures=bad)


def test_a2a_version_header_emitted():
    assert client.get("/health").headers["A2A-Version"] == "1.0"
    r = client.post("/ask", json={"question": "hi"})   # even a 422 carries it
    assert r.status_code == 422 and r.headers["A2A-Version"] == "1.0"


# ---------- Model + routing ----------

def test_ask_happy_path_grounded(monkeypatch):
    monkeypatch.setattr(llm, "chat", grounded_model())
    r = client.post("/ask", json={"question": "What is AgentForge?"})
    assert r.status_code == 200
    body = r.json()
    assert body["grounded"] is True
    assert body["skill_matched"] == "capstone_qa"
    assert body["citations"] == ["AGENTS.md · overview"]
    assert [s["tool"] for s in body["tools_called"]] == ["get_architecture"]
    assert body["model_calls"] == 2                    # the tool turn, then the answer
    assert "cost_usd" in body and "baseline_cost_usd" in body


def test_ask_out_of_scope_ungrounded(monkeypatch):
    monkeypatch.setattr(llm, "chat", scripted(answer(
        "OUT_OF_SCOPE I can only answer questions about this capstone.", ctok=12)))
    r = client.post("/ask", json={"question": "What is the capital of France?"})
    assert r.status_code == 200
    body = r.json()
    assert body["grounded"] is False
    assert body["skill_matched"] is None
    assert body["tools_called"] == [] and body["model_calls"] == 1


@pytest.mark.parametrize("question,tier,routed_up", [
    ("What is the cost ceiling?", "small", False),
    ("Why does the small tier run on nano? Compare the tiers.", "frontier", False),
    # long, no reasoning cue -> SIMPLE at low confidence -> routed up to the frontier
    ("Please restate the following configuration values back to me " + "x " * 120,
     "frontier", True),
], ids=["simple", "reasoning", "low_confidence"])
def test_routing_tiers(monkeypatch, question, tier, routed_up):
    model = grounded_model()
    monkeypatch.setattr(llm, "chat", model)
    body = client.post("/ask", json={"question": question}).json()
    assert body["tier"] == tier and body["routed_up"] is routed_up
    expected = SMALL if tier == "small" else FRONTIER
    assert body["model"] == expected
    assert set(model.seen["models"]) == {expected}       # the WHOLE loop runs on that tier
    # A small-tier answer saved money vs the frontier baseline; a frontier one saved nothing.
    assert (body["saved_usd"] > 0) is (tier == "small")


# ---------- the iteration cap ----------

def test_iteration_cap_returns_429(monkeypatch):
    """A model that never stops calling tools is stopped by the cap, not by luck."""
    looping = scripted(calls("get_build", week=4))       # the same tool call, forever
    monkeypatch.setattr(llm, "chat", looping)
    r = client.post("/ask", json={"question": "What did you build in Week 4?"})
    assert r.status_code == 429
    assert r.json()["detail"] == {"error": "iteration_cap_exceeded", "max_iterations": 8}
    assert len(looping.seen["models"]) == 8              # eight calls happened, the ninth did not


# ---------- the context pack ----------

def test_get_context_does_not_cache(tmp_path, monkeypatch):
    from app import context
    for name in context.PACK_FILES:
        (tmp_path / name).write_text('{"sample": 1}' if name.endswith(".json") else "sample",
                                     encoding="utf-8")
    monkeypatch.setattr(context, "get_settings",
                        lambda: config.Settings(openai_api_key="test-key",
                                                data_dir=str(tmp_path)))
    assert "sample" in get_context()
    (tmp_path / context.PACK_FILES[0]).unlink()
    with pytest.raises(ContextPackError):
        get_context()


@pytest.mark.parametrize("body", ["{not json", "[1, 2]", bytes([0xFF, 0xFE]) + b" bad utf-8"])
def test_a_corrupt_eval_results_file_is_a_503_not_a_500(tmp_path, monkeypatch, body):
    import shutil
    for name in ("AGENTS.md", "architecture.md"):
        shutil.copy(ROOT / "data" / name, tmp_path / name)
    target = tmp_path / "eval_results.json"
    target.write_bytes(body if isinstance(body, bytes) else body.encode())
    _settings(monkeypatch, DATA_DIR=str(tmp_path))
    monkeypatch.setattr(llm, "chat", lambda *a, **k: pytest.fail("no call with a broken pack"))
    assert client.get("/portfolio").status_code == 503
    assert client.get("/portfolio/5").status_code == 503
    r = client.post("/ask", json={"question": "What did you build in Week 4?"})
    assert r.status_code == 503 and "context pack not loaded" in r.json()["detail"]


def test_missing_pack_is_a_503_before_any_model_call(tmp_path, monkeypatch):
    _settings(monkeypatch, DATA_DIR=str(tmp_path))
    monkeypatch.setattr(llm, "chat", lambda *a, **k: pytest.fail("no call without a pack"))
    r = client.post("/ask", json={"question": "What did you build in Week 4?"})
    assert r.status_code == 503 and "context pack not loaded" in r.json()["detail"]


# ---------- the approval gate (request_intro) ----------

def _propose():
    from app import tools
    env = tools.execute_tool("request_intro", dict(INTRO), "alice", "s1")
    assert env.success, env.error
    return env.data["action_id"]


def test_approval_gate_lifecycle():
    first, second, third = _propose(), _propose(), _propose()
    assert {a["id"] for a in client.get("/actions").json()} == {first, second, third}
    approved = client.post("/approve", json={"action_id": first, "approve": True}).json()
    assert approved["status"] == "executed" and "Intro accepted" in approved["result"]
    rejected = client.post("/approve", json={"action_id": second, "approve": False}).json()
    assert rejected["status"] == "rejected" and rejected["result"] is None   # does nothing
    assert [a["id"] for a in client.get("/actions").json()] == [third]       # only open ones
    assert client.post("/approve", json={"action_id": "nope", "approve": True}).status_code \
        == 404


# ---------- Memory layer ----------

def test_memory_boundaries_over_the_wire():
    """The red-team pass's probes, end to end: write/recall, cross-user, cross-session,
    delete-then-query - the value goes, the audit entries stay."""
    mine = {"user_id": "alice", "session_id": "sessionA", "key": "pref"}
    r = client.post("/memory", json={**mine, "value": "dark mode"})
    assert r.status_code == 200 and r.json()["stored"] is True
    got = client.get("/memory", params=mine).json()
    assert got["found"] is True and got["value"] == "dark mode"
    for other in ({**mine, "user_id": "bob"}, {**mine, "session_id": "sessionB"}):
        got = client.get("/memory", params=other).json()
        assert got["found"] is False and got["value"] is None, other   # nothing bleeds
    assert client.request("DELETE", "/memory", json=mine).json()["deleted"] is True
    assert client.get("/memory", params=mine).json()["found"] is False
    kinds = [e["kind"] for e in client.get("/audit/alice").json()]
    assert "memory:wrote" in kinds and "memory:deleted" in kinds


def test_the_oldest_user_is_dropped_past_the_user_cap(monkeypatch):
    monkeypatch.setattr(memory, "MAX_USERS", 3)
    for user in ("u1", "u2", "u3", "u4"):
        memory.remember(user, "s", "k", user)
    assert memory.recall("u1", "s", "k") is None                     # least recently written
    assert [memory.recall(u, "s", "k") for u in ("u2", "u3", "u4")] == ["u2", "u3", "u4"]
    assert len(memory._STATE) == 3


def test_audit_is_scoped_per_user_and_records_the_trace(monkeypatch):
    monkeypatch.setattr(llm, "chat", grounded_model())
    client.post("/ask", json={"question": "What is the cost ceiling?", "user_id": "alice"})
    alice = client.get("/audit/alice").json()
    bob = client.get("/audit/bob").json()
    assert len(alice) == 1 and alice[0]["kind"] == "ask"
    assert "get_architecture(overview)" in alice[0]["detail"]   # what the agent did, too
    assert bob == []                                            # bob sees nothing of alice's


# ---------- PII scrubbing ----------

@pytest.mark.parametrize("dirty,clean", [
    ("mail jane.doe@example.com now", "mail [REDACTED_EMAIL] now"),
    ("call +1 (555) 123-4567 now", f"call {PHONE_REDACTION} now"),
    ("card 4111 1111 1111 1111 on file", f"card {CARD_REDACTION} on file"),
    # ISO dates, times and dated model pins are not phone numbers
    ("at 2026-09-11 ok", "at 2026-09-11 ok"),
    ("at 2026-09-11 14:23:05 ok", "at 2026-09-11 14:23:05 ok"),
    ("at 2026-09-11T14:23:05Z ok", "at 2026-09-11T14:23:05Z ok"),
    ("model=gpt-5.4-nano-2026-03-17 loaded", "model=gpt-5.4-nano-2026-03-17 loaded"),
])
def test_scrub_patterns(dirty, clean):
    assert scrub.scrub(dirty) == clean


def test_digits_that_fail_luhn_are_still_never_logged_raw():
    assert "1234" not in scrub.scrub("ref 1234 5678 9012 3456")


def test_scrubbing_a_long_run_is_fast():
    import time
    started = time.monotonic()
    scrub.scrub("a" * 50_000)
    assert time.monotonic() - started < 0.2                      # was quadratic


def test_pii_filter_scrubs_a_message_and_its_traceback():
    """`logger.exception()` carries the trace in `exc_info`, which the formatter renders AFTER
    the message - a msg-only filter let a raw email inside a provider error reach the log."""
    import io
    import logging

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    handler.addFilter(scrub.PIIScrubFilter())
    log = logging.getLogger("test-pii-traceback")
    log.handlers = [handler]
    log.propagate = False
    log.setLevel(logging.DEBUG)

    try:
        raise ValueError("provider rejected jane.doe@example.com / +1 (415) 555-2671")
    except ValueError:
        log.exception("model call failed for %s", "user@corp.com")   # main.py's 502 path

    out = stream.getvalue()
    assert "Traceback" in out                     # still debuggable - only the PII is gone
    for raw in ("jane.doe@example.com", "555-2671", "user@corp.com"):
        assert raw not in out, raw
    assert "[REDACTED_EMAIL]" in out and "[REDACTED_PHONE]" in out


def test_a_provider_error_does_not_leak_pii_to_the_client(monkeypatch):
    """The 502 detail is an egress path: a provider that echoes the prompt must not
    hand the caller back an email address it happened to quote."""
    from app import main as main_mod

    def _boom(*args, **kwargs):
        raise RuntimeError("upstream rejected prompt for jane.doe@example.com")

    monkeypatch.setattr(main_mod, "route", _boom)
    r = client.post("/ask", json={"question": "how does the memory layer work?",
                                  "user_id": "u-1", "session_id": "s-1"})

    assert r.status_code == 502
    detail = r.json()["detail"]
    assert "jane.doe@example.com" not in detail, detail
    assert "REDACTED" in detail, detail
