"""Smoke + contract tests for all four layers - no real API calls, no network, no key.

The model is scripted at `app.llm.chat` (tests/scripted.py), so routing, the agent loop, the
tools, the approval gate, memory and the caps are all exercised offline in a few seconds.
The eval gate is run in-process too.
"""
import pytest
from fastapi.testclient import TestClient

from app import llm, memory, scrub
from app.budget import IterationBudget, IterationCapExceeded
from app.context import ContextPackError, build_context, get_context
from app.main import app
from tests.scripted import answer, calls, grounded_model, scripted

client = TestClient(app)

SMALL = "gpt-5.4-nano-2026-03-17"
FRONTIER = "gpt-5.4-mini-2026-03-17"
INTRO = {"name": "Jane", "company": "Acme", "contact": "jane@example.com", "reason": "hiring",
         "message": "We are hiring an AI engineer."}


# ---------- health, the card, the legacy alias ----------

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["small_model"] == SMALL
    assert body["frontier_model"] == FRONTIER
    assert body["max_iterations"] == 8
    assert body["cost_ceiling_usd"] == 0.05
    # The price table is on /health too - a price you cannot see is a price nobody trusts.
    assert body["pricing_per_1m"]["frontier"]["input"] == 0.75
    assert body["pricing_per_1m"]["small"]["input"] == 0.20


def test_agent_card_well_formed():
    card = client.get("/.well-known/agent-card.json").json()
    assert card["name"]
    # A2A 1.0: WHERE to call the agent is an interface, not a top-level url.
    assert card["supportedInterfaces"][0]["url"] == "http://localhost:8000/a2a"
    assert "url" not in card
    ids = {s["id"] for s in card["skills"]}
    assert ids == {"capstone_qa", "request_intro"}   # the gated skill is advertised


def test_legacy_alias_serves_identical_card():
    canonical = client.get("/.well-known/agent-card.json")
    legacy = client.get("/.well-known/agent.json")
    assert legacy.status_code == 200
    assert legacy.content == canonical.content   # byte-for-byte, one card, two paths


def test_card_capabilities_are_protocol_flags_only():
    caps = client.get("/.well-known/agent-card.json").json()["capabilities"]
    assert set(caps) == {"streaming", "pushNotifications", "extendedAgentCard"}
    assert all(isinstance(v, bool) for v in caps.values())


def test_card_skill_has_tags_and_modes():
    skill = client.get("/.well-known/agent-card.json").json()["skills"][0]
    for key in ("id", "name", "description", "tags", "examples", "inputModes", "outputModes"):
        assert key in skill, f"skills[0] is missing {key}"
    assert skill["inputModes"] == ["text/plain"]   # a MEDIA TYPE, not the word "text"


def test_card_protocol_version_is_1_0():
    card = client.get("/.well-known/agent-card.json").json()
    iface = card["supportedInterfaces"][0]
    assert iface["protocolVersion"] == "1.0"          # the version THAT endpoint speaks
    assert iface["protocolBinding"] == "JSONRPC"
    assert "protocolVersion" not in card              # 1.0 moved it onto the interface
    assert card["version"] == "1.0.0"                 # the AGENT's own version


def test_card_is_unsigned_and_says_nothing_else():
    """This build does not sign its card - so the card carries no `signatures` at all."""
    card = client.get("/.well-known/agent-card.json").json()
    assert "signatures" not in card and "signature" not in card


def test_card_modes_are_media_types():
    """inputModes/outputModes are MEDIA TYPES ("text/plain"), never a bare word or a schema."""
    card = client.get("/.well-known/agent-card.json").json()
    for field in ("defaultInputModes", "defaultOutputModes"):
        assert card[field], f"{field} is empty"
        assert all("/" in mode for mode in card[field]), card[field]
    for skill in card["skills"]:
        for field in ("inputModes", "outputModes"):
            assert all("/" in mode for mode in skill[field]), (skill["id"], skill[field])


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
    assert r.headers["A2A-Version"] == "1.0"


def test_serve_ui():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


# ---------- Model + routing ----------

def test_ask_rejects_short_question(monkeypatch):
    monkeypatch.setattr(llm, "chat", lambda *a, **k: pytest.fail("no model call on a 422"))
    r = client.post("/ask", json={"question": "hi"})
    assert r.status_code == 422  # Pydantic min_length, before any spend


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


def test_routing_simple_goes_to_small(monkeypatch):
    model = grounded_model()
    monkeypatch.setattr(llm, "chat", model)
    body = client.post("/ask", json={"question": "What is the cost ceiling?"}).json()
    assert body["tier"] == "small"
    assert body["model"] == SMALL
    assert set(model.seen["models"]) == {SMALL}          # the WHOLE loop runs on that tier
    # A cheap question on the small tier saved money vs the frontier baseline.
    assert body["saved_usd"] > 0


def test_routing_reasoning_goes_to_frontier(monkeypatch):
    monkeypatch.setattr(llm, "chat", grounded_model())
    body = client.post(
        "/ask", json={"question": "Why does the small tier run on nano? Compare the tiers."}).json()
    assert body["tier"] == "frontier"
    assert body["model"] == FRONTIER
    assert body["saved_usd"] == 0   # already on the frontier - nothing to save


def test_routing_low_confidence_escalates(monkeypatch):
    monkeypatch.setattr(llm, "chat", grounded_model())
    # Long, no reasoning cue -> classified SIMPLE at low confidence -> route-up to frontier.
    long_q = "Please restate the following configuration values back to me " + ("x " * 120)
    body = client.post("/ask", json={"question": long_q}).json()
    assert body["routed_up"] is True
    assert body["tier"] == "frontier"


# ---------- the two caps ----------

def test_ask_oversized_input_returns_413(monkeypatch):
    monkeypatch.setattr(llm, "chat", lambda *a, **k: pytest.fail("no call once the ceiling trips"))
    r = client.post("/ask", json={"question": "Describe your architecture. " * 45_000})
    assert r.status_code == 413
    detail = r.json()["detail"]
    assert detail["error"] == "cost_ceiling_exceeded"
    assert detail["ceiling_usd"] == 0.05
    assert detail["projected_usd"] > 0.05


def test_iteration_budget_caps_at_eight():
    budget = IterationBudget()
    assert budget.limit == 8
    with pytest.raises(IterationCapExceeded):
        for _ in range(9):
            budget.step()
    assert budget.used == 8


def test_iteration_cap_returns_429(monkeypatch):
    """A model that never stops calling tools is stopped by the cap, not by luck."""
    looping = scripted(calls("get_build", week=4))       # the same tool call, forever
    monkeypatch.setattr(llm, "chat", looping)
    r = client.post("/ask", json={"question": "What did you build in Week 4?"})
    assert r.status_code == 429
    assert r.json()["detail"] == {"error": "iteration_cap_exceeded", "max_iterations": 8}
    assert len(looping.seen["models"]) == 8              # eight calls happened, the ninth did not


# ---------- the context pack ----------

def test_context_pack_loads():
    ctx = build_context()
    assert "AGENTS" in ctx
    assert "EVAL_RESULTS" in ctx


def test_context_pack_missing_raises(monkeypatch):
    monkeypatch.setattr("app.context.get_settings",
                        lambda: __import__("app.config", fromlist=["Settings"]).Settings(
                            openai_api_key="test-key", data_dir="does_not_exist"))
    with pytest.raises(ContextPackError):
        build_context()


def test_get_context_does_not_cache(tmp_path, monkeypatch):
    from app import config, context
    for name in context.PACK_FILES:
        (tmp_path / name).write_text("sample", encoding="utf-8")
    monkeypatch.setattr(context, "get_settings",
                        lambda: config.Settings(openai_api_key="test-key",
                                                data_dir=str(tmp_path)))
    assert "sample" in get_context()
    (tmp_path / context.PACK_FILES[0]).unlink()
    with pytest.raises(ContextPackError):
        get_context()


def test_missing_pack_is_a_503_before_any_model_call(tmp_path, monkeypatch):
    from app import config
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    config.get_settings.cache_clear()
    monkeypatch.setattr(llm, "chat", lambda *a, **k: pytest.fail("no call without a pack"))
    r = client.post("/ask", json={"question": "What did you build in Week 4?"})
    assert r.status_code == 503 and "context pack not loaded" in r.json()["detail"]


# ---------- the approval gate (request_intro) ----------

def _propose():
    from app import tools
    env = tools.execute_tool("request_intro", dict(INTRO), "alice", "s1")
    assert env.success, env.error
    return env.data["action_id"]


def test_intro_pauses_at_approval_gate():
    from app import tools
    action = tools.get_action(_propose())
    assert action.status == "input-required"   # paused - nothing sent
    assert action.result is None


def test_approve_executes_and_reject_does_nothing():
    approved = client.post("/approve", json={"action_id": _propose(), "approve": True}).json()
    assert approved["status"] == "executed"
    assert "Intro accepted" in approved["result"]

    rejected = client.post("/approve", json={"action_id": _propose(), "approve": False}).json()
    assert rejected["status"] == "rejected"
    assert rejected["result"] is None   # a rejected action does nothing


def test_approve_unknown_action_404():
    r = client.post("/approve", json={"action_id": "nope", "approve": True})
    assert r.status_code == 404


def test_inbox_lists_only_open_requests():
    first, second = _propose(), _propose()
    client.post("/approve", json={"action_id": first, "approve": True})
    inbox = client.get("/actions").json()
    assert [a["id"] for a in inbox] == [second]


def test_the_agent_proposes_an_intro_and_audits_it_scrubbed(monkeypatch):
    """The model chooses request_intro; the action pauses; the audit line hides the contact."""
    monkeypatch.setattr(llm, "chat", scripted(
        calls("request_intro", **INTRO),
        answer("Thanks Jane - your request is waiting for the student's approval.")))
    question = "I'm hiring - please have the student reach me at jane@example.com"
    body = client.post("/ask", json={"question": question, "user_id": "jane"}).json()
    assert body["pending_action"]["status"] == "input-required"
    assert body["skill_matched"] == "request_intro"
    audit = client.get("/audit/jane").json()
    intro = [e for e in audit if e["kind"] == "intro:proposed"]
    assert intro and "jane@example.com" not in intro[0]["detail"]
    assert "[REDACTED_EMAIL]" in intro[0]["detail"]


# ---------- Memory layer ----------

def test_memory_cross_user_isolation():
    memory.remember("alice", "s1", "secret", "alice-only")
    assert memory.recall("alice", "s1", "secret") == "alice-only"
    assert memory.recall("bob", "s1", "secret") is None   # user B cannot read user A


def test_memory_session_scoping():
    memory.remember("alice", "sessionA", "note", "from-A")
    assert memory.recall("alice", "sessionB", "note") is None   # session B cannot see session A


def test_memory_hard_delete_retains_audit():
    memory.remember("alice", "s1", "k", "v")
    assert memory.delete("alice", "s1", "k") is True
    assert memory.recall("alice", "s1", "k") is None            # genuinely gone
    kinds = [e.kind for e in memory.audit_log("alice")]
    assert "memory:deleted" in kinds                            # the audit entry is retained


def test_memory_write_then_recall_over_the_wire():
    """POST /memory writes; GET /memory reads it back for the SAME user and session."""
    r = client.post("/memory", json={"user_id": "alice", "session_id": "s1",
                                     "key": "pref", "value": "dark mode"})
    assert r.status_code == 200
    assert r.json()["stored"] is True
    got = client.get("/memory", params={"user_id": "alice", "session_id": "s1",
                                        "key": "pref"}).json()
    assert got["found"] is True
    assert got["value"] == "dark mode"


def test_memory_cross_user_read_is_denied_over_the_wire():
    """The red-team pass's cross-user probe, executable end to end."""
    client.post("/memory", json={"user_id": "alice", "session_id": "s1",
                                 "key": "secret", "value": "alice-only"})
    got = client.get("/memory", params={"user_id": "bob", "session_id": "s1",
                                        "key": "secret"}).json()
    assert got["found"] is False and got["value"] is None   # user B cannot read user A


def test_memory_cross_session_read_is_denied_over_the_wire():
    """The cross-session probe: same user, a different session, nothing bleeds."""
    client.post("/memory", json={"user_id": "alice", "session_id": "sessionA",
                                 "key": "note", "value": "from-A"})
    got = client.get("/memory", params={"user_id": "alice", "session_id": "sessionB",
                                        "key": "note"}).json()
    assert got["found"] is False and got["value"] is None


def test_memory_delete_then_query_over_the_wire():
    """The delete-then-query probe: the value is gone, the audit entries are retained."""
    body = {"user_id": "alice", "session_id": "s1", "key": "pref"}
    client.post("/memory", json={**body, "value": "dark mode"})
    deleted = client.request("DELETE", "/memory", json=body).json()
    assert deleted["deleted"] is True             # it really was there, and really went
    got = client.get("/memory", params=body).json()
    assert got["found"] is False
    kinds = [e["kind"] for e in client.get("/audit/alice").json()]
    assert "memory:wrote" in kinds and "memory:deleted" in kinds


def test_audit_is_scoped_per_user_and_records_the_trace(monkeypatch):
    monkeypatch.setattr(llm, "chat", grounded_model())
    client.post("/ask", json={"question": "What is the cost ceiling?", "user_id": "alice"})
    alice = client.get("/audit/alice").json()
    bob = client.get("/audit/bob").json()
    assert len(alice) == 1 and alice[0]["kind"] == "ask"
    assert "get_architecture(overview)" in alice[0]["detail"]   # what the agent did, too
    assert bob == []                                            # bob sees nothing of alice's


# ---------- PII scrubbing ----------

def test_scrub_redacts_email_and_phone():
    dirty = "contact jane.doe@example.com or +1 (415) 555-2671 now"
    clean = scrub.scrub(dirty)
    assert "jane.doe@example.com" not in clean
    assert "555-2671" not in clean
    assert "[REDACTED_EMAIL]" in clean and "[REDACTED_PHONE]" in clean


def test_scrub_leaves_dated_model_ids_alone():
    # A model pin has an ISO date (8 digits) - it must NOT be mistaken for a phone number.
    text = "startup model=gpt-5.4-nano-2026-03-17 loaded"
    assert scrub.scrub(text) == text


def test_pii_filter_scrubs_a_log_record():
    import logging
    handler = logging.StreamHandler()
    handler.addFilter(scrub.PIIScrubFilter())
    rec = logging.LogRecord("x", logging.INFO, __file__, 1,
                            "leak %s", ("user@corp.com",), None)
    assert handler.filters[0].filter(rec) is True
    assert "user@corp.com" not in rec.getMessage()


def test_pii_filter_scrubs_an_exception_traceback():
    """The half a msg-only filter misses.

    `logger.exception()` carries the trace in `exc_info`, which the formatter renders AFTER
    the message - so a raw email inside a provider error reached the log file untouched.
    """
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
        log.exception("model call failed")        # exactly app/main.py's 502 path

    out = stream.getvalue()
    assert "Traceback" in out                     # still debuggable - only the PII is gone
    assert "jane.doe@example.com" not in out
    assert "555-2671" not in out
    assert "[REDACTED_EMAIL]" in out and "[REDACTED_PHONE]" in out


def test_audit_detail_is_scrubbed(monkeypatch):
    """The audit log is a side channel too - and the UI renders it back on screen."""
    monkeypatch.setattr(llm, "chat", grounded_model())
    question = "Email jane.doe@example.com or call +1 (415) 555-2671 about the cost ceiling"
    assert len(question) <= 80        # the whole PII survives main.py's [:80] truncation
    client.post("/ask", json={"question": question, "user_id": "alice"})
    detail = client.get("/audit/alice").json()[0]["detail"]
    assert "jane.doe@example.com" not in detail
    assert "555-2671" not in detail
    assert "[REDACTED_EMAIL]" in detail and "[REDACTED_PHONE]" in detail


# ---------- Eval gate ----------

def test_eval_gate_passes():
    import eval_run
    results = eval_run.run()
    assert results["overall_score"] >= 0.90        # the frozen routing set stays green
    assert results["passed"] == round(results["overall_score"] * results["cases"])
    assert results["cost"]["savings_ratio_x"] > 1  # routing is cheaper than all-frontier


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
