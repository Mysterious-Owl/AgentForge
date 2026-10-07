"""Regression tests for the code audit: each one pins a defect that was found and fixed.

Offline: the model is scripted (tests/scripted.py), nothing reaches the network.
"""
import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from app import config, guard, llm, memory, signing, tools
from app.agent import EMPTY_ANSWER
from app.main import app
from app.schemas import AskRequest, ModelTurn, ToolCall
from app.scrub import scrub
from tests.scripted import answer, calls, scripted

client = TestClient(app)
ROOT = Path(__file__).resolve().parent.parent
INTRO = {"name": "X", "contact": "x@y.example", "reason": "other", "message": "hi"}


def _settings(monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    config.get_settings.cache_clear()


def test_one_question_creates_at_most_one_intro(monkeypatch):
    many = [("request_intro", dict(INTRO)) for _ in range(15)]
    monkeypatch.setattr(llm, "chat", scripted(calls(*many), answer("Waiting for the student.")))
    body = client.post("/ask", json={"question": "Please have the student contact me",
                                     "user_id": "spam"}).json()
    assert len(tools.pending_actions()) == 1
    assert [s["success"] for s in body["tools_called"]] == [True] + [False] * 14
    assert sum(e.kind == "intro:proposed" for e in memory.audit_log("spam")) == 1


def test_an_intro_is_audited_even_if_a_cap_stops_the_request(monkeypatch):
    monkeypatch.setattr(llm, "chat", scripted(calls("request_intro", ptok=70_000, **INTRO),
                                              answer("never reached")))
    r = client.post("/ask", json={"question": "Why not ask the student to contact me?",
                                  "user_id": "late"})
    assert r.status_code == 413                                  # the ceiling stopped call 2
    assert [e.kind for e in memory.audit_log("late")] == ["intro:proposed"]


def test_the_rate_limit_table_is_bounded(monkeypatch):
    _settings(monkeypatch, TRUST_FORWARDED_FOR="true")
    monkeypatch.setattr(guard, "MAX_TRACKED_CLIENTS", 5)
    for i in range(20):                                          # 20 forged addresses
        client.post("/ask", json={"question": "x"}, headers={"X-Forwarded-For": f"10.0.0.{i}"})
    assert len(guard._HITS) == 5


def test_scrubbing_a_long_run_is_fast_and_still_finds_emails():
    started = time.monotonic()
    scrub("a" * 50_000)
    assert time.monotonic() - started < 0.2                      # was quadratic
    assert scrub("mail jane.doe@example.com now") == "mail [REDACTED_EMAIL] now"


def test_a_non_ascii_token_is_a_401_not_a_500(monkeypatch):
    _settings(monkeypatch, ADMIN_TOKEN="tok")
    raw = TestClient(app, raise_server_exceptions=False)
    r = raw.get("/actions", headers={"Authorization": "Bearer café".encode("latin-1")})
    assert r.status_code == 401


def test_an_empty_answer_gets_a_message(monkeypatch):
    monkeypatch.setattr(llm, "chat", scripted(answer("")))
    body = client.post("/ask", json={"question": "What did you build in Week 4?"}).json()
    assert body["answer"] == EMPTY_ANSWER and body["grounded"] is False


def test_deeply_nested_arguments_are_a_tool_error(monkeypatch):
    deep = ModelTurn(tool_calls=[ToolCall(id="c0", name="get_build",
                                          arguments="[" * 100_000 + "]" * 100_000)])
    monkeypatch.setattr(llm, "chat", scripted(deep, answer("OUT_OF_SCOPE sorry.")))
    r = client.post("/ask", json={"question": "What did you build in Week 4?"})
    assert r.status_code == 200
    assert r.json()["tools_called"][0]["error"] == "arguments were not valid JSON"


def test_deeply_nested_a2a_json_is_a_parse_error():
    r = client.post("/a2a", content=b"[" * 100_000, headers={"Content-Type": "application/json"})
    assert r.status_code == 200 and r.json()["error"]["code"] == -32700


def test_a_huge_week_is_a_tool_error():
    env = tools.execute_tool("get_build", {"week": 10 ** 300})
    assert env.success is False and "no Week" in env.error


def test_the_audit_line_is_scrubbed_before_it_is_cut(monkeypatch):
    monkeypatch.setattr(llm, "chat", scripted(answer("OUT_OF_SCOPE no.")))
    question = "Hello there, I really want the student to get in touch, please email " \
               "jane.doe@example.com"
    assert len(question) > 80                                    # the email crosses the cut
    client.post("/ask", json={"question": question, "user_id": "pii"})
    detail = memory.audit_log("pii")[0].detail
    assert "jane.doe" not in detail and "[REDACTED_E" in detail   # gone; marker is cut too


def test_half_an_emoji_never_reaches_the_provider():
    req = AskRequest.model_validate({"question": "What did W4 build \ud83d?",
                                     "history": [{"question": "q", "answer": "ok \ud83d"}]})
    for text in (req.question, req.history[0].answer):
        text.encode("utf-8")                                     # would raise on a surrogate


def test_verify_card_says_false_to_a_hostile_card(monkeypatch):
    _settings(monkeypatch, CARD_SIGNING_SEED="seed")
    keys = client.get("/.well-known/jwks.json").json()
    for bad in ({"signatures": "x"},
                {"signatures": [{"protected": signing.b64url(b"[1]"), "signature": "AA"}]}):
        assert signing.verify_card(bad, keys) is False


def test_a_profile_that_is_not_an_object_is_a_503(tmp_path, monkeypatch):
    for name in ("AGENTS.md", "architecture.md", "eval_results.json"):
        (tmp_path / name).write_text((ROOT / "data" / name).read_text(encoding="utf-8"),
                                     encoding="utf-8")
    (tmp_path / "profile.json").write_text(json.dumps([]), encoding="utf-8")
    _settings(monkeypatch, DATA_DIR=str(tmp_path))
    assert client.get("/portfolio").status_code == 503


def test_the_pages_handle_the_audited_cases():
    page = (ROOT / "index.html").read_text(encoding="utf-8")
    assert "Array.from(t).slice(0, HISTORY_CHARS)" in page       # cut by characters
    assert "let BUSY = 0" in page                                # no early re-enable
    assert "/^\\s*OUT_OF_SCOPE/" in page
    admin = (ROOT / "admin.html").read_text(encoding="utf-8")
    assert "buttons.forEach(x => x.disabled = true)" in admin    # one decision per click
    assert "user === '..'" in admin
