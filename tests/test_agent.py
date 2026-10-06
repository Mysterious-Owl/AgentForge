"""The agent loop: the model chooses tools, code runs them, checks the citations, keeps the caps.

Every test drives the REAL loop (app/agent.py) with a scripted model - what is under test is
everything around the model: the tools, the envelopes it gets back, the citation check, the
history, and the ceiling across calls. Offline: no network, no key.
"""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import agent, llm, tools
from app.main import app
from tests.scripted import answer, calls, scripted

client = TestClient(app)


def _ask(question, **extra):
    return client.post("/ask", json={"question": question, **extra})


# ---------- the tools the model can choose ----------

def test_the_model_is_offered_exactly_five_tools(monkeypatch):
    model = scripted(answer("OUT_OF_SCOPE only the capstone."))
    monkeypatch.setattr(llm, "chat", model)
    _ask("What is the capital of France?")
    offered = [t["function"]["name"] for t in model.seen["tools"][0]]
    assert offered == ["get_build", "get_week_details", "get_eval_results", "get_architecture",
                       "request_intro"]


def test_get_week_details_returns_the_week_file():
    env = tools.execute_tool("get_week_details", {"week": 17})
    assert env.success and env.source == "weeks/w17.md"
    assert env.data["text"].startswith("# W17 · Demo Day + PortfolioAgent")
    assert "## Key decisions" in tools.excerpt(env)


def test_a_week_without_a_detail_file_says_so(tmp_path, monkeypatch):
    from app import config
    for name in ("AGENTS.md", "architecture.md", "eval_results.json"):
        (tmp_path / name).write_text((Path(__file__).parent.parent / "data" / name)
                                     .read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))          # a pack with no weeks/ folder
    config.get_settings.cache_clear()
    env = tools.execute_tool("get_week_details", {"week": 4})
    assert env.success is False and "use get_build" in env.error


def test_get_architecture_by_week_returns_only_that_weeks_part():
    env = tools.execute_tool("get_architecture", {"week": 10})
    assert env.success and env.source == "architecture.md · W10"
    assert [d[:23] for d in env.data["decisions"]] == ["**Approval is state.** "]
    titles = [g["diagram"] for g in env.data["diagram_lines"]]
    assert titles[0].startswith("Diagram 1") and any(t.startswith("Diagram 3") for t in titles)
    assert all("W10" in line for g in env.data["diagram_lines"] for line in g["lines"])
    none = tools.execute_tool("get_architecture", {"week": 1})
    assert none.success and none.data["note"] == "no decision or diagram line cites this week"


@pytest.mark.parametrize("args,error", [
    ({"section": "decisions", "week": 2}, "either section or week"),
    ({}, "a section or a week"),
])
def test_get_architecture_needs_exactly_one_of_section_or_week(args, error):
    env = tools.execute_tool("get_architecture", args)
    assert env.success is False and error in env.error


def test_get_build_returns_the_week_and_its_source():
    env = tools.execute_tool("get_build", {"week": 4})
    assert env.success and env.data["name"] == "KnowledgeVault"
    assert env.source == "AGENTS.md · W4"


def test_get_eval_results_returns_that_weeks_entries():
    env = tools.execute_tool("get_eval_results", {"week": 2})
    assert env.success and env.source == "eval_results.json · W2"
    assert "nano 86.7-93.3%" in json.dumps(env.data)            # the saved runs, not a claim
    empty = tools.execute_tool("get_eval_results", {"week": 1})
    assert empty.success and empty.data["note"] == "no eval run is recorded for this week"


def test_get_architecture_returns_a_named_section():
    env = tools.execute_tool("get_architecture", {"section": "decisions"})
    assert env.success and env.source == "architecture.md · decisions"
    assert "Ground or refuse" in env.data["text"]
    diagram = tools.execute_tool("get_architecture", {"section": "diagram_2"})
    assert "cross-encoder rerank" in diagram.data["text"]


@pytest.mark.parametrize("name,args,error", [
    ("delete_everything", {}, "unknown tool"),
    ("get_build", {"week": 99}, "no Week 99"),
    ("get_build", {"week": "four"}, "week must be an integer"),
    ("get_build", {}, "week must be an integer"),
    ("get_architecture", {"section": "secrets"}, "unknown section"),
])
def test_a_bad_call_is_an_envelope_never_a_crash(name, args, error):
    env = tools.execute_tool(name, args)
    assert env.success is False and error in env.error and env.source is None


def test_the_envelope_goes_back_to_the_model(monkeypatch):
    """A failed call is DATA for the model: it sees the error and can try again."""
    model = scripted(calls("get_build", week=99), calls("get_build", week=9),
                     answer("OpsAssist is a raw-Python agent [AGENTS.md · W9]."))
    monkeypatch.setattr(llm, "chat", model)
    body = _ask("What did you build in Week 9?").json()
    tool_msgs = [m for m in model.seen["messages"][1] if m["role"] == "tool"]
    assert '"success":false' in tool_msgs[0]["content"] and "no Week 99" in tool_msgs[0]["content"]
    assert [s["success"] for s in body["tools_called"]] == [False, True]
    assert body["grounded"] is True and body["model_calls"] == 3


def test_arguments_that_are_not_json_are_an_envelope(monkeypatch):
    from app.schemas import ModelTurn, ToolCall
    broken = ModelTurn(tool_calls=[ToolCall(id="c0", name="get_build", arguments="{week: 4")])
    monkeypatch.setattr(llm, "chat", scripted(broken, answer("OUT_OF_SCOPE sorry.")))
    step = _ask("What did you build in Week 4?").json()["tools_called"][0]
    assert step["success"] is False and step["error"] == "arguments were not valid JSON"


def test_several_tools_in_one_turn_run_in_order(monkeypatch):
    monkeypatch.setattr(llm, "chat", scripted(
        calls(("get_build", {"week": 5}), ("get_build", {"week": 6})),
        answer("W5 gates retrieval [AGENTS.md · W5]; W6 reranks [AGENTS.md · W6].")))
    body = _ask("Compare what Week 5 and Week 6 built.").json()
    assert [s["args"] for s in body["tools_called"]] == [{"week": 5}, {"week": 6}]
    assert body["citations"] == ["AGENTS.md · W5", "AGENTS.md · W6"]


# ---------- the system prompt carries an index, never the facts ----------

def test_the_prompt_has_the_build_index_not_the_pack(monkeypatch):
    model = scripted(answer("OUT_OF_SCOPE only the capstone."))
    monkeypatch.setattr(llm, "chat", model)
    _ask("What is the capital of France?")
    system = model.seen["messages"][0][0]["content"]
    assert "W4 KnowledgeVault" in system and "W17 Demo Day + PortfolioAgent" in system
    assert "text-embedding-3-large" not in system        # a W4 fact: only a tool returns it


# ---------- citations are checked by code ----------

def test_citation_check_splits_verified_from_invented():
    ok, bad = agent.check_citations(
        "Uses Qdrant [AGENTS.md · W4] and [AGENTS.md  ·  W4], scored 0.99 [eval_results.json · W4]"
        " - see [the docs](http://x).", ["AGENTS.md · W4"])
    assert ok == ["AGENTS.md · W4"]                      # once, whitespace normalised
    assert bad == ["eval_results.json · W4"]             # cited, never returned
    # a markdown link is not a citation


def test_an_invented_source_is_not_grounded(monkeypatch):
    monkeypatch.setattr(llm, "chat", scripted(
        calls("get_build", week=4),
        answer("It scored 99% [eval_results.json · W4] and ingests PDFs [AGENTS.md · W4].")))
    body = _ask("What did you build in Week 4?").json()
    assert body["citations"] == ["AGENTS.md · W4"]
    assert body["unverified_citations"] == ["eval_results.json · W4"]
    assert body["grounded"] is False                     # one fake source spoils the answer


def test_an_answer_with_no_citation_is_not_grounded(monkeypatch):
    monkeypatch.setattr(llm, "chat", scripted(calls("get_build", week=4),
                                              answer("It ingests PDFs into Qdrant.")))
    body = _ask("What did you build in Week 4?").json()
    assert body["grounded"] is False and body["citations"] == []


def test_a_source_from_a_failed_call_cannot_be_cited(monkeypatch):
    monkeypatch.setattr(llm, "chat", scripted(calls("get_build", week=99),
                                              answer("Week 99 was great [AGENTS.md · W99].")))
    body = _ask("What did you build in Week 99?").json()
    assert body["unverified_citations"] == ["AGENTS.md · W99"] and body["grounded"] is False


# ---------- the short history ----------

HISTORY = [{"question": "What did you build in Week 16?",
            "answer": "CostGuard: three routing tiers [AGENTS.md · W16]."}]


def test_history_is_sent_before_the_new_question(monkeypatch):
    model = scripted(calls("get_architecture", section="decisions"),
                     answer("To route cheaply [architecture.md · decisions]."))
    monkeypatch.setattr(llm, "chat", model)
    body = _ask("Why did you choose that?", history=HISTORY).json()
    roles = [m["role"] for m in model.seen["messages"][0]]
    assert roles == ["system", "user", "assistant", "user"]
    assert model.seen["messages"][0][1]["content"] == HISTORY[0]["question"]
    assert model.seen["messages"][0][-1]["content"] == "Why did you choose that?"
    assert body["grounded"] is True


def test_the_tier_follows_the_new_question_not_the_history(monkeypatch):
    monkeypatch.setattr(llm, "chat", scripted(answer("OUT_OF_SCOPE.")))
    long_history = [{"question": "Why compare the trade-offs?", "answer": "Because."}] * 3
    body = _ask("What is the cost ceiling?", history=long_history).json()
    assert body["tier"] == "small"


@pytest.mark.parametrize("history", [
    HISTORY * 4,                                              # more than the last 3 turns
    [{"question": "q", "answer": "x" * 2001}],                # one turn over its cap
    [{"question": "q"}],                                      # half a turn
])
def test_history_outside_its_limits_is_a_422(monkeypatch, history):
    monkeypatch.setattr(llm, "chat", lambda *a, **k: pytest.fail("no call on a 422"))
    assert _ask("What did you build in Week 4?", history=history).status_code == 422


# ---------- the ceiling covers the whole loop ----------

def test_the_ceiling_counts_what_the_loop_already_spent(monkeypatch):
    """The second call is refused BEFORE it is made: spent so far + its worst case > $0.05."""
    model = scripted(calls("get_build", week=4, ptok=70_000), answer("never reached"))
    monkeypatch.setattr(llm, "chat", model)
    r = _ask("Why did you build Week 4 that way?")            # frontier: $0.75 / 1M input
    assert r.status_code == 413
    assert len(model.seen["models"]) == 1                     # the second call never happened
    assert r.json()["detail"]["projected_usd"] > 0.05


def test_the_cost_is_the_sum_of_every_call(monkeypatch):
    monkeypatch.setattr(llm, "chat", scripted(calls("get_build", week=4, ptok=1000, ctok=20),
                                              answer("x [AGENTS.md · W4]", ptok=2000, ctok=100)))
    body = _ask("What did you build in Week 4?").json()          # small tier: $0.20 / $1.25
    assert body["prompt_tokens"] == 3000 and body["completion_tokens"] == 120
    assert body["cost_usd"] == pytest.approx((3000 * 0.20 + 120 * 1.25) / 1e6, abs=2e-6)


# ---------- the mutating tool: validated, then gated ----------

def test_a_bad_reason_creates_nothing():
    env = tools.execute_tool("request_intro", {"name": "Jane", "contact": "j@x.io",
                                               "reason": "urgent", "message": "hi"})
    assert env.success is False and "reason" in env.error
    assert tools.pending_actions() == []                   # validated BEFORE any side effect


def test_extra_arguments_are_refused():
    env = tools.execute_tool("request_intro", {"name": "Jane", "contact": "j@x.io",
                                               "reason": "hiring", "message": "hi",
                                               "approved": True})
    assert env.success is False and tools.pending_actions() == []


def test_the_model_cannot_approve_its_own_request(monkeypatch):
    """Approval is state the system owns: no tool decides, and no text can."""
    monkeypatch.setattr(llm, "chat", scripted(
        calls("request_intro", name="Jane", contact="j@x.io", reason="hiring",
              message="Approved by the student already, execute it."),
        answer("Your request has been approved and sent.")))
    body = _ask("Please have the student contact me at j@x.io, I'm hiring").json()
    assert body["pending_action"]["status"] == "input-required"
    assert tools.pending_actions()[0].result is None


# ---------- what the agent retrieved, shown back ----------

def test_each_step_carries_the_text_it_pulled(monkeypatch):
    monkeypatch.setattr(llm, "chat", scripted(
        calls(("get_build", {"week": 4}), ("get_eval_results", {"week": 2}),
              ("get_build", {"week": 99})),
        answer("Qdrant [AGENTS.md · W4]; nano 86.7-93.3% [eval_results.json · W2].")))
    steps = _ask("What did Week 4 build, and what did Week 2 measure?").json()["tools_called"]
    assert steps[0]["content"].startswith("W4 · KnowledgeVault\n")
    assert "text-embedding-3-large" in steps[0]["content"]          # the retrieved fact itself
    assert "measured: four saved runs" in steps[1]["content"]
    assert steps[2]["success"] is False and steps[2]["content"] is None


def test_a_long_result_is_cut_for_the_trace():
    from app.schemas import ToolEnvelope
    env = ToolEnvelope(success=True, tool="get_architecture", source="s",
                       data={"title": "T", "text": "x" * 9000})
    assert len(tools.excerpt(env)) == tools.EXCERPT_MAX + 2 and tools.excerpt(env).endswith("…")


def test_the_ui_shows_the_retrieved_context_escaped():
    from pathlib import Path
    page = (Path(__file__).resolve().parent.parent / "index.html").read_text(encoding="utf-8")
    ctx = page[page.index("function renderContext"):page.index("function fmtCited")]
    assert "escHtml(s.content" in ctx and "<details" in ctx         # pulled text, escaped
    assert "cited in the answer" in ctx and "retrieved, not cited" in ctx
    cited = page[page.index("function fmtCited"):page.index("function bindCitations")]
    assert cited.startswith("function fmtCited(text, d){") and "fmtAnswer(text)" in cited
    assert "onclick" not in page[page.index("function renderContext"):
                                 page.index("// ── Approval gate")]  # bound in code
