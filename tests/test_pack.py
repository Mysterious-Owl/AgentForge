"""The context pack is what the tools read - these tests keep it, and the numbers quoted about
it, honest. Each failure message says what to run, so a pack edit never leaves a stale figure.
"""
import json
from pathlib import Path

import pytest

import eval_run
import eval_tools
from app import llm, portfolio, tools
from app.budget import project_cost
from app.config import get_settings
from app.schemas import AskResponse, Turn

DATA = Path(__file__).parent.parent / "data"


def test_a_long_tool_heavy_loop_still_fits_the_ceiling():
    """A realistic worst case on the frontier tier: a long question, a full 3-turn history and
    the four biggest tool results - the last call of that loop still clears $0.05."""
    s = get_settings()
    question = "Walk me through the trade-offs in your retrieval design. " * 20   # ~1,150 chars
    history = [Turn(question="q" * 2000, answer="a" * 2000)] * 3
    messages = llm.build_messages(question, history, portfolio.get_portfolio())
    for env in (tools.get_architecture("diagram_1"), tools.get_architecture("decisions"),
                tools.get_build(16), tools.get_eval_results(17)):
        messages.append(llm.tool_result_message("c", env.model_dump_json()))
    worst = project_cost(llm.message_texts(messages, tools.TOOL_SPECS), s.frontier_model,
                         message_count=len(messages))
    assert worst < s.cost_ceiling_usd / 2, f"one call could cost ${worst} - trim the pack"


def test_golden_prompt_tokens_track_the_pack():
    """The bill is priced on these token counts - a pack edit must not silently skew it."""
    for row in eval_run._load_golden(eval_run.golden_path()):
        live = eval_run.pack_prompt_tokens(row["question"])
        # --refresh-tokens rounds each row to the nearest 10, so honest drift is a few tokens.
        assert abs(row["prompt_tokens"] - live) <= max(10, 0.02 * live), (
            f"{row['id']}: {row['prompt_tokens']} vs ~{live} for the current pack - "
            "run `python eval_run.py --refresh-tokens`, then `python eval_run.py`")


def test_saved_routing_results_match_a_fresh_run():
    """The numbers the pack, README and video quote are the numbers the code produces."""
    saved = json.loads((DATA / "eval_results.json").read_text(encoding="utf-8"))
    assert saved["portfolioagent_routing_gate"] == eval_run.run(), \
        "data/eval_results.json is stale - run `python eval_run.py`"


def test_eval_run_merges_only_its_own_section(tmp_path):
    out = tmp_path / "eval_results.json"
    out.write_text(json.dumps({"weekly": {"w16": "kept"}, "portfolioagent_routing_gate": {}}),
                   encoding="utf-8")
    eval_run._write_section(out, {"overall_score": 1.0})
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["weekly"] == {"w16": "kept"}                     # the capstone's results survive
    assert doc["portfolioagent_routing_gate"] == {"overall_score": 1.0}


# ---------- the tool-choice golden set ----------

@pytest.mark.parametrize("row", eval_tools.load_rows(), ids=lambda r: r["id"])
def test_tool_golden_rows_ask_for_tools_that_exist(row):
    """Every expected call is a real tool with real arguments, and every expected source is
    exactly what that tool returns - so the gate can fail only on the MODEL."""
    names = [t["function"]["name"] for t in tools.TOOL_SPECS]
    returned = set()
    for want in row["tools"]:
        assert want["name"] in names
        if want["name"] != "request_intro":
            env = tools.execute_tool(want["name"], want["args"])
            assert env.success, (row["id"], env.error)
            returned.add(env.source)
    assert set(row["sources"]) <= returned


def _resp(**kw):
    base = {"answer": "x", "model": "m", "tier": "small", "complexity": "simple",
            "grounded": False}
    return AskResponse(**{**base, **kw})


def test_the_tool_gate_scorer():
    row = eval_tools.load_rows()[0]                       # get_build(4), cite [AGENTS.md · W4]
    step = {"tool": "get_build", "args": {"week": 4}, "success": True, "source": "AGENTS.md · W4"}
    good = _resp(grounded=True, tools_called=[step], citations=["AGENTS.md · W4"])
    assert eval_tools.score_row(row, good) == []
    wrong_week = _resp(grounded=True, tools_called=[{**step, "args": {"week": 5}}],
                       citations=["AGENTS.md · W4"])
    assert "expected get_build({'week': 4})" in eval_tools.score_row(row, wrong_week)
    assert "not grounded" in eval_tools.score_row(row, _resp(tools_called=[step]))
    off = next(r for r in eval_tools.load_rows() if r["expect"] == "out_of_scope")
    assert eval_tools.score_row(off, _resp(answer="OUT_OF_SCOPE no.")) == []
    assert eval_tools.score_row(off, _resp(answer="Paris.")) == ["no OUT_OF_SCOPE sentinel"]
    typo = {**off, "expect": "refused"}                    # a golden-set typo fails loudly
    assert eval_tools.score_row(typo, _resp(answer="OUT_OF_SCOPE no."))[0].startswith(
        "unknown expect 'refused'")


def test_a_row_that_errors_after_spending_still_counts_in_the_cost(capsys):
    """A model that keeps calling tools hits the iteration cap - the row fails, and the eight
    calls it made are still on the run's bill."""
    from tests.scripted import calls, scripted
    from app.budget import cost_of
    row = eval_tools.load_rows()[0]
    out = eval_tools.run([row], scripted(calls("get_build", week=4)))
    assert out["passed"] == 0 and out["rows"][0]["fails"][0].startswith("error:")
    small = get_settings().small_model
    assert out["cost_usd"] == pytest.approx(8 * cost_of(900, 30, small), abs=1e-6)


def test_the_tool_gate_refuses_to_spend_without_live(capsys):
    assert eval_tools.main([]) == 2
    assert "--live" in capsys.readouterr().out


def test_readme_quotes_the_saved_routing_numbers():
    """The README states the routing bill - it must be the bill eval_run.py produced."""
    gate = json.loads((DATA / "eval_results.json").read_text(encoding="utf-8"))
    cost = gate["portfolioagent_routing_gate"]["cost"]
    readme = (DATA.parent / "README.md").read_text(encoding="utf-8")
    for figure in (f"${cost['routed_total_usd']:.6f}", f"${cost['all_frontier_total_usd']:.6f}",
                   f"{cost['savings_ratio_x']}x"):
        assert figure in readme, f"README does not quote {figure} - update it from eval_run.py"
