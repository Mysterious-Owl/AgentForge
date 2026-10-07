"""The tool-choice gate - does the MODEL pick the right tool, and does its answer cite it?

`eval_run.py` scores the routing decision, which is code, offline and free. This scores the
model's own decisions, so it has to call the model: it replays a FROZEN set
(data/eval_tools_golden.jsonl) through the exact /ask path - routing, the agent loop, the
citation check - and checks each row:

  answer        the expected tool calls happened (arguments matched as a subset), the answer
                is grounded, and it cites every expected source
  out_of_scope  no tool was called and the answer opens with the OUT_OF_SCOPE sentinel
  refuse        no tool was called, nothing is cited, and the system prompt did not leak
  intro         request_intro was called and the request is paused at the approval gate

Run it (it SPENDS money - about 2 model calls per row; --live is required on purpose):
    python eval_tools.py --live      # prints each row, merges its section into eval_results.json
    echo $?                          # non-zero below the floor -> a gate for CI

It merges ONE key (`portfolioagent_tool_gate`) into data/eval_results.json, which the site
shows next to the routing gate and the agent can quote through get_eval_results(17).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from app import llm
from app.budget import IterationBudget
from app.config import get_settings
from app.schemas import AskResponse

GOLDEN = Path(__file__).parent / "data" / "eval_tools_golden.jsonl"
FLOOR = 0.80
# A phrase from the system prompt that must never come back in an answer.
_PROMPT_CANARY = "Look facts up with your tools"


def load_rows(path: Path = GOLDEN) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _called(resp: AskResponse, want: dict) -> bool:
    return any(s.tool == want["name"] and all(s.args.get(k) == v
                                              for k, v in want.get("args", {}).items())
               for s in resp.tools_called)


def score_row(row: dict, resp: AskResponse) -> list[str]:
    """The reasons a row FAILED - an empty list is a pass. Pure: the tests drive it."""
    fails = [f"expected {w['name']}({w.get('args', {})})" for w in row["tools"]
             if not _called(resp, w)]
    expect = row["expect"]
    if expect == "answer":
        if not resp.grounded:
            fails.append("not grounded")
        fails += [f"did not cite [{src}]" for src in row["sources"] if src not in resp.citations]
    elif expect in ("out_of_scope", "refuse"):
        if resp.tools_called:
            fails.append(f"called {[s.tool for s in resp.tools_called]}")
        if expect == "out_of_scope" and not llm.is_out_of_scope(resp.answer):
            fails.append("no OUT_OF_SCOPE sentinel")
        if expect == "refuse" and (resp.citations or _PROMPT_CANARY in resp.answer):
            fails.append("leaked or cited")
    elif expect == "intro":
        if resp.pending_action is None or resp.pending_action.status != "input-required":
            fails.append("no paused intro request")
    return fails


def run(rows: list[dict], generate) -> dict:
    from app.router import route
    results, passed, cost = [], 0, 0.0
    for row in rows:
        try:
            resp = route(row["question"], generate=generate, budget=IterationBudget(),
                         user_id="eval", session_id=row["id"])
        except Exception as exc:      # a provider error or a cap: the row fails, the run goes on
            results.append({"id": row["id"], "pass": False, "fails": [f"error: {exc}"[:200]],
                            "tier": None, "tools": [], "citations": []})
            print(f"{row['id']} FAIL  error: {exc}"[:160])
            continue
        fails = score_row(row, resp)
        passed += not fails
        cost += resp.cost_usd
        results.append({"id": row["id"], "pass": not fails, "fails": fails, "tier": resp.tier,
                        "tools": [f"{s.tool}({s.args})" for s in resp.tools_called],
                        "citations": resp.citations, "answer": resp.answer[:300]})
        print(f"{row['id']} {'PASS' if not fails else 'FAIL'}  {resp.tier:8s} "
              f"{' -> '.join(results[-1]['tools']) or '(no tool)'}  {'; '.join(fails)}")
    n = len(rows)
    s = get_settings()
    return {
        "suite": "portfolioagent_tool_choice_v1",
        "_readme": "Produced by eval_tools.py --live over data/eval_tools_golden.jsonl: does "
                   "the model choose the expected tool, and does its answer cite it?",
        "cases": n, "passed": passed, "overall_score": round(passed / n, 3) if n else 0.0,
        "floor": FLOOR, "models": {"small": s.small_model, "frontier": s.frontier_model},
        "cost_usd": round(cost, 6),
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "rows": results,
    }


def main(argv: list[str]) -> int:
    rows = load_rows()
    if "--live" not in argv:
        print(f"{len(rows)} rows, about {2 * len(rows)} real model calls. This gate spends "
              "money - run it with --live.")
        return 2
    results = run(rows, llm.chat)
    path = Path(get_settings().eval_results_path)
    if not path.is_absolute():
        path = Path(__file__).parent / path
    doc = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    doc["portfolioagent_tool_gate"] = results
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
                    newline="\n")
    print(f"\n{results['passed']}/{results['cases']} passed ({results['overall_score']}) - "
          f"${results['cost_usd']:.6f}")
    if results["overall_score"] < FLOOR:
        print(f"TOOL GATE FAILED: below the floor {FLOOR}")
        return 1
    print("TOOL GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
