"""Offline eval harness + the routing-quality gate the walkthrough points at.

Replays a FROZEN golden set (data/eval_golden.jsonl) through the SAME routing logic the live
`/ask` uses, and measures two things without a network call or an API key:

  1. Routing quality - did each question land in the tier a human labelled it? The pass rate
     is the headline `overall_score`, checked against `eval_score_floor` (the gate).
  2. The bill - what the routed mix costs vs an all-frontier baseline on the same tokens,
     priced from the ONE rate table in config.py. That ratio is the "I routed the easy
     majority down" number you say out loud.

Run it:
    python eval_run.py           # prints the report, updates its section of eval_results.json
    echo $?                      # non-zero if overall_score < the floor -> an eval GATE for CI

It merges ONE key (`portfolioagent_routing_gate`) into data/eval_results.json; after a pack
change, `--refresh-tokens` re-sizes the golden set's prompt tokens (the tests flag drift).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Offline: no model is called, so no key is needed - settings just require the field.
os.environ.setdefault("OPENAI_API_KEY", "not-needed-offline")

from app.budget import cost_of
from app.config import get_settings
from app.router import resolve_tier


def _load_golden(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def run() -> dict:
    settings = get_settings()
    golden_path = Path(settings.eval_golden_path)
    if not golden_path.is_absolute():
        golden_path = Path(__file__).parent / golden_path
    rows = _load_golden(golden_path)

    correct = 0
    routed_total = 0.0
    frontier_total = 0.0
    by_tier: dict[str, int] = {"small": 0, "frontier": 0}

    for row in rows:
        tier, model, _ = resolve_tier(row["question"])
        by_tier[tier] = by_tier.get(tier, 0) + 1
        if tier == row["expected_tier"]:
            correct += 1
        ptok, ctok = row["prompt_tokens"], row["completion_tokens"]
        routed_total += cost_of(ptok, ctok, model)
        frontier_total += cost_of(ptok, ctok, settings.frontier_model)

    n = len(rows)
    overall_score = round(correct / n, 3) if n else 0.0
    routed_total = round(routed_total, 6)
    frontier_total = round(frontier_total, 6)
    ratio = round(frontier_total / routed_total, 2) if routed_total else 0.0

    results = {
        "SAMPLE_DATA": True,
        "_readme": (
            f"Produced by eval_run.py over data/eval_golden.jsonl: {n} frozen questions about the "
            "capstone, labelled with the tier a human expects. Prompt tokens are the first "
            "call's input (system prompt, tool schemas, question); replace the golden set "
            "with your own before you quote these."
        ),
        "suite": "portfolioagent_routing_v1",
        "cases": n,
        "passed": correct,
        "overall_score": overall_score,
        "metrics": {
            "routing_accuracy": overall_score,
            "by_tier": by_tier,
        },
        "cost": {
            "avg_usd_per_request": round(routed_total / n, 6) if n else 0.0,
            "routed_total_usd": routed_total,
            "all_frontier_total_usd": frontier_total,
            "savings_ratio_x": ratio,
            "ceiling_usd_per_request": settings.cost_ceiling_usd,
            "_note": (
                "ceiling_usd_per_request is the one number here that is real and enforced: "
                "app/budget.py rejects any request whose projected cost exceeds it (413), and "
                "GET /health reports it. savings_ratio_x is the all-frontier bill divided by "
                "the routed-mix bill on the same tokens."
            ),
        },
        "tests": {
            "unit_and_contract": "run `pytest -q`",
            "_note": "The eval gate below fails CI if overall_score drops under the floor.",
        },
    }
    return results


def main() -> int:
    settings = get_settings()
    results = run()

    out_path = Path(settings.eval_results_path)
    if not out_path.is_absolute():
        out_path = Path(__file__).parent / out_path
    _write_section(out_path, results)

    c = results["cost"]
    print(f"cases            : {results['cases']}")
    print(f"overall_score    : {results['overall_score']:.3f}  (floor {settings.eval_score_floor})")
    print(f"routing by tier  : {results['metrics']['by_tier']}")
    print(f"routed bill      : ${c['routed_total_usd']:.6f}")
    print(f"all-frontier bill: ${c['all_frontier_total_usd']:.6f}")
    print(f"savings ratio    : {c['savings_ratio_x']}x cheaper than all-frontier")
    print(f"avg $ / request  : ${c['avg_usd_per_request']:.6f}")
    print(f"wrote            : {out_path}")

    # THE EVAL GATE: a routing-quality regression fails the build, loud, before a demo.
    if results["overall_score"] < settings.eval_score_floor:
        floor = settings.eval_score_floor
        print(f"\nEVAL GATE FAILED: {results['overall_score']:.3f} < floor {floor}")
        return 1
    print("\nEVAL GATE PASSED")
    return 0


def _write_section(out_path: Path, results: dict) -> None:
    """Merge this gate's results into eval_results.json - one key, everything else kept."""
    doc = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    doc["portfolioagent_routing_gate"] = results
    # newline="\n": on Windows, text mode would otherwise rewrite every line ending as CRLF.
    out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
                        newline="\n")


def golden_path() -> Path:
    path = Path(get_settings().eval_golden_path)
    return path if path.is_absolute() else Path(__file__).parent / path


def pack_prompt_tokens(question: str) -> int:
    """The input tokens of the FIRST call the live /ask makes for this question - counted the
    same way the cost ceiling counts them (the model's tokenizer + the chat framing). The
    tool turns after it depend on what the model chooses, so they are not sized here."""
    from app import llm, portfolio, tools
    from app.budget import count_input_tokens
    messages = llm.build_messages(question, [], portfolio.get_portfolio())
    return count_input_tokens(llm.message_texts(messages, tools.TOOL_SPECS),
                              message_count=len(messages))


def refresh_golden_tokens() -> int:
    """Re-size every row's prompt_tokens to the current pack, to the nearest 10 tokens - so a
    one-character change in the pack (this script's own output is part of it) cannot make the
    sizing flip back and forth. Labels and answers stay frozen."""
    path = golden_path()
    rows = _load_golden(path)
    for row in rows:
        row["prompt_tokens"] = round(pack_prompt_tokens(row["question"]), -1)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8", newline="\n")
    print(f"refreshed prompt_tokens for {len(rows)} rows in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(refresh_golden_tokens() if "--refresh-tokens" in sys.argv else main())
