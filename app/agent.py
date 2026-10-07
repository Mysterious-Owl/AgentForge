"""The agent loop - the model chooses tools by function calling, the code runs them, checks the
answer's citations and keeps both caps.

    messages -> COST CEILING -> model call -> tool calls?  yes -> run each -> results back -> loop
                                                           no  -> the answer -> check citations

Three guarantees, all code, none of them prompt text:

  * **The ceiling covers the whole loop.** Before EVERY call, what has been spent so far plus
    the worst case of the next call is checked against the per-request ceiling ($0.05); over
    it is a 413 before that call is made. An oversized question fails on the first check.
  * **The cap bounds the loop.** Every model call spends one iteration (app/budget.py); the
    9th is a 429. A model that keeps calling tools cannot run up a bill.
  * **Citations are verified.** The answer may cite only sources a tool returned in THIS run.
    `grounded` is true only if it cites at least one and every one checks out - the Week 5
    rule, applied to an agent: a wrong answer with a fake source is worse than no answer.

`generate(messages, model, tools, budget) -> ModelTurn` is injected, so the tests and the
eval drive the loop with a scripted model and no network; the route passes `llm.chat`.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable

from app import llm, memory, tools
from app.budget import IterationBudget, cost_of, guard_cost
from app.schemas import AuditEntry, ModelTurn, PendingAction, ToolEnvelope, ToolStep, Turn

logger = logging.getLogger(__name__)

GenerateFn = Callable[[list[dict], str, list[dict], IterationBudget | None], ModelTurn]

# [AGENTS.md · W4] - a bracketed tag, not a markdown link "[text](url)".
_CITATION = re.compile(r"\[([^\[\]\n]{1,80})\](?!\()")
# Only a tag shaped like a source (a pack file, optionally " · part") is a citation. Brackets
# quoted from the pack - `[doc#N]`, `[chunk-id]`, `["action_execute"]` - are text, not claims.
_SOURCE_SHAPE = re.compile(r"[\w./-]+\.(?:md|json)(?: · \S.*)?")


class EmptyAnswer(RuntimeError):
    """The model finished with no text (blank, whitespace, or out of completion budget) - an
    empty message is never passed on as an answer; the route returns it as a 502."""
_ONE_INTRO = "one intro request per question - it is already waiting for the student"


@dataclass
class AgentRun:
    answer: str = ""
    steps: list[ToolStep] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)
    pending: PendingAction | None = None
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    baseline_usd: float = 0.0

    @property
    def grounded(self) -> bool:
        return (not llm.is_out_of_scope(self.answer) and bool(self.citations)
                and not self.unverified)


def check_citations(answer: str, sources: list[str]) -> tuple[list[str], list[str]]:
    """(verified, unverified) - each cited tag, once, in order. Whitespace is normalised; the
    tag must otherwise match a returned source exactly."""
    returned = {" ".join(s.split()) for s in sources}
    verified, unverified = [], []
    for raw in _CITATION.findall(answer):
        tag = " ".join(raw.split())
        if not _SOURCE_SHAPE.fullmatch(tag):
            continue
        bucket = verified if tag in returned else unverified
        if tag not in bucket:
            bucket.append(tag)
    return verified, unverified


def run(question: str, history: list[Turn], pack: dict, model: str, frontier_model: str,
        generate: GenerateFn, budget: IterationBudget, user_id: str = "anon",
        session_id: str = "default") -> AgentRun:
    """Answer one question: let the model call tools until it answers, within both caps."""
    out = AgentRun()
    messages = llm.build_messages(question, history, pack)
    specs = tools.TOOL_SPECS
    while True:
        # THE CEILING, before every call: spent so far + this call's worst case.
        guard_cost(llm.message_texts(messages, specs), model,
                   message_count=len(messages), already_spent=out.cost_usd)
        turn = generate(messages, model, specs, budget)
        out.calls += 1
        out.prompt_tokens += turn.prompt_tokens
        out.completion_tokens += turn.completion_tokens
        out.cost_usd = round(out.cost_usd + cost_of(turn.prompt_tokens, turn.completion_tokens,
                                                    model), 6)
        out.baseline_usd = round(out.baseline_usd + cost_of(
            turn.prompt_tokens, turn.completion_tokens, frontier_model), 6)
        if not turn.tool_calls:
            out.answer = turn.text.strip()
            if not out.answer:
                raise EmptyAnswer("the model returned an empty answer")
            break
        messages.append(llm.assistant_tool_message(turn))
        for call in turn.tool_calls:
            try:
                args = json.loads(call.arguments or "{}")
            except (json.JSONDecodeError, RecursionError):
                args = None
            if call.name == "request_intro" and out.pending is not None:
                # One question creates at most one intro - a prompt-injected visitor cannot
                # fill the student's inbox with one request.
                env = ToolEnvelope(success=False, tool=call.name, error=_ONE_INTRO)
            else:
                env = tools.execute_tool(call.name, args, user_id, session_id)
            if args is None:
                env = env.model_copy(update={"error": "arguments were not valid JSON"})
            out.steps.append(ToolStep(tool=call.name, args=args if isinstance(args, dict) else {},
                                      success=env.success, source=env.source, error=env.error,
                                      content=tools.excerpt(env)))
            if env.source and env.source not in out.sources:
                out.sources.append(env.source)
            if env.success and call.name == "request_intro":
                out.pending = tools.get_action(env.data["action_id"])
                # Audited the moment it exists - even if a later cap stops this request.
                memory.log_audit(AuditEntry(
                    user_id=user_id, session_id=session_id, kind="intro:proposed",
                    detail=f"{out.pending.id} from {out.pending.name} ({out.pending.reason}) - "
                           f"{out.pending.contact}"))
            messages.append(llm.tool_result_message(call.id, env.model_dump_json()))
    out.citations, out.unverified = check_citations(out.answer, out.sources)
    logger.info("agent done: %s call(s), tools=%s, citations=%s, unverified=%s",
                out.calls, [s.tool for s in out.steps], out.citations, out.unverified)
    return out
