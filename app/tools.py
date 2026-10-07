"""Tool layer - what the MODEL can call: four read tools over the pack, one mutating tool
behind an approval gate, and the dispatcher that runs whichever it chooses.

  get_build(week)            -> that week's milestone from AGENTS.md         [AGENTS.md · W4]
  get_week_details(week)     -> the week in depth, data/weeks/w04.md         [weeks/w04.md]
  get_eval_results(week)     -> that week's entries in eval_results.json
                                                                     [eval_results.json · W4]
  get_architecture(section)  -> a named section: overview, decisions, a diagram ...
                                                                   [architecture.md · decisions]
  get_architecture(week)     -> the decisions and diagram lines tagged with that week
                                                                   [architecture.md · W4]
  request_intro(...)         -> MUTATING: a visitor asks to reach the student. It creates a
                                paused action and does nothing else until the student decides.

The model chooses by function calling - nothing here maps a question to a tool. Four
properties the red-team pass checks, all enforced here:

  * **Structured errors** - `execute_tool()` returns a `ToolEnvelope`; an unknown tool, bad
    arguments or a failing tool come back to the model as `{success: false, error}`. It never
    raises (except a missing pack file - that is the route's 503) and never crashes the run.
  * **A citable source** - every successful read carries `source`, the only tag the answer's
    citation check accepts.
  * **Enum validation** - `request_intro`'s `reason` is a Literal: a bad value is a tool error,
    BEFORE any side effect.
  * **The approval gate** - `request_intro` creates an action at `input-required` and mutates
    NOTHING. It executes only when `decide(approve=True)` is called by the student (/approve,
    admin token). Approval is a state the system owns, not a sentence anyone can type.

Pending actions live in an in-process dict behind a lock (a real deploy swaps this seam for a
durable store, where the database row carries the same check-then-act guarantee).
"""
from __future__ import annotations

import logging
import threading
import uuid
from typing import Any

from pydantic import ValidationError

from app import portfolio
from app.schemas import IntroRequest, PendingAction, ToolEnvelope

logger = logging.getLogger(__name__)


# ── The schemas the model sees (OpenAI function calling) ─────────────────────────

def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required,
                       "additionalProperties": False}}}


_WEEK = {"type": "integer", "minimum": 1, "maximum": 17,
         "description": "Course week of the build, 1-17."}

TOOL_SPECS: list[dict] = [
    _fn("get_build", "What the student built in one week: the milestone's name and a "
        "one-paragraph summary from AGENTS.md.", {"week": _WEEK}, ["week"]),
    _fn("get_week_details", "One week's build in depth: what it is, how it works, its key "
        "decisions, its measured results and its stack. Use it for how/why questions about a "
        "week, or when the summary from get_build is not enough.", {"week": _WEEK}, ["week"]),
    _fn("get_eval_results", "The measured results and eval gates recorded for one week in "
        "eval_results.json (Week 17 includes this agent's own gates).", {"week": _WEEK},
        ["week"]),
    _fn("get_architecture", "The capstone's design. Pass `section` for one named section "
        "(the overview, the operating rules, `portfolio_agent` for this agent itself - its model "
        "tiers and routing, why the small tier runs on nano, the open-model slot, its caps and "
        "gates - the four-layer spine, the key decisions, one of four architecture diagrams, "
        "where the numbers come from), or pass "
        "`week` for just that week's part: the decisions that cite it and the diagram lines "
        "tagged with it.",
        {"section": {"type": "string", "enum": list(portfolio.SECTIONS)}, "week": _WEEK}, []),
    _fn("request_intro", "Ask the student to get in touch with this visitor. Use ONLY when the "
        "visitor explicitly asks to contact, hire or meet the student and gives a way to "
        "reach them. It does not send anything: it waits for the student's approval.",
        {"name": {"type": "string"}, "company": {"type": "string"},
         "contact": {"type": "string", "description": "Email or other contact the visitor gave."},
         "reason": {"type": "string", "enum": ["hiring", "collaboration", "feedback", "other"]},
         "message": {"type": "string", "description": "What the visitor wants, in one or two "
                     "sentences."}},
        ["name", "contact", "reason", "message"]),
]


# ── Read tools (no side effects) ─────────────────────────────────────────────────

def _week(args: dict) -> int:
    week = args.get("week")
    if isinstance(week, bool) or not isinstance(week, int):
        raise ValueError("week must be an integer")
    if not 1 <= week <= 17:              # the course's weeks - the schema says the same
        raise ValueError(f"no Week {week} in the build history - weeks are 1-17")
    return week


def get_build(week: int) -> ToolEnvelope:
    build = next((b for b in portfolio.get_portfolio()["builds"] if b["week"] == week), None)
    if build is None:
        return ToolEnvelope(success=False, tool="get_build",
                            error=f"no Week {week} in the build history")
    return ToolEnvelope(success=True, tool="get_build", data=build,
                        source=f"AGENTS.md · W{week}")


def get_eval_results(week: int) -> ToolEnvelope:
    results = portfolio.week_results(week)
    note = None if results else "no eval run is recorded for this week"
    return ToolEnvelope(success=True, tool="get_eval_results",
                        data={"week": week, "results": results, "note": note},
                        source=f"eval_results.json · W{week}")


def get_week_details(week: int) -> ToolEnvelope:
    text = portfolio.week_details(week)
    if text is None:
        return ToolEnvelope(success=False, tool="get_week_details",
                            error=f"no detail file for Week {week}; use get_build for its summary")
    return ToolEnvelope(success=True, tool="get_week_details", data={"week": week, "text": text},
                        source=f"weeks/w{week:02d}.md")


def get_architecture_for_week(week: int) -> ToolEnvelope:
    found = portfolio.week_architecture(week)
    if not found["decisions"] and not found["diagram_lines"]:
        return ToolEnvelope(success=True, tool="get_architecture",
                            data={**found, "note": "no decision or diagram line cites this week"},
                            source=f"architecture.md · W{week}")
    return ToolEnvelope(success=True, tool="get_architecture", data=found,
                        source=f"architecture.md · W{week}")


def _architecture(args: dict) -> ToolEnvelope:
    if args.get("section") and args.get("week") is not None:
        raise ValueError("pass either section or week, not both")
    if args.get("week") is not None:
        return get_architecture_for_week(_week(args))
    if not args.get("section"):
        raise ValueError("pass a section or a week")
    return get_architecture(str(args["section"]))


def get_architecture(section: str) -> ToolEnvelope:
    found = portfolio.section(section)
    if found is None:
        return ToolEnvelope(success=False, tool="get_architecture",
                            error=f"unknown section '{section}'; known: {list(portfolio.SECTIONS)}")
    return ToolEnvelope(success=True, tool="get_architecture",
                        data={"title": found["title"], "text": found["text"]},
                        source=found["source"])


_READ_TOOLS = {
    "get_build": lambda a: get_build(_week(a)),
    "get_eval_results": lambda a: get_eval_results(_week(a)),
    "get_week_details": lambda a: get_week_details(_week(a)),
    "get_architecture": _architecture,
}


def execute_tool(name: str, args: Any, user_id: str = "anon",
                 session_id: str = "default") -> ToolEnvelope:
    """Run the tool the model chose. Unknown tool, bad arguments or a failure -> a structured
    envelope back to the model, never a crash. A missing pack file is NOT swallowed: it raises
    ContextPackError, which the route turns into a 503 - a broken pack is not a tool error."""
    if not isinstance(args, dict):
        return ToolEnvelope(success=False, tool=name, error="arguments must be a JSON object")
    if name == "request_intro":
        return request_intro(args, user_id, session_id)
    fn = _READ_TOOLS.get(name)
    if fn is None:
        logger.warning("unknown tool requested: %r", name)   # %r: no forged log lines
        known = [t["function"]["name"] for t in TOOL_SPECS]
        return ToolEnvelope(success=False, tool=name,
                            error=f"unknown tool '{name}'; known: {known}")
    try:
        return fn(args)
    except (ValueError, TypeError) as exc:   # bad arguments are data for the model to fix
        return ToolEnvelope(success=False, tool=name, error=str(exc))


EXCERPT_MAX = 4_000   # characters of one tool result shown back in the trace


def excerpt(env: ToolEnvelope) -> str | None:
    """A read tool's result as plain text - what the UI shows as the retrieved context. The
    model got the same data as JSON; this is that data, readable. None for a failed call."""
    if not env.success or env.tool == "request_intro" or env.data is None:
        return None
    data = env.data
    if env.tool == "get_build":
        text = f"W{data['week']} · {data['name']}\n{data['summary']}"
    elif env.tool == "get_week_details":
        text = data["text"]
    elif env.tool == "get_architecture" and "week" in data:
        lines = [data["note"]] if data.get("note") else []
        lines += [f"Decision: {d}" for d in data["decisions"]]
        for g in data["diagram_lines"]:
            lines.append(g["diagram"])
            lines += [f"  {line}" for line in g["lines"]]
        text = "\n".join(lines)
    elif env.tool == "get_architecture":
        text = f"{data['title']}\n{data['text']}"
    elif env.tool == "get_eval_results":
        lines = [data["note"]] if data.get("note") else []
        for r in data.get("results", []):
            lines.append(r["label"])
            lines += [f"  {k}: {', '.join(map(str, v)) if isinstance(v, list) else v}"
                      for k, v in r["fields"].items() if not str(k).startswith("_")]
        text = "\n".join(lines)
    else:
        return None
    return text if len(text) <= EXCERPT_MAX else text[:EXCERPT_MAX] + " …"


# ── The mutating tool + the approval gate ────────────────────────────────────────

_PENDING: dict[str, PendingAction] = {}
# Approvals arrive on a threadpool. The lock makes decide() check-then-act atomic, so two
# simultaneous "approve" clicks execute the action exactly once.
_LOCK = threading.Lock()
# /ask is public, so the store is bounded: a decided action makes room for a new one; an
# open one never does - it waits for its human. All open -> the next request is refused.
MAX_ACTIONS = 1_000


class TooManyPendingActions(Exception):
    def __init__(self, limit: int) -> None:
        super().__init__(f"{limit} actions are waiting for a decision")
        self.limit = limit


class AlreadyDecided(Exception):
    def __init__(self, status: str) -> None:
        super().__init__(f"action already {status}")
        self.status = status


def _make_room() -> None:
    """Evict the oldest DECIDED action if the store is full. Call with the lock held."""
    if len(_PENDING) < MAX_ACTIONS:
        return
    oldest = next((k for k, a in _PENDING.items() if a.status in ("executed", "rejected")), None)
    if oldest is None:
        raise TooManyPendingActions(MAX_ACTIONS)
    del _PENDING[oldest]


def request_intro(args: dict, user_id: str, session_id: str) -> ToolEnvelope:
    """Create a PAUSED intro request. Validates first (a bad `reason` creates nothing), then
    stops at the approval gate - nothing reaches the student's inbox until they approve."""
    try:
        req = IntroRequest(**args)
    except ValidationError as exc:
        first = exc.errors()[0]
        field = ".".join(str(x) for x in first["loc"]) or "arguments"
        return ToolEnvelope(success=False, tool="request_intro",
                            error=f"invalid {field}: {first['msg']}")
    action = PendingAction(id=uuid.uuid4().hex[:8], user_id=user_id, session_id=session_id,
                           **req.model_dump())
    with _LOCK:
        try:
            _make_room()
        except TooManyPendingActions as exc:
            return ToolEnvelope(success=False, tool="request_intro", error=str(exc))
        _PENDING[action.id] = action
    logger.info("intro request %s proposed (status=input-required, awaiting the student)",
                action.id)
    return ToolEnvelope(success=True, tool="request_intro",
                        data={"action_id": action.id, "status": action.status,
                              "note": "waiting for the student's approval; nothing is sent yet"})


def _execute(action: PendingAction, approver: str) -> str:
    """The side effect itself - the ONLY place a mutating action runs. Canned in this build:
    a real deploy would email the student here."""
    return f"Intro accepted by {approver}: the student will reply to {action.name}."


def decide(action_id: str, approve: bool, approver: str = "owner") -> PendingAction:
    """Resume a paused action: approve -> execute, reject -> stop. Decided once: a second
    decision raises AlreadyDecided and changes nothing."""
    with _LOCK:
        action = _PENDING.get(action_id)
        if action is None:
            raise KeyError(action_id)
        if action.status in ("executed", "rejected"):
            logger.info("action %s already %s - refused", action_id, action.status)
            raise AlreadyDecided(action.status)
        if not approve:
            action.status = "rejected"
            logger.info("action %s rejected by %s - nothing sent", action_id, approver)
            return action
        action.result = _execute(action, approver)
        action.status = "executed"
        logger.info("action %s approved by %s - executed", action_id, approver)
        return action


def get_action(action_id: str) -> PendingAction | None:
    with _LOCK:
        return _PENDING.get(action_id)


def pending_actions() -> list[PendingAction]:
    """The student's inbox: every action still waiting at the gate, oldest first."""
    with _LOCK:
        return [a for a in _PENDING.values() if a.status == "input-required"]


def reset_actions() -> None:
    """Test/ops helper - clear the pending-action store."""
    with _LOCK:
        _PENDING.clear()
