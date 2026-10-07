"""FastAPI routes for the PortfolioAgent - an agent over the student's capstone, wired thin.

A2A surface (discovery, then the call itself):
  GET  /.well-known/agent-card.json  -> the Agent Card, signed (+ /.well-known/agent.json, legacy)
  GET  /.well-known/jwks.json        -> the public key that verifies the card's signature
  POST /a2a                          -> A2A 1.0 JSON-RPC: SendMessage answers exactly like /ask

UI contract:
  GET  /                        -> serve browser UI (index.html)
  GET  /admin                   -> the owner's page: token, intro inbox, audit log (admin.html)
  GET  /health                  -> liveness + both model tiers, both caps, and the price table
  GET  /readme                  -> render README.md as dark-themed HTML
  GET  /portfolio               -> the build cards, parsed from data/AGENTS.md (no model call)
  GET  /portfolio/{week}        -> one build's read page: results, decisions, diagrams (HTML)

The agent, layer by layer:
  POST /ask                     -> the agent: routed to a tier, the MODEL picks the tools
                                   (get_build / get_eval_results / get_architecture, or the
                                   gated request_intro), the answer cites what they returned
  GET  /actions           [admin] -> Tool: the student's inbox - intro requests at the gate
  POST /approve           [admin] -> Tool: the student's YES/NO that resumes a paused action
  GET  /audit/{user_id}   [admin] -> Memory: one user's audit log (never another user's)
  POST /memory            [admin] -> Memory: remember one value, scoped to (user, session)
  GET  /memory            [admin] -> Memory: recall one value - only that user's own session
  DELETE /memory          [admin] -> Memory: hard-delete one remembered value

`A2A-Version` is on every response; app/budget.py caps ONE request (413, 429) and app/guard.py
caps the DAY, rate-limits POSTs, limits the body size and checks the [admin] token; PII is
scrubbed from the logs and the audit trail.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from app import a2a, guard, llm, memory, portfolio, signing, tools
from app.budget import (
    BudgetExceeded,
    IterationBudget,
    IterationCapExceeded,
)
from app.config import get_settings
from app.context import ContextPackError, get_context
from app.router import route
from app.schemas import (
    ID_MAX,
    MEMORY_VALUE_MAX,
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    AskRequest,
    AskResponse,
    ApprovalDecision,
    AuditEntry,
    PendingAction,
)
from app.scrub import install_pii_scrubbing, scrub

logging.basicConfig(
    level="INFO",
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("portfolioagent")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Fail loudly at boot, not on the first request.

    Loading settings here means a missing env var stops the server before it accepts
    traffic; installing the PII scrubber here means no log line is ever emitted un-scrubbed;
    probing the context pack here turns a bad deploy into a boot failure.
    """
    settings = get_settings()
    logging.getLogger().setLevel(settings.log_level)
    install_pii_scrubbing()   # scrub every log line, whichever module writes it
    get_context()             # probe: raises ContextPackError -> the boot fails loudly
    logger.info("startup complete: config loaded, context pack loaded, small=%s frontier=%s",
                settings.small_model, settings.frontier_model)
    yield


app = FastAPI(title="PortfolioAgent", version="0.1.0", lifespan=lifespan)

# Middleware runs outermost-LAST-added. The two guards go on first, so they sit INSIDE CORS
# and the A2A-Version header: a 429 or a 413 still carries both, and a browser client sees
# the real refusal rather than a CORS error.
app.middleware("http")(guard.rate_limit)        # POSTs per client per minute -> 429
app.middleware("http")(guard.limit_body_size)   # body over the limit -> 413, no length -> 411
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def a2a_version_header(request: Request, call_next):
    """Stamp `A2A-Version` on EVERY response. Emitted, never negotiated."""
    response = await call_next(request)
    response.headers["A2A-Version"] = get_settings().a2a_protocol_version
    return response


# ── A2A Agent Card ───────────────────────────────────────────────────────────────

def build_agent_card() -> AgentCard:
    """Build the A2A Agent Card from the capstone's identity."""
    s = get_settings()
    return AgentCard(
        supported_interfaces=[{"url": f"{s.agent_base_url}/a2a",
                               "protocol_version": s.a2a_protocol_version}],
        description=(
            "Answers questions about a student's capstone - what each week built, its results "
            "and its design decisions - by calling tools over the project's AGENTS.md, "
            "architecture and eval results, and cites the source of every fact. Routes cheap "
            "questions to a small model and escalates reasoning to a frontier model; an intro "
            "request pauses for the student's approval."
        ),
        name=s.agent_name, documentation_url=f"{s.agent_base_url}/readme",
        # capabilities = PROTOCOL FLAGS ONLY. Never a list of what the agent can do.
        capabilities=AgentCapabilities(
            streaming=False,
            push_notifications=False,
            extended_agent_card=False,
        ),
        # skills[] = what the agent can DO.
        skills=[
            AgentSkill(
                id="capstone_qa",
                name="Capstone Q&A",
                description=(
                    "Answer a question about what the capstone built, week by week, its eval "
                    "results or its design - looked up with tools, every fact cited."
                ),
                tags=["capstone", "portfolio", "eval", "architecture"],
                examples=[
                    "What did you build in Week 4?",
                    "Why does the small tier run on nano instead of a local model?",
                ],
            ),
            AgentSkill(
                id="request_intro",
                name="Request an intro",
                description=(
                    "Ask the student to get in touch (hiring, collaboration, feedback). The "
                    "request pauses at an approval gate; nothing reaches the student's inbox "
                    "until they approve it."
                ),
                tags=["tool", "approval-gate", "contact"],
                examples=["I'm hiring for an AI engineer role - please reach me at "
                          "jane@example.com"],
            ),
        ],
    )


def card_document() -> dict[str, Any]:
    """The card as served: its JSON, plus `signatures` - a detached JWS over exactly that JSON -
    when this deployment has a signing seed (app/signing.py). Unsigned, it carries no
    `signatures` at all."""
    card = build_agent_card().model_dump(mode="json", by_alias=True, exclude_none=True)
    signatures = signing.sign_card(card, get_settings().agent_base_url)
    return {**card, "signatures": signatures} if signatures else card


def _card_json() -> Response:
    import json
    return Response(content=json.dumps(card_document(), ensure_ascii=False),
                    media_type="application/json")


@app.get("/.well-known/agent-card.json", include_in_schema=False)
def agent_card() -> Response:
    """A2A discovery surface at the canonical (>=0.3) well-known path. No auth."""
    return _card_json()


@app.get("/.well-known/agent.json", include_in_schema=False)
def agent_card_legacy() -> Response:
    """Legacy pre-0.3 discovery path - kept as an alias for older clients."""
    return _card_json()


@app.get(signing.JWKS_PATH, include_in_schema=False)
def card_keys() -> dict[str, Any]:
    """The JWK Set a caller verifies the card's signature with - the `jku` in its header."""
    return signing.jwks()


# ── UI / health / readme ─────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness + both tiers, both caps, and the price table. A cap you cannot read is a cap
    nobody trusts - and neither is a price you cannot see."""
    s = get_settings()
    return {
        "status": "ok",
        "small_model": s.small_model,
        "frontier_model": s.frontier_model,
        "max_iterations": s.max_iterations,
        "cost_ceiling_usd": s.cost_ceiling_usd,
        # Lets the UI label the owner-only buttons on a deployment instead of failing on them.
        "admin_token_required": bool(s.admin_token),
        "card_signed": bool(s.card_signing_seed),
        "pricing_per_1m": {
            "small": {"input": s.small_input_cost_per_1m, "output": s.small_output_cost_per_1m},
            "frontier": {"input": s.frontier_input_cost_per_1m,
                         "output": s.frontier_output_cost_per_1m},
        },
    }


@app.get("/", include_in_schema=False)
def serve_ui():
    idx = Path(__file__).parent.parent / "index.html"
    if not idx.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(idx, media_type="text/html")


@app.get("/admin", include_in_schema=False)
def serve_admin():
    """The owner's page. Serving it is harmless: everything it SHOWS comes from [admin] routes,
    which need the bearer token on a deployment - the page only holds the token box."""
    page = Path(__file__).parent.parent / "admin.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="admin.html not found")
    return FileResponse(page, media_type="text/html")


@app.get("/portfolio")
def read_portfolio() -> dict[str, Any]:
    """The build cards the page opens on - one per week, from the pack's own AGENTS.md, so a
    visitor with no time can scroll what was built - plus the two eval gates' pass rates.
    Text parsing only: free, no model call."""
    try:
        return {**portfolio.get_portfolio(), "gates": portfolio.gates(),
                "profile": portfolio.profile()}
    except ContextPackError as exc:
        raise HTTPException(status_code=503, detail=f"context pack not loaded: {exc}")


@app.get("/portfolio/{week}", include_in_schema=False)
def read_build(week: int) -> Response:
    """One build to read in its own tab - what was built, its results, the decisions that cite
    it and its lines in the diagrams, all from the pack. HTML, every pack string escaped."""
    try:
        detail = portfolio.build_detail(week)
    except ContextPackError as exc:
        raise HTTPException(status_code=503, detail=f"context pack not loaded: {exc}")
    if detail is None:
        raise HTTPException(status_code=404, detail=f"no Week {week} in the build history")
    return Response(content=portfolio.render_build_page(detail),
                    media_type="text/html; charset=utf-8")


@app.get("/readme", include_in_schema=False)
def serve_readme() -> Response:
    try:
        import markdown as _md
    except ImportError:
        raise HTTPException(status_code=503, detail="pip install markdown==3.7")
    readme = Path(__file__).parent.parent / "README.md"
    if not readme.exists():
        raise HTTPException(status_code=404, detail="README.md not found")
    body = _md.markdown(
        readme.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "toc"],
    )
    page = (
        "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>"
        "<title>PortfolioAgent - README</title>"
        "<style>body{background:#0d1117;color:#e6edf3;"
        "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
        "max-width:900px;margin:40px auto;padding:0 24px;line-height:1.7}"
        "h1,h2,h3{color:#58a6ff;border-bottom:1px solid #30363d;padding-bottom:6px}"
        "code{background:#21262d;padding:2px 6px;border-radius:4px}"
        "pre{background:#161b22;padding:16px;border-radius:8px;overflow-x:auto}"
        "pre code{background:none;padding:0}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #30363d;padding:8px 12px;text-align:left}"
        "th{background:#161b22;color:#58a6ff}"
        "a{color:#58a6ff}</style></head><body>" + body + "</body></html>"
    )
    return Response(content=page, media_type="text/html; charset=utf-8")


# ── Model + Cost: the routed, grounded answer ────────────────────────────────────

@app.post("/ask", response_model=AskResponse, dependencies=[Depends(guard.require_budget)])
def ask(req: AskRequest) -> AskResponse:
    """Answer one question about the capstone: routed to the cheapest capable tier, where the
    model calls the tools it needs and cites what they returned.

    Guard order, cheapest first:
      1. min_length=8, history caps -> 422 (Pydantic) before this function is even entered.
      2. context pack   -> 503 if a pack file is missing (fail-loud, never a half-answer).
      3. COST CEILING   -> 413 before EVERY model call of the loop (spent so far + worst case).
      4. ITERATION CAP  -> 429 if the loop asks for a 9th model call.
      5. provider error -> 502, structured, never a stack trace.
    """
    budget = IterationBudget()
    try:
        get_context()            # all three pack files present, or a 503 before any spend
        resp = route(req.question, generate=guard.metered(llm.chat), budget=budget,
                     history=req.history, user_id=req.user_id, session_id=req.session_id)
    except ContextPackError as exc:
        logger.error("Context pack unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=f"context pack not loaded: {exc}")
    except BudgetExceeded as exc:
        logger.warning("refused oversized request: %s", exc)
        raise HTTPException(
            status_code=413,
            detail={
                "error": "cost_ceiling_exceeded",
                "projected_usd": exc.projected,
                "ceiling_usd": exc.ceiling,
            },
        )
    except IterationCapExceeded as exc:
        logger.error("iteration cap exceeded")
        raise HTTPException(
            status_code=429,
            detail={"error": "iteration_cap_exceeded", "max_iterations": exc.limit},
        )
    except Exception as exc:  # provider/key/network error - structured, never a stack trace
        logger.exception("model call failed")
        # The provider echoes the prompt back in some error strings, so this detail is
        # an egress path like any other. The log is scrubbed by the filter; the client
        # response has to be scrubbed here, or the leak just changes direction.
        raise HTTPException(status_code=502, detail=scrub(f"model call failed: {exc}"))

    trace = " -> ".join(f"{s.tool}({','.join(map(str, s.args.values()))})"
                        for s in resp.tools_called if s.tool != "request_intro")
    memory.log_audit(AuditEntry(       # log_audit scrubs every detail
        user_id=req.user_id, session_id=req.session_id, kind="ask",
        detail=f"[{resp.tier}] {req.question[:80]}" + (f" | {trace}" if trace else ""),
    ))
    if resp.pending_action is not None:
        a = resp.pending_action
        memory.log_audit(AuditEntry(
            user_id=req.user_id, session_id=req.session_id, kind="intro:proposed",
            detail=f"{a.id} from {a.name} ({a.reason}) - {a.contact}",
        ))
    return resp


# ── Tool layer: the approval gate ────────────────────────────────────────────────

@app.get("/actions", response_model=list[PendingAction],
         dependencies=[Depends(guard.require_admin)])
def list_actions() -> list[PendingAction]:
    """The student's inbox: every intro request still waiting at the gate. The agent creates
    them (request_intro); only the token holder sees and decides them."""
    return tools.pending_actions()


@app.post("/approve", response_model=PendingAction, dependencies=[Depends(guard.require_admin)])
def approve(decision: ApprovalDecision) -> PendingAction:
    """The student's YES/NO that resumes a paused action. Approve -> execute, reject -> nothing.

    A decision is made once: deciding it again is a 409, and nothing is audited for it - the
    audit trail records decisions that happened, not clicks."""
    try:
        action = tools.decide(decision.action_id, decision.approve, decision.approver)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no pending action '{decision.action_id}'")
    except tools.AlreadyDecided as exc:
        raise HTTPException(status_code=409, detail={"error": "already_decided",
                                                     "status": exc.status})
    memory.log_audit(AuditEntry(
        user_id=action.user_id, session_id=action.session_id,
        kind=f"intro:{action.status}", detail=f"{action.id} by {decision.approver}",
    ))
    return action


# ── Memory layer ─────────────────────────────────────────────────────────────────

@app.get("/audit/{user_id:path}", response_model=list[AuditEntry],
         dependencies=[Depends(guard.require_admin)])
def audit(user_id: str) -> list[AuditEntry]:
    """One user's audit log - scoped, never another user's."""
    return memory.audit_log(user_id)


class MemoryWriteRequest(BaseModel):
    """POST /memory body - the (user, session, key) triple memory.py is keyed by, plus a value."""

    user_id: str = Field("anon", max_length=ID_MAX)
    session_id: str = Field("default", max_length=ID_MAX)
    key: str = Field(..., min_length=1, max_length=ID_MAX)
    value: str = Field(..., min_length=1, max_length=MEMORY_VALUE_MAX)


class MemoryDeleteRequest(BaseModel):
    user_id: str = Field("anon", max_length=ID_MAX)
    session_id: str = Field("default", max_length=ID_MAX)
    key: str = Field(..., min_length=1, max_length=ID_MAX)


@app.post("/memory", dependencies=[Depends(guard.require_admin)])
def write_memory(req: MemoryWriteRequest) -> dict[str, Any]:
    """Remember one value, scoped to (user, session).

    The write half of the Memory layer, on the wire - without it the boundary probes the
    red-team pass runs (cross-session, cross-user, delete-then-query) are unreachable from
    outside the test suite. The audit line records the key and the size, never the value.
    """
    memory.remember(req.user_id, req.session_id, req.key, req.value)
    memory.log_audit(AuditEntry(
        user_id=req.user_id, session_id=req.session_id, kind="memory:wrote",
        detail=f"{req.key} ({len(req.value)} chars)",
    ))
    return {"stored": True, "user_id": req.user_id, "session_id": req.session_id,
            "key": req.key}


@app.get("/memory", dependencies=[Depends(guard.require_admin)])
def read_memory(key: str = Query(..., max_length=ID_MAX),
                user_id: str = Query("anon", max_length=ID_MAX),
                session_id: str = Query("default", max_length=ID_MAX)) -> dict[str, Any]:
    """Recall one value - ONLY from the asking user's own session.

    A miss is a 200 with `value: null`, not a 404: the boundary is enforced by the store's
    shape, so "another user's key" and "no such key" are the same answer, and the response
    leaks nothing about what exists in a bucket the caller cannot reach.
    """
    value = memory.recall(user_id, session_id, key)
    return {"found": value is not None, "value": value,
            "user_id": user_id, "session_id": session_id, "key": key}


@app.delete("/memory", dependencies=[Depends(guard.require_admin)])
def delete_memory(req: MemoryDeleteRequest) -> dict[str, Any]:
    """Hard-delete one remembered value. The value is gone; the audit entry is retained."""
    deleted = memory.delete(req.user_id, req.session_id, req.key)
    return {"deleted": deleted, "key": req.key}


# ── A2A 1.0 over JSON-RPC, and the public-deploy guards ─────────────────────────

from fastapi.concurrency import run_in_threadpool  # noqa: E402 - used only by /a2a below


def _a2a_answer(req: AskRequest) -> AskResponse:
    """SendMessage IS /ask: same loop, same ceiling, same audit line - and the daily budget
    that /ask's dependency enforces is checked here, because this call skips the decorator."""
    guard.require_budget()
    return ask(req)


@app.post("/a2a")
async def a2a_rpc(request: Request) -> dict[str, Any]:
    """A2A 1.0 JSON-RPC - the `url` the Agent Card's supportedInterfaces advertises.

    HTTP 200 with a JSON-RPC body - success in `result`, failure in `error` - unless the
    rate limiter has already answered 429 for this client.
    """
    try:
        payload = await request.json()
    except ValueError:
        return a2a.error_response(None, a2a.RpcError(a2a.PARSE_ERROR, "invalid JSON"))
    return await run_in_threadpool(
        a2a.handle, payload, request.headers.get("A2A-Version"),
        get_settings().a2a_protocol_version, _a2a_answer)
