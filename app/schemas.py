"""Pydantic models - the contracts on the wire and the internal data shapes.

Layers represented here:
  - Model / routing: QuestionComplexity, Classification, the Ask request/response (with the
    short chat history the browser sends), ModelTurn (one model call: text or tool calls).
  - Tool: ToolCall, ToolEnvelope (every tool's result, with the source it can be cited by),
    ToolStep (the trace the UI shows), IntroRequest (the mutating tool's arguments, with the
    `reason` enum) and PendingAction (the approval gate's paused state).
  - Memory: AuditEntry.
  - A2A: the Agent Card wire models served at /.well-known/agent-card.json.

The A2A split people get wrong, and that this file gets right:

    skills[]      = what the agent can DO   (id, name, description, tags, examples,
                                             inputModes, outputModes)
    capabilities  = PROTOCOL FLAGS ONLY     (streaming, pushNotifications,
                                             extendedAgentCard)
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Field caps - a public URL takes text from strangers, and every string here is kept in
# memory (the pending actions, the audit log). The question has no cap of its own: the
# body-size limit bounds it, and the cost ceiling is what refuses an enormous one.
ID_MAX = 128               # user_id, session_id, memory keys, action ids
MEMORY_VALUE_MAX = 4_000   # one remembered value
HISTORY_TURNS = 3          # earlier turns the browser may send with a question
HISTORY_TEXT_MAX = 2_000   # one earlier question or answer


# ---------- Model / routing ----------

class QuestionComplexity(str, Enum):
    """Complexity IS the routing key - each value maps to a different model tier."""

    SIMPLE = "simple"       # cheap factual lookup -> the small tier
    FRONTIER = "frontier"   # needs reasoning -> the frontier tier


class ToolCall(BaseModel):
    """One function call the model asked for - `arguments` is the model's raw JSON string."""

    id: str
    name: str
    arguments: str = "{}"


class ModelTurn(BaseModel):
    """What one model call returns: tool calls to run, or the final text - plus the token
    usage that call is priced from."""

    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Classification(BaseModel):
    complexity: QuestionComplexity
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str                       # the plain-English trigger the classifier fired on
    routed_up: bool = False           # did the low-confidence safety valve escalate the tier?


# ---------- ask / answer ----------

def _no_lone_surrogates(v):
    """Half an emoji (a lone UTF-16 surrogate, e.g. from text cut mid-character) is valid JSON
    but cannot be encoded for the provider - it would fail every later call. Replace it."""
    return v.encode("utf-8", "replace").decode("utf-8") if isinstance(v, str) else v


class Turn(BaseModel):
    """One earlier exchange, sent back by the browser so a follow-up ("why did you choose
    that?") has something to refer to. The server stores none of it."""

    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(..., min_length=1, max_length=HISTORY_TEXT_MAX)
    answer: str = Field(..., min_length=1, max_length=HISTORY_TEXT_MAX)

    @field_validator("question", "answer", mode="before")
    @classmethod
    def _whole_characters(cls, v):
        return _no_lone_surrogates(v)


class AskRequest(BaseModel):
    """POST /ask body. `min_length=8` rejects a too-short question ("hi") before any spend -
    counted after surrounding whitespace is stripped, so eight spaces are not a question."""

    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(..., min_length=8, description="A question about the capstone.")
    # Short history, kept in the BROWSER: at most the last 3 turns, each capped. It is client
    # text like the question - it can say anything, so it is data, never instructions.
    history: list[Turn] = Field(default_factory=list, max_length=HISTORY_TURNS)
    # Memory scoping. A question carries WHO is asking and WHICH session, so the audit log
    # and any remembered state are keyed per-user and per-session, never shared across users.
    user_id: str = Field("anon", max_length=ID_MAX)
    session_id: str = Field("default", max_length=ID_MAX)

    @field_validator("question", mode="before")
    @classmethod
    def _whole_characters(cls, v):
        return _no_lone_surrogates(v)


class ToolStep(BaseModel):
    """One step of the trace: the tool the model chose, its arguments, and how it went -
    the chip under every answer."""

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    success: bool
    source: str | None = None         # what an answer may cite this result as
    error: str | None = None
    # What the tool pulled, as text - the retrieved context the UI shows under the answer.
    content: str | None = None


class PendingAction(BaseModel):
    """A proposed intro request, paused at the approval gate.

    Created at `input-required` by the `request_intro` tool and does NOTHING until the student
    decides it. Approval is a state the system owns (this row's status), not a sentence a
    visitor - or the model - can type.
    """

    id: str
    kind: Literal["request_intro"] = "request_intro"
    name: str
    company: str = ""
    contact: str                      # PII: kept for the student, scrubbed from logs and audit
    reason: str
    message: str
    # "input-required" (at the gate) -> "executed" (approved) | "rejected" (declined)
    status: str = "input-required"
    result: str | None = None
    user_id: str = "anon"
    session_id: str = "default"       # where it was proposed - the decision is audited there too


class AskResponse(BaseModel):
    answer: str
    model: str                        # the exact pinned model id that ran the loop
    tier: str                         # "small" | "frontier"
    complexity: QuestionComplexity
    grounded: bool                    # cited at least one source, and every citation checked out
    skill_matched: str | None = None
    routed_up: bool = False
    # The agent's trace: the tools the MODEL chose, in order, and what the answer cites.
    tools_called: list[ToolStep] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)              # verified by code
    unverified_citations: list[str] = Field(default_factory=list)   # cited, never returned
    pending_action: PendingAction | None = None                     # request_intro, at the gate
    model_calls: int = 0              # iterations of the loop this answer used (cap: 8)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0             # what the whole loop actually cost
    baseline_cost_usd: float = 0.0   # the same tokens priced on the frontier tier
    saved_usd: float = 0.0           # baseline_cost_usd - cost_usd (0 when it took the frontier)


# ---------- Tool layer: the intro request + the approval gate ----------

class IntroRequest(BaseModel):
    """The arguments of the mutating tool `request_intro`, validated BEFORE anything happens.

    `reason` is a Literal enum: a bad argument from the model (e.g. "urgent") goes back to it
    as a tool error and nothing is created - the Tool-layer probe from the red-team pass.
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    name: str = Field(..., min_length=1, max_length=80)
    company: str = Field("", max_length=80)
    contact: str = Field(..., min_length=3, max_length=120)
    reason: Literal["hiring", "collaboration", "feedback", "other"]
    message: str = Field(..., min_length=1, max_length=500)


class ApprovalDecision(BaseModel):
    """POST /approve body - the student's YES/NO that resumes a paused action."""

    action_id: str = Field(..., max_length=ID_MAX)
    approve: bool
    approver: str = Field("owner", max_length=ID_MAX)


# ---------- Tool dispatcher envelope ----------

class ToolEnvelope(BaseModel):
    """The uniform result shape every tool returns - success or a structured error.

    An unknown tool name, bad arguments or a failing tool return `{success: false, error}` to
    the model; it never raises and never crashes the run. `source` is the tag an answer may
    cite this result by - the citation check accepts nothing else.
    """

    success: bool
    tool: str
    data: dict | list | None = None
    error: str | None = None
    source: str | None = None


# ---------- Memory layer ----------

class AuditEntry(BaseModel):
    """One line in a user's audit log - what was asked/done, scoped to that user."""

    user_id: str
    session_id: str
    kind: str                         # "ask" | "intro:proposed" | "intro:executed" | ...
    detail: str


# ---------- A2A Agent Card (the discovery document) ----------

class AgentSkill(BaseModel):
    """One advertised skill on the Agent Card - what the agent can DO."""

    id: str
    name: str
    description: str
    tags: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    # inputModes / outputModes are MEDIA TYPES, not JSON schemas - the A2A field people
    # most often misread. This skill takes a plain-text question and returns plain prose.
    input_modes: list[str] = Field(
        default_factory=lambda: ["text/plain"], serialization_alias="inputModes"
    )
    output_modes: list[str] = Field(
        default_factory=lambda: ["text/plain"], serialization_alias="outputModes"
    )


class AgentCapabilities(BaseModel):
    """A2A capabilities block - PROTOCOL FLAGS ONLY, never a skill list."""

    streaming: bool = False
    push_notifications: bool = Field(default=False, serialization_alias="pushNotifications")
    extended_agent_card: bool = Field(default=False, serialization_alias="extendedAgentCard")


class AgentInterface(BaseModel):
    """One way to CALL the agent - A2A 1.0 lists these instead of a single top-level url."""

    url: str
    protocol_binding: str = Field(default="JSONRPC", serialization_alias="protocolBinding")
    protocol_version: str = Field(default="1.0", serialization_alias="protocolVersion")


class AgentCardSignature(BaseModel):
    """A2A 1.0 card signature: a detached JWS - base64url header, signature, optional extras."""

    protected: str
    signature: str
    header: dict[str, Any] | None = None


class AgentCard(BaseModel):
    """The JSON served at /.well-known/agent-card.json - the A2A 1.0 discovery contract.

    Field names serialize in the spec's camelCase (supportedInterfaces, defaultInputModes, ...)
    so a real A2A client can consume the card as-is.
    """

    name: str
    description: str
    # WHERE and HOW to call the agent. The protocol version lives on each interface - it is
    # the version that endpoint speaks, which is what a client has to match.
    supported_interfaces: list[AgentInterface] = Field(serialization_alias="supportedInterfaces")
    version: str = "1.0.0"          # the AGENT's version, not the protocol's
    documentation_url: str | None = Field(default=None, serialization_alias="documentationUrl")
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    # Media types again - the card-level default a skill inherits when it declares none.
    default_input_modes: list[str] = Field(
        default_factory=lambda: ["text/plain"], serialization_alias="defaultInputModes"
    )
    default_output_modes: list[str] = Field(
        default_factory=lambda: ["text/plain"], serialization_alias="defaultOutputModes"
    )
    skills: list[AgentSkill] = Field(default_factory=list)

    # A2A 1.0 lets a publisher sign the card with JWS so a client can verify who published it.
    # app/signing.py fills it when CARD_SIGNING_SEED is set; without a seed it stays None and
    # is left out of the JSON - an unsigned card claims no signature.
    signatures: list[AgentCardSignature] | None = None
