# AgentForge™ - Capstone AGENTS.md

> **This is the course's reference capstone.** It assembles the sixteen weekly builds of the
> Applied GenAI & Agentic AI Engineering Course into one system, so the PortfolioAgent has a
> real capstone to talk about. Replace `data/` with your own capstone's AGENTS.md, architecture
> diagrams and eval results before you deploy. Every number below comes from a run recorded in
> that week's package; where a week ships a gate but no committed result, it says so.

## What AgentForge is
An engineering-operations copilot for a platform team. It answers questions grounded in
runbooks and design documents with checked citations, triages incoming incidents with a small
team of agents, proposes remediations that wait at a human approval gate, routes every model
call to the cheapest capable tier, and is deployed behind CI with an eval gate, tracing,
reliability guards and a cost report. The PortfolioAgent is its public front door.

## Operating rules (for any agent working on or speaking for AgentForge)
- Every mutating action pauses at a human approval gate; approval is server state, never a
  sentence a user can type. Proposals are free; execution needs an approver.
- Answer only from this context pack; refuse with the `OUT_OF_SCOPE` sentinel rather than guess.
- Model ids are dated pins, never `-latest`; an upgrade is a deliberate, gated change.
- An eval gate scores the shipped code path, not a copy of it, and blocks the deploy on a drop.
- Two caps bound every request: a pre-flight cost ceiling ($0.05) and an iteration cap (8).
- PII is scrubbed at ingestion and from every log line, not only from the answer.
- Report measured numbers only. A gate with no recorded run is described as a gate, not a score.

## Build history - one milestone per week
**W1 · ReleaseBot** - the three patterns every later build reuses: streaming (`POST
/summarize-stream`), structured output validated by Pydantic, and a tool call (`send_email`).
Settings load through pydantic-settings and fail loud on a missing key. Its four-layer baseline
architecture (Model, Retrieval, Tool, Memory) was drawn this week; only the Model and Tool
layers ship in code, with a Postgres fetch and per-recipient Redis memory left as reserved
slots.

**W2 · IntentIQ** - a provider abstraction and a 30-row hand-labelled golden set, benchmarked
across gpt-5.4-mini, gpt-5.4-nano and a local qwen3:0.6b. The README reports a run where nano
scored 93.3% (p95 1,185 ms) against mini's 90.0% (p95 4,665 ms) and qwen3's 66.7%, after an
earlier run tied mini and nano at 90.0%; no output of either run is committed. Four benchmark
runs saved in the API notebook put mini and nano between 86.7% and 93.3% and qwen3 between 66.7%
and 80.0%, with mini/nano p95 between 2.2 and 4.4 s. Lesson kept: measure accuracy and tail
latency together, and expect run-to-run variance.

**W3 · TicketStream** - typed intake: a discriminated-union ticket with a
`Literal` priority, validated BEFORE any field is streamed over SSE, then routed by a
side-effect tool only after validation passes. At most 2 correction retries; SSE over
WebSockets because the flow is one-way.

**W4 · KnowledgeVault** - multimodal ingestion of engineering PDFs: a vision
model forced through function-calling describes figures, tables become rows, prose is chunked
into 300-token children inside 1,500-token parents (only children are embedded). Embeddings
are `text-embedding-3-large` (3,072 dims) in Qdrant; retrieval is dense + BM25 fused with RRF.

**W5 · CitationRAG** - grounded answers with citations. Retrieval is gated before the model is
called (top-1 dense cosine >= 0.55 and a top-1/top-3 spread >= 0.08, else refuse); every
citation id and quote is checked deterministically; one exact refusal string. A five-metric
eval: groundedness, citation precision, citation recall, false-answer rate, false-refusal rate -
0.80 / 1.00 / 0.80 / 0.00 / 0.20 on the 10-row golden set.

**W6 · RAGOptimizer** - HyDE (a nano probe, unioned with the original query's candidates so
drift cannot erase recall), a CPU cross-encoder reranker (BAAI/bge-reranker-base, all 30 pairs
scored in one batch; ~110 ms is a design estimate, not a measured run) and extractive
compression that keeps 80% and never drops a citation marker. `/eval/compare` runs the Week 5
baseline and the full pipeline on the same 10 golden questions.

**W7 · BreakRAG™** - the adversarial eval harness: 10 seeds x 7 = 70 cases from five
deterministic mutations (typo, jailbreak, multi-hop, conflict, hostile), two judges (nano
screens, mini scores) with Krippendorff's alpha, three leakage checks, and a CI workflow. An
earlier red run was withdrawn because the harness, not the system, was broken - the retrieval
gate saw the whole adversarial prompt, the abstention check knew one sentence, and two refusal
probes were mislabelled. After those fixes no CI scorecard is committed, and both saved 70-case
runs are still RED: judge agreement alpha 0.365 and 0.225 against a 0.60 floor, typo 0.55 and
multi-hop 0.60 / 0.40 against 0.70.

**W8 · SpecialistTuner** - LoRA (r=8, alpha=16, q/k/v/o; ~2.3M trainable params, 0.38%) on
Qwen3-0.6B on CPU, with completion-only loss masking; a Vertex AI Gemini tuning variant
alongside. The shipped 4-epoch run overfits on purpose: eval loss 2.27 -> 2.21 (best, epoch 2)
-> 2.24 -> 2.30 while train loss falls 2.86 -> ~1.5, so the best checkpoint is kept. On the same
42-question set the tuned Gemini scored 29/42 and the LoRA 13/42 (final checkpoint); no
base-model accuracy run is on record, so the rule - fine-tune only where it beats base
prompting - is stated, not yet demonstrated.

**W9 · OpsAssist** - a raw-Python tool-using ops agent (about 1,200 lines, no framework):
`get_runbook`, `query_metrics`, `propose_remediation` (with an idempotency key); `Literal` tool
arguments; a dispatcher that returns an error envelope instead of raising; seven stop conditions
and `max_iters` 10. Its offline eval is built so 3/5 scenarios pass (two are built to fail); no
run of it is recorded.

**W10 · TriageFlow™** - a LangGraph team of three agents (Triage, Knowledge,
Action). The Action agent only proposes; `interrupt_before` pauses the graph at
`action_execute` until `/approve` resumes it (409 if nothing is paused). Four memory layers:
session memory on Redis is the one real backend; external, episodic and procedural are
production-shaped stubs. Runbook and episodic search are real vector search (in-process, or
pgvector); `DELETE /user/{id}` erases a user from all four layers.

**W11 · AgentMesh™ + CohortMCP** - an MCP tool server with
structured `{code, message, retryable}` errors and an 8-case tool-pick eval built to show a vague
tool description lowering accuracy; and an A2A remote agent: an Agent Card with `skills[]` for
work and `capabilities` for protocol flags, task streaming over SSE with `Last-Event-ID`
replay, a swappable task store (memory or SQLite) and JWT scopes.

**W12 · GuardianAI™** - hardening: a regex injection classifier plus a
retrieval sanitiser, Presidio PII scrubbing at ingestion with a visible `<REDACTED>` (redact
at >= 0.65 confidence, human review 0.35-0.65, phone matches re-scored to 0.85), RBAC with a
per-document ACL pushed into both retrieval channels, and an append-only audit log. Its
12-row eval scores the live pipeline and requires zero PII leaks.

**W13 · WorkbenchAI™** - a streaming operator UI: typed SSE frames for tokens,
citations, tool traces and approval prompts. The model chooses its tools; a side-effecting
`restart_service` is parked as a pending action that only `/approve` executes. A thumbs-down or
a correction becomes a BreakRAG-shaped eval case.

**W14 · DeployCore** - CI runs test -> eval-gate -> build ->
deploy. The gate scores the shipped answer path on a 5-case golden set: prompt `answer-v4`
passes 1.00 (5/5) against a 0.80 threshold, and `answer-v3` falls to 0.00 and blocks the
deploy. Prompts carry their version in the filename; a pin guard rejects `-latest` model ids.
Background jobs get retries with backoff, idempotency, a dead-letter queue and OpenTelemetry
spans.

**W15 · ReliabilityKit™** - prompts versioned as files with a regression gate (12 frozen cases,
threshold 0.8, promotion stays a human decision); a deterministic hallucination-debugging tree;
a Pydantic enforcement ladder (strict parse -> JSON salvage -> up to 2 correction prompts -> a
process-level circuit breaker that trips after 3 consecutive failures and half-opens after
30 s); and a model-upgrade A/B on 10 frozen cases that exits non-zero on a drop beyond 0.02.

**W16 · CostGuard™** - three routing tiers priced per 1M tokens: gpt-5.4-nano $0.20 in / $1.25
out, gpt-5.4-mini $0.75 / $4.50, gpt-5.4 $2.50 / $15.00 (cached input is a tenth). A
deterministic classifier with a 0.60 route-up valve, an exact-match cache that costs $0, a
per-request cost meter and a weekly report. Its routing gate (a stub model, token counts from
prompt length): 10/10 correct, $0.004559 routed vs $0.010722 all-frontier - 57.5% saved, about
2.35x cheaper. Plus an event-driven SRE bot: webhook -> diagnosis -> proposal -> approval gate
-> execution, with fingerprint dedupe.

**W17 · Demo Day + PortfolioAgent** - a red-team pass over five surfaces (adversarial inputs,
PII edge cases, tool misuse, memory boundaries, cost spikes), the architecture walkthrough,
and this PortfolioAgent.

## The PortfolioAgent (Week 17) - what answers you
A FastAPI service over this pack, built as a small agent. A deterministic classifier picks the
model tier: short lookups go to the small tier (gpt-5.4-nano, $0.20 / $1.25 per 1M), reasoning
questions to the frontier tier (gpt-5.4-mini, $0.75 / $4.50) - a 3.75x input-price gap. It is
two tiers, not CostGuard's three: it uses CostGuard's small and medium tiers, because answering
from a fixed pack never needs gpt-5.4. The chosen model then decides, by function calling,
which tools to call: get_build(week), get_week_details(week), get_eval_results(week) and
get_architecture(section or week) read this pack, and every result carries a source the answer
must cite - code checks each citation against what the tools returned. A fifth tool,
request_intro, lets a visitor ask to reach the student; it pauses at an approval gate until the
student decides.

**Why a small model for routing - and the open-model slot.** Most questions are short lookups
that a small model answers as well as a large one - in Week 2's saved runs nano scored within a
row or two of mini, 86.7-93.3% each - so paying frontier prices for them buys nothing. The small
tier is also the open-model slot: set `SMALL_BASE_URL` and `SMALL_MODEL` to run it on a local
open model (Qwen through Ollama) with no per-token bill and no data leaving the machine, while
the frontier tier stays on the vendor. It ships on nano because across Week 2's four saved runs
on the 30-row golden set nano scored 86.7-93.3% and the local qwen3:0.6b 66.7-80.0% - a local
model earns the slot by matching nano on the tool-choice gate: picking the right tool and citing
it. The routing gate cannot judge that - it checks which tier each question lands in, and never
calls a model. The routing decision itself is deterministic code, not a model call:
reproducible, free, and testable offline.

**What else it carries.** Two eval gates: its routing gate replays 27 frozen questions
offline, and its tool-choice gate replays 14 against the live model - did it pick the right
tool, and cite it (see `eval_results.json`). It keeps the last three turns of a conversation
in the visitor's browser, never on the server. It speaks A2A 1.0 (Agent Card plus JSON-RPC
`SendMessage` at `/a2a`), keeps a per-user audit log with PII scrubbed, and on its public URL
has an admin token, a rate limit and a daily spend budget. It is deployed on Render from
GitHub.
