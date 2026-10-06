# AgentForge™ - Architecture

## Spine: Model · Retrieval · Tool · Memory
Every component sits in one of the four layers the course builds on, plus three cross-cutting
guards: cost, safety (PII, injection, access control) and evaluation. Complexity is a
decision, not a default: one agent with tools until a measured pressure earns a second one.

## Key decisions
1. **Ground or refuse.** Retrieval is gated on the dense score before the model runs, every
   citation is checked by code, and the refusal is one exact string - a wrong answer with a
   fake source is worse than no answer (W5).
2. **Retrieve wide, rerank narrow.** Hybrid dense + BM25 with RRF, HyDE unioned with the raw
   query, a cross-encoder for precision, extractive compression for tokens (W4, W6).
3. **Approval is state.** Agents propose; a paused graph node or a pending-action row waits
   for `/approve`. No sentence a user types can execute a mutating tool (W9, W10, W13).
4. **Route to the cheapest capable model.** A deterministic classifier, a low-confidence
   route-up valve, a $0 exact-match cache and a ledger that prices every call (W16).
5. **Gate on the shipped path.** Eval gates call the same function the route calls, and block
   CI on a drop: deploy gate (W14), prompt regression (W15), model upgrade (W15), routing (W16).
6. **Pin everything.** Dated model ids, versioned prompt files, hashed frozen eval sets.

## Diagram 1 - System architecture
```
users: engineers (web UI, SSE) · incident webhooks · hiring managers / other agents (A2A)
                |
        FastAPI services, each route one thin wire to a seam
  Model ...... provider abstraction · pinned ids · router: nano / mini / gpt-5.4 (W2, W16)
  Retrieval .. KnowledgeVault index (Qdrant) -> CitationRAG + RAGOptimizer answers (W4-W6)
  Tool ....... read tools · MCP server (W11) · mutating tools behind the approval gate
  Memory ..... session (Redis) · episodic + procedural (vector) · per-user audit log (W10)
  Agents ..... OpsAssist loop (W9) · TriageFlow: Triage -> Knowledge -> Action (W10)
  Interop .... A2A Agent Card + task endpoint (W11) · PortfolioAgent front door (W17)
  Guards ..... injection + PII + RBAC/ACL (W12) · cost ceiling + iteration cap (W16)
                |
  Ship ....... CI eval gates -> deploy · OTel traces · job queue + dead letter (W14)
  Measure .... BreakRAG harness (W7) · feedback -> eval cases (W13) · cost report (W16)
```

## Diagram 2 - Retrieval and grounding
```
PDF / runbook -> vision describes figures · tables to rows · 300-token children in   (W4)
                 1,500-token parents -> PII scrubbed -> text-embedding-3-large -> Qdrant  (W4, W12)
question -> (hostile text stripped: query focus) -> HyDE probe (nano) + raw query       (W6)
         -> dense + BM25 -> RRF fuse -> union -> cross-encoder rerank -> compress (keep 80%)  (W4-W6)
         -> gate: top-1 cosine >= 0.55 AND spread >= 0.08 ?  no -> exact refusal string    (W5)
         -> yes -> answer with [chunk-id] citations -> ids + quotes checked by code       (W5)
         -> grounded answer + citation cards        (ACL filter applied in both channels) (W12)
```

## Diagram 3 - Agents, the approval gate and memory
```
incident -> Triage agent (classify) -> Knowledge agent (runbooks, episodic memory)    (W10)
         -> Action agent PROPOSES {tool, args} -> graph pauses before action_execute    (W10)
         -> human: POST /approve {approve: true}  -> execute once (idempotency key)     (W9, W10)
                   POST /approve {approve: false} -> nothing runs                       (W10)
tool calls -> dispatcher: unknown tool / bad args -> error envelope, never a crash    (W9)
loop stops on: end_turn · max_iters · token or cost budget · fatal tool error · timeout ·  (W9)
               consent revoked
memory: session (Redis) | external | episodic (vector) | procedural - scoped per user;  (W10)
        DELETE /user/{id} erases all four; every step lands in the audit log            (W10, W12)
```

## Diagram 4 - Ship, measure and cost
```
pull request -> pytest -> eval gate (golden set, shipped path) -> build -> deploy   (W14)
             -> prompt regression gate (12 frozen) · model upgrade A/B (10 frozen, 0.02)  (W15)
request -> classify -> tier: nano | mini | gpt-5.4 (route up below 0.60 confidence)  (W16)
        -> exact-match cache hit? $0 : priced call -> cost ceiling $0.05 pre-flight -> 413  (W16)
        -> ledger + cost header -> weekly report: actual vs all-frontier             (W16)
failure -> enforcement ladder: retry -> correction -> circuit breaker (half-open ~30 s)  (W15)
traces -> OpenTelemetry spans · background jobs: retry with backoff -> dead letter   (W14)
users -> thumbs-down / correction -> new eval case for the adversarial harness       (W7, W13)
```

## Where the numbers come from
`eval_results.json` lists each week's gate, the size of its frozen set, its threshold, and
the measured result where a run was recorded. The PortfolioAgent's own routing numbers are
produced by `eval_run.py` over `data/eval_golden.jsonl`; prices come from one table in
`config.py`. Nothing is extrapolated from a single run.
