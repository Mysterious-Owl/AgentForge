"""Cost-aware router - classify the question, route to the cheapest capable tier, run the
agent loop on that tier, measure.

The flow, top to bottom:
    classify -> (route-up safety valve) -> pick tier + model
             -> agent loop on that model (the model picks the tools; ceiling before each call)
             -> price every call + the frontier counterfactual -> AskResponse

The TIER is a code decision (deterministic, testable, free); the TOOLS are the model's
decision (function calling). `generate(messages, model, tools, budget)` is injected so tests
and the eval drive the loop with a scripted model and no network; `main.py` passes
`app/llm.py:chat`. Every response carries what it cost AND what the same tokens would have
cost on the frontier tier - the saving is measured per request, never promised.
"""
from __future__ import annotations

import logging

from app import agent, portfolio
from app.agent import GenerateFn
from app.budget import IterationBudget
from app.classifier import classify
from app.config import get_settings
from app.schemas import AskResponse, Classification, QuestionComplexity, Turn

logger = logging.getLogger(__name__)

# Complexity -> (tier label, settings attribute holding the pinned model id).
_TIER_BY_COMPLEXITY: dict[QuestionComplexity, tuple[str, str]] = {
    QuestionComplexity.SIMPLE: ("small", "small_model"),
    QuestionComplexity.FRONTIER: ("frontier", "frontier_model"),
}


def _resolve_complexity(question: str) -> Classification:
    """Classify, then apply the low-confidence route-up safety valve."""
    c = classify(question)
    settings = get_settings()
    if c.confidence < settings.route_up_threshold and c.complexity == QuestionComplexity.SIMPLE:
        logger.warning("route-up: confidence %.2f < %.2f - escalating simple -> frontier",
                       c.confidence, settings.route_up_threshold)
        return Classification(complexity=QuestionComplexity.FRONTIER, confidence=c.confidence,
                              reason=f"routed up from simple ({c.reason}, low confidence)",
                              routed_up=True)
    return c


def resolve_tier(question: str) -> tuple[str, str, Classification]:
    """Decide the tier for a question: (tier_label, pinned_model_id, classification).

    The single source of the routing decision - `route()` uses it to serve a request and
    `eval_run.py` uses the SAME logic to score routing quality offline.
    """
    classification = _resolve_complexity(question)
    tier, model_attr = _TIER_BY_COMPLEXITY[classification.complexity]
    return tier, getattr(get_settings(), model_attr), classification


def route(question: str, generate: GenerateFn, budget: IterationBudget | None = None,
          history: list[Turn] | None = None, user_id: str = "anon",
          session_id: str = "default") -> AskResponse:
    """Route one question to its tier and run the agent loop there.

    THE COST CEILING is enforced inside the loop BEFORE every model call, and the ITERATION
    CAP on every call - an oversized request is refused before a single token is bought.
    The tier is decided on the NEW question only: a cheap follow-up stays cheap.
    """
    settings = get_settings()
    tier, model, classification = resolve_tier(question)
    run = agent.run(question, history or [], portfolio.get_portfolio(), model,
                    settings.frontier_model, generate, budget or IterationBudget(),
                    user_id, session_id)

    if run.pending is not None:
        skill = "request_intro"
    elif run.grounded:
        skill = "capstone_qa"
    else:
        skill = None
    logger.info("route ok: tier=%s model=%s calls=%s grounded=%s cost=$%.6f baseline=$%.6f",
                tier, model, run.calls, run.grounded, run.cost_usd, run.baseline_usd)
    return AskResponse(
        answer=run.answer,
        model=model,
        tier=tier,
        complexity=classification.complexity,
        grounded=run.grounded,
        skill_matched=skill,
        routed_up=classification.routed_up,
        tools_called=run.steps,
        citations=run.citations,
        unverified_citations=run.unverified,
        pending_action=run.pending,
        model_calls=run.calls,
        prompt_tokens=run.prompt_tokens,
        completion_tokens=run.completion_tokens,
        cost_usd=run.cost_usd,
        baseline_cost_usd=run.baseline_usd,
        saved_usd=round(run.baseline_usd - run.cost_usd, 6),
    )
