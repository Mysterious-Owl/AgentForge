"""Question-complexity classifier - the routing key, as deterministic code.

Routing must be reproducible: the same question must always land in the same tier, or the
cost story is noise. So the first pass is pure Python - cheap signals (length, reasoning
cues), no LLM, no network. Cheap lookups ("what model do you use?", "how does the memory
layer work?") stay on the small tier; anything that needs real reasoning ("why did you
choose X over Y?", "compare...", "walk me through the trade-off") escalates to the frontier.

The safety valve lives in the router: when confidence is low, escalate rather than risk a
weak answer on the small tier. A wrong cheap answer is more expensive than a right dear one.
"""
from __future__ import annotations

import logging
import re

from app.config import get_settings
from app.schemas import Classification, QuestionComplexity

logger = logging.getLogger(__name__)

# Reasoning cues -> the question needs analysis / comparison / judgement -> FRONTIER.
# Matched as WHOLE words, in the forms people actually type: a cue inside another word
# ("reason" in "reasonable", "architect" in "architecture") is not a cue. Nouns this
# capstone is ABOUT - design, architecture, decision - are not cues either: "tell me about
# your architecture" is a lookup.
_REASON_KW = (
    "why", "compare", "compared", "compares", "comparing", "comparison",
    "contrast", "contrasts", "contrasting", "trade-off", "trade-offs", "tradeoff",
    "tradeoffs", "analyze", "analyse", "analyzing", "analysing", "evaluate", "evaluating",
    "justify", "justified", "justifying", "assess", "assessing", "critique",
    "alternative", "alternatives", "walk me through", "how would you", "what if",
    "pros and cons", "deep dive", "root cause",
)
_REASON_RE = re.compile(r"\b(?:" + "|".join(map(re.escape, _REASON_KW)) + r")\b")


def classify(question: str) -> Classification:
    """Return the tier for a question, plus the confidence and the trigger that fired.

    Priority order (most decisive signal first):
      1. FRONTIER - a reasoning cue is present.
      2. SIMPLE   - short, no reasoning cue (a cheap lookup the small tier answers well).
      3. SIMPLE (low confidence) - longer but no cue; the router's safety valve may escalate.
    """
    settings = get_settings()
    t = question.lower()
    n = len(question)

    has_reason = _REASON_RE.search(t) is not None

    # 1. FRONTIER - an explicit reasoning cue.
    if has_reason:
        logger.info("classify -> frontier (reasoning cue)")
        return Classification(complexity=QuestionComplexity.FRONTIER, confidence=0.9,
                              reason="reasoning cue")

    # 2. SIMPLE - short and cue-free: a cheap lookup, answered on the small tier.
    if n <= settings.simple_max_chars:
        logger.info("classify -> simple (short lookup)")
        return Classification(complexity=QuestionComplexity.SIMPLE, confidence=0.9,
                              reason="short factual lookup")

    # 3. SIMPLE but LOW confidence - long, no cue. Lean small, but let the router's
    #    route-up safety valve escalate to the frontier tier if it wants the headroom.
    logger.info("classify -> simple (long, low confidence)")
    return Classification(complexity=QuestionComplexity.SIMPLE, confidence=0.5,
                          reason="long input, no clear reasoning cue")
