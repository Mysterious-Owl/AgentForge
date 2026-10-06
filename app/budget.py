"""The two caps every agent needs - an iteration cap and a cost ceiling - plus the
tiered price table both the ceiling and the eval harness are computed from.

An agent without a cap is a bill without a limit. Two failure shapes, two guards:

  * **Runaway loop** - the agent keeps calling tools. `IterationBudget` counts every model
    call and raises `IterationCapExceeded` at `max_iterations` (8) -> HTTP 429.
  * **Runaway prompt** - a deliberately enormous input tries to bypass the token budget.
    `guard_cost()` COUNTS the input with the model's tokenizer, PRICES the call before it is
    made, and raises `BudgetExceeded` if the projection is over `cost_ceiling_usd` (0.05)
    -> HTTP 413, rejected before any spend.

Pricing is keyed by MODEL ID (not tier label): the ledger bills the model that actually
served the call, so re-pointing a tier at a pricier model shows up as real cost, never a
smiling report. An unknown id bills at the frontier rate - fail expensive, never cheap.
Both caps live in `config.py`, both are surfaced on `GET /health`.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

from app.config import get_settings

logger = logging.getLogger(__name__)

# The completion budget of a single answer call - the same value app/llm.py asks for.
DEFAULT_MAX_COMPLETION_TOKENS = 700

# A chat call is billed for its message texts PLUS framing: a few tokens per call and a few
# per message. Every recorded two-message call in this build billed exactly 10 more than its
# texts' own count; the ceiling allows 16 per call and 8 per message, so it can only
# over-estimate. (The tool schemas are counted as text - the provider bills them as input.)
CHAT_FRAMING_TOKENS = 16
MESSAGE_FRAMING_TOKENS = 8

# The tokenizer's vocabulary ships with the package (data/tiktoken/), under the file name
# tiktoken's own cache uses, so counting never downloads anything and the tests stay offline.
# tiktoken checks the file against the SHA-256 it expects before using it. It is pointed at
# for the one load whatever TIKTOKEN_CACHE_DIR says - an empty cache set elsewhere would
# otherwise send tiktoken to the network.
_TIKTOKEN_DIR = Path(__file__).resolve().parent.parent / "data" / "tiktoken"


class BudgetExceeded(Exception):
    """The projected cost of ONE request is over the ceiling. Raised BEFORE the call."""

    def __init__(self, projected: float, ceiling: float) -> None:
        super().__init__(f"projected ${projected:.6f} exceeds ceiling ${ceiling:.6f}")
        self.projected = projected
        self.ceiling = ceiling


class IterationCapExceeded(Exception):
    """The answer loop asked for one model call more than it is allowed."""

    def __init__(self, limit: int) -> None:
        super().__init__(f"iteration_cap_exceeded (max {limit})")
        self.limit = limit


def max_iterations() -> int:
    return get_settings().max_iterations


def cost_ceiling() -> float:
    return get_settings().cost_ceiling_usd


# --- The price table (single source of truth: config.py) --------------------------

def _model_rates() -> dict[str, tuple[float, float]]:
    """model id -> (input USD per 1M tokens, output USD per 1M tokens).

    Built cheap -> expensive so a duplicate id keeps the HIGHER rate: you cannot make
    a bill cheaper by labelling a frontier model "small".
    """
    s = get_settings()
    return {
        s.small_model: (s.small_input_cost_per_1m, s.small_output_cost_per_1m),
        s.frontier_model: (s.frontier_input_cost_per_1m, s.frontier_output_cost_per_1m),
    }


def rates_for_model(model: str) -> tuple[float, float]:
    """(input, output) USD per 1M tokens for a model id. Unknown id -> frontier rate."""
    s = get_settings()
    return _model_rates().get(
        model, (s.frontier_input_cost_per_1m, s.frontier_output_cost_per_1m)
    )


@lru_cache(maxsize=1)
def _encoding():
    """o200k_base, loaded once from the shipped file - the GPT-5.4 family's tokenizer."""
    import tiktoken  # deferred - importing the app stays light

    previous = os.environ.get("TIKTOKEN_CACHE_DIR")
    os.environ["TIKTOKEN_CACHE_DIR"] = str(_TIKTOKEN_DIR)
    try:
        return tiktoken.get_encoding("o200k_base")
    finally:
        if previous is None:
            os.environ.pop("TIKTOKEN_CACHE_DIR", None)
        else:
            os.environ["TIKTOKEN_CACHE_DIR"] = previous


# tiktoken's BPE is quadratic on a long run with no whitespace (a 100k-char run takes ~15 s)
# and overflows its stack past ~0.5-1M chars (an uncatchable Rust panic) - so a request made
# of "xxxx..." could stall or crash the very guard meant to refuse it. Counting in pieces of at
# most 512 chars, each cut placed right after a newline that starts a line of text, keeps every
# piece small. o200k never merges a token across that boundary, so ordinary text counts
# EXACTLY; only a line longer than a piece is hard-cut, which can add at most one token per
# cut: the safe direction for a ceiling. A 1.26M-char run counts in ~0.25 s.
_COUNT_PIECE_CHARS = 512


def _pieces(text: str):
    start, end_of_text = 0, len(text)
    while start < end_of_text:
        end = min(start + _COUNT_PIECE_CHARS, end_of_text)
        if end < end_of_text:
            # Cut after a newline that starts a line of text: o200k merges a run of whitespace
            # ("\n\n", "\n  "), so a cut inside one would split a token and count one too many.
            cut = end
            while (newline := text.rfind("\n", start, cut)) > start:
                if not text[newline + 1].isspace():
                    end = newline + 1
                    break
                cut = newline
        yield text[start:end]
        start = end


def count_input_tokens(messages: list[str], message_count: int = 0) -> int:
    """The input tokens a chat call with these message texts is billed for - counted, not guessed.

    A characters-per-token guess reads low on exactly the input a runaway request carries -
    hex or base64 dumps, CJK, emoji - so the ceiling counts with the real tokenizer.
    `encode_ordinary` treats a string like "<|endoftext|>" in user text as plain text.
    """
    enc = _encoding()
    pieces = (piece for text in messages for piece in _pieces(text))
    framing = CHAT_FRAMING_TOKENS + MESSAGE_FRAMING_TOKENS * message_count
    return sum(len(enc.encode_ordinary(piece)) for piece in pieces) + framing


def cost_of(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    """Full-price cost of one call: input tokens x input rate + output tokens x output rate."""
    in_rate, out_rate = rates_for_model(model)
    usd = (prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000
    return round(usd, 6)


def project_cost(messages: list[str], model: str,
                 max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
                 message_count: int = 0) -> float:
    """What this call could cost AT WORST: its counted input + the full completion budget,
    priced at `model`'s split input/output rates.

    `messages` is exactly what is about to be sent - `llm.message_texts()` of the call.
    """
    return cost_of(count_input_tokens(messages, message_count), max_completion_tokens, model)


def guard_cost(messages: list[str], model: str,
               max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
               message_count: int = 0, already_spent: float = 0.0) -> float:
    """THE LINE THAT ENFORCES THE CEILING. Raises before a single token is bought.

    The ceiling is per REQUEST, and a request is a loop: what the earlier calls already cost
    plus this call's worst case must fit under it."""
    projected = round(already_spent + project_cost(messages, model, max_completion_tokens,
                                                   message_count), 6)
    ceiling = cost_ceiling()
    if projected > ceiling:
        logger.warning("cost ceiling hit: projected $%.6f > ceiling $%.6f", projected, ceiling)
        raise BudgetExceeded(projected, ceiling)
    return projected


def actual_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    """Price a call that already happened, from real usage - for the request log."""
    return cost_of(prompt_tokens, completion_tokens, model)


class IterationBudget:
    """A step counter that refuses to count past the cap."""

    def __init__(self, limit: int | None = None) -> None:
        self._limit = limit if limit is not None else max_iterations()
        self._used = 0

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def used(self) -> int:
        return self._used

    def step(self) -> int:
        """THE LINE THAT ENFORCES THE ITERATION CAP.

        Every model call goes through here - the first attempt AND each network retry.
        A lookup uses 2 of 8 (tools, then the answer). The 9th raises, and the run stops.
        """
        if self._used >= self._limit:
            logger.warning("iteration cap hit: %s of %s used", self._used, self._limit)
            raise IterationCapExceeded(self._limit)
        self._used += 1
        return self._used
