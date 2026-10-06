"""The context pack - the agent's entire brain.

Three files in `data/` become one block of system context, re-read per request.
If any file is missing, `build_context` raises `ContextPackError` - the route
turns that into a 503. Never a silent half-answer from an empty brain.
"""
from __future__ import annotations

import logging
from pathlib import Path

from app.config import get_settings

logger = logging.getLogger(__name__)

PACK_FILES = ("AGENTS.md", "architecture.md", "eval_results.json")


class ContextPackError(RuntimeError):
    """Raised when a required context-pack file is missing or unreadable."""


def _pack_dir() -> Path:
    settings = get_settings()
    root = Path(settings.data_dir)
    if not root.is_absolute():
        root = Path(__file__).parent.parent / root
    return root


def build_context() -> str:
    """Read the three pack files and concatenate them behind clear headers.

    Raises ContextPackError if any file is missing - fail loud, never a partial pack.
    Nothing is cached: `get_context()` calls this on every request.
    """
    root = _pack_dir()
    sections: list[str] = []
    for name in PACK_FILES:
        path = root / name
        if not path.exists():
            raise ContextPackError(f"missing context-pack file: {name}")
        header = name.replace(".md", "").replace(".json", "").upper()
        sections.append(f"===== {header} =====\n{path.read_text(encoding='utf-8')}")
    logger.info("Context pack loaded: %s files from %s", len(PACK_FILES), root)
    return "\n\n".join(sections)


def get_context() -> str:
    """Assemble the pack fresh on every request.

    Three small files - microseconds, no cache. The payoff: a broken pack
    surfaces as a 503 on the very NEXT ask, never from behind a stale cache.
    (Startup ALSO probes the pack once, so a bad deploy fails loudly at boot.)
    """
    return build_context()
