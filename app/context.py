"""The context pack - the three files every answer is looked up in.

The model never sees the pack whole: the tools (app/tools.py, app/portfolio.py) read the
files they need, fresh, on every call. This module is the PROBE the routes run first: if a
pack file is missing, unreadable, or `eval_results.json` is not a JSON object,
`build_context` raises `ContextPackError` - the route turns that into a 503, and startup
into a boot failure. Never a silent half-answer from a broken pack.
"""
from __future__ import annotations

import json
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


def read_pack_file(name: str) -> str:
    """One pack file's text. Missing or unreadable (bad encoding, permissions) ->
    ContextPackError, never a raw OSError the route would turn into a 500."""
    path = _pack_dir() / name
    if not path.exists():
        raise ContextPackError(f"missing context-pack file: {name}")
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ContextPackError(f"unreadable context-pack file: {name} ({exc})") from exc


def read_pack_json(name: str) -> dict:
    """A JSON pack file, which must hold a JSON object - anything else is a broken pack."""
    try:
        data = json.loads(read_pack_file(name))
    except json.JSONDecodeError as exc:
        raise ContextPackError(f"{name} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ContextPackError(f"{name} must be a JSON object")
    return data


def build_context() -> str:
    """Check the three pack files and return them behind clear headers.

    Raises ContextPackError if any file is missing, unreadable or (eval_results.json) not a
    JSON object - fail loud, never a partial pack. Nothing is cached: `get_context()` calls
    this on every request.
    """
    root = _pack_dir()
    sections: list[str] = []
    for name in PACK_FILES:
        text = read_pack_file(name)
        if name.endswith(".json"):
            read_pack_json(name)
        header = name.replace(".md", "").replace(".json", "").upper()
        sections.append(f"===== {header} =====\n{text}")
    logger.info("Context pack loaded: %s files from %s", len(PACK_FILES), root)
    return "\n\n".join(sections)


def get_context() -> str:
    """Probe the pack fresh on every request (the routes discard the text; the tools read
    what they need). Three small files - microseconds, no cache. The payoff: a broken pack
    surfaces as a 503 on the very NEXT ask, never from behind a stale cache.
    (Startup ALSO probes the pack once, so a bad deploy fails loudly at boot.)
    """
    return build_context()
