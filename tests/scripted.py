"""A scripted model for the tests: it plays back ModelTurns in order through the real agent
loop - so the tools, the citation check and both caps run for real, with no network.

    model = scripted(calls("get_build", week=4), answer("It ingests PDFs [AGENTS.md · W4]."))
    monkeypatch.setattr(llm, "chat", model)

Each call spends one iteration, like `llm.chat` does, and records what it was sent.
"""
from __future__ import annotations

import json

from app.schemas import ModelTurn, ToolCall


def calls(*specs, ptok: int = 900, ctok: int = 30, **args) -> ModelTurn:
    """A turn that asks for tools: calls("get_build", week=4) for one, or
    calls(("get_build", {"week": 5}), ("get_build", {"week": 6})) for several."""
    if specs and isinstance(specs[0], str):
        specs = ((specs[0], args),)
    return ModelTurn(tool_calls=[ToolCall(id=f"c{i}", name=n, arguments=json.dumps(a))
                                 for i, (n, a) in enumerate(specs)],
                     prompt_tokens=ptok, completion_tokens=ctok)


def answer(text: str, ptok: int = 1300, ctok: int = 80) -> ModelTurn:
    return ModelTurn(text=text, prompt_tokens=ptok, completion_tokens=ctok)


def scripted(*turns: ModelTurn):
    """A stand-in for llm.chat that returns `turns` in order (the last one repeats)."""
    seen = {"models": [], "messages": [], "tools": []}

    def _chat(messages, model, tools, budget=None):
        if budget is not None:
            budget.step()
        i = len(seen["models"])
        seen["models"].append(model)
        seen["messages"].append([dict(m) for m in messages])
        seen["tools"].append(tools)
        return turns[min(i, len(turns) - 1)]

    _chat.seen = seen
    return _chat


def grounded_model(text: str = "AgentForge is an engineering-operations copilot "
                                "[AGENTS.md · overview].", ptok: int = 1300, ctok: int = 80):
    """The everyday case: one tool call, then an answer that cites what it returned."""
    return scripted(calls("get_architecture", section="overview"), answer(text, ptok, ctok))
