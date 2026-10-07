"""Correctness the walkthrough relies on but does not narrate: closed clients, read-only reads,
thread-safe state, honest token counts, safe PII patterns, and a pinned route surface."""
import inspect
import threading
import time

from fastapi.testclient import TestClient

from app import budget, llm, memory, router, tools
from app.main import app
from app.schemas import AuditEntry
from app.scrub import CARD_REDACTION, PHONE_REDACTION, scrub

client = TestClient(app)

# FastAPI adds these on its own; they are not part of the surface the diagram claims.
DOC_ROUTES = {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}
SURFACE = {
    ("GET", "/"), ("GET", "/admin"), ("GET", "/health"), ("GET", "/readme"), ("GET", "/portfolio"),
    ("GET", "/portfolio/{week}"),
    ("GET", "/.well-known/agent-card.json"), ("GET", "/.well-known/agent.json"),
    ("GET", "/.well-known/jwks.json"),
    ("POST", "/a2a"), ("POST", "/ask"), ("GET", "/actions"),
    ("POST", "/approve"), ("GET", "/audit/{user_id:path}"),
    ("POST", "/memory"), ("GET", "/memory"), ("DELETE", "/memory"),
}


def test_route_surface_is_exactly_the_seventeen_routes():
    routes = {(m, r.path) for r in app.routes if r.path not in DOC_ROUTES
              for m in getattr(r, "methods", set()) - {"HEAD"}}
    assert routes == SURFACE


# ---------- the provider seam: one client per call, closed, with a timeout, no hidden retries --

def test_chat_opens_and_closes_one_client(monkeypatch):
    seen = {}

    class FakeClient:
        def __init__(self, **kwargs):
            seen["kwargs"] = kwargs

        def __enter__(self):
            seen["entered"] = True
            return self

        def __exit__(self, *exc):
            seen["closed"] = True
            return False

        class chat:                                   # noqa: N801 - mirrors the SDK's shape
            class completions:                        # noqa: N801
                @staticmethod
                def create(**kwargs):
                    seen["create"] = kwargs
                    usage = type("U", (), {"prompt_tokens": 12, "completion_tokens": 3})
                    fn = type("F", (), {"name": "get_build", "arguments": '{"week": 4}'})
                    call = type("T", (), {"id": "call_1", "function": fn})
                    msg = type("M", (), {"content": None, "tool_calls": [call]})
                    return type("R", (), {"choices": [type("C", (), {"message": msg})],
                                          "usage": usage})

    monkeypatch.setattr(llm, "OpenAI", FakeClient)
    messages = [{"role": "user", "content": "What did you build in Week 4?"}]
    turn = llm.chat(messages, "gpt-5.4-nano-2026-03-17", tools.TOOL_SPECS)
    assert turn.tool_calls[0].name == "get_build" and turn.prompt_tokens == 12
    assert seen["create"]["tools"] == tools.TOOL_SPECS     # the model is OFFERED the tools
    assert seen["entered"] and seen["closed"]              # the pool is released, not leaked
    assert seen["kwargs"]["max_retries"] == 0              # tenacity owns retries - no stacking
    assert seen["kwargs"]["timeout"] > 0                   # a hung provider cannot hang /ask


# ---------- the cost ceiling counts tokens with the model's tokenizer ----------

def test_ceiling_counts_input_with_the_tokenizer():
    enc = budget._encoding()
    system, question = "You answer from the pack.", "Describe <|endoftext|> politely."
    expected = (len(enc.encode_ordinary(system)) + len(enc.encode_ordinary(question))
                + budget.CHAT_FRAMING_TOKENS + 2 * budget.MESSAGE_FRAMING_TOKENS)
    assert budget.count_input_tokens([system, question], message_count=2) == expected


def test_a_long_text_is_counted_exactly_despite_the_pieces():
    from pathlib import Path
    data = Path(__file__).resolve().parent.parent / "data"
    enc = budget._encoding()
    for name in ("AGENTS.md", "architecture.md"):          # ordinary prose: cut at newlines
        text = (data / name).read_text(encoding="utf-8")
        assert budget.count_input_tokens([text]) == len(enc.encode_ordinary(text)) + \
            budget.CHAT_FRAMING_TOKENS, name


def test_a_line_longer_than_a_piece_is_never_under_counted():
    from app.context import get_context
    pack = get_context()                       # eval_results.json has 500+ character lines
    enc = budget._encoding()
    exact = len(enc.encode_ordinary(pack)) + budget.CHAT_FRAMING_TOKENS
    counted = budget.count_input_tokens([pack])
    assert exact <= counted <= exact + len(pack) // budget._COUNT_PIECE_CHARS + 1


def test_the_ceiling_prices_the_tool_schemas_too():
    from app import portfolio
    messages = llm.build_messages("What did you build in Week 4?", [], portfolio.get_portfolio())
    with_tools = budget.count_input_tokens(llm.message_texts(messages, tools.TOOL_SPECS))
    without = budget.count_input_tokens(llm.message_texts(messages, []))
    assert with_tools - without > 200                      # the schemas are billed as input


def test_dense_input_is_not_under_counted():
    dense = "".join(f"{i:08x}" for i in range(5000))          # 40,000 chars of hex
    assert budget.count_input_tokens([dense]) > len(dense) // 4  # chars/4 would under-price it


def test_a_huge_unbroken_input_gets_a_413_not_a_crash(monkeypatch):
    """One unbroken 1.2M-char run used to overflow the tokenizer's stack (a Rust panic that
    `except Exception` cannot catch). It must be counted - and refused by the ceiling."""
    started = time.monotonic()
    assert budget.count_input_tokens(["x" * 1_200_000]) > 100_000
    assert time.monotonic() - started < 5          # linear, not quadratic, in the run length
    monkeypatch.setattr(llm, "chat", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    r = client.post("/ask", json={"question": "x" * 1_200_000})
    assert r.status_code == 413 and r.json()["detail"]["error"] == "cost_ceiling_exceeded"


# ---------- memory: reads never create state; tools: decide is atomic ----------

def test_reads_do_not_create_memory_buckets():
    memory.recall("ghost", "s1", "k")
    memory.audit_log("ghost")
    memory.delete("ghost", "s1", "k")
    assert "ghost" not in memory._STATE


def test_concurrent_approvals_execute_once(monkeypatch):
    executions = []

    def slow_execute(action, approver):
        executions.append(action.id)
        time.sleep(0.05)                                # widen any check-then-act window
        return f"Intro accepted by {approver}"

    monkeypatch.setattr(tools, "_execute", slow_execute)
    env = tools.request_intro({"name": "Jane", "contact": "j@x.io", "reason": "hiring",
                               "message": "hi"}, "u-1", "web")
    action = tools.get_action(env.data["action_id"])
    outcomes = []

    def click():
        try:
            outcomes.append(tools.decide(action.id, True).status)
        except tools.AlreadyDecided:
            outcomes.append("refused")

    threads = [threading.Thread(target=click) for _ in range(12)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert executions == [action.id]
    assert sorted(outcomes) == ["executed"] + ["refused"] * 11
    assert tools.get_action(action.id).status == "executed"


def test_approval_is_audited_in_the_proposing_session(monkeypatch):
    from tests.scripted import answer, calls, scripted
    monkeypatch.setattr(llm, "chat", scripted(
        calls("request_intro", name="Jane", contact="j@x.io", reason="hiring", message="hi"),
        answer("Your request is waiting for the student's approval.")))
    proposed = client.post("/ask", json={"question": "Please have the student contact me at "
                                                     "j@x.io", "user_id": "u-9",
                                         "session_id": "web"}).json()["pending_action"]
    client.post("/approve", json={"action_id": proposed["id"], "approve": True})
    sessions = {e.kind: e.session_id for e in memory.audit_log("u-9")}
    assert sessions == {"ask": "web", "intro:proposed": "web", "intro:executed": "web"}


def test_user_ids_with_a_slash_reach_their_own_audit_log():
    memory.log_audit(AuditEntry(user_id="team/alice", session_id="s", kind="ask", detail="x"))
    r = client.get("/audit/team/alice")
    assert r.status_code == 200 and [e["user_id"] for e in r.json()] == ["team/alice"]


# ---------- the router has one public signature; nothing dead ----------

def test_router_has_no_unused_force_tier():
    assert "force_tier" not in inspect.signature(router.resolve_tier).parameters
    assert "force_tier" not in inspect.signature(router.route).parameters


# ---------- PII patterns: what they promise, and nothing more ----------

def test_iso_dates_and_times_are_not_phones():
    for stamp in ("2026-09-11", "2026-09-11 14:23:05", "2026-09-11T14:23:05Z",
                  "gpt-5.4-nano-2026-03-17"):
        assert scrub(f"at {stamp} ok") == f"at {stamp} ok", stamp


def test_phones_and_cards_are_redacted_under_their_own_labels():
    assert scrub("call +1 (555) 123-4567 now") == f"call {PHONE_REDACTION} now"
    assert scrub("card 4111 1111 1111 1111 on file") == f"card {CARD_REDACTION} on file"
    # 16 digits that fail the Luhn check are not a card - but they are still never logged raw
    assert "1234" not in scrub("ref 1234 5678 9012 3456")


# ---------- the UI: one brand, and no server value inside an inline handler ----------

def test_ui_credits_the_course_under_the_student_and_has_no_inline_handler_ids():
    page = client.get("/").text
    assert "<title>AgentForge - PortfolioAgent</title>" in page   # the profile's name replaces it
    assert "Applied GenAI &amp; Agentic AI Engineering Course" in page   # credited, underneath
    assert "CoreSmart AI" not in page
    assert "decideAction('${" not in page


# ---------- the open-source path: only the SMALL tier moves to the local server ----------

def _capture_clients(monkeypatch):
    seen = []

    class FakeClient:
        def __init__(self, **kwargs):
            self.base_url = kwargs.get("base_url")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        @property
        def chat(self):
            return self

        @property
        def completions(self):
            return self

        def create(self, **kwargs):
            seen.append((self.base_url, kwargs["model"]))
            usage = type("U", (), {"prompt_tokens": 1, "completion_tokens": 1})
            msg = type("M", (), {"content": "ok", "tool_calls": None})
            return type("R", (), {"choices": [type("C", (), {"message": msg})], "usage": usage})

    monkeypatch.setattr(llm, "OpenAI", FakeClient)
    return seen


def test_small_base_url_moves_only_the_small_tier(monkeypatch):
    from app import config
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("SMALL_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("SMALL_MODEL", "qwen3:8b")
    config.get_settings.cache_clear()
    seen = _capture_clients(monkeypatch)
    llm.chat([{"role": "user", "content": "q"}], "qwen3:8b", tools.TOOL_SPECS)
    llm.chat([{"role": "user", "content": "q"}], "gpt-5.4-mini-2026-03-17", tools.TOOL_SPECS)
    assert seen == [("http://localhost:11434/v1", "qwen3:8b"),
                    (None, "gpt-5.4-mini-2026-03-17")]      # the frontier stays on the vendor


# ---------- the tokenizer never leaves the machine ----------

def test_the_shipped_vocabulary_is_used_even_with_a_cache_dir_set(monkeypatch, tmp_path):
    import tiktoken.registry
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path))          # an empty cache
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")          # and no network
    monkeypatch.setattr(tiktoken.registry, "ENCODINGS", {})
    budget._encoding.cache_clear()
    try:
        assert budget.count_input_tokens(["hello world"]) == 2 + budget.CHAT_FRAMING_TOKENS
        assert list(tmp_path.iterdir()) == []                        # nothing downloaded
    finally:
        budget._encoding.cache_clear()


# ---------- routing cues are whole words ----------

def test_cues_match_whole_words_not_parts_of_words():
    from app.classifier import classify
    small = ["Tell me about your architecture", "Is the default ceiling reasonable?",
             "What is in the decision log?", "Which week designed the gate?",
             "What design system does the UI use?"]
    frontier = ["Why does the small tier run on nano?", "Compare the two tiers",
                "Comparing nano and mini, which is cheaper?", "What are the trade-offs here?",
                "Walk me through the approval gate", "Justify the cost ceiling",
                "What tradeoffs did Week 16 make?", "Explain why the gate is server state"]
    assert {q: classify(q).complexity.value for q in small} == dict.fromkeys(small, "simple")
    assert {q: classify(q).complexity.value for q in frontier} \
        == dict.fromkeys(frontier, "frontier")


# ---------- every attempt is one counted iteration, retries included ----------

def test_a_network_retry_spends_an_iteration(monkeypatch):
    from openai import APIConnectionError
    attempts = []

    class Flaky:
        class chat:                                   # noqa: N801 - mirrors the SDK's shape
            class completions:                        # noqa: N801
                @staticmethod
                def create(**kwargs):
                    attempts.append(1)
                    if len(attempts) < 3:
                        raise APIConnectionError(request=None)
                    return "ok"

    monkeypatch.setattr(llm._create_completion.retry, "sleep", lambda _s: None)
    spent = budget.IterationBudget()
    assert llm._create_completion(Flaky(), budget=spent, model="m") == "ok"
    assert len(attempts) == 3 and spent.used == 3
