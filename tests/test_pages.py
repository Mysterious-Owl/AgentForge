"""The pages: the visitor's index.html, the owner's admin.html and the rendered README.

Two kinds of check. The source greps pin one stable token per property: escaping, no inline
handler built from server data, text (never HTML) for the student's name, the owner token kept
in this tab, one decision per click. The behaviour tests run the pages' own script in Node
against a fake DOM (tests/page_harness.js) with hostile text in every field - skipped when
Node.js is not installed.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import signing
from app.main import app

client = TestClient(app)
ROOT = Path(__file__).resolve().parent.parent
ESC_HTML = ("function escHtml(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;')"
            ".replace(/>/g,'&gt;').replace(/\"/g,'&quot;').replace(/'/g,'&#39;'); }")


def _page(name):
    return (ROOT / name).read_text(encoding="utf-8")


def _esc_html(page):
    """escHtml's whole definition, whitespace removed - every replacement must be there."""
    start = page.index("function escHtml")
    return re.sub(r"\s+", "", page[start:page.index("}", start) + 1])


def _between(page, start, end):
    return page[page.index(start):page.index(end, page.index(start))]


def test_pages_are_served():
    for path in ("/", "/admin"):
        r = client.get(path)
        assert r.status_code == 200 and "text/html" in r.headers["content-type"], path
    admin, public = client.get("/admin").text, client.get("/").text
    for needle in ('id="owner-token"', "'/actions'", "'/approve'", "/audit/"):
        assert needle in admin, needle
    # the owner's tools live on /admin only; the public page links to it
    assert 'id="owner-token"' not in public and "/actions" not in public
    assert 'href="/admin"' in public


def test_readme_renders():
    r = client.get("/readme")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "<title>PortfolioAgent - README</title>" in r.text
    assert "<h2" in r.text and "<table>" in r.text                 # markdown, not raw text


def test_index_page_safety_greps():
    page = _page("index.html")
    script = page[page.index("<script>"):]
    assert _esc_html(page) == re.sub(r"\s+", "", ESC_HTML)      # all five characters escaped
    assert not re.search(r"onclick\s*=", script)              # no handler built from data
    assert "escHtml(s.content" in page                        # retrieved context, escaped
    assert "escHtml(b.name)" in page                          # build cards, escaped
    assert "fmtAnswer(b.summary)" in page
    profile = _between(page, "function renderProfile", "function aboutLine")
    assert ".textContent = `${p.name}" in profile             # the name is text ...
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML"):
        assert sink not in profile, sink                      # ... never HTML
    assert "escHtml(p.name)" in page and "escHtml(p[k])" in page
    assert 'rel="noopener"' in page                           # new tabs cannot reach back
    assert 'href="/portfolio/${+b.week}"' in page             # relative: the page's server
    assert "fetch(" not in _between(page, "function askAboutBuild", "// ── Tool layer")
    assert "crypto.subtle.verify" in page                     # the card verified in the browser
    assert "new URL(header.jku).pathname" in page             # with the key the header names
    assert "Array.from(t).slice(0, HISTORY_CHARS)" in page    # history cut by characters
    assert "BUSY + (a ? 1 : -1)" in page                      # Ask re-enables when ALL are done
    assert "/^\\s*OUT_OF_SCOPE/" in page
    assert "if(!(d.answer || '').trim()) throw" in page       # an empty answer is refused


def test_admin_page_safety_greps():
    page = _page("admin.html")
    assert _esc_html(page) == re.sub(r"\s+", "", ESC_HTML)
    assert not re.search(r"onclick\s*=", page)
    assert "r.status === 401)" in page                        # a 401 is said, not rendered blank
    assert "sessionStorage.setItem(OWNER_KEY" in page         # the token stays in this tab
    assert "localStorage" not in page
    assert "'Authorization': `Bearer ${t}`" in page           # sent on the owner-only calls
    assert "buttons.forEach(x => x.disabled = true)" in page  # one decision per click
    assert "user === '..'" in page                            # no path walk in the audit lookup


# ---------- behaviour: the pages' own script, run in Node against a fake DOM ----------

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
XSS = "\"'><img src=x onerror=alert(1)>"


def _run_page(page, fixtures, tmp_path):
    """Every HTML string the page's script writes for these render calls (page_harness.js)."""
    spec = tmp_path / "fixtures.json"
    spec.write_text(json.dumps(fixtures), encoding="utf-8")
    out = subprocess.run([NODE, str(ROOT / "tests" / "page_harness.js"), str(ROOT / page),
                          str(spec)], capture_output=True, text=True, encoding="utf-8",
                         timeout=60, check=True)
    return json.loads(out.stdout)


@needs_node
def test_no_server_or_model_text_reaches_the_index_page_as_html(tmp_path):
    """Hostile text in EVERY field the visitor's page renders - the answer, the trace, the
    retrieved context, citations, the intro card, build cards, the profile, logs, errors -
    comes out as text. Removing one escHtml anywhere on these paths fails this test."""
    step = {"tool": XSS, "args": {XSS: XSS}, "success": False, "source": XSS, "error": XSS,
            "content": XSS}
    intro = {"id": XSS, "name": XSS, "company": XSS, "contact": XSS, "reason": XSS,
             "message": XSS, "status": XSS, "result": XSS}
    answer = {"answer": f"{XSS} [{XSS}]", "model": XSS, "tier": XSS, "complexity": XSS,
              "grounded": True, "routed_up": True, "tools_called": [step],
              "citations": [XSS], "unverified_citations": [XSS], "pending_action": intro,
              "model_calls": 2, "cost_usd": 0.001, "baseline_cost_usd": 0.002,
              "saved_usd": 0.001}
    builds = {"title": XSS, "overview": XSS, "builds": [{"week": 4, "name": XSS, "summary": XSS}],
              "gates": {"routing": {"score": 1.0, "cases": 27, "passed": 27},
                        "tool_choice": {"score": None, "cases": None, "passed": None}},
              "profile": {"name": XSS, "headline": XSS, "linkedin": "https://x.example/" + XSS}}
    written = _run_page("index.html", [["renderAnswer", answer], ["renderAnswer", {**answer,
                        "pending_action": None}], ["renderBuilds", builds],
                        ["renderAction", intro], ["addLog", XSS], ["showError", XSS]], tmp_path)
    assert len(written) >= 6
    for html in written:
        assert "<img" not in html, html[:300]


@needs_node
def test_no_server_text_reaches_the_admin_page_as_html(tmp_path):
    intro = {"id": XSS, "name": XSS, "company": XSS, "contact": XSS, "reason": XSS,
             "message": XSS, "status": "input-required", "result": XSS}
    written = _run_page("admin.html", [["renderAction", intro]], tmp_path)
    assert written and all("<img" not in html for html in written)


@needs_node
@pytest.mark.parametrize("gate,shown", [
    ({"score": 1.0, "cases": 27, "passed": 27}, "27/27 (100%)"),
    ({"score": 0.963, "cases": 27, "passed": 26}, "26/27 (96.3%)"),
    ({"score": None, "cases": None, "passed": None}, "not run yet"),
])
def test_the_page_shows_a_gate_as_passed_of_cases_and_a_percentage(tmp_path, gate, shown):
    [chips] = _run_page("index.html", [["gateChips", {"routing": gate, "tool_choice": gate}]],
                        tmp_path)
    assert chips.count(shown) == 2


@needs_node
def test_the_browser_rebuilds_exactly_the_bytes_the_server_signed(tmp_path):
    """The page verifies the card with its OWN canonical() - it must produce the same bytes
    as app/signing.canonical, or every visitor sees "signature does not match"."""
    card = {k: v for k, v in client.get("/.well-known/agent-card.json").json().items()
            if k != "signatures"}
    [text] = _run_page("index.html", [["canonical", card]], tmp_path)
    assert text.encode() == signing.canonical(card)
