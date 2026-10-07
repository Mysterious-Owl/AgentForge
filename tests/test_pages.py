"""The pages: the visitor's index.html, the owner's admin.html and the rendered README.

There is no browser in the suite, so the page-safety properties are checked on the source -
one stable token per assertion: escaping, no inline handler built from server data, text
(never HTML) for the student's name, the owner token kept in this tab, one decision per click.
"""
import re
from pathlib import Path

from fastapi.testclient import TestClient

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
