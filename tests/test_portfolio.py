"""The build cards the page opens on (GET /portfolio) and each build's read page
(GET /portfolio/{week}), parsed from the context pack.

They come from the same files the agent answers from, cost nothing, show no internal package
ids, and render safely. Offline: no model is ever called.
"""
import re
from pathlib import Path

from fastapi.testclient import TestClient

from app import config, llm
from app.main import app
from app.portfolio import parse_portfolio, weeks_in

client = TestClient(app)
ROOT = Path(__file__).resolve().parent.parent


def test_portfolio_lists_every_week_of_the_pack():
    r = client.get("/portfolio")
    assert r.status_code == 200
    d = r.json()
    assert d["title"] == "AgentForge™"
    assert d["overview"].startswith("An engineering-operations copilot")
    assert [b["week"] for b in d["builds"]] == list(range(1, 18))
    assert d["builds"][0]["name"] == "ReleaseBot"
    assert d["builds"][-1]["name"] == "Demo Day + PortfolioAgent"
    assert set(d["builds"][0]) == {"week", "name", "summary"}


def test_no_internal_package_id_reaches_a_visitor():
    pid = re.compile(r"w\d\dv\d\dc\d\d")
    assert not pid.search(client.get("/portfolio").text)
    for week in (1, 8, 11, 17):
        assert not pid.search(client.get(f"/portfolio/{week}").text)


def test_no_milestone_is_silently_dropped():
    pack = (ROOT / "data" / "AGENTS.md").read_text(encoding="utf-8")
    assert len(client.get("/portfolio").json()["builds"]) == len(re.findall(r"^\*\*W\d+ · ", pack,
                                                                             re.M))


def test_a_summary_is_one_line_of_the_pack_text():
    w2 = client.get("/portfolio").json()["builds"][1]
    assert "\n" not in w2["summary"]
    assert "nano scored 93.3% (p95 1,185 ms)" in w2["summary"]


def test_a_students_own_pack_drives_the_cards():
    text = ("# MyAgent - AGENTS.md\n\n## What MyAgent is\nA helper.\n\n"
            "## Build history\n**W1 · First** - did a thing\nover two lines.\n\n"
            "**W2 · Second** (`w02`) - another.\n\nA loose note, not a milestone.\n\n"
            "## Later section\n**W9 · Not a build** - outside the section.\n")
    d = parse_portfolio(text)
    assert d["title"] == "MyAgent" and d["overview"] == "A helper."
    assert d["builds"] == [
        {"week": 1, "name": "First", "summary": "did a thing over two lines."},
        {"week": 2, "name": "Second", "summary": "another."},
    ]


def test_missing_pack_is_a_503(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    config.get_settings.cache_clear()
    r = client.get("/portfolio")
    assert r.status_code == 503 and "AGENTS.md" in r.json()["detail"]


def test_portfolio_never_calls_a_model(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the cards must not call a model")
    monkeypatch.setattr(llm, "chat", boom)
    assert client.get("/portfolio").status_code == 200


def test_portfolio_is_a_free_get_under_the_rate_limit(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1")
    config.get_settings.cache_clear()
    assert [client.get("/portfolio").status_code for _ in range(3)] == [200, 200, 200]


def test_page_opens_on_the_cards_and_renders_them_safely():
    page = (ROOT / "index.html").read_text(encoding="utf-8")
    assert re.search(r"checkHealth\(\);\s*showHistory\(\);\s*showBuilds\(\);\s*</script>", page)
    assert "fetch(`${API}/portfolio`)" in page
    render = page[page.index("function renderBuilds"):page.index("function askAboutBuild")]
    assert "escHtml(b.name)" in render and "fmtAnswer(b.summary)" in render
    assert "onclick" not in render           # build text never passes through an attribute
    ask = page[page.index("function askAboutBuild"):page.index("// ── Tool layer")]
    assert "fetch(" not in ask               # filling the question spends nothing


# ---------- one build's read page (new tab) ----------

def test_week_tags_cover_ranges():
    assert weeks_in("router (W2, W16)") == {2, 16}
    assert weeks_in("answers (W4-W6) and W10") == {4, 5, 6, 10}
    assert weeks_in("no tag, Week 5 spelled out") == set()


def test_read_page_shows_results_decisions_and_tagged_diagram_lines():
    r = client.get("/portfolio/5")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    page = r.text
    assert "<h1><span class='week'>W5</span> CitationRAG</h1>" in page
    assert "Grounded answers with citations." in page                   # the full description
    assert "top-1 cosine &gt;= 0.55 and spread &gt;= 0.08" in page      # eval_results w05_*
    assert "<b>Ground or refuse.</b>" in page                           # the decision citing W5
    assert "<mark>  Retrieval .. KnowledgeVault index" in page          # (W4-W6) covers W5
    assert page.count("<details>") == 4                                 # all four diagrams
    assert "href='/portfolio/4'" in page and "href='/portfolio/6'" in page


def test_week_17_shows_its_own_routing_gate():
    page = client.get("/portfolio/17").text
    assert "PortfolioAgent routing gate" in page and "27 frozen questions" in page
    assert "routing accuracy 0.963" in page


def test_a_week_with_nothing_recorded_says_so():
    page = client.get("/portfolio/1").text
    assert "No eval run is recorded for this week" in page
    assert "href='/portfolio/0'" not in page                            # no link before W1


def test_unknown_week_is_404_and_a_bad_week_422():
    assert client.get("/portfolio/99").status_code == 404
    assert client.get("/portfolio/abc").status_code == 422


def test_read_page_escapes_the_pack(tmp_path, monkeypatch):
    for name in ("architecture.md", "eval_results.json"):
        (tmp_path / name).write_text((ROOT / "data" / name).read_text(encoding="utf-8"),
                                     encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text(
        "# X\n\n## Build history\n**W1 · <script>alert(1)</script>** - a <img src=x> b\n",
        encoding="utf-8")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    config.get_settings.cache_clear()
    page = client.get("/portfolio/1").text
    assert "<script>alert" not in page and "<img" not in page
    assert "&lt;script&gt;" in page


def test_cards_link_each_build_to_its_page_in_a_new_tab():
    page = (ROOT / "index.html").read_text(encoding="utf-8")
    assert 'href="/portfolio/${+b.week}" target="_blank"' in page   # relative: the page's server
    assert 'rel="noopener"' in page and "build-pkg" not in page


def test_a_read_page_shows_the_week_in_depth_when_there_is_a_file():
    page = client.get("/portfolio/17").text
    assert "<h2>In depth</h2>" in page and "<h3>Key decisions</h3>" in page
    assert "<li><b>Tools instead of the whole pack in the prompt</b>" in page
