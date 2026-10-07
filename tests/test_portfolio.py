"""The build cards the page opens on (GET /portfolio) and each build's read page
(GET /portfolio/{week}), parsed from the context pack.

They come from the same files the agent answers from, cost nothing, show no internal package
ids, and render safely. Offline: no model is ever called.
"""
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import config
from app.main import app
from app.portfolio import parse_portfolio, pass_rate, weeks_in

client = TestClient(app)
ROOT = Path(__file__).resolve().parent.parent


def test_portfolio_cards():
    r = client.get("/portfolio")
    assert r.status_code == 200
    d = r.json()
    assert d["title"] == "AgentForge™"
    assert d["overview"].startswith("An engineering-operations copilot")
    assert [b["week"] for b in d["builds"]] == list(range(1, 18))   # no milestone dropped
    assert d["builds"][0]["name"] == "ReleaseBot"
    assert d["builds"][-1]["name"] == "Demo Day + PortfolioAgent"
    assert set(d["builds"][0]) == {"week", "name", "summary"}
    w2 = d["builds"][1]["summary"]                                  # one line of the pack text
    assert "\n" not in w2 and "Nano scored 93.3% (p95 1,185 ms" in w2


def test_no_internal_package_id_reaches_a_visitor():
    pid = re.compile(r"w\d\dv\d\dc\d\d")
    assert not pid.search(client.get("/portfolio").text)
    for week in (1, 8, 11, 17):
        assert not pid.search(client.get(f"/portfolio/{week}").text)


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


# ---------- one build's read page (new tab) ----------

def test_week_tags_cover_ranges():
    assert weeks_in("router (W2, W16)") == {2, 16}
    assert weeks_in("answers (W4-W6) and W10") == {4, 5, 6, 10}
    assert weeks_in("no tag, Week 5 spelled out") == set()


@pytest.mark.parametrize("week,needles,absent", [
    (5, ["<h1><span class='week'>W5</span> CitationRAG</h1>",
         "Grounded answers with citations.",                     # the full description
         "top-1 cosine &gt;= 0.55 and spread &gt;= 0.08",         # eval_results w05_*
         "<b>Ground or refuse.</b>",                              # the decision citing W5
         "<mark>  Retrieval .. KnowledgeVault index",             # (W4-W6) covers W5
         "href='/portfolio/4'", "href='/portfolio/6'"], []),
    (17, ["PortfolioAgent routing gate", "27 frozen questions", "routing 27/27 (100%)",
          "<h2>In depth</h2>", "<h3>Key decisions</h3>",           # the week file, when present
          "<li><b>Tools instead of the whole pack in the prompt</b>"], []),
    (1, ["No eval run is recorded for this week"], ["href='/portfolio/0'"]),   # no link before W1
])
def test_read_page_content(week, needles, absent):
    r = client.get(f"/portfolio/{week}")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    for needle in needles:
        assert needle in r.text, needle
    for needle in absent:
        assert needle not in r.text, needle
    assert r.text.count("<details>") == 4                                # all four diagrams


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


@pytest.mark.parametrize("gate,text", [
    ({"passed": 27, "cases": 27, "overall_score": 1.0}, "27/27 (100%)"),
    ({"passed": 26, "cases": 27, "overall_score": 0.963}, "26/27 (96.3%)"),
    ({"passed": 10, "cases": 14, "overall_score": 0.714}, "10/14 (71.4%)"),
    ({"passed": None, "cases": None, "overall_score": None}, "not run yet"),
])
def test_a_gate_reads_as_passed_of_cases_and_a_percentage(gate, text):
    assert pass_rate(gate) == text
