"""Reading the pack - for the build cards, the read pages, and the agent's read tools.

Everything comes from the three pack files, so the cards, the read pages and what the tools
return share one source: replace the pack and all three follow. Pure text parsing - no model
call. `section()`, `week_results()` and `get_portfolio()` are what `app/tools.py` serves.

- The cards: the "Build history" of `data/AGENTS.md`. Each milestone is one paragraph that opens
  with `**W<n> · <name>**`, then an optional parenthetical (skipped), then ` - ` and
  the description.
- A build's read page adds that week's entries from `eval_results.json` (keys `w05_...`; Week 17
  shows the routing gate), the key decisions in `architecture.md` that cite the week, and the
  diagram lines tagged with it - `(W10)`, `(W4-W6)`.
"""
from __future__ import annotations

import html
import json
import re

from app.context import ContextPackError, _pack_dir

_SECTION = re.compile(r"^## Build history.*?$(.*?)(?=^## |\Z)", re.M | re.S)
_ENTRY = re.compile(r"^\*\*W(\d+) · (.+?)\*\*\s*(?:\([^)]*\))?\s*-\s*(.*)$", re.S)
_OVERVIEW = re.compile(r"^## What .+? is\s*$(.*?)(?=^## |\Z)", re.M | re.S)
_TITLE = re.compile(r"^# (.+?)(?: - .*)?$", re.M)
_DECISIONS = re.compile(r"^## Key decisions\s*$(.*?)(?=^## |\Z)", re.M | re.S)
_DIAGRAM = re.compile(r"^## (Diagram \d+ - .+?)\s*$\s*```\n(.*?)```", re.M | re.S)
_WEEK_TAG = re.compile(r"\bW(\d+)(?:-W?(\d+))?\b")


def _read(name: str) -> str:
    path = _pack_dir() / name
    if not path.exists():
        raise ContextPackError(f"missing context-pack file: {name}")
    return path.read_text(encoding="utf-8")


def parse_portfolio(text: str) -> dict:
    """AGENTS.md text -> {"title", "overview", "builds": [{week, name, summary}]}."""
    title = _TITLE.search(text)
    overview = _OVERVIEW.search(text)
    builds = []
    section = _SECTION.search(text)
    for para in re.split(r"\n\s*\n", section.group(1) if section else ""):
        entry = _ENTRY.match(para.strip())
        if entry:
            week, name, summary = entry.groups()
            builds.append({"week": int(week), "name": name.strip(),
                           "summary": " ".join(summary.split())})
    return {
        "title": title.group(1).strip() if title else "",
        "overview": " ".join(overview.group(1).split()) if overview else "",
        "builds": builds,
    }


def get_portfolio() -> dict:
    """Read AGENTS.md fresh (like the pack itself); a missing file is the pack's 503."""
    return parse_portfolio(_read("AGENTS.md"))


def weeks_in(text: str) -> set[int]:
    """Every week a line cites: `W10` -> {10}, `W4-W6` -> {4, 5, 6}."""
    weeks: set[int] = set()
    for start, end in _WEEK_TAG.findall(text):
        weeks.update(range(int(start), int(end or start) + 1))
    return weeks


# The named sections get_architecture(section) can return: (file, heading that opens it).
# A student's own pack keeps these headings, or edits this table - one place.
SECTIONS: dict[str, tuple[str, str]] = {
    "overview": ("AGENTS.md", r"## What .+? is"),
    "rules": ("AGENTS.md", r"## Operating rules"),
    "portfolio_agent": ("AGENTS.md", r"## The PortfolioAgent"),
    "spine": ("architecture.md", r"## Spine"),
    "decisions": ("architecture.md", r"## Key decisions"),
    "diagram_1": ("architecture.md", r"## Diagram 1 "),
    "diagram_2": ("architecture.md", r"## Diagram 2 "),
    "diagram_3": ("architecture.md", r"## Diagram 3 "),
    "diagram_4": ("architecture.md", r"## Diagram 4 "),
    "numbers": ("architecture.md", r"## Where the numbers"),
}


def section(name: str) -> dict | None:
    """One named section of the pack: {"title", "text", "source"}, or None if it is absent."""
    if name not in SECTIONS:
        return None
    file, heading = SECTIONS[name]
    found = re.search(rf"^({heading}.*?)$(.*?)(?=^## |\Z)", _read(file), re.M | re.S)
    if not found:
        return None
    return {"title": found.group(1).lstrip("# ").strip(), "text": found.group(2).strip(),
            "source": f"{file} · {name}"}


_PROFILE_TEXT = ("name", "headline")
_PROFILE_LINKS = ("linkedin", "github", "resume", "photo")


def profile() -> dict:
    """`data/profile.json` - who built this: name, headline, LinkedIn, GitHub, resume, photo.

    Optional (no file -> {}). Text is capped; a link is kept only if it is an http(s) URL, so a
    `javascript:` value in the file can never reach the page as a link."""
    path = _pack_dir() / "profile.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContextPackError(f"profile.json is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ContextPackError("profile.json must be a JSON object")
    out = {}
    for key in _PROFILE_TEXT:
        if isinstance(data.get(key), str) and data[key].strip():
            out[key] = data[key].strip()[:120]
    for key in _PROFILE_LINKS:
        value = data.get(key)
        if isinstance(value, str) and value.strip().lower().startswith(("https://", "http://")):
            out[key] = value.strip()[:500]
    return out


def week_architecture(week: int) -> dict:
    """The architecture slice for ONE week: the key decisions that cite it and every diagram
    line tagged with it - the same filter the read page uses, served to the model."""
    arch = _read("architecture.md")
    found = _DECISIONS.search(arch)
    decisions = [" ".join(d.split()) for d in
                 re.split(r"^\d+\. ", found.group(1) if found else "", flags=re.M)[1:]]
    diagrams = []
    for title, body in _DIAGRAM.findall(arch):
        lines = [line.strip() for line in body.split("\n") if week in weeks_in(line)]
        if lines:
            diagrams.append({"diagram": title, "lines": lines})
    return {"week": week, "decisions": [d for d in decisions if week in weeks_in(d)],
            "diagram_lines": diagrams}


def week_details(week: int) -> str | None:
    """`data/weeks/wNN.md` - the week in depth (what, how, decisions, results, stack), or None
    when the pack has no such file. Optional: a pack without the folder still works."""
    path = _pack_dir() / "weeks" / f"w{week:02d}.md"
    return path.read_text(encoding="utf-8") if path.exists() else None


def week_results(week: int) -> list[dict]:
    """That week's eval entries from eval_results.json (see `_results`)."""
    return _results(json.loads(_read("eval_results.json")), week)


def gates() -> dict:
    """The two gates the site shows: routing (offline, every run) and tool choice (live)."""
    evals = json.loads(_read("eval_results.json"))
    out = {}
    for key, label in (("portfolioagent_routing_gate", "routing"),
                       ("portfolioagent_tool_gate", "tool_choice")):
        g = evals.get(key)
        out[label] = ({"score": g.get("overall_score"), "cases": g.get("cases"),
                       "passed": g.get("passed")} if isinstance(g, dict)
                      else {"score": None, "cases": None, "passed": None})
    return out


def _results(evals: dict, week: int) -> list[dict]:
    """That week's eval entries as label/fields; Week 17 shows its own routing gate."""
    out = []
    for key, entry in evals.get("weekly", {}).items():
        if key.startswith(f"w{week:02d}_") and isinstance(entry, dict):
            out.append({"label": key[4:].replace("_", " "), "fields": entry})
    gate = evals.get("portfolioagent_routing_gate")
    if week == 17 and isinstance(gate, dict):
        cost = gate.get("cost", {})
        out.append({"label": "PortfolioAgent routing gate", "fields": {
            "set": f"{gate.get('cases')} frozen questions",
            "measured": f"routing accuracy {gate.get('overall_score')}, "
                        f"split {gate.get('metrics', {}).get('by_tier')}",
            "cost": f"routed ${cost.get('routed_total_usd')} vs all-frontier "
                    f"${cost.get('all_frontier_total_usd')}"}})
    tool_gate = evals.get("portfolioagent_tool_gate")
    if week == 17 and isinstance(tool_gate, dict):
        out.append({"label": "PortfolioAgent tool-choice gate", "fields": {
            "set": f"{tool_gate.get('cases')} frozen questions, run against the live model",
            "measured": f"{tool_gate.get('passed')}/{tool_gate.get('cases')} passed "
                        f"({tool_gate.get('overall_score')})"}})
    return out


def build_detail(week: int) -> dict | None:
    """One build's read page as data, or None for a week the pack does not list."""
    pack = get_portfolio()
    build = next((b for b in pack["builds"] if b["week"] == week), None)
    if build is None:
        return None
    arch = _read("architecture.md")
    evals = json.loads(_read("eval_results.json"))
    section = _DECISIONS.search(arch)
    decisions = [" ".join(d.split()) for d in
                 re.split(r"^\d+\. ", section.group(1) if section else "", flags=re.M)[1:]]
    diagrams = [{"title": t, "lines": body.rstrip("\n").split("\n")}
                for t, body in _DIAGRAM.findall(arch)]
    weeks = [b["week"] for b in pack["builds"]]
    i = weeks.index(week)
    return {
        "title": pack["title"], "build": build, "details": week_details(week),
        "results": _results(evals, week),
        "decisions": [d for d in decisions if week in weeks_in(d)],
        "diagrams": diagrams,
        "prev": pack["builds"][i - 1] if i > 0 else None,
        "next": pack["builds"][i + 1] if i + 1 < len(weeks) else None,
    }


# ── The read page: plain HTML, every pack string escaped ──────────────────────────

_STYLE = (
    "body{background:#0d1117;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,"
    "'Segoe UI',sans-serif;max-width:900px;margin:40px auto;padding:0 24px;line-height:1.7}"
    "h1{margin:4px 0 16px}h2{color:#58a6ff;border-bottom:1px solid #30363d;padding-bottom:6px;"
    "margin-top:32px}a{color:#58a6ff;text-decoration:none}a:hover{text-decoration:underline}"
    "code{background:#21262d;padding:2px 6px;border-radius:4px}"
    ".nav{display:flex;justify-content:space-between;gap:12px;font-size:14px}"
    ".week{background:#1f3a5f;color:#58a6ff;border:1px solid #1f4080;border-radius:6px;"
    "padding:1px 8px;font-size:13px;font-weight:700}"
    ".muted{color:#7d8590}table{border-collapse:collapse;width:100%;margin:8px 0 16px}"
    "th,td{border:1px solid #30363d;padding:8px 12px;text-align:left;vertical-align:top}"
    "th{background:#161b22;color:#7d8590;width:120px;font-weight:600}"
    "pre{background:#161b22;padding:16px;border-radius:8px;overflow-x:auto;font-size:12.5px;"
    "line-height:1.55}mark{background:#1f3a5f;color:#e6edf3;border-radius:3px}"
    "details{margin:10px 0}summary{cursor:pointer;color:#58a6ff}"
)


def _inline(text: str) -> str:
    """Escape, then the two inline marks the pack uses: **bold** and `code`."""
    out = html.escape(text)
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out)
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", out)


def _details(text: str) -> str:
    """A week file's light markdown as HTML: ## headings, - bullets, paragraphs; escaped."""
    out, items = [], []
    for line in text.splitlines()[1:]:          # the first line repeats the page title
        stripped = line.strip()
        if stripped.startswith("- "):
            items.append(f"<li>{_inline(stripped[2:])}</li>")
            continue
        if items:
            out.append("<ul>" + "".join(items) + "</ul>")
            items = []
        if stripped.startswith("## "):
            out.append(f"<h3>{html.escape(stripped[3:])}</h3>")
        elif stripped:
            out.append(f"<p>{_inline(stripped)}</p>")
    if items:
        out.append("<ul>" + "".join(items) + "</ul>")
    return "".join(out)


def _diagram(d: dict, week: int | None) -> str:
    rows = []
    for line in d["lines"]:
        row = html.escape(line)
        rows.append(f"<mark>{row}</mark>" if week is not None and week in weeks_in(line) else row)
    return f"<pre>{chr(10).join(rows)}</pre>"


def _field(value) -> str:
    if isinstance(value, list):
        return html.escape(", ".join(map(str, value)))
    if isinstance(value, dict):
        return html.escape(json.dumps(value))
    return html.escape(str(value))


def render_build_page(detail: dict) -> str:
    b, week = detail["build"], detail["build"]["week"]

    def link(other, arrow_first):
        if not other:
            return "<span></span>"
        label = f"W{other['week']} · {html.escape(other['name'])}"
        text = f"← {label}" if arrow_first else f"{label} →"
        return f"<a href='/portfolio/{other['week']}'>{text}</a>"

    results = "".join(
        f"<h3>{html.escape(r['label'])}</h3><table>" + "".join(
            f"<tr><th>{html.escape(str(k))}</th><td>{_field(v)}</td></tr>"
            for k, v in r["fields"].items() if not str(k).startswith("_")) + "</table>"
        for r in detail["results"]
    ) or "<p class='muted'>No eval run is recorded for this week in the pack.</p>"
    decisions = ("<ul>" + "".join(f"<li>{_inline(d)}</li>" for d in detail["decisions"])
                 + "</ul>") if detail["decisions"] else \
        "<p class='muted'>No key decision cites this week.</p>"
    tagged = [d for d in detail["diagrams"] if any(week in weeks_in(x) for x in d["lines"])]
    where = "".join(f"<h3>{html.escape(d['title'])}</h3>{_diagram(d, week)}" for d in tagged) \
        or "<p class='muted'>No diagram line is tagged with this week.</p>"
    every = "".join(f"<details><summary>{html.escape(d['title'])}</summary>{_diagram(d, None)}"
                    "</details>" for d in detail["diagrams"])
    title = html.escape(detail["title"])
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>W{week} {html.escape(b['name'])} - {title}</title><style>{_STYLE}</style>"
        "</head><body>"
        f"<div class='nav'><a href='/'>← All builds</a>"
        f"<span>{link(detail['prev'], True)} &nbsp; {link(detail['next'], False)}</span></div>"
        f"<p class='muted' style='margin-top:24px'>{title} · build history</p>"
        f"<h1><span class='week'>W{week}</span> {html.escape(b['name'])}</h1>"
        f"<h2>What I built</h2><p>{_inline(b['summary'][:1].upper() + b['summary'][1:])}</p>"
        + (f"<h2>In depth</h2>{_details(detail['details'])}" if detail.get("details") else "")
        +
        f"<h2>Results</h2>{results}"
        f"<h2>Design decisions</h2>{decisions}"
        "<h2>Where it sits in the architecture</h2>"
        "<p class='muted'>Lines tagged with this week are highlighted.</p>"
        f"{where}<h2>All architecture diagrams</h2>{every}"
        "<p class='muted' style='margin-top:32px'>From <code>data/AGENTS.md</code>, "
        "<code>architecture.md</code> and <code>eval_results.json</code> - the same pack the "
        "agent answers from.</p></body></html>"
    )
