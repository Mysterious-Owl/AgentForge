"""The signed Agent Card, the owner's /admin page, and the student's profile.

Offline: no model is called; the signing key is derived from a test seed.
"""
import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import config, signing
from app.main import app

client = TestClient(app)
ROOT = Path(__file__).resolve().parent.parent


def _settings(monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    config.get_settings.cache_clear()


# ---------- the signed Agent Card ----------

def test_without_a_seed_the_card_is_unsigned_and_the_key_set_empty():
    assert "signatures" not in client.get("/.well-known/agent-card.json").json()
    assert client.get("/.well-known/jwks.json").json() == {"keys": []}
    assert client.get("/health").json()["card_signed"] is False


def test_a_signed_card_verifies_against_the_published_key(monkeypatch):
    _settings(monkeypatch, CARD_SIGNING_SEED="test-seed")
    card = client.get("/.well-known/agent-card.json").json()
    keys = client.get("/.well-known/jwks.json").json()
    assert signing.verify_card(card, keys)
    header = json.loads(signing.b64url_decode(card["signatures"][0]["protected"]))
    assert header == {"alg": "ES256", "jku": "http://localhost:8000/.well-known/jwks.json",
                      "kid": keys["keys"][0]["kid"], "typ": "JOSE"}
    assert keys["keys"][0]["kty"] == "EC" and keys["keys"][0]["crv"] == "P-256"
    assert client.get("/health").json()["card_signed"] is True


def test_a_changed_card_or_another_key_fails_verification(monkeypatch):
    _settings(monkeypatch, CARD_SIGNING_SEED="test-seed")
    card = client.get("/.well-known/agent-card.json").json()
    keys = client.get("/.well-known/jwks.json").json()
    tampered = {**card, "supportedInterfaces": [{"url": "https://evil.example/a2a",
                                                "protocolBinding": "JSONRPC",
                                                "protocolVersion": "1.0"}]}
    assert not signing.verify_card(tampered, keys)          # a redirected interface is caught
    _settings(monkeypatch, CARD_SIGNING_SEED="someone-else")
    foreign = client.get("/.well-known/jwks.json").json()["keys"][0]
    same_kid = {"keys": [{**foreign, "kid": keys["keys"][0]["kid"]}]}   # right kid, wrong key
    assert not signing.verify_card(card, same_kid)                      # the maths says no
    assert not signing.verify_card({k: v for k, v in card.items() if k != "signatures"}, keys)


def test_the_legacy_alias_serves_the_same_signed_bytes(monkeypatch):
    _settings(monkeypatch, CARD_SIGNING_SEED="test-seed")
    assert client.get("/.well-known/agent.json").content == \
        client.get("/.well-known/agent-card.json").content


def test_the_ui_verifies_the_signature_in_the_browser():
    page = (ROOT / "index.html").read_text(encoding="utf-8")
    verify = page[page.index("async function verifyCard"):]
    assert "crypto.subtle.verify" in verify and "P-256" in verify
    assert "new URL(header.jku).pathname" in verify          # the key the header names


# ---------- the owner's page ----------

def test_the_admin_page_holds_the_owner_tools_and_the_public_page_does_not():
    admin = client.get("/admin")
    assert admin.status_code == 200 and "text/html" in admin.headers["content-type"]
    for needle in ('id="owner-token"', "'/actions'", "'/approve'", "/audit/"):
        assert needle in admin.text, needle
    public = client.get("/").text
    assert 'id="owner-token"' not in public and "/actions" not in public
    assert 'href="/admin"' in public and 'data-decide' not in public


def test_the_admin_page_is_harmless_without_the_token(monkeypatch):
    _settings(monkeypatch, ADMIN_TOKEN="t0ken")
    assert client.get("/admin").status_code == 200           # the page is just a token box
    assert client.get("/actions").status_code == 401         # its data still needs the token


# ---------- the student's profile ----------

def test_the_profile_is_served_with_the_build_cards():
    shipped = json.loads((ROOT / "data" / "profile.json").read_text(encoding="utf-8"))
    profile = client.get("/portfolio").json()["profile"]
    assert profile["name"] == shipped["name"]                 # whatever the student wrote
    assert all(v.startswith(("https://", "http://")) for k, v in profile.items()
               if k in ("linkedin", "github", "resume", "photo"))


def _pack_with_profile(tmp_path, monkeypatch, text):
    for name in ("AGENTS.md", "architecture.md", "eval_results.json"):
        (tmp_path / name).write_text((ROOT / "data" / name).read_text(encoding="utf-8"),
                                     encoding="utf-8")
    if text is not None:
        (tmp_path / "profile.json").write_text(text, encoding="utf-8")
    _settings(monkeypatch, DATA_DIR=str(tmp_path))


def test_only_http_links_reach_the_page(tmp_path, monkeypatch):
    _pack_with_profile(tmp_path, monkeypatch, json.dumps({
        "name": "<b>Ada</b>", "linkedin": "javascript:alert(1)", "github": "https://github.com/a",
        "photo": "data:image/png;base64,AAAA", "resume": "  HTTPS://x.io/cv.pdf "}))
    profile = client.get("/portfolio").json()["profile"]
    assert profile == {"name": "<b>Ada</b>", "github": "https://github.com/a",
                       "resume": "HTTPS://x.io/cv.pdf"}


@pytest.mark.parametrize("text,status", [(None, 200), ("{not json", 503)])
def test_the_profile_is_optional_but_must_be_valid(tmp_path, monkeypatch, text, status):
    _pack_with_profile(tmp_path, monkeypatch, text)
    r = client.get("/portfolio")
    assert r.status_code == status
    if status == 200:
        assert r.json()["profile"] == {}


def test_the_header_sets_the_name_as_text_never_as_html():
    page = (ROOT / "index.html").read_text(encoding="utf-8")
    render = page[page.index("function renderProfile"):page.index("function aboutLine")]
    assert "textContent = `${p.name} · AgentForge`" in render and "innerHTML" not in render
    about = page[page.index("function aboutLine"):page.index("function renderBuilds")]
    assert "escHtml(p.name)" in about and "escHtml(p[k])" in about
    assert re.search(r"rel=\"noopener\"", about)
