"""The signed Agent Card and the student's profile.

Offline: no model is called; the signing key is derived from a test seed.
"""
import json
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


def test_verify_card_says_false_to_a_hostile_card(monkeypatch):
    _settings(monkeypatch, CARD_SIGNING_SEED="seed")
    keys = client.get("/.well-known/jwks.json").json()
    for bad in ({"signatures": "x"},
                {"signatures": [{"protected": signing.b64url(b"[1]"), "signature": "AA"}]}):
        assert signing.verify_card(bad, keys) is False


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


@pytest.mark.parametrize("text,status", [(None, 200), ("{not json", 503), ("[]", 503)],
                         ids=["missing", "not_json", "not_an_object"])
def test_the_profile_is_optional_but_must_be_valid(tmp_path, monkeypatch, text, status):
    _pack_with_profile(tmp_path, monkeypatch, text)
    r = client.get("/portfolio")
    assert r.status_code == status
    if status == 200:
        assert r.json()["profile"] == {}
