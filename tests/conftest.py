"""Stub env so tests need no real key, no network. Real get_settings() then works
everywhere (health, agent card, context) reading these env vars. In-process stores
(memory, pending actions, guard counters) are reset between tests so each starts clean."""
import pytest

from app import config, guard, memory, tools


@pytest.fixture(autouse=True)
def fake_env(monkeypatch):
    # Never read the developer's .env or shell: a local FRONTIER_MODEL or ADMIN_TOKEN would
    # otherwise change what the suite asserts. Tests see the defaults plus what they set here.
    monkeypatch.setitem(config.Settings.model_config, "env_file", None)
    for name in [*config.Settings.model_fields, "RENDER_EXTERNAL_URL"]:
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("DATA_DIR", "data")
    monkeypatch.setenv("AGENT_BASE_URL", "http://localhost:8000")
    # A dead local address: if a stub is ever missed, the call fails fast instead of
    # reaching a real provider.
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9")
    config.get_settings.cache_clear()
    memory.reset()
    tools.reset_actions()
    guard.reset()
    yield
    config.get_settings.cache_clear()
    memory.reset()
    tools.reset_actions()
    guard.reset()
