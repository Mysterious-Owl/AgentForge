"""Typed settings, loaded once.

All configuration goes through this module. No `os.environ` reads anywhere
else in the codebase. Missing required env vars fail loudly at startup.

Two model tiers are pinned here, one per routing decision (dated ids, NEVER
`-latest`). The routing thesis of the capstone is that most questions do not
need the frontier model - a small model answers the cheap majority and
only reasoning-heavy or mutating requests escalate. Pricing lives here too, so
`eval_run.py` and the cost ceiling are computed from a single source of truth.
"""
from __future__ import annotations
from functools import lru_cache

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Single source of truth for runtime configuration.

    Loaded from a `.env` file in the project root. See `.env.example`
    for the complete list of variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # --- Provider key (required - process exits loud on a missing key) ---
    openai_api_key: str

    # --- Where the calls go ---
    # OPENAI_BASE_URL moves EVERY call (a proxy, a gateway). Empty = the hosted API.
    openai_base_url: str | None = None
    # The open-source path: SMALL_BASE_URL moves only the SMALL tier to a local
    # OpenAI-compatible server (Qwen via Ollama / vLLM); the frontier stays on the vendor.
    #   SMALL_BASE_URL=http://localhost:11434/v1
    #   SMALL_MODEL=qwen3:8b
    small_base_url: str | None = None

    # --- Model pins, one per routing tier (dated ids, NEVER `-latest`) ---
    # SMALL is the cheap tier that answers the easy majority (a local open model can take it).
    # FRONTIER is the stronger model, reserved for questions that need reasoning.
    small_model: str = "gpt-5.4-nano-2026-03-17"
    frontier_model: str = "gpt-5.4-mini-2026-03-17"

    # --- Cost rates (USD per 1M tokens) - the vendor's published list price ---
    # INPUT and OUTPUT are priced separately - the same list as Week 16's CostGuard. The
    # frontier tier is 3.75x the small tier on input ($0.75 vs $0.20) and 3.6x on output;
    # routing the easy majority down is what turns that gap into a smaller BILL.
    small_input_cost_per_1m: float = 0.20       # gpt-5.4-nano
    small_output_cost_per_1m: float = 1.25
    frontier_input_cost_per_1m: float = 0.75    # gpt-5.4-mini (the frontier tier)
    frontier_output_cost_per_1m: float = 4.50

    # --- Classifier thresholds (deterministic routing key, see app/classifier.py) ---
    simple_max_chars: int = 160          # short + no reasoning cue -> answer on the small tier
    route_up_threshold: float = 0.60     # below this confidence, escalate to frontier (the valve)

    # --- The context pack (the agent's brain) ---
    data_dir: str = "data"

    # --- Agent Card identity (A2A discovery) ---
    agent_name: str = "agentforge-portfolio"
    agent_base_url: str = Field("http://localhost:8000", validation_alias=AliasChoices(
        "AGENT_BASE_URL", "RENDER_EXTERNAL_URL"))   # Render sets RENDER_EXTERNAL_URL itself
    # A2A protocol version: stamped on the card's interface (`protocolVersion`) and the
    # `A2A-Version` header of every response; /a2a refuses a client asking for another.
    a2a_protocol_version: str = "1.0"
    # Signs the Agent Card (app/signing.py): its SHA-256 is the P-256 key. Render generates it
    # (render.yaml); unset = the card is served unsigned.
    card_signing_seed: str | None = None

    # --- The approval gate (Tool layer) ---
    # Every mutating tool waits for an explicit human YES before it executes.
    # Approval is state the system owns, not a sentence the user can assert.
    approval_required: bool = True

    # --- The two caps (app/budget.py enforces BOTH; /health surfaces BOTH) ---
    max_iterations: int = 8            # bounded agent loop - no runaway re-planning
    cost_ceiling_usd: float = 0.05     # per-request projected-cost ceiling, pre-flight
    model_timeout_s: float = 30.0      # one model call; tenacity retries it on a timeout

    # --- Public-deploy guards (app/guard.py) - a shared URL spends YOUR key ---
    # ADMIN_TOKEN puts the routes that decide or read private state (/approve, /audit,
    # /memory) behind `Authorization: Bearer <token>`. Unset = open: localhost only.
    admin_token: str | None = None
    rate_limit_per_minute: int = 30    # POSTs per client per minute; 0 switches it off
    daily_budget_usd: float = 1.00     # model spend per UTC day, all callers together
    trust_forwarded_for: bool = False  # behind Render's proxy: key on the forwarded client IP
    # Above the UI's 1.26M-character oversized demo, so that one is refused by the COST
    # CEILING (its lesson); far below Render Free's 512 MB.
    max_body_bytes: int = 2 * 1024 * 1024
    render: bool = False               # Render sets RENDER=true on every service

    # --- Eval gate (eval_run.py) ---
    eval_score_floor: float = 0.90       # min overall score on the frozen golden set
    eval_golden_path: str = "data/eval_golden.jsonl"
    eval_results_path: str = "data/eval_results.json"

    log_level: str = "INFO"

    @field_validator("agent_base_url")
    @classmethod
    def _no_trailing_slash(cls, url: str) -> str:
        """`https://x.onrender.com/` must not advertise `https://x.onrender.com//a2a`."""
        return url.strip().rstrip("/")

    @model_validator(mode="after")
    def check_deploy_guards(self) -> "Settings":
        """On Render the URL is public - refuse to boot with the gate's routes left open."""
        if self.render and not self.admin_token:
            raise ValueError("ADMIN_TOKEN must be set when deployed (RENDER=true)")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the (effectively singleton) settings object."""
    return Settings()
