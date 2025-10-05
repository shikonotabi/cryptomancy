# cryptomancy/api/config.py
from __future__ import annotations

from typing import List, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Load environment from .env at project root
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # tolerate unknown env keys
    )

    # App basics
    app_env: str = "codex"
    chain_id: int = 1
    database_url: str = "sqlite+aiosqlite:///./data/cryptomancy.db"
    data_dir: str = "./data"
    index_from_block: int | None = None

    # Required
    eth_rpc_url: str

    # Optional (The Graph)
    graph_api_key: str | None = None
    uniswap_v3_subgraph_id: str | None = None

    # Optional (pricing + metrics)
    coingecko_api_key: str | None = None

    # Tuning knobs
    metrics_refresh_seconds: int = 120
    rpc_logs_batch: int = 2000
    rpc_request_sleep_ms: int = 100
    enable_background_jobs: bool = True

    # ---------- NEW: API/Auth/CORS/limits ----------
    # Public read-only token; if set and enforcement is turned on, clients must send X-API-Key
    public_read_token: str | None = None
    # Admin API key for privileged endpoints (if/when you add them)
    admin_api_key: str | None = None
    # If true, require X-API-Key=PUBLIC_READ_TOKEN on public GET routes
    require_api_key_for_public: bool = False

    # CORS allow-list (comma-separated in .env or list). Example:
    # ALLOWED_ORIGINS=http://localhost:5173,https://your.site
    allowed_origins: List[str] = []
    # Optional regex to match ephemeral tunnel hosts, e.g. ^https://.*\.trycloudflare\.com$
    allowed_origin_regex: str | None = None
    cors_allow_credentials: bool = False
    cors_allow_headers: List[str] = ["*"]
    cors_allow_methods: List[str] = ["*"]

    # Simple global rate limit (requests/minute) that your ASGI middleware can read
    rate_limit_per_minute: int = 120
    # Default network timeout your code can reuse
    request_timeout_seconds: int = 15
    # ---------- /NEW ----------

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _split_csv(cls, v):
        if v is None or v == "":
            return []
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        if isinstance(v, (list, tuple)):
            return list(v)
        raise TypeError("allowed_origins must be a list or a comma-separated string")

    @property
    def subgraph_url(self) -> str | None:
        if self.graph_api_key and self.uniswap_v3_subgraph_id:
            return (
                f"https://gateway.thegraph.com/api/{self.graph_api_key}"
                f"/subgraphs/id/{self.uniswap_v3_subgraph_id}"
            )
        return None


settings = Settings()
