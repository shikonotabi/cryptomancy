# api/config.py
from __future__ import annotations

import json
import re
from typing import List, Optional, Pattern

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _parse_origins(raw: Optional[str]) -> List[str]:
    """
    Accepts:
      "*"                  -> ["*"]
      '["http://a", ...]'  -> JSON list
      "http://a,http://b"  -> split on comma/semicolon/space
      "" or None           -> ["*"]
      "http://a"           -> ["http://a"]
    """
    if raw is None:
        return ["*"]
    s = raw.strip()
    if not s:
        return ["*"]
    if s in {"*", '"*"', "'*'"}:
        return ["*"]
    # JSON list?
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass
    # Comma/semicolon/space separated
    for sep in (",", ";", " "):
        if sep in s:
            parts = [p.strip() for p in s.split(sep) if p.strip()]
            if parts:
                return parts
    return [s]


class Settings(BaseSettings):
    # App / bind
    env: str = Field(default="dev", alias="ENV")
    host: str = Field(default="127.0.0.1", alias="HOST")  # bind locally (Cloudflare tunnel will proxy)
    port: int = Field(default=8000, alias="PORT")

    # Chain / RPC
    chain_id: int = Field(default=1, alias="CHAIN_ID")
    eth_rpc_url: str = Field(default="http://127.0.0.1:8545", alias="ETH_RPC_URL")

    # The Graph
    graph_api_key: Optional[str] = Field(default=None, alias="GRAPH_API_KEY")
    uniswap_v3_subgraph_id: Optional[str] = Field(default=None, alias="UNISWAP_V3_SUBGRAPH_ID")

    # Database (async SQLAlchemy URL)
    database_url: str = Field(default="sqlite+aiosqlite:///./data/cryptomancy.db", alias="DATABASE_URL")

    # API auth
    public_read_token: str = Field(default="dev-public", alias="PUBLIC_READ_TOKEN")
    admin_token: str = Field(default="dev-admin", alias="ADMIN_TOKEN")

    # CORS
    allowed_origins_raw: Optional[str] = Field(default="*", alias="ALLOWED_ORIGINS")
    allowed_origin_regex: Optional[str] = Field(default=None, alias="ALLOWED_ORIGIN_REGEX")

    # Feature flags / jobs
    pricing_enabled: bool = Field(default=True, alias="PRICING_ENABLED")
    metrics_enabled: bool = Field(default=True, alias="METRICS_ENABLED")
    metrics_interval_minutes: int = Field(default=60, alias="METRICS_INTERVAL_MINUTES")

    # Pricing sources
    coingecko_base_url: str = Field(default="https://api.coingecko.com/api/v3", alias="COINGECKO_BASE_URL")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- computed helpers ----
    @computed_field  # type: ignore[misc]
    @property
    def allowed_origins(self) -> List[str]:
        return _parse_origins(self.allowed_origins_raw)

    @computed_field  # type: ignore[misc]
    @property
    def allowed_origin_pattern(self) -> Optional[Pattern[str]]:
        return re.compile(self.allowed_origin_regex) if self.allowed_origin_regex else None

    @computed_field  # type: ignore[misc]
    @property
    def cors_kwargs(self) -> dict:
        return {
            "allow_origins": self.allowed_origins or ["*"],
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }

    # ---- back-compat for older main.py ----
    @property
    def enable_background_jobs(self) -> bool:
        # Treat “background jobs enabled” as “pricing OR metrics enabled”
        return bool(self.pricing_enabled or self.metrics_enabled)


settings = Settings()
