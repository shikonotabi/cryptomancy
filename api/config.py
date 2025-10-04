# cryptomancy/api/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Load environment from .env at project root
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # App basics
    app_env: str = "codex"
    chain_id: int = 1
    database_url: str = "sqlite+aiosqlite:///./data/cryptomancy.db"
    data_dir: str = "./data"

    # Required
    eth_rpc_url: str

    # Optional (The Graph)
    graph_api_key: str | None = None
    uniswap_v3_subgraph_id: str | None = None

    @property
    def subgraph_url(self) -> str | None:
        if self.graph_api_key and self.uniswap_v3_subgraph_id:
            return (
                f"https://gateway.thegraph.com/api/{self.graph_api_key}"
                f"/subgraphs/id/{self.uniswap_v3_subgraph_id}"
            )
        return None

settings = Settings()
