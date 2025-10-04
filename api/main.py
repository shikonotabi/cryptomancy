# cryptomancy/api/main.py
from fastapi import Depends, FastAPI, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from web3 import Web3
import requests

from .config import settings
from .db import Pool, init_db, get_session

app = FastAPI()


@app.on_event("startup")
async def startup_event() -> None:
    await init_db()

@app.get("/health")
def health():
    w3 = Web3(Web3.HTTPProvider(settings.eth_rpc_url, request_kwargs={"timeout": 15}))
    out = {"chain_id": w3.eth.chain_id, "latest_block": w3.eth.block_number}
    if settings.subgraph_url:
        try:
            r = requests.post(settings.subgraph_url, json={"query":"{ _meta { hasIndexingErrors } }"}, timeout=15)
            out["graph_status"] = r.status_code
        except Exception as e:
            out["graph_error"] = str(e)
    else:
        out["graph_status"] = "not_configured"
    return out


@app.get("/pools")
async def list_pools(
    limit: int = Query(50, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
):
    stmt = (
        select(Pool)
        .options(selectinload(Pool.token0), selectinload(Pool.token1))
        .order_by(Pool.id.asc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    pools = result.scalars().all()

    payload = []
    for pool in pools:
        token0 = pool.token0
        token1 = pool.token1
        payload.append(
            {
                "address": pool.address,
                "fee": pool.fee,
                "tick_spacing": pool.tick_spacing,
                "created_block": pool.created_block,
                "token0": {
                    "address": token0.address if token0 else None,
                    "symbol": token0.symbol if token0 else None,
                    "decimals": token0.decimals if token0 else None,
                },
                "token1": {
                    "address": token1.address if token1 else None,
                    "symbol": token1.symbol if token1 else None,
                    "decimals": token1.decimals if token1 else None,
                },
                "tvl_usd": None,
                "volume_24h": None,
                "volatility_30d": None,
                "rho_pair": None,
                "score": None,
            }
        )

    return payload
