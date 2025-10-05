# cryptomancy/api/main.py
from fastapi import Depends, FastAPI, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from web3 import Web3
import requests

from .config import settings
from .db import LiquidityEvent, Pool, Swap, init_db, get_session

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


async def _resolve_pool(
    session: AsyncSession,
    address: str,
) -> Pool:
    try:
        checksum = Web3.to_checksum_address(address)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid pool address") from exc

    stmt = select(Pool).where(Pool.chain_id == settings.chain_id, Pool.address == checksum)
    result = await session.execute(stmt)
    pool = result.scalar_one_or_none()
    if not pool:
        raise HTTPException(status_code=404, detail="Pool not found")
    return pool


@app.get("/pools/{address}/swaps")
async def pool_swaps(
    address: str,
    limit: int = Query(100, ge=1, le=1000),
    before_block: int | None = Query(None),
    after_block: int | None = Query(None),
    session: AsyncSession = Depends(get_session),
):
    pool = await _resolve_pool(session, address)

    stmt = select(Swap).where(Swap.pool_id == pool.id)
    if before_block is not None:
        stmt = stmt.where(Swap.block_number < before_block)
    if after_block is not None:
        stmt = stmt.where(Swap.block_number > after_block)
    stmt = stmt.order_by(Swap.block_number.desc(), Swap.log_index.desc()).limit(limit)

    result = await session.execute(stmt)
    rows = result.scalars().all()

    return [
        {
            "tx_hash": row.tx_hash,
            "log_index": row.log_index,
            "block_number": row.block_number,
            "block_time": row.block_time,
            "sender": row.sender,
            "recipient": row.recipient,
            "amount0": row.amount0,
            "amount1": row.amount1,
            "sqrt_price_x96": row.sqrt_price_x96,
            "liquidity_after": row.liquidity_after,
            "tick": row.tick,
        }
        for row in rows
    ]


@app.get("/pools/{address}/liquidity")
async def pool_liquidity_events(
    address: str,
    limit: int = Query(100, ge=1, le=1000),
    type: str = Query("any"),
    session: AsyncSession = Depends(get_session),
):
    pool = await _resolve_pool(session, address)

    event_type = type.lower()
    if event_type not in {"any", "mint", "burn"}:
        raise HTTPException(status_code=400, detail="Invalid liquidity event type")

    stmt = select(LiquidityEvent).where(LiquidityEvent.pool_id == pool.id)
    if event_type in {"mint", "burn"}:
        stmt = stmt.where(LiquidityEvent.etype == event_type)
    stmt = stmt.order_by(LiquidityEvent.block_number.desc(), LiquidityEvent.log_index.desc()).limit(limit)

    result = await session.execute(stmt)
    rows = result.scalars().all()

    return [
        {
            "tx_hash": row.tx_hash,
            "log_index": row.log_index,
            "block_number": row.block_number,
            "block_time": row.block_time,
            "etype": row.etype,
            "owner": row.owner,
            "tick_lower": row.tick_lower,
            "tick_upper": row.tick_upper,
            "amount_liquidity": row.amount_liquidity,
            "amount0": row.amount0,
            "amount1": row.amount1,
        }
        for row in rows
    ]
