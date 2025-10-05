#!/usr/bin/env python
"""Backfill Uniswap v3 pool Swap/Mint/Burn events into SQLite."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.ext.asyncio import AsyncSession
from web3 import Web3
from web3.exceptions import BlockNotFound

from api.config import settings
from api.db import LiquidityEvent, Pool, Swap, init_db, session_scope


SWAP_TOPIC = Web3.keccak(text="Swap(address,address,int256,int256,uint160,uint128,int24)").hex()
MINT_TOPIC = Web3.keccak(text="Mint(address,address,int24,int24,uint128,uint256,uint256)").hex()
BURN_TOPIC = Web3.keccak(text="Burn(address,int24,int24,uint128,uint256,uint256)").hex()


@dataclass(slots=True)
class BatchStats:
    swaps: int = 0
    mints: int = 0
    burns: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Uniswap v3 pool activity events")
    parser.add_argument(
        "--pool",
        dest="pools",
        action="append",
        required=True,
        help="Pool address to ingest (repeatable)",
    )
    parser.add_argument("--from-block", default="auto", help="Start block (int) or 'auto'")
    parser.add_argument("--to-block", default="latest", help="End block (int) or 'latest'")
    parser.add_argument(
        "--since-hours",
        type=int,
        default=None,
        help="Convenience shortcut: approximate starting block from hours back",
    )
    parser.add_argument("--batch", type=int, default=2000, help="Blocks per eth_getLogs batch")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def dedupe_preserve_order(seq: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for item in seq:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(item)
    return ordered


async def get_block_number_range(
    w3: Web3,
    start_arg: str,
    end_arg: str,
    since_hours: Optional[int],
) -> Tuple[int, int]:
    latest_block = await asyncio.to_thread(w3.eth.get_block, "latest")
    latest_number: int = latest_block["number"]

    if since_hours is not None:
        approx_blocks = max(int(since_hours * 3600 / 12), 1)
        start_block = max(latest_number - approx_blocks, 0)
    elif start_arg == "auto":
        env_val = os.getenv("INDEX_FROM_BLOCK") or (
            settings.index_from_block and str(settings.index_from_block)
        )
        start_block = None
        if env_val:
            try:
                start_block = int(env_val)
            except ValueError:
                logging.warning("Invalid INDEX_FROM_BLOCK=%s, defaulting to latest-100000", env_val)
        if start_block is None:
            start_block = max(latest_number - 100_000, 0)
    else:
        start_block = int(start_arg)

    if end_arg == "latest":
        end_block = latest_number
    else:
        end_block = int(end_arg)

    if start_block > end_block:
        raise ValueError(f"from-block {start_block} is greater than to-block {end_block}")

    return start_block, end_block


def topic_to_address(topic: bytes) -> str:
    hex_str = topic.hex()
    return Web3.to_checksum_address("0x" + hex_str[-40:])


def int256_from_bytes(data: bytes) -> int:
    value = int.from_bytes(data, "big", signed=False)
    return value if value < (1 << 255) else value - (1 << 256)


def uint_from_bytes(data: bytes) -> int:
    return int.from_bytes(data, "big", signed=False)


def int24_from_bytes(data: bytes) -> int:
    last3 = data[-3:]
    unsigned = int.from_bytes(last3, "big", signed=False)
    return unsigned if unsigned < (1 << 23) else unsigned - (1 << 24)


async def get_block_timestamp(
    w3: Web3,
    block_number: int,
    cache: Dict[int, int],
) -> int:
    if block_number in cache:
        return cache[block_number]
    try:
        block = await asyncio.to_thread(w3.eth.get_block, block_number)
    except BlockNotFound:
        logging.warning("Block %s not found, defaulting timestamp to 0", block_number)
        cache[block_number] = 0
        return 0
    timestamp = int(block["timestamp"])
    cache[block_number] = timestamp
    return timestamp


async def get_logs_with_retry(w3: Web3, params: Dict, label: str, attempts: int = 3) -> List[Dict]:
    for attempt in range(1, attempts + 1):
        try:
            logging.debug("eth_getLogs params (%s): %s", label, params)
            return await asyncio.to_thread(w3.eth.get_logs, params)
        except Exception as exc:  # broad: RPC errors vary
            logging.warning(
                "eth_getLogs failed for %s attempt %s/%s: %s",
                label,
                attempt,
                attempts,
                exc,
            )
            await asyncio.sleep(min(2 ** attempt, 10))
    logging.error("eth_getLogs giving up for %s after %s attempts", label, attempts)
    return []


def decode_swap_log(log: Dict) -> Dict[str, str | int | None]:
    topics: List[bytes] = log["topics"]
    data_bytes = bytes.fromhex(log["data"][2:])
    words = [data_bytes[i : i + 32] for i in range(0, len(data_bytes), 32)]

    amount0 = int256_from_bytes(words[0])
    amount1 = int256_from_bytes(words[1])
    sqrt_price = uint_from_bytes(words[2])
    liquidity = uint_from_bytes(words[3][-16:])
    tick = int24_from_bytes(words[4])

    sender = topic_to_address(topics[1]) if len(topics) > 1 else None
    recipient = topic_to_address(topics[2]) if len(topics) > 2 else None

    return {
        "sender": sender,
        "recipient": recipient,
        "amount0": str(amount0),
        "amount1": str(amount1),
        "sqrt_price_x96": str(sqrt_price),
        "liquidity_after": str(liquidity),
        "tick": tick,
    }


def decode_mint_log(log: Dict) -> Dict[str, str | int]:
    topics: List[bytes] = log["topics"]
    data_bytes = bytes.fromhex(log["data"][2:])
    words = [data_bytes[i : i + 32] for i in range(0, len(data_bytes), 32)]

    owner = topic_to_address(topics[1]) if len(topics) > 1 else "0x0000000000000000000000000000000000000000"
    tick_lower = int24_from_bytes(topics[2]) if len(topics) > 2 else 0
    tick_upper = int24_from_bytes(topics[3]) if len(topics) > 3 else 0

    amount = uint_from_bytes(words[1][-16:])
    amount0 = uint_from_bytes(words[2])
    amount1 = uint_from_bytes(words[3])

    return {
        "etype": "mint",
        "owner": owner,
        "tick_lower": tick_lower,
        "tick_upper": tick_upper,
        "amount_liquidity": str(amount),
        "amount0": str(amount0),
        "amount1": str(amount1),
    }


def decode_burn_log(log: Dict) -> Dict[str, str | int]:
    topics: List[bytes] = log["topics"]
    data_bytes = bytes.fromhex(log["data"][2:])
    words = [data_bytes[i : i + 32] for i in range(0, len(data_bytes), 32)]

    owner = topic_to_address(topics[1]) if len(topics) > 1 else "0x0000000000000000000000000000000000000000"
    tick_lower = int24_from_bytes(topics[2]) if len(topics) > 2 else 0
    tick_upper = int24_from_bytes(topics[3]) if len(topics) > 3 else 0

    amount = uint_from_bytes(words[0][-16:])
    amount0 = uint_from_bytes(words[1])
    amount1 = uint_from_bytes(words[2])

    return {
        "etype": "burn",
        "owner": owner,
        "tick_lower": tick_lower,
        "tick_upper": tick_upper,
        "amount_liquidity": str(amount),
        "amount0": str(amount0),
        "amount1": str(amount1),
    }


async def resolve_pools(
    session: AsyncSession,
    addresses: Iterable[str],
    chain_id: int,
) -> List[Pool]:
    ordered = dedupe_preserve_order(list(addresses))
    checksummed = [Web3.to_checksum_address(addr) for addr in ordered]
    if not checksummed:
        return []

    stmt = select(Pool).where(Pool.chain_id == chain_id, Pool.address.in_(checksummed))
    result = await session.execute(stmt)
    rows = result.scalars().all()
    mapping = {row.address: row for row in rows}

    resolved: List[Pool] = []
    for addr in checksummed:
        pool = mapping.get(addr)
        if pool:
            resolved.append(pool)
        else:
            logging.warning("Pool %s not found in database; skipping", addr)

    return resolved


async def process_chunk(
    session: AsyncSession,
    w3: Web3,
    pool: Pool,
    start_block: int,
    end_block: int,
    timestamp_cache: Dict[int, int],
) -> BatchStats:
    stats = BatchStats()
    base_params = {
        "address": pool.address,
        "fromBlock": start_block,
        "toBlock": end_block,
    }

    swap_logs = await get_logs_with_retry(w3, {**base_params, "topics": [SWAP_TOPIC]}, "Swap")
    mint_logs = await get_logs_with_retry(w3, {**base_params, "topics": [MINT_TOPIC]}, "Mint")
    burn_logs = await get_logs_with_retry(w3, {**base_params, "topics": [BURN_TOPIC]}, "Burn")

    if swap_logs:
        logging.info(
            "Processing %s swap logs for %s blocks %s-%s",
            len(swap_logs),
            pool.address,
            start_block,
            end_block,
        )
        swap_rows = []
        for log in swap_logs:
            decoded = decode_swap_log(log)
            block_number = int(log["blockNumber"])
            block_time = await get_block_timestamp(w3, block_number, timestamp_cache)
            swap_rows.append(
                {
                    "chain_id": settings.chain_id,
                    "pool_id": pool.id,
                    "tx_hash": log["transactionHash"].hex(),
                    "log_index": int(log["logIndex"]),
                    "block_number": block_number,
                    "block_time": block_time,
                    **decoded,
                }
            )
        stmt = insert(Swap).values(swap_rows).on_conflict_do_nothing(index_elements=["tx_hash", "log_index"])
        await session.execute(stmt)
        stats.swaps += len(swap_rows)

    liq_rows: List[Dict] = []

    if mint_logs:
        logging.info(
            "Processing %s mint logs for %s blocks %s-%s",
            len(mint_logs),
            pool.address,
            start_block,
            end_block,
        )
        for log in mint_logs:
            decoded = decode_mint_log(log)
            block_number = int(log["blockNumber"])
            block_time = await get_block_timestamp(w3, block_number, timestamp_cache)
            liq_rows.append(
                {
                    "chain_id": settings.chain_id,
                    "pool_id": pool.id,
                    "tx_hash": log["transactionHash"].hex(),
                    "log_index": int(log["logIndex"]),
                    "block_number": block_number,
                    "block_time": block_time,
                    **decoded,
                }
            )
        stats.mints += len(mint_logs)

    if burn_logs:
        logging.info(
            "Processing %s burn logs for %s blocks %s-%s",
            len(burn_logs),
            pool.address,
            start_block,
            end_block,
        )
        for log in burn_logs:
            decoded = decode_burn_log(log)
            block_number = int(log["blockNumber"])
            block_time = await get_block_timestamp(w3, block_number, timestamp_cache)
            liq_rows.append(
                {
                    "chain_id": settings.chain_id,
                    "pool_id": pool.id,
                    "tx_hash": log["transactionHash"].hex(),
                    "log_index": int(log["logIndex"]),
                    "block_number": block_number,
                    "block_time": block_time,
                    **decoded,
                }
            )
        stats.burns += len(burn_logs)

    if liq_rows:
        stmt = (
            insert(LiquidityEvent)
            .values(liq_rows)
            .on_conflict_do_nothing(index_elements=["tx_hash", "log_index"])
        )
        await session.execute(stmt)

    return stats


async def ingest_pool(
    session: AsyncSession,
    w3: Web3,
    pool: Pool,
    start_block: int,
    end_block: int,
    batch_size: int,
    timestamp_cache: Dict[int, int],
) -> None:
    current = start_block
    while current <= end_block:
        chunk_end = min(current + batch_size - 1, end_block)
        stats = await process_chunk(session, w3, pool, current, chunk_end, timestamp_cache)
        await session.commit()
        if stats.swaps or stats.mints or stats.burns:
            logging.info(
                "Pool %s [%s-%s] swaps=%s mints=%s burns=%s",
                pool.address,
                current,
                chunk_end,
                stats.swaps,
                stats.mints,
                stats.burns,
            )
        current = chunk_end + 1


async def async_main(args: argparse.Namespace) -> None:
    if not settings.eth_rpc_url:
        raise RuntimeError("ETH_RPC_URL must be configured")
    if args.batch <= 0:
        raise ValueError("--batch must be positive")

    w3 = Web3(Web3.HTTPProvider(settings.eth_rpc_url, request_kwargs={"timeout": 30}))
    start_block, end_block = await get_block_number_range(
        w3, args.from_block, args.to_block, args.since_hours
    )

    logging.info(
        "Ingesting pool activity from blocks %s to %s (batch size %s)",
        start_block,
        end_block,
        args.batch,
    )

    await init_db()

    async with session_scope() as session:
        pools = await resolve_pools(session, args.pools, settings.chain_id)
        if not pools:
            logging.error("No pools resolved; exiting")
            return

        timestamp_cache: Dict[int, int] = {}
        for pool in pools:
            logging.info("Processing pool %s", pool.address)
            await ingest_pool(session, w3, pool, start_block, end_block, args.batch, timestamp_cache)


def main() -> None:
    setup_logging()
    args = parse_args()
    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        logging.warning("Interrupted")


if __name__ == "__main__":
    main()
