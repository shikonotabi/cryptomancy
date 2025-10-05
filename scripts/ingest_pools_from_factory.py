#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Iterable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from web3 import Web3
from web3.exceptions import BadFunctionCallOutput, ContractLogicError
from eth_utils import keccak, to_hex
from hexbytes import HexBytes

from api.config import settings
from api.db import Pool, Token, init_db, session_scope

FACTORY_ADDRESS = "0x1F98431c8aD98523631AE4a59f267346ea31F984"
POOL_CREATED_SIG = "PoolCreated(address,address,uint24,int24,address)"
TOPIC0: str = to_hex(keccak(text=POOL_CREATED_SIG))  # "0x" + 64 hex chars

ERC20_ABI = [
    {"constant": True, "inputs": [], "name": "symbol", "outputs": [{"name": "", "type": "string"}],
     "payable": False, "stateMutability": "view", "type": "function"},
    {"constant": True, "inputs": [], "name": "decimals", "outputs": [{"name": "", "type": "uint8"}],
     "payable": False, "stateMutability": "view", "type": "function"},
]

@dataclass(slots=True)
class UpsertStats:
    logs_total: int = 0
    pools_created: int = 0
    tokens_created: int = 0

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest Uniswap v3 pools from factory events")
    p.add_argument("--from-block", default="auto", help="Starting block number or 'auto'")
    p.add_argument("--to-block", default="latest", help="Ending block number or 'latest'")
    p.add_argument("--batch", type=int, default=2000, help="Block range per get_logs request")
    return p.parse_args()

def resolve_block_range(w3: Web3, start_arg: str, end_arg: str) -> Tuple[int, int]:
    latest_block = int(w3.eth.block_number)
    if start_arg == "auto":
        env_val = os.getenv("INDEX_FROM_BLOCK") or getattr(settings, "index_from_block", None)
        try:
            start_block = int(env_val) if env_val is not None else max(latest_block - 100_000, 0)
        except ValueError:
            logging.warning("Invalid INDEX_FROM_BLOCK=%s, falling back to latest-100000", env_val)
            start_block = max(latest_block - 100_000, 0)
    else:
        start_block = int(start_arg)
    end_block = latest_block if end_arg == "latest" else int(end_arg)
    if start_block > end_block:
        raise ValueError(f"from-block {start_block} > to-block {end_block}")
    return start_block, end_block

def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def normalize_symbol(value: object) -> str | None:
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            value = value.decode("latin-1", "ignore")
        value = value.rstrip("\x00")
    if isinstance(value, str):
        v = value.strip()
        return v or None
    return None

def fetch_token_metadata(w3: Web3, address: str) -> Tuple[str | None, int | None]:
    c = w3.eth.contract(address=address, abi=ERC20_ABI)
    symbol: str | None = None
    decimals: int | None = None
    try:
        symbol = normalize_symbol(c.functions.symbol().call())
    except (BadFunctionCallOutput, ContractLogicError, ValueError):
        symbol = None
    try:
        decimals = c.functions.decimals().call()
        if isinstance(decimals, bytes):
            decimals = int.from_bytes(decimals, "big", signed=False)
    except (BadFunctionCallOutput, ContractLogicError, ValueError):
        decimals = None
    return symbol, decimals

def _topic_hex(t) -> str:
    if isinstance(t, (bytes, bytearray, HexBytes)):
        return "0x" + bytes(t).hex()
    return str(t)

def _addr_from_topic(t) -> str:
    hx = _topic_hex(t)
    return Web3.to_checksum_address("0x" + hx[-40:])

def _int24_from_word_last3_signed(word: bytes) -> int:
    u = int.from_bytes(word[-3:], "big", signed=False)
    return u if u < (1 << 23) else u - (1 << 24)

def _uint24_from_topic_last3(t) -> int:
    b = bytes(HexBytes(_topic_hex(t)))
    return int.from_bytes(b[-3:], "big", signed=False)

def decode_pool_created(log: dict) -> Tuple[str, str, int, int, str]:
    """
    topics:
      [0] signature
      [1] token0 (indexed)
      [2] token1 (indexed)
      [3] fee    (indexed uint24)
    data (2 words = 64 bytes):
      word0: int24 tickSpacing (use last 3 bytes, signed)
      word1: address pool      (use last 20 bytes)
    """
    topics = log.get("topics") or []
    if len(topics) < 4:
        raise ValueError(f"Unexpected topics length: {len(topics)}")
    if _topic_hex(topics[0]).lower() != TOPIC0.lower():
        raise ValueError("Mismatched topic0 signature")

    token0 = _addr_from_topic(topics[1])
    token1 = _addr_from_topic(topics[2])
    fee    = _uint24_from_topic_last3(topics[3])

    data_bytes = bytes(HexBytes(log.get("data", "0x")))
    if len(data_bytes) < 64:
        raise ValueError(f"Unexpected data length {len(data_bytes)} for PoolCreated")

    w0 = data_bytes[0:32]
    w1 = data_bytes[32:64]

    tick_spacing = _int24_from_word_last3_signed(w0)
    pool = Web3.to_checksum_address("0x" + w1[-20:].hex())
    return token0, token1, fee, tick_spacing, pool

async def get_or_create_token(
    session: AsyncSession, w3: Web3, chain_id: int, address: str,
    meta_cache: Dict[str, Tuple[str | None, int | None]],
) -> Tuple[Token, bool]:
    checksum = Web3.to_checksum_address(address)
    result = await session.execute(select(Token).where(Token.chain_id == chain_id, Token.address == checksum))
    token = result.scalar_one_or_none()
    if token:
        return token, False
    if checksum not in meta_cache:
        meta_cache[checksum] = fetch_token_metadata(w3, checksum)
    symbol, decimals = meta_cache[checksum]
    token = Token(chain_id=chain_id, address=checksum, symbol=symbol, decimals=decimals)
    session.add(token)
    await session.flush()
    return token, True

async def upsert_pool(
    session: AsyncSession, chain_id: int, pool_address: str, token0: Token, token1: Token,
    fee: int, tick_spacing: int, created_block: int,
) -> bool:
    checksum = Web3.to_checksum_address(pool_address)
    result = await session.execute(select(Pool).where(Pool.chain_id == chain_id, Pool.address == checksum))
    pool = result.scalar_one_or_none()
    if pool:
        changed = False
        if pool.token0_id != token0.id: pool.token0_id = token0.id; changed = True
        if pool.token1_id != token1.id: pool.token1_id = token1.id; changed = True
        if pool.created_block is None:  pool.created_block = created_block; changed = True
        if pool.fee != fee:             pool.fee = fee; changed = True
        if pool.tick_spacing != tick_spacing: pool.tick_spacing = tick_spacing; changed = True
        return False
    session.add(Pool(
        chain_id=chain_id, address=checksum, token0_id=token0.id, token1_id=token1.id,
        fee=fee, tick_spacing=tick_spacing, created_block=created_block
    ))
    return True

def build_log_filter(factory_addr: str, start_block: int, end_block: int) -> dict:
    return {
        "address": Web3.to_checksum_address(factory_addr),
        "fromBlock": int(start_block),  # ints OK; Web3 hex-encodes
        "toBlock": int(end_block),
        "topics": [TOPIC0],
    }

async def process_batches(
    session: AsyncSession, w3: Web3, start_block: int, end_block: int, batch_size: int, stats: UpsertStats
) -> None:
    chain_id = settings.chain_id
    meta_cache: Dict[str, Tuple[str | None, int | None]] = {}
    current = start_block
    printed_dbg = False

    while current <= end_block:
        batch_end = min(current + batch_size - 1, end_block)
        params = build_log_filter(FACTORY_ADDRESS, current, batch_end)
        if not printed_dbg:
            logging.info("DBG eth_getLogs params: %s", params)
            printed_dbg = True

        logs = w3.eth.get_logs(params)
        if logs:
            logging.info("Processing %s logs between blocks %s-%s", len(logs), current, batch_end)

        for log in logs:
            stats.logs_total += 1
            token0_addr, token1_addr, fee, tick_spacing, pool_addr = decode_pool_created(log)

            token0, created0 = await get_or_create_token(session, w3, chain_id, token0_addr, meta_cache)
            token1, created1 = await get_or_create_token(session, w3, chain_id, token1_addr, meta_cache)
            if created0: stats.tokens_created += 1
            if created1: stats.tokens_created += 1

            if await upsert_pool(session, chain_id, pool_addr, token0, token1, fee, tick_spacing, log["blockNumber"]):
                stats.pools_created += 1

        current = batch_end + 1

async def async_main(args: argparse.Namespace) -> None:
    if not settings.eth_rpc_url:
        raise RuntimeError("ETH_RPC_URL is required")

    w3 = Web3(Web3.HTTPProvider(settings.eth_rpc_url, request_kwargs={"timeout": 20}))
    start_block, end_block = resolve_block_range(w3, args.from_block, args.to_block)
    logging.info("Fetching PoolCreated events from blocks %s to %s (batch size %s)", start_block, end_block, args.batch)

    await init_db()
    stats = UpsertStats()
    async with session_scope() as session:
        await process_batches(session, w3, start_block, end_block, args.batch, stats)

    logging.info("Done. Logs processed: %s, new pools: %s, new tokens: %s",
                 stats.logs_total, stats.pools_created, stats.tokens_created)

def main() -> None:
    setup_logging()
    args = parse_args()
    if args.batch <= 0:
        raise ValueError("--batch must be positive")
    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        logging.warning("Interrupted by user")

if __name__ == "__main__":
    main()
