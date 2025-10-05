# Cryptomancy – Product & Technical Spec

_Last updated: <REPLACE ME>_

## 1) Problem & Goal
Discover promising Uniswap V3 pools early and monitor them in near-real time. Provide:
- Fresh pool discovery (from factory events)
- Swap & liquidity activity ingestion
- Basic pricing (USD) + subgraph TVL/volume context
- A simple ranking (“top pools”) and pool detail views via API + web UI

Non-goals (for v0):
- Cross-chain support beyond Ethereum mainnet
- On-chain execution, trading, or alerting
- Deep risk modeling; only lightweight metrics now

## 2) Users & Jobs-To-Be-Done
- **DeFi researcher/trader:** Find new/active pools, scan recent swaps/liquidity, gauge momentum.
- **Bot builder / analyst:** Query a clean API for pools, swaps, mints/burns, and lightweight ranks.

## 3) High-Level Features
- **Pool Discovery:** Read Uniswap V3 `PoolCreated` from the factory; store `token0/1`, fee, tick_spacing, created_block.
- **Activity Ingestion:** For selected pools ingest `Swap`, `Mint`, `Burn` logs in block ranges with retry/throttle.
- **Pricing:** Resolve token USD price (Coingecko, cached). _(On-chain fallback TWAP: TODO)_
- **Metrics & Ranking:** Fuse swaps + pricing + subgraph TVL/volume to compute basic z-scored “top pools”.
- **API:** Public read endpoints for pools, swaps, liquidity, stats, health.
- **Web UI:** Top pools table with filters/sort; pool detail (latest swaps + recent liquidity changes).
- **Ops:** CORS + API-key gate for public reads, `.env` configuration, Cloudflare “quick tunnel” for remote testing.

## 4) Architecture
- **Backend:** FastAPI (Uvicorn), SQLAlchemy async, SQLite by default (`data/cryptomancy.db`).
- **Chain Access:** web3.py (Infura/Alchemy via `ETH_RPC_URL`).
- **Context/TVL/Volume:** The Graph Gateway (`GRAPH_API_KEY` + `UNISWAP_V3_SUBGRAPH_ID`) where available.
- **Pricing:** Coingecko (best-effort; 5-minute TTL cache).
- **Frontend:** Vite + React + Tailwind (consumes API; sends `X-API-Key` header).

## 5) Data Model (simplified)
- **tokens**: `chain_id`, `address (pk)`, `symbol`, `decimals`
- **pools**: `address (pk)`, `token0_addr`, `token1_addr`, `fee`, `tick_spacing`, `created_block`
- **swaps**: `tx_hash+log_index (pk)`, `pool_addr`, `block_number`, `block_time`, `sender`, `recipient`, `amount0`, `amount1`, `sqrt_price_x96`, `liquidity_after`, `tick`
- **liquidity_events**: `tx_hash+log_index (pk)`, `pool_addr`, `etype(mint|burn)`, `owner`, `tick_lower`, `tick_upper`, `amount_liquidity`, `amount0`, `amount1`, `block_number`, `block_time`
- **prices_hourly**: `token_addr`, `hour_ts`, `price_usd`
- **pool_stats**: period snapshots (h1/d1), blended metrics and ranks (TVL, volume, velocity, z-scores)

## 6) Ingestion Pipelines
### 6.1 Pool Discovery
- Query Uniswap V3 Factory `PoolCreated` logs in batches (`RPC_MAX_BATCH`) from `--from-block auto`.
- Manual upsert for specific pools supported (helper snippet).
- Throttling via `RPC_SLEEP_MS` to avoid 429s.

### 6.2 Pool Activity
- Chunked `getLogs` for target pool: `Swap`, `Mint`, `Burn` topics.
- Robust decoding (HexBytes / topic order), retries with backoff.
- Idempotent upserts for logs.

### 6.3 Pricing & Metrics
- Coingecko price map for top tokens; cached in memory & persisted hourly.
- Subgraph snapshots for TVL/volume where keys are present.
- `scripts/compute_metrics.py` folds the above into `pool_stats` with simple z-score ranks.

## 7) API Endpoints (public read)
- `GET /health` → `{ chain_id, latest_block, graph_status }`
- `GET /pools?limit=&offset=` → list pools (basic info)
- `GET /pools/{pool}/swaps?limit=&since_block=` → recent swaps
- `GET /pools/{pool}/liquidity?type=any|mint|burn&limit=&since_block=` → recent liquidity events
- `GET /stats/top?window=h1|d1&limit=&offset=` → ranked pools (after metrics job)
- All public reads accept `X-API-Key: $PUBLIC_READ_TOKEN`.

## 8) Configuration (env)
See `.env.example` for:
