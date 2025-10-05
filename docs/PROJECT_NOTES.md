# Cryptomancy – Working Notes

_Last updated: <REPLACE ME with date/time>_

## What this does (current status)
- **Backend (FastAPI)** at `http://127.0.0.1:8000`
- **SQLite DB** at `data/cryptomancy.db`
- **On-chain ingestion**:
  - `scripts/ingest_pools_from_factory.py` seeds Uniswap V3 pools from the factory.
  - `scripts/ingest_pool_activity.py` ingests swaps/mints/burns for specific pools.
- **Metrics** (hourly) + **pricing** (Coingecko, cached ~5m) – endpoints exposed.
- **Frontend** (Vite/React/Tailwind) under `/web` (talks to the API).

We’ve verified:
- Pool factory ingestion **works** (≈549 pools from recent blocks).
- Per-pool activity ingestion **works** (USDC/WETH 0.05% demo).
- API endpoints `/health`, `/pools`, `/pools/{address}/swaps`, `/pools/{address}/liquidity` **work** locally.
- Cloudflare “quick tunnel” can expose the local API to Codex (ephemeral URL).

---

## Quick start (local)

```bash
# 0) Clone and enter
git clone https://github.com/shikonotabi/cryptomancy
cd cryptomancy

# 1) Python env & deps
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -r requirements.txt

# 2) Create .env from example and fill in your keys
cp .env.example .env
# (edit .env with your editor)

# 3) Start API (bind to loopback for tunnel)
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

Health check:

curl http://127.0.0.1:8000/health


Seed recent pools (one-time or occasionally):

python -m scripts.ingest_pools_from_factory --from-block auto --batch 2000
sqlite3 data/cryptomancy.db 'select count(*) from pools;'


Ingest a high-volume demo pool (USDC/WETH 0.05%):

export POOL=0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640
python -m scripts.ingest_pool_activity --pool "$POOL" --since-hours 12 --batch 2000
curl "http://127.0.0.1:8000/pools/$POOL/swaps?limit=5"
curl "http://127.0.0.1:8000/pools/$POOL/liquidity?type=any&limit=5"

Expose to Codex (Option A: quick Cloudflare tunnel)

Ephemeral; changes each run.

In a separate terminal (keep API running):

./cloudflared tunnel --url http://localhost:8000
# copy the https://<something>.trycloudflare.com address it prints


In your .env, set:

ALLOWED_ORIGINS=http://localhost:5173
ALLOWED_ORIGIN_REGEX=^https://.*\.trycloudflare\.com$


Test the tunnel:

export TUNNEL=https://<your-subdomain>.trycloudflare.com
curl "$TUNNEL/health"
curl "$TUNNEL/pools?limit=5" -H "X-API-Key: $PUBLIC_READ_TOKEN"


If you see Cloudflare 1033 errors, wait a few seconds, ensure the API is running, or restart the tunnel to get a fresh URL.

Environment variables

Create .env from the template below:

# --- Chain + Subgraph ---
# Your Ethereum RPC (Infura/Alchemy/other)
ETH_RPC_URL=https://mainnet.infura.io/v3/<YOUR_PROJECT_ID>

# The Graph gateway (optional but used by /health + metrics)
GRAPH_API_KEY=<your_graph_gateway_key>
UNISWAP_V3_SUBGRAPH_ID=<subgraph_id_for_uniswap_v3>

# --- Database ---
# Default is SQLite under ./data, but explicit is safer:
DATABASE_URL=sqlite+aiosqlite:///./data/cryptomancy.db

# --- API / CORS ---
PUBLIC_READ_TOKEN=changeme-public             # This is safe-to-share (read-only gate)
ALLOWED_ORIGINS=http://localhost:5173
ALLOWED_ORIGIN_REGEX=^https://.*\.trycloudflare\.com$
ENABLE_BACKGROUND_JOBS=0                      # 0 while Codex tests; 1 for auto jobs

# --- Pricing ---
COINGECKO_API_KEY=                            # optional; free tier works unauth’d
PRICING_CACHE_TTL_SECONDS=300

# --- Ingestion throttles ---
RPC_SLEEP_MS=250
RPC_MAX_BATCH=2000


Notes:

PUBLIC_READ_TOKEN is treated as public; it just gates endpoints so random scrapers don’t hammer you.

Keep your real secrets (ETH_RPC_URL, GRAPH_API_KEY) out of version control; .env is in .gitignore.

Useful endpoints

GET /health → { chain_id, latest_block, graph_status }

GET /pools?limit=&offset= → basic pool list

GET /pools/{pool}/swaps?limit=&since_block= → recent swaps

GET /pools/{pool}/liquidity?type=any|mint|burn&limit= → recent mints/burns

GET /stats/top?window=h1|d1&limit= → pool rankings (after metrics computed)

All public reads accept X-API-Key: $PUBLIC_READ_TOKEN (recommended when exposed)

Frontend (optional local run)
cd web
# if nvm complains about prefix/globalconfig, do:
#   nvm use --delete-prefix v20 --silent
npm install
npm run dev   # runs on http://localhost:5173


The UI expects the API at http://127.0.0.1:8000 (or your tunnel URL) and sends the X-API-Key.

Jobs & metrics

Run the hourly metrics job manually (Codex can also call this):

python -m scripts.compute_metrics


Pricing CLI sanity:

python -m api.pricing --help
python -m api.graph --help

Common gotchas

Port already in use → fuser -k 8000/tcp || true and re-run uvicorn.

429 Too Many Requests (Infura) → lower RPC_MAX_BATCH and increase RPC_SLEEP_MS.

eth_getLogs "data types must start with 0x" → we now hex-format block ranges; fixed in scripts.

Cloudflare 1033 error → your quick tunnel died or isn’t yet ready; restart the tunnel, keep API running.

Multiple venvs → which python should resolve to .../cryptomancy/.venv/bin/python.

Security

.env is ignored by git (verified).

Only read endpoints are public; we gate with X-API-Key.

CORS is strict: local dev + *.trycloudflare.com.

Git workflow (for Codex / PRs)
git checkout -b codex-sync
git add -A
git commit -m "chore: sync for Codex testing"
git push origin codex-sync
# open PR or let Codex target this branch

Next tasks (handoff to Codex)

Use the tunnel URL and X-API-Key to hit:

/health

/pools?limit=25

/stats/top?window=h1&limit=25 (after running compute_metrics)

Exercise /pools/{pool}/swaps and /liquidity on an active pool (USDC/WETH 0.05%).

Build and point /web to the tunnel; verify UI table renders (filters, sorts).

Implement on-chain fallback in get_twap_price_usd (pricing placeholder).


---

# 2) Updated `.env.example`

If you don’t already have it in the repo root, create/replace with:

```dotenv
# --- Chain + Subgraph ---
ETH_RPC_URL=
GRAPH_API_KEY=
UNISWAP_V3_SUBGRAPH_ID=

# --- Database ---
DATABASE_URL=sqlite+aiosqlite:///./data/cryptomancy.db

# --- API / CORS ---
PUBLIC_READ_TOKEN=changeme-public
ALLOWED_ORIGINS=http://localhost:5173
ALLOWED_ORIGIN_REGEX=^https://.*\.trycloudflare\.com$
ENABLE_BACKGROUND_JOBS=0

# --- Pricing ---
COINGECKO_API_KEY=
PRICING_CACHE_TTL_SECONDS=300

# --- Ingestion throttles ---
RPC_SLEEP_MS=250
RPC_MAX_BATCH=2000
