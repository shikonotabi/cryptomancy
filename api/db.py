# cryptomancy/api/db.py
"""Database models and helpers for the Cryptomancy API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from sqlalchemy import (
    BigInteger,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    UniqueConstraint,
    inspect,
    text,
)
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from .config import settings


class Base(DeclarativeBase):
    pass


class Token(Base):
    __tablename__ = "tokens"
    __table_args__ = (UniqueConstraint("chain_id", "address", name="uq_token_chain_address"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chain_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    address: Mapped[str] = mapped_column(String(64), nullable=False)
    symbol: Mapped[str | None] = mapped_column(String(32))
    decimals: Mapped[int | None] = mapped_column(Integer)

    pools_token0: Mapped[list[Pool]] = relationship(
        "Pool",
        back_populates="token0",
        foreign_keys="Pool.token0_id",
    )
    pools_token1: Mapped[list[Pool]] = relationship(
        "Pool",
        back_populates="token1",
        foreign_keys="Pool.token1_id",
    )


class Pool(Base):
    __tablename__ = "pools"
    __table_args__ = (
        UniqueConstraint("chain_id", "address", name="uq_pool_chain_address"),
        Index("ix_pools_token0_token1", "token0_id", "token1_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chain_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    address: Mapped[str] = mapped_column(String(64), nullable=False)
    token0_id: Mapped[int] = mapped_column(ForeignKey("tokens.id"), nullable=False)
    token1_id: Mapped[int] = mapped_column(ForeignKey("tokens.id"), nullable=False)
    fee: Mapped[int] = mapped_column(Integer, nullable=False)
    tick_spacing: Mapped[int] = mapped_column(Integer, nullable=False)
    created_block: Mapped[int | None] = mapped_column(BigInteger)

    token0: Mapped[Token] = relationship("Token", foreign_keys=[token0_id], back_populates="pools_token0")
    token1: Mapped[Token] = relationship("Token", foreign_keys=[token1_id], back_populates="pools_token1")
    swaps: Mapped[list[Swap]] = relationship("Swap", back_populates="pool", cascade="all, delete-orphan")
    liquidity_events: Mapped[list[LiquidityEvent]] = relationship(
        "LiquidityEvent", back_populates="pool", cascade="all, delete-orphan"
    )
    stats_daily: Mapped[list[PoolStatsDaily]] = relationship(
        "PoolStatsDaily", back_populates="pool", cascade="all, delete-orphan"
    )


class Swap(Base):
    __tablename__ = "swaps"
    __table_args__ = (
        UniqueConstraint("tx_hash", "log_index", name="uq_swaps_tx_log"),
        Index("ix_swaps_pool_block", "pool_id", "block_number"),
        Index("ix_swaps_pool_time", "pool_id", "block_time"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chain_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    pool_id: Mapped[int] = mapped_column(ForeignKey("pools.id"), nullable=False)
    tx_hash: Mapped[str] = mapped_column(String(66), nullable=False)
    log_index: Mapped[int] = mapped_column(Integer, nullable=False)
    block_number: Mapped[int] = mapped_column(BigInteger, nullable=False)
    block_time: Mapped[int] = mapped_column(Integer, nullable=False)
    sender: Mapped[str | None] = mapped_column(String(42))
    recipient: Mapped[str | None] = mapped_column(String(42))
    amount0: Mapped[str] = mapped_column(String, nullable=False)
    amount1: Mapped[str] = mapped_column(String, nullable=False)
    sqrt_price_x96: Mapped[str] = mapped_column(String, nullable=False)
    liquidity_after: Mapped[str] = mapped_column(String, nullable=False)
    tick: Mapped[int] = mapped_column(Integer, nullable=False)

    pool: Mapped[Pool] = relationship("Pool", back_populates="swaps")


class LiquidityEvent(Base):
    __tablename__ = "liquidity_events"
    __table_args__ = (
        UniqueConstraint("tx_hash", "log_index", name="uq_liquidity_tx_log"),
        Index("ix_liquidity_pool_block", "pool_id", "block_number"),
        Index("ix_liquidity_pool_time", "pool_id", "block_time"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chain_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    pool_id: Mapped[int] = mapped_column(ForeignKey("pools.id"), nullable=False)
    tx_hash: Mapped[str] = mapped_column(String(66), nullable=False)
    log_index: Mapped[int] = mapped_column(Integer, nullable=False)
    block_number: Mapped[int] = mapped_column(BigInteger, nullable=False)
    block_time: Mapped[int] = mapped_column(Integer, nullable=False)
    etype: Mapped[str] = mapped_column(String(8), nullable=False)
    owner: Mapped[str] = mapped_column(String(42), nullable=False)
    tick_lower: Mapped[int] = mapped_column(Integer, nullable=False)
    tick_upper: Mapped[int] = mapped_column(Integer, nullable=False)
    amount_liquidity: Mapped[str] = mapped_column(String, nullable=False)
    amount0: Mapped[str] = mapped_column(String, nullable=False)
    amount1: Mapped[str] = mapped_column(String, nullable=False)

    pool: Mapped[Pool] = relationship("Pool", back_populates="liquidity_events")


class Price(Base):
    __tablename__ = "prices"
    __table_args__ = (Index("ix_prices_token_time", "token_id", "time"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    time: Mapped[DateTime] = mapped_column(DateTime, nullable=False, index=True)
    token_id: Mapped[int] = mapped_column(ForeignKey("tokens.id"), nullable=False)
    price_usd: Mapped[float] = mapped_column(Float, nullable=False)

    token: Mapped[Token] = relationship("Token")


class PoolStatsDaily(Base):
    __tablename__ = "pool_stats_daily"
    __table_args__ = (UniqueConstraint("pool_id", "day", name="uq_pool_stats_day"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    pool_id: Mapped[int] = mapped_column(ForeignKey("pools.id"), nullable=False)
    day: Mapped[Date] = mapped_column(Date, nullable=False)
    tvl_usd: Mapped[float | None] = mapped_column(Float)
    volume_usd: Mapped[float | None] = mapped_column(Float)
    volatility_30d: Mapped[float | None] = mapped_column(Float)
    rho_pair: Mapped[float | None] = mapped_column(Float)

    pool: Mapped[Pool] = relationship("Pool", back_populates="stats_daily")


class SimResult(Base):
    __tablename__ = "sim_results"

    request_hash: Mapped[str] = mapped_column(String(128), primary_key=True)
    summary_json: Mapped[dict | None] = mapped_column(JSON)

    series: Mapped[list[SimSeries]] = relationship("SimSeries", back_populates="result", cascade="all, delete-orphan")


class SimSeries(Base):
    __tablename__ = "sim_series"
    __table_args__ = (Index("ix_sim_series_request_time", "request_hash", "t"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    request_hash: Mapped[str] = mapped_column(ForeignKey("sim_results.request_hash"), nullable=False)
    t: Mapped[DateTime] = mapped_column(DateTime, nullable=False)
    lp_value_usd: Mapped[float | None] = mapped_column(Float)
    hodl_value_usd: Mapped[float | None] = mapped_column(Float)
    fees_usd_cum: Mapped[float | None] = mapped_column(Float)
    price: Mapped[float | None] = mapped_column(Float)

    result: Mapped[SimResult] = relationship("SimResult", back_populates="series")


class Portfolio(Base):
    __tablename__ = "portfolios"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)

    items: Mapped[list[PortfolioItem]] = relationship("PortfolioItem", back_populates="portfolio", cascade="all, delete-orphan")


class PortfolioItem(Base):
    __tablename__ = "portfolio_items"
    __table_args__ = (Index("ix_portfolio_items_portfolio", "portfolio_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id"), nullable=False)
    sim_request_id: Mapped[str] = mapped_column(ForeignKey("sim_results.request_hash"), nullable=False)
    weight: Mapped[float] = mapped_column(Float, nullable=False)

    portfolio: Mapped[Portfolio] = relationship("Portfolio", back_populates="items")
    sim_result: Mapped[SimResult] = relationship("SimResult")


_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _ensure_data_dir(database_url: str) -> None:
    url = make_url(database_url)
    if not url.database or url.database == ":memory:":
        return
    if not url.drivername.startswith("sqlite"):
        return
    db_path = Path(url.database)
    if not db_path.is_absolute():
        db_path = Path.cwd() / db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)


def _prepare_schema(sync_conn) -> None:
    inspector = inspect(sync_conn)

    expected_swaps = {
        "id",
        "chain_id",
        "pool_id",
        "tx_hash",
        "log_index",
        "block_number",
        "block_time",
        "sender",
        "recipient",
        "amount0",
        "amount1",
        "sqrt_price_x96",
        "liquidity_after",
        "tick",
    }
    expected_liquidity = {
        "id",
        "chain_id",
        "pool_id",
        "tx_hash",
        "log_index",
        "block_number",
        "block_time",
        "etype",
        "owner",
        "tick_lower",
        "tick_upper",
        "amount_liquidity",
        "amount0",
        "amount1",
    }

    existing_tables = set(inspector.get_table_names())

    if "swaps" in existing_tables:
        swap_cols = {col["name"] for col in inspector.get_columns("swaps")}
        if swap_cols != expected_swaps:
            sync_conn.execute(text("DROP TABLE swaps"))

    if "liquidity_events" in existing_tables:
        liq_cols = {col["name"] for col in inspector.get_columns("liquidity_events")}
        if liq_cols != expected_liquidity:
            sync_conn.execute(text("DROP TABLE liquidity_events"))

    Base.metadata.create_all(sync_conn)


async def init_db(database_url: str | None = None) -> AsyncEngine:
    """Initialize the async engine, create tables, and return the engine."""

    global _engine, _session_factory

    if _engine is not None:
        return _engine

    resolved_url = database_url or settings.database_url
    _ensure_data_dir(resolved_url)

    _engine = create_async_engine(resolved_url, echo=False, future=True)
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)

    async with _engine.begin() as conn:
        await conn.run_sync(_prepare_schema)

    return _engine


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")

    session = _session_factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def get_session() -> AsyncIterator[AsyncSession]:
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    async with _session_factory() as session:
        yield session
