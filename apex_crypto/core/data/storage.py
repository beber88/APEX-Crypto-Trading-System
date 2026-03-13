"""Database interface module for the APEX crypto trading system.

Provides the StorageManager class that handles all database operations
across three backends:
    - TimescaleDB (via psycopg2) for time-series market data
    - SQLite for the local trade log
    - Redis for real-time signal cache and position state

Usage::

    config = {
        "timescaledb_url": "postgresql://user:pass@localhost:5432/apex",
        "sqlite_path": "/data/trades.db",
        "redis_url": "redis://localhost:6379/0",
    }
    with StorageManager(config) as storage:
        storage.store_ohlcv("BTCUSDT", "1h", df)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import psycopg2
import psycopg2.extras
import psycopg2.pool
import redis

logger = logging.getLogger(__name__)


def _json_log(level: str, msg: str, **kwargs: Any) -> None:
    """Emit a structured JSON log line.

    Args:
        level: Log level string (debug, info, warning, error).
        msg: Human-readable message.
        **kwargs: Arbitrary key-value pairs attached to the log entry.
    """
    payload = {
        "ts": datetime.now(tz=timezone.utc).isoformat(),
        "level": level,
        "msg": msg,
        **kwargs,
    }
    getattr(logger, level)(json.dumps(payload, default=str))


# ---------------------------------------------------------------------------
# SQL constants
# ---------------------------------------------------------------------------

_TIMESCALE_INIT_SQL = """
CREATE TABLE IF NOT EXISTS ohlcv (
    symbol      TEXT        NOT NULL,
    timeframe   TEXT        NOT NULL,
    timestamp   TIMESTAMPTZ NOT NULL,
    open        DOUBLE PRECISION NOT NULL,
    high        DOUBLE PRECISION NOT NULL,
    low         DOUBLE PRECISION NOT NULL,
    close       DOUBLE PRECISION NOT NULL,
    volume      DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (symbol, timeframe, timestamp)
);

CREATE TABLE IF NOT EXISTS funding_rates (
    symbol          TEXT        NOT NULL,
    timestamp       TIMESTAMPTZ NOT NULL,
    rate            DOUBLE PRECISION NOT NULL,
    predicted_rate  DOUBLE PRECISION,
    PRIMARY KEY (symbol, timestamp)
);

CREATE TABLE IF NOT EXISTS liquidations (
    id          BIGSERIAL PRIMARY KEY,
    symbol      TEXT            NOT NULL,
    timestamp   TIMESTAMPTZ     NOT NULL,
    side        TEXT            NOT NULL,
    quantity    DOUBLE PRECISION NOT NULL,
    price       DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS sentiment (
    id          BIGSERIAL PRIMARY KEY,
    symbol      TEXT            NOT NULL,
    timestamp   TIMESTAMPTZ     NOT NULL,
    source      TEXT            NOT NULL,
    score       DOUBLE PRECISION NOT NULL,
    raw_data    JSONB
);

CREATE TABLE IF NOT EXISTS regime (
    id          BIGSERIAL PRIMARY KEY,
    symbol      TEXT            NOT NULL,
    timestamp   TIMESTAMPTZ     NOT NULL,
    timeframe   TEXT            NOT NULL,
    regime      TEXT            NOT NULL,
    confidence  DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS open_interest (
    symbol      TEXT            NOT NULL,
    timestamp   TIMESTAMPTZ     NOT NULL,
    oi_value    DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (symbol, timestamp)
);

CREATE TABLE IF NOT EXISTS equity_snapshots (
    id                  BIGSERIAL PRIMARY KEY,
    timestamp           TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    total_equity        DOUBLE PRECISION NOT NULL,
    available_balance   DOUBLE PRECISION NOT NULL,
    unrealized_pnl      DOUBLE PRECISION NOT NULL,
    positions_count     INTEGER          NOT NULL
);

CREATE TABLE IF NOT EXISTS signals (
    id          BIGSERIAL PRIMARY KEY,
    timestamp   TIMESTAMPTZ     NOT NULL,
    symbol      TEXT            NOT NULL,
    strategy    TEXT            NOT NULL,
    timeframe   TEXT            NOT NULL,
    score       DOUBLE PRECISION NOT NULL,
    direction   TEXT            NOT NULL,
    metadata    JSONB
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_sym_tf_ts
    ON ohlcv (symbol, timeframe, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_funding_sym_ts
    ON funding_rates (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_liquidations_sym_ts
    ON liquidations (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_sentiment_sym_ts
    ON sentiment (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_regime_sym_ts
    ON regime (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_open_interest_sym_ts
    ON open_interest (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_equity_ts
    ON equity_snapshots (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signals_sym_ts
    ON signals (symbol, timestamp DESC);
"""

_SQLITE_INIT_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    trade_id        TEXT PRIMARY KEY,
    symbol          TEXT NOT NULL,
    strategy        TEXT NOT NULL,
    direction       TEXT NOT NULL,
    entry_price     REAL,
    exit_price      REAL,
    quantity         REAL,
    entry_time      TEXT,
    exit_time       TEXT,
    status          TEXT NOT NULL DEFAULT 'open',
    pnl             REAL,
    pnl_pct         REAL,
    r_multiple      REAL,
    fees            REAL DEFAULT 0,
    stop_loss       REAL,
    take_profit     REAL,
    timeframe       TEXT,
    metadata        TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_trades_status ON trades (status);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades (symbol);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades (strategy);
CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades (entry_time);
"""


class StorageManager:
    """Unified database interface for the APEX trading system.

    Manages connections to TimescaleDB, SQLite, and Redis, exposing
    domain-specific methods for storing and retrieving market data,
    trade records, and real-time state.

    Args:
        config: Dictionary containing connection parameters.
            Required keys:
                - ``timescaledb_url`` (str): PostgreSQL connection string.
                - ``sqlite_path`` (str): Path to the SQLite database file.
                - ``redis_url`` (str): Redis connection URL.
            Optional keys:
                - ``pg_pool_min`` (int): Minimum pool connections (default 2).
                - ``pg_pool_max`` (int): Maximum pool connections (default 10).

    Example::

        config = {
            "timescaledb_url": "postgresql://user:pw@localhost/apex",
            "sqlite_path": "./trades.db",
            "redis_url": "redis://localhost:6379/0",
        }
        sm = StorageManager(config)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._pg_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None
        self._sqlite_conn: Optional[sqlite3.Connection] = None
        self._redis: Optional[redis.Redis] = None

        self._init_timescaledb()
        self._init_sqlite()
        self._init_redis()

        _json_log("info", "StorageManager initialised")

    # ------------------------------------------------------------------
    # Connection initialisation
    # ------------------------------------------------------------------

    def _init_timescaledb(self) -> None:
        """Create the TimescaleDB connection pool and ensure schema exists."""
        url = self._config["timescaledb_url"]
        pool_min = self._config.get("pg_pool_min", 2)
        pool_max = self._config.get("pg_pool_max", 10)

        self._pg_pool = psycopg2.pool.ThreadedConnectionPool(
            pool_min, pool_max, dsn=url
        )
        _json_log("info", "TimescaleDB pool created", min=pool_min, max=pool_max)

        conn = self._pg_pool.getconn()
        try:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(_TIMESCALE_INIT_SQL)
            _json_log("info", "TimescaleDB schema verified")
        finally:
            self._pg_pool.putconn(conn)

    def _init_sqlite(self) -> None:
        """Open (or create) the SQLite trade-log database."""
        path = self._config["sqlite_path"]
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        self._sqlite_conn = sqlite3.connect(path, check_same_thread=False)
        self._sqlite_conn.row_factory = sqlite3.Row
        self._sqlite_conn.execute("PRAGMA journal_mode=WAL")
        self._sqlite_conn.execute("PRAGMA foreign_keys=ON")
        self._sqlite_conn.executescript(_SQLITE_INIT_SQL)
        self._sqlite_conn.commit()
        _json_log("info", "SQLite trade log ready", path=path)

    def _init_redis(self) -> None:
        """Connect to Redis."""
        url = self._config["redis_url"]
        self._redis = redis.Redis.from_url(url, decode_responses=True)
        self._redis.ping()
        _json_log("info", "Redis connected", url=url)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "StorageManager":
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        self.close()

    def close(self) -> None:
        """Cleanly shut down all database connections.

        Closes the TimescaleDB connection pool, SQLite connection,
        and Redis client. Safe to call multiple times.
        """
        if self._pg_pool is not None:
            self._pg_pool.closeall()
            self._pg_pool = None
            _json_log("info", "TimescaleDB pool closed")

        if self._sqlite_conn is not None:
            self._sqlite_conn.close()
            self._sqlite_conn = None
            _json_log("info", "SQLite connection closed")

        if self._redis is not None:
            self._redis.close()
            self._redis = None
            _json_log("info", "Redis connection closed")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _pg_cursor(self, commit: bool = True):
        """Yield a TimescaleDB cursor from the pool.

        Args:
            commit: Whether to commit the transaction on success.

        Yields:
            A ``psycopg2`` cursor instance.

        Raises:
            RuntimeError: If the connection pool is not available.
        """
        if self._pg_pool is None:
            raise RuntimeError("TimescaleDB pool is not initialised")
        conn = self._pg_pool.getconn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                yield cur
                if commit:
                    conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pg_pool.putconn(conn)

    # ------------------------------------------------------------------
    # TimescaleDB — OHLCV
    # ------------------------------------------------------------------

    def store_ohlcv(
        self, symbol: str, timeframe: str, data: pd.DataFrame
    ) -> int:
        """Bulk-insert OHLCV candle data into TimescaleDB.

        Duplicate rows (same symbol/timeframe/timestamp) are silently
        skipped via ``ON CONFLICT DO NOTHING``.

        Args:
            symbol: Trading pair identifier (e.g. ``"BTCUSDT"``).
            timeframe: Candle interval (e.g. ``"1h"``, ``"4h"``).
            data: DataFrame with columns
                ``[timestamp, open, high, low, close, volume]``.
                The ``timestamp`` column must be timezone-aware or
                Unix-epoch milliseconds.

        Returns:
            Number of rows actually inserted.
        """
        if data.empty:
            return 0

        required = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

        rows = [
            (
                symbol,
                timeframe,
                row["timestamp"],
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(row["volume"]),
            )
            for _, row in data.iterrows()
        ]

        sql = """
            INSERT INTO ohlcv (symbol, timeframe, timestamp, open, high, low, close, volume)
            VALUES %s
            ON CONFLICT (symbol, timeframe, timestamp) DO NOTHING
        """
        with self._pg_cursor() as cur:
            psycopg2.extras.execute_values(cur, sql, rows, page_size=1000)
            inserted = cur.rowcount

        _json_log(
            "info",
            "Stored OHLCV data",
            symbol=symbol,
            timeframe=timeframe,
            offered=len(rows),
            inserted=inserted,
        )
        return inserted

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Retrieve OHLCV candle data from TimescaleDB.

        Args:
            symbol: Trading pair identifier.
            timeframe: Candle interval.
            start: Inclusive lower bound on timestamp.
            end: Inclusive upper bound on timestamp.
            limit: Maximum number of rows to return (most recent first
                before re-sorting).

        Returns:
            DataFrame with columns
            ``[timestamp, open, high, low, close, volume]`` sorted by
            timestamp ascending.
        """
        clauses = ["symbol = %s", "timeframe = %s"]
        params: list[Any] = [symbol, timeframe]

        if start is not None:
            clauses.append("timestamp >= %s")
            params.append(start)
        if end is not None:
            clauses.append("timestamp <= %s")
            params.append(end)

        where = " AND ".join(clauses)
        sql = f"""
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv
            WHERE {where}
            ORDER BY timestamp ASC
        """
        if limit is not None:
            sql += " LIMIT %s"
            params.append(limit)

        with self._pg_cursor(commit=False) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        if not rows:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        df = pd.DataFrame(rows)
        _json_log(
            "debug",
            "Retrieved OHLCV data",
            symbol=symbol,
            timeframe=timeframe,
            rows=len(df),
        )
        return df

    def get_latest_timestamp(
        self, symbol: str, timeframe: str
    ) -> Optional[datetime]:
        """Return the most recent OHLCV timestamp for a symbol/timeframe.

        Useful for incremental data downloads — fetch only candles newer
        than the returned value.

        Args:
            symbol: Trading pair identifier.
            timeframe: Candle interval.

        Returns:
            The latest ``datetime`` if data exists, otherwise ``None``.
        """
        sql = """
            SELECT MAX(timestamp) AS latest
            FROM ohlcv
            WHERE symbol = %s AND timeframe = %s
        """
        with self._pg_cursor(commit=False) as cur:
            cur.execute(sql, (symbol, timeframe))
            row = cur.fetchone()

        latest = row["latest"] if row else None
        _json_log(
            "debug",
            "Latest timestamp lookup",
            symbol=symbol,
            timeframe=timeframe,
            latest=str(latest),
        )
        return latest

    # ------------------------------------------------------------------
    # TimescaleDB — Funding rates
    # ------------------------------------------------------------------

    def store_funding_rate(
        self,
        symbol: str,
        timestamp: datetime,
        rate: float,
        predicted_rate: Optional[float] = None,
    ) -> None:
        """Insert a single funding-rate observation.

        Args:
            symbol: Trading pair identifier.
            timestamp: Funding event timestamp.
            rate: Realised funding rate.
            predicted_rate: Next predicted funding rate (may be ``None``).
        """
        sql = """
            INSERT INTO funding_rates (symbol, timestamp, rate, predicted_rate)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (symbol, timestamp) DO NOTHING
        """
        with self._pg_cursor() as cur:
            cur.execute(sql, (symbol, timestamp, rate, predicted_rate))

        _json_log(
            "debug",
            "Stored funding rate",
            symbol=symbol,
            timestamp=str(timestamp),
            rate=rate,
        )

    def get_funding_rates(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Retrieve funding-rate history for a symbol.

        Args:
            symbol: Trading pair identifier.
            start: Inclusive lower bound on timestamp.
            end: Inclusive upper bound on timestamp.

        Returns:
            DataFrame with columns
            ``[timestamp, rate, predicted_rate]`` sorted ascending.
        """
        clauses = ["symbol = %s"]
        params: list[Any] = [symbol]

        if start is not None:
            clauses.append("timestamp >= %s")
            params.append(start)
        if end is not None:
            clauses.append("timestamp <= %s")
            params.append(end)

        where = " AND ".join(clauses)
        sql = f"""
            SELECT timestamp, rate, predicted_rate
            FROM funding_rates
            WHERE {where}
            ORDER BY timestamp ASC
        """
        with self._pg_cursor(commit=False) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        if not rows:
            return pd.DataFrame(columns=["timestamp", "rate", "predicted_rate"])

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # TimescaleDB — Liquidations
    # ------------------------------------------------------------------

    def store_liquidation(
        self,
        symbol: str,
        timestamp: datetime,
        side: str,
        quantity: float,
        price: float,
    ) -> None:
        """Record a liquidation event.

        Args:
            symbol: Trading pair identifier.
            timestamp: When the liquidation occurred.
            side: ``"buy"`` or ``"sell"``.
            quantity: Size of the liquidated position.
            price: Price at which the liquidation was executed.
        """
        sql = """
            INSERT INTO liquidations (symbol, timestamp, side, quantity, price)
            VALUES (%s, %s, %s, %s, %s)
        """
        with self._pg_cursor() as cur:
            cur.execute(sql, (symbol, timestamp, side, quantity, price))

        _json_log(
            "debug",
            "Stored liquidation",
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
        )

    # ------------------------------------------------------------------
    # TimescaleDB — Sentiment
    # ------------------------------------------------------------------

    def store_sentiment(
        self,
        symbol: str,
        timestamp: datetime,
        source: str,
        score: float,
        raw_data: Optional[dict[str, Any]] = None,
    ) -> None:
        """Store a sentiment observation.

        Args:
            symbol: Trading pair identifier.
            timestamp: Observation timestamp.
            source: Data source label (e.g. ``"twitter"``, ``"news"``).
            score: Normalised sentiment score (typically -1.0 to 1.0).
            raw_data: Optional raw payload stored as JSONB.
        """
        sql = """
            INSERT INTO sentiment (symbol, timestamp, source, score, raw_data)
            VALUES (%s, %s, %s, %s, %s)
        """
        raw_json = json.dumps(raw_data, default=str) if raw_data else None
        with self._pg_cursor() as cur:
            cur.execute(sql, (symbol, timestamp, source, score, raw_json))

        _json_log(
            "debug",
            "Stored sentiment",
            symbol=symbol,
            source=source,
            score=score,
        )

    # ------------------------------------------------------------------
    # TimescaleDB — Regime
    # ------------------------------------------------------------------

    def store_regime(
        self,
        symbol: str,
        timestamp: datetime,
        timeframe: str,
        regime: str,
        confidence: float,
    ) -> None:
        """Record a detected market regime.

        Args:
            symbol: Trading pair identifier.
            timestamp: Detection timestamp.
            timeframe: Candle interval the regime was derived from.
            regime: Regime label (e.g. ``"trending"``, ``"ranging"``).
            confidence: Model confidence in the classification (0.0-1.0).
        """
        sql = """
            INSERT INTO regime (symbol, timestamp, timeframe, regime, confidence)
            VALUES (%s, %s, %s, %s, %s)
        """
        with self._pg_cursor() as cur:
            cur.execute(sql, (symbol, timestamp, timeframe, regime, confidence))

        _json_log(
            "debug",
            "Stored regime",
            symbol=symbol,
            regime=regime,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # TimescaleDB — Equity snapshots
    # ------------------------------------------------------------------

    def store_equity_snapshot(
        self,
        total_equity: float,
        available_balance: float,
        unrealized_pnl: float,
        positions_count: int,
    ) -> None:
        """Persist a point-in-time equity snapshot.

        Args:
            total_equity: Total account equity (USDT).
            available_balance: Free margin / available balance.
            unrealized_pnl: Sum of unrealised PnL across positions.
            positions_count: Number of open positions.
        """
        sql = """
            INSERT INTO equity_snapshots
                (total_equity, available_balance, unrealized_pnl, positions_count)
            VALUES (%s, %s, %s, %s)
        """
        with self._pg_cursor() as cur:
            cur.execute(
                sql,
                (total_equity, available_balance, unrealized_pnl, positions_count),
            )

        _json_log(
            "info",
            "Stored equity snapshot",
            total_equity=total_equity,
            positions_count=positions_count,
        )

    # ------------------------------------------------------------------
    # TimescaleDB — Signals
    # ------------------------------------------------------------------

    def store_signal(
        self,
        timestamp: datetime,
        symbol: str,
        strategy: str,
        timeframe: str,
        score: float,
        direction: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Persist a trading signal for audit/analysis.

        Args:
            timestamp: Signal generation timestamp.
            symbol: Trading pair identifier.
            strategy: Strategy name that produced the signal.
            timeframe: Candle interval used by the strategy.
            score: Signal strength / conviction score.
            direction: ``"long"`` or ``"short"``.
            metadata: Arbitrary signal details stored as JSONB.
        """
        meta_json = json.dumps(metadata, default=str) if metadata else None
        sql = """
            INSERT INTO signals
                (timestamp, symbol, strategy, timeframe, score, direction, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        with self._pg_cursor() as cur:
            cur.execute(
                sql,
                (timestamp, symbol, strategy, timeframe, score, direction, meta_json),
            )

        _json_log(
            "info",
            "Stored signal",
            symbol=symbol,
            strategy=strategy,
            direction=direction,
            score=score,
        )

    # ------------------------------------------------------------------
    # SQLite — Trade log
    # ------------------------------------------------------------------

    def record_trade(self, trade: dict[str, Any]) -> str:
        """Insert a new trade record into the SQLite trade log.

        Args:
            trade: Dictionary with trade details. Must include at minimum
                ``trade_id``, ``symbol``, ``strategy``, and ``direction``.

        Returns:
            The ``trade_id`` of the inserted record.

        Raises:
            ValueError: If required keys are missing.
        """
        required = {"trade_id", "symbol", "strategy", "direction"}
        missing = required - set(trade.keys())
        if missing:
            raise ValueError(f"Trade dict missing required keys: {missing}")

        columns = [
            "trade_id", "symbol", "strategy", "direction",
            "entry_price", "exit_price", "quantity",
            "entry_time", "exit_time", "status",
            "pnl", "pnl_pct", "r_multiple", "fees",
            "stop_loss", "take_profit", "timeframe", "metadata",
        ]

        present_cols = [c for c in columns if c in trade]
        placeholders = ", ".join("?" for _ in present_cols)
        col_names = ", ".join(present_cols)
        values = []
        for c in present_cols:
            val = trade[c]
            if c == "metadata" and isinstance(val, dict):
                val = json.dumps(val, default=str)
            values.append(val)

        sql = f"INSERT INTO trades ({col_names}) VALUES ({placeholders})"
        self._sqlite_conn.execute(sql, values)
        self._sqlite_conn.commit()

        _json_log(
            "info",
            "Recorded trade",
            trade_id=trade["trade_id"],
            symbol=trade["symbol"],
            strategy=trade["strategy"],
            direction=trade["direction"],
        )
        return trade["trade_id"]

    def update_trade(self, trade_id: str, updates: dict[str, Any]) -> None:
        """Update an existing trade record.

        Args:
            trade_id: Identifier of the trade to update.
            updates: Key-value pairs to set. Keys must correspond to
                column names in the ``trades`` table.
        """
        if not updates:
            return

        updates["updated_at"] = datetime.now(tz=timezone.utc).isoformat()

        set_parts = []
        values = []
        for key, val in updates.items():
            set_parts.append(f"{key} = ?")
            if key == "metadata" and isinstance(val, dict):
                val = json.dumps(val, default=str)
            values.append(val)

        values.append(trade_id)
        sql = f"UPDATE trades SET {', '.join(set_parts)} WHERE trade_id = ?"
        self._sqlite_conn.execute(sql, values)
        self._sqlite_conn.commit()

        _json_log("info", "Updated trade", trade_id=trade_id, fields=list(updates.keys()))

    def get_open_trades(self) -> list[dict[str, Any]]:
        """Return all trades with status ``'open'``.

        Returns:
            List of trade dictionaries.
        """
        cur = self._sqlite_conn.execute(
            "SELECT * FROM trades WHERE status = 'open' ORDER BY entry_time ASC"
        )
        rows = cur.fetchall()
        trades = [self._sqlite_row_to_dict(r) for r in rows]
        _json_log("debug", "Fetched open trades", count=len(trades))
        return trades

    def get_trade_history(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve historical trade records with optional filters.

        Args:
            start: ISO-format lower bound on ``entry_time``.
            end: ISO-format upper bound on ``entry_time``.
            symbol: Filter by trading pair.
            strategy: Filter by strategy name.

        Returns:
            List of trade dictionaries sorted by entry_time descending.
        """
        clauses: list[str] = []
        params: list[Any] = []

        if start is not None:
            clauses.append("entry_time >= ?")
            params.append(start)
        if end is not None:
            clauses.append("entry_time <= ?")
            params.append(end)
        if symbol is not None:
            clauses.append("symbol = ?")
            params.append(symbol)
        if strategy is not None:
            clauses.append("strategy = ?")
            params.append(strategy)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM trades{where} ORDER BY entry_time DESC"
        cur = self._sqlite_conn.execute(sql, params)
        rows = cur.fetchall()
        trades = [self._sqlite_row_to_dict(r) for r in rows]
        _json_log("debug", "Fetched trade history", count=len(trades))
        return trades

    def get_trade_stats(self, window_trades: int = 60) -> dict[str, Any]:
        """Compute rolling performance statistics over recent closed trades.

        Args:
            window_trades: Number of most-recent closed trades to
                include in the calculation.

        Returns:
            Dictionary with keys:

            - ``total_trades`` (int)
            - ``winning_trades`` (int)
            - ``losing_trades`` (int)
            - ``win_rate`` (float): 0.0-1.0
            - ``avg_r_multiple`` (float)
            - ``profit_factor`` (float): gross profit / gross loss
            - ``total_pnl`` (float)
            - ``avg_pnl`` (float)
        """
        cur = self._sqlite_conn.execute(
            """
            SELECT pnl, r_multiple
            FROM trades
            WHERE status = 'closed' AND pnl IS NOT NULL
            ORDER BY exit_time DESC
            LIMIT ?
            """,
            (window_trades,),
        )
        rows = cur.fetchall()

        if not rows:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "avg_r_multiple": 0.0,
                "profit_factor": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
            }

        pnls = [float(r["pnl"]) for r in rows]
        r_multiples = [
            float(r["r_multiple"]) for r in rows if r["r_multiple"] is not None
        ]

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0

        stats = {
            "total_trades": len(pnls),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": len(wins) / len(pnls) if pnls else 0.0,
            "avg_r_multiple": (
                sum(r_multiples) / len(r_multiples) if r_multiples else 0.0
            ),
            "profit_factor": (
                gross_profit / gross_loss if gross_loss > 0 else float("inf")
            ),
            "total_pnl": sum(pnls),
            "avg_pnl": sum(pnls) / len(pnls) if pnls else 0.0,
        }

        _json_log("debug", "Computed trade stats", **stats)
        return stats

    @staticmethod
    def _sqlite_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        """Convert a sqlite3.Row to a plain dictionary.

        Parses the ``metadata`` column back from JSON if present.

        Args:
            row: A sqlite3.Row object.

        Returns:
            Dictionary representation of the row.
        """
        d = dict(row)
        if d.get("metadata") and isinstance(d["metadata"], str):
            try:
                d["metadata"] = json.loads(d["metadata"])
            except (json.JSONDecodeError, TypeError):
                pass
        return d

    # ------------------------------------------------------------------
    # Redis — Signal cache
    # ------------------------------------------------------------------

    def cache_signal(
        self,
        symbol: str,
        strategy: str,
        signal_data: dict[str, Any],
        ttl: int = 300,
    ) -> None:
        """Cache a trading signal in Redis with a time-to-live.

        Args:
            symbol: Trading pair identifier.
            strategy: Strategy name.
            signal_data: Signal payload to cache.
            ttl: Time-to-live in seconds (default 300 = 5 minutes).
        """
        key = f"apex:signal:{symbol}:{strategy}"
        self._redis.setex(key, ttl, json.dumps(signal_data, default=str))
        _json_log(
            "debug",
            "Cached signal",
            key=key,
            ttl=ttl,
        )

    def get_cached_signal(
        self, symbol: str, strategy: str
    ) -> Optional[dict[str, Any]]:
        """Retrieve a cached signal from Redis.

        Args:
            symbol: Trading pair identifier.
            strategy: Strategy name.

        Returns:
            The cached signal dictionary, or ``None`` if expired / absent.
        """
        key = f"apex:signal:{symbol}:{strategy}"
        raw = self._redis.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    # ------------------------------------------------------------------
    # Redis — Position state
    # ------------------------------------------------------------------

    def set_position_state(
        self, symbol: str, state: dict[str, Any]
    ) -> None:
        """Store current position state for a symbol in Redis.

        Args:
            symbol: Trading pair identifier.
            state: Position state dictionary (size, entry, side, etc.).
        """
        key = f"apex:position:{symbol}"
        self._redis.set(key, json.dumps(state, default=str))
        _json_log("debug", "Set position state", symbol=symbol)

    def get_position_state(
        self, symbol: str
    ) -> Optional[dict[str, Any]]:
        """Retrieve current position state for a symbol.

        Args:
            symbol: Trading pair identifier.

        Returns:
            Position state dictionary, or ``None`` if no state is stored.
        """
        key = f"apex:position:{symbol}"
        raw = self._redis.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    # ------------------------------------------------------------------
    # Redis — Daily trade count
    # ------------------------------------------------------------------

    def increment_daily_trade_count(self) -> int:
        """Atomically increment the daily trade counter.

        The counter key expires at midnight UTC so it resets naturally.

        Returns:
            The new count after incrementing.
        """
        key = self._daily_trade_key()
        count = self._redis.incr(key)
        if count == 1:
            seconds_until_midnight = self._seconds_until_utc_midnight()
            self._redis.expire(key, seconds_until_midnight)
        _json_log("debug", "Incremented daily trade count", count=count)
        return count

    def get_daily_trade_count(self) -> int:
        """Return the current daily trade count.

        Returns:
            Number of trades executed today (UTC), or 0 if no trades yet.
        """
        key = self._daily_trade_key()
        val = self._redis.get(key)
        return int(val) if val is not None else 0

    # ------------------------------------------------------------------
    # Redis — Daily loss tracking
    # ------------------------------------------------------------------

    def set_daily_loss(self, amount: float) -> None:
        """Record the cumulative daily realised loss.

        Args:
            amount: Absolute loss amount (positive number).
        """
        key = self._daily_loss_key()
        self._redis.set(key, str(amount))
        seconds_until_midnight = self._seconds_until_utc_midnight()
        self._redis.expire(key, seconds_until_midnight)
        _json_log("debug", "Set daily loss", amount=amount)

    def get_daily_loss(self) -> float:
        """Retrieve the cumulative daily realised loss.

        Returns:
            Loss amount, or ``0.0`` if not set.
        """
        key = self._daily_loss_key()
        val = self._redis.get(key)
        return float(val) if val is not None else 0.0

    # ------------------------------------------------------------------
    # Redis — System state flags
    # ------------------------------------------------------------------

    def set_system_state(self, key: str, value: str) -> None:
        """Store a system-level state flag in Redis.

        Useful for pause/halt switches, maintenance mode, etc.

        Args:
            key: State key name (e.g. ``"trading_paused"``).
            value: State value (e.g. ``"true"``).
        """
        redis_key = f"apex:system:{key}"
        self._redis.set(redis_key, value)
        _json_log("info", "Set system state", key=key, value=value)

    def get_system_state(self, key: str) -> Optional[str]:
        """Retrieve a system-level state flag.

        Args:
            key: State key name.

        Returns:
            The stored value as a string, or ``None`` if not set.
        """
        redis_key = f"apex:system:{key}"
        return self._redis.get(redis_key)

    # ------------------------------------------------------------------
    # Async wrappers for AlternativeDataManager compatibility
    # ------------------------------------------------------------------

    async def redis_get(self, key: str) -> Optional[str]:
        """Async wrapper around Redis GET."""
        if self._redis is None:
            raise RuntimeError("Redis is not initialised")
        return self._redis.get(key)

    async def redis_set(
        self, key: str, value: str, ttl: Optional[int] = None
    ) -> None:
        """Async wrapper around Redis SET/SETEX."""
        if self._redis is None:
            raise RuntimeError("Redis is not initialised")
        if ttl:
            self._redis.setex(key, ttl, value)
        else:
            self._redis.set(key, value)

    async def timescaledb_insert(
        self, table: str, record: dict[str, Any]
    ) -> None:
        """Async wrapper for inserting a single record into TimescaleDB."""
        if self._pg_pool is None:
            raise RuntimeError("TimescaleDB pool is not initialised")
        columns = list(record.keys())
        values = list(record.values())
        col_str = ", ".join(columns)
        placeholders = ", ".join(["%s"] * len(columns))
        sql = f"INSERT INTO {table} ({col_str}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"
        with self._pg_cursor() as cur:
            cur.execute(sql, values)

    async def timescaledb_query(
        self, query: str, params: Optional[list] = None
    ) -> list[dict[str, Any]]:
        """Async wrapper for running a SELECT query on TimescaleDB."""
        if self._pg_pool is None:
            raise RuntimeError("TimescaleDB pool is not initialised")
        # Convert $1, $2 style params to %s style for psycopg2
        converted_query = query
        if params:
            for i in range(len(params), 0, -1):
                converted_query = converted_query.replace(f"${i}", "%s")
        with self._pg_cursor(commit=False) as cur:
            cur.execute(converted_query, params or [])
            rows = cur.fetchall()
        return [dict(r) for r in rows] if rows else []

    # ------------------------------------------------------------------
    # Redis helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _daily_trade_key() -> str:
        """Build the Redis key for today's trade counter.

        Returns:
            Key string in the form ``apex:daily_trades:YYYY-MM-DD``.
        """
        today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        return f"apex:daily_trades:{today}"

    @staticmethod
    def _daily_loss_key() -> str:
        """Build the Redis key for today's loss tracker.

        Returns:
            Key string in the form ``apex:daily_loss:YYYY-MM-DD``.
        """
        today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        return f"apex:daily_loss:{today}"

    @staticmethod
    def _seconds_until_utc_midnight() -> int:
        """Calculate seconds remaining until the next UTC midnight.

        Returns:
            Number of seconds (minimum 1).
        """
        now = datetime.now(tz=timezone.utc)
        midnight = now.replace(
            hour=0, minute=0, second=0, microsecond=0
        ) + pd.Timedelta(days=1)
        remaining = int((midnight - now).total_seconds())
        return max(remaining, 1)
