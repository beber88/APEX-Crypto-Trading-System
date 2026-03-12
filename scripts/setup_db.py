#!/usr/bin/env python3
"""TimescaleDB schema setup script for the APEX Crypto Trading System.

Creates all required tables, hypertables, compression policies, and indexes
in a TimescaleDB instance. Safe to run multiple times — all operations use
IF NOT EXISTS semantics.

Usage:
    python -m apex_crypto.scripts.setup_db
    # or
    TIMESCALEDB_URL="postgresql://..." python apex_crypto/scripts/setup_db.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2 import sql
from psycopg2.extensions import connection as PgConnection

# ---------------------------------------------------------------------------
# Structured JSON logging
# ---------------------------------------------------------------------------

class _JsonFormatter(logging.Formatter):
    """Emits each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def _configure_logging() -> logging.Logger:
    """Configure and return the module logger with JSON output on stdout.

    Returns:
        A configured ``logging.Logger`` instance.
    """
    logger = logging.getLogger("apex.setup_db")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)

    return logger


logger = _configure_logging()

# ---------------------------------------------------------------------------
# Default connection URL
# ---------------------------------------------------------------------------

_DEFAULT_URL = "postgresql://apex:apex_password@timescaledb:5432/apex_db"

# ---------------------------------------------------------------------------
# SQL definitions
# ---------------------------------------------------------------------------

_CREATE_EXTENSION = "CREATE EXTENSION IF NOT EXISTS timescaledb;"

_TABLE_DEFINITIONS: list[dict[str, Any]] = [
    # 1. ohlcv
    {
        "name": "ohlcv",
        "ddl": """
            CREATE TABLE IF NOT EXISTS ohlcv (
                symbol      VARCHAR(20)      NOT NULL,
                timeframe   VARCHAR(5)       NOT NULL,
                timestamp   TIMESTAMPTZ      NOT NULL,
                open        DOUBLE PRECISION NOT NULL,
                high        DOUBLE PRECISION NOT NULL,
                low         DOUBLE PRECISION NOT NULL,
                close       DOUBLE PRECISION NOT NULL,
                volume      DOUBLE PRECISION NOT NULL,
                PRIMARY KEY (symbol, timeframe, timestamp)
            );
        """,
        "hypertable": {
            "column": "timestamp",
            "options": "if_not_exists => TRUE",
        },
        "compression": {
            "segment_by": "symbol, timeframe",
            "order_by": "timestamp DESC",
            "after_interval": "INTERVAL '30 days'",
        },
        "indexes": [
            "CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol ON ohlcv (symbol);",
            "CREATE INDEX IF NOT EXISTS idx_ohlcv_timeframe ON ohlcv (timeframe);",
            "CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timeframe ON ohlcv (symbol, timeframe);",
        ],
    },
    # 2. trades
    {
        "name": "trades",
        "ddl": """
            CREATE TABLE IF NOT EXISTS trades (
                id               SERIAL           PRIMARY KEY,
                symbol           VARCHAR(20)      NOT NULL,
                side             VARCHAR(10)      NOT NULL,
                price            DOUBLE PRECISION NOT NULL,
                amount           DOUBLE PRECISION NOT NULL,
                cost             DOUBLE PRECISION,
                fee              DOUBLE PRECISION,
                order_id         VARCHAR(64),
                strategy         VARCHAR(50),
                signal_score     DOUBLE PRECISION,
                entry_timestamp  TIMESTAMPTZ,
                exit_timestamp   TIMESTAMPTZ,
                exit_price       DOUBLE PRECISION,
                exit_reason      VARCHAR(50),
                pnl_usdt         DOUBLE PRECISION,
                pnl_pct          DOUBLE PRECISION,
                r_multiple       DOUBLE PRECISION,
                status           VARCHAR(10)      NOT NULL DEFAULT 'open'
                                     CHECK (status IN ('open', 'closed', 'cancelled'))
            );
        """,
        "hypertable": None,
        "compression": None,
        "indexes": [
            "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades (symbol);",
            "CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades (strategy);",
            "CREATE INDEX IF NOT EXISTS idx_trades_status ON trades (status);",
            "CREATE INDEX IF NOT EXISTS idx_trades_entry_ts ON trades (entry_timestamp);",
        ],
    },
    # 3. order_book_snapshots
    {
        "name": "order_book_snapshots",
        "ddl": """
            CREATE TABLE IF NOT EXISTS order_book_snapshots (
                symbol    VARCHAR(20)  NOT NULL,
                timestamp TIMESTAMPTZ  NOT NULL,
                bids      JSONB        NOT NULL,
                asks      JSONB        NOT NULL,
                PRIMARY KEY (symbol, timestamp)
            );
        """,
        "hypertable": {
            "column": "timestamp",
            "options": "if_not_exists => TRUE",
        },
        "compression": None,
        "indexes": [
            "CREATE INDEX IF NOT EXISTS idx_obs_symbol ON order_book_snapshots (symbol);",
        ],
    },
    # 4. funding_rates
    {
        "name": "funding_rates",
        "ddl": """
            CREATE TABLE IF NOT EXISTS funding_rates (
                symbol         VARCHAR(20)      NOT NULL,
                timestamp      TIMESTAMPTZ      NOT NULL,
                rate           DOUBLE PRECISION NOT NULL,
                predicted_rate DOUBLE PRECISION,
                PRIMARY KEY (symbol, timestamp)
            );
        """,
        "hypertable": {
            "column": "timestamp",
            "options": "if_not_exists => TRUE",
        },
        "compression": None,
        "indexes": [
            "CREATE INDEX IF NOT EXISTS idx_fr_symbol ON funding_rates (symbol);",
        ],
    },
    # 5. liquidations
    {
        "name": "liquidations",
        "ddl": """
            CREATE TABLE IF NOT EXISTS liquidations (
                symbol    VARCHAR(20)      NOT NULL,
                timestamp TIMESTAMPTZ      NOT NULL,
                side      VARCHAR(10)      NOT NULL,
                quantity  DOUBLE PRECISION NOT NULL,
                price     DOUBLE PRECISION NOT NULL,
                PRIMARY KEY (symbol, timestamp)
            );
        """,
        "hypertable": {
            "column": "timestamp",
            "options": "if_not_exists => TRUE",
        },
        "compression": None,
        "indexes": [
            "CREATE INDEX IF NOT EXISTS idx_liq_symbol ON liquidations (symbol);",
        ],
    },
    # 6. sentiment_scores
    {
        "name": "sentiment_scores",
        "ddl": """
            CREATE TABLE IF NOT EXISTS sentiment_scores (
                symbol    VARCHAR(20)      NOT NULL,
                timestamp TIMESTAMPTZ      NOT NULL,
                source    VARCHAR(50)      NOT NULL,
                score     DOUBLE PRECISION NOT NULL,
                raw_data  JSONB,
                PRIMARY KEY (symbol, timestamp, source)
            );
        """,
        "hypertable": {
            "column": "timestamp",
            "options": "if_not_exists => TRUE",
        },
        "compression": None,
        "indexes": [
            "CREATE INDEX IF NOT EXISTS idx_sent_symbol ON sentiment_scores (symbol);",
            "CREATE INDEX IF NOT EXISTS idx_sent_source ON sentiment_scores (source);",
        ],
    },
    # 7. regime_labels
    {
        "name": "regime_labels",
        "ddl": """
            CREATE TABLE IF NOT EXISTS regime_labels (
                symbol     VARCHAR(20)      NOT NULL,
                timestamp  TIMESTAMPTZ      NOT NULL,
                timeframe  VARCHAR(5)       NOT NULL,
                regime     VARCHAR(20)      NOT NULL,
                confidence DOUBLE PRECISION,
                PRIMARY KEY (symbol, timestamp, timeframe)
            );
        """,
        "hypertable": {
            "column": "timestamp",
            "options": "if_not_exists => TRUE",
        },
        "compression": None,
        "indexes": [
            "CREATE INDEX IF NOT EXISTS idx_regime_symbol ON regime_labels (symbol);",
            "CREATE INDEX IF NOT EXISTS idx_regime_timeframe ON regime_labels (timeframe);",
        ],
    },
    # 8. equity_snapshots
    {
        "name": "equity_snapshots",
        "ddl": """
            CREATE TABLE IF NOT EXISTS equity_snapshots (
                timestamp         TIMESTAMPTZ      NOT NULL PRIMARY KEY,
                total_equity      DOUBLE PRECISION NOT NULL,
                available_balance DOUBLE PRECISION NOT NULL,
                unrealized_pnl    DOUBLE PRECISION NOT NULL,
                positions_count   INTEGER          NOT NULL DEFAULT 0
            );
        """,
        "hypertable": {
            "column": "timestamp",
            "options": "if_not_exists => TRUE",
        },
        "compression": None,
        "indexes": [],
    },
    # 9. signals
    {
        "name": "signals",
        "ddl": """
            CREATE TABLE IF NOT EXISTS signals (
                timestamp TIMESTAMPTZ  NOT NULL,
                symbol    VARCHAR(20)  NOT NULL,
                strategy  VARCHAR(50)  NOT NULL,
                timeframe VARCHAR(5)   NOT NULL,
                score     INTEGER      NOT NULL,
                direction VARCHAR(10)  NOT NULL,
                metadata  JSONB,
                PRIMARY KEY (timestamp, symbol, strategy, timeframe)
            );
        """,
        "hypertable": {
            "column": "timestamp",
            "options": "if_not_exists => TRUE",
        },
        "compression": None,
        "indexes": [
            "CREATE INDEX IF NOT EXISTS idx_sig_symbol ON signals (symbol);",
            "CREATE INDEX IF NOT EXISTS idx_sig_strategy ON signals (strategy);",
            "CREATE INDEX IF NOT EXISTS idx_sig_timeframe ON signals (timeframe);",
        ],
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_connection_url() -> str:
    """Resolve the TimescaleDB connection URL.

    Resolution order:
        1. ``TIMESCALEDB_URL`` environment variable.
        2. ``data.timescaledb_url`` in ``config/config.yaml`` (relative to
           the project root).
        3. Hard-coded default.

    Returns:
        A PostgreSQL connection string.
    """
    env_url = os.environ.get("TIMESCALEDB_URL")
    if env_url:
        logger.info("Using connection URL from TIMESCALEDB_URL environment variable.")
        return env_url

    # Attempt to read from config.yaml
    config_candidates = [
        Path(__file__).resolve().parent.parent / "config" / "config.yaml",
        Path.cwd() / "apex_crypto" / "config" / "config.yaml",
    ]

    for config_path in config_candidates:
        if config_path.is_file():
            try:
                import yaml  # type: ignore[import-untyped]

                with open(config_path, "r") as fh:
                    cfg = yaml.safe_load(fh)
                url = cfg.get("data", {}).get("timescaledb_url")
                if url:
                    logger.info(
                        "Using connection URL from config.yaml.",
                        extra={"config_path": str(config_path)},
                    )
                    return str(url)
            except ImportError:
                logger.warning(
                    "PyYAML not installed — cannot read config.yaml. "
                    "Falling back to default URL."
                )
            except Exception:
                logger.warning(
                    "Failed to parse config.yaml — falling back to default URL.",
                    exc_info=True,
                )

    logger.info("Using default connection URL.")
    return _DEFAULT_URL


def _connect(dsn: str, retries: int = 5, backoff: float = 2.0) -> PgConnection:
    """Open a psycopg2 connection with retry logic.

    Args:
        dsn: PostgreSQL connection string.
        retries: Maximum number of connection attempts.
        backoff: Base seconds for exponential back-off between attempts.

    Returns:
        An open ``psycopg2`` connection.

    Raises:
        psycopg2.OperationalError: If all retry attempts are exhausted.
    """
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            conn = psycopg2.connect(dsn)
            conn.autocommit = True
            logger.info(
                "Connected to TimescaleDB.",
                extra={"attempt": attempt},
            )
            return conn
        except psycopg2.OperationalError as exc:
            last_err = exc
            wait = backoff ** attempt
            logger.warning(
                "Connection attempt %d/%d failed — retrying in %.1fs.",
                attempt,
                retries,
                wait,
            )
            time.sleep(wait)

    raise psycopg2.OperationalError(
        f"Could not connect after {retries} attempts: {last_err}"
    )


def _create_extension(conn: PgConnection) -> None:
    """Enable the TimescaleDB extension.

    Args:
        conn: An open psycopg2 connection.
    """
    with conn.cursor() as cur:
        cur.execute(_CREATE_EXTENSION)
    logger.info("TimescaleDB extension enabled.")


def _create_table(conn: PgConnection, table: dict[str, Any]) -> bool:
    """Create a single table, optional hypertable, compression, and indexes.

    Args:
        conn: An open psycopg2 connection.
        table: A table definition dictionary from ``_TABLE_DEFINITIONS``.

    Returns:
        ``True`` if the table was newly created, ``False`` if it already
        existed.
    """
    name: str = table["name"]

    # Check whether the table already exists before CREATE so we can
    # report accurately (CREATE IF NOT EXISTS doesn't tell us).
    with conn.cursor() as cur:
        cur.execute(
            "SELECT EXISTS ("
            "  SELECT 1 FROM information_schema.tables "
            "  WHERE table_schema = 'public' AND table_name = %s"
            ");",
            (name,),
        )
        already_exists: bool = cur.fetchone()[0]  # type: ignore[index]

    with conn.cursor() as cur:
        # -- DDL --
        cur.execute(table["ddl"])

        # -- Hypertable --
        hyper = table.get("hypertable")
        if hyper:
            cur.execute(
                f"SELECT create_hypertable('{name}', '{hyper['column']}', "
                f"{hyper['options']});"
            )

        # -- Compression policy --
        comp = table.get("compression")
        if comp:
            cur.execute(
                f"ALTER TABLE {name} SET ("
                f"  timescaledb.compress,"
                f"  timescaledb.compress_segmentby = '{comp['segment_by']}',"
                f"  timescaledb.compress_orderby = '{comp['order_by']}'"
                f");"
            )
            cur.execute(
                f"SELECT add_compression_policy('{name}', {comp['after_interval']},"
                f" if_not_exists => TRUE);"
            )

        # -- Indexes --
        for idx_sql in table.get("indexes", []):
            cur.execute(idx_sql)

    status = "already existed" if already_exists else "created"
    logger.info("Table '%s' %s.", name, status)
    return not already_exists


def _print_summary(created: list[str], existing: list[str]) -> None:
    """Log a human-readable summary of the setup run.

    Args:
        created: Names of tables that were newly created.
        existing: Names of tables that already existed.
    """
    total = len(created) + len(existing)
    summary: dict[str, Any] = {
        "total_tables": total,
        "newly_created": created,
        "already_existed": existing,
    }
    logger.info("Schema setup complete. %s", json.dumps(summary))

    # Also print a plain-text recap for quick visual scanning.
    print("\n" + "=" * 60)
    print("  APEX Crypto — TimescaleDB Schema Setup Summary")
    print("=" * 60)
    print(f"  Total tables processed : {total}")
    print(f"  Newly created          : {len(created)}")
    print(f"  Already existed        : {len(existing)}")
    if created:
        print("\n  New tables:")
        for t in created:
            print(f"    + {t}")
    if existing:
        print("\n  Existing tables (no changes):")
        for t in existing:
            print(f"    - {t}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full TimescaleDB schema setup.

    Connects to the database, creates the TimescaleDB extension, creates all
    tables (with hypertables, compression policies, and indexes), and prints
    a summary of results.

    Raises:
        SystemExit: On unrecoverable errors (connection failure, SQL errors).
    """
    logger.info("Starting APEX Crypto TimescaleDB schema setup.")

    dsn = _load_connection_url()

    try:
        conn = _connect(dsn)
    except psycopg2.OperationalError:
        logger.critical("Failed to connect to TimescaleDB. Aborting.", exc_info=True)
        sys.exit(1)

    try:
        _create_extension(conn)

        created: list[str] = []
        existing: list[str] = []

        for table_def in _TABLE_DEFINITIONS:
            try:
                was_new = _create_table(conn, table_def)
                if was_new:
                    created.append(table_def["name"])
                else:
                    existing.append(table_def["name"])
            except psycopg2.Error:
                logger.error(
                    "Failed to create table '%s'.",
                    table_def["name"],
                    exc_info=True,
                )
                existing.append(table_def["name"])

        _print_summary(created, existing)

    except psycopg2.Error:
        logger.critical("Unrecoverable database error during setup.", exc_info=True)
        sys.exit(1)
    finally:
        conn.close()
        logger.info("Database connection closed.")


if __name__ == "__main__":
    main()
