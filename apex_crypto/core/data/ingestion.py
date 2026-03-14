"""
APEX Crypto Trading System — OHLCV Data Ingestion Module.

Downloads, paginates, and stores market data from MEXC exchange via ccxt.
Supports incremental history downloads, dynamic watchlist generation,
and real-time ticker/order-book retrieval.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import ccxt.async_support as ccxt
import pandas as pd

from apex_crypto.core.data.storage import StorageManager

logger = logging.getLogger(__name__)

# MEXC hard cap per single fetch_ohlcv call
_MEXC_MAX_CANDLES_PER_REQUEST: int = 1000

# Stablecoin quote/base tokens excluded from the dynamic watchlist
_STABLECOIN_TOKENS: set[str] = {
    "USDT", "USDC", "BUSD", "DAI", "TUSD", "USDP", "FDUSD",
}

# Mapping from ccxt timeframe strings to millisecond durations
_TIMEFRAME_MS: dict[str, int] = {
    "1m":  60_000,
    "3m":  3 * 60_000,
    "5m":  5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h":  60 * 60_000,
    "2h":  2 * 60 * 60_000,
    "4h":  4 * 60 * 60_000,
    "1d":  24 * 60 * 60_000,
    "1w":  7 * 24 * 60 * 60_000,
}


def _structured_log(
    level: int,
    event: str,
    **kwargs: Any,
) -> None:
    """Emit a structured JSON log line.

    Args:
        level: Python logging level (e.g. ``logging.INFO``).
        event: Short event descriptor.
        **kwargs: Arbitrary key/value pairs attached to the log record.
    """
    payload: dict[str, Any] = {
        "event": event,
        "ts": datetime.now(timezone.utc).isoformat(),
        **kwargs,
    }
    logger.log(level, json.dumps(payload, default=str))


class MarketDataManager:
    """Manages OHLCV data ingestion from MEXC via ccxt.

    Responsibilities:
        * Download historical and incremental OHLCV candles.
        * Persist data to TimescaleDB through a ``StorageManager``.
        * Provide real-time ticker and order-book snapshots.
        * Generate a dynamic watchlist filtered by volume and price.

    Args:
        config: Full system configuration dictionary (parsed from
            ``config.yaml``).
        storage: A ``StorageManager`` instance used for database I/O.

    Example::

        mgr = MarketDataManager(config, storage)
        df = await mgr.download_ohlcv("BTC/USDT", "1h")
    """

    def __init__(self, config: dict[str, Any], storage: StorageManager) -> None:
        self._config: dict[str, Any] = config
        self._storage: StorageManager = storage

        exchange_cfg: dict[str, Any] = config.get("exchange", {})

        self._rate_limit_ms: int = int(exchange_cfg.get("rate_limit_ms", 50))
        self._max_retries: int = int(exchange_cfg.get("max_retries", 5))
        self._retry_backoff_base: float = float(
            exchange_cfg.get("retry_backoff_base", 2.0)
        )

        # History depth pulled from data.history_years (default 5)
        data_cfg: dict[str, Any] = config.get("data", {})
        self._history_years: int = int(data_cfg.get("history_years", 5))

        # Dynamic watchlist filter thresholds
        assets_cfg: dict[str, Any] = config.get("assets", {})
        self._min_daily_volume: float = float(
            assets_cfg.get("min_daily_volume_usd", 8_000_000)
        )
        self._min_price: float = float(assets_cfg.get("min_price", 0.001))
        self._watchlist_size: int = 30

        # Build the ccxt MEXC exchange instance
        api_key: str = os.environ.get("MEXC_API_KEY", "")
        secret_key: str = os.environ.get("MEXC_SECRET_KEY", "")

        self._exchange: ccxt.mexc = ccxt.mexc(
            {
                "apiKey": api_key,
                "secret": secret_key,
                "enableRateLimit": True,
                "rateLimit": self._rate_limit_ms,
                "options": {"defaultType": "spot", "fetchCurrencies": False},
            }
        )

        if exchange_cfg.get("testnet", False):
            _structured_log(logging.WARNING, "mexc_sandbox_not_supported",
                            msg="MEXC does not support sandbox mode in ccxt; skipping")

        _structured_log(
            logging.INFO,
            "market_data_manager_initialized",
            rate_limit_ms=self._rate_limit_ms,
            max_retries=self._max_retries,
            testnet=exchange_cfg.get("testnet", False),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _close_exchange(self) -> None:
        """Gracefully close the underlying ccxt exchange session."""
        try:
            await self._exchange.close()
        except Exception:
            pass

    async def _retry_with_backoff(
        self,
        coro_factory: Any,
        operation: str,
        **log_ctx: Any,
    ) -> Any:
        """Execute an async callable with exponential-backoff retry.

        Retries on ``ccxt.NetworkError``, ``ccxt.ExchangeError``, and
        ``ccxt.RequestTimeout``.  All other exceptions propagate immediately.

        Args:
            coro_factory: A zero-argument callable that returns an awaitable.
            operation: Human-readable operation name for logging.
            **log_ctx: Extra key/value pairs included in log output.

        Returns:
            The result of the awaitable on success.

        Raises:
            The last caught exception after exhausting retries.
        """
        last_exc: Optional[Exception] = None

        for attempt in range(1, self._max_retries + 1):
            try:
                return await coro_factory()
            except (ccxt.RateLimitExceeded,) as exc:
                last_exc = exc
                wait: float = self._retry_backoff_base ** attempt
                _structured_log(
                    logging.WARNING,
                    "rate_limit_hit",
                    operation=operation,
                    attempt=attempt,
                    wait_s=wait,
                    **log_ctx,
                )
                await asyncio.sleep(wait)
            except (ccxt.NetworkError, ccxt.RequestTimeout) as exc:
                last_exc = exc
                wait = self._retry_backoff_base ** attempt
                _structured_log(
                    logging.WARNING,
                    "network_error_retry",
                    operation=operation,
                    attempt=attempt,
                    wait_s=wait,
                    error=str(exc),
                    **log_ctx,
                )
                await asyncio.sleep(wait)
            except ccxt.ExchangeError as exc:
                last_exc = exc
                wait = self._retry_backoff_base ** attempt
                _structured_log(
                    logging.ERROR,
                    "exchange_error_retry",
                    operation=operation,
                    attempt=attempt,
                    wait_s=wait,
                    error=str(exc),
                    **log_ctx,
                )
                await asyncio.sleep(wait)

        # All retries exhausted
        _structured_log(
            logging.ERROR,
            "retries_exhausted",
            operation=operation,
            max_retries=self._max_retries,
            error=str(last_exc),
            **log_ctx,
        )
        raise last_exc  # type: ignore[misc]

    @staticmethod
    def _ohlcv_to_dataframe(raw: list[list[float | int]]) -> pd.DataFrame:
        """Convert raw ccxt OHLCV rows to a typed DataFrame.

        Args:
            raw: List of ``[timestamp_ms, open, high, low, close, volume]``
                rows as returned by ``fetch_ohlcv``.

        Returns:
            A ``pd.DataFrame`` with columns
            ``[timestamp, open, high, low, close, volume]`` where
            ``timestamp`` is a timezone-aware UTC datetime.
        """
        df = pd.DataFrame(
            raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        # Remove exact-duplicate timestamps that can occur at page boundaries
        df = df.drop_duplicates(subset=["timestamp"], keep="last").reset_index(
            drop=True
        )
        return df

    async def _rate_limit_pause(self) -> None:
        """Sleep for the configured rate-limit interval."""
        await asyncio.sleep(self._rate_limit_ms / 1000.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def download_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Download OHLCV candles from MEXC and persist them.

        Handles automatic pagination because MEXC returns at most 1 000
        candles per request.  The method loops from ``since`` until the
        present, issuing successive requests with an advancing ``since``
        parameter.

        Args:
            symbol: Trading pair in ccxt format (e.g. ``"BTC/USDT"``).
            timeframe: Candle interval string (e.g. ``"1h"``, ``"1d"``).
            since: Earliest candle open time to fetch.  Defaults to
                ``history_years`` ago.
            limit: Optional total-candle cap.  ``None`` means download
                everything available from ``since`` to now.

        Returns:
            A ``pd.DataFrame`` containing all fetched candles with
            columns ``[timestamp, open, high, low, close, volume]``.
        """
        tf_ms: int = _TIMEFRAME_MS.get(timeframe, 60_000)

        if since is None:
            since = datetime.now(timezone.utc) - timedelta(
                days=self._history_years * 365
            )

        since_ms: int = int(since.timestamp() * 1000)
        now_ms: int = int(datetime.now(timezone.utc).timestamp() * 1000)

        all_rows: list[list[float | int]] = []
        total_fetched: int = 0
        page: int = 0

        _structured_log(
            logging.INFO,
            "download_ohlcv_start",
            symbol=symbol,
            timeframe=timeframe,
            since=since.isoformat(),
            limit=limit,
        )

        while since_ms < now_ms:
            page_limit: int = _MEXC_MAX_CANDLES_PER_REQUEST
            if limit is not None:
                remaining = limit - total_fetched
                if remaining <= 0:
                    break
                page_limit = min(page_limit, remaining)

            raw: list[list[float | int]] = await self._retry_with_backoff(
                lambda _since=since_ms, _pl=page_limit: self._exchange.fetch_ohlcv(
                    symbol, timeframe, since=_since, limit=_pl
                ),
                operation="fetch_ohlcv",
                symbol=symbol,
                timeframe=timeframe,
                page=page,
            )

            if not raw:
                break

            all_rows.extend(raw)
            total_fetched += len(raw)
            page += 1

            # Advance the cursor past the last received candle
            last_ts: int = int(raw[-1][0])
            since_ms = last_ts + tf_ms

            _structured_log(
                logging.DEBUG,
                "download_ohlcv_page",
                symbol=symbol,
                timeframe=timeframe,
                page=page,
                rows_in_page=len(raw),
                total_fetched=total_fetched,
            )

            # Fewer rows than requested means we have reached the end
            if len(raw) < page_limit:
                break

            await self._rate_limit_pause()

        if not all_rows:
            _structured_log(
                logging.WARNING,
                "download_ohlcv_empty",
                symbol=symbol,
                timeframe=timeframe,
            )
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        df: pd.DataFrame = self._ohlcv_to_dataframe(all_rows)

        # Persist to TimescaleDB via storage manager
        await self._storage.store_ohlcv(symbol, timeframe, df)

        _structured_log(
            logging.INFO,
            "download_ohlcv_complete",
            symbol=symbol,
            timeframe=timeframe,
            rows=len(df),
            first_ts=str(df["timestamp"].iloc[0]),
            last_ts=str(df["timestamp"].iloc[-1]),
        )

        return df

    async def download_all_history(
        self,
        symbols: list[str],
        timeframes: list[str],
    ) -> dict[str, dict[str, int]]:
        """Download full OHLCV history for every symbol/timeframe pair.

        For each combination the method first queries ``StorageManager``
        for the latest stored timestamp and performs an incremental
        download from that point forward.  If no data exists, it
        downloads from ``history_years`` ago.

        Args:
            symbols: List of trading pairs (e.g.
                ``["BTC/USDT", "ETH/USDT"]``).
            timeframes: List of candle intervals (e.g.
                ``["1h", "4h", "1d"]``).

        Returns:
            A nested dict mapping
            ``{symbol: {timeframe: row_count}}`` for every downloaded
            combination.
        """
        results: dict[str, dict[str, int]] = {}

        _structured_log(
            logging.INFO,
            "download_all_history_start",
            symbols=symbols,
            timeframes=timeframes,
            history_years=self._history_years,
        )

        for symbol in symbols:
            results[symbol] = {}
            for timeframe in timeframes:
                # Determine the starting point for this pair
                latest_ts: Optional[datetime] = await self._storage.get_latest_timestamp(
                    symbol, timeframe
                )

                if latest_ts is not None:
                    # Small overlap of one candle to avoid gaps
                    since = latest_ts
                    _structured_log(
                        logging.INFO,
                        "incremental_download",
                        symbol=symbol,
                        timeframe=timeframe,
                        since=since.isoformat(),
                    )
                else:
                    since = datetime.now(timezone.utc) - timedelta(
                        days=self._history_years * 365
                    )
                    _structured_log(
                        logging.INFO,
                        "full_history_download",
                        symbol=symbol,
                        timeframe=timeframe,
                        since=since.isoformat(),
                    )

                df: pd.DataFrame = await self.download_ohlcv(
                    symbol, timeframe, since=since
                )
                results[symbol][timeframe] = len(df)

                # Rate-limit between symbol/timeframe pairs
                await self._rate_limit_pause()

        _structured_log(
            logging.INFO,
            "download_all_history_complete",
            results=results,
        )

        return results

    async def refresh_latest(
        self,
        symbols: list[str],
        timeframes: list[str],
    ) -> None:
        """Fetch the most recent candles since the last stored timestamp.

        Intended to be called periodically (e.g. every minute) to keep
        the local database current without re-downloading the entire
        history.

        Args:
            symbols: Trading pairs to refresh.
            timeframes: Candle intervals to refresh.
        """
        _structured_log(
            logging.INFO,
            "refresh_latest_start",
            symbols=symbols,
            timeframes=timeframes,
        )

        for symbol in symbols:
            for timeframe in timeframes:
                latest_ts: Optional[datetime] = await self._storage.get_latest_timestamp(
                    symbol, timeframe
                )

                if latest_ts is not None:
                    since = latest_ts
                else:
                    # No stored data — pull last 200 candles as a sensible
                    # default so downstream consumers have something to work
                    # with immediately.
                    tf_ms: int = _TIMEFRAME_MS.get(timeframe, 60_000)
                    since = datetime.now(timezone.utc) - timedelta(
                        milliseconds=tf_ms * 200
                    )

                df: pd.DataFrame = await self.download_ohlcv(
                    symbol, timeframe, since=since
                )

                _structured_log(
                    logging.DEBUG,
                    "refresh_latest_pair",
                    symbol=symbol,
                    timeframe=timeframe,
                    rows=len(df),
                )

                await self._rate_limit_pause()

        _structured_log(logging.INFO, "refresh_latest_complete")

    async def get_dynamic_watchlist(self) -> list[str]:
        """Build a watchlist of the top coins ranked by 24 h volume.

        Fetches all USDT-quoted tickers from MEXC, filters out
        stablecoins, low-volume, and low-price pairs, then returns
        the top 30 symbols sorted by descending quoteVolume.

        Returns:
            A list of up to 30 symbol strings
            (e.g. ``["BTC/USDT", "ETH/USDT", ...]``).
        """
        _structured_log(logging.INFO, "get_dynamic_watchlist_start")

        tickers: dict[str, Any] = await self._retry_with_backoff(
            lambda: self._exchange.fetch_tickers(),
            operation="fetch_tickers",
        )

        candidates: list[tuple[str, float]] = []

        for symbol, ticker in tickers.items():
            # Only consider USDT-quoted spot pairs
            if not symbol.endswith("/USDT"):
                continue

            # Extract the base token and skip stablecoins
            base: str = symbol.split("/")[0]
            if base in _STABLECOIN_TOKENS:
                continue

            quote_volume: float = float(ticker.get("quoteVolume") or 0)
            last_price: float = float(ticker.get("last") or 0)

            if quote_volume < self._min_daily_volume:
                continue
            if last_price < self._min_price:
                continue

            candidates.append((symbol, quote_volume))

        # Sort descending by 24 h quote volume and take top N
        candidates.sort(key=lambda x: x[1], reverse=True)
        watchlist: list[str] = [sym for sym, _ in candidates[: self._watchlist_size]]

        _structured_log(
            logging.INFO,
            "get_dynamic_watchlist_complete",
            count=len(watchlist),
            top_5=watchlist[:5],
        )

        return watchlist

    async def get_ticker(self, symbol: str) -> dict[str, Any]:
        """Return the current ticker snapshot for a symbol.

        Args:
            symbol: Trading pair (e.g. ``"BTC/USDT"``).

        Returns:
            A dict with keys ``price``, ``volume``, ``bid``, ``ask``,
            and ``spread``.
        """
        raw: dict[str, Any] = await self._retry_with_backoff(
            lambda: self._exchange.fetch_ticker(symbol),
            operation="fetch_ticker",
            symbol=symbol,
        )

        bid: float = float(raw.get("bid") or 0)
        ask: float = float(raw.get("ask") or 0)
        spread: float = ask - bid if (ask and bid) else 0.0

        result: dict[str, Any] = {
            "symbol": symbol,
            "price": raw.get("last"),
            "volume": raw.get("quoteVolume"),
            "bid": bid,
            "ask": ask,
            "spread": spread,
        }

        _structured_log(
            logging.DEBUG,
            "get_ticker",
            **result,
        )

        return result

    async def get_order_book(
        self,
        symbol: str,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Return the current order book for a symbol.

        Args:
            symbol: Trading pair (e.g. ``"BTC/USDT"``).
            limit: Number of price levels on each side (default 20).

        Returns:
            A dict with keys ``symbol``, ``bids``, ``asks``,
            ``timestamp``, ``bid_depth``, and ``ask_depth``.  Each of
            ``bids`` and ``asks`` is a list of ``[price, amount]``
            pairs.
        """
        raw: dict[str, Any] = await self._retry_with_backoff(
            lambda: self._exchange.fetch_order_book(symbol, limit=limit),
            operation="fetch_order_book",
            symbol=symbol,
            limit=limit,
        )

        bids: list[list[float]] = raw.get("bids", [])
        asks: list[list[float]] = raw.get("asks", [])

        result: dict[str, Any] = {
            "symbol": symbol,
            "bids": bids,
            "asks": asks,
            "timestamp": raw.get("timestamp"),
            "bid_depth": sum(b[1] for b in bids) if bids else 0.0,
            "ask_depth": sum(a[1] for a in asks) if asks else 0.0,
        }

        _structured_log(
            logging.DEBUG,
            "get_order_book",
            symbol=symbol,
            bid_levels=len(bids),
            ask_levels=len(asks),
        )

        return result
