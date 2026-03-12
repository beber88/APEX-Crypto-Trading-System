"""WebSocket streaming handler for real-time MEXC exchange data feeds.

Provides the ``MarketStreamManager`` class which connects to MEXC via
ccxt.pro (async WebSocket support) and dispatches real-time trade, ticker,
order-book, funding-rate, and liquidation events to callbacks and persistent
storage.

Falls back to the raw ``websockets`` library when ccxt.pro is not installed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

from apex_crypto.core.data.storage import StorageManager

# ---------------------------------------------------------------------------
# Structured JSON logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


class _JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


def _configure_logger() -> None:
    """Attach the JSON formatter to the module logger if not already configured."""
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(_JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)


_configure_logger()

# ---------------------------------------------------------------------------
# Try to import ccxt.pro; fall back to raw websockets
# ---------------------------------------------------------------------------

try:
    import ccxt.pro as ccxtpro  # type: ignore[import-untyped]

    _HAS_CCXT_PRO = True
    logger.info("ccxt.pro available – using native async WebSocket support")
except ImportError:
    _HAS_CCXT_PRO = False
    logger.warning("ccxt.pro not available – falling back to raw websockets")

try:
    import websockets  # type: ignore[import-untyped]
except ImportError:
    websockets = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

TradeCallback = Callable[[Dict[str, Any]], Coroutine[Any, Any, None] | None]
TickerCallback = Callable[[Dict[str, Any]], Coroutine[Any, Any, None] | None]
OrderbookCallback = Callable[[Dict[str, Any]], Coroutine[Any, Any, None] | None]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_HEARTBEAT_INTERVAL: int = 30  # seconds
_MAX_RECONNECT_DELAY: int = 60  # seconds
_BASE_RECONNECT_DELAY: float = 1.0  # seconds
_DEFAULT_ORDERBOOK_SNAPSHOT_INTERVAL: int = 10  # seconds
_LIQUIDATION_ALERT_THRESHOLD_USD: float = 100_000.0

# MEXC raw WebSocket endpoints (used only in fallback mode)
_MEXC_WS_SPOT = "wss://wbs.mexc.com/ws"
_MEXC_WS_FUTURES = "wss://contract.mexc.com/edge"


class MarketStreamManager:
    """Manages real-time WebSocket data feeds from the MEXC exchange.

    Connects via ccxt.pro when available, otherwise falls back to the raw
    ``websockets`` library.  Provides automatic reconnection with exponential
    back-off, periodic heartbeats, and thread-safe callback dispatching.

    Args:
        config: Exchange / streaming configuration dictionary.  Recognised
            keys include ``api_key``, ``secret``, ``heartbeat_interval``,
            ``orderbook_snapshot_interval``, ``liquidation_alert_threshold``,
            and ``sandbox`` (bool).
        storage: A ``StorageManager`` instance used for persisting snapshots
            and events to TimescaleDB.
        on_trade_callback: Optional async/sync callable invoked on every
            trade event.
        on_ticker_callback: Optional async/sync callable invoked on every
            ticker update.
        on_orderbook_callback: Optional async/sync callable invoked on every
            order-book update.
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(
        self,
        config: dict,
        storage: StorageManager,
        on_trade_callback: Optional[TradeCallback] = None,
        on_ticker_callback: Optional[TickerCallback] = None,
        on_orderbook_callback: Optional[OrderbookCallback] = None,
    ) -> None:
        self._config = config
        self._storage = storage

        # Callbacks ---------------------------------------------------------
        self._on_trade_callback = on_trade_callback
        self._on_ticker_callback = on_ticker_callback
        self._on_orderbook_callback = on_orderbook_callback

        # Tunables ----------------------------------------------------------
        self._heartbeat_interval: int = int(
            config.get("heartbeat_interval", _DEFAULT_HEARTBEAT_INTERVAL)
        )
        self._orderbook_snapshot_interval: int = int(
            config.get("orderbook_snapshot_interval", _DEFAULT_ORDERBOOK_SNAPSHOT_INTERVAL)
        )
        self._liquidation_alert_threshold: float = float(
            config.get("liquidation_alert_threshold", _LIQUIDATION_ALERT_THRESHOLD_USD)
        )

        # Internal state ----------------------------------------------------
        self._exchange: Optional[Any] = None  # ccxt.pro exchange instance
        self._running: bool = False
        self._tasks: Dict[str, asyncio.Task[None]] = {}
        self._subscribed_symbols: Set[str] = set()
        self._reconnect_attempts: Dict[str, int] = {}
        self._last_orderbook_snapshot: Dict[str, float] = {}
        self._raw_ws_connections: Dict[str, Any] = {}
        self._lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Exchange bootstrap
    # ------------------------------------------------------------------

    def _create_exchange(self) -> Any:
        """Create and return a ccxt.pro MEXC exchange instance.

        Returns:
            A configured ``ccxtpro.mexc`` exchange object.

        Raises:
            RuntimeError: If neither ccxt.pro nor raw websockets are available.
        """
        if not _HAS_CCXT_PRO:
            if websockets is None:
                raise RuntimeError(
                    "Neither ccxt.pro nor the websockets library is installed. "
                    "Install one of them to use MarketStreamManager."
                )
            logger.info("Operating in raw-websockets fallback mode")
            return None

        exchange = ccxtpro.mexc(
            {
                "apiKey": self._config.get("api_key", ""),
                "secret": self._config.get("secret", ""),
                "enableRateLimit": True,
                "options": {
                    "defaultType": self._config.get("default_type", "spot"),
                },
            }
        )
        if self._config.get("sandbox", False):
            exchange.set_sandbox_mode(True)
        return exchange

    # ------------------------------------------------------------------
    # Public API – lifecycle
    # ------------------------------------------------------------------

    async def start(self, symbols: List[str]) -> None:
        """Start WebSocket connections for all specified symbols.

        Subscribes each symbol to the trade, ticker, and order-book streams.
        Tasks run concurrently inside the current asyncio event loop.

        Args:
            symbols: List of unified symbol strings (e.g. ``["BTC/USDT"]``).
        """
        if self._running:
            logger.warning("MarketStreamManager is already running")
            return

        self._exchange = self._create_exchange()
        self._running = True

        logger.info("Starting streams for symbols: %s", symbols)

        for symbol in symbols:
            self._subscribed_symbols.add(symbol)

            trade_key = f"{symbol}:trades"
            ticker_key = f"{symbol}:ticker"
            ob_key = f"{symbol}:orderbook"
            funding_key = f"{symbol}:funding"
            liq_key = f"{symbol}:liquidations"

            self._tasks[trade_key] = asyncio.create_task(
                self._run_with_reconnect(symbol, "trades", self.subscribe_trades)
            )
            self._tasks[ticker_key] = asyncio.create_task(
                self._run_with_reconnect(symbol, "ticker", self.subscribe_ticker)
            )
            self._tasks[ob_key] = asyncio.create_task(
                self._run_with_reconnect(symbol, "orderbook", self.subscribe_orderbook)
            )
            self._tasks[funding_key] = asyncio.create_task(
                self._run_with_reconnect(symbol, "funding", self.subscribe_funding_rate)
            )
            self._tasks[liq_key] = asyncio.create_task(
                self._run_with_reconnect(symbol, "liquidations", self.subscribe_liquidations)
            )

        # Heartbeat task
        self._tasks["__heartbeat__"] = asyncio.create_task(self._heartbeat_loop())

        logger.info("All stream tasks launched")

    async def stop(self) -> None:
        """Gracefully close all WebSocket connections and cancel tasks."""
        logger.info("Stopping MarketStreamManager")
        self._running = False

        for name, task in self._tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.debug("Task %s cancelled", name)

        self._tasks.clear()

        # Close ccxt.pro exchange
        if self._exchange is not None and _HAS_CCXT_PRO:
            try:
                await self._exchange.close()
            except Exception as exc:
                logger.error("Error closing exchange: %s", exc)

        # Close raw websocket connections
        for key, ws in list(self._raw_ws_connections.items()):
            try:
                await ws.close()
            except Exception as exc:
                logger.error("Error closing raw ws %s: %s", key, exc)
        self._raw_ws_connections.clear()

        self._subscribed_symbols.clear()
        logger.info("MarketStreamManager stopped")

    # ------------------------------------------------------------------
    # Public API – subscriptions
    # ------------------------------------------------------------------

    async def subscribe_trades(self, symbol: str) -> None:
        """Subscribe to real-time trade stream for a symbol.

        Each trade event contains ``price``, ``amount``, ``side``, and
        ``timestamp`` fields.  If ``on_trade_callback`` was provided at
        construction time it is invoked for every trade.

        Args:
            symbol: Unified symbol string (e.g. ``"BTC/USDT"``).
        """
        logger.info("Subscribing to trades: %s", symbol)

        if _HAS_CCXT_PRO and self._exchange is not None:
            while self._running:
                trades = await self._exchange.watch_trades(symbol)
                for trade in trades:
                    normalized = self._normalize_trade(trade)
                    await self._dispatch_callback(self._on_trade_callback, normalized)
        else:
            await self._raw_subscribe_trades(symbol)

    async def subscribe_ticker(self, symbol: str) -> None:
        """Subscribe to real-time price ticker updates for a symbol.

        If ``on_ticker_callback`` was provided it is invoked on each update.

        Args:
            symbol: Unified symbol string.
        """
        logger.info("Subscribing to ticker: %s", symbol)

        if _HAS_CCXT_PRO and self._exchange is not None:
            while self._running:
                ticker = await self._exchange.watch_ticker(symbol)
                normalized = self._normalize_ticker(ticker)
                await self._dispatch_callback(self._on_ticker_callback, normalized)
        else:
            await self._raw_subscribe_ticker(symbol)

    async def subscribe_orderbook(self, symbol: str, depth: int = 20) -> None:
        """Subscribe to real-time order-book updates for a symbol.

        Stores snapshots to the storage layer at the configured interval.
        If ``on_orderbook_callback`` was provided it is invoked on each
        update.

        Args:
            symbol: Unified symbol string.
            depth: Number of price levels per side (default 20).
        """
        logger.info("Subscribing to orderbook: %s (depth=%d)", symbol, depth)

        if _HAS_CCXT_PRO and self._exchange is not None:
            while self._running:
                orderbook = await self._exchange.watch_order_book(symbol, depth)
                normalized = self._normalize_orderbook(orderbook, symbol)
                await self._dispatch_callback(self._on_orderbook_callback, normalized)
                await self._maybe_store_orderbook_snapshot(symbol, normalized)
        else:
            await self._raw_subscribe_orderbook(symbol, depth)

    async def subscribe_funding_rate(self, symbol: str) -> None:
        """Subscribe to funding-rate updates for perpetual futures.

        Stores each funding-rate record to TimescaleDB via the storage
        manager.

        Args:
            symbol: Unified perpetual-futures symbol (e.g.
                ``"BTC/USDT:USDT"``).
        """
        logger.info("Subscribing to funding rate: %s", symbol)

        if _HAS_CCXT_PRO and self._exchange is not None:
            while self._running:
                try:
                    funding = await self._exchange.watch_funding_rate(symbol)
                    normalized = self._normalize_funding_rate(funding, symbol)
                    await self._store_funding_rate(normalized)
                except Exception as exc:
                    logger.error("Funding rate error for %s: %s", symbol, exc)
                    await asyncio.sleep(5)
        else:
            await self._raw_subscribe_funding_rate(symbol)

    async def subscribe_liquidations(self, symbol: str) -> None:
        """Subscribe to liquidation events for a symbol.

        Large liquidation events (exceeding the configured threshold) are
        stored to TimescaleDB and trigger an alert via the storage manager.

        Args:
            symbol: Unified symbol string.
        """
        logger.info("Subscribing to liquidations: %s", symbol)

        if _HAS_CCXT_PRO and self._exchange is not None:
            while self._running:
                try:
                    liquidations = await self._exchange.watch_liquidations(symbol)
                    for liq in liquidations:
                        normalized = self._normalize_liquidation(liq, symbol)
                        await self._store_liquidation(normalized)
                except Exception as exc:
                    logger.error("Liquidation stream error for %s: %s", symbol, exc)
                    await asyncio.sleep(5)
        else:
            await self._raw_subscribe_liquidations(symbol)

    # ------------------------------------------------------------------
    # Reconnection
    # ------------------------------------------------------------------

    async def reconnect(self, symbol: str, stream_type: str) -> None:
        """Reconnect a single stream with exponential back-off.

        The delay between attempts doubles on each failure up to a maximum
        of ``_MAX_RECONNECT_DELAY`` seconds.

        Args:
            symbol: The symbol whose stream disconnected.
            stream_type: One of ``"trades"``, ``"ticker"``, ``"orderbook"``,
                ``"funding"``, ``"liquidations"``.
        """
        key = f"{symbol}:{stream_type}"
        attempts = self._reconnect_attempts.get(key, 0)
        delay = min(_BASE_RECONNECT_DELAY * (2 ** attempts), _MAX_RECONNECT_DELAY)

        logger.warning(
            "Reconnecting %s (attempt %d, delay %.1fs)", key, attempts + 1, delay
        )
        await asyncio.sleep(delay)

        self._reconnect_attempts[key] = attempts + 1

        handler_map: Dict[str, Callable[..., Coroutine[Any, Any, None]]] = {
            "trades": self.subscribe_trades,
            "ticker": self.subscribe_ticker,
            "orderbook": self.subscribe_orderbook,
            "funding": self.subscribe_funding_rate,
            "liquidations": self.subscribe_liquidations,
        }

        handler = handler_map.get(stream_type)
        if handler is None:
            logger.error("Unknown stream type for reconnect: %s", stream_type)
            return

        await handler(symbol)

    # ------------------------------------------------------------------
    # Internal – reconnection wrapper
    # ------------------------------------------------------------------

    async def _run_with_reconnect(
        self,
        symbol: str,
        stream_type: str,
        handler: Callable[..., Coroutine[Any, Any, None]],
    ) -> None:
        """Run *handler* in a loop, reconnecting on failure.

        Args:
            symbol: Target symbol.
            stream_type: Logical stream name.
            handler: The subscription coroutine to run.
        """
        key = f"{symbol}:{stream_type}"
        while self._running:
            try:
                await handler(symbol)
            except asyncio.CancelledError:
                logger.debug("Stream %s cancelled", key)
                return
            except Exception as exc:
                if not self._running:
                    return
                logger.error("Stream %s error: %s", key, exc, exc_info=True)
                attempts = self._reconnect_attempts.get(key, 0)
                delay = min(
                    _BASE_RECONNECT_DELAY * (2 ** attempts), _MAX_RECONNECT_DELAY
                )
                logger.warning(
                    "Reconnecting %s (attempt %d, backoff %.1fs)",
                    key,
                    attempts + 1,
                    delay,
                )
                await asyncio.sleep(delay)
                self._reconnect_attempts[key] = attempts + 1
            else:
                # Handler returned cleanly (shouldn't happen unless stopped)
                break

        # Reset attempt counter on clean exit
        self._reconnect_attempts.pop(key, None)

    # ------------------------------------------------------------------
    # Internal – heartbeat
    # ------------------------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        """Periodically send pings / log health status."""
        while self._running:
            await asyncio.sleep(self._heartbeat_interval)
            active = sum(1 for t in self._tasks.values() if not t.done())
            logger.debug(
                "Heartbeat – %d/%d tasks active", active, len(self._tasks)
            )
            # Ping raw WS connections to keep them alive
            for key, ws in list(self._raw_ws_connections.items()):
                try:
                    pong_waiter = await ws.ping()
                    await asyncio.wait_for(pong_waiter, timeout=10)
                except Exception as exc:
                    logger.warning("Heartbeat ping failed for %s: %s", key, exc)

    # ------------------------------------------------------------------
    # Internal – callback dispatch
    # ------------------------------------------------------------------

    async def _dispatch_callback(
        self,
        callback: Optional[Callable[..., Any]],
        data: Dict[str, Any],
    ) -> None:
        """Invoke a callback in a thread-safe manner.

        Supports both sync and async callbacks.

        Args:
            callback: The callable to invoke (may be ``None``).
            data: The payload to pass.
        """
        if callback is None:
            return

        try:
            result = callback(data)
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:
            logger.error("Callback error: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Internal – normalisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_trade(trade: Dict[str, Any]) -> Dict[str, Any]:
        """Normalise a ccxt trade dict into the canonical APEX format.

        Args:
            trade: Raw ccxt trade dictionary.

        Returns:
            Normalised trade dictionary.
        """
        return {
            "symbol": trade.get("symbol", ""),
            "price": float(trade.get("price", 0)),
            "amount": float(trade.get("amount", 0)),
            "side": trade.get("side", ""),
            "timestamp": trade.get("timestamp", int(time.time() * 1000)),
            "datetime": trade.get("datetime", datetime.now(timezone.utc).isoformat()),
            "trade_id": trade.get("id", ""),
            "source": "mexc",
        }

    @staticmethod
    def _normalize_ticker(ticker: Dict[str, Any]) -> Dict[str, Any]:
        """Normalise a ccxt ticker dict.

        Args:
            ticker: Raw ccxt ticker dictionary.

        Returns:
            Normalised ticker dictionary.
        """
        return {
            "symbol": ticker.get("symbol", ""),
            "last": float(ticker.get("last", 0)),
            "bid": float(ticker.get("bid", 0)),
            "ask": float(ticker.get("ask", 0)),
            "high": float(ticker.get("high", 0)),
            "low": float(ticker.get("low", 0)),
            "volume": float(ticker.get("baseVolume", 0)),
            "quote_volume": float(ticker.get("quoteVolume", 0)),
            "timestamp": ticker.get("timestamp", int(time.time() * 1000)),
            "datetime": ticker.get("datetime", datetime.now(timezone.utc).isoformat()),
            "source": "mexc",
        }

    @staticmethod
    def _normalize_orderbook(
        orderbook: Dict[str, Any], symbol: str
    ) -> Dict[str, Any]:
        """Normalise a ccxt order-book dict.

        Args:
            orderbook: Raw ccxt order-book dictionary.
            symbol: The symbol this order book belongs to.

        Returns:
            Normalised order-book dictionary.
        """
        return {
            "symbol": symbol,
            "bids": [
                {"price": float(level[0]), "amount": float(level[1])}
                for level in (orderbook.get("bids") or [])
            ],
            "asks": [
                {"price": float(level[0]), "amount": float(level[1])}
                for level in (orderbook.get("asks") or [])
            ],
            "timestamp": orderbook.get("timestamp", int(time.time() * 1000)),
            "datetime": orderbook.get(
                "datetime", datetime.now(timezone.utc).isoformat()
            ),
            "nonce": orderbook.get("nonce"),
            "source": "mexc",
        }

    @staticmethod
    def _normalize_funding_rate(
        funding: Dict[str, Any], symbol: str
    ) -> Dict[str, Any]:
        """Normalise a funding-rate event.

        Args:
            funding: Raw ccxt funding-rate dictionary.
            symbol: Symbol string.

        Returns:
            Normalised funding-rate dictionary.
        """
        return {
            "symbol": symbol,
            "funding_rate": float(funding.get("fundingRate", 0)),
            "funding_timestamp": funding.get("fundingTimestamp"),
            "next_funding_timestamp": funding.get("nextFundingTimestamp"),
            "timestamp": funding.get("timestamp", int(time.time() * 1000)),
            "datetime": funding.get(
                "datetime", datetime.now(timezone.utc).isoformat()
            ),
            "source": "mexc",
        }

    @staticmethod
    def _normalize_liquidation(
        liq: Dict[str, Any], symbol: str
    ) -> Dict[str, Any]:
        """Normalise a liquidation event.

        Args:
            liq: Raw liquidation dictionary.
            symbol: Symbol string.

        Returns:
            Normalised liquidation dictionary.
        """
        price = float(liq.get("price", 0))
        amount = float(liq.get("amount", liq.get("contracts", 0)))
        return {
            "symbol": symbol,
            "side": liq.get("side", ""),
            "price": price,
            "amount": amount,
            "notional_usd": price * amount,
            "timestamp": liq.get("timestamp", int(time.time() * 1000)),
            "datetime": liq.get(
                "datetime", datetime.now(timezone.utc).isoformat()
            ),
            "source": "mexc",
        }

    # ------------------------------------------------------------------
    # Internal – storage helpers
    # ------------------------------------------------------------------

    async def _maybe_store_orderbook_snapshot(
        self, symbol: str, orderbook: Dict[str, Any]
    ) -> None:
        """Persist an order-book snapshot if the configured interval has elapsed.

        Args:
            symbol: The symbol for this order book.
            orderbook: Normalised order-book dictionary.
        """
        now = time.time()
        last = self._last_orderbook_snapshot.get(symbol, 0.0)
        if now - last < self._orderbook_snapshot_interval:
            return

        self._last_orderbook_snapshot[symbol] = now
        try:
            await self._run_storage_call(
                self._storage.store_orderbook_snapshot, orderbook
            )
            logger.debug("Stored orderbook snapshot for %s", symbol)
        except Exception as exc:
            logger.error(
                "Failed to store orderbook snapshot for %s: %s", symbol, exc
            )

    async def _store_funding_rate(self, funding: Dict[str, Any]) -> None:
        """Persist a funding-rate record.

        Args:
            funding: Normalised funding-rate dictionary.
        """
        try:
            await self._run_storage_call(
                self._storage.store_funding_rate, funding
            )
            logger.debug(
                "Stored funding rate for %s: %s",
                funding["symbol"],
                funding["funding_rate"],
            )
        except Exception as exc:
            logger.error("Failed to store funding rate: %s", exc)

    async def _store_liquidation(self, liquidation: Dict[str, Any]) -> None:
        """Persist a liquidation event and trigger alerts for large ones.

        Args:
            liquidation: Normalised liquidation dictionary.
        """
        try:
            await self._run_storage_call(
                self._storage.store_liquidation, liquidation
            )
            logger.info(
                "Stored liquidation for %s: side=%s notional=$%.2f",
                liquidation["symbol"],
                liquidation["side"],
                liquidation["notional_usd"],
            )

            if liquidation["notional_usd"] >= self._liquidation_alert_threshold:
                logger.warning(
                    "LARGE LIQUIDATION ALERT – %s %s $%.2f",
                    liquidation["symbol"],
                    liquidation["side"],
                    liquidation["notional_usd"],
                )
                await self._run_storage_call(
                    self._storage.store_alert,
                    {
                        "type": "large_liquidation",
                        "symbol": liquidation["symbol"],
                        "side": liquidation["side"],
                        "notional_usd": liquidation["notional_usd"],
                        "timestamp": liquidation["timestamp"],
                        "datetime": liquidation["datetime"],
                    },
                )
        except Exception as exc:
            logger.error("Failed to store liquidation: %s", exc)

    @staticmethod
    async def _run_storage_call(
        func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Invoke a storage method, awaiting it if it returns a coroutine.

        This allows the streaming module to work with both sync and async
        ``StorageManager`` implementations.

        Args:
            func: The storage method to call.
            *args: Positional arguments forwarded to *func*.
            **kwargs: Keyword arguments forwarded to *func*.

        Returns:
            Whatever *func* returns.
        """
        result = func(*args, **kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result

    # ------------------------------------------------------------------
    # Raw websockets fallback – MEXC protocol
    # ------------------------------------------------------------------

    async def _raw_connect(self, url: str, key: str) -> Any:
        """Open a raw WebSocket connection and store it.

        Args:
            url: WebSocket endpoint URL.
            key: Internal key for tracking the connection.

        Returns:
            The ``websockets`` connection object.
        """
        if websockets is None:
            raise RuntimeError(
                "The 'websockets' library is required for fallback mode"
            )
        ws = await websockets.connect(url, ping_interval=self._heartbeat_interval)
        self._raw_ws_connections[key] = ws
        logger.info("Raw WS connected: %s", key)
        return ws

    async def _raw_send(self, ws: Any, payload: Dict[str, Any]) -> None:
        """Send a JSON payload over a raw WebSocket.

        Args:
            ws: The websocket connection.
            payload: Dictionary to serialise and send.
        """
        await ws.send(json.dumps(payload))

    async def _raw_subscribe_trades(self, symbol: str) -> None:
        """Fallback trade subscription using raw MEXC WebSocket protocol.

        Args:
            symbol: Unified symbol string (converted to MEXC format).
        """
        mexc_symbol = symbol.replace("/", "_").upper()
        key = f"raw:{symbol}:trades"
        ws = await self._raw_connect(_MEXC_WS_SPOT, key)

        subscribe_msg = {
            "method": "SUBSCRIPTION",
            "params": [f"spot@public.deals.v3.api@{mexc_symbol}"],
        }
        await self._raw_send(ws, subscribe_msg)

        async for raw_msg in ws:
            if not self._running:
                break
            try:
                msg = json.loads(raw_msg)
                if "d" not in msg:
                    continue
                data = msg["d"]
                deals = data.get("deals", [])
                for deal in deals:
                    normalized = {
                        "symbol": symbol,
                        "price": float(deal.get("p", 0)),
                        "amount": float(deal.get("v", 0)),
                        "side": "buy" if deal.get("S") == 1 else "sell",
                        "timestamp": int(deal.get("t", time.time() * 1000)),
                        "datetime": datetime.fromtimestamp(
                            int(deal.get("t", time.time() * 1000)) / 1000,
                            tz=timezone.utc,
                        ).isoformat(),
                        "trade_id": str(deal.get("t", "")),
                        "source": "mexc",
                    }
                    await self._dispatch_callback(
                        self._on_trade_callback, normalized
                    )
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                logger.warning("Raw trade parse error: %s", exc)

    async def _raw_subscribe_ticker(self, symbol: str) -> None:
        """Fallback ticker subscription via raw MEXC WebSocket.

        Args:
            symbol: Unified symbol string.
        """
        mexc_symbol = symbol.replace("/", "_").upper()
        key = f"raw:{symbol}:ticker"
        ws = await self._raw_connect(_MEXC_WS_SPOT, key)

        subscribe_msg = {
            "method": "SUBSCRIPTION",
            "params": [f"spot@public.miniTicker.v3.api@{mexc_symbol}"],
        }
        await self._raw_send(ws, subscribe_msg)

        async for raw_msg in ws:
            if not self._running:
                break
            try:
                msg = json.loads(raw_msg)
                if "d" not in msg:
                    continue
                data = msg["d"]
                normalized = {
                    "symbol": symbol,
                    "last": float(data.get("c", 0)),
                    "bid": 0.0,
                    "ask": 0.0,
                    "high": float(data.get("h", 0)),
                    "low": float(data.get("l", 0)),
                    "volume": float(data.get("v", 0)),
                    "quote_volume": float(data.get("qv", 0)),
                    "timestamp": int(data.get("t", time.time() * 1000)),
                    "datetime": datetime.fromtimestamp(
                        int(data.get("t", time.time() * 1000)) / 1000,
                        tz=timezone.utc,
                    ).isoformat(),
                    "source": "mexc",
                }
                await self._dispatch_callback(
                    self._on_ticker_callback, normalized
                )
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                logger.warning("Raw ticker parse error: %s", exc)

    async def _raw_subscribe_orderbook(self, symbol: str, depth: int = 20) -> None:
        """Fallback order-book subscription via raw MEXC WebSocket.

        Args:
            symbol: Unified symbol string.
            depth: Number of levels (used in channel name).
        """
        mexc_symbol = symbol.replace("/", "_").upper()
        key = f"raw:{symbol}:orderbook"
        ws = await self._raw_connect(_MEXC_WS_SPOT, key)

        subscribe_msg = {
            "method": "SUBSCRIPTION",
            "params": [
                f"spot@public.limit.v3.api@{mexc_symbol}@{depth}"
            ],
        }
        await self._raw_send(ws, subscribe_msg)

        async for raw_msg in ws:
            if not self._running:
                break
            try:
                msg = json.loads(raw_msg)
                if "d" not in msg:
                    continue
                data = msg["d"]
                bids_raw = data.get("bids", [])
                asks_raw = data.get("asks", [])
                normalized = {
                    "symbol": symbol,
                    "bids": [
                        {"price": float(b.get("p", 0)), "amount": float(b.get("v", 0))}
                        for b in bids_raw
                    ],
                    "asks": [
                        {"price": float(a.get("p", 0)), "amount": float(a.get("v", 0))}
                        for a in asks_raw
                    ],
                    "timestamp": int(data.get("t", time.time() * 1000)),
                    "datetime": datetime.fromtimestamp(
                        int(data.get("t", time.time() * 1000)) / 1000,
                        tz=timezone.utc,
                    ).isoformat(),
                    "nonce": data.get("r"),
                    "source": "mexc",
                }
                await self._dispatch_callback(
                    self._on_orderbook_callback, normalized
                )
                await self._maybe_store_orderbook_snapshot(symbol, normalized)
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                logger.warning("Raw orderbook parse error: %s", exc)

    async def _raw_subscribe_funding_rate(self, symbol: str) -> None:
        """Fallback funding-rate subscription via MEXC futures WebSocket.

        Args:
            symbol: Unified perpetual-futures symbol.
        """
        mexc_symbol = symbol.replace("/", "_").replace(":", "_").upper()
        key = f"raw:{symbol}:funding"
        ws = await self._raw_connect(_MEXC_WS_FUTURES, key)

        subscribe_msg = {
            "method": "sub.funding.rate",
            "param": {"symbol": mexc_symbol},
        }
        await self._raw_send(ws, subscribe_msg)

        async for raw_msg in ws:
            if not self._running:
                break
            try:
                msg = json.loads(raw_msg)
                data = msg.get("data", {})
                if not data:
                    continue
                normalized = {
                    "symbol": symbol,
                    "funding_rate": float(data.get("fundingRate", 0)),
                    "funding_timestamp": data.get("nextSettleTime"),
                    "next_funding_timestamp": data.get("nextSettleTime"),
                    "timestamp": int(data.get("timestamp", time.time() * 1000)),
                    "datetime": datetime.fromtimestamp(
                        int(data.get("timestamp", time.time() * 1000)) / 1000,
                        tz=timezone.utc,
                    ).isoformat(),
                    "source": "mexc",
                }
                await self._store_funding_rate(normalized)
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                logger.warning("Raw funding rate parse error: %s", exc)

    async def _raw_subscribe_liquidations(self, symbol: str) -> None:
        """Fallback liquidation subscription via MEXC futures WebSocket.

        Args:
            symbol: Unified symbol string.
        """
        mexc_symbol = symbol.replace("/", "_").replace(":", "_").upper()
        key = f"raw:{symbol}:liquidations"
        ws = await self._raw_connect(_MEXC_WS_FUTURES, key)

        subscribe_msg = {
            "method": "sub.liquidation.order",
            "param": {"symbol": mexc_symbol},
        }
        await self._raw_send(ws, subscribe_msg)

        async for raw_msg in ws:
            if not self._running:
                break
            try:
                msg = json.loads(raw_msg)
                data = msg.get("data", {})
                if not data:
                    continue
                price = float(data.get("price", 0))
                amount = float(data.get("vol", 0))
                normalized = {
                    "symbol": symbol,
                    "side": "buy" if data.get("side") == 1 else "sell",
                    "price": price,
                    "amount": amount,
                    "notional_usd": price * amount,
                    "timestamp": int(data.get("timestamp", time.time() * 1000)),
                    "datetime": datetime.fromtimestamp(
                        int(data.get("timestamp", time.time() * 1000)) / 1000,
                        tz=timezone.utc,
                    ).isoformat(),
                    "source": "mexc",
                }
                await self._store_liquidation(normalized)
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                logger.warning("Raw liquidation parse error: %s", exc)
