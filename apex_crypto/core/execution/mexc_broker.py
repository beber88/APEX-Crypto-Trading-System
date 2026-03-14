"""MEXC exchange broker for the APEX Crypto Trading System.

Handles all order execution on MEXC via the ccxt library, supporting
both spot and futures markets.  Includes paper trading mode for
simulation without hitting the live exchange.

Usage::

    config = {
        "exchange": {"testnet": False},
        "paper_trading": False,
        "rate_limit_ms": 100,
    }
    broker = MEXCBroker(config)
    order = await broker.place_limit_order("BTC/USDT", "buy", 0.01, 42000.0)
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import ccxt.async_support as ccxt

from apex_crypto.core.logging import get_logger, log_with_data

logger = get_logger("execution.mexc_broker")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_RETRIES = 5
_BASE_BACKOFF_SECONDS = 0.5
_PAPER_FILL_SLIPPAGE_PCT = 0.0005  # 0.05 % simulated slippage


class MEXCBroker:
    """Broker that routes orders to MEXC via ccxt.

    Supports limit, market, and stop-limit orders for both spot and
    futures.  When ``paper_trading`` is enabled in the config, all orders
    are simulated locally and no requests are sent to the exchange.

    Args:
        config: System configuration dictionary.  Relevant keys:

            - ``exchange.testnet`` (bool): Use MEXC sandbox.
            - ``paper_trading`` (bool): Simulate fills locally.
            - ``rate_limit_ms`` (int): Minimum ms between requests.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._paper_trading: bool = config.get("paper_trading", False)

        exchange_cfg = config.get("exchange", {})
        testnet: bool = exchange_cfg.get("testnet", False)

        api_key = os.environ.get("MEXC_API_KEY", "")
        secret_key = os.environ.get("MEXC_SECRET_KEY", "")

        self._exchange: ccxt.mexc = ccxt.mexc({
            "apiKey": api_key,
            "secret": secret_key,
            "enableRateLimit": True,
            "rateLimit": config.get("rate_limit_ms", 100),
            "options": {
                "defaultType": exchange_cfg.get("default_type", "swap"),
                "fetchCurrencies": False,
            },
        })

        if testnet:
            log_with_data(logger, "warning", "MEXC does not support sandbox mode in ccxt; skipping set_sandbox_mode. Paper trading is handled internally.")

        # Paper trading bookkeeping
        self._paper_orders: dict[str, dict[str, Any]] = {}
        self._paper_balance: dict[str, float] = {
            "total_usdt": config.get("paper_initial_balance", 10_000.0),
            "free_usdt": config.get("paper_initial_balance", 10_000.0),
            "used_usdt": 0.0,
        }
        self._paper_positions: dict[str, dict[str, Any]] = {}

        log_with_data(logger, "info", "MEXCBroker initialised", {
            "testnet": testnet,
            "paper_trading": self._paper_trading,
            "rate_limit_ms": config.get("rate_limit_ms", 100),
        })

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying ccxt exchange connection."""
        await self._exchange.close()
        log_with_data(logger, "info", "MEXCBroker connection closed")

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    async def place_limit_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Place a limit order on MEXC spot or futures.

        Args:
            symbol: Trading pair (e.g. ``"BTC/USDT"``).
            side: ``"buy"`` or ``"sell"``.
            amount: Order quantity in base currency.
            price: Limit price.
            params: Extra ccxt parameters forwarded to the exchange.

        Returns:
            Normalised order dict with keys: ``order_id``, ``symbol``,
            ``side``, ``type``, ``price``, ``amount``, ``status``,
            ``timestamp``.
        """
        if self._paper_trading:
            return self._simulate_order(symbol, side, "limit", amount, price)

        raw = await self._retry(
            self._exchange.create_limit_order,
            symbol, side, amount, price, params or {},
        )

        order = self._normalise_order(raw)
        log_with_data(logger, "info", "Limit order placed", order)
        return order

    async def place_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Place a market order (for breakout entries).

        Args:
            symbol: Trading pair.
            side: ``"buy"`` or ``"sell"``.
            amount: Order quantity in base currency.
            params: Extra ccxt parameters.

        Returns:
            Normalised order dict.
        """
        if self._paper_trading:
            return self._simulate_order(symbol, side, "market", amount)

        raw = await self._retry(
            self._exchange.create_market_order,
            symbol, side, amount, params or {},
        )

        order = self._normalise_order(raw)
        log_with_data(logger, "info", "Market order placed", order)
        return order

    async def place_stop_limit(
        self,
        symbol: str,
        side: str,
        amount: float,
        stop_price: float,
        limit_price: float,
    ) -> dict[str, Any]:
        """Place a stop-limit order for stop losses.

        The limit price is set slightly beyond the stop price to
        maximise fill probability during volatile moves.

        Args:
            symbol: Trading pair.
            side: ``"buy"`` or ``"sell"``.
            amount: Order quantity in base currency.
            stop_price: Trigger / stop price.
            limit_price: Limit price placed once triggered.

        Returns:
            Normalised order dict.
        """
        if self._paper_trading:
            return self._simulate_order(
                symbol, side, "stop_limit", amount,
                price=limit_price, stop_price=stop_price,
            )

        params: dict[str, Any] = {
            "stopPrice": stop_price,
        }

        raw = await self._retry(
            self._exchange.create_order,
            symbol, "limit", side, amount, limit_price, params,
        )

        order = self._normalise_order(raw)
        order["stop_price"] = stop_price
        log_with_data(logger, "info", "Stop-limit order placed", order)
        return order

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    async def cancel_order(
        self, order_id: str, symbol: str
    ) -> dict[str, Any]:
        """Cancel an open order.

        Args:
            order_id: Exchange order identifier.
            symbol: Trading pair the order belongs to.

        Returns:
            Normalised order dict reflecting the cancelled state.
        """
        if self._paper_trading:
            return self._paper_cancel(order_id)

        raw = await self._retry(
            self._exchange.cancel_order, order_id, symbol,
        )

        result = self._normalise_order(raw)
        log_with_data(logger, "info", "Order cancelled", result)
        return result

    async def cancel_all_orders(self, symbol: str) -> list[dict[str, Any]]:
        """Cancel all open orders for a symbol.

        Args:
            symbol: Trading pair.

        Returns:
            List of normalised order dicts for each cancelled order.
        """
        if self._paper_trading:
            return self._paper_cancel_all(symbol)

        open_orders = await self.get_open_orders(symbol)
        results: list[dict[str, Any]] = []
        for order in open_orders:
            try:
                cancelled = await self.cancel_order(order["order_id"], symbol)
                results.append(cancelled)
            except Exception as exc:
                log_with_data(logger, "warning", "Failed to cancel order", {
                    "order_id": order["order_id"],
                    "symbol": symbol,
                    "error": str(exc),
                })

        log_with_data(logger, "info", "All orders cancelled for symbol", {
            "symbol": symbol,
            "cancelled_count": len(results),
        })
        return results

    async def get_open_orders(
        self, symbol: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """Return all open orders, optionally filtered by symbol.

        Args:
            symbol: Trading pair filter.  ``None`` returns all symbols.

        Returns:
            List of normalised order dicts.
        """
        if self._paper_trading:
            return self._paper_get_open_orders(symbol)

        raw_orders = await self._retry(
            self._exchange.fetch_open_orders, symbol,
        )

        orders = [self._normalise_order(o) for o in raw_orders]
        log_with_data(logger, "debug", "Fetched open orders", {
            "symbol": symbol,
            "count": len(orders),
        })
        return orders

    # ------------------------------------------------------------------
    # Account information
    # ------------------------------------------------------------------

    async def get_balance(self) -> dict[str, Any]:
        """Return account balance summary.

        Returns:
            Dict with keys: ``total_usdt``, ``free_usdt``,
            ``used_usdt``, ``positions``.
        """
        if self._paper_trading:
            return {
                **self._paper_balance,
                "positions": dict(self._paper_positions),
            }

        raw = await self._retry(self._exchange.fetch_balance)

        usdt = raw.get("USDT", {})
        balance: dict[str, Any] = {
            "total_usdt": float(usdt.get("total", 0.0)),
            "free_usdt": float(usdt.get("free", 0.0)),
            "used_usdt": float(usdt.get("used", 0.0)),
            "positions": {},
        }

        # Attach futures positions if available
        if "info" in raw and isinstance(raw["info"], dict):
            positions_raw = raw["info"].get("positions", [])
            for pos in positions_raw:
                pos_symbol = pos.get("symbol", "")
                pos_amt = float(pos.get("positionAmt", 0))
                if pos_amt != 0.0:
                    balance["positions"][pos_symbol] = {
                        "amount": pos_amt,
                        "entry_price": float(pos.get("entryPrice", 0)),
                        "unrealized_pnl": float(pos.get("unrealizedProfit", 0)),
                    }

        log_with_data(logger, "debug", "Balance fetched", {
            "total_usdt": balance["total_usdt"],
            "free_usdt": balance["free_usdt"],
        })
        return balance

    async def get_position(self, symbol: str) -> dict[str, Any]:
        """Return the current futures position for a symbol.

        Args:
            symbol: Trading pair.

        Returns:
            Position dict with keys: ``symbol``, ``side``, ``amount``,
            ``entry_price``, ``unrealized_pnl``, ``leverage``,
            ``liquidation_price``.  Returns an empty-position dict
            if no position is open.
        """
        if self._paper_trading:
            return self._paper_positions.get(symbol, {
                "symbol": symbol,
                "side": "none",
                "amount": 0.0,
                "entry_price": 0.0,
                "unrealized_pnl": 0.0,
                "leverage": 1,
                "liquidation_price": 0.0,
            })

        positions = await self._retry(
            self._exchange.fetch_positions, [symbol],
        )

        for pos in positions:
            contracts = float(pos.get("contracts", 0))
            if contracts != 0.0:
                result: dict[str, Any] = {
                    "symbol": symbol,
                    "side": pos.get("side", "none"),
                    "amount": contracts,
                    "entry_price": float(pos.get("entryPrice", 0)),
                    "unrealized_pnl": float(pos.get("unrealizedPnl", 0)),
                    "leverage": int(pos.get("leverage", 1)),
                    "liquidation_price": float(pos.get("liquidationPrice", 0)),
                }
                log_with_data(logger, "debug", "Position fetched", result)
                return result

        empty: dict[str, Any] = {
            "symbol": symbol,
            "side": "none",
            "amount": 0.0,
            "entry_price": 0.0,
            "unrealized_pnl": 0.0,
            "leverage": 1,
            "liquidation_price": 0.0,
        }
        return empty

    async def set_leverage(
        self, symbol: str, leverage: int
    ) -> dict[str, Any]:
        """Set leverage for a futures position.

        Args:
            symbol: Trading pair.
            leverage: Desired leverage multiplier.

        Returns:
            Dict confirming the leverage change with keys ``symbol``
            and ``leverage``.
        """
        if self._paper_trading:
            log_with_data(logger, "info", "Paper leverage set", {
                "symbol": symbol,
                "leverage": leverage,
            })
            return {"symbol": symbol, "leverage": leverage}

        raw = await self._retry(
            self._exchange.set_leverage, leverage, symbol,
        )

        result: dict[str, Any] = {
            "symbol": symbol,
            "leverage": leverage,
            "raw": raw,
        }
        log_with_data(logger, "info", "Leverage set", result)
        return result

    async def close_position(self, symbol: str) -> dict[str, Any]:
        """Close the entire position for a symbol at market.

        Args:
            symbol: Trading pair.

        Returns:
            Normalised order dict for the closing market order.
            Returns a status dict if no position exists.
        """
        position = await self.get_position(symbol)

        if position["amount"] == 0.0:
            log_with_data(logger, "info", "No position to close", {
                "symbol": symbol,
            })
            return {
                "symbol": symbol,
                "status": "no_position",
                "message": "No open position found",
            }

        close_side = "sell" if position["side"] == "long" else "buy"
        amount = abs(position["amount"])

        if self._paper_trading:
            entry_price = position.get("entry_price", 0)
            current_price = self._estimate_market_price(symbol)
            if current_price <= 0:
                current_price = entry_price

            # Calculate realized P&L
            if position["side"] == "long":
                realized_pnl = (current_price - entry_price) * amount
            else:
                realized_pnl = (entry_price - current_price) * amount

            # Update paper balance: release margin and apply P&L
            entry_cost = entry_price * amount
            self._paper_balance["used_usdt"] = max(0.0, self._paper_balance["used_usdt"] - entry_cost)
            self._paper_balance["free_usdt"] += entry_cost + realized_pnl
            self._paper_balance["total_usdt"] += realized_pnl

            self._paper_positions.pop(symbol, None)

            order_id = f"paper_{uuid.uuid4().hex[:12]}"
            order: dict[str, Any] = {
                "order_id": order_id,
                "symbol": symbol,
                "side": close_side,
                "type": "market",
                "price": current_price,
                "amount": amount,
                "filled": amount,
                "remaining": 0.0,
                "status": "closed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "realized_pnl": realized_pnl,
            }
            self._paper_orders[order_id] = order
            log_with_data(logger, "info", "Paper position closed", {
                "symbol": symbol,
                "pnl": round(realized_pnl, 2),
                "total_equity": round(self._paper_balance["total_usdt"], 2),
            })
            return order

        order = await self.place_market_order(
            symbol, close_side, amount, {"reduceOnly": True},
        )
        log_with_data(logger, "info", "Position closed", {
            "symbol": symbol,
            "side": close_side,
            "amount": amount,
        })
        return order

    # ------------------------------------------------------------------
    # Full entry execution
    # ------------------------------------------------------------------

    async def execute_entry(self, signal: dict[str, Any]) -> dict[str, Any]:
        """Execute a complete trade entry from a signal.

        Performs the full entry workflow:

        1. Set leverage for the symbol.
        2. Place the entry order (market or limit).
        3. Place the stop-loss order.
        4. Place take-profit order(s).

        Args:
            signal: Signal dictionary with keys:

                - ``symbol`` (str): Trading pair.
                - ``direction`` (str): ``"long"`` or ``"short"``.
                - ``entry_price`` (float): Desired entry price.
                - ``entry_type`` (str): ``"market"`` or ``"limit"``.
                - ``amount`` (float): Position size in base currency.
                - ``stop_loss`` (float): Stop-loss price.
                - ``take_profit`` (float | list[dict]): TP price or
                  list of ``{price, pct}`` dicts.
                - ``leverage`` (int): Desired leverage.
                - ``strategy`` (str): Name of the originating strategy.
                - ``signal_score`` (float): Signal conviction score.

        Returns:
            Complete trade record dict with entry order, SL/TP order
            IDs, and metadata.
        """
        symbol: str = signal["symbol"]
        direction: str = signal["direction"]
        side = "buy" if direction == "long" else "sell"
        entry_type: str = signal.get("entry_type", "limit")
        amount: float = signal["amount"]
        leverage: int = signal.get("leverage", 1)
        stop_loss: float = signal["stop_loss"]
        take_profit = signal.get("take_profit")

        log_with_data(logger, "info", "Executing entry", {
            "symbol": symbol,
            "direction": direction,
            "entry_type": entry_type,
            "amount": amount,
            "leverage": leverage,
        })

        # 1. Set leverage
        await self.set_leverage(symbol, leverage)

        # 2. Place entry order
        if entry_type == "market":
            entry_order = await self.place_market_order(symbol, side, amount)
        else:
            entry_price: float = signal["entry_price"]
            entry_order = await self.place_limit_order(
                symbol, side, amount, entry_price,
            )

        # 3. Place stop-loss order
        sl_side = "sell" if direction == "long" else "buy"
        sl_buffer = stop_loss * (0.002 if direction == "long" else -0.002)
        sl_limit_price = stop_loss - sl_buffer if direction == "long" else stop_loss + abs(sl_buffer)
        sl_order = await self.place_stop_limit(
            symbol, sl_side, amount, stop_loss, sl_limit_price,
        )

        # 4. Place take-profit order(s)
        tp_orders: list[dict[str, Any]] = []
        tp_side = sl_side
        if take_profit is not None:
            if isinstance(take_profit, list):
                for tp_level in take_profit:
                    tp_price = tp_level["price"]
                    tp_amount = amount * tp_level.get("pct", 1.0)
                    tp_order = await self.place_limit_order(
                        symbol, tp_side, tp_amount, tp_price,
                    )
                    tp_orders.append(tp_order)
            else:
                tp_order = await self.place_limit_order(
                    symbol, tp_side, amount, float(take_profit),
                )
                tp_orders.append(tp_order)

        trade_record: dict[str, Any] = {
            "trade_id": str(uuid.uuid4()),
            "symbol": symbol,
            "direction": direction,
            "strategy": signal.get("strategy", "unknown"),
            "signal_score": signal.get("signal_score", 0.0),
            "entry_order": entry_order,
            "entry_order_id": entry_order["order_id"],
            "sl_order_id": sl_order["order_id"],
            "sl_price": stop_loss,
            "tp_order_ids": [tp["order_id"] for tp in tp_orders],
            "tp_orders": tp_orders,
            "amount": amount,
            "leverage": leverage,
            "status": "open",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Track paper positions so get_position / close_position work
        if self._paper_trading:
            entry_price_actual = entry_order.get("price", signal.get("entry_price", 0))
            self._paper_positions[symbol] = {
                "symbol": symbol,
                "side": "long" if direction == "long" else "short",
                "amount": amount,
                "entry_price": entry_price_actual,
                "unrealized_pnl": 0.0,
                "leverage": leverage,
                "liquidation_price": 0.0,
            }

        log_with_data(logger, "info", "Entry execution complete", {
            "trade_id": trade_record["trade_id"],
            "symbol": symbol,
            "entry_order_id": entry_order["order_id"],
            "sl_order_id": sl_order["order_id"],
            "tp_count": len(tp_orders),
        })

        return trade_record

    # ------------------------------------------------------------------
    # Retry logic with exponential backoff
    # ------------------------------------------------------------------

    async def _retry(self, func, *args, **kwargs) -> Any:
        """Execute a ccxt call with exponential backoff on rate limits.

        Retries up to ``_MAX_RETRIES`` times when the exchange returns
        a rate-limit or temporary error.  Raises the final exception
        if all retries are exhausted.

        Args:
            func: Async ccxt method to call.
            *args: Positional arguments forwarded to *func*.
            **kwargs: Keyword arguments forwarded to *func*.

        Returns:
            The result of the ccxt call.

        Raises:
            ccxt.BaseError: If all retries fail.
        """
        last_exception: Optional[Exception] = None

        for attempt in range(_MAX_RETRIES):
            try:
                return await func(*args, **kwargs)
            except (
                ccxt.RateLimitExceeded,
                ccxt.DDoSProtection,
                ccxt.RequestTimeout,
                ccxt.NetworkError,
            ) as exc:
                last_exception = exc
                wait = _BASE_BACKOFF_SECONDS * (2 ** attempt)
                log_with_data(logger, "warning", "Rate limit / transient error, retrying", {
                    "attempt": attempt + 1,
                    "max_retries": _MAX_RETRIES,
                    "wait_seconds": wait,
                    "error": str(exc),
                })
                await asyncio.sleep(wait)
            except ccxt.InsufficientFunds as exc:
                log_with_data(logger, "error", "Insufficient funds", {
                    "error": str(exc),
                })
                raise
            except ccxt.InvalidOrder as exc:
                log_with_data(logger, "error", "Invalid order parameters", {
                    "error": str(exc),
                })
                raise
            except ccxt.AuthenticationError as exc:
                log_with_data(logger, "error", "Authentication failed", {
                    "error": str(exc),
                })
                raise
            except ccxt.ExchangeNotAvailable as exc:
                last_exception = exc
                wait = _BASE_BACKOFF_SECONDS * (2 ** attempt)
                log_with_data(logger, "warning", "Exchange unavailable, retrying", {
                    "attempt": attempt + 1,
                    "wait_seconds": wait,
                    "error": str(exc),
                })
                await asyncio.sleep(wait)

        log_with_data(logger, "error", "All retries exhausted", {
            "function": func.__name__,
            "error": str(last_exception),
        })
        raise last_exception  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_order(raw: dict[str, Any]) -> dict[str, Any]:
        """Convert a raw ccxt order response to a normalised format.

        Args:
            raw: Raw order dict from ccxt.

        Returns:
            Normalised order dict.
        """
        return {
            "order_id": str(raw.get("id", "")),
            "symbol": raw.get("symbol", ""),
            "side": raw.get("side", ""),
            "type": raw.get("type", ""),
            "price": float(raw.get("price", 0) or 0),
            "amount": float(raw.get("amount", 0) or 0),
            "filled": float(raw.get("filled", 0) or 0),
            "remaining": float(raw.get("remaining", 0) or 0),
            "status": raw.get("status", "unknown"),
            "timestamp": raw.get("datetime", datetime.now(timezone.utc).isoformat()),
        }

    # ------------------------------------------------------------------
    # Paper trading simulation
    # ------------------------------------------------------------------

    def _simulate_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: float = 0.0,
        stop_price: float = 0.0,
    ) -> dict[str, Any]:
        """Simulate an order fill for paper trading.

        Creates a virtual order record and updates paper balance.
        Market orders are filled immediately with simulated slippage.
        Limit and stop-limit orders are recorded as open.

        Args:
            symbol: Trading pair.
            side: ``"buy"`` or ``"sell"``.
            order_type: ``"market"``, ``"limit"``, or ``"stop_limit"``.
            amount: Order quantity.
            price: Limit price (0 for market).
            stop_price: Stop trigger price for stop-limit orders.

        Returns:
            Simulated order dict.
        """
        order_id = f"paper_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()

        if order_type == "market":
            fill_price = price if price > 0 else self._estimate_market_price(symbol)
            slippage = fill_price * _PAPER_FILL_SLIPPAGE_PCT
            if side == "buy":
                fill_price += slippage
            else:
                fill_price -= slippage

            status = "closed"
            filled = amount
            remaining = 0.0

            cost = fill_price * amount
            if side == "buy":
                self._paper_balance["free_usdt"] -= cost
                self._paper_balance["used_usdt"] += cost
            else:
                self._paper_balance["free_usdt"] += cost
                self._paper_balance["used_usdt"] -= cost
        else:
            fill_price = price
            status = "open"
            filled = 0.0
            remaining = amount

        order: dict[str, Any] = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "price": fill_price,
            "amount": amount,
            "filled": filled,
            "remaining": remaining,
            "status": status,
            "timestamp": now,
        }

        if stop_price > 0:
            order["stop_price"] = stop_price

        self._paper_orders[order_id] = order

        log_with_data(logger, "info", "Paper order simulated", order)
        return order

    def _estimate_market_price(self, symbol: str) -> float:
        """Estimate the current market price for paper trading.

        Uses the most recent paper order price for the symbol, or
        falls back to a default placeholder.

        Args:
            symbol: Trading pair.

        Returns:
            Estimated price as a float.
        """
        for oid in reversed(list(self._paper_orders.keys())):
            order = self._paper_orders[oid]
            if order["symbol"] == symbol and order["price"] > 0:
                return order["price"]
        return 0.0

    def _paper_cancel(self, order_id: str) -> dict[str, Any]:
        """Cancel a paper trading order.

        Args:
            order_id: Paper order identifier.

        Returns:
            Updated order dict with ``status`` set to ``"cancelled"``.
        """
        if order_id in self._paper_orders:
            self._paper_orders[order_id]["status"] = "cancelled"
            log_with_data(logger, "info", "Paper order cancelled", {
                "order_id": order_id,
            })
            return self._paper_orders[order_id]

        return {
            "order_id": order_id,
            "status": "not_found",
            "message": "Paper order not found",
        }

    def _paper_cancel_all(self, symbol: str) -> list[dict[str, Any]]:
        """Cancel all open paper orders for a symbol.

        Args:
            symbol: Trading pair.

        Returns:
            List of cancelled paper order dicts.
        """
        cancelled: list[dict[str, Any]] = []
        for oid, order in self._paper_orders.items():
            if order["symbol"] == symbol and order["status"] == "open":
                order["status"] = "cancelled"
                cancelled.append(order)

        log_with_data(logger, "info", "All paper orders cancelled", {
            "symbol": symbol,
            "cancelled_count": len(cancelled),
        })
        return cancelled

    def _paper_get_open_orders(
        self, symbol: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """Return open paper orders, optionally filtered by symbol.

        Args:
            symbol: Optional trading pair filter.

        Returns:
            List of open paper order dicts.
        """
        results: list[dict[str, Any]] = []
        for order in self._paper_orders.values():
            if order["status"] != "open":
                continue
            if symbol is not None and order["symbol"] != symbol:
                continue
            results.append(order)
        return results
