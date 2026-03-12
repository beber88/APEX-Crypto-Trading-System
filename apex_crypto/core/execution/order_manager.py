"""Order lifecycle manager for the APEX Crypto Trading System.

Manages OCO (one-cancels-other) simulation, multi-target take-profit
ladders, trailing stops, and order-fill monitoring.  Works in
conjunction with :class:`MEXCBroker` for exchange communication.

Usage::

    broker = MEXCBroker(config)
    manager = OrderManager(config, broker)

    oco = await manager.setup_oco(
        "BTC/USDT", "sell", 0.01, stop_price=40000, take_profit_price=50000,
    )
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional

from apex_crypto.core.execution.mexc_broker import MEXCBroker
from apex_crypto.core.logging import get_logger, log_with_data

logger = get_logger("execution.order_manager")


class OrderManager:
    """Manages order lifecycle, OCO simulation, and trailing stops.

    Provides higher-level order management on top of the raw broker
    interface, including coordinated stop-loss / take-profit pairs
    and automatic cancellation of counterpart orders upon fills.

    Args:
        config: System configuration dictionary.
        broker: An initialised :class:`MEXCBroker` instance.
    """

    def __init__(
        self,
        config: dict[str, Any],
        broker: MEXCBroker,
    ) -> None:
        self._config = config
        self._broker = broker

        # Maps trade_id -> {sl_order_id, tp_order_ids, symbol, side, amount, ...}
        self._oco_groups: dict[str, dict[str, Any]] = {}

        # Maps order_id -> trade_id for quick reverse lookups
        self._order_to_trade: dict[str, str] = {}

        # Trailing stop state: symbol -> {current_stop_id, current_stop_price, trail_distance, side, best_price}
        self._trailing_stops: dict[str, dict[str, Any]] = {}

        log_with_data(logger, "info", "OrderManager initialised")

    # ------------------------------------------------------------------
    # OCO simulation
    # ------------------------------------------------------------------

    async def setup_oco(
        self,
        symbol: str,
        side: str,
        amount: float,
        stop_price: float,
        take_profit_price: float,
    ) -> dict[str, Any]:
        """Set up an OCO (one-cancels-other) pair of SL and TP orders.

        Places both a stop-loss and a take-profit order.  When one
        fills, the :meth:`monitor_orders` loop cancels the other
        automatically.

        Args:
            symbol: Trading pair (e.g. ``"BTC/USDT"``).
            side: Exit side — ``"sell"`` for long positions,
                ``"buy"`` for short positions.
            amount: Position size in base currency.
            stop_price: Stop-loss trigger price.
            take_profit_price: Take-profit limit price.

        Returns:
            Dict with keys ``sl_order_id``, ``tp_order_id``,
            ``trade_group_id``.
        """
        # Place SL (stop-limit with small buffer for fill certainty)
        if side == "sell":
            sl_limit = stop_price * 0.998
        else:
            sl_limit = stop_price * 1.002

        sl_order = await self._broker.place_stop_limit(
            symbol, side, amount, stop_price, sl_limit,
        )

        # Place TP (limit order)
        tp_order = await self._broker.place_limit_order(
            symbol, side, amount, take_profit_price,
        )

        trade_group_id = f"oco_{sl_order['order_id']}_{tp_order['order_id']}"

        group: dict[str, Any] = {
            "trade_group_id": trade_group_id,
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "sl_order_id": sl_order["order_id"],
            "tp_order_ids": [tp_order["order_id"]],
            "stop_price": stop_price,
            "take_profit_price": take_profit_price,
            "status": "active",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        self._oco_groups[trade_group_id] = group
        self._order_to_trade[sl_order["order_id"]] = trade_group_id
        self._order_to_trade[tp_order["order_id"]] = trade_group_id

        log_with_data(logger, "info", "OCO pair set up", {
            "trade_group_id": trade_group_id,
            "symbol": symbol,
            "sl_order_id": sl_order["order_id"],
            "tp_order_id": tp_order["order_id"],
            "stop_price": stop_price,
            "take_profit_price": take_profit_price,
        })

        return {
            "sl_order_id": sl_order["order_id"],
            "tp_order_id": tp_order["order_id"],
            "trade_group_id": trade_group_id,
        }

    # ------------------------------------------------------------------
    # Multi-target take profit
    # ------------------------------------------------------------------

    async def setup_multi_tp(
        self,
        symbol: str,
        side: str,
        total_amount: float,
        stop_price: float,
        tp_levels: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Set up a stop-loss with multiple take-profit levels.

        Places one SL order for the full position and multiple TP
        orders at different price levels, each for a fraction of
        the total position.

        Args:
            symbol: Trading pair.
            side: Exit side (``"sell"`` for longs, ``"buy"`` for shorts).
            total_amount: Total position size in base currency.
            stop_price: Stop-loss trigger price.
            tp_levels: List of take-profit level dicts, each with
                ``price`` (float) and ``pct`` (float, fraction of
                total position).  Example::

                    [
                        {"price": 50000, "pct": 0.35},
                        {"price": 52000, "pct": 0.35},
                        {"price": 55000, "pct": 0.30},
                    ]

        Returns:
            Dict with keys ``sl_order_id``, ``tp_order_ids`` (list),
            ``trade_group_id``.

        Raises:
            ValueError: If TP percentages do not sum to approximately 1.0.
        """
        total_pct = sum(tp["pct"] for tp in tp_levels)
        if abs(total_pct - 1.0) > 0.01:
            raise ValueError(
                f"TP level percentages must sum to ~1.0, got {total_pct:.4f}"
            )

        # Place SL for full position
        if side == "sell":
            sl_limit = stop_price * 0.998
        else:
            sl_limit = stop_price * 1.002

        sl_order = await self._broker.place_stop_limit(
            symbol, side, total_amount, stop_price, sl_limit,
        )

        # Place TP orders at each level
        tp_order_ids: list[str] = []
        tp_orders: list[dict[str, Any]] = []
        for tp_level in tp_levels:
            tp_amount = total_amount * tp_level["pct"]
            tp_order = await self._broker.place_limit_order(
                symbol, side, tp_amount, tp_level["price"],
            )
            tp_order_ids.append(tp_order["order_id"])
            tp_orders.append(tp_order)

        trade_group_id = f"multi_tp_{sl_order['order_id']}"

        group: dict[str, Any] = {
            "trade_group_id": trade_group_id,
            "symbol": symbol,
            "side": side,
            "total_amount": total_amount,
            "sl_order_id": sl_order["order_id"],
            "tp_order_ids": tp_order_ids,
            "tp_levels": tp_levels,
            "stop_price": stop_price,
            "status": "active",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        self._oco_groups[trade_group_id] = group
        self._order_to_trade[sl_order["order_id"]] = trade_group_id
        for tp_id in tp_order_ids:
            self._order_to_trade[tp_id] = trade_group_id

        log_with_data(logger, "info", "Multi-TP orders set up", {
            "trade_group_id": trade_group_id,
            "symbol": symbol,
            "sl_order_id": sl_order["order_id"],
            "tp_count": len(tp_order_ids),
            "tp_order_ids": tp_order_ids,
        })

        return {
            "sl_order_id": sl_order["order_id"],
            "tp_order_ids": tp_order_ids,
            "trade_group_id": trade_group_id,
        }

    # ------------------------------------------------------------------
    # Trailing stop
    # ------------------------------------------------------------------

    async def update_trailing_stop(
        self,
        symbol: str,
        side: str,
        current_price: float,
        trail_distance: float,
        current_stop_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Update a trailing stop, tightening it as price moves favourably.

        The trailing stop only moves in the profitable direction — it
        never loosens.  When the stop needs to be moved, the existing
        stop order is cancelled and a new one is placed.

        Args:
            symbol: Trading pair.
            side: Position side — ``"long"`` or ``"short"``.
            current_price: The current market price.
            trail_distance: Absolute price distance to trail behind.
            current_stop_id: Existing stop order ID to cancel if the
                stop needs to move.  If ``None``, uses the internally
                tracked stop for the symbol.

        Returns:
            Dict with keys ``new_stop_id``, ``new_stop_price``,
            ``moved`` (bool indicating whether the stop was updated).
        """
        state = self._trailing_stops.get(symbol)

        if side == "long":
            new_stop_price = current_price - trail_distance
        else:
            new_stop_price = current_price + trail_distance

        # Only tighten, never loosen
        if state is not None:
            if side == "long" and new_stop_price <= state["current_stop_price"]:
                return {
                    "new_stop_id": state["current_stop_id"],
                    "new_stop_price": state["current_stop_price"],
                    "moved": False,
                }
            if side == "short" and new_stop_price >= state["current_stop_price"]:
                return {
                    "new_stop_id": state["current_stop_id"],
                    "new_stop_price": state["current_stop_price"],
                    "moved": False,
                }

        # Cancel old stop if it exists
        stop_to_cancel = current_stop_id
        if stop_to_cancel is None and state is not None:
            stop_to_cancel = state.get("current_stop_id")

        if stop_to_cancel is not None:
            try:
                await self._broker.cancel_order(stop_to_cancel, symbol)
            except Exception as exc:
                log_with_data(logger, "warning", "Failed to cancel old trailing stop", {
                    "order_id": stop_to_cancel,
                    "symbol": symbol,
                    "error": str(exc),
                })

        # Place new stop
        exit_side = "sell" if side == "long" else "buy"
        amount = state["amount"] if state is not None else 0.0
        if amount == 0.0:
            position = await self._broker.get_position(symbol)
            amount = abs(position.get("amount", 0.0))

        if side == "long":
            sl_limit = new_stop_price * 0.998
        else:
            sl_limit = new_stop_price * 1.002

        new_stop_order = await self._broker.place_stop_limit(
            symbol, exit_side, amount, new_stop_price, sl_limit,
        )

        # Update tracking state
        self._trailing_stops[symbol] = {
            "current_stop_id": new_stop_order["order_id"],
            "current_stop_price": new_stop_price,
            "trail_distance": trail_distance,
            "side": side,
            "amount": amount,
            "best_price": current_price,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        log_with_data(logger, "info", "Trailing stop updated", {
            "symbol": symbol,
            "side": side,
            "new_stop_price": new_stop_price,
            "new_stop_id": new_stop_order["order_id"],
            "current_price": current_price,
            "trail_distance": trail_distance,
        })

        return {
            "new_stop_id": new_stop_order["order_id"],
            "new_stop_price": new_stop_price,
            "moved": True,
        }

    # ------------------------------------------------------------------
    # Order monitoring
    # ------------------------------------------------------------------

    async def monitor_orders(
        self, open_trades: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Check the status of all orders tied to open trades.

        Detects fills and partial fills across all OCO groups.  When
        a take-profit fills, the corresponding stop-loss is cancelled.
        When a stop-loss fills, all take-profit orders for that
        position are cancelled.

        Args:
            open_trades: List of trade dicts, each containing at
                minimum ``symbol``, ``sl_order_id``, and
                ``tp_order_ids``.

        Returns:
            List of event dicts.  Each event contains:

                - ``event_type``: ``"sl_filled"``, ``"tp_filled"``,
                  ``"partial_fill"``, or ``"order_cancelled"``.
                - ``trade_id``: Identifier linking back to the trade.
                - ``order_id``: The order that triggered the event.
                - ``fill_price``: Price at which the fill occurred.
                - ``filled_amount``: Quantity filled.
                - ``timestamp``: ISO-format event timestamp.
        """
        events: list[dict[str, Any]] = []

        for trade in open_trades:
            symbol: str = trade["symbol"]
            trade_id: str = trade.get("trade_id", trade.get("trade_group_id", ""))
            sl_order_id: str = trade.get("sl_order_id", "")
            tp_order_ids: list[str] = trade.get("tp_order_ids", [])

            all_order_ids = []
            if sl_order_id:
                all_order_ids.append(("sl", sl_order_id))
            for tp_id in tp_order_ids:
                all_order_ids.append(("tp", tp_id))

            open_orders = await self._broker.get_open_orders(symbol)
            open_order_ids = {o["order_id"] for o in open_orders}

            # Build a lookup of current order states
            order_states: dict[str, dict[str, Any]] = {}
            for o in open_orders:
                order_states[o["order_id"]] = o

            for order_type, order_id in all_order_ids:
                if order_id in open_order_ids:
                    # Check for partial fills
                    order_info = order_states.get(order_id, {})
                    filled = order_info.get("filled", 0.0)
                    if filled > 0.0:
                        events.append({
                            "event_type": "partial_fill",
                            "trade_id": trade_id,
                            "order_id": order_id,
                            "order_type": order_type,
                            "fill_price": order_info.get("price", 0.0),
                            "filled_amount": filled,
                            "remaining": order_info.get("remaining", 0.0),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })
                    continue

                # Order is no longer open — it was filled or cancelled
                # Determine if it was a fill (not in open orders and was active)
                event_type = f"{order_type}_filled"

                event: dict[str, Any] = {
                    "event_type": event_type,
                    "trade_id": trade_id,
                    "order_id": order_id,
                    "order_type": order_type,
                    "fill_price": 0.0,
                    "filled_amount": 0.0,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

                if order_type == "sl":
                    # SL filled: cancel all TP orders
                    for tp_id in tp_order_ids:
                        try:
                            await self._broker.cancel_order(tp_id, symbol)
                            log_with_data(logger, "info", "TP cancelled after SL fill", {
                                "trade_id": trade_id,
                                "cancelled_tp_id": tp_id,
                            })
                        except Exception as exc:
                            log_with_data(logger, "warning", "Failed to cancel TP after SL fill", {
                                "trade_id": trade_id,
                                "tp_order_id": tp_id,
                                "error": str(exc),
                            })

                elif order_type == "tp":
                    # TP filled: check if all TPs are filled
                    remaining_tps = [
                        tid for tid in tp_order_ids
                        if tid != order_id and tid in open_order_ids
                    ]

                    if not remaining_tps:
                        # All TPs filled — cancel SL
                        if sl_order_id:
                            try:
                                await self._broker.cancel_order(sl_order_id, symbol)
                                log_with_data(logger, "info", "SL cancelled after all TPs filled", {
                                    "trade_id": trade_id,
                                    "cancelled_sl_id": sl_order_id,
                                })
                            except Exception as exc:
                                log_with_data(logger, "warning", "Failed to cancel SL after TP fill", {
                                    "trade_id": trade_id,
                                    "sl_order_id": sl_order_id,
                                    "error": str(exc),
                                })

                events.append(event)

        if events:
            log_with_data(logger, "info", "Order monitoring cycle complete", {
                "events_detected": len(events),
                "event_types": [e["event_type"] for e in events],
            })

        return events

    # ------------------------------------------------------------------
    # Partial fill handling
    # ------------------------------------------------------------------

    async def handle_partial_fill(
        self,
        order_id: str,
        filled_amount: float,
        remaining: float,
    ) -> dict[str, Any]:
        """Handle a partial fill by adjusting linked SL/TP orders.

        When an entry or TP order partially fills, the corresponding
        SL and remaining TP orders must be adjusted proportionally
        to reflect the actual position size.

        Args:
            order_id: The order that was partially filled.
            filled_amount: Quantity already filled.
            remaining: Quantity still open.

        Returns:
            Dict describing adjustments made, with keys
            ``order_id``, ``filled_amount``, ``remaining``,
            ``adjustments`` (list of modified order details).
        """
        trade_group_id = self._order_to_trade.get(order_id)
        adjustments: list[dict[str, Any]] = []

        if trade_group_id is None:
            log_with_data(logger, "warning", "Partial fill for untracked order", {
                "order_id": order_id,
            })
            return {
                "order_id": order_id,
                "filled_amount": filled_amount,
                "remaining": remaining,
                "adjustments": [],
            }

        group = self._oco_groups.get(trade_group_id)
        if group is None:
            return {
                "order_id": order_id,
                "filled_amount": filled_amount,
                "remaining": remaining,
                "adjustments": [],
            }

        symbol: str = group["symbol"]
        side: str = group["side"]
        original_amount: float = group.get("total_amount", group.get("amount", 0.0))

        if original_amount <= 0.0:
            return {
                "order_id": order_id,
                "filled_amount": filled_amount,
                "remaining": remaining,
                "adjustments": [],
            }

        fill_ratio = filled_amount / original_amount

        # If a TP filled partially, reduce the SL amount
        if order_id in group.get("tp_order_ids", []):
            sl_order_id = group.get("sl_order_id")
            if sl_order_id:
                try:
                    await self._broker.cancel_order(sl_order_id, symbol)
                except Exception as exc:
                    log_with_data(logger, "warning", "Failed to cancel SL for partial fill adjustment", {
                        "sl_order_id": sl_order_id,
                        "error": str(exc),
                    })

                new_sl_amount = original_amount - filled_amount
                if new_sl_amount > 0:
                    stop_price = group.get("stop_price", 0.0)
                    if side == "sell":
                        sl_limit = stop_price * 0.998
                    else:
                        sl_limit = stop_price * 1.002

                    new_sl_order = await self._broker.place_stop_limit(
                        symbol, side, new_sl_amount, stop_price, sl_limit,
                    )

                    group["sl_order_id"] = new_sl_order["order_id"]
                    self._order_to_trade[new_sl_order["order_id"]] = trade_group_id

                    adjustments.append({
                        "type": "sl_resized",
                        "old_order_id": sl_order_id,
                        "new_order_id": new_sl_order["order_id"],
                        "new_amount": new_sl_amount,
                    })

                    log_with_data(logger, "info", "SL resized after partial TP fill", {
                        "trade_group_id": trade_group_id,
                        "new_sl_amount": new_sl_amount,
                        "new_sl_order_id": new_sl_order["order_id"],
                    })

        result: dict[str, Any] = {
            "order_id": order_id,
            "filled_amount": filled_amount,
            "remaining": remaining,
            "adjustments": adjustments,
        }

        log_with_data(logger, "info", "Partial fill handled", result)
        return result

    # ------------------------------------------------------------------
    # Emergency close
    # ------------------------------------------------------------------

    async def emergency_close_all(self) -> list[dict[str, Any]]:
        """Close ALL positions at market and cancel ALL open orders.

        Intended for circuit-breaker or panic scenarios.  Iterates
        over every tracked symbol, cancels all orders, and then
        closes any remaining position at market.

        Returns:
            List of result dicts for each symbol, containing
            ``symbol``, ``orders_cancelled``, and ``close_result``.
        """
        log_with_data(logger, "warning", "EMERGENCY CLOSE ALL initiated")

        results: list[dict[str, Any]] = []

        # Collect all symbols from tracked groups and trailing stops
        symbols: set[str] = set()
        for group in self._oco_groups.values():
            symbols.add(group["symbol"])
        for sym in self._trailing_stops:
            symbols.add(sym)

        # Also fetch open orders to discover any untracked symbols
        try:
            all_open = await self._broker.get_open_orders()
            for order in all_open:
                symbols.add(order["symbol"])
        except Exception as exc:
            log_with_data(logger, "error", "Failed to fetch open orders during emergency close", {
                "error": str(exc),
            })

        for symbol in symbols:
            result: dict[str, Any] = {"symbol": symbol}

            # Cancel all orders
            try:
                cancelled = await self._broker.cancel_all_orders(symbol)
                result["orders_cancelled"] = len(cancelled)
            except Exception as exc:
                log_with_data(logger, "error", "Failed to cancel orders during emergency close", {
                    "symbol": symbol,
                    "error": str(exc),
                })
                result["orders_cancelled"] = 0
                result["cancel_error"] = str(exc)

            # Close position
            try:
                close_result = await self._broker.close_position(symbol)
                result["close_result"] = close_result
            except Exception as exc:
                log_with_data(logger, "error", "Failed to close position during emergency close", {
                    "symbol": symbol,
                    "error": str(exc),
                })
                result["close_result"] = {"status": "error", "error": str(exc)}

            results.append(result)

        # Clear internal state
        self._oco_groups.clear()
        self._order_to_trade.clear()
        self._trailing_stops.clear()

        log_with_data(logger, "warning", "EMERGENCY CLOSE ALL completed", {
            "symbols_processed": len(results),
            "results": [
                {"symbol": r["symbol"], "cancelled": r.get("orders_cancelled", 0)}
                for r in results
            ],
        })

        return results
