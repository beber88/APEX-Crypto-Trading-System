"""Position tracker for the APEX Crypto Trading System.

Tracks all open positions and their lifecycle, including entry,
partial closes, exits, and performance statistics.  Positions are
persisted in SQLite (via :class:`StorageManager`) and cached in
Redis for fast lookup.

Usage::

    from apex_crypto.core.data.storage import StorageManager
    storage = StorageManager(config)
    tracker = PositionTracker(config, storage)

    trade_id = tracker.open_position({
        "symbol": "BTC/USDT",
        "direction": "long",
        "entry_price": 42000.0,
        "amount": 0.01,
        "strategy": "breakout",
        "signal_score": 0.85,
        "sl_price": 40000.0,
        "tp_prices": [44000.0, 46000.0],
    })
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from apex_crypto.core.logging import get_logger, log_with_data

logger = get_logger("execution.position_tracker")


class PositionTracker:
    """Tracks all open positions and their full lifecycle.

    Maintains a dual-write approach: SQLite for durable trade logs
    and Redis for low-latency position lookups.  Provides real-time
    statistics for risk management decisions.

    Args:
        config: System configuration dictionary.
        storage: An initialised :class:`StorageManager` instance
            providing access to SQLite and Redis.
    """

    def __init__(self, config: dict[str, Any], storage: Any) -> None:
        self._config = config
        self._storage = storage

        # In-memory cache for fast access (mirrors Redis)
        self._positions: dict[str, dict[str, Any]] = {}

        # Equity tracking
        self._peak_equity: float = config.get("initial_equity", 10_000.0)
        self._current_equity: float = self._peak_equity

        # Load existing open positions from storage
        self._load_open_positions()

        log_with_data(logger, "info", "PositionTracker initialised", {
            "open_positions": len(self._positions),
            "peak_equity": self._peak_equity,
        })

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _load_open_positions(self) -> None:
        """Load open positions from SQLite and Redis into memory.

        Reconciles state between the two stores, preferring Redis
        for runtime fields and SQLite for durable records.
        """
        try:
            open_trades = self._storage.get_open_trades()
            for trade in open_trades:
                trade_id = trade["trade_id"]
                self._positions[trade_id] = trade

                # Also populate from Redis if available
                redis_state = self._storage.get_position_state(trade.get("symbol", ""))
                if redis_state and redis_state.get("trade_id") == trade_id:
                    self._positions[trade_id].update(redis_state)

            log_with_data(logger, "info", "Loaded open positions from storage", {
                "count": len(self._positions),
            })
        except Exception as exc:
            log_with_data(logger, "error", "Failed to load open positions", {
                "error": str(exc),
            })

    # ------------------------------------------------------------------
    # Position lifecycle
    # ------------------------------------------------------------------

    def open_position(self, trade_data: dict[str, Any]) -> str:
        """Record a new position opening.

        Creates a trade record in SQLite and caches the position
        state in Redis for fast access by the order manager and
        risk engine.

        Args:
            trade_data: Position details.  Required keys:

                - ``symbol`` (str): Trading pair.
                - ``direction`` (str): ``"long"`` or ``"short"``.
                - ``entry_price`` (float): Entry fill price.
                - ``amount`` (float): Position size in base currency.
                - ``strategy`` (str): Originating strategy name.

                Optional keys:

                - ``signal_score`` (float): Signal conviction score.
                - ``sl_price`` (float): Stop-loss price.
                - ``tp_prices`` (list[float]): Take-profit price(s).
                - ``order_ids`` (dict): Related exchange order IDs.
                - ``leverage`` (int): Applied leverage.
                - ``timeframe`` (str): Signal timeframe.

        Returns:
            The generated ``trade_id`` string.
        """
        trade_id = trade_data.get("trade_id", str(uuid.uuid4()))
        now = datetime.now(timezone.utc).isoformat()

        # Build the trade record for SQLite
        trade_record: dict[str, Any] = {
            "trade_id": trade_id,
            "symbol": trade_data["symbol"],
            "strategy": trade_data["strategy"],
            "direction": trade_data["direction"],
            "entry_price": trade_data["entry_price"],
            "quantity": trade_data["amount"],
            "entry_time": now,
            "status": "open",
            "stop_loss": trade_data.get("sl_price"),
            "take_profit": (
                trade_data["tp_prices"][0]
                if isinstance(trade_data.get("tp_prices"), list) and trade_data["tp_prices"]
                else trade_data.get("tp_prices")
            ),
            "timeframe": trade_data.get("timeframe"),
            "metadata": {
                "signal_score": trade_data.get("signal_score"),
                "leverage": trade_data.get("leverage", 1),
                "order_ids": trade_data.get("order_ids", {}),
                "tp_prices": trade_data.get("tp_prices", []),
            },
        }

        # Store in SQLite
        self._storage.record_trade(trade_record)

        # Build Redis state for fast lookups
        position_state: dict[str, Any] = {
            "trade_id": trade_id,
            "symbol": trade_data["symbol"],
            "direction": trade_data["direction"],
            "entry_price": trade_data["entry_price"],
            "amount": trade_data["amount"],
            "remaining_amount": trade_data["amount"],
            "strategy": trade_data["strategy"],
            "signal_score": trade_data.get("signal_score", 0.0),
            "sl_price": trade_data.get("sl_price"),
            "tp_prices": trade_data.get("tp_prices", []),
            "order_ids": trade_data.get("order_ids", {}),
            "leverage": trade_data.get("leverage", 1),
            "status": "open",
            "entry_time": now,
        }

        # Store in Redis
        self._storage.set_position_state(trade_data["symbol"], position_state)

        # Update in-memory cache
        self._positions[trade_id] = position_state

        # Increment daily trade counter
        self._storage.increment_daily_trade_count()

        log_with_data(logger, "info", "Position opened", {
            "trade_id": trade_id,
            "symbol": trade_data["symbol"],
            "direction": trade_data["direction"],
            "entry_price": trade_data["entry_price"],
            "amount": trade_data["amount"],
            "strategy": trade_data["strategy"],
        })

        return trade_id

    def update_position(
        self, trade_id: str, updates: dict[str, Any]
    ) -> None:
        """Update an existing position's state.

        Propagates changes to SQLite, Redis, and the in-memory cache.

        Args:
            trade_id: The trade identifier to update.
            updates: Key-value pairs to merge into the position state.
                Common updates include adjusted SL/TP prices,
                ``remaining_amount`` after partial closes, and
                order ID changes.
        """
        if trade_id not in self._positions:
            log_with_data(logger, "warning", "Attempted to update unknown position", {
                "trade_id": trade_id,
            })
            return

        position = self._positions[trade_id]
        position.update(updates)

        # Persist to SQLite (only fields that map to columns)
        sqlite_updates: dict[str, Any] = {}
        column_map = {
            "sl_price": "stop_loss",
            "entry_price": "entry_price",
            "amount": "quantity",
            "status": "status",
        }
        for key, col in column_map.items():
            if key in updates:
                sqlite_updates[col] = updates[key]

        # Pack remaining updates into metadata
        if updates:
            existing_meta = position.get("metadata", {})
            if isinstance(existing_meta, str):
                try:
                    existing_meta = json.loads(existing_meta)
                except (json.JSONDecodeError, TypeError):
                    existing_meta = {}
            for key, val in updates.items():
                if key not in column_map:
                    existing_meta[key] = val
            sqlite_updates["metadata"] = existing_meta

        if sqlite_updates:
            self._storage.update_trade(trade_id, sqlite_updates)

        # Update Redis
        symbol = position.get("symbol", "")
        if symbol:
            self._storage.set_position_state(symbol, position)

        log_with_data(logger, "info", "Position updated", {
            "trade_id": trade_id,
            "updated_fields": list(updates.keys()),
        })

    def close_position(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str,
    ) -> dict[str, Any]:
        """Close a position and compute final PnL.

        Calculates realised PnL in both USDT and R-multiple terms,
        updates all storage layers, and returns the trade summary.

        Args:
            trade_id: The trade identifier to close.
            exit_price: Price at which the position was exited.
            exit_reason: Reason for closing (e.g. ``"stop_loss"``,
                ``"take_profit"``, ``"manual"``, ``"trailing_stop"``).

        Returns:
            Dict with keys:

            - ``pnl_usdt`` (float): Absolute PnL in USDT.
            - ``pnl_pct`` (float): Percentage return on position.
            - ``r_multiple`` (float): PnL divided by initial risk.
            - ``hold_duration`` (str): ISO duration string.
            - ``exit_reason`` (str): The provided exit reason.
        """
        if trade_id not in self._positions:
            log_with_data(logger, "warning", "Attempted to close unknown position", {
                "trade_id": trade_id,
            })
            return {
                "pnl_usdt": 0.0,
                "pnl_pct": 0.0,
                "r_multiple": 0.0,
                "hold_duration": "PT0S",
                "exit_reason": exit_reason,
            }

        position = self._positions[trade_id]
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()

        entry_price: float = position.get("entry_price", 0.0)
        amount: float = position.get("remaining_amount", position.get("amount", 0.0))
        direction: str = position.get("direction", "long")
        sl_price: Optional[float] = position.get("sl_price")

        # Calculate PnL
        if direction == "long":
            pnl_per_unit = exit_price - entry_price
        else:
            pnl_per_unit = entry_price - exit_price

        leverage = position.get("leverage", 1)
        pnl_usdt = pnl_per_unit * amount * leverage
        pnl_pct = (pnl_per_unit / entry_price * 100.0) * leverage if entry_price > 0 else 0.0

        # R-multiple: PnL relative to initial risk
        r_multiple = 0.0
        if sl_price is not None and entry_price > 0:
            risk_per_unit = abs(entry_price - sl_price)
            if risk_per_unit > 0:
                r_multiple = pnl_per_unit / risk_per_unit

        # Hold duration
        entry_time_str = position.get("entry_time", now_iso)
        try:
            entry_time = datetime.fromisoformat(entry_time_str)
            if entry_time.tzinfo is None:
                entry_time = entry_time.replace(tzinfo=timezone.utc)
            hold_duration = now - entry_time
            hold_duration_str = str(hold_duration)
        except (ValueError, TypeError):
            hold_duration_str = "unknown"

        # Update SQLite
        self._storage.update_trade(trade_id, {
            "exit_price": exit_price,
            "exit_time": now_iso,
            "status": "closed",
            "pnl": pnl_usdt,
            "pnl_pct": pnl_pct,
            "r_multiple": r_multiple,
            "metadata": {
                **position.get("metadata", {}),
                "exit_reason": exit_reason,
                "hold_duration": hold_duration_str,
            } if isinstance(position.get("metadata"), dict) else {
                "exit_reason": exit_reason,
                "hold_duration": hold_duration_str,
            },
        })

        # Clear Redis position cache
        symbol = position.get("symbol", "")
        if symbol:
            self._storage.set_position_state(symbol, {
                "trade_id": trade_id,
                "symbol": symbol,
                "status": "closed",
            })

        # Update daily loss tracking
        if pnl_usdt < 0:
            current_daily_loss = self._storage.get_daily_loss()
            self._storage.set_daily_loss(current_daily_loss + abs(pnl_usdt))

        # Remove from in-memory cache
        del self._positions[trade_id]

        result: dict[str, Any] = {
            "pnl_usdt": round(pnl_usdt, 4),
            "pnl_pct": round(pnl_pct, 4),
            "r_multiple": round(r_multiple, 4),
            "hold_duration": hold_duration_str,
            "exit_reason": exit_reason,
        }

        log_with_data(logger, "info", "Position closed", {
            "trade_id": trade_id,
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            **result,
        })

        return result

    # ------------------------------------------------------------------
    # Position queries
    # ------------------------------------------------------------------

    def get_open_positions(self) -> list[dict[str, Any]]:
        """Return all currently open positions.

        Returns:
            List of position state dicts, each containing symbol,
            direction, entry_price, amount, and related metadata.
        """
        positions = list(self._positions.values())
        log_with_data(logger, "debug", "Retrieved open positions", {
            "count": len(positions),
        })
        return positions

    def get_position(self, symbol: str) -> Optional[dict[str, Any]]:
        """Return the current position for a symbol, if any.

        Searches the in-memory cache by symbol.  If multiple positions
        exist for the same symbol (not expected in normal operation),
        returns the most recently opened one.

        Args:
            symbol: Trading pair (e.g. ``"BTC/USDT"``).

        Returns:
            Position state dict or ``None`` if no position exists.
        """
        for position in self._positions.values():
            if position.get("symbol") == symbol:
                return position
        return None

    def has_position(self, symbol: str) -> bool:
        """Check if an open position exists for a symbol.

        Args:
            symbol: Trading pair.

        Returns:
            ``True`` if a position is open for the symbol.
        """
        for position in self._positions.values():
            if position.get("symbol") == symbol:
                return True
        return False

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_daily_stats(self) -> dict[str, Any]:
        """Return performance statistics for the current trading day.

        Queries SQLite for today's closed trades and computes
        win/loss counts, cumulative PnL, and consecutive loss
        streaks.

        Returns:
            Dict with keys:

            - ``trades_today`` (int): Number of trades closed today.
            - ``wins`` (int): Winning trades today.
            - ``losses`` (int): Losing trades today.
            - ``pnl_today`` (float): Cumulative PnL today (USDT).
            - ``consecutive_losses`` (int): Current losing streak.
            - ``last_loss_time`` (str | None): ISO timestamp of the
              most recent loss.
        """
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        trades = self._storage.get_trade_history(start=f"{today_str}T00:00:00")

        closed_today = [t for t in trades if t.get("status") == "closed"]

        wins = 0
        losses = 0
        pnl_today = 0.0
        consecutive_losses = 0
        last_loss_time: Optional[str] = None
        counting_streak = True

        # Sort by exit time descending for streak calculation
        sorted_trades = sorted(
            closed_today,
            key=lambda t: t.get("exit_time", ""),
            reverse=True,
        )

        for trade in sorted_trades:
            pnl = trade.get("pnl", 0.0) or 0.0
            pnl_today += pnl

            if pnl > 0:
                wins += 1
                counting_streak = False
            else:
                losses += 1
                if counting_streak:
                    consecutive_losses += 1
                if last_loss_time is None:
                    last_loss_time = trade.get("exit_time")

        stats: dict[str, Any] = {
            "trades_today": len(closed_today),
            "wins": wins,
            "losses": losses,
            "pnl_today": round(pnl_today, 4),
            "consecutive_losses": consecutive_losses,
            "last_loss_time": last_loss_time,
        }

        log_with_data(logger, "debug", "Daily stats computed", stats)
        return stats

    def get_equity_stats(
        self, current_balance: float
    ) -> dict[str, Any]:
        """Compute equity and drawdown statistics.

        Tracks the all-time equity peak and calculates the current
        drawdown percentage.  The peak is updated whenever the
        current balance exceeds the previous peak.

        Args:
            current_balance: Current account equity in USDT.

        Returns:
            Dict with keys:

            - ``current_equity`` (float): The provided balance.
            - ``peak_equity`` (float): All-time high equity.
            - ``drawdown_pct`` (float): Current drawdown as a
              percentage (0.0 if at peak).
            - ``drawdown_from_peak`` (float): Absolute USDT amount
              below the peak.
        """
        self._current_equity = current_balance

        if current_balance > self._peak_equity:
            self._peak_equity = current_balance

        drawdown_from_peak = self._peak_equity - current_balance
        drawdown_pct = (
            (drawdown_from_peak / self._peak_equity * 100.0)
            if self._peak_equity > 0
            else 0.0
        )

        stats: dict[str, Any] = {
            "current_equity": round(current_balance, 4),
            "peak_equity": round(self._peak_equity, 4),
            "drawdown_pct": round(drawdown_pct, 4),
            "drawdown_from_peak": round(drawdown_from_peak, 4),
        }

        log_with_data(logger, "debug", "Equity stats computed", stats)
        return stats

    # ------------------------------------------------------------------
    # Portfolio correlation
    # ------------------------------------------------------------------

    def compute_portfolio_correlation(
        self,
        symbols: list[str],
        price_data: dict[str, list[float]],
    ) -> dict[str, Any]:
        """Compute a correlation matrix between open position returns.

        Uses the last 30 days of daily closing prices to calculate
        pairwise Pearson correlations between the specified symbols.

        Args:
            symbols: List of trading pair identifiers to include.
            price_data: Dictionary mapping each symbol to a list of
                daily closing prices (most recent last).  Each list
                should contain at least 30 values for meaningful
                results.

        Returns:
            Dict with keys:

            - ``correlation_matrix`` (dict[str, dict[str, float]]):
              Nested dict of pairwise correlations.
            - ``avg_correlation`` (float): Mean of all off-diagonal
              correlations.
            - ``max_correlation`` (tuple): The most correlated pair
              and its value.
            - ``highly_correlated_pairs`` (list): Pairs with
              correlation above 0.7.
        """
        if len(symbols) < 2:
            return {
                "correlation_matrix": {},
                "avg_correlation": 0.0,
                "max_correlation": ([], 0.0),
                "highly_correlated_pairs": [],
            }

        # Compute daily returns for each symbol
        returns: dict[str, np.ndarray] = {}
        lookback = 30

        for sym in symbols:
            prices = price_data.get(sym, [])
            if len(prices) < 2:
                continue

            price_arr = np.array(prices[-lookback - 1:], dtype=np.float64)
            if len(price_arr) < 2:
                continue

            daily_returns = np.diff(price_arr) / price_arr[:-1]
            returns[sym] = daily_returns

        valid_symbols = [s for s in symbols if s in returns]
        if len(valid_symbols) < 2:
            return {
                "correlation_matrix": {},
                "avg_correlation": 0.0,
                "max_correlation": ([], 0.0),
                "highly_correlated_pairs": [],
            }

        # Align return series to the same length
        min_len = min(len(returns[s]) for s in valid_symbols)
        aligned: dict[str, np.ndarray] = {
            s: returns[s][-min_len:] for s in valid_symbols
        }

        # Build correlation matrix
        n = len(valid_symbols)
        return_matrix = np.column_stack([aligned[s] for s in valid_symbols])
        corr_matrix = np.corrcoef(return_matrix, rowvar=False)

        correlation_dict: dict[str, dict[str, float]] = {}
        for i, sym_i in enumerate(valid_symbols):
            correlation_dict[sym_i] = {}
            for j, sym_j in enumerate(valid_symbols):
                correlation_dict[sym_i][sym_j] = round(float(corr_matrix[i, j]), 4)

        # Compute aggregate statistics
        off_diagonal: list[float] = []
        max_corr_value = -1.0
        max_corr_pair: list[str] = []
        highly_correlated: list[dict[str, Any]] = []

        for i in range(n):
            for j in range(i + 1, n):
                corr_val = float(corr_matrix[i, j])
                off_diagonal.append(corr_val)

                if corr_val > max_corr_value:
                    max_corr_value = corr_val
                    max_corr_pair = [valid_symbols[i], valid_symbols[j]]

                if corr_val > 0.7:
                    highly_correlated.append({
                        "pair": [valid_symbols[i], valid_symbols[j]],
                        "correlation": round(corr_val, 4),
                    })

        avg_correlation = float(np.mean(off_diagonal)) if off_diagonal else 0.0

        result: dict[str, Any] = {
            "correlation_matrix": correlation_dict,
            "avg_correlation": round(avg_correlation, 4),
            "max_correlation": (max_corr_pair, round(max_corr_value, 4)),
            "highly_correlated_pairs": highly_correlated,
        }

        log_with_data(logger, "info", "Portfolio correlation computed", {
            "symbols": valid_symbols,
            "avg_correlation": result["avg_correlation"],
            "highly_correlated_count": len(highly_correlated),
        })

        return result
