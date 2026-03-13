"""Dynamic entry optimization for APEX.

Selects order type, implements partial entries, tracks fill rates,
and adjusts entry based on volatility and sentiment.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from apex_crypto.core.logging import get_logger, log_with_data

logger = get_logger("signals.entry_optimizer")


class EntryOptimizer:
    """Optimizes trade entry mechanics: order type selection, partial entries,
    fill-rate tracking, and sentiment-based threshold overrides.

    Args:
        config: Dictionary containing ``signals`` and ``entry`` configuration
            sections from ``config.yaml``.
    """

    def __init__(self, config: dict) -> None:
        entry_cfg = config.get("entry", {})

        # --- Fill-rate tracking ---
        self.fill_rate_window: int = entry_cfg.get("fill_rate_window", 100)
        self.fill_rate_threshold: float = entry_cfg.get("fill_rate_threshold", 0.60)

        # --- Partial entry settings ---
        self.partial_entry_enabled: bool = entry_cfg.get("partial_entry_enabled", True)
        self.partial_entry_first_pct: float = entry_cfg.get(
            "partial_entry_first_pct", 0.50
        )
        self.partial_entry_pullback_pct: float = entry_cfg.get(
            "partial_entry_pullback_pct", 0.003
        )
        self.partial_entry_timeout_seconds: int = entry_cfg.get(
            "partial_entry_timeout_seconds", 300
        )

        # --- Volatility percentile thresholds ---
        self.high_vol_atr_percentile: float = entry_cfg.get(
            "high_vol_atr_percentile", 70
        )
        self.low_vol_atr_percentile: float = entry_cfg.get(
            "low_vol_atr_percentile", 30
        )

        # --- Sentiment override ---
        self.sentiment_override_threshold: float = entry_cfg.get(
            "sentiment_override_threshold", 0.8
        )
        self.sentiment_score_reduction: int = entry_cfg.get(
            "sentiment_score_reduction", 5
        )

        # --- Internal tracking state ---
        self._limit_orders_placed: int = 0
        self._limit_orders_filled: int = 0
        self._fill_history: list[dict] = []

        log_with_data(logger, "info", "EntryOptimizer initialized", {
            "fill_rate_window": self.fill_rate_window,
            "fill_rate_threshold": self.fill_rate_threshold,
            "partial_entry_enabled": self.partial_entry_enabled,
            "partial_entry_first_pct": self.partial_entry_first_pct,
            "partial_entry_pullback_pct": self.partial_entry_pullback_pct,
            "partial_entry_timeout_seconds": self.partial_entry_timeout_seconds,
            "high_vol_atr_percentile": self.high_vol_atr_percentile,
            "low_vol_atr_percentile": self.low_vol_atr_percentile,
            "sentiment_override_threshold": self.sentiment_override_threshold,
            "sentiment_score_reduction": self.sentiment_score_reduction,
        })

    # ------------------------------------------------------------------
    # Order type selection
    # ------------------------------------------------------------------

    def select_order_type(
        self,
        atr_percentile: float,
        signal_score: float,
        is_breakout: bool,
    ) -> dict:
        """Choose between MARKET and LIMIT order based on conditions.

        Priority logic:
        1. Breakout signals always use MARKET (speed matters).
        2. If recent limit fill-rate < threshold, fall back to MARKET.
        3. High-volatility regimes use LIMIT with wider offset.
        4. Low-volatility regimes use MARKET (price improvement minimal).
        5. Default: LIMIT with narrow offset.

        Returns:
            ``{"order_type": "market"|"limit", "offset_pct": float, "reason": str}``
        """
        fill_rate = self.get_fill_rate()

        # 1. Breakout — always market
        if is_breakout:
            result = {
                "order_type": "market",
                "offset_pct": 0.0,
                "reason": "Breakout strategy: market order for immediate fill",
            }
            log_with_data(logger, "info", "Order type selected", {
                "order_type": result["order_type"],
                "reason": result["reason"],
                "atr_percentile": atr_percentile,
                "signal_score": signal_score,
                "is_breakout": is_breakout,
                "fill_rate": round(fill_rate, 4),
            })
            return result

        # 2. Poor fill rate — switch to market
        if fill_rate < self.fill_rate_threshold and self._limit_orders_placed > 0:
            result = {
                "order_type": "market",
                "offset_pct": 0.0,
                "reason": (
                    f"Fill rate {fill_rate:.1%} below threshold "
                    f"{self.fill_rate_threshold:.1%}: switching to market"
                ),
            }
            log_with_data(logger, "info", "Order type selected", {
                "order_type": result["order_type"],
                "reason": result["reason"],
                "fill_rate": round(fill_rate, 4),
                "limit_orders_placed": self._limit_orders_placed,
                "limit_orders_filled": self._limit_orders_filled,
            })
            return result

        # 3. High volatility — limit with wider offset
        if atr_percentile > self.high_vol_atr_percentile:
            offset = 0.003 + (atr_percentile - self.high_vol_atr_percentile) / 100 * 0.002
            offset = round(min(offset, 0.005), 4)
            result = {
                "order_type": "limit",
                "offset_pct": offset,
                "reason": (
                    f"High volatility (ATR pctl {atr_percentile:.0f}): "
                    f"limit with {offset:.2%} offset"
                ),
            }
            log_with_data(logger, "info", "Order type selected", {
                "order_type": result["order_type"],
                "offset_pct": offset,
                "atr_percentile": atr_percentile,
            })
            return result

        # 4. Low volatility — market (price improvement minimal)
        if atr_percentile < self.low_vol_atr_percentile:
            result = {
                "order_type": "market",
                "offset_pct": 0.0,
                "reason": (
                    f"Low volatility (ATR pctl {atr_percentile:.0f}): "
                    "market order, price improvement minimal"
                ),
            }
            log_with_data(logger, "info", "Order type selected", {
                "order_type": result["order_type"],
                "atr_percentile": atr_percentile,
            })
            return result

        # 5. Default — limit with narrow offset
        offset = 0.001 + (atr_percentile - self.low_vol_atr_percentile) / 100 * 0.001
        offset = round(min(offset, 0.002), 4)
        result = {
            "order_type": "limit",
            "offset_pct": offset,
            "reason": (
                f"Normal conditions (ATR pctl {atr_percentile:.0f}): "
                f"limit with {offset:.2%} offset"
            ),
        }
        log_with_data(logger, "info", "Order type selected", {
            "order_type": result["order_type"],
            "offset_pct": offset,
            "atr_percentile": atr_percentile,
            "signal_score": signal_score,
        })
        return result

    # ------------------------------------------------------------------
    # Partial entry calculation
    # ------------------------------------------------------------------

    def calculate_partial_entry(
        self,
        signal_score: float,
        current_price: float,
        direction: str,
    ) -> dict:
        """Determine whether to enter in a single order or split into parts.

        High-conviction signals (score >= 75) enter fully at market.
        Lower-conviction signals split: 50% immediately, 50% on pullback.

        Args:
            signal_score: Aggregated signal score (0-100).
            current_price: Current market price for the asset.
            direction: ``"long"`` or ``"short"``.

        Returns:
            Dictionary with ``entries`` list describing each tranche.
        """
        if not self.partial_entry_enabled or signal_score >= 75:
            result = {
                "partial": False,
                "entries": [
                    {
                        "pct": 1.0,
                        "type": "market",
                        "price": current_price,
                    }
                ],
            }
            log_with_data(logger, "info", "Full entry calculated", {
                "signal_score": signal_score,
                "current_price": current_price,
                "direction": direction,
                "reason": (
                    "High conviction signal"
                    if signal_score >= 75
                    else "Partial entry disabled"
                ),
            })
            return result

        # Calculate pullback price depending on direction
        if direction == "long":
            pullback_price = round(
                current_price * (1 - self.partial_entry_pullback_pct), 8
            )
        else:
            pullback_price = round(
                current_price * (1 + self.partial_entry_pullback_pct), 8
            )

        first_pct = self.partial_entry_first_pct
        second_pct = round(1.0 - first_pct, 4)

        result = {
            "partial": True,
            "entries": [
                {
                    "pct": first_pct,
                    "type": "market",
                    "price": current_price,
                },
                {
                    "pct": second_pct,
                    "type": "limit",
                    "price": pullback_price,
                    "timeout": self.partial_entry_timeout_seconds,
                },
            ],
        }

        log_with_data(logger, "info", "Partial entry calculated", {
            "signal_score": signal_score,
            "current_price": current_price,
            "direction": direction,
            "first_pct": first_pct,
            "pullback_price": pullback_price,
            "pullback_offset_pct": self.partial_entry_pullback_pct,
            "timeout_seconds": self.partial_entry_timeout_seconds,
        })
        return result

    # ------------------------------------------------------------------
    # Sentiment override
    # ------------------------------------------------------------------

    def check_sentiment_override(
        self,
        signal_score: float,
        sentiment_score: float,
        min_score_threshold: int,
    ) -> dict:
        """Check whether strong sentiment justifies lowering the entry threshold.

        If sentiment exceeds the override threshold and the signal score is
        within ``sentiment_score_reduction`` points of ``min_score_threshold``,
        the effective threshold is lowered so the trade can proceed.

        Returns:
            ``{"override": True/False, ...}`` with adjusted threshold when
            override is active.
        """
        adjusted_threshold = min_score_threshold - self.sentiment_score_reduction

        if (
            sentiment_score > self.sentiment_override_threshold
            and signal_score > adjusted_threshold
            and signal_score <= min_score_threshold
        ):
            result = {
                "override": True,
                "adjusted_threshold": adjusted_threshold,
                "original_threshold": min_score_threshold,
                "signal_score": signal_score,
                "sentiment_score": sentiment_score,
                "reason": (
                    f"Strong sentiment ({sentiment_score:.2f}) override: "
                    f"threshold lowered from {min_score_threshold} to "
                    f"{adjusted_threshold}"
                ),
            }
            log_with_data(logger, "info", "Sentiment override triggered", {
                "signal_score": signal_score,
                "sentiment_score": sentiment_score,
                "original_threshold": min_score_threshold,
                "adjusted_threshold": adjusted_threshold,
            })
            return result

        result = {
            "override": False,
            "signal_score": signal_score,
            "sentiment_score": sentiment_score,
            "min_score_threshold": min_score_threshold,
        }
        log_with_data(logger, "debug", "Sentiment override not triggered", {
            "signal_score": signal_score,
            "sentiment_score": sentiment_score,
            "min_score_threshold": min_score_threshold,
            "needed_for_override": (
                f"sentiment > {self.sentiment_override_threshold} AND "
                f"score > {adjusted_threshold}"
            ),
        })
        return result

    # ------------------------------------------------------------------
    # Fill tracking
    # ------------------------------------------------------------------

    def record_fill(self, order_type: str, filled: bool) -> None:
        """Record whether a limit order was filled or missed.

        Args:
            order_type: ``"market"`` or ``"limit"``.
            filled: Whether the order was filled.
        """
        record = {
            "timestamp": time.time(),
            "order_type": order_type,
            "filled": filled,
        }
        self._fill_history.append(record)

        if order_type == "limit":
            self._limit_orders_placed += 1
            if filled:
                self._limit_orders_filled += 1

        # Trim history to window size
        if len(self._fill_history) > self.fill_rate_window * 2:
            self._fill_history = self._fill_history[-self.fill_rate_window:]

        log_with_data(logger, "debug", "Fill recorded", {
            "order_type": order_type,
            "filled": filled,
            "running_fill_rate": round(self.get_fill_rate(), 4),
            "history_size": len(self._fill_history),
        })

    def get_fill_rate(self) -> float:
        """Return the fill rate for limit orders within the recent window.

        Returns:
            Float between 0.0 and 1.0.  Returns 1.0 when no limit orders
            have been placed (optimistic default).
        """
        recent = [
            r for r in self._fill_history[-self.fill_rate_window:]
            if r["order_type"] == "limit"
        ]
        if not recent:
            return 1.0
        filled_count = sum(1 for r in recent if r["filled"])
        return filled_count / len(recent)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return summary statistics for the dashboard.

        Returns:
            Dictionary with fill-rate stats, partial entry config, and
            recent order breakdown.
        """
        recent = self._fill_history[-self.fill_rate_window:]
        limit_orders = [r for r in recent if r["order_type"] == "limit"]
        market_orders = [r for r in recent if r["order_type"] == "market"]

        stats = {
            "fill_rate": round(self.get_fill_rate(), 4),
            "total_limit_orders_placed": self._limit_orders_placed,
            "total_limit_orders_filled": self._limit_orders_filled,
            "recent_limit_count": len(limit_orders),
            "recent_market_count": len(market_orders),
            "recent_limit_filled": sum(1 for r in limit_orders if r["filled"]),
            "partial_entry_enabled": self.partial_entry_enabled,
            "partial_entry_first_pct": self.partial_entry_first_pct,
            "partial_entry_pullback_pct": self.partial_entry_pullback_pct,
            "fill_rate_threshold": self.fill_rate_threshold,
            "history_size": len(self._fill_history),
        }

        log_with_data(logger, "debug", "Entry optimizer stats requested", stats)
        return stats
