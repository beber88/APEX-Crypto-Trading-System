"""Dynamic exit optimization for APEX.

Implements regime-based dynamic TP, time-based exits, re-entry after TP1,
and post-exit analysis.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from apex_crypto.core.logging import get_logger, log_with_data

logger = get_logger("signals.exit_optimizer")


# ------------------------------------------------------------------
# Regime-based TP override tables
# ------------------------------------------------------------------

_REGIME_TP_OVERRIDES: dict[str, dict[str, float]] = {
    "STRONG_BULL": {"tp1_r": 2.0, "tp2_r": 4.0, "tp3_trail_atr": 2.0},
    "RANGING": {"tp1_r": 1.0, "tp2_r": 1.5, "tp3_trail_atr": 1.0},
    "HIGH_VOL": {"tp1_r": 1.0, "tp2_r": 2.0, "tp3_trail_atr": 0.75},
    "CHAOS": {"tp1_r": 1.0, "tp2_r": 2.0, "tp3_trail_atr": 0.75},
}

_DEFAULT_TP: dict[str, float] = {
    "tp1_r": 1.5,
    "tp2_r": 2.5,
    "tp3_trail_atr": 1.5,
}


class ExitOptimizer:
    """Optimizes trade exit mechanics: regime-aware take-profit levels,
    time-based stops, re-entry logic after TP1, and post-exit analysis.

    Args:
        config: Dictionary containing ``exit`` configuration section
            from ``config.yaml``.
    """

    def __init__(self, config: dict) -> None:
        exit_cfg = config.get("exit", {})

        # --- Default TP config (overridden per-regime) ---
        self.default_tp1_r: float = exit_cfg.get("tp1_r", _DEFAULT_TP["tp1_r"])
        self.default_tp2_r: float = exit_cfg.get("tp2_r", _DEFAULT_TP["tp2_r"])
        self.default_tp3_trail_atr: float = exit_cfg.get(
            "tp3_trail_atr", _DEFAULT_TP["tp3_trail_atr"]
        )

        # --- Time-based stop ---
        self.time_stop_hours: int = exit_cfg.get("time_stop_hours", 48)
        self.time_stop_breakeven_buffer_pct: float = exit_cfg.get(
            "time_stop_breakeven_buffer_pct", 0.001
        )

        # --- Re-entry after TP1 ---
        self.reentry_enabled: bool = exit_cfg.get("reentry_enabled", True)
        self.reentry_pullback_r: float = exit_cfg.get("reentry_pullback_r", 0.3)
        self.reentry_min_score: int = exit_cfg.get("reentry_min_score", 50)
        self.reentry_size_pct: float = exit_cfg.get("reentry_size_pct", 0.35)

        # --- Regime TP overrides (allow config-level customisation) ---
        self.regime_tp_overrides: dict[str, dict[str, float]] = exit_cfg.get(
            "regime_tp_overrides", _REGIME_TP_OVERRIDES
        )

        # --- Internal tracking state ---
        self._trade_analysis: list[dict] = []

        log_with_data(logger, "info", "ExitOptimizer initialized", {
            "default_tp1_r": self.default_tp1_r,
            "default_tp2_r": self.default_tp2_r,
            "default_tp3_trail_atr": self.default_tp3_trail_atr,
            "time_stop_hours": self.time_stop_hours,
            "time_stop_breakeven_buffer_pct": self.time_stop_breakeven_buffer_pct,
            "reentry_enabled": self.reentry_enabled,
            "reentry_pullback_r": self.reentry_pullback_r,
            "reentry_min_score": self.reentry_min_score,
            "reentry_size_pct": self.reentry_size_pct,
            "regime_overrides_configured": list(self.regime_tp_overrides.keys()),
        })

    # ------------------------------------------------------------------
    # Dynamic take-profit levels
    # ------------------------------------------------------------------

    def get_dynamic_tp(
        self,
        regime: str,
        base_tp1_r: float,
        base_tp2_r: float,
        base_tp3_atr: float,
    ) -> dict:
        """Return regime-adjusted take-profit levels.

        If the current market regime has a known override entry the
        override values are used; otherwise the supplied base values
        (or instance defaults) are returned as-is.

        Args:
            regime: Market regime label (e.g. ``"STRONG_BULL"``).
            base_tp1_r: Caller-supplied TP1 in R-multiples.
            base_tp2_r: Caller-supplied TP2 in R-multiples.
            base_tp3_atr: Caller-supplied trailing TP3 in ATR multiples.

        Returns:
            ``{"tp1_r": X, "tp2_r": Y, "tp3_trail_atr": Z,
              "regime": str, "reason": str}``
        """
        override = self.regime_tp_overrides.get(regime)

        if override:
            result = {
                "tp1_r": override.get("tp1_r", base_tp1_r),
                "tp2_r": override.get("tp2_r", base_tp2_r),
                "tp3_trail_atr": override.get("tp3_trail_atr", base_tp3_atr),
                "regime": regime,
                "reason": (
                    f"Regime override for {regime}: TP levels adjusted "
                    f"(TP1={override.get('tp1_r', base_tp1_r)}R, "
                    f"TP2={override.get('tp2_r', base_tp2_r)}R, "
                    f"trail={override.get('tp3_trail_atr', base_tp3_atr)} ATR)"
                ),
            }
        else:
            result = {
                "tp1_r": base_tp1_r,
                "tp2_r": base_tp2_r,
                "tp3_trail_atr": base_tp3_atr,
                "regime": regime,
                "reason": (
                    f"No regime override for '{regime}': using base TP levels "
                    f"(TP1={base_tp1_r}R, TP2={base_tp2_r}R, "
                    f"trail={base_tp3_atr} ATR)"
                ),
            }

        log_with_data(logger, "info", "Dynamic TP levels computed", {
            "regime": regime,
            "tp1_r": result["tp1_r"],
            "tp2_r": result["tp2_r"],
            "tp3_trail_atr": result["tp3_trail_atr"],
            "override_applied": override is not None,
            "base_tp1_r": base_tp1_r,
            "base_tp2_r": base_tp2_r,
            "base_tp3_atr": base_tp3_atr,
        })
        return result

    # ------------------------------------------------------------------
    # Time-based stop
    # ------------------------------------------------------------------

    def check_time_stop(self, position: dict) -> dict:
        """Evaluate whether a position should be closed based on elapsed time.

        Rules:
        1. If position has been open longer than ``time_stop_hours`` **and**
           is not profitable, exit at breakeven + buffer.
        2. If position has been open longer than ``2 * time_stop_hours``
           (extended stop), exit regardless of P/L.

        Args:
            position: Dict with at least ``open_time`` (epoch float),
                ``entry_price`` (float), ``current_price`` (float), and
                ``direction`` (``"long"`` | ``"short"``).

        Returns:
            ``{"exit": True/False, ...}`` with reason when exit is triggered.
        """
        now = time.time()
        open_time = position.get("open_time", now)
        hours_open = (now - open_time) / 3600.0

        entry_price = position.get("entry_price", 0.0)
        current_price = position.get("current_price", 0.0)
        direction = position.get("direction", "long")

        # Determine profitability
        if direction == "long":
            pnl_pct = (current_price - entry_price) / entry_price if entry_price else 0.0
        else:
            pnl_pct = (entry_price - current_price) / entry_price if entry_price else 0.0

        is_profitable = pnl_pct > self.time_stop_breakeven_buffer_pct

        extended_hours = self.time_stop_hours * 2

        # Rule 2: extended time stop — close regardless
        if hours_open > extended_hours:
            result = {
                "exit": True,
                "reason": (
                    f"Extended time stop: position open {hours_open:.1f}h "
                    f"(limit {extended_hours}h) — closing regardless"
                ),
                "hours_open": round(hours_open, 2),
                "pnl_pct": round(pnl_pct, 6),
                "is_profitable": is_profitable,
            }
            log_with_data(logger, "warning", "Extended time stop triggered", {
                "hours_open": round(hours_open, 2),
                "extended_limit_hours": extended_hours,
                "pnl_pct": round(pnl_pct, 6),
                "direction": direction,
                "entry_price": entry_price,
                "current_price": current_price,
            })
            return result

        # Rule 1: normal time stop — only if not profitable
        if hours_open > self.time_stop_hours and not is_profitable:
            result = {
                "exit": True,
                "reason": (
                    f"Time stop: position unprofitable after "
                    f"{hours_open:.1f}h (limit {self.time_stop_hours}h, "
                    f"P/L {pnl_pct:.2%})"
                ),
                "hours_open": round(hours_open, 2),
                "pnl_pct": round(pnl_pct, 6),
                "is_profitable": is_profitable,
            }
            log_with_data(logger, "info", "Time stop triggered", {
                "hours_open": round(hours_open, 2),
                "time_stop_hours": self.time_stop_hours,
                "pnl_pct": round(pnl_pct, 6),
                "is_profitable": is_profitable,
                "direction": direction,
                "entry_price": entry_price,
                "current_price": current_price,
            })
            return result

        result: dict[str, Any] = {
            "exit": False,
            "hours_open": round(hours_open, 2),
            "pnl_pct": round(pnl_pct, 6),
            "is_profitable": is_profitable,
            "hours_until_time_stop": round(self.time_stop_hours - hours_open, 2),
        }
        log_with_data(logger, "debug", "Time stop not triggered", {
            "hours_open": round(hours_open, 2),
            "time_stop_hours": self.time_stop_hours,
            "pnl_pct": round(pnl_pct, 6),
            "is_profitable": is_profitable,
        })
        return result

    # ------------------------------------------------------------------
    # Re-entry after TP1
    # ------------------------------------------------------------------

    def check_reentry(
        self,
        position: dict,
        current_price: float,
        current_score: float,
    ) -> dict:
        """Determine whether to re-enter after TP1 was hit and price pulled back.

        Conditions (all must be true):
        - Re-entry is enabled.
        - TP1 was already hit on this position.
        - Price has pulled back by at least ``reentry_pullback_r`` * R from TP1.
        - Current signal score exceeds ``reentry_min_score``.

        Args:
            position: Dict with ``tp1_hit`` (bool), ``tp1_price`` (float),
                ``risk_r`` (float per 1R in price), and ``direction``.
            current_price: Latest market price.
            current_score: Current aggregated signal score.

        Returns:
            ``{"reenter": True/False, ...}`` with sizing when re-entry fires.
        """
        if not self.reentry_enabled:
            result = {"reenter": False, "reason": "Re-entry disabled"}
            log_with_data(logger, "debug", "Re-entry check skipped (disabled)", {})
            return result

        tp1_hit = position.get("tp1_hit", False)
        if not tp1_hit:
            result = {"reenter": False, "reason": "TP1 not yet hit"}
            log_with_data(logger, "debug", "Re-entry check: TP1 not hit", {
                "current_price": current_price,
                "current_score": current_score,
            })
            return result

        tp1_price = position.get("tp1_price", 0.0)
        risk_r = position.get("risk_r", 0.0)
        direction = position.get("direction", "long")

        if risk_r <= 0:
            result = {"reenter": False, "reason": "Invalid risk_r value"}
            log_with_data(logger, "warning", "Re-entry check: invalid risk_r", {
                "risk_r": risk_r,
            })
            return result

        # Calculate pullback distance in R-multiples from TP1
        pullback_threshold = self.reentry_pullback_r * risk_r

        if direction == "long":
            pullback_distance = tp1_price - current_price
        else:
            pullback_distance = current_price - tp1_price

        has_pulled_back = pullback_distance >= pullback_threshold
        score_sufficient = current_score >= self.reentry_min_score

        if has_pulled_back and score_sufficient:
            result = {
                "reenter": True,
                "size_pct": self.reentry_size_pct,
                "reason": (
                    f"Re-entry after TP1 pullback: "
                    f"{pullback_distance / risk_r:.2f}R pullback from TP1, "
                    f"score {current_score:.0f} >= {self.reentry_min_score}"
                ),
                "pullback_r": round(pullback_distance / risk_r, 4),
                "current_price": current_price,
                "tp1_price": tp1_price,
                "current_score": current_score,
            }
            log_with_data(logger, "info", "Re-entry triggered after TP1", {
                "direction": direction,
                "tp1_price": tp1_price,
                "current_price": current_price,
                "pullback_r": round(pullback_distance / risk_r, 4),
                "pullback_threshold_r": self.reentry_pullback_r,
                "current_score": current_score,
                "reentry_size_pct": self.reentry_size_pct,
            })
            return result

        reasons = []
        if not has_pulled_back:
            reasons.append(
                f"pullback {pullback_distance / risk_r:.2f}R < "
                f"required {self.reentry_pullback_r}R"
            )
        if not score_sufficient:
            reasons.append(
                f"score {current_score:.0f} < required {self.reentry_min_score}"
            )

        result = {
            "reenter": False,
            "reason": f"Re-entry conditions not met: {'; '.join(reasons)}",
            "pullback_r": round(pullback_distance / risk_r, 4) if risk_r else 0.0,
            "current_score": current_score,
        }
        log_with_data(logger, "debug", "Re-entry conditions not met", {
            "direction": direction,
            "tp1_price": tp1_price,
            "current_price": current_price,
            "pullback_r": round(pullback_distance / risk_r, 4) if risk_r else 0.0,
            "pullback_threshold_r": self.reentry_pullback_r,
            "current_score": current_score,
            "min_score_required": self.reentry_min_score,
        })
        return result

    # ------------------------------------------------------------------
    # Post-exit analysis
    # ------------------------------------------------------------------

    def analyze_post_exit(
        self,
        trade: dict,
        prices_after_exit: list[float],
    ) -> dict:
        """Analyse price action after a trade was closed to quantify
        whether profit was left on the table.

        Args:
            trade: Dict with ``exit_price``, ``entry_price``, ``risk_r``
                (price per 1R), and ``direction``.
            prices_after_exit: Chronological list of prices observed
                after the exit.

        Returns:
            ``{"continuation_r": float, "max_favorable": float,
              "max_adverse": float, "left_on_table": bool}``
        """
        exit_price = trade.get("exit_price", 0.0)
        entry_price = trade.get("entry_price", 0.0)
        risk_r = trade.get("risk_r", 0.0)
        direction = trade.get("direction", "long")

        if not prices_after_exit or risk_r <= 0:
            result = {
                "continuation_r": 0.0,
                "max_favorable": 0.0,
                "max_adverse": 0.0,
                "left_on_table": False,
                "reason": "Insufficient data for post-exit analysis",
            }
            log_with_data(logger, "debug", "Post-exit analysis skipped", {
                "prices_count": len(prices_after_exit),
                "risk_r": risk_r,
            })
            return result

        if direction == "long":
            max_price = max(prices_after_exit)
            min_price = min(prices_after_exit)
            final_price = prices_after_exit[-1]

            continuation = (final_price - exit_price) / risk_r
            max_favorable = (max_price - exit_price) / risk_r
            max_adverse = (exit_price - min_price) / risk_r
        else:
            max_price = max(prices_after_exit)
            min_price = min(prices_after_exit)
            final_price = prices_after_exit[-1]

            continuation = (exit_price - final_price) / risk_r
            max_favorable = (exit_price - min_price) / risk_r
            max_adverse = (max_price - exit_price) / risk_r

        left_on_table = max_favorable > 1.0

        result = {
            "continuation_r": round(continuation, 4),
            "max_favorable": round(max_favorable, 4),
            "max_adverse": round(max_adverse, 4),
            "left_on_table": left_on_table,
            "exit_price": exit_price,
            "final_price_after": final_price,
            "direction": direction,
        }

        self._trade_analysis.append(result)

        log_with_data(logger, "info", "Post-exit analysis completed", {
            "direction": direction,
            "exit_price": exit_price,
            "entry_price": entry_price,
            "continuation_r": round(continuation, 4),
            "max_favorable_r": round(max_favorable, 4),
            "max_adverse_r": round(max_adverse, 4),
            "left_on_table": left_on_table,
            "prices_analysed": len(prices_after_exit),
            "total_analyses": len(self._trade_analysis),
        })
        return result

    # ------------------------------------------------------------------
    # Aggregate exit analysis summary
    # ------------------------------------------------------------------

    def get_exit_analysis_summary(self) -> dict:
        """Aggregate all post-exit analyses into a summary report.

        Returns:
            Dictionary with average continuation, percentage of trades
            where money was left on the table, and recommended TP
            adjustments.
        """
        if not self._trade_analysis:
            result = {
                "total_analyses": 0,
                "avg_continuation_r": 0.0,
                "avg_max_favorable_r": 0.0,
                "avg_max_adverse_r": 0.0,
                "pct_left_on_table": 0.0,
                "recommended_adjustments": [],
                "reason": "No post-exit analyses recorded yet",
            }
            log_with_data(logger, "debug", "Exit analysis summary (empty)", {})
            return result

        total = len(self._trade_analysis)
        avg_continuation = sum(
            a["continuation_r"] for a in self._trade_analysis
        ) / total
        avg_max_favorable = sum(
            a["max_favorable"] for a in self._trade_analysis
        ) / total
        avg_max_adverse = sum(
            a["max_adverse"] for a in self._trade_analysis
        ) / total
        left_count = sum(1 for a in self._trade_analysis if a["left_on_table"])
        pct_left = left_count / total

        # Build recommendations based on patterns
        recommendations: list[str] = []

        if pct_left > 0.50:
            recommendations.append(
                f"Over {pct_left:.0%} of trades left >1R on the table — "
                "consider widening TP2/TP3 targets or using a looser trail"
            )
        if avg_max_favorable > 2.0 and avg_continuation > 0.5:
            recommendations.append(
                f"Avg max favorable excursion is {avg_max_favorable:.1f}R — "
                "price continues significantly after exit, consider adding "
                "a TP3 trailing component"
            )
        if avg_max_adverse > 0.5:
            recommendations.append(
                f"Avg max adverse excursion post-exit is {avg_max_adverse:.1f}R — "
                "exits are well-timed relative to drawdowns"
            )
        if avg_continuation < -0.5:
            recommendations.append(
                f"Avg post-exit continuation is {avg_continuation:.2f}R — "
                "price reverses after exit, current exit timing is good"
            )

        if not recommendations:
            recommendations.append("No significant adjustments recommended at this time")

        result = {
            "total_analyses": total,
            "avg_continuation_r": round(avg_continuation, 4),
            "avg_max_favorable_r": round(avg_max_favorable, 4),
            "avg_max_adverse_r": round(avg_max_adverse, 4),
            "pct_left_on_table": round(pct_left, 4),
            "trades_left_on_table": left_count,
            "recommended_adjustments": recommendations,
        }

        log_with_data(logger, "info", "Exit analysis summary generated", {
            "total_analyses": total,
            "avg_continuation_r": round(avg_continuation, 4),
            "avg_max_favorable_r": round(avg_max_favorable, 4),
            "pct_left_on_table": round(pct_left, 4),
            "recommendation_count": len(recommendations),
        })
        return result
