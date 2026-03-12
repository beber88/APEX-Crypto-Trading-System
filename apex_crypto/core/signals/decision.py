"""Trade decision engine for the APEX Crypto Trading System.

Evaluates aggregated signals against risk rules, position limits, and
cooldown logic to produce actionable trade decisions.  Also monitors
open positions for exit conditions.
"""

from __future__ import annotations

import time
from typing import Any

from apex_crypto.core.logging import get_logger, log_with_data
from apex_crypto.core.strategies.base import SignalDirection, TradeSignal

logger = get_logger("signals.decision")


class TradeDecisionEngine:
    """Decides whether to enter, exit, or skip trades based on aggregated
    signal scores, active risk constraints, and portfolio state.

    Args:
        config: Dictionary containing both the ``signals`` and ``risk``
            sections from ``config.yaml``.
    """

    def __init__(self, config: dict) -> None:
        signals_cfg = config.get("signals", {})
        risk_cfg = config.get("risk", {})

        # --- Signal thresholds ---
        self._full_position_score: int = signals_cfg.get("full_position_score", 75)
        self._half_position_score: int = signals_cfg.get("half_position_score", 60)
        self._min_agreeing: int = signals_cfg.get("min_agreeing_strategies", 2)

        # --- Risk limits ---
        self._max_trades_per_day: int = risk_cfg.get("max_trades_per_day", 25)
        self._max_open_positions: int = risk_cfg.get("max_open_positions", 8)
        self._daily_loss_limit_pct: float = risk_cfg.get("daily_loss_limit_pct", 3.0)
        self._max_drawdown_pct: float = risk_cfg.get("max_drawdown_pct", 12.0)
        self._consecutive_loss_threshold: int = risk_cfg.get("consecutive_loss_threshold", 3)
        self._cooldown_minutes: int = risk_cfg.get("consecutive_loss_cooldown_minutes", 120)
        self._cooldown_elevated_score: int = risk_cfg.get("cooldown_elevated_score", 80)

        # --- Position sizing ---
        self._risk_per_trade_pct: float = risk_cfg.get("risk_per_trade_pct", 1.0)
        self._max_position_pct: float = risk_cfg.get("max_position_pct", 5.0)

        # --- Take-profit percentages ---
        self._tp1_close_pct: float = risk_cfg.get("tp1_close_pct", 0.35)
        self._tp2_close_pct: float = risk_cfg.get("tp2_close_pct", 0.35)

        # --- Swing trade limits ---
        self._swing_max_hold_days: int = config.get("strategies", {}).get(
            "swing", {}
        ).get("hold_days_max", 10)

        log_with_data(logger, "info", "TradeDecisionEngine initialized", {
            "full_position_score": self._full_position_score,
            "half_position_score": self._half_position_score,
            "max_trades_per_day": self._max_trades_per_day,
            "max_open_positions": self._max_open_positions,
            "daily_loss_limit_pct": self._daily_loss_limit_pct,
        })

    # ------------------------------------------------------------------
    # Entry evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        aggregated: dict[str, Any],
        current_positions: list[dict[str, Any]],
        daily_stats: dict[str, Any],
    ) -> dict[str, Any]:
        """Decide whether to enter a trade for the given aggregated signal.

        Runs a series of checks in order.  The first failing check results
        in a ``"skip"`` action.  If all checks pass, the action is either
        ``"enter_full"`` or ``"enter_half"`` depending on the score.

        Args:
            aggregated: Output of ``SignalAggregator.aggregate_signals``
                (optionally with bonuses applied).
            current_positions: List of dicts describing currently open
                positions.  Each dict must contain at least ``"symbol"``
                and ``"direction"`` keys.
            daily_stats: Dictionary with keys:

                * ``"trades_today"`` (int) — number of trades executed today.
                * ``"daily_pnl_pct"`` (float) — unrealised + realised P&L
                  as a percentage of starting equity.
                * ``"consecutive_losses"`` (int) — current streak of
                  consecutive losing trades.
                * ``"last_loss_ts"`` (float | None) — timestamp of the most
                  recent loss, or ``None``.

        Returns:
            Decision dict::

                {
                    "action": "enter_full" | "enter_half" | "skip",
                    "reason": str,
                    "symbol": str,
                    "direction": str,
                    "score": float,
                    "position_size_pct": float,
                    "checks_passed": [str, ...],
                    "checks_failed": [str, ...],
                }
        """
        symbol = aggregated.get("symbol", "UNKNOWN")
        score = abs(aggregated.get("weighted_score", 0))
        direction = aggregated.get("direction", SignalDirection.NEUTRAL.value)
        has_conflict = aggregated.get("has_conflict", False)
        num_agreeing = aggregated.get("num_agreeing", 0)

        checks_passed: list[str] = []
        checks_failed: list[str] = []

        # ----- Check (a): Score threshold -----------------------------------
        if score >= self._full_position_score:
            action = "enter_full"
            position_size_pct = self._max_position_pct
            checks_passed.append(
                f"score_full_threshold (score={score} >= {self._full_position_score})"
            )
        elif score >= self._half_position_score:
            action = "enter_half"
            position_size_pct = self._max_position_pct / 2.0
            checks_passed.append(
                f"score_half_threshold (score={score} >= {self._half_position_score})"
            )
        else:
            reason = (
                f"Score {score} below minimum threshold "
                f"{self._half_position_score}"
            )
            checks_failed.append(f"score_threshold (score={score})")
            return self._skip(symbol, direction, score, reason,
                              checks_passed, checks_failed)

        # ----- Check (b): Conflict ------------------------------------------
        if has_conflict:
            reason = "Strategy conflict detected (disagreement exceeds threshold)"
            checks_failed.append("conflict_check")
            return self._skip(symbol, direction, score, reason,
                              checks_passed, checks_failed)
        checks_passed.append("conflict_check")

        # ----- Check (c): Minimum agreeing strategies -----------------------
        if num_agreeing < self._min_agreeing:
            reason = (
                f"Only {num_agreeing} strategies agree; "
                f"minimum is {self._min_agreeing}"
            )
            checks_failed.append(
                f"min_agreeing (need={self._min_agreeing}, have={num_agreeing})"
            )
            return self._skip(symbol, direction, score, reason,
                              checks_passed, checks_failed)
        checks_passed.append(
            f"min_agreeing ({num_agreeing} >= {self._min_agreeing})"
        )

        # ----- Check (d): Daily limits --------------------------------------
        trades_today = daily_stats.get("trades_today", 0)
        if trades_today >= self._max_trades_per_day:
            reason = (
                f"Daily trade limit reached ({trades_today}/"
                f"{self._max_trades_per_day})"
            )
            checks_failed.append("max_trades_per_day")
            return self._skip(symbol, direction, score, reason,
                              checks_passed, checks_failed)
        checks_passed.append(
            f"max_trades_per_day ({trades_today}/{self._max_trades_per_day})"
        )

        if len(current_positions) >= self._max_open_positions:
            reason = (
                f"Max open positions reached ({len(current_positions)}/"
                f"{self._max_open_positions})"
            )
            checks_failed.append("max_open_positions")
            return self._skip(symbol, direction, score, reason,
                              checks_passed, checks_failed)
        checks_passed.append(
            f"max_open_positions ({len(current_positions)}/{self._max_open_positions})"
        )

        daily_pnl_pct = daily_stats.get("daily_pnl_pct", 0.0)
        if daily_pnl_pct <= -self._daily_loss_limit_pct:
            reason = (
                f"Daily loss limit breached (PnL={daily_pnl_pct:.2f}%, "
                f"limit=-{self._daily_loss_limit_pct}%)"
            )
            checks_failed.append("daily_loss_limit")
            return self._skip(symbol, direction, score, reason,
                              checks_passed, checks_failed)
        checks_passed.append(
            f"daily_loss_limit (pnl={daily_pnl_pct:.2f}%)"
        )

        # ----- Check (e): Consecutive-loss cooldown -------------------------
        consecutive_losses = daily_stats.get("consecutive_losses", 0)
        if consecutive_losses >= self._consecutive_loss_threshold:
            last_loss_ts = daily_stats.get("last_loss_ts")
            now = time.time()
            elapsed_minutes = (
                (now - last_loss_ts) / 60.0 if last_loss_ts else float("inf")
            )
            cooldown_ok = elapsed_minutes >= self._cooldown_minutes
            score_ok = score >= self._cooldown_elevated_score

            if not (cooldown_ok and score_ok):
                reasons: list[str] = []
                if not cooldown_ok:
                    remaining = self._cooldown_minutes - elapsed_minutes
                    reasons.append(
                        f"cooldown not elapsed ({elapsed_minutes:.0f}/"
                        f"{self._cooldown_minutes} min)"
                    )
                if not score_ok:
                    reasons.append(
                        f"elevated score not met (score={score}, "
                        f"need={self._cooldown_elevated_score})"
                    )
                reason = (
                    f"Consecutive-loss cooldown active after "
                    f"{consecutive_losses} losses: {'; '.join(reasons)}"
                )
                checks_failed.append("consecutive_loss_cooldown")
                return self._skip(symbol, direction, score, reason,
                                  checks_passed, checks_failed)

            checks_passed.append(
                f"consecutive_loss_cooldown (losses={consecutive_losses}, "
                f"elapsed={elapsed_minutes:.0f}min, score={score})"
            )
        else:
            checks_passed.append(
                f"consecutive_loss_cooldown (losses={consecutive_losses}, "
                f"below threshold)"
            )

        # ----- Check (f): Not already in position for this symbol -----------
        position_symbols = {p.get("symbol") for p in current_positions}
        if symbol in position_symbols:
            reason = f"Already in a position for {symbol}"
            checks_failed.append("duplicate_symbol")
            return self._skip(symbol, direction, score, reason,
                              checks_passed, checks_failed)
        checks_passed.append("no_duplicate_position")

        # ----- All checks passed --------------------------------------------
        result: dict[str, Any] = {
            "action": action,
            "reason": f"All checks passed — {action}",
            "symbol": symbol,
            "direction": direction,
            "score": score,
            "position_size_pct": position_size_pct,
            "checks_passed": checks_passed,
            "checks_failed": checks_failed,
        }

        log_with_data(logger, "info", "Trade decision: ENTER", {
            "symbol": symbol,
            "action": action,
            "direction": direction,
            "score": score,
            "position_size_pct": position_size_pct,
            "checks_passed_count": len(checks_passed),
        })

        return result

    # ------------------------------------------------------------------
    # Exit evaluation
    # ------------------------------------------------------------------

    def check_exit_conditions(
        self,
        position: dict[str, Any],
        indicators: dict[str, Any],
        signals: list[TradeSignal],
    ) -> dict[str, Any]:
        """Check whether an open position should be closed (partially or fully).

        Exit checks are evaluated in priority order.  The first triggered
        condition determines the action.

        Args:
            position: Dictionary describing the open position with keys:

                * ``"symbol"`` (str)
                * ``"direction"`` (str) — ``"long"`` or ``"short"``.
                * ``"entry_price"`` (float)
                * ``"current_price"`` (float)
                * ``"stop_loss"`` (float)
                * ``"take_profit_1"`` (float)
                * ``"take_profit_2"`` (float)
                * ``"take_profit_3"`` (float)
                * ``"tp1_hit"`` (bool) — whether TP1 has already been taken.
                * ``"tp2_hit"`` (bool) — whether TP2 has already been taken.
                * ``"open_timestamp"`` (float) — epoch seconds.
                * ``"strategy"`` (str) — originating strategy name.
                * ``"regime_at_entry"`` (str)

            indicators: Current indicator values.  Expected keys:

                * ``"current_regime"`` (str) — latest regime classification.
                * ``"trailing_stop"`` (float | None) — dynamically
                  computed trailing stop price, if active.

            signals: Latest ``TradeSignal`` objects for this symbol
                from the current evaluation cycle.

        Returns:
            Exit decision dict::

                {
                    "action": "hold" | "close_partial" | "close_full",
                    "reason": str,
                    "close_pct": float,
                    "symbol": str,
                    "direction": str,
                    "current_price": float,
                }
        """
        symbol = position.get("symbol", "UNKNOWN")
        direction = position.get("direction", "neutral")
        entry_price = position.get("entry_price", 0.0)
        current_price = position.get("current_price", 0.0)
        stop_loss = position.get("stop_loss", 0.0)
        tp1 = position.get("take_profit_1", 0.0)
        tp2 = position.get("take_profit_2", 0.0)
        tp1_hit = position.get("tp1_hit", False)
        tp2_hit = position.get("tp2_hit", False)

        base_result: dict[str, Any] = {
            "symbol": symbol,
            "direction": direction,
            "current_price": current_price,
        }

        # ----- (a) Stop loss hit --------------------------------------------
        if self._stop_loss_hit(direction, current_price, stop_loss):
            result = {
                **base_result,
                "action": "close_full",
                "reason": (
                    f"Stop loss hit at {current_price} "
                    f"(stop={stop_loss})"
                ),
                "close_pct": 1.0,
            }
            log_with_data(logger, "warning", "Exit: stop loss hit", {
                "symbol": symbol, "current_price": current_price,
                "stop_loss": stop_loss,
            })
            return result

        # ----- (b) Take profit levels ---------------------------------------
        # TP1: close 35%
        if not tp1_hit and self._take_profit_hit(direction, current_price, tp1):
            result = {
                **base_result,
                "action": "close_partial",
                "reason": (
                    f"TP1 reached at {current_price} "
                    f"(target={tp1})"
                ),
                "close_pct": self._tp1_close_pct,
            }
            log_with_data(logger, "info", "Exit: TP1 reached", {
                "symbol": symbol, "current_price": current_price,
                "tp1": tp1, "close_pct": self._tp1_close_pct,
            })
            return result

        # TP2: close 35%
        if tp1_hit and not tp2_hit and self._take_profit_hit(direction, current_price, tp2):
            result = {
                **base_result,
                "action": "close_partial",
                "reason": (
                    f"TP2 reached at {current_price} "
                    f"(target={tp2})"
                ),
                "close_pct": self._tp2_close_pct,
            }
            log_with_data(logger, "info", "Exit: TP2 reached", {
                "symbol": symbol, "current_price": current_price,
                "tp2": tp2, "close_pct": self._tp2_close_pct,
            })
            return result

        # TP3: trailing stop on the remaining position
        if tp1_hit and tp2_hit:
            trailing_stop = indicators.get("trailing_stop")
            if trailing_stop is not None and self._stop_loss_hit(
                direction, current_price, trailing_stop
            ):
                result = {
                    **base_result,
                    "action": "close_full",
                    "reason": (
                        f"TP3 trailing stop hit at {current_price} "
                        f"(trailing_stop={trailing_stop})"
                    ),
                    "close_pct": 1.0,
                }
                log_with_data(logger, "info", "Exit: TP3 trailing stop", {
                    "symbol": symbol, "current_price": current_price,
                    "trailing_stop": trailing_stop,
                })
                return result

        # ----- (c) Strategy reversal signal ---------------------------------
        if signals:
            avg_score = sum(s.score for s in signals) / len(signals)
            is_reversal = (
                (direction == SignalDirection.LONG.value and avg_score < -50)
                or (direction == SignalDirection.SHORT.value and avg_score > 50)
            )
            if is_reversal:
                result = {
                    **base_result,
                    "action": "close_full",
                    "reason": (
                        f"Strategy reversal detected "
                        f"(avg_signal_score={avg_score:.1f}, "
                        f"position_direction={direction})"
                    ),
                    "close_pct": 1.0,
                }
                log_with_data(logger, "warning", "Exit: strategy reversal", {
                    "symbol": symbol, "avg_score": avg_score,
                    "direction": direction,
                })
                return result

        # ----- (d) Regime changed to unfavorable ----------------------------
        current_regime = indicators.get("current_regime", "")
        regime_at_entry = position.get("regime_at_entry", "")
        if current_regime and regime_at_entry and current_regime != regime_at_entry:
            unfavorable = self._regime_unfavorable(direction, current_regime)
            if unfavorable:
                result = {
                    **base_result,
                    "action": "close_full",
                    "reason": (
                        f"Regime changed from {regime_at_entry} to "
                        f"{current_regime} (unfavorable for {direction})"
                    ),
                    "close_pct": 1.0,
                }
                log_with_data(logger, "warning", "Exit: unfavorable regime", {
                    "symbol": symbol, "old_regime": regime_at_entry,
                    "new_regime": current_regime, "direction": direction,
                })
                return result

        # ----- (e) Time-based exit for swing trades -------------------------
        open_ts = position.get("open_timestamp", 0.0)
        strategy_name = position.get("strategy", "")
        if strategy_name == "swing" and open_ts > 0:
            days_held = (time.time() - open_ts) / 86400.0
            if days_held >= self._swing_max_hold_days:
                result = {
                    **base_result,
                    "action": "close_full",
                    "reason": (
                        f"Swing trade exceeded max hold time "
                        f"({days_held:.1f} days >= "
                        f"{self._swing_max_hold_days} days)"
                    ),
                    "close_pct": 1.0,
                }
                log_with_data(logger, "info", "Exit: swing time limit", {
                    "symbol": symbol, "days_held": round(days_held, 1),
                    "max_days": self._swing_max_hold_days,
                })
                return result

        # ----- No exit condition met ----------------------------------------
        return {
            **base_result,
            "action": "hold",
            "reason": "No exit condition triggered",
            "close_pct": 0.0,
        }

    # ------------------------------------------------------------------
    # Trading pause check
    # ------------------------------------------------------------------

    def should_pause_trading(
        self,
        daily_stats: dict[str, Any],
        equity_stats: dict[str, Any],
    ) -> tuple[bool, str]:
        """Determine whether the system should pause all trading activity.

        Three independent circuit breakers are checked:

        1. Daily loss exceeds ``daily_loss_limit_pct``.
        2. Portfolio drawdown exceeds ``max_drawdown_pct``.
        3. Consecutive losses reach ``consecutive_loss_threshold``.

        Args:
            daily_stats: Dictionary with keys:

                * ``"daily_pnl_pct"`` (float)
                * ``"consecutive_losses"`` (int)

            equity_stats: Dictionary with keys:

                * ``"current_drawdown_pct"`` (float) — current drawdown
                  from peak equity as a positive percentage.

        Returns:
            Tuple of ``(should_pause, reason)``.  If trading should continue,
            *reason* is an empty string.
        """
        daily_pnl_pct = daily_stats.get("daily_pnl_pct", 0.0)
        consecutive_losses = daily_stats.get("consecutive_losses", 0)
        drawdown_pct = equity_stats.get("current_drawdown_pct", 0.0)

        # --- Check 1: Daily loss limit ---
        if daily_pnl_pct <= -self._daily_loss_limit_pct:
            reason = (
                f"Daily loss limit breached: PnL={daily_pnl_pct:.2f}% "
                f"exceeds -{self._daily_loss_limit_pct}% limit"
            )
            log_with_data(logger, "warning", "Trading paused: daily loss", {
                "daily_pnl_pct": daily_pnl_pct,
                "limit": self._daily_loss_limit_pct,
            })
            return True, reason

        # --- Check 2: Max drawdown ---
        if drawdown_pct >= self._max_drawdown_pct:
            reason = (
                f"Max drawdown breached: drawdown={drawdown_pct:.2f}% "
                f"exceeds {self._max_drawdown_pct}% limit"
            )
            log_with_data(logger, "error", "Trading paused: max drawdown", {
                "drawdown_pct": drawdown_pct,
                "limit": self._max_drawdown_pct,
            })
            return True, reason

        # --- Check 3: Consecutive losses ---
        if consecutive_losses >= self._consecutive_loss_threshold:
            reason = (
                f"Consecutive loss threshold reached: "
                f"{consecutive_losses} losses >= "
                f"{self._consecutive_loss_threshold} threshold"
            )
            log_with_data(logger, "warning", "Trading paused: consecutive losses", {
                "consecutive_losses": consecutive_losses,
                "threshold": self._consecutive_loss_threshold,
            })
            return True, reason

        return False, ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stop_loss_hit(direction: str, current_price: float, stop: float) -> bool:
        """Check if price has hit or crossed the stop loss level.

        Args:
            direction: ``"long"`` or ``"short"``.
            current_price: Latest market price.
            stop: Stop loss price level.

        Returns:
            ``True`` if stop loss has been triggered.
        """
        if stop <= 0:
            return False
        if direction == SignalDirection.LONG.value:
            return current_price <= stop
        if direction == SignalDirection.SHORT.value:
            return current_price >= stop
        return False

    @staticmethod
    def _take_profit_hit(
        direction: str, current_price: float, target: float
    ) -> bool:
        """Check if price has reached a take-profit target.

        Args:
            direction: ``"long"`` or ``"short"``.
            current_price: Latest market price.
            target: Take profit price level.

        Returns:
            ``True`` if take-profit level has been reached.
        """
        if target <= 0:
            return False
        if direction == SignalDirection.LONG.value:
            return current_price >= target
        if direction == SignalDirection.SHORT.value:
            return current_price <= target
        return False

    @staticmethod
    def _regime_unfavorable(direction: str, regime: str) -> bool:
        """Determine if the current regime is unfavorable for the position.

        A simple heuristic mapping:
        - Longs are unfavorable in bearish / high-volatility-down regimes.
        - Shorts are unfavorable in bullish / low-volatility-up regimes.

        Args:
            direction: ``"long"`` or ``"short"``.
            regime: Current market regime string.

        Returns:
            ``True`` if the regime is considered unfavorable.
        """
        regime_lower = regime.lower()

        unfavorable_long = {"bearish", "crash", "high_volatility_down", "distribution"}
        unfavorable_short = {"bullish", "strong_uptrend", "low_volatility_up", "accumulation"}

        if direction == SignalDirection.LONG.value:
            return regime_lower in unfavorable_long
        if direction == SignalDirection.SHORT.value:
            return regime_lower in unfavorable_short
        return False

    @staticmethod
    def _skip(
        symbol: str,
        direction: str,
        score: float,
        reason: str,
        checks_passed: list[str],
        checks_failed: list[str],
    ) -> dict[str, Any]:
        """Build a 'skip' decision result.

        Args:
            symbol: Trading pair symbol.
            direction: Signal direction.
            score: Absolute weighted score.
            reason: Human-readable skip reason.
            checks_passed: List of check descriptions that passed.
            checks_failed: List of check descriptions that failed.

        Returns:
            Decision dict with ``action="skip"``.
        """
        result: dict[str, Any] = {
            "action": "skip",
            "reason": reason,
            "symbol": symbol,
            "direction": direction,
            "score": score,
            "position_size_pct": 0.0,
            "checks_passed": checks_passed,
            "checks_failed": checks_failed,
        }

        log_with_data(logger, "info", "Trade decision: SKIP", {
            "symbol": symbol,
            "reason": reason,
            "direction": direction,
            "score": score,
        })

        return result
