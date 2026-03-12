"""Risk limit enforcement for APEX Crypto Trading System.

Tracks daily P&L, drawdown, position counts, trade counts, consecutive
losses, leverage, and asset concentration.  Every limit check returns a
``(bool, str)`` tuple so callers can programmatically decide whether to
proceed and surface a human-readable reason when a limit is breached.
"""

import time
import logging
from typing import Any, Optional

from apex_crypto.core.logging import get_logger, log_with_data

logger = get_logger("risk.limits")


class RiskLimits:
    """Hard and soft risk limits that gate every trade decision.

    All thresholds are loaded from the ``risk`` section of config.yaml.
    State (e.g. peak equity) is tracked in-memory with an optional Redis
    persistence layer for crash recovery.

    Attributes:
        daily_loss_limit_pct: Max daily loss as % of portfolio before pause.
        max_drawdown_pct: Max drawdown from ATH before full system halt.
        max_open_positions: Maximum number of concurrent positions.
        max_trades_per_day: Maximum trades allowed per calendar day.
        consecutive_loss_threshold: Losses in a row before cooldown kicks in.
        cooldown_minutes: Mandatory cooldown duration after consecutive losses.
        max_leverage: Hard ceiling on leverage.
        default_leverage: Leverage used when none is specified.
        max_asset_concentration_pct: Max % of portfolio in any single asset.
    """

    def __init__(self, config: dict) -> None:
        """Initialise risk limits from the risk config section.

        Args:
            config: The ``risk`` section of config.yaml.
        """
        self.daily_loss_limit_pct: float = config.get("daily_loss_limit_pct", 3.0)
        self.max_drawdown_pct: float = config.get("max_drawdown_pct", 12.0)
        self.max_open_positions: int = int(config.get("max_open_positions", 8))
        self.max_trades_per_day: int = int(config.get("max_trades_per_day", 25))
        self.consecutive_loss_threshold: int = int(
            config.get("consecutive_loss_threshold", 3)
        )
        self.cooldown_minutes: int = int(
            config.get("consecutive_loss_cooldown_minutes", 120)
        )
        self.max_leverage: float = config.get("max_leverage", 3.0)
        self.default_leverage: float = config.get("default_leverage", 1.0)
        self.max_asset_concentration_pct: float = config.get(
            "max_asset_concentration_pct", 10.0
        )

        # In-memory state — can be hydrated from Redis on restart
        self._peak_equity: float = 0.0
        self._system_halted: bool = False

        log_with_data(
            logger,
            "info",
            "RiskLimits initialised",
            {
                "daily_loss_limit_pct": self.daily_loss_limit_pct,
                "max_drawdown_pct": self.max_drawdown_pct,
                "max_open_positions": self.max_open_positions,
                "max_trades_per_day": self.max_trades_per_day,
                "consecutive_loss_threshold": self.consecutive_loss_threshold,
                "cooldown_minutes": self.cooldown_minutes,
                "max_leverage": self.max_leverage,
                "max_asset_concentration_pct": self.max_asset_concentration_pct,
            },
        )

    # ------------------------------------------------------------------
    # Aggregate check
    # ------------------------------------------------------------------

    def check_all_limits(
        self,
        daily_stats: dict[str, Any],
        equity_stats: dict[str, Any],
        open_positions: list[dict[str, Any]],
    ) -> tuple[bool, list[str]]:
        """Run every limit check and return a combined verdict.

        Args:
            daily_stats: Must contain keys ``daily_pnl`` (float),
                ``portfolio_value`` (float), ``trade_count`` (int),
                ``consecutive_losses`` (int), and
                ``last_loss_time`` (float, epoch seconds).
            equity_stats: Must contain ``current_equity`` (float) and
                ``peak_equity`` (float).
            open_positions: List of open-position dicts, each with at
                least ``symbol`` (str) and ``notional_usdt`` (float).

        Returns:
            Tuple of ``(can_trade, violations)`` where *violations* is a
            list of human-readable strings for every breached limit.
        """
        violations: list[str] = []

        # 1. Daily loss
        ok, msg = self.check_daily_loss_limit(
            daily_stats.get("daily_pnl", 0.0),
            daily_stats.get("portfolio_value", 0.0),
        )
        if not ok:
            violations.append(msg)

        # 2. Drawdown
        ok, msg = self.check_max_drawdown(
            equity_stats.get("current_equity", 0.0),
            equity_stats.get("peak_equity", 0.0),
        )
        if not ok:
            violations.append(msg)

        # 3. Position count
        ok, msg = self.check_position_count(open_positions)
        if not ok:
            violations.append(msg)

        # 4. Daily trade count
        ok, msg = self.check_daily_trade_count(
            daily_stats.get("trade_count", 0)
        )
        if not ok:
            violations.append(msg)

        # 5. Consecutive losses
        ok, msg = self.check_consecutive_losses(
            daily_stats.get("consecutive_losses", 0),
            daily_stats.get("last_loss_time", 0.0),
        )
        if not ok:
            violations.append(msg)

        can_trade = len(violations) == 0

        log_with_data(
            logger,
            "info" if can_trade else "warning",
            "All limits checked",
            {
                "can_trade": can_trade,
                "violation_count": len(violations),
                "violations": violations,
            },
        )

        return can_trade, violations

    # ------------------------------------------------------------------
    # Individual limit checks
    # ------------------------------------------------------------------

    def check_daily_loss_limit(
        self,
        daily_pnl: float,
        portfolio_value: float,
    ) -> tuple[bool, str]:
        """Check whether the daily loss limit has been breached.

        Args:
            daily_pnl: Realised + unrealised P&L for the current day
                (negative means a loss).
            portfolio_value: Current total portfolio value in USDT.

        Returns:
            ``(within_limit, message)``.  ``within_limit`` is ``False``
            when the loss exceeds the configured threshold.
        """
        if portfolio_value <= 0:
            return False, "Portfolio value is zero or negative — cannot assess daily loss limit"

        loss_pct = abs(daily_pnl) / portfolio_value * 100.0 if daily_pnl < 0 else 0.0

        if loss_pct >= self.daily_loss_limit_pct:
            msg = (
                f"DAILY LOSS LIMIT BREACHED: lost {loss_pct:.2f}% "
                f"(limit {self.daily_loss_limit_pct:.1f}%) — trading paused"
            )
            log_with_data(
                logger,
                "error",
                msg,
                {
                    "daily_pnl": daily_pnl,
                    "portfolio_value": portfolio_value,
                    "loss_pct": round(loss_pct, 4),
                    "limit_pct": self.daily_loss_limit_pct,
                },
            )
            return False, msg

        msg = (
            f"Daily loss within limit: {loss_pct:.2f}% / "
            f"{self.daily_loss_limit_pct:.1f}%"
        )
        log_with_data(logger, "debug", msg, {"loss_pct": round(loss_pct, 4)})
        return True, msg

    def check_max_drawdown(
        self,
        current_equity: float,
        peak_equity: float,
    ) -> tuple[bool, str]:
        """Check whether the maximum drawdown from all-time high is breached.

        If the drawdown exceeds ``max_drawdown_pct`` the system must
        perform a **full halt** — no new trades until manual review.

        Args:
            current_equity: Current portfolio equity in USDT.
            peak_equity: All-time high portfolio equity in USDT.

        Returns:
            ``(within_limit, message)``.
        """
        # Track peak equity internally
        if peak_equity > self._peak_equity:
            self._peak_equity = peak_equity
        effective_peak = max(self._peak_equity, peak_equity)

        if effective_peak <= 0:
            return False, "Peak equity is zero or negative — cannot assess drawdown"

        drawdown_pct = ((effective_peak - current_equity) / effective_peak) * 100.0

        if drawdown_pct >= self.max_drawdown_pct:
            self._system_halted = True
            msg = (
                f"MAX DRAWDOWN BREACHED: {drawdown_pct:.2f}% from ATH "
                f"(limit {self.max_drawdown_pct:.1f}%) — FULL SYSTEM HALT"
            )
            log_with_data(
                logger,
                "critical",
                msg,
                {
                    "current_equity": current_equity,
                    "peak_equity": effective_peak,
                    "drawdown_pct": round(drawdown_pct, 4),
                    "limit_pct": self.max_drawdown_pct,
                    "system_halted": True,
                },
            )
            return False, msg

        msg = (
            f"Drawdown within limit: {drawdown_pct:.2f}% / "
            f"{self.max_drawdown_pct:.1f}%"
        )
        log_with_data(logger, "debug", msg, {"drawdown_pct": round(drawdown_pct, 4)})
        return True, msg

    def check_position_count(
        self,
        open_positions: list[dict[str, Any]],
    ) -> tuple[bool, str]:
        """Check whether the maximum concurrent position count is reached.

        Args:
            open_positions: List of currently open position dicts.

        Returns:
            ``(within_limit, message)``.
        """
        count = len(open_positions)

        if count >= self.max_open_positions:
            msg = (
                f"MAX POSITIONS REACHED: {count} / {self.max_open_positions} "
                f"— no new entries allowed"
            )
            log_with_data(
                logger,
                "warning",
                msg,
                {"open_positions": count, "limit": self.max_open_positions},
            )
            return False, msg

        msg = f"Position count within limit: {count} / {self.max_open_positions}"
        log_with_data(logger, "debug", msg, {"open_positions": count})
        return True, msg

    def check_daily_trade_count(
        self,
        trade_count: int,
    ) -> tuple[bool, str]:
        """Check whether the daily trade count has been exceeded.

        Args:
            trade_count: Number of trades executed today.

        Returns:
            ``(within_limit, message)``.
        """
        if trade_count >= self.max_trades_per_day:
            msg = (
                f"MAX DAILY TRADES REACHED: {trade_count} / "
                f"{self.max_trades_per_day} — no new trades today"
            )
            log_with_data(
                logger,
                "warning",
                msg,
                {"trade_count": trade_count, "limit": self.max_trades_per_day},
            )
            return False, msg

        msg = f"Daily trade count within limit: {trade_count} / {self.max_trades_per_day}"
        log_with_data(logger, "debug", msg, {"trade_count": trade_count})
        return True, msg

    def check_consecutive_losses(
        self,
        consecutive_losses: int,
        last_loss_time: float,
    ) -> tuple[bool, str]:
        """Enforce a cooldown after consecutive losing trades.

        After ``consecutive_loss_threshold`` (default 3) consecutive
        losses, trading is paused for ``cooldown_minutes`` (default 120
        minutes / 2 hours).

        Args:
            consecutive_losses: Number of consecutive losing trades.
            last_loss_time: Epoch timestamp of the most recent loss.

        Returns:
            ``(can_trade, message)``.
        """
        if consecutive_losses < self.consecutive_loss_threshold:
            msg = (
                f"Consecutive losses within limit: {consecutive_losses} / "
                f"{self.consecutive_loss_threshold}"
            )
            log_with_data(
                logger,
                "debug",
                msg,
                {"consecutive_losses": consecutive_losses},
            )
            return True, msg

        cooldown_seconds = self.cooldown_minutes * 60
        elapsed = time.time() - last_loss_time

        if elapsed < cooldown_seconds:
            remaining_minutes = (cooldown_seconds - elapsed) / 60.0
            msg = (
                f"CONSECUTIVE LOSS COOLDOWN: {consecutive_losses} losses in a row "
                f"— {remaining_minutes:.0f} min remaining of "
                f"{self.cooldown_minutes} min cooldown"
            )
            log_with_data(
                logger,
                "warning",
                msg,
                {
                    "consecutive_losses": consecutive_losses,
                    "threshold": self.consecutive_loss_threshold,
                    "cooldown_minutes": self.cooldown_minutes,
                    "remaining_minutes": round(remaining_minutes, 1),
                    "elapsed_seconds": round(elapsed, 1),
                },
            )
            return False, msg

        msg = (
            f"Consecutive loss cooldown expired after {consecutive_losses} losses "
            f"— trading may resume"
        )
        log_with_data(logger, "info", msg, {"consecutive_losses": consecutive_losses})
        return True, msg

    def check_leverage_limit(
        self,
        requested_leverage: float,
        regime: str,
    ) -> float:
        """Clamp requested leverage to the allowed maximum for the regime.

        In the ``CHAOS`` regime the maximum leverage is always 1x
        regardless of configuration.

        Args:
            requested_leverage: Leverage the strategy wants to use.
            regime: Current market regime label (e.g. ``"TRENDING"``,
                ``"RANGING"``, ``"VOLATILE"``, ``"CHAOS"``).

        Returns:
            The allowed leverage value (clamped).
        """
        effective_max = 1.0 if regime.upper() == "CHAOS" else self.max_leverage

        allowed = min(max(requested_leverage, 1.0), effective_max)

        if allowed < requested_leverage:
            log_with_data(
                logger,
                "warning",
                "Leverage clamped",
                {
                    "requested": requested_leverage,
                    "allowed": allowed,
                    "effective_max": effective_max,
                    "regime": regime,
                },
            )
        else:
            log_with_data(
                logger,
                "debug",
                "Leverage within limit",
                {
                    "requested": requested_leverage,
                    "allowed": allowed,
                    "regime": regime,
                },
            )

        return allowed

    def check_asset_concentration(
        self,
        symbol: str,
        open_positions: list[dict[str, Any]],
        portfolio_value: float,
    ) -> tuple[bool, str]:
        """Check that adding to a symbol won't exceed concentration limits.

        Sums the notional value of all open positions for the given
        ``symbol`` and rejects the trade if it would push total exposure
        above ``max_asset_concentration_pct`` of portfolio.

        Args:
            symbol: Ticker to check (e.g. ``"BTC/USDT"``).
            open_positions: List of dicts with ``symbol`` and
                ``notional_usdt`` keys.
            portfolio_value: Current portfolio value in USDT.

        Returns:
            ``(within_limit, message)``.
        """
        if portfolio_value <= 0:
            return False, "Portfolio value is zero or negative — cannot assess concentration"

        current_exposure = sum(
            pos.get("notional_usdt", 0.0)
            for pos in open_positions
            if pos.get("symbol") == symbol
        )
        concentration_pct = (current_exposure / portfolio_value) * 100.0

        if concentration_pct >= self.max_asset_concentration_pct:
            msg = (
                f"ASSET CONCENTRATION LIMIT: {symbol} already at "
                f"{concentration_pct:.2f}% of portfolio "
                f"(limit {self.max_asset_concentration_pct:.1f}%)"
            )
            log_with_data(
                logger,
                "warning",
                msg,
                {
                    "symbol": symbol,
                    "current_exposure_usdt": round(current_exposure, 4),
                    "concentration_pct": round(concentration_pct, 4),
                    "limit_pct": self.max_asset_concentration_pct,
                    "portfolio_value": portfolio_value,
                },
            )
            return False, msg

        msg = (
            f"Asset concentration within limit: {symbol} at "
            f"{concentration_pct:.2f}% / {self.max_asset_concentration_pct:.1f}%"
        )
        log_with_data(
            logger,
            "debug",
            msg,
            {
                "symbol": symbol,
                "concentration_pct": round(concentration_pct, 4),
            },
        )
        return True, msg

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    @property
    def is_system_halted(self) -> bool:
        """Whether the system has been halted due to max drawdown breach."""
        return self._system_halted

    def reset_halt(self) -> None:
        """Manually reset the system halt flag after operator review.

        This should only be called after a human has reviewed the
        drawdown event and confirmed it is safe to resume trading.
        """
        self._system_halted = False
        log_with_data(
            logger,
            "warning",
            "System halt flag manually reset by operator",
            {"peak_equity": self._peak_equity},
        )
