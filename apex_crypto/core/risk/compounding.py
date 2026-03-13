"""Compounding and adaptive position sizing engine for APEX.

Implements anti-martingale sizing, drawdown-adjusted sizing,
volatility-adjusted sizing, and portfolio compounding.
"""

from __future__ import annotations

import random
import time
from typing import Any, Optional

from apex_crypto.core.logging import get_logger, log_with_data

logger = get_logger("risk.compounding")


class CompoundingEngine:
    """Adaptive position sizing engine with anti-martingale compounding.

    Combines four sizing systems to dynamically adjust risk per trade:

    1. **Anti-martingale**: Increase risk after wins, decrease after losses.
    2. **Drawdown-adjusted**: Reduce risk proportionally during drawdowns.
    3. **Volatility-adjusted**: Size up in low-vol regimes, down in high-vol.
    4. **Portfolio compounding**: Periodically recalibrate to grown equity.

    Attributes:
        base_risk_pct: Starting risk percentage per trade.
        anti_martingale_step: Fractional step to increase/decrease risk.
        anti_martingale_floor: Minimum allowed risk percentage.
        anti_martingale_ceiling: Maximum allowed risk percentage.
        drawdown_tiers: List of (low, high, multiplier) drawdown ranges.
        vol_low_percentile: Percentile below which volatility is "low".
        vol_high_percentile: Percentile above which volatility is "high".
        vol_low_multiplier: Risk multiplier in low-volatility regimes.
        vol_high_multiplier: Risk multiplier in high-volatility regimes.
        compound_frequency: Number of closed trades between equity resizes.
    """

    def __init__(self, config: dict) -> None:
        """Initialise the compounding engine from the risk config section.

        Args:
            config: The ``risk`` or ``compounding`` section of config.yaml.
        """
        self.base_risk_pct: float = config.get("base_risk_pct", 1.0)
        self.anti_martingale_step: float = config.get(
            "anti_martingale_step", 0.10
        )
        self.anti_martingale_floor: float = config.get(
            "anti_martingale_floor", 0.5
        )
        self.anti_martingale_ceiling: float = config.get(
            "anti_martingale_ceiling", 3.0
        )
        self.drawdown_tiers: list[tuple[float, float, float]] = config.get(
            "drawdown_tiers",
            [
                (0, 3, 1.0),
                (3, 6, 0.7),
                (6, 9, 0.5),
                (9, 12, 0.25),
            ],
        )
        self.vol_low_percentile: float = config.get("vol_low_percentile", 30)
        self.vol_high_percentile: float = config.get("vol_high_percentile", 70)
        self.vol_low_multiplier: float = config.get("vol_low_multiplier", 1.5)
        self.vol_high_multiplier: float = config.get("vol_high_multiplier", 0.5)
        self.compound_frequency: int = config.get("compound_frequency", 10)

        # Profit locking: lock a fraction of profits every N% growth
        self.profit_lock_step: float = config.get("profit_lock_step", 0.20)
        self.profit_lock_fraction: float = config.get("profit_lock_fraction", 0.10)
        self.rebalance_every_hours: int = config.get("rebalance_every_hours", 6)

        # Ultra mode: faster compounding, no profit locking, growth^0.7
        self.ultra_mode: bool = config.get("ultra_mode", False)
        self.growth_exponent: float = config.get("growth_exponent", 0.5)

        # Internal state
        self._consecutive_wins: int = 0
        self._consecutive_losses: int = 0
        self._current_risk_pct: float = self.base_risk_pct
        self._trades_since_resize: int = 0
        self._equity_snapshots: list[tuple[float, float]] = []

        # Profit locking state
        self._start_equity: Optional[float] = None
        self._last_locked_equity: Optional[float] = None
        self._locked_profit_total: float = 0.0
        self._last_rebalance_ts: float = 0.0

        log_with_data(
            logger,
            "info",
            "CompoundingEngine initialised",
            {
                "base_risk_pct": self.base_risk_pct,
                "anti_martingale_step": self.anti_martingale_step,
                "anti_martingale_floor": self.anti_martingale_floor,
                "anti_martingale_ceiling": self.anti_martingale_ceiling,
                "drawdown_tiers": self.drawdown_tiers,
                "vol_low_percentile": self.vol_low_percentile,
                "vol_high_percentile": self.vol_high_percentile,
                "vol_low_multiplier": self.vol_low_multiplier,
                "vol_high_multiplier": self.vol_high_multiplier,
                "compound_frequency": self.compound_frequency,
                "profit_lock_step": self.profit_lock_step,
                "profit_lock_fraction": self.profit_lock_fraction,
                "rebalance_every_hours": self.rebalance_every_hours,
            },
        )

    # ------------------------------------------------------------------
    # Core adaptive risk calculation
    # ------------------------------------------------------------------

    def calculate_adaptive_risk(
        self,
        base_risk_pct: float,
        equity_stats: dict[str, Any],
        market_vol_percentile: float,
        last_trade_won: Optional[bool],
    ) -> dict[str, Any]:
        """Calculate the fully-adjusted risk percentage for the next trade.

        Applies three layers of adjustment on top of the base risk:

        1. Anti-martingale step (increase after wins, decrease after losses).
        2. Drawdown multiplier (reduce risk during portfolio drawdowns).
        3. Volatility multiplier (size up in calm markets, down in wild ones).

        Args:
            base_risk_pct: Starting risk percentage before adjustments.
            equity_stats: Dictionary containing at least
                ``current_drawdown_pct`` (positive float, e.g. 4.5 for 4.5%).
            market_vol_percentile: Current market volatility expressed as a
                percentile (0-100) relative to historical range.
            last_trade_won: ``True`` if the most recent trade was a winner,
                ``False`` if it was a loser, ``None`` if no prior trade.

        Returns:
            Dictionary with the computed ``risk_pct`` and all intermediate
            multipliers and streak counters.
        """
        # Step 1 — Anti-martingale adjustment
        anti_martingale_risk = self._current_risk_pct

        if last_trade_won is True:
            self._consecutive_wins += 1
            self._consecutive_losses = 0
            anti_martingale_risk += self.anti_martingale_step
            log_with_data(
                logger,
                "debug",
                "Anti-martingale: win streak — increasing risk",
                {
                    "consecutive_wins": self._consecutive_wins,
                    "step": self.anti_martingale_step,
                    "new_risk_before_clamp": anti_martingale_risk,
                },
            )
        elif last_trade_won is False:
            self._consecutive_losses += 1
            self._consecutive_wins = 0
            anti_martingale_risk -= self.anti_martingale_step
            log_with_data(
                logger,
                "debug",
                "Anti-martingale: loss streak — decreasing risk",
                {
                    "consecutive_losses": self._consecutive_losses,
                    "step": self.anti_martingale_step,
                    "new_risk_before_clamp": anti_martingale_risk,
                },
            )
        else:
            log_with_data(
                logger,
                "debug",
                "Anti-martingale: no prior trade result — keeping current risk",
                {"current_risk_pct": anti_martingale_risk},
            )

        # Clamp to floor and ceiling
        anti_martingale_risk = max(
            self.anti_martingale_floor,
            min(anti_martingale_risk, self.anti_martingale_ceiling),
        )
        self._current_risk_pct = anti_martingale_risk

        # Step 2 — Drawdown adjustment
        current_drawdown_pct = equity_stats.get("current_drawdown_pct", 0.0)
        drawdown_multiplier = self.get_drawdown_multiplier(current_drawdown_pct)

        log_with_data(
            logger,
            "debug",
            "Drawdown adjustment applied",
            {
                "current_drawdown_pct": current_drawdown_pct,
                "drawdown_multiplier": drawdown_multiplier,
            },
        )

        # Step 3 — Volatility adjustment
        volatility_multiplier = self.get_volatility_multiplier(
            market_vol_percentile
        )

        log_with_data(
            logger,
            "debug",
            "Volatility adjustment applied",
            {
                "market_vol_percentile": market_vol_percentile,
                "volatility_multiplier": volatility_multiplier,
            },
        )

        # Step 4 — Combine all layers
        final_risk = anti_martingale_risk * drawdown_multiplier * volatility_multiplier
        final_risk = min(final_risk, self.anti_martingale_ceiling)

        result: dict[str, Any] = {
            "risk_pct": round(final_risk, 4),
            "base_risk": base_risk_pct,
            "anti_martingale_risk": round(anti_martingale_risk, 4),
            "drawdown_multiplier": round(drawdown_multiplier, 4),
            "volatility_multiplier": round(volatility_multiplier, 4),
            "consecutive_wins": self._consecutive_wins,
            "consecutive_losses": self._consecutive_losses,
        }

        log_with_data(
            logger,
            "info",
            "Adaptive risk calculated",
            result,
        )

        return result

    # ------------------------------------------------------------------
    # Trade result tracking
    # ------------------------------------------------------------------

    def record_trade_result(
        self, won: bool, pnl_pct: float, equity: float
    ) -> None:
        """Record a completed trade and update internal state.

        After every ``compound_frequency`` trades, a resize event is
        triggered so that position sizes naturally scale with equity.

        Args:
            won: Whether the trade was a winner.
            pnl_pct: Profit or loss of the trade as a percentage.
            equity: Portfolio equity after the trade closed.
        """
        # Update streak counters (these are also set in calculate_adaptive_risk,
        # but record_trade_result can be called independently)
        if won:
            self._consecutive_wins += 1
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            self._consecutive_wins = 0

        self._trades_since_resize += 1

        # Snapshot equity with timestamp
        snapshot = (time.time(), equity)
        self._equity_snapshots.append(snapshot)

        log_with_data(
            logger,
            "info",
            "Trade result recorded",
            {
                "won": won,
                "pnl_pct": round(pnl_pct, 4),
                "equity": round(equity, 2),
                "consecutive_wins": self._consecutive_wins,
                "consecutive_losses": self._consecutive_losses,
                "trades_since_resize": self._trades_since_resize,
                "total_snapshots": len(self._equity_snapshots),
            },
        )

        # Check if it is time to compound-resize
        if self._trades_since_resize >= self.compound_frequency:
            self._compound_resize(equity)

    # ------------------------------------------------------------------
    # Compound resize
    # ------------------------------------------------------------------

    def _compound_resize(self, equity: float) -> None:
        """Trigger a compounding resize event.

        Position sizes naturally grow with equity — this method simply
        resets the trade counter and logs the event so operators can
        track when resizes occur.

        Args:
            equity: Current portfolio equity at the time of resize.
        """
        log_with_data(
            logger,
            "info",
            "Compound resize triggered — position sizes recalibrated to current equity",
            {
                "equity": round(equity, 2),
                "trades_since_last_resize": self._trades_since_resize,
                "current_risk_pct": round(self._current_risk_pct, 4),
                "consecutive_wins": self._consecutive_wins,
                "consecutive_losses": self._consecutive_losses,
            },
        )
        self._trades_since_resize = 0

    # ------------------------------------------------------------------
    # Drawdown multiplier lookup
    # ------------------------------------------------------------------

    def get_drawdown_multiplier(self, drawdown_pct: float) -> float:
        """Look up the risk multiplier for the current drawdown level.

        Searches ``drawdown_tiers`` for the tier whose range contains
        ``drawdown_pct`` and returns its multiplier.  If no tier matches
        (drawdown exceeds all defined ranges), the most conservative
        (last) tier multiplier is returned.

        Args:
            drawdown_pct: Current portfolio drawdown as a positive
                percentage (e.g. 5.0 for a 5% drawdown).

        Returns:
            Risk multiplier in the range (0, 1].
        """
        for low, high, multiplier in self.drawdown_tiers:
            if low <= drawdown_pct < high:
                log_with_data(
                    logger,
                    "debug",
                    "Drawdown tier matched",
                    {
                        "drawdown_pct": drawdown_pct,
                        "tier_low": low,
                        "tier_high": high,
                        "multiplier": multiplier,
                    },
                )
                return multiplier

        # Drawdown exceeds all tiers — use the most conservative multiplier
        if self.drawdown_tiers:
            fallback = self.drawdown_tiers[-1][2]
            log_with_data(
                logger,
                "warning",
                "Drawdown exceeds all defined tiers — using most conservative multiplier",
                {
                    "drawdown_pct": drawdown_pct,
                    "fallback_multiplier": fallback,
                },
            )
            return fallback

        log_with_data(
            logger,
            "warning",
            "No drawdown tiers configured — returning 1.0",
            {"drawdown_pct": drawdown_pct},
        )
        return 1.0

    # ------------------------------------------------------------------
    # Volatility multiplier
    # ------------------------------------------------------------------

    def get_volatility_multiplier(self, vol_percentile: float) -> float:
        """Determine the risk multiplier based on market volatility regime.

        Low volatility environments receive a size boost (larger
        positions when markets are calm), while high volatility
        environments receive a size reduction.

        Args:
            vol_percentile: Market volatility as a percentile (0-100)
                relative to historical range.

        Returns:
            Risk multiplier: ``vol_low_multiplier`` (1.5) when calm,
            ``vol_high_multiplier`` (0.5) when volatile, or 1.0 for
            normal conditions.
        """
        if vol_percentile < self.vol_low_percentile:
            log_with_data(
                logger,
                "debug",
                "Volatility regime: LOW — increasing position size",
                {
                    "vol_percentile": vol_percentile,
                    "threshold": self.vol_low_percentile,
                    "multiplier": self.vol_low_multiplier,
                },
            )
            return self.vol_low_multiplier

        if vol_percentile > self.vol_high_percentile:
            log_with_data(
                logger,
                "debug",
                "Volatility regime: HIGH — decreasing position size",
                {
                    "vol_percentile": vol_percentile,
                    "threshold": self.vol_high_percentile,
                    "multiplier": self.vol_high_multiplier,
                },
            )
            return self.vol_high_multiplier

        log_with_data(
            logger,
            "debug",
            "Volatility regime: NORMAL — no adjustment",
            {
                "vol_percentile": vol_percentile,
                "low_threshold": self.vol_low_percentile,
                "high_threshold": self.vol_high_percentile,
                "multiplier": 1.0,
            },
        )
        return 1.0

    # ------------------------------------------------------------------
    # Profit locking
    # ------------------------------------------------------------------

    def maybe_lock_profits(self, current_equity: float) -> float:
        """Lock a fraction of profits when equity grows past a threshold.

        When equity grows by ``profit_lock_step`` (default 20%) above the
        last locked level, ``profit_lock_fraction`` of the current equity
        is "locked" — reducing the tradable equity.

        In ultra mode with profit_lock_fraction=0, all equity stays in play.

        Args:
            current_equity: Current total portfolio equity.

        Returns:
            Effective equity available for trading (after locking).
        """
        # Ultra mode with zero lock fraction: all equity available
        if self.profit_lock_fraction <= 0:
            if self._start_equity is None:
                self._start_equity = current_equity
                self._last_locked_equity = current_equity
            return current_equity

        if self._start_equity is None:
            self._start_equity = current_equity
            self._last_locked_equity = current_equity
            return current_equity

        base = self._last_locked_equity or self._start_equity
        if base <= 0:
            return current_equity

        growth = (current_equity - base) / base

        if growth >= self.profit_lock_step:
            lock_amount = current_equity * self.profit_lock_fraction
            new_equity_for_trading = current_equity - lock_amount
            self._last_locked_equity = new_equity_for_trading
            self._locked_profit_total += lock_amount

            log_with_data(
                logger,
                "info",
                "Profit locked — reducing tradable equity",
                {
                    "current_equity": round(current_equity, 2),
                    "growth_pct": round(growth * 100, 2),
                    "lock_amount": round(lock_amount, 2),
                    "new_trading_equity": round(new_equity_for_trading, 2),
                    "total_locked": round(self._locked_profit_total, 2),
                },
            )

            return new_equity_for_trading

        return current_equity

    # ------------------------------------------------------------------
    # Time-based rebalance check
    # ------------------------------------------------------------------

    def should_rebalance(self, now_ts: float) -> bool:
        """Check if enough time or trades have passed for a rebalance.

        Args:
            now_ts: Current Unix timestamp.

        Returns:
            True if a rebalance should be triggered.
        """
        if self._last_rebalance_ts == 0.0:
            self._last_rebalance_ts = now_ts
            return False

        # Trade-count trigger
        if self._trades_since_resize >= self.compound_frequency:
            return True

        # Time trigger
        hours_elapsed = (now_ts - self._last_rebalance_ts) / 3600.0
        if hours_elapsed >= self.rebalance_every_hours:
            return True

        return False

    def on_rebalance(self, equity: float, now_ts: float) -> None:
        """Execute a rebalance event: recalibrate risk to current equity.

        Scales base_risk_pct proportionally to equity growth.
        Normal mode: growth^0.5 (sqrt scaling, conservative).
        Ultra mode: growth^0.7 (aggressive — equity doubles -> risk grows 62%).

        Args:
            equity: Current tradable equity.
            now_ts: Current Unix timestamp.
        """
        if self._start_equity and self._start_equity > 0:
            growth_factor = equity / self._start_equity
            exponent = self.growth_exponent
            new_base_risk = min(
                self.anti_martingale_ceiling,
                self.base_risk_pct * (growth_factor ** exponent),
            )
            old_base = self.base_risk_pct
            self.base_risk_pct = new_base_risk

            log_with_data(
                logger,
                "info",
                "Rebalance: base risk recalibrated to equity growth",
                {
                    "old_base_risk": round(old_base, 4),
                    "new_base_risk": round(new_base_risk, 4),
                    "growth_factor": round(growth_factor, 4),
                    "equity": round(equity, 2),
                },
            )

        self._last_rebalance_ts = now_ts
        self._trades_since_resize = 0

    @property
    def locked_profit_total(self) -> float:
        """Total profit amount locked so far."""
        return self._locked_profit_total

    # ------------------------------------------------------------------
    # Equity projection (Monte Carlo)
    # ------------------------------------------------------------------

    def project_equity(
        self,
        starting_equity: float,
        win_rate: float,
        avg_win_pct: float,
        avg_loss_pct: float,
        trades_per_day: int,
        days: int,
    ) -> list[dict[str, Any]]:
        """Project equity growth using Monte Carlo-style simulation.

        Runs three scenarios — conservative (win_rate - 5%), base, and
        aggressive (win_rate + 5%) — each simulating ``trades_per_day``
        trades over ``days`` days.  Equity compounds after every trade.

        Args:
            starting_equity: Initial portfolio value in USDT.
            win_rate: Expected win rate (0.0 – 1.0).
            avg_win_pct: Average winning trade return as a percentage
                (e.g. 2.0 for 2%).
            avg_loss_pct: Average losing trade return as a positive
                percentage (e.g. 1.0 for 1%).
            trades_per_day: Number of trades expected per day.
            days: Number of days to project.

        Returns:
            List of dictionaries, one per day, each containing
            ``day``, ``conservative``, ``base``, and ``aggressive``
            equity values plus ``cumulative_return_pct`` for each
            scenario.
        """
        log_with_data(
            logger,
            "info",
            "Starting equity projection",
            {
                "starting_equity": starting_equity,
                "win_rate": win_rate,
                "avg_win_pct": avg_win_pct,
                "avg_loss_pct": avg_loss_pct,
                "trades_per_day": trades_per_day,
                "days": days,
            },
        )

        scenarios = {
            "conservative": max(0.0, min(win_rate - 0.05, 1.0)),
            "base": win_rate,
            "aggressive": max(0.0, min(win_rate + 0.05, 1.0)),
        }

        # Seed for reproducibility within a single call
        rng = random.Random(42)

        projection: list[dict[str, Any]] = []

        # Track running equity for each scenario
        equities = {
            "conservative": starting_equity,
            "base": starting_equity,
            "aggressive": starting_equity,
        }

        for day in range(1, days + 1):
            for scenario_name, scenario_wr in scenarios.items():
                equity = equities[scenario_name]

                for _ in range(trades_per_day):
                    if rng.random() < scenario_wr:
                        # Winning trade
                        equity *= 1.0 + (avg_win_pct / 100.0)
                    else:
                        # Losing trade
                        equity *= 1.0 - (avg_loss_pct / 100.0)

                equities[scenario_name] = equity

            day_result: dict[str, Any] = {
                "day": day,
                "conservative": round(equities["conservative"], 2),
                "base": round(equities["base"], 2),
                "aggressive": round(equities["aggressive"], 2),
                "cumulative_return_pct": {
                    "conservative": round(
                        ((equities["conservative"] - starting_equity)
                         / starting_equity) * 100.0,
                        2,
                    ),
                    "base": round(
                        ((equities["base"] - starting_equity)
                         / starting_equity) * 100.0,
                        2,
                    ),
                    "aggressive": round(
                        ((equities["aggressive"] - starting_equity)
                         / starting_equity) * 100.0,
                        2,
                    ),
                },
            }
            projection.append(day_result)

        log_with_data(
            logger,
            "info",
            "Equity projection complete",
            {
                "days_projected": days,
                "final_conservative": projection[-1]["conservative"] if projection else None,
                "final_base": projection[-1]["base"] if projection else None,
                "final_aggressive": projection[-1]["aggressive"] if projection else None,
            },
        )

        return projection

    # ------------------------------------------------------------------
    # Equity snapshots (for dashboard)
    # ------------------------------------------------------------------

    def get_equity_snapshots(self) -> list[dict[str, Any]]:
        """Return formatted equity snapshots for the dashboard.

        Each snapshot includes a timestamp and equity value recorded
        when ``record_trade_result`` was called.

        Returns:
            List of dictionaries with ``timestamp`` and ``equity`` keys,
            ordered chronologically.
        """
        snapshots = [
            {
                "timestamp": ts,
                "equity": round(eq, 2),
            }
            for ts, eq in self._equity_snapshots
        ]

        log_with_data(
            logger,
            "debug",
            "Equity snapshots retrieved",
            {"snapshot_count": len(snapshots)},
        )

        return snapshots
