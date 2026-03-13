"""Kelly Criterion + Anti-Martingale position sizing engine.

Complements the existing PositionSizer (sizing.py) by providing
strategy-aware dynamic risk percentages based on:
1. Half-Kelly optimal fraction per strategy
2. Anti-martingale streak adjustments (increase after wins, decrease after losses)
3. Hard floor/ceiling clamping
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from apex_crypto.core.logging import get_logger, log_with_data

logger = get_logger("risk.position_sizing")


@dataclass
class StrategyStats:
    """Rolling performance statistics for a single strategy."""

    win_rate: float  # 0–1
    avg_win_r: float  # Average win in R-multiples
    avg_loss_r: float  # Average loss in R-multiples (positive)
    recent_win_streak: int  # Consecutive wins
    recent_loss_streak: int  # Consecutive losses


class PositionSizingEngine:
    """Kelly + Anti-Martingale dynamic risk calculator.

    Given a strategy's historical stats, computes an optimal risk
    percentage that adapts to both edge quality (Kelly) and recent
    momentum (anti-martingale streaks).

    Attributes:
        base_risk_pct: Default risk when no stats available (fraction, e.g. 0.01).
        max_risk_pct: Hard ceiling (fraction).
        min_risk_pct: Hard floor (fraction).
        kelly_fraction: Fraction of full Kelly to use (0.5 = half-Kelly).
        win_streak_boost: Per-win multiplier boost (e.g. 0.10 = +10% per win).
        loss_streak_cut: Per-loss multiplier reduction.
    """

    def __init__(self, config: dict) -> None:
        self.base_risk_pct: float = config.get("base_risk_pct", 0.01)
        self.max_risk_pct: float = config.get("max_risk_pct", 0.03)
        self.min_risk_pct: float = config.get("min_risk_pct", 0.005)
        self.kelly_fraction: float = config.get("kelly_fraction", 0.5)
        self.win_streak_boost: float = config.get("win_streak_boost", 0.10)
        self.loss_streak_cut: float = config.get("loss_streak_cut", 0.10)

        log_with_data(logger, "info", "PositionSizingEngine initialised", {
            "base_risk_pct": self.base_risk_pct,
            "max_risk_pct": self.max_risk_pct,
            "min_risk_pct": self.min_risk_pct,
            "kelly_fraction": self.kelly_fraction,
            "win_streak_boost": self.win_streak_boost,
            "loss_streak_cut": self.loss_streak_cut,
        })

    # ------------------------------------------------------------------
    # Kelly
    # ------------------------------------------------------------------

    def _kelly_for_strategy(self, stats: StrategyStats) -> float:
        """Compute half-Kelly optimal risk fraction.

        Kelly formula: f* = W - (1 - W) / R
        where W = win_rate, R = avg_win / avg_loss.
        """
        if stats.win_rate <= 0 or stats.avg_loss_r <= 0:
            return self.base_risk_pct

        r = stats.avg_win_r / stats.avg_loss_r
        f_star = stats.win_rate - (1 - stats.win_rate) / r

        if f_star <= 0:
            log_with_data(logger, "debug", "Negative Kelly — no edge, using base risk", {
                "win_rate": stats.win_rate,
                "r_ratio": round(r, 4),
                "f_star": round(f_star, 6),
            })
            return self.base_risk_pct

        kelly_risk = f_star * self.kelly_fraction

        log_with_data(logger, "debug", "Kelly risk computed", {
            "win_rate": stats.win_rate,
            "r_ratio": round(r, 4),
            "full_kelly": round(f_star, 6),
            "half_kelly": round(kelly_risk, 6),
        })

        return kelly_risk

    # ------------------------------------------------------------------
    # Anti-Martingale
    # ------------------------------------------------------------------

    def _apply_anti_martingale(self, risk_pct: float, stats: StrategyStats) -> float:
        """Scale risk up after wins and down after losses.

        Boost is proportional to streak length:
        - After 3 consecutive wins: risk * (1 + 0.10 * 3) = risk * 1.30
        - After 2 consecutive losses: risk * (1 - 0.10 * 2) = risk * 0.80
        """
        adjusted = risk_pct

        if stats.recent_win_streak > 0 and stats.recent_loss_streak == 0:
            adjusted *= (1 + self.win_streak_boost * stats.recent_win_streak)
            log_with_data(logger, "debug", "Anti-martingale: win streak boost", {
                "streak": stats.recent_win_streak,
                "base_risk": round(risk_pct, 6),
                "adjusted_risk": round(adjusted, 6),
            })

        if stats.recent_loss_streak > 0 and stats.recent_win_streak == 0:
            adjusted *= (1 - self.loss_streak_cut * stats.recent_loss_streak)
            # Floor at 0 to prevent negative
            adjusted = max(adjusted, 0.0)
            log_with_data(logger, "debug", "Anti-martingale: loss streak cut", {
                "streak": stats.recent_loss_streak,
                "base_risk": round(risk_pct, 6),
                "adjusted_risk": round(adjusted, 6),
            })

        return adjusted

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_risk_pct(self, stats: Optional[StrategyStats]) -> float:
        """Compute the final risk percentage for a trade.

        Applies Kelly → Anti-Martingale → Clamping.

        Args:
            stats: Strategy performance stats, or None for base risk.

        Returns:
            Risk as a fraction (e.g. 0.015 for 1.5%).
        """
        if stats is None:
            return self.base_risk_pct

        kelly_risk = self._kelly_for_strategy(stats)
        adjusted = self._apply_anti_martingale(kelly_risk, stats)

        # Clamp to [min, max]
        final = max(self.min_risk_pct, min(self.max_risk_pct, adjusted))

        log_with_data(logger, "info", "Position sizing: risk computed", {
            "kelly_risk": round(kelly_risk, 6),
            "anti_martingale_adjusted": round(adjusted, 6),
            "final_clamped": round(final, 6),
            "win_rate": stats.win_rate,
            "win_streak": stats.recent_win_streak,
            "loss_streak": stats.recent_loss_streak,
        })

        return final

    def get_risk_pct_from_raw(
        self,
        win_rate: float,
        avg_win_r: float,
        avg_loss_r: float,
        recent_win_streak: int = 0,
        recent_loss_streak: int = 0,
    ) -> float:
        """Convenience wrapper that builds StrategyStats internally."""
        stats = StrategyStats(
            win_rate=win_rate,
            avg_win_r=avg_win_r,
            avg_loss_r=avg_loss_r,
            recent_win_streak=recent_win_streak,
            recent_loss_streak=recent_loss_streak,
        )
        return self.get_risk_pct(stats)
