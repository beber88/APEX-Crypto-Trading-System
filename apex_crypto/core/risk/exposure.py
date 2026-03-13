"""Adaptive leverage and exposure controller.

Adjusts risk multipliers, max positions, and leverage caps based on
the current market regime, volatility conditions, and ultra mode setting.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from apex_crypto.core.logging import get_logger, log_with_data

logger = get_logger("risk.exposure")


class MarketRegime(str, Enum):
    """Market regime classifications from the regime classifier."""

    STRONG_BULL = "STRONG_BULL"
    WEAK_BULL = "WEAK_BULL"
    RANGING = "RANGING"
    WEAK_BEAR = "WEAK_BEAR"
    STRONG_BEAR = "STRONG_BEAR"
    HIGH_VOL = "HIGH_VOL"
    CHAOS = "CHAOS"


# Normal mode regime mappings
_DEFAULT_REGIME_MULTIPLIERS: dict[str, float] = {
    "STRONG_BULL": 1.5,
    "WEAK_BULL": 1.2,
    "RANGING": 1.0,
    "WEAK_BEAR": 0.7,
    "STRONG_BEAR": 0.5,
    "HIGH_VOL": 0.5,
    "CHAOS": 0.25,
}

_DEFAULT_REGIME_LEVERAGE: dict[str, float] = {
    "STRONG_BULL": 3.0,
    "WEAK_BULL": 2.0,
    "RANGING": 2.0,
    "WEAK_BEAR": 1.5,
    "STRONG_BEAR": 1.0,
    "HIGH_VOL": 1.0,
    "CHAOS": 1.0,
}

# Ultra mode regime mappings — much more aggressive
_ULTRA_REGIME_MULTIPLIERS: dict[str, float] = {
    "STRONG_BULL": 3.0,   # 2x normal
    "WEAK_BULL": 2.0,
    "RANGING": 1.2,
    "WEAK_BEAR": 0.3,     # still cautious on bear
    "STRONG_BEAR": 0.2,
    "HIGH_VOL": 1.8,      # ultra loves vol
    "CHAOS": 0.3,
}

_ULTRA_REGIME_LEVERAGE: dict[str, float] = {
    "STRONG_BULL": 8.0,
    "WEAK_BULL": 5.0,
    "RANGING": 3.0,
    "WEAK_BEAR": 2.0,
    "STRONG_BEAR": 1.0,
    "HIGH_VOL": 5.0,
    "CHAOS": 1.0,
}


class ExposureController:
    """Controls portfolio exposure based on market regime and ultra mode.

    Provides:
    - Risk multipliers that scale position sizes per regime
    - Dynamic max position counts (more in bull, fewer in bear)
    - Regime-aware leverage caps
    - Ultra mode toggle for aggressive trading
    - Asset eligibility checks for ultra mode

    Attributes:
        max_positions_base: Default max open positions.
        max_positions_aggressive: Max positions in bullish regimes.
        max_positions_ultra: Max positions in ultra mode.
        max_portfolio_leverage: Absolute leverage ceiling.
        ultra_mode: Whether ultra-aggressive mode is active.
    """

    def __init__(self, config: dict) -> None:
        self.max_positions_base: int = config.get("max_positions_base", 12)
        self.max_positions_aggressive: int = config.get("max_positions_aggressive", 15)
        self.max_positions_ultra: int = config.get("max_positions_ultra", 25)
        self.max_portfolio_leverage: float = config.get("max_portfolio_leverage", 3.0)
        self.max_portfolio_leverage_ultra: float = config.get("max_portfolio_leverage_ultra", 8.0)
        self.ultra_mode: bool = config.get("ultra_mode", False)

        # Ultra mode asset eligibility thresholds
        self.ultra_min_volume_24h: float = config.get("ultra_min_volume_24h", 5_000_000)
        self.ultra_max_spread_pct: float = config.get("ultra_max_spread_pct", 0.08)
        self.ultra_min_signal_score: float = config.get("ultra_min_signal_score", 50)

        # Ultra drawdown kill switch (tighter than normal 12%)
        self.ultra_max_drawdown_pct: float = config.get("ultra_max_drawdown_pct", 8.0)

        # Build regime mappings — allow config overrides
        self.regime_multipliers: dict[str, float] = {
            **_DEFAULT_REGIME_MULTIPLIERS,
            **config.get("regime_multipliers", {}),
        }
        self.regime_leverage: dict[str, float] = {
            **_DEFAULT_REGIME_LEVERAGE,
            **config.get("regime_leverage", {}),
        }
        self.ultra_regime_multipliers: dict[str, float] = {
            **_ULTRA_REGIME_MULTIPLIERS,
            **config.get("ultra_regime_multipliers", {}),
        }
        self.ultra_regime_leverage: dict[str, float] = {
            **_ULTRA_REGIME_LEVERAGE,
            **config.get("ultra_regime_leverage", {}),
        }

        log_with_data(logger, "info", "ExposureController initialised", {
            "max_positions_base": self.max_positions_base,
            "max_positions_aggressive": self.max_positions_aggressive,
            "max_positions_ultra": self.max_positions_ultra,
            "max_portfolio_leverage": self.max_portfolio_leverage,
            "ultra_mode": self.ultra_mode,
        })

    # ------------------------------------------------------------------
    # Ultra mode toggle
    # ------------------------------------------------------------------

    def set_ultra_mode(self, enabled: bool) -> None:
        """Enable or disable ultra-aggressive trading mode."""
        old = self.ultra_mode
        self.ultra_mode = enabled
        log_with_data(logger, "warning", "ULTRA MODE toggled", {
            "old": old,
            "new": enabled,
        })

    # ------------------------------------------------------------------
    # Risk multiplier
    # ------------------------------------------------------------------

    def get_risk_multiplier(self, regime: str) -> float:
        """Return the position-size multiplier for the given regime.

        In ultra mode, uses much more aggressive multipliers.
        """
        regime_upper = regime.upper()

        if self.ultra_mode:
            multiplier = self.ultra_regime_multipliers.get(regime_upper, 1.0)
        else:
            multiplier = self.regime_multipliers.get(regime_upper, 1.0)

        log_with_data(logger, "debug", "Regime risk multiplier", {
            "regime": regime_upper,
            "multiplier": multiplier,
            "ultra_mode": self.ultra_mode,
        })

        return multiplier

    # ------------------------------------------------------------------
    # Max positions
    # ------------------------------------------------------------------

    def get_max_positions(self, regime: str) -> int:
        """Return the maximum number of open positions for the regime."""
        regime_upper = regime.upper()

        if self.ultra_mode:
            if regime_upper in ("STRONG_BULL", "WEAK_BULL", "HIGH_VOL"):
                max_pos = self.max_positions_ultra
            elif regime_upper in ("STRONG_BEAR", "CHAOS"):
                max_pos = max(3, self.max_positions_base // 2)
            else:
                max_pos = self.max_positions_aggressive
        else:
            if regime_upper in ("STRONG_BULL", "WEAK_BULL"):
                max_pos = self.max_positions_aggressive
            elif regime_upper in ("STRONG_BEAR", "CHAOS"):
                max_pos = max(3, self.max_positions_base // 2)
            else:
                max_pos = self.max_positions_base

        log_with_data(logger, "debug", "Max positions for regime", {
            "regime": regime_upper,
            "max_positions": max_pos,
            "ultra_mode": self.ultra_mode,
        })

        return max_pos

    # ------------------------------------------------------------------
    # Leverage
    # ------------------------------------------------------------------

    def get_max_leverage(self, regime: str) -> float:
        """Return the maximum allowed leverage for the regime."""
        regime_upper = regime.upper()

        if self.ultra_mode:
            regime_lev = self.ultra_regime_leverage.get(regime_upper, 1.0)
            ceiling = self.max_portfolio_leverage_ultra
        else:
            regime_lev = self.regime_leverage.get(regime_upper, 1.0)
            ceiling = self.max_portfolio_leverage

        clamped = min(regime_lev, ceiling)

        log_with_data(logger, "debug", "Max leverage for regime", {
            "regime": regime_upper,
            "regime_leverage": regime_lev,
            "clamped": clamped,
            "ultra_mode": self.ultra_mode,
        })

        return clamped

    def clamp_leverage(self, requested_leverage: float) -> float:
        """Clamp requested leverage to the portfolio ceiling."""
        ceiling = self.max_portfolio_leverage_ultra if self.ultra_mode else self.max_portfolio_leverage
        return min(requested_leverage, ceiling)

    # ------------------------------------------------------------------
    # Ultra mode asset eligibility
    # ------------------------------------------------------------------

    def is_asset_eligible_ultra(
        self,
        volume_24h: float,
        spread_pct: float,
        signal_score: float,
    ) -> bool:
        """Check if an asset meets ultra mode trading criteria.

        Args:
            volume_24h: 24-hour trading volume in USD.
            spread_pct: Current bid-ask spread as percentage.
            signal_score: Aggregated signal score for this asset.

        Returns:
            True if the asset is eligible for ultra mode trading.
        """
        eligible = (
            volume_24h >= self.ultra_min_volume_24h
            and spread_pct <= self.ultra_max_spread_pct
            and signal_score >= self.ultra_min_signal_score
        )

        if eligible:
            log_with_data(logger, "debug", "Asset eligible for ultra mode", {
                "volume_24h": volume_24h,
                "spread_pct": spread_pct,
                "signal_score": signal_score,
            })

        return eligible

    # ------------------------------------------------------------------
    # Ultra safety checks
    # ------------------------------------------------------------------

    def check_ultra_safeties(
        self,
        current_drawdown_pct: float,
        open_position_count: int,
    ) -> dict[str, Any]:
        """Run ultra mode safety checks.

        Returns:
            Dictionary with emergency_close (bool), pause_trading (bool),
            disable_ultra (bool), and reason (str).
        """
        result: dict[str, Any] = {
            "emergency_close": False,
            "pause_trading": False,
            "disable_ultra": False,
            "reason": "",
        }

        if not self.ultra_mode:
            return result

        # Tighter drawdown limit in ultra mode
        if current_drawdown_pct > self.ultra_max_drawdown_pct:
            result["emergency_close"] = True
            result["disable_ultra"] = True
            result["reason"] = (
                f"ULTRA DD LIMIT: {current_drawdown_pct:.1f}% exceeds "
                f"{self.ultra_max_drawdown_pct:.1f}% — emergency close + ultra OFF"
            )
            log_with_data(logger, "critical", result["reason"], {
                "drawdown_pct": current_drawdown_pct,
                "limit_pct": self.ultra_max_drawdown_pct,
            })
            return result

        # Max positions check
        if open_position_count > self.max_positions_ultra:
            result["pause_trading"] = True
            result["reason"] = (
                f"ULTRA MAX POSITIONS: {open_position_count} > {self.max_positions_ultra}"
            )
            log_with_data(logger, "warning", result["reason"])

        return result

    # ------------------------------------------------------------------
    # Composite query
    # ------------------------------------------------------------------

    def get_exposure_params(self, regime: str) -> dict[str, Any]:
        """Return all exposure parameters for the current regime."""
        result = {
            "regime": regime.upper(),
            "risk_multiplier": self.get_risk_multiplier(regime),
            "max_positions": self.get_max_positions(regime),
            "max_leverage": self.get_max_leverage(regime),
            "ultra_mode": self.ultra_mode,
        }

        log_with_data(logger, "info", "Exposure parameters computed", result)

        return result
