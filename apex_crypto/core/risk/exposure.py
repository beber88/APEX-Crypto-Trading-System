"""Adaptive leverage and exposure controller.

Adjusts risk multipliers, max positions, and leverage caps based on
the current market regime and volatility conditions.
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


# Default regime → risk multiplier mapping
_DEFAULT_REGIME_MULTIPLIERS: dict[str, float] = {
    "STRONG_BULL": 1.5,
    "WEAK_BULL": 1.2,
    "RANGING": 1.0,
    "WEAK_BEAR": 0.7,
    "STRONG_BEAR": 0.5,
    "HIGH_VOL": 0.5,
    "CHAOS": 0.25,
}

# Default regime → max leverage mapping
_DEFAULT_REGIME_LEVERAGE: dict[str, float] = {
    "STRONG_BULL": 3.0,
    "WEAK_BULL": 2.0,
    "RANGING": 2.0,
    "WEAK_BEAR": 1.5,
    "STRONG_BEAR": 1.0,
    "HIGH_VOL": 1.0,
    "CHAOS": 1.0,
}


class ExposureController:
    """Controls portfolio exposure based on market regime.

    Provides:
    - Risk multipliers that scale position sizes per regime
    - Dynamic max position counts (more in bull, fewer in bear)
    - Regime-aware leverage caps
    - Overall portfolio leverage ceiling

    Attributes:
        max_positions_base: Default max open positions.
        max_positions_aggressive: Max positions in bullish regimes.
        max_portfolio_leverage: Absolute leverage ceiling.
        regime_multipliers: Regime → risk multiplier mapping.
        regime_leverage: Regime → max leverage mapping.
    """

    def __init__(self, config: dict) -> None:
        self.max_positions_base: int = config.get("max_positions_base", 8)
        self.max_positions_aggressive: int = config.get("max_positions_aggressive", 15)
        self.max_portfolio_leverage: float = config.get("max_portfolio_leverage", 3.0)

        # Allow overriding regime multipliers from config
        self.regime_multipliers: dict[str, float] = {
            **_DEFAULT_REGIME_MULTIPLIERS,
            **config.get("regime_multipliers", {}),
        }
        self.regime_leverage: dict[str, float] = {
            **_DEFAULT_REGIME_LEVERAGE,
            **config.get("regime_leverage", {}),
        }

        log_with_data(logger, "info", "ExposureController initialised", {
            "max_positions_base": self.max_positions_base,
            "max_positions_aggressive": self.max_positions_aggressive,
            "max_portfolio_leverage": self.max_portfolio_leverage,
            "regime_count": len(self.regime_multipliers),
        })

    # ------------------------------------------------------------------
    # Risk multiplier
    # ------------------------------------------------------------------

    def get_risk_multiplier(self, regime: str) -> float:
        """Return the position-size multiplier for the given regime.

        Bull markets → larger positions (1.2–1.5x).
        Bear/chaos → smaller positions (0.25–0.7x).
        """
        regime_upper = regime.upper()
        multiplier = self.regime_multipliers.get(regime_upper, 1.0)

        log_with_data(logger, "debug", "Regime risk multiplier", {
            "regime": regime_upper,
            "multiplier": multiplier,
        })

        return multiplier

    # ------------------------------------------------------------------
    # Max positions
    # ------------------------------------------------------------------

    def get_max_positions(self, regime: str) -> int:
        """Return the maximum number of open positions for the regime."""
        regime_upper = regime.upper()

        if regime_upper in ("STRONG_BULL", "WEAK_BULL"):
            max_pos = self.max_positions_aggressive
        elif regime_upper in ("STRONG_BEAR", "CHAOS"):
            max_pos = max(3, self.max_positions_base // 2)
        else:
            max_pos = self.max_positions_base

        log_with_data(logger, "debug", "Max positions for regime", {
            "regime": regime_upper,
            "max_positions": max_pos,
        })

        return max_pos

    # ------------------------------------------------------------------
    # Leverage
    # ------------------------------------------------------------------

    def get_max_leverage(self, regime: str) -> float:
        """Return the maximum allowed leverage for the regime."""
        regime_upper = regime.upper()
        regime_lev = self.regime_leverage.get(regime_upper, 1.0)
        clamped = min(regime_lev, self.max_portfolio_leverage)

        log_with_data(logger, "debug", "Max leverage for regime", {
            "regime": regime_upper,
            "regime_leverage": regime_lev,
            "clamped": clamped,
        })

        return clamped

    def clamp_leverage(self, requested_leverage: float) -> float:
        """Clamp requested leverage to the portfolio ceiling."""
        return min(requested_leverage, self.max_portfolio_leverage)

    # ------------------------------------------------------------------
    # Composite query
    # ------------------------------------------------------------------

    def get_exposure_params(self, regime: str) -> dict[str, Any]:
        """Return all exposure parameters for the current regime.

        Returns:
            Dictionary with risk_multiplier, max_positions, and max_leverage.
        """
        result = {
            "regime": regime.upper(),
            "risk_multiplier": self.get_risk_multiplier(regime),
            "max_positions": self.get_max_positions(regime),
            "max_leverage": self.get_max_leverage(regime),
        }

        log_with_data(logger, "info", "Exposure parameters computed", result)

        return result
