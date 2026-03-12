"""Funding rate strategy for the APEX Crypto Trading System.

Exploits extreme perpetual funding rates as mean-reversion signals.
Negative funding (shorts paying longs) suggests squeeze risk and favours
longs; excessively positive funding (longs overpaying) signals crowding
and favours shorts.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from apex_crypto.core.logging import get_logger, log_with_data
from apex_crypto.core.strategies.base import (
    BaseStrategy,
    SignalDirection,
    TradeSignal,
)

logger = get_logger("strategies.funding_rate")

# Funding rate thresholds (expressed as percentages, e.g. 0.05 → 0.05%)
_LONG_THRESHOLD: float = -0.05   # funding < -0.05% → long bias
_SHORT_THRESHOLD: float = 0.10   # funding > 0.10% → short bias
_EXTREME_MULTIPLIER: float = 2.0  # 2x threshold counts as extreme

# Regime alignment maps
_BULLISH_REGIMES: set[str] = {"STRONG_BULL", "WEAK_BULL"}
_BEARISH_REGIMES: set[str] = {"STRONG_BEAR", "WEAK_BEAR"}


class FundingRateStrategy(BaseStrategy):
    """Trade perpetual contracts based on extreme funding rates.

    When funding is deeply negative, shorts are paying longs, indicating
    a crowded-short market ripe for a squeeze. When funding is excessively
    positive, longs are overleveraged and a correction becomes likely.

    The strategy only triggers when the prevailing market regime *agrees*
    with the funding-implied direction, reducing false signals.

    Attributes:
        name: Strategy identifier.
        active_regimes: Market regimes where this strategy operates.
        primary_timeframe: Timeframe used for OHLCV context.
    """

    name: str = "funding_rate"
    active_regimes: list[str] = [
        "STRONG_BULL",
        "WEAK_BULL",
        "RANGING",
        "WEAK_BEAR",
        "STRONG_BEAR",
    ]
    primary_timeframe: str = "4h"

    def __init__(self, config: dict) -> None:
        """Initialize the funding rate strategy.

        Args:
            config: Strategy-specific configuration dict.
        """
        super().__init__(config)

        fr_cfg: dict = config.get("strategies", {}).get("funding_rate", {})
        self._long_threshold: float = fr_cfg.get("long_threshold", _LONG_THRESHOLD)
        self._short_threshold: float = fr_cfg.get("short_threshold", _SHORT_THRESHOLD)
        self._extreme_mult: float = fr_cfg.get("extreme_multiplier", _EXTREME_MULTIPLIER)

        logger.info(
            "FundingRateStrategy configured",
            extra={
                "data": {
                    "long_threshold": self._long_threshold,
                    "short_threshold": self._short_threshold,
                    "extreme_multiplier": self._extreme_mult,
                }
            },
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_signal(
        self,
        symbol: str,
        data: dict[str, pd.DataFrame],
        indicators: dict[str, pd.DataFrame],
        regime: str,
        alt_data: Optional[dict] = None,
    ) -> TradeSignal:
        """Generate a funding-rate-based trading signal.

        Args:
            symbol: Trading pair symbol (e.g. ``'BTC/USDT'``).
            data: OHLCV DataFrames keyed by timeframe.
            indicators: Pre-computed indicator DataFrames keyed by timeframe.
                Expected columns on the 4h frame: ``atr``.
            regime: Current market regime string.
            alt_data: Alternative data dict.  Required keys:
                ``funding_rate`` (float, percentage),
                ``open_interest`` (float), and optionally
                ``open_interest_change`` (float, percentage).

        Returns:
            TradeSignal with score of 0 (neutral) when conditions are not
            met, or 50-90 when a valid funding setup is detected.
        """
        # ----- Gate checks ------------------------------------------------
        if not self.is_active(regime):
            logger.debug("Regime %s not active for funding_rate strategy", regime)
            return self._neutral_signal(symbol)

        if alt_data is None:
            logger.debug("No alt_data provided for %s — cannot evaluate funding", symbol)
            return self._neutral_signal(symbol)

        funding_rate: Optional[float] = alt_data.get("funding_rate")
        if funding_rate is None:
            logger.debug("funding_rate missing in alt_data for %s", symbol)
            return self._neutral_signal(symbol)

        # ----- Determine directional bias ---------------------------------
        direction: Optional[SignalDirection] = None
        is_extreme: bool = False

        if funding_rate < self._long_threshold:
            direction = SignalDirection.LONG
            is_extreme = funding_rate < self._long_threshold * self._extreme_mult
        elif funding_rate > self._short_threshold:
            direction = SignalDirection.SHORT
            is_extreme = funding_rate > self._short_threshold * self._extreme_mult

        if direction is None:
            logger.debug(
                "Funding rate %.4f%% within normal range for %s",
                funding_rate,
                symbol,
            )
            return self._neutral_signal(symbol)

        # ----- Regime alignment check -------------------------------------
        regime_aligned: bool = self._regime_aligns(direction, regime)

        if not regime_aligned and regime != "RANGING":
            logger.info(
                "Funding signal for %s (%s) conflicts with regime %s — skipping",
                symbol,
                direction.value,
                regime,
            )
            return self._neutral_signal(symbol)

        # ----- OI divergence bonus ----------------------------------------
        oi_divergence: bool = False
        oi_change: Optional[float] = alt_data.get("open_interest_change")
        if oi_change is not None:
            # OI divergence: funding extreme but OI moving against the crowd
            if direction == SignalDirection.LONG and oi_change < 0:
                oi_divergence = True
            elif direction == SignalDirection.SHORT and oi_change > 0:
                oi_divergence = True

        # ----- Score ------------------------------------------------------
        score: int = 50

        if is_extreme:
            score += 20

        if regime_aligned:
            score += 10

        if oi_divergence:
            score += 10

        score = min(score, 90)

        # ----- Price levels -----------------------------------------------
        df = data.get(self.primary_timeframe)
        ind = indicators.get(self.primary_timeframe)

        if df is None or df.empty:
            logger.warning("No 4h OHLCV data for %s", symbol)
            return self._neutral_signal(symbol)

        entry_price: float = float(df["close"].iloc[-1])

        atr: float = 0.0
        if ind is not None and not ind.empty and "atr" in ind.columns:
            atr = float(ind["atr"].iloc[-1])

        if atr > 0:
            stop_loss = self.compute_stop_loss(
                entry_price, direction, atr, atr_multiplier=1.5, max_stop_pct=0.02
            )
        else:
            # Fallback: percentage-based stop
            pct = 0.02
            if direction == SignalDirection.LONG:
                stop_loss = entry_price * (1 - pct)
            else:
                stop_loss = entry_price * (1 + pct)

        tp1, tp2, tp3 = self.compute_take_profits(
            entry_price, stop_loss, direction, tp1_r=1.5, tp2_r=2.5, tp3_r=4.0
        )

        confidence: float = round(score / 100.0, 2)

        signal = TradeSignal(
            symbol=symbol,
            direction=direction,
            score=score if direction == SignalDirection.LONG else -score,
            strategy=self.name,
            timeframe=self.primary_timeframe,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            confidence=confidence,
            metadata={
                "funding_rate": funding_rate,
                "is_extreme_funding": is_extreme,
                "regime_aligned": regime_aligned,
                "oi_divergence": oi_divergence,
                "oi_change": oi_change,
                "atr": round(atr, 4) if atr else None,
            },
        )

        log_with_data(
            logger,
            "info",
            f"Funding rate signal generated for {symbol}",
            data=signal.to_dict(),
        )

        return signal

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _regime_aligns(direction: SignalDirection, regime: str) -> bool:
        """Check whether the regime agrees with the signal direction.

        Args:
            direction: Proposed signal direction.
            regime: Current market regime string.

        Returns:
            True if the regime supports the proposed direction.
        """
        if direction == SignalDirection.LONG:
            return regime in _BULLISH_REGIMES or regime == "RANGING"
        if direction == SignalDirection.SHORT:
            return regime in _BEARISH_REGIMES or regime == "RANGING"
        return False
