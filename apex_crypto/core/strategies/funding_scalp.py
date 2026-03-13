"""Funding Rate Extreme Scalp strategy for the APEX Crypto Trading System.

Captures reversals that follow extreme perpetual-swap funding rates.
When funding is excessively positive (longs pay shorts) or negative
(shorts pay longs), the crowded side tends to unwind, creating a
predictable counter-move.

Entry triggers:
- SHORT when funding > +0.15% and a 5m bearish candle prints after peak.
- LONG  when funding < -0.15% and a 5m bullish candle prints after trough.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from apex_crypto.core.logging import get_logger
from apex_crypto.core.strategies.base import (
    BaseStrategy,
    SignalDirection,
    TradeSignal,
)

logger = get_logger("strategies.funding_scalp")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TIMEFRAME: str = "5m"

_FUNDING_THRESHOLD: float = 0.0015         # 0.15%
_FUNDING_EXTREME: float = 0.0025           # 0.25%
_STOP_PCT: float = 0.005                   # 0.5%
_TP_MIN_PCT: float = 0.005                 # 0.5%
_TP_MAX_PCT: float = 0.010                 # 1.0%

_BASE_SCORE: int = 55
_EXTREME_FUNDING_BONUS: int = 20
_REGIME_ALIGNMENT_BONUS: int = 10

# Regime alignment mappings
_BULLISH_REGIMES: set[str] = {"STRONG_BULL", "WEAK_BULL"}
_BEARISH_REGIMES: set[str] = {"STRONG_BEAR", "WEAK_BEAR"}


class FundingScalpStrategy(BaseStrategy):
    """Scalp reversals driven by extreme perpetual-swap funding rates.

    When the funding rate reaches extreme levels, the over-leveraged side
    is forced to unwind, creating a predictable counter-directional move.
    This strategy captures that move on 5m bars.

    Attributes:
        name: Strategy identifier.
        active_regimes: Empty list means active in all regimes.
        primary_timeframe: 5m bars for entry timing.
    """

    name: str = "funding_scalp"
    active_regimes: list[str] = []
    primary_timeframe: str = _TIMEFRAME
    confirmation_timeframe: str = "15m"
    entry_timeframe: str = "5m"

    def __init__(self, config: dict) -> None:
        """Initialize FundingScalpStrategy.

        Args:
            config: Strategy-specific configuration dictionary.
        """
        super().__init__(config)
        cfg = config.get("strategies", {}).get("funding_scalp", {})
        self.funding_threshold: float = cfg.get(
            "funding_threshold", _FUNDING_THRESHOLD
        )
        self.funding_extreme: float = cfg.get(
            "funding_extreme", _FUNDING_EXTREME
        )
        self.stop_pct: float = cfg.get("stop_pct", _STOP_PCT)
        self.base_score: int = cfg.get("base_score", _BASE_SCORE)

        # Track previous funding to detect peak/trough
        self._prev_funding: dict[str, float] = {}

        logger.info(
            "FundingScalpStrategy configured",
            extra={
                "funding_threshold": self.funding_threshold,
                "funding_extreme": self.funding_extreme,
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
        """Generate a funding-rate scalp signal.

        Args:
            symbol: Trading pair symbol.
            data: OHLCV DataFrames keyed by timeframe.
            indicators: Pre-computed indicator DataFrames keyed by timeframe.
            regime: Current market regime string.
            alt_data: Alternative data dict; expected key ``funding_rate``
                mapping symbol to current funding rate (as a decimal, e.g.
                0.001 for 0.1%).

        Returns:
            TradeSignal with direction and score, or NEUTRAL.
        """
        if not self.is_active(regime):
            return self._neutral_signal(symbol)

        # Require funding rate from alt_data
        funding_rate = self._extract_funding_rate(symbol, alt_data)
        if funding_rate is None:
            logger.debug("No funding rate available for %s", symbol)
            return self._neutral_signal(symbol)

        abs_funding = abs(funding_rate)
        if abs_funding < self.funding_threshold:
            self._prev_funding[symbol] = funding_rate
            return self._neutral_signal(symbol)

        # Require 5m OHLCV data
        tf = self.primary_timeframe
        if tf not in data or data[tf].empty:
            logger.warning("Missing %s data for %s", tf, symbol)
            return self._neutral_signal(symbol)

        df = data[tf]
        if len(df) < 2:
            return self._neutral_signal(symbol)

        # Detect peak/trough (funding started declining from extreme)
        prev_funding = self._prev_funding.get(symbol, 0.0)
        past_peak = funding_rate > 0 and funding_rate <= prev_funding
        past_trough = funding_rate < 0 and funding_rate >= prev_funding

        # Update tracked funding for next iteration
        self._prev_funding[symbol] = funding_rate

        # Latest candle analysis
        last_open = float(df["open"].iloc[-1])
        last_close = float(df["close"].iloc[-1])
        is_bearish = last_close < last_open
        is_bullish = last_close > last_open

        # Determine direction
        direction: Optional[SignalDirection] = None

        if funding_rate > self.funding_threshold and (past_peak or is_bearish):
            # High positive funding -> crowded longs -> SHORT
            if is_bearish:
                direction = SignalDirection.SHORT
        elif funding_rate < -self.funding_threshold and (past_trough or is_bullish):
            # High negative funding -> crowded shorts -> LONG
            if is_bullish:
                direction = SignalDirection.LONG

        if direction is None:
            return self._neutral_signal(symbol)

        # Score computation
        score = self.base_score
        if abs_funding >= self.funding_extreme:
            score += _EXTREME_FUNDING_BONUS
        if self._regime_aligns(regime, direction):
            score += _REGIME_ALIGNMENT_BONUS
        score = min(score, 100)

        # Entry, stop, targets
        entry_price = last_close
        if direction == SignalDirection.LONG:
            stop_loss = entry_price * (1 - self.stop_pct)
            tp1 = entry_price * (1 + _TP_MIN_PCT)
            tp2 = entry_price * (1 + _TP_MAX_PCT)
            tp3 = entry_price * (1 + _TP_MAX_PCT * 1.5)
        else:
            stop_loss = entry_price * (1 + self.stop_pct)
            tp1 = entry_price * (1 - _TP_MIN_PCT)
            tp2 = entry_price * (1 - _TP_MAX_PCT)
            tp3 = entry_price * (1 - _TP_MAX_PCT * 1.5)

        confidence = round(score / 100.0, 2)

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
                "signal_type": "entry",
                "funding_rate": round(funding_rate, 6),
                "prev_funding": round(prev_funding, 6),
                "past_peak": past_peak,
                "past_trough": past_trough,
                "candle_bullish": is_bullish,
                "candle_bearish": is_bearish,
                "regime_aligned": self._regime_aligns(regime, direction),
            },
        )

        logger.info(
            "Funding scalp signal generated",
            extra={
                "symbol": symbol,
                "direction": direction.value,
                "score": signal.score,
                "funding_rate": round(funding_rate, 6),
            },
        )
        return signal

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_funding_rate(
        symbol: str,
        alt_data: Optional[dict],
    ) -> Optional[float]:
        """Extract the current funding rate for *symbol* from alt_data.

        Tries several common key layouts:
        - alt_data["funding_rate"][symbol]
        - alt_data["funding"][symbol]
        - alt_data["funding_rates"][symbol]

        Args:
            symbol: Trading pair.
            alt_data: Alternative data dictionary.

        Returns:
            Funding rate as a float, or None if unavailable.
        """
        if alt_data is None:
            return None

        for key in ("funding_rate", "funding", "funding_rates"):
            container = alt_data.get(key)
            if isinstance(container, dict) and symbol in container:
                val = container[symbol]
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue

        # Flat key fallback
        if "funding_rate" in alt_data:
            try:
                return float(alt_data["funding_rate"])
            except (TypeError, ValueError):
                pass

        return None

    @staticmethod
    def _regime_aligns(regime: str, direction: SignalDirection) -> bool:
        """Check whether the current regime aligns with the trade direction.

        A SHORT during a bearish regime (or LONG during bullish) is
        considered aligned.

        Args:
            regime: Current market regime string.
            direction: Proposed trade direction.

        Returns:
            True if the regime supports the direction.
        """
        if direction == SignalDirection.LONG and regime in _BULLISH_REGIMES:
            return True
        if direction == SignalDirection.SHORT and regime in _BEARISH_REGIMES:
            return True
        return False
