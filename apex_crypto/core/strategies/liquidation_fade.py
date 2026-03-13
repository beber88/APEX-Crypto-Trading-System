"""Liquidation Cluster Fade strategy for the APEX Crypto Trading System.

Fades large liquidation cascades on the assumption that forced liquidations
cause price to overshoot fair value, creating a mean-reversion opportunity.

Entry triggers:
- LONG : a large SHORT liquidation cluster (>$5M in 5 min) drives price
         below equilibrium; we buy the overextension.
- SHORT: a large LONG liquidation cluster (>$5M in 5 min) drives price
         above equilibrium; we sell the overextension.

Stop: 1.0% against entry.  Target: 1.5-2.0% profit.
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

logger = get_logger("strategies.liquidation_fade")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TIMEFRAME: str = "5m"

_LIQUIDATION_THRESHOLD: float = 5_000_000.0    # $5M cluster
_LIQUIDATION_EXTREME: float = 10_000_000.0     # $10M extreme cluster
_STOP_PCT: float = 0.010                        # 1.0%
_TP_MIN_PCT: float = 0.015                      # 1.5%
_TP_MAX_PCT: float = 0.020                      # 2.0%

_BASE_SCORE: int = 55
_EXTREME_LIQUIDATION_BONUS: int = 15
_RSI_CONFIRMATION_BONUS: int = 10

_RSI_PERIOD: int = 14
_RSI_OVERSOLD: float = 30.0
_RSI_OVERBOUGHT: float = 70.0

# Volume look-back for z-score calculation
_VOLUME_LOOKBACK: int = 50


class LiquidationFadeStrategy(BaseStrategy):
    """Fade large liquidation cascades by trading the reversion.

    When a large cluster of liquidations forces price away from fair value
    the strategy enters in the opposite direction, expecting a snap-back.

    Attributes:
        name: Strategy identifier.
        active_regimes: Empty list means active in all regimes.
        primary_timeframe: 5m bars for entry timing.
    """

    name: str = "liquidation_fade"
    active_regimes: list[str] = []
    primary_timeframe: str = _TIMEFRAME
    confirmation_timeframe: str = "15m"
    entry_timeframe: str = "5m"

    def __init__(self, config: dict) -> None:
        """Initialize LiquidationFadeStrategy.

        Args:
            config: Strategy-specific configuration dictionary.
        """
        super().__init__(config)
        cfg = config.get("strategies", {}).get("liquidation_fade", {})
        self.liquidation_threshold: float = cfg.get(
            "liquidation_threshold", _LIQUIDATION_THRESHOLD
        )
        self.liquidation_extreme: float = cfg.get(
            "liquidation_extreme", _LIQUIDATION_EXTREME
        )
        self.stop_pct: float = cfg.get("stop_pct", _STOP_PCT)
        self.tp_min_pct: float = cfg.get("tp_min_pct", _TP_MIN_PCT)
        self.tp_max_pct: float = cfg.get("tp_max_pct", _TP_MAX_PCT)
        self.base_score: int = cfg.get("base_score", _BASE_SCORE)
        self.rsi_oversold: float = cfg.get("rsi_oversold", _RSI_OVERSOLD)
        self.rsi_overbought: float = cfg.get("rsi_overbought", _RSI_OVERBOUGHT)

        logger.info(
            "LiquidationFadeStrategy configured",
            extra={
                "liquidation_threshold": self.liquidation_threshold,
                "liquidation_extreme": self.liquidation_extreme,
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
        """Generate a liquidation-fade signal.

        Args:
            symbol: Trading pair symbol.
            data: OHLCV DataFrames keyed by timeframe.
            indicators: Pre-computed indicator DataFrames keyed by timeframe.
            regime: Current market regime string.
            alt_data: Alternative data dict; expected keys:
                - ``liquidations`` or ``liquidation_data``: dict mapping
                  symbol to a dict with ``long_liquidations`` and
                  ``short_liquidations`` (USD values over the last 5 min).

        Returns:
            TradeSignal with direction and score, or NEUTRAL.
        """
        if not self.is_active(regime):
            return self._neutral_signal(symbol)

        # Extract liquidation data from alt_data
        long_liqs, short_liqs = self._extract_liquidation_data(symbol, alt_data)
        if long_liqs is None and short_liqs is None:
            logger.debug("No liquidation data available for %s", symbol)
            return self._neutral_signal(symbol)

        long_liqs = long_liqs or 0.0
        short_liqs = short_liqs or 0.0

        # Determine which side had a large liquidation cluster
        has_long_cluster = long_liqs >= self.liquidation_threshold
        has_short_cluster = short_liqs >= self.liquidation_threshold

        if not has_long_cluster and not has_short_cluster:
            return self._neutral_signal(symbol)

        # Require OHLCV data for price context
        tf = self.primary_timeframe
        if tf not in data or data[tf].empty:
            logger.warning("Missing %s data for %s", tf, symbol)
            return self._neutral_signal(symbol)

        df = data[tf]
        if len(df) < _VOLUME_LOOKBACK:
            logger.debug("Insufficient data length for %s", symbol)
            return self._neutral_signal(symbol)

        ind = indicators.get(tf)

        # Verify price overextension with recent candle range
        close = float(df["close"].iloc[-1])
        recent_high = float(df["high"].iloc[-5:].max())
        recent_low = float(df["low"].iloc[-5:].min())
        recent_range = recent_high - recent_low
        if recent_range <= 0:
            return self._neutral_signal(symbol)

        # Determine direction
        direction: Optional[SignalDirection] = None
        liq_volume: float = 0.0

        if has_long_cluster and close > recent_low + recent_range * 0.7:
            # Large LONG liquidation cluster -> price overextended up -> SHORT
            direction = SignalDirection.SHORT
            liq_volume = long_liqs
        elif has_short_cluster and close < recent_high - recent_range * 0.7:
            # Large SHORT liquidation cluster -> price overextended down -> LONG
            direction = SignalDirection.LONG
            liq_volume = short_liqs

        if direction is None:
            return self._neutral_signal(symbol)

        # Score computation
        score = self.base_score

        # Extreme liquidation volume bonus
        if liq_volume >= self.liquidation_extreme:
            score += _EXTREME_LIQUIDATION_BONUS

        # RSI confirmation bonus
        rsi = self._get_rsi(df, ind)
        rsi_confirms = False
        if rsi is not None:
            if direction == SignalDirection.LONG and rsi < self.rsi_oversold:
                score += _RSI_CONFIRMATION_BONUS
                rsi_confirms = True
            elif direction == SignalDirection.SHORT and rsi > self.rsi_overbought:
                score += _RSI_CONFIRMATION_BONUS
                rsi_confirms = True

        score = min(score, 100)

        # Entry, stop, targets
        entry_price = close
        if direction == SignalDirection.LONG:
            stop_loss = entry_price * (1 - self.stop_pct)
            tp1 = entry_price * (1 + self.tp_min_pct)
            tp2 = entry_price * (1 + self.tp_max_pct)
            tp3 = entry_price * (1 + self.tp_max_pct * 1.5)
        else:
            stop_loss = entry_price * (1 + self.stop_pct)
            tp1 = entry_price * (1 - self.tp_min_pct)
            tp2 = entry_price * (1 - self.tp_max_pct)
            tp3 = entry_price * (1 - self.tp_max_pct * 1.5)

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
                "long_liquidations_usd": round(long_liqs, 2),
                "short_liquidations_usd": round(short_liqs, 2),
                "liq_volume": round(liq_volume, 2),
                "rsi_confirms": rsi_confirms,
                "rsi": round(rsi, 2) if rsi is not None else None,
                "recent_high": round(recent_high, 4),
                "recent_low": round(recent_low, 4),
            },
        )

        logger.info(
            "Liquidation fade signal generated",
            extra={
                "symbol": symbol,
                "direction": direction.value,
                "score": signal.score,
                "liq_volume_usd": round(liq_volume, 2),
                "rsi_confirms": rsi_confirms,
            },
        )
        return signal

    # ------------------------------------------------------------------
    # Liquidation data extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_liquidation_data(
        symbol: str,
        alt_data: Optional[dict],
    ) -> tuple[Optional[float], Optional[float]]:
        """Extract long and short liquidation volumes from alt_data.

        Tries several common key layouts:
        - alt_data["liquidations"][symbol]
        - alt_data["liquidation_data"][symbol]

        Each symbol entry is expected to contain ``long_liquidations``
        and ``short_liquidations`` keys with USD values.

        Args:
            symbol: Trading pair.
            alt_data: Alternative data dictionary.

        Returns:
            Tuple of (long_liquidations_usd, short_liquidations_usd),
            either or both may be None.
        """
        if alt_data is None:
            return None, None

        for key in ("liquidations", "liquidation_data", "liq_data"):
            container = alt_data.get(key)
            if isinstance(container, dict):
                sym_data = container.get(symbol)
                if isinstance(sym_data, dict):
                    try:
                        long_liqs = float(sym_data.get("long_liquidations", 0))
                        short_liqs = float(sym_data.get("short_liquidations", 0))
                        return long_liqs, short_liqs
                    except (TypeError, ValueError):
                        continue

        return None, None

    # ------------------------------------------------------------------
    # RSI helper
    # ------------------------------------------------------------------

    @staticmethod
    def _get_rsi(
        df: pd.DataFrame,
        ind: Optional[pd.DataFrame],
    ) -> Optional[float]:
        """Retrieve or compute RSI(14).

        First checks pre-computed indicators for ``rsi_14``.  Falls back to
        manual computation from close prices.

        Args:
            df: OHLCV DataFrame.
            ind: Optional pre-computed indicator DataFrame.

        Returns:
            RSI(14) value, or None on failure.
        """
        # Try pre-computed
        if ind is not None and not ind.empty and "rsi_14" in ind.columns:
            val = ind["rsi_14"].iloc[-1]
            if not pd.isna(val):
                return float(val)

        # Manual RSI(14)
        try:
            close = df["close"]
            if len(close) < _RSI_PERIOD + 1:
                return None
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.ewm(span=_RSI_PERIOD, min_periods=_RSI_PERIOD).mean()
            avg_loss = loss.ewm(span=_RSI_PERIOD, min_periods=_RSI_PERIOD).mean()
            last_loss = float(avg_loss.iloc[-1])
            if last_loss == 0:
                return 100.0
            rs = float(avg_gain.iloc[-1]) / last_loss
            return 100.0 - (100.0 / (1.0 + rs))
        except Exception:
            logger.exception("Error computing RSI(14)")
            return None
