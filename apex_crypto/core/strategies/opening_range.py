"""Opening Range Breakout strategy for the APEX Crypto Trading System.

Defines the first 30-minute range after 00:00 UTC (the crypto "open")
and trades breakouts above the range high or below the range low when
confirmed by elevated volume.

Entry triggers:
- LONG : price breaks above the 00:00-00:30 UTC range high with
         volume z-score > 2.0.
- SHORT: price breaks below the 00:00-00:30 UTC range low with
         volume z-score > 2.0.

Stop: opposite side of the opening range.
Target: 2x the range height.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

from apex_crypto.core.logging import get_logger
from apex_crypto.core.strategies.base import (
    BaseStrategy,
    SignalDirection,
    TradeSignal,
)

logger = get_logger("strategies.opening_range")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TIMEFRAME: str = "5m"

_VOLUME_ZSCORE_THRESHOLD: float = 2.0
_VOLUME_LOOKBACK: int = 50

_BASE_SCORE: int = 55
_STRONG_VOLUME_BONUS: int = 15
_TREND_ALIGNMENT_BONUS: int = 10

_RANGE_BARS: int = 6  # 6 x 5m = 30 minutes (00:00-00:30 UTC)

# Regime alignment mappings
_BULLISH_REGIMES: set[str] = {"STRONG_BULL", "WEAK_BULL"}
_BEARISH_REGIMES: set[str] = {"STRONG_BEAR", "WEAK_BEAR"}


class OpeningRangeBreakoutStrategy(BaseStrategy):
    """Opening Range Breakout on the crypto 00:00 UTC session.

    At 00:00 UTC the first 30 minutes of trading define a range (high
    and low).  When price subsequently breaks out of this range with
    strong volume, the strategy enters in the breakout direction.

    Attributes:
        name: Strategy identifier.
        active_regimes: Empty list means active in all regimes.
        primary_timeframe: 5m bars.
    """

    name: str = "opening_range_breakout"
    active_regimes: list[str] = []
    primary_timeframe: str = _TIMEFRAME
    confirmation_timeframe: str = "15m"
    entry_timeframe: str = "5m"

    def __init__(self, config: dict) -> None:
        """Initialize OpeningRangeBreakoutStrategy.

        Args:
            config: Strategy-specific configuration dictionary.
        """
        super().__init__(config)
        cfg = config.get("strategies", {}).get("opening_range_breakout", {})
        self.volume_zscore_threshold: float = cfg.get(
            "volume_zscore_threshold", _VOLUME_ZSCORE_THRESHOLD
        )
        self.volume_lookback: int = cfg.get("volume_lookback", _VOLUME_LOOKBACK)
        self.base_score: int = cfg.get("base_score", _BASE_SCORE)
        self.range_bars: int = cfg.get("range_bars", _RANGE_BARS)

        # Cache the detected opening range per symbol per day
        self._range_cache: dict[str, dict[str, Any]] = {}

        logger.info(
            "OpeningRangeBreakoutStrategy configured",
            extra={
                "volume_zscore_threshold": self.volume_zscore_threshold,
                "range_bars": self.range_bars,
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
        """Generate an opening-range breakout signal.

        Args:
            symbol: Trading pair symbol.
            data: OHLCV DataFrames keyed by timeframe.
            indicators: Pre-computed indicator DataFrames keyed by timeframe.
            regime: Current market regime string.
            alt_data: Optional alternative data (unused).

        Returns:
            TradeSignal with direction and score, or NEUTRAL.
        """
        if not self.is_active(regime):
            return self._neutral_signal(symbol)

        tf = self.primary_timeframe
        if tf not in data or data[tf].empty:
            logger.warning("Missing %s data for %s", tf, symbol)
            return self._neutral_signal(symbol)

        df = data[tf]
        if len(df) < self.volume_lookback:
            logger.debug("Insufficient data for %s", symbol)
            return self._neutral_signal(symbol)

        # Compute or retrieve the opening range
        range_high, range_low = self._get_opening_range(symbol, df)
        if range_high is None or range_low is None:
            logger.debug("Could not determine opening range for %s", symbol)
            return self._neutral_signal(symbol)

        range_height = range_high - range_low
        if range_height <= 0:
            return self._neutral_signal(symbol)

        close = float(df["close"].iloc[-1])

        # Check for breakout
        broke_high = close > range_high
        broke_low = close < range_low

        if not broke_high and not broke_low:
            return self._neutral_signal(symbol)

        # Volume z-score confirmation
        vol_zscore = self._compute_volume_zscore(df)
        if vol_zscore is None or vol_zscore < self.volume_zscore_threshold:
            return self._neutral_signal(symbol)

        # Determine direction
        if broke_high:
            direction = SignalDirection.LONG
        else:
            direction = SignalDirection.SHORT

        # Score computation
        score = self.base_score
        if vol_zscore >= 3.0:
            score += _STRONG_VOLUME_BONUS
        if self._trend_aligns(regime, direction):
            score += _TREND_ALIGNMENT_BONUS
        score = min(score, 100)

        # Entry, stop, targets
        entry_price = close
        if direction == SignalDirection.LONG:
            stop_loss = range_low
            tp1 = entry_price + range_height
            tp2 = entry_price + range_height * 2.0
            tp3 = entry_price + range_height * 3.0
        else:
            stop_loss = range_high
            tp1 = entry_price - range_height
            tp2 = entry_price - range_height * 2.0
            tp3 = entry_price - range_height * 3.0

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
                "range_high": round(range_high, 4),
                "range_low": round(range_low, 4),
                "range_height": round(range_height, 4),
                "volume_zscore": round(vol_zscore, 2),
                "trend_aligned": self._trend_aligns(regime, direction),
            },
        )

        logger.info(
            "Opening range breakout signal generated",
            extra={
                "symbol": symbol,
                "direction": direction.value,
                "score": signal.score,
                "range_high": round(range_high, 4),
                "range_low": round(range_low, 4),
                "volume_zscore": round(vol_zscore, 2),
            },
        )
        return signal

    # ------------------------------------------------------------------
    # Opening range detection
    # ------------------------------------------------------------------

    def _get_opening_range(
        self,
        symbol: str,
        df: pd.DataFrame,
    ) -> tuple[Optional[float], Optional[float]]:
        """Compute the 00:00-00:30 UTC opening range for today.

        The opening range is defined as the high and low of the first
        ``range_bars`` 5m bars after midnight UTC.  Results are cached
        per symbol per calendar day.

        Args:
            symbol: Trading pair symbol.
            df: OHLCV DataFrame with DatetimeIndex or ``timestamp`` column.

        Returns:
            Tuple of (range_high, range_low), or (None, None) on failure.
        """
        try:
            idx = df.index
            if not isinstance(idx, pd.DatetimeIndex):
                if "timestamp" in df.columns:
                    idx = pd.DatetimeIndex(df["timestamp"])
                else:
                    return None, None

            now_utc = datetime.now(timezone.utc)
            today_str = now_utc.strftime("%Y-%m-%d")

            # Return cached result if available for today
            cache_key = f"{symbol}_{today_str}"
            if cache_key in self._range_cache:
                cached = self._range_cache[cache_key]
                return cached["high"], cached["low"]

            # Find bars from 00:00 to 00:30 UTC today
            day_start = now_utc.replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            range_end = now_utc.replace(
                hour=0, minute=30, second=0, microsecond=0
            )

            day_start_ts = pd.Timestamp(day_start, tz=timezone.utc)
            range_end_ts = pd.Timestamp(range_end, tz=timezone.utc)

            mask = (idx >= day_start_ts) & (idx < range_end_ts)
            range_bars = df.loc[mask]

            if range_bars.empty or len(range_bars) < self.range_bars:
                # Not enough bars yet -- range not fully formed
                return None, None

            range_high = float(range_bars["high"].max())
            range_low = float(range_bars["low"].min())

            # Cache for the day
            self._range_cache[cache_key] = {
                "high": range_high,
                "low": range_low,
            }

            logger.debug(
                "Opening range computed for %s: %.4f - %.4f",
                symbol,
                range_low,
                range_high,
            )
            return range_high, range_low

        except Exception:
            logger.exception("Error computing opening range for %s", symbol)
            return None, None

    # ------------------------------------------------------------------
    # Volume z-score
    # ------------------------------------------------------------------

    def _compute_volume_zscore(self, df: pd.DataFrame) -> Optional[float]:
        """Compute the z-score of the most recent bar's volume.

        Args:
            df: OHLCV DataFrame with a ``volume`` column.

        Returns:
            Volume z-score, or None on failure.
        """
        try:
            vol = df["volume"].iloc[-self.volume_lookback:]
            if len(vol) < 10:
                return None
            mean = float(vol.mean())
            std = float(vol.std())
            if std <= 0:
                return None
            current = float(vol.iloc[-1])
            return (current - mean) / std
        except Exception:
            logger.exception("Error computing volume z-score")
            return None

    # ------------------------------------------------------------------
    # Trend alignment
    # ------------------------------------------------------------------

    @staticmethod
    def _trend_aligns(regime: str, direction: SignalDirection) -> bool:
        """Check whether the market regime aligns with the trade direction.

        A LONG during a bullish regime or SHORT during a bearish regime
        is considered aligned.

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
