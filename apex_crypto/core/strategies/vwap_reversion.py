"""VWAP Reversion strategy for the APEX Crypto Trading System.

Trades mean-reversion setups around the intraday VWAP on 5m bars for
BTC/USDT, ETH/USDT, and SOL/USDT.  Enters when price deviates from
VWAP by more than 0.4% with elevated volume and RSI(7) confirmation,
targeting a return to VWAP.
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

logger = get_logger("strategies.vwap_reversion")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_ALLOWED_ASSETS: list[str] = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
_TIMEFRAME: str = "5m"

_VWAP_DEVIATION_THRESHOLD: float = 0.004   # 0.4%
_VWAP_EXTREME_DEVIATION: float = 0.006     # 0.6%
_VOLUME_ZSCORE_THRESHOLD: float = 1.5
_RSI_OVERSOLD: float = 35.0
_RSI_OVERBOUGHT: float = 65.0
_RSI_PERIOD: int = 7

_STOP_PCT: float = 0.0035                  # 0.35%
_TP_MIN_PCT: float = 0.004                 # 0.4%
_TP_MAX_PCT: float = 0.008                 # 0.8%

_BASE_SCORE: int = 55
_EXTREME_DEVIATION_BONUS: int = 15
_VOLUME_SPIKE_BONUS: int = 10
_VOLUME_LOOKBACK: int = 50                 # bars for volume z-score


class VWAPReversionStrategy(BaseStrategy):
    """Mean-reversion around the intraday VWAP on 5-minute bars.

    Entry conditions:
    - LONG : price < VWAP by > 0.4%, volume z-score > 1.5, RSI(7) < 35.
    - SHORT: price > VWAP by > 0.4%, volume z-score > 1.5, RSI(7) > 65.

    Target is a return to VWAP (0.4-0.8%).  Stop is 0.35% from entry.
    Active in ALL market regimes.

    Attributes:
        name: Strategy identifier.
        active_regimes: Empty list means active in all regimes.
        primary_timeframe: 5m bars.
    """

    name: str = "vwap_reversion"
    active_regimes: list[str] = []
    primary_timeframe: str = _TIMEFRAME
    confirmation_timeframe: str = "15m"
    entry_timeframe: str = "5m"

    def __init__(self, config: dict) -> None:
        """Initialize VWAPReversionStrategy.

        Args:
            config: Strategy-specific configuration dictionary.
        """
        super().__init__(config)
        cfg = config.get("strategies", {}).get("vwap_reversion", {})
        self.deviation_threshold: float = cfg.get(
            "deviation_threshold", _VWAP_DEVIATION_THRESHOLD
        )
        self.extreme_deviation: float = cfg.get(
            "extreme_deviation", _VWAP_EXTREME_DEVIATION
        )
        self.volume_zscore_threshold: float = cfg.get(
            "volume_zscore_threshold", _VOLUME_ZSCORE_THRESHOLD
        )
        self.rsi_oversold: float = cfg.get("rsi_oversold", _RSI_OVERSOLD)
        self.rsi_overbought: float = cfg.get("rsi_overbought", _RSI_OVERBOUGHT)
        self.stop_pct: float = cfg.get("stop_pct", _STOP_PCT)
        self.base_score: int = cfg.get("base_score", _BASE_SCORE)
        self.volume_lookback: int = cfg.get("volume_lookback", _VOLUME_LOOKBACK)

        logger.info(
            "VWAPReversionStrategy configured",
            extra={
                "deviation_threshold": self.deviation_threshold,
                "volume_zscore_threshold": self.volume_zscore_threshold,
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
        """Generate a VWAP-reversion signal.

        Args:
            symbol: Trading pair symbol.
            data: OHLCV DataFrames keyed by timeframe.
            indicators: Pre-computed indicator DataFrames keyed by timeframe.
            regime: Current market regime string.
            alt_data: Optional alternative data (unused).

        Returns:
            TradeSignal with direction and score, or NEUTRAL.
        """
        # Gate checks
        if not self.is_active(regime):
            return self._neutral_signal(symbol)

        if symbol not in _ALLOWED_ASSETS:
            logger.debug("%s not in allowed VWAP reversion assets", symbol)
            return self._neutral_signal(symbol)

        tf = self.primary_timeframe
        if tf not in data or data[tf].empty:
            logger.warning("Missing %s OHLCV data for %s", tf, symbol)
            return self._neutral_signal(symbol)

        df = data[tf]
        ind = indicators.get(tf)

        if len(df) < self.volume_lookback:
            logger.debug("Insufficient data length for %s", symbol)
            return self._neutral_signal(symbol)

        # Compute VWAP from day start (00:00 UTC)
        vwap = self._compute_intraday_vwap(df)
        if vwap is None or np.isnan(vwap):
            logger.debug("Could not compute VWAP for %s", symbol)
            return self._neutral_signal(symbol)

        close = float(df["close"].iloc[-1])
        vwap_deviation = (close - vwap) / vwap if vwap != 0 else 0.0
        abs_deviation = abs(vwap_deviation)

        # Check deviation threshold
        if abs_deviation < self.deviation_threshold:
            return self._neutral_signal(symbol)

        # Volume z-score
        vol_zscore = self._compute_volume_zscore(df)
        if vol_zscore is None or vol_zscore < self.volume_zscore_threshold:
            return self._neutral_signal(symbol)

        # RSI(7) — prefer pre-computed, else compute from close
        rsi = self._get_rsi(df, ind)
        if rsi is None:
            return self._neutral_signal(symbol)

        # Determine direction
        direction: Optional[SignalDirection] = None
        if vwap_deviation < 0 and rsi < self.rsi_oversold:
            direction = SignalDirection.LONG
        elif vwap_deviation > 0 and rsi > self.rsi_overbought:
            direction = SignalDirection.SHORT

        if direction is None:
            return self._neutral_signal(symbol)

        # Score computation
        score = self.base_score
        if abs_deviation >= self.extreme_deviation:
            score += _EXTREME_DEVIATION_BONUS
        if vol_zscore >= 2.5:
            score += _VOLUME_SPIKE_BONUS
        score = min(score, 100)

        # Entry, stop, targets
        entry_price = close
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
                "vwap": round(vwap, 4),
                "vwap_deviation": round(vwap_deviation, 6),
                "volume_zscore": round(vol_zscore, 2),
                "rsi_7": round(rsi, 2),
            },
        )

        logger.info(
            "VWAP reversion signal generated",
            extra={
                "symbol": symbol,
                "direction": direction.value,
                "score": signal.score,
                "vwap_deviation": round(vwap_deviation, 6),
                "volume_zscore": round(vol_zscore, 2),
                "rsi": round(rsi, 2),
            },
        )
        return signal

    # ------------------------------------------------------------------
    # VWAP calculation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_intraday_vwap(df: pd.DataFrame) -> Optional[float]:
        """Compute cumulative VWAP from the most recent 00:00 UTC bar.

        VWAP = cumulative(typical_price * volume) / cumulative(volume)
        Typical price = (high + low + close) / 3

        Args:
            df: OHLCV DataFrame (must have a DatetimeIndex or a datetime column).

        Returns:
            Current VWAP value, or None on failure.
        """
        try:
            idx = df.index
            if not isinstance(idx, pd.DatetimeIndex):
                if "timestamp" in df.columns:
                    idx = pd.DatetimeIndex(df["timestamp"])
                else:
                    return None

            # Locate the start of today (00:00 UTC)
            now_utc = datetime.now(timezone.utc)
            day_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)

            mask = idx >= pd.Timestamp(day_start, tz=timezone.utc)
            if not mask.any():
                # Fallback: use all available data
                mask = pd.Series(True, index=df.index)

            intraday = df.loc[mask]
            if intraday.empty:
                return None

            typical_price = (
                intraday["high"] + intraday["low"] + intraday["close"]
            ) / 3.0
            cum_tp_vol = (typical_price * intraday["volume"]).cumsum()
            cum_vol = intraday["volume"].cumsum()

            # Avoid division by zero
            last_cum_vol = float(cum_vol.iloc[-1])
            if last_cum_vol <= 0:
                return None

            vwap = float(cum_tp_vol.iloc[-1] / last_cum_vol)
            return vwap
        except Exception:
            logger.exception("Error computing intraday VWAP")
            return None

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
    # RSI helper
    # ------------------------------------------------------------------

    @staticmethod
    def _get_rsi(
        df: pd.DataFrame,
        ind: Optional[pd.DataFrame],
    ) -> Optional[float]:
        """Retrieve or compute RSI(7).

        First checks pre-computed indicators for ``rsi_7``.  Falls back to
        manual computation from close prices.

        Args:
            df: OHLCV DataFrame.
            ind: Optional pre-computed indicator DataFrame.

        Returns:
            RSI(7) value, or None on failure.
        """
        # Try pre-computed
        if ind is not None and not ind.empty and "rsi_7" in ind.columns:
            val = ind["rsi_7"].iloc[-1]
            if not pd.isna(val):
                return float(val)

        # Manual RSI(7)
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
            logger.exception("Error computing RSI(7)")
            return None
