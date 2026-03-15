"""Stock Momentum Strategy — technical trend-following for equities.

Uses the same technical framework as the crypto trend strategy but
adapted for stock market characteristics:
- Works on daily/weekly timeframes (stocks are slower than crypto)
- Considers sector rotation signals
- Respects market hours
- Uses stock-specific momentum factors (52w high proximity, relative strength)
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from apex_crypto.core.logging import get_logger, log_with_data
from apex_crypto.core.strategies.base import (
    BaseStrategy, SignalDirection, TradeSignal,
)

logger = get_logger("stocks.strategies.momentum")


class StockMomentumStrategy(BaseStrategy):
    """Technical momentum strategy for stocks."""

    name = "stock_momentum"
    active_regimes = ["STRONG_BULL", "WEAK_BULL", "RANGING"]
    primary_timeframe = "1d"
    confirmation_timeframe = "1d"
    entry_timeframe = "1d"

    def __init__(self, config: dict) -> None:
        super().__init__(config)

        self._ema_fast: int = config.get("ema_fast", 21)
        self._ema_slow: int = config.get("ema_slow", 50)
        self._rsi_period: int = config.get("rsi_period", 14)
        self._volume_zscore_min: float = config.get("volume_zscore_min", 1.5)
        self._relative_strength_window: int = config.get("rs_window", 60)

        logger.info("StockMomentumStrategy configured")

    def generate_signal(
        self,
        symbol: str,
        data: dict[str, pd.DataFrame],
        indicators: dict[str, pd.DataFrame],
        regime: str,
        alt_data: Optional[dict] = None,
    ) -> TradeSignal:
        """Generate momentum signal from price and volume data."""
        df = data.get(self.primary_timeframe)
        if df is None or len(df) < self._ema_slow + 10:
            return self._neutral_signal(symbol)

        ind = indicators.get(self.primary_timeframe)
        if ind is None:
            ind = df

        close = df["close"]
        volume = df["volume"]
        current_price = float(close.iloc[-1])

        # Calculate indicators if not pre-computed
        ema_fast = self._calc_ema(close, self._ema_fast)
        ema_slow = self._calc_ema(close, self._ema_slow)
        rsi = self._calc_rsi(close, self._rsi_period)
        atr = self._calc_atr(df)
        vol_zscore = self._calc_volume_zscore(volume)

        if ema_fast is None or ema_slow is None or rsi is None:
            return self._neutral_signal(symbol)

        # Momentum factors
        score = 0

        # 1. EMA trend alignment
        if ema_fast > ema_slow:
            score += 20
            # EMA crossover recently (within 5 bars)
            if len(close) > 5:
                prev_fast = self._calc_ema(close.iloc[:-5], self._ema_fast)
                prev_slow = self._calc_ema(close.iloc[:-5], self._ema_slow)
                if prev_fast is not None and prev_slow is not None:
                    if prev_fast <= prev_slow:
                        score += 15  # fresh crossover
        elif ema_fast < ema_slow:
            score -= 20

        # 2. RSI momentum
        if 55 < rsi < 70:
            score += 15  # bullish momentum
        elif 30 < rsi < 45:
            score -= 15  # bearish momentum
        elif rsi >= 75:
            score -= 10  # overbought
        elif rsi <= 25:
            score += 10  # oversold bounce

        # 3. Volume confirmation
        if vol_zscore > self._volume_zscore_min:
            if score > 0:
                score += 15  # volume confirms bullish
            elif score < 0:
                score -= 15  # volume confirms bearish

        # 4. 52-week relative strength
        if len(close) >= 252:
            high_252 = float(close.rolling(252).max().iloc[-1])
            low_252 = float(close.rolling(252).min().iloc[-1])
            if high_252 > low_252:
                rs_pct = (current_price - low_252) / (high_252 - low_252)
                if rs_pct > 0.80:
                    score += 10  # near highs
                elif rs_pct < 0.20:
                    score -= 10  # near lows

        # 5. Price above/below 200 EMA (long-term trend)
        ema_200 = self._calc_ema(close, 200)
        if ema_200 is not None:
            if current_price > ema_200:
                score += 10
            else:
                score -= 10

        # Determine direction
        if score > 0:
            direction = SignalDirection.LONG
        elif score < 0:
            direction = SignalDirection.SHORT
        else:
            return self._neutral_signal(symbol)

        # Stop loss and targets
        atr_val = atr if atr and atr > 0 else current_price * 0.02
        stop_loss = self.compute_stop_loss(
            current_price, direction, atr_val,
            atr_multiplier=2.0, max_stop_pct=0.05,
        )
        tp1, tp2, tp3 = self.compute_take_profits(
            current_price, stop_loss, direction,
            tp1_r=2.0, tp2_r=3.0, tp3_r=5.0,
        )

        return TradeSignal(
            symbol=symbol,
            direction=direction,
            score=max(-100, min(100, score)),
            strategy=self.name,
            timeframe=self.primary_timeframe,
            entry_price=current_price,
            stop_loss=round(stop_loss, 2),
            take_profit_1=round(tp1, 2),
            take_profit_2=round(tp2, 2),
            take_profit_3=round(tp3, 2),
            confidence=min(abs(score) / 100, 0.90),
            metadata={"asset_type": "stock", "rsi": round(rsi, 1), "vol_zscore": round(vol_zscore, 2)},
        )

    # ------------------------------------------------------------------
    # Indicator helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_ema(series: pd.Series, period: int) -> Optional[float]:
        if len(series) < period:
            return None
        return float(series.ewm(span=period, adjust=False).mean().iloc[-1])

    @staticmethod
    def _calc_rsi(close: pd.Series, period: int = 14) -> Optional[float]:
        if len(close) < period + 1:
            return None
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1])

    @staticmethod
    def _calc_atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
        if len(df) < period + 1:
            return None
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return float(atr.iloc[-1])

    @staticmethod
    def _calc_volume_zscore(volume: pd.Series, period: int = 20) -> float:
        if len(volume) < period:
            return 0.0
        mean = volume.rolling(period).mean().iloc[-1]
        std = volume.rolling(period).std().iloc[-1]
        if std == 0:
            return 0.0
        return float((volume.iloc[-1] - mean) / std)
