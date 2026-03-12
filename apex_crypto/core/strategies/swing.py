"""Swing structure strategy for the APEX Crypto Trading System.

Identifies higher-timeframe pullback entries at confluent support/resistance
levels — the 50 EMA on the daily chart and the 0.618 Fibonacci retracement —
and confirms with reversal candlestick patterns on the 4h entry timeframe.
Hold time: 2-10 days.
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

logger = get_logger("strategies.swing")

# Fibonacci retracement level of interest
_FIB_LEVEL: float = 0.618

# EMA proximity threshold — price within this percentage of the 50 EMA counts
_EMA_PROXIMITY_PCT: float = 0.005  # 0.5%

# Fibonacci proximity threshold
_FIB_PROXIMITY_PCT: float = 0.005  # 0.5%

# Volume confirmation: current bar vs 20-bar average
_VOLUME_SPIKE_MULT: float = 1.3

# Minimum candle body-to-total ratio for reversal candle detection
_MIN_BODY_RATIO: float = 0.35

# Wick-to-body ratio threshold for pin bars / hammers
_PIN_WICK_RATIO: float = 2.0


class SwingStructureStrategy(BaseStrategy):
    """Swing trading strategy on daily pullbacks with 4h entry.

    Enters when price retraces to the 50 EMA on the 1d chart *or* to the
    0.618 Fibonacci retracement of the last major swing, and a reversal
    candlestick pattern (hammer, engulfing, pin bar) appears on the 4h
    chart.

    Attributes:
        name: Strategy identifier.
        active_regimes: Market regimes where this strategy operates.
        primary_timeframe: Higher timeframe for structure analysis.
        entry_timeframe: Lower timeframe for entry confirmation.
    """

    name: str = "swing"
    active_regimes: list[str] = ["WEAK_BULL", "WEAK_BEAR", "RANGING"]
    primary_timeframe: str = "1d"
    entry_timeframe: str = "4h"

    def __init__(self, config: dict) -> None:
        """Initialize the swing structure strategy.

        Args:
            config: Strategy-specific configuration dict.
        """
        super().__init__(config)

        swing_cfg: dict = config.get("strategies", {}).get("swing", {})
        self._ema_proximity: float = swing_cfg.get(
            "ema_proximity_pct", _EMA_PROXIMITY_PCT
        )
        self._fib_proximity: float = swing_cfg.get(
            "fib_proximity_pct", _FIB_PROXIMITY_PCT
        )
        self._swing_lookback: int = swing_cfg.get("swing_lookback", 50)

        logger.info(
            "SwingStructureStrategy configured",
            extra={
                "data": {
                    "ema_proximity_pct": self._ema_proximity,
                    "fib_proximity_pct": self._fib_proximity,
                    "swing_lookback": self._swing_lookback,
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
        """Generate a swing structure signal for *symbol*.

        Args:
            symbol: Trading pair symbol (e.g. ``'BTC/USDT'``).
            data: OHLCV DataFrames keyed by timeframe (``'1d'``, ``'4h'``).
            indicators: Pre-computed indicator DataFrames keyed by timeframe.
                Expected columns on 1d: ``ema_50``, ``atr``.
                Expected columns on 4h: ``atr``, ``volume``.
            regime: Current market regime string.
            alt_data: Optional alternative data dict (unused by this strategy).

        Returns:
            TradeSignal with score of 0 (neutral) when conditions are not met,
            or 55-100 when a valid swing setup is detected.
        """
        # ----- Gate checks ------------------------------------------------
        if not self.is_active(regime):
            logger.debug("Regime %s not active for swing strategy", regime)
            return self._neutral_signal(symbol)

        df_1d = data.get(self.primary_timeframe)
        ind_1d = indicators.get(self.primary_timeframe)
        df_4h = data.get(self.entry_timeframe)
        ind_4h = indicators.get(self.entry_timeframe)

        if df_1d is None or df_1d.empty or ind_1d is None or ind_1d.empty:
            logger.debug("Missing 1d data/indicators for %s", symbol)
            return self._neutral_signal(symbol)

        if df_4h is None or df_4h.empty:
            logger.debug("Missing 4h data for %s", symbol)
            return self._neutral_signal(symbol)

        # ----- Daily context ----------------------------------------------
        close_1d: float = float(df_1d["close"].iloc[-1])
        ema_50: float = float(ind_1d["ema_50"].iloc[-1])

        # Swing high / low from the lookback window
        lookback_df = df_1d.tail(self._swing_lookback)
        swing_high: float = float(lookback_df["high"].max())
        swing_low: float = float(lookback_df["low"].min())

        # Fibonacci 0.618 retracement from swing high to swing low
        fib_range: float = swing_high - swing_low
        if fib_range <= 0:
            logger.debug("No swing range for %s", symbol)
            return self._neutral_signal(symbol)

        fib_618: float = swing_high - fib_range * _FIB_LEVEL

        # ----- Pullback to key level? -------------------------------------
        at_ema: bool = abs(close_1d - ema_50) / ema_50 <= self._ema_proximity
        at_fib: bool = abs(close_1d - fib_618) / fib_618 <= self._fib_proximity
        ema_fib_confluence: bool = at_ema and at_fib

        if not at_ema and not at_fib:
            logger.debug(
                "Price not at key level for %s (EMA=%.2f, Fib618=%.2f, close=%.2f)",
                symbol,
                ema_50,
                fib_618,
                close_1d,
            )
            return self._neutral_signal(symbol)

        # ----- Direction from regime + level context ----------------------
        # In WEAK_BULL / RANGING: pullback to support → look for longs
        # In WEAK_BEAR: pullback to resistance → look for shorts
        if regime in ("WEAK_BULL", "RANGING"):
            direction = SignalDirection.LONG
        else:
            direction = SignalDirection.SHORT

        # ----- 4h reversal candle confirmation ----------------------------
        reversal_type: Optional[str] = self._detect_reversal_candle(df_4h, direction)
        if reversal_type is None:
            logger.debug("No reversal candle on 4h for %s", symbol)
            return self._neutral_signal(symbol)

        # ----- Volume confirmation on 4h ----------------------------------
        volume_confirmed: bool = False
        if ind_4h is not None and not ind_4h.empty and "volume" in df_4h.columns:
            vol_current: float = float(df_4h["volume"].iloc[-1])
            vol_mean: float = float(df_4h["volume"].rolling(20).mean().iloc[-1])
            if vol_mean > 0 and vol_current > vol_mean * _VOLUME_SPIKE_MULT:
                volume_confirmed = True

        # ----- Higher timeframe trend alignment ---------------------------
        htf_aligned: bool = False
        if "ema_50" in ind_1d.columns and len(ind_1d) >= 2:
            ema_slope = float(ind_1d["ema_50"].iloc[-1]) - float(
                ind_1d["ema_50"].iloc[-2]
            )
            if direction == SignalDirection.LONG and ema_slope > 0:
                htf_aligned = True
            elif direction == SignalDirection.SHORT and ema_slope < 0:
                htf_aligned = True

        # ----- Score ------------------------------------------------------
        score: int = 55

        if ema_fib_confluence:
            score += 15

        if reversal_type is not None:
            score += 10

        if volume_confirmed:
            score += 10

        if htf_aligned:
            score += 10

        score = min(score, 100)

        # ----- Entry, stop, targets on 4h ---------------------------------
        entry_price: float = float(df_4h["close"].iloc[-1])

        atr_4h: float = 0.0
        if ind_4h is not None and not ind_4h.empty and "atr" in ind_4h.columns:
            atr_4h = float(ind_4h["atr"].iloc[-1])

        # Structure-based swing level for stop
        if direction == SignalDirection.LONG:
            swing_level = swing_low
        else:
            swing_level = swing_high

        if atr_4h > 0:
            stop_loss = self.compute_stop_loss(
                entry_price,
                direction,
                atr_4h,
                swing_level=swing_level,
                atr_multiplier=2.0,
                max_stop_pct=0.04,
            )
        else:
            pct = 0.03
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
                "ema_50": round(ema_50, 4),
                "fib_618": round(fib_618, 4),
                "at_ema": at_ema,
                "at_fib": at_fib,
                "ema_fib_confluence": ema_fib_confluence,
                "reversal_candle": reversal_type,
                "volume_confirmed": volume_confirmed,
                "htf_aligned": htf_aligned,
                "swing_high": round(swing_high, 4),
                "swing_low": round(swing_low, 4),
                "hold_days": "2-10",
            },
        )

        log_with_data(
            logger,
            "info",
            f"Swing signal generated for {symbol}",
            data=signal.to_dict(),
        )

        return signal

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_reversal_candle(
        df: pd.DataFrame,
        direction: SignalDirection,
    ) -> Optional[str]:
        """Detect reversal candlestick patterns on the latest two bars.

        Looks for hammer, bullish/bearish engulfing, and pin bars.

        Args:
            df: OHLCV DataFrame (4h).
            direction: Expected trade direction (determines which patterns
                are relevant).

        Returns:
            Name of the detected pattern, or ``None`` if nothing found.
        """
        if len(df) < 2:
            return None

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        c_open: float = float(curr["open"])
        c_close: float = float(curr["close"])
        c_high: float = float(curr["high"])
        c_low: float = float(curr["low"])
        c_body: float = abs(c_close - c_open)
        c_range: float = c_high - c_low

        if c_range == 0:
            return None

        body_ratio: float = c_body / c_range

        p_open: float = float(prev["open"])
        p_close: float = float(prev["close"])
        p_body: float = abs(p_close - p_open)

        if direction == SignalDirection.LONG:
            # Hammer: small body at top, long lower wick
            lower_wick: float = min(c_open, c_close) - c_low
            upper_wick: float = c_high - max(c_open, c_close)
            if (
                c_body > 0
                and lower_wick >= c_body * _PIN_WICK_RATIO
                and upper_wick < c_body
                and c_close > c_open
            ):
                return "hammer"

            # Bullish engulfing: current green candle engulfs previous red
            if (
                p_close < p_open
                and c_close > c_open
                and c_body > p_body
                and c_close > p_open
                and c_open <= p_close
            ):
                return "bullish_engulfing"

            # Bullish pin bar: long lower wick, body in upper third
            if (
                body_ratio >= _MIN_BODY_RATIO
                and lower_wick >= c_body * _PIN_WICK_RATIO
                and c_close > c_open
            ):
                return "bullish_pin_bar"

        elif direction == SignalDirection.SHORT:
            # Inverted hammer / shooting star
            upper_wick = c_high - max(c_open, c_close)
            lower_wick = min(c_open, c_close) - c_low
            if (
                c_body > 0
                and upper_wick >= c_body * _PIN_WICK_RATIO
                and lower_wick < c_body
                and c_close < c_open
            ):
                return "shooting_star"

            # Bearish engulfing
            if (
                p_close > p_open
                and c_close < c_open
                and c_body > p_body
                and c_close < p_open
                and c_open >= p_close
            ):
                return "bearish_engulfing"

            # Bearish pin bar
            if (
                body_ratio >= _MIN_BODY_RATIO
                and upper_wick >= c_body * _PIN_WICK_RATIO
                and c_close < c_open
            ):
                return "bearish_pin_bar"

        return None
