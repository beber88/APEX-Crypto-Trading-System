"""Trend Momentum strategy for the APEX Crypto Trading System.

Uses multi-timeframe EMA alignment, MACD momentum, RSI filtering,
VWAP confirmation, and volume z-score thresholds to enter high-conviction
trend-following trades in strongly trending regimes.
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

logger = get_logger("strategies.trend_momentum")


class TrendMomentumStrategy(BaseStrategy):
    """Trend-following strategy combining EMA alignment with momentum filters.

    Enters trades only during strongly trending markets (STRONG_BULL or
    STRONG_BEAR) when multi-timeframe EMA stacking, MACD histogram
    expansion, RSI positioning, VWAP confirmation, and elevated volume
    all agree on direction.

    Attributes:
        name: Strategy identifier.
        active_regimes: Regimes in which signals are generated.
        primary_timeframe: Timeframe used for primary trend assessment.
        confirmation_timeframe: Higher timeframe for trend confirmation.
        entry_timeframe: Lower timeframe for precise entry timing.
    """

    name: str = "trend_momentum"
    active_regimes: list[str] = ["STRONG_BULL", "STRONG_BEAR"]
    primary_timeframe: str = "4h"
    confirmation_timeframe: str = "1d"
    entry_timeframe: str = "1h"

    # ------------------------------------------------------------------
    # Configuration defaults
    # ------------------------------------------------------------------
    _DEFAULT_RSI_PERIOD: int = 14
    _DEFAULT_RSI_LONG_MIN: float = 50.0
    _DEFAULT_RSI_LONG_MAX: float = 75.0
    _DEFAULT_RSI_SHORT_MIN: float = 25.0
    _DEFAULT_RSI_SHORT_MAX: float = 50.0
    _DEFAULT_VOLUME_ZSCORE_THRESHOLD: float = 1.5
    _DEFAULT_ADX_THRESHOLD: float = 30.0
    _DEFAULT_BASE_SCORE: int = 50

    def __init__(self, config: dict) -> None:
        """Initialize TrendMomentumStrategy.

        Args:
            config: Strategy-specific configuration dictionary.  Recognised
                keys (all optional):
                - rsi_long_min / rsi_long_max: RSI bounds for longs.
                - rsi_short_min / rsi_short_max: RSI bounds for shorts.
                - volume_zscore_threshold: Minimum volume z-score.
                - adx_threshold: ADX level considered "strong".
        """
        super().__init__(config)
        self.rsi_long_min: float = config.get("rsi_long_min", self._DEFAULT_RSI_LONG_MIN)
        self.rsi_long_max: float = config.get("rsi_long_max", self._DEFAULT_RSI_LONG_MAX)
        self.rsi_short_min: float = config.get("rsi_short_min", self._DEFAULT_RSI_SHORT_MIN)
        self.rsi_short_max: float = config.get("rsi_short_max", self._DEFAULT_RSI_SHORT_MAX)
        self.volume_zscore_threshold: float = config.get(
            "volume_zscore_threshold", self._DEFAULT_VOLUME_ZSCORE_THRESHOLD
        )
        self.adx_threshold: float = config.get("adx_threshold", self._DEFAULT_ADX_THRESHOLD)
        self.base_score: int = config.get("base_score", self._DEFAULT_BASE_SCORE)
        logger.info(
            "TrendMomentumStrategy configured",
            extra={
                "rsi_long_range": [self.rsi_long_min, self.rsi_long_max],
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
        """Generate a trend-momentum trading signal.

        Evaluates multi-timeframe EMA alignment, MACD histogram momentum,
        RSI positioning, VWAP relationship, and volume z-score.  When all
        conditions align, a directional signal with a scored conviction is
        returned.

        Args:
            symbol: Trading pair symbol (e.g., ``'BTC/USDT'``).
            data: OHLCV DataFrames keyed by timeframe string.
            indicators: Pre-computed indicator DataFrames keyed by timeframe.
            regime: Current market regime string.
            alt_data: Optional alternative data (unused by this strategy).

        Returns:
            A ``TradeSignal`` reflecting the assessment.  Direction is NEUTRAL
            when conditions are not met.
        """
        if not self.is_active(regime):
            logger.debug("Strategy inactive for regime %s", regime)
            return self._neutral_signal(symbol)

        # Validate required timeframes are present
        for tf in (self.primary_timeframe, self.confirmation_timeframe, self.entry_timeframe):
            if tf not in data or tf not in indicators:
                logger.warning(
                    "Missing timeframe data",
                    extra={"timeframe": tf, "symbol": symbol},
                )
                return self._neutral_signal(symbol)

        primary_data = data[self.primary_timeframe]
        primary_ind = indicators[self.primary_timeframe]
        confirm_ind = indicators[self.confirmation_timeframe]
        entry_data = data[self.entry_timeframe]

        if primary_data.empty or primary_ind.empty:
            logger.warning("Empty primary data for %s", symbol)
            return self._neutral_signal(symbol)

        # --- Evaluate long conditions ---
        long_ok, long_extras = self._check_long_conditions(primary_data, primary_ind)
        if long_ok:
            score = self._compute_score(
                primary_ind, confirm_ind, SignalDirection.LONG
            )
            return self._build_signal(
                symbol, SignalDirection.LONG, score, entry_data, primary_ind, long_extras
            )

        # --- Evaluate short conditions ---
        short_ok, short_extras = self._check_short_conditions(primary_data, primary_ind)
        if short_ok:
            score = self._compute_score(
                primary_ind, confirm_ind, SignalDirection.SHORT
            )
            return self._build_signal(
                symbol, SignalDirection.SHORT, score, entry_data, primary_ind, short_extras
            )

        # --- Check for exit conditions on existing positions ---
        exit_direction = self._check_exit_conditions(primary_ind)
        if exit_direction is not None:
            logger.info(
                "Exit signal detected",
                extra={"symbol": symbol, "exit_direction": exit_direction.value},
            )
            return TradeSignal(
                symbol=symbol,
                direction=exit_direction,
                score=0,
                strategy=self.name,
                timeframe=self.primary_timeframe,
                metadata={"signal_type": "exit"},
            )

        return self._neutral_signal(symbol)

    # ------------------------------------------------------------------
    # Long / short condition checks
    # ------------------------------------------------------------------

    def _check_long_conditions(
        self,
        primary_data: pd.DataFrame,
        primary_ind: pd.DataFrame,
    ) -> tuple[bool, dict[str, Any]]:
        """Evaluate all long-entry conditions on the primary timeframe.

        All five conditions must be satisfied simultaneously:
        1. EMA21 > EMA50 > EMA200 (bullish EMA stack).
        2. MACD histogram is rising (current > previous bar).
        3. RSI(14) between ``rsi_long_min`` and ``rsi_long_max``.
        4. Price above VWAP.
        5. Volume z-score > threshold.

        Args:
            primary_data: OHLCV DataFrame for the primary timeframe.
            primary_ind: Indicator DataFrame for the primary timeframe.

        Returns:
            Tuple of (conditions_met, metadata_dict).
        """
        extras: dict[str, Any] = {}

        # 1. EMA alignment
        ema21 = self._safe_last(primary_ind, "ema_21")
        ema50 = self._safe_last(primary_ind, "ema_50")
        ema200 = self._safe_last(primary_ind, "ema_200")
        if ema21 is None or ema50 is None or ema200 is None:
            return False, extras
        ema_aligned = ema21 > ema50 > ema200
        if not ema_aligned:
            return False, extras
        extras["ema_alignment"] = "bullish"

        # 2. MACD histogram rising
        macd_hist = self._safe_last(primary_ind, "macd_histogram")
        macd_hist_prev = self._safe_last(primary_ind, "macd_histogram", offset=1)
        if macd_hist is None or macd_hist_prev is None:
            return False, extras
        if macd_hist <= macd_hist_prev:
            return False, extras
        extras["macd_histogram"] = float(macd_hist)
        extras["macd_histogram_prev"] = float(macd_hist_prev)

        # 3. RSI in range
        rsi = self._safe_last(primary_ind, "rsi_14")
        if rsi is None:
            return False, extras
        if not (self.rsi_long_min < rsi < self.rsi_long_max):
            return False, extras
        extras["rsi"] = float(rsi)

        # 4. Price above VWAP
        close = self._safe_last(primary_data, "close")
        vwap = self._safe_last(primary_ind, "vwap")
        if close is None or vwap is None:
            return False, extras
        if close <= vwap:
            return False, extras
        extras["close"] = float(close)
        extras["vwap"] = float(vwap)

        # 5. Volume z-score
        vol_z = self._safe_last(primary_ind, "volume_zscore")
        if vol_z is None:
            vol_z = self._compute_volume_zscore(primary_data)
        if vol_z is None or vol_z < self.volume_zscore_threshold:
            return False, extras
        extras["volume_zscore"] = float(vol_z)

        logger.info(
            "Long conditions met",
            extra={"conditions": extras},
        )
        return True, extras

    def _check_short_conditions(
        self,
        primary_data: pd.DataFrame,
        primary_ind: pd.DataFrame,
    ) -> tuple[bool, dict[str, Any]]:
        """Evaluate all short-entry conditions on the primary timeframe.

        Mirror of long conditions:
        1. EMA21 < EMA50 < EMA200 (bearish EMA stack).
        2. MACD histogram is falling (current < previous bar).
        3. RSI(14) between ``rsi_short_min`` and ``rsi_short_max``.
        4. Price below VWAP.
        5. Volume z-score > threshold.

        Args:
            primary_data: OHLCV DataFrame for the primary timeframe.
            primary_ind: Indicator DataFrame for the primary timeframe.

        Returns:
            Tuple of (conditions_met, metadata_dict).
        """
        extras: dict[str, Any] = {}

        # 1. EMA alignment (bearish)
        ema21 = self._safe_last(primary_ind, "ema_21")
        ema50 = self._safe_last(primary_ind, "ema_50")
        ema200 = self._safe_last(primary_ind, "ema_200")
        if ema21 is None or ema50 is None or ema200 is None:
            return False, extras
        ema_aligned = ema21 < ema50 < ema200
        if not ema_aligned:
            return False, extras
        extras["ema_alignment"] = "bearish"

        # 2. MACD histogram falling
        macd_hist = self._safe_last(primary_ind, "macd_histogram")
        macd_hist_prev = self._safe_last(primary_ind, "macd_histogram", offset=1)
        if macd_hist is None or macd_hist_prev is None:
            return False, extras
        if macd_hist >= macd_hist_prev:
            return False, extras
        extras["macd_histogram"] = float(macd_hist)
        extras["macd_histogram_prev"] = float(macd_hist_prev)

        # 3. RSI in range
        rsi = self._safe_last(primary_ind, "rsi_14")
        if rsi is None:
            return False, extras
        if not (self.rsi_short_min < rsi < self.rsi_short_max):
            return False, extras
        extras["rsi"] = float(rsi)

        # 4. Price below VWAP
        close = self._safe_last(primary_data, "close")
        vwap = self._safe_last(primary_ind, "vwap")
        if close is None or vwap is None:
            return False, extras
        if close >= vwap:
            return False, extras
        extras["close"] = float(close)
        extras["vwap"] = float(vwap)

        # 5. Volume z-score
        vol_z = self._safe_last(primary_ind, "volume_zscore")
        if vol_z is None:
            vol_z = self._compute_volume_zscore(primary_data)
        if vol_z is None or vol_z < self.volume_zscore_threshold:
            return False, extras
        extras["volume_zscore"] = float(vol_z)

        logger.info(
            "Short conditions met",
            extra={"conditions": extras},
        )
        return True, extras

    # ------------------------------------------------------------------
    # Exit logic
    # ------------------------------------------------------------------

    def _check_exit_conditions(
        self,
        primary_ind: pd.DataFrame,
    ) -> Optional[SignalDirection]:
        """Detect exit signals on the primary (4h) timeframe.

        Exit triggers:
        - EMA21/EMA50 cross reversal (bearish cross -> exit long, bullish
          cross -> exit short).
        - RSI bearish divergence (price making higher high but RSI making
          lower high) or bullish divergence for shorts.

        Args:
            primary_ind: Indicator DataFrame for the primary timeframe.

        Returns:
            ``SignalDirection.NEUTRAL`` to signal an exit for longs or shorts,
            or ``None`` if no exit is warranted.
        """
        ema21_curr = self._safe_last(primary_ind, "ema_21")
        ema50_curr = self._safe_last(primary_ind, "ema_50")
        ema21_prev = self._safe_last(primary_ind, "ema_21", offset=1)
        ema50_prev = self._safe_last(primary_ind, "ema_50", offset=1)

        if None in (ema21_curr, ema50_curr, ema21_prev, ema50_prev):
            return None

        # Bearish EMA cross -> exit longs
        if ema21_prev > ema50_prev and ema21_curr <= ema50_curr:
            logger.info("Bearish EMA21/EMA50 cross detected — exit long signal")
            return SignalDirection.NEUTRAL

        # Bullish EMA cross -> exit shorts
        if ema21_prev < ema50_prev and ema21_curr >= ema50_curr:
            logger.info("Bullish EMA21/EMA50 cross detected — exit short signal")
            return SignalDirection.NEUTRAL

        # RSI divergence check
        rsi_div = self._detect_rsi_divergence(primary_ind)
        if rsi_div is not None:
            logger.info("RSI divergence detected — exit signal", extra={"type": rsi_div})
            return SignalDirection.NEUTRAL

        return None

    def _detect_rsi_divergence(
        self,
        ind: pd.DataFrame,
        lookback: int = 10,
    ) -> Optional[str]:
        """Detect RSI divergence over the last ``lookback`` bars.

        A bearish divergence occurs when price makes a higher high but
        RSI makes a lower high.  A bullish divergence is the mirror.

        Args:
            ind: Indicator DataFrame (must contain ``rsi_14``).
            lookback: Number of bars to inspect.

        Returns:
            ``'bearish'`` or ``'bullish'`` if divergence found, else ``None``.
        """
        if "rsi_14" not in ind.columns or "close" not in ind.columns:
            # If close not in indicators, divergence cannot be checked here
            if "rsi_14" not in ind.columns:
                return None
            return None

        if len(ind) < lookback + 1:
            return None

        recent = ind.iloc[-(lookback + 1):]
        rsi_vals = recent["rsi_14"].values

        # Use the indicator frame's own close if available; otherwise skip
        if "close" not in ind.columns:
            return None
        close_vals = recent["close"].values

        # Find two most recent local highs in close
        highs_idx = []
        for i in range(1, len(close_vals) - 1):
            if close_vals[i] > close_vals[i - 1] and close_vals[i] > close_vals[i + 1]:
                highs_idx.append(i)

        if len(highs_idx) >= 2:
            i1, i2 = highs_idx[-2], highs_idx[-1]
            if close_vals[i2] > close_vals[i1] and rsi_vals[i2] < rsi_vals[i1]:
                return "bearish"

        # Find two most recent local lows in close
        lows_idx = []
        for i in range(1, len(close_vals) - 1):
            if close_vals[i] < close_vals[i - 1] and close_vals[i] < close_vals[i + 1]:
                lows_idx.append(i)

        if len(lows_idx) >= 2:
            i1, i2 = lows_idx[-2], lows_idx[-1]
            if close_vals[i2] < close_vals[i1] and rsi_vals[i2] > rsi_vals[i1]:
                return "bullish"

        return None

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _compute_score(
        self,
        primary_ind: pd.DataFrame,
        confirm_ind: pd.DataFrame,
        direction: SignalDirection,
    ) -> int:
        """Compute conviction score for a signal.

        Starts at ``base_score`` (50) and adds up to +30 bonus points:
        - +10 for strong MACD (histogram > 2x previous bar).
        - +10 for ADX > ``adx_threshold``.
        - +10 for 1d EMA alignment confirming direction.

        The total is clamped to [0, 100].

        Args:
            primary_ind: Primary timeframe indicators.
            confirm_ind: Confirmation (1d) timeframe indicators.
            direction: Trade direction being scored.

        Returns:
            Integer score in [0, 100].
        """
        score = self.base_score

        # +10: Strong MACD — histogram > 2x previous bar
        macd_hist = self._safe_last(primary_ind, "macd_histogram")
        macd_hist_prev = self._safe_last(primary_ind, "macd_histogram", offset=1)
        if macd_hist is not None and macd_hist_prev is not None and macd_hist_prev != 0:
            if direction == SignalDirection.LONG:
                if macd_hist > 0 and macd_hist > 2.0 * abs(macd_hist_prev):
                    score += 10
                    logger.debug("Score +10: strong MACD histogram expansion")
            else:
                if macd_hist < 0 and abs(macd_hist) > 2.0 * abs(macd_hist_prev):
                    score += 10
                    logger.debug("Score +10: strong MACD histogram expansion (short)")

        # +10: ADX > threshold
        adx = self._safe_last(primary_ind, "adx")
        if adx is not None and adx > self.adx_threshold:
            score += 10
            logger.debug("Score +10: ADX %.1f > %.1f", adx, self.adx_threshold)

        # +10: 1d EMA alignment confirms direction
        ema21_1d = self._safe_last(confirm_ind, "ema_21")
        ema50_1d = self._safe_last(confirm_ind, "ema_50")
        ema200_1d = self._safe_last(confirm_ind, "ema_200")
        if ema21_1d is not None and ema50_1d is not None and ema200_1d is not None:
            if direction == SignalDirection.LONG and ema21_1d > ema50_1d > ema200_1d:
                score += 10
                logger.debug("Score +10: 1d bullish EMA alignment")
            elif direction == SignalDirection.SHORT and ema21_1d < ema50_1d < ema200_1d:
                score += 10
                logger.debug("Score +10: 1d bearish EMA alignment")

        return int(np.clip(score, 0, 100))

    # ------------------------------------------------------------------
    # Signal construction
    # ------------------------------------------------------------------

    def _build_signal(
        self,
        symbol: str,
        direction: SignalDirection,
        score: int,
        entry_data: pd.DataFrame,
        primary_ind: pd.DataFrame,
        extras: dict[str, Any],
    ) -> TradeSignal:
        """Build a complete TradeSignal with entry, stop, and take-profit levels.

        Args:
            symbol: Trading pair.
            direction: Long or short.
            score: Computed conviction score.
            entry_data: OHLCV DataFrame for the entry timeframe.
            primary_ind: Primary timeframe indicators (used for ATR).
            extras: Additional metadata from condition checks.

        Returns:
            Fully populated ``TradeSignal``.
        """
        entry_price = float(entry_data["close"].iloc[-1])
        atr = self._safe_last(primary_ind, "atr")
        atr_val = float(atr) if atr is not None else entry_price * 0.015

        # Swing-based structure stop
        swing_level: Optional[float] = None
        if direction == SignalDirection.LONG:
            swing_level = float(entry_data["low"].rolling(20).min().iloc[-1])
        else:
            swing_level = float(entry_data["high"].rolling(20).max().iloc[-1])

        stop_loss = self.compute_stop_loss(
            entry_price, direction, atr_val, swing_level=swing_level
        )
        tp1, tp2, tp3 = self.compute_take_profits(entry_price, stop_loss, direction)
        confidence = score / 100.0

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
                "regime_required": self.active_regimes,
                **extras,
            },
        )

        logger.info(
            "Signal generated",
            extra={
                "symbol": symbol,
                "direction": direction.value,
                "score": signal.score,
                "entry": entry_price,
                "stop": stop_loss,
                "tp1": tp1,
                "r_multiple": signal.r_multiple(),
            },
        )
        return signal

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_last(
        df: pd.DataFrame,
        column: str,
        offset: int = 0,
    ) -> Optional[float]:
        """Safely retrieve the last (or offset-from-last) value of a column.

        Args:
            df: DataFrame to read from.
            column: Column name.
            offset: How many rows back from the last row (0 = last row).

        Returns:
            The float value, or ``None`` if unavailable or NaN.
        """
        if column not in df.columns:
            return None
        idx = -(1 + offset)
        if abs(idx) > len(df):
            return None
        val = df[column].iloc[idx]
        if pd.isna(val):
            return None
        return float(val)

    @staticmethod
    def _compute_volume_zscore(
        data: pd.DataFrame,
        window: int = 20,
    ) -> Optional[float]:
        """Compute volume z-score from raw OHLCV data as a fallback.

        Args:
            data: OHLCV DataFrame with a ``volume`` column.
            window: Rolling window for mean/std computation.

        Returns:
            Z-score of the latest volume bar, or ``None`` if insufficient data.
        """
        if "volume" not in data.columns or len(data) < window:
            return None
        vol = data["volume"].astype(float)
        rolling_mean = vol.rolling(window).mean().iloc[-1]
        rolling_std = vol.rolling(window).std().iloc[-1]
        if pd.isna(rolling_std) or rolling_std == 0:
            return None
        return float((vol.iloc[-1] - rolling_mean) / rolling_std)
