"""Mean Reversion strategy for the APEX Crypto Trading System.

Trades counter-trend bounces in ranging markets using RSI extremes,
Bollinger Band touches, and Chaikin Money Flow confirmation, while
filtering against the higher-timeframe trend to avoid catching falling
knives.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import pandas as pd

from apex_crypto.core.logging import get_logger, log_with_data
from apex_crypto.core.strategies.base import (
    BaseStrategy,
    SignalDirection,
    TradeSignal,
)

logger = get_logger("strategies.mean_reversion")


class MeanReversionStrategy(BaseStrategy):
    """Mean-reversion strategy for range-bound markets.

    Enters long when price is oversold (RSI < 28, touching lower Bollinger
    Band) with positive money flow, and short when overbought (RSI > 72,
    touching upper BB) with negative money flow.  All entries are filtered
    against the 1d trend to avoid counter-trend trades in trending markets.

    Attributes:
        name: Strategy identifier.
        active_regimes: Only active in RANGING markets.
        primary_timeframe: Timeframe for signal detection.
        entry_timeframe: Lower timeframe for precise entry.
    """

    name: str = "mean_reversion"
    active_regimes: list[str] = ["RANGING"]
    primary_timeframe: str = "1h"
    confirmation_timeframe: str = "1d"
    entry_timeframe: str = "15m"

    # ------------------------------------------------------------------
    # Thresholds
    # ------------------------------------------------------------------
    _DEFAULT_RSI_OVERSOLD: float = 28.0
    _DEFAULT_RSI_OVERBOUGHT: float = 72.0
    _DEFAULT_RSI_EXTREME_LOW: float = 20.0
    _DEFAULT_RSI_EXTREME_HIGH: float = 80.0
    _DEFAULT_RSI_EXIT: float = 50.0
    _DEFAULT_BASE_SCORE: int = 60

    def __init__(self, config: dict) -> None:
        """Initialize MeanReversionStrategy.

        Args:
            config: Strategy-specific configuration dictionary.  Recognised
                keys (all optional):
                - rsi_oversold / rsi_overbought: RSI entry thresholds.
                - rsi_extreme_low / rsi_extreme_high: Extreme RSI bonus levels.
                - base_score: Starting conviction score.
        """
        super().__init__(config)
        self.rsi_oversold: float = config.get("rsi_oversold", self._DEFAULT_RSI_OVERSOLD)
        self.rsi_overbought: float = config.get("rsi_overbought", self._DEFAULT_RSI_OVERBOUGHT)
        self.rsi_extreme_low: float = config.get("rsi_extreme_low", self._DEFAULT_RSI_EXTREME_LOW)
        self.rsi_extreme_high: float = config.get("rsi_extreme_high", self._DEFAULT_RSI_EXTREME_HIGH)
        self.rsi_exit: float = config.get("rsi_exit", self._DEFAULT_RSI_EXIT)
        self.base_score: int = config.get("base_score", self._DEFAULT_BASE_SCORE)
        logger.info(
            "MeanReversionStrategy configured",
            extra={
                "rsi_oversold": self.rsi_oversold,
                "rsi_overbought": self.rsi_overbought,
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
        """Generate a mean-reversion trading signal.

        Checks for oversold/overbought extremes on the primary timeframe,
        confirms with Bollinger Band position and Chaikin Money Flow, and
        filters against the daily trend.

        Args:
            symbol: Trading pair symbol (e.g., ``'ETH/USDT'``).
            data: OHLCV DataFrames keyed by timeframe string.
            indicators: Pre-computed indicator DataFrames keyed by timeframe.
            regime: Current market regime string.
            alt_data: Optional alternative data (unused by this strategy).

        Returns:
            A ``TradeSignal`` with direction and score, or NEUTRAL.
        """
        if not self.is_active(regime):
            logger.debug("Strategy inactive for regime %s", regime)
            return self._neutral_signal(symbol)

        # Validate required timeframes
        for tf in (self.primary_timeframe, self.entry_timeframe):
            if tf not in data or tf not in indicators:
                logger.warning("Missing timeframe data", extra={"timeframe": tf})
                return self._neutral_signal(symbol)

        primary_data = data[self.primary_timeframe]
        primary_ind = indicators[self.primary_timeframe]
        entry_data = data[self.entry_timeframe]

        # Daily trend filter (optional — allow signal if 1d data missing)
        daily_ind = indicators.get("1d")
        daily_trend_bullish = self._daily_trend_is_bullish(daily_ind)
        daily_trend_bearish = self._daily_trend_is_bearish(daily_ind)

        if primary_data.empty or primary_ind.empty:
            return self._neutral_signal(symbol)

        # --- Long entry ---
        long_ok, long_extras = self._check_long_conditions(
            primary_data, primary_ind, daily_trend_bearish
        )
        if long_ok:
            score = self._compute_score(primary_ind, SignalDirection.LONG)
            return self._build_signal(
                symbol, SignalDirection.LONG, score, entry_data, primary_data, primary_ind, long_extras
            )

        # --- Short entry ---
        short_ok, short_extras = self._check_short_conditions(
            primary_data, primary_ind, daily_trend_bullish
        )
        if short_ok:
            score = self._compute_score(primary_ind, SignalDirection.SHORT)
            return self._build_signal(
                symbol, SignalDirection.SHORT, score, entry_data, primary_data, primary_ind, short_extras
            )

        # --- Exit check ---
        exit_signal = self._check_exit_conditions(primary_data, primary_ind)
        if exit_signal is not None:
            return TradeSignal(
                symbol=symbol,
                direction=SignalDirection.NEUTRAL,
                score=0,
                strategy=self.name,
                timeframe=self.primary_timeframe,
                metadata={"signal_type": "exit", "exit_reason": exit_signal},
            )

        return self._neutral_signal(symbol)

    # ------------------------------------------------------------------
    # Condition checks
    # ------------------------------------------------------------------

    def _check_long_conditions(
        self,
        primary_data: pd.DataFrame,
        primary_ind: pd.DataFrame,
        daily_bearish: bool,
    ) -> tuple[bool, dict[str, Any]]:
        """Evaluate long-entry conditions.

        Conditions (ALL must be true):
        1. RSI(14) < ``rsi_oversold`` (28 by default).
        2. Price at or below lower Bollinger Band.
        3. Chaikin Money Flow > 0 (money flowing in despite low price).
        4. 1d trend must NOT be bearish (EMA50 < EMA200 on daily).

        Args:
            primary_data: OHLCV DataFrame for the primary timeframe.
            primary_ind: Indicator DataFrame for the primary timeframe.
            daily_bearish: True if the daily trend is bearish.

        Returns:
            Tuple of (conditions_met, metadata_dict).
        """
        extras: dict[str, Any] = {}

        # Filter: do not go long if daily trend is bearish
        if daily_bearish:
            logger.debug("Long blocked by bearish daily trend")
            return False, extras

        # 1. RSI oversold
        rsi = self._safe_last(primary_ind, "rsi_14")
        if rsi is None or rsi >= self.rsi_oversold:
            return False, extras
        extras["rsi"] = float(rsi)

        # 2. Price at or below lower Bollinger Band
        close = self._safe_last(primary_data, "close")
        bb_lower = self._safe_last(primary_ind, "bb_lower")
        if close is None or bb_lower is None:
            return False, extras
        if close > bb_lower:
            return False, extras
        extras["close"] = float(close)
        extras["bb_lower"] = float(bb_lower)

        # 3. CMF > 0
        cmf = self._safe_last(primary_ind, "cmf")
        if cmf is None or cmf <= 0:
            return False, extras
        extras["cmf"] = float(cmf)

        logger.info("Long mean-reversion conditions met", extra={"conditions": extras})
        return True, extras

    def _check_short_conditions(
        self,
        primary_data: pd.DataFrame,
        primary_ind: pd.DataFrame,
        daily_bullish: bool,
    ) -> tuple[bool, dict[str, Any]]:
        """Evaluate short-entry conditions.

        Conditions (ALL must be true):
        1. RSI(14) > ``rsi_overbought`` (72 by default).
        2. Price at or above upper Bollinger Band.
        3. Chaikin Money Flow < 0 (money flowing out despite high price).
        4. 1d trend must NOT be bullish (EMA50 > EMA200 on daily).

        Args:
            primary_data: OHLCV DataFrame for the primary timeframe.
            primary_ind: Indicator DataFrame for the primary timeframe.
            daily_bullish: True if the daily trend is bullish.

        Returns:
            Tuple of (conditions_met, metadata_dict).
        """
        extras: dict[str, Any] = {}

        # Filter: do not go short if daily trend is bullish
        if daily_bullish:
            logger.debug("Short blocked by bullish daily trend")
            return False, extras

        # 1. RSI overbought
        rsi = self._safe_last(primary_ind, "rsi_14")
        if rsi is None or rsi <= self.rsi_overbought:
            return False, extras
        extras["rsi"] = float(rsi)

        # 2. Price at or above upper Bollinger Band
        close = self._safe_last(primary_data, "close")
        bb_upper = self._safe_last(primary_ind, "bb_upper")
        if close is None or bb_upper is None:
            return False, extras
        if close < bb_upper:
            return False, extras
        extras["close"] = float(close)
        extras["bb_upper"] = float(bb_upper)

        # 3. CMF < 0
        cmf = self._safe_last(primary_ind, "cmf")
        if cmf is None or cmf >= 0:
            return False, extras
        extras["cmf"] = float(cmf)

        logger.info("Short mean-reversion conditions met", extra={"conditions": extras})
        return True, extras

    # ------------------------------------------------------------------
    # Daily trend helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _daily_trend_is_bullish(daily_ind: Optional[pd.DataFrame]) -> bool:
        """Return True if the 1d EMA50 > EMA200 (bullish daily trend).

        Args:
            daily_ind: 1d indicator DataFrame, or None.

        Returns:
            True when the daily trend is confirmed bullish.
        """
        if daily_ind is None or daily_ind.empty:
            return False
        ema50 = MeanReversionStrategy._safe_last(daily_ind, "ema_50")
        ema200 = MeanReversionStrategy._safe_last(daily_ind, "ema_200")
        if ema50 is None or ema200 is None:
            return False
        return ema50 > ema200

    @staticmethod
    def _daily_trend_is_bearish(daily_ind: Optional[pd.DataFrame]) -> bool:
        """Return True if the 1d EMA50 < EMA200 (bearish daily trend).

        Args:
            daily_ind: 1d indicator DataFrame, or None.

        Returns:
            True when the daily trend is confirmed bearish.
        """
        if daily_ind is None or daily_ind.empty:
            return False
        ema50 = MeanReversionStrategy._safe_last(daily_ind, "ema_50")
        ema200 = MeanReversionStrategy._safe_last(daily_ind, "ema_200")
        if ema50 is None or ema200 is None:
            return False
        return ema50 < ema200

    # ------------------------------------------------------------------
    # Exit logic
    # ------------------------------------------------------------------

    def _check_exit_conditions(
        self,
        primary_data: pd.DataFrame,
        primary_ind: pd.DataFrame,
    ) -> Optional[str]:
        """Check for mean-reversion exit conditions.

        Exit triggers:
        - RSI crosses 50 (mean reverted).
        - Price reaches middle Bollinger Band.

        Args:
            primary_data: OHLCV DataFrame for the primary timeframe.
            primary_ind: Indicator DataFrame for the primary timeframe.

        Returns:
            Exit reason string, or ``None`` if no exit warranted.
        """
        rsi = self._safe_last(primary_ind, "rsi_14")
        rsi_prev = self._safe_last(primary_ind, "rsi_14", offset=1)

        # RSI crossing 50
        if rsi is not None and rsi_prev is not None:
            crossed_up = rsi_prev < self.rsi_exit and rsi >= self.rsi_exit
            crossed_down = rsi_prev > self.rsi_exit and rsi <= self.rsi_exit
            if crossed_up or crossed_down:
                logger.info("Exit: RSI crossed %.0f", self.rsi_exit)
                return "rsi_cross_50"

        # Price reaching middle BB
        close = self._safe_last(primary_data, "close")
        bb_mid = self._safe_last(primary_ind, "bb_middle")
        close_prev = self._safe_last(primary_data, "close", offset=1)
        if close is not None and bb_mid is not None and close_prev is not None:
            # Long exit: price crosses up through middle BB
            if close_prev < bb_mid and close >= bb_mid:
                logger.info("Exit: price reached middle Bollinger Band (long)")
                return "middle_bb_reached"
            # Short exit: price crosses down through middle BB
            if close_prev > bb_mid and close <= bb_mid:
                logger.info("Exit: price reached middle Bollinger Band (short)")
                return "middle_bb_reached"

        return None

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _compute_score(
        self,
        primary_ind: pd.DataFrame,
        direction: SignalDirection,
    ) -> int:
        """Compute conviction score for a mean-reversion signal.

        Starts at ``base_score`` (60) and adds bonus points:
        - +15 for extreme RSI (< 20 for longs, > 80 for shorts).
        - +10 for MFI confirmation (MFI oversold/overbought aligning).

        The total is clamped to [0, 100].

        Args:
            primary_ind: Primary timeframe indicators.
            direction: Trade direction being scored.

        Returns:
            Integer score in [0, 100].
        """
        score = self.base_score

        rsi = self._safe_last(primary_ind, "rsi_14")
        if rsi is not None:
            if direction == SignalDirection.LONG and rsi < self.rsi_extreme_low:
                score += 15
                logger.debug("Score +15: extreme oversold RSI %.1f", rsi)
            elif direction == SignalDirection.SHORT and rsi > self.rsi_extreme_high:
                score += 15
                logger.debug("Score +15: extreme overbought RSI %.1f", rsi)

        # MFI confirmation
        mfi = self._safe_last(primary_ind, "mfi")
        if mfi is not None:
            if direction == SignalDirection.LONG and mfi < 30.0:
                score += 10
                logger.debug("Score +10: MFI oversold confirmation %.1f", mfi)
            elif direction == SignalDirection.SHORT and mfi > 70.0:
                score += 10
                logger.debug("Score +10: MFI overbought confirmation %.1f", mfi)

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
        primary_data: pd.DataFrame,
        primary_ind: pd.DataFrame,
        extras: dict[str, Any],
    ) -> TradeSignal:
        """Build a complete TradeSignal with entry, stop, and take-profit levels.

        For mean-reversion trades the first take-profit target is the middle
        Bollinger Band (a natural mean-reversion target).

        Args:
            symbol: Trading pair.
            direction: Long or short.
            score: Computed conviction score.
            entry_data: OHLCV DataFrame for the entry timeframe.
            primary_data: OHLCV DataFrame for the primary timeframe.
            primary_ind: Primary timeframe indicators (ATR, BB).
            extras: Additional metadata from condition checks.

        Returns:
            Fully populated ``TradeSignal``.
        """
        entry_price = float(entry_data["close"].iloc[-1])
        atr = self._safe_last(primary_ind, "atr")
        atr_val = float(atr) if atr is not None else entry_price * 0.01

        # Structure-based stop
        swing_level: Optional[float] = None
        if direction == SignalDirection.LONG:
            swing_level = float(primary_data["low"].rolling(10).min().iloc[-1])
        else:
            swing_level = float(primary_data["high"].rolling(10).max().iloc[-1])

        stop_loss = self.compute_stop_loss(
            entry_price, direction, atr_val, swing_level=swing_level
        )

        # Mean-reversion targets: middle BB as TP1, upper/lower BB as TP2
        bb_mid = self._safe_last(primary_ind, "bb_middle")
        bb_upper = self._safe_last(primary_ind, "bb_upper")
        bb_lower = self._safe_last(primary_ind, "bb_lower")

        risk = abs(entry_price - stop_loss)
        if direction == SignalDirection.LONG:
            tp1 = float(bb_mid) if bb_mid is not None else entry_price + risk * 1.5
            tp2 = float(bb_upper) if bb_upper is not None else entry_price + risk * 2.5
            tp3 = entry_price + risk * 4.0
        else:
            tp1 = float(bb_mid) if bb_mid is not None else entry_price - risk * 1.5
            tp2 = float(bb_lower) if bb_lower is not None else entry_price - risk * 2.5
            tp3 = entry_price - risk * 4.0

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


# ══════════════════════════════════════════════════════════════════════
# SIMONS UPGRADE: Statistical Mean Reversion with Kalman Filter
# ══════════════════════════════════════════════════════════════════════


class StatisticalMeanReversion(BaseStrategy):
    """Statistical mean reversion using Kalman Filter dynamic mean estimation.

    Combines Ornstein-Uhlenbeck half-life validation with ADX trending
    market filtering and z-score based entry/exit signals.

    Features:
    - Kalman Filter for adaptive mean estimation
    - OU half-life validation (only trade when mean reversion is statistically likely)
    - ADX < 30 filter to avoid trending markets
    - Z-score based entry (>2.0) and exit (<0.5)

    Attributes:
        name: Strategy identifier.
        active_regimes: Only active in RANGING markets.
    """

    name: str = "stat_mean_reversion"
    active_regimes: list[str] = ["RANGING"]
    primary_timeframe: str = "1h"
    confirmation_timeframe: str = "4h"
    entry_timeframe: str = "15m"

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    _DEFAULT_ZSCORE_ENTRY: float = 2.0
    _DEFAULT_ZSCORE_EXIT: float = 0.5
    _DEFAULT_ZSCORE_STOP: float = 3.5
    _DEFAULT_ADX_MAX: float = 30.0
    _DEFAULT_KALMAN_Q: float = 0.001    # process noise
    _DEFAULT_KALMAN_R: float = 0.1      # measurement noise
    _DEFAULT_MIN_HALFLIFE: float = 2.0  # hours
    _DEFAULT_MAX_HALFLIFE: float = 72.0 # hours
    _DEFAULT_LOOKBACK: int = 480        # 20 days in hours
    _DEFAULT_BASE_SCORE: int = 60

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.zscore_entry: float = config.get("zscore_entry", self._DEFAULT_ZSCORE_ENTRY)
        self.zscore_exit: float = config.get("zscore_exit", self._DEFAULT_ZSCORE_EXIT)
        self.zscore_stop: float = config.get("zscore_stop", self._DEFAULT_ZSCORE_STOP)
        self.adx_max: float = config.get("adx_max", self._DEFAULT_ADX_MAX)
        self.kalman_q: float = config.get("kalman_q", self._DEFAULT_KALMAN_Q)
        self.kalman_r: float = config.get("kalman_r", self._DEFAULT_KALMAN_R)
        self.min_halflife: float = config.get("min_halflife", self._DEFAULT_MIN_HALFLIFE)
        self.max_halflife: float = config.get("max_halflife", self._DEFAULT_MAX_HALFLIFE)
        self.lookback: int = config.get("lookback", self._DEFAULT_LOOKBACK)
        self.base_score: int = config.get("base_score", self._DEFAULT_BASE_SCORE)

        # Kalman state per symbol
        self._kalman_state: dict[str, dict[str, float]] = {}

        log_with_data(logger, "info", "StatisticalMeanReversion initialized", {
            "zscore_entry": self.zscore_entry,
            "adx_max": self.adx_max,
            "kalman_q": self.kalman_q,
        })

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
        """Generate a statistical mean-reversion signal."""
        if not self.is_active(regime):
            return self._neutral_signal(symbol)

        if self.primary_timeframe not in data or self.primary_timeframe not in indicators:
            return self._neutral_signal(symbol)

        primary_data = data[self.primary_timeframe]
        primary_ind = indicators[self.primary_timeframe]

        if primary_data.empty or len(primary_data) < self.lookback:
            return self._neutral_signal(symbol)

        # ADX filter: reject if market is trending
        adx = self._safe_last(primary_ind, "adx")
        if adx is not None and adx > self.adx_max:
            logger.debug("ADX filter: market trending", extra={
                "symbol": symbol, "adx": round(adx, 1),
            })
            return self._neutral_signal(symbol)

        close = primary_data["close"].iloc[-self.lookback:]

        # Kalman Filter mean estimation
        kalman_mean = self._kalman_filter(symbol, close)

        # Compute z-score
        residuals = close.values - kalman_mean
        residual_std = np.std(residuals)
        if residual_std <= 0:
            return self._neutral_signal(symbol)

        current_zscore = residuals[-1] / residual_std

        # OU half-life validation
        halflife = self._ou_halflife(residuals)
        if halflife is None:
            return self._neutral_signal(symbol)

        if halflife < self.min_halflife or halflife > self.max_halflife:
            logger.debug("OU half-life out of range", extra={
                "symbol": symbol, "halflife": round(halflife, 2),
            })
            return self._neutral_signal(symbol)

        # Z-score based signal generation
        abs_z = abs(current_zscore)

        # Stop condition
        if abs_z > self.zscore_stop:
            return self._neutral_signal(symbol)

        # Exit condition
        if abs_z < self.zscore_exit:
            return self._neutral_signal(symbol)

        # Entry condition
        if abs_z < self.zscore_entry:
            return self._neutral_signal(symbol)

        # Positive z-score = price above mean → short (expect reversion down)
        # Negative z-score = price below mean → long (expect reversion up)
        if current_zscore > self.zscore_entry:
            direction = SignalDirection.SHORT
        elif current_zscore < -self.zscore_entry:
            direction = SignalDirection.LONG
        else:
            return self._neutral_signal(symbol)

        # Score
        score = self._compute_score(current_zscore, halflife, adx)

        # Build signal
        entry_tf = self.entry_timeframe if self.entry_timeframe in data else self.primary_timeframe
        entry_data = data[entry_tf]

        return self._build_signal(
            symbol, direction, score, entry_data, primary_data, primary_ind,
            {
                "zscore": round(current_zscore, 4),
                "kalman_mean": round(kalman_mean[-1], 2),
                "halflife": round(halflife, 2),
                "adx": round(adx, 1) if adx is not None else None,
            },
        )

    # ------------------------------------------------------------------
    # Kalman Filter
    # ------------------------------------------------------------------

    def _kalman_filter(self, symbol: str, close: pd.Series) -> np.ndarray:
        """Apply 1-D Kalman filter to estimate dynamic mean.

        State: x_t (estimated mean price)
        Measurement: z_t = close_t

        Returns:
            Array of Kalman-filtered mean estimates.
        """
        prices = close.values
        n = len(prices)
        filtered = np.zeros(n)

        # Initialize or retrieve state
        state = self._kalman_state.get(symbol, {
            "x": prices[0],
            "P": 1.0,
        })

        x = state["x"]
        P = state["P"]
        Q = self.kalman_q
        R = self.kalman_r

        for i in range(n):
            # Predict
            x_pred = x
            P_pred = P + Q

            # Update
            K = P_pred / (P_pred + R)
            x = x_pred + K * (prices[i] - x_pred)
            P = (1 - K) * P_pred

            filtered[i] = x

        # Save state
        self._kalman_state[symbol] = {"x": x, "P": P}

        return filtered

    # ------------------------------------------------------------------
    # Ornstein-Uhlenbeck half-life
    # ------------------------------------------------------------------

    @staticmethod
    def _ou_halflife(residuals: np.ndarray) -> Optional[float]:
        """Estimate OU half-life from residuals.

        Fits: delta_r = a + b * r_lag
        Half-life = -ln(2) / b
        """
        if len(residuals) < 10:
            return None

        lag = residuals[:-1]
        delta = np.diff(residuals)

        X = np.column_stack([np.ones(len(lag)), lag])
        try:
            beta = np.linalg.lstsq(X, delta, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None

        b = beta[1]
        if b >= 0:
            return None

        halflife = -math.log(2) / b
        return float(halflife)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _compute_score(
        self,
        zscore: float,
        halflife: float,
        adx: Optional[float],
    ) -> int:
        """Compute conviction score."""
        score = self.base_score

        # Z-score magnitude bonus
        abs_z = abs(zscore)
        if abs_z > 3.0:
            score += 20
        elif abs_z > 2.5:
            score += 10

        # Good half-life bonus (faster mean reversion = better)
        if halflife < 12.0:
            score += 10
        elif halflife < 24.0:
            score += 5

        # Low ADX bonus (very flat market = ideal for MR)
        if adx is not None and adx < 20.0:
            score += 5

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
        primary_data: pd.DataFrame,
        primary_ind: pd.DataFrame,
        extras: dict[str, Any],
    ) -> TradeSignal:
        """Build a TradeSignal for statistical mean reversion."""
        entry_price = float(entry_data["close"].iloc[-1])
        atr = self._safe_last(primary_ind, "atr")
        atr_val = float(atr) if atr is not None else entry_price * 0.01

        if direction == SignalDirection.LONG:
            swing_level = float(primary_data["low"].rolling(10).min().iloc[-1])
        else:
            swing_level = float(primary_data["high"].rolling(10).max().iloc[-1])

        stop_loss = self.compute_stop_loss(
            entry_price, direction, atr_val, swing_level=swing_level,
        )

        # Mean-reversion targets based on Kalman mean
        kalman_mean = extras.get("kalman_mean", entry_price)
        risk = abs(entry_price - stop_loss)

        if direction == SignalDirection.LONG:
            tp1 = kalman_mean if kalman_mean > entry_price else entry_price + risk * 1.5
            tp2 = entry_price + risk * 2.5
            tp3 = entry_price + risk * 4.0
        else:
            tp1 = kalman_mean if kalman_mean < entry_price else entry_price - risk * 1.5
            tp2 = entry_price - risk * 2.5
            tp3 = entry_price - risk * 4.0

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

        log_with_data(logger, "info", "Stat MR signal generated", {
            "symbol": symbol,
            "direction": direction.value,
            "score": signal.score,
            "zscore": extras.get("zscore"),
            "halflife": extras.get("halflife"),
        })

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
        if column not in df.columns:
            return None
        idx = -(1 + offset)
        if abs(idx) > len(df):
            return None
        val = df[column].iloc[idx]
        if pd.isna(val):
            return None
        return float(val)
