"""Breakout strategy for the APEX Crypto Trading System.

Detects consolidation ranges on the 4h timeframe, waits for a decisive
breakout with volume confirmation, filters false breakouts by requiring
consecutive closes beyond the level, and optionally waits for a retest
before entry.
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

logger = get_logger("strategies.breakout")


class BreakoutStrategy(BaseStrategy):
    """Breakout strategy for transitional and range-bound markets.

    Identifies tight consolidation ranges (Bollinger Band bandwidth below
    the 10th percentile over a look-back window), then signals when price
    breaks out with elevated volume and two consecutive closes beyond the
    range boundary.

    Attributes:
        name: Strategy identifier.
        active_regimes: Active during RANGING, WEAK_BULL, and WEAK_BEAR.
        primary_timeframe: Timeframe for consolidation and breakout detection.
        entry_timeframe: Lower timeframe for entry timing.
    """

    name: str = "breakout"
    active_regimes: list[str] = ["RANGING", "WEAK_BULL", "WEAK_BEAR"]
    primary_timeframe: str = "4h"
    confirmation_timeframe: str = "1d"
    entry_timeframe: str = "1h"

    # ------------------------------------------------------------------
    # Configuration defaults
    # ------------------------------------------------------------------
    _DEFAULT_CONSOLIDATION_BARS: int = 20
    _DEFAULT_BB_WIDTH_PERCENTILE: float = 10.0
    _DEFAULT_BB_WIDTH_LOOKBACK: int = 100
    _DEFAULT_VOLUME_ZSCORE_THRESHOLD: float = 2.5
    _DEFAULT_CONSECUTIVE_CLOSES: int = 2
    _DEFAULT_WAIT_FOR_RETEST: bool = False
    _DEFAULT_RETEST_TOLERANCE_PCT: float = 0.003
    _DEFAULT_BASE_SCORE: int = 55
    _DEFAULT_ADX_THRESHOLD: float = 25.0

    def __init__(self, config: dict) -> None:
        """Initialize BreakoutStrategy.

        Args:
            config: Strategy-specific configuration dictionary.  Recognised
                keys (all optional):
                - consolidation_bars: Minimum bars for range detection.
                - bb_width_percentile: BB bandwidth percentile threshold.
                - bb_width_lookback: Look-back for percentile computation.
                - volume_zscore_threshold: Min volume z-score on breakout.
                - consecutive_closes: Required closes beyond the level.
                - wait_for_retest: Whether to wait for a retest of the
                  breakout level before entering.
                - retest_tolerance_pct: How close price must come to the
                  breakout level to count as a retest.
                - base_score: Starting conviction score.
        """
        super().__init__(config)
        self.consolidation_bars: int = config.get(
            "consolidation_bars", self._DEFAULT_CONSOLIDATION_BARS
        )
        self.bb_width_percentile: float = config.get(
            "bb_width_percentile", self._DEFAULT_BB_WIDTH_PERCENTILE
        )
        self.bb_width_lookback: int = config.get(
            "bb_width_lookback", self._DEFAULT_BB_WIDTH_LOOKBACK
        )
        self.volume_zscore_threshold: float = config.get(
            "volume_zscore_threshold", self._DEFAULT_VOLUME_ZSCORE_THRESHOLD
        )
        self.consecutive_closes: int = config.get(
            "consecutive_closes", self._DEFAULT_CONSECUTIVE_CLOSES
        )
        self.wait_for_retest: bool = config.get(
            "wait_for_retest", self._DEFAULT_WAIT_FOR_RETEST
        )
        self.retest_tolerance_pct: float = config.get(
            "retest_tolerance_pct", self._DEFAULT_RETEST_TOLERANCE_PCT
        )
        self.base_score: int = config.get("base_score", self._DEFAULT_BASE_SCORE)
        self.adx_threshold: float = config.get("adx_threshold", self._DEFAULT_ADX_THRESHOLD)
        logger.info(
            "BreakoutStrategy configured",
            extra={
                "consolidation_bars": self.consolidation_bars,
                "bb_width_percentile": self.bb_width_percentile,
                "volume_zscore_threshold": self.volume_zscore_threshold,
                "wait_for_retest": self.wait_for_retest,
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
        """Generate a breakout trading signal.

        Steps:
        1. Detect a consolidation range on the primary timeframe.
        2. Determine if price has broken above or below that range.
        3. Confirm with volume z-score and consecutive closes.
        4. Optionally wait for a retest of the breakout level.
        5. Score and build the signal.

        Args:
            symbol: Trading pair symbol (e.g., ``'BTC/USDT'``).
            data: OHLCV DataFrames keyed by timeframe string.
            indicators: Pre-computed indicator DataFrames keyed by timeframe.
            regime: Current market regime string.
            alt_data: Optional alternative data (unused by this strategy).

        Returns:
            A ``TradeSignal`` reflecting the assessment.
        """
        if not self.is_active(regime):
            logger.debug("Strategy inactive for regime %s", regime)
            return self._neutral_signal(symbol)

        for tf in (self.primary_timeframe, self.entry_timeframe):
            if tf not in data or tf not in indicators:
                logger.warning("Missing timeframe data", extra={"timeframe": tf})
                return self._neutral_signal(symbol)

        primary_data = data[self.primary_timeframe]
        primary_ind = indicators[self.primary_timeframe]
        entry_data = data[self.entry_timeframe]
        confirm_ind = indicators.get(self.confirmation_timeframe)

        if len(primary_data) < self.consolidation_bars + 5:
            logger.debug("Insufficient primary data bars")
            return self._neutral_signal(symbol)

        # Step 1: Check for consolidation
        is_consolidating, range_high, range_low = self._detect_consolidation(
            primary_data, primary_ind
        )
        if not is_consolidating:
            logger.debug("No consolidation detected for %s", symbol)
            return self._neutral_signal(symbol)

        # Step 2 & 3: Check for breakout with volume and consecutive closes
        direction, breakout_extras = self._detect_breakout(
            primary_data, primary_ind, range_high, range_low
        )
        if direction is None:
            return self._neutral_signal(symbol)

        # Step 4: Optional retest filter
        if self.wait_for_retest:
            retest_level = range_high if direction == SignalDirection.LONG else range_low
            if not self._check_retest(primary_data, retest_level, direction):
                logger.debug("Waiting for retest of breakout level %.4f", retest_level)
                return self._neutral_signal(symbol)
            breakout_extras["retest_confirmed"] = True

        # Step 5: Score and build
        score = self._compute_score(primary_ind, confirm_ind, direction)
        range_height = range_high - range_low
        breakout_extras["range_high"] = float(range_high)
        breakout_extras["range_low"] = float(range_low)
        breakout_extras["range_height"] = float(range_height)

        return self._build_signal(
            symbol, direction, score, entry_data, primary_ind,
            range_high, range_low, range_height, breakout_extras,
        )

    # ------------------------------------------------------------------
    # Consolidation detection
    # ------------------------------------------------------------------

    def _detect_consolidation(
        self,
        primary_data: pd.DataFrame,
        primary_ind: pd.DataFrame,
    ) -> tuple[bool, float, float]:
        """Detect a consolidation range on the primary timeframe.

        A consolidation is identified when:
        - The last ``consolidation_bars`` candles trade within a tight range.
        - The Bollinger Band bandwidth is below its ``bb_width_percentile``
          percentile over the look-back window.

        Args:
            primary_data: OHLCV DataFrame for the primary timeframe.
            primary_ind: Indicator DataFrame for the primary timeframe.

        Returns:
            Tuple of (is_consolidating, range_high, range_low).  If not
            consolidating, range values are 0.0.
        """
        n = self.consolidation_bars
        recent = primary_data.iloc[-n:]
        range_high = float(recent["high"].max())
        range_low = float(recent["low"].min())

        # Check BB bandwidth percentile
        bb_width = self._get_bb_bandwidth(primary_ind)
        if bb_width is None:
            logger.debug("BB bandwidth unavailable — falling back to range check")
            # Fallback: use price range relative to midpoint
            midpoint = (range_high + range_low) / 2.0
            if midpoint == 0:
                return False, 0.0, 0.0
            range_pct = (range_high - range_low) / midpoint * 100.0
            is_tight = range_pct < 5.0  # less than 5% range
            return is_tight, range_high, range_low

        # Compute historical percentile of BB bandwidth
        lookback = min(self.bb_width_lookback, len(primary_ind))
        if lookback < n:
            return False, 0.0, 0.0

        bb_width_series = self._get_bb_bandwidth_series(primary_ind, lookback)
        if bb_width_series is None or len(bb_width_series) < lookback:
            return False, 0.0, 0.0

        percentile_value = float(np.percentile(bb_width_series.dropna().values, self.bb_width_percentile))
        current_bw = float(bb_width_series.iloc[-1]) if not pd.isna(bb_width_series.iloc[-1]) else None
        if current_bw is None:
            return False, 0.0, 0.0

        is_tight = current_bw <= percentile_value
        if is_tight:
            logger.info(
                "Consolidation detected",
                extra={
                    "range_high": range_high,
                    "range_low": range_low,
                    "bb_width": current_bw,
                    "threshold": percentile_value,
                },
            )
        return is_tight, range_high, range_low

    def _get_bb_bandwidth(self, ind: pd.DataFrame) -> Optional[float]:
        """Get the latest Bollinger Band bandwidth value.

        Bandwidth = (upper - lower) / middle * 100.

        Args:
            ind: Indicator DataFrame.

        Returns:
            BB bandwidth percentage, or ``None`` if columns are missing.
        """
        bb_upper = self._safe_last(ind, "bb_upper")
        bb_lower = self._safe_last(ind, "bb_lower")
        bb_mid = self._safe_last(ind, "bb_middle")
        if bb_upper is None or bb_lower is None or bb_mid is None or bb_mid == 0:
            return None
        return (bb_upper - bb_lower) / bb_mid * 100.0

    def _get_bb_bandwidth_series(
        self,
        ind: pd.DataFrame,
        lookback: int,
    ) -> Optional[pd.Series]:
        """Compute a BB bandwidth series over the look-back window.

        Args:
            ind: Indicator DataFrame.
            lookback: Number of bars to include.

        Returns:
            Series of bandwidth values, or ``None`` if columns missing.
        """
        for col in ("bb_upper", "bb_lower", "bb_middle"):
            if col not in ind.columns:
                return None
        recent = ind.iloc[-lookback:]
        mid = recent["bb_middle"].replace(0, np.nan)
        bandwidth = (recent["bb_upper"] - recent["bb_lower"]) / mid * 100.0
        return bandwidth

    # ------------------------------------------------------------------
    # Breakout detection
    # ------------------------------------------------------------------

    def _detect_breakout(
        self,
        primary_data: pd.DataFrame,
        primary_ind: pd.DataFrame,
        range_high: float,
        range_low: float,
    ) -> tuple[Optional[SignalDirection], dict[str, Any]]:
        """Detect and validate a breakout from the consolidation range.

        Validation requires:
        - ``consecutive_closes`` candle closes beyond the range boundary.
        - Volume z-score above ``volume_zscore_threshold``.

        Args:
            primary_data: OHLCV DataFrame for the primary timeframe.
            primary_ind: Indicator DataFrame for the primary timeframe.
            range_high: Upper boundary of the consolidation range.
            range_low: Lower boundary of the consolidation range.

        Returns:
            Tuple of (direction_or_None, extras_dict).
        """
        extras: dict[str, Any] = {}
        closes = primary_data["close"].values

        # Check consecutive closes above range_high (bullish breakout)
        n = self.consecutive_closes
        if len(closes) < n:
            return None, extras

        recent_closes = closes[-n:]
        bullish_breakout = all(c > range_high for c in recent_closes)
        bearish_breakout = all(c < range_low for c in recent_closes)

        if not bullish_breakout and not bearish_breakout:
            return None, extras

        # Volume z-score check
        vol_z = self._safe_last(primary_ind, "volume_zscore")
        if vol_z is None:
            vol_z = self._compute_volume_zscore(primary_data)
        if vol_z is None or vol_z < self.volume_zscore_threshold:
            logger.debug(
                "Breakout rejected: insufficient volume z-score %.2f < %.2f",
                vol_z if vol_z is not None else 0.0,
                self.volume_zscore_threshold,
            )
            return None, extras

        extras["volume_zscore"] = float(vol_z)
        extras["consecutive_closes_confirmed"] = n

        if bullish_breakout:
            extras["breakout_direction"] = "bullish"
            extras["breakout_level"] = float(range_high)
            logger.info(
                "Bullish breakout confirmed",
                extra={
                    "level": range_high,
                    "consecutive": n,
                    "volume_zscore": vol_z,
                },
            )
            return SignalDirection.LONG, extras

        extras["breakout_direction"] = "bearish"
        extras["breakout_level"] = float(range_low)
        logger.info(
            "Bearish breakout confirmed",
            extra={
                "level": range_low,
                "consecutive": n,
                "volume_zscore": vol_z,
            },
        )
        return SignalDirection.SHORT, extras

    # ------------------------------------------------------------------
    # Retest filter
    # ------------------------------------------------------------------

    def _check_retest(
        self,
        primary_data: pd.DataFrame,
        breakout_level: float,
        direction: SignalDirection,
        lookback: int = 10,
    ) -> bool:
        """Check whether price has retested the breakout level.

        After a bullish breakout above ``range_high``, price should pull
        back near that level before continuing higher.  The retest is
        confirmed when the low of a subsequent candle comes within
        ``retest_tolerance_pct`` of the breakout level.

        Args:
            primary_data: OHLCV DataFrame for the primary timeframe.
            breakout_level: The price level that was broken.
            direction: Direction of the breakout.
            lookback: Number of recent bars to scan for a retest.

        Returns:
            True if a valid retest has occurred.
        """
        if len(primary_data) < lookback + self.consecutive_closes:
            return False

        tolerance = breakout_level * self.retest_tolerance_pct
        # Scan bars after the breakout (skip the breakout bars themselves)
        scan_start = -(lookback + self.consecutive_closes)
        scan_end = -self.consecutive_closes
        scan = primary_data.iloc[scan_start:scan_end]

        if direction == SignalDirection.LONG:
            # Low should come close to breakout_level from above
            for _, row in scan.iterrows():
                low = float(row["low"])
                if abs(low - breakout_level) <= tolerance and float(row["close"]) > breakout_level:
                    logger.debug("Retest confirmed at %.4f (long)", low)
                    return True
        else:
            # High should come close to breakout_level from below
            for _, row in scan.iterrows():
                high = float(row["high"])
                if abs(high - breakout_level) <= tolerance and float(row["close"]) < breakout_level:
                    logger.debug("Retest confirmed at %.4f (short)", high)
                    return True

        return False

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _compute_score(
        self,
        primary_ind: pd.DataFrame,
        confirm_ind: Optional[pd.DataFrame],
        direction: SignalDirection,
    ) -> int:
        """Compute conviction score for a breakout signal.

        Starts at ``base_score`` (55) and adds bonus points:
        - +15 for volume confirmation (z-score > 3.0 — strong volume).
        - +10 for 1d trend alignment (EMA stack matching direction).
        - +10 for rising ADX (current ADX > previous ADX and above threshold).

        Total clamped to [0, 100].

        Args:
            primary_ind: Primary timeframe indicators.
            confirm_ind: 1d confirmation timeframe indicators (may be None).
            direction: Trade direction being scored.

        Returns:
            Integer score in [0, 100].
        """
        score = self.base_score

        # +15: Strong volume (z-score > 3.0)
        vol_z = self._safe_last(primary_ind, "volume_zscore")
        if vol_z is not None and vol_z > 3.0:
            score += 15
            logger.debug("Score +15: strong breakout volume z-score %.2f", vol_z)

        # +10: 1d trend alignment
        if confirm_ind is not None and not confirm_ind.empty:
            ema50_1d = self._safe_last(confirm_ind, "ema_50")
            ema200_1d = self._safe_last(confirm_ind, "ema_200")
            if ema50_1d is not None and ema200_1d is not None:
                if direction == SignalDirection.LONG and ema50_1d > ema200_1d:
                    score += 10
                    logger.debug("Score +10: 1d bullish trend alignment")
                elif direction == SignalDirection.SHORT and ema50_1d < ema200_1d:
                    score += 10
                    logger.debug("Score +10: 1d bearish trend alignment")

        # +10: Rising ADX
        adx_curr = self._safe_last(primary_ind, "adx")
        adx_prev = self._safe_last(primary_ind, "adx", offset=1)
        if adx_curr is not None and adx_prev is not None:
            if adx_curr > adx_prev and adx_curr > self.adx_threshold:
                score += 10
                logger.debug("Score +10: rising ADX %.1f > %.1f", adx_curr, adx_prev)

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
        range_high: float,
        range_low: float,
        range_height: float,
        extras: dict[str, Any],
    ) -> TradeSignal:
        """Build a breakout TradeSignal with measured-move targets.

        The take-profit targets use the measured-move technique: the
        consolidation range height is projected from the breakout point.

        Args:
            symbol: Trading pair.
            direction: Long or short.
            score: Computed conviction score.
            entry_data: OHLCV DataFrame for the entry timeframe.
            primary_ind: Primary timeframe indicators (ATR).
            range_high: Upper boundary of the consolidation range.
            range_low: Lower boundary of the consolidation range.
            range_height: Height of the consolidation range.
            extras: Additional metadata from breakout detection.

        Returns:
            Fully populated ``TradeSignal``.
        """
        entry_price = float(entry_data["close"].iloc[-1])
        atr = self._safe_last(primary_ind, "atr")
        atr_val = float(atr) if atr is not None else entry_price * 0.015

        # Stop loss: just inside the range
        if direction == SignalDirection.LONG:
            structure_stop = range_low
            stop_loss = self.compute_stop_loss(
                entry_price, direction, atr_val, swing_level=structure_stop
            )
            # Measured-move targets
            tp1 = range_high + range_height * 1.0
            tp2 = range_high + range_height * 1.618
            tp3 = range_high + range_height * 2.618
        else:
            structure_stop = range_high
            stop_loss = self.compute_stop_loss(
                entry_price, direction, atr_val, swing_level=structure_stop
            )
            tp1 = range_low - range_height * 1.0
            tp2 = range_low - range_height * 1.618
            tp3 = range_low - range_height * 2.618

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
                "target_method": "measured_move",
                **extras,
            },
        )

        logger.info(
            "Breakout signal generated",
            extra={
                "symbol": symbol,
                "direction": direction.value,
                "score": signal.score,
                "entry": entry_price,
                "stop": stop_loss,
                "tp1": tp1,
                "range": f"{range_low:.4f}-{range_high:.4f}",
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
