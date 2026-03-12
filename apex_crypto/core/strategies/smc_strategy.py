"""Smart Money Concepts (SMC) strategy for the APEX Crypto Trading System.

Uses institutional order flow concepts -- Order Blocks, Fair Value Gaps,
liquidity sweeps, and Change of Character -- combined with higher-timeframe
bias to enter high-probability reversal and continuation trades.
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

logger = get_logger("strategies.smc")


class SMCStrategy(BaseStrategy):
    """Smart Money Concepts strategy.

    Determines a higher-timeframe (HTF) bias from 1d/4h EMA structure,
    then scans for three entry patterns on the 15m timeframe:

    a) **Order Block retest** -- price revisits a prior OB zone and prints
       a reversal candle.
    b) **Fair Value Gap fill** -- price enters an unfilled FVG and prints
       a reversal candle near the midpoint.
    c) **Liquidity sweep fade** -- price sweeps equal highs/lows and
       immediately reverses.

    The highest-scoring pattern that aligns with HTF bias is selected.
    Pre-computed SMC structures can be supplied via ``alt_data``; otherwise
    the strategy detects them internally from OHLCV data.

    Attributes:
        name: Strategy identifier.
        active_regimes: Active in every regime.
        primary_timeframe: Timeframe for structural analysis.
        confirmation_timeframe: Timeframe for HTF bias.
        entry_timeframe: Timeframe for entry-pattern detection.
    """

    name: str = "smc"
    active_regimes: list[str] = [
        "STRONG_BULL", "WEAK_BULL", "RANGING", "WEAK_BEAR", "STRONG_BEAR",
    ]
    primary_timeframe: str = "4h"
    confirmation_timeframe: str = "1d"
    entry_timeframe: str = "15m"

    # -----------------------------------------------------------------
    # Configuration defaults
    # -----------------------------------------------------------------
    _DEFAULT_OB_LOOKBACK: int = 50
    _DEFAULT_FVG_LOOKBACK: int = 30
    _DEFAULT_LIQUIDITY_LOOKBACK: int = 40
    _DEFAULT_EQUAL_LEVEL_TOLERANCE: float = 0.001
    _DEFAULT_REVERSAL_BODY_RATIO: float = 0.55
    _DEFAULT_MIN_R_MULTIPLE: float = 3.0
    _DEFAULT_VOLUME_SPIKE_RATIO: float = 2.0

    def __init__(self, config: dict) -> None:
        """Initialize SMCStrategy.

        Args:
            config: Strategy-specific configuration dictionary.  Recognised
                keys (all optional):
                - ob_lookback: Bars to scan for order blocks.
                - fvg_lookback: Bars to scan for fair value gaps.
                - liquidity_lookback: Bars to scan for liquidity levels.
                - equal_level_tolerance: Fractional tolerance for equal
                  highs/lows detection.
                - reversal_body_ratio: Minimum body-to-range ratio for a
                  candle to qualify as a reversal candle.
                - min_r_multiple: Minimum acceptable R:R for a trade.
                - volume_spike_ratio: Multiplier over average volume to
                  qualify as a volume spike.
        """
        super().__init__(config)
        self.ob_lookback: int = config.get(
            "ob_lookback", self._DEFAULT_OB_LOOKBACK
        )
        self.fvg_lookback: int = config.get(
            "fvg_lookback", self._DEFAULT_FVG_LOOKBACK
        )
        self.liquidity_lookback: int = config.get(
            "liquidity_lookback", self._DEFAULT_LIQUIDITY_LOOKBACK
        )
        self.equal_level_tolerance: float = config.get(
            "equal_level_tolerance", self._DEFAULT_EQUAL_LEVEL_TOLERANCE
        )
        self.reversal_body_ratio: float = config.get(
            "reversal_body_ratio", self._DEFAULT_REVERSAL_BODY_RATIO
        )
        self.min_r_multiple: float = config.get(
            "min_r_multiple", self._DEFAULT_MIN_R_MULTIPLE
        )
        self.volume_spike_ratio: float = config.get(
            "volume_spike_ratio", self._DEFAULT_VOLUME_SPIKE_RATIO
        )
        logger.info(
            "SMCStrategy configured",
            extra={
                "ob_lookback": self.ob_lookback,
                "fvg_lookback": self.fvg_lookback,
                "min_r_multiple": self.min_r_multiple,
            },
        )

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def generate_signal(
        self,
        symbol: str,
        data: dict[str, pd.DataFrame],
        indicators: dict[str, pd.DataFrame],
        regime: str,
        alt_data: Optional[dict] = None,
    ) -> TradeSignal:
        """Generate an SMC-based trading signal.

        Determines HTF bias from 1d/4h EMA structure, then scans for
        Order Block retests, FVG fills, and liquidity sweep fades on the
        entry timeframe.  The highest-scoring pattern that aligns with
        bias is selected.

        Pre-computed SMC structures (``order_blocks``, ``fvgs``,
        ``liquidity_sweeps``) can be provided via *alt_data*; otherwise
        they are detected from the 4h OHLCV data.

        Args:
            symbol: Trading pair symbol (e.g., ``'BTC/USDT'``).
            data: OHLCV DataFrames keyed by timeframe string.
            indicators: Pre-computed indicator DataFrames keyed by
                timeframe.
            regime: Current market regime string.
            alt_data: Optional dict that may contain pre-computed SMC
                analysis results such as ``order_blocks``, ``fvgs``,
                and ``liquidity_sweeps``.

        Returns:
            A ``TradeSignal`` reflecting the assessment.
        """
        if not self.is_active(regime):
            logger.debug("Strategy inactive for regime %s", regime)
            return self._neutral_signal(symbol)

        # Validate required timeframe data is present
        for tf in (self.primary_timeframe, self.entry_timeframe):
            if tf not in data or tf not in indicators:
                logger.warning(
                    "Missing timeframe data",
                    extra={"timeframe": tf, "symbol": symbol},
                )
                return self._neutral_signal(symbol)

        primary_data = data[self.primary_timeframe]
        entry_data = data[self.entry_timeframe]

        if primary_data.empty or entry_data.empty:
            logger.debug("Empty dataframe for %s", symbol)
            return self._neutral_signal(symbol)

        # Step 1: Determine HTF bias from 1d/4h EMA structure
        htf_bias = self._determine_htf_bias(data, indicators)
        if htf_bias == "neutral":
            logger.debug("No clear HTF bias for %s", symbol)
            return self._neutral_signal(symbol)

        logger.debug("HTF bias: %s for %s", htf_bias, symbol)

        # Step 2: Gather SMC structures (prefer alt_data if available)
        smc = alt_data if alt_data is not None else {}
        order_blocks: list[dict[str, Any]] = smc.get(
            "order_blocks",
            self._detect_order_blocks(primary_data),
        )
        fvgs: list[dict[str, Any]] = smc.get(
            "fvgs",
            self._detect_fair_value_gaps(primary_data),
        )
        liquidity_sweeps: list[dict[str, Any]] = smc.get(
            "liquidity_sweeps",
            self._detect_liquidity_levels(primary_data),
        )

        # Step 3: Current price on the entry timeframe
        price = float(entry_data["close"].iloc[-1])

        # Step 4: Scan for entry patterns (check all, keep the best)
        candidates: list[tuple[int, str, dict[str, Any]]] = []

        ob_hit, ob_info = self._check_ob_retest(price, order_blocks, entry_data)
        if ob_hit:
            ob_direction = ob_info.get("direction", "bullish")
            if (ob_direction == "bullish" and htf_bias == "bullish") or (
                ob_direction == "bearish" and htf_bias == "bearish"
            ):
                score = 60
                # +10 for OB confluence with FVG or liquidity level
                if self._ob_has_confluence(ob_info, fvgs, liquidity_sweeps):
                    score += 10
                    ob_info["ob_confluence"] = True
                candidates.append((score, "ob_retest", ob_info))

        fvg_hit, fvg_info = self._check_fvg_fill(price, fvgs, entry_data)
        if fvg_hit:
            fvg_direction = fvg_info.get("direction", "bullish")
            if (fvg_direction == "bullish" and htf_bias == "bullish") or (
                fvg_direction == "bearish" and htf_bias == "bearish"
            ):
                score = 60
                # +10 for FVG + OB confluence
                if self._fvg_has_ob_confluence(fvg_info, order_blocks):
                    score += 10
                    fvg_info["fvg_ob_confluence"] = True
                candidates.append((score, "fvg_fill", fvg_info))

        sweep_hit, sweep_info = self._check_liquidity_sweep(
            liquidity_sweeps, entry_data
        )
        if sweep_hit:
            sweep_direction = sweep_info.get("direction", "bullish")
            if (sweep_direction == "bullish" and htf_bias == "bullish") or (
                sweep_direction == "bearish" and htf_bias == "bearish"
            ):
                score = 65
                # +10 for volume spike during the sweep
                if self._has_volume_spike(entry_data):
                    score += 10
                    sweep_info["volume_spike"] = True
                candidates.append((score, "liquidity_sweep", sweep_info))

        if not candidates:
            logger.debug("No SMC entry patterns found for %s", symbol)
            return self._neutral_signal(symbol)

        # Step 5: Select the highest-scoring candidate
        candidates.sort(key=lambda c: c[0], reverse=True)
        best_score, best_type, best_info = candidates[0]

        # Resolve direction
        if htf_bias == "bullish":
            direction = SignalDirection.LONG
        else:
            direction = SignalDirection.SHORT

        # Step 6: Apply additional confluence scoring
        final_score = self._apply_confluence_bonuses(
            best_score,
            direction,
            price,
            primary_data,
            entry_data,
            indicators,
            order_blocks,
            best_info,
        )

        # Step 7: Compute stop loss and targets
        stop_loss = self._compute_stop(
            price, direction, order_blocks, best_info, indicators
        )
        risk = abs(price - stop_loss)
        if risk == 0:
            atr = self._safe_last(
                indicators[self.primary_timeframe], "atr"
            )
            risk = float(atr) if atr is not None else price * 0.015

        tp1, tp2, tp3 = self._compute_targets(
            price, direction, risk, order_blocks, fvgs
        )

        # Enforce minimum 3R on TP1
        min_tp1 = (
            price + risk * self.min_r_multiple
            if direction == SignalDirection.LONG
            else price - risk * self.min_r_multiple
        )
        if direction == SignalDirection.LONG:
            tp1 = max(tp1, min_tp1)
        else:
            tp1 = min(tp1, min_tp1)

        confidence = final_score / 100.0

        signal = TradeSignal(
            symbol=symbol,
            direction=direction,
            score=final_score if direction == SignalDirection.LONG else -final_score,
            strategy=self.name,
            timeframe=self.primary_timeframe,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            confidence=confidence,
            metadata={
                "signal_type": "entry",
                "entry_type": best_type,
                "htf_bias": htf_bias,
                "regime": regime,
                **best_info,
            },
        )

        logger.info(
            "SMC signal generated",
            extra={
                "symbol": symbol,
                "direction": direction.value,
                "entry_type": best_type,
                "score": signal.score,
                "entry_price": price,
                "stop_loss": stop_loss,
                "tp1": tp1,
                "r_multiple": signal.r_multiple(),
                "htf_bias": htf_bias,
            },
        )
        return signal

    # -----------------------------------------------------------------
    # HTF bias
    # -----------------------------------------------------------------

    def _determine_htf_bias(
        self,
        data: dict[str, pd.DataFrame],
        indicators: dict[str, pd.DataFrame],
    ) -> str:
        """Determine higher-timeframe directional bias.

        Uses 1d EMA50 vs EMA200 as the primary filter.  Falls back to
        4h EMA50 vs EMA200 if daily data is unavailable.  When both
        timeframes are present they must agree for a clear bias.

        Args:
            data: OHLCV DataFrames keyed by timeframe string.
            indicators: Pre-computed indicator DataFrames keyed by
                timeframe.

        Returns:
            ``'bullish'`` if EMA50 > EMA200, ``'bearish'`` if
            EMA50 < EMA200, or ``'neutral'`` when no clear direction
            can be established.
        """
        daily_ind = indicators.get(self.confirmation_timeframe)
        primary_ind = indicators.get(self.primary_timeframe)

        # Attempt 1d bias
        bias_1d: Optional[str] = None
        if daily_ind is not None and not daily_ind.empty:
            ema50_1d = self._safe_last(daily_ind, "ema_50")
            ema200_1d = self._safe_last(daily_ind, "ema_200")
            if ema50_1d is not None and ema200_1d is not None:
                bias_1d = "bullish" if ema50_1d > ema200_1d else "bearish"
                logger.debug(
                    "1d EMA bias: %s (EMA50=%.2f, EMA200=%.2f)",
                    bias_1d,
                    ema50_1d,
                    ema200_1d,
                )

        # Attempt 4h bias
        bias_4h: Optional[str] = None
        if primary_ind is not None and not primary_ind.empty:
            ema50_4h = self._safe_last(primary_ind, "ema_50")
            ema200_4h = self._safe_last(primary_ind, "ema_200")
            if ema50_4h is not None and ema200_4h is not None:
                bias_4h = "bullish" if ema50_4h > ema200_4h else "bearish"
                logger.debug(
                    "4h EMA bias: %s (EMA50=%.2f, EMA200=%.2f)",
                    bias_4h,
                    ema50_4h,
                    ema200_4h,
                )

        # Both present -- must agree
        if bias_1d is not None and bias_4h is not None:
            if bias_1d == bias_4h:
                return bias_1d
            logger.debug("1d and 4h bias disagree -- returning neutral")
            return "neutral"

        # Only one available
        if bias_1d is not None:
            return bias_1d
        if bias_4h is not None:
            return bias_4h

        return "neutral"

    # -----------------------------------------------------------------
    # Entry pattern checks
    # -----------------------------------------------------------------

    def _check_ob_retest(
        self,
        price: float,
        order_blocks: list[dict[str, Any]],
        df_entry: pd.DataFrame,
    ) -> tuple[bool, dict[str, Any]]:
        """Check for an Order Block retest entry on the entry timeframe.

        Conditions:
        - Current price is inside an OB zone.
        - The latest candle on 15m is a reversal candle (engulfing,
          hammer, or pin bar pattern).

        Args:
            price: Current close price on the entry timeframe.
            order_blocks: List of order block dicts, each with at
                minimum ``high``, ``low``, and ``direction`` keys.
            df_entry: 15m OHLCV DataFrame.

        Returns:
            Tuple of (matched, info_dict).  ``matched`` is True when a
            valid OB retest with reversal confirmation is found.
        """
        if not order_blocks or df_entry.empty:
            return False, {}

        last_idx = len(df_entry) - 1

        for ob in reversed(order_blocks):
            ob_high = float(ob.get("high", 0))
            ob_low = float(ob.get("low", 0))
            ob_dir = ob.get("direction", "")

            # Price must be inside the OB zone
            if not (ob_low <= price <= ob_high):
                continue

            # Reversal candle confirmation
            if not self._has_reversal_candle(df_entry, last_idx):
                continue

            info: dict[str, Any] = {
                "direction": ob_dir,
                "ob_high": ob_high,
                "ob_low": ob_low,
                "ob_zone": dict(ob),
            }
            logger.info(
                "OB retest detected",
                extra={
                    "price": price,
                    "ob_high": ob_high,
                    "ob_low": ob_low,
                    "ob_direction": ob_dir,
                },
            )
            return True, info

        return False, {}

    def _check_fvg_fill(
        self,
        price: float,
        fvgs: list[dict[str, Any]],
        df_entry: pd.DataFrame,
    ) -> tuple[bool, dict[str, Any]]:
        """Check for a Fair Value Gap fill entry on the entry timeframe.

        Conditions:
        - Current price is inside an FVG zone.
        - A reversal candle prints at or near the FVG midpoint.

        Args:
            price: Current close price on the entry timeframe.
            fvgs: List of FVG dicts, each with at minimum ``high``,
                ``low``, and ``direction`` keys.
            df_entry: 15m OHLCV DataFrame.

        Returns:
            Tuple of (matched, info_dict).
        """
        if not fvgs or df_entry.empty:
            return False, {}

        last_idx = len(df_entry) - 1

        for fvg in reversed(fvgs):
            fvg_high = float(fvg.get("high", 0))
            fvg_low = float(fvg.get("low", 0))
            fvg_dir = fvg.get("direction", "")
            midpoint = (fvg_high + fvg_low) / 2.0

            # Price must be inside the FVG
            if not (fvg_low <= price <= fvg_high):
                continue

            # Price should be near the midpoint (within the zone is
            # sufficient since the zone itself is the imbalance)
            fvg_range = fvg_high - fvg_low
            if fvg_range > 0:
                distance_to_mid = abs(price - midpoint) / fvg_range
                # Allow the full zone but prefer being near midpoint
                if distance_to_mid > 0.75:
                    continue

            # Reversal candle confirmation
            if not self._has_reversal_candle(df_entry, last_idx):
                continue

            info: dict[str, Any] = {
                "direction": fvg_dir,
                "fvg_high": fvg_high,
                "fvg_low": fvg_low,
                "fvg_midpoint": midpoint,
                "fvg_zone": dict(fvg),
            }
            logger.info(
                "FVG fill detected",
                extra={
                    "price": price,
                    "fvg_high": fvg_high,
                    "fvg_low": fvg_low,
                    "fvg_direction": fvg_dir,
                },
            )
            return True, info

        return False, {}

    def _check_liquidity_sweep(
        self,
        sweeps: list[dict[str, Any]],
        df_entry: pd.DataFrame,
    ) -> tuple[bool, dict[str, Any]]:
        """Check for a liquidity sweep fade entry on the entry timeframe.

        Conditions:
        - A sweep of equal highs/lows is detected (price wicked through
          the liquidity level and closed back inside).
        - A reversal candle follows.

        Args:
            sweeps: List of liquidity level dicts, each with at minimum
                ``high``, ``low``, and ``direction`` keys.
            df_entry: 15m OHLCV DataFrame.

        Returns:
            Tuple of (matched, info_dict).
        """
        if not sweeps or len(df_entry) < 2:
            return False, {}

        current_high = float(df_entry["high"].iloc[-1])
        current_low = float(df_entry["low"].iloc[-1])
        current_close = float(df_entry["close"].iloc[-1])
        prev_close = float(df_entry["close"].iloc[-2])
        last_idx = len(df_entry) - 1

        for liq in reversed(sweeps):
            liq_high = float(liq.get("high", 0))
            liq_low = float(liq.get("low", 0))
            liq_dir = liq.get("direction", "")

            if liq_dir == "bullish":
                # Liquidity below equal lows -- sweep down then rally
                swept = current_low < liq_low
                rejected = current_close > liq_high and prev_close > liq_low
            elif liq_dir == "bearish":
                # Liquidity above equal highs -- sweep up then drop
                swept = current_high > liq_high
                rejected = current_close < liq_low and prev_close < liq_high
            else:
                continue

            if not (swept and rejected):
                continue

            # Reversal candle confirmation
            if not self._has_reversal_candle(df_entry, last_idx):
                continue

            info: dict[str, Any] = {
                "direction": liq_dir,
                "liq_high": liq_high,
                "liq_low": liq_low,
                "liquidity_zone": dict(liq),
            }
            logger.info(
                "Liquidity sweep fade detected",
                extra={
                    "current_low": current_low,
                    "current_high": current_high,
                    "liq_direction": liq_dir,
                },
            )
            return True, info

        return False, {}

    # -----------------------------------------------------------------
    # Reversal candle detection
    # -----------------------------------------------------------------

    def _has_reversal_candle(
        self,
        df: pd.DataFrame,
        index: int,
    ) -> bool:
        """Check whether the candle at *index* qualifies as a reversal.

        Detects engulfing candles, hammers, and pin bars.  A candle is
        considered a reversal if it satisfies any of the following:

        - **Engulfing**: body covers >= ``reversal_body_ratio`` of the
          total range and the body direction opposes the prior candle.
        - **Hammer / inverted hammer**: one wick is >= 2x the body size
          with the body in the expected direction.
        - **Pin bar**: total wick on one side >= 66% of the candle range
          with a small body (body < 33% of range).

        Args:
            df: OHLCV DataFrame for the entry timeframe.
            index: Row index to inspect.

        Returns:
            True if the candle at *index* matches a reversal pattern.
        """
        if index < 1 or index >= len(df):
            return False

        o = float(df["open"].iloc[index])
        c = float(df["close"].iloc[index])
        h = float(df["high"].iloc[index])
        low = float(df["low"].iloc[index])
        candle_range = h - low
        if candle_range == 0:
            return False

        body = abs(c - o)
        body_ratio = body / candle_range

        # Previous candle for engulfing check
        prev_o = float(df["open"].iloc[index - 1])
        prev_c = float(df["close"].iloc[index - 1])
        prev_body = abs(prev_c - prev_o)

        is_bullish = c > o
        is_bearish = c < o
        prev_is_bearish = prev_c < prev_o
        prev_is_bullish = prev_c > prev_o

        # --- Engulfing ---
        if body_ratio >= self.reversal_body_ratio and body > prev_body:
            if (is_bullish and prev_is_bearish) or (
                is_bearish and prev_is_bullish
            ):
                return True

        # --- Hammer / inverted hammer ---
        if body > 0:
            upper_wick = h - max(o, c)
            lower_wick = min(o, c) - low

            # Bullish hammer: long lower wick, small upper wick
            if is_bullish and lower_wick >= 2 * body and upper_wick < body:
                return True
            # Bearish inverted hammer (shooting star)
            if is_bearish and upper_wick >= 2 * body and lower_wick < body:
                return True

        # --- Pin bar ---
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - low
        if body_ratio < 0.33:
            # Bullish pin bar: long lower wick
            if lower_wick / candle_range >= 0.66:
                return True
            # Bearish pin bar: long upper wick
            if upper_wick / candle_range >= 0.66:
                return True

        return False

    # -----------------------------------------------------------------
    # Confluence scoring
    # -----------------------------------------------------------------

    def _apply_confluence_bonuses(
        self,
        base_score: int,
        direction: SignalDirection,
        price: float,
        primary_data: pd.DataFrame,
        entry_data: pd.DataFrame,
        indicators: dict[str, pd.DataFrame],
        order_blocks: list[dict[str, Any]],
        extras: dict[str, Any],
    ) -> int:
        """Apply additional confluence bonuses to the base score.

        Bonuses:
        - +10 for premium/discount zone alignment (longs in discount,
          shorts in premium).
        - +10 for CHoCH (Change of Character) confirmation on the
          entry timeframe.
        - +5 for multiple-timeframe OB alignment (4h OB overlaps with
          entry-timeframe OB).

        The total is clamped to [0, 100].

        Args:
            base_score: Starting score from the entry pattern.
            direction: Trade direction.
            price: Current entry price.
            primary_data: 4h OHLCV DataFrame.
            entry_data: 15m OHLCV DataFrame.
            indicators: Indicator DataFrames keyed by timeframe.
            order_blocks: Detected order block structures.
            extras: Metadata dict from the matched entry pattern.

        Returns:
            Final integer score in [0, 100].
        """
        score = base_score

        # +10: Premium / discount zone alignment
        if len(primary_data) >= 50:
            recent_high = float(primary_data["high"].iloc[-50:].max())
            recent_low = float(primary_data["low"].iloc[-50:].min())
            mid = (recent_high + recent_low) / 2.0
            if direction == SignalDirection.LONG and price < mid:
                score += 10
                extras["premium_discount"] = "discount_zone"
                logger.debug("Score +10: long entry in discount zone")
            elif direction == SignalDirection.SHORT and price > mid:
                score += 10
                extras["premium_discount"] = "premium_zone"
                logger.debug("Score +10: short entry in premium zone")

        # +10: CHoCH confirmation on entry timeframe
        if self._detect_choch(entry_data, direction):
            score += 10
            extras["choch_confirmed"] = True
            logger.debug("Score +10: CHoCH confirmed on entry timeframe")

        # +5: Multiple timeframe OB alignment
        if self._check_mtf_ob_alignment(
            price, order_blocks, entry_data
        ):
            score += 5
            extras["mtf_ob_alignment"] = True
            logger.debug("Score +5: multiple-timeframe OB alignment")

        return int(np.clip(score, 0, 100))

    def _detect_choch(
        self,
        entry_data: pd.DataFrame,
        expected_direction: SignalDirection,
        lookback: int = 20,
    ) -> bool:
        """Detect a Change of Character (CHoCH) on the entry timeframe.

        A bullish CHoCH occurs when price breaks above a recent swing
        high after a series of lower highs.  Bearish is the mirror.

        Args:
            entry_data: Entry-timeframe OHLCV DataFrame.
            expected_direction: Direction the CHoCH should confirm.
            lookback: Number of bars to scan.

        Returns:
            True if a CHoCH matching the expected direction is detected.
        """
        if len(entry_data) < lookback:
            return False

        recent = entry_data.iloc[-lookback:]
        highs = recent["high"].values
        lows = recent["low"].values
        closes = recent["close"].values

        swing_highs: list[tuple[int, float]] = []
        swing_lows: list[tuple[int, float]] = []

        for i in range(1, len(highs) - 1):
            if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
                swing_highs.append((i, float(highs[i])))
            if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
                swing_lows.append((i, float(lows[i])))

        if expected_direction == SignalDirection.LONG:
            if len(swing_highs) >= 2:
                prev_sh = swing_highs[-2][1]
                last_sh = swing_highs[-1][1]
                # Structure was bearish (lower highs)
                if last_sh < prev_sh and float(closes[-1]) > last_sh:
                    return True
        else:
            if len(swing_lows) >= 2:
                prev_sl = swing_lows[-2][1]
                last_sl = swing_lows[-1][1]
                # Structure was bullish (higher lows)
                if last_sl > prev_sl and float(closes[-1]) < last_sl:
                    return True

        return False

    def _check_mtf_ob_alignment(
        self,
        price: float,
        htf_order_blocks: list[dict[str, Any]],
        entry_data: pd.DataFrame,
    ) -> bool:
        """Check for multiple-timeframe order block alignment.

        Returns True when the current price sits inside both a
        higher-timeframe OB and an entry-timeframe OB detected from
        the 15m data.

        Args:
            price: Current close price.
            htf_order_blocks: 4h order block dicts.
            entry_data: 15m OHLCV DataFrame.

        Returns:
            True if OBs on both timeframes overlap at the current price.
        """
        # Check if price is in any HTF OB
        in_htf_ob = False
        for ob in htf_order_blocks:
            ob_high = float(ob.get("high", 0))
            ob_low = float(ob.get("low", 0))
            if ob_low <= price <= ob_high:
                in_htf_ob = True
                break

        if not in_htf_ob:
            return False

        # Detect entry-timeframe OBs and check overlap
        entry_obs = self._detect_order_blocks(entry_data)
        for ob in entry_obs:
            ob_high = float(ob.get("high", 0))
            ob_low = float(ob.get("low", 0))
            if ob_low <= price <= ob_high:
                return True

        return False

    # -----------------------------------------------------------------
    # Confluence helpers for entry patterns
    # -----------------------------------------------------------------

    def _ob_has_confluence(
        self,
        ob_info: dict[str, Any],
        fvgs: list[dict[str, Any]],
        liquidity_levels: list[dict[str, Any]],
    ) -> bool:
        """Check if an OB zone overlaps with any FVG or liquidity level.

        Args:
            ob_info: The matched OB info dict (must have ``ob_high``
                and ``ob_low``).
            fvgs: FVG zone dicts.
            liquidity_levels: Liquidity level dicts.

        Returns:
            True when the OB overlaps another structural zone.
        """
        ob_high = float(ob_info.get("ob_high", 0))
        ob_low = float(ob_info.get("ob_low", 0))

        for fvg in fvgs:
            fvg_high = float(fvg.get("high", 0))
            fvg_low = float(fvg.get("low", 0))
            if ob_low <= fvg_high and fvg_low <= ob_high:
                return True

        for liq in liquidity_levels:
            liq_high = float(liq.get("high", 0))
            liq_low = float(liq.get("low", 0))
            if ob_low <= liq_high and liq_low <= ob_high:
                return True

        return False

    def _fvg_has_ob_confluence(
        self,
        fvg_info: dict[str, Any],
        order_blocks: list[dict[str, Any]],
    ) -> bool:
        """Check if an FVG zone overlaps with any order block.

        Args:
            fvg_info: The matched FVG info dict (must have ``fvg_high``
                and ``fvg_low``).
            order_blocks: Order block dicts.

        Returns:
            True when the FVG overlaps an OB zone.
        """
        fvg_high = float(fvg_info.get("fvg_high", 0))
        fvg_low = float(fvg_info.get("fvg_low", 0))

        for ob in order_blocks:
            ob_high = float(ob.get("high", 0))
            ob_low = float(ob.get("low", 0))
            if fvg_low <= ob_high and ob_low <= fvg_high:
                return True

        return False

    def _has_volume_spike(
        self,
        df_entry: pd.DataFrame,
        lookback: int = 20,
    ) -> bool:
        """Detect a volume spike on the entry timeframe.

        A spike is defined as the latest bar's volume exceeding
        ``volume_spike_ratio`` times the rolling average.

        Args:
            df_entry: 15m OHLCV DataFrame.
            lookback: Number of bars for the rolling average.

        Returns:
            True if the latest bar shows a volume spike.
        """
        if "volume" not in df_entry.columns or len(df_entry) < lookback + 1:
            return False

        volumes = df_entry["volume"].values
        avg_volume = float(np.mean(volumes[-(lookback + 1):-1]))
        if avg_volume == 0:
            return False

        current_volume = float(volumes[-1])
        return current_volume >= avg_volume * self.volume_spike_ratio

    # -----------------------------------------------------------------
    # SMC structure detection (fallback when alt_data is absent)
    # -----------------------------------------------------------------

    def _detect_order_blocks(
        self,
        data: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """Detect Order Blocks from OHLCV data.

        An Order Block is the last opposing candle before a strong
        impulsive move.  A bullish OB is a bearish candle followed by a
        strong bullish impulse; a bearish OB is the mirror.

        Args:
            data: OHLCV DataFrame.

        Returns:
            List of dicts with ``high``, ``low``, ``direction``, and
            ``bar_index`` keys representing unmitigated order blocks.
        """
        obs: list[dict[str, Any]] = []
        if len(data) < 3:
            return obs

        opens = data["open"].values
        closes = data["close"].values
        highs = data["high"].values
        lows = data["low"].values

        lookback = min(self.ob_lookback, len(data) - 2)

        for i in range(len(data) - lookback, len(data) - 1):
            if i < 1:
                continue

            body_prev = abs(closes[i - 1] - opens[i - 1])
            body_curr = abs(closes[i] - opens[i])
            range_curr = highs[i] - lows[i]

            if range_curr == 0:
                continue

            # Strong impulsive move: current candle body > 1.5x prev body
            if body_curr < body_prev * 1.5:
                continue

            is_bullish_move = closes[i] > opens[i]
            is_prev_bearish = closes[i - 1] < opens[i - 1]
            is_bearish_move = closes[i] < opens[i]
            is_prev_bullish = closes[i - 1] > opens[i - 1]

            if is_bullish_move and is_prev_bearish:
                obs.append({
                    "high": float(highs[i - 1]),
                    "low": float(lows[i - 1]),
                    "direction": "bullish",
                    "bar_index": i - 1,
                })
            elif is_bearish_move and is_prev_bullish:
                obs.append({
                    "high": float(highs[i - 1]),
                    "low": float(lows[i - 1]),
                    "direction": "bearish",
                    "bar_index": i - 1,
                })

        # Filter out mitigated OBs
        current_close = float(closes[-1])
        unmitigated: list[dict[str, Any]] = []
        for ob in obs:
            if ob["direction"] == "bullish" and current_close > ob["low"]:
                unmitigated.append(ob)
            elif ob["direction"] == "bearish" and current_close < ob["high"]:
                unmitigated.append(ob)

        logger.debug("Detected %d unmitigated order blocks", len(unmitigated))
        return unmitigated

    def _detect_fair_value_gaps(
        self,
        data: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """Detect Fair Value Gaps (imbalances) from OHLCV data.

        A bullish FVG exists when ``candle[i-2].high < candle[i].low``,
        leaving a gap that price may revisit.  Bearish is the mirror.

        Args:
            data: OHLCV DataFrame.

        Returns:
            List of dicts with ``high``, ``low``, ``direction``, and
            ``bar_index`` keys representing unfilled FVGs.
        """
        fvgs: list[dict[str, Any]] = []
        if len(data) < 3:
            return fvgs

        highs = data["high"].values
        lows = data["low"].values
        closes = data["close"].values

        lookback = min(self.fvg_lookback, len(data) - 2)

        for i in range(len(data) - lookback, len(data)):
            if i < 2:
                continue

            # Bullish FVG
            if highs[i - 2] < lows[i]:
                fvgs.append({
                    "high": float(lows[i]),
                    "low": float(highs[i - 2]),
                    "direction": "bullish",
                    "bar_index": i - 1,
                })

            # Bearish FVG
            if lows[i - 2] > highs[i]:
                fvgs.append({
                    "high": float(lows[i - 2]),
                    "low": float(highs[i]),
                    "direction": "bearish",
                    "bar_index": i - 1,
                })

        # Filter out filled FVGs
        current_close = float(closes[-1])
        unfilled: list[dict[str, Any]] = []
        for fvg in fvgs:
            if fvg["direction"] == "bullish" and current_close > fvg["high"]:
                continue
            if fvg["direction"] == "bearish" and current_close < fvg["low"]:
                continue
            unfilled.append(fvg)

        logger.debug("Detected %d unfilled FVGs", len(unfilled))
        return unfilled

    def _detect_liquidity_levels(
        self,
        data: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """Detect equal highs/lows as liquidity pools from OHLCV data.

        When two or more swing highs (or lows) rest at approximately the
        same price, liquidity is pooled above (or below) those levels.

        Args:
            data: OHLCV DataFrame.

        Returns:
            List of dicts with ``high``, ``low``, ``direction``, and
            ``bar_index`` keys marking liquidity pools.
        """
        levels: list[dict[str, Any]] = []
        if len(data) < 5:
            return levels

        highs = data["high"].values
        lows = data["low"].values

        lookback = min(self.liquidity_lookback, len(data) - 2)
        start = len(data) - lookback

        # Collect swing highs and swing lows
        swing_highs: list[tuple[int, float]] = []
        swing_lows: list[tuple[int, float]] = []

        for i in range(max(start, 2), len(data) - 2):
            if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
                swing_highs.append((i, float(highs[i])))
            elif highs[i] > highs[i - 2] and highs[i] > highs[i + 2]:
                swing_highs.append((i, float(highs[i])))

            if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
                swing_lows.append((i, float(lows[i])))
            elif lows[i] < lows[i - 2] and lows[i] < lows[i + 2]:
                swing_lows.append((i, float(lows[i])))

        # Equal highs -> liquidity above (bearish sweep expected)
        for idx_a in range(len(swing_highs)):
            for idx_b in range(idx_a + 1, len(swing_highs)):
                _, price_a = swing_highs[idx_a]
                bar_b, price_b = swing_highs[idx_b]
                if price_a == 0:
                    continue
                if abs(price_a - price_b) / price_a <= self.equal_level_tolerance:
                    avg_price = (price_a + price_b) / 2.0
                    levels.append({
                        "high": avg_price * (1 + self.equal_level_tolerance),
                        "low": avg_price * (1 - self.equal_level_tolerance),
                        "direction": "bearish",
                        "bar_index": bar_b,
                    })
                    break

        # Equal lows -> liquidity below (bullish sweep expected)
        for idx_a in range(len(swing_lows)):
            for idx_b in range(idx_a + 1, len(swing_lows)):
                _, price_a = swing_lows[idx_a]
                bar_b, price_b = swing_lows[idx_b]
                if price_a == 0:
                    continue
                if abs(price_a - price_b) / price_a <= self.equal_level_tolerance:
                    avg_price = (price_a + price_b) / 2.0
                    levels.append({
                        "high": avg_price * (1 + self.equal_level_tolerance),
                        "low": avg_price * (1 - self.equal_level_tolerance),
                        "direction": "bullish",
                        "bar_index": bar_b,
                    })
                    break

        logger.debug("Detected %d liquidity levels", len(levels))
        return levels

    # -----------------------------------------------------------------
    # Stop loss and targets
    # -----------------------------------------------------------------

    def _compute_stop(
        self,
        entry_price: float,
        direction: SignalDirection,
        order_blocks: list[dict[str, Any]],
        extras: dict[str, Any],
        indicators: dict[str, pd.DataFrame],
    ) -> float:
        """Compute stop loss using OB structure with ATR fallback.

        For longs the stop is placed below the low of the nearest
        relevant bullish OB.  For shorts it is placed above the high
        of the nearest bearish OB.

        Args:
            entry_price: Trade entry price.
            direction: Trade direction.
            order_blocks: Detected order block dicts.
            extras: Entry pattern metadata.
            indicators: Indicator DataFrames keyed by timeframe.

        Returns:
            Stop loss price.
        """
        if direction == SignalDirection.LONG:
            relevant = [
                ob for ob in order_blocks
                if ob.get("direction") == "bullish"
                and float(ob.get("low", 0)) < entry_price
            ]
            if relevant:
                closest = max(relevant, key=lambda ob: float(ob["low"]))
                ob_low = float(closest["low"])
                ob_high = float(closest["high"])
                buffer = (ob_high - ob_low) * 0.1
                return ob_low - buffer
        else:
            relevant = [
                ob for ob in order_blocks
                if ob.get("direction") == "bearish"
                and float(ob.get("high", 0)) > entry_price
            ]
            if relevant:
                closest = min(relevant, key=lambda ob: float(ob["high"]))
                ob_high = float(closest["high"])
                ob_low = float(closest["low"])
                buffer = (ob_high - ob_low) * 0.1
                return ob_high + buffer

        # ATR-based fallback
        primary_ind = indicators.get(self.primary_timeframe)
        atr = self._safe_last(primary_ind, "atr") if primary_ind is not None else None
        atr_val = float(atr) if atr is not None else entry_price * 0.015
        return self.compute_stop_loss(entry_price, direction, atr_val)

    def _compute_targets(
        self,
        entry_price: float,
        direction: SignalDirection,
        risk: float,
        order_blocks: list[dict[str, Any]],
        fvgs: list[dict[str, Any]],
    ) -> tuple[float, float, float]:
        """Compute take-profit targets using opposing SMC zones.

        TP1 targets the next FVG or OB in the opposing direction.
        TP2 and TP3 use further zones or R-multiple extensions.
        All targets enforce a minimum of ``min_r_multiple`` R.

        Args:
            entry_price: Trade entry price.
            direction: Trade direction.
            risk: Risk per unit (|entry - stop|).
            order_blocks: Order block dicts.
            fvgs: FVG dicts.

        Returns:
            Tuple of (tp1, tp2, tp3) prices.
        """
        target_zones: list[float] = []

        if direction == SignalDirection.LONG:
            for zone in order_blocks:
                if zone.get("direction") == "bearish":
                    zone_low = float(zone.get("low", 0))
                    if zone_low > entry_price:
                        target_zones.append(zone_low)
            for zone in fvgs:
                if zone.get("direction") == "bearish":
                    zone_low = float(zone.get("low", 0))
                    if zone_low > entry_price:
                        target_zones.append(zone_low)
            target_zones.sort()
        else:
            for zone in order_blocks:
                if zone.get("direction") == "bullish":
                    zone_high = float(zone.get("high", 0))
                    if zone_high < entry_price:
                        target_zones.append(zone_high)
            for zone in fvgs:
                if zone.get("direction") == "bullish":
                    zone_high = float(zone.get("high", 0))
                    if zone_high < entry_price:
                        target_zones.append(zone_high)
            target_zones.sort(reverse=True)

        # R-multiple fallbacks
        stop_for_calc = (
            entry_price - risk
            if direction == SignalDirection.LONG
            else entry_price + risk
        )
        r_targets = self.compute_take_profits(
            entry_price,
            stop_for_calc,
            direction,
            tp1_r=self.min_r_multiple,
            tp2_r=self.min_r_multiple * 1.5,
            tp3_r=self.min_r_multiple * 2.0,
        )

        tp1 = target_zones[0] if len(target_zones) > 0 else r_targets[0]
        tp2 = target_zones[1] if len(target_zones) > 1 else r_targets[1]
        tp3 = target_zones[2] if len(target_zones) > 2 else r_targets[2]

        return tp1, tp2, tp3

    # -----------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------

    @staticmethod
    def _safe_last(
        df: Optional[pd.DataFrame],
        column: str,
        offset: int = 0,
    ) -> Optional[float]:
        """Safely retrieve the last (or offset-from-last) value of a column.

        Args:
            df: DataFrame to read from (may be None).
            column: Column name.
            offset: How many rows back from the last row (0 = last).

        Returns:
            The float value, or ``None`` if unavailable or NaN.
        """
        if df is None or df.empty:
            return None
        if column not in df.columns:
            return None
        idx = -(1 + offset)
        if abs(idx) > len(df):
            return None
        val = df[column].iloc[idx]
        if pd.isna(val):
            return None
        return float(val)
