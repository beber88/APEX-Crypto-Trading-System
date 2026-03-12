"""Candlestick and chart pattern recognition module.

Provides vectorized candlestick pattern detection and geometric chart pattern
recognition for technical analysis of OHLCV price data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import linregress


class PatternRecognition:
    """Detects candlestick patterns and geometric chart patterns.

    Candlestick methods return a pandas Series of signal strength from -100
    (strongly bearish) to +100 (strongly bullish), with 0 meaning no pattern.

    Chart-pattern methods return a list of dicts describing each detected
    formation with probability scores, measured-move targets, and
    invalidation levels.

    Args:
        config: Indicator configuration dictionary.  Recognised keys:
            - ``volume_avg_period`` (int): lookback for volume average,
              default 20.
            - ``volume_boost_threshold`` (float): volume ratio above which
              the signal is boosted, default 1.2.
            - ``volume_boost_pct`` (float): percentage boost applied when
              volume confirms, default 0.20.
            - ``swing_order`` (int): number of bars on each side used to
              identify swing highs/lows, default 5.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, config: dict) -> None:
        """Initialise with indicator configuration.

        Args:
            config: Dictionary of configuration parameters.
        """
        self.config = config
        self.volume_avg_period: int = config.get("volume_avg_period", 20)
        self.volume_boost_threshold: float = config.get(
            "volume_boost_threshold", 1.2
        )
        self.volume_boost_pct: float = config.get("volume_boost_pct", 0.20)
        self.swing_order: int = config.get("swing_order", 5)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _body(self, df: pd.DataFrame) -> pd.Series:
        """Absolute body size (close - open)."""
        return (df["close"] - df["open"]).abs()

    def _range(self, df: pd.DataFrame) -> pd.Series:
        """Full candle range (high - low), floored to avoid division by 0."""
        return (df["high"] - df["low"]).replace(0, np.nan)

    def _upper_shadow(self, df: pd.DataFrame) -> pd.Series:
        """Upper shadow length."""
        return df["high"] - df[["open", "close"]].max(axis=1)

    def _lower_shadow(self, df: pd.DataFrame) -> pd.Series:
        """Lower shadow length."""
        return df[["open", "close"]].min(axis=1) - df["low"]

    def _is_bullish(self, df: pd.DataFrame) -> pd.Series:
        """True where close >= open."""
        return df["close"] >= df["open"]

    def _volume_boost(self, df: pd.DataFrame) -> pd.Series:
        """Multiplier Series: 1.0 + boost where volume confirms.

        Returns:
            Series of multipliers (1.0 or 1.0 + volume_boost_pct).
        """
        if "volume" not in df.columns:
            return pd.Series(1.0, index=df.index)
        avg_vol = df["volume"].rolling(
            window=self.volume_avg_period, min_periods=1
        ).mean()
        ratio = df["volume"] / avg_vol.replace(0, np.nan)
        boost = np.where(
            ratio > self.volume_boost_threshold,
            1.0 + self.volume_boost_pct,
            1.0,
        )
        return pd.Series(boost, index=df.index)

    def _clamp(self, series: pd.Series) -> pd.Series:
        """Clamp values to [-100, +100]."""
        return series.clip(-100, 100)

    def _swing_highs(self, highs: np.ndarray, order: int) -> np.ndarray:
        """Return boolean array marking swing high positions.

        A swing high at index *i* means ``highs[i]`` is the maximum in the
        window ``[i - order, i + order]``.
        """
        n = len(highs)
        result = np.zeros(n, dtype=bool)
        for i in range(order, n - order):
            window = highs[i - order: i + order + 1]
            if highs[i] == np.max(window):
                result[i] = True
        return result

    def _swing_lows(self, lows: np.ndarray, order: int) -> np.ndarray:
        """Return boolean array marking swing low positions."""
        n = len(lows)
        result = np.zeros(n, dtype=bool)
        for i in range(order, n - order):
            window = lows[i - order: i + order + 1]
            if lows[i] == np.min(window):
                result[i] = True
        return result

    def _fit_line(
        self, indices: np.ndarray, values: np.ndarray
    ) -> tuple[float, float, float]:
        """Fit a line via linear regression.

        Args:
            indices: X values (bar indices).
            values: Y values (prices).

        Returns:
            Tuple of (slope, intercept, r_squared).
        """
        if len(indices) < 2:
            return 0.0, 0.0, 0.0
        slope, intercept, r_value, _, _ = linregress(indices, values)
        return slope, intercept, r_value ** 2

    # ------------------------------------------------------------------
    # A. CANDLESTICK PATTERNS
    # ------------------------------------------------------------------

    def detect_doji(self, df: pd.DataFrame) -> pd.Series:
        """Detect doji candles — body < 10% of the candle range.

        Args:
            df: OHLCV DataFrame.

        Returns:
            Series of signal strengths (-100 to +100).  Doji near support
            zones are bullish; this basic detector returns +50 when the
            close is above the midpoint and -50 otherwise, boosted by
            volume.
        """
        body = self._body(df)
        rng = self._range(df)
        ratio = body / rng
        is_doji = ratio < 0.10

        mid = (df["high"] + df["low"]) / 2
        raw = np.where(
            is_doji,
            np.where(df["close"] >= mid, 50.0, -50.0),
            0.0,
        )
        signal = pd.Series(raw, index=df.index) * self._volume_boost(df)
        return self._clamp(signal)

    def detect_hammer(self, df: pd.DataFrame) -> pd.Series:
        """Detect hammer candles — small body at top, long lower shadow.

        A hammer has a lower shadow at least 2x the body and a small upper
        shadow.  It is a bullish reversal signal.

        Args:
            df: OHLCV DataFrame.

        Returns:
            Series of signal strengths (0 or positive up to +100).
        """
        body = self._body(df)
        rng = self._range(df)
        lower = self._lower_shadow(df)
        upper = self._upper_shadow(df)

        body_safe = body.replace(0, np.nan)
        is_hammer = (
            (lower >= 2 * body)
            & (upper <= body * 0.5)
            & (body / rng < 0.35)
        )
        shadow_ratio = (lower / body_safe).fillna(0).clip(upper=5)
        strength = (shadow_ratio / 5 * 100).clip(upper=100)

        raw = np.where(is_hammer, strength, 0.0)
        signal = pd.Series(raw, index=df.index) * self._volume_boost(df)
        return self._clamp(signal)

    def detect_shooting_star(self, df: pd.DataFrame) -> pd.Series:
        """Detect shooting star — small body at bottom, long upper shadow.

        Bearish reversal signal.

        Args:
            df: OHLCV DataFrame.

        Returns:
            Series of signal strengths (0 or negative down to -100).
        """
        body = self._body(df)
        rng = self._range(df)
        upper = self._upper_shadow(df)
        lower = self._lower_shadow(df)

        body_safe = body.replace(0, np.nan)
        is_star = (
            (upper >= 2 * body)
            & (lower <= body * 0.5)
            & (body / rng < 0.35)
        )
        shadow_ratio = (upper / body_safe).fillna(0).clip(upper=5)
        strength = (shadow_ratio / 5 * 100).clip(upper=100)

        raw = np.where(is_star, -strength, 0.0)
        signal = pd.Series(raw, index=df.index) * self._volume_boost(df)
        return self._clamp(signal)

    def detect_engulfing(self, df: pd.DataFrame) -> pd.Series:
        """Detect bullish and bearish engulfing patterns.

        Bullish engulfing: bearish candle followed by a bullish candle whose
        body fully engulfs the previous body.  Bearish engulfing is the
        mirror.

        Args:
            df: OHLCV DataFrame.

        Returns:
            Series — +100 bullish, -100 bearish, 0 otherwise.
        """
        prev_open = df["open"].shift(1)
        prev_close = df["close"].shift(1)
        prev_bullish = prev_close >= prev_open
        curr_bullish = self._is_bullish(df)

        curr_body_top = df[["open", "close"]].max(axis=1)
        curr_body_bot = df[["open", "close"]].min(axis=1)
        prev_body_top = np.maximum(prev_open, prev_close)
        prev_body_bot = np.minimum(prev_open, prev_close)

        bullish_engulf = (
            (~prev_bullish)
            & curr_bullish
            & (curr_body_top > prev_body_top)
            & (curr_body_bot < prev_body_bot)
        )
        bearish_engulf = (
            prev_bullish
            & (~curr_bullish)
            & (curr_body_top > prev_body_top)
            & (curr_body_bot < prev_body_bot)
        )

        raw = np.where(bullish_engulf, 100.0, np.where(bearish_engulf, -100.0, 0.0))
        signal = pd.Series(raw, index=df.index) * self._volume_boost(df)
        return self._clamp(signal)

    def detect_harami(self, df: pd.DataFrame) -> pd.Series:
        """Detect harami (inside bar after a large candle).

        Bullish harami: large bearish candle followed by small bullish candle
        inside it.  Bearish harami is the mirror.

        Args:
            df: OHLCV DataFrame.

        Returns:
            Series of signal strengths.
        """
        prev_open = df["open"].shift(1)
        prev_close = df["close"].shift(1)
        prev_body = (prev_close - prev_open).abs()
        prev_bullish = prev_close >= prev_open

        curr_body_top = df[["open", "close"]].max(axis=1)
        curr_body_bot = df[["open", "close"]].min(axis=1)
        prev_body_top = np.maximum(prev_open, prev_close)
        prev_body_bot = np.minimum(prev_open, prev_close)

        curr_bullish = self._is_bullish(df)

        # Current body must be inside previous body
        inside = (curr_body_top <= prev_body_top) & (curr_body_bot >= prev_body_bot)

        # Previous candle should be sizeable relative to range
        prev_range = (df["high"].shift(1) - df["low"].shift(1)).replace(0, np.nan)
        large_prev = prev_body / prev_range > 0.5

        bullish_harami = (~prev_bullish) & curr_bullish & inside & large_prev
        bearish_harami = prev_bullish & (~curr_bullish) & inside & large_prev

        raw = np.where(
            bullish_harami, 80.0, np.where(bearish_harami, -80.0, 0.0)
        )
        signal = pd.Series(raw, index=df.index) * self._volume_boost(df)
        return self._clamp(signal)

    def detect_morning_star(self, df: pd.DataFrame) -> pd.Series:
        """Detect morning star — 3-candle bullish reversal.

        Bar 0: large bearish candle.
        Bar 1: small body (star) that gaps down.
        Bar 2: large bullish candle that closes into bar-0 body.

        Args:
            df: OHLCV DataFrame.

        Returns:
            Series of signal strengths (signal placed on bar 2).
        """
        body = self._body(df)
        rng = self._range(df)
        body_ratio = body / rng

        o0 = df["open"].shift(2)
        c0 = df["close"].shift(2)
        body0 = (c0 - o0).abs()
        range0 = (df["high"].shift(2) - df["low"].shift(2)).replace(0, np.nan)
        bearish0 = c0 < o0
        large0 = body0 / range0 > 0.5

        body1_ratio = body.shift(1) / rng.shift(1)
        small1 = body1_ratio < 0.30

        bullish2 = self._is_bullish(df)
        large2 = body_ratio > 0.5
        closes_into = df["close"] > (o0 + c0) / 2

        is_morning = bearish0 & large0 & small1 & bullish2 & large2 & closes_into

        raw = np.where(is_morning, 100.0, 0.0)
        signal = pd.Series(raw, index=df.index) * self._volume_boost(df)
        return self._clamp(signal)

    def detect_evening_star(self, df: pd.DataFrame) -> pd.Series:
        """Detect evening star — 3-candle bearish reversal.

        Mirror of morning star.

        Args:
            df: OHLCV DataFrame.

        Returns:
            Series of signal strengths (signal placed on bar 2).
        """
        body = self._body(df)
        rng = self._range(df)
        body_ratio = body / rng

        o0 = df["open"].shift(2)
        c0 = df["close"].shift(2)
        body0 = (c0 - o0).abs()
        range0 = (df["high"].shift(2) - df["low"].shift(2)).replace(0, np.nan)
        bullish0 = c0 > o0
        large0 = body0 / range0 > 0.5

        body1_ratio = body.shift(1) / rng.shift(1)
        small1 = body1_ratio < 0.30

        bearish2 = ~self._is_bullish(df)
        large2 = body_ratio > 0.5
        closes_into = df["close"] < (o0 + c0) / 2

        is_evening = bullish0 & large0 & small1 & bearish2 & large2 & closes_into

        raw = np.where(is_evening, -100.0, 0.0)
        signal = pd.Series(raw, index=df.index) * self._volume_boost(df)
        return self._clamp(signal)

    def detect_pin_bar(self, df: pd.DataFrame) -> pd.Series:
        """Detect pin bars — very long wick with a tiny body.

        A pin bar has one shadow at least 3x the body and the body is
        less than 15% of the total range.

        Args:
            df: OHLCV DataFrame.

        Returns:
            Series of signal strengths.
        """
        body = self._body(df)
        rng = self._range(df)
        upper = self._upper_shadow(df)
        lower = self._lower_shadow(df)
        body_ratio = body / rng

        tiny_body = body_ratio < 0.15

        bullish_pin = tiny_body & (lower >= 3 * body) & (lower > upper)
        bearish_pin = tiny_body & (upper >= 3 * body) & (upper > lower)

        raw = np.where(
            bullish_pin, 90.0, np.where(bearish_pin, -90.0, 0.0)
        )
        signal = pd.Series(raw, index=df.index) * self._volume_boost(df)
        return self._clamp(signal)

    def detect_inside_bar(self, df: pd.DataFrame) -> pd.Series:
        """Detect inside bars — entire range within previous bar.

        Args:
            df: OHLCV DataFrame.

        Returns:
            Series — +60 if bullish close, -60 if bearish close, 0 otherwise.
        """
        prev_high = df["high"].shift(1)
        prev_low = df["low"].shift(1)

        inside = (df["high"] <= prev_high) & (df["low"] >= prev_low)
        bullish = self._is_bullish(df)

        raw = np.where(inside, np.where(bullish, 60.0, -60.0), 0.0)
        signal = pd.Series(raw, index=df.index) * self._volume_boost(df)
        return self._clamp(signal)

    def detect_outside_bar(self, df: pd.DataFrame) -> pd.Series:
        """Detect outside bars — range engulfs previous bar.

        Args:
            df: OHLCV DataFrame.

        Returns:
            Series — +70 if bullish close, -70 if bearish close, 0 otherwise.
        """
        prev_high = df["high"].shift(1)
        prev_low = df["low"].shift(1)

        outside = (df["high"] > prev_high) & (df["low"] < prev_low)
        bullish = self._is_bullish(df)

        raw = np.where(outside, np.where(bullish, 70.0, -70.0), 0.0)
        signal = pd.Series(raw, index=df.index) * self._volume_boost(df)
        return self._clamp(signal)

    def detect_three_white_soldiers(self, df: pd.DataFrame) -> pd.Series:
        """Detect three white soldiers — 3 consecutive bullish candles.

        Each candle opens within the prior body and closes at a new high.

        Args:
            df: OHLCV DataFrame.

        Returns:
            Series of signal strengths (signal on bar 3).
        """
        bullish = self._is_bullish(df)
        body = self._body(df)
        rng = self._range(df)

        b0 = bullish.shift(2)
        b1 = bullish.shift(1)
        b2 = bullish

        # Higher closes
        hc1 = df["close"].shift(1) > df["close"].shift(2)
        hc2 = df["close"] > df["close"].shift(1)

        # Opens within previous body
        ow1 = (
            (df["open"].shift(1) >= df["open"].shift(2))
            & (df["open"].shift(1) <= df["close"].shift(2))
        )
        ow2 = (
            (df["open"] >= df["open"].shift(1))
            & (df["open"] <= df["close"].shift(1))
        )

        # Reasonable body size
        decent0 = body.shift(2) / rng.shift(2) > 0.4
        decent1 = body.shift(1) / rng.shift(1) > 0.4
        decent2 = body / rng > 0.4

        is_pattern = (
            b0 & b1 & b2 & hc1 & hc2 & ow1 & ow2
            & decent0 & decent1 & decent2
        )

        raw = np.where(is_pattern, 100.0, 0.0)
        signal = pd.Series(raw, index=df.index) * self._volume_boost(df)
        return self._clamp(signal)

    def detect_three_black_crows(self, df: pd.DataFrame) -> pd.Series:
        """Detect three black crows — 3 consecutive bearish candles.

        Each candle opens within the prior body and closes at a new low.

        Args:
            df: OHLCV DataFrame.

        Returns:
            Series of signal strengths (signal on bar 3).
        """
        bearish = ~self._is_bullish(df)
        body = self._body(df)
        rng = self._range(df)

        b0 = bearish.shift(2)
        b1 = bearish.shift(1)
        b2 = bearish

        # Lower closes
        lc1 = df["close"].shift(1) < df["close"].shift(2)
        lc2 = df["close"] < df["close"].shift(1)

        # Opens within previous body (bearish: open <= prev open, open >= prev close)
        ow1 = (
            (df["open"].shift(1) <= df["open"].shift(2))
            & (df["open"].shift(1) >= df["close"].shift(2))
        )
        ow2 = (
            (df["open"] <= df["open"].shift(1))
            & (df["open"] >= df["close"].shift(1))
        )

        decent0 = body.shift(2) / rng.shift(2) > 0.4
        decent1 = body.shift(1) / rng.shift(1) > 0.4
        decent2 = body / rng > 0.4

        is_pattern = (
            b0 & b1 & b2 & lc1 & lc2 & ow1 & ow2
            & decent0 & decent1 & decent2
        )

        raw = np.where(is_pattern, -100.0, 0.0)
        signal = pd.Series(raw, index=df.index) * self._volume_boost(df)
        return self._clamp(signal)

    # ------------------------------------------------------------------
    # B. CHART PATTERNS (geometric detection)
    # ------------------------------------------------------------------

    def detect_head_and_shoulders(
        self, df: pd.DataFrame, lookback: int = 100
    ) -> list[dict[str, Any]]:
        """Detect head-and-shoulders (bearish) and inverse (bullish).

        Uses swing-point detection to find left shoulder, head, and right
        shoulder.  The neckline is drawn through the troughs (or peaks for
        inverse) between the shoulders.

        Args:
            df: OHLCV DataFrame.
            lookback: Number of bars to scan.

        Returns:
            List of pattern dictionaries.
        """
        patterns: list[dict[str, Any]] = []
        if len(df) < lookback:
            lookback = len(df)
        data = df.iloc[-lookback:].reset_index(drop=True)
        highs = data["high"].values.astype(float)
        lows = data["low"].values.astype(float)

        order = self.swing_order
        sh = self._swing_highs(highs, order)
        sl = self._swing_lows(lows, order)

        sh_idx = np.where(sh)[0]
        sl_idx = np.where(sl)[0]

        # --- Regular H&S (bearish) ---
        if len(sh_idx) >= 3 and len(sl_idx) >= 2:
            for i in range(len(sh_idx) - 2):
                ls_i, h_i, rs_i = sh_idx[i], sh_idx[i + 1], sh_idx[i + 2]
                ls_v, h_v, rs_v = highs[ls_i], highs[h_i], highs[rs_i]

                # Head must be highest
                if h_v <= ls_v or h_v <= rs_v:
                    continue
                # Shoulders roughly equal (within 5% of head height)
                if abs(ls_v - rs_v) / h_v > 0.05:
                    continue

                # Find troughs between shoulders
                t1_candidates = sl_idx[(sl_idx > ls_i) & (sl_idx < h_i)]
                t2_candidates = sl_idx[(sl_idx > h_i) & (sl_idx < rs_i)]
                if len(t1_candidates) == 0 or len(t2_candidates) == 0:
                    continue
                t1_i = t1_candidates[np.argmin(lows[t1_candidates])]
                t2_i = t2_candidates[np.argmin(lows[t2_candidates])]
                t1_v, t2_v = lows[t1_i], lows[t2_i]

                neckline = (t1_v + t2_v) / 2
                pattern_height = h_v - neckline
                target = neckline - pattern_height

                # Probability based on symmetry and volume
                symmetry = 1.0 - abs(ls_v - rs_v) / h_v
                prob = min(100, int(symmetry * 80 + 10))

                start_abs = len(df) - lookback + int(ls_i)
                end_abs = len(df) - lookback + int(rs_i)
                expected_bars = int((rs_i - ls_i) * 0.5)

                patterns.append({
                    "pattern_type": "head_and_shoulders",
                    "start_index": start_abs,
                    "end_index": end_abs,
                    "probability_score": prob,
                    "measured_move_target": float(target),
                    "invalidation_level": float(h_v),
                    "expected_completion_bars": max(1, expected_bars),
                    "direction": "bearish",
                })

        # --- Inverse H&S (bullish) ---
        if len(sl_idx) >= 3 and len(sh_idx) >= 2:
            for i in range(len(sl_idx) - 2):
                ls_i, h_i, rs_i = sl_idx[i], sl_idx[i + 1], sl_idx[i + 2]
                ls_v, h_v, rs_v = lows[ls_i], lows[h_i], lows[rs_i]

                # Head must be lowest
                if h_v >= ls_v or h_v >= rs_v:
                    continue
                avg_shoulder = (ls_v + rs_v) / 2
                if avg_shoulder == 0:
                    continue
                if abs(ls_v - rs_v) / avg_shoulder > 0.05:
                    continue

                # Peaks between troughs
                p1_candidates = sh_idx[(sh_idx > ls_i) & (sh_idx < h_i)]
                p2_candidates = sh_idx[(sh_idx > h_i) & (sh_idx < rs_i)]
                if len(p1_candidates) == 0 or len(p2_candidates) == 0:
                    continue
                p1_i = p1_candidates[np.argmax(highs[p1_candidates])]
                p2_i = p2_candidates[np.argmax(highs[p2_candidates])]
                p1_v, p2_v = highs[p1_i], highs[p2_i]

                neckline = (p1_v + p2_v) / 2
                pattern_height = neckline - h_v
                target = neckline + pattern_height

                symmetry = 1.0 - abs(ls_v - rs_v) / avg_shoulder
                prob = min(100, int(symmetry * 80 + 10))

                start_abs = len(df) - lookback + int(ls_i)
                end_abs = len(df) - lookback + int(rs_i)
                expected_bars = int((rs_i - ls_i) * 0.5)

                patterns.append({
                    "pattern_type": "inverse_head_and_shoulders",
                    "start_index": start_abs,
                    "end_index": end_abs,
                    "probability_score": prob,
                    "measured_move_target": float(target),
                    "invalidation_level": float(h_v),
                    "expected_completion_bars": max(1, expected_bars),
                    "direction": "bullish",
                })

        return patterns

    def detect_double_top_bottom(
        self, df: pd.DataFrame, lookback: int = 100
    ) -> list[dict[str, Any]]:
        """Detect double top and double bottom patterns.

        Two peaks (or troughs) at similar levels with a valley (or peak)
        between them.  Tolerance is 1% of price.

        Args:
            df: OHLCV DataFrame.
            lookback: Number of bars to scan.

        Returns:
            List of pattern dictionaries.
        """
        patterns: list[dict[str, Any]] = []
        if len(df) < 10:
            return patterns
        if len(df) < lookback:
            lookback = len(df)
        data = df.iloc[-lookback:].reset_index(drop=True)
        highs = data["high"].values.astype(float)
        lows = data["low"].values.astype(float)
        order = self.swing_order

        sh_idx = np.where(self._swing_highs(highs, order))[0]
        sl_idx = np.where(self._swing_lows(lows, order))[0]

        tolerance = 0.01

        # Double top (bearish)
        for i in range(len(sh_idx) - 1):
            p1, p2 = sh_idx[i], sh_idx[i + 1]
            v1, v2 = highs[p1], highs[p2]
            avg = (v1 + v2) / 2
            if avg == 0:
                continue
            if abs(v1 - v2) / avg > tolerance:
                continue
            # Find trough between peaks
            between = sl_idx[(sl_idx > p1) & (sl_idx < p2)]
            if len(between) == 0:
                continue
            trough_i = between[np.argmin(lows[between])]
            neckline = lows[trough_i]
            pattern_height = avg - neckline
            target = neckline - pattern_height

            start_abs = len(df) - lookback + int(p1)
            end_abs = len(df) - lookback + int(p2)

            patterns.append({
                "pattern_type": "double_top",
                "start_index": start_abs,
                "end_index": end_abs,
                "probability_score": 70,
                "measured_move_target": float(target),
                "invalidation_level": float(max(v1, v2)),
                "expected_completion_bars": max(1, int((p2 - p1) * 0.5)),
                "direction": "bearish",
            })

        # Double bottom (bullish)
        for i in range(len(sl_idx) - 1):
            t1, t2 = sl_idx[i], sl_idx[i + 1]
            v1, v2 = lows[t1], lows[t2]
            avg = (v1 + v2) / 2
            if avg == 0:
                continue
            if abs(v1 - v2) / avg > tolerance:
                continue
            between = sh_idx[(sh_idx > t1) & (sh_idx < t2)]
            if len(between) == 0:
                continue
            peak_i = between[np.argmax(highs[between])]
            neckline = highs[peak_i]
            pattern_height = neckline - avg
            target = neckline + pattern_height

            start_abs = len(df) - lookback + int(t1)
            end_abs = len(df) - lookback + int(t2)

            patterns.append({
                "pattern_type": "double_bottom",
                "start_index": start_abs,
                "end_index": end_abs,
                "probability_score": 70,
                "measured_move_target": float(target),
                "invalidation_level": float(min(v1, v2)),
                "expected_completion_bars": max(1, int((t2 - t1) * 0.5)),
                "direction": "bullish",
            })

        return patterns

    def detect_triple_top_bottom(
        self, df: pd.DataFrame, lookback: int = 150
    ) -> list[dict[str, Any]]:
        """Detect triple top and triple bottom patterns.

        Three peaks (or troughs) at similar levels.

        Args:
            df: OHLCV DataFrame.
            lookback: Number of bars to scan.

        Returns:
            List of pattern dictionaries.
        """
        patterns: list[dict[str, Any]] = []
        if len(df) < 15:
            return patterns
        if len(df) < lookback:
            lookback = len(df)
        data = df.iloc[-lookback:].reset_index(drop=True)
        highs = data["high"].values.astype(float)
        lows = data["low"].values.astype(float)
        order = self.swing_order

        sh_idx = np.where(self._swing_highs(highs, order))[0]
        sl_idx = np.where(self._swing_lows(lows, order))[0]

        tolerance = 0.015

        # Triple top
        for i in range(len(sh_idx) - 2):
            p1, p2, p3 = sh_idx[i], sh_idx[i + 1], sh_idx[i + 2]
            v1, v2, v3 = highs[p1], highs[p2], highs[p3]
            avg = (v1 + v2 + v3) / 3
            if avg == 0:
                continue
            if (
                abs(v1 - avg) / avg > tolerance
                or abs(v2 - avg) / avg > tolerance
                or abs(v3 - avg) / avg > tolerance
            ):
                continue

            # Find lowest trough between first and last peak
            between = sl_idx[(sl_idx > p1) & (sl_idx < p3)]
            if len(between) == 0:
                continue
            neckline = float(np.min(lows[between]))
            pattern_height = avg - neckline
            target = neckline - pattern_height

            start_abs = len(df) - lookback + int(p1)
            end_abs = len(df) - lookback + int(p3)

            patterns.append({
                "pattern_type": "triple_top",
                "start_index": start_abs,
                "end_index": end_abs,
                "probability_score": 75,
                "measured_move_target": float(target),
                "invalidation_level": float(max(v1, v2, v3)),
                "expected_completion_bars": max(1, int((p3 - p1) * 0.4)),
                "direction": "bearish",
            })

        # Triple bottom
        for i in range(len(sl_idx) - 2):
            t1, t2, t3 = sl_idx[i], sl_idx[i + 1], sl_idx[i + 2]
            v1, v2, v3 = lows[t1], lows[t2], lows[t3]
            avg = (v1 + v2 + v3) / 3
            if avg == 0:
                continue
            if (
                abs(v1 - avg) / avg > tolerance
                or abs(v2 - avg) / avg > tolerance
                or abs(v3 - avg) / avg > tolerance
            ):
                continue

            between = sh_idx[(sh_idx > t1) & (sh_idx < t3)]
            if len(between) == 0:
                continue
            neckline = float(np.max(highs[between]))
            pattern_height = neckline - avg
            target = neckline + pattern_height

            start_abs = len(df) - lookback + int(t1)
            end_abs = len(df) - lookback + int(t3)

            patterns.append({
                "pattern_type": "triple_bottom",
                "start_index": start_abs,
                "end_index": end_abs,
                "probability_score": 75,
                "measured_move_target": float(target),
                "invalidation_level": float(min(v1, v2, v3)),
                "expected_completion_bars": max(1, int((t3 - t1) * 0.4)),
                "direction": "bullish",
            })

        return patterns

    def detect_triangle(
        self, df: pd.DataFrame, lookback: int = 80
    ) -> list[dict[str, Any]]:
        """Detect ascending, descending, and symmetrical triangles.

        Uses linear regression on swing highs and swing lows to determine
        trendline slopes.  Classification:
        - Ascending: higher lows (positive low-slope), flat highs
        - Descending: flat lows, lower highs (negative high-slope)
        - Symmetrical: converging trendlines (positive low-slope, negative
          high-slope)

        Args:
            df: OHLCV DataFrame.
            lookback: Number of bars to scan.

        Returns:
            List of pattern dictionaries.
        """
        patterns: list[dict[str, Any]] = []
        if len(df) < 20:
            return patterns
        if len(df) < lookback:
            lookback = len(df)
        data = df.iloc[-lookback:].reset_index(drop=True)
        highs = data["high"].values.astype(float)
        lows = data["low"].values.astype(float)
        closes = data["close"].values.astype(float)
        order = self.swing_order

        sh_idx = np.where(self._swing_highs(highs, order))[0]
        sl_idx = np.where(self._swing_lows(lows, order))[0]

        if len(sh_idx) < 3 or len(sl_idx) < 3:
            return patterns

        # Use the most recent swing points
        sh_vals = highs[sh_idx]
        sl_vals = lows[sl_idx]

        h_slope, h_intercept, h_r2 = self._fit_line(
            sh_idx.astype(float), sh_vals
        )
        l_slope, l_intercept, l_r2 = self._fit_line(
            sl_idx.astype(float), sl_vals
        )

        # Minimum fit quality
        if h_r2 < 0.4 or l_r2 < 0.4:
            return patterns

        # Normalise slopes by average price to compare
        avg_price = np.mean(closes)
        if avg_price == 0:
            return patterns
        h_slope_norm = h_slope / avg_price
        l_slope_norm = l_slope / avg_price

        flat_threshold = 0.0002  # roughly flat
        converging = h_slope < 0 or l_slope > 0  # at least one converging

        # Must be converging to be a triangle
        if h_slope_norm - l_slope_norm >= 0:
            # trendlines diverging — not a triangle
            return patterns

        start_abs = len(df) - lookback + int(min(sh_idx[0], sl_idx[0]))
        end_abs = len(df) - lookback + int(max(sh_idx[-1], sl_idx[-1]))
        last_bar = max(sh_idx[-1], sl_idx[-1])

        # Apex approximation
        if abs(h_slope - l_slope) > 1e-12:
            apex_bar = (l_intercept - h_intercept) / (h_slope - l_slope)
        else:
            apex_bar = last_bar + 30

        expected_bars = max(1, int(apex_bar - last_bar))

        # Resistance / support at the last bar
        resistance = h_slope * last_bar + h_intercept
        support = l_slope * last_bar + l_intercept
        pattern_height = resistance - support

        if abs(h_slope_norm) < flat_threshold and l_slope_norm > flat_threshold:
            # Ascending triangle (bullish)
            patterns.append({
                "pattern_type": "ascending_triangle",
                "start_index": start_abs,
                "end_index": end_abs,
                "probability_score": min(100, int(min(h_r2, l_r2) * 100)),
                "measured_move_target": float(resistance + pattern_height),
                "invalidation_level": float(support),
                "expected_completion_bars": expected_bars,
                "direction": "bullish",
            })
        elif abs(l_slope_norm) < flat_threshold and h_slope_norm < -flat_threshold:
            # Descending triangle (bearish)
            patterns.append({
                "pattern_type": "descending_triangle",
                "start_index": start_abs,
                "end_index": end_abs,
                "probability_score": min(100, int(min(h_r2, l_r2) * 100)),
                "measured_move_target": float(support - pattern_height),
                "invalidation_level": float(resistance),
                "expected_completion_bars": expected_bars,
                "direction": "bearish",
            })
        elif h_slope_norm < -flat_threshold and l_slope_norm > flat_threshold:
            # Symmetrical triangle — direction depends on prior trend
            prior_trend = closes[-1] - closes[0]
            direction = "bullish" if prior_trend > 0 else "bearish"
            target = (
                float(resistance + pattern_height)
                if direction == "bullish"
                else float(support - pattern_height)
            )
            inv = (
                float(support) if direction == "bullish" else float(resistance)
            )
            patterns.append({
                "pattern_type": "symmetrical_triangle",
                "start_index": start_abs,
                "end_index": end_abs,
                "probability_score": min(100, int(min(h_r2, l_r2) * 90)),
                "measured_move_target": target,
                "invalidation_level": inv,
                "expected_completion_bars": expected_bars,
                "direction": direction,
            })

        return patterns

    def detect_flag_pennant(
        self, df: pd.DataFrame, lookback: int = 50
    ) -> list[dict[str, Any]]:
        """Detect bull/bear flags and pennants.

        A flag/pennant consists of a strong directional move (pole) followed
        by a tight consolidation.  A flag has roughly parallel trendlines
        while a pennant has converging trendlines.

        Args:
            df: OHLCV DataFrame.
            lookback: Number of bars to scan.

        Returns:
            List of pattern dictionaries.
        """
        patterns: list[dict[str, Any]] = []
        if len(df) < 15:
            return patterns
        if len(df) < lookback:
            lookback = len(df)
        data = df.iloc[-lookback:].reset_index(drop=True)
        highs = data["high"].values.astype(float)
        lows = data["low"].values.astype(float)
        closes = data["close"].values.astype(float)
        n = len(data)

        # Try different pole lengths
        for pole_len in range(5, min(25, n - 10)):
            consol_start = pole_len
            consol_end = n
            consol_len = consol_end - consol_start
            if consol_len < 5:
                continue

            pole_move = closes[pole_len - 1] - closes[0]
            avg_price = np.mean(closes[:pole_len])
            if avg_price == 0:
                continue
            pole_pct = abs(pole_move) / avg_price

            # Pole must be a strong move (>3%)
            if pole_pct < 0.03:
                continue

            # Consolidation range should be tight relative to pole
            consol_highs = highs[consol_start:consol_end]
            consol_lows = lows[consol_start:consol_end]
            consol_range = np.max(consol_highs) - np.min(consol_lows)
            if consol_range == 0:
                continue
            if consol_range / abs(pole_move) > 0.50:
                continue

            # Fit trendlines on consolidation
            x = np.arange(consol_len, dtype=float)
            h_slope, h_int, h_r2 = self._fit_line(x, consol_highs)
            l_slope, l_int, l_r2 = self._fit_line(x, consol_lows)

            is_bull = pole_move > 0
            direction = "bullish" if is_bull else "bearish"

            # Classify flag vs pennant
            slope_diff = abs(h_slope - l_slope) / avg_price
            converging = (h_slope < 0 and l_slope > 0) or (
                abs(h_slope - l_slope) > abs(h_slope + l_slope) * 0.3
            )

            if converging:
                ptype = "bull_pennant" if is_bull else "bear_pennant"
            else:
                ptype = "bull_flag" if is_bull else "bear_flag"

            target_move = abs(pole_move)
            if is_bull:
                target = float(closes[-1] + target_move)
                inv = float(np.min(consol_lows))
            else:
                target = float(closes[-1] - target_move)
                inv = float(np.max(consol_highs))

            prob = min(100, int(50 + pole_pct * 200))

            start_abs = len(df) - lookback
            end_abs = len(df) - 1

            patterns.append({
                "pattern_type": ptype,
                "start_index": start_abs,
                "end_index": end_abs,
                "probability_score": prob,
                "measured_move_target": target,
                "invalidation_level": inv,
                "expected_completion_bars": max(1, consol_len),
                "direction": direction,
            })
            # Take best pole length only
            break

        return patterns

    def detect_wedge(
        self, df: pd.DataFrame, lookback: int = 80
    ) -> list[dict[str, Any]]:
        """Detect rising and falling wedge patterns.

        A wedge has both trendlines sloping in the same direction but
        converging.  Rising wedge is bearish; falling wedge is bullish.

        Args:
            df: OHLCV DataFrame.
            lookback: Number of bars to scan.

        Returns:
            List of pattern dictionaries.
        """
        patterns: list[dict[str, Any]] = []
        if len(df) < 20:
            return patterns
        if len(df) < lookback:
            lookback = len(df)
        data = df.iloc[-lookback:].reset_index(drop=True)
        highs = data["high"].values.astype(float)
        lows = data["low"].values.astype(float)
        closes = data["close"].values.astype(float)
        order = self.swing_order

        sh_idx = np.where(self._swing_highs(highs, order))[0]
        sl_idx = np.where(self._swing_lows(lows, order))[0]

        if len(sh_idx) < 3 or len(sl_idx) < 3:
            return patterns

        h_slope, h_int, h_r2 = self._fit_line(
            sh_idx.astype(float), highs[sh_idx]
        )
        l_slope, l_int, l_r2 = self._fit_line(
            sl_idx.astype(float), lows[sl_idx]
        )

        if h_r2 < 0.5 or l_r2 < 0.5:
            return patterns

        # Both slopes same direction and converging
        same_dir = (h_slope > 0 and l_slope > 0) or (h_slope < 0 and l_slope < 0)
        if not same_dir:
            return patterns

        # Must converge: the steeper one is the outer trendline
        # For rising wedge: l_slope > h_slope (lows rising faster)
        # For falling wedge: h_slope < l_slope in magnitude (highs falling slower)
        converging = abs(h_slope - l_slope) > 1e-10

        if not converging:
            return patterns

        last_bar = max(sh_idx[-1], sl_idx[-1])
        resistance = h_slope * last_bar + h_int
        support = l_slope * last_bar + l_int
        pattern_height = resistance - support

        start_abs = len(df) - lookback + int(min(sh_idx[0], sl_idx[0]))
        end_abs = len(df) - lookback + int(last_bar)

        # Apex
        if abs(h_slope - l_slope) > 1e-12:
            apex_bar = (l_int - h_int) / (h_slope - l_slope)
        else:
            apex_bar = last_bar + 30
        expected_bars = max(1, int(apex_bar - last_bar))

        prob = min(100, int(min(h_r2, l_r2) * 100))

        if h_slope > 0 and l_slope > 0:
            # Rising wedge — bearish
            patterns.append({
                "pattern_type": "rising_wedge",
                "start_index": start_abs,
                "end_index": end_abs,
                "probability_score": prob,
                "measured_move_target": float(support - pattern_height),
                "invalidation_level": float(resistance),
                "expected_completion_bars": expected_bars,
                "direction": "bearish",
            })
        else:
            # Falling wedge — bullish
            patterns.append({
                "pattern_type": "falling_wedge",
                "start_index": start_abs,
                "end_index": end_abs,
                "probability_score": prob,
                "measured_move_target": float(resistance + pattern_height),
                "invalidation_level": float(support),
                "expected_completion_bars": expected_bars,
                "direction": "bullish",
            })

        return patterns

    def detect_cup_and_handle(
        self, df: pd.DataFrame, lookback: int = 120
    ) -> list[dict[str, Any]]:
        """Detect cup-and-handle pattern (bullish continuation).

        Looks for a U-shaped cup followed by a small pullback (handle).
        The cup is identified by a decline, rounded bottom, and recovery
        back to the rim level.  The handle is a small retracement.

        Args:
            df: OHLCV DataFrame.
            lookback: Number of bars to scan.

        Returns:
            List of pattern dictionaries.
        """
        patterns: list[dict[str, Any]] = []
        if len(df) < 30:
            return patterns
        if len(df) < lookback:
            lookback = len(df)
        data = df.iloc[-lookback:].reset_index(drop=True)
        closes = data["close"].values.astype(float)
        highs = data["high"].values.astype(float)
        lows = data["low"].values.astype(float)
        n = len(data)

        # Scan for potential cups: left rim, bottom, right rim
        order = self.swing_order
        sh_idx = np.where(self._swing_highs(highs, order))[0]
        sl_idx = np.where(self._swing_lows(lows, order))[0]

        if len(sh_idx) < 2 or len(sl_idx) < 1:
            return patterns

        for i in range(len(sh_idx) - 1):
            left_rim_i = sh_idx[i]
            right_rim_i = sh_idx[i + 1]
            left_rim_v = highs[left_rim_i]
            right_rim_v = highs[right_rim_i]

            # Rims should be at similar level (within 3%)
            rim_avg = (left_rim_v + right_rim_v) / 2
            if rim_avg == 0:
                continue
            if abs(left_rim_v - right_rim_v) / rim_avg > 0.03:
                continue

            # Cup bottom: lowest low between rims
            cup_lows = sl_idx[
                (sl_idx > left_rim_i) & (sl_idx < right_rim_i)
            ]
            if len(cup_lows) == 0:
                continue
            bottom_i = cup_lows[np.argmin(lows[cup_lows])]
            bottom_v = lows[bottom_i]

            # Cup depth should be meaningful (at least 10% of rim)
            cup_depth = rim_avg - bottom_v
            if cup_depth / rim_avg < 0.05:
                continue

            # Bottom should be roughly centred (within 40%-60% of cup width)
            cup_width = right_rim_i - left_rim_i
            if cup_width < 10:
                continue
            bottom_pos = (bottom_i - left_rim_i) / cup_width
            if bottom_pos < 0.25 or bottom_pos > 0.75:
                continue

            # Check for handle after right rim: small pullback
            handle_end = min(right_rim_i + int(cup_width * 0.3), n - 1)
            if handle_end <= right_rim_i:
                continue

            handle_section = closes[right_rim_i: handle_end + 1]
            handle_low = np.min(handle_section)
            handle_retrace = (right_rim_v - handle_low) / cup_depth

            # Handle should retrace < 50% of cup depth
            if handle_retrace > 0.50:
                continue

            target = float(rim_avg + cup_depth)
            inv = float(handle_low)

            start_abs = len(df) - lookback + int(left_rim_i)
            end_abs = len(df) - lookback + int(handle_end)
            expected_bars = max(1, int(cup_width * 0.3))

            prob = min(100, max(50, int(70 + (1.0 - handle_retrace) * 30)))

            patterns.append({
                "pattern_type": "cup_and_handle",
                "start_index": start_abs,
                "end_index": end_abs,
                "probability_score": prob,
                "measured_move_target": target,
                "invalidation_level": inv,
                "expected_completion_bars": expected_bars,
                "direction": "bullish",
            })

        return patterns

    def detect_rectangle(
        self, df: pd.DataFrame, lookback: int = 60
    ) -> list[dict[str, Any]]:
        """Detect rectangle (range-bound) patterns.

        Horizontal support and resistance with price oscillating between
        them.  Breakout direction is determined by the prior trend.

        Args:
            df: OHLCV DataFrame.
            lookback: Number of bars to scan.

        Returns:
            List of pattern dictionaries.
        """
        patterns: list[dict[str, Any]] = []
        if len(df) < 15:
            return patterns
        if len(df) < lookback:
            lookback = len(df)
        data = df.iloc[-lookback:].reset_index(drop=True)
        highs = data["high"].values.astype(float)
        lows = data["low"].values.astype(float)
        closes = data["close"].values.astype(float)
        order = self.swing_order

        sh_idx = np.where(self._swing_highs(highs, order))[0]
        sl_idx = np.where(self._swing_lows(lows, order))[0]

        if len(sh_idx) < 2 or len(sl_idx) < 2:
            return patterns

        # Fit lines — both should be roughly flat
        h_slope, h_int, h_r2 = self._fit_line(
            sh_idx.astype(float), highs[sh_idx]
        )
        l_slope, l_int, l_r2 = self._fit_line(
            sl_idx.astype(float), lows[sl_idx]
        )

        avg_price = np.mean(closes)
        if avg_price == 0:
            return patterns

        h_slope_norm = abs(h_slope / avg_price)
        l_slope_norm = abs(l_slope / avg_price)

        flat_threshold = 0.0003

        if h_slope_norm > flat_threshold or l_slope_norm > flat_threshold:
            return patterns

        resistance = float(np.mean(highs[sh_idx]))
        support = float(np.mean(lows[sl_idx]))
        rect_height = resistance - support

        if rect_height <= 0:
            return patterns

        # Prior trend determines expected breakout
        pre_start = max(0, int(min(sh_idx[0], sl_idx[0])) - 10)
        pre_end = int(min(sh_idx[0], sl_idx[0]))
        if pre_end > pre_start:
            prior_trend = closes[pre_end] - closes[pre_start]
        else:
            prior_trend = 0.0

        direction = "bullish" if prior_trend > 0 else "bearish"
        if direction == "bullish":
            target = resistance + rect_height
            inv = support
        else:
            target = support - rect_height
            inv = resistance

        start_abs = len(df) - lookback + int(min(sh_idx[0], sl_idx[0]))
        end_abs = len(df) - lookback + int(max(sh_idx[-1], sl_idx[-1]))
        span = int(max(sh_idx[-1], sl_idx[-1]) - min(sh_idx[0], sl_idx[0]))

        prob = min(100, int(min(h_r2, l_r2) * 80 + 20))

        patterns.append({
            "pattern_type": "rectangle",
            "start_index": start_abs,
            "end_index": end_abs,
            "probability_score": prob,
            "measured_move_target": float(target),
            "invalidation_level": float(inv),
            "expected_completion_bars": max(1, int(span * 0.3)),
            "direction": direction,
        })

        return patterns

    # ------------------------------------------------------------------
    # C. COMPREHENSIVE
    # ------------------------------------------------------------------

    def detect_all_candlestick(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all candlestick pattern detectors on the DataFrame.

        Adds one column per pattern to a copy of the input DataFrame.

        Args:
            df: OHLCV DataFrame.

        Returns:
            DataFrame with original columns plus one column per candlestick
            pattern, each containing signal strengths from -100 to +100.
        """
        result = df.copy()
        result["doji"] = self.detect_doji(df)
        result["hammer"] = self.detect_hammer(df)
        result["shooting_star"] = self.detect_shooting_star(df)
        result["engulfing"] = self.detect_engulfing(df)
        result["harami"] = self.detect_harami(df)
        result["morning_star"] = self.detect_morning_star(df)
        result["evening_star"] = self.detect_evening_star(df)
        result["pin_bar"] = self.detect_pin_bar(df)
        result["inside_bar"] = self.detect_inside_bar(df)
        result["outside_bar"] = self.detect_outside_bar(df)
        result["three_white_soldiers"] = self.detect_three_white_soldiers(df)
        result["three_black_crows"] = self.detect_three_black_crows(df)
        return result

    def detect_all_chart_patterns(
        self, df: pd.DataFrame
    ) -> list[dict[str, Any]]:
        """Run all chart-pattern detectors and return a combined list.

        Results are sorted by ``probability_score`` in descending order.

        Args:
            df: OHLCV DataFrame.

        Returns:
            List of pattern dictionaries from all detectors, sorted by
            probability score (highest first).
        """
        all_patterns: list[dict[str, Any]] = []
        all_patterns.extend(self.detect_head_and_shoulders(df))
        all_patterns.extend(self.detect_double_top_bottom(df))
        all_patterns.extend(self.detect_triple_top_bottom(df))
        all_patterns.extend(self.detect_triangle(df))
        all_patterns.extend(self.detect_flag_pennant(df))
        all_patterns.extend(self.detect_wedge(df))
        all_patterns.extend(self.detect_cup_and_handle(df))
        all_patterns.extend(self.detect_rectangle(df))
        all_patterns.sort(key=lambda p: p["probability_score"], reverse=True)
        return all_patterns
