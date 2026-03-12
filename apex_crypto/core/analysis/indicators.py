"""
Technical Indicator Library for the APEX Crypto Trading System.

Computes trend, momentum, volatility, and volume indicators on OHLCV
DataFrames.  Uses TA-Lib when available, falls back to pure
numpy/pandas implementations.

Input DataFrame expected columns:
    timestamp, open, high, low, close, volume
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional TA-Lib import
# ---------------------------------------------------------------------------
try:
    import talib

    _HAS_TALIB = True
except ImportError:
    _HAS_TALIB = False

logger = logging.getLogger(__name__)


def _log_timing(indicator: str, timeframe: str, elapsed: float, rows: int) -> None:
    """Emit a structured JSON log line for computation timing."""
    logger.debug(
        json.dumps(
            {
                "event": "indicator_computed",
                "indicator": indicator,
                "timeframe": timeframe,
                "elapsed_ms": round(elapsed * 1000, 3),
                "rows": rows,
            }
        )
    )


class IndicatorEngine:
    """Compute all technical indicators consumed by the APEX trading system.

    Args:
        config: The ``indicators`` section of ``config.yaml``, e.g.::

            {
                "ema_periods": [9, 21, 50, 100, 200],
                "rsi_periods": [7, 14, 21],
                "macd": {"fast": 12, "slow": 26, "signal": 9},
                ...
            }
    """

    def __init__(self, config: dict) -> None:
        self.config: dict = config

    # ===================================================================== #
    #  HELPERS                                                               #
    # ===================================================================== #

    @staticmethod
    def _ensure_float(series: pd.Series) -> np.ndarray:
        """Return a contiguous float64 ndarray (required by TA-Lib)."""
        return np.ascontiguousarray(series.values, dtype=np.float64)

    @staticmethod
    def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Compute True Range."""
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    @staticmethod
    def _ema_series(series: pd.Series, period: int) -> pd.Series:
        """Raw EMA on an arbitrary Series."""
        return series.ewm(span=period, adjust=False).mean()

    # ===================================================================== #
    #  A.  TREND INDICATORS                                                  #
    # ===================================================================== #

    def compute_ema(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Compute Exponential Moving Average.

        Args:
            df: OHLCV DataFrame.
            period: EMA look-back window.

        Returns:
            pd.Series with EMA values.
        """
        close = df["close"]
        if _HAS_TALIB:
            arr = talib.EMA(self._ensure_float(close), timeperiod=period)
            return pd.Series(arr, index=df.index, name=f"ema_{period}")
        return close.ewm(span=period, adjust=False).mean().rename(f"ema_{period}")

    def compute_dema(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Compute Double Exponential Moving Average.

        DEMA = 2 * EMA(close, p) - EMA(EMA(close, p), p)

        Args:
            df: OHLCV DataFrame.
            period: Look-back window.

        Returns:
            pd.Series with DEMA values.
        """
        if _HAS_TALIB:
            arr = talib.DEMA(self._ensure_float(df["close"]), timeperiod=period)
            return pd.Series(arr, index=df.index, name=f"dema_{period}")
        ema1 = df["close"].ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        return (2 * ema1 - ema2).rename(f"dema_{period}")

    def compute_tema(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Compute Triple Exponential Moving Average.

        TEMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))

        Args:
            df: OHLCV DataFrame.
            period: Look-back window.

        Returns:
            pd.Series with TEMA values.
        """
        if _HAS_TALIB:
            arr = talib.TEMA(self._ensure_float(df["close"]), timeperiod=period)
            return pd.Series(arr, index=df.index, name=f"tema_{period}")
        ema1 = df["close"].ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        return (3 * ema1 - 3 * ema2 + ema3).rename(f"tema_{period}")

    def compute_supertrend(
        self, df: pd.DataFrame, period: int = 10, multiplier: float = 2.0
    ) -> pd.DataFrame:
        """Compute Supertrend indicator.

        Args:
            df: OHLCV DataFrame.
            period: ATR look-back period.
            multiplier: ATR multiplier for band width.

        Returns:
            DataFrame with columns ``supertrend`` and ``direction``
            (1 = bullish, -1 = bearish).
        """
        hl2 = (df["high"] + df["low"]) / 2.0
        atr = self.compute_atr(df, period)

        upper_band = hl2 + multiplier * atr
        lower_band = hl2 - multiplier * atr

        n = len(df)
        direction = np.ones(n, dtype=np.float64)
        supertrend = np.empty(n, dtype=np.float64)
        supertrend[:] = np.nan

        ub = upper_band.values.copy()
        lb = lower_band.values.copy()
        close = df["close"].values

        for i in range(1, n):
            # Lower band: keep previous if previous was higher
            if lb[i] < lb[i - 1] and close[i - 1] > lb[i - 1]:
                lb[i] = lb[i - 1]
            # Upper band: keep previous if previous was lower
            if ub[i] > ub[i - 1] and close[i - 1] < ub[i - 1]:
                ub[i] = ub[i - 1]

            if direction[i - 1] == 1.0:
                # Bullish — switch to bearish when close drops below lower band
                if close[i] < lb[i]:
                    direction[i] = -1.0
                    supertrend[i] = ub[i]
                else:
                    direction[i] = 1.0
                    supertrend[i] = lb[i]
            else:
                # Bearish — switch to bullish when close rises above upper band
                if close[i] > ub[i]:
                    direction[i] = 1.0
                    supertrend[i] = lb[i]
                else:
                    direction[i] = -1.0
                    supertrend[i] = ub[i]

        return pd.DataFrame(
            {"supertrend": supertrend, "direction": direction}, index=df.index
        )

    def compute_ichimoku(
        self,
        df: pd.DataFrame,
        tenkan: int = 9,
        kijun: int = 26,
        senkou: int = 52,
        chikou: int = 26,
    ) -> pd.DataFrame:
        """Compute Ichimoku Cloud components.

        Args:
            df: OHLCV DataFrame.
            tenkan: Tenkan-sen (conversion line) period.
            kijun: Kijun-sen (base line) period.
            senkou: Senkou Span B period.
            chikou: Chikou Span displacement.

        Returns:
            DataFrame with columns tenkan_sen, kijun_sen, senkou_span_a,
            senkou_span_b, chikou_span.
        """
        high = df["high"]
        low = df["low"]

        tenkan_sen = (
            high.rolling(window=tenkan).max() + low.rolling(window=tenkan).min()
        ) / 2.0
        kijun_sen = (
            high.rolling(window=kijun).max() + low.rolling(window=kijun).min()
        ) / 2.0

        senkou_span_a = ((tenkan_sen + kijun_sen) / 2.0).shift(kijun)
        senkou_span_b = (
            (
                high.rolling(window=senkou).max()
                + low.rolling(window=senkou).min()
            )
            / 2.0
        ).shift(kijun)

        chikou_span = df["close"].shift(-chikou)

        return pd.DataFrame(
            {
                "tenkan_sen": tenkan_sen,
                "kijun_sen": kijun_sen,
                "senkou_span_a": senkou_span_a,
                "senkou_span_b": senkou_span_b,
                "chikou_span": chikou_span,
            },
            index=df.index,
        )

    def compute_parabolic_sar(self, df: pd.DataFrame) -> pd.Series:
        """Compute Parabolic SAR.

        Uses acceleration factor starting at 0.02, incrementing by 0.02,
        with a maximum of 0.20.

        Args:
            df: OHLCV DataFrame.

        Returns:
            pd.Series of SAR values.
        """
        if _HAS_TALIB:
            arr = talib.SAR(
                self._ensure_float(df["high"]),
                self._ensure_float(df["low"]),
                acceleration=0.02,
                maximum=0.20,
            )
            return pd.Series(arr, index=df.index, name="parabolic_sar")

        high = df["high"].values.astype(np.float64)
        low = df["low"].values.astype(np.float64)
        n = len(df)
        sar = np.empty(n, dtype=np.float64)
        sar[:] = np.nan

        if n < 2:
            return pd.Series(sar, index=df.index, name="parabolic_sar")

        af_start = 0.02
        af_inc = 0.02
        af_max = 0.20

        # Initialise: assume first trend is bullish
        is_bull = True
        af = af_start
        ep = high[0]
        sar[0] = low[0]

        for i in range(1, n):
            prev_sar = sar[i - 1]
            sar[i] = prev_sar + af * (ep - prev_sar)

            if is_bull:
                # Clamp SAR below previous two lows
                if i >= 2:
                    sar[i] = min(sar[i], low[i - 1], low[i - 2])
                else:
                    sar[i] = min(sar[i], low[i - 1])

                if low[i] < sar[i]:
                    # Reverse to bearish
                    is_bull = False
                    sar[i] = ep
                    ep = low[i]
                    af = af_start
                else:
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + af_inc, af_max)
            else:
                # Clamp SAR above previous two highs
                if i >= 2:
                    sar[i] = max(sar[i], high[i - 1], high[i - 2])
                else:
                    sar[i] = max(sar[i], high[i - 1])

                if high[i] > sar[i]:
                    # Reverse to bullish
                    is_bull = True
                    sar[i] = ep
                    ep = high[i]
                    af = af_start
                else:
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + af_inc, af_max)

        return pd.Series(sar, index=df.index, name="parabolic_sar")

    def compute_linear_regression_channel(
        self, df: pd.DataFrame, period: int = 20
    ) -> pd.DataFrame:
        """Compute rolling linear-regression channel.

        Args:
            df: OHLCV DataFrame.
            period: Rolling window for regression.

        Returns:
            DataFrame with columns ``upper``, ``middle``, ``lower``, ``slope``.
        """
        close = df["close"].values.astype(np.float64)
        n = len(close)
        middle = np.full(n, np.nan)
        slope_arr = np.full(n, np.nan)
        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)

        x = np.arange(period, dtype=np.float64)
        x_mean = x.mean()
        ss_xx = ((x - x_mean) ** 2).sum()

        for i in range(period - 1, n):
            y = close[i - period + 1: i + 1]
            y_mean = y.mean()
            ss_xy = ((x - x_mean) * (y - y_mean)).sum()
            b = ss_xy / ss_xx
            a = y_mean - b * x_mean
            fitted = a + b * x
            residuals = y - fitted
            std_dev = residuals.std()
            middle[i] = fitted[-1]
            slope_arr[i] = b
            upper[i] = fitted[-1] + 2.0 * std_dev
            lower[i] = fitted[-1] - 2.0 * std_dev

        return pd.DataFrame(
            {
                "upper": upper,
                "middle": middle,
                "lower": lower,
                "slope": slope_arr,
            },
            index=df.index,
        )

    def compute_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Compute Average Directional Index with +DI / -DI.

        Args:
            df: OHLCV DataFrame.
            period: Smoothing period.

        Returns:
            DataFrame with columns ``adx``, ``plus_di``, ``minus_di``.
        """
        if _HAS_TALIB:
            h = self._ensure_float(df["high"])
            l_ = self._ensure_float(df["low"])
            c = self._ensure_float(df["close"])
            adx = talib.ADX(h, l_, c, timeperiod=period)
            plus_di = talib.PLUS_DI(h, l_, c, timeperiod=period)
            minus_di = talib.MINUS_DI(h, l_, c, timeperiod=period)
            return pd.DataFrame(
                {"adx": adx, "plus_di": plus_di, "minus_di": minus_di},
                index=df.index,
            )

        high = df["high"]
        low = df["low"]
        close = df["close"]

        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr = self._true_range(high, low, close)

        atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        plus_dm_smooth = plus_dm.ewm(
            alpha=1.0 / period, min_periods=period, adjust=False
        ).mean()
        minus_dm_smooth = minus_dm.ewm(
            alpha=1.0 / period, min_periods=period, adjust=False
        ).mean()

        plus_di = 100.0 * plus_dm_smooth / atr
        minus_di = 100.0 * minus_dm_smooth / atr

        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        return pd.DataFrame(
            {"adx": adx, "plus_di": plus_di, "minus_di": minus_di},
            index=df.index,
        )

    # ===================================================================== #
    #  B.  MOMENTUM INDICATORS                                               #
    # ===================================================================== #

    def compute_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute Relative Strength Index.

        Args:
            df: OHLCV DataFrame.
            period: RSI look-back period.

        Returns:
            pd.Series with RSI values in [0, 100].
        """
        if _HAS_TALIB:
            arr = talib.RSI(self._ensure_float(df["close"]), timeperiod=period)
            return pd.Series(arr, index=df.index, name=f"rsi_{period}")

        delta = df["close"].diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)

        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - 100.0 / (1.0 + rs)
        return rsi.rename(f"rsi_{period}")

    def detect_rsi_divergence(
        self, df: pd.DataFrame, rsi: pd.Series, lookback: int = 20
    ) -> pd.DataFrame:
        """Detect bullish and bearish RSI divergence.

        Bullish divergence: price makes lower low while RSI makes higher low.
        Bearish divergence: price makes higher high while RSI makes lower high.

        Args:
            df: OHLCV DataFrame.
            rsi: Pre-computed RSI Series.
            lookback: Window to compare swings.

        Returns:
            DataFrame with boolean columns ``bullish_div`` and ``bearish_div``.
        """
        close = df["close"].values.astype(np.float64)
        rsi_vals = rsi.values.astype(np.float64)
        n = len(close)

        bullish = np.zeros(n, dtype=bool)
        bearish = np.zeros(n, dtype=bool)

        for i in range(lookback, n):
            window_close = close[i - lookback: i + 1]
            window_rsi = rsi_vals[i - lookback: i + 1]

            # Find local minima / maxima in the window
            price_min_idx = np.nanargmin(window_close)
            price_max_idx = np.nanargmax(window_close)

            # Bullish: current price near low of window AND lower than
            # previous low but RSI is higher than at previous low
            current_price = close[i]
            current_rsi = rsi_vals[i]
            window_price_low = window_close[price_min_idx]
            window_rsi_at_low = window_rsi[price_min_idx]

            if (
                price_min_idx != lookback
                and current_price <= window_price_low * 1.005
                and current_rsi > window_rsi_at_low
            ):
                bullish[i] = True

            # Bearish: current price near high of window AND higher than
            # previous high but RSI is lower than at previous high
            window_price_high = window_close[price_max_idx]
            window_rsi_at_high = window_rsi[price_max_idx]

            if (
                price_max_idx != lookback
                and current_price >= window_price_high * 0.995
                and current_rsi < window_rsi_at_high
            ):
                bearish[i] = True

        return pd.DataFrame(
            {"bullish_div": bullish, "bearish_div": bearish}, index=df.index
        )

    def compute_macd(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        """Compute MACD, Signal line, Histogram, and Histogram slope.

        Args:
            df: OHLCV DataFrame.
            fast: Fast EMA period.
            slow: Slow EMA period.
            signal: Signal EMA period.

        Returns:
            DataFrame with columns ``macd_line``, ``signal_line``,
            ``histogram``, ``histogram_slope``.
        """
        if _HAS_TALIB:
            c = self._ensure_float(df["close"])
            macd_line, signal_line, histogram = talib.MACD(
                c, fastperiod=fast, slowperiod=slow, signalperiod=signal
            )
            macd_line = pd.Series(macd_line, index=df.index)
            signal_line = pd.Series(signal_line, index=df.index)
            histogram = pd.Series(histogram, index=df.index)
        else:
            ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
            ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            histogram = macd_line - signal_line

        histogram_slope = histogram.diff()

        return pd.DataFrame(
            {
                "macd_line": macd_line,
                "signal_line": signal_line,
                "histogram": histogram,
                "histogram_slope": histogram_slope,
            },
            index=df.index,
        )

    def compute_stochastic(
        self, df: pd.DataFrame, k: int = 14, d: int = 3, smooth: int = 3
    ) -> pd.DataFrame:
        """Compute Stochastic Oscillator (%K and %D).

        Args:
            df: OHLCV DataFrame.
            k: %K look-back period.
            d: %D smoothing period.
            smooth: %K smoothing period.

        Returns:
            DataFrame with columns ``k`` and ``d``.
        """
        if _HAS_TALIB:
            slowk, slowd = talib.STOCH(
                self._ensure_float(df["high"]),
                self._ensure_float(df["low"]),
                self._ensure_float(df["close"]),
                fastk_period=k,
                slowk_period=smooth,
                slowk_matype=0,
                slowd_period=d,
                slowd_matype=0,
            )
            return pd.DataFrame(
                {"k": slowk, "d": slowd}, index=df.index
            )

        lowest_low = df["low"].rolling(window=k).min()
        highest_high = df["high"].rolling(window=k).max()

        fast_k = 100.0 * (df["close"] - lowest_low) / (
            (highest_high - lowest_low).replace(0, np.nan)
        )
        slow_k = fast_k.rolling(window=smooth).mean()
        slow_d = slow_k.rolling(window=d).mean()

        return pd.DataFrame({"k": slow_k, "d": slow_d}, index=df.index)

    def compute_stochastic_rsi(
        self,
        df: pd.DataFrame,
        period: int = 14,
        k: int = 3,
        d: int = 3,
    ) -> pd.DataFrame:
        """Compute Stochastic RSI.

        StochRSI = (RSI - min(RSI)) / (max(RSI) - min(RSI)) over *period*.

        Args:
            df: OHLCV DataFrame.
            period: RSI and stochastic period.
            k: %K smoothing.
            d: %D smoothing.

        Returns:
            DataFrame with columns ``k`` and ``d`` in [0, 100].
        """
        rsi = self.compute_rsi(df, period)
        rsi_min = rsi.rolling(window=period).min()
        rsi_max = rsi.rolling(window=period).max()
        stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)

        k_line = stoch_rsi.rolling(window=k).mean() * 100.0
        d_line = k_line.rolling(window=d).mean()

        return pd.DataFrame({"k": k_line, "d": d_line}, index=df.index)

    def compute_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute Williams %R.

        Args:
            df: OHLCV DataFrame.
            period: Look-back period.

        Returns:
            pd.Series with values in [-100, 0].
        """
        if _HAS_TALIB:
            arr = talib.WILLR(
                self._ensure_float(df["high"]),
                self._ensure_float(df["low"]),
                self._ensure_float(df["close"]),
                timeperiod=period,
            )
            return pd.Series(arr, index=df.index, name="williams_r")

        highest_high = df["high"].rolling(window=period).max()
        lowest_low = df["low"].rolling(window=period).min()
        wr = -100.0 * (highest_high - df["close"]) / (
            (highest_high - lowest_low).replace(0, np.nan)
        )
        return wr.rename("williams_r")

    def compute_cci(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Compute Commodity Channel Index.

        Args:
            df: OHLCV DataFrame.
            period: Look-back period.

        Returns:
            pd.Series with CCI values.
        """
        if _HAS_TALIB:
            arr = talib.CCI(
                self._ensure_float(df["high"]),
                self._ensure_float(df["low"]),
                self._ensure_float(df["close"]),
                timeperiod=period,
            )
            return pd.Series(arr, index=df.index, name="cci")

        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        tp_sma = tp.rolling(window=period).mean()
        tp_mad = tp.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )
        cci = (tp - tp_sma) / (0.015 * tp_mad.replace(0, np.nan))
        return cci.rename("cci")

    def compute_roc(self, df: pd.DataFrame, period: int = 10) -> pd.Series:
        """Compute Rate of Change.

        Args:
            df: OHLCV DataFrame.
            period: Look-back period.

        Returns:
            pd.Series with ROC as percentage.
        """
        if _HAS_TALIB:
            arr = talib.ROC(self._ensure_float(df["close"]), timeperiod=period)
            return pd.Series(arr, index=df.index, name="roc")

        roc = df["close"].pct_change(periods=period) * 100.0
        return roc.rename("roc")

    def compute_tsi(
        self, df: pd.DataFrame, long_period: int = 25, short_period: int = 13
    ) -> pd.Series:
        """Compute True Strength Index.

        TSI = 100 * EMA(EMA(momentum, long), short) /
              EMA(EMA(|momentum|, long), short)

        Args:
            df: OHLCV DataFrame.
            long_period: Long EMA period.
            short_period: Short EMA period.

        Returns:
            pd.Series with TSI values.
        """
        momentum = df["close"].diff()
        smooth1 = momentum.ewm(span=long_period, adjust=False).mean()
        smooth2 = smooth1.ewm(span=short_period, adjust=False).mean()

        abs_smooth1 = momentum.abs().ewm(span=long_period, adjust=False).mean()
        abs_smooth2 = abs_smooth1.ewm(span=short_period, adjust=False).mean()

        tsi = 100.0 * smooth2 / abs_smooth2.replace(0, np.nan)
        return tsi.rename("tsi")

    def compute_awesome_oscillator(self, df: pd.DataFrame) -> pd.Series:
        """Compute Awesome Oscillator (AO).

        AO = SMA(median_price, 5) - SMA(median_price, 34)

        Args:
            df: OHLCV DataFrame.

        Returns:
            pd.Series with AO values.
        """
        median_price = (df["high"] + df["low"]) / 2.0
        ao = median_price.rolling(5).mean() - median_price.rolling(34).mean()
        return ao.rename("awesome_oscillator")

    # ===================================================================== #
    #  C.  VOLATILITY INDICATORS                                             #
    # ===================================================================== #

    def compute_bollinger_bands(
        self, df: pd.DataFrame, period: int = 20, std: float = 2.0
    ) -> pd.DataFrame:
        """Compute Bollinger Bands with bandwidth and %B.

        Args:
            df: OHLCV DataFrame.
            period: SMA look-back period.
            std: Standard-deviation multiplier.

        Returns:
            DataFrame with columns ``upper``, ``middle``, ``lower``,
            ``bandwidth_pct``, ``pct_b``.
        """
        if _HAS_TALIB:
            c = self._ensure_float(df["close"])
            upper, middle, lower = talib.BBANDS(
                c, timeperiod=period, nbdevup=std, nbdevdn=std, matype=0
            )
            upper = pd.Series(upper, index=df.index)
            middle = pd.Series(middle, index=df.index)
            lower = pd.Series(lower, index=df.index)
        else:
            middle = df["close"].rolling(window=period).mean()
            rolling_std = df["close"].rolling(window=period).std()
            upper = middle + std * rolling_std
            lower = middle - std * rolling_std

        bandwidth_pct = ((upper - lower) / middle.replace(0, np.nan)) * 100.0
        pct_b = (df["close"] - lower) / (upper - lower).replace(0, np.nan)

        return pd.DataFrame(
            {
                "upper": upper,
                "middle": middle,
                "lower": lower,
                "bandwidth_pct": bandwidth_pct,
                "pct_b": pct_b,
            },
            index=df.index,
        )

    def compute_keltner_channels(
        self, df: pd.DataFrame, period: int = 20, atr_mult: float = 1.5
    ) -> pd.DataFrame:
        """Compute Keltner Channels.

        Args:
            df: OHLCV DataFrame.
            period: EMA period for middle line.
            atr_mult: ATR multiplier for band width.

        Returns:
            DataFrame with columns ``upper``, ``middle``, ``lower``.
        """
        middle = df["close"].ewm(span=period, adjust=False).mean()
        atr = self.compute_atr(df, period)
        upper = middle + atr_mult * atr
        lower = middle - atr_mult * atr

        return pd.DataFrame(
            {"upper": upper, "middle": middle, "lower": lower},
            index=df.index,
        )

    def compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute Average True Range.

        Args:
            df: OHLCV DataFrame.
            period: Smoothing period.

        Returns:
            pd.Series with ATR values.
        """
        if _HAS_TALIB:
            arr = talib.ATR(
                self._ensure_float(df["high"]),
                self._ensure_float(df["low"]),
                self._ensure_float(df["close"]),
                timeperiod=period,
            )
            return pd.Series(arr, index=df.index, name="atr")

        tr = self._true_range(df["high"], df["low"], df["close"])
        atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        return atr.rename("atr")

    def compute_historical_volatility(
        self, df: pd.DataFrame, period: int = 20
    ) -> pd.Series:
        """Compute annualised historical volatility.

        Uses log returns and assumes 365 trading days (crypto markets).

        Args:
            df: OHLCV DataFrame.
            period: Rolling window.

        Returns:
            pd.Series with annualised volatility.
        """
        log_ret = np.log(df["close"] / df["close"].shift(1))
        hv = log_ret.rolling(window=period).std() * np.sqrt(365)
        return hv.rename("historical_volatility")

    def compute_volatility_percentile(
        self, df: pd.DataFrame, period: int = 20, lookback: int = 90
    ) -> pd.Series:
        """Compute percentile rank of current volatility over a lookback window.

        Args:
            df: OHLCV DataFrame.
            period: Period for the volatility measure (ATR-based).
            lookback: Window over which to rank.

        Returns:
            pd.Series with values in [0, 100].
        """
        atr = self.compute_atr(df, period)
        close = df["close"].replace(0, np.nan)
        norm_atr = atr / close  # normalise by price

        def _pct_rank(arr: np.ndarray) -> float:
            if np.isnan(arr[-1]):
                return np.nan
            valid = arr[~np.isnan(arr)]
            if len(valid) < 2:
                return np.nan
            return float(np.sum(valid < valid[-1]) / (len(valid) - 1) * 100.0)

        vol_pctile = norm_atr.rolling(window=lookback, min_periods=lookback).apply(
            _pct_rank, raw=True
        )
        return vol_pctile.rename("volatility_percentile")

    # ===================================================================== #
    #  D.  VOLUME INDICATORS                                                 #
    # ===================================================================== #

    def compute_obv(self, df: pd.DataFrame) -> pd.Series:
        """Compute On-Balance Volume.

        Args:
            df: OHLCV DataFrame.

        Returns:
            pd.Series with cumulative OBV.
        """
        if _HAS_TALIB:
            arr = talib.OBV(
                self._ensure_float(df["close"]),
                self._ensure_float(df["volume"]),
            )
            return pd.Series(arr, index=df.index, name="obv")

        sign = np.sign(df["close"].diff())
        sign.iloc[0] = 0
        obv = (sign * df["volume"]).cumsum()
        return obv.rename("obv")

    def compute_obv_slope(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Compute slope of On-Balance Volume over a rolling window.

        Uses a simple linear-regression slope on the OBV values.

        Args:
            df: OHLCV DataFrame.
            period: Rolling window.

        Returns:
            pd.Series with OBV slope values.
        """
        obv = self.compute_obv(df)
        x = np.arange(period, dtype=np.float64)
        x_mean = x.mean()
        ss_xx = ((x - x_mean) ** 2).sum()

        def _slope(y: np.ndarray) -> float:
            if np.any(np.isnan(y)):
                return np.nan
            y_mean = y.mean()
            ss_xy = ((x - x_mean) * (y - y_mean)).sum()
            return ss_xy / ss_xx

        slope = obv.rolling(window=period).apply(_slope, raw=True)
        return slope.rename("obv_slope")

    def compute_vwap(
        self, df: pd.DataFrame, anchor: str = "session"
    ) -> pd.Series:
        """Compute Volume-Weighted Average Price.

        Anchors to session (daily), week, or month boundaries.  Falls back
        to cumulative VWAP when ``timestamp`` column is missing.

        Args:
            df: OHLCV DataFrame.
            anchor: One of ``'session'``, ``'week'``, ``'month'``.

        Returns:
            pd.Series with VWAP values.
        """
        typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
        tp_vol = typical_price * df["volume"]

        if "timestamp" not in df.columns:
            cum_tp_vol = tp_vol.cumsum()
            cum_vol = df["volume"].cumsum()
            return (cum_tp_vol / cum_vol.replace(0, np.nan)).rename("vwap")

        ts = pd.to_datetime(df["timestamp"])

        if anchor == "week":
            group_key = ts.dt.isocalendar().week.astype(str) + "_" + ts.dt.isocalendar().year.astype(str)
        elif anchor == "month":
            group_key = ts.dt.to_period("M").astype(str)
        else:
            # session = daily
            group_key = ts.dt.date.astype(str)

        cum_tp_vol = tp_vol.groupby(group_key).cumsum()
        cum_vol = df["volume"].groupby(group_key).cumsum()
        vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
        return vwap.rename("vwap")

    def compute_volume_profile(
        self, df: pd.DataFrame, num_bins: int = 50
    ) -> Dict[str, Any]:
        """Compute Volume Profile (volume at price).

        Args:
            df: OHLCV DataFrame.
            num_bins: Number of price bins.

        Returns:
            Dictionary with keys:

            - ``poc`` (float): Point of Control price.
            - ``vah`` (float): Value Area High.
            - ``val`` (float): Value Area Low.
            - ``profile`` (np.ndarray): Volume per bin.
            - ``bin_edges`` (np.ndarray): Bin edge prices.
        """
        close = df["close"].dropna().values
        volume = df["volume"].dropna().values
        if len(close) == 0 or len(volume) == 0:
            return {"poc": np.nan, "vah": np.nan, "val": np.nan}

        min_price = float(np.nanmin(close))
        max_price = float(np.nanmax(close))
        if min_price == max_price:
            return {"poc": min_price, "vah": min_price, "val": min_price}

        bin_edges = np.linspace(min_price, max_price, num_bins + 1)
        bin_indices = np.digitize(close, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)

        profile = np.zeros(num_bins, dtype=np.float64)
        np.add.at(profile, bin_indices, volume[: len(bin_indices)])

        # Point of Control — bin with highest volume
        poc_idx = int(np.argmax(profile))
        poc = float((bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2.0)

        # Value Area — 70% of total volume centered around POC
        total_vol = profile.sum()
        target_vol = 0.70 * total_vol
        accumulated = profile[poc_idx]
        lo, hi = poc_idx, poc_idx

        while accumulated < target_vol and (lo > 0 or hi < num_bins - 1):
            expand_lo = profile[lo - 1] if lo > 0 else 0.0
            expand_hi = profile[hi + 1] if hi < num_bins - 1 else 0.0
            if expand_lo >= expand_hi and lo > 0:
                lo -= 1
                accumulated += profile[lo]
            elif hi < num_bins - 1:
                hi += 1
                accumulated += profile[hi]
            else:
                lo -= 1
                accumulated += profile[lo]

        val_ = float((bin_edges[lo] + bin_edges[lo + 1]) / 2.0)
        vah = float((bin_edges[hi] + bin_edges[hi + 1]) / 2.0)

        return {
            "poc": poc,
            "vah": vah,
            "val": val_,
            "profile": profile,
            "bin_edges": bin_edges,
        }

    def compute_cmf(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Compute Chaikin Money Flow.

        Args:
            df: OHLCV DataFrame.
            period: Rolling period.

        Returns:
            pd.Series with CMF values in [-1, 1].
        """
        hl_range = (df["high"] - df["low"]).replace(0, np.nan)
        mf_multiplier = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / hl_range
        mf_volume = mf_multiplier * df["volume"]

        cmf = mf_volume.rolling(window=period).sum() / df["volume"].rolling(
            window=period
        ).sum().replace(0, np.nan)
        return cmf.rename("cmf")

    def compute_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute Money Flow Index.

        Args:
            df: OHLCV DataFrame.
            period: Look-back period.

        Returns:
            pd.Series with MFI values in [0, 100].
        """
        if _HAS_TALIB:
            arr = talib.MFI(
                self._ensure_float(df["high"]),
                self._ensure_float(df["low"]),
                self._ensure_float(df["close"]),
                self._ensure_float(df["volume"]),
                timeperiod=period,
            )
            return pd.Series(arr, index=df.index, name="mfi")

        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        raw_mf = tp * df["volume"]
        tp_diff = tp.diff()

        pos_mf = raw_mf.where(tp_diff > 0, 0.0).rolling(window=period).sum()
        neg_mf = raw_mf.where(tp_diff < 0, 0.0).rolling(window=period).sum()

        mfi = 100.0 - 100.0 / (1.0 + pos_mf / neg_mf.replace(0, np.nan))
        return mfi.rename("mfi")

    def compute_volume_zscore(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Compute Z-score of current volume relative to rolling window.

        Args:
            df: OHLCV DataFrame.
            period: Rolling window for mean and std.

        Returns:
            pd.Series with Z-score values.
        """
        vol = df["volume"]
        vol_mean = vol.rolling(window=period).mean()
        vol_std = vol.rolling(window=period).std().replace(0, np.nan)
        zscore = (vol - vol_mean) / vol_std
        return zscore.rename("volume_zscore")

    def compute_trade_flow_imbalance(self, df: pd.DataFrame) -> pd.Series:
        """Compute trade-flow imbalance.

        If ``buy_volume`` and ``sell_volume`` columns exist, uses those
        directly.  Otherwise approximates buy/sell split by the position
        of close within the high-low range.

        Args:
            df: OHLCV DataFrame (optionally with buy_volume, sell_volume).

        Returns:
            pd.Series with values in [-1, 1].  Positive = buy pressure.
        """
        if "buy_volume" in df.columns and "sell_volume" in df.columns:
            total = (df["buy_volume"] + df["sell_volume"]).replace(0, np.nan)
            imbalance = (df["buy_volume"] - df["sell_volume"]) / total
            return imbalance.rename("trade_flow_imbalance")

        # Approximate using close position in bar
        hl_range = (df["high"] - df["low"]).replace(0, np.nan)
        close_position = (df["close"] - df["low"]) / hl_range  # 0..1
        buy_pct = close_position
        sell_pct = 1.0 - close_position
        buy_vol = buy_pct * df["volume"]
        sell_vol = sell_pct * df["volume"]
        total = df["volume"].replace(0, np.nan)
        imbalance = (buy_vol - sell_vol) / total
        return imbalance.rename("trade_flow_imbalance")

    # ===================================================================== #
    #  E.  COMPREHENSIVE — compute_all                                       #
    # ===================================================================== #

    def compute_all(
        self, df: pd.DataFrame, timeframe: str = ""
    ) -> pd.DataFrame:
        """Compute ALL configured indicators and append as new columns.

        Reads periods and parameters from the ``config`` dict passed at
        construction time.  The returned DataFrame is a copy of the input
        with all indicator columns added.

        Args:
            df: OHLCV DataFrame.
            timeframe: Label for logging (e.g. ``'4h'``).

        Returns:
            pd.DataFrame — original columns plus all indicator columns.
        """
        if df.empty:
            logger.warning("compute_all called with empty DataFrame")
            return df.copy()

        out = df.copy()
        cfg = self.config

        # ---- EMA --------------------------------------------------------
        for period in cfg.get("ema_periods", [9, 21, 50, 100, 200]):
            t0 = time.monotonic()
            out[f"ema_{period}"] = self.compute_ema(df, period)
            _log_timing(f"ema_{period}", timeframe, time.monotonic() - t0, len(df))

        # ---- DEMA / TEMA for first EMA period ----------------------------
        first_ema = (cfg.get("ema_periods") or [21])[0]
        t0 = time.monotonic()
        out[f"dema_{first_ema}"] = self.compute_dema(df, first_ema)
        _log_timing(f"dema_{first_ema}", timeframe, time.monotonic() - t0, len(df))

        t0 = time.monotonic()
        out[f"tema_{first_ema}"] = self.compute_tema(df, first_ema)
        _log_timing(f"tema_{first_ema}", timeframe, time.monotonic() - t0, len(df))

        # ---- RSI ---------------------------------------------------------
        for period in cfg.get("rsi_periods", [7, 14, 21]):
            t0 = time.monotonic()
            rsi = self.compute_rsi(df, period)
            out[f"rsi_{period}"] = rsi
            _log_timing(f"rsi_{period}", timeframe, time.monotonic() - t0, len(df))

        # RSI divergence on default period
        default_rsi_period = cfg.get("rsi_periods", [14])[0]
        t0 = time.monotonic()
        rsi_for_div = self.compute_rsi(df, default_rsi_period)
        div = self.detect_rsi_divergence(df, rsi_for_div)
        out["rsi_bullish_div"] = div["bullish_div"]
        out["rsi_bearish_div"] = div["bearish_div"]
        _log_timing("rsi_divergence", timeframe, time.monotonic() - t0, len(df))

        # ---- MACD --------------------------------------------------------
        macd_cfg = cfg.get("macd", {"fast": 12, "slow": 26, "signal": 9})
        t0 = time.monotonic()
        macd_df = self.compute_macd(
            df,
            fast=macd_cfg.get("fast", 12),
            slow=macd_cfg.get("slow", 26),
            signal=macd_cfg.get("signal", 9),
        )
        for col in macd_df.columns:
            out[col] = macd_df[col]
        _log_timing("macd", timeframe, time.monotonic() - t0, len(df))

        # ---- Stochastic --------------------------------------------------
        stoch_cfg = cfg.get("stochastic", {"k": 14, "d": 3, "smooth": 3})
        t0 = time.monotonic()
        stoch = self.compute_stochastic(
            df,
            k=stoch_cfg.get("k", 14),
            d=stoch_cfg.get("d", 3),
            smooth=stoch_cfg.get("smooth", 3),
        )
        out["stoch_k"] = stoch["k"]
        out["stoch_d"] = stoch["d"]
        _log_timing("stochastic", timeframe, time.monotonic() - t0, len(df))

        # ---- Stochastic RSI ----------------------------------------------
        t0 = time.monotonic()
        stoch_rsi = self.compute_stochastic_rsi(df)
        out["stoch_rsi_k"] = stoch_rsi["k"]
        out["stoch_rsi_d"] = stoch_rsi["d"]
        _log_timing("stochastic_rsi", timeframe, time.monotonic() - t0, len(df))

        # ---- Williams %R -------------------------------------------------
        t0 = time.monotonic()
        out["williams_r"] = self.compute_williams_r(df)
        _log_timing("williams_r", timeframe, time.monotonic() - t0, len(df))

        # ---- CCI ---------------------------------------------------------
        t0 = time.monotonic()
        out["cci"] = self.compute_cci(df)
        _log_timing("cci", timeframe, time.monotonic() - t0, len(df))

        # ---- ROC ---------------------------------------------------------
        t0 = time.monotonic()
        out["roc"] = self.compute_roc(df)
        _log_timing("roc", timeframe, time.monotonic() - t0, len(df))

        # ---- TSI ---------------------------------------------------------
        t0 = time.monotonic()
        out["tsi"] = self.compute_tsi(df)
        _log_timing("tsi", timeframe, time.monotonic() - t0, len(df))

        # ---- Awesome Oscillator ------------------------------------------
        t0 = time.monotonic()
        out["awesome_oscillator"] = self.compute_awesome_oscillator(df)
        _log_timing("awesome_oscillator", timeframe, time.monotonic() - t0, len(df))

        # ---- ADX ---------------------------------------------------------
        adx_period = cfg.get("adx_period", 14)
        t0 = time.monotonic()
        adx_df = self.compute_adx(df, adx_period)
        out["adx"] = adx_df["adx"]
        out["plus_di"] = adx_df["plus_di"]
        out["minus_di"] = adx_df["minus_di"]
        _log_timing("adx", timeframe, time.monotonic() - t0, len(df))

        # ---- Supertrend --------------------------------------------------
        for mult in cfg.get("supertrend_multipliers", [2.0, 3.0]):
            t0 = time.monotonic()
            st = self.compute_supertrend(df, multiplier=mult)
            suffix = str(mult).replace(".", "_")
            out[f"supertrend_{suffix}"] = st["supertrend"]
            out[f"supertrend_dir_{suffix}"] = st["direction"]
            _log_timing(
                f"supertrend_{mult}", timeframe, time.monotonic() - t0, len(df)
            )

        # ---- Ichimoku ----------------------------------------------------
        ichi_cfg = cfg.get(
            "ichimoku", {"tenkan": 9, "kijun": 26, "senkou": 52, "chikou": 26}
        )
        t0 = time.monotonic()
        ichi = self.compute_ichimoku(df, **ichi_cfg)
        for col in ichi.columns:
            out[col] = ichi[col]
        _log_timing("ichimoku", timeframe, time.monotonic() - t0, len(df))

        # ---- Parabolic SAR -----------------------------------------------
        t0 = time.monotonic()
        out["parabolic_sar"] = self.compute_parabolic_sar(df)
        _log_timing("parabolic_sar", timeframe, time.monotonic() - t0, len(df))

        # ---- Linear Regression Channel -----------------------------------
        t0 = time.monotonic()
        lrc = self.compute_linear_regression_channel(df)
        out["lr_upper"] = lrc["upper"]
        out["lr_middle"] = lrc["middle"]
        out["lr_lower"] = lrc["lower"]
        out["lr_slope"] = lrc["slope"]
        _log_timing("linear_regression_channel", timeframe, time.monotonic() - t0, len(df))

        # ---- Bollinger Bands ---------------------------------------------
        bb_cfg = cfg.get("bollinger", {"period": 20, "std": 2.0})
        t0 = time.monotonic()
        bb = self.compute_bollinger_bands(
            df, period=bb_cfg.get("period", 20), std=bb_cfg.get("std", 2.0)
        )
        out["bb_upper"] = bb["upper"]
        out["bb_middle"] = bb["middle"]
        out["bb_lower"] = bb["lower"]
        out["bb_bandwidth_pct"] = bb["bandwidth_pct"]
        out["bb_pct_b"] = bb["pct_b"]
        _log_timing("bollinger_bands", timeframe, time.monotonic() - t0, len(df))

        # ---- Keltner Channels --------------------------------------------
        kc_cfg = cfg.get("keltner", {"period": 20, "atr_mult": 1.5})
        t0 = time.monotonic()
        kc = self.compute_keltner_channels(
            df,
            period=kc_cfg.get("period", 20),
            atr_mult=kc_cfg.get("atr_mult", 1.5),
        )
        out["kc_upper"] = kc["upper"]
        out["kc_middle"] = kc["middle"]
        out["kc_lower"] = kc["lower"]
        _log_timing("keltner_channels", timeframe, time.monotonic() - t0, len(df))

        # ---- ATR ---------------------------------------------------------
        atr_period = cfg.get("atr_period", 14)
        t0 = time.monotonic()
        out["atr"] = self.compute_atr(df, atr_period)
        _log_timing("atr", timeframe, time.monotonic() - t0, len(df))

        # ---- Historical Volatility ---------------------------------------
        t0 = time.monotonic()
        out["historical_volatility"] = self.compute_historical_volatility(df)
        _log_timing("historical_volatility", timeframe, time.monotonic() - t0, len(df))

        # ---- Volatility Percentile ---------------------------------------
        t0 = time.monotonic()
        out["volatility_percentile"] = self.compute_volatility_percentile(df)
        _log_timing("volatility_percentile", timeframe, time.monotonic() - t0, len(df))

        # ---- OBV ---------------------------------------------------------
        t0 = time.monotonic()
        out["obv"] = self.compute_obv(df)
        _log_timing("obv", timeframe, time.monotonic() - t0, len(df))

        # ---- OBV Slope ---------------------------------------------------
        t0 = time.monotonic()
        out["obv_slope"] = self.compute_obv_slope(df)
        _log_timing("obv_slope", timeframe, time.monotonic() - t0, len(df))

        # ---- VWAP --------------------------------------------------------
        t0 = time.monotonic()
        out["vwap"] = self.compute_vwap(df)
        _log_timing("vwap", timeframe, time.monotonic() - t0, len(df))

        # ---- CMF ---------------------------------------------------------
        cmf_period = cfg.get("cmf_period", 20)
        t0 = time.monotonic()
        out["cmf"] = self.compute_cmf(df, cmf_period)
        _log_timing("cmf", timeframe, time.monotonic() - t0, len(df))

        # ---- MFI ---------------------------------------------------------
        mfi_period = cfg.get("mfi_period", 14)
        t0 = time.monotonic()
        out["mfi"] = self.compute_mfi(df, mfi_period)
        _log_timing("mfi", timeframe, time.monotonic() - t0, len(df))

        # ---- Volume Z-Score ----------------------------------------------
        vz_period = cfg.get("volume_zscore_period", 20)
        t0 = time.monotonic()
        out["volume_zscore"] = self.compute_volume_zscore(df, vz_period)
        _log_timing("volume_zscore", timeframe, time.monotonic() - t0, len(df))

        # ---- Trade Flow Imbalance ----------------------------------------
        t0 = time.monotonic()
        out["trade_flow_imbalance"] = self.compute_trade_flow_imbalance(df)
        _log_timing("trade_flow_imbalance", timeframe, time.monotonic() - t0, len(df))

        logger.info(
            json.dumps(
                {
                    "event": "compute_all_complete",
                    "timeframe": timeframe,
                    "rows": len(df),
                    "columns_added": len(out.columns) - len(df.columns),
                }
            )
        )

        return out
