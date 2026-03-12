"""Market regime classification wrapper for the APEX Crypto Trading System.

Provides live-trading regime detection by wrapping a trained XGBoost model
with a rule-based fallback.  The classifier accepts either a pre-computed
feature dictionary or a raw OHLCV DataFrame and returns a regime label
with an associated confidence score.

Regime labels:
    STRONG_BULL, WEAK_BULL, RANGING, WEAK_BEAR, STRONG_BEAR, CHAOS

Typical usage::

    from apex_crypto.core.analysis.regime import RegimeClassifier

    classifier = RegimeClassifier(config)
    result = classifier.classify_from_df(ohlcv_df, alt_data=alt_dict)
    print(result)  # {"regime": "STRONG_BULL", "confidence": 0.87}
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

REGIMES: list[str] = [
    "STRONG_BULL",
    "WEAK_BULL",
    "RANGING",
    "WEAK_BEAR",
    "STRONG_BEAR",
    "CHAOS",
]

FEATURE_NAMES: list[str] = [
    "adx",
    "volatility_pct",
    "ema_slope",
    "bb_width",
    "rsi_14",
    "oi_change_pct",
    "funding_rate",
    "fear_greed",
    "volume_zscore",
    "btc_correlation",
]

# Default percentile thresholds used by the rule-based classifier.
_VOLATILITY_50TH_DEFAULT: float = 2.5
_VOLATILITY_90TH_DEFAULT: float = 6.0


def _json_log(level: str, msg: str, **kwargs: Any) -> None:
    """Emit a structured JSON log line.

    Args:
        level: Log level string (debug, info, warning, error).
        msg: Human-readable message.
        **kwargs: Arbitrary key-value pairs attached to the log entry.
    """
    payload = {
        "ts": datetime.now(tz=timezone.utc).isoformat(),
        "level": level,
        "component": "regime_classifier",
        "msg": msg,
        **kwargs,
    }
    getattr(logger, level)(json.dumps(payload, default=str))


class RegimeClassifier:
    """Live-trading market regime classifier.

    Wraps a trained XGBoost model for regime prediction.  If the model
    file does not exist on disk, classification falls back to a
    deterministic rule-based approach so that the trading system can
    always obtain a regime label.

    Args:
        config: System configuration dictionary.  The key
            ``ml.regime_model_path`` is consulted for the default model
            location when *model_path* is not supplied.
        model_path: Explicit filesystem path to a saved model artifact
            (joblib bundle).  Takes precedence over the config value.
    """

    def __init__(self, config: dict, model_path: Optional[str] = None) -> None:
        self._config: dict = config
        self._model: Any = None
        self._label_encoder: Any = None
        self._feature_names: list[str] = list(FEATURE_NAMES)
        self._use_ml: bool = False

        # Volatility percentile thresholds (may be overridden by config).
        ml_cfg: dict = config.get("ml", {})
        self._vol_50th: float = float(
            ml_cfg.get("volatility_50th_pct", _VOLATILITY_50TH_DEFAULT)
        )
        self._vol_90th: float = float(
            ml_cfg.get("volatility_90th_pct", _VOLATILITY_90TH_DEFAULT)
        )

        # Resolve model path.
        resolved_path: Optional[str] = model_path or ml_cfg.get(
            "regime_model_path"
        )

        if resolved_path and Path(resolved_path).is_file():
            self._load_model(resolved_path)
        else:
            _json_log(
                "warning",
                "No trained regime model found; using rule-based fallback",
                attempted_path=str(resolved_path),
            )

    # ── Model loading ────────────────────────────────────────────────────

    def _load_model(self, path: str) -> None:
        """Load a saved model bundle from disk.

        Args:
            path: Filesystem path to the joblib artifact produced by
                :class:`~apex_crypto.ml.regime_model.RegimeModelTrainer`.
        """
        try:
            import joblib

            bundle: dict = joblib.load(path)
            self._model = bundle["model"]
            self._label_encoder = bundle["label_encoder"]
            self._feature_names = bundle.get("feature_names", list(FEATURE_NAMES))
            self._use_ml = True

            _json_log(
                "info",
                "Loaded trained regime model",
                path=path,
                feature_count=len(self._feature_names),
            )
        except Exception as exc:
            _json_log(
                "error",
                "Failed to load regime model; falling back to rules",
                path=path,
                error=str(exc),
            )
            self._use_ml = False

    # ── Public API ───────────────────────────────────────────────────────

    def classify(self, features: dict) -> dict:
        """Classify the current market regime from pre-computed features.

        Args:
            features: Dictionary mapping each feature name to its numeric
                value.  Expected keys: ``adx``, ``volatility_pct``,
                ``ema_slope``, ``bb_width``, ``rsi_14``,
                ``oi_change_pct``, ``funding_rate``, ``fear_greed``,
                ``volume_zscore``, ``btc_correlation``.

        Returns:
            Dictionary with keys:

            - **regime** (str): One of the six regime labels.
            - **confidence** (float): Confidence score in ``[0, 1]``.
        """
        if self._use_ml:
            return self._ml_classify(features)
        return self.rule_based_classify(features)

    def classify_from_df(
        self,
        df: pd.DataFrame,
        alt_data: Optional[dict] = None,
    ) -> dict:
        """Classify regime from a raw OHLCV DataFrame.

        Extracts all required features from *df* and optional alternative
        data, then delegates to :meth:`classify`.

        Args:
            df: OHLCV DataFrame with columns
                ``[open, high, low, close, volume]`` indexed or sorted
                chronologically.  A minimum of 50 rows is recommended.
            alt_data: Optional dictionary of alternative-data values.
                Recognised keys: ``oi_change_pct``, ``funding_rate``,
                ``fear_greed``, ``btc_correlation``.

        Returns:
            Same structure as :meth:`classify`.
        """
        features: dict = self.extract_features(df, alt_data)
        result: dict = self.classify(features)

        _json_log(
            "info",
            "Regime classified from DataFrame",
            regime=result["regime"],
            confidence=result["confidence"],
            features={k: round(v, 4) if isinstance(v, float) else v for k, v in features.items()},
        )
        return result

    def rule_based_classify(self, features: dict) -> dict:
        """Deterministic rule-based regime classification.

        Used as a fallback when no trained ML model is available and
        also to generate training labels for supervised learning.

        Rules (evaluated top to bottom, first match wins):

        1. ADX > 40 and EMA slope > 0 => STRONG_BULL
        2. ADX > 40 and EMA slope < 0 => STRONG_BEAR
        3. 25 <= ADX <= 40 and EMA slope > 0 => WEAK_BULL
        4. 25 <= ADX <= 40 and EMA slope < 0 => WEAK_BEAR
        5. ADX < 25 and volatility < 50th percentile => RANGING
        6. Volatility > 90th percentile and ADX < 20 => CHAOS
        7. Default => RANGING

        Args:
            features: Feature dictionary (same schema as :meth:`classify`).

        Returns:
            Dictionary with ``regime`` and ``confidence`` keys.
        """
        adx: float = float(features.get("adx", 0.0))
        ema_slope: float = float(features.get("ema_slope", 0.0))
        volatility: float = float(features.get("volatility_pct", 0.0))

        regime: str
        confidence: float

        if adx > 40 and ema_slope > 0:
            regime = "STRONG_BULL"
            confidence = min(0.6 + (adx - 40) / 100, 0.95)
        elif adx > 40 and ema_slope < 0:
            regime = "STRONG_BEAR"
            confidence = min(0.6 + (adx - 40) / 100, 0.95)
        elif 25 <= adx <= 40 and ema_slope > 0:
            regime = "WEAK_BULL"
            confidence = 0.45 + (adx - 25) / 60
        elif 25 <= adx <= 40 and ema_slope < 0:
            regime = "WEAK_BEAR"
            confidence = 0.45 + (adx - 25) / 60
        elif volatility > self._vol_90th and adx < 20:
            regime = "CHAOS"
            confidence = min(0.5 + (volatility - self._vol_90th) / 10, 0.90)
        elif adx < 25 and volatility < self._vol_50th:
            regime = "RANGING"
            confidence = 0.50 + (25 - adx) / 100
        else:
            regime = "RANGING"
            confidence = 0.35

        confidence = round(max(0.0, min(confidence, 1.0)), 4)

        _json_log(
            "debug",
            "Rule-based classification",
            regime=regime,
            confidence=confidence,
            adx=adx,
            ema_slope=round(ema_slope, 6),
            volatility=round(volatility, 4),
        )

        return {"regime": regime, "confidence": confidence}

    def extract_features(
        self,
        df: pd.DataFrame,
        alt_data: Optional[dict] = None,
    ) -> dict:
        """Compute all regime features from OHLCV data and alternative data.

        Args:
            df: OHLCV DataFrame with columns ``[open, high, low, close,
                volume]``.  Must contain at least 20 rows for meaningful
                indicator computation.
            alt_data: Optional dictionary supplying external features.
                Recognised keys: ``oi_change_pct``, ``funding_rate``,
                ``fear_greed``, ``btc_correlation``.

        Returns:
            Dictionary with all 10 feature values ready for classification.
        """
        if alt_data is None:
            alt_data = {}

        close: pd.Series = df["close"].astype(float)
        high: pd.Series = df["high"].astype(float)
        low: pd.Series = df["low"].astype(float)
        volume: pd.Series = df["volume"].astype(float)

        # ── ADX (14-period) ──────────────────────────────────────────────
        adx_value: float = self._compute_adx(high, low, close, period=14)

        # ── Volatility (percentage, 20-period rolling std of returns) ────
        returns: pd.Series = close.pct_change().dropna()
        volatility_pct: float = float(returns.tail(20).std() * 100) if len(returns) >= 20 else 0.0

        # ── EMA slope (20-period EMA, slope of last 5 values) ───────────
        ema_20: pd.Series = close.ewm(span=20, adjust=False).mean()
        ema_slope: float = self._compute_slope(ema_20, window=5)

        # ── Bollinger Band width ────────────────────────────────────────
        sma_20: pd.Series = close.rolling(window=20).mean()
        std_20: pd.Series = close.rolling(window=20).std()
        upper_band: pd.Series = sma_20 + 2 * std_20
        lower_band: pd.Series = sma_20 - 2 * std_20
        last_sma: float = float(sma_20.iloc[-1]) if not sma_20.empty else 1.0
        bb_width: float = (
            float((upper_band.iloc[-1] - lower_band.iloc[-1]) / last_sma * 100)
            if last_sma != 0 and not np.isnan(last_sma)
            else 0.0
        )

        # ── RSI (14-period) ─────────────────────────────────────────────
        rsi_14: float = self._compute_rsi(close, period=14)

        # ── Volume z-score (20-period) ──────────────────────────────────
        vol_mean: float = float(volume.tail(20).mean()) if len(volume) >= 20 else 0.0
        vol_std: float = float(volume.tail(20).std()) if len(volume) >= 20 else 1.0
        vol_std = vol_std if vol_std > 0 else 1.0
        volume_zscore: float = float((volume.iloc[-1] - vol_mean) / vol_std) if not volume.empty else 0.0

        # ── Alternative data features ───────────────────────────────────
        oi_change_pct: float = float(alt_data.get("oi_change_pct", 0.0))
        funding_rate: float = float(alt_data.get("funding_rate", 0.0))
        fear_greed: float = float(alt_data.get("fear_greed", 50.0))
        btc_correlation: float = float(alt_data.get("btc_correlation", 0.0))

        features: dict = {
            "adx": round(adx_value, 4),
            "volatility_pct": round(volatility_pct, 4),
            "ema_slope": round(ema_slope, 6),
            "bb_width": round(bb_width, 4),
            "rsi_14": round(rsi_14, 4),
            "oi_change_pct": round(oi_change_pct, 4),
            "funding_rate": round(funding_rate, 6),
            "fear_greed": round(fear_greed, 2),
            "volume_zscore": round(volume_zscore, 4),
            "btc_correlation": round(btc_correlation, 4),
        }

        _json_log("debug", "Features extracted", features=features)
        return features

    # ── ML prediction ────────────────────────────────────────────────────

    def _ml_classify(self, features: dict) -> dict:
        """Run the trained XGBoost model on the given features.

        Args:
            features: Feature dictionary with all 10 regime features.

        Returns:
            Dictionary with ``regime`` and ``confidence`` keys.
        """
        try:
            feature_vector: np.ndarray = np.array(
                [[float(features.get(f, 0.0)) for f in self._feature_names]]
            )
            probabilities: np.ndarray = self._model.predict_proba(feature_vector)[0]
            predicted_class: int = int(np.argmax(probabilities))
            confidence: float = float(probabilities[predicted_class])
            regime: str = self._label_encoder.inverse_transform([predicted_class])[0]

            _json_log(
                "debug",
                "ML classification",
                regime=regime,
                confidence=round(confidence, 4),
                probabilities={
                    self._label_encoder.inverse_transform([i])[0]: round(float(p), 4)
                    for i, p in enumerate(probabilities)
                },
            )

            return {"regime": regime, "confidence": round(confidence, 4)}

        except Exception as exc:
            _json_log(
                "error",
                "ML classification failed; falling back to rules",
                error=str(exc),
            )
            return self.rule_based_classify(features)

    # ── Technical indicator helpers ──────────────────────────────────────

    @staticmethod
    def _compute_adx(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> float:
        """Compute the Average Directional Index (ADX).

        Args:
            high: Series of high prices.
            low: Series of low prices.
            close: Series of close prices.
            period: Lookback period for smoothing.

        Returns:
            The most recent ADX value, or 0.0 if insufficient data.
        """
        if len(close) < period + 1:
            return 0.0

        plus_dm: pd.Series = high.diff()
        minus_dm: pd.Series = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr1: pd.Series = high - low
        tr2: pd.Series = (high - close.shift(1)).abs()
        tr3: pd.Series = (low - close.shift(1)).abs()
        true_range: pd.Series = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr: pd.Series = true_range.ewm(span=period, adjust=False).mean()
        plus_di: pd.Series = 100 * (
            plus_dm.ewm(span=period, adjust=False).mean() / atr
        )
        minus_di: pd.Series = 100 * (
            minus_dm.ewm(span=period, adjust=False).mean() / atr
        )

        dx: pd.Series = (
            100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
        )
        adx: pd.Series = dx.ewm(span=period, adjust=False).mean()

        last_adx: float = float(adx.iloc[-1])
        return last_adx if not np.isnan(last_adx) else 0.0

    @staticmethod
    def _compute_rsi(close: pd.Series, period: int = 14) -> float:
        """Compute the Relative Strength Index (RSI).

        Args:
            close: Series of close prices.
            period: Lookback period.

        Returns:
            The most recent RSI value, or 50.0 if insufficient data.
        """
        if len(close) < period + 1:
            return 50.0

        delta: pd.Series = close.diff()
        gain: pd.Series = delta.where(delta > 0, 0.0)
        loss: pd.Series = (-delta).where(delta < 0, 0.0)

        avg_gain: pd.Series = gain.ewm(span=period, adjust=False).mean()
        avg_loss: pd.Series = loss.ewm(span=period, adjust=False).mean()

        rs: pd.Series = avg_gain / (avg_loss + 1e-10)
        rsi: pd.Series = 100 - (100 / (1 + rs))

        last_rsi: float = float(rsi.iloc[-1])
        return last_rsi if not np.isnan(last_rsi) else 50.0

    @staticmethod
    def _compute_slope(series: pd.Series, window: int = 5) -> float:
        """Compute the linear regression slope over the last *window* values.

        Args:
            series: Input time series.
            window: Number of trailing values to fit.

        Returns:
            Slope coefficient, or 0.0 if insufficient data.
        """
        if len(series) < window:
            return 0.0

        tail: np.ndarray = series.tail(window).values.astype(float)
        if np.any(np.isnan(tail)):
            return 0.0

        x: np.ndarray = np.arange(window, dtype=float)
        x_mean: float = x.mean()
        y_mean: float = tail.mean()

        numerator: float = float(np.sum((x - x_mean) * (tail - y_mean)))
        denominator: float = float(np.sum((x - x_mean) ** 2))

        if denominator == 0:
            return 0.0

        return numerator / denominator
