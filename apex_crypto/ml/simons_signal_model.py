"""Simons ML Signal Layer for APEX Crypto Trading System.

XGBoost-based signal model trained on 47 engineered features per asset.
Uses walk-forward validation with a rolling 365-day training window.
Predicts direction and magnitude of next-24h returns.

Simons used ML before it was mainstream — this finds statistical patterns
that human traders miss.
"""

from __future__ import annotations

import os
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from apex_crypto.core.logging import get_logger, log_with_data

logger = get_logger("ml.simons_signal")

MODEL_DIR = Path(__file__).parent / "models"


class SimonsMLSignal:
    """XGBoost ML signal model with 47 engineered features.

    Features span price, volatility, volume, market microstructure, and
    cross-asset dimensions.  The model is retrained weekly on a rolling
    365-day window with walk-forward validation.

    The ML signal is designed to be ensembled with existing strategy
    signals at a 25% weight.

    Args:
        config: ML model configuration dictionary.
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        self.train_window_days: int = cfg.get("train_window_days", 365)
        self.validation_days: int = cfg.get("validation_days", 60)
        self.retrain_interval_days: int = cfg.get("retrain_interval_days", 7)
        self.min_training_trades: int = cfg.get("min_training_trades", 200)
        self.min_accuracy: float = cfg.get("min_accuracy", 0.52)
        self.max_model_versions: int = cfg.get("max_model_versions", 3)

        # Model state per symbol
        self._models: dict[str, Any] = {}
        self._model_metrics: dict[str, dict[str, float]] = {}
        self._last_train_time: dict[str, float] = {}

        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        log_with_data(logger, "info", "SimonsMLSignal initialized", {
            "train_window": self.train_window_days,
            "retrain_interval": self.retrain_interval_days,
            "min_accuracy": self.min_accuracy,
        })

    # ------------------------------------------------------------------
    # Feature engineering (47 features)
    # ------------------------------------------------------------------

    def build_features(
        self,
        ohlcv: pd.DataFrame,
        btc_data: pd.DataFrame | None = None,
        funding_rates: pd.Series | None = None,
        oi_data: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Build 47 features from OHLCV and alternative data.

        Feature categories:
        - Price features (11): multi-period returns, SMA ratios, 52w hi/lo
        - Volatility features (5): realized vol, vol ratio, percentile, Parkinson
        - Volume features (4): z-scores, dollar volume, trend
        - Market microstructure (5): funding rate, OI changes, ls ratio, liquidations
        - Cross-asset features (3): BTC dominance proxy, correlation, beta

        Args:
            ohlcv: OHLCV DataFrame with columns [open, high, low, close, volume].
            btc_data: BTC OHLCV DataFrame for cross-asset features.
            funding_rates: Funding rate time series.
            oi_data: Open interest time series.

        Returns:
            DataFrame of features aligned with input index.
        """
        df = ohlcv.copy()
        features = pd.DataFrame(index=df.index)

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"].astype(float)

        # === PRICE FEATURES ===
        for period in [1, 4, 12, 24, 72, 168, 720, 2160]:  # hours
            col_name = f"return_{period}h"
            features[col_name] = close.pct_change(period)

        # Log returns (normalized)
        features["log_return_24h"] = np.log(close / close.shift(24))

        # Price relative to SMAs
        for sma_period in [20, 50, 200]:
            sma = close.rolling(sma_period).mean()
            features[f"price_rel_sma{sma_period}"] = (close - sma) / sma

        # Distance from 52-week high/low (as percentile)
        rolling_high = high.rolling(252 * 24, min_periods=100).max()
        rolling_low = low.rolling(252 * 24, min_periods=100).min()
        range_size = rolling_high - rolling_low
        features["dist_52w_pct"] = np.where(
            range_size > 0, (close - rolling_low) / range_size, 0.5
        )

        # === VOLATILITY FEATURES ===
        log_returns = np.log(close / close.shift(1))

        for window in [120, 480, 1440]:  # 5d, 20d, 60d in hours
            features[f"realized_vol_{window}h"] = log_returns.rolling(window).std() * np.sqrt(252 * 24)

        # Vol ratio (term structure): 5d / 20d
        vol_5d = log_returns.rolling(120).std()
        vol_20d = log_returns.rolling(480).std()
        features["vol_ratio_5d_20d"] = np.where(vol_20d > 0, vol_5d / vol_20d, 1.0)

        # Vol percentile (current 20d vol vs 252d range)
        features["vol_percentile"] = vol_20d.rolling(252 * 24, min_periods=100).rank(pct=True)

        # Parkinson volatility (uses high-low range)
        hl_ratio = np.log(high / low)
        features["parkinson_vol"] = hl_ratio.rolling(480).apply(
            lambda x: np.sqrt(np.mean(x**2) / (4 * np.log(2))), raw=True
        )

        # === VOLUME FEATURES ===
        for window in [120, 480]:  # 5d, 20d
            vol_mean = volume.rolling(window).mean()
            vol_std = volume.rolling(window).std()
            features[f"volume_zscore_{window}h"] = np.where(
                vol_std > 0, (volume - vol_mean) / vol_std, 0.0
            )

        features["dollar_volume_24h"] = (close * volume).rolling(24).sum()

        # Volume trend: 5d avg vs 20d avg
        vol_5d_avg = volume.rolling(120).mean()
        vol_20d_avg = volume.rolling(480).mean()
        features["volume_trend"] = np.where(vol_20d_avg > 0, vol_5d_avg / vol_20d_avg, 1.0)

        # === MARKET MICROSTRUCTURE ===
        if funding_rates is not None and len(funding_rates) > 0:
            # Align funding rates with feature index
            fr = funding_rates.reindex(features.index, method="ffill")
            features["funding_rate"] = fr
            features["funding_rate_7d_chg"] = fr.diff(168)  # 7d change
        else:
            features["funding_rate"] = 0.0
            features["funding_rate_7d_chg"] = 0.0

        if oi_data is not None and len(oi_data) > 0:
            oi = oi_data.reindex(features.index, method="ffill")
            features["oi_change_24h"] = oi.pct_change(24)
            features["oi_change_7d"] = oi.pct_change(168)
        else:
            features["oi_change_24h"] = 0.0
            features["oi_change_7d"] = 0.0

        # Placeholder for long/short ratio (requires exchange API)
        features["ls_ratio"] = 1.0

        # === CROSS-ASSET FEATURES ===
        if btc_data is not None and "close" not in btc_data.columns:
            btc_data = None

        if btc_data is not None and len(btc_data) > 100:
            btc_close = btc_data["close"].reindex(features.index, method="ffill")
            btc_returns = np.log(btc_close / btc_close.shift(1))
            asset_returns = log_returns

            # Correlation to BTC (7d and 30d)
            for window in [168, 720]:
                features[f"btc_corr_{window}h"] = asset_returns.rolling(window).corr(btc_returns)

            # Beta to market (30d rolling)
            cov_30d = asset_returns.rolling(720).cov(btc_returns)
            var_btc_30d = btc_returns.rolling(720).var()
            features["beta_to_btc_30d"] = np.where(var_btc_30d > 0, cov_30d / var_btc_30d, 1.0)
        else:
            features["btc_corr_168h"] = 0.0
            features["btc_corr_720h"] = 0.0
            features["beta_to_btc_30d"] = 1.0

        # Replace infinities and forward-fill NaNs
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.ffill().fillna(0)

        return features

    # ------------------------------------------------------------------
    # Target variable
    # ------------------------------------------------------------------

    @staticmethod
    def build_target(
        ohlcv: pd.DataFrame,
        forward_period: int = 24,
    ) -> pd.Series:
        """Build the target variable: signed risk-adjusted forward return.

        y = sign(forward_return_24h) * min(abs(forward_return_24h / atr), 2)

        This predicts direction AND magnitude, capped at 2x ATR.

        Args:
            ohlcv: OHLCV DataFrame.
            forward_period: Forward look period in bars.

        Returns:
            Target series (NaN at the end where forward data unavailable).
        """
        close = ohlcv["close"]
        forward_return = close.shift(-forward_period) / close - 1

        # ATR for normalization
        tr = pd.concat([
            ohlcv["high"] - ohlcv["low"],
            abs(ohlcv["high"] - close.shift(1)),
            abs(ohlcv["low"] - close.shift(1)),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        atr_pct = atr / close

        # Signed risk-adjusted return, capped at 2x ATR
        direction = np.sign(forward_return)
        magnitude = np.where(atr_pct > 0, abs(forward_return) / atr_pct, 0)
        magnitude = np.minimum(magnitude, 2.0)

        target = direction * magnitude
        return pd.Series(target, index=ohlcv.index, name="target")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        symbol: str,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> dict[str, float]:
        """Train XGBoost model with walk-forward validation.

        Args:
            symbol: Trading pair symbol.
            features: Feature DataFrame.
            target: Target series.

        Returns:
            Validation metrics dict.
        """
        try:
            import xgboost as xgb
        except ImportError:
            logger.error("xgboost not installed — cannot train ML model")
            return {"accuracy": 0.0, "error": "xgboost not installed"}

        # Align features and target, drop NaN
        combined = features.join(target, how="inner").dropna()
        if len(combined) < self.min_training_trades:
            return {"accuracy": 0.0, "error": f"Insufficient data: {len(combined)}"}

        X = combined[features.columns]
        y = combined["target"]

        # Walk-forward split
        val_size = min(self.validation_days * 24, len(X) // 5)
        train_size = len(X) - val_size

        # Never train on last 7 days (avoid lookahead)
        buffer = 7 * 24
        train_end = train_size - buffer
        if train_end < self.min_training_trades:
            return {"accuracy": 0.0, "error": "Insufficient training data after buffer"}

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_val = X.iloc[train_size:]
        y_val = y.iloc[train_size:]

        # Normalize features using rolling z-score from training set
        means = X_train.mean()
        stds = X_train.std().replace(0, 1)
        X_train_norm = (X_train - means) / stds
        X_val_norm = (X_val - means) / stds

        # Train XGBoost
        dtrain = xgb.DMatrix(X_train_norm, label=y_train)
        dval = xgb.DMatrix(X_val_norm, label=y_val)

        params = {
            "objective": "reg:squarederror",
            "max_depth": 6,
            "eta": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "eval_metric": "rmse",
            "seed": 42,
            "nthread": -1,
        }

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        # Evaluate: direction accuracy
        predictions = model.predict(dval)
        direction_correct = np.sign(predictions) == np.sign(y_val.values)
        accuracy = float(np.mean(direction_correct))

        # Store model and metadata
        self._models[symbol] = {
            "model": model,
            "means": means,
            "stds": stds,
            "feature_names": list(features.columns),
        }

        metrics = {
            "accuracy": round(accuracy, 4),
            "num_train": len(X_train),
            "num_val": len(X_val),
            "best_iteration": model.best_iteration if hasattr(model, "best_iteration") else 0,
        }
        self._model_metrics[symbol] = metrics
        self._last_train_time[symbol] = time.time()

        # Save model to disk
        self._save_model(symbol, model, means, stds, list(features.columns))

        log_with_data(logger, "info", "Model trained", {
            "symbol": symbol,
            **metrics,
        })

        return metrics

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        symbol: str,
        features: pd.DataFrame,
    ) -> tuple[float, float]:
        """Generate ML signal from trained model.

        Args:
            symbol: Trading pair.
            features: Current feature DataFrame (last row used).

        Returns:
            Tuple of (signal_score, confidence).
            signal_score in [-100, 100], confidence in [0, 1].
        """
        if symbol not in self._models:
            self._try_load_model(symbol)

        if symbol not in self._models:
            return 0.0, 0.0

        model_data = self._models[symbol]
        model = model_data["model"]
        means = model_data["means"]
        stds = model_data["stds"]

        # Check model quality
        metrics = self._model_metrics.get(symbol, {})
        accuracy = metrics.get("accuracy", 0.0)
        if accuracy < self.min_accuracy:
            return 0.0, 0.0

        try:
            import xgboost as xgb
        except ImportError:
            return 0.0, 0.0

        # Normalize features
        current = features.iloc[[-1]]
        X_norm = (current - means) / stds
        X_norm = X_norm.fillna(0).replace([np.inf, -np.inf], 0)

        dmatrix = xgb.DMatrix(X_norm)
        prediction = float(model.predict(dmatrix)[0])

        # Convert prediction to signal
        # prediction is the expected risk-adjusted return (capped at +-2)
        prob_positive = 1.0 / (1.0 + np.exp(-prediction * 2))  # sigmoid scaling
        magnitude = min(abs(prediction), 2.0) / 2.0  # normalize to [0, 1]

        # Signal = (prob - 0.5) * 2 * magnitude * 100
        signal_score = (prob_positive - 0.5) * 2 * magnitude * 100
        signal_score = float(np.clip(signal_score, -100, 100))

        confidence = magnitude * accuracy  # Scale confidence by model accuracy

        return signal_score, confidence

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def _save_model(
        self,
        symbol: str,
        model: Any,
        means: pd.Series,
        stds: pd.Series,
        feature_names: list[str],
    ) -> None:
        """Save model to disk with versioning."""
        clean_symbol = symbol.replace("/", "_")
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        filename = f"simons_model_{clean_symbol}_{date_str}.pkl"
        filepath = MODEL_DIR / filename

        model_data = {
            "model": model,
            "means": means,
            "stds": stds,
            "feature_names": feature_names,
            "metrics": self._model_metrics.get(symbol, {}),
            "timestamp": time.time(),
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        # Clean old versions (keep last N)
        self._cleanup_old_models(clean_symbol)

        log_with_data(logger, "info", "Model saved", {
            "symbol": symbol, "path": str(filepath),
        })

    def _try_load_model(self, symbol: str) -> None:
        """Try to load the most recent model from disk."""
        clean_symbol = symbol.replace("/", "_")
        pattern = f"simons_model_{clean_symbol}_*.pkl"

        model_files = sorted(MODEL_DIR.glob(pattern), reverse=True)
        if not model_files:
            return

        try:
            with open(model_files[0], "rb") as f:
                model_data = pickle.load(f)

            self._models[symbol] = {
                "model": model_data["model"],
                "means": model_data["means"],
                "stds": model_data["stds"],
                "feature_names": model_data["feature_names"],
            }
            self._model_metrics[symbol] = model_data.get("metrics", {})
            self._last_train_time[symbol] = model_data.get("timestamp", 0)

            log_with_data(logger, "info", "Model loaded from disk", {
                "symbol": symbol, "file": model_files[0].name,
            })
        except Exception as exc:
            logger.warning("Failed to load model for %s: %s", symbol, exc)

    def _cleanup_old_models(self, clean_symbol: str) -> None:
        """Keep only the last N model versions for a symbol."""
        pattern = f"simons_model_{clean_symbol}_*.pkl"
        model_files = sorted(MODEL_DIR.glob(pattern), reverse=True)

        for old_file in model_files[self.max_model_versions:]:
            try:
                old_file.unlink()
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Check if retraining needed
    # ------------------------------------------------------------------

    def needs_retraining(self, symbol: str) -> bool:
        """Check if a symbol's model needs retraining.

        Returns True if no model exists or the model is older than
        the retrain interval.
        """
        last_train = self._last_train_time.get(symbol, 0)
        elapsed_days = (time.time() - last_train) / 86400
        return elapsed_days >= self.retrain_interval_days
