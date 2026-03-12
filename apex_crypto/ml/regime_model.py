"""XGBoost regime classification model for the APEX Crypto Trading System.

Provides training, persistence, inference, and scheduled retraining for a
multiclass XGBoost classifier that predicts market regimes.  Training labels
are generated from historical data using the same rule-based logic exposed
by :class:`~apex_crypto.core.analysis.regime.RegimeClassifier`.

Regime labels (6 classes):
    STRONG_BULL, WEAK_BULL, RANGING, WEAK_BEAR, STRONG_BEAR, CHAOS

Typical usage::

    from apex_crypto.ml.regime_model import RegimeModelTrainer

    trainer = RegimeModelTrainer(config)
    X, y = trainer.prepare_training_data(ohlcv_data)
    metrics = trainer.train(X, y)
    trainer.save_model("/models/regime_xgb.joblib")
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from apex_crypto.core.analysis.regime import (
    FEATURE_NAMES,
    REGIMES,
    RegimeClassifier,
)

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_DEFAULT_MODEL_DIR: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models"
)

_ROLLING_WINDOW_MONTHS: int = 24


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
        "component": "regime_model_trainer",
        "msg": msg,
        **kwargs,
    }
    getattr(logger, level)(json.dumps(payload, default=str))


class RegimeModelTrainer:
    """Trains and manages an XGBoost multiclass regime classifier.

    The trainer generates labels from historical data using the
    rule-based classifier, trains a 6-class XGBoost model with
    stratified splitting, and provides save/load/predict utilities.

    Args:
        config: System configuration dictionary.  Relevant keys under
            ``ml``: ``regime_model_path``, ``volatility_50th_pct``,
            ``volatility_90th_pct``.
    """

    def __init__(self, config: dict) -> None:
        self._config: dict = config
        self._model: Optional[XGBClassifier] = None
        self._label_encoder: LabelEncoder = LabelEncoder()
        self._label_encoder.fit(REGIMES)
        self._feature_names: list[str] = list(FEATURE_NAMES)

        # Rule-based classifier used to generate training labels.
        self._rule_classifier: RegimeClassifier = RegimeClassifier(
            config, model_path="__skip__"
        )

        _json_log("info", "RegimeModelTrainer initialised")

    # ── Data preparation ─────────────────────────────────────────────────

    def prepare_training_data(
        self,
        ohlcv_data: dict[str, pd.DataFrame],
        alt_data_history: Optional[pd.DataFrame] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build feature matrix and label vector from historical OHLCV data.

        For each symbol in *ohlcv_data*, a sliding window is advanced one
        bar at a time.  At each step the 10 regime features are computed
        and the rule-based classifier assigns a label.

        Args:
            ohlcv_data: Mapping of symbol names to OHLCV DataFrames.
                Each DataFrame must have columns ``[open, high, low,
                close, volume]`` and a ``timestamp`` column or a
                datetime index.  Data should cover at least 24 months.
            alt_data_history: Optional DataFrame indexed by timestamp
                with columns ``oi_change_pct``, ``funding_rate``,
                ``fear_greed``, ``btc_correlation``.  When provided the
                values are looked up by the closest preceding timestamp.

        Returns:
            Tuple of ``(X, y)`` where *X* has shape
            ``(n_samples, 10)`` and *y* has shape ``(n_samples,)``
            with integer-encoded regime labels.
        """
        start_time: float = time.monotonic()
        all_features: list[np.ndarray] = []
        all_labels: list[int] = []

        # Minimum number of bars needed to compute indicators reliably.
        min_bars: int = 50

        for symbol, df in ohlcv_data.items():
            df = df.copy()

            # Ensure chronological order.
            if "timestamp" in df.columns:
                df = df.sort_values("timestamp").reset_index(drop=True)
            elif isinstance(df.index, pd.DatetimeIndex):
                df = df.sort_index().reset_index()

            # Apply rolling window filter (most recent 24 months).
            if "timestamp" in df.columns:
                cutoff = pd.Timestamp.now(tz="UTC") - pd.DateOffset(
                    months=_ROLLING_WINDOW_MONTHS
                )
                ts_col = pd.to_datetime(df["timestamp"], utc=True)
                df = df.loc[ts_col >= cutoff].reset_index(drop=True)

            if len(df) < min_bars:
                _json_log(
                    "warning",
                    "Skipping symbol: insufficient data",
                    symbol=symbol,
                    rows=len(df),
                    min_required=min_bars,
                )
                continue

            _json_log(
                "info",
                "Processing symbol for training data",
                symbol=symbol,
                rows=len(df),
            )

            for end_idx in range(min_bars, len(df)):
                window_df: pd.DataFrame = df.iloc[: end_idx + 1]

                # Look up alternative data for the current bar.
                alt_dict: dict = self._lookup_alt_data(
                    window_df, alt_data_history
                )

                try:
                    features: dict = self._rule_classifier.extract_features(
                        window_df, alt_data=alt_dict
                    )
                    label_result: dict = self._rule_classifier.rule_based_classify(
                        features
                    )

                    feature_vector: np.ndarray = np.array(
                        [float(features.get(f, 0.0)) for f in self._feature_names]
                    )
                    label_int: int = int(
                        self._label_encoder.transform([label_result["regime"]])[0]
                    )

                    all_features.append(feature_vector)
                    all_labels.append(label_int)
                except Exception as exc:
                    _json_log(
                        "warning",
                        "Feature extraction failed for bar",
                        symbol=symbol,
                        end_idx=end_idx,
                        error=str(exc),
                    )
                    continue

        if not all_features:
            _json_log("error", "No training samples generated")
            return np.empty((0, len(self._feature_names))), np.empty(0, dtype=int)

        X: np.ndarray = np.vstack(all_features)
        y: np.ndarray = np.array(all_labels, dtype=int)

        elapsed: float = round(time.monotonic() - start_time, 2)

        # Log class distribution.
        unique, counts = np.unique(y, return_counts=True)
        distribution: dict[str, int] = {}
        for cls_idx, cnt in zip(unique, counts):
            label_name: str = self._label_encoder.inverse_transform([cls_idx])[0]
            distribution[label_name] = int(cnt)

        _json_log(
            "info",
            "Training data prepared",
            samples=int(X.shape[0]),
            features=int(X.shape[1]),
            class_distribution=distribution,
            elapsed_seconds=elapsed,
        )

        return X, y

    # ── Training ─────────────────────────────────────────────────────────

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
    ) -> dict:
        """Train a multiclass XGBoost classifier on the provided data.

        Uses stratified train/test splitting to maintain class balance
        in both partitions.  The trained model is stored on ``self`` and
        can be persisted via :meth:`save_model`.

        Args:
            X: Feature matrix of shape ``(n_samples, 10)``.
            y: Integer-encoded label vector of shape ``(n_samples,)``.
            test_size: Fraction of data reserved for evaluation.

        Returns:
            Dictionary of training metrics:

            - **accuracy** (float)
            - **f1_macro** (float)
            - **classification_report** (str)
            - **confusion_matrix** (list[list[int]])
        """
        start_time: float = time.monotonic()

        if X.shape[0] == 0:
            _json_log("error", "Cannot train on empty dataset")
            return {
                "accuracy": 0.0,
                "f1_macro": 0.0,
                "classification_report": "",
                "confusion_matrix": [],
            }

        _json_log(
            "info",
            "Starting model training",
            samples=int(X.shape[0]),
            test_size=test_size,
        )

        # Stratified split.
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=y,
            random_state=42,
        )

        # Determine the number of classes actually present.
        n_classes: int = len(np.unique(y))

        # XGBoost hyperparameters.
        self._model = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

        # Train with early stopping on validation set.
        self._model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # Evaluate.
        y_pred: np.ndarray = self._model.predict(X_test)
        acc: float = float(accuracy_score(y_test, y_pred))
        f1: float = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

        # Build human-readable labels for the report.
        present_classes: list[int] = sorted(np.unique(np.concatenate([y_test, y_pred])).tolist())
        target_names: list[str] = [
            self._label_encoder.inverse_transform([c])[0] for c in present_classes
        ]

        cls_report: str = classification_report(
            y_test,
            y_pred,
            labels=present_classes,
            target_names=target_names,
            zero_division=0,
        )
        conf_matrix: list[list[int]] = confusion_matrix(
            y_test, y_pred, labels=present_classes
        ).tolist()

        elapsed: float = round(time.monotonic() - start_time, 2)

        metrics: dict = {
            "accuracy": round(acc, 4),
            "f1_macro": round(f1, 4),
            "classification_report": cls_report,
            "confusion_matrix": conf_matrix,
        }

        _json_log(
            "info",
            "Model training complete",
            accuracy=metrics["accuracy"],
            f1_macro=metrics["f1_macro"],
            train_samples=int(X_train.shape[0]),
            test_samples=int(X_test.shape[0]),
            elapsed_seconds=elapsed,
        )

        return metrics

    # ── Persistence ──────────────────────────────────────────────────────

    def save_model(self, path: str) -> None:
        """Save the trained model, label encoder, and feature names to disk.

        The artifact is a single joblib file containing a dictionary with
        keys ``model``, ``label_encoder``, ``feature_names``, and
        ``saved_at``.

        Args:
            path: Filesystem path for the output file.

        Raises:
            RuntimeError: If no model has been trained yet.
        """
        if self._model is None:
            raise RuntimeError(
                "No trained model to save. Call train() first."
            )

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        bundle: dict[str, Any] = {
            "model": self._model,
            "label_encoder": self._label_encoder,
            "feature_names": self._feature_names,
            "saved_at": datetime.now(tz=timezone.utc).isoformat(),
        }

        joblib.dump(bundle, path)

        _json_log(
            "info",
            "Model saved",
            path=path,
            size_bytes=os.path.getsize(path),
        )

    def load_model(self, path: str) -> None:
        """Load a previously saved model bundle from disk.

        After loading, the trainer can be used for prediction via
        :meth:`predict`.

        Args:
            path: Filesystem path to the joblib artifact.

        Raises:
            FileNotFoundError: If *path* does not exist.
            KeyError: If the bundle is missing required keys.
        """
        if not Path(path).is_file():
            raise FileNotFoundError(f"Model file not found: {path}")

        bundle: dict = joblib.load(path)

        self._model = bundle["model"]
        self._label_encoder = bundle["label_encoder"]
        self._feature_names = bundle.get("feature_names", list(FEATURE_NAMES))

        _json_log(
            "info",
            "Model loaded",
            path=path,
            saved_at=bundle.get("saved_at", "unknown"),
            feature_count=len(self._feature_names),
        )

    # ── Inference ────────────────────────────────────────────────────────

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        """Predict the regime label and confidence for a single sample.

        Args:
            features: 1-D or 2-D array of feature values.  If 1-D it is
                reshaped to ``(1, n_features)`` automatically.

        Returns:
            Tuple of ``(regime_label, confidence_score)`` where
            *confidence_score* is the maximum class probability from
            ``predict_proba``.

        Raises:
            RuntimeError: If no model has been trained or loaded.
        """
        if self._model is None:
            raise RuntimeError(
                "No model available. Call train() or load_model() first."
            )

        if features.ndim == 1:
            features = features.reshape(1, -1)

        probabilities: np.ndarray = self._model.predict_proba(features)[0]
        predicted_class: int = int(np.argmax(probabilities))
        confidence: float = float(probabilities[predicted_class])
        regime_label: str = self._label_encoder.inverse_transform(
            [predicted_class]
        )[0]

        _json_log(
            "debug",
            "Prediction made",
            regime=regime_label,
            confidence=round(confidence, 4),
        )

        return regime_label, round(confidence, 4)

    # ── Scheduled retraining ─────────────────────────────────────────────

    def retrain_monthly(self, storage: Any) -> dict:
        """Pull latest data from storage, retrain, and save a new model.

        Designed to be invoked by a monthly cron/scheduler.  Fetches all
        available OHLCV data from the storage backend, prepares training
        data, trains a fresh model, and persists it to the configured
        model path.

        Args:
            storage: A :class:`~apex_crypto.core.data.storage.StorageManager`
                instance (or compatible object) exposing ``get_ohlcv``
                for each tracked symbol.

        Returns:
            Training metrics dictionary (same schema as :meth:`train`).
        """
        start_time: float = time.monotonic()
        _json_log("info", "Starting monthly retraining")

        # Collect OHLCV data for all configured symbols.
        ml_cfg: dict = self._config.get("ml", {})
        symbols: list[str] = ml_cfg.get(
            "regime_symbols",
            self._config.get("symbols", []),
        )
        timeframe: str = ml_cfg.get("regime_timeframe", "4h")

        ohlcv_data: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                df: pd.DataFrame = storage.get_ohlcv(
                    symbol=symbol, timeframe=timeframe
                )
                if not df.empty:
                    ohlcv_data[symbol] = df
                    _json_log(
                        "info",
                        "Fetched OHLCV for retraining",
                        symbol=symbol,
                        rows=len(df),
                    )
                else:
                    _json_log(
                        "warning",
                        "No OHLCV data available for symbol",
                        symbol=symbol,
                    )
            except Exception as exc:
                _json_log(
                    "error",
                    "Failed to fetch OHLCV for retraining",
                    symbol=symbol,
                    error=str(exc),
                )

        if not ohlcv_data:
            _json_log("error", "No data available for retraining")
            return {
                "accuracy": 0.0,
                "f1_macro": 0.0,
                "classification_report": "",
                "confusion_matrix": [],
            }

        # Prepare data and train.
        X, y = self.prepare_training_data(ohlcv_data)

        if X.shape[0] == 0:
            _json_log("error", "Training data preparation yielded zero samples")
            return {
                "accuracy": 0.0,
                "f1_macro": 0.0,
                "classification_report": "",
                "confusion_matrix": [],
            }

        metrics: dict = self.train(X, y)

        # Save the retrained model.
        model_path: str = ml_cfg.get(
            "regime_model_path",
            os.path.join(_DEFAULT_MODEL_DIR, "regime_xgb.joblib"),
        )
        self.save_model(model_path)

        elapsed: float = round(time.monotonic() - start_time, 2)

        _json_log(
            "info",
            "Monthly retraining complete",
            model_path=model_path,
            accuracy=metrics["accuracy"],
            f1_macro=metrics["f1_macro"],
            total_elapsed_seconds=elapsed,
        )

        return metrics

    # ── Internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _lookup_alt_data(
        window_df: pd.DataFrame,
        alt_data_history: Optional[pd.DataFrame],
    ) -> dict:
        """Look up alternative data values for the last bar in the window.

        Uses the closest preceding timestamp from *alt_data_history*.

        Args:
            window_df: The current OHLCV window being processed.
            alt_data_history: Historical alternative data DataFrame, or
                ``None`` if unavailable.

        Returns:
            Dictionary with keys ``oi_change_pct``, ``funding_rate``,
            ``fear_greed``, ``btc_correlation``.  Missing values default
            to neutral values.
        """
        defaults: dict = {
            "oi_change_pct": 0.0,
            "funding_rate": 0.0,
            "fear_greed": 50.0,
            "btc_correlation": 0.0,
        }

        if alt_data_history is None or alt_data_history.empty:
            return defaults

        try:
            # Determine the timestamp of the last bar.
            if "timestamp" in window_df.columns:
                last_ts = pd.to_datetime(window_df["timestamp"].iloc[-1], utc=True)
            elif isinstance(window_df.index, pd.DatetimeIndex):
                last_ts = window_df.index[-1]
            else:
                return defaults

            # Find the closest preceding row in alt_data_history.
            alt_index = pd.to_datetime(alt_data_history.index, utc=True)
            mask = alt_index <= last_ts
            if not mask.any():
                return defaults

            closest_idx = alt_index[mask][-1]
            row = alt_data_history.loc[closest_idx]

            result: dict = {}
            for key, default_val in defaults.items():
                val = row.get(key, default_val) if hasattr(row, "get") else default_val
                result[key] = float(val) if not pd.isna(val) else default_val

            return result

        except Exception:
            return defaults
