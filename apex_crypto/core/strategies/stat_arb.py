"""Pairs Trading / Statistical Arbitrage strategy for APEX Crypto Trading System.

Implements Engle-Granger cointegration-based pairs trading:
- Cointegration testing between crypto pairs
- Rolling OLS hedge ratios
- Z-score based entry/exit on log-price spreads
- Default pairs: BTC/ETH, ETH/SOL, BNB/ETH, SOL/AVAX, LINK/DOT
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

logger = get_logger("strategies.stat_arb")


# Default pairs for cointegration trading
DEFAULT_PAIRS: list[tuple[str, str]] = [
    ("BTC/USDT", "ETH/USDT"),
    ("ETH/USDT", "SOL/USDT"),
    ("BNB/USDT", "ETH/USDT"),
    ("SOL/USDT", "AVAX/USDT"),
    ("LINK/USDT", "DOT/USDT"),
]


class PairsTrading(BaseStrategy):
    """Statistical arbitrage via cointegrated pairs trading.

    Identifies cointegrated pairs and trades the spread when it deviates
    significantly from its mean.  Uses rolling OLS to compute dynamic
    hedge ratios and z-score thresholds for entry/exit.

    Attributes:
        name: Strategy identifier.
        active_regimes: Active in all regimes (market-neutral).
        primary_timeframe: Timeframe for spread computation.
    """

    name: str = "stat_arb"
    active_regimes: list[str] = ["STRONG_BULL", "WEAK_BULL", "RANGING", "WEAK_BEAR", "STRONG_BEAR"]
    primary_timeframe: str = "1h"
    confirmation_timeframe: str = "4h"
    entry_timeframe: str = "15m"

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    _DEFAULT_COINT_LOOKBACK: int = 720      # hours (30 days)
    _DEFAULT_ZSCORE_ENTRY: float = 2.0      # z-score to enter
    _DEFAULT_ZSCORE_EXIT: float = 0.5       # z-score to exit
    _DEFAULT_ZSCORE_STOP: float = 3.5       # z-score to stop out
    _DEFAULT_HEDGE_WINDOW: int = 120        # hours for rolling hedge ratio
    _DEFAULT_SPREAD_WINDOW: int = 480       # hours for spread mean/std
    _DEFAULT_MIN_HALFLIFE: float = 2.0      # minimum OU half-life (hours)
    _DEFAULT_MAX_HALFLIFE: float = 168.0    # maximum OU half-life (hours)
    _DEFAULT_COINT_PVALUE: float = 0.05     # p-value threshold
    _DEFAULT_BASE_SCORE: int = 60

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.pairs: list[tuple[str, str]] = config.get("pairs", DEFAULT_PAIRS)
        self.coint_lookback: int = config.get("coint_lookback", self._DEFAULT_COINT_LOOKBACK)
        self.zscore_entry: float = config.get("zscore_entry", self._DEFAULT_ZSCORE_ENTRY)
        self.zscore_exit: float = config.get("zscore_exit", self._DEFAULT_ZSCORE_EXIT)
        self.zscore_stop: float = config.get("zscore_stop", self._DEFAULT_ZSCORE_STOP)
        self.hedge_window: int = config.get("hedge_window", self._DEFAULT_HEDGE_WINDOW)
        self.spread_window: int = config.get("spread_window", self._DEFAULT_SPREAD_WINDOW)
        self.min_halflife: float = config.get("min_halflife", self._DEFAULT_MIN_HALFLIFE)
        self.max_halflife: float = config.get("max_halflife", self._DEFAULT_MAX_HALFLIFE)
        self.coint_pvalue: float = config.get("coint_pvalue", self._DEFAULT_COINT_PVALUE)
        self.base_score: int = config.get("base_score", self._DEFAULT_BASE_SCORE)

        # State: track active pair spreads and hedge ratios
        self._hedge_ratios: dict[str, float] = {}
        self._spread_stats: dict[str, dict[str, float]] = {}
        self._coint_cache: dict[str, dict[str, Any]] = {}

        log_with_data(logger, "info", "PairsTrading initialized", {
            "num_pairs": len(self.pairs),
            "zscore_entry": self.zscore_entry,
            "zscore_exit": self.zscore_exit,
            "coint_lookback": self.coint_lookback,
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
        """Generate a pairs trading signal for the given symbol.

        Checks all configured pairs where *symbol* is one leg.
        For each valid pair, computes the spread z-score and generates
        a signal when the z-score exceeds entry thresholds.
        """
        if not self.is_active(regime):
            return self._neutral_signal(symbol)

        if self.primary_timeframe not in data:
            return self._neutral_signal(symbol)

        alt = alt_data or {}
        best_signal: Optional[TradeSignal] = None
        best_zscore: float = 0.0

        for leg_a, leg_b in self.pairs:
            # Check if this symbol is part of this pair
            if symbol not in (leg_a, leg_b):
                continue

            # Get data for both legs
            pair_key = f"{leg_a}_{leg_b}"
            data_a = data.get(self.primary_timeframe)
            data_b_key = leg_b if symbol == leg_a else leg_a

            # Try to get partner data from alt_data
            partner_data = alt.get(f"ohlcv_{data_b_key}")
            if partner_data is None:
                partner_data = alt.get(data_b_key)
            if partner_data is None or not isinstance(partner_data, pd.DataFrame):
                continue

            if data_a is None or data_a.empty or partner_data.empty:
                continue

            # Ensure sufficient data
            min_len = max(self.coint_lookback, self.spread_window)
            if len(data_a) < min_len or len(partner_data) < min_len:
                continue

            # Align data
            close_a = data_a["close"].iloc[-min_len:]
            close_b = partner_data["close"].iloc[-min_len:]

            if len(close_a) != len(close_b):
                # Align by taking the shorter length
                common_len = min(len(close_a), len(close_b))
                close_a = close_a.iloc[-common_len:]
                close_b = close_b.iloc[-common_len:]

            # Test cointegration
            is_coint, coint_info = self._test_cointegration(pair_key, close_a, close_b)
            if not is_coint:
                continue

            # Compute hedge ratio and spread
            hedge_ratio = self._compute_hedge_ratio(close_a, close_b)
            self._hedge_ratios[pair_key] = hedge_ratio

            # Log-price spread
            log_a = np.log(close_a.values)
            log_b = np.log(close_b.values)
            spread = log_a - hedge_ratio * log_b

            # Compute z-score of spread
            spread_mean = np.mean(spread[-self.spread_window:])
            spread_std = np.std(spread[-self.spread_window:])
            if spread_std <= 0:
                continue

            current_zscore = (spread[-1] - spread_mean) / spread_std

            # OU half-life check
            halflife = self._ou_halflife(spread)
            if halflife is not None:
                if halflife < self.min_halflife or halflife > self.max_halflife:
                    logger.debug("Half-life out of range", extra={
                        "pair": pair_key, "halflife": round(halflife, 2),
                    })
                    continue

            self._spread_stats[pair_key] = {
                "zscore": current_zscore,
                "spread_mean": spread_mean,
                "spread_std": spread_std,
                "hedge_ratio": hedge_ratio,
                "halflife": halflife or 0.0,
            }

            # Generate signal based on z-score
            signal = self._zscore_signal(
                symbol, current_zscore, pair_key, data_a,
                indicators.get(self.primary_timeframe, pd.DataFrame()),
                coint_info,
            )

            if signal is not None and abs(current_zscore) > abs(best_zscore):
                best_signal = signal
                best_zscore = current_zscore

        return best_signal if best_signal is not None else self._neutral_signal(symbol)

    # ------------------------------------------------------------------
    # Cointegration testing
    # ------------------------------------------------------------------

    def _test_cointegration(
        self,
        pair_key: str,
        close_a: pd.Series,
        close_b: pd.Series,
    ) -> tuple[bool, dict[str, Any]]:
        """Test for cointegration using Engle-Granger method."""
        try:
            from statsmodels.tsa.stattools import coint
            stat, pvalue, crit_values = coint(close_a.values, close_b.values)

            info = {
                "statistic": round(float(stat), 4),
                "pvalue": round(float(pvalue), 4),
                "critical_1pct": round(float(crit_values[0]), 4),
                "critical_5pct": round(float(crit_values[1]), 4),
            }

            is_cointegrated = pvalue < self.coint_pvalue
            self._coint_cache[pair_key] = info

            if is_cointegrated:
                log_with_data(logger, "debug", "Pair is cointegrated", {
                    "pair": pair_key, **info,
                })

            return is_cointegrated, info

        except ImportError:
            logger.warning("statsmodels not installed — cannot test cointegration")
            return False, {}
        except Exception as exc:
            logger.warning("Cointegration test failed for %s: %s", pair_key, exc)
            return False, {}

    # ------------------------------------------------------------------
    # Hedge ratio (rolling OLS)
    # ------------------------------------------------------------------

    def _compute_hedge_ratio(
        self,
        close_a: pd.Series,
        close_b: pd.Series,
    ) -> float:
        """Compute hedge ratio using OLS on log prices."""
        log_a = np.log(close_a.values[-self.hedge_window:])
        log_b = np.log(close_b.values[-self.hedge_window:])

        # Simple OLS: log_a = alpha + beta * log_b
        b_mean = np.mean(log_b)
        a_mean = np.mean(log_a)
        numerator = np.sum((log_b - b_mean) * (log_a - a_mean))
        denominator = np.sum((log_b - b_mean) ** 2)

        if denominator == 0:
            return 1.0

        beta = numerator / denominator
        return float(beta)

    # ------------------------------------------------------------------
    # Ornstein-Uhlenbeck half-life
    # ------------------------------------------------------------------

    @staticmethod
    def _ou_halflife(spread: np.ndarray) -> Optional[float]:
        """Estimate OU half-life from spread series.

        Fits: delta_spread = a + b * spread_lag
        Half-life = -ln(2) / b
        """
        if len(spread) < 10:
            return None

        spread_lag = spread[:-1]
        delta_spread = np.diff(spread)

        # OLS: delta = a + b * lag
        X = np.column_stack([np.ones(len(spread_lag)), spread_lag])
        try:
            beta = np.linalg.lstsq(X, delta_spread, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None

        b = beta[1]
        if b >= 0:
            return None  # Not mean-reverting

        halflife = -math.log(2) / b
        return float(halflife)

    # ------------------------------------------------------------------
    # Z-score signal generation
    # ------------------------------------------------------------------

    def _zscore_signal(
        self,
        symbol: str,
        zscore: float,
        pair_key: str,
        primary_data: pd.DataFrame,
        primary_ind: pd.DataFrame,
        coint_info: dict[str, Any],
    ) -> Optional[TradeSignal]:
        """Generate signal based on spread z-score."""
        abs_z = abs(zscore)

        # Exit signal
        if abs_z < self.zscore_exit:
            return None

        # Stop signal (z-score too extreme — spread diverging)
        if abs_z > self.zscore_stop:
            logger.warning("Z-score exceeds stop threshold", extra={
                "pair": pair_key, "zscore": round(zscore, 2),
            })
            return None

        # Entry signal
        if abs_z < self.zscore_entry:
            return None

        # Z-score > entry: spread is extended, expect mean reversion
        # Positive z-score (A overvalued vs B) → short A / long B
        # Negative z-score (A undervalued vs B) → long A / short B
        if zscore > self.zscore_entry:
            direction = SignalDirection.SHORT
        elif zscore < -self.zscore_entry:
            direction = SignalDirection.LONG
        else:
            return None

        # Compute score
        score = self._compute_score(zscore, coint_info)

        # Build signal
        entry_price = float(primary_data["close"].iloc[-1])
        atr = self._safe_last(primary_ind, "atr")
        atr_val = float(atr) if atr is not None else entry_price * 0.015

        stop_loss = self.compute_stop_loss(
            entry_price, direction, atr_val,
            atr_multiplier=2.0,
        )
        tp1, tp2, tp3 = self.compute_take_profits(
            entry_price, stop_loss, direction,
            tp1_r=1.5, tp2_r=2.5, tp3_r=3.5,
        )
        confidence = score / 100.0

        stats = self._spread_stats.get(pair_key, {})

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
                "pair": pair_key,
                "zscore": round(zscore, 4),
                "hedge_ratio": round(stats.get("hedge_ratio", 0), 4),
                "halflife": round(stats.get("halflife", 0), 2),
                "coint_pvalue": coint_info.get("pvalue", 0),
                "regime_required": self.active_regimes,
            },
        )

        log_with_data(logger, "info", "Stat arb signal generated", {
            "symbol": symbol,
            "pair": pair_key,
            "direction": direction.value,
            "zscore": round(zscore, 4),
            "score": signal.score,
            "halflife": round(stats.get("halflife", 0), 2),
        })

        return signal

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _compute_score(
        self,
        zscore: float,
        coint_info: dict[str, Any],
    ) -> int:
        """Compute conviction score for a stat arb signal."""
        score = self.base_score

        # Z-score strength bonus: +10 for z > 2.5, +20 for z > 3.0
        abs_z = abs(zscore)
        if abs_z > 3.0:
            score += 20
        elif abs_z > 2.5:
            score += 10

        # Strong cointegration bonus: +10 for p < 0.01
        pvalue = coint_info.get("pvalue", 1.0)
        if pvalue < 0.01:
            score += 10
        elif pvalue < 0.03:
            score += 5

        return int(np.clip(score, 0, 100))

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_last(
        df: pd.DataFrame,
        column: str,
        offset: int = 0,
    ) -> Optional[float]:
        """Safely retrieve the last value of a column."""
        if column not in df.columns:
            return None
        idx = -(1 + offset)
        if abs(idx) > len(df):
            return None
        val = df[column].iloc[idx]
        if pd.isna(val):
            return None
        return float(val)
