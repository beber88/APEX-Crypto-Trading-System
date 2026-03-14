"""Quantitative Momentum strategy for the APEX Crypto Trading System.

Implements cross-sectional and time-series momentum factors modelled
after Renaissance Technologies' quantitative approach.

Features:
- Cross-sectional momentum: rank assets by 90-day return, skip last 7 days
- Time-series momentum: risk-adjusted 12-month return
- Crash protection: BTC drawdown >10% in 20d pauses longs
- Volatility filtering and funding-rate crowding detection
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

logger = get_logger("strategies.quant_momentum")


class QuantMomentum(BaseStrategy):
    """Cross-sectional and time-series momentum strategy.

    Uses multi-horizon return signals with volatility normalisation,
    crash protection, and crowding filters.

    Attributes:
        name: Strategy identifier.
        active_regimes: Active in trending and ranging regimes.
        primary_timeframe: Timeframe for signal computation.
    """

    name: str = "quant_momentum"
    active_regimes: list[str] = ["STRONG_BULL", "WEAK_BULL", "RANGING", "WEAK_BEAR"]
    primary_timeframe: str = "4h"
    confirmation_timeframe: str = "1d"
    entry_timeframe: str = "1h"

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    _DEFAULT_LOOKBACK_LONG: int = 90       # days for momentum calc
    _DEFAULT_SKIP_RECENT: int = 7          # skip last 7 days (reversal effect)
    _DEFAULT_VOL_WINDOW: int = 20          # days for vol normalisation
    _DEFAULT_CRASH_DRAWDOWN: float = 0.10  # 10% BTC drawdown = crash
    _DEFAULT_CRASH_WINDOW: int = 20        # days to measure drawdown
    _DEFAULT_MAX_FUNDING: float = 0.001    # max 8h funding rate (crowding)
    _DEFAULT_MIN_VOL_PCT: float = 0.10     # min vol percentile
    _DEFAULT_MAX_VOL_PCT: float = 0.90     # max vol percentile
    _DEFAULT_BASE_SCORE: int = 55

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.lookback_long: int = config.get("lookback_long", self._DEFAULT_LOOKBACK_LONG)
        self.skip_recent: int = config.get("skip_recent", self._DEFAULT_SKIP_RECENT)
        self.vol_window: int = config.get("vol_window", self._DEFAULT_VOL_WINDOW)
        self.crash_drawdown: float = config.get("crash_drawdown", self._DEFAULT_CRASH_DRAWDOWN)
        self.crash_window: int = config.get("crash_window", self._DEFAULT_CRASH_WINDOW)
        self.max_funding: float = config.get("max_funding_rate", self._DEFAULT_MAX_FUNDING)
        self.min_vol_pct: float = config.get("min_vol_percentile", self._DEFAULT_MIN_VOL_PCT)
        self.max_vol_pct: float = config.get("max_vol_percentile", self._DEFAULT_MAX_VOL_PCT)
        self.base_score: int = config.get("base_score", self._DEFAULT_BASE_SCORE)

        log_with_data(logger, "info", "QuantMomentum initialized", {
            "lookback": self.lookback_long,
            "skip_recent": self.skip_recent,
            "crash_drawdown": self.crash_drawdown,
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
        """Generate a momentum-based trading signal."""
        if not self.is_active(regime):
            return self._neutral_signal(symbol)

        if self.primary_timeframe not in data:
            return self._neutral_signal(symbol)

        primary_data = data[self.primary_timeframe]
        if primary_data.empty or len(primary_data) < (self.lookback_long * 6 + self.skip_recent * 6):
            return self._neutral_signal(symbol)

        alt = alt_data or {}

        # Crash protection: check BTC drawdown
        if self._is_crash(alt):
            logger.info("Crash protection active — no longs", extra={"symbol": symbol})
            return self._neutral_signal(symbol)

        # Funding rate crowding filter
        funding_rate = alt.get("funding_rate", 0.0)
        if abs(funding_rate) > self.max_funding:
            logger.debug("Funding rate crowding detected", extra={
                "symbol": symbol, "funding_rate": funding_rate,
            })
            return self._neutral_signal(symbol)

        # Compute momentum signals
        close = primary_data["close"]

        # Time-series momentum: return over lookback, skipping recent days
        # Convert days to bars (4h = 6 bars/day)
        bars_per_day = 6
        lookback_bars = self.lookback_long * bars_per_day
        skip_bars = self.skip_recent * bars_per_day

        if len(close) < lookback_bars + skip_bars:
            return self._neutral_signal(symbol)

        # Momentum = return from t-lookback to t-skip
        price_start = close.iloc[-(lookback_bars + skip_bars)]
        price_end = close.iloc[-skip_bars] if skip_bars > 0 else close.iloc[-1]
        raw_momentum = (price_end / price_start) - 1.0

        # Volatility-adjusted momentum (t-statistic of trend)
        log_returns = np.log(close / close.shift(1)).dropna()
        vol_window_bars = self.vol_window * bars_per_day
        if len(log_returns) < vol_window_bars:
            return self._neutral_signal(symbol)

        realised_vol = float(log_returns.iloc[-vol_window_bars:].std())
        if realised_vol <= 0:
            return self._neutral_signal(symbol)

        annualised_vol = realised_vol * np.sqrt(252 * bars_per_day)
        vol_adj_momentum = raw_momentum / (realised_vol * np.sqrt(lookback_bars))

        # Volatility percentile filter
        vol_pct = self._vol_percentile(log_returns, vol_window_bars)
        if vol_pct < self.min_vol_pct or vol_pct > self.max_vol_pct:
            logger.debug("Vol percentile filter", extra={
                "symbol": symbol, "vol_pct": round(vol_pct, 2),
            })
            return self._neutral_signal(symbol)

        # Determine direction based on vol-adjusted momentum
        if vol_adj_momentum > 0.5:
            direction = SignalDirection.LONG
        elif vol_adj_momentum < -0.5:
            direction = SignalDirection.SHORT
        else:
            return self._neutral_signal(symbol)

        # In BEAR regime, only allow shorts
        if regime == "BEAR" and direction == SignalDirection.LONG:
            return self._neutral_signal(symbol)

        # Score computation
        score = self._compute_score(vol_adj_momentum, raw_momentum, regime)

        # Build signal
        entry_tf = self.entry_timeframe if self.entry_timeframe in data else self.primary_timeframe
        entry_data = data[entry_tf]
        primary_ind = indicators.get(self.primary_timeframe, pd.DataFrame())

        return self._build_signal(
            symbol, direction, score, entry_data, primary_ind,
            {
                "raw_momentum": round(raw_momentum, 4),
                "vol_adj_momentum": round(vol_adj_momentum, 4),
                "realised_vol": round(annualised_vol, 4),
                "vol_percentile": round(vol_pct, 2),
                "funding_rate": funding_rate,
            },
        )

    # ------------------------------------------------------------------
    # Crash protection
    # ------------------------------------------------------------------

    def _is_crash(self, alt_data: dict) -> bool:
        """Check if BTC has drawn down more than threshold in recent window."""
        btc_prices = alt_data.get("btc_close")
        if btc_prices is None:
            return False

        if isinstance(btc_prices, pd.Series) and len(btc_prices) > self.crash_window:
            recent = btc_prices.iloc[-self.crash_window:]
            peak = recent.max()
            current = recent.iloc[-1]
            if peak > 0:
                drawdown = (peak - current) / peak
                if drawdown > self.crash_drawdown:
                    return True
        return False

    # ------------------------------------------------------------------
    # Volatility percentile
    # ------------------------------------------------------------------

    @staticmethod
    def _vol_percentile(log_returns: pd.Series, window: int) -> float:
        """Compute current volatility percentile over history."""
        current_vol = log_returns.iloc[-window:].std()
        rolling_vol = log_returns.rolling(window).std().dropna()
        if len(rolling_vol) < 10:
            return 0.5
        return float((rolling_vol < current_vol).mean())

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _compute_score(
        self,
        vol_adj_mom: float,
        raw_mom: float,
        regime: str,
    ) -> int:
        """Compute conviction score from momentum strength."""
        score = self.base_score

        # Strong momentum bonus: +15 for vol-adj > 1.5
        if abs(vol_adj_mom) > 1.5:
            score += 15

        # Consistent direction bonus: +10 if raw and vol-adj agree strongly
        if abs(raw_mom) > 0.10 and np.sign(raw_mom) == np.sign(vol_adj_mom):
            score += 10

        # Regime alignment bonus
        if regime in ("STRONG_BULL",) and vol_adj_mom > 0:
            score += 10
        elif regime in ("BEAR",) and vol_adj_mom < 0:
            score += 10

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
        """Build a TradeSignal with entry, stop, and take-profit levels."""
        entry_price = float(entry_data["close"].iloc[-1])

        atr = self._safe_last(primary_ind, "atr")
        atr_val = float(atr) if atr is not None else entry_price * 0.02

        # Wider stops for momentum trades
        if direction == SignalDirection.LONG:
            swing_level = float(entry_data["low"].rolling(30).min().iloc[-1])
        else:
            swing_level = float(entry_data["high"].rolling(30).max().iloc[-1])

        stop_loss = self.compute_stop_loss(
            entry_price, direction, atr_val, swing_level=swing_level,
            atr_multiplier=2.0,
        )
        tp1, tp2, tp3 = self.compute_take_profits(
            entry_price, stop_loss, direction,
            tp1_r=2.0, tp2_r=3.5, tp3_r=5.0,
        )
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

        log_with_data(logger, "info", "Momentum signal generated", {
            "symbol": symbol,
            "direction": direction.value,
            "score": signal.score,
            "entry": entry_price,
            "vol_adj_mom": extras.get("vol_adj_momentum"),
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
