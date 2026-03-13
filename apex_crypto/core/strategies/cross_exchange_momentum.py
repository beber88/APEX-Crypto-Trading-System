"""Cross-Exchange Momentum (Lead-Lag) strategy for the APEX Crypto Trading System.

Exploits the price-discovery latency between a leading exchange (e.g.
Binance, Bybit) and the execution venue (MEXC).  When the lead exchange
prints a rapid directional move (>0.15% in 30 seconds), this strategy
enters in the same direction on MEXC before the lagging venue catches up.

Applicable to BTC/USDT, ETH/USDT, SOL/USDT only.

Stop: 0.2%.  Target: 0.2-0.4%.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np
import pandas as pd

from apex_crypto.core.logging import get_logger
from apex_crypto.core.strategies.base import (
    BaseStrategy,
    SignalDirection,
    TradeSignal,
)

logger = get_logger("strategies.cross_exchange_momentum")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_ALLOWED_ASSETS: list[str] = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
_TIMEFRAME: str = "5m"

_LEAD_MOVE_THRESHOLD: float = 0.0015          # 0.15%
_LEAD_MOVE_STRONG: float = 0.0025             # 0.25%
_LEAD_WINDOW_SECONDS: int = 30                # 30-second window

_STOP_PCT: float = 0.002                       # 0.2%
_TP_MIN_PCT: float = 0.002                     # 0.2%
_TP_MAX_PCT: float = 0.004                     # 0.4%

_BASE_SCORE: int = 55
_STRONG_MOMENTUM_BONUS: int = 15
_VOLUME_CONFIRMATION_BONUS: int = 10

_VOLUME_LOOKBACK: int = 50
_VOLUME_ZSCORE_THRESHOLD: float = 1.0


class CrossExchangeMomentumStrategy(BaseStrategy):
    """Cross-exchange lead-lag momentum strategy.

    Monitors a leading exchange for rapid price moves and enters in the
    same direction on the execution venue (MEXC), capturing the latency
    spread before the lagging venue reprices.

    Attributes:
        name: Strategy identifier.
        active_regimes: Empty list means active in all regimes.
        primary_timeframe: 5m bars for volume context.
    """

    name: str = "cross_exchange_momentum"
    active_regimes: list[str] = []
    primary_timeframe: str = _TIMEFRAME
    confirmation_timeframe: str = "1m"
    entry_timeframe: str = "5m"

    def __init__(self, config: dict) -> None:
        """Initialize CrossExchangeMomentumStrategy.

        Args:
            config: Strategy-specific configuration dictionary.
        """
        super().__init__(config)
        cfg = config.get("strategies", {}).get("cross_exchange_momentum", {})
        self.lead_move_threshold: float = cfg.get(
            "lead_move_threshold", _LEAD_MOVE_THRESHOLD
        )
        self.lead_move_strong: float = cfg.get(
            "lead_move_strong", _LEAD_MOVE_STRONG
        )
        self.lead_window_seconds: int = cfg.get(
            "lead_window_seconds", _LEAD_WINDOW_SECONDS
        )
        self.stop_pct: float = cfg.get("stop_pct", _STOP_PCT)
        self.tp_min_pct: float = cfg.get("tp_min_pct", _TP_MIN_PCT)
        self.tp_max_pct: float = cfg.get("tp_max_pct", _TP_MAX_PCT)
        self.base_score: int = cfg.get("base_score", _BASE_SCORE)
        self.volume_lookback: int = cfg.get("volume_lookback", _VOLUME_LOOKBACK)

        logger.info(
            "CrossExchangeMomentumStrategy configured",
            extra={
                "lead_move_threshold": self.lead_move_threshold,
                "lead_window_seconds": self.lead_window_seconds,
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
        """Generate a cross-exchange momentum signal.

        Args:
            symbol: Trading pair symbol.
            data: OHLCV DataFrames keyed by timeframe.
            indicators: Pre-computed indicator DataFrames keyed by timeframe.
            regime: Current market regime string.
            alt_data: Alternative data dict; expected keys:
                - ``lead_exchange`` or ``external_prices``: dict mapping
                  symbol to a dict with ``price``, ``prev_price``, and
                  optionally ``timestamp`` (epoch seconds).

        Returns:
            TradeSignal with direction and score, or NEUTRAL.
        """
        if not self.is_active(regime):
            return self._neutral_signal(symbol)

        if symbol not in _ALLOWED_ASSETS:
            logger.debug("%s not in allowed cross-exchange assets", symbol)
            return self._neutral_signal(symbol)

        # Extract lead-exchange price data
        lead_price, lead_prev_price, lead_ts = self._extract_lead_data(
            symbol, alt_data
        )
        if lead_price is None or lead_prev_price is None:
            logger.debug("No lead exchange data for %s", symbol)
            return self._neutral_signal(symbol)

        # Check staleness: reject if data is older than 2x the window
        if lead_ts is not None:
            age = time.time() - lead_ts
            if age > self.lead_window_seconds * 2:
                logger.debug(
                    "Lead data stale for %s (%.1fs old)", symbol, age
                )
                return self._neutral_signal(symbol)

        # Compute lead-exchange move
        if lead_prev_price <= 0:
            return self._neutral_signal(symbol)
        lead_move = (lead_price - lead_prev_price) / lead_prev_price
        abs_move = abs(lead_move)

        if abs_move < self.lead_move_threshold:
            return self._neutral_signal(symbol)

        # Require OHLCV data for volume confirmation
        tf = self.primary_timeframe
        if tf not in data or data[tf].empty:
            logger.warning("Missing %s data for %s", tf, symbol)
            return self._neutral_signal(symbol)

        df = data[tf]
        mexc_close = float(df["close"].iloc[-1])

        # Direction follows the lead exchange
        if lead_move > 0:
            direction = SignalDirection.LONG
        else:
            direction = SignalDirection.SHORT

        # Score computation
        score = self.base_score

        # Strong momentum bonus
        if abs_move >= self.lead_move_strong:
            score += _STRONG_MOMENTUM_BONUS

        # Volume confirmation bonus
        vol_zscore = self._compute_volume_zscore(df)
        if vol_zscore is not None and vol_zscore >= _VOLUME_ZSCORE_THRESHOLD:
            score += _VOLUME_CONFIRMATION_BONUS

        score = min(score, 100)

        # Entry, stop, targets
        entry_price = mexc_close
        if direction == SignalDirection.LONG:
            stop_loss = entry_price * (1 - self.stop_pct)
            tp1 = entry_price * (1 + self.tp_min_pct)
            tp2 = entry_price * (1 + self.tp_max_pct)
            tp3 = entry_price * (1 + self.tp_max_pct * 1.5)
        else:
            stop_loss = entry_price * (1 + self.stop_pct)
            tp1 = entry_price * (1 - self.tp_min_pct)
            tp2 = entry_price * (1 - self.tp_max_pct)
            tp3 = entry_price * (1 - self.tp_max_pct * 1.5)

        confidence = round(score / 100.0, 2)

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
                "lead_price": round(lead_price, 4),
                "lead_prev_price": round(lead_prev_price, 4),
                "lead_move_pct": round(lead_move * 100, 4),
                "mexc_price": round(mexc_close, 4),
                "volume_zscore": round(vol_zscore, 2) if vol_zscore else None,
                "lead_data_age_s": (
                    round(time.time() - lead_ts, 1) if lead_ts else None
                ),
            },
        )

        logger.info(
            "Cross-exchange momentum signal generated",
            extra={
                "symbol": symbol,
                "direction": direction.value,
                "score": signal.score,
                "lead_move_pct": round(lead_move * 100, 4),
                "volume_zscore": round(vol_zscore, 2) if vol_zscore else None,
            },
        )
        return signal

    # ------------------------------------------------------------------
    # Lead-exchange data extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_lead_data(
        symbol: str,
        alt_data: Optional[dict],
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """Extract lead-exchange price data from alt_data.

        Tries several common key layouts:
        - alt_data["lead_exchange"][symbol]
        - alt_data["external_prices"][symbol]
        - alt_data["cross_exchange"][symbol]

        Each symbol entry should contain ``price`` and ``prev_price``
        keys (floats) and optionally ``timestamp`` (epoch seconds).

        Args:
            symbol: Trading pair.
            alt_data: Alternative data dictionary.

        Returns:
            Tuple of (current_price, previous_price, timestamp).
            Any or all may be None.
        """
        if alt_data is None:
            return None, None, None

        for key in ("lead_exchange", "external_prices", "cross_exchange"):
            container = alt_data.get(key)
            if isinstance(container, dict):
                sym_data = container.get(symbol)
                if isinstance(sym_data, dict):
                    try:
                        price = float(sym_data["price"])
                        prev_price = float(sym_data["prev_price"])
                        ts = sym_data.get("timestamp")
                        ts = float(ts) if ts is not None else None
                        return price, prev_price, ts
                    except (KeyError, TypeError, ValueError):
                        continue

        return None, None, None

    # ------------------------------------------------------------------
    # Volume z-score
    # ------------------------------------------------------------------

    def _compute_volume_zscore(self, df: pd.DataFrame) -> Optional[float]:
        """Compute the z-score of the most recent bar's volume.

        Args:
            df: OHLCV DataFrame with a ``volume`` column.

        Returns:
            Volume z-score, or None on failure.
        """
        try:
            vol = df["volume"].iloc[-self.volume_lookback:]
            if len(vol) < 10:
                return None
            mean = float(vol.mean())
            std = float(vol.std())
            if std <= 0:
                return None
            current = float(vol.iloc[-1])
            return (current - mean) / std
        except Exception:
            logger.exception("Error computing volume z-score")
            return None
