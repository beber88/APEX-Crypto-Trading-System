"""Open interest divergence strategy for the APEX Crypto Trading System.

Detects divergences between price action and open interest to identify
capitulation (short liquidation cascades) and overcrowding (leveraged
long blowoff) setups on Tier 1 perpetual assets.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from apex_crypto.core.logging import get_logger, log_with_data
from apex_crypto.core.strategies.base import (
    BaseStrategy,
    SignalDirection,
    TradeSignal,
)

logger = get_logger("strategies.oi_divergence")

# Default Tier 1 assets
_DEFAULT_TIER1_ASSETS: list[str] = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "XRP/USDT",
]

# RSI thresholds
_RSI_OVERSOLD: float = 35.0
_RSI_EXTREME_OVERSOLD: float = 25.0
_RSI_OVERBOUGHT: float = 70.0
_RSI_EXTREME_OVERBOUGHT: float = 80.0

# OI rate-of-change threshold for "rising fast"
_OI_FAST_RISE_PCT: float = 5.0  # 5% increase over lookback

# Funding rate threshold for short signal confirmation
_FUNDING_SHORT_THRESHOLD: float = 0.08  # 0.08%

# Price change lookback (number of 4h bars)
_PRICE_LOOKBACK: int = 6  # 24 hours of 4h bars

# Volume divergence: current vs mean ratio threshold
_VOLUME_DIV_THRESHOLD: float = 0.7  # volume declining relative to move

# Regime alignment maps
_BULLISH_REGIMES: set[str] = {"STRONG_BULL", "WEAK_BULL"}
_BEARISH_REGIMES: set[str] = {"STRONG_BEAR", "WEAK_BEAR"}


class OIDivergenceStrategy(BaseStrategy):
    """Trade open interest divergences on Tier 1 perpetual assets.

    Long signal: price falling + OI falling (short capitulation) + RSI < 35.
    Short signal: price rising + OI rising fast + funding > 0.08% + RSI > 70.

    These divergences often precede sharp reversals as leveraged positions
    unwind, creating liquidity cascades.

    Attributes:
        name: Strategy identifier.
        active_regimes: Market regimes where this strategy operates.
        primary_timeframe: Timeframe used for analysis.
    """

    name: str = "oi_divergence"
    active_regimes: list[str] = [
        "STRONG_BULL",
        "WEAK_BULL",
        "WEAK_BEAR",
        "STRONG_BEAR",
    ]
    primary_timeframe: str = "4h"

    def __init__(self, config: dict) -> None:
        """Initialize the OI divergence strategy.

        Args:
            config: Strategy-specific configuration dict.
        """
        super().__init__(config)

        oi_cfg: dict = config.get("strategies", {}).get("oi_divergence", {})
        self._tier1_assets: list[str] = oi_cfg.get("tier1_assets", _DEFAULT_TIER1_ASSETS)
        self._rsi_oversold: float = oi_cfg.get("rsi_oversold", _RSI_OVERSOLD)
        self._rsi_overbought: float = oi_cfg.get("rsi_overbought", _RSI_OVERBOUGHT)
        self._oi_fast_rise_pct: float = oi_cfg.get(
            "oi_fast_rise_pct", _OI_FAST_RISE_PCT
        )
        self._funding_threshold: float = oi_cfg.get(
            "funding_short_threshold", _FUNDING_SHORT_THRESHOLD
        )
        self._price_lookback: int = oi_cfg.get("price_lookback", _PRICE_LOOKBACK)

        logger.info(
            "OIDivergenceStrategy configured",
            extra={
                "data": {
                    "tier1_assets": self._tier1_assets,
                    "rsi_oversold": self._rsi_oversold,
                    "rsi_overbought": self._rsi_overbought,
                    "oi_fast_rise_pct": self._oi_fast_rise_pct,
                    "funding_short_threshold": self._funding_threshold,
                }
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
        """Generate an OI divergence signal for *symbol*.

        Args:
            symbol: Trading pair symbol (e.g. ``'BTC/USDT'``).
            data: OHLCV DataFrames keyed by timeframe. Requires ``'4h'``.
            indicators: Pre-computed indicator DataFrames keyed by timeframe.
                Expected columns on the 4h frame: ``rsi``, ``atr``, ``volume``.
            regime: Current market regime string.
            alt_data: Alternative data dict. Required keys:
                ``open_interest`` (float — current OI),
                ``open_interest_change`` (float — percentage change over
                lookback), ``funding_rate`` (float — percentage).

        Returns:
            TradeSignal with score of 0 (neutral) when conditions are not met,
            or 55-100 when a valid OI divergence setup is detected.
        """
        # ----- Gate checks ------------------------------------------------
        if not self.is_active(regime):
            logger.debug("Regime %s not active for OI divergence strategy", regime)
            return self._neutral_signal(symbol)

        if symbol not in self._tier1_assets:
            logger.debug("%s is not a Tier 1 asset — skipping OI divergence", symbol)
            return self._neutral_signal(symbol)

        if alt_data is None:
            logger.debug("No alt_data provided for %s", symbol)
            return self._neutral_signal(symbol)

        oi_change: Optional[float] = alt_data.get("open_interest_change")
        funding_rate: Optional[float] = alt_data.get("funding_rate")

        if oi_change is None:
            logger.debug("open_interest_change missing in alt_data for %s", symbol)
            return self._neutral_signal(symbol)

        # ----- OHLCV + indicators -----------------------------------------
        df = data.get(self.primary_timeframe)
        ind = indicators.get(self.primary_timeframe)

        if df is None or df.empty or ind is None or ind.empty:
            logger.debug("Missing 4h data/indicators for %s", symbol)
            return self._neutral_signal(symbol)

        if "rsi" not in ind.columns:
            logger.debug("RSI indicator missing for %s", symbol)
            return self._neutral_signal(symbol)

        rsi: float = float(ind["rsi"].iloc[-1])
        close: float = float(df["close"].iloc[-1])

        # Price change over lookback
        lookback_idx: int = max(0, len(df) - self._price_lookback - 1)
        price_prev: float = float(df["close"].iloc[lookback_idx])
        price_change_pct: float = (
            ((close - price_prev) / price_prev) * 100 if price_prev > 0 else 0.0
        )

        # ----- Signal detection -------------------------------------------
        direction: Optional[SignalDirection] = None

        # Long: price falling + OI falling (short capitulation) + RSI oversold
        if price_change_pct < 0 and oi_change < 0 and rsi < self._rsi_oversold:
            direction = SignalDirection.LONG
            logger.debug(
                "Long divergence detected for %s: price=%.2f%%, OI=%.2f%%, RSI=%.1f",
                symbol,
                price_change_pct,
                oi_change,
                rsi,
            )

        # Short: price rising + OI rising fast + high funding + RSI overbought
        elif (
            price_change_pct > 0
            and oi_change > self._oi_fast_rise_pct
            and funding_rate is not None
            and funding_rate > self._funding_threshold
            and rsi > self._rsi_overbought
        ):
            direction = SignalDirection.SHORT
            logger.debug(
                "Short divergence detected for %s: price=%.2f%%, OI=%.2f%%, "
                "funding=%.4f%%, RSI=%.1f",
                symbol,
                price_change_pct,
                oi_change,
                funding_rate,
                rsi,
            )

        if direction is None:
            return self._neutral_signal(symbol)

        # ----- Score computation ------------------------------------------
        score: int = 55

        # Extreme RSI bonus
        if direction == SignalDirection.LONG and rsi < _RSI_EXTREME_OVERSOLD:
            score += 15
        elif direction == SignalDirection.SHORT and rsi > _RSI_EXTREME_OVERBOUGHT:
            score += 15

        # Funding rate confirmation bonus
        funding_confirmed: bool = False
        if funding_rate is not None:
            if direction == SignalDirection.LONG and funding_rate < -0.03:
                funding_confirmed = True
                score += 10
            elif direction == SignalDirection.SHORT and funding_rate > self._funding_threshold:
                funding_confirmed = True
                score += 10

        # Volume divergence: declining volume into the move suggests exhaustion
        volume_divergence: bool = False
        if "volume" in df.columns and len(df) >= 20:
            vol_current: float = float(df["volume"].iloc[-1])
            vol_mean: float = float(df["volume"].rolling(20).mean().iloc[-1])
            if vol_mean > 0:
                vol_ratio: float = vol_current / vol_mean
                if direction == SignalDirection.LONG and vol_ratio < _VOLUME_DIV_THRESHOLD:
                    volume_divergence = True
                    score += 10
                elif (
                    direction == SignalDirection.SHORT
                    and vol_ratio < _VOLUME_DIV_THRESHOLD
                ):
                    volume_divergence = True
                    score += 10

        # Regime alignment bonus
        regime_aligned: bool = self._regime_aligns(direction, regime)
        if regime_aligned:
            score += 10

        score = min(score, 100)

        # ----- Entry, stop, targets ---------------------------------------
        entry_price: float = close

        atr: float = 0.0
        if "atr" in ind.columns:
            atr = float(ind["atr"].iloc[-1])

        if atr > 0:
            stop_loss = self.compute_stop_loss(
                entry_price, direction, atr, atr_multiplier=2.0, max_stop_pct=0.03
            )
        else:
            pct = 0.025
            if direction == SignalDirection.LONG:
                stop_loss = entry_price * (1 - pct)
            else:
                stop_loss = entry_price * (1 + pct)

        tp1, tp2, tp3 = self.compute_take_profits(
            entry_price, stop_loss, direction, tp1_r=1.5, tp2_r=2.5, tp3_r=4.0
        )

        confidence: float = round(score / 100.0, 2)

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
                "price_change_pct": round(price_change_pct, 4),
                "oi_change_pct": round(oi_change, 4),
                "rsi": round(rsi, 2),
                "funding_rate": funding_rate,
                "funding_confirmed": funding_confirmed,
                "volume_divergence": volume_divergence,
                "regime_aligned": regime_aligned,
                "atr": round(atr, 4) if atr else None,
            },
        )

        log_with_data(
            logger,
            "info",
            f"OI divergence signal generated for {symbol}",
            data=signal.to_dict(),
        )

        return signal

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _regime_aligns(direction: SignalDirection, regime: str) -> bool:
        """Check whether the regime supports the proposed direction.

        Args:
            direction: Proposed signal direction.
            regime: Current market regime string.

        Returns:
            True if the regime agrees with the signal direction.
        """
        if direction == SignalDirection.LONG:
            return regime in _BULLISH_REGIMES
        if direction == SignalDirection.SHORT:
            return regime in _BEARISH_REGIMES
        return False
