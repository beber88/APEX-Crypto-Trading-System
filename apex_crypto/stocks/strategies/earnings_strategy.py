"""JPMorgan-inspired Earnings Strategy for stocks.

Generates pre-earnings trade signals based on:
- Historical beat/miss patterns
- Price reaction patterns around earnings
- Earnings quality score
- Bull/Bear case analysis

Designed to capture the earnings announcement drift.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Optional

import pandas as pd

from apex_crypto.core.logging import get_logger, log_with_data
from apex_crypto.core.strategies.base import (
    BaseStrategy, SignalDirection, TradeSignal,
)

logger = get_logger("stocks.strategies.earnings")


class StockEarningsStrategy(BaseStrategy):
    """Pre-earnings momentum/contrarian strategy."""

    name = "stock_earnings"
    active_regimes = []
    primary_timeframe = "1d"
    confirmation_timeframe = "1d"
    entry_timeframe = "1d"

    def __init__(self, config: dict) -> None:
        super().__init__(config)

        self._days_before_earnings: int = config.get("days_before_earnings", 10)
        self._min_quality_score: float = config.get("min_quality_score", 55)
        self._min_beat_rate: float = config.get("min_beat_rate", 0.60)

        self._analyzer = None
        logger.info("StockEarningsStrategy configured")

    def _ensure_analyzer(self) -> None:
        if self._analyzer is None:
            from apex_crypto.stocks.analysis.earnings import EarningsAnalyzer
            self._analyzer = EarningsAnalyzer()

    def generate_signal(
        self,
        symbol: str,
        data: dict[str, pd.DataFrame],
        indicators: dict[str, pd.DataFrame],
        regime: str,
        alt_data: Optional[dict] = None,
    ) -> TradeSignal:
        """Generate signal based on earnings analysis."""
        fundamentals = (alt_data or {}).get("fundamentals")
        if not fundamentals or fundamentals.get("error"):
            return self._neutral_signal(symbol)

        # Check if earnings are coming up
        next_earnings = fundamentals.get("next_earnings_date")
        if not next_earnings:
            return self._neutral_signal(symbol)

        try:
            earnings_date = pd.Timestamp(next_earnings)
            if earnings_date.tzinfo:
                earnings_date = earnings_date.tz_localize(None)
            now = pd.Timestamp.now()
            days_until = (earnings_date - now).days
        except Exception:
            return self._neutral_signal(symbol)

        # Only generate signals within the pre-earnings window
        if days_until < 0 or days_until > self._days_before_earnings:
            return self._neutral_signal(symbol)

        self._ensure_analyzer()

        # Get daily OHLCV for price reaction analysis
        daily_df = data.get("1d")

        # Run full analysis
        analysis = self._analyzer.analyze(fundamentals, daily_df)

        quality_score = analysis.get("quality_score", 50)
        recommendation = analysis.get("recommendation", {})
        action = recommendation.get("action", "HOLD")
        confidence = recommendation.get("confidence", 0.5)
        history = analysis.get("earnings_history", {})
        beat_rate = history.get("beat_rate", 0.5)

        current_price = fundamentals.get("current_price", 0)
        if current_price <= 0:
            return self._neutral_signal(symbol)

        # Score calculation
        score = 0

        if action == "BUY BEFORE EARNINGS":
            score = 50
            direction = SignalDirection.LONG
        elif action == "REDUCE BEFORE EARNINGS":
            score = -40
            direction = SignalDirection.SHORT
        else:
            # Mild signal based on quality
            if quality_score >= 70:
                score = 25
                direction = SignalDirection.LONG
            elif quality_score < 35:
                score = -25
                direction = SignalDirection.SHORT
            else:
                return self._neutral_signal(symbol)

        # Confidence adjustment
        score = int(score * confidence)

        # Beat rate bonus
        if beat_rate >= 0.80:
            score += 15 if score > 0 else -15
        elif beat_rate <= 0.30:
            score += -15 if score > 0 else 15

        # Calculate targets based on expected move
        price_reactions = analysis.get("price_reactions", {})
        expected_move = price_reactions.get("avg_absolute_move_pct", 3.0)

        if direction == SignalDirection.LONG:
            stop_loss = current_price * (1 - expected_move / 100 * 1.5)
            tp1 = current_price * (1 + expected_move / 100)
            tp2 = current_price * (1 + expected_move / 100 * 1.5)
            tp3 = current_price * (1 + expected_move / 100 * 2.5)
        else:
            stop_loss = current_price * (1 + expected_move / 100 * 1.5)
            tp1 = current_price * (1 - expected_move / 100)
            tp2 = current_price * (1 - expected_move / 100 * 1.5)
            tp3 = current_price * (1 - expected_move / 100 * 2.5)

        signal = TradeSignal(
            symbol=symbol,
            direction=direction,
            score=max(-100, min(100, score)),
            strategy=self.name,
            timeframe=self.primary_timeframe,
            entry_price=current_price,
            stop_loss=round(stop_loss, 2),
            take_profit_1=round(tp1, 2),
            take_profit_2=round(tp2, 2),
            take_profit_3=round(tp3, 2),
            confidence=confidence,
            metadata={
                "days_until_earnings": days_until,
                "earnings_date": str(next_earnings),
                "beat_rate": beat_rate,
                "quality_score": quality_score,
                "recommendation": action,
                "expected_move_pct": expected_move,
                "bull_case": analysis.get("bull_case", {}),
                "bear_case": analysis.get("bear_case", {}),
                "asset_type": "stock",
            },
        )

        log_with_data(logger, "info", "Earnings signal generated", {
            "symbol": symbol,
            "score": signal.score,
            "days_until": days_until,
            "action": action,
            "quality": quality_score,
        })

        return signal
