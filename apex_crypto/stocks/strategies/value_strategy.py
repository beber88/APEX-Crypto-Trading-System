"""Goldman Sachs-inspired Value Strategy for stocks.

Combines the stock screener score with DCF valuation to find
undervalued stocks with strong fundamentals.

Signal generation:
- Screen stocks using GS screener (composite score)
- Run DCF valuation to check intrinsic value vs market price
- Generate BUY signals for undervalued stocks with high screener scores
- Generate SELL signals for overvalued stocks with low scores
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from apex_crypto.core.logging import get_logger, log_with_data
from apex_crypto.core.strategies.base import (
    BaseStrategy, SignalDirection, TradeSignal,
)

logger = get_logger("stocks.strategies.value")


class StockValueStrategy(BaseStrategy):
    """Value investing strategy using fundamental analysis."""

    name = "stock_value"
    active_regimes = []  # active in all regimes
    primary_timeframe = "1d"
    confirmation_timeframe = "1d"
    entry_timeframe = "1d"

    def __init__(self, config: dict) -> None:
        super().__init__(config)

        self._min_composite_score: float = config.get("min_composite_score", 60)
        self._min_dcf_upside: float = config.get("min_dcf_upside_pct", 15)
        self._max_risk_rating: int = config.get("max_risk_rating", 7)

        # Lazy-loaded analysis modules
        self._screener = None
        self._dcf = None

        logger.info("StockValueStrategy configured")

    def _ensure_modules(self) -> None:
        if self._screener is None:
            from apex_crypto.stocks.analysis.screener import StockScreener
            self._screener = StockScreener()
        if self._dcf is None:
            from apex_crypto.stocks.analysis.dcf import DCFValuation
            self._dcf = DCFValuation()

    def generate_signal(
        self,
        symbol: str,
        data: dict[str, pd.DataFrame],
        indicators: dict[str, pd.DataFrame],
        regime: str,
        alt_data: Optional[dict] = None,
    ) -> TradeSignal:
        """Generate signal based on fundamental value analysis."""
        fundamentals = (alt_data or {}).get("fundamentals")
        if not fundamentals or fundamentals.get("error"):
            return self._neutral_signal(symbol)

        self._ensure_modules()

        # Run screener
        screen_result = self._screener.screen(fundamentals)
        composite = screen_result["composite_score"]
        risk_rating = screen_result["risk_rating"]

        # Run DCF
        dcf_result = self._dcf.valuate(fundamentals)
        upside_pct = dcf_result.get("upside_pct", 0)
        verdict = dcf_result.get("verdict", "FAIR VALUE")

        current_price = fundamentals.get("current_price", 0)
        if current_price <= 0:
            return self._neutral_signal(symbol)

        # Score calculation
        score = 0

        # Screener component (0-50)
        if composite >= 80:
            score += 40
        elif composite >= 70:
            score += 30
        elif composite >= 60:
            score += 20
        elif composite < 40:
            score -= 20

        # DCF component (0-50)
        if upside_pct >= 30:
            score += 40
        elif upside_pct >= 20:
            score += 30
        elif upside_pct >= 10:
            score += 20
        elif upside_pct <= -20:
            score -= 30
        elif upside_pct <= -10:
            score -= 15

        # Risk adjustment
        if risk_rating >= 8:
            score = int(score * 0.6)
        elif risk_rating >= 6:
            score = int(score * 0.8)

        # Determine direction
        if score > 0:
            direction = SignalDirection.LONG
        elif score < 0:
            direction = SignalDirection.SHORT
        else:
            return self._neutral_signal(symbol)

        # Calculate stop loss and targets
        stop_loss = screen_result.get("stop_loss", current_price * 0.93)
        targets = screen_result.get("price_targets", {})
        tp1 = targets.get("pessimistic", current_price * 1.05)
        tp2 = targets.get("base", current_price * 1.10)
        tp3 = targets.get("optimistic", current_price * 1.20)

        signal = TradeSignal(
            symbol=symbol,
            direction=direction,
            score=max(-100, min(100, score)),
            strategy=self.name,
            timeframe=self.primary_timeframe,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            confidence=min(composite / 100, 0.95),
            metadata={
                "composite_score": composite,
                "dcf_upside_pct": upside_pct,
                "dcf_verdict": verdict,
                "moat_rating": screen_result.get("moat_rating"),
                "risk_rating": risk_rating,
                "recommendation": screen_result.get("recommendation"),
                "asset_type": "stock",
            },
        )

        log_with_data(logger, "info", "Value signal generated", {
            "symbol": symbol,
            "score": signal.score,
            "composite": composite,
            "dcf_upside": upside_pct,
        })

        return signal
