"""Goldman Sachs-style Stock Screener.

Screens and ranks stocks based on institutional-grade criteria:
- P/E ratio vs sector average
- Revenue growth trends (5 years)
- Financial health (debt-to-equity ratio)
- Dividend yield and payout sustainability
- Competitive moat rating (weak/medium/strong)
- 12-month price targets (optimistic/pessimistic)
- Risk rating (1-10)
- Entry price zones and stop-loss suggestions

Inspired by the Goldman Sachs Stock Screener methodology.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from apex_crypto.core.logging import get_logger, log_with_data

logger = get_logger("stocks.screener")

# Sector average P/E ratios (approximate)
SECTOR_AVG_PE: dict[str, float] = {
    "Technology": 28.0,
    "Healthcare": 22.0,
    "Financial Services": 14.0,
    "Consumer Cyclical": 20.0,
    "Consumer Defensive": 24.0,
    "Energy": 12.0,
    "Industrials": 18.0,
    "Communication Services": 20.0,
    "Utilities": 16.0,
    "Real Estate": 30.0,
    "Basic Materials": 15.0,
    "Unknown": 20.0,
}


class StockScreener:
    """Screens stocks using Goldman Sachs-inspired criteria.

    Produces a composite score (0-100) and detailed breakdown for each stock.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._min_market_cap: float = cfg.get("min_market_cap", 1e9)  # $1B
        self._min_volume: int = cfg.get("min_avg_volume", 500_000)
        self._max_debt_to_equity: float = cfg.get("max_debt_to_equity", 200)

    def screen(self, fundamentals: dict[str, Any]) -> dict[str, Any]:
        """Screen a single stock and return a detailed rating.

        Args:
            fundamentals: Dict from StockBroker.fetch_fundamentals().

        Returns:
            Dict with composite_score, sub_scores, moat_rating, price_targets,
            risk_rating, entry_zone, stop_loss, and recommendation.
        """
        symbol = fundamentals.get("symbol", "?")
        sector = fundamentals.get("sector", "Unknown")

        # --- Sub-scores (each 0-100) ---
        valuation_score = self._score_valuation(fundamentals, sector)
        growth_score = self._score_growth(fundamentals)
        health_score = self._score_financial_health(fundamentals)
        dividend_score = self._score_dividend(fundamentals)
        momentum_score = self._score_momentum(fundamentals)
        analyst_score = self._score_analyst_consensus(fundamentals)

        # Weighted composite
        composite = (
            valuation_score * 0.20
            + growth_score * 0.25
            + health_score * 0.20
            + dividend_score * 0.10
            + momentum_score * 0.15
            + analyst_score * 0.10
        )

        # Moat rating
        moat = self._rate_moat(fundamentals, composite)

        # Risk rating (1-10, 1 = lowest risk)
        risk_rating = self._calculate_risk_rating(fundamentals)

        # Price targets
        current_price = fundamentals.get("current_price", 0)
        targets = self._calculate_price_targets(fundamentals, current_price)

        # Entry zone and stop loss
        entry_zone = self._calculate_entry_zone(fundamentals, current_price)
        stop_loss = self._calculate_stop_loss(fundamentals, current_price)

        result = {
            "symbol": symbol,
            "name": fundamentals.get("name", symbol),
            "sector": sector,
            "composite_score": round(composite, 1),
            "sub_scores": {
                "valuation": round(valuation_score, 1),
                "growth": round(growth_score, 1),
                "financial_health": round(health_score, 1),
                "dividend": round(dividend_score, 1),
                "momentum": round(momentum_score, 1),
                "analyst_consensus": round(analyst_score, 1),
            },
            "moat_rating": moat,
            "risk_rating": risk_rating,
            "price_targets": targets,
            "entry_zone": entry_zone,
            "stop_loss": stop_loss,
            "current_price": current_price,
            "recommendation": self._get_recommendation(composite, risk_rating),
        }

        log_with_data(logger, "info", "Stock screened", {
            "symbol": symbol,
            "composite_score": result["composite_score"],
            "moat": moat,
            "risk": risk_rating,
        })

        return result

    def screen_multiple(
        self, fundamentals_list: list[dict], top_n: int = 10
    ) -> list[dict[str, Any]]:
        """Screen multiple stocks and return top N ranked by composite score."""
        results = []
        for f in fundamentals_list:
            try:
                result = self.screen(f)
                results.append(result)
            except Exception as exc:
                logger.warning("Screening failed for %s: %s", f.get("symbol"), exc)

        # Sort by composite score descending
        results.sort(key=lambda x: x["composite_score"], reverse=True)
        return results[:top_n]

    # ------------------------------------------------------------------
    # Scoring functions
    # ------------------------------------------------------------------

    def _score_valuation(self, f: dict, sector: str) -> float:
        """Score based on P/E relative to sector, PEG ratio, P/B."""
        score = 50.0  # neutral starting point

        pe = f.get("pe_ratio", 0)
        sector_pe = SECTOR_AVG_PE.get(sector, 20.0)

        if pe and pe > 0:
            pe_ratio_vs_sector = pe / sector_pe
            if pe_ratio_vs_sector < 0.6:
                score += 30  # deeply undervalued
            elif pe_ratio_vs_sector < 0.8:
                score += 20
            elif pe_ratio_vs_sector < 1.0:
                score += 10
            elif pe_ratio_vs_sector > 1.5:
                score -= 20
            elif pe_ratio_vs_sector > 1.2:
                score -= 10

        peg = f.get("peg_ratio", 0)
        if peg and 0 < peg < 1.0:
            score += 15  # undervalued relative to growth
        elif peg and 1.0 <= peg < 1.5:
            score += 5
        elif peg and peg > 2.0:
            score -= 10

        pb = f.get("price_to_book", 0)
        if pb and 0 < pb < 1.0:
            score += 10
        elif pb and pb > 5.0:
            score -= 5

        return max(0, min(100, score))

    def _score_growth(self, f: dict) -> float:
        """Score based on revenue growth, earnings growth."""
        score = 50.0

        rev_growth = f.get("revenue_growth", 0)
        if rev_growth:
            if rev_growth > 0.25:
                score += 30
            elif rev_growth > 0.15:
                score += 20
            elif rev_growth > 0.05:
                score += 10
            elif rev_growth < -0.05:
                score -= 15
            elif rev_growth < -0.15:
                score -= 25

        earn_growth = f.get("earnings_growth", 0)
        if earn_growth:
            if earn_growth > 0.25:
                score += 15
            elif earn_growth > 0.10:
                score += 10
            elif earn_growth < -0.10:
                score -= 10

        # Check multi-year revenue trend from income statement
        income = f.get("income_statement", {})
        revenues = income.get("total_revenue", [])
        if len(revenues) >= 3:
            # Check if revenue is consistently growing
            growing = all(revenues[i] > revenues[i + 1] for i in range(len(revenues) - 1) if revenues[i + 1] > 0)
            if growing:
                score += 10

        return max(0, min(100, score))

    def _score_financial_health(self, f: dict) -> float:
        """Score based on debt/equity, current ratio, ROE."""
        score = 50.0

        dte = f.get("debt_to_equity", 0)
        if dte is not None:
            if dte < 30:
                score += 25
            elif dte < 80:
                score += 15
            elif dte < 150:
                score += 5
            elif dte > 200:
                score -= 20
            elif dte > 300:
                score -= 30

        cr = f.get("current_ratio", 0)
        if cr:
            if cr > 2.0:
                score += 15
            elif cr > 1.5:
                score += 10
            elif cr < 1.0:
                score -= 15

        roe = f.get("roe", 0)
        if roe:
            if roe > 0.20:
                score += 10
            elif roe > 0.10:
                score += 5
            elif roe < 0:
                score -= 15

        return max(0, min(100, score))

    def _score_dividend(self, f: dict) -> float:
        """Score based on dividend yield and payout sustainability."""
        div_yield = f.get("dividend_yield", 0) or 0
        payout = f.get("payout_ratio", 0) or 0

        if div_yield == 0:
            return 40.0  # neutral — growth stocks don't pay dividends

        score = 50.0

        if div_yield > 0.05:
            score += 15
        elif div_yield > 0.03:
            score += 20  # sweet spot
        elif div_yield > 0.01:
            score += 10

        # Payout sustainability
        if 0 < payout < 0.5:
            score += 15  # very sustainable
        elif 0.5 <= payout < 0.75:
            score += 5
        elif payout > 0.90:
            score -= 15  # unsustainable

        return max(0, min(100, score))

    def _score_momentum(self, f: dict) -> float:
        """Score based on price relative to moving averages and 52w range."""
        score = 50.0

        current = f.get("current_price", 0)
        avg_50 = f.get("50d_avg", 0)
        avg_200 = f.get("200d_avg", 0)
        high_52 = f.get("52w_high", 0)
        low_52 = f.get("52w_low", 0)

        if current and avg_50 and avg_200:
            # Price above both MAs = bullish
            if current > avg_50 > avg_200:
                score += 20
            elif current > avg_50:
                score += 10
            elif current < avg_50 < avg_200:
                score -= 15

        if current and high_52 and low_52 and high_52 > low_52:
            range_pct = (current - low_52) / (high_52 - low_52)
            if 0.4 <= range_pct <= 0.7:
                score += 10  # healthy mid-range
            elif range_pct > 0.95:
                score -= 10  # near ATH — risky entry
            elif range_pct < 0.2:
                score += 5  # possible value

        return max(0, min(100, score))

    def _score_analyst_consensus(self, f: dict) -> float:
        """Score based on analyst recommendations and price targets."""
        score = 50.0

        rec = f.get("recommendation", "none")
        if rec == "strong_buy":
            score += 25
        elif rec == "buy":
            score += 15
        elif rec == "hold":
            score += 0
        elif rec == "sell":
            score -= 20
        elif rec == "strong_sell":
            score -= 30

        current = f.get("current_price", 0)
        target = f.get("target_mean", 0)
        if current and target and current > 0:
            upside = (target - current) / current
            if upside > 0.30:
                score += 15
            elif upside > 0.15:
                score += 10
            elif upside < -0.10:
                score -= 10

        num_analysts = f.get("num_analysts", 0)
        if num_analysts and num_analysts >= 10:
            score += 5  # well-covered stock

        return max(0, min(100, score))

    # ------------------------------------------------------------------
    # Moat, risk, targets
    # ------------------------------------------------------------------

    def _rate_moat(self, f: dict, composite: float) -> str:
        """Rate competitive moat: weak/medium/strong."""
        moat_score = 0

        margin = f.get("profit_margin", 0) or 0
        if margin > 0.20:
            moat_score += 3
        elif margin > 0.10:
            moat_score += 2
        elif margin > 0.05:
            moat_score += 1

        roe = f.get("roe", 0) or 0
        if roe > 0.20:
            moat_score += 2
        elif roe > 0.10:
            moat_score += 1

        cap = f.get("market_cap", 0) or 0
        if cap > 100e9:
            moat_score += 2  # mega-cap = harder to disrupt
        elif cap > 10e9:
            moat_score += 1

        if composite > 70:
            moat_score += 1

        if moat_score >= 6:
            return "strong"
        elif moat_score >= 3:
            return "medium"
        return "weak"

    def _calculate_risk_rating(self, f: dict) -> int:
        """Calculate risk on scale 1-10 (1 = lowest risk)."""
        risk = 5  # neutral

        beta = f.get("beta", 1.0) or 1.0
        if beta > 1.5:
            risk += 2
        elif beta > 1.2:
            risk += 1
        elif beta < 0.6:
            risk -= 2
        elif beta < 0.8:
            risk -= 1

        dte = f.get("debt_to_equity", 0) or 0
        if dte > 200:
            risk += 2
        elif dte > 100:
            risk += 1
        elif dte < 30:
            risk -= 1

        cap = f.get("market_cap", 0) or 0
        if cap < 2e9:
            risk += 1
        elif cap > 100e9:
            risk -= 1

        return max(1, min(10, risk))

    def _calculate_price_targets(
        self, f: dict, current: float
    ) -> dict[str, float]:
        """Calculate 12-month price targets (optimistic/pessimistic)."""
        target_mean = f.get("target_mean", 0) or current
        target_high = f.get("target_high", 0) or target_mean * 1.3
        target_low = f.get("target_low", 0) or target_mean * 0.7

        return {
            "optimistic": round(target_high, 2),
            "base": round(target_mean, 2),
            "pessimistic": round(target_low, 2),
        }

    def _calculate_entry_zone(
        self, f: dict, current: float
    ) -> dict[str, float]:
        """Calculate ideal entry price zone."""
        avg_50 = f.get("50d_avg", current)
        avg_200 = f.get("200d_avg", current)

        # Entry zone: between current pullback support and current price
        support = min(avg_50, avg_200) * 0.97 if avg_50 and avg_200 else current * 0.95
        resistance = current * 1.02

        return {
            "lower": round(support, 2),
            "upper": round(resistance, 2),
        }

    def _calculate_stop_loss(self, f: dict, current: float) -> float:
        """Calculate suggested stop-loss price."""
        beta = f.get("beta", 1.0) or 1.0
        # Wider stop for higher-beta stocks
        stop_pct = 0.05 + (beta - 1.0) * 0.02
        stop_pct = max(0.03, min(0.10, stop_pct))
        return round(current * (1 - stop_pct), 2)

    def _get_recommendation(self, composite: float, risk: int) -> str:
        """Get actionable recommendation string."""
        if composite >= 75 and risk <= 5:
            return "STRONG BUY"
        elif composite >= 65:
            return "BUY"
        elif composite >= 50:
            return "HOLD"
        elif composite >= 35:
            return "REDUCE"
        else:
            return "SELL"
