"""JPMorgan-style Earnings Analyzer.

Analyzes earnings reports and generates pre-earnings trade signals:
- Last 4 quarters earnings vs expectations (beat/miss history)
- Consensus EPS and revenue forecasts
- Revenue breakdown by segments
- Management guidance summary
- Historical price reaction to earnings
- Bull/Bear case scenarios
- Recommendation: buy before, sell before, or hold

Inspired by JPMorgan Chase equity research methodology.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

from apex_crypto.core.logging import get_logger, log_with_data

logger = get_logger("stocks.earnings")


class EarningsAnalyzer:
    """Analyzes stock earnings and generates pre-earnings insights.

    Args:
        config: Optional configuration dict.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._min_surprise_for_signal: float = cfg.get("min_surprise_pct", 5.0)
        self._lookback_quarters: int = cfg.get("lookback_quarters", 8)

    def analyze(
        self,
        fundamentals: dict[str, Any],
        ohlcv_daily: Optional[pd.DataFrame] = None,
    ) -> dict[str, Any]:
        """Run full earnings analysis.

        Args:
            fundamentals: Dict from StockBroker.fetch_fundamentals().
            ohlcv_daily: Daily OHLCV DataFrame for historical price reaction.

        Returns:
            Comprehensive earnings analysis report.
        """
        symbol = fundamentals.get("symbol", "?")

        # Earnings beat/miss history
        history = self._analyze_earnings_history(fundamentals)

        # Estimate next quarter
        estimates = self._get_estimates(fundamentals)

        # Historical price reactions around earnings
        price_reactions = self._analyze_price_reactions(
            fundamentals, ohlcv_daily
        )

        # Bull/Bear case
        bull_bear = self._build_bull_bear_case(fundamentals, history)

        # Overall earnings quality score
        quality_score = self._score_earnings_quality(
            history, fundamentals
        )

        # Pre-earnings recommendation
        recommendation = self._recommend(
            history, price_reactions, quality_score, fundamentals
        )

        result = {
            "symbol": symbol,
            "next_earnings_date": fundamentals.get("next_earnings_date"),
            "earnings_history": history,
            "estimates": estimates,
            "price_reactions": price_reactions,
            "bull_case": bull_bear["bull"],
            "bear_case": bull_bear["bear"],
            "quality_score": round(quality_score, 1),
            "recommendation": recommendation,
        }

        log_with_data(logger, "info", "Earnings analysis complete", {
            "symbol": symbol,
            "quality_score": result["quality_score"],
            "recommendation": recommendation["action"],
            "beat_rate": history.get("beat_rate", 0),
        })

        return result

    # ------------------------------------------------------------------
    # Earnings history analysis
    # ------------------------------------------------------------------

    def _analyze_earnings_history(self, f: dict) -> dict[str, Any]:
        """Analyze beat/miss history from earnings data."""
        earnings = f.get("earnings_history", [])

        beats = 0
        misses = 0
        in_line = 0
        surprises = []

        for entry in earnings[:self._lookback_quarters]:
            estimate = entry.get("eps_estimate")
            actual = entry.get("eps_actual")
            surprise = entry.get("surprise_pct")

            if estimate is not None and actual is not None:
                if actual > estimate:
                    beats += 1
                elif actual < estimate:
                    misses += 1
                else:
                    in_line += 1

            if surprise is not None:
                surprises.append(surprise)

        total = beats + misses + in_line
        beat_rate = beats / total if total > 0 else 0.5
        avg_surprise = np.mean(surprises) if surprises else 0

        # Trend — are surprises improving or declining?
        surprise_trend = "stable"
        if len(surprises) >= 4:
            recent = np.mean(surprises[:2])
            older = np.mean(surprises[2:4])
            if recent > older + 2:
                surprise_trend = "improving"
            elif recent < older - 2:
                surprise_trend = "declining"

        return {
            "beats": beats,
            "misses": misses,
            "in_line": in_line,
            "total_quarters": total,
            "beat_rate": round(beat_rate, 2),
            "avg_surprise_pct": round(avg_surprise, 2),
            "surprise_trend": surprise_trend,
            "details": earnings[:self._lookback_quarters],
        }

    def _get_estimates(self, f: dict) -> dict[str, Any]:
        """Get consensus estimates for next quarter."""
        return {
            "forward_eps": f.get("forward_eps", 0),
            "forward_pe": f.get("forward_pe", 0),
            "revenue_growth": f.get("revenue_growth", 0),
            "earnings_growth": f.get("earnings_growth", 0),
            "num_analysts": f.get("num_analysts", 0),
            "target_mean": f.get("target_mean", 0),
            "target_high": f.get("target_high", 0),
            "target_low": f.get("target_low", 0),
        }

    # ------------------------------------------------------------------
    # Price reaction analysis
    # ------------------------------------------------------------------

    def _analyze_price_reactions(
        self,
        f: dict,
        ohlcv: Optional[pd.DataFrame],
    ) -> dict[str, Any]:
        """Analyze how the stock price reacted to past earnings."""
        if ohlcv is None or ohlcv.empty:
            return {"available": False}

        earnings = f.get("earnings_history", [])
        reactions = []

        for entry in earnings[:4]:
            date_str = entry.get("date")
            if not date_str:
                continue

            try:
                # Parse earnings date
                earnings_date = pd.Timestamp(date_str)
                if earnings_date.tzinfo:
                    earnings_date = earnings_date.tz_localize(None)

                # Find closest trading day
                idx = ohlcv.index.get_indexer([earnings_date], method="nearest")[0]
                if idx < 1 or idx >= len(ohlcv) - 1:
                    continue

                # Price change on earnings day
                pre_close = float(ohlcv.iloc[idx - 1]["close"])
                post_close = float(ohlcv.iloc[idx]["close"])
                day_after = float(ohlcv.iloc[idx + 1]["close"])

                day_change_pct = ((post_close - pre_close) / pre_close) * 100
                next_day_change = ((day_after - post_close) / post_close) * 100

                reactions.append({
                    "date": str(earnings_date.date()),
                    "surprise_pct": entry.get("surprise_pct"),
                    "day_change_pct": round(day_change_pct, 2),
                    "next_day_change_pct": round(next_day_change, 2),
                    "direction": "up" if day_change_pct > 0 else "down",
                })
            except Exception:
                continue

        # Calculate average reaction
        avg_move = np.mean([abs(r["day_change_pct"]) for r in reactions]) if reactions else 0
        positive_reactions = sum(1 for r in reactions if r["day_change_pct"] > 0)

        return {
            "available": True,
            "reactions": reactions,
            "avg_absolute_move_pct": round(avg_move, 2),
            "positive_reaction_rate": positive_reactions / len(reactions) if reactions else 0.5,
            "expected_move_range": {
                "low": round(-avg_move * 1.2, 2),
                "high": round(avg_move * 1.2, 2),
            },
        }

    # ------------------------------------------------------------------
    # Bull/Bear case
    # ------------------------------------------------------------------

    def _build_bull_bear_case(
        self, f: dict, history: dict
    ) -> dict[str, dict]:
        """Build bull and bear case scenarios."""
        current_price = f.get("current_price", 0)
        target_high = f.get("target_high", 0)
        target_low = f.get("target_low", 0)

        bull = {
            "price_target": target_high or current_price * 1.25,
            "upside_pct": round(((target_high or current_price * 1.25) - current_price) / current_price * 100, 1) if current_price else 0,
            "catalysts": [],
        }

        bear = {
            "price_target": target_low or current_price * 0.80,
            "downside_pct": round((current_price - (target_low or current_price * 0.80)) / current_price * 100, 1) if current_price else 0,
            "risks": [],
        }

        # Bull catalysts
        if history.get("beat_rate", 0) > 0.75:
            bull["catalysts"].append("Strong earnings beat track record")
        if f.get("revenue_growth", 0) and f["revenue_growth"] > 0.15:
            bull["catalysts"].append(f"Strong revenue growth ({f['revenue_growth']:.0%})")
        if f.get("recommendation") in ("buy", "strong_buy"):
            bull["catalysts"].append("Positive analyst consensus")
        if not bull["catalysts"]:
            bull["catalysts"].append("Market momentum and sector tailwinds")

        # Bear risks
        if history.get("beat_rate", 0) < 0.50:
            bear["risks"].append("Poor earnings execution history")
        if f.get("debt_to_equity", 0) and f["debt_to_equity"] > 150:
            bear["risks"].append(f"High leverage (D/E: {f['debt_to_equity']:.0f})")
        if f.get("pe_ratio", 0) and f["pe_ratio"] > 35:
            bear["risks"].append(f"Rich valuation (P/E: {f['pe_ratio']:.1f})")
        if not bear["risks"]:
            bear["risks"].append("Macroeconomic slowdown could impact earnings")

        return {"bull": bull, "bear": bear}

    # ------------------------------------------------------------------
    # Scoring and recommendation
    # ------------------------------------------------------------------

    def _score_earnings_quality(self, history: dict, f: dict) -> float:
        """Score overall earnings quality (0-100)."""
        score = 50.0

        beat_rate = history.get("beat_rate", 0.5)
        if beat_rate > 0.80:
            score += 20
        elif beat_rate > 0.60:
            score += 10
        elif beat_rate < 0.40:
            score -= 15

        avg_surprise = history.get("avg_surprise_pct", 0)
        if avg_surprise > 10:
            score += 15
        elif avg_surprise > 5:
            score += 10
        elif avg_surprise < -5:
            score -= 15

        trend = history.get("surprise_trend", "stable")
        if trend == "improving":
            score += 10
        elif trend == "declining":
            score -= 10

        margin = f.get("profit_margin", 0) or 0
        if margin > 0.15:
            score += 5
        elif margin < 0:
            score -= 10

        return max(0, min(100, score))

    def _recommend(
        self,
        history: dict,
        price_reactions: dict,
        quality_score: float,
        f: dict,
    ) -> dict[str, Any]:
        """Generate pre-earnings recommendation."""
        action = "HOLD"
        confidence = 0.5
        reasoning = []

        beat_rate = history.get("beat_rate", 0.5)
        avg_surprise = history.get("avg_surprise_pct", 0)
        trend = history.get("surprise_trend", "stable")

        # Strong beat history + improving trend = buy before
        if beat_rate >= 0.75 and avg_surprise > 5 and trend != "declining":
            action = "BUY BEFORE EARNINGS"
            confidence = 0.7
            reasoning.append(f"Beat rate: {beat_rate:.0%} with avg surprise +{avg_surprise:.1f}%")

        # Weak beat history = sell before to reduce risk
        elif beat_rate < 0.50 or (avg_surprise < -3 and trend == "declining"):
            action = "REDUCE BEFORE EARNINGS"
            confidence = 0.65
            reasoning.append(f"Poor beat rate: {beat_rate:.0%}, trend: {trend}")

        # Mixed signals
        else:
            action = "HOLD THROUGH EARNINGS"
            confidence = 0.5
            reasoning.append("Mixed earnings signals — no clear edge")

        # Adjust for price reaction patterns
        if price_reactions.get("available"):
            pos_rate = price_reactions.get("positive_reaction_rate", 0.5)
            avg_move = price_reactions.get("avg_absolute_move_pct", 0)

            if pos_rate > 0.75 and avg_move > 3:
                reasoning.append(f"Historical: {pos_rate:.0%} positive reactions, avg move ±{avg_move:.1f}%")
                if action != "BUY BEFORE EARNINGS":
                    confidence += 0.1

        # Quality score adjustment
        if quality_score >= 70:
            reasoning.append(f"Earnings quality score: {quality_score:.0f}/100")
        elif quality_score < 40:
            reasoning.append(f"Low earnings quality: {quality_score:.0f}/100")

        return {
            "action": action,
            "confidence": round(min(confidence, 0.95), 2),
            "reasoning": reasoning,
        }
