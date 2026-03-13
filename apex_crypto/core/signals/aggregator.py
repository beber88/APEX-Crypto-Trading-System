"""Signal aggregation for the APEX Crypto Trading System.

Collects signals from all active strategies for each symbol, weights them
by rolling performance, applies contextual bonuses, and ranks the resulting
trade opportunities.

SIMONS UPGRADE: Integrates five new quantitative strategies with configurable
weights and pre-trade cost model filtering.
"""

from __future__ import annotations

import time
from typing import Any

from apex_crypto.core.logging import get_logger, log_with_data
from apex_crypto.core.strategies.base import BaseStrategy, SignalDirection, TradeSignal

logger = get_logger("signals.aggregator")

# Simons strategy signal weights (used when Simons strategies are active)
SIMONS_STRATEGY_WEIGHTS: dict[str, float] = {
    "stat_mean_reversion": 0.20,
    "quant_momentum": 0.20,
    "simons_trend": 0.15,
    "stat_arb": 0.25,
    "ml_signal": 0.20,
}


class SignalAggregator:
    """Aggregates, weights, and ranks trade signals from multiple strategies."""

    def __init__(self, config: dict, strategies: list[BaseStrategy]) -> None:
        self._config = config
        self._strategies = strategies
        self._strategy_map: dict[str, BaseStrategy] = {s.name: s for s in strategies}
        self._conflict_threshold: int = config.get("conflict_threshold", 40)
        self._half_position_score: int = config.get("half_position_score", 60)
        self._win_rate_window: int = config.get("win_rate_window_trades", 60)
        self._tf_alignment_bonus: int = config.get("timeframe_alignment_bonus", 20)
        self._sentiment_bonus: int = config.get("sentiment_alignment_bonus", 10)
        self._fg_extreme_bonus: int = config.get("fear_greed_extreme_bonus", 15)
        self._fg_extreme_low: int = config.get("fear_greed_extreme_low", 15)
        self._fg_extreme_high: int = config.get("fear_greed_extreme_high", 85)

        log_with_data(logger, "info", "SignalAggregator initialized", {
            "num_strategies": len(strategies),
            "conflict_threshold": self._conflict_threshold,
            "half_position_score": self._half_position_score,
            "win_rate_window": self._win_rate_window,
        })

    def _rolling_win_rate(self, strategy_name: str) -> float:
        strategy = self._strategy_map.get(strategy_name)
        if strategy is None:
            return 0.5
        history = strategy._trade_history
        if not history:
            return 0.5
        window = history[-self._win_rate_window:]
        wins = sum(1 for t in window if t["pnl"] > 0)
        return wins / len(window)

    def aggregate_signals(self, symbol: str, signals: list[TradeSignal]) -> dict[str, Any]:
        if not signals:
            return self._empty_aggregation(symbol)

        strategy_scores: dict[str, int] = {}
        weighted_sum: float = 0.0
        total_weight: float = 0.0

        for signal in signals:
            win_rate = self._rolling_win_rate(signal.strategy)
            weight = max(win_rate, 0.1)
            weighted_sum += signal.score * weight
            total_weight += weight
            strategy_scores[signal.strategy] = signal.score

        weighted_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        if weighted_score > 0:
            direction = SignalDirection.LONG.value
        elif weighted_score < 0:
            direction = SignalDirection.SHORT.value
        else:
            direction = SignalDirection.NEUTRAL.value

        if direction == SignalDirection.LONG.value:
            num_agreeing = sum(1 for s in signals if s.score > 0)
            num_disagreeing = sum(1 for s in signals if s.score < 0)
        elif direction == SignalDirection.SHORT.value:
            num_agreeing = sum(1 for s in signals if s.score < 0)
            num_disagreeing = sum(1 for s in signals if s.score > 0)
        else:
            num_agreeing = 0
            num_disagreeing = 0

        sorted_by_abs = sorted(signals, key=lambda s: abs(s.score), reverse=True)
        strongest_signal = sorted_by_abs[0].to_dict()
        weakest_signal = sorted_by_abs[-1].to_dict()

        scores = [s.score for s in signals]
        has_conflict = False
        if len(scores) >= 2:
            max_score = max(scores)
            min_score = min(scores)
            if abs(max_score - min_score) > self._conflict_threshold:
                has_conflict = True

        result: dict[str, Any] = {
            "symbol": symbol,
            "weighted_score": round(weighted_score, 2),
            "direction": direction,
            "num_agreeing": num_agreeing,
            "num_disagreeing": num_disagreeing,
            "strongest_signal": strongest_signal,
            "weakest_signal": weakest_signal,
            "strategy_scores": strategy_scores,
            "has_conflict": has_conflict,
            "timestamp": time.time(),
        }

        log_with_data(logger, "info", "Signals aggregated", {
            "symbol": symbol,
            "weighted_score": result["weighted_score"],
            "direction": direction,
            "num_agreeing": num_agreeing,
            "num_disagreeing": num_disagreeing,
            "has_conflict": has_conflict,
            "strategy_count": len(signals),
        })

        return result

    def apply_bonuses(self, aggregated: dict[str, Any], timeframe_alignment: dict[str, Any], sentiment: dict[str, Any], fear_greed: int) -> dict[str, Any]:
        bonus_breakdown: dict[str, int] = {}
        total_bonus: int = 0
        direction = aggregated["direction"]

        aligned_1h = timeframe_alignment.get("1h", False)
        aligned_4h = timeframe_alignment.get("4h", False)
        aligned_1d = timeframe_alignment.get("1d", False)

        if aligned_1h and aligned_4h and aligned_1d:
            bonus_breakdown["timeframe_alignment"] = self._tf_alignment_bonus
            total_bonus += self._tf_alignment_bonus

        sentiment_direction = sentiment.get("direction", "neutral")
        if sentiment_direction == direction and direction != SignalDirection.NEUTRAL.value:
            bonus_breakdown["sentiment_alignment"] = self._sentiment_bonus
            total_bonus += self._sentiment_bonus

        if fear_greed < self._fg_extreme_low and direction == SignalDirection.LONG.value:
            bonus_breakdown["fear_greed_extreme"] = self._fg_extreme_bonus
            total_bonus += self._fg_extreme_bonus
        elif fear_greed > self._fg_extreme_high and direction == SignalDirection.SHORT.value:
            bonus_breakdown["fear_greed_extreme"] = self._fg_extreme_bonus
            total_bonus += self._fg_extreme_bonus

        original_score = aggregated["weighted_score"]
        if direction == SignalDirection.SHORT.value:
            adjusted_score = original_score - total_bonus
        else:
            adjusted_score = original_score + total_bonus

        aggregated["weighted_score"] = round(adjusted_score, 2)
        aggregated["bonus_breakdown"] = bonus_breakdown
        aggregated["total_bonus"] = total_bonus

        log_with_data(logger, "info", "Bonuses applied", {
            "symbol": aggregated.get("symbol"),
            "original_score": original_score,
            "adjusted_score": aggregated["weighted_score"],
            "bonus_breakdown": bonus_breakdown,
            "fear_greed": fear_greed,
        })

        return aggregated

    def rank_opportunities(self, aggregated_signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
        qualified = [sig for sig in aggregated_signals if abs(sig.get("weighted_score", 0)) >= self._half_position_score]
        ranked = sorted(qualified, key=lambda s: abs(s.get("weighted_score", 0)), reverse=True)

        log_with_data(logger, "info", "Opportunities ranked", {
            "total_symbols": len(aggregated_signals),
            "qualified": len(ranked),
            "top_symbol": ranked[0]["symbol"] if ranked else None,
            "top_score": ranked[0]["weighted_score"] if ranked else None,
        })

        return ranked

    def aggregate_with_simons(self, symbol: str, base_signals: list[TradeSignal], simons_signals: list[TradeSignal], base_weight: float = 0.60, simons_weight: float = 0.40) -> dict[str, Any]:
        base_agg = self.aggregate_signals(symbol, base_signals)
        base_score = base_agg["weighted_score"]

        simons_weighted_sum: float = 0.0
        simons_total_weight: float = 0.0
        simons_scores: dict[str, int] = {}

        for signal in simons_signals:
            strategy_weight = SIMONS_STRATEGY_WEIGHTS.get(signal.strategy, 0.15)
            win_rate = self._rolling_win_rate(signal.strategy)
            combined_weight = strategy_weight * max(win_rate, 0.1)
            simons_weighted_sum += signal.score * combined_weight
            simons_total_weight += combined_weight
            simons_scores[signal.strategy] = signal.score

        simons_score = simons_weighted_sum / simons_total_weight if simons_total_weight > 0 else 0.0

        final_score = base_score * base_weight + simons_score * simons_weight

        if final_score > 0:
            direction = SignalDirection.LONG.value
        elif final_score < 0:
            direction = SignalDirection.SHORT.value
        else:
            direction = SignalDirection.NEUTRAL.value

        all_signals = base_signals + simons_signals
        all_scores = [s.score for s in all_signals]

        has_conflict = False
        if len(all_scores) >= 2:
            if abs(max(all_scores) - min(all_scores)) > self._conflict_threshold:
                has_conflict = True

        merged_scores = {**base_agg.get("strategy_scores", {}), **simons_scores}

        result: dict[str, Any] = {
            "symbol": symbol,
            "weighted_score": round(final_score, 2),
            "direction": direction,
            "num_agreeing": sum(1 for s in all_signals if (s.score > 0) == (final_score > 0)),
            "num_disagreeing": sum(1 for s in all_signals if (s.score > 0) != (final_score > 0) and s.score != 0),
            "strongest_signal": sorted(all_signals, key=lambda s: abs(s.score), reverse=True)[0].to_dict() if all_signals else {},
            "weakest_signal": sorted(all_signals, key=lambda s: abs(s.score))[0].to_dict() if all_signals else {},
            "strategy_scores": merged_scores,
            "has_conflict": has_conflict,
            "base_score": round(base_score, 2),
            "simons_score": round(simons_score, 2),
            "simons_strategies": simons_scores,
            "timestamp": time.time(),
        }

        log_with_data(logger, "info", "Simons-enhanced aggregation", {
            "symbol": symbol,
            "base_score": round(base_score, 2),
            "simons_score": round(simons_score, 2),
            "final_score": round(final_score, 2),
            "direction": direction,
            "num_strategies": len(all_signals),
        })

        return result

    def _empty_aggregation(self, symbol: str) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "weighted_score": 0.0,
            "direction": SignalDirection.NEUTRAL.value,
            "num_agreeing": 0,
            "num_disagreeing": 0,
            "strongest_signal": {},
            "weakest_signal": {},
            "strategy_scores": {},
            "has_conflict": False,
            "timestamp": time.time(),
        }
