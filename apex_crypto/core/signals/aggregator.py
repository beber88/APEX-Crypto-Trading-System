"""Signal aggregation for the APEX Crypto Trading System.

Collects signals from all active strategies for each symbol, weights them
by rolling performance, applies contextual bonuses, and ranks the resulting
trade opportunities.
"""

from __future__ import annotations

import time
from typing import Any

from apex_crypto.core.logging import get_logger, log_with_data
from apex_crypto.core.strategies.base import BaseStrategy, SignalDirection, TradeSignal

logger = get_logger("signals.aggregator")


class SignalAggregator:
    """Aggregates, weights, and ranks trade signals from multiple strategies.

    Each signal is weighted by the originating strategy's rolling win rate
    over the most recent N trades (configured via ``win_rate_window_trades``).
    Additional bonuses are layered on for timeframe alignment, sentiment
    confirmation, and extreme Fear & Greed readings.

    Args:
        config: The ``signals`` section from ``config.yaml``.
        strategies: All instantiated strategy objects.
    """

    def __init__(self, config: dict, strategies: list[BaseStrategy]) -> None:
        self._config = config
        self._strategies = strategies

        # Build a fast name -> strategy lookup for win-rate retrieval.
        self._strategy_map: dict[str, BaseStrategy] = {
            s.name: s for s in strategies
        }

        # Pull thresholds from config with sensible defaults.
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

    # ------------------------------------------------------------------
    # Rolling win rate helper
    # ------------------------------------------------------------------

    def _rolling_win_rate(self, strategy_name: str) -> float:
        """Compute the rolling win rate over the last N trades for a strategy.

        If the strategy has fewer than ``win_rate_window_trades`` recorded
        trades the full history is used.  Strategies with zero history
        default to 0.5 (neutral weighting).

        Args:
            strategy_name: Name of the strategy.

        Returns:
            Win rate between 0.0 and 1.0.
        """
        strategy = self._strategy_map.get(strategy_name)
        if strategy is None:
            return 0.5

        history = strategy._trade_history
        if not history:
            return 0.5

        window = history[-self._win_rate_window:]
        wins = sum(1 for t in window if t["pnl"] > 0)
        return wins / len(window)

    # ------------------------------------------------------------------
    # Core aggregation
    # ------------------------------------------------------------------

    def aggregate_signals(
        self,
        symbol: str,
        signals: list[TradeSignal],
    ) -> dict[str, Any]:
        """Aggregate signals from all active strategies for a single symbol.

        Each signal's absolute score is weighted by the originating strategy's
        rolling win rate.  The weighted average determines overall direction
        and conviction.

        Args:
            symbol: Trading pair symbol (e.g. ``"BTC/USDT"``).
            signals: All ``TradeSignal`` objects produced for *symbol*
                in the current evaluation cycle.

        Returns:
            Aggregation result dictionary::

                {
                    "symbol": str,
                    "weighted_score": float,
                    "direction": "long" | "short" | "neutral",
                    "num_agreeing": int,
                    "num_disagreeing": int,
                    "strongest_signal": dict,
                    "weakest_signal": dict,
                    "strategy_scores": {strategy_name: score, ...},
                    "has_conflict": bool,
                    "timestamp": float,
                }
        """
        if not signals:
            return self._empty_aggregation(symbol)

        # Gather per-strategy scores and win-rate weights.
        strategy_scores: dict[str, int] = {}
        weighted_sum: float = 0.0
        total_weight: float = 0.0

        for signal in signals:
            win_rate = self._rolling_win_rate(signal.strategy)
            # Ensure a minimum weight so new strategies still contribute.
            weight = max(win_rate, 0.1)
            weighted_sum += signal.score * weight
            total_weight += weight
            strategy_scores[signal.strategy] = signal.score

        weighted_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Determine consensus direction from the weighted score.
        if weighted_score > 0:
            direction = SignalDirection.LONG.value
        elif weighted_score < 0:
            direction = SignalDirection.SHORT.value
        else:
            direction = SignalDirection.NEUTRAL.value

        # Count agreement / disagreement relative to the consensus.
        if direction == SignalDirection.LONG.value:
            num_agreeing = sum(1 for s in signals if s.score > 0)
            num_disagreeing = sum(1 for s in signals if s.score < 0)
        elif direction == SignalDirection.SHORT.value:
            num_agreeing = sum(1 for s in signals if s.score < 0)
            num_disagreeing = sum(1 for s in signals if s.score > 0)
        else:
            num_agreeing = 0
            num_disagreeing = 0

        # Identify strongest and weakest signals by absolute score.
        sorted_by_abs = sorted(signals, key=lambda s: abs(s.score), reverse=True)
        strongest_signal = sorted_by_abs[0].to_dict()
        weakest_signal = sorted_by_abs[-1].to_dict()

        # Conflict detection: any two strategies disagree by more than
        # conflict_threshold points.
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

    # ------------------------------------------------------------------
    # Bonus application
    # ------------------------------------------------------------------

    def apply_bonuses(
        self,
        aggregated: dict[str, Any],
        timeframe_alignment: dict[str, Any],
        sentiment: dict[str, Any],
        fear_greed: int,
    ) -> dict[str, Any]:
        """Layer contextual bonuses on top of the weighted aggregation score.

        Bonuses are additive and each is individually capped.  The bonus
        breakdown is attached to the result for audit trail purposes.

        Args:
            aggregated: Output of :meth:`aggregate_signals`.
            timeframe_alignment: Dict with boolean keys for each timeframe
                indicating whether that timeframe's bias aligns with the
                signal direction.  Expected keys: ``"1h"``, ``"4h"``,
                ``"1d"`` (values are ``True`` / ``False``).
            sentiment: FinBERT sentiment dict with at minimum a
                ``"direction"`` key (``"long"`` / ``"short"`` /
                ``"neutral"``) and an optional ``"confidence"`` float.
            fear_greed: Current Fear & Greed index value (0-100).

        Returns:
            Updated *aggregated* dict with ``"bonus_breakdown"`` added and
            ``"weighted_score"`` adjusted.
        """
        bonus_breakdown: dict[str, int] = {}
        total_bonus: int = 0
        direction = aggregated["direction"]

        # --- Timeframe alignment bonus ---
        aligned_1h = timeframe_alignment.get("1h", False)
        aligned_4h = timeframe_alignment.get("4h", False)
        aligned_1d = timeframe_alignment.get("1d", False)

        if aligned_1h and aligned_4h and aligned_1d:
            bonus_breakdown["timeframe_alignment"] = self._tf_alignment_bonus
            total_bonus += self._tf_alignment_bonus

        # --- Sentiment (FinBERT) bonus ---
        sentiment_direction = sentiment.get("direction", "neutral")
        if sentiment_direction == direction and direction != SignalDirection.NEUTRAL.value:
            bonus_breakdown["sentiment_alignment"] = self._sentiment_bonus
            total_bonus += self._sentiment_bonus

        # --- Fear & Greed extreme bonus ---
        if fear_greed < self._fg_extreme_low and direction == SignalDirection.LONG.value:
            bonus_breakdown["fear_greed_extreme"] = self._fg_extreme_bonus
            total_bonus += self._fg_extreme_bonus
        elif fear_greed > self._fg_extreme_high and direction == SignalDirection.SHORT.value:
            bonus_breakdown["fear_greed_extreme"] = self._fg_extreme_bonus
            total_bonus += self._fg_extreme_bonus

        # Apply bonuses (preserve sign: bonuses push the score further
        # in the consensus direction).
        original_score = aggregated["weighted_score"]
        if direction == SignalDirection.SHORT.value:
            # Short scores are negative; subtract bonus to push further negative.
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

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def rank_opportunities(
        self,
        aggregated_signals: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Rank and filter trade opportunities by final weighted score.

        Symbols whose absolute weighted score falls below
        ``half_position_score`` are discarded.  The remaining
        opportunities are sorted from highest absolute score to lowest.

        Args:
            aggregated_signals: List of aggregation dicts (output of
                :meth:`aggregate_signals`, optionally with bonuses applied).

        Returns:
            Filtered and sorted list (best opportunity first).
        """
        qualified: list[dict[str, Any]] = [
            sig for sig in aggregated_signals
            if abs(sig.get("weighted_score", 0)) >= self._half_position_score
        ]

        ranked = sorted(
            qualified,
            key=lambda s: abs(s.get("weighted_score", 0)),
            reverse=True,
        )

        log_with_data(logger, "info", "Opportunities ranked", {
            "total_symbols": len(aggregated_signals),
            "qualified": len(ranked),
            "top_symbol": ranked[0]["symbol"] if ranked else None,
            "top_score": ranked[0]["weighted_score"] if ranked else None,
        })

        return ranked

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _empty_aggregation(self, symbol: str) -> dict[str, Any]:
        """Return a neutral aggregation result when no signals are present.

        Args:
            symbol: Trading pair symbol.

        Returns:
            Aggregation dict with zero score and neutral direction.
        """
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
