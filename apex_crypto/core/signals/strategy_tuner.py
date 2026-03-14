"""Strategy performance tuner for the APEX Crypto Trading System.

Monitors rolling strategy performance and adjusts strategy weights
and activation based on recent trade results.
"""

from __future__ import annotations

import time
from typing import Any

from apex_crypto.core.logging import get_logger, log_with_data
from apex_crypto.core.strategies.base import BaseStrategy

logger = get_logger("signals.strategy_tuner")


class StrategyTuner:
    """Analyzes strategy performance and adjusts weights dynamically.

    Tracks win rates, profit factors, and R-multiples per strategy
    over a rolling window to determine which strategies should remain
    active and with what weight.
    """

    def __init__(self, config: dict, strategies: list[BaseStrategy]) -> None:
        self._config = config
        self._strategies = strategies
        self._strategy_map: dict[str, BaseStrategy] = {s.name: s for s in strategies}

        self._min_trades_for_analysis: int = config.get("min_trades_for_analysis", 0)
        self._min_win_rate: float = config.get("min_win_rate", 0.30)
        self._min_profit_factor: float = config.get("min_profit_factor", 0.8)
        self._analysis_window: int = config.get("analysis_window_trades", 50)
        self._last_analysis_time: float = 0.0
        self._analysis_interval: int = config.get("analysis_interval_seconds", 3600)

        self._strategy_weights: dict[str, float] = {s.name: 1.0 for s in strategies}
        self._disabled_strategies: set[str] = set()

        log_with_data(logger, "info", "StrategyTuner initialized", {
            "num_strategies": len(strategies),
            "min_trades_for_analysis": self._min_trades_for_analysis,
            "min_win_rate": self._min_win_rate,
            "analysis_window": self._analysis_window,
        })

    def analyze_performance(self) -> dict[str, Any]:
        """Analyze all strategy performance and adjust weights.

        Returns:
            Summary of analysis results.
        """
        now = time.time()
        if now - self._last_analysis_time < self._analysis_interval:
            return {"skipped": True, "reason": "Too soon since last analysis"}

        self._last_analysis_time = now
        results: dict[str, Any] = {}
        active_count = 0

        for strategy in self._strategies:
            name = strategy.name
            trades = len(strategy._trade_history)

            if trades < self._min_trades_for_analysis:
                log_with_data(logger, "debug", "Insufficient trades for analysis", {
                    "strategy": name,
                    "trades": trades,
                    "minimum": self._min_trades_for_analysis,
                })
                # Still count as active — don't disable strategies with no history
                active_count += 1
                self._strategy_weights[name] = 1.0
                results[name] = {
                    "status": "active",
                    "trades": trades,
                    "reason": "insufficient_data",
                }
                continue

            win_rate = strategy.win_rate
            profit_factor = strategy.profit_factor
            avg_r = strategy.avg_r_multiple

            weight = 1.0
            status = "active"

            if win_rate < self._min_win_rate and trades >= 10:
                weight = 0.5
                status = "reduced"

            if profit_factor < self._min_profit_factor and trades >= 10:
                weight *= 0.5
                status = "reduced"

            if avg_r > 1.0:
                weight *= 1.2

            self._strategy_weights[name] = max(weight, 0.1)
            active_count += 1

            results[name] = {
                "status": status,
                "trades": trades,
                "win_rate": round(win_rate, 3),
                "profit_factor": round(profit_factor, 3),
                "avg_r": round(avg_r, 3),
                "weight": round(self._strategy_weights[name], 3),
            }

        log_with_data(logger, "info", "Performance analysis complete", {
            "strategies_analyzed": len(results),
            "active_strategies": active_count,
        })

        return {
            "strategies": results,
            "active_strategies": active_count,
            "timestamp": now,
        }

    def get_weight(self, strategy_name: str) -> float:
        """Get the current weight for a strategy."""
        return self._strategy_weights.get(strategy_name, 1.0)

    def is_enabled(self, strategy_name: str) -> bool:
        """Check if a strategy is currently enabled by the tuner."""
        return strategy_name not in self._disabled_strategies
