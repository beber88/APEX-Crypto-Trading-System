"""Strategy auto-tuner for APEX Crypto Trading System.

Analyzes historical trade performance by strategy, time of day, regime,
and asset to automatically adjust strategy weights, trading hours,
and per-asset sizing.

This is the single highest-impact optimization: eliminating losing
strategy × regime × time combinations.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Optional

from apex_crypto.core.logging import get_logger, log_with_data

logger = get_logger("signals.strategy_tuner")


class StrategyTuner:
    """Automatically adjusts strategy weights based on rolling performance.

    Runs periodic analysis on trade history and adjusts:
    1. Strategy weights (kill losers, boost winners)
    2. Trading hours (only trade during profitable hours)
    3. Strategy-regime matrix (disable losing combinations)
    4. Per-asset sizing (reduce size on losing assets)
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._min_trades_for_analysis: int = config.get("min_trades_for_analysis", 20)
        self._analysis_interval_hours: int = config.get("analysis_interval_hours", 24)
        self._last_analysis_time: float = 0.0

        # Strategy weight adjustments (1.0 = normal, 0.0 = disabled, 2.0 = doubled)
        self._strategy_weights: dict[str, float] = {}

        # Profitable trading hours (UTC)
        self._profitable_hours: list[int] = list(range(24))  # All hours initially

        # Strategy-regime matrix: strategy → set of profitable regimes
        self._regime_matrix: dict[str, set[str]] = {}

        # Per-asset size multiplier
        self._asset_multipliers: dict[str, float] = {}

        # Analysis results
        self._last_analysis: dict[str, Any] = {}

        log_with_data(logger, "info", "StrategyTuner initialized", {
            "min_trades": self._min_trades_for_analysis,
            "analysis_interval_hours": self._analysis_interval_hours,
        })

    def analyze_performance(self, trade_history: list[dict[str, Any]]) -> dict[str, Any]:
        """Run full performance analysis on trade history.

        Args:
            trade_history: List of closed trade dicts with keys:
                strategy, symbol, direction, entry_price, exit_price,
                pnl_usdt, pnl_pct, r_multiple, entry_time, exit_time,
                regime, duration_seconds

        Returns:
            Complete analysis results.
        """
        if len(trade_history) < self._min_trades_for_analysis:
            log_with_data(logger, "info", "Insufficient trades for analysis", {
                "trades": len(trade_history),
                "minimum": self._min_trades_for_analysis,
            })
            return {"status": "insufficient_data", "trades": len(trade_history)}

        analysis = {
            "timestamp": time.time(),
            "total_trades": len(trade_history),
            "strategy_analysis": self._analyze_strategies(trade_history),
            "time_of_day_analysis": self._analyze_time_of_day(trade_history),
            "regime_matrix": self._analyze_regime_matrix(trade_history),
            "asset_analysis": self._analyze_assets(trade_history),
        }

        # Apply tuning based on analysis
        self._apply_strategy_weights(analysis["strategy_analysis"])
        self._apply_trading_hours(analysis["time_of_day_analysis"])
        self._apply_regime_matrix(analysis["regime_matrix"])
        self._apply_asset_multipliers(analysis["asset_analysis"])

        self._last_analysis = analysis
        self._last_analysis_time = time.time()

        log_with_data(logger, "info", "Performance analysis complete", {
            "strategies_analyzed": len(analysis["strategy_analysis"]),
            "active_strategies": sum(1 for w in self._strategy_weights.values() if w > 0),
            "profitable_hours": len(self._profitable_hours),
        })

        return analysis

    def _analyze_strategies(self, trades: list[dict]) -> list[dict]:
        """Analyze performance by strategy."""
        strategy_stats: dict[str, dict] = {}

        for trade in trades:
            strat = trade.get("strategy", "unknown")
            if strat not in strategy_stats:
                strategy_stats[strat] = {
                    "strategy": strat,
                    "trades": 0, "wins": 0, "losses": 0,
                    "total_pnl": 0.0, "pnl_list": [],
                    "best_trade": 0.0, "worst_trade": 0.0,
                    "total_r": 0.0,
                }
            s = strategy_stats[strat]
            pnl = trade.get("pnl_usdt", 0.0)
            r_mult = trade.get("r_multiple", 0.0)
            s["trades"] += 1
            s["total_pnl"] += pnl
            s["pnl_list"].append(pnl)
            s["total_r"] += r_mult
            if pnl > 0:
                s["wins"] += 1
            else:
                s["losses"] += 1
            s["best_trade"] = max(s["best_trade"], pnl)
            s["worst_trade"] = min(s["worst_trade"], pnl)

        results = []
        for strat, s in strategy_stats.items():
            win_rate = s["wins"] / s["trades"] if s["trades"] > 0 else 0
            avg_pnl = s["total_pnl"] / s["trades"] if s["trades"] > 0 else 0
            avg_r = s["total_r"] / s["trades"] if s["trades"] > 0 else 0

            # Determine action
            if win_rate < 0.40 and s["total_pnl"] < 0:
                action = "DISABLE"
                weight = 0.0
            elif win_rate < 0.45 and s["total_pnl"] > 0:
                action = "REDUCE_50"
                weight = 0.5
            elif win_rate > 0.60:
                action = "DOUBLE"
                weight = 2.0
            elif win_rate > 0.55 and s["total_pnl"] > 0:
                action = "BOOST_50"
                weight = 1.5
            else:
                action = "KEEP"
                weight = 1.0

            results.append({
                "strategy": strat,
                "trades": s["trades"],
                "wins": s["wins"],
                "losses": s["losses"],
                "win_rate": round(win_rate, 4),
                "avg_pnl": round(avg_pnl, 2),
                "total_pnl": round(s["total_pnl"], 2),
                "avg_r": round(avg_r, 3),
                "best_trade": round(s["best_trade"], 2),
                "worst_trade": round(s["worst_trade"], 2),
                "action": action,
                "weight": weight,
            })

        results.sort(key=lambda x: x["total_pnl"], reverse=True)
        return results

    def _analyze_time_of_day(self, trades: list[dict]) -> list[dict]:
        """Analyze performance by hour of day (UTC)."""
        hour_stats: dict[int, dict] = {h: {"hour": h, "trades": 0, "total_pnl": 0.0, "wins": 0} for h in range(24)}

        for trade in trades:
            entry_time = trade.get("entry_time")
            if entry_time is None:
                continue
            if isinstance(entry_time, str):
                try:
                    dt = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    continue
            elif isinstance(entry_time, (int, float)):
                dt = datetime.fromtimestamp(entry_time, tz=timezone.utc)
            else:
                continue

            hour = dt.hour
            pnl = trade.get("pnl_usdt", 0.0)
            hour_stats[hour]["trades"] += 1
            hour_stats[hour]["total_pnl"] += pnl
            if pnl > 0:
                hour_stats[hour]["wins"] += 1

        results = []
        for h, s in hour_stats.items():
            avg_pnl = s["total_pnl"] / s["trades"] if s["trades"] > 0 else 0
            win_rate = s["wins"] / s["trades"] if s["trades"] > 0 else 0
            results.append({
                "hour_utc": h,
                "trades": s["trades"],
                "total_pnl": round(s["total_pnl"], 2),
                "avg_pnl": round(avg_pnl, 2),
                "win_rate": round(win_rate, 4),
            })

        results.sort(key=lambda x: x["avg_pnl"], reverse=True)
        return results

    def _analyze_regime_matrix(self, trades: list[dict]) -> list[dict]:
        """Analyze performance by strategy x regime combination."""
        matrix: dict[str, dict[str, dict]] = {}

        for trade in trades:
            strat = trade.get("strategy", "unknown")
            regime = trade.get("regime", "UNKNOWN")
            pnl = trade.get("pnl_usdt", 0.0)

            if strat not in matrix:
                matrix[strat] = {}
            if regime not in matrix[strat]:
                matrix[strat][regime] = {"trades": 0, "total_pnl": 0.0, "wins": 0}

            matrix[strat][regime]["trades"] += 1
            matrix[strat][regime]["total_pnl"] += pnl
            if pnl > 0:
                matrix[strat][regime]["wins"] += 1

        results = []
        for strat, regimes in matrix.items():
            for regime, s in regimes.items():
                avg_pnl = s["total_pnl"] / s["trades"] if s["trades"] > 0 else 0
                win_rate = s["wins"] / s["trades"] if s["trades"] > 0 else 0
                profitable = avg_pnl > 0
                results.append({
                    "strategy": strat,
                    "regime": regime,
                    "trades": s["trades"],
                    "total_pnl": round(s["total_pnl"], 2),
                    "avg_pnl": round(avg_pnl, 2),
                    "win_rate": round(win_rate, 4),
                    "profitable": profitable,
                    "action": "ACTIVE" if profitable else "DISABLE_IN_REGIME",
                })

        results.sort(key=lambda x: x["avg_pnl"], reverse=True)
        return results

    def _analyze_assets(self, trades: list[dict]) -> list[dict]:
        """Analyze performance by asset."""
        asset_stats: dict[str, dict] = {}

        for trade in trades:
            symbol = trade.get("symbol", "UNKNOWN")
            pnl = trade.get("pnl_usdt", 0.0)

            if symbol not in asset_stats:
                asset_stats[symbol] = {"symbol": symbol, "trades": 0, "total_pnl": 0.0, "wins": 0}

            asset_stats[symbol]["trades"] += 1
            asset_stats[symbol]["total_pnl"] += pnl
            if pnl > 0:
                asset_stats[symbol]["wins"] += 1

        results = []
        for sym, s in asset_stats.items():
            avg_pnl = s["total_pnl"] / s["trades"] if s["trades"] > 0 else 0
            win_rate = s["wins"] / s["trades"] if s["trades"] > 0 else 0
            results.append({
                "symbol": sym,
                "trades": s["trades"],
                "total_pnl": round(s["total_pnl"], 2),
                "avg_pnl": round(avg_pnl, 2),
                "win_rate": round(win_rate, 4),
            })

        results.sort(key=lambda x: x["total_pnl"], reverse=True)
        return results

    def _apply_strategy_weights(self, strategy_analysis: list[dict]) -> None:
        """Apply strategy weight adjustments from analysis."""
        for sa in strategy_analysis:
            self._strategy_weights[sa["strategy"]] = sa["weight"]
            if sa["action"] == "DISABLE":
                log_with_data(logger, "warning", "Strategy DISABLED by tuner", {
                    "strategy": sa["strategy"],
                    "win_rate": sa["win_rate"],
                    "total_pnl": sa["total_pnl"],
                })
            elif sa["action"] == "DOUBLE":
                log_with_data(logger, "info", "Strategy DOUBLED by tuner", {
                    "strategy": sa["strategy"],
                    "win_rate": sa["win_rate"],
                    "total_pnl": sa["total_pnl"],
                })

    def _apply_trading_hours(self, tod_analysis: list[dict]) -> None:
        """Set profitable trading hours (top 12 by avg P&L)."""
        profitable = [h for h in tod_analysis if h["trades"] >= 3 and h["avg_pnl"] > 0]
        if len(profitable) >= 6:
            self._profitable_hours = [h["hour_utc"] for h in profitable[:18]]
        else:
            self._profitable_hours = list(range(24))

    def _apply_regime_matrix(self, regime_analysis: list[dict]) -> None:
        """Build strategy-regime activation matrix."""
        self._regime_matrix = {}
        for entry in regime_analysis:
            strat = entry["strategy"]
            regime = entry["regime"]
            if strat not in self._regime_matrix:
                self._regime_matrix[strat] = set()
            if entry["profitable"] and entry["trades"] >= 3:
                self._regime_matrix[strat].add(regime)

    def _apply_asset_multipliers(self, asset_analysis: list[dict]) -> None:
        """Set per-asset size multipliers based on performance ranking."""
        if len(asset_analysis) < 3:
            return

        # Bottom 3: reduce to 50%
        for asset in asset_analysis[-3:]:
            if asset["total_pnl"] < 0:
                self._asset_multipliers[asset["symbol"]] = 0.5

        # Top 3: increase to 150%
        for asset in asset_analysis[:3]:
            if asset["total_pnl"] > 0:
                self._asset_multipliers[asset["symbol"]] = 1.5

    # ---------------------------------------------------------------
    # Query methods used by engine
    # ---------------------------------------------------------------

    def get_strategy_weight(self, strategy_name: str) -> float:
        """Get the current weight multiplier for a strategy."""
        return self._strategy_weights.get(strategy_name, 1.0)

    def is_profitable_hour(self, hour_utc: Optional[int] = None) -> bool:
        """Check if the current hour is in the profitable hours list."""
        if hour_utc is None:
            hour_utc = datetime.now(timezone.utc).hour
        return hour_utc in self._profitable_hours

    def is_strategy_active_in_regime(self, strategy_name: str, regime: str) -> bool:
        """Check if a strategy should be active in the given regime."""
        if strategy_name not in self._regime_matrix:
            return True  # No data yet, allow all
        return regime in self._regime_matrix[strategy_name]

    def get_asset_multiplier(self, symbol: str) -> float:
        """Get the position size multiplier for an asset."""
        return self._asset_multipliers.get(symbol, 1.0)

    def should_run_analysis(self) -> bool:
        """Check if it's time for a new analysis run."""
        elapsed_hours = (time.time() - self._last_analysis_time) / 3600
        return elapsed_hours >= self._analysis_interval_hours

    def get_analysis_summary(self) -> dict[str, Any]:
        """Get summary for dashboard display."""
        return {
            "last_analysis_time": self._last_analysis_time,
            "strategy_weights": dict(self._strategy_weights),
            "profitable_hours": list(self._profitable_hours),
            "regime_matrix": {k: list(v) for k, v in self._regime_matrix.items()},
            "asset_multipliers": dict(self._asset_multipliers),
            "analysis": self._last_analysis,
        }
