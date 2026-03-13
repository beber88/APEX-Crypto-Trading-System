"""APEX Trading Engine — Core orchestration loop.

Wires together all system components: data ingestion, indicator computation,
regime classification, strategy signal generation, signal aggregation,
trade decision evaluation, and order execution on MEXC.

This is the beating heart of the system — it runs continuously, scanning
for opportunities and managing positions.
"""

from __future__ import annotations

import asyncio
import os
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

from apex_crypto.core.logging import get_logger, log_with_data

logger = get_logger("engine")

# Strategy name → class import path
STRATEGY_REGISTRY: dict[str, tuple[str, str]] = {
    "trend_momentum": ("apex_crypto.core.strategies.trend", "TrendMomentumStrategy"),
    "mean_reversion": ("apex_crypto.core.strategies.mean_reversion", "MeanReversionStrategy"),
    "breakout": ("apex_crypto.core.strategies.breakout", "BreakoutStrategy"),
    "smc": ("apex_crypto.core.strategies.smc_strategy", "SMCStrategy"),
    "scalping": ("apex_crypto.core.strategies.scalping", "ScalpingStrategy"),
    "funding_rate": ("apex_crypto.core.strategies.funding_rate", "FundingRateStrategy"),
    "swing": ("apex_crypto.core.strategies.swing", "SwingStructureStrategy"),
    "oi_divergence": ("apex_crypto.core.strategies.oi_divergence", "OIDivergenceStrategy"),
    "quant_momentum": ("apex_crypto.core.strategies.momentum_factor", "QuantMomentum"),
    "stat_arb": ("apex_crypto.core.strategies.stat_arb", "PairsTrading"),
    # New HF strategies (Research Area 5)
    "vwap_reversion": ("apex_crypto.core.strategies.vwap_reversion", "VWAPReversionStrategy"),
    "funding_scalp": ("apex_crypto.core.strategies.funding_scalp", "FundingScalpStrategy"),
    "liquidation_fade": ("apex_crypto.core.strategies.liquidation_fade", "LiquidationFadeStrategy"),
    "opening_range": ("apex_crypto.core.strategies.opening_range", "OpeningRangeBreakout"),
    "cross_exchange_momentum": ("apex_crypto.core.strategies.cross_exchange_momentum", "CrossExchangeMomentum"),
}


class TradingEngine:
    """Main trading orchestration engine.

    Runs a continuous loop that:
    1. Fetches latest OHLCV data for all watched symbols
    2. Computes technical indicators
    3. Classifies market regime
    4. Runs all enabled strategies to generate signals
    5. Aggregates and weights signals
    6. Evaluates trade decisions (risk checks)
    7. Executes trades via MEXC broker
    8. Monitors open positions for exits
    """

    def __init__(self, config: dict, full_config: dict) -> None:
        self._config = config
        self._full_config = full_config
        self._running = False

        # Cycle interval in seconds
        self._cycle_interval: int = config.get("cycle_interval_seconds", 60)
        self._data_refresh_interval: int = config.get("data_refresh_seconds", 300)
        self._last_data_refresh: float = 0.0

        # Components (initialized in setup())
        self._broker = None
        self._indicator_engine = None
        self._regime_classifier = None
        self._strategies: list = []
        self._aggregator = None
        self._decision_engine = None
        self._alt_data_manager = None
        self._storage = None
        self._risk_guards = None
        # New optimization modules
        self._compounding_engine = None
        self._entry_optimizer = None
        self._exit_optimizer = None
        self._strategy_tuner = None
        self._speed_layer = None
        self._position_sizing_engine = None
        self._exposure_controller = None
        self._stats_repo = None
        self._last_trade_won: bool = True
        self._market_vol_percentile: float = 50.0

        # Ultra mode
        self._ultra_mode: bool = full_config.get("ultra_mode", {}).get("enabled", False)

        # BTC price history for kill switch (Rule 30)
        self._btc_price_history: list[tuple[float, float]] = []  # (timestamp, price)

        # Runtime state
        self._open_positions: list[dict[str, Any]] = []
        self._daily_stats: dict[str, Any] = {
            "trades_today": 0,
            "daily_pnl_pct": 0.0,
            "consecutive_losses": 0,
            "last_loss_ts": None,
        }
        self._equity_stats: dict[str, Any] = {
            "current_drawdown_pct": 0.0,
            "peak_equity": 0.0,
            "current_equity": 0.0,
        }
        self._current_signals: list[dict] = []
        self._current_regimes: dict[str, dict] = {}
        self._ohlcv_cache: dict[str, dict[str, pd.DataFrame]] = {}
        self._last_cycle_time: float = 0.0
        self._cycle_count: int = 0

        log_with_data(logger, "info", "TradingEngine created", {
            "cycle_interval": self._cycle_interval,
        })

    async def setup(self) -> None:
        """Initialize all sub-components."""
        log_with_data(logger, "info", "Setting up TradingEngine components")

        # 1. Broker
        from apex_crypto.core.execution.mexc_broker import MEXCBroker
        mode = self._full_config.get("system", {}).get("mode", "paper")
        exchange_cfg = self._full_config.get("exchange", {})
        broker_config = {
            "exchange": exchange_cfg,
            "paper_trading": mode == "paper",
            "rate_limit_ms": exchange_cfg.get("rate_limit_ms", 100),
            "paper_initial_balance": 10_000.0,
        }
        self._broker = MEXCBroker(broker_config)

        # 2. Indicator engine
        from apex_crypto.core.analysis.indicators import IndicatorEngine
        self._indicator_engine = IndicatorEngine(
            self._full_config.get("indicators", {})
        )

        # 3. Regime classifier
        from apex_crypto.core.analysis.regime import RegimeClassifier
        self._regime_classifier = RegimeClassifier(self._full_config)

        # 4. Strategies
        self._strategies = self._load_strategies()

        # 5. Signal aggregator
        from apex_crypto.core.signals.aggregator import SignalAggregator
        self._aggregator = SignalAggregator(
            self._full_config.get("signals", {}),
            self._strategies,
        )

        # 6. Decision engine
        from apex_crypto.core.signals.decision import TradeDecisionEngine
        self._decision_engine = TradeDecisionEngine(self._full_config)

        # 6b. Risk guards (Rules 7, 10, 15, 30)
        from apex_crypto.core.risk.guards import RiskGuards
        self._risk_guards = RiskGuards(self._full_config)

        # 6c. Compounding engine (anti-martingale + drawdown-adjusted sizing)
        try:
            from apex_crypto.core.risk.compounding import CompoundingEngine
            comp_cfg = dict(self._full_config.get("compounding", {}))
            comp_cfg["ultra_mode"] = self._ultra_mode
            if self._ultra_mode:
                # Ultra defaults: faster rebalance, no profit lock, growth^0.7
                comp_cfg.setdefault("compound_frequency", 5)
                comp_cfg.setdefault("rebalance_every_hours", 3)
                comp_cfg.setdefault("profit_lock_fraction", 0.0)
                comp_cfg.setdefault("growth_exponent", 0.7)
            self._compounding_engine = CompoundingEngine(comp_cfg)
        except Exception as exc:
            logger.warning("CompoundingEngine not available: %s", exc)

        # 6d. Entry optimizer (dynamic order type, partial entries)
        try:
            from apex_crypto.core.signals.entry_optimizer import EntryOptimizer
            self._entry_optimizer = EntryOptimizer(
                self._full_config.get("entry_optimizer", {})
            )
        except Exception as exc:
            logger.warning("EntryOptimizer not available: %s", exc)

        # 6e. Exit optimizer (dynamic TP, time-stop, re-entry)
        try:
            from apex_crypto.core.signals.exit_optimizer import ExitOptimizer
            self._exit_optimizer = ExitOptimizer(
                self._full_config.get("exit_optimizer", self._full_config.get("risk", {}))
            )
        except Exception as exc:
            logger.warning("ExitOptimizer not available: %s", exc)

        # 6f. Strategy auto-tuner (kill losers, boost winners)
        try:
            from apex_crypto.core.signals.strategy_tuner import StrategyTuner
            self._strategy_tuner = StrategyTuner(
                self._full_config.get("strategy_tuner", {})
            )
        except Exception as exc:
            logger.warning("StrategyTuner not available: %s", exc)

        # 6g. Speed layer (pre-computation cache)
        try:
            from apex_crypto.core.execution.speed_layer import PreComputeCache
            self._speed_layer = PreComputeCache(
                self._full_config.get("speed_layer", {})
            )
        except Exception as exc:
            logger.warning("SpeedLayer not available: %s", exc)

        # 6h. Position sizing engine (Kelly + Anti-Martingale)
        try:
            from apex_crypto.core.risk.position_sizing import PositionSizingEngine
            ps_cfg = self._full_config.get("position_sizing", {})
            ps_config = {
                "base_risk_pct": ps_cfg.get("base_risk_pct",
                    self._full_config.get("risk", {}).get("risk_per_trade_pct", 1.0) / 100.0),
                "max_risk_pct": ps_cfg.get("max_risk_pct", 0.08),
                "min_risk_pct": ps_cfg.get("min_risk_pct", 0.003),
                "kelly_fraction": ps_cfg.get("kelly_fraction", 0.8),
                "win_streak_boost": ps_cfg.get("win_streak_boost", 0.20),
                "loss_streak_cut": ps_cfg.get("loss_streak_cut", 0.15),
                "ultra_mode": self._ultra_mode,
            }
            self._position_sizing_engine = PositionSizingEngine(ps_config)
        except Exception as exc:
            logger.warning("PositionSizingEngine not available: %s", exc)

        # 6i. Exposure controller (adaptive leverage & regime-based sizing)
        try:
            from apex_crypto.core.risk.exposure import ExposureController
            exp_cfg = self._full_config.get("exposure", {})
            exposure_config = {
                "max_positions_base": exp_cfg.get("max_positions_base",
                    self._full_config.get("risk", {}).get("max_open_positions", 12)),
                "max_positions_aggressive": exp_cfg.get("max_positions_aggressive", 15),
                "max_positions_ultra": exp_cfg.get("max_positions_ultra", 25),
                "max_portfolio_leverage": exp_cfg.get("max_portfolio_leverage",
                    self._full_config.get("risk", {}).get("max_leverage", 3.0)),
                "max_portfolio_leverage_ultra": exp_cfg.get("max_portfolio_leverage_ultra", 8.0),
                "ultra_mode": self._ultra_mode,
                "ultra_max_drawdown_pct": exp_cfg.get("ultra_max_drawdown_pct", 8.0),
                "ultra_min_volume_24h": exp_cfg.get("ultra_min_volume_24h", 5_000_000),
                "ultra_max_spread_pct": exp_cfg.get("ultra_max_spread_pct", 0.08),
                "ultra_min_signal_score": exp_cfg.get("ultra_min_signal_score", 50),
                **{k: v for k, v in exp_cfg.items() if k.startswith("regime_") or k.startswith("ultra_regime_")},
            }
            self._exposure_controller = ExposureController(exposure_config)
        except Exception as exc:
            logger.warning("ExposureController not available: %s", exc)

        # 6j. Stats repository (DB-backed strategy stats for Kelly sizing)
        try:
            from apex_crypto.core.risk.stats_repository import StatsRepository
            self._stats_repo = StatsRepository(self._full_config.get("data", {}))
        except Exception as exc:
            logger.warning("StatsRepository not available: %s", exc)

        # 7. Storage manager (optional — may fail without DB connections)
        try:
            from apex_crypto.core.data.storage import StorageManager
            data_cfg = self._full_config.get("data", {})
            self._storage = StorageManager(data_cfg)
            log_with_data(logger, "info", "StorageManager initialized")
        except Exception as exc:
            logger.warning("StorageManager not available: %s", exc)
            self._storage = None

        # 8. Alt data manager (optional — may fail without API keys)
        try:
            from apex_crypto.core.data.alt_data import AlternativeDataManager
            self._alt_data_manager = AlternativeDataManager(
                self._full_config.get("data", {}), self._storage
            )
        except Exception as exc:
            logger.warning("Alt data manager not available: %s", exc)
            self._alt_data_manager = None

        # 9. Try to get initial balance (timeout after 10s to avoid blocking startup)
        try:
            balance = await asyncio.wait_for(
                self._broker.get_balance(), timeout=10.0
            )
            equity = balance.get("total_usdt", 10_000.0)
            self._equity_stats["current_equity"] = equity
            self._equity_stats["peak_equity"] = equity
            log_with_data(logger, "info", "Initial balance fetched", {
                "equity": equity,
            })
        except Exception as exc:
            logger.warning("Could not fetch initial balance: %s — using default", exc)
            self._equity_stats["current_equity"] = 10_000.0
            self._equity_stats["peak_equity"] = 10_000.0

        log_with_data(logger, "info", "TradingEngine setup complete", {
            "strategies_loaded": len(self._strategies),
            "mode": mode,
        })

    def _load_strategies(self) -> list:
        """Dynamically load and instantiate all enabled strategies."""
        import importlib
        strategies = []
        strategies_cfg = self._full_config.get("strategies", {})

        for name, (module_path, class_name) in STRATEGY_REGISTRY.items():
            strat_cfg = strategies_cfg.get(name, {})
            if not strat_cfg.get("enabled", True):
                logger.info("Strategy %s disabled, skipping", name)
                continue

            try:
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)
                instance = cls(strat_cfg)
                strategies.append(instance)
                log_with_data(logger, "info", "Strategy loaded", {
                    "name": name, "class": class_name,
                })
            except Exception as exc:
                logger.warning("Failed to load strategy %s: %s", name, exc)

        return strategies

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self, symbols: list[str], timeframes: list[str]) -> None:
        """Run the main trading loop.

        Args:
            symbols: List of trading pairs to monitor.
            timeframes: List of timeframes to fetch data for.
        """
        self._running = True
        log_with_data(logger, "info", "Trading engine started", {
            "symbols": len(symbols),
            "timeframes": timeframes,
        })

        # Initial data load (timeout so dashboard isn't blocked)
        try:
            await asyncio.wait_for(
                self._refresh_market_data(symbols, timeframes),
                timeout=30.0,
            )
        except (asyncio.TimeoutError, Exception) as exc:
            logger.warning("Initial data load incomplete: %s — will retry in loop", exc)

        while self._running:
            try:
                cycle_start = time.time()
                self._cycle_count += 1

                # Refresh data periodically
                if time.time() - self._last_data_refresh > self._data_refresh_interval:
                    await self._refresh_market_data(symbols, timeframes)

                # Run the trading cycle
                await self._trading_cycle(symbols, timeframes)

                self._last_cycle_time = time.time() - cycle_start

                if self._cycle_count % 10 == 0:
                    log_with_data(logger, "info", "Engine heartbeat", {
                        "cycle": self._cycle_count,
                        "cycle_time_ms": round(self._last_cycle_time * 1000),
                        "open_positions": len(self._open_positions),
                        "equity": self._equity_stats.get("current_equity", 0),
                    })

                # Wait for next cycle
                elapsed = time.time() - cycle_start
                sleep_time = max(0, self._cycle_interval - elapsed)
                await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Trading cycle error: %s\n%s", exc, traceback.format_exc())
                await asyncio.sleep(10)

        log_with_data(logger, "info", "Trading engine stopped")

    async def stop(self) -> None:
        """Gracefully stop the engine."""
        self._running = False
        if self._broker:
            await self._broker.close()
        if self._alt_data_manager:
            try:
                await self._alt_data_manager.close()
            except Exception:
                pass
        if self._storage:
            try:
                self._storage.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Data refresh
    # ------------------------------------------------------------------

    async def _refresh_market_data(
        self, symbols: list[str], timeframes: list[str]
    ) -> None:
        """Fetch latest OHLCV data for all symbols and timeframes."""
        log_with_data(logger, "info", "Refreshing market data", {
            "symbols": len(symbols), "timeframes": timeframes,
        })

        for symbol in symbols:
            self._ohlcv_cache.setdefault(symbol, {})
            for tf in timeframes:
                try:
                    df = await self._fetch_ohlcv(symbol, tf)
                    if df is not None and not df.empty:
                        self._ohlcv_cache[symbol][tf] = df
                except Exception as exc:
                    logger.warning("Failed to fetch %s %s: %s", symbol, tf, exc)

            # Small delay to respect rate limits
            await asyncio.sleep(0.2)

        self._last_data_refresh = time.time()
        log_with_data(logger, "info", "Market data refreshed", {
            "cached_symbols": len(self._ohlcv_cache),
        })

    async def _fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int = 500
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV candles from MEXC via ccxt."""
        try:
            ohlcv = await self._broker._exchange.fetch_ohlcv(
                symbol, timeframe, limit=limit
            )
            if not ohlcv:
                return None

            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.set_index("timestamp")
            return df

        except Exception as exc:
            logger.warning("OHLCV fetch error %s %s: %s", symbol, timeframe, exc)
            return None

    # ------------------------------------------------------------------
    # Trading cycle
    # ------------------------------------------------------------------

    async def _trading_cycle(
        self, symbols: list[str], timeframes: list[str]
    ) -> None:
        """Execute one full trading cycle across all symbols."""

        # Rule 30: Kill switch check — emergency close all on broad market crash
        await self._check_kill_switch()
        if self._risk_guards and self._risk_guards.is_kill_switch_triggered:
            log_with_data(logger, "critical",
                          "KILL SWITCH ACTIVE — closing all positions")
            await self._emergency_close_all()
            return

        # Ultra mode safety checks (tighter drawdown limit)
        if self._ultra_mode and self._exposure_controller:
            try:
                ultra_safety = self._exposure_controller.check_ultra_safeties(
                    self._equity_stats.get("current_drawdown_pct", 0.0),
                    len(self._open_positions),
                )
                if ultra_safety.get("emergency_close"):
                    log_with_data(logger, "critical", ultra_safety["reason"])
                    await self._emergency_close_all()
                    if ultra_safety.get("disable_ultra"):
                        self._ultra_mode = False
                        self._exposure_controller.set_ultra_mode(False)
                        if self._position_sizing_engine:
                            self._position_sizing_engine.ultra_mode = False
                    return
                if ultra_safety.get("pause_trading"):
                    log_with_data(logger, "warning", ultra_safety["reason"])
                    return
            except Exception as exc:
                logger.debug("Ultra safety check error: %s", exc)

        # Check if trading should be paused
        should_pause, pause_reason = self._decision_engine.should_pause_trading(
            self._daily_stats, self._equity_stats,
        )
        if should_pause:
            log_with_data(logger, "warning", "Trading paused", {
                "reason": pause_reason,
            })
            return

        # Update equity
        await self._update_equity()

        # Rule 15: Check break-even stops for open positions
        self._update_breakeven_stops()

        # Monitor existing positions for exits
        await self._check_exits(symbols)

        # Pre-compute cache refresh (speed layer)
        if self._speed_layer:
            try:
                await self._speed_layer.refresh(
                    symbols, self._ohlcv_cache,
                    self._indicator_engine,
                    self._full_config.get("risk", {}),
                    self._equity_stats.get("current_equity", 10_000.0),
                )
            except Exception as exc:
                logger.debug("Speed layer refresh error: %s", exc)

        # Strategy tuner: check if analysis should run
        if self._strategy_tuner and self._strategy_tuner.should_run_analysis():
            try:
                # In production, pull from trade history DB
                self._strategy_tuner.analyze_performance([])
            except Exception as exc:
                logger.debug("Strategy tuner analysis error: %s", exc)

        # Scan for new entry signals (PARALLEL for speed)
        all_aggregated: list[dict[str, Any]] = []
        self._current_signals = []

        # Parallel scan all symbols simultaneously
        scan_tasks = []
        for symbol in symbols:
            scan_tasks.append(self._scan_symbol(symbol, timeframes))

        scan_results = await asyncio.gather(*scan_tasks, return_exceptions=True)
        for result in scan_results:
            if result is not None and not isinstance(result, Exception):
                all_aggregated.append(result)

        # Rule 2 (Eugene Ng): Boost signals for relatively strong coins
        if all_aggregated:
            self._apply_relative_strength_bonus(all_aggregated)

        # Rank opportunities and execute
        if all_aggregated:
            ranked = self._aggregator.rank_opportunities(all_aggregated)
            for opportunity in ranked:
                await self._execute_opportunity(opportunity)

    async def _scan_symbol(
        self, symbol: str, timeframes: list[str]
    ) -> Optional[dict[str, Any]]:
        """Run all strategies on a single symbol and aggregate signals."""
        data = self._ohlcv_cache.get(symbol, {})
        if not data:
            return None

        # Compute indicators for each timeframe
        indicators: dict[str, pd.DataFrame] = {}
        for tf, df in data.items():
            if df is not None and not df.empty:
                try:
                    ind_df = df.copy()
                    ind_df = ind_df.reset_index()
                    computed = self._indicator_engine.compute_all(ind_df, tf)
                    indicators[tf] = computed
                except Exception as exc:
                    logger.debug("Indicator error %s %s: %s", symbol, tf, exc)
                    indicators[tf] = df

        # Classify regime from the 4h or 1d timeframe
        regime = "RANGING"
        regime_confidence = 0.5
        regime_tf = "4h" if "4h" in data else ("1d" if "1d" in data else None)
        if regime_tf and regime_tf in data:
            try:
                regime_df = data[regime_tf].reset_index()
                alt_data_dict = await self._get_alt_data(symbol)
                regime_result = self._regime_classifier.classify_from_df(
                    regime_df, alt_data_dict
                )
                regime = regime_result.get("regime", "RANGING")
                regime_confidence = regime_result.get("confidence", 0.5)
            except Exception as exc:
                logger.debug("Regime classification error %s: %s", symbol, exc)

        self._current_regimes[symbol] = {
            "regime": regime,
            "confidence": regime_confidence,
            "timestamp": time.time(),
        }

        # Generate signals from all active strategies
        signals = []
        alt_data = await self._get_alt_data(symbol)

        for strategy in self._strategies:
            try:
                if not strategy.is_active(regime):
                    continue
                signal = strategy.generate_signal(
                    symbol, data, indicators, regime, alt_data
                )
                if signal.direction.value != "neutral" and signal.score != 0:
                    signals.append(signal)
                    self._current_signals.append(signal.to_dict())
            except Exception as exc:
                logger.debug("Strategy %s error on %s: %s",
                             strategy.name, symbol, exc)

        if not signals:
            logger.debug("No non-neutral signals for %s from %d strategies",
                         symbol, len(self._strategies))
            return None

        # Log signal details for debugging
        for sig in signals:
            logger.info("Signal: %s %s dir=%s score=%d strategy=%s",
                        symbol, sig.timeframe, sig.direction.value,
                        sig.score, sig.strategy)

        # Aggregate signals
        aggregated = self._aggregator.aggregate_signals(symbol, signals)

        # Apply bonuses
        tf_alignment = self._check_timeframe_alignment(indicators, regime)
        sentiment = {"direction": "neutral"}
        fear_greed = 50

        if alt_data:
            fg = alt_data.get("fear_greed", 50)
            if isinstance(fg, dict):
                fear_greed = int(fg.get("value", 50))
            elif isinstance(fg, (int, float)):
                fear_greed = int(fg)

        # Get current price and recent candles for Rule 3/6
        current_price = 0.0
        recent_candles: list[dict] = []
        entry_tf = "1h" if "1h" in data else ("4h" if "4h" in data else None)
        if entry_tf and entry_tf in data:
            df = data[entry_tf]
            if not df.empty:
                current_price = float(df["close"].iloc[-1])
                # Last 5 candles for Rule 6 (three candle confirmation)
                tail = df.tail(5)
                recent_candles = [
                    {"open": float(r["open"]), "close": float(r["close"])}
                    for _, r in tail.iterrows()
                ]

        aggregated = self._aggregator.apply_bonuses(
            aggregated, tf_alignment, sentiment, fear_greed,
            current_price=current_price,
            recent_candles=recent_candles,
        )

        return aggregated

    async def _get_alt_data(self, symbol: str) -> dict:
        """Get alternative data for a symbol (with caching)."""
        if not self._alt_data_manager:
            return {}
        try:
            # Use funding rate if available
            funding_rates = await self._alt_data_manager.fetch_funding_rates([symbol])
            fg = await self._alt_data_manager.fetch_fear_greed_index()

            # funding_rates keys use futures format (e.g. "BTC/USDT:USDT")
            futures_symbol = f"{symbol}:USDT" if ":USDT" not in symbol and "/USDT" in symbol else symbol
            rate = funding_rates.get(futures_symbol, funding_rates.get(symbol, 0.0))

            return {
                "funding_rate": rate,
                "fear_greed": fg.get("value", 50) if isinstance(fg, dict) else 50,
            }
        except Exception:
            return {}

    def _check_timeframe_alignment(
        self, indicators: dict[str, pd.DataFrame], regime: str
    ) -> dict[str, bool]:
        """Check if multiple timeframes agree on direction."""
        alignment = {}
        for tf in ["1h", "4h", "1d"]:
            if tf not in indicators:
                alignment[tf] = False
                continue
            ind = indicators[tf]
            if "ema_9" in ind.columns and "ema_21" in ind.columns:
                last_ema9 = ind["ema_9"].iloc[-1] if len(ind) > 0 else 0
                last_ema21 = ind["ema_21"].iloc[-1] if len(ind) > 0 else 0
                if regime in ("STRONG_BULL", "WEAK_BULL"):
                    alignment[tf] = last_ema9 > last_ema21
                elif regime in ("STRONG_BEAR", "WEAK_BEAR"):
                    alignment[tf] = last_ema9 < last_ema21
                else:
                    alignment[tf] = False
            else:
                alignment[tf] = False
        return alignment

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    async def _check_exits(self, symbols: list[str]) -> None:
        """Check all open positions for exit conditions."""
        if not self._open_positions:
            return

        positions_to_close = []

        for position in self._open_positions:
            symbol = position.get("symbol", "")
            try:
                # Get current price
                current_price = await self._get_current_price(symbol)
                if current_price <= 0:
                    continue
                position["current_price"] = current_price

                # Get latest signals for this symbol
                symbol_signals = [
                    s for s in self._strategies
                    if hasattr(s, '_last_signal') and s._last_signal
                ]

                # Check time-stop from exit optimizer
                if self._exit_optimizer:
                    try:
                        time_stop = self._exit_optimizer.check_time_stop(position)
                        if time_stop.get("exit", False):
                            positions_to_close.append((position, 1.0, {
                                "action": "close_full",
                                "reason": time_stop["reason"],
                            }))
                            continue
                    except Exception:
                        pass

                # Check exit conditions
                indicator_state = {
                    "current_regime": self._current_regimes.get(symbol, {}).get("regime", ""),
                    "trailing_stop": position.get("trailing_stop"),
                }

                exit_decision = self._decision_engine.check_exit_conditions(
                    position, indicator_state, []
                )

                action = exit_decision.get("action", "hold")
                if action in ("close_full", "close_partial"):
                    close_pct = exit_decision.get("close_pct", 1.0)
                    positions_to_close.append((position, close_pct, exit_decision))

            except Exception as exc:
                logger.warning("Exit check error for %s: %s", symbol, exc)

        # Execute closes
        for position, close_pct, decision in positions_to_close:
            await self._close_position(position, close_pct, decision)

    async def _close_position(
        self, position: dict, close_pct: float, decision: dict
    ) -> None:
        """Close (partially or fully) a position."""
        symbol = position["symbol"]
        try:
            if close_pct >= 1.0:
                order = await self._broker.close_position(symbol)
                self._open_positions = [
                    p for p in self._open_positions if p["symbol"] != symbol
                ]
                # Calculate PnL
                entry_price = position.get("entry_price", 0)
                current_price = position.get("current_price", 0)
                direction = position.get("direction", "long")
                if entry_price > 0 and current_price > 0:
                    if direction == "long":
                        pnl_pct = (current_price - entry_price) / entry_price * 100
                    else:
                        pnl_pct = (entry_price - current_price) / entry_price * 100

                    self._daily_stats["daily_pnl_pct"] += pnl_pct
                    if pnl_pct < 0:
                        self._daily_stats["consecutive_losses"] += 1
                        self._daily_stats["last_loss_ts"] = time.time()
                        self._last_trade_won = False
                    else:
                        self._daily_stats["consecutive_losses"] = 0
                        self._last_trade_won = True

                    # Record in compounding engine + check rebalance
                    if self._compounding_engine:
                        try:
                            current_eq = self._equity_stats.get("current_equity", 10_000.0)
                            self._compounding_engine.record_trade_result(
                                won=pnl_pct > 0,
                                pnl_pct=pnl_pct,
                                equity=current_eq,
                            )
                            # Check if time/trade-count triggers a rebalance
                            now_ts = time.time()
                            if self._compounding_engine.should_rebalance(now_ts):
                                effective_eq = self._compounding_engine.maybe_lock_profits(current_eq)
                                self._compounding_engine.on_rebalance(effective_eq, now_ts)
                                logger.info("Compounding rebalance triggered (equity=%.2f)", effective_eq)
                        except Exception:
                            pass

                self._daily_stats["trades_today"] += 1

                log_with_data(logger, "info", "Position closed", {
                    "symbol": symbol,
                    "reason": decision.get("reason", ""),
                    "close_pct": close_pct,
                })
            else:
                # Partial close — mark TP hit
                amount = position.get("amount", 0) * close_pct
                direction = position.get("direction", "long")
                side = "sell" if direction == "long" else "buy"
                await self._broker.place_market_order(symbol, side, amount)

                if not position.get("tp1_hit", False):
                    position["tp1_hit"] = True
                elif not position.get("tp2_hit", False):
                    position["tp2_hit"] = True

                log_with_data(logger, "info", "Partial close executed", {
                    "symbol": symbol,
                    "close_pct": close_pct,
                })

        except Exception as exc:
            logger.error("Failed to close position %s: %s", symbol, exc)

    async def _execute_opportunity(self, opportunity: dict[str, Any]) -> None:
        """Evaluate and execute a ranked trading opportunity."""
        symbol = opportunity.get("symbol", "")
        direction = opportunity.get("direction", "neutral")

        if direction == "neutral":
            return

        # Already in position?
        position_symbols = {p.get("symbol") for p in self._open_positions}
        if symbol in position_symbols:
            return

        # Run decision engine
        decision = self._decision_engine.evaluate(
            opportunity, self._open_positions, self._daily_stats
        )

        action = decision.get("action", "skip")
        if action == "skip":
            logger.info("Trade SKIPPED for %s: %s (score=%.1f, agreeing=%d)",
                        symbol, decision.get("reason", ""),
                        decision.get("score", 0),
                        opportunity.get("num_agreeing", 0))
            return

        # Rule 12 (Eugene Ng): Conviction-based position sizing
        # Enhanced with Kelly + Anti-Martingale + Exposure Controller + Compounding
        equity = self._equity_stats.get("current_equity", 10_000.0)
        score = abs(opportunity.get("weighted_score", 0))
        num_agreeing = opportunity.get("num_agreeing", 0)
        strategy_name = opportunity.get("strongest_signal", {}).get("strategy", "aggregated")
        base_risk_pct = self._full_config.get("risk", {}).get("risk_per_trade_pct", 1.0)

        # Layer 1: Kelly + Anti-Martingale from DB stats
        if self._position_sizing_engine and self._stats_repo:
            try:
                db_stats = self._stats_repo.get_strategy_stats(strategy_name)
                if db_stats:
                    from apex_crypto.core.risk.position_sizing import StrategyStats
                    stats = StrategyStats(
                        win_rate=db_stats["win_rate"],
                        avg_win_r=db_stats["avg_win_r"],
                        avg_loss_r=db_stats["avg_loss_r"],
                        recent_win_streak=db_stats["recent_win_streak"],
                        recent_loss_streak=db_stats["recent_loss_streak"],
                    )
                    kelly_risk = self._position_sizing_engine.get_risk_pct(
                        stats, atr_percentile=self._market_vol_percentile
                    )
                    # Convert fraction to percentage for downstream code
                    base_risk_pct = kelly_risk * 100.0
                    logger.info("Kelly+AM risk for %s: %.2f%% (WR=%.1f%%, streak W%d/L%d)",
                                strategy_name, base_risk_pct,
                                db_stats["win_rate"] * 100,
                                db_stats["recent_win_streak"],
                                db_stats["recent_loss_streak"])
            except Exception as exc:
                logger.debug("PositionSizingEngine error: %s", exc)

        # Layer 2: Compounding engine adaptive risk (drawdown + vol)
        if self._compounding_engine:
            try:
                adaptive = self._compounding_engine.calculate_adaptive_risk(
                    base_risk_pct,
                    self._equity_stats,
                    self._market_vol_percentile,
                    self._last_trade_won,
                )
                base_risk_pct = adaptive.get("risk_pct", base_risk_pct)
                logger.info("Adaptive risk: %.2f%% (DD mult=%.2f, Vol mult=%.2f)",
                            base_risk_pct,
                            adaptive.get("drawdown_multiplier", 1.0),
                            adaptive.get("volatility_multiplier", 1.0))
            except Exception as exc:
                logger.debug("Compounding engine error: %s", exc)

        # Layer 3: Exposure controller — regime-based risk multiplier
        current_regime = self._current_regimes.get(symbol, {}).get("regime", "RANGING")
        if self._exposure_controller:
            try:
                regime_mult = self._exposure_controller.get_risk_multiplier(current_regime)
                base_risk_pct *= regime_mult
                logger.info("Regime %s risk multiplier: %.2f → effective risk: %.2f%%",
                            current_regime, regime_mult, base_risk_pct)
            except Exception as exc:
                logger.debug("ExposureController error: %s", exc)

        # Layer 4: Strategy tuner asset multiplier
        if self._strategy_tuner:
            asset_mult = self._strategy_tuner.get_asset_multiplier(symbol)
            base_risk_pct *= asset_mult

        # Layer 5: Conviction scaling
        if score >= 75 and num_agreeing >= 3:
            conviction_risk_pct = min(base_risk_pct * 5, 5.0)
            logger.info("HIGH CONVICTION trade for %s: risk=%.1f%% (score=%d, agreeing=%d)",
                        symbol, conviction_risk_pct, score, num_agreeing)
        elif score >= 60 and num_agreeing >= 2:
            conviction_risk_pct = min(base_risk_pct * 2.5, 3.0)
        else:
            conviction_risk_pct = base_risk_pct

        # Apply profit locking to reduce tradable equity
        effective_equity = equity
        if self._compounding_engine:
            try:
                effective_equity = self._compounding_engine.maybe_lock_profits(equity)
            except Exception:
                pass

        position_size_pct = decision.get("position_size_pct", conviction_risk_pct) / 100.0
        position_value = effective_equity * position_size_pct

        # Get entry price
        current_price = await self._get_current_price(symbol)
        if current_price <= 0:
            return

        amount = position_value / current_price
        if amount <= 0:
            return

        # Get stop loss and take profits from the strongest signal
        strongest = opportunity.get("strongest_signal", {})
        stop_loss = strongest.get("stop_loss", 0)
        tp1 = strongest.get("take_profit_1", 0)
        tp2 = strongest.get("take_profit_2", 0)
        tp3 = strongest.get("take_profit_3", 0)

        # Fallback SL/TP if not set
        if stop_loss <= 0:
            if direction == "long":
                stop_loss = current_price * 0.98
            else:
                stop_loss = current_price * 1.02

        # Adaptive leverage from exposure controller
        leverage = int(self._full_config.get("risk", {}).get("default_leverage", 1))
        if self._exposure_controller:
            try:
                max_leverage = self._exposure_controller.get_max_leverage(current_regime)
            except Exception:
                max_leverage = self._full_config.get("risk", {}).get("max_leverage", 3.0)
        else:
            max_leverage = self._full_config.get("risk", {}).get("max_leverage", 3.0)
        leverage = min(leverage, int(max_leverage))

        # Check max positions from exposure controller
        if self._exposure_controller:
            try:
                max_pos = self._exposure_controller.get_max_positions(current_regime)
                if len(self._open_positions) >= max_pos:
                    logger.info("Max positions (%d) reached for regime %s — skipping %s",
                                max_pos, current_regime, symbol)
                    return
            except Exception:
                pass

        # Build entry signal for broker
        entry_signal = {
            "symbol": symbol,
            "direction": direction,
            "entry_price": current_price,
            "entry_type": "market",
            "amount": amount,
            "stop_loss": stop_loss,
            "take_profit": [
                {"price": tp1, "pct": 0.35} if tp1 > 0 else None,
                {"price": tp2, "pct": 0.35} if tp2 > 0 else None,
            ],
            "leverage": leverage,
            "strategy": strongest.get("strategy", "aggregated"),
            "signal_score": opportunity.get("weighted_score", 0),
        }
        entry_signal["take_profit"] = [
            tp for tp in entry_signal["take_profit"] if tp is not None
        ]
        if not entry_signal["take_profit"]:
            entry_signal.pop("take_profit")

        try:
            trade_record = await self._broker.execute_entry(entry_signal)

            # Track position
            self._open_positions.append({
                "symbol": symbol,
                "direction": direction,
                "entry_price": current_price,
                "current_price": current_price,
                "stop_loss": stop_loss,
                "take_profit_1": tp1,
                "take_profit_2": tp2,
                "take_profit_3": tp3,
                "tp1_hit": False,
                "tp2_hit": False,
                "amount": amount,
                "leverage": leverage,
                "strategy": strongest.get("strategy", "aggregated"),
                "regime_at_entry": self._current_regimes.get(symbol, {}).get("regime", ""),
                "open_timestamp": time.time(),
                "trade_id": trade_record.get("trade_id", ""),
                "trailing_stop": None,
            })

            self._daily_stats["trades_today"] += 1

            log_with_data(logger, "info", "TRADE EXECUTED", {
                "symbol": symbol,
                "direction": direction,
                "entry_price": current_price,
                "amount": round(amount, 6),
                "stop_loss": stop_loss,
                "leverage": leverage,
                "score": opportunity.get("weighted_score", 0),
            })

        except Exception as exc:
            logger.error("Trade execution failed for %s: %s", symbol, exc)

    async def _get_current_price(self, symbol: str) -> float:
        """Get the latest price for a symbol."""
        try:
            ticker = await self._broker._exchange.fetch_ticker(symbol)
            return float(ticker.get("last", 0) or 0)
        except Exception:
            # Fallback to cached OHLCV
            data = self._ohlcv_cache.get(symbol, {})
            for tf in ["1m", "5m", "15m", "1h", "4h"]:
                if tf in data and not data[tf].empty:
                    return float(data[tf]["close"].iloc[-1])
            return 0.0

    async def _update_equity(self) -> None:
        """Update current equity from broker."""
        try:
            balance = await self._broker.get_balance()
            equity = balance.get("total_usdt", self._equity_stats["current_equity"])
            self._equity_stats["current_equity"] = equity
            if equity > self._equity_stats["peak_equity"]:
                self._equity_stats["peak_equity"] = equity
            peak = self._equity_stats["peak_equity"]
            if peak > 0:
                self._equity_stats["current_drawdown_pct"] = (
                    (peak - equity) / peak * 100
                )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Rule 2: Relative Strength Scanner (Eugene Ng)
    # ------------------------------------------------------------------

    def _apply_relative_strength_bonus(
        self, aggregated_signals: list[dict[str, Any]]
    ) -> None:
        """Boost scores for coins showing relative strength vs BTC.

        When BTC is pulling back, coins that drop less show relative
        strength and will likely rally harder on recovery. Add a bonus
        to their aggregated scores.
        """
        btc_data = self._ohlcv_cache.get("BTC/USDT", {})
        btc_1h = btc_data.get("1h") if btc_data else None
        if btc_1h is None or len(btc_1h) < 24:
            return

        # Check if BTC is in a pullback (24h return < -2%)
        btc_close_now = float(btc_1h["close"].iloc[-1])
        btc_close_24h = float(btc_1h["close"].iloc[-24])
        if btc_close_24h <= 0:
            return
        btc_return = (btc_close_now - btc_close_24h) / btc_close_24h * 100.0

        if btc_return >= -2.0:
            return  # BTC not in pullback, no relative strength analysis

        # Calculate each symbol's 24h return
        symbol_returns: dict[str, float] = {}
        for sig in aggregated_signals:
            symbol = sig["symbol"]
            sym_data = self._ohlcv_cache.get(symbol, {})
            sym_1h = sym_data.get("1h")
            if sym_1h is None or len(sym_1h) < 24:
                continue
            close_now = float(sym_1h["close"].iloc[-1])
            close_24h = float(sym_1h["close"].iloc[-24])
            if close_24h > 0:
                symbol_returns[symbol] = (close_now - close_24h) / close_24h * 100.0

        if not symbol_returns:
            return

        # Rank by relative strength (least negative = strongest)
        sorted_symbols = sorted(symbol_returns.items(), key=lambda x: x[1], reverse=True)
        top_strong = {s for s, _ in sorted_symbols[:max(1, len(sorted_symbols) // 3)]}

        for sig in aggregated_signals:
            if sig["symbol"] in top_strong:
                old_score = sig["weighted_score"]
                bonus = 10 if old_score > 0 else -10  # same sign as direction
                sig["weighted_score"] = round(old_score + bonus, 2)
                sig.setdefault("bonus_breakdown", {})["relative_strength"] = abs(bonus)
                logger.info("Rule 2: Relative strength bonus for %s (24h: %.1f%% vs BTC: %.1f%%)",
                            sig["symbol"],
                            symbol_returns.get(sig["symbol"], 0),
                            btc_return)

    # ------------------------------------------------------------------
    # Rule 30: Kill switch helpers (Goodman)
    # ------------------------------------------------------------------

    async def _check_kill_switch(self) -> None:
        """Track BTC price and check for black swan event."""
        if not self._risk_guards:
            return
        try:
            btc_price = await self._get_current_price("BTC/USDT")
            if btc_price <= 0:
                return
            now = time.time()
            self._btc_price_history.append((now, btc_price))
            # Keep only last 2 hours of data
            cutoff = now - 7200
            self._btc_price_history = [
                (ts, p) for ts, p in self._btc_price_history if ts > cutoff
            ]
            # Find price from 1 hour ago
            target_ts = now - 3600
            btc_price_1h_ago = 0.0
            for ts, p in self._btc_price_history:
                if ts <= target_ts:
                    btc_price_1h_ago = p
            if btc_price_1h_ago > 0:
                self._risk_guards.check_kill_switch(btc_price, btc_price_1h_ago)
        except Exception as exc:
            logger.debug("Kill switch check error: %s", exc)

    async def _emergency_close_all(self) -> None:
        """Emergency close all positions (Rule 30 kill switch)."""
        log_with_data(logger, "critical",
                      "EMERGENCY CLOSE ALL — Black Swan Kill Switch activated",
                      {"open_positions": len(self._open_positions)})
        for position in list(self._open_positions):
            try:
                await self._broker.close_position(position["symbol"])
                log_with_data(logger, "info", "Emergency closed position", {
                    "symbol": position["symbol"],
                })
            except Exception as exc:
                logger.error("Failed to emergency close %s: %s",
                             position["symbol"], exc)
        self._open_positions.clear()

    # ------------------------------------------------------------------
    # Rule 15: Break-even stop management (Eugene Ng)
    # ------------------------------------------------------------------

    def _update_breakeven_stops(self) -> None:
        """Move stop loss to entry price when profit exceeds threshold."""
        if not self._risk_guards:
            return
        for position in self._open_positions:
            entry_price = position.get("entry_price", 0)
            current_price = position.get("current_price", 0)
            direction = position.get("direction", "long")
            current_stop = position.get("stop_loss", 0)

            if current_price <= 0 or entry_price <= 0:
                continue

            # Skip if already at break-even or better
            if direction == "long" and current_stop >= entry_price:
                continue
            if direction == "short" and 0 < current_stop <= entry_price:
                continue

            if self._risk_guards.should_move_to_breakeven(
                entry_price, current_price, direction
            ):
                position["stop_loss"] = entry_price
                log_with_data(logger, "info", "Stop moved to break-even", {
                    "symbol": position.get("symbol"),
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "direction": direction,
                })

    # ------------------------------------------------------------------
    # State accessors (for dashboard)
    # ------------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        """Return current engine state for the dashboard."""
        state = {
            "running": self._running,
            "ultra_mode": self._ultra_mode,
            "cycle_count": self._cycle_count,
            "last_cycle_time_ms": round(self._last_cycle_time * 1000, 1),
            "open_positions": list(self._open_positions),
            "daily_stats": dict(self._daily_stats),
            "equity_stats": dict(self._equity_stats),
            "current_signals": list(self._current_signals),
            "current_regimes": dict(self._current_regimes),
            "strategies_loaded": len(self._strategies),
        }

        # Add compounding engine stats
        if self._compounding_engine:
            try:
                state["compounding"] = {
                    "current_risk_pct": getattr(self._compounding_engine, '_current_risk_pct', 1.0),
                    "consecutive_wins": getattr(self._compounding_engine, '_consecutive_wins', 0),
                    "consecutive_losses": getattr(self._compounding_engine, '_consecutive_losses', 0),
                    "locked_profit": getattr(self._compounding_engine, '_locked_profit_total', 0.0),
                    "base_risk_pct": getattr(self._compounding_engine, 'base_risk_pct', 1.0),
                }
            except Exception:
                pass

        # Add exposure controller state
        if self._exposure_controller:
            try:
                # Use first regime or default
                sample_regime = "RANGING"
                for r in self._current_regimes.values():
                    sample_regime = r.get("regime", "RANGING")
                    break
                state["exposure"] = self._exposure_controller.get_exposure_params(sample_regime)
            except Exception:
                pass

        # Add strategy tuner summary
        if self._strategy_tuner:
            try:
                state["tuner"] = self._strategy_tuner.get_analysis_summary()
            except Exception:
                pass

        # Add entry optimizer stats
        if self._entry_optimizer:
            try:
                state["entry_optimizer"] = self._entry_optimizer.get_stats()
            except Exception:
                pass

        # Add speed layer cache info
        if self._speed_layer:
            try:
                cached = self._speed_layer.get_all_cached()
                state["precomputed_symbols"] = len(cached)
            except Exception:
                pass

        return state
