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

        # Monitor existing positions for exits
        await self._check_exits(symbols)

        # Scan for new entry signals
        all_aggregated: list[dict[str, Any]] = []
        self._current_signals = []

        for symbol in symbols:
            try:
                result = await self._scan_symbol(symbol, timeframes)
                if result:
                    all_aggregated.append(result)
            except Exception as exc:
                logger.warning("Error scanning %s: %s", symbol, exc)

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

        aggregated = self._aggregator.apply_bonuses(
            aggregated, tf_alignment, sentiment, fear_greed
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
                    else:
                        self._daily_stats["consecutive_losses"] = 0

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

        # Calculate position size
        equity = self._equity_stats.get("current_equity", 10_000.0)
        position_size_pct = decision.get("position_size_pct", 2.5) / 100.0
        position_value = equity * position_size_pct

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

        leverage = int(self._full_config.get("risk", {}).get("default_leverage", 1))
        max_leverage = int(self._full_config.get("risk", {}).get("max_leverage", 3))
        leverage = min(leverage, max_leverage)

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
    # State accessors (for dashboard)
    # ------------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        """Return current engine state for the dashboard."""
        return {
            "running": self._running,
            "cycle_count": self._cycle_count,
            "last_cycle_time_ms": round(self._last_cycle_time * 1000, 1),
            "open_positions": list(self._open_positions),
            "daily_stats": dict(self._daily_stats),
            "equity_stats": dict(self._equity_stats),
            "current_signals": list(self._current_signals),
            "current_regimes": dict(self._current_regimes),
            "strategies_loaded": len(self._strategies),
        }
