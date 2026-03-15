"""APEX Stock Trading Engine — orchestrates stock trading alongside crypto.

Runs a parallel trading loop for US equities:
1. Fetches OHLCV data via yfinance
2. Fetches fundamentals for screener/DCF/earnings analysis
3. Runs stock-specific strategies (value, momentum, earnings)
4. Aggregates signals and evaluates trades
5. Executes via Alpaca broker (paper or live)
6. Monitors positions and manages exits

Integrates Goldman Sachs screener, Morgan Stanley DCF, JPMorgan earnings,
Bridgewater risk assessment, and BlackRock portfolio builder.
"""

from __future__ import annotations

import asyncio
import importlib
import time
import traceback
from typing import Any, Optional

import pandas as pd

from apex_crypto.core.logging import get_logger, log_with_data
from apex_crypto.core.signals.aggregator import SignalAggregator
from apex_crypto.core.signals.decision import TradeDecisionEngine
from apex_crypto.core.strategies.base import BaseStrategy, SignalDirection, TradeSignal

logger = get_logger("stocks.engine")

# Stock strategy registry
STOCK_STRATEGY_REGISTRY: dict[str, tuple[str, str]] = {
    "stock_value": (
        "apex_crypto.stocks.strategies.value_strategy",
        "StockValueStrategy",
    ),
    "stock_momentum": (
        "apex_crypto.stocks.strategies.momentum_strategy",
        "StockMomentumStrategy",
    ),
    "stock_earnings": (
        "apex_crypto.stocks.strategies.earnings_strategy",
        "StockEarningsStrategy",
    ),
}


class StockTradingEngine:
    """Trading engine for US equities.

    Runs alongside the crypto TradingEngine, sharing the signal
    aggregation and decision engine infrastructure.
    """

    def __init__(self, stock_config: dict, full_config: dict) -> None:
        self._config = stock_config
        self._full_config = full_config
        self._running = False

        self._cycle_interval = stock_config.get("cycle_interval_seconds", 300)
        self._data_refresh_interval = 600  # 10 minutes

        # Components
        self._broker = None
        self._strategies: list[BaseStrategy] = []
        self._aggregator = None
        self._decision_engine = None
        self._screener = None
        self._risk_analyzer = None

        # State
        self._ohlcv_cache: dict[str, dict[str, pd.DataFrame]] = {}
        self._fundamentals_cache: dict[str, dict] = {}
        self._screen_results: list[dict] = []
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
        self._cycle_count: int = 0
        self._last_data_refresh: float = 0.0
        self._last_screen_time: float = 0.0
        self._last_cycle_time: float = 0.0

        log_with_data(logger, "info", "StockTradingEngine created", {
            "cycle_interval": self._cycle_interval,
        })

    async def setup(self) -> None:
        """Initialize all stock trading components."""
        log_with_data(logger, "info", "Setting up StockTradingEngine")

        # 1. Stock broker
        from apex_crypto.stocks.broker import StockBroker
        mode = self._config.get("mode", "paper")
        broker_cfg = self._config.get("broker", {})
        broker_cfg["paper_trading"] = mode == "paper"
        self._broker = StockBroker(broker_cfg)
        await self._broker.initialize()

        # 2. Load stock strategies
        self._strategies = self._load_strategies()

        # 3. Signal aggregator (reuse crypto infrastructure)
        signals_cfg = self._full_config.get("signals", {})
        self._aggregator = SignalAggregator(signals_cfg, self._strategies)

        # 4. Decision engine
        self._decision_engine = TradeDecisionEngine(self._full_config)

        # 5. Stock screener
        from apex_crypto.stocks.analysis.screener import StockScreener
        self._screener = StockScreener(self._config.get("screener", {}))

        # 6. Risk analyzer
        from apex_crypto.stocks.analysis.risk_analyzer import PortfolioRiskAnalyzer
        self._risk_analyzer = PortfolioRiskAnalyzer(self._config.get("risk_analysis", {}))

        # 7. Initial balance
        try:
            balance = await self._broker.get_balance()
            equity = balance.get("total_equity", 100_000)
            self._equity_stats["current_equity"] = equity
            self._equity_stats["peak_equity"] = equity
            log_with_data(logger, "info", "Stock initial balance", {"equity": equity})
        except Exception:
            self._equity_stats["current_equity"] = 100_000
            self._equity_stats["peak_equity"] = 100_000

        log_with_data(logger, "info", "StockTradingEngine setup complete", {
            "strategies": len(self._strategies),
            "mode": mode,
        })

    def _load_strategies(self) -> list[BaseStrategy]:
        """Load enabled stock strategies."""
        strategies = []
        strategies_cfg = self._config.get("strategies", {})

        for name, (module_path, class_name) in STOCK_STRATEGY_REGISTRY.items():
            strat_cfg = strategies_cfg.get(name, {})
            if not strat_cfg.get("enabled", True):
                continue

            try:
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)
                instance = cls(strat_cfg)
                strategies.append(instance)
                log_with_data(logger, "info", "Stock strategy loaded", {
                    "name": name, "class": class_name,
                })
            except Exception as exc:
                logger.warning("Failed to load stock strategy %s: %s", name, exc)

        return strategies

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Run the stock trading loop."""
        self._running = True

        assets_cfg = self._config.get("assets", {})
        tier1 = assets_cfg.get("tier1", [])
        tier2 = assets_cfg.get("tier2", [])
        all_symbols = tier1 + tier2
        timeframes = self._config.get("timeframes", ["1d"])

        log_with_data(logger, "info", "Stock trading engine started", {
            "symbols": len(all_symbols),
            "timeframes": timeframes,
        })

        # Initial data load
        try:
            await asyncio.wait_for(
                self._refresh_data(all_symbols, timeframes),
                timeout=120.0,
            )
        except Exception as exc:
            logger.warning("Initial stock data load incomplete: %s", exc)

        while self._running:
            try:
                cycle_start = time.time()
                self._cycle_count += 1

                # Refresh data periodically
                if time.time() - self._last_data_refresh > self._data_refresh_interval:
                    await self._refresh_data(all_symbols, timeframes)

                # Run daily screener
                screen_interval = self._config.get("screener", {}).get("run_interval_hours", 24) * 3600
                if time.time() - self._last_screen_time > screen_interval:
                    await self._run_screener()

                # Trading cycle
                await self._trading_cycle(all_symbols, timeframes)

                self._last_cycle_time = time.time() - cycle_start

                if self._cycle_count % 5 == 0:
                    log_with_data(logger, "info", "Stock engine heartbeat", {
                        "cycle": self._cycle_count,
                        "symbols_with_data": sum(1 for v in self._ohlcv_cache.values() if v),
                        "open_positions": len(self._open_positions),
                        "signals": len(self._current_signals),
                    })

                elapsed = time.time() - cycle_start
                sleep_time = max(0, self._cycle_interval - elapsed)
                await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Stock cycle error: %s\n%s", exc, traceback.format_exc())
                await asyncio.sleep(30)

        log_with_data(logger, "info", "Stock trading engine stopped")

    async def stop(self) -> None:
        """Stop the stock engine."""
        self._running = False
        if self._broker:
            await self._broker.close()

    # ------------------------------------------------------------------
    # Data refresh
    # ------------------------------------------------------------------

    async def _refresh_data(
        self, symbols: list[str], timeframes: list[str]
    ) -> None:
        """Fetch OHLCV and fundamentals for all symbols."""
        log_with_data(logger, "info", "Refreshing stock data", {
            "symbols": len(symbols),
        })

        for symbol in symbols:
            self._ohlcv_cache.setdefault(symbol, {})

            # Fetch OHLCV
            for tf in timeframes:
                try:
                    df = await self._broker.fetch_ohlcv(symbol, tf)
                    if df is not None and not df.empty:
                        self._ohlcv_cache[symbol][tf] = df
                except Exception as exc:
                    logger.warning("Stock OHLCV error %s %s: %s", symbol, tf, exc)

            # Fetch fundamentals
            try:
                fundamentals = await self._broker.fetch_fundamentals(symbol)
                if fundamentals and not fundamentals.get("error"):
                    self._fundamentals_cache[symbol] = fundamentals
            except Exception as exc:
                logger.warning("Fundamentals error %s: %s", symbol, exc)

            # Rate limit
            await asyncio.sleep(0.5)

        self._last_data_refresh = time.time()
        symbols_with_data = sum(1 for v in self._ohlcv_cache.values() if v)
        log_with_data(logger, "info", "Stock data refreshed", {
            "symbols_with_data": symbols_with_data,
            "fundamentals_cached": len(self._fundamentals_cache),
        })

    # ------------------------------------------------------------------
    # Screener
    # ------------------------------------------------------------------

    async def _run_screener(self) -> None:
        """Run the Goldman Sachs screener on the screening watchlist."""
        watchlist = self._config.get("assets", {}).get("screening_watchlist", [])
        if not watchlist:
            return

        fundamentals_list = []
        for sym in watchlist:
            f = self._fundamentals_cache.get(sym)
            if f:
                fundamentals_list.append(f)

        if fundamentals_list:
            self._screen_results = self._screener.screen_multiple(
                fundamentals_list, top_n=10
            )
            self._last_screen_time = time.time()

            log_with_data(logger, "info", "Stock screener complete", {
                "screened": len(fundamentals_list),
                "top_picks": [r["symbol"] for r in self._screen_results[:5]],
            })

    # ------------------------------------------------------------------
    # Trading cycle
    # ------------------------------------------------------------------

    async def _trading_cycle(
        self, symbols: list[str], timeframes: list[str]
    ) -> None:
        """Execute one stock trading cycle."""
        self._current_signals = []
        all_aggregated: list[dict[str, Any]] = []

        for symbol in symbols:
            try:
                result = await self._scan_symbol(symbol)
                if result:
                    all_aggregated.append(result)
            except Exception as exc:
                logger.warning("Stock scan error %s: %s", symbol, exc)

        # Rank and execute
        if all_aggregated:
            ranked = self._aggregator.rank_opportunities(all_aggregated)
            for opportunity in ranked:
                await self._execute_opportunity(opportunity)

    async def _scan_symbol(self, symbol: str) -> Optional[dict[str, Any]]:
        """Run all stock strategies on a symbol."""
        data = self._ohlcv_cache.get(symbol, {})
        if not data:
            return None

        fundamentals = self._fundamentals_cache.get(symbol, {})

        # Build alt_data with fundamentals
        alt_data = {"fundamentals": fundamentals}

        # Compute simple indicators
        indicators: dict[str, pd.DataFrame] = {}
        for tf, df in data.items():
            indicators[tf] = df

        # Generate signals
        signals = []
        for strategy in self._strategies:
            try:
                signal = strategy.generate_signal(
                    symbol, data, indicators, "RANGING", alt_data
                )
                if signal.direction != SignalDirection.NEUTRAL and signal.score != 0:
                    signals.append(signal)
                    self._current_signals.append(signal.to_dict())
            except Exception as exc:
                logger.warning("Stock strategy %s error on %s: %s",
                               strategy.name, symbol, exc)

        if not signals:
            return None

        # Aggregate
        aggregated = self._aggregator.aggregate_signals(symbol, signals)
        return aggregated

    async def _execute_opportunity(self, opportunity: dict[str, Any]) -> None:
        """Execute a stock trade opportunity."""
        symbol = opportunity.get("symbol", "")
        direction = opportunity.get("direction", "neutral")

        if direction == "neutral":
            return

        # Skip if already in position
        position_symbols = {p.get("symbol") for p in self._open_positions}
        if symbol in position_symbols:
            return

        # Decision engine evaluation
        decision = self._decision_engine.evaluate(
            opportunity, self._open_positions, self._daily_stats
        )
        if decision.get("action") == "skip":
            return

        # Position sizing
        equity = self._equity_stats.get("current_equity", 100_000)
        max_pct = self._config.get("risk", {}).get("max_position_pct", 8.0)
        position_size_pct = min(decision.get("position_size_pct", 5.0), max_pct) / 100
        position_value = equity * position_size_pct

        # Get current price from fundamentals or last OHLCV
        fundamentals = self._fundamentals_cache.get(symbol, {})
        current_price = fundamentals.get("current_price", 0)
        if current_price <= 0:
            data = self._ohlcv_cache.get(symbol, {})
            for tf_df in data.values():
                if not tf_df.empty:
                    current_price = float(tf_df["close"].iloc[-1])
                    break

        if current_price <= 0:
            return

        shares = int(position_value / current_price)
        if shares <= 0:
            return

        # Get stop/targets from strongest signal
        strongest = opportunity.get("strongest_signal", {})
        stop_loss = strongest.get("stop_loss", current_price * 0.95)

        entry_signal = {
            "symbol": symbol,
            "direction": direction,
            "entry_price": current_price,
            "amount": shares,
            "stop_loss": stop_loss,
            "strategy": strongest.get("strategy", "aggregated"),
        }

        try:
            trade = await self._broker.execute_entry(entry_signal)

            self._open_positions.append({
                "symbol": symbol,
                "direction": direction,
                "entry_price": current_price,
                "current_price": current_price,
                "stop_loss": stop_loss,
                "shares": shares,
                "strategy": entry_signal["strategy"],
                "open_timestamp": time.time(),
                "trade_id": trade.get("trade_id", ""),
                "asset_type": "stock",
            })

            self._daily_stats["trades_today"] += 1

            log_with_data(logger, "info", "STOCK TRADE EXECUTED", {
                "symbol": symbol,
                "direction": direction,
                "shares": shares,
                "price": current_price,
                "value": round(shares * current_price, 2),
            })

        except Exception as exc:
            logger.error("Stock trade failed for %s: %s", symbol, exc)

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        """Return current stock engine state for the dashboard."""
        return {
            "running": self._running,
            "cycle_count": self._cycle_count,
            "open_positions": list(self._open_positions),
            "daily_stats": dict(self._daily_stats),
            "equity_stats": dict(self._equity_stats),
            "current_signals": list(self._current_signals),
            "screen_results": list(self._screen_results[:10]),
            "strategies_loaded": len(self._strategies),
            "symbols_with_data": sum(1 for v in self._ohlcv_cache.values() if v),
            "fundamentals_cached": len(self._fundamentals_cache),
        }
