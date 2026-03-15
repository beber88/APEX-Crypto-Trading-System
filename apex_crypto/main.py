"""APEX Trading System — Main Entry Point.

Initializes all system components and starts the autonomous trading loop.
Supports both crypto trading (via MEXC) and stock trading (via Alpaca/yfinance).
Runs the trading engines, dashboard API, and optional Telegram bot.
"""

import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import Optional

from apex_crypto.core.logging import get_logger

logger = get_logger("main")


class ApexTradingSystem:
    """Main orchestrator for the APEX Trading System.

    Manages lifecycle of all subsystems: crypto trading, stock trading,
    data ingestion, analysis, strategy execution, risk management, and monitoring.
    """

    def __init__(self) -> None:
        self._running: bool = False
        self._config = None
        self._tasks: list[asyncio.Task] = []
        self._engine = None
        self._stock_engine = None

    async def initialize(self) -> None:
        """Initialize all system components."""
        logger.info("Initializing APEX Trading System")

        # Load .env file if present
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            self._load_env(env_path)
            logger.info("Loaded .env file")

        # Load configuration
        from apex_crypto.config.loader import Config
        self._config = Config()
        mode = self._config.get("system.mode", "paper")
        logger.info(f"System mode: {mode}")

        # Ensure data directories exist
        Path("./data").mkdir(parents=True, exist_ok=True)
        Path("./reports").mkdir(parents=True, exist_ok=True)

        # Initialize the crypto trading engine
        from apex_crypto.core.engine import TradingEngine
        engine_cfg = self._config._data.get("engine", {})
        self._engine = TradingEngine(engine_cfg, self._config._data)
        await self._engine.setup()

        # Initialize the stock trading engine (if enabled)
        stocks_cfg = self._config._data.get("stocks", {})
        if stocks_cfg.get("enabled", False):
            try:
                from apex_crypto.stocks.engine import StockTradingEngine
                self._stock_engine = StockTradingEngine(stocks_cfg, self._config._data)
                await self._stock_engine.setup()
                logger.info("Stock trading engine initialized")
            except Exception as exc:
                logger.warning("Stock engine initialization failed: %s — stocks disabled", exc)
                self._stock_engine = None
        else:
            logger.info("Stock trading is disabled in config")

        logger.info("All components initialized successfully")

    async def start(self) -> None:
        """Start the trading system main loop."""
        self._running = True
        logger.info("Starting APEX Trading System")

        # Get crypto asset lists
        tier1 = self._config.get("assets.tier1", [])
        tier2 = self._config.get("assets.tier2", [])
        all_crypto_symbols = tier1 + tier2

        # Timeframes for crypto
        timeframes = ["15m", "1h", "4h", "1d"]

        mode = self._config.get("system.mode", "paper")

        # Start crypto trading engine
        engine_task = asyncio.create_task(
            self._engine.run(all_crypto_symbols, timeframes),
            name="crypto_engine",
        )
        self._tasks.append(engine_task)

        # Start stock trading engine (if initialized)
        stock_symbols_count = 0
        stock_strategies_count = 0
        if self._stock_engine:
            stock_task = asyncio.create_task(
                self._stock_engine.run(),
                name="stock_engine",
            )
            self._tasks.append(stock_task)
            stocks_cfg = self._config._data.get("stocks", {}).get("assets", {})
            stock_symbols_count = len(stocks_cfg.get("tier1", [])) + len(stocks_cfg.get("tier2", []))
            stock_strategies_count = len(self._stock_engine._strategies)

        # Start dashboard
        dashboard_task = asyncio.create_task(
            self._run_dashboard(),
            name="dashboard",
        )
        self._tasks.append(dashboard_task)

        logger.info("=" * 60)
        logger.info("  APEX TRADING SYSTEM — RUNNING")
        logger.info(f"  Mode: {mode}")
        logger.info(f"  Crypto Symbols: {len(all_crypto_symbols)}")
        logger.info(f"  Crypto Strategies: {len(self._engine._strategies)}")
        if self._stock_engine:
            stock_mode = self._config._data.get("stocks", {}).get("mode", "paper")
            logger.info(f"  Stock Symbols: {stock_symbols_count}")
            logger.info(f"  Stock Strategies: {stock_strategies_count}")
            logger.info(f"  Stock Mode: {stock_mode}")
        logger.info(f"  Dashboard: http://0.0.0.0:8000")
        logger.info("=" * 60)

        # Wait for tasks
        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            pass

    async def _run_dashboard(self) -> None:
        """Run the FastAPI dashboard server."""
        try:
            import uvicorn
            from apex_crypto.dashboard.app import create_app

            app = create_app(self._config._data, self._engine)

            config = uvicorn.Config(
                app,
                host="0.0.0.0",
                port=int(self._config.get("dashboard.port", 8000)),
                log_level="warning",
            )
            server = uvicorn.Server(config)
            await server.serve()
        except ImportError as exc:
            logger.warning("Dashboard not available (missing deps): %s", exc)
        except Exception as exc:
            logger.error("Dashboard error: %s", exc)

    async def stop(self) -> None:
        """Gracefully stop the trading system."""
        logger.info("Stopping APEX Trading System")
        self._running = False

        if self._engine:
            try:
                await asyncio.wait_for(self._engine.stop(), timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning("Crypto engine shutdown timed out after 30s")
            except Exception as exc:
                logger.error("Error during crypto engine shutdown: %s", exc)

        if self._stock_engine:
            try:
                await asyncio.wait_for(self._stock_engine.stop(), timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning("Stock engine shutdown timed out after 30s")
            except Exception as exc:
                logger.error("Error during stock engine shutdown: %s", exc)

        for task in self._tasks:
            task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        logger.info("System stopped gracefully")

    @staticmethod
    def _load_env(path: Path) -> None:
        """Load environment variables from a .env file."""
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ.setdefault(key, value)


async def main() -> None:
    """Main async entry point."""
    system = ApexTradingSystem()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(system.stop()))

    try:
        await system.initialize()
        await system.start()
    except KeyboardInterrupt:
        pass
    finally:
        await system.stop()


if __name__ == "__main__":
    asyncio.run(main())
