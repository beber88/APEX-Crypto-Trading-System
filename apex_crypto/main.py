"""APEX Crypto Trading System — Main Entry Point.

Initializes all system components and starts the autonomous trading loop.
Runs the trading engine, dashboard API, and optional Telegram bot.
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
    """Main orchestrator for the APEX Crypto Trading System.

    Manages lifecycle of all subsystems: data ingestion, analysis,
    strategy execution, risk management, and monitoring.
    """

    def __init__(self) -> None:
        self._running: bool = False
        self._config = None
        self._tasks: list[asyncio.Task] = []
        self._engine = None
        self._telegram_bot = None

    async def initialize(self) -> None:
        """Initialize all system components."""
        logger.info("Initializing APEX Crypto Trading System")

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

        # Initialize the trading engine (replaces old storage/stream/data managers)
        from apex_crypto.core.engine import TradingEngine
        engine_cfg = self._config._data.get("engine", {})
        self._engine = TradingEngine(engine_cfg, self._config._data)
        await self._engine.setup()

        # Initialize Telegram bot if enabled
        telegram_cfg = self._config._data.get("telegram", {})
        if telegram_cfg.get("enabled", False):
            try:
                from apex_crypto.telegram.bot import ApexTelegramBot
                self._telegram_bot = ApexTelegramBot(config=telegram_cfg)
                self._telegram_bot.set_system(
                    get_portfolio_stats=self._get_portfolio_stats,
                    get_open_positions=self._get_open_positions,
                    get_regimes=self._get_regimes,
                )
                logger.info("Telegram bot initialized")
            except Exception as exc:
                logger.warning("Telegram bot not available: %s", exc)
                self._telegram_bot = None

        logger.info("All components initialized successfully")

    async def start(self) -> None:
        """Start the trading system main loop."""
        self._running = True
        logger.info("Starting APEX Crypto Trading System")

        # Get asset lists
        tier1 = self._config.get("assets.tier1", [])
        tier2 = self._config.get("assets.tier2", [])
        all_symbols = tier1 + tier2

        # Use all configured timeframes from config.yaml
        # Includes lower TFs needed by strategies (SMC→15m, scalping→1m/3m)
        timeframes = self._config.get_timeframes()
        if not timeframes:
            timeframes = ["1m", "3m", "5m", "15m", "1h", "4h", "1d"]
        logger.info(f"Timeframes: {timeframes}")

        mode = self._config.get("system.mode", "paper")

        # Start trading engine
        engine_task = asyncio.create_task(
            self._engine.run(all_symbols, timeframes),
            name="trading_engine",
        )
        self._tasks.append(engine_task)

        # Start dashboard
        dashboard_task = asyncio.create_task(
            self._run_dashboard(),
            name="dashboard",
        )
        self._tasks.append(dashboard_task)

        # Start Telegram bot
        if self._telegram_bot:
            try:
                await self._telegram_bot.start()
                logger.info("Telegram bot started")
            except Exception as exc:
                logger.warning("Telegram bot start failed: %s", exc)

        logger.info("=" * 60)
        logger.info("  APEX CRYPTO TRADING SYSTEM — RUNNING")
        logger.info(f"  Mode: {mode}")
        logger.info(f"  Symbols: {len(all_symbols)}")
        logger.info(f"  Strategies: {len(self._engine._strategies)}")
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
        logger.info("Stopping APEX Crypto Trading System")
        self._running = False

        if self._telegram_bot:
            try:
                await self._telegram_bot.stop()
            except Exception:
                pass

        if self._engine:
            await self._engine.stop()

        for task in self._tasks:
            task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        logger.info("System stopped gracefully")

    # ------------------------------------------------------------------
    # Telegram callback methods
    # ------------------------------------------------------------------

    def _get_portfolio_stats(self) -> dict:
        """Return portfolio stats for Telegram /status command."""
        if not self._engine:
            return {}
        state = self._engine.get_state()
        mode = self._config.get("system.mode", "paper") if self._config else "unknown"
        daily_stats = state.get("daily_stats", {})
        equity_stats = state.get("equity_stats", {})
        positions = state.get("open_positions", [])

        # Realized P&L from closed trades today
        realized_pnl_pct = daily_stats.get("daily_pnl_pct", 0.0)

        # Unrealized P&L from open positions
        total_unrealized = sum(p.get("unrealized_pnl", 0.0) for p in positions)
        equity = equity_stats.get("current_equity", 0.0)

        # Total daily P&L = realized + unrealized
        total_pnl = total_unrealized  # dollar value from unrealized
        total_pnl_pct = realized_pnl_pct
        if equity > 0 and total_unrealized != 0:
            total_pnl_pct += (total_unrealized / equity) * 100

        return {
            "total_equity": equity,
            "daily_pnl": total_pnl,
            "daily_pnl_pct": total_pnl_pct,
            "open_positions_count": len(positions),
            "current_drawdown_pct": equity_stats.get("current_drawdown_pct", 0.0),
            "mode": mode,
        }

    def _get_open_positions(self) -> list:
        """Return open positions for Telegram /positions command."""
        if not self._engine:
            return []
        state = self._engine.get_state()
        return state.get("open_positions", [])

    def _get_regimes(self) -> dict:
        """Return current regimes for Telegram /regime command."""
        if not self._engine:
            return {}
        state = self._engine.get_state()
        return state.get("current_regimes", {})

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
