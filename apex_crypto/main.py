"""APEX Crypto Trading System — Main Entry Point.

Initializes all system components and starts the autonomous trading loop.
"""

import asyncio
import signal
import sys
import os
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
        """Initialize the trading system."""
        self._running: bool = False
        self._config: Optional[dict] = None
        self._tasks: list[asyncio.Task] = []

    async def initialize(self) -> None:
        """Initialize all system components.

        Loads configuration, connects to databases, initializes
        exchange connections, and prepares all subsystems.
        """
        logger.info("Initializing APEX Crypto Trading System")

        # Load configuration
        from apex_crypto.config.loader import Config
        self._config = Config.load()
        mode = self._config.get("system.mode", "paper")
        logger.info(f"System mode: {mode}")

        # Initialize storage
        from apex_crypto.core.data.storage import StorageManager
        self._storage = StorageManager(
            timescaledb_url=self._config.get("data.timescaledb_url"),
            sqlite_path=self._config.get("data.sqlite_path"),
            redis_url=self._config.get("data.redis_url"),
        )

        # Initialize data ingestion
        from apex_crypto.core.data.ingestion import MarketDataManager
        self._data_manager = MarketDataManager(
            config=self._config._data,
            storage=self._storage,
        )

        # Initialize streaming
        from apex_crypto.core.data.streaming import MarketStreamManager
        self._stream_manager = MarketStreamManager(
            config=self._config._data,
            storage=self._storage,
        )

        # Initialize alternative data
        from apex_crypto.core.data.alt_data import AlternativeDataManager
        self._alt_data = AlternativeDataManager(
            config=self._config._data,
            storage=self._storage,
        )

        logger.info("All components initialized successfully")

    async def start(self) -> None:
        """Start the trading system main loop.

        Begins data streaming, analysis pipeline, and trade execution.
        """
        self._running = True
        logger.info("Starting APEX Crypto Trading System")

        # Get asset lists
        tier1 = self._config.get("assets.tier1", [])
        tier2 = self._config.get("assets.tier2", [])
        all_symbols = tier1 + tier2

        # Refresh latest data
        all_timeframes = self._config.get_timeframes()
        self._tasks.append(
            asyncio.create_task(
                self._data_manager.refresh_latest(all_symbols, all_timeframes)
            )
        )

        # Start WebSocket streams
        self._tasks.append(
            asyncio.create_task(
                self._stream_manager.start(all_symbols)
            )
        )

        # Main loop
        while self._running:
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                break

    async def stop(self) -> None:
        """Gracefully stop the trading system."""
        logger.info("Stopping APEX Crypto Trading System")
        self._running = False

        for task in self._tasks:
            task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        if hasattr(self, "_stream_manager"):
            await self._stream_manager.stop()

        if hasattr(self, "_storage"):
            self._storage.close()

        logger.info("System stopped gracefully")


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
