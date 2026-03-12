"""Download historical OHLCV data from MEXC for all configured assets.

Usage:
    python -m apex_crypto.scripts.download_history
    python -m apex_crypto.scripts.download_history --symbols BTC/USDT ETH/USDT
    python -m apex_crypto.scripts.download_history --timeframes 1h 4h 1d
"""

import asyncio
import argparse
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from apex_crypto.core.logging import get_logger
from apex_crypto.config.loader import Config
from apex_crypto.core.data.storage import StorageManager
from apex_crypto.core.data.ingestion import MarketDataManager

logger = get_logger("download_history")


async def run_download(
    symbols: Optional[list[str]] = None,
    timeframes: Optional[list[str]] = None,
) -> None:
    """Execute the historical data download.

    Args:
        symbols: Optional list of symbols to download. Uses config if None.
        timeframes: Optional list of timeframes. Uses config if None.
    """
    config = Config.load()

    if symbols is None:
        tier1 = config.get("assets.tier1", [])
        tier2 = config.get("assets.tier2", [])
        symbols = tier1 + tier2

    if timeframes is None:
        timeframes = config.get_timeframes()

    logger.info(
        f"Starting historical download: {len(symbols)} symbols, "
        f"{len(timeframes)} timeframes"
    )

    storage = StorageManager(
        timescaledb_url=config.get("data.timescaledb_url"),
        sqlite_path=config.get("data.sqlite_path"),
        redis_url=config.get("data.redis_url"),
    )

    try:
        data_manager = MarketDataManager(
            config=config._data,
            storage=storage,
        )

        results = await data_manager.download_all_history(symbols, timeframes)

        total_rows = 0
        for symbol, tf_data in results.items():
            for tf, count in tf_data.items():
                total_rows += count
                logger.info(f"  {symbol} {tf}: {count:,} candles")

        logger.info(f"Download complete. Total: {total_rows:,} candles stored.")

    finally:
        storage.close()


def main() -> None:
    """CLI entry point for historical data download."""
    parser = argparse.ArgumentParser(
        description="Download historical OHLCV data from MEXC"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Symbols to download (e.g., BTC/USDT ETH/USDT)",
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        help="Timeframes to download (e.g., 1h 4h 1d)",
    )

    args = parser.parse_args()
    asyncio.run(run_download(args.symbols, args.timeframes))


if __name__ == "__main__":
    main()
