"""CLI runner for backtesting strategies.

Usage:
    python -m apex_crypto.scripts.run_backtest --assets BTC/USDT ETH/USDT
    python -m apex_crypto.scripts.run_backtest --all
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from apex_crypto.core.logging import get_logger

logger = get_logger("run_backtest")


def main() -> None:
    """CLI entry point for backtesting."""
    parser = argparse.ArgumentParser(description="Run APEX backtester")
    parser.add_argument(
        "--assets",
        nargs="+",
        help="Assets to backtest (e.g., BTC/USDT ETH/USDT)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Backtest all configured assets",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        help="Run only a specific strategy",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./reports",
        help="Output directory for reports",
    )

    args = parser.parse_args()

    logger.info("Backtest engine will be implemented in Phase E")
    logger.info(f"Assets: {args.assets or 'all'}")
    logger.info(f"Strategy: {args.strategy or 'all'}")
    logger.info(f"Output: {args.output}")


if __name__ == "__main__":
    main()
