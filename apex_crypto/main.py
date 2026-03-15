"""APEX Trading System — Main Entry Point.

Initializes all system components and starts the autonomous trading loop.
Supports both crypto trading (via MEXC) and stock trading (via Alpaca/yfinance).
Runs the trading engines, dashboard API, and optional Telegram bot.

Usage:
    # Run indefinitely (default):
    python -m apex_crypto.main

    # Run for 8 hours then auto-stop and generate report:
    python -m apex_crypto.main --duration 8h

    # Run for 30 minutes:
    python -m apex_crypto.main --duration 30m

    # Generate status report from existing data:
    python -m apex_crypto.main --report
"""

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from apex_crypto.core.logging import get_logger

logger = get_logger("main")


def _parse_duration(s: str) -> float:
    """Parse a duration string like '8h', '30m', '2d' into seconds."""
    s = s.strip().lower()
    multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    if s[-1] in multipliers:
        return float(s[:-1]) * multipliers[s[-1]]
    return float(s)


class ApexTradingSystem:
    """Main orchestrator for the APEX Trading System.

    Manages lifecycle of all subsystems: crypto trading, stock trading,
    data ingestion, analysis, strategy execution, risk management, and monitoring.
    """

    def __init__(self, duration_seconds: Optional[float] = None) -> None:
        self._running: bool = False
        self._config = None
        self._tasks: list[asyncio.Task] = []
        self._engine = None
        self._stock_engine = None
        self._duration_seconds = duration_seconds
        self._start_time: Optional[float] = None

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
        self._start_time = time.monotonic()
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

        # Start duration watchdog (if set)
        if self._duration_seconds:
            duration_task = asyncio.create_task(
                self._duration_watchdog(),
                name="duration_watchdog",
            )
            self._tasks.append(duration_task)

        duration_str = f"{self._duration_seconds / 3600:.1f}h" if self._duration_seconds else "unlimited"
        logger.info("=" * 60)
        logger.info("  APEX TRADING SYSTEM — RUNNING")
        logger.info(f"  Mode: {mode}")
        logger.info(f"  Duration: {duration_str}")
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

    async def _duration_watchdog(self) -> None:
        """Auto-stop the system after the configured duration."""
        await asyncio.sleep(self._duration_seconds)
        elapsed_h = self._duration_seconds / 3600
        logger.info(f"Duration limit reached ({elapsed_h:.1f}h) — stopping system and generating report")
        await self.generate_report()
        await self.stop()

    async def _run_dashboard(self) -> None:
        """Run the FastAPI dashboard server."""
        try:
            import uvicorn
            from apex_crypto.dashboard.app import create_app

            port = int(self._config.get("dashboard.port", 8000))
            app = create_app(self._config._data, self._engine)

            config = uvicorn.Config(
                app,
                host="0.0.0.0",
                port=port,
                log_level="warning",
            )
            server = uvicorn.Server(config)
            await server.serve()
        except ImportError as exc:
            logger.warning("Dashboard not available (missing deps): %s", exc)
        except SystemExit:
            logger.warning("Dashboard failed to start (port %s likely in use) — trading continues without dashboard", port)
        except Exception as exc:
            logger.error("Dashboard error: %s — trading continues without dashboard", exc)

    async def generate_report(self) -> str:
        """Generate a status report of the trading session."""
        report_dir = Path("./reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"session_report_{timestamp}.txt"

        lines = []
        lines.append("=" * 60)
        lines.append("  APEX TRADING SYSTEM — SESSION REPORT")
        lines.append(f"  Generated: {datetime.now(timezone.utc).isoformat()}")
        lines.append("=" * 60)

        # Duration
        if self._start_time:
            elapsed = time.monotonic() - self._start_time
            hours = elapsed / 3600
            lines.append(f"\n  Session Duration: {hours:.2f} hours")

        mode = self._config.get("system.mode", "paper") if self._config else "unknown"
        lines.append(f"  Trading Mode: {mode}")

        # Engine stats
        if self._engine:
            broker = getattr(self._engine, "_broker", None)
            if broker:
                # Paper trading balance
                paper_bal = getattr(broker, "_paper_balance", None)
                if paper_bal:
                    lines.append("\n--- PAPER TRADING BALANCE ---")
                    lines.append(f"  Total USDT:  ${paper_bal.get('total_usdt', 0):,.2f}")
                    lines.append(f"  Free USDT:   ${paper_bal.get('free_usdt', 0):,.2f}")
                    lines.append(f"  In Positions: ${paper_bal.get('used_usdt', 0):,.2f}")

                # Paper positions
                paper_pos = getattr(broker, "_paper_positions", {})
                if paper_pos:
                    lines.append(f"\n--- OPEN POSITIONS ({len(paper_pos)}) ---")
                    for sym, pos in paper_pos.items():
                        side = pos.get("side", "?")
                        size = pos.get("contracts", pos.get("amount", 0))
                        entry = pos.get("entryPrice", 0)
                        lines.append(f"  {sym}: {side} {size} @ ${entry:,.4f}")
                else:
                    lines.append("\n--- OPEN POSITIONS: None ---")

                # Paper orders history
                paper_orders = getattr(broker, "_paper_orders", {})
                closed_orders = {k: v for k, v in paper_orders.items() if v.get("status") == "closed"}
                lines.append(f"\n--- COMPLETED TRADES: {len(closed_orders)} ---")
                total_pnl = 0.0
                for oid, order in closed_orders.items():
                    sym = order.get("symbol", "?")
                    side = order.get("side", "?")
                    filled = order.get("filled", 0)
                    price = order.get("average", order.get("price", 0))
                    lines.append(f"  {sym}: {side} {filled} @ ${price:,.4f}")

            # Strategy stats
            strategies = getattr(self._engine, "_strategies", [])
            if strategies:
                lines.append(f"\n--- ACTIVE STRATEGIES ({len(strategies)}) ---")
                for strat in strategies:
                    name = getattr(strat, "name", type(strat).__name__)
                    lines.append(f"  - {name}")

            # Cycle count
            cycle_count = getattr(self._engine, "_cycle_count", 0)
            lines.append(f"\n  Total Trading Cycles: {cycle_count}")

        lines.append("\n" + "=" * 60)
        lines.append("  END OF REPORT")
        lines.append("=" * 60)

        report_text = "\n".join(lines)

        # Write to file
        report_path.write_text(report_text)
        logger.info(f"Session report saved to {report_path}")

        # Also write JSON summary
        json_path = report_dir / f"session_report_{timestamp}.json"
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "duration_hours": (time.monotonic() - self._start_time) / 3600 if self._start_time else 0,
        }
        if self._engine:
            broker = getattr(self._engine, "_broker", None)
            if broker:
                summary["paper_balance"] = getattr(broker, "_paper_balance", {})
                summary["open_positions"] = len(getattr(broker, "_paper_positions", {}))
                summary["completed_trades"] = len(
                    {k: v for k, v in getattr(broker, "_paper_orders", {}).items()
                     if v.get("status") == "closed"}
                )
            summary["strategies"] = len(getattr(self._engine, "_strategies", []))
            summary["cycles"] = getattr(self._engine, "_cycle_count", 0)

        json_path.write_text(json.dumps(summary, indent=2, default=str))
        logger.info(f"JSON report saved to {json_path}")

        # Print to console
        print(report_text)
        return report_text

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="APEX Crypto Trading System")
    parser.add_argument(
        "--duration",
        type=str,
        default=None,
        help="Auto-stop after duration (e.g. 8h, 30m, 2d). Default: run indefinitely.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate a status report from existing data and exit.",
    )
    parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        default=None,
        help="Override trading mode (paper or live).",
    )
    return parser.parse_args()


async def main() -> None:
    """Main async entry point."""
    args = parse_args()

    # Override mode via CLI if specified
    if args.mode:
        os.environ["APEX_MODE_OVERRIDE"] = args.mode

    duration = _parse_duration(args.duration) if args.duration else None
    system = ApexTradingSystem(duration_seconds=duration)

    if args.report:
        await system.initialize()
        await system.generate_report()
        return

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
