"""Dashboard application factory for the APEX Crypto Trading System.

Creates and configures the FastAPI app, wiring it to the live trading engine
so that dashboard endpoints can serve real-time data.
"""

from __future__ import annotations

import time
from typing import Any, Optional


class EngineDataStore:
    """Adapter that exposes trading engine state as attributes
    expected by the dashboard routes.

    The routes in ``routes.py`` access ``app.state.data_store`` using
    ``getattr(store, "current_equity", 0.0)`` etc. This class bridges
    the engine's ``get_state()`` dict to those attribute lookups.
    """

    def __init__(self, engine) -> None:
        self._engine = engine
        self._trading_paused = False

    @property
    def mode(self) -> str:
        cfg = self._engine._full_config.get("system", {})
        return cfg.get("mode", "paper")

    @property
    def current_equity(self) -> float:
        return self._engine._equity_stats.get("current_equity", 0.0)

    @property
    def peak_equity(self) -> float:
        return self._engine._equity_stats.get("peak_equity", 0.0)

    @property
    def equity_curve(self) -> list[dict[str, Any]]:
        return [
            {"timestamp": time.time(), "value": self.current_equity}
        ]

    @property
    def open_positions(self) -> list[dict[str, Any]]:
        return list(self._engine._open_positions)

    @property
    def trade_history(self) -> list[dict[str, Any]]:
        return []

    @property
    def current_signals(self) -> list[dict[str, Any]]:
        return list(self._engine._current_signals)

    @property
    def current_regimes(self) -> dict[str, Any]:
        return dict(self._engine._current_regimes)

    @property
    def fear_greed(self) -> dict[str, Any]:
        return {"value": 50, "classification": "Neutral", "timestamp": time.time()}

    @property
    def funding_rates(self) -> dict[str, Any]:
        return {}

    @property
    def risk_metrics(self) -> dict[str, Any]:
        stats = self._engine._equity_stats
        daily = self._engine._daily_stats
        return {
            "drawdown_pct": stats.get("current_drawdown_pct", 0.0),
            "daily_loss_pct": daily.get("daily_pnl_pct", 0.0),
            "positions_count": len(self._engine._open_positions),
            "trades_today": daily.get("trades_today", 0),
            "consecutive_losses": daily.get("consecutive_losses", 0),
        }

    @property
    def performance_30d(self) -> dict[str, Any]:
        return {
            "sharpe_30d": 0.0,
            "win_rate_30d": 0.5,
            "profit_factor_30d": 0.0,
            "total_trades_30d": 0,
            "total_pnl_30d": 0.0,
        }

    def set_trading_paused(self, paused: bool) -> None:
        self._trading_paused = paused
        self._engine._running = not paused


def create_app(config: dict, engine) -> Any:
    """Create the FastAPI app with engine state injected.

    Args:
        config: Full system configuration dict.
        engine: The TradingEngine instance.

    Returns:
        Configured FastAPI application.
    """
    from apex_crypto.dashboard.api.routes import app

    # Inject the data store so routes can access engine state
    app.state.data_store = EngineDataStore(engine)

    return app
