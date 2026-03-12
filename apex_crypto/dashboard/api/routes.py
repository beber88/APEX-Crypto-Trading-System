"""FastAPI application with REST and WebSocket endpoints for the APEX dashboard.

Provides system status, equity data, positions, trade history, signals,
risk metrics, regime information, Fear & Greed index, funding rates,
performance stats, trading controls, Prometheus metrics, and real-time
WebSocket updates.

Authentication is handled via HTTP Basic Auth with credentials sourced
from the ``DASHBOARD_USER`` and ``DASHBOARD_PASSWORD`` environment
variables.  The ``/api/metrics`` endpoint is exempt from authentication
so that Prometheus scrapers can reach it without credentials.

CORS is enabled for all origins by default but can be narrowed via the
``CORS_ALLOWED_ORIGINS`` environment variable (comma-separated list).
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import time
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from apex_crypto.core.logging import get_logger, log_with_data
from apex_crypto.dashboard.api.websocket_manager import WebSocketManager

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = get_logger("dashboard.api.routes")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_VERSION: str = "1.0.0"
_SYSTEM_NAME: str = "APEX Crypto Trading System"
_WS_BROADCAST_INTERVAL: float = 30.0

# ---------------------------------------------------------------------------
# Module-level mutable state
# ---------------------------------------------------------------------------
_system_start_time: float = time.time()
_trading_paused: bool = False
_broadcast_task: Optional[asyncio.Task[None]] = None

# ---------------------------------------------------------------------------
# WebSocket manager singleton
# ---------------------------------------------------------------------------
ws_manager = WebSocketManager()

# ---------------------------------------------------------------------------
# Authentication helpers
# ---------------------------------------------------------------------------
_security = HTTPBasic(auto_error=False)

_DASHBOARD_USER: str = os.environ.get("DASHBOARD_USER", "admin")
_DASHBOARD_PASSWORD: str = os.environ.get("DASHBOARD_PASSWORD", "changeme")


async def _verify_credentials(
    credentials: Optional[HTTPBasicCredentials] = Depends(_security),
) -> Optional[str]:
    """Verify HTTP Basic Auth credentials from environment variables.

    Compares the supplied username and password against the values of
    ``DASHBOARD_USER`` and ``DASHBOARD_PASSWORD`` using constant-time
    comparison to prevent timing attacks.

    Args:
        credentials: HTTP Basic credentials extracted by FastAPI's
            ``HTTPBasic`` security scheme.

    Returns:
        The authenticated username when credentials are valid, or
        ``None`` if no credentials are required (should not happen
        when this dependency is active).

    Raises:
        HTTPException: 401 Unauthorized when credentials are missing
            or invalid.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Basic"},
        )

    username_ok = secrets.compare_digest(
        credentials.username.encode("utf-8"),
        _DASHBOARD_USER.encode("utf-8"),
    )
    password_ok = secrets.compare_digest(
        credentials.password.encode("utf-8"),
        _DASHBOARD_PASSWORD.encode("utf-8"),
    )

    if not (username_ok and password_ok):
        log_with_data(
            logger,
            "warning",
            "Failed authentication attempt",
            {"username": credentials.username},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )

    return credentials.username


# ---------------------------------------------------------------------------
# CORS configuration
# ---------------------------------------------------------------------------
def _cors_origins() -> list[str]:
    """Return the list of allowed CORS origins.

    Reads from the ``CORS_ALLOWED_ORIGINS`` environment variable.  When
    the variable is set to ``"*"`` (default) all origins are permitted.

    Returns:
        List of origin strings.
    """
    raw = os.environ.get("CORS_ALLOWED_ORIGINS", "*")
    if raw.strip() == "*":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]


# ---------------------------------------------------------------------------
# Data-store accessor
# ---------------------------------------------------------------------------
def _store(app: FastAPI) -> Any:
    """Return the runtime data store attached to ``app.state``.

    Args:
        app: The FastAPI application instance.

    Returns:
        The data store object, or ``None`` if not yet injected.
    """
    return getattr(app.state, "data_store", None)


# ---------------------------------------------------------------------------
# Background task helpers
# ---------------------------------------------------------------------------
async def _log_action_async(action: str, detail: dict[str, Any]) -> None:
    """Log a trading control action asynchronously.

    Designed to be scheduled via FastAPI's ``BackgroundTasks`` so that
    the HTTP response is not delayed by logging I/O.

    Args:
        action: Human-readable action description (e.g. ``"pause"``).
        detail: Structured data to attach to the log entry.
    """
    log_with_data(logger, "info", f"Background log: {action}", detail)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="APEX Crypto Trading System - Dashboard API",
    description="REST and WebSocket API for the APEX trading dashboard.",
    version=_VERSION,
)

# ---- CORS ----------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================================
# REST Endpoints
# ==========================================================================


@app.get("/api/status")
async def get_status(
    _user: Optional[str] = Depends(_verify_credentials),
) -> dict[str, Any]:
    """Return current system status information.

    Returns:
        Dictionary containing ``mode``, ``uptime_seconds``, ``version``,
        ``system_name``, and ``is_trading``.
    """
    store = _store(app)
    mode: str = "paper"
    if store is not None and hasattr(store, "mode"):
        mode = store.mode

    uptime = time.time() - _system_start_time

    log_with_data(logger, "debug", "Status endpoint called", {"mode": mode})

    return {
        "mode": mode,
        "uptime_seconds": round(uptime, 2),
        "version": _VERSION,
        "system_name": _SYSTEM_NAME,
        "is_trading": not _trading_paused,
    }


@app.get("/api/equity")
async def get_equity(
    _user: Optional[str] = Depends(_verify_credentials),
) -> dict[str, Any]:
    """Return equity curve data, current equity, peak, and drawdown.

    Returns:
        Dictionary with ``equity_curve`` (list of ``{timestamp, value}``
        dicts), ``current_equity``, ``peak_equity``, and
        ``drawdown_pct``.
    """
    store = _store(app)
    if store is None:
        return {
            "equity_curve": [],
            "current_equity": 0.0,
            "peak_equity": 0.0,
            "drawdown_pct": 0.0,
        }

    equity_curve: list[dict[str, Any]] = getattr(store, "equity_curve", [])
    current_equity: float = getattr(store, "current_equity", 0.0)
    peak_equity: float = getattr(store, "peak_equity", 0.0)

    if peak_equity > 0.0:
        drawdown_pct = ((peak_equity - current_equity) / peak_equity) * 100.0
    else:
        drawdown_pct = 0.0

    return {
        "equity_curve": [
            {"timestamp": point.get("timestamp", ""), "value": point.get("value", 0.0)}
            for point in equity_curve
        ],
        "current_equity": current_equity,
        "peak_equity": peak_equity,
        "drawdown_pct": round(drawdown_pct, 4),
    }


@app.get("/api/positions")
async def get_positions(
    _user: Optional[str] = Depends(_verify_credentials),
) -> dict[str, Any]:
    """Return all currently open positions.

    Returns:
        Dictionary with ``positions`` (list of position dicts containing
        ``symbol``, ``direction``, ``entry_price``, ``current_price``,
        ``unrealized_pnl``, ``stop_loss``, ``take_profit``,
        ``strategy``, ``signal_score``, and ``entry_time``).
    """
    store = _store(app)
    if store is None:
        return {"positions": []}

    raw_positions: list[dict[str, Any]] = getattr(store, "open_positions", [])

    positions = [
        {
            "symbol": pos.get("symbol", ""),
            "direction": pos.get("direction", ""),
            "entry_price": pos.get("entry_price", 0.0),
            "current_price": pos.get("current_price", 0.0),
            "unrealized_pnl": pos.get("unrealized_pnl", 0.0),
            "stop_loss": pos.get("stop_loss", 0.0),
            "take_profit": pos.get("take_profit", 0.0),
            "strategy": pos.get("strategy", ""),
            "signal_score": pos.get("signal_score", 0.0),
            "entry_time": pos.get("entry_time", ""),
        }
        for pos in raw_positions
    ]

    return {"positions": positions}


@app.get("/api/trades")
async def get_trades(
    _user: Optional[str] = Depends(_verify_credentials),
    limit: int = Query(50, ge=1, le=500, description="Number of trades to return"),
    offset: int = Query(0, ge=0, description="Number of trades to skip"),
    symbol: Optional[str] = Query(None, description="Filter by trading symbol"),
    strategy: Optional[str] = Query(None, description="Filter by strategy name"),
) -> dict[str, Any]:
    """Return paginated trade history with optional filters.

    Args:
        limit: Maximum number of trades to return per page.
        offset: Number of trades to skip from the beginning.
        symbol: Optional filter to restrict results to a specific symbol.
        strategy: Optional filter to restrict results to a specific strategy.

    Returns:
        Dictionary with ``trades`` (list of trade dicts), ``total``,
        ``limit``, and ``offset``.
    """
    store = _store(app)
    if store is None:
        return {
            "trades": [],
            "total": 0,
            "limit": limit,
            "offset": offset,
        }

    all_trades: list[dict[str, Any]] = getattr(store, "trade_history", [])

    # Apply filters
    filtered = all_trades
    if symbol:
        filtered = [t for t in filtered if t.get("symbol") == symbol]
    if strategy:
        filtered = [t for t in filtered if t.get("strategy") == strategy]

    total = len(filtered)
    page_trades = filtered[offset : offset + limit]

    return {
        "trades": page_trades,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@app.get("/api/signals")
async def get_signals(
    _user: Optional[str] = Depends(_verify_credentials),
) -> dict[str, Any]:
    """Return current trading signals for all monitored assets.

    Returns:
        Dictionary with ``signals`` (list of dicts containing
        ``symbol``, ``strategy``, ``score``, ``direction``,
        ``timeframe``, and ``timestamp``).
    """
    store = _store(app)
    if store is None:
        return {"signals": []}

    raw_signals: list[dict[str, Any]] = getattr(store, "current_signals", [])

    signals = [
        {
            "symbol": sig.get("symbol", ""),
            "strategy": sig.get("strategy", ""),
            "score": sig.get("score", 0.0),
            "direction": sig.get("direction", ""),
            "timeframe": sig.get("timeframe", ""),
            "timestamp": sig.get("timestamp", ""),
        }
        for sig in raw_signals
    ]

    return {"signals": signals}


@app.get("/api/risk")
async def get_risk(
    _user: Optional[str] = Depends(_verify_credentials),
) -> dict[str, Any]:
    """Return current risk metrics.

    Returns:
        Dictionary with ``drawdown_pct``, ``daily_loss_pct``,
        ``positions_count``, ``max_positions``, ``trades_today``,
        ``max_trades``, and ``consecutive_losses``.
    """
    store = _store(app)
    if store is None:
        return {
            "drawdown_pct": 0.0,
            "daily_loss_pct": 0.0,
            "positions_count": 0,
            "max_positions": 0,
            "trades_today": 0,
            "max_trades": 0,
            "consecutive_losses": 0,
        }

    risk_metrics: dict[str, Any] = getattr(store, "risk_metrics", {})

    return {
        "drawdown_pct": risk_metrics.get("drawdown_pct", 0.0),
        "daily_loss_pct": risk_metrics.get("daily_loss_pct", 0.0),
        "positions_count": risk_metrics.get("positions_count", 0),
        "max_positions": risk_metrics.get("max_positions", 0),
        "trades_today": risk_metrics.get("trades_today", 0),
        "max_trades": risk_metrics.get("max_trades", 0),
        "consecutive_losses": risk_metrics.get("consecutive_losses", 0),
    }


@app.get("/api/regimes")
async def get_regimes(
    _user: Optional[str] = Depends(_verify_credentials),
) -> dict[str, Any]:
    """Return the current market regime for each monitored asset.

    Returns:
        Dictionary with ``regimes`` (dict mapping symbol to
        ``{regime, confidence, timestamp}``).
    """
    store = _store(app)
    if store is None:
        return {"regimes": {}}

    raw_regimes: dict[str, Any] = getattr(store, "current_regimes", {})

    regimes: dict[str, dict[str, Any]] = {}
    for symbol, regime_data in raw_regimes.items():
        if isinstance(regime_data, dict):
            regimes[symbol] = {
                "regime": regime_data.get("regime", "unknown"),
                "confidence": regime_data.get("confidence", 0.0),
                "timestamp": regime_data.get("timestamp", ""),
            }
        else:
            regimes[symbol] = {
                "regime": str(regime_data),
                "confidence": 0.0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    return {"regimes": regimes}


@app.get("/api/fear-greed")
async def get_fear_greed(
    _user: Optional[str] = Depends(_verify_credentials),
) -> dict[str, Any]:
    """Return the current Fear and Greed index.

    Returns:
        Dictionary with ``value``, ``classification``, and ``timestamp``.
    """
    store = _store(app)
    if store is None:
        return {
            "value": 50,
            "classification": "Neutral",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    fg_data: dict[str, Any] = getattr(store, "fear_greed", {})

    return {
        "value": fg_data.get("value", 50),
        "classification": fg_data.get("classification", "Neutral"),
        "timestamp": fg_data.get(
            "timestamp", datetime.now(timezone.utc).isoformat()
        ),
    }


@app.get("/api/funding-rates")
async def get_funding_rates(
    _user: Optional[str] = Depends(_verify_credentials),
) -> dict[str, Any]:
    """Return current funding rates for perpetual contracts.

    Returns:
        Dictionary with ``rates`` (dict mapping symbol to
        ``{rate, predicted_rate, timestamp}``).
    """
    store = _store(app)
    if store is None:
        return {"rates": {}}

    raw_rates: dict[str, Any] = getattr(store, "funding_rates", {})

    rates: dict[str, dict[str, Any]] = {}
    for symbol, rate_data in raw_rates.items():
        if isinstance(rate_data, dict):
            rates[symbol] = {
                "rate": rate_data.get("rate", 0.0),
                "predicted_rate": rate_data.get("predicted_rate", 0.0),
                "timestamp": rate_data.get("timestamp", ""),
            }
        else:
            rates[symbol] = {
                "rate": float(rate_data) if rate_data is not None else 0.0,
                "predicted_rate": 0.0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    return {"rates": rates}


@app.get("/api/performance")
async def get_performance(
    _user: Optional[str] = Depends(_verify_credentials),
) -> dict[str, Any]:
    """Return rolling 30-day performance statistics.

    Returns:
        Dictionary with ``sharpe_30d``, ``win_rate_30d``,
        ``profit_factor_30d``, ``total_return``, and
        ``avg_r_multiple``.
    """
    store = _store(app)
    if store is None:
        return {
            "sharpe_30d": 0.0,
            "win_rate_30d": 0.0,
            "profit_factor_30d": 0.0,
            "total_return": 0.0,
            "avg_r_multiple": 0.0,
        }

    perf: dict[str, Any] = getattr(store, "performance_30d", {})

    return {
        "sharpe_30d": perf.get("sharpe_30d", 0.0),
        "win_rate_30d": perf.get("win_rate_30d", 0.0),
        "profit_factor_30d": perf.get("profit_factor_30d", 0.0),
        "total_return": perf.get("total_return", 0.0),
        "avg_r_multiple": perf.get("avg_r_multiple", 0.0),
    }


@app.post("/api/pause")
async def pause_trading(
    background_tasks: BackgroundTasks,
    _user: Optional[str] = Depends(_verify_credentials),
) -> dict[str, str]:
    """Pause all trading activity.

    Notifies the runtime data store (if present), broadcasts a system
    event to all WebSocket clients, and logs the action via a
    background task.

    Args:
        background_tasks: FastAPI background task scheduler.

    Returns:
        Dictionary with ``status`` set to ``"paused"``.
    """
    global _trading_paused
    _trading_paused = True

    log_with_data(
        logger,
        "warning",
        "Trading paused via dashboard API",
        {"paused_at": datetime.now(timezone.utc).isoformat()},
    )

    store = _store(app)
    if store is not None and hasattr(store, "set_trading_paused"):
        store.set_trading_paused(True)

    await ws_manager.broadcast({
        "type": "system_event",
        "event": "trading_paused",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    background_tasks.add_task(
        _log_action_async,
        "pause",
        {"paused_at": datetime.now(timezone.utc).isoformat()},
    )

    return {"status": "paused"}


@app.post("/api/resume")
async def resume_trading(
    background_tasks: BackgroundTasks,
    _user: Optional[str] = Depends(_verify_credentials),
) -> dict[str, str]:
    """Resume trading activity.

    Re-enables trading, notifies the runtime data store, broadcasts
    a system event, and logs via a background task.

    Args:
        background_tasks: FastAPI background task scheduler.

    Returns:
        Dictionary with ``status`` set to ``"resumed"``.
    """
    global _trading_paused
    _trading_paused = False

    log_with_data(
        logger,
        "info",
        "Trading resumed via dashboard API",
        {"resumed_at": datetime.now(timezone.utc).isoformat()},
    )

    store = _store(app)
    if store is not None and hasattr(store, "set_trading_paused"):
        store.set_trading_paused(False)

    await ws_manager.broadcast({
        "type": "system_event",
        "event": "trading_resumed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    background_tasks.add_task(
        _log_action_async,
        "resume",
        {"resumed_at": datetime.now(timezone.utc).isoformat()},
    )

    return {"status": "resumed"}


@app.get("/api/metrics", response_class=PlainTextResponse)
async def get_metrics() -> str:
    """Return Prometheus-compatible metrics in text exposition format.

    This endpoint is exempt from authentication so that Prometheus
    scrapers can reach it without credentials.

    Returns:
        Plain text string in Prometheus exposition format.
    """
    store = _store(app)
    uptime = time.time() - _system_start_time

    lines: list[str] = [
        "# HELP apex_uptime_seconds System uptime in seconds",
        "# TYPE apex_uptime_seconds gauge",
        f"apex_uptime_seconds {uptime:.2f}",
        "",
        "# HELP apex_trading_active Whether trading is active (1=active, 0=paused)",
        "# TYPE apex_trading_active gauge",
        f"apex_trading_active {0 if _trading_paused else 1}",
        "",
        "# HELP apex_websocket_connections Number of active WebSocket connections",
        "# TYPE apex_websocket_connections gauge",
        f"apex_websocket_connections {ws_manager.get_connection_count()}",
    ]

    if store is not None:
        current_equity: float = getattr(store, "current_equity", 0.0)
        risk_metrics: dict[str, Any] = getattr(store, "risk_metrics", {})
        open_positions: list[dict[str, Any]] = getattr(store, "open_positions", [])
        perf: dict[str, Any] = getattr(store, "performance_30d", {})

        lines.extend([
            "",
            "# HELP apex_equity_usdt Current portfolio equity in USDT",
            "# TYPE apex_equity_usdt gauge",
            f"apex_equity_usdt {current_equity:.4f}",
            "",
            "# HELP apex_drawdown_pct Current drawdown percentage",
            "# TYPE apex_drawdown_pct gauge",
            f'apex_drawdown_pct {risk_metrics.get("drawdown_pct", 0.0):.4f}',
            "",
            "# HELP apex_open_positions Number of open positions",
            "# TYPE apex_open_positions gauge",
            f"apex_open_positions {len(open_positions)}",
            "",
            "# HELP apex_daily_loss_pct Daily loss percentage",
            "# TYPE apex_daily_loss_pct gauge",
            f'apex_daily_loss_pct {risk_metrics.get("daily_loss_pct", 0.0):.4f}',
            "",
            "# HELP apex_positions_count Current position count",
            "# TYPE apex_positions_count gauge",
            f'apex_positions_count {risk_metrics.get("positions_count", 0)}',
            "",
            "# HELP apex_trades_today Number of trades executed today",
            "# TYPE apex_trades_today gauge",
            f'apex_trades_today {risk_metrics.get("trades_today", 0)}',
            "",
            "# HELP apex_consecutive_losses Current streak of consecutive losses",
            "# TYPE apex_consecutive_losses gauge",
            f'apex_consecutive_losses {risk_metrics.get("consecutive_losses", 0)}',
            "",
            "# HELP apex_sharpe_30d Rolling 30-day Sharpe ratio",
            "# TYPE apex_sharpe_30d gauge",
            f'apex_sharpe_30d {perf.get("sharpe_30d", 0.0):.4f}',
            "",
            "# HELP apex_win_rate_30d Rolling 30-day win rate",
            "# TYPE apex_win_rate_30d gauge",
            f'apex_win_rate_30d {perf.get("win_rate_30d", 0.0):.4f}',
            "",
            "# HELP apex_profit_factor_30d Rolling 30-day profit factor",
            "# TYPE apex_profit_factor_30d gauge",
            f'apex_profit_factor_30d {perf.get("profit_factor_30d", 0.0):.4f}',
            "",
            "# HELP apex_total_return Total return percentage",
            "# TYPE apex_total_return gauge",
            f'apex_total_return {perf.get("total_return", 0.0):.4f}',
            "",
            "# HELP apex_avg_r_multiple Average R-multiple",
            "# TYPE apex_avg_r_multiple gauge",
            f'apex_avg_r_multiple {perf.get("avg_r_multiple", 0.0):.4f}',
        ])

    lines.append("")
    return "\n".join(lines)


# ==========================================================================
# WebSocket Endpoint
# ==========================================================================


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time dashboard updates.

    After the connection is accepted, sends an update payload every
    30 seconds containing equity, positions, signals, and risk metrics.
    Also listens for incoming messages to handle keep-alive pings.

    Args:
        websocket: The incoming WebSocket connection.
    """
    await ws_manager.connect(websocket)
    try:
        # Send initial snapshot immediately upon connection
        initial_payload = _build_update_payload(store=_store(app))
        await ws_manager.send_personal(websocket, initial_payload)

        # Run periodic sender and message listener concurrently
        update_task = asyncio.create_task(
            _periodic_client_updates(websocket)
        )
        listen_task = asyncio.create_task(
            _listen_for_messages(websocket)
        )

        done, pending = await asyncio.wait(
            {update_task, listen_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    except WebSocketDisconnect:
        log_with_data(
            logger,
            "info",
            "WebSocket client disconnected normally",
            {"client": _ws_client_id(websocket)},
        )
    except Exception as exc:
        log_with_data(
            logger,
            "error",
            "WebSocket error",
            {"client": _ws_client_id(websocket), "error": str(exc)},
        )
    finally:
        await ws_manager.disconnect(websocket)


# ==========================================================================
# Startup / Shutdown Events
# ==========================================================================


@app.on_event("startup")
async def on_startup() -> None:
    """Initialise module state and start the WebSocket broadcast loop.

    Resets the system start time and launches a background task that
    periodically broadcasts updates to all connected WebSocket clients.
    """
    global _system_start_time, _broadcast_task
    _system_start_time = time.time()

    _broadcast_task = asyncio.create_task(_broadcast_loop())

    log_with_data(
        logger,
        "info",
        "Dashboard API started",
        {
            "version": _VERSION,
            "cors_origins": _cors_origins(),
            "auth_user": _DASHBOARD_USER,
            "ws_broadcast_interval": _WS_BROADCAST_INTERVAL,
        },
    )


@app.on_event("shutdown")
async def on_shutdown() -> None:
    """Cancel the broadcast task and notify connected WebSocket clients.

    Sends a shutdown system event to all connected clients before
    cancelling the background broadcast loop.
    """
    global _broadcast_task

    if _broadcast_task is not None:
        _broadcast_task.cancel()
        try:
            await _broadcast_task
        except asyncio.CancelledError:
            pass
        _broadcast_task = None

    await ws_manager.broadcast({
        "type": "system_event",
        "event": "shutdown",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    log_with_data(logger, "info", "Dashboard API shutting down", {})


# ==========================================================================
# Internal Helpers
# ==========================================================================


async def _broadcast_loop() -> None:
    """Continuously broadcast update payloads to all WebSocket clients.

    Runs indefinitely, sending the current state snapshot every
    ``_WS_BROADCAST_INTERVAL`` seconds.  Designed to be launched as
    an ``asyncio.Task`` during startup and cancelled during shutdown.
    """
    while True:
        await asyncio.sleep(_WS_BROADCAST_INTERVAL)
        try:
            payload = _build_update_payload(store=_store(app))
            await ws_manager.broadcast(payload)
        except Exception as exc:
            log_with_data(
                logger,
                "error",
                "Error during WebSocket broadcast",
                {"error": str(exc)},
            )


async def _periodic_client_updates(
    websocket: WebSocket,
    interval: float = _WS_BROADCAST_INTERVAL,
) -> None:
    """Push periodic updates to a single WebSocket client.

    This runs per-client inside the WebSocket handler so that each
    client receives updates even if the global broadcast misses
    timing due to connection churn.

    Args:
        websocket: The target WebSocket connection.
        interval: Seconds between update pushes.
    """
    while True:
        await asyncio.sleep(interval)
        payload = _build_update_payload(store=_store(app))
        try:
            await websocket.send_text(json.dumps(payload, default=str))
        except (WebSocketDisconnect, RuntimeError, ConnectionError):
            break


async def _listen_for_messages(websocket: WebSocket) -> None:
    """Listen for incoming WebSocket messages.

    Handles ``"ping"`` messages by replying with ``{"type": "pong"}``.
    Any other messages are logged and discarded.

    Args:
        websocket: The WebSocket connection to listen on.
    """
    while True:
        try:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            else:
                log_with_data(
                    logger,
                    "debug",
                    "Received WebSocket message",
                    {"client": _ws_client_id(websocket), "message": data[:200]},
                )
        except (WebSocketDisconnect, RuntimeError):
            break


def _build_update_payload(store: Any = None) -> dict[str, Any]:
    """Build the standard WebSocket update payload.

    Assembles a snapshot of the current system state including equity,
    open positions, active signals, and risk metrics.

    Args:
        store: The runtime data store, or ``None`` if not yet attached.

    Returns:
        Dictionary with ``type`` set to ``"update"`` and a ``data``
        sub-dictionary containing ``equity``, ``positions``,
        ``signals``, ``risk_metrics``, and ``timestamp``.
    """
    now = datetime.now(timezone.utc).isoformat()

    if store is None:
        return {
            "type": "update",
            "data": {
                "equity": 0.0,
                "positions": [],
                "signals": [],
                "risk_metrics": {},
                "timestamp": now,
            },
        }

    return {
        "type": "update",
        "data": {
            "equity": getattr(store, "current_equity", 0.0),
            "positions": getattr(store, "open_positions", []),
            "signals": getattr(store, "current_signals", []),
            "risk_metrics": getattr(store, "risk_metrics", {}),
            "timestamp": now,
        },
    }


def _ws_client_id(websocket: WebSocket) -> str:
    """Derive a human-readable identifier for a WebSocket client.

    Args:
        websocket: The WebSocket connection.

    Returns:
        String identifier in ``host:port`` format, or ``"unknown"``
        if the client address is unavailable.
    """
    try:
        client = websocket.client
        if client is not None:
            return f"{client.host}:{client.port}"
    except Exception:
        pass
    return "unknown"
