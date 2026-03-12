"""WebSocket connection manager for the APEX Crypto Trading System dashboard.

Manages WebSocket connections for real-time updates, supporting
broadcast to all connected clients and personal messages to
individual connections.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from apex_crypto.core.logging import get_logger, log_with_data

logger = get_logger("dashboard.websocket_manager")


class WebSocketManager:
    """Manages WebSocket connections for real-time dashboard updates.

    Maintains a set of active WebSocket connections and provides
    methods to broadcast data to all clients or send messages to
    individual connections.  Thread-safe through asyncio's event
    loop model.

    Attributes:
        _active_connections: Set of currently connected WebSocket instances.
    """

    def __init__(self) -> None:
        """Initialize the WebSocket manager with an empty connection set."""
        self._active_connections: set[WebSocket] = set()
        log_with_data(
            logger,
            "info",
            "WebSocketManager initialised",
            {"initial_connections": 0},
        )

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a new WebSocket connection.

        Accepts the incoming WebSocket handshake and adds the
        connection to the active set.

        Args:
            websocket: The incoming WebSocket connection to accept.
        """
        await websocket.accept()
        self._active_connections.add(websocket)
        log_with_data(
            logger,
            "info",
            "WebSocket client connected",
            {
                "client": _client_id(websocket),
                "total_connections": len(self._active_connections),
            },
        )

    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection from the active set.

        Silently ignores connections that are not in the active set
        (e.g. if already removed by a prior error handler).

        Args:
            websocket: The WebSocket connection to remove.
        """
        self._active_connections.discard(websocket)
        log_with_data(
            logger,
            "info",
            "WebSocket client disconnected",
            {
                "client": _client_id(websocket),
                "total_connections": len(self._active_connections),
            },
        )

    async def broadcast(self, data: dict) -> None:
        """Send a JSON message to all connected WebSocket clients.

        Connections that fail during send are automatically removed
        from the active set.  Sends are dispatched concurrently to
        all clients.

        Args:
            data: Dictionary payload to serialise as JSON and send.
        """
        if not self._active_connections:
            return

        message = json.dumps(data, default=str)
        stale_connections: list[WebSocket] = []

        send_tasks = []
        for connection in self._active_connections.copy():
            send_tasks.append(
                self._safe_send(connection, message, stale_connections)
            )

        if send_tasks:
            await asyncio.gather(*send_tasks)

        for stale in stale_connections:
            self._active_connections.discard(stale)

        if stale_connections:
            log_with_data(
                logger,
                "warning",
                "Removed stale WebSocket connections during broadcast",
                {
                    "removed_count": len(stale_connections),
                    "remaining_connections": len(self._active_connections),
                },
            )

    async def send_personal(self, websocket: WebSocket, data: dict) -> None:
        """Send a JSON message to a single WebSocket client.

        If the send fails the connection is removed from the active
        set.

        Args:
            websocket: The target WebSocket connection.
            data: Dictionary payload to serialise as JSON and send.
        """
        message = json.dumps(data, default=str)
        try:
            await websocket.send_text(message)
        except (WebSocketDisconnect, RuntimeError, ConnectionError) as exc:
            log_with_data(
                logger,
                "warning",
                "Failed to send personal message, removing connection",
                {
                    "client": _client_id(websocket),
                    "error": str(exc),
                },
            )
            self._active_connections.discard(websocket)

    def get_connection_count(self) -> int:
        """Return the number of currently active WebSocket connections.

        Returns:
            Integer count of active connections.
        """
        return len(self._active_connections)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _safe_send(
        websocket: WebSocket,
        message: str,
        stale_list: list[WebSocket],
    ) -> None:
        """Attempt to send a text message, appending to stale_list on failure.

        Args:
            websocket: Target WebSocket connection.
            message: Pre-serialised JSON string to send.
            stale_list: Mutable list to which failed connections are
                appended for later cleanup.
        """
        try:
            await websocket.send_text(message)
        except (WebSocketDisconnect, RuntimeError, ConnectionError) as exc:
            log_with_data(
                logger,
                "debug",
                "WebSocket send failed, marking connection as stale",
                {
                    "client": _client_id(websocket),
                    "error": str(exc),
                },
            )
            stale_list.append(websocket)


def _client_id(websocket: WebSocket) -> str:
    """Derive a human-readable identifier for a WebSocket client.

    Args:
        websocket: The WebSocket connection.

    Returns:
        A string identifier based on the client's address, or
        ``"unknown"`` if the address is unavailable.
    """
    try:
        client = websocket.client
        if client is not None:
            return f"{client.host}:{client.port}"
    except Exception:
        pass
    return "unknown"
