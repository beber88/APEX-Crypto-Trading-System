"""Automatic Telegram alert system for the APEX Crypto Trading System.

Sends real-time notifications for trade events, daily summaries, risk
warnings, system errors, and high-conviction signals.  All messages
are dispatched asynchronously with retry logic.

Uses python-telegram-bot v20+ async API exclusively.

Typical usage::

    from apex_crypto.telegram.alerts import AlertManager

    alerts = AlertManager(config=cfg["telegram"])
    await alerts.send_trade_opened(trade_dict)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional

from telegram import Bot
from telegram.error import (
    NetworkError,
    RetryAfter,
    TelegramError,
    TimedOut,
)

from apex_crypto.core.logging import get_logger, log_with_data

logger = get_logger("telegram.alerts")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_RETRIES: int = 3
_BASE_RETRY_DELAY: float = 2.0  # seconds

_DRAWDOWN_THRESHOLDS: list[float] = [5.0, 8.0, 12.0]


class AlertManager:
    """Sends automatic Telegram alerts for trading events and system status.

    Each public method corresponds to a specific alert type with a
    predefined format.  All methods use :meth:`send_message` internally,
    which provides retry logic with exponential back-off.

    Args:
        config: Dictionary containing ``bot_token`` and ``chat_id``
            from the ``telegram`` section of ``config.yaml``.
    """

    def __init__(self, config: dict) -> None:
        """Initialise the alert manager with Telegram credentials.

        Args:
            config: Must contain keys ``bot_token`` (str) and
                ``chat_id`` (int or str).
        """
        self._bot_token: str = config["bot_token"]
        self._chat_id: int = int(config["chat_id"])
        self._bot: Bot = Bot(token=self._bot_token)

        log_with_data(
            logger,
            "info",
            "AlertManager initialised",
            {"chat_id": self._chat_id},
        )

    # ------------------------------------------------------------------
    # Trade alerts
    # ------------------------------------------------------------------

    async def send_trade_opened(self, trade: dict) -> None:
        """Send an alert when a new trade is opened.

        Alert format example::

            LONG BTC/USDT | Entry: $67,500 | Size: 0.15 BTC ($10,125)
            | SL: $66,200 | TP1: $69,450 | Strategy: trend_momentum | Score: 82

        Args:
            trade: Dictionary with trade details. Expected keys:
                ``symbol``, ``direction``, ``entry_price``, ``size_units``,
                ``size_usdt``, ``stop_loss``, ``take_profit``,
                ``strategy``, ``score``.
        """
        symbol: str = trade.get("symbol", "???")
        direction: str = trade.get("direction", "long").upper()
        entry: float = trade.get("entry_price", 0.0)
        size_units: float = trade.get("size_units", 0.0)
        size_usdt: float = trade.get("size_usdt", 0.0)
        sl: float = trade.get("stop_loss", 0.0)
        tp: float = trade.get("take_profit", 0.0)
        strategy: str = trade.get("strategy", "unknown")
        score: int = int(trade.get("score", 0))

        # Extract base asset from symbol (e.g. "BTC" from "BTC/USDT")
        base_asset: str = symbol.split("/")[0] if "/" in symbol else symbol

        dir_emoji = "\U0001f7e2" if direction == "LONG" else "\U0001f534"

        text = (
            f"{dir_emoji} <b>{direction} {symbol}</b>\n"
            f"Entry: <b>${entry:,.2f}</b>\n"
            f"Size: {size_units:.4f} {base_asset} (${size_usdt:,.2f})\n"
            f"SL: ${sl:,.2f}\n"
            f"TP1: ${tp:,.2f}\n"
            f"Strategy: {strategy}\n"
            f"Score: <b>{score}</b>"
        )

        await self.send_message(text)

        log_with_data(
            logger,
            "info",
            "Trade opened alert sent",
            {
                "symbol": symbol,
                "direction": direction,
                "entry_price": entry,
                "score": score,
            },
        )

    async def send_trade_closed(self, trade: dict) -> None:
        """Send an alert when a trade is closed.

        Alert format example::

            CLOSED BTC/USDT LONG | Exit: $69,400 (TP1)
            | P&L: +$285.00 (+2.8%) | R: +1.5R | Hold: 8h 23m

        Args:
            trade: Dictionary with trade details. Expected keys:
                ``symbol``, ``direction``, ``exit_price``, ``exit_reason``,
                ``realized_pnl``, ``realized_pnl_pct``, ``r_multiple``,
                ``hold_duration_seconds``.
        """
        symbol: str = trade.get("symbol", "???")
        direction: str = trade.get("direction", "long").upper()
        exit_price: float = trade.get("exit_price", 0.0)
        exit_reason: str = trade.get("exit_reason", "manual")
        pnl: float = trade.get("realized_pnl", 0.0)
        pnl_pct: float = trade.get("realized_pnl_pct", 0.0)
        r_mult: float = trade.get("r_multiple", 0.0)
        hold_secs: int = int(trade.get("hold_duration_seconds", 0))

        pnl_sign = "+" if pnl >= 0 else ""
        r_sign = "+" if r_mult >= 0 else ""
        result_emoji = "\U0001f7e2" if pnl >= 0 else "\U0001f534"
        hold_str = _format_duration(hold_secs)

        text = (
            f"{result_emoji} <b>CLOSED {symbol} {direction}</b>\n"
            f"Exit: ${exit_price:,.2f} ({exit_reason})\n"
            f"P&L: <b>{pnl_sign}${pnl:,.2f} ({pnl_sign}{pnl_pct:.1f}%)</b>\n"
            f"R: {r_sign}{r_mult:.1f}R\n"
            f"Hold: {hold_str}"
        )

        await self.send_message(text)

        log_with_data(
            logger,
            "info",
            "Trade closed alert sent",
            {
                "symbol": symbol,
                "direction": direction,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "r_multiple": r_mult,
            },
        )

    # ------------------------------------------------------------------
    # Daily summary
    # ------------------------------------------------------------------

    async def send_daily_summary(self, stats: dict) -> None:
        """Send the daily performance summary (typically at 00:00 UTC).

        Alert format example::

            Daily Summary | P&L: +$450 (+1.2%) | Trades: 5W / 2L
            | Equity: $38,450 | Drawdown: 2.1%

        Args:
            stats: Dictionary with daily statistics. Expected keys:
                ``daily_pnl``, ``daily_pnl_pct``, ``wins``, ``losses``,
                ``total_equity``, ``current_drawdown_pct``,
                ``best_trade``, ``worst_trade``.
        """
        pnl: float = stats.get("daily_pnl", 0.0)
        pnl_pct: float = stats.get("daily_pnl_pct", 0.0)
        wins: int = int(stats.get("wins", 0))
        losses: int = int(stats.get("losses", 0))
        equity: float = stats.get("total_equity", 0.0)
        drawdown: float = stats.get("current_drawdown_pct", 0.0)
        best: str = stats.get("best_trade", "N/A")
        worst: str = stats.get("worst_trade", "N/A")

        pnl_sign = "+" if pnl >= 0 else ""
        day_emoji = "\U0001f7e2" if pnl >= 0 else "\U0001f534"
        today_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

        text = (
            f"\U0001f4ca <b>Daily Summary — {today_str}</b>\n"
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
            f"{day_emoji} P&L: <b>{pnl_sign}${pnl:,.2f} "
            f"({pnl_sign}{pnl_pct:.1f}%)</b>\n"
            f"\U0001f3af Trades: <b>{wins}W / {losses}L</b>\n"
            f"\U0001f4b0 Equity: <b>${equity:,.2f}</b>\n"
            f"\U0001f4c9 Drawdown: <b>{drawdown:.1f}%</b>\n"
            f"\u2b06\ufe0f Best: {best}\n"
            f"\u2b07\ufe0f Worst: {worst}"
        )

        await self.send_message(text)

        log_with_data(
            logger,
            "info",
            "Daily summary alert sent",
            {
                "date": today_str,
                "pnl": pnl,
                "wins": wins,
                "losses": losses,
                "equity": equity,
            },
        )

    # ------------------------------------------------------------------
    # Risk alerts
    # ------------------------------------------------------------------

    async def send_drawdown_warning(self, level: float, equity: float) -> None:
        """Send a drawdown warning at predefined thresholds (5%, 8%, 12%).

        Alert format example::

            DRAWDOWN WARNING: 8.0% from peak | Equity: $36,800 | Peak: $40,000

        Args:
            level: Current drawdown percentage from peak (e.g. 8.0).
            equity: Current portfolio equity in USDT.
        """
        peak: float = equity / (1.0 - level / 100.0) if level < 100.0 else equity

        if level >= 12.0:
            severity = "\U0001f534\U0001f534\U0001f534 CRITICAL"
        elif level >= 8.0:
            severity = "\U0001f534\U0001f534 HIGH"
        else:
            severity = "\U0001f534 ELEVATED"

        text = (
            f"\u26a0\ufe0f <b>DRAWDOWN WARNING: {level:.1f}% from peak</b>\n"
            f"Severity: {severity}\n"
            f"Equity: ${equity:,.2f}\n"
            f"Peak: ${peak:,.2f}\n"
            f"Threshold: {_next_threshold(level)}"
        )

        await self.send_message(text)

        log_with_data(
            logger,
            "warning",
            "Drawdown warning alert sent",
            {"level": level, "equity": equity, "peak": peak},
        )

    async def send_system_halt(self, reason: str) -> None:
        """Send a system halt notification.

        Alert format example::

            SYSTEM HALT — Max drawdown 12% reached.
            All trading paused. Manual intervention required.

        Args:
            reason: Human-readable reason for the halt (e.g.
                ``"Max drawdown 12% reached"``).
        """
        text = (
            f"\U0001f6d1 <b>SYSTEM HALT</b>\n\n"
            f"Reason: {reason}\n\n"
            f"All trading has been paused.\n"
            f"Manual intervention required.\n\n"
            f"Use /resume after reviewing the situation."
        )

        await self.send_message(text)

        log_with_data(
            logger,
            "error",
            "System halt alert sent",
            {"reason": reason},
        )

    # ------------------------------------------------------------------
    # Signal alerts
    # ------------------------------------------------------------------

    async def send_high_conviction_signal(self, signal: dict) -> None:
        """Send an alert for a high-conviction signal (score > 85).

        Alert format example::

            HIGH CONVICTION: BTC/USDT LONG | Score: 92
            | Strategies: trend_momentum (95), smc (88), swing (85)

        Args:
            signal: Dictionary with signal details. Expected keys:
                ``symbol``, ``direction``, ``score``,
                ``strategy_scores`` (dict mapping strategy name to score).
        """
        symbol: str = signal.get("symbol", "???")
        direction: str = signal.get("direction", "long").upper()
        score: int = int(signal.get("score", 0))
        strategy_scores: dict = signal.get("strategy_scores", {})

        strategies_str = ", ".join(
            f"{name} ({s})" for name, s in sorted(
                strategy_scores.items(), key=lambda x: x[1], reverse=True
            )
        )

        text = (
            f"\U0001f3af <b>HIGH CONVICTION: {symbol} {direction}</b>\n"
            f"Score: <b>{score}</b>\n"
            f"Strategies: {strategies_str}"
        )

        await self.send_message(text)

        log_with_data(
            logger,
            "info",
            "High conviction signal alert sent",
            {
                "symbol": symbol,
                "direction": direction,
                "score": score,
                "strategies": strategy_scores,
            },
        )

    # ------------------------------------------------------------------
    # System alerts
    # ------------------------------------------------------------------

    async def send_error(self, error: str, module: str) -> None:
        """Send a system error notification.

        Alert format example::

            ERROR in data_streaming: MEXC WebSocket disconnected.
            Reconnecting...

        Args:
            error: Human-readable error description.
            module: Name of the module where the error occurred.
        """
        text = (
            f"\u274c <b>ERROR in {module}</b>\n"
            f"{error}"
        )

        await self.send_message(text)

        log_with_data(
            logger,
            "error",
            "Error alert sent",
            {"module": module, "error": error},
        )

    async def send_regime_change(
        self, symbol: str, old_regime: str, new_regime: str
    ) -> None:
        """Send a regime change notification.

        Alert format example::

            REGIME CHANGE: BTC/USDT | RANGING -> STRONG_BULL

        Args:
            symbol: The asset symbol (e.g. ``"BTC/USDT"``).
            old_regime: Previous regime label.
            new_regime: New regime label.
        """
        old_emoji = _regime_emoji(old_regime)
        new_emoji = _regime_emoji(new_regime)

        text = (
            f"\U0001f504 <b>REGIME CHANGE: {symbol}</b>\n"
            f"{old_emoji} {old_regime} \u2192 {new_emoji} {new_regime}"
        )

        await self.send_message(text)

        log_with_data(
            logger,
            "info",
            "Regime change alert sent",
            {
                "symbol": symbol,
                "old_regime": old_regime,
                "new_regime": new_regime,
            },
        )

    # ------------------------------------------------------------------
    # Generic message sender with retry
    # ------------------------------------------------------------------

    async def send_message(
        self, text: str, parse_mode: str = "HTML"
    ) -> None:
        """Send a Telegram message with retry logic and exponential back-off.

        Retries up to ``_MAX_RETRIES`` times on transient network errors,
        timeouts, and Telegram rate-limit responses.  Non-retryable errors
        are logged and re-raised.

        Args:
            text: The message text to send (supports HTML formatting).
            parse_mode: Telegram parse mode.  Defaults to ``"HTML"``.

        Raises:
            TelegramError: If all retry attempts are exhausted or a
                non-retryable error occurs.
        """
        last_error: Optional[Exception] = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                await self._bot.send_message(
                    chat_id=self._chat_id,
                    text=text,
                    parse_mode=parse_mode,
                    disable_web_page_preview=True,
                )
                if attempt > 1:
                    log_with_data(
                        logger,
                        "info",
                        "Message sent after retry",
                        {"attempt": attempt},
                    )
                return

            except RetryAfter as exc:
                wait = exc.retry_after + 1
                log_with_data(
                    logger,
                    "warning",
                    "Rate limited by Telegram",
                    {
                        "retry_after": exc.retry_after,
                        "wait": wait,
                        "attempt": attempt,
                    },
                )
                last_error = exc
                await asyncio.sleep(wait)

            except TimedOut as exc:
                delay = _BASE_RETRY_DELAY * (2 ** (attempt - 1))
                log_with_data(
                    logger,
                    "warning",
                    "Telegram request timed out",
                    {"attempt": attempt, "retry_delay": delay},
                )
                last_error = exc
                await asyncio.sleep(delay)

            except NetworkError as exc:
                delay = _BASE_RETRY_DELAY * (2 ** (attempt - 1))
                log_with_data(
                    logger,
                    "warning",
                    "Network error sending Telegram message",
                    {
                        "attempt": attempt,
                        "retry_delay": delay,
                        "error": str(exc),
                    },
                )
                last_error = exc
                await asyncio.sleep(delay)

            except TelegramError as exc:
                # Non-retryable Telegram errors (bad request, unauthorized, etc.)
                log_with_data(
                    logger,
                    "error",
                    "Non-retryable Telegram error",
                    {"error": str(exc), "attempt": attempt},
                )
                raise

        # All retries exhausted
        log_with_data(
            logger,
            "error",
            "Failed to send Telegram message after all retries",
            {"max_retries": _MAX_RETRIES, "last_error": str(last_error)},
        )
        if last_error is not None:
            raise last_error


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _format_duration(total_seconds: int) -> str:
    """Format a duration in seconds as a human-readable string.

    Args:
        total_seconds: Duration in seconds.

    Returns:
        Formatted string like ``"8h 23m"`` or ``"2d 5h 10m"``.
    """
    if total_seconds < 0:
        total_seconds = 0

    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, _ = divmod(remainder, 60)

    parts: list[str] = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0 or days > 0:
        parts.append(f"{hours}h")
    parts.append(f"{minutes}m")

    return " ".join(parts)


def _regime_emoji(regime: str) -> str:
    """Return an emoji for a regime label.

    Args:
        regime: Regime label string.

    Returns:
        Corresponding emoji character.
    """
    mapping = {
        "STRONG_BULL": "\U0001f7e2",
        "WEAK_BULL": "\U0001f7e1",
        "RANGING": "\u26aa",
        "WEAK_BEAR": "\U0001f7e0",
        "STRONG_BEAR": "\U0001f534",
        "CHAOS": "\U0001f535",
    }
    return mapping.get(regime, "\u2753")


def _next_threshold(current_level: float) -> str:
    """Determine the next drawdown threshold and describe it.

    Args:
        current_level: Current drawdown percentage.

    Returns:
        Human-readable description of the next threshold or halt level.
    """
    for threshold in _DRAWDOWN_THRESHOLDS:
        if current_level < threshold:
            return f"Next alert at {threshold:.0f}%"
    return "SYSTEM HALT level reached (12%+)"
