"""Telegram bot interface for the APEX Crypto Trading System.

Provides a full two-way command interface over Telegram, allowing the
operator to monitor portfolio status, manage positions, adjust risk
parameters, and control system behaviour in real time.

Uses python-telegram-bot v20+ async API exclusively.

Typical usage::

    from apex_crypto.telegram.bot import ApexTelegramBot

    bot = ApexTelegramBot(config=cfg["telegram"])
    bot.set_system(trading_system)
    await bot.start()
"""

from __future__ import annotations

import asyncio
import io
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from telegram import Update
from telegram.ext import (
    Application,
    CallbackContext,
    CommandHandler,
    ContextTypes,
)

from apex_crypto.core.logging import get_logger, log_with_data

logger = get_logger("telegram.bot")

# ---------------------------------------------------------------------------
# Regime display helpers
# ---------------------------------------------------------------------------

_REGIME_EMOJI: Dict[str, str] = {
    "STRONG_BULL": "\U0001f7e2",   # green circle
    "WEAK_BULL": "\U0001f7e1",     # yellow circle
    "RANGING": "\u26aa",            # white circle
    "WEAK_BEAR": "\U0001f7e0",     # orange circle
    "STRONG_BEAR": "\U0001f534",   # red circle
    "CHAOS": "\U0001f535",          # blue circle
}

_DIRECTION_ARROW: Dict[str, str] = {
    "long": "\u2b06\ufe0f",   # up arrow
    "short": "\u2b07\ufe0f",  # down arrow
}


class ApexTelegramBot:
    """Two-way Telegram bot for the APEX Crypto Trading System.

    Registers command handlers for portfolio monitoring, position
    management, risk adjustment, and emergency controls.  The bot
    communicates with the trading system through callback references
    set via :meth:`set_system`.

    Args:
        config: Dictionary containing ``bot_token`` and ``chat_id``
            from the ``telegram`` section of ``config.yaml``.
    """

    def __init__(self, config: dict) -> None:
        """Initialise the bot with Telegram credentials and register handlers.

        Args:
            config: Must contain keys ``bot_token`` (str) and
                ``chat_id`` (int or str).
        """
        self._bot_token: str = config["bot_token"]
        self._chat_id: int = int(config["chat_id"])
        self._app: Application = (
            Application.builder().token(self._bot_token).build()
        )

        # Trading system callback references (set via set_system)
        self._get_portfolio_stats: Optional[Callable] = None
        self._get_open_positions: Optional[Callable] = None
        self._pause_trading: Optional[Callable] = None
        self._resume_trading: Optional[Callable] = None
        self._close_all_positions: Optional[Callable] = None
        self._close_position: Optional[Callable] = None
        self._generate_report: Optional[Callable] = None
        self._set_risk_per_trade: Optional[Callable] = None
        self._set_max_leverage: Optional[Callable] = None
        self._set_trading_mode: Optional[Callable] = None
        self._run_backtest: Optional[Callable] = None
        self._get_regimes: Optional[Callable] = None

        # Emergency stop confirmation state
        self._stop_pending: bool = False
        self._stop_pending_user_id: Optional[int] = None

        self._register_handlers()

        log_with_data(
            logger,
            "info",
            "ApexTelegramBot initialised",
            {"chat_id": self._chat_id},
        )

    # ------------------------------------------------------------------
    # System binding
    # ------------------------------------------------------------------

    def set_system(
        self,
        *,
        get_portfolio_stats: Optional[Callable] = None,
        get_open_positions: Optional[Callable] = None,
        pause_trading: Optional[Callable] = None,
        resume_trading: Optional[Callable] = None,
        close_all_positions: Optional[Callable] = None,
        close_position: Optional[Callable] = None,
        generate_report: Optional[Callable] = None,
        set_risk_per_trade: Optional[Callable] = None,
        set_max_leverage: Optional[Callable] = None,
        set_trading_mode: Optional[Callable] = None,
        run_backtest: Optional[Callable] = None,
        get_regimes: Optional[Callable] = None,
    ) -> None:
        """Bind trading-system callbacks so commands can interact with the engine.

        All parameters are keyword-only callables (sync or async) that the
        bot invokes in response to operator commands.

        Args:
            get_portfolio_stats: Returns a dict with keys ``total_equity``,
                ``daily_pnl``, ``daily_pnl_pct``, ``open_positions_count``,
                ``current_drawdown_pct``, ``mode``.
            get_open_positions: Returns a list of position dicts.
            pause_trading: Pauses new entries.
            resume_trading: Resumes trading.
            close_all_positions: Closes every open position at market.
            close_position: Accepts a symbol string, closes that position.
            generate_report: Returns PDF bytes of the daily report.
            set_risk_per_trade: Accepts a float (percentage).
            set_max_leverage: Accepts a float.
            set_trading_mode: Accepts ``"paper"`` or ``"live"``.
            run_backtest: Accepts a symbol string, returns a results dict.
            get_regimes: Returns a dict mapping symbol to regime info.
        """
        self._get_portfolio_stats = get_portfolio_stats
        self._get_open_positions = get_open_positions
        self._pause_trading = pause_trading
        self._resume_trading = resume_trading
        self._close_all_positions = close_all_positions
        self._close_position = close_position
        self._generate_report = generate_report
        self._set_risk_per_trade = set_risk_per_trade
        self._set_max_leverage = set_max_leverage
        self._set_trading_mode = set_trading_mode
        self._run_backtest = run_backtest
        self._get_regimes = get_regimes

        log_with_data(
            logger,
            "info",
            "Trading system callbacks bound",
            {
                "bound": [
                    name
                    for name, cb in {
                        "get_portfolio_stats": get_portfolio_stats,
                        "get_open_positions": get_open_positions,
                        "pause_trading": pause_trading,
                        "resume_trading": resume_trading,
                        "close_all_positions": close_all_positions,
                        "close_position": close_position,
                        "generate_report": generate_report,
                        "set_risk_per_trade": set_risk_per_trade,
                        "set_max_leverage": set_max_leverage,
                        "set_trading_mode": set_trading_mode,
                        "run_backtest": run_backtest,
                        "get_regimes": get_regimes,
                    }.items()
                    if cb is not None
                ]
            },
        )

    # ------------------------------------------------------------------
    # Handler registration
    # ------------------------------------------------------------------

    def _register_handlers(self) -> None:
        """Register all command handlers with the application."""
        commands: List[tuple[str, Callable]] = [
            ("status", self._cmd_status),
            ("positions", self._cmd_positions),
            ("pause", self._cmd_pause),
            ("resume", self._cmd_resume),
            ("stop", self._cmd_stop),
            ("confirm_stop", self._cmd_confirm_stop),
            ("close", self._cmd_close),
            ("report", self._cmd_report),
            ("risk", self._cmd_risk),
            ("leverage", self._cmd_leverage),
            ("mode", self._cmd_mode),
            ("backtest", self._cmd_backtest),
            ("regime", self._cmd_regime),
            ("help", self._cmd_help),
            ("start", self._cmd_help),
        ]
        for name, callback in commands:
            self._app.add_handler(CommandHandler(name, callback))

        log_with_data(
            logger,
            "debug",
            "Command handlers registered",
            {"commands": [c[0] for c in commands]},
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the bot polling loop.

        Initialises the application, begins polling for updates, and
        sends a startup notification to the configured chat.
        """
        log_with_data(logger, "info", "Starting Telegram bot polling", {})
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)

        await self._app.bot.send_message(
            chat_id=self._chat_id,
            text=(
                "\u2705 <b>APEX Crypto Trading System</b> — Bot online.\n"
                "Type /help for available commands."
            ),
            parse_mode="HTML",
        )
        log_with_data(logger, "info", "Telegram bot polling started", {})

    async def stop(self) -> None:
        """Stop the bot gracefully.

        Sends a shutdown notification, then stops the updater and
        shuts down the application.
        """
        log_with_data(logger, "info", "Stopping Telegram bot", {})
        try:
            await self._app.bot.send_message(
                chat_id=self._chat_id,
                text="\u274c <b>APEX Bot</b> shutting down.",
                parse_mode="HTML",
            )
        except Exception:
            pass  # Best-effort notification

        if self._app.updater and self._app.updater.running:
            await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()
        log_with_data(logger, "info", "Telegram bot stopped", {})

    # ------------------------------------------------------------------
    # Authorization helper
    # ------------------------------------------------------------------

    def _is_authorized(self, update: Update) -> bool:
        """Check whether the incoming message originates from the authorised chat.

        Args:
            update: The incoming Telegram update.

        Returns:
            True if the message chat ID matches the configured chat ID.
        """
        if update.effective_chat is None:
            return False
        return update.effective_chat.id == self._chat_id

    # ------------------------------------------------------------------
    # Async callback helper
    # ------------------------------------------------------------------

    @staticmethod
    async def _invoke(callback: Optional[Callable], *args: Any, **kwargs: Any) -> Any:
        """Invoke a callback, handling both sync and async callables.

        Args:
            callback: The callable to invoke (may be None).
            *args: Positional arguments forwarded to the callback.
            **kwargs: Keyword arguments forwarded to the callback.

        Returns:
            The return value of the callback, or None if callback is None.

        Raises:
            RuntimeError: If the callback is not set.
        """
        if callback is None:
            raise RuntimeError("System callback not configured")
        result = callback(*args, **kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    async def _cmd_status(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /status — show portfolio summary.

        Args:
            update: Incoming Telegram update.
            context: Callback context from python-telegram-bot.
        """
        if not self._is_authorized(update):
            return

        try:
            stats: dict = await self._invoke(self._get_portfolio_stats)
            text = self._format_status(stats)
            await update.message.reply_text(text, parse_mode="HTML")
            log_with_data(logger, "info", "Status command served", stats)
        except RuntimeError:
            await update.message.reply_text(
                "\u26a0\ufe0f System not connected. Portfolio stats unavailable."
            )
        except Exception as exc:
            log_with_data(
                logger, "error", "Status command failed", {"error": str(exc)}
            )
            await update.message.reply_text(
                f"\u274c Error fetching status: {exc}"
            )

    async def _cmd_positions(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /positions — show open positions table.

        Args:
            update: Incoming Telegram update.
            context: Callback context from python-telegram-bot.
        """
        if not self._is_authorized(update):
            return

        try:
            positions: list = await self._invoke(self._get_open_positions)
            if not positions:
                await update.message.reply_text("No open positions.")
                return
            text = self._format_positions_table(positions)
            await update.message.reply_text(
                f"<pre>{text}</pre>", parse_mode="HTML"
            )
            log_with_data(
                logger,
                "info",
                "Positions command served",
                {"count": len(positions)},
            )
        except RuntimeError:
            await update.message.reply_text(
                "\u26a0\ufe0f System not connected. Positions unavailable."
            )
        except Exception as exc:
            log_with_data(
                logger, "error", "Positions command failed", {"error": str(exc)}
            )
            await update.message.reply_text(
                f"\u274c Error fetching positions: {exc}"
            )

    async def _cmd_pause(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /pause — stop new trade entries while managing existing positions.

        Args:
            update: Incoming Telegram update.
            context: Callback context from python-telegram-bot.
        """
        if not self._is_authorized(update):
            return

        try:
            await self._invoke(self._pause_trading)
            await update.message.reply_text(
                "\u23f8\ufe0f <b>Trading PAUSED.</b>\n"
                "No new entries will be opened.\n"
                "Existing positions continue to be managed (SL/TP active).\n"
                "Use /resume to restart trading.",
                parse_mode="HTML",
            )
            log_with_data(logger, "info", "Trading paused via Telegram", {})
        except RuntimeError:
            await update.message.reply_text(
                "\u26a0\ufe0f System not connected."
            )
        except Exception as exc:
            log_with_data(
                logger, "error", "Pause command failed", {"error": str(exc)}
            )
            await update.message.reply_text(f"\u274c Error pausing: {exc}")

    async def _cmd_resume(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /resume — resume trading after a pause.

        Args:
            update: Incoming Telegram update.
            context: Callback context from python-telegram-bot.
        """
        if not self._is_authorized(update):
            return

        try:
            await self._invoke(self._resume_trading)
            await update.message.reply_text(
                "\u25b6\ufe0f <b>Trading RESUMED.</b>\n"
                "The system will now open new positions when signals fire.",
                parse_mode="HTML",
            )
            log_with_data(logger, "info", "Trading resumed via Telegram", {})
        except RuntimeError:
            await update.message.reply_text(
                "\u26a0\ufe0f System not connected."
            )
        except Exception as exc:
            log_with_data(
                logger, "error", "Resume command failed", {"error": str(exc)}
            )
            await update.message.reply_text(f"\u274c Error resuming: {exc}")

    async def _cmd_stop(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /stop — initiate emergency stop (requires confirmation).

        Sets a pending-stop flag and asks the user to confirm with
        ``/confirm_stop``.

        Args:
            update: Incoming Telegram update.
            context: Callback context from python-telegram-bot.
        """
        if not self._is_authorized(update):
            return

        self._stop_pending = True
        self._stop_pending_user_id = (
            update.effective_user.id if update.effective_user else None
        )
        await update.message.reply_text(
            "\U0001f6a8 <b>EMERGENCY STOP requested.</b>\n\n"
            "This will:\n"
            "  1. Close ALL open positions at market price\n"
            "  2. Pause the entire trading system\n\n"
            "\u26a0\ufe0f Type /confirm_stop to proceed.\n"
            "Any other command cancels this request.",
            parse_mode="HTML",
        )
        log_with_data(logger, "warning", "Emergency stop requested — awaiting confirmation", {})

    async def _cmd_confirm_stop(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /confirm_stop — execute the emergency stop.

        Only proceeds if a stop was previously requested via ``/stop``
        by the same user.

        Args:
            update: Incoming Telegram update.
            context: Callback context from python-telegram-bot.
        """
        if not self._is_authorized(update):
            return

        if not self._stop_pending:
            await update.message.reply_text(
                "No pending stop request. Use /stop first."
            )
            return

        user_id = update.effective_user.id if update.effective_user else None
        if self._stop_pending_user_id is not None and user_id != self._stop_pending_user_id:
            await update.message.reply_text(
                "\u274c Only the user who initiated /stop can confirm."
            )
            return

        self._stop_pending = False
        self._stop_pending_user_id = None

        try:
            await update.message.reply_text(
                "\U0001f6d1 Closing all positions at market..."
            )
            await self._invoke(self._close_all_positions)
            await self._invoke(self._pause_trading)
            await update.message.reply_text(
                "\U0001f6d1 <b>EMERGENCY STOP EXECUTED.</b>\n"
                "All positions closed. System paused.\n"
                "Use /resume to restart when ready.",
                parse_mode="HTML",
            )
            log_with_data(
                logger, "warning", "Emergency stop executed via Telegram", {}
            )
        except RuntimeError:
            await update.message.reply_text(
                "\u26a0\ufe0f System not connected. Cannot execute emergency stop."
            )
        except Exception as exc:
            log_with_data(
                logger,
                "error",
                "Emergency stop failed",
                {"error": str(exc)},
            )
            await update.message.reply_text(
                f"\u274c CRITICAL — Emergency stop failed: {exc}\n"
                "Manual intervention required!"
            )

    async def _cmd_close(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /close <symbol> — close a specific position.

        Args:
            update: Incoming Telegram update.
            context: Callback context; ``context.args`` should contain
                the symbol (e.g. ``["BTC"]``).
        """
        if not self._is_authorized(update):
            return

        # Cancel any pending stop request
        self._stop_pending = False

        if not context.args:
            await update.message.reply_text(
                "Usage: /close <symbol>\n"
                "Example: /close BTC"
            )
            return

        symbol: str = context.args[0].upper().strip()
        # Normalise bare symbol to trading pair
        if "/" not in symbol and not symbol.endswith("USDT"):
            symbol = f"{symbol}/USDT"

        try:
            result = await self._invoke(self._close_position, symbol)
            await update.message.reply_text(
                f"\u2705 Position <b>{symbol}</b> closed.\n"
                f"{self._format_close_result(result)}",
                parse_mode="HTML",
            )
            log_with_data(
                logger,
                "info",
                "Position closed via Telegram",
                {"symbol": symbol},
            )
        except RuntimeError:
            await update.message.reply_text(
                "\u26a0\ufe0f System not connected."
            )
        except Exception as exc:
            log_with_data(
                logger,
                "error",
                "Close command failed",
                {"symbol": symbol, "error": str(exc)},
            )
            await update.message.reply_text(
                f"\u274c Error closing {symbol}: {exc}"
            )

    async def _cmd_report(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /report — generate and send the daily PDF report.

        Args:
            update: Incoming Telegram update.
            context: Callback context from python-telegram-bot.
        """
        if not self._is_authorized(update):
            return

        # Cancel any pending stop request
        self._stop_pending = False

        try:
            await update.message.reply_text(
                "\U0001f4ca Generating report... please wait."
            )
            pdf_bytes: bytes = await self._invoke(self._generate_report)

            today_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
            filename = f"apex_report_{today_str}.pdf"

            await update.message.reply_document(
                document=io.BytesIO(pdf_bytes),
                filename=filename,
                caption=f"APEX Daily Report — {today_str}",
            )
            log_with_data(
                logger,
                "info",
                "Report sent via Telegram",
                {"filename": filename, "size_bytes": len(pdf_bytes)},
            )
        except RuntimeError:
            await update.message.reply_text(
                "\u26a0\ufe0f System not connected. Cannot generate report."
            )
        except Exception as exc:
            log_with_data(
                logger, "error", "Report command failed", {"error": str(exc)}
            )
            await update.message.reply_text(
                f"\u274c Error generating report: {exc}"
            )

    async def _cmd_risk(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /risk <value> — set risk per trade percentage.

        Args:
            update: Incoming Telegram update.
            context: Callback context; ``context.args`` should contain
                the new risk percentage (e.g. ``["1.5"]``).
        """
        if not self._is_authorized(update):
            return

        # Cancel any pending stop request
        self._stop_pending = False

        if not context.args:
            await update.message.reply_text(
                "Usage: /risk <value>\n"
                "Example: /risk 1.5  (sets risk to 1.5% per trade)"
            )
            return

        try:
            value = float(context.args[0])
        except ValueError:
            await update.message.reply_text(
                "\u274c Invalid number. Example: /risk 1.5"
            )
            return

        if value <= 0 or value > 10:
            await update.message.reply_text(
                "\u274c Risk must be between 0.01% and 10%."
            )
            return

        try:
            await self._invoke(self._set_risk_per_trade, value)
            await update.message.reply_text(
                f"\u2705 Risk per trade set to <b>{value:.2f}%</b>",
                parse_mode="HTML",
            )
            log_with_data(
                logger,
                "info",
                "Risk per trade updated via Telegram",
                {"risk_pct": value},
            )
        except RuntimeError:
            await update.message.reply_text(
                "\u26a0\ufe0f System not connected."
            )
        except Exception as exc:
            log_with_data(
                logger, "error", "Risk command failed", {"error": str(exc)}
            )
            await update.message.reply_text(
                f"\u274c Error setting risk: {exc}"
            )

    async def _cmd_leverage(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /leverage <value> — set maximum leverage.

        Args:
            update: Incoming Telegram update.
            context: Callback context; ``context.args`` should contain
                the new leverage value (e.g. ``["2"]``).
        """
        if not self._is_authorized(update):
            return

        # Cancel any pending stop request
        self._stop_pending = False

        if not context.args:
            await update.message.reply_text(
                "Usage: /leverage <value>\n"
                "Example: /leverage 2  (sets max leverage to 2x)"
            )
            return

        try:
            value = float(context.args[0])
        except ValueError:
            await update.message.reply_text(
                "\u274c Invalid number. Example: /leverage 2"
            )
            return

        if value < 1 or value > 20:
            await update.message.reply_text(
                "\u274c Leverage must be between 1x and 20x."
            )
            return

        try:
            await self._invoke(self._set_max_leverage, value)
            await update.message.reply_text(
                f"\u2705 Max leverage set to <b>{value:.1f}x</b>",
                parse_mode="HTML",
            )
            log_with_data(
                logger,
                "info",
                "Max leverage updated via Telegram",
                {"leverage": value},
            )
        except RuntimeError:
            await update.message.reply_text(
                "\u26a0\ufe0f System not connected."
            )
        except Exception as exc:
            log_with_data(
                logger,
                "error",
                "Leverage command failed",
                {"error": str(exc)},
            )
            await update.message.reply_text(
                f"\u274c Error setting leverage: {exc}"
            )

    async def _cmd_mode(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /mode <paper|live> — switch trading mode.

        Args:
            update: Incoming Telegram update.
            context: Callback context; ``context.args`` should contain
                ``"paper"`` or ``"live"``.
        """
        if not self._is_authorized(update):
            return

        # Cancel any pending stop request
        self._stop_pending = False

        if not context.args or context.args[0].lower() not in ("paper", "live"):
            await update.message.reply_text(
                "Usage: /mode <paper|live>\n"
                "Example: /mode paper"
            )
            return

        mode: str = context.args[0].lower()

        try:
            await self._invoke(self._set_trading_mode, mode)
            mode_label = "\U0001f4dd Paper" if mode == "paper" else "\U0001f4b0 Live"
            await update.message.reply_text(
                f"\u2705 Trading mode switched to <b>{mode_label}</b>",
                parse_mode="HTML",
            )
            log_with_data(
                logger,
                "info",
                "Trading mode changed via Telegram",
                {"mode": mode},
            )
        except RuntimeError:
            await update.message.reply_text(
                "\u26a0\ufe0f System not connected."
            )
        except Exception as exc:
            log_with_data(
                logger, "error", "Mode command failed", {"error": str(exc)}
            )
            await update.message.reply_text(
                f"\u274c Error switching mode: {exc}"
            )

    async def _cmd_backtest(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /backtest <symbol> — run a simplified backtest.

        Args:
            update: Incoming Telegram update.
            context: Callback context; ``context.args`` should contain
                the symbol (e.g. ``["BTC"]``).
        """
        if not self._is_authorized(update):
            return

        # Cancel any pending stop request
        self._stop_pending = False

        if not context.args:
            await update.message.reply_text(
                "Usage: /backtest <symbol>\n"
                "Example: /backtest BTC"
            )
            return

        symbol: str = context.args[0].upper().strip()
        if "/" not in symbol and not symbol.endswith("USDT"):
            symbol = f"{symbol}/USDT"

        try:
            await update.message.reply_text(
                f"\u23f3 Running backtest for <b>{symbol}</b>...",
                parse_mode="HTML",
            )
            results: dict = await self._invoke(self._run_backtest, symbol)
            text = self._format_backtest_results(symbol, results)
            await update.message.reply_text(text, parse_mode="HTML")
            log_with_data(
                logger,
                "info",
                "Backtest completed via Telegram",
                {"symbol": symbol},
            )
        except RuntimeError:
            await update.message.reply_text(
                "\u26a0\ufe0f System not connected. Cannot run backtest."
            )
        except Exception as exc:
            log_with_data(
                logger,
                "error",
                "Backtest command failed",
                {"symbol": symbol, "error": str(exc)},
            )
            await update.message.reply_text(
                f"\u274c Error running backtest for {symbol}: {exc}"
            )

    async def _cmd_regime(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /regime — show current market regime for all assets.

        Args:
            update: Incoming Telegram update.
            context: Callback context from python-telegram-bot.
        """
        if not self._is_authorized(update):
            return

        # Cancel any pending stop request
        self._stop_pending = False

        try:
            regimes: dict = await self._invoke(self._get_regimes)
            text = self._format_regime(regimes)
            await update.message.reply_text(text, parse_mode="HTML")
            log_with_data(
                logger,
                "info",
                "Regime command served",
                {"asset_count": len(regimes)},
            )
        except RuntimeError:
            await update.message.reply_text(
                "\u26a0\ufe0f System not connected. Regime data unavailable."
            )
        except Exception as exc:
            log_with_data(
                logger, "error", "Regime command failed", {"error": str(exc)}
            )
            await update.message.reply_text(
                f"\u274c Error fetching regimes: {exc}"
            )

    async def _cmd_help(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /help — list all available commands.

        Args:
            update: Incoming Telegram update.
            context: Callback context from python-telegram-bot.
        """
        if not self._is_authorized(update):
            return

        # Cancel any pending stop request
        self._stop_pending = False

        help_text = (
            "<b>APEX Crypto Trading System</b>\n"
            "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
            "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
            "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\n"
            "<b>Monitoring</b>\n"
            "/status — Portfolio summary\n"
            "/positions — Open positions table\n"
            "/regime — Market regime for all assets\n"
            "/report — Generate and send daily PDF report\n\n"
            "<b>Trading Control</b>\n"
            "/pause — Stop new entries (keeps managing existing)\n"
            "/resume — Resume trading\n"
            "/stop — EMERGENCY: close all positions + pause\n"
            "/close &lt;symbol&gt; — Close specific position\n\n"
            "<b>Configuration</b>\n"
            "/risk &lt;value&gt; — Set risk per trade %\n"
            "/leverage &lt;value&gt; — Set max leverage\n"
            "/mode &lt;paper|live&gt; — Switch trading mode\n\n"
            "<b>Analysis</b>\n"
            "/backtest &lt;symbol&gt; — Run backtest on asset\n\n"
            "<b>Example:</b> /risk 1.5 /close BTC /mode paper"
        )
        await update.message.reply_text(help_text, parse_mode="HTML")

    # ------------------------------------------------------------------
    # Formatting methods
    # ------------------------------------------------------------------

    @staticmethod
    def _format_status(stats: dict) -> str:
        """Format portfolio status into a readable Telegram message.

        Args:
            stats: Dictionary with keys ``total_equity``, ``daily_pnl``,
                ``daily_pnl_pct``, ``open_positions_count``,
                ``current_drawdown_pct``, ``mode``.

        Returns:
            HTML-formatted status string.
        """
        equity = stats.get("total_equity", 0.0)
        daily_pnl = stats.get("daily_pnl", 0.0)
        daily_pnl_pct = stats.get("daily_pnl_pct", 0.0)
        total_pnl = stats.get("total_pnl", 0.0)
        total_pnl_pct = stats.get("total_pnl_pct", 0.0)
        positions_count = stats.get("open_positions_count", 0)
        drawdown = stats.get("current_drawdown_pct", 0.0)
        mode = stats.get("mode", "unknown")

        pnl_sign = "+" if daily_pnl >= 0 else ""
        pnl_emoji = "\U0001f7e2" if daily_pnl >= 0 else "\U0001f534"
        total_sign = "+" if total_pnl >= 0 else ""
        total_emoji = "\U0001f7e2" if total_pnl >= 0 else "\U0001f534"
        mode_label = "\U0001f4dd Paper" if mode == "paper" else "\U0001f4b0 Live"

        return (
            f"<b>APEX Portfolio Status</b>\n"
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
            f"\U0001f4b0 Equity: <b>${equity:,.2f}</b>\n"
            f"{pnl_emoji} Today P&L: <b>{pnl_sign}${daily_pnl:,.2f} "
            f"({pnl_sign}{daily_pnl_pct:.2f}%)</b>\n"
            f"{total_emoji} Total P&L: <b>{total_sign}${total_pnl:,.2f} "
            f"({total_sign}{total_pnl_pct:.2f}%)</b>\n"
            f"\U0001f4c8 Open Positions: <b>{positions_count}</b>\n"
            f"\U0001f4c9 Drawdown: <b>{drawdown:.1f}%</b>\n"
            f"\u2699\ufe0f Mode: <b>{mode_label}</b>"
        )

    @staticmethod
    def _format_positions_table(positions: list) -> str:
        """Format open positions as an aligned text table.

        Args:
            positions: List of position dicts, each containing keys
                ``symbol``, ``direction``, ``entry_price``,
                ``current_price``, ``unrealized_pnl``,
                ``unrealized_pnl_pct``, ``stop_loss``, ``target``,
                ``strategy``.

        Returns:
            Plain-text table suitable for monospace display.
        """
        if not positions:
            return "No open positions."

        header = (
            f"{'Symbol':<12} {'Dir':<6} {'Entry':>10} {'Current':>10} "
            f"{'uP&L':>10} {'SL':>10} {'TP':>10} {'Strategy':<16}"
        )
        separator = "\u2500" * len(header)
        lines: List[str] = [header, separator]

        for pos in positions:
            symbol = pos.get("symbol", "???")
            direction = pos.get("direction", "?")
            entry = pos.get("entry_price", 0.0)
            current = pos.get("current_price", 0.0)
            upnl = pos.get("unrealized_pnl", 0.0)
            upnl_pct = pos.get("unrealized_pnl_pct", 0.0)
            sl = pos.get("stop_loss", 0.0)
            tp = pos.get("target", 0.0)
            strategy = pos.get("strategy", "n/a")

            dir_arrow = _DIRECTION_ARROW.get(direction.lower(), "")
            pnl_sign = "+" if upnl >= 0 else ""
            pnl_str = f"{pnl_sign}${upnl:,.1f} ({pnl_sign}{upnl_pct:.1f}%)"

            lines.append(
                f"{symbol:<12} {dir_arrow}{direction.upper():<4} "
                f"${entry:>9,.2f} ${current:>9,.2f} "
                f"{pnl_str:>10} ${sl:>9,.2f} ${tp:>9,.2f} {strategy:<16}"
            )

        total_upnl = sum(p.get("unrealized_pnl", 0.0) for p in positions)
        sign = "+" if total_upnl >= 0 else ""
        lines.append(separator)
        lines.append(f"Total uP&L: {sign}${total_upnl:,.2f}")

        return "\n".join(lines)

    @staticmethod
    def _format_regime(regimes: dict) -> str:
        """Format regime information for all assets.

        Args:
            regimes: Dictionary mapping symbol strings to regime info
                dicts with keys ``regime`` and ``confidence``.

        Returns:
            HTML-formatted regime display with emoji indicators.
        """
        if not regimes:
            return "No regime data available."

        lines: List[str] = [
            "<b>Market Regimes</b>",
            "\u2500" * 30,
        ]

        for symbol, info in regimes.items():
            regime_name: str = info.get("regime", "UNKNOWN")
            confidence: float = info.get("confidence", 0.0)
            emoji = _REGIME_EMOJI.get(regime_name, "\u2753")
            conf_bar = _confidence_bar(confidence)
            lines.append(
                f"{emoji} <b>{symbol}</b>: {regime_name} "
                f"({confidence:.0%}) {conf_bar}"
            )

        return "\n".join(lines)

    @staticmethod
    def _format_close_result(result: Any) -> str:
        """Format the result of closing a position.

        Args:
            result: Result from the close_position callback — may be a
                dict with trade details or None.

        Returns:
            Human-readable summary of the close.
        """
        if result is None or not isinstance(result, dict):
            return ""

        pnl = result.get("realized_pnl", 0.0)
        pnl_pct = result.get("realized_pnl_pct", 0.0)
        exit_price = result.get("exit_price", 0.0)
        sign = "+" if pnl >= 0 else ""

        return (
            f"Exit: ${exit_price:,.2f}\n"
            f"P&L: {sign}${pnl:,.2f} ({sign}{pnl_pct:.2f}%)"
        )

    @staticmethod
    def _format_backtest_results(symbol: str, results: dict) -> str:
        """Format backtest results into a Telegram message.

        Args:
            symbol: The asset symbol that was backtested.
            results: Backtest results dict with keys ``total_return_pct``,
                ``sharpe_ratio``, ``max_drawdown_pct``, ``win_rate``,
                ``total_trades``, ``profit_factor``, ``period``.

        Returns:
            HTML-formatted backtest summary.
        """
        total_return = results.get("total_return_pct", 0.0)
        sharpe = results.get("sharpe_ratio", 0.0)
        max_dd = results.get("max_drawdown_pct", 0.0)
        win_rate = results.get("win_rate", 0.0)
        total_trades = results.get("total_trades", 0)
        pf = results.get("profit_factor", 0.0)
        period = results.get("period", "N/A")

        return_sign = "+" if total_return >= 0 else ""

        return (
            f"<b>Backtest Results: {symbol}</b>\n"
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
            f"\U0001f4c5 Period: {period}\n"
            f"\U0001f4c8 Return: <b>{return_sign}{total_return:.2f}%</b>\n"
            f"\U0001f4ca Sharpe Ratio: <b>{sharpe:.2f}</b>\n"
            f"\U0001f4c9 Max Drawdown: <b>{max_dd:.1f}%</b>\n"
            f"\U0001f3af Win Rate: <b>{win_rate:.1f}%</b>\n"
            f"\U0001f4b9 Profit Factor: <b>{pf:.2f}</b>\n"
            f"\U0001f504 Total Trades: <b>{total_trades}</b>"
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _confidence_bar(confidence: float, width: int = 10) -> str:
    """Render a text-based confidence bar.

    Args:
        confidence: Value between 0.0 and 1.0.
        width: Number of characters in the bar.

    Returns:
        String like ``[########  ]`` representing the confidence.
    """
    filled = int(round(confidence * width))
    empty = width - filled
    return f"[{'#' * filled}{' ' * empty}]"
