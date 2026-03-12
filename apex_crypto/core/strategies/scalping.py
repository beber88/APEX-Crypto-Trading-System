"""Scalping strategy for the APEX Crypto Trading System.

Captures quick profits from VWAP deviations during high-volume sessions
on BTC/USDT and ETH/USDT. Uses StochasticRSI extremes and trade flow
imbalance for entry confirmation on 1m/3m timeframes.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

from apex_crypto.core.logging import get_logger, log_with_data
from apex_crypto.core.strategies.base import (
    BaseStrategy,
    SignalDirection,
    TradeSignal,
)

logger = get_logger("strategies.scalping")

# High-volume session windows (UTC hours, inclusive start, exclusive end)
_SESSION_WINDOWS: list[tuple[int, int]] = [(0, 4), (8, 12)]

# Default allowed assets
_DEFAULT_ASSETS: list[str] = ["BTC/USDT", "ETH/USDT"]

# Strategy thresholds
_VWAP_DEVIATION_MIN: float = 0.003  # 0.3%
_VWAP_EXTREME_DEVIATION: float = 0.006  # 0.6% — "extreme"
_STOCH_RSI_OVERSOLD: float = 20.0
_STOCH_RSI_OVERBOUGHT: float = 80.0
_STOP_PCT: float = 0.004  # 0.4%
_TP_MIN_R: float = 1.5
_TP_MAX_R: float = 2.0
_MAX_TRADES_PER_SESSION: int = 10
_MIN_WIN_RATE: float = 0.42
_ROLLING_WINDOW: int = 20


class ScalpingStrategy(BaseStrategy):
    """High-frequency scalping on 1m charts with 3m confirmation.

    Only active during high-volume UTC sessions (00:00-04:00, 08:00-12:00).
    Trades exclusively BTC/USDT and ETH/USDT. Auto-disables when the rolling
    20-trade win rate drops below 42%.

    Attributes:
        name: Strategy identifier.
        active_regimes: Market regimes where this strategy operates.
        primary_timeframe: Main chart timeframe for signal generation.
        confirmation_timeframe: Secondary timeframe for confirmation.
    """

    name: str = "scalping"
    active_regimes: list[str] = [
        "STRONG_BULL",
        "WEAK_BULL",
        "RANGING",
        "WEAK_BEAR",
        "STRONG_BEAR",
    ]
    primary_timeframe: str = "1m"
    confirmation_timeframe: str = "3m"

    def __init__(self, config: dict) -> None:
        """Initialize the scalping strategy.

        Args:
            config: Strategy-specific configuration dict. Expected key
                ``strategies.scalping.assets`` for the tradable asset list.
        """
        super().__init__(config)

        scalping_cfg: dict = config.get("strategies", {}).get("scalping", {})
        self._allowed_assets: list[str] = scalping_cfg.get("assets", _DEFAULT_ASSETS)
        self._max_trades: int = scalping_cfg.get(
            "max_trades_per_session", _MAX_TRADES_PER_SESSION
        )
        self._min_win_rate: float = scalping_cfg.get("min_win_rate", _MIN_WIN_RATE)
        self._rolling_window: int = scalping_cfg.get("rolling_window", _ROLLING_WINDOW)

        # Per-asset, per-session trade counters.  Key = (symbol, session_id)
        self._session_trade_counts: dict[tuple[str, str], int] = defaultdict(int)

        # Rolling recent trade outcomes (True = win, False = loss)
        self._recent_outcomes: list[bool] = []

        # Auto-disable flag
        self._auto_disabled: bool = False

        logger.info(
            "ScalpingStrategy configured",
            extra={
                "data": {
                    "assets": self._allowed_assets,
                    "max_trades_per_session": self._max_trades,
                    "min_win_rate": self._min_win_rate,
                }
            },
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_signal(
        self,
        symbol: str,
        data: dict[str, pd.DataFrame],
        indicators: dict[str, pd.DataFrame],
        regime: str,
        alt_data: Optional[dict] = None,
    ) -> TradeSignal:
        """Generate a scalping signal for *symbol*.

        Args:
            symbol: Trading pair symbol (e.g. ``'BTC/USDT'``).
            data: OHLCV DataFrames keyed by timeframe (``'1m'``, ``'3m'``).
            indicators: Pre-computed indicator DataFrames keyed by timeframe.
                Expected columns on the 1m frame: ``vwap``, ``stoch_rsi_k``,
                ``stoch_rsi_d``, ``volume``, ``buy_volume``, ``sell_volume``.
            regime: Current market regime string.
            alt_data: Optional alternative data dict (unused by this strategy).

        Returns:
            TradeSignal with a score of 0 (neutral) when conditions are not
            met, or 55-90 when a valid scalp setup is detected.
        """
        # ----- Gate checks ------------------------------------------------
        if not self._passes_gate_checks(symbol, regime):
            return self._neutral_signal(symbol)

        # ----- Ensure required data is present ----------------------------
        df_1m = data.get(self.primary_timeframe)
        ind_1m = indicators.get(self.primary_timeframe)
        if df_1m is None or ind_1m is None or df_1m.empty or ind_1m.empty:
            logger.debug("Missing 1m data or indicators for %s", symbol)
            return self._neutral_signal(symbol)

        # ----- Extract latest values -------------------------------------
        close: float = float(df_1m["close"].iloc[-1])
        vwap: float = float(ind_1m["vwap"].iloc[-1])
        stoch_k: float = float(ind_1m["stoch_rsi_k"].iloc[-1])
        volume_current: float = float(df_1m["volume"].iloc[-1])
        volume_mean: float = float(df_1m["volume"].rolling(20).mean().iloc[-1])

        # Trade flow imbalance
        buy_vol: float = float(ind_1m["buy_volume"].iloc[-1])
        sell_vol: float = float(ind_1m["sell_volume"].iloc[-1])
        total_vol: float = buy_vol + sell_vol
        flow_imbalance: float = (
            (buy_vol - sell_vol) / total_vol if total_vol > 0 else 0.0
        )

        # ----- VWAP deviation ---------------------------------------------
        vwap_deviation: float = (close - vwap) / vwap if vwap != 0 else 0.0
        abs_deviation: float = abs(vwap_deviation)

        if abs_deviation < _VWAP_DEVIATION_MIN:
            logger.debug(
                "VWAP deviation %.4f below threshold for %s",
                abs_deviation,
                symbol,
            )
            return self._neutral_signal(symbol)

        # ----- Determine direction from StochRSI + flow -------------------
        direction: Optional[SignalDirection] = None

        if stoch_k < _STOCH_RSI_OVERSOLD and vwap_deviation < 0 and flow_imbalance > 0:
            # Price below VWAP, oversold, buyers stepping in → long
            direction = SignalDirection.LONG
        elif (
            stoch_k > _STOCH_RSI_OVERBOUGHT
            and vwap_deviation > 0
            and flow_imbalance < 0
        ):
            # Price above VWAP, overbought, sellers stepping in → short
            direction = SignalDirection.SHORT

        if direction is None:
            logger.debug(
                "No directional confluence for %s (stoch_k=%.1f, flow=%.4f)",
                symbol,
                stoch_k,
                flow_imbalance,
            )
            return self._neutral_signal(symbol)

        # ----- 3m confirmation --------------------------------------------
        ind_3m = indicators.get(self.confirmation_timeframe)
        has_3m_confirmation: bool = False
        if ind_3m is not None and not ind_3m.empty:
            stoch_k_3m: float = float(ind_3m["stoch_rsi_k"].iloc[-1])
            if direction == SignalDirection.LONG and stoch_k_3m < 40:
                has_3m_confirmation = True
            elif direction == SignalDirection.SHORT and stoch_k_3m > 60:
                has_3m_confirmation = True

        # ----- Score computation ------------------------------------------
        score: int = 55

        if abs_deviation >= _VWAP_EXTREME_DEVIATION:
            score += 15

        volume_spike: bool = volume_current > 2.0 * volume_mean if volume_mean > 0 else False
        if volume_spike:
            score += 10

        if has_3m_confirmation:
            score += 10

        score = min(score, 90)

        # ----- Entry, stop, targets ---------------------------------------
        entry_price: float = close

        if direction == SignalDirection.LONG:
            stop_loss = entry_price * (1 - _STOP_PCT)
            tp1 = entry_price * (1 + _STOP_PCT * _TP_MIN_R)
            tp2 = entry_price * (1 + _STOP_PCT * _TP_MAX_R)
            tp3 = entry_price * (1 + _STOP_PCT * 2.5)
        else:
            stop_loss = entry_price * (1 + _STOP_PCT)
            tp1 = entry_price * (1 - _STOP_PCT * _TP_MIN_R)
            tp2 = entry_price * (1 - _STOP_PCT * _TP_MAX_R)
            tp3 = entry_price * (1 - _STOP_PCT * 2.5)

        # ----- Track session usage ----------------------------------------
        session_id = self._current_session_id()
        self._session_trade_counts[(symbol, session_id)] += 1

        confidence: float = round(score / 100.0, 2)

        signal = TradeSignal(
            symbol=symbol,
            direction=direction,
            score=score if direction == SignalDirection.LONG else -score,
            strategy=self.name,
            timeframe=self.primary_timeframe,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            confidence=confidence,
            metadata={
                "vwap_deviation": round(vwap_deviation, 6),
                "stoch_rsi_k": round(stoch_k, 2),
                "flow_imbalance": round(flow_imbalance, 4),
                "volume_spike": volume_spike,
                "confirmation_3m": has_3m_confirmation,
                "session_id": session_id,
                "session_trade_count": self._session_trade_counts[
                    (symbol, session_id)
                ],
            },
        )

        log_with_data(
            logger,
            "info",
            f"Scalp signal generated for {symbol}",
            data=signal.to_dict(),
        )

        return signal

    # ------------------------------------------------------------------
    # Trade result tracking (extends base)
    # ------------------------------------------------------------------

    def record_trade_result(self, pnl: float, r_multiple: float) -> None:
        """Record a completed trade and update the rolling win-rate tracker.

        Args:
            pnl: Profit/loss in USDT.
            r_multiple: R-multiple of the trade.
        """
        super().record_trade_result(pnl, r_multiple)

        self._recent_outcomes.append(pnl > 0)
        if len(self._recent_outcomes) > self._rolling_window:
            self._recent_outcomes.pop(0)

        if len(self._recent_outcomes) >= self._rolling_window:
            rolling_wr = sum(self._recent_outcomes) / len(self._recent_outcomes)
            if rolling_wr < self._min_win_rate:
                self._auto_disabled = True
                logger.warning(
                    "ScalpingStrategy auto-disabled: rolling %d-trade win rate "
                    "%.1f%% < %.1f%%",
                    self._rolling_window,
                    rolling_wr * 100,
                    self._min_win_rate * 100,
                )
            else:
                # Re-enable if win rate recovers
                if self._auto_disabled:
                    logger.info(
                        "ScalpingStrategy re-enabled: rolling win rate %.1f%%",
                        rolling_wr * 100,
                    )
                self._auto_disabled = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _passes_gate_checks(self, symbol: str, regime: str) -> bool:
        """Run all pre-signal gate checks.

        Args:
            symbol: Trading pair symbol.
            regime: Current market regime string.

        Returns:
            True if all gates pass.
        """
        if self._auto_disabled:
            logger.debug("ScalpingStrategy is auto-disabled due to low win rate")
            return False

        if not self.is_active(regime):
            logger.debug("Regime %s not active for scalping", regime)
            return False

        if symbol not in self._allowed_assets:
            logger.debug("%s not in allowed scalping assets", symbol)
            return False

        if not self._is_high_volume_session():
            logger.debug("Outside high-volume session window")
            return False

        session_id = self._current_session_id()
        if self._session_trade_counts[(symbol, session_id)] >= self._max_trades:
            logger.info(
                "Max scalp trades (%d) reached for %s in session %s",
                self._max_trades,
                symbol,
                session_id,
            )
            return False

        return True

    @staticmethod
    def _is_high_volume_session() -> bool:
        """Check whether the current UTC hour falls within a high-volume window.

        Returns:
            True if the current time is inside one of the defined session windows.
        """
        current_hour: int = datetime.now(timezone.utc).hour
        return any(start <= current_hour < end for start, end in _SESSION_WINDOWS)

    @staticmethod
    def _current_session_id() -> str:
        """Build a unique identifier for the current trading session.

        The session id combines the UTC date with the active session window
        index so that trade counts reset between sessions.

        Returns:
            String session identifier, e.g. ``'2025-06-15_S0'``.
        """
        now = datetime.now(timezone.utc)
        current_hour = now.hour
        for idx, (start, end) in enumerate(_SESSION_WINDOWS):
            if start <= current_hour < end:
                return f"{now.strftime('%Y-%m-%d')}_S{idx}"
        # Fallback (outside session — should not reach here after gate check)
        return f"{now.strftime('%Y-%m-%d')}_OOB"
