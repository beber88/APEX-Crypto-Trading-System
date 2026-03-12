"""Base strategy class for the APEX Crypto Trading System.

All trading strategies inherit from BaseStrategy and implement
the generate_signal() method.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd

from apex_crypto.core.logging import get_logger

logger = get_logger("strategies.base")


class SignalDirection(Enum):
    """Trade signal direction."""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


@dataclass
class TradeSignal:
    """Represents a trading signal from a strategy.

    Attributes:
        symbol: Trading pair symbol.
        direction: Signal direction (long/short/neutral).
        score: Signal strength from -100 (max short) to +100 (max long).
        strategy: Name of the strategy that generated this signal.
        timeframe: Primary timeframe used.
        entry_price: Suggested entry price.
        stop_loss: Suggested stop loss price.
        take_profit_1: First take profit target.
        take_profit_2: Second take profit target.
        take_profit_3: Third take profit target.
        confidence: Strategy confidence in the signal (0.0 to 1.0).
        metadata: Additional signal metadata.
        timestamp: Signal generation timestamp.
    """
    symbol: str
    direction: SignalDirection
    score: int
    strategy: str
    timeframe: str
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit_1: float = 0.0
    take_profit_2: float = 0.0
    take_profit_3: float = 0.0
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def r_multiple(self) -> float:
        """Calculate the risk-reward ratio to TP1.

        Returns:
            R:R ratio, or 0 if stop loss not set.
        """
        if self.stop_loss == 0 or self.entry_price == 0:
            return 0.0
        risk = abs(self.entry_price - self.stop_loss)
        if risk == 0:
            return 0.0
        if self.direction == SignalDirection.LONG:
            reward = self.take_profit_1 - self.entry_price
        else:
            reward = self.entry_price - self.take_profit_1
        return reward / risk

    def to_dict(self) -> dict[str, Any]:
        """Convert signal to dictionary.

        Returns:
            Dictionary representation of the signal.
        """
        return {
            "symbol": self.symbol,
            "direction": self.direction.value,
            "score": self.score,
            "strategy": self.strategy,
            "timeframe": self.timeframe,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit_1": self.take_profit_1,
            "take_profit_2": self.take_profit_2,
            "take_profit_3": self.take_profit_3,
            "confidence": self.confidence,
            "r_multiple": self.r_multiple(),
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies.

    Subclasses must implement generate_signal() and define
    the regimes in which they are active.
    """

    # Override in subclasses
    name: str = "base"
    active_regimes: list[str] = []
    primary_timeframe: str = "4h"
    confirmation_timeframe: str = "1d"
    entry_timeframe: str = "1h"

    def __init__(self, config: dict) -> None:
        """Initialize the strategy.

        Args:
            config: Strategy-specific configuration dict.
        """
        self.config = config
        self.enabled: bool = config.get("enabled", True)
        self._trade_history: list[dict] = []
        self._win_count: int = 0
        self._loss_count: int = 0
        self._total_r: float = 0.0
        logger.info(f"Strategy initialized: {self.name}")

    @abstractmethod
    def generate_signal(
        self,
        symbol: str,
        data: dict[str, pd.DataFrame],
        indicators: dict[str, pd.DataFrame],
        regime: str,
        alt_data: Optional[dict] = None,
    ) -> TradeSignal:
        """Generate a trading signal for the given symbol.

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT').
            data: OHLCV DataFrames keyed by timeframe.
            indicators: Pre-computed indicator DataFrames keyed by timeframe.
            regime: Current market regime string.
            alt_data: Optional alternative data (sentiment, funding, etc.).

        Returns:
            TradeSignal with score from -100 to +100.
        """
        ...

    def is_active(self, regime: str) -> bool:
        """Check if this strategy is active in the given regime.

        Args:
            regime: Current market regime string.

        Returns:
            True if strategy should generate signals.
        """
        if not self.enabled:
            return False
        if not self.active_regimes:
            return True
        return regime in self.active_regimes

    def record_trade_result(self, pnl: float, r_multiple: float) -> None:
        """Record a completed trade result for performance tracking.

        Args:
            pnl: Profit/loss in USDT.
            r_multiple: R-multiple of the trade.
        """
        self._trade_history.append({"pnl": pnl, "r_multiple": r_multiple})
        if pnl > 0:
            self._win_count += 1
        else:
            self._loss_count += 1
        self._total_r += r_multiple

    @property
    def win_rate(self) -> float:
        """Calculate rolling win rate.

        Returns:
            Win rate as a float between 0 and 1.
        """
        total = self._win_count + self._loss_count
        if total == 0:
            return 0.5  # Default assumption
        return self._win_count / total

    @property
    def avg_r_multiple(self) -> float:
        """Calculate average R-multiple.

        Returns:
            Average R-multiple across all recorded trades.
        """
        total = len(self._trade_history)
        if total == 0:
            return 0.0
        return self._total_r / total

    @property
    def profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss).

        Returns:
            Profit factor ratio.
        """
        gross_profit = sum(
            t["pnl"] for t in self._trade_history if t["pnl"] > 0
        )
        gross_loss = abs(
            sum(t["pnl"] for t in self._trade_history if t["pnl"] < 0)
        )
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    def compute_stop_loss(
        self,
        entry_price: float,
        direction: SignalDirection,
        atr: float,
        swing_level: Optional[float] = None,
        atr_multiplier: float = 1.5,
        max_stop_pct: float = 0.02,
    ) -> float:
        """Compute the tightest stop loss from multiple methods.

        Uses the tightest of: ATR-based, structure-based, and percentage-based stops.

        Args:
            entry_price: Trade entry price.
            direction: Trade direction.
            atr: Current ATR value.
            swing_level: Nearest swing low (longs) or swing high (shorts).
            atr_multiplier: ATR multiplier for stop distance.
            max_stop_pct: Maximum stop loss as percentage of entry.

        Returns:
            Stop loss price (the tightest of the three methods).
        """
        if direction == SignalDirection.LONG:
            atr_stop = entry_price - (atr * atr_multiplier)
            pct_stop = entry_price * (1 - max_stop_pct)
            structure_stop = swing_level if swing_level else 0.0
            # Use the tightest (highest for longs)
            candidates = [s for s in [atr_stop, pct_stop, structure_stop] if s > 0]
            return max(candidates) if candidates else pct_stop
        else:
            atr_stop = entry_price + (atr * atr_multiplier)
            pct_stop = entry_price * (1 + max_stop_pct)
            structure_stop = swing_level if swing_level else float("inf")
            candidates = [atr_stop, pct_stop]
            if structure_stop != float("inf"):
                candidates.append(structure_stop)
            return min(candidates)

    def compute_take_profits(
        self,
        entry_price: float,
        stop_loss: float,
        direction: SignalDirection,
        tp1_r: float = 1.5,
        tp2_r: float = 2.5,
        tp3_r: float = 4.0,
    ) -> tuple[float, float, float]:
        """Compute take profit levels based on R-multiples.

        Args:
            entry_price: Trade entry price.
            stop_loss: Stop loss price.
            direction: Trade direction.
            tp1_r: R-multiple for first target.
            tp2_r: R-multiple for second target.
            tp3_r: R-multiple for third target.

        Returns:
            Tuple of (tp1, tp2, tp3) prices.
        """
        risk = abs(entry_price - stop_loss)
        if direction == SignalDirection.LONG:
            tp1 = entry_price + (risk * tp1_r)
            tp2 = entry_price + (risk * tp2_r)
            tp3 = entry_price + (risk * tp3_r)
        else:
            tp1 = entry_price - (risk * tp1_r)
            tp2 = entry_price - (risk * tp2_r)
            tp3 = entry_price - (risk * tp3_r)
        return tp1, tp2, tp3

    def _neutral_signal(self, symbol: str) -> TradeSignal:
        """Create a neutral (no trade) signal.

        Args:
            symbol: Trading pair symbol.

        Returns:
            Neutral TradeSignal with score 0.
        """
        return TradeSignal(
            symbol=symbol,
            direction=SignalDirection.NEUTRAL,
            score=0,
            strategy=self.name,
            timeframe=self.primary_timeframe,
        )
