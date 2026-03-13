"""Transaction cost model for the APEX Crypto Trading System.

Estimates total trading costs (fees, slippage, funding) before every trade
and enforces a minimum-edge requirement.

Inspired by Jim Simons' obsession with transaction costs — the single
largest drag on high-frequency returns.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from apex_crypto.core.logging import get_logger, log_with_data

logger = get_logger("execution.cost_model")


@dataclass
class CostEstimate:
    """Breakdown of estimated transaction costs for a single trade."""

    maker_fee_pct: float = 0.0
    taker_fee_pct: float = 0.0
    slippage_pct: float = 0.0
    funding_cost_pct: float = 0.0
    total_cost_pct: float = 0.0
    expected_profit_pct: float = 0.0
    net_edge_pct: float = 0.0
    has_edge: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "maker_fee_pct": round(self.maker_fee_pct, 6),
            "taker_fee_pct": round(self.taker_fee_pct, 6),
            "slippage_pct": round(self.slippage_pct, 6),
            "funding_cost_pct": round(self.funding_cost_pct, 6),
            "total_cost_pct": round(self.total_cost_pct, 6),
            "expected_profit_pct": round(self.expected_profit_pct, 6),
            "net_edge_pct": round(self.net_edge_pct, 6),
            "has_edge": self.has_edge,
        }


@dataclass
class TradeLog:
    """Record of a single trade's estimated vs actual costs."""

    symbol: str
    timestamp: float
    estimated_cost_pct: float
    actual_cost_pct: float = 0.0
    estimated_slippage_pct: float = 0.0
    actual_slippage_pct: float = 0.0
    pnl_pct: float = 0.0


class TransactionCostModel:
    """Estimates and tracks transaction costs across all trades.

    Uses MEXC fee tiers, square-root market impact for slippage estimation,
    and funding rate projections to compute total round-trip costs.  Trades
    are only approved when the expected edge exceeds costs by a configurable
    minimum threshold.

    Args:
        config: Cost model configuration dictionary.
    """

    def __init__(self, config: dict | None = None) -> None:
        config = config or {}
        self.maker_fee: float = config.get("maker_fee", 0.0002)
        self.taker_fee: float = config.get("taker_fee", 0.0006)
        self.min_edge_pct: float = config.get("min_edge_pct", 0.0015)
        self.default_hold_hours: float = config.get("default_hold_hours", 8.0)

        self._trade_log: list[TradeLog] = []
        self._total_fees_usdt: float = 0.0
        self._total_slippage_usdt: float = 0.0

        log_with_data(logger, "info", "TransactionCostModel initialized", {
            "maker_fee": self.maker_fee,
            "taker_fee": self.taker_fee,
            "min_edge_pct": self.min_edge_pct,
        })

    def estimate_slippage(
        self,
        size_usdt: float,
        daily_volume_usdt: float,
        volatility: float,
        spread_pct: float = 0.0005,
    ) -> float:
        """Estimate slippage using square-root market impact model."""
        if daily_volume_usdt <= 0 or size_usdt <= 0:
            return spread_pct

        participation_rate = size_usdt / daily_volume_usdt
        market_impact = math.sqrt(participation_rate) * volatility

        total_slippage = spread_pct + market_impact
        return total_slippage

    def estimate_costs(
        self,
        signal_score: int,
        atr_pct: float,
        size_usdt: float,
        daily_volume_usdt: float,
        volatility: float,
        current_funding_rate: float = 0.0,
        hold_hours: float | None = None,
        is_maker: bool = False,
        spread_pct: float = 0.0005,
    ) -> CostEstimate:
        """Estimate total round-trip costs and determine if the trade has edge."""
        hold_hours = hold_hours or self.default_hold_hours

        fee_pct = self.maker_fee if is_maker else self.taker_fee
        total_fee_pct = fee_pct * 2

        slippage_pct = self.estimate_slippage(
            size_usdt, daily_volume_usdt, volatility, spread_pct
        )
        total_slippage_pct = slippage_pct * 2

        funding_periods = hold_hours / 8.0
        funding_cost_pct = abs(current_funding_rate) * funding_periods

        total_cost_pct = total_fee_pct + total_slippage_pct + funding_cost_pct

        expected_profit_pct = (abs(signal_score) / 100.0) * atr_pct * 0.5

        net_edge = expected_profit_pct - total_cost_pct
        has_edge = net_edge >= self.min_edge_pct

        estimate = CostEstimate(
            maker_fee_pct=fee_pct if is_maker else 0.0,
            taker_fee_pct=fee_pct if not is_maker else 0.0,
            slippage_pct=total_slippage_pct,
            funding_cost_pct=funding_cost_pct,
            total_cost_pct=total_cost_pct,
            expected_profit_pct=expected_profit_pct,
            net_edge_pct=net_edge,
            has_edge=has_edge,
        )

        log_with_data(logger, "debug", "Cost estimate computed", {
            "signal_score": signal_score,
            "total_cost_pct": round(total_cost_pct, 6),
            "expected_profit_pct": round(expected_profit_pct, 6),
            "net_edge_pct": round(net_edge, 6),
            "has_edge": has_edge,
        })

        return estimate

    def check_edge(
        self,
        signal_score: int,
        atr_pct: float,
        size_usdt: float,
        daily_volume_usdt: float,
        volatility: float,
        current_funding_rate: float = 0.0,
        hold_hours: float | None = None,
        spread_pct: float = 0.0005,
    ) -> tuple[bool, CostEstimate]:
        """Check whether a trade has sufficient edge after costs."""
        estimate = self.estimate_costs(
            signal_score=signal_score,
            atr_pct=atr_pct,
            size_usdt=size_usdt,
            daily_volume_usdt=daily_volume_usdt,
            volatility=volatility,
            current_funding_rate=current_funding_rate,
            hold_hours=hold_hours,
            spread_pct=spread_pct,
        )

        if not estimate.has_edge:
            log_with_data(logger, "info", "Trade SKIPPED — insufficient edge", {
                "net_edge_pct": round(estimate.net_edge_pct, 6),
                "min_required": self.min_edge_pct,
                "total_cost_pct": round(estimate.total_cost_pct, 6),
            })

        return estimate.has_edge, estimate

    def log_trade(
        self,
        symbol: str,
        estimated_cost_pct: float,
        actual_cost_pct: float,
        estimated_slippage_pct: float,
        actual_slippage_pct: float,
        pnl_pct: float,
        size_usdt: float,
    ) -> None:
        """Log a completed trade with estimated vs actual costs."""
        entry = TradeLog(
            symbol=symbol,
            timestamp=time.time(),
            estimated_cost_pct=estimated_cost_pct,
            actual_cost_pct=actual_cost_pct,
            estimated_slippage_pct=estimated_slippage_pct,
            actual_slippage_pct=actual_slippage_pct,
            pnl_pct=pnl_pct,
        )
        self._trade_log.append(entry)
        self._total_fees_usdt += actual_cost_pct * size_usdt
        self._total_slippage_usdt += actual_slippage_pct * size_usdt

        log_with_data(logger, "info", "Trade cost logged", {
            "symbol": symbol,
            "est_cost": round(estimated_cost_pct, 6),
            "actual_cost": round(actual_cost_pct, 6),
            "est_slip": round(estimated_slippage_pct, 6),
            "actual_slip": round(actual_slippage_pct, 6),
        })

    def generate_cost_report(self) -> dict[str, Any]:
        """Generate a summary report of trading costs."""
        if not self._trade_log:
            return {"total_trades": 0, "message": "No trades logged yet."}

        total_trades = len(self._trade_log)
        avg_estimated = sum(t.estimated_cost_pct for t in self._trade_log) / total_trades
        avg_actual = sum(t.actual_cost_pct for t in self._trade_log) / total_trades
        avg_est_slip = sum(t.estimated_slippage_pct for t in self._trade_log) / total_trades
        avg_act_slip = sum(t.actual_slippage_pct for t in self._trade_log) / total_trades
        total_pnl = sum(t.pnl_pct for t in self._trade_log)
        cost_drag = avg_actual * total_trades

        report = {
            "total_trades": total_trades,
            "total_fees_usdt": round(self._total_fees_usdt, 2),
            "total_slippage_usdt": round(self._total_slippage_usdt, 2),
            "avg_estimated_cost_pct": round(avg_estimated, 6),
            "avg_actual_cost_pct": round(avg_actual, 6),
            "avg_estimated_slippage_pct": round(avg_est_slip, 6),
            "avg_actual_slippage_pct": round(avg_act_slip, 6),
            "estimation_accuracy": round(avg_actual / avg_estimated, 4) if avg_estimated > 0 else 0,
            "total_pnl_pct": round(total_pnl, 4),
            "total_cost_drag_pct": round(cost_drag, 4),
        }

        log_with_data(logger, "info", "Cost report generated", report)
        return report
