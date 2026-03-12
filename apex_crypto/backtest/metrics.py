"""Performance metrics for the APEX Crypto Trading System.

Provides comprehensive backtesting and live-trading performance analytics.
All methods are stateless class/static methods that operate on equity curves
and trade DataFrames produced by :class:`VectorizedBacktester`.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from apex_crypto.core.logging import get_logger, log_with_data

logger = get_logger("backtest.metrics")

# Annualisation constants (crypto markets trade 365 days/year, 24/7).
_TRADING_DAYS_YEAR: int = 365
_HOURS_PER_YEAR: int = _TRADING_DAYS_YEAR * 24


class PerformanceMetrics:
    """Stateless container for performance metric computations.

    All public methods are ``@staticmethod`` or ``@classmethod`` and can
    be called without instantiation, though the class may be instantiated
    as a no-op for API consistency.
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Aggregate entry point
    # ------------------------------------------------------------------

    @classmethod
    def compute_all(
        cls,
        equity_curve: pd.Series,
        trades: pd.DataFrame,
        benchmark: Optional[pd.Series] = None,
    ) -> dict[str, Any]:
        """Compute all available performance metrics.

        Args:
            equity_curve: Cumulative equity series (indexed by datetime).
            trades: Trade records DataFrame as returned by
                :meth:`VectorizedBacktester._extract_trades`.
            benchmark: Optional benchmark equity curve for relative
                metrics.  When supplied, alpha and beta are included.

        Returns:
            Comprehensive dictionary of performance metrics.
        """
        returns = equity_curve.pct_change().fillna(0.0)

        metrics: dict[str, Any] = {
            # Returns
            "total_return": cls.total_return(equity_curve),
            "cagr": cls.cagr(equity_curve),

            # Risk-adjusted
            "sharpe_ratio": cls.sharpe_ratio(returns),
            "sortino_ratio": cls.sortino_ratio(returns),
            "calmar_ratio": cls.calmar_ratio(equity_curve),

            # Drawdown
            "max_drawdown": cls.max_drawdown(equity_curve),
            "max_drawdown_duration": cls.max_drawdown_duration(equity_curve),

            # Trade statistics
            "win_rate": cls.win_rate(trades),
            "profit_factor": cls.profit_factor(trades),
            "average_expectancy": cls.average_expectancy(trades),
            "total_trades": len(trades),
            "trade_frequency": cls.trade_frequency(trades, len(equity_curve)),
            "average_hold_time": cls.average_hold_time(trades),

            # Risk
            "value_at_risk_95": cls.value_at_risk(returns, confidence=0.95),
            "conditional_var_95": cls.conditional_var(returns, confidence=0.95),

            # Breakdown
            "monthly_pnl": cls.monthly_pnl_heatmap(equity_curve).to_dict(),
            "strategy_breakdown": cls.strategy_breakdown(trades),
        }

        # Benchmark-relative metrics.
        if benchmark is not None and not benchmark.empty:
            bench_returns = benchmark.pct_change().fillna(0.0)
            # Align series on common index.
            aligned_returns, aligned_bench = returns.align(
                bench_returns, join="inner", fill_value=0.0,
            )
            if len(aligned_returns) > 1:
                cov_matrix = np.cov(
                    aligned_returns.values, aligned_bench.values,
                )
                bench_var = cov_matrix[1, 1]
                beta = float(cov_matrix[0, 1] / bench_var) if bench_var > 0 else 0.0
                alpha = float(
                    aligned_returns.mean() - beta * aligned_bench.mean()
                ) * _TRADING_DAYS_YEAR
                metrics["beta"] = round(beta, 4)
                metrics["alpha"] = round(alpha, 6)

        log_with_data(logger, "info", "All metrics computed", {
            "total_return": metrics["total_return"],
            "sharpe": metrics["sharpe_ratio"],
            "max_dd": metrics["max_drawdown"],
            "win_rate": metrics["win_rate"],
            "total_trades": metrics["total_trades"],
        })

        return metrics

    # ------------------------------------------------------------------
    # Return metrics
    # ------------------------------------------------------------------

    @staticmethod
    def total_return(equity: pd.Series) -> float:
        """Total return as a decimal fraction.

        Args:
            equity: Equity curve series.

        Returns:
            Total return (e.g. 0.50 = +50%).
        """
        if equity.empty or equity.iloc[0] == 0:
            return 0.0
        return float(equity.iloc[-1] / equity.iloc[0] - 1)

    @staticmethod
    def cagr(equity: pd.Series) -> float:
        """Compound annual growth rate.

        Uses the time span between the first and last index entry,
        assuming a DatetimeIndex.  Falls back to bar-count estimation
        when the index is not datetime-typed.

        Args:
            equity: Equity curve series.

        Returns:
            Annualised CAGR as a decimal fraction.
        """
        if equity.empty or len(equity) < 2 or equity.iloc[0] == 0:
            return 0.0

        total_ret = equity.iloc[-1] / equity.iloc[0]

        if isinstance(equity.index, pd.DatetimeIndex):
            delta = equity.index[-1] - equity.index[0]
            years = delta.total_seconds() / (365.25 * 86400)
        else:
            # Assume daily bars as fallback.
            years = len(equity) / _TRADING_DAYS_YEAR

        if years <= 0:
            return 0.0

        return float(np.sign(total_ret) * (abs(total_ret) ** (1 / years) - 1))

    # ------------------------------------------------------------------
    # Risk-adjusted metrics
    # ------------------------------------------------------------------

    @staticmethod
    def sharpe_ratio(
        returns: pd.Series,
        risk_free: float = 0.02,
    ) -> float:
        """Annualised Sharpe ratio.

        Args:
            returns: Series of per-period returns.
            risk_free: Annual risk-free rate (decimal).

        Returns:
            Sharpe ratio.  Returns 0 when volatility is zero.
        """
        if returns.empty or returns.std() == 0:
            return 0.0

        # Daily risk-free rate.
        rf_daily = (1 + risk_free) ** (1 / _TRADING_DAYS_YEAR) - 1
        excess = returns - rf_daily
        return float(excess.mean() / excess.std() * np.sqrt(_TRADING_DAYS_YEAR))

    @staticmethod
    def sortino_ratio(
        returns: pd.Series,
        risk_free: float = 0.02,
    ) -> float:
        """Annualised Sortino ratio (downside deviation only).

        Args:
            returns: Series of per-period returns.
            risk_free: Annual risk-free rate (decimal).

        Returns:
            Sortino ratio.
        """
        if returns.empty:
            return 0.0

        rf_daily = (1 + risk_free) ** (1 / _TRADING_DAYS_YEAR) - 1
        excess = returns - rf_daily
        downside = excess[excess < 0]

        if downside.empty or downside.std() == 0:
            return 0.0 if excess.mean() <= 0 else float("inf")

        downside_std = float(np.sqrt((downside ** 2).mean()))
        return float(excess.mean() / downside_std * np.sqrt(_TRADING_DAYS_YEAR))

    @staticmethod
    def calmar_ratio(equity: pd.Series) -> float:
        """Calmar ratio: CAGR divided by maximum drawdown.

        Args:
            equity: Equity curve series.

        Returns:
            Calmar ratio.  Returns 0 when max drawdown is zero.
        """
        cagr_val = PerformanceMetrics.cagr(equity)
        mdd = PerformanceMetrics.max_drawdown(equity)
        if mdd == 0:
            return 0.0
        return float(cagr_val / abs(mdd))

    # ------------------------------------------------------------------
    # Drawdown metrics
    # ------------------------------------------------------------------

    @staticmethod
    def max_drawdown(equity: pd.Series) -> float:
        """Maximum drawdown as a decimal fraction (negative value).

        Args:
            equity: Equity curve series.

        Returns:
            Max drawdown as a negative float (e.g. -0.15 = -15%).
        """
        if equity.empty:
            return 0.0

        running_max = equity.cummax()
        drawdowns = (equity - running_max) / running_max
        return float(drawdowns.min())

    @staticmethod
    def max_drawdown_duration(equity: pd.Series) -> int:
        """Duration of the longest drawdown in bars.

        Measures the number of bars between a peak and the recovery to
        a new high.  If the equity never recovers, the duration extends
        to the end of the series.

        Args:
            equity: Equity curve series.

        Returns:
            Maximum drawdown duration in bars.
        """
        if equity.empty:
            return 0

        running_max = equity.cummax()
        is_at_peak = equity >= running_max

        # Track consecutive non-peak bars.
        peak_indices = np.where(is_at_peak.values)[0]
        if len(peak_indices) == 0:
            return len(equity)

        # Gaps between consecutive peaks.
        gaps = np.diff(peak_indices)
        max_gap = int(gaps.max()) if len(gaps) > 0 else 0

        # Check trailing drawdown (from last peak to end).
        trailing = len(equity) - 1 - peak_indices[-1]
        return max(max_gap, trailing)

    # ------------------------------------------------------------------
    # Trade statistics
    # ------------------------------------------------------------------

    @staticmethod
    def win_rate(trades: pd.DataFrame) -> float:
        """Fraction of winning trades.

        Args:
            trades: Trades DataFrame with a ``pnl`` column.

        Returns:
            Win rate between 0.0 and 1.0.
        """
        if trades.empty or "pnl" not in trades.columns:
            return 0.0
        total = len(trades)
        winners = int((trades["pnl"] > 0).sum())
        return float(winners / total)

    @staticmethod
    def profit_factor(trades: pd.DataFrame) -> float:
        """Gross profit divided by gross loss.

        Args:
            trades: Trades DataFrame with a ``pnl`` column.

        Returns:
            Profit factor.  Returns ``inf`` when gross loss is zero and
            there is profit, or 0.0 when there are no trades.
        """
        if trades.empty or "pnl" not in trades.columns:
            return 0.0

        gross_profit = float(trades.loc[trades["pnl"] > 0, "pnl"].sum())
        gross_loss = float(abs(trades.loc[trades["pnl"] < 0, "pnl"].sum()))

        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @staticmethod
    def average_expectancy(trades: pd.DataFrame) -> float:
        """Average profit per trade (expectancy).

        Args:
            trades: Trades DataFrame with a ``pnl`` column.

        Returns:
            Mean P&L across all trades.
        """
        if trades.empty or "pnl" not in trades.columns:
            return 0.0
        return float(trades["pnl"].mean())

    @staticmethod
    def trade_frequency(trades: pd.DataFrame, total_bars: int) -> float:
        """Average number of trades per day.

        Assumes the bars represent daily granularity.  For intraday bars,
        callers should adjust *total_bars* accordingly.

        Args:
            trades: Trades DataFrame.
            total_bars: Total number of bars in the backtest.

        Returns:
            Trades per day.
        """
        if total_bars <= 0:
            return 0.0
        return float(len(trades) / total_bars)

    @staticmethod
    def average_hold_time(trades: pd.DataFrame) -> float:
        """Average trade holding time in bars.

        Args:
            trades: Trades DataFrame with a ``hold_bars`` column.

        Returns:
            Mean holding time in bars.
        """
        if trades.empty or "hold_bars" not in trades.columns:
            return 0.0
        return float(trades["hold_bars"].mean())

    # ------------------------------------------------------------------
    # Strategy-level breakdown
    # ------------------------------------------------------------------

    @staticmethod
    def strategy_breakdown(trades: pd.DataFrame) -> dict[str, Any]:
        """Per-strategy performance breakdown.

        If the trades DataFrame contains a ``strategy`` column, metrics
        are computed per strategy.  Otherwise returns a single "unknown"
        bucket.

        Args:
            trades: Trades DataFrame, optionally with a ``strategy`` column.

        Returns:
            Dict mapping strategy name to a sub-dict of metrics.
        """
        if trades.empty or "pnl" not in trades.columns:
            return {}

        if "strategy" not in trades.columns:
            trades = trades.copy()
            trades["strategy"] = "unknown"

        breakdown: dict[str, Any] = {}
        for name, group in trades.groupby("strategy"):
            total = len(group)
            winners = int((group["pnl"] > 0).sum())
            gross_profit = float(group.loc[group["pnl"] > 0, "pnl"].sum())
            gross_loss = float(abs(group.loc[group["pnl"] < 0, "pnl"].sum()))
            pf = (
                gross_profit / gross_loss
                if gross_loss > 0
                else (float("inf") if gross_profit > 0 else 0.0)
            )

            breakdown[str(name)] = {
                "total_trades": total,
                "win_rate": round(winners / total, 4) if total > 0 else 0.0,
                "profit_factor": round(pf, 4),
                "total_pnl": round(float(group["pnl"].sum()), 4),
                "avg_pnl": round(float(group["pnl"].mean()), 4),
                "avg_hold_bars": (
                    round(float(group["hold_bars"].mean()), 2)
                    if "hold_bars" in group.columns
                    else 0.0
                ),
            }

        return breakdown

    # ------------------------------------------------------------------
    # Monthly P&L heatmap
    # ------------------------------------------------------------------

    @staticmethod
    def monthly_pnl_heatmap(equity: pd.Series) -> pd.DataFrame:
        """Generate a month-by-year P&L grid.

        Args:
            equity: Equity curve with a DatetimeIndex.

        Returns:
            DataFrame where rows are years and columns are months (1-12),
            with values being the monthly return as a decimal fraction.
            Non-datetime indices produce an empty DataFrame.
        """
        if equity.empty or not isinstance(equity.index, pd.DatetimeIndex):
            return pd.DataFrame()

        # Resample to monthly end-of-period equity.
        monthly_equity = equity.resample("ME").last().dropna()
        monthly_returns = monthly_equity.pct_change().dropna()

        if monthly_returns.empty:
            return pd.DataFrame()

        # Build year x month pivot.
        df = pd.DataFrame({
            "year": monthly_returns.index.year,
            "month": monthly_returns.index.month,
            "return": monthly_returns.values,
        })

        heatmap = df.pivot_table(
            index="year", columns="month", values="return", aggfunc="sum",
        ).fillna(0.0)

        # Ensure all 12 months exist as columns.
        for m in range(1, 13):
            if m not in heatmap.columns:
                heatmap[m] = 0.0
        heatmap = heatmap[sorted(heatmap.columns)]

        return heatmap

    # ------------------------------------------------------------------
    # Risk metrics
    # ------------------------------------------------------------------

    @staticmethod
    def value_at_risk(
        returns: pd.Series,
        confidence: float = 0.95,
    ) -> float:
        """Historical Value at Risk at the given confidence level.

        Args:
            returns: Series of per-period returns.
            confidence: Confidence level (e.g. 0.95 for 95th percentile).

        Returns:
            VaR as a negative float representing the loss threshold.
        """
        if returns.empty:
            return 0.0
        return float(np.percentile(returns.values, (1 - confidence) * 100))

    @staticmethod
    def conditional_var(
        returns: pd.Series,
        confidence: float = 0.95,
    ) -> float:
        """Conditional Value at Risk (Expected Shortfall).

        The average return in the worst ``(1 - confidence)`` fraction of
        observations.

        Args:
            returns: Series of per-period returns.
            confidence: Confidence level.

        Returns:
            CVaR as a negative float.
        """
        if returns.empty:
            return 0.0
        var_threshold = PerformanceMetrics.value_at_risk(returns, confidence)
        tail = returns[returns <= var_threshold]
        if tail.empty:
            return float(var_threshold)
        return float(tail.mean())
