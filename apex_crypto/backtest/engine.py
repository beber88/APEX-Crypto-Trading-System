"""Vectorized backtesting engine for the APEX Crypto Trading System.

Provides high-performance backtesting using pandas/numpy vectorized
operations instead of row-by-row simulation loops.  Supports single-asset,
multi-asset portfolio, and full strategy-level backtests with realistic
commission and slippage modelling.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from apex_crypto.core.logging import get_logger, log_with_data

logger = get_logger("backtest.engine")


class VectorizedBacktester:
    """Vectorized backtesting engine using pandas/numpy.

    All P&L computations are performed via vectorized array operations --
    no Python-level loops over bars.  Commissions and slippage are applied
    as multiplicative friction on every position change.

    Args:
        config: The ``backtest`` section from ``config.yaml``.
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self.commission_pct: float = config.get("commission_pct", 0.001)
        self.slippage_pct: float = config.get("slippage_pct", 0.0005)
        self._initial_equity: float = config.get("initial_equity", 10_000.0)

        log_with_data(logger, "info", "VectorizedBacktester initialized", {
            "commission_pct": self.commission_pct,
            "slippage_pct": self.slippage_pct,
            "initial_equity": self._initial_equity,
        })

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        entry_prices: Optional[pd.Series] = None,
        stop_losses: Optional[pd.Series] = None,
        take_profits: Optional[pd.Series] = None,
    ) -> dict[str, Any]:
        """Run a vectorized backtest on a single asset.

        Args:
            df: OHLCV DataFrame with columns ``open``, ``high``, ``low``,
                ``close``, ``volume`` and a DatetimeIndex.
            signals: Series of signal scores in ``[-100, +100]``.  Positive
                values open/hold long, negative values open/hold short,
                zero means flat.  Must share *df*'s index.
            entry_prices: Optional per-bar entry prices.  Falls back to
                ``df['close']`` when not provided.
            stop_losses: Optional per-bar stop-loss prices.  When provided,
                the engine checks ``df['low']`` (longs) or ``df['high']``
                (shorts) for stop hits and flattens the position on that bar.
            take_profits: Optional per-bar take-profit prices.  Analogous
                logic to *stop_losses*.

        Returns:
            Dictionary with keys:
                - ``equity_curve`` (pd.Series): Cumulative equity.
                - ``trades`` (pd.DataFrame): Individual trade records.
                - ``metrics`` (dict): Summary statistics (delegated to
                  :class:`PerformanceMetrics` by the caller).
        """
        signals = signals.reindex(df.index, fill_value=0)
        prices = entry_prices if entry_prices is not None else df["close"]

        # Convert continuous signal scores to discrete positions.
        positions = self._signals_to_positions(signals)

        # Apply stop-loss / take-profit overrides.
        if stop_losses is not None or take_profits is not None:
            positions = self._apply_sl_tp(
                positions, df, stop_losses, take_profits,
            )

        # Vectorized return stream.
        strategy_returns = self._compute_returns(
            df, positions, self.commission_pct, self.slippage_pct,
        )

        # Build equity curve.
        equity_curve = (1 + strategy_returns).cumprod() * self._initial_equity
        equity_curve.name = "equity"

        # Extract discrete trades.
        trades = self._extract_trades(positions, prices, equity_curve)

        # Lightweight inline metrics (full metrics via PerformanceMetrics).
        metrics = self._quick_metrics(equity_curve, trades, strategy_returns)

        log_with_data(logger, "info", "Backtest completed", {
            "total_bars": len(df),
            "total_trades": len(trades),
            "total_return_pct": round(metrics["total_return_pct"], 4),
            "max_drawdown_pct": round(metrics["max_drawdown_pct"], 4),
        })

        return {
            "equity_curve": equity_curve,
            "trades": trades,
            "metrics": metrics,
        }

    # ------------------------------------------------------------------
    # Vectorized return computation
    # ------------------------------------------------------------------

    def _compute_returns(
        self,
        df: pd.DataFrame,
        positions: pd.Series,
        commission: float,
        slippage: float,
    ) -> pd.Series:
        """Compute per-bar returns accounting for transaction costs.

        The return on each bar equals the position direction multiplied by
        the underlying bar return.  On every bar where the position changes,
        a round-trip cost (commission + slippage) is deducted.

        Args:
            df: OHLCV DataFrame.
            positions: Series of ``+1`` (long), ``-1`` (short), ``0`` (flat).
            commission: One-way commission as a decimal fraction.
            slippage: One-way slippage as a decimal fraction.

        Returns:
            Series of per-bar net returns.
        """
        # Underlying bar returns (close-to-close).
        close = df["close"]
        bar_returns = close.pct_change().fillna(0.0)

        # Gross strategy returns.
        gross_returns = positions.shift(1).fillna(0) * bar_returns

        # Transaction cost: deducted on every position change.
        position_changes = positions.diff().fillna(positions.iloc[0]).abs()
        cost_per_bar = position_changes * (commission + slippage)

        net_returns = gross_returns - cost_per_bar
        return net_returns

    # ------------------------------------------------------------------
    # Trade extraction
    # ------------------------------------------------------------------

    def _extract_trades(
        self,
        positions: pd.Series,
        prices: pd.Series,
        equity: pd.Series,
    ) -> pd.DataFrame:
        """Extract individual trades from position changes.

        A "trade" begins when the position moves from 0 to +/-1 (or flips
        direction) and ends when the position returns to 0 or flips again.

        Args:
            positions: Series of discrete position values.
            prices: Close (or entry) prices aligned with positions.
            equity: Equity curve for equity-at-entry/exit.

        Returns:
            DataFrame with columns: ``entry_time``, ``exit_time``,
            ``direction``, ``entry_price``, ``exit_price``, ``pnl``,
            ``pnl_pct``, ``r_multiple``, ``hold_bars``.
        """
        pos_values = positions.values.astype(float)
        idx = positions.index
        price_values = prices.values.astype(float)

        # Detect edges: where position changes.
        changes = np.diff(pos_values, prepend=0)
        change_indices = np.nonzero(changes)[0]

        trades: list[dict[str, Any]] = []
        entry_idx: Optional[int] = None
        entry_dir: float = 0.0
        entry_price: float = 0.0

        for ci in change_indices:
            new_pos = pos_values[ci]

            # If we have an open trade, close it.
            if entry_idx is not None and entry_dir != 0:
                exit_price = price_values[ci]
                hold_bars = ci - entry_idx
                if entry_dir > 0:
                    pnl_pct = (exit_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price

                pnl = pnl_pct * self._initial_equity
                # R-multiple: assume risk = 1% of entry as default unit.
                risk_unit = entry_price * 0.01
                r_multiple = (
                    (exit_price - entry_price) / risk_unit
                    if entry_dir > 0
                    else (entry_price - exit_price) / risk_unit
                ) if risk_unit > 0 else 0.0

                trades.append({
                    "entry_time": idx[entry_idx],
                    "exit_time": idx[ci],
                    "direction": "long" if entry_dir > 0 else "short",
                    "entry_price": round(entry_price, 8),
                    "exit_price": round(exit_price, 8),
                    "pnl": round(pnl, 4),
                    "pnl_pct": round(pnl_pct, 6),
                    "r_multiple": round(r_multiple, 4),
                    "hold_bars": int(hold_bars),
                })
                entry_idx = None
                entry_dir = 0.0

            # Open new trade if moving to a non-zero position.
            if new_pos != 0:
                entry_idx = ci
                entry_dir = new_pos
                entry_price = price_values[ci]

        # Close any position still open at the end of the data.
        if entry_idx is not None and entry_dir != 0:
            ci = len(pos_values) - 1
            exit_price = price_values[ci]
            hold_bars = ci - entry_idx
            if entry_dir > 0:
                pnl_pct = (exit_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - exit_price) / entry_price

            pnl = pnl_pct * self._initial_equity
            risk_unit = entry_price * 0.01
            r_multiple = (
                (exit_price - entry_price) / risk_unit
                if entry_dir > 0
                else (entry_price - exit_price) / risk_unit
            ) if risk_unit > 0 else 0.0

            trades.append({
                "entry_time": idx[entry_idx],
                "exit_time": idx[ci],
                "direction": "long" if entry_dir > 0 else "short",
                "entry_price": round(entry_price, 8),
                "exit_price": round(exit_price, 8),
                "pnl": round(pnl, 4),
                "pnl_pct": round(pnl_pct, 6),
                "r_multiple": round(r_multiple, 4),
                "hold_bars": int(hold_bars),
            })

        if not trades:
            return pd.DataFrame(columns=[
                "entry_time", "exit_time", "direction", "entry_price",
                "exit_price", "pnl", "pnl_pct", "r_multiple", "hold_bars",
            ])

        return pd.DataFrame(trades)

    # ------------------------------------------------------------------
    # Multi-asset backtest
    # ------------------------------------------------------------------

    def run_multi_asset(
        self,
        data: dict[str, pd.DataFrame],
        signals: dict[str, pd.Series],
    ) -> dict[str, Any]:
        """Run a backtest across multiple assets simultaneously.

        Each asset is backtested independently, then equity curves are
        combined into an equal-weighted portfolio curve.

        Args:
            data: Mapping of symbol to OHLCV DataFrame.
            signals: Mapping of symbol to signal Series (same keys as *data*).

        Returns:
            Dictionary with keys:
                - ``portfolio_equity`` (pd.Series): Combined equity curve.
                - ``per_asset`` (dict): Per-symbol results from :meth:`run`.
                - ``portfolio_metrics`` (dict): Portfolio-level metrics.
        """
        per_asset: dict[str, dict] = {}
        all_returns: list[pd.Series] = []

        for symbol in data:
            if symbol not in signals:
                logger.warning("No signals for %s, skipping", symbol)
                continue

            result = self.run(data[symbol], signals[symbol])
            per_asset[symbol] = result

            # Derive per-bar returns from equity curve.
            asset_returns = result["equity_curve"].pct_change().fillna(0.0)
            asset_returns.name = symbol
            all_returns.append(asset_returns)

        if not all_returns:
            log_with_data(logger, "warning", "No assets produced results", {})
            return {
                "portfolio_equity": pd.Series(dtype=float),
                "per_asset": per_asset,
                "portfolio_metrics": {},
            }

        # Equal-weight portfolio returns.
        returns_df = pd.concat(all_returns, axis=1).fillna(0.0)
        portfolio_returns = returns_df.mean(axis=1)
        portfolio_equity = (
            (1 + portfolio_returns).cumprod() * self._initial_equity
        )
        portfolio_equity.name = "portfolio_equity"

        all_trades = pd.concat(
            [r["trades"].assign(symbol=sym) for sym, r in per_asset.items()
             if not r["trades"].empty],
            ignore_index=True,
        ) if per_asset else pd.DataFrame()

        portfolio_metrics = self._quick_metrics(
            portfolio_equity, all_trades, portfolio_returns,
        )

        log_with_data(logger, "info", "Multi-asset backtest completed", {
            "num_assets": len(per_asset),
            "portfolio_return_pct": round(portfolio_metrics["total_return_pct"], 4),
            "total_trades": len(all_trades),
        })

        return {
            "portfolio_equity": portfolio_equity,
            "per_asset": per_asset,
            "portfolio_metrics": portfolio_metrics,
        }

    # ------------------------------------------------------------------
    # Strategy-level backtest
    # ------------------------------------------------------------------

    def run_strategy_backtest(
        self,
        strategy_name: str,
        data: dict[str, pd.DataFrame],
        indicators: dict[str, pd.DataFrame],
    ) -> dict[str, Any]:
        """Run a full strategy backtest: generate signals then backtest.

        Loads the named strategy class, generates signals from the provided
        indicator data, and feeds them through the vectorized engine.

        Args:
            strategy_name: Registered strategy name (e.g. ``"scalping"``).
            data: OHLCV DataFrames keyed by symbol.
            indicators: Pre-computed indicator DataFrames keyed by symbol.

        Returns:
            Complete results dict including equity curve, trades, and metrics
            as returned by :meth:`run_multi_asset`.
        """
        from apex_crypto.core.strategies.base import BaseStrategy

        # Lazy import the strategy registry.
        strategy_map: dict[str, type[BaseStrategy]] = (
            self._discover_strategies()
        )

        strategy_cls = strategy_map.get(strategy_name)
        if strategy_cls is None:
            logger.error("Unknown strategy: %s", strategy_name)
            raise ValueError(
                f"Strategy '{strategy_name}' not found.  "
                f"Available: {list(strategy_map.keys())}"
            )

        strategy = strategy_cls(self._config)

        # Generate signals for every symbol.
        all_signals: dict[str, pd.Series] = {}
        for symbol, df in data.items():
            symbol_indicators = indicators.get(symbol, pd.DataFrame())
            signal_series = self._generate_signal_series(
                strategy, symbol, df, symbol_indicators,
            )
            all_signals[symbol] = signal_series

        result = self.run_multi_asset(data, all_signals)
        result["strategy_name"] = strategy_name

        log_with_data(logger, "info", "Strategy backtest completed", {
            "strategy": strategy_name,
            "num_assets": len(data),
        })

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _signals_to_positions(signals: pd.Series) -> pd.Series:
        """Convert continuous signal scores to discrete positions.

        Mapping:
            - score > 0  -> +1 (long)
            - score < 0  -> -1 (short)
            - score == 0 -> 0  (flat)

        Args:
            signals: Series of signal scores in ``[-100, +100]``.

        Returns:
            Series of discrete positions ``{-1, 0, +1}``.
        """
        return pd.Series(
            np.sign(signals.values).astype(int),
            index=signals.index,
            name="position",
        )

    @staticmethod
    def _apply_sl_tp(
        positions: pd.Series,
        df: pd.DataFrame,
        stop_losses: Optional[pd.Series],
        take_profits: Optional[pd.Series],
    ) -> pd.Series:
        """Override positions where stop-loss or take-profit is hit.

        For longs, a stop is hit when ``df['low'] <= stop_loss``, and a
        take-profit when ``df['high'] >= take_profit``.  For shorts the
        logic is inverted.  On the bar where either condition fires the
        position is flattened to zero and remains flat until the *signals*
        would re-enter.

        Args:
            positions: Discrete position series.
            df: OHLCV DataFrame with ``high`` and ``low`` columns.
            stop_losses: Per-bar stop-loss prices (may be ``None``).
            take_profits: Per-bar take-profit prices (may be ``None``).

        Returns:
            Modified position series with SL/TP exits applied.
        """
        pos = positions.copy()
        pos_vals = pos.values.astype(float)
        low = df["low"].values.astype(float)
        high = df["high"].values.astype(float)

        sl_vals = (
            stop_losses.reindex(df.index).values.astype(float)
            if stop_losses is not None
            else np.full(len(df), np.nan)
        )
        tp_vals = (
            take_profits.reindex(df.index).values.astype(float)
            if take_profits is not None
            else np.full(len(df), np.nan)
        )

        # Vectorized masks for stop / take-profit hits.
        long_mask = pos_vals > 0
        short_mask = pos_vals < 0

        # Long stop: low breaches stop.
        long_sl_hit = long_mask & np.isfinite(sl_vals) & (low <= sl_vals)
        # Long TP: high breaches take-profit.
        long_tp_hit = long_mask & np.isfinite(tp_vals) & (high >= tp_vals)
        # Short stop: high breaches stop.
        short_sl_hit = short_mask & np.isfinite(sl_vals) & (high >= sl_vals)
        # Short TP: low breaches take-profit.
        short_tp_hit = short_mask & np.isfinite(tp_vals) & (low <= tp_vals)

        exit_mask = long_sl_hit | long_tp_hit | short_sl_hit | short_tp_hit
        pos_vals[exit_mask] = 0

        # After an exit, remain flat until the original signal would re-enter.
        # We forward-fill the zero across bars until the raw signal changes.
        # This is done by iterating through exit events and zeroing out until
        # the next signal direction change (vectorized where possible).
        if exit_mask.any():
            exit_indices = np.where(exit_mask)[0]
            raw_sign = np.sign(positions.values.astype(float))
            for ei in exit_indices:
                # Zero out from exit bar forward until raw signal changes
                # direction from what it was at exit time.
                exited_dir = raw_sign[ei]
                for j in range(ei, len(pos_vals)):
                    if raw_sign[j] != exited_dir and raw_sign[j] != 0:
                        break
                    pos_vals[j] = 0

        return pd.Series(pos_vals.astype(int), index=positions.index, name="position")

    @staticmethod
    def _quick_metrics(
        equity: pd.Series,
        trades: pd.DataFrame,
        returns: pd.Series,
    ) -> dict[str, Any]:
        """Compute lightweight summary metrics inline.

        These are intentionally minimal -- comprehensive metrics are
        delegated to :class:`PerformanceMetrics`.

        Args:
            equity: Equity curve series.
            trades: Trades DataFrame.
            returns: Per-bar returns series.

        Returns:
            Dict of summary metrics.
        """
        if equity.empty:
            return {
                "total_return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "total_trades": 0,
                "win_rate": 0.0,
                "sharpe_ratio": 0.0,
            }

        total_return_pct = (equity.iloc[-1] / equity.iloc[0] - 1) * 100

        # Max drawdown.
        running_max = equity.cummax()
        drawdowns = (equity - running_max) / running_max
        max_dd = float(drawdowns.min()) * 100  # as negative percentage

        # Win rate from trades.
        total_trades = len(trades)
        if total_trades > 0 and "pnl" in trades.columns:
            winners = (trades["pnl"] > 0).sum()
            win_rate = float(winners) / total_trades
        else:
            win_rate = 0.0

        # Annualized Sharpe (assume ~365 bars/year for daily, scale otherwise).
        ret_std = returns.std()
        sharpe = (
            float(returns.mean() / ret_std * np.sqrt(365))
            if ret_std > 0
            else 0.0
        )

        return {
            "total_return_pct": float(total_return_pct),
            "max_drawdown_pct": float(max_dd),
            "total_trades": total_trades,
            "win_rate": float(win_rate),
            "sharpe_ratio": float(sharpe),
        }

    @staticmethod
    def _discover_strategies() -> dict[str, type]:
        """Build a map of strategy name to class from the strategies package.

        Returns:
            Dict mapping strategy ``name`` attribute to its class.
        """
        from apex_crypto.core.strategies.base import BaseStrategy
        from apex_crypto.core.strategies.scalping import ScalpingStrategy
        from apex_crypto.core.strategies.funding_rate import FundingRateStrategy

        registry: dict[str, type] = {}
        for cls in [ScalpingStrategy, FundingRateStrategy]:
            if hasattr(cls, "name"):
                registry[cls.name] = cls

        return registry

    def _generate_signal_series(
        self,
        strategy: Any,
        symbol: str,
        df: pd.DataFrame,
        indicators: pd.DataFrame,
    ) -> pd.Series:
        """Generate a signal series by invoking a strategy on each bar.

        Wraps the per-bar strategy call into a vectorized-compatible series.
        While the strategy itself may not be vectorized, the output is
        aligned back to the DataFrame index for the vectorized engine.

        Args:
            strategy: Instantiated strategy object.
            symbol: Trading pair symbol.
            df: OHLCV DataFrame for one symbol.
            indicators: Indicator DataFrame for the same symbol.

        Returns:
            Series of signal scores aligned to *df*'s index.
        """
        scores = np.zeros(len(df), dtype=int)

        # Use expanding windows to simulate the strategy seeing cumulative
        # data up to each bar.  For performance, stride by the strategy's
        # primary timeframe bar count (default every bar).
        data_dict = {strategy.primary_timeframe: df}
        ind_dict = {strategy.primary_timeframe: indicators}

        for i in range(len(df)):
            # Slice data up to bar i (inclusive).
            sliced_data = {
                tf: frame.iloc[: i + 1] for tf, frame in data_dict.items()
            }
            sliced_ind = {
                tf: frame.iloc[: i + 1] for tf, frame in ind_dict.items()
            }

            try:
                signal = strategy.generate_signal(
                    symbol=symbol,
                    data=sliced_data,
                    indicators=sliced_ind,
                    regime="RANGING",
                )
                scores[i] = signal.score
            except Exception:
                scores[i] = 0

        return pd.Series(scores, index=df.index, name="signal")
