"""Walk-forward optimisation for the APEX Crypto Trading System.

Implements rolling in-sample / out-of-sample validation to guard against
overfitting.  Includes grid-search parameter optimisation and sensitivity
analysis for individual parameters.
"""

from __future__ import annotations

import itertools
from typing import Any, Optional

import numpy as np
import pandas as pd

from apex_crypto.backtest.engine import VectorizedBacktester
from apex_crypto.backtest.metrics import PerformanceMetrics
from apex_crypto.core.logging import get_logger, log_with_data
from apex_crypto.core.strategies.base import BaseStrategy

logger = get_logger("backtest.wfo")


class WalkForwardOptimizer:
    """Rolling walk-forward optimiser with parameter sensitivity analysis.

    Splits historical data into sequential in-sample (IS) / out-of-sample
    (OOS) windows.  For each window the parameter grid is searched to
    maximise the Sharpe ratio in-sample, then the best parameters are
    validated out-of-sample.

    Args:
        config: The ``backtest`` (or ``wfo``) section from ``config.yaml``.
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self.wfo_insample_months: int = config.get("wfo_insample_months", 12)
        self.wfo_outsample_months: int = config.get("wfo_outsample_months", 4)
        self._objective: str = config.get("wfo_objective", "sharpe_ratio")
        self._max_combinations: int = config.get("wfo_max_combinations", 500)

        log_with_data(logger, "info", "WalkForwardOptimizer initialized", {
            "insample_months": self.wfo_insample_months,
            "outsample_months": self.wfo_outsample_months,
            "objective": self._objective,
            "max_combinations": self._max_combinations,
        })

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def run_wfo(
        self,
        data: dict[str, pd.DataFrame],
        strategy_class: type[BaseStrategy],
        param_grid: dict[str, list[Any]],
    ) -> dict[str, Any]:
        """Execute walk-forward optimisation across rolling windows.

        Args:
            data: OHLCV DataFrames keyed by symbol.  All frames must share
                a ``DatetimeIndex``.
            strategy_class: The strategy class to optimise (not an instance).
            param_grid: Mapping of parameter name to a list of candidate
                values.  Example::

                    {
                        "vwap_deviation_min": [0.002, 0.003, 0.004],
                        "stop_pct": [0.003, 0.004, 0.005],
                    }

        Returns:
            Dictionary with keys:
                - ``windows`` (list[dict]): Per-window IS/OOS results.
                - ``oos_equity`` (pd.Series): Concatenated OOS equity curve.
                - ``oos_metrics`` (dict): Aggregate OOS performance metrics.
                - ``best_params_per_window`` (list[dict]): Best parameters
                  found in each IS window.
        """
        # Determine date range from data.
        all_dates = self._collect_dates(data)
        if all_dates.empty:
            logger.error("No datetime data found for WFO windows")
            return self._empty_wfo_result()

        windows = self._build_windows(all_dates)
        if not windows:
            logger.error("Insufficient data for even one WFO window")
            return self._empty_wfo_result()

        window_results: list[dict[str, Any]] = []
        best_params_list: list[dict[str, Any]] = []
        oos_equities: list[pd.Series] = []

        for i, (is_start, is_end, oos_start, oos_end) in enumerate(windows):
            log_with_data(logger, "info", f"WFO window {i + 1}/{len(windows)}", {
                "is_start": str(is_start),
                "is_end": str(is_end),
                "oos_start": str(oos_start),
                "oos_end": str(oos_end),
            })

            # In-sample optimisation.
            is_result = self._optimize_window(
                data, strategy_class, param_grid, is_start, is_end,
            )
            best_params = is_result["best_params"]
            best_params_list.append(best_params)

            # Out-of-sample validation.
            oos_result = self._validate_window(
                data, strategy_class, best_params, oos_start, oos_end,
            )

            window_results.append({
                "window_index": i,
                "is_start": str(is_start),
                "is_end": str(is_end),
                "oos_start": str(oos_start),
                "oos_end": str(oos_end),
                "best_params": best_params,
                "is_metrics": is_result["metrics"],
                "oos_metrics": oos_result["metrics"],
            })

            if not oos_result["equity"].empty:
                oos_equities.append(oos_result["equity"])

        # Concatenate OOS equity curves.
        if oos_equities:
            oos_equity = self._stitch_equity_curves(oos_equities)
            oos_returns = oos_equity.pct_change().fillna(0.0)
            oos_all_trades = pd.concat(
                [w.get("oos_trades", pd.DataFrame()) for w in window_results
                 if "oos_trades" in w and not w.get("oos_trades", pd.DataFrame()).empty],
                ignore_index=True,
            ) if any("oos_trades" in w for w in window_results) else pd.DataFrame()

            oos_metrics = PerformanceMetrics.compute_all(
                oos_equity, oos_all_trades,
            )
        else:
            oos_equity = pd.Series(dtype=float)
            oos_metrics = {}

        log_with_data(logger, "info", "WFO completed", {
            "num_windows": len(windows),
            "oos_total_return": oos_metrics.get("total_return", 0),
            "oos_sharpe": oos_metrics.get("sharpe_ratio", 0),
        })

        return {
            "windows": window_results,
            "oos_equity": oos_equity,
            "oos_metrics": oos_metrics,
            "best_params_per_window": best_params_list,
        }

    # ------------------------------------------------------------------
    # In-sample optimisation
    # ------------------------------------------------------------------

    def _optimize_window(
        self,
        data: dict[str, pd.DataFrame],
        strategy_class: type[BaseStrategy],
        param_grid: dict[str, list[Any]],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> dict[str, Any]:
        """Grid search over *param_grid* on the in-sample period.

        When the total number of parameter combinations exceeds
        ``wfo_max_combinations``, a random subset is sampled.

        Args:
            data: Full OHLCV data dict.
            strategy_class: Strategy class to instantiate.
            param_grid: Parameter search space.
            start: IS period start (inclusive).
            end: IS period end (inclusive).

        Returns:
            Dict with ``best_params`` and ``metrics`` for the best run.
        """
        # Slice data to IS window.
        is_data = self._slice_data(data, start, end)

        # Build all parameter combinations.
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combos = list(itertools.product(*param_values))

        # Random subsample if too many.
        rng = np.random.default_rng(42)
        if len(all_combos) > self._max_combinations:
            indices = rng.choice(
                len(all_combos), size=self._max_combinations, replace=False,
            )
            all_combos = [all_combos[i] for i in indices]

        best_score: float = -np.inf
        best_params: dict[str, Any] = {}
        best_metrics: dict[str, Any] = {}

        backtester = VectorizedBacktester(self._config)

        for combo in all_combos:
            params = dict(zip(param_names, combo))

            try:
                # Merge params into strategy config.
                strategy_config = self._build_strategy_config(
                    strategy_class, params,
                )
                strategy = strategy_class(strategy_config)

                # Generate signals and run backtest.
                signals: dict[str, pd.Series] = {}
                for symbol, df in is_data.items():
                    sig = backtester._generate_signal_series(
                        strategy, symbol, df, df,
                    )
                    signals[symbol] = sig

                result = backtester.run_multi_asset(is_data, signals)
                metrics = result.get("portfolio_metrics", {})
                score = metrics.get(self._objective, 0.0)

                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    best_metrics = metrics.copy()

            except Exception as exc:
                logger.debug(
                    "Parameter combo failed: %s — %s", params, str(exc),
                )
                continue

        log_with_data(logger, "info", "IS optimisation complete", {
            "combos_tested": len(all_combos),
            "best_score": round(best_score, 6),
            "best_params": best_params,
        })

        return {"best_params": best_params, "metrics": best_metrics}

    # ------------------------------------------------------------------
    # Out-of-sample validation
    # ------------------------------------------------------------------

    def _validate_window(
        self,
        data: dict[str, pd.DataFrame],
        strategy_class: type[BaseStrategy],
        params: dict[str, Any],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> dict[str, Any]:
        """Validate a parameter set on the out-of-sample period.

        Args:
            data: Full OHLCV data dict.
            strategy_class: Strategy class.
            params: Optimised parameter dict from IS.
            start: OOS period start.
            end: OOS period end.

        Returns:
            Dict with ``equity`` (Series), ``metrics`` (dict), and
            ``trades`` (DataFrame).
        """
        oos_data = self._slice_data(data, start, end)

        if not oos_data:
            logger.warning("No OOS data in window %s -> %s", start, end)
            return {
                "equity": pd.Series(dtype=float),
                "metrics": {},
                "trades": pd.DataFrame(),
            }

        backtester = VectorizedBacktester(self._config)

        strategy_config = self._build_strategy_config(strategy_class, params)
        strategy = strategy_class(strategy_config)

        signals: dict[str, pd.Series] = {}
        for symbol, df in oos_data.items():
            sig = backtester._generate_signal_series(strategy, symbol, df, df)
            signals[symbol] = sig

        result = backtester.run_multi_asset(oos_data, signals)

        equity = result.get("portfolio_equity", pd.Series(dtype=float))
        metrics = result.get("portfolio_metrics", {})

        # Collect trades from per-asset results.
        all_trades = []
        for sym, asset_res in result.get("per_asset", {}).items():
            t = asset_res.get("trades", pd.DataFrame())
            if not t.empty:
                t = t.copy()
                t["symbol"] = sym
                all_trades.append(t)
        trades = (
            pd.concat(all_trades, ignore_index=True) if all_trades
            else pd.DataFrame()
        )

        log_with_data(logger, "info", "OOS validation complete", {
            "start": str(start),
            "end": str(end),
            "oos_return": metrics.get("total_return_pct", 0),
            "oos_sharpe": metrics.get("sharpe_ratio", 0),
            "oos_trades": len(trades),
        })

        return {"equity": equity, "metrics": metrics, "trades": trades}

    # ------------------------------------------------------------------
    # Parameter sensitivity
    # ------------------------------------------------------------------

    def parameter_sensitivity(
        self,
        data: dict[str, pd.DataFrame],
        strategy_class: type[BaseStrategy],
        base_params: dict[str, Any],
        vary_pct: float = 0.2,
    ) -> dict[str, Any]:
        """Analyse sensitivity by varying each parameter +/- *vary_pct*.

        For each parameter in *base_params*, five values are tested:
        ``base * (1 - vary_pct)``, ``base * (1 - vary_pct/2)``, ``base``,
        ``base * (1 + vary_pct/2)``, ``base * (1 + vary_pct)``.

        Args:
            data: OHLCV DataFrames keyed by symbol.
            strategy_class: Strategy class to test.
            base_params: Baseline parameter dictionary.
            vary_pct: Fractional variation (0.2 = +/-20%).

        Returns:
            Dict mapping parameter name to::

                {
                    "values_tested": list[float],
                    "metrics_at_each_value": list[dict],
                }
        """
        sensitivity: dict[str, Any] = {}
        backtester = VectorizedBacktester(self._config)

        for param_name, base_value in base_params.items():
            # Skip non-numeric parameters.
            if not isinstance(base_value, (int, float)):
                logger.debug(
                    "Skipping non-numeric param '%s' for sensitivity",
                    param_name,
                )
                continue

            # Build test values.
            multipliers = [
                1 - vary_pct,
                1 - vary_pct / 2,
                1.0,
                1 + vary_pct / 2,
                1 + vary_pct,
            ]
            test_values = [base_value * m for m in multipliers]

            # Preserve int type.
            if isinstance(base_value, int):
                test_values = [max(1, int(round(v))) for v in test_values]
                # Deduplicate while preserving order.
                seen: set[int] = set()
                deduped: list[int] = []
                for v in test_values:
                    if v not in seen:
                        seen.add(v)
                        deduped.append(v)
                test_values = deduped

            metrics_list: list[dict[str, Any]] = []

            for value in test_values:
                test_params = base_params.copy()
                test_params[param_name] = value

                try:
                    strategy_config = self._build_strategy_config(
                        strategy_class, test_params,
                    )
                    strategy = strategy_class(strategy_config)

                    signals: dict[str, pd.Series] = {}
                    for symbol, df in data.items():
                        sig = backtester._generate_signal_series(
                            strategy, symbol, df, df,
                        )
                        signals[symbol] = sig

                    result = backtester.run_multi_asset(data, signals)
                    metrics = result.get("portfolio_metrics", {})

                except Exception as exc:
                    logger.debug(
                        "Sensitivity test failed for %s=%s: %s",
                        param_name, value, str(exc),
                    )
                    metrics = {"error": str(exc)}

                metrics_list.append(metrics)

            sensitivity[param_name] = {
                "values_tested": test_values,
                "metrics_at_each_value": metrics_list,
            }

        log_with_data(logger, "info", "Parameter sensitivity analysis complete", {
            "num_params_tested": len(sensitivity),
            "vary_pct": vary_pct,
        })

        return sensitivity

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_dates(data: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
        """Extract the union of all datetime indices across assets.

        Args:
            data: OHLCV DataFrames keyed by symbol.

        Returns:
            Sorted ``DatetimeIndex`` covering all assets.
        """
        all_idx: list[pd.DatetimeIndex] = []
        for df in data.values():
            if isinstance(df.index, pd.DatetimeIndex):
                all_idx.append(df.index)

        if not all_idx:
            return pd.DatetimeIndex([])

        combined = all_idx[0]
        for idx in all_idx[1:]:
            combined = combined.union(idx)
        return combined.sort_values()

    def _build_windows(
        self,
        dates: pd.DatetimeIndex,
    ) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Build rolling IS/OOS window boundaries.

        Args:
            dates: Sorted datetime index of all available dates.

        Returns:
            List of ``(is_start, is_end, oos_start, oos_end)`` tuples.
        """
        start = dates.min()
        end = dates.max()
        windows: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []

        is_start = start
        while True:
            is_end = is_start + pd.DateOffset(months=self.wfo_insample_months)
            oos_start = is_end
            oos_end = oos_start + pd.DateOffset(months=self.wfo_outsample_months)

            if oos_end > end:
                break

            windows.append((
                pd.Timestamp(is_start),
                pd.Timestamp(is_end),
                pd.Timestamp(oos_start),
                pd.Timestamp(oos_end),
            ))

            # Roll forward by the OOS period length.
            is_start = is_start + pd.DateOffset(months=self.wfo_outsample_months)

        return windows

    @staticmethod
    def _slice_data(
        data: dict[str, pd.DataFrame],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> dict[str, pd.DataFrame]:
        """Slice all DataFrames to the given date range.

        Args:
            data: Full OHLCV data dict.
            start: Start timestamp (inclusive).
            end: End timestamp (inclusive).

        Returns:
            Sliced data dict (symbols with no data in range are omitted).
        """
        sliced: dict[str, pd.DataFrame] = {}
        for symbol, df in data.items():
            if not isinstance(df.index, pd.DatetimeIndex):
                sliced[symbol] = df
                continue
            mask = (df.index >= start) & (df.index <= end)
            subset = df.loc[mask]
            if not subset.empty:
                sliced[symbol] = subset
        return sliced

    @staticmethod
    def _build_strategy_config(
        strategy_class: type[BaseStrategy],
        params: dict[str, Any],
    ) -> dict:
        """Merge optimisation parameters into a strategy config dict.

        Nests parameters under the strategy's ``name`` attribute within
        the ``strategies`` config section, matching the pattern used by
        :class:`ScalpingStrategy` and others.

        Args:
            strategy_class: The strategy class.
            params: Flat parameter dict to inject.

        Returns:
            Config dict suitable for strategy initialisation.
        """
        name = getattr(strategy_class, "name", "unknown")
        return {
            "enabled": True,
            "strategies": {
                name: params,
            },
        }

    @staticmethod
    def _stitch_equity_curves(
        curves: list[pd.Series],
    ) -> pd.Series:
        """Concatenate sequential equity curves into a continuous series.

        Each subsequent curve is rescaled so that its starting value
        matches the ending value of the previous curve.

        Args:
            curves: Ordered list of OOS equity curve segments.

        Returns:
            Single continuous equity series.
        """
        if not curves:
            return pd.Series(dtype=float)

        stitched_parts: list[pd.Series] = [curves[0]]
        last_value = float(curves[0].iloc[-1])

        for curve in curves[1:]:
            if curve.empty:
                continue
            scale = last_value / float(curve.iloc[0]) if curve.iloc[0] != 0 else 1.0
            scaled = curve * scale
            stitched_parts.append(scaled)
            last_value = float(scaled.iloc[-1])

        return pd.concat(stitched_parts)

    @staticmethod
    def _empty_wfo_result() -> dict[str, Any]:
        """Return a zeroed-out WFO result dict.

        Returns:
            Empty WFO result structure.
        """
        return {
            "windows": [],
            "oos_equity": pd.Series(dtype=float),
            "oos_metrics": {},
            "best_params_per_window": [],
        }
