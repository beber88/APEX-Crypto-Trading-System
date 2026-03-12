"""Monte Carlo simulation for the APEX Crypto Trading System.

Assesses strategy robustness by bootstrapping trade sequences and
injecting execution noise.  Produces confidence intervals, ruin
probabilities, and fan-chart visualisation data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from apex_crypto.core.logging import get_logger, log_with_data

logger = get_logger("backtest.montecarlo")


class MonteCarloSimulator:
    """Monte Carlo bootstrap simulator for trade-level analysis.

    Generates thousands of synthetic equity curves by randomly
    reordering observed trades, then derives distributional statistics
    such as probability of profit, probability of ruin, and expected
    maximum drawdown.

    Args:
        config: The ``backtest`` (or ``monte_carlo``) section from
            ``config.yaml``.
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self.monte_carlo_runs: int = config.get("monte_carlo_runs", 1000)

        log_with_data(logger, "info", "MonteCarloSimulator initialized", {
            "default_runs": self.monte_carlo_runs,
        })

    # ------------------------------------------------------------------
    # Primary simulation (trade reordering)
    # ------------------------------------------------------------------

    def run_simulation(
        self,
        trades: pd.DataFrame,
        num_simulations: int = 1000,
        initial_equity: float = 10_000.0,
    ) -> dict[str, Any]:
        """Run Monte Carlo simulation via bootstrap trade reordering.

        Each simulation shuffles the sequence of observed trades, then
        reconstructs an equity curve by applying each trade's percentage
        return in the new random order.

        Args:
            trades: Trades DataFrame produced by the backtester.  Must
                contain a ``pnl_pct`` column (per-trade return as a
                decimal fraction).
            num_simulations: Number of random permutations to run.
            initial_equity: Starting equity for each simulation path.

        Returns:
            Dictionary with keys:
                - ``simulations`` (np.ndarray): Array of shape
                  ``(num_simulations, num_trades + 1)`` containing
                  equity values for every path (column 0 is initial
                  equity).
                - ``median_equity`` (pd.Series): 50th percentile path.
                - ``p5_equity`` (pd.Series): 5th percentile path.
                - ``p25_equity`` (pd.Series): 25th percentile path.
                - ``p75_equity`` (pd.Series): 75th percentile path.
                - ``p95_equity`` (pd.Series): 95th percentile path.
                - ``probability_of_profit`` (float): Fraction of paths
                  ending above *initial_equity*.
                - ``probability_of_ruin`` (float): Fraction of paths
                  experiencing a drawdown > 50%.
                - ``expected_max_drawdown`` (float): Mean of per-path
                  maximum drawdowns (as a negative decimal).
                - ``confidence_intervals`` (dict): Terminal equity at
                  various percentiles.
        """
        if trades.empty or "pnl_pct" not in trades.columns:
            logger.warning("No trades or missing pnl_pct column for MC sim")
            return self._empty_simulation_result(initial_equity)

        trade_returns = trades["pnl_pct"].values.astype(np.float64)
        num_trades = len(trade_returns)
        rng = np.random.default_rng()

        # Pre-allocate simulation matrix: (num_sims x num_trades + 1).
        simulations = np.empty(
            (num_simulations, num_trades + 1), dtype=np.float64,
        )
        simulations[:, 0] = initial_equity

        # Generate all shuffled index arrays at once for performance.
        # Each row of shuffle_indices is a permutation of [0, num_trades).
        shuffle_indices = np.array([
            rng.permutation(num_trades) for _ in range(num_simulations)
        ])

        # Gather shuffled returns: (num_sims x num_trades).
        shuffled_returns = trade_returns[shuffle_indices]

        # Vectorized cumulative product to build equity curves.
        # equity[i] = initial * prod(1 + r_j for j in 0..i-1)
        growth_factors = 1.0 + shuffled_returns  # (num_sims x num_trades)
        cum_growth = np.cumprod(growth_factors, axis=1)  # (num_sims x num_trades)
        simulations[:, 1:] = initial_equity * cum_growth

        # Percentile paths.
        p5 = np.percentile(simulations, 5, axis=0)
        p25 = np.percentile(simulations, 25, axis=0)
        p50 = np.percentile(simulations, 50, axis=0)
        p75 = np.percentile(simulations, 75, axis=0)
        p95 = np.percentile(simulations, 95, axis=0)

        trade_index = pd.RangeIndex(num_trades + 1, name="trade_num")

        # Terminal equity stats.
        terminal_equities = simulations[:, -1]
        probability_of_profit = float(
            np.mean(terminal_equities > initial_equity)
        )

        # Per-path max drawdown.
        max_drawdowns = self._compute_max_drawdowns(simulations)
        probability_of_ruin = float(np.mean(max_drawdowns < -0.50))
        expected_max_drawdown = float(np.mean(max_drawdowns))

        # Confidence intervals on terminal equity.
        confidence_intervals = {
            "p1": float(np.percentile(terminal_equities, 1)),
            "p5": float(np.percentile(terminal_equities, 5)),
            "p10": float(np.percentile(terminal_equities, 10)),
            "p25": float(np.percentile(terminal_equities, 25)),
            "p50": float(np.percentile(terminal_equities, 50)),
            "p75": float(np.percentile(terminal_equities, 75)),
            "p90": float(np.percentile(terminal_equities, 90)),
            "p95": float(np.percentile(terminal_equities, 95)),
            "p99": float(np.percentile(terminal_equities, 99)),
            "mean": float(np.mean(terminal_equities)),
            "std": float(np.std(terminal_equities)),
        }

        log_with_data(logger, "info", "Monte Carlo simulation complete", {
            "num_simulations": num_simulations,
            "num_trades": num_trades,
            "probability_of_profit": round(probability_of_profit, 4),
            "probability_of_ruin": round(probability_of_ruin, 4),
            "expected_max_drawdown": round(expected_max_drawdown, 4),
            "median_terminal": round(confidence_intervals["p50"], 2),
        })

        return {
            "simulations": simulations,
            "median_equity": pd.Series(p50, index=trade_index, name="p50"),
            "p5_equity": pd.Series(p5, index=trade_index, name="p5"),
            "p25_equity": pd.Series(p25, index=trade_index, name="p25"),
            "p75_equity": pd.Series(p75, index=trade_index, name="p75"),
            "p95_equity": pd.Series(p95, index=trade_index, name="p95"),
            "probability_of_profit": probability_of_profit,
            "probability_of_ruin": probability_of_ruin,
            "expected_max_drawdown": expected_max_drawdown,
            "confidence_intervals": confidence_intervals,
        }

    # ------------------------------------------------------------------
    # Noise injection simulation
    # ------------------------------------------------------------------

    def run_with_noise(
        self,
        trades: pd.DataFrame,
        noise_pct: float = 0.1,
        num_simulations: int = 1000,
    ) -> dict[str, Any]:
        """Run Monte Carlo with random noise added to each trade's PnL.

        Tests robustness to execution variance (slippage variability,
        partial fills, timing jitter) by perturbing each trade's
        ``pnl_pct`` by a uniform random amount in
        ``[-noise_pct, +noise_pct]`` relative to the trade's return.

        Args:
            trades: Trades DataFrame with ``pnl_pct`` column.
            noise_pct: Maximum fractional perturbation applied to each
                trade's PnL percentage (e.g. 0.1 = +/-10% of each
                trade's return magnitude).
            num_simulations: Number of noisy simulations.

        Returns:
            Same structure as :meth:`run_simulation`, plus an additional
            ``noise_pct`` key recording the perturbation level.
        """
        if trades.empty or "pnl_pct" not in trades.columns:
            logger.warning("No trades for noise simulation")
            return {**self._empty_simulation_result(10_000.0), "noise_pct": noise_pct}

        rng = np.random.default_rng()
        trade_returns = trades["pnl_pct"].values.astype(np.float64)
        num_trades = len(trade_returns)
        initial_equity = 10_000.0

        # Pre-allocate.
        simulations = np.empty(
            (num_simulations, num_trades + 1), dtype=np.float64,
        )
        simulations[:, 0] = initial_equity

        # For each simulation: shuffle order AND add noise.
        shuffle_indices = np.array([
            rng.permutation(num_trades) for _ in range(num_simulations)
        ])
        shuffled_returns = trade_returns[shuffle_indices]

        # Add noise: uniform perturbation proportional to absolute return.
        abs_returns = np.abs(shuffled_returns)
        noise = rng.uniform(
            -noise_pct, noise_pct,
            size=(num_simulations, num_trades),
        ) * abs_returns
        noisy_returns = shuffled_returns + noise

        # Build equity curves.
        growth_factors = 1.0 + noisy_returns
        cum_growth = np.cumprod(growth_factors, axis=1)
        simulations[:, 1:] = initial_equity * cum_growth

        # Statistics.
        p5 = np.percentile(simulations, 5, axis=0)
        p25 = np.percentile(simulations, 25, axis=0)
        p50 = np.percentile(simulations, 50, axis=0)
        p75 = np.percentile(simulations, 75, axis=0)
        p95 = np.percentile(simulations, 95, axis=0)

        trade_index = pd.RangeIndex(num_trades + 1, name="trade_num")
        terminal_equities = simulations[:, -1]

        probability_of_profit = float(
            np.mean(terminal_equities > initial_equity)
        )
        max_drawdowns = self._compute_max_drawdowns(simulations)
        probability_of_ruin = float(np.mean(max_drawdowns < -0.50))
        expected_max_drawdown = float(np.mean(max_drawdowns))

        confidence_intervals = {
            "p1": float(np.percentile(terminal_equities, 1)),
            "p5": float(np.percentile(terminal_equities, 5)),
            "p10": float(np.percentile(terminal_equities, 10)),
            "p25": float(np.percentile(terminal_equities, 25)),
            "p50": float(np.percentile(terminal_equities, 50)),
            "p75": float(np.percentile(terminal_equities, 75)),
            "p90": float(np.percentile(terminal_equities, 90)),
            "p95": float(np.percentile(terminal_equities, 95)),
            "p99": float(np.percentile(terminal_equities, 99)),
            "mean": float(np.mean(terminal_equities)),
            "std": float(np.std(terminal_equities)),
        }

        log_with_data(logger, "info", "Noise simulation complete", {
            "num_simulations": num_simulations,
            "noise_pct": noise_pct,
            "probability_of_profit": round(probability_of_profit, 4),
            "probability_of_ruin": round(probability_of_ruin, 4),
            "expected_max_drawdown": round(expected_max_drawdown, 4),
        })

        return {
            "simulations": simulations,
            "median_equity": pd.Series(p50, index=trade_index, name="p50"),
            "p5_equity": pd.Series(p5, index=trade_index, name="p5"),
            "p25_equity": pd.Series(p25, index=trade_index, name="p25"),
            "p75_equity": pd.Series(p75, index=trade_index, name="p75"),
            "p95_equity": pd.Series(p95, index=trade_index, name="p95"),
            "probability_of_profit": probability_of_profit,
            "probability_of_ruin": probability_of_ruin,
            "expected_max_drawdown": expected_max_drawdown,
            "confidence_intervals": confidence_intervals,
            "noise_pct": noise_pct,
        }

    # ------------------------------------------------------------------
    # Fan chart data
    # ------------------------------------------------------------------

    @staticmethod
    def plot_fan_chart(simulations: np.ndarray) -> dict[str, np.ndarray]:
        """Extract percentile bands for a fan chart visualisation.

        Args:
            simulations: Array of shape ``(num_sims, num_steps)`` as
                produced by :meth:`run_simulation`.

        Returns:
            Dictionary with keys:
                - ``x`` (np.ndarray): Step indices.
                - ``p5`` (np.ndarray): 5th percentile at each step.
                - ``p25`` (np.ndarray): 25th percentile.
                - ``p50`` (np.ndarray): 50th percentile (median).
                - ``p75`` (np.ndarray): 75th percentile.
                - ``p95`` (np.ndarray): 95th percentile.
        """
        if simulations.size == 0:
            empty = np.array([], dtype=np.float64)
            return {
                "x": empty,
                "p5": empty,
                "p25": empty,
                "p50": empty,
                "p75": empty,
                "p95": empty,
            }

        num_steps = simulations.shape[1]
        x = np.arange(num_steps)

        return {
            "x": x,
            "p5": np.percentile(simulations, 5, axis=0),
            "p25": np.percentile(simulations, 25, axis=0),
            "p50": np.percentile(simulations, 50, axis=0),
            "p75": np.percentile(simulations, 75, axis=0),
            "p95": np.percentile(simulations, 95, axis=0),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_max_drawdowns(simulations: np.ndarray) -> np.ndarray:
        """Compute per-path maximum drawdown from a simulation matrix.

        Uses vectorized numpy operations across all paths simultaneously.

        Args:
            simulations: Array of shape ``(num_sims, num_steps)``.

        Returns:
            1-D array of length ``num_sims`` containing the max drawdown
            for each path as a negative decimal fraction.
        """
        # Running maximum along each path.
        running_max = np.maximum.accumulate(simulations, axis=1)

        # Drawdown at every point.
        drawdowns = np.where(
            running_max > 0,
            (simulations - running_max) / running_max,
            0.0,
        )

        # Per-path max drawdown (most negative value).
        return np.min(drawdowns, axis=1)

    @staticmethod
    def _empty_simulation_result(
        initial_equity: float,
    ) -> dict[str, Any]:
        """Return an empty simulation result structure.

        Args:
            initial_equity: Starting equity value.

        Returns:
            Zeroed-out simulation result dict.
        """
        empty_series = pd.Series([initial_equity], name="equity")
        return {
            "simulations": np.array([[initial_equity]]),
            "median_equity": empty_series,
            "p5_equity": empty_series,
            "p25_equity": empty_series,
            "p75_equity": empty_series,
            "p95_equity": empty_series,
            "probability_of_profit": 0.0,
            "probability_of_ruin": 0.0,
            "expected_max_drawdown": 0.0,
            "confidence_intervals": {
                "p1": initial_equity,
                "p5": initial_equity,
                "p10": initial_equity,
                "p25": initial_equity,
                "p50": initial_equity,
                "p75": initial_equity,
                "p90": initial_equity,
                "p95": initial_equity,
                "p99": initial_equity,
                "mean": initial_equity,
                "std": 0.0,
            },
        }
