"""Portfolio optimization for the APEX Crypto Trading System.

Implements Simons' portfolio construction approach: mean-variance optimization
with Ledoit-Wolf covariance shrinkage.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from apex_crypto.core.logging import get_logger, log_with_data

logger = get_logger("signals.portfolio_optimizer")


class MeanVarianceOptimizer:
    """Mean-variance portfolio optimizer with Ledoit-Wolf shrinkage.

    Solves: maximize  w' * mu - (lambda/2) * w' * Sigma * w
    subject to:
        sum(|w|) <= max_gross_exposure
        -max_position <= w_i <= max_position
        sum(w) = net_exposure_target
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        self.risk_aversion: float = cfg.get("risk_aversion", 2.0)
        self.max_gross_exposure: float = cfg.get("max_gross_exposure", 1.0)
        self.max_position: float = cfg.get("max_position", 0.05)
        self.net_exposure_target: float = cfg.get("net_exposure_target", 0.5)
        self.rebalance_threshold: float = cfg.get("rebalance_threshold", 0.005)
        self.cov_lookback: int = cfg.get("cov_lookback", 60)
        self.min_history: int = cfg.get("min_history", 30)

        self._current_weights: dict[str, float] = {}
        self._last_optimization: float = 0.0

        log_with_data(logger, "info", "MeanVarianceOptimizer initialized", {
            "risk_aversion": self.risk_aversion,
            "max_gross_exposure": self.max_gross_exposure,
            "max_position": self.max_position,
            "net_exposure_target": self.net_exposure_target,
        })

    def estimate_covariance(self, returns_matrix: pd.DataFrame) -> np.ndarray:
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf().fit(returns_matrix.values)
            return lw.covariance_
        except ImportError:
            logger.warning("sklearn not available — falling back to sample covariance")
            return returns_matrix.cov().values

    @staticmethod
    def signals_to_expected_returns(signal_scores: dict[str, float], scale: float = 0.01) -> dict[str, float]:
        return {symbol: (score / 100.0) * scale for symbol, score in signal_scores.items()}

    def optimize(self, signal_scores: dict[str, float], returns_data: dict[str, pd.Series]) -> dict[str, float]:
        symbols = sorted(signal_scores.keys())
        n = len(symbols)

        if n == 0:
            return {}

        if n == 1:
            symbol = symbols[0]
            score = signal_scores[symbol]
            weight = np.clip(score / 100.0 * self.max_position, -self.max_position, self.max_position)
            return {symbol: float(weight)}

        returns_df = pd.DataFrame({s: returns_data[s] for s in symbols if s in returns_data})
        returns_df = returns_df.dropna()

        if len(returns_df) < self.min_history:
            return self._equal_weight_fallback(signal_scores, symbols)

        cov_matrix = self.estimate_covariance(returns_df[symbols])

        mu_dict = self.signals_to_expected_returns(signal_scores)
        mu = np.array([mu_dict.get(s, 0.0) for s in symbols])

        try:
            weights = self._solve_optimization(mu, cov_matrix, n)
        except Exception as exc:
            logger.warning("Optimization failed: %s — using fallback", exc)
            return self._equal_weight_fallback(signal_scores, symbols)

        result = {symbols[i]: float(weights[i]) for i in range(n)}

        log_with_data(logger, "info", "Portfolio optimized", {
            "num_assets": n,
            "gross_exposure": round(sum(abs(w) for w in weights), 4),
            "net_exposure": round(sum(weights), 4),
            "top_long": max(result.items(), key=lambda x: x[1]) if result else None,
            "top_short": min(result.items(), key=lambda x: x[1]) if result else None,
        })

        self._current_weights = result
        return result

    def _solve_optimization(self, mu: np.ndarray, cov: np.ndarray, n: int) -> np.ndarray:
        from scipy.optimize import minimize

        lam = self.risk_aversion

        def objective(w):
            portfolio_return = np.dot(w, mu)
            portfolio_risk = np.dot(w, np.dot(cov, w))
            return -(portfolio_return - (lam / 2) * portfolio_risk)

        def objective_jac(w):
            return -(mu - lam * np.dot(cov, w))

        constraints = [
            {"type": "ineq", "fun": lambda w: self.max_gross_exposure - np.sum(np.abs(w))},
            {"type": "eq", "fun": lambda w: np.sum(w) - self.net_exposure_target},
        ]

        bounds = [(-self.max_position, self.max_position)] * n

        w0 = np.sign(mu) * (self.max_position / 2)
        w0_adj = w0 + (self.net_exposure_target - np.sum(w0)) / n
        w0 = np.clip(w0_adj, -self.max_position, self.max_position)

        result = minimize(objective, w0, jac=objective_jac, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 1000, "ftol": 1e-12})

        if not result.success:
            logger.warning("Optimization did not converge: %s", result.message)

        return result.x

    def _equal_weight_fallback(self, signal_scores, symbols):
        n = len(symbols)
        if n == 0:
            return {}
        weight_per_asset = min(self.max_position, self.max_gross_exposure / n)
        result = {}
        for s in symbols:
            direction = np.sign(signal_scores.get(s, 0))
            result[s] = float(direction * weight_per_asset)
        log_with_data(logger, "info", "Using equal-weight fallback", {"num_assets": n, "weight_per_asset": round(weight_per_asset, 4)})
        return result

    def needs_rebalance(self, current_weights, target_weights):
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())
        for symbol in all_symbols:
            current = current_weights.get(symbol, 0.0)
            target = target_weights.get(symbol, 0.0)
            if abs(current - target) > self.rebalance_threshold:
                return True
        return False

    def compute_portfolio_risk(self, weights, returns_data):
        symbols = sorted(weights.keys())
        if not symbols:
            return {"portfolio_vol": 0.0, "sharpe_estimate": 0.0}
        returns_df = pd.DataFrame({s: returns_data[s] for s in symbols if s in returns_data})
        returns_df = returns_df.dropna()
        if returns_df.empty:
            return {"portfolio_vol": 0.0, "sharpe_estimate": 0.0}
        w = np.array([weights.get(s, 0.0) for s in returns_df.columns])
        cov = self.estimate_covariance(returns_df)
        portfolio_var = float(np.dot(w, np.dot(cov, w)))
        portfolio_vol = float(np.sqrt(portfolio_var) * np.sqrt(252))
        mean_returns = returns_df.mean().values
        portfolio_return = float(np.dot(w, mean_returns) * 252)
        sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0.0
        return {
            "portfolio_vol": round(portfolio_vol, 6),
            "portfolio_return_annual": round(portfolio_return, 6),
            "sharpe_estimate": round(sharpe, 4),
            "gross_exposure": round(sum(abs(v) for v in weights.values()), 4),
            "net_exposure": round(sum(weights.values()), 4),
        }
