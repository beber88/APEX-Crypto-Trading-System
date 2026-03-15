"""Bridgewater-style Risk Assessment + BlackRock Portfolio Builder.

Comprehensive portfolio risk analysis:
- Correlation analysis between holdings
- Sector concentration risk
- Geographic/currency exposure
- Interest rate sensitivity (via beta)
- Drawdown stress test (max expected drawdown)
- Risk rating per holding
- Individual stock risk and position sizing recommendations
- Tail risk scenarios with probability estimates
- Hedging strategies for top 3 risks
- Rebalancing suggestions with trigger rules

BlackRock Portfolio Builder features:
- Optimal asset allocation (stocks, bonds, alternatives)
- Core vs satellite position marking
- Expected annual return based on historical data
- DCA plan recommendations
- Benchmark tracking (S&P 500)

Inspired by Bridgewater Associates and BlackRock institutional methodologies.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from apex_crypto.core.logging import get_logger, log_with_data

logger = get_logger("stocks.risk_analyzer")


class PortfolioRiskAnalyzer:
    """Analyzes portfolio risk using institutional-grade methodology.

    Args:
        config: Optional configuration dict.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._max_sector_concentration: float = cfg.get("max_sector_pct", 0.30)
        self._max_single_position: float = cfg.get("max_single_pct", 0.10)
        self._rebalance_threshold: float = cfg.get("rebalance_threshold_pct", 0.05)
        self._var_confidence: float = cfg.get("var_confidence", 0.95)
        self._stress_scenarios = cfg.get("stress_scenarios", [
            {"name": "Market Crash (-20%)", "market_drop": -0.20},
            {"name": "Rate Shock (+2%)", "rate_change": 0.02},
            {"name": "Sector Rotation", "rotation_pct": -0.15},
        ])

    def analyze_portfolio(
        self,
        holdings: list[dict[str, Any]],
        fundamentals_map: dict[str, dict],
        ohlcv_map: dict[str, pd.DataFrame],
    ) -> dict[str, Any]:
        """Run full portfolio risk assessment.

        Args:
            holdings: List of dicts with {symbol, shares, avg_price, current_price, weight_pct}.
            fundamentals_map: Dict mapping symbol -> fundamentals.
            ohlcv_map: Dict mapping symbol -> daily OHLCV DataFrame.

        Returns:
            Comprehensive risk assessment report.
        """
        if not holdings:
            return {"error": "No holdings to analyze"}

        # 1. Correlation matrix
        correlation = self._build_correlation_matrix(holdings, ohlcv_map)

        # 2. Sector concentration
        sector_analysis = self._analyze_sectors(holdings, fundamentals_map)

        # 3. Individual stock risk ratings
        stock_risks = self._rate_individual_risks(holdings, fundamentals_map)

        # 4. Portfolio-level metrics
        portfolio_metrics = self._calculate_portfolio_metrics(
            holdings, ohlcv_map, fundamentals_map
        )

        # 5. Drawdown stress test
        stress_test = self._run_stress_test(
            holdings, fundamentals_map, portfolio_metrics
        )

        # 6. Tail risk analysis
        tail_risk = self._analyze_tail_risk(holdings, ohlcv_map)

        # 7. Top 3 risks and hedging suggestions
        top_risks = self._identify_top_risks(
            sector_analysis, correlation, portfolio_metrics, stock_risks
        )

        # 8. Rebalancing suggestions
        rebalance = self._generate_rebalance_suggestions(
            holdings, sector_analysis, stock_risks
        )

        # 9. Position sizing recommendations
        sizing = self._recommend_position_sizes(holdings, stock_risks, portfolio_metrics)

        # Overall portfolio risk score (1-10)
        overall_risk = self._calculate_overall_risk_score(
            portfolio_metrics, sector_analysis, correlation
        )

        result = {
            "overall_risk_score": overall_risk,
            "portfolio_metrics": portfolio_metrics,
            "correlation_matrix": correlation,
            "sector_analysis": sector_analysis,
            "individual_risks": stock_risks,
            "stress_test": stress_test,
            "tail_risk": tail_risk,
            "top_risks_and_hedges": top_risks,
            "rebalancing": rebalance,
            "position_sizing": sizing,
        }

        log_with_data(logger, "info", "Portfolio risk analysis complete", {
            "num_holdings": len(holdings),
            "overall_risk": overall_risk,
            "portfolio_beta": portfolio_metrics.get("portfolio_beta", 1.0),
        })

        return result

    # ------------------------------------------------------------------
    # Correlation analysis
    # ------------------------------------------------------------------

    def _build_correlation_matrix(
        self, holdings: list[dict], ohlcv_map: dict[str, pd.DataFrame]
    ) -> dict[str, Any]:
        """Build correlation matrix between holdings."""
        symbols = [h["symbol"] for h in holdings]
        returns_data = {}

        for sym in symbols:
            df = ohlcv_map.get(sym)
            if df is not None and len(df) > 20:
                returns_data[sym] = df["close"].pct_change().dropna()

        if len(returns_data) < 2:
            return {"available": False}

        # Align dates
        returns_df = pd.DataFrame(returns_data).dropna()
        if returns_df.empty or len(returns_df) < 20:
            return {"available": False}

        corr_matrix = returns_df.corr()

        # Find highly correlated pairs
        high_corr_pairs = []
        for i, sym1 in enumerate(corr_matrix.columns):
            for j, sym2 in enumerate(corr_matrix.columns):
                if i < j:
                    corr_val = float(corr_matrix.loc[sym1, sym2])
                    if abs(corr_val) > 0.70:
                        high_corr_pairs.append({
                            "pair": f"{sym1}/{sym2}",
                            "correlation": round(corr_val, 3),
                            "risk": "high" if corr_val > 0.85 else "moderate",
                        })

        avg_correlation = float(corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)].mean())

        return {
            "available": True,
            "matrix": {
                sym: {s2: round(float(corr_matrix.loc[sym, s2]), 3) for s2 in corr_matrix.columns}
                for sym in corr_matrix.index
            },
            "high_corr_pairs": high_corr_pairs,
            "avg_correlation": round(avg_correlation, 3),
            "diversification_score": round(max(0, 1 - avg_correlation) * 100, 1),
        }

    # ------------------------------------------------------------------
    # Sector analysis
    # ------------------------------------------------------------------

    def _analyze_sectors(
        self, holdings: list[dict], fundamentals_map: dict[str, dict]
    ) -> dict[str, Any]:
        """Analyze sector concentration risk."""
        sector_weights: dict[str, float] = {}
        total_value = sum(h.get("weight_pct", 0) for h in holdings) or 1.0

        for h in holdings:
            sym = h["symbol"]
            f = fundamentals_map.get(sym, {})
            sector = f.get("sector", "Unknown")
            weight = h.get("weight_pct", 0) / total_value
            sector_weights[sector] = sector_weights.get(sector, 0) + weight

        # Identify over-concentrated sectors
        concentrated = []
        for sector, weight in sector_weights.items():
            if weight > self._max_sector_concentration:
                concentrated.append({
                    "sector": sector,
                    "weight_pct": round(weight * 100, 1),
                    "limit_pct": round(self._max_sector_concentration * 100, 1),
                    "excess_pct": round((weight - self._max_sector_concentration) * 100, 1),
                })

        return {
            "sector_weights": {k: round(v * 100, 1) for k, v in sorted(sector_weights.items(), key=lambda x: -x[1])},
            "num_sectors": len(sector_weights),
            "concentrated_sectors": concentrated,
            "is_well_diversified": len(concentrated) == 0 and len(sector_weights) >= 3,
        }

    # ------------------------------------------------------------------
    # Individual risk ratings
    # ------------------------------------------------------------------

    def _rate_individual_risks(
        self, holdings: list[dict], fundamentals_map: dict[str, dict]
    ) -> list[dict[str, Any]]:
        """Rate risk for each individual holding."""
        risks = []

        for h in holdings:
            sym = h["symbol"]
            f = fundamentals_map.get(sym, {})

            risk_score = 5  # neutral

            beta = f.get("beta", 1.0) or 1.0
            if beta > 1.5:
                risk_score += 2
            elif beta > 1.2:
                risk_score += 1
            elif beta < 0.7:
                risk_score -= 1

            dte = f.get("debt_to_equity", 0) or 0
            if dte > 200:
                risk_score += 1
            elif dte < 50:
                risk_score -= 1

            weight = h.get("weight_pct", 0)
            if weight > self._max_single_position * 100:
                risk_score += 1

            risk_score = max(1, min(10, risk_score))

            risks.append({
                "symbol": sym,
                "risk_score": risk_score,
                "beta": round(beta, 2),
                "weight_pct": round(weight, 1),
                "sector": f.get("sector", "Unknown"),
                "flags": self._get_risk_flags(f, weight),
            })

        return sorted(risks, key=lambda x: -x["risk_score"])

    @staticmethod
    def _get_risk_flags(f: dict, weight: float) -> list[str]:
        flags = []
        if (f.get("beta", 1) or 1) > 1.5:
            flags.append("high_beta")
        if (f.get("debt_to_equity", 0) or 0) > 200:
            flags.append("high_leverage")
        if weight > 10:
            flags.append("over_concentrated")
        if (f.get("pe_ratio", 0) or 0) > 40:
            flags.append("expensive_valuation")
        return flags

    # ------------------------------------------------------------------
    # Portfolio metrics
    # ------------------------------------------------------------------

    def _calculate_portfolio_metrics(
        self, holdings: list[dict], ohlcv_map: dict[str, pd.DataFrame],
        fundamentals_map: dict[str, dict]
    ) -> dict[str, Any]:
        """Calculate portfolio-level risk metrics."""
        total_value = sum(h.get("weight_pct", 0) for h in holdings) or 1.0

        # Weighted beta
        weighted_beta = 0
        for h in holdings:
            f = fundamentals_map.get(h["symbol"], {})
            beta = f.get("beta", 1.0) or 1.0
            weight = h.get("weight_pct", 0) / total_value
            weighted_beta += beta * weight

        # Portfolio returns for VaR calculation
        portfolio_returns = self._calculate_portfolio_returns(holdings, ohlcv_map, total_value)

        var_95 = 0.0
        max_drawdown = 0.0
        sharpe = 0.0
        annual_return = 0.0

        if portfolio_returns is not None and len(portfolio_returns) > 20:
            var_95 = float(np.percentile(portfolio_returns, 5))
            annual_return = float(portfolio_returns.mean() * 252)
            annual_vol = float(portfolio_returns.std() * np.sqrt(252))
            sharpe = annual_return / annual_vol if annual_vol > 0 else 0

            # Max drawdown
            cumulative = (1 + portfolio_returns).cumprod()
            peak = cumulative.cummax()
            drawdown = (cumulative - peak) / peak
            max_drawdown = float(drawdown.min())

        return {
            "portfolio_beta": round(weighted_beta, 2),
            "var_95_daily_pct": round(var_95 * 100, 2),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "sharpe_ratio": round(sharpe, 2),
            "annual_return_pct": round(annual_return * 100, 2),
            "num_holdings": len(holdings),
        }

    def _calculate_portfolio_returns(
        self, holdings: list[dict], ohlcv_map: dict[str, pd.DataFrame],
        total_value: float
    ) -> Optional[pd.Series]:
        """Calculate weighted portfolio daily returns."""
        returns_data = {}
        weights = {}

        for h in holdings:
            sym = h["symbol"]
            df = ohlcv_map.get(sym)
            if df is not None and len(df) > 20:
                returns_data[sym] = df["close"].pct_change().dropna()
                weights[sym] = h.get("weight_pct", 0) / total_value

        if not returns_data:
            return None

        returns_df = pd.DataFrame(returns_data).dropna()
        if returns_df.empty:
            return None

        portfolio_returns = sum(
            returns_df[sym] * weights.get(sym, 0) for sym in returns_df.columns
        )
        return portfolio_returns

    # ------------------------------------------------------------------
    # Stress test
    # ------------------------------------------------------------------

    def _run_stress_test(
        self, holdings: list[dict], fundamentals_map: dict[str, dict],
        metrics: dict
    ) -> list[dict[str, Any]]:
        """Run stress test scenarios."""
        results = []
        portfolio_beta = metrics.get("portfolio_beta", 1.0)

        for scenario in self._stress_scenarios:
            name = scenario["name"]
            market_drop = scenario.get("market_drop", 0)
            rate_change = scenario.get("rate_change", 0)
            rotation = scenario.get("rotation_pct", 0)

            # Estimate portfolio impact
            if market_drop:
                impact_pct = market_drop * portfolio_beta * 100
            elif rate_change:
                # Higher beta = more rate sensitive
                impact_pct = -rate_change * portfolio_beta * 50
            elif rotation:
                impact_pct = rotation * 100 * 0.5
            else:
                impact_pct = 0

            results.append({
                "scenario": name,
                "estimated_impact_pct": round(impact_pct, 1),
                "severity": "HIGH" if abs(impact_pct) > 15 else "MEDIUM" if abs(impact_pct) > 8 else "LOW",
            })

        return results

    # ------------------------------------------------------------------
    # Tail risk
    # ------------------------------------------------------------------

    def _analyze_tail_risk(
        self, holdings: list[dict], ohlcv_map: dict[str, pd.DataFrame]
    ) -> dict[str, Any]:
        """Analyze tail risk scenarios."""
        worst_days = []

        for h in holdings:
            sym = h["symbol"]
            df = ohlcv_map.get(sym)
            if df is not None and len(df) > 60:
                daily_returns = df["close"].pct_change().dropna()
                worst = float(daily_returns.min())
                percentile_1 = float(np.percentile(daily_returns, 1))
                worst_days.append({
                    "symbol": sym,
                    "worst_daily_pct": round(worst * 100, 2),
                    "1st_percentile_pct": round(percentile_1 * 100, 2),
                })

        return {
            "worst_days_by_stock": sorted(worst_days, key=lambda x: x["worst_daily_pct"]),
            "portfolio_1pct_var": round(np.mean([w["1st_percentile_pct"] for w in worst_days]), 2) if worst_days else 0,
        }

    # ------------------------------------------------------------------
    # Top risks and hedging
    # ------------------------------------------------------------------

    def _identify_top_risks(
        self, sector_analysis: dict, correlation: dict,
        metrics: dict, stock_risks: list[dict]
    ) -> list[dict[str, Any]]:
        """Identify top 3 risks with hedging suggestions."""
        risks = []

        # 1. Sector concentration risk
        concentrated = sector_analysis.get("concentrated_sectors", [])
        if concentrated:
            top_sector = concentrated[0]
            risks.append({
                "risk": f"Sector concentration: {top_sector['sector']} ({top_sector['weight_pct']}%)",
                "severity": "HIGH",
                "hedge": f"Reduce {top_sector['sector']} exposure by {top_sector['excess_pct']}% or add sector ETF hedge (e.g., inverse sector ETF)",
            })

        # 2. High correlation risk
        if correlation.get("available"):
            if correlation.get("avg_correlation", 0) > 0.60:
                risks.append({
                    "risk": f"High portfolio correlation ({correlation['avg_correlation']:.2f}) — holdings move together",
                    "severity": "HIGH",
                    "hedge": "Add uncorrelated assets: bonds (TLT), gold (GLD), or international stocks (VXUS)",
                })

        # 3. Beta/volatility risk
        beta = metrics.get("portfolio_beta", 1.0)
        if beta > 1.3:
            risks.append({
                "risk": f"High portfolio beta ({beta:.2f}) — amplified market sensitivity",
                "severity": "MEDIUM",
                "hedge": "Add low-beta defensive stocks (utilities, consumer staples) or buy put options on SPY",
            })

        # 4. Individual over-concentration
        for sr in stock_risks[:2]:
            if sr["weight_pct"] > 15:
                risks.append({
                    "risk": f"Over-concentrated in {sr['symbol']} ({sr['weight_pct']:.1f}%)",
                    "severity": "MEDIUM",
                    "hedge": f"Trim {sr['symbol']} position to under 10%, consider covered calls to reduce risk",
                })

        if not risks:
            risks.append({
                "risk": "General market risk",
                "severity": "LOW",
                "hedge": "Portfolio appears well-balanced. Consider protective puts on SPY for tail risk",
            })

        return risks[:3]

    # ------------------------------------------------------------------
    # Rebalancing
    # ------------------------------------------------------------------

    def _generate_rebalance_suggestions(
        self, holdings: list[dict], sector_analysis: dict,
        stock_risks: list[dict]
    ) -> dict[str, Any]:
        """Generate rebalancing recommendations."""
        suggestions = []

        # Check for position limit violations
        for h in holdings:
            weight = h.get("weight_pct", 0)
            if weight > self._max_single_position * 100:
                target = self._max_single_position * 100
                suggestions.append({
                    "symbol": h["symbol"],
                    "action": "TRIM",
                    "current_pct": round(weight, 1),
                    "target_pct": round(target, 1),
                    "reason": "Exceeds maximum single position limit",
                })

        # Sector rebalancing
        for conc in sector_analysis.get("concentrated_sectors", []):
            suggestions.append({
                "action": "REDUCE SECTOR",
                "sector": conc["sector"],
                "current_pct": conc["weight_pct"],
                "target_pct": conc["limit_pct"],
                "reason": "Sector over-concentration",
            })

        trigger_rules = [
            f"Rebalance when any position drifts >{self._rebalance_threshold * 100:.0f}% from target",
            "Review quarterly or after major market events",
            "Rebalance when sector exceeds 30% of portfolio",
        ]

        return {
            "suggestions": suggestions,
            "trigger_rules": trigger_rules,
            "needs_rebalancing": len(suggestions) > 0,
        }

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def _recommend_position_sizes(
        self, holdings: list[dict], stock_risks: list[dict],
        metrics: dict
    ) -> list[dict[str, Any]]:
        """Recommend optimal position sizes based on risk."""
        recommendations = []
        risk_map = {r["symbol"]: r["risk_score"] for r in stock_risks}

        for h in holdings:
            sym = h["symbol"]
            risk = risk_map.get(sym, 5)

            # Inverse risk sizing — lower risk = larger position
            if risk <= 3:
                target = 8.0
            elif risk <= 5:
                target = 5.0
            elif risk <= 7:
                target = 3.0
            else:
                target = 2.0

            current = h.get("weight_pct", 0)
            recommendations.append({
                "symbol": sym,
                "current_pct": round(current, 1),
                "recommended_pct": target,
                "risk_score": risk,
                "action": "INCREASE" if current < target * 0.8 else "DECREASE" if current > target * 1.2 else "HOLD",
            })

        return recommendations

    # ------------------------------------------------------------------
    # Overall risk score
    # ------------------------------------------------------------------

    def _calculate_overall_risk_score(
        self, metrics: dict, sector_analysis: dict, correlation: dict
    ) -> int:
        """Calculate overall portfolio risk score (1-10)."""
        score = 5

        beta = metrics.get("portfolio_beta", 1.0)
        if beta > 1.3:
            score += 2
        elif beta > 1.1:
            score += 1
        elif beta < 0.8:
            score -= 1

        if sector_analysis.get("concentrated_sectors"):
            score += 1

        if not sector_analysis.get("is_well_diversified"):
            score += 1

        if correlation.get("available"):
            avg_corr = correlation.get("avg_correlation", 0)
            if avg_corr > 0.70:
                score += 1
            elif avg_corr < 0.30:
                score -= 1

        dd = metrics.get("max_drawdown_pct", 0)
        if dd < -30:
            score += 1

        return max(1, min(10, score))


class PortfolioBuilder:
    """BlackRock-style portfolio construction.

    Given an investor profile, builds an optimal asset allocation.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._risk_profiles = {
            "conservative": {"stocks": 30, "bonds": 55, "alternatives": 15},
            "moderate": {"stocks": 50, "bonds": 35, "alternatives": 15},
            "balanced": {"stocks": 60, "bonds": 30, "alternatives": 10},
            "growth": {"stocks": 75, "bonds": 15, "alternatives": 10},
            "aggressive": {"stocks": 90, "bonds": 5, "alternatives": 5},
        }

    def build_portfolio(
        self,
        risk_tolerance: str = "balanced",
        investment_amount: float = 100_000,
        monthly_contribution: float = 0,
        time_horizon_years: int = 10,
    ) -> dict[str, Any]:
        """Build an optimal portfolio allocation.

        Args:
            risk_tolerance: One of conservative/moderate/balanced/growth/aggressive.
            investment_amount: Total initial investment.
            monthly_contribution: Monthly DCA amount.
            time_horizon_years: Investment time horizon.

        Returns:
            Portfolio construction plan.
        """
        profile = self._risk_profiles.get(risk_tolerance, self._risk_profiles["balanced"])

        # Asset allocation
        allocation = {
            "stocks": {
                "pct": profile["stocks"],
                "amount": investment_amount * profile["stocks"] / 100,
                "etfs": self._recommend_stock_etfs(risk_tolerance),
            },
            "bonds": {
                "pct": profile["bonds"],
                "amount": investment_amount * profile["bonds"] / 100,
                "etfs": self._recommend_bond_etfs(risk_tolerance),
            },
            "alternatives": {
                "pct": profile["alternatives"],
                "amount": investment_amount * profile["alternatives"] / 100,
                "etfs": self._recommend_alternative_etfs(),
            },
        }

        # Core vs satellite
        core_satellite = {
            "core_pct": 70,
            "satellite_pct": 30,
            "core": "Broad market index funds (SPY, VTI, AGG)",
            "satellite": "Individual stocks, sector ETFs, crypto allocation",
        }

        # Expected returns
        expected = self._estimate_returns(profile, time_horizon_years, investment_amount, monthly_contribution)

        # DCA plan
        dca_plan = None
        if monthly_contribution > 0:
            dca_plan = {
                "monthly_amount": monthly_contribution,
                "allocation": {
                    "stocks": round(monthly_contribution * profile["stocks"] / 100, 2),
                    "bonds": round(monthly_contribution * profile["bonds"] / 100, 2),
                    "alternatives": round(monthly_contribution * profile["alternatives"] / 100, 2),
                },
                "frequency": "Monthly on 1st/15th",
                "rebalance": "Quarterly",
            }

        # Benchmark
        benchmark = "S&P 500 (SPY)" if profile["stocks"] > 50 else "60/40 Portfolio (VBINX)"

        return {
            "risk_tolerance": risk_tolerance,
            "allocation": allocation,
            "core_satellite": core_satellite,
            "expected_returns": expected,
            "dca_plan": dca_plan,
            "benchmark": benchmark,
            "rebalance_schedule": "Quarterly or when drift exceeds 5%",
            "tax_strategy": "Use tax-advantaged accounts for bonds; hold stocks for long-term capital gains",
        }

    @staticmethod
    def _recommend_stock_etfs(risk: str) -> list[dict]:
        etfs = [
            {"ticker": "VTI", "name": "Vanguard Total Stock Market", "type": "core", "pct": 40},
            {"ticker": "VXUS", "name": "Vanguard Int'l Stock", "type": "core", "pct": 20},
        ]
        if risk in ("growth", "aggressive"):
            etfs.append({"ticker": "QQQ", "name": "Nasdaq 100", "type": "satellite", "pct": 25})
            etfs.append({"ticker": "ARKK", "name": "ARK Innovation", "type": "satellite", "pct": 15})
        else:
            etfs.append({"ticker": "VIG", "name": "Dividend Appreciation", "type": "satellite", "pct": 25})
            etfs.append({"ticker": "SCHD", "name": "Schwab Dividend", "type": "satellite", "pct": 15})
        return etfs

    @staticmethod
    def _recommend_bond_etfs(risk: str) -> list[dict]:
        if risk in ("conservative", "moderate"):
            return [
                {"ticker": "AGG", "name": "US Aggregate Bond", "type": "core", "pct": 50},
                {"ticker": "TIP", "name": "TIPS (Inflation)", "type": "core", "pct": 30},
                {"ticker": "BND", "name": "Total Bond Market", "type": "core", "pct": 20},
            ]
        return [
            {"ticker": "AGG", "name": "US Aggregate Bond", "type": "core", "pct": 60},
            {"ticker": "TLT", "name": "20+ Year Treasury", "type": "satellite", "pct": 40},
        ]

    @staticmethod
    def _recommend_alternative_etfs() -> list[dict]:
        return [
            {"ticker": "GLD", "name": "SPDR Gold", "type": "hedge", "pct": 40},
            {"ticker": "VNQ", "name": "Vanguard Real Estate", "type": "satellite", "pct": 30},
            {"ticker": "DBC", "name": "Commodities", "type": "satellite", "pct": 30},
        ]

    @staticmethod
    def _estimate_returns(
        profile: dict, years: int, initial: float, monthly: float
    ) -> dict[str, Any]:
        # Historical average returns by asset class
        stock_return = 0.10  # 10% annual
        bond_return = 0.04
        alt_return = 0.06

        weighted_return = (
            profile["stocks"] / 100 * stock_return
            + profile["bonds"] / 100 * bond_return
            + profile["alternatives"] / 100 * alt_return
        )

        # Compound growth
        future_lump = initial * (1 + weighted_return) ** years

        # DCA future value
        if monthly > 0:
            monthly_rate = weighted_return / 12
            months = years * 12
            future_dca = monthly * ((1 + monthly_rate) ** months - 1) / monthly_rate
        else:
            future_dca = 0

        total_future = future_lump + future_dca
        total_invested = initial + monthly * years * 12

        return {
            "expected_annual_return_pct": round(weighted_return * 100, 1),
            "total_invested": round(total_invested, 2),
            "projected_value": round(total_future, 2),
            "total_gain": round(total_future - total_invested, 2),
            "time_horizon_years": years,
            "worst_year_estimate_pct": round(-weighted_return * 3, 1),  # ~3x annual vol
        }
