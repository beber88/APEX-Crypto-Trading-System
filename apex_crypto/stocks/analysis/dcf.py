"""Morgan Stanley-style DCF Valuation Model.

Builds a Discounted Cash Flow model to estimate intrinsic value:
- 5-year revenue forecast with growth assumptions
- Operating margin estimation based on historical trends
- Free Cash Flow (FCF) calculation year by year
- WACC (Weighted Average Cost of Capital) estimation
- Terminal value via Exit Multiple and Perpetuity Growth methods
- Sensitivity table showing fair value at different discount rates
- Comparison of DCF value vs current market price
- Key assumptions that could "break" the model

Inspired by Morgan Stanley's investment banking DCF methodology.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from apex_crypto.core.logging import get_logger, log_with_data

logger = get_logger("stocks.dcf")

# Default assumptions
DEFAULT_RISK_FREE_RATE = 0.04      # 4% (US 10Y Treasury)
DEFAULT_EQUITY_RISK_PREMIUM = 0.055 # 5.5%
DEFAULT_TERMINAL_GROWTH = 0.025     # 2.5% perpetuity growth
DEFAULT_EXIT_MULTIPLE = 12.0        # EV/EBITDA exit multiple
DEFAULT_TAX_RATE = 0.21             # US corporate tax rate


class DCFValuation:
    """Discounted Cash Flow valuation engine.

    Args:
        config: Optional configuration overrides.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._risk_free_rate = cfg.get("risk_free_rate", DEFAULT_RISK_FREE_RATE)
        self._equity_risk_premium = cfg.get("equity_risk_premium", DEFAULT_EQUITY_RISK_PREMIUM)
        self._terminal_growth = cfg.get("terminal_growth", DEFAULT_TERMINAL_GROWTH)
        self._exit_multiple = cfg.get("exit_multiple", DEFAULT_EXIT_MULTIPLE)
        self._forecast_years = cfg.get("forecast_years", 5)

    def valuate(self, fundamentals: dict[str, Any]) -> dict[str, Any]:
        """Run full DCF valuation on a stock.

        Args:
            fundamentals: Dict from StockBroker.fetch_fundamentals().

        Returns:
            Comprehensive DCF valuation report.
        """
        symbol = fundamentals.get("symbol", "?")
        current_price = fundamentals.get("current_price", 0)

        # Step 1: Extract historical financials
        historical = self._extract_historical(fundamentals)

        # Step 2: Project revenue
        revenue_forecast = self._forecast_revenue(historical)

        # Step 3: Project operating margins and FCF
        fcf_forecast = self._forecast_fcf(historical, revenue_forecast)

        # Step 4: Calculate WACC
        wacc = self._calculate_wacc(fundamentals)

        # Step 5: Calculate terminal value (both methods)
        terminal_exit = self._terminal_value_exit_multiple(fcf_forecast)
        terminal_perp = self._terminal_value_perpetuity(fcf_forecast, wacc)

        # Step 6: Discount cash flows
        dcf_exit = self._discount_cash_flows(fcf_forecast, terminal_exit, wacc)
        dcf_perp = self._discount_cash_flows(fcf_forecast, terminal_perp, wacc)

        # Step 7: Per-share value
        shares = fundamentals.get("market_cap", 0) / current_price if current_price > 0 else 1
        fair_value_exit = dcf_exit / shares if shares > 0 else 0
        fair_value_perp = dcf_perp / shares if shares > 0 else 0
        fair_value_avg = (fair_value_exit + fair_value_perp) / 2

        # Step 8: Sensitivity analysis
        sensitivity = self._sensitivity_table(fcf_forecast, terminal_perp, shares)

        # Step 9: Valuation verdict
        if current_price > 0:
            upside_pct = (fair_value_avg - current_price) / current_price * 100
        else:
            upside_pct = 0

        if upside_pct > 20:
            verdict = "UNDERVALUED"
        elif upside_pct > -10:
            verdict = "FAIR VALUE"
        else:
            verdict = "OVERVALUED"

        # Step 10: Key assumptions that could break the model
        risk_factors = self._identify_risk_factors(fundamentals, historical)

        result = {
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "fair_value_exit_multiple": round(fair_value_exit, 2),
            "fair_value_perpetuity": round(fair_value_perp, 2),
            "fair_value_average": round(fair_value_avg, 2),
            "upside_pct": round(upside_pct, 1),
            "verdict": verdict,
            "wacc": round(wacc * 100, 2),
            "revenue_forecast": revenue_forecast,
            "fcf_forecast": fcf_forecast,
            "terminal_value_exit": round(terminal_exit / 1e9, 2),  # in billions
            "terminal_value_perpetuity": round(terminal_perp / 1e9, 2),
            "sensitivity_table": sensitivity,
            "assumptions": {
                "risk_free_rate": self._risk_free_rate,
                "equity_risk_premium": self._equity_risk_premium,
                "terminal_growth": self._terminal_growth,
                "exit_multiple": self._exit_multiple,
                "tax_rate": DEFAULT_TAX_RATE,
                "forecast_years": self._forecast_years,
            },
            "risk_factors": risk_factors,
        }

        log_with_data(logger, "info", "DCF valuation complete", {
            "symbol": symbol,
            "fair_value": result["fair_value_average"],
            "current_price": current_price,
            "upside_pct": result["upside_pct"],
            "verdict": verdict,
        })

        return result

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _extract_historical(self, f: dict) -> dict[str, Any]:
        """Extract historical financial data."""
        income = f.get("income_statement", {})
        cashflow = f.get("cash_flow", {})

        revenues = income.get("total_revenue", [])
        net_incomes = income.get("net_income", [])
        ebitda_list = income.get("ebitda", [])
        op_cfs = cashflow.get("operating_cf", [])
        free_cfs = cashflow.get("free_cf", [])
        capex_list = cashflow.get("capex", [])

        # Calculate growth rates
        rev_growth_rates = []
        for i in range(len(revenues) - 1):
            if revenues[i + 1] and revenues[i + 1] > 0:
                growth = (revenues[i] - revenues[i + 1]) / revenues[i + 1]
                rev_growth_rates.append(growth)

        # Calculate margins
        op_margins = []
        for i, rev in enumerate(revenues):
            if rev and rev > 0 and i < len(ebitda_list):
                op_margins.append(ebitda_list[i] / rev)

        return {
            "revenues": revenues,
            "net_incomes": net_incomes,
            "ebitda": ebitda_list,
            "operating_cf": op_cfs,
            "free_cf": free_cfs,
            "capex": capex_list,
            "rev_growth_rates": rev_growth_rates,
            "operating_margins": op_margins,
            "latest_revenue": revenues[0] if revenues else 0,
            "latest_fcf": free_cfs[0] if free_cfs else 0,
            "latest_ebitda": ebitda_list[0] if ebitda_list else 0,
        }

    def _forecast_revenue(self, historical: dict) -> list[dict[str, Any]]:
        """Project revenue for next 5 years."""
        latest_rev = historical.get("latest_revenue", 0)
        growth_rates = historical.get("rev_growth_rates", [])

        if not latest_rev or latest_rev <= 0:
            return [{"year": i + 1, "revenue": 0, "growth_rate": 0} for i in range(self._forecast_years)]

        # Estimate future growth — declining towards terminal
        if growth_rates:
            avg_growth = np.mean(growth_rates)
            recent_growth = growth_rates[0] if growth_rates else avg_growth
        else:
            avg_growth = 0.05
            recent_growth = 0.05

        # Growth decays towards terminal rate over forecast period
        forecast = []
        current_rev = latest_rev
        for year in range(1, self._forecast_years + 1):
            # Linear interpolation from recent growth to terminal growth
            weight = year / self._forecast_years
            growth = recent_growth * (1 - weight) + self._terminal_growth * weight
            growth = max(growth, self._terminal_growth)  # floor at terminal
            current_rev *= (1 + growth)
            forecast.append({
                "year": year,
                "revenue": round(current_rev),
                "growth_rate": round(growth, 4),
            })

        return forecast

    def _forecast_fcf(
        self, historical: dict, revenue_forecast: list[dict]
    ) -> list[dict[str, Any]]:
        """Project Free Cash Flow based on revenue and margins."""
        margins = historical.get("operating_margins", [])
        avg_margin = np.mean(margins) if margins else 0.15

        # Capex as % of revenue
        capex_list = historical.get("capex", [])
        revenues = historical.get("revenues", [])
        capex_pcts = []
        for i in range(min(len(capex_list), len(revenues))):
            if revenues[i] and revenues[i] > 0:
                capex_pcts.append(abs(capex_list[i]) / revenues[i])

        avg_capex_pct = np.mean(capex_pcts) if capex_pcts else 0.05

        forecast = []
        for entry in revenue_forecast:
            rev = entry["revenue"]
            ebitda = rev * avg_margin
            tax = ebitda * DEFAULT_TAX_RATE
            capex = rev * avg_capex_pct
            fcf = ebitda - tax - capex

            forecast.append({
                "year": entry["year"],
                "revenue": round(rev),
                "ebitda": round(ebitda),
                "fcf": round(fcf),
                "margin": round(avg_margin, 4),
            })

        return forecast

    def _calculate_wacc(self, f: dict) -> float:
        """Estimate WACC from beta and capital structure."""
        beta = f.get("beta", 1.0) or 1.0
        beta = max(0.5, min(beta, 2.5))  # clamp

        # Cost of equity (CAPM)
        cost_of_equity = self._risk_free_rate + beta * self._equity_risk_premium

        # Cost of debt (approximate)
        dte = f.get("debt_to_equity", 0) or 0
        if dte > 0:
            cost_of_debt = self._risk_free_rate + 0.02  # risk-free + 2% spread
            debt_weight = dte / (100 + dte)
            equity_weight = 1 - debt_weight
            wacc = (equity_weight * cost_of_equity
                    + debt_weight * cost_of_debt * (1 - DEFAULT_TAX_RATE))
        else:
            wacc = cost_of_equity

        return max(0.05, min(wacc, 0.20))  # floor/cap

    def _terminal_value_exit_multiple(self, fcf_forecast: list[dict]) -> float:
        """Terminal value using exit multiple method."""
        if not fcf_forecast:
            return 0
        last_ebitda = fcf_forecast[-1].get("ebitda", 0)
        return last_ebitda * self._exit_multiple

    def _terminal_value_perpetuity(
        self, fcf_forecast: list[dict], wacc: float
    ) -> float:
        """Terminal value using perpetuity growth method."""
        if not fcf_forecast:
            return 0
        last_fcf = fcf_forecast[-1].get("fcf", 0)
        denominator = wacc - self._terminal_growth
        if denominator <= 0:
            return last_fcf * 20  # fallback
        return last_fcf * (1 + self._terminal_growth) / denominator

    def _discount_cash_flows(
        self, fcf_forecast: list[dict], terminal_value: float, wacc: float
    ) -> float:
        """Discount all future cash flows to present value."""
        pv_total = 0.0

        for entry in fcf_forecast:
            year = entry["year"]
            fcf = entry["fcf"]
            pv = fcf / (1 + wacc) ** year
            pv_total += pv

        # Discount terminal value
        n = len(fcf_forecast)
        pv_terminal = terminal_value / (1 + wacc) ** n
        pv_total += pv_terminal

        return pv_total

    def _sensitivity_table(
        self, fcf_forecast: list[dict], base_terminal: float, shares: float
    ) -> list[dict[str, Any]]:
        """Generate sensitivity table with different WACC/growth combos."""
        wacc_range = [0.07, 0.08, 0.09, 0.10, 0.11, 0.12]
        growth_range = [0.015, 0.020, 0.025, 0.030, 0.035]

        table = []
        for wacc in wacc_range:
            row = {"wacc_pct": round(wacc * 100, 1)}
            for growth in growth_range:
                if not fcf_forecast:
                    fair_value = 0
                else:
                    last_fcf = fcf_forecast[-1].get("fcf", 0)
                    denom = wacc - growth
                    if denom <= 0:
                        tv = last_fcf * 20
                    else:
                        tv = last_fcf * (1 + growth) / denom
                    dcf = self._discount_cash_flows(fcf_forecast, tv, wacc)
                    fair_value = dcf / shares if shares > 0 else 0

                row[f"growth_{growth:.1%}"] = round(fair_value, 2)
            table.append(row)

        return table

    def _identify_risk_factors(
        self, f: dict, historical: dict
    ) -> list[str]:
        """Identify key assumptions that could break the DCF model."""
        risks = []

        growth_rates = historical.get("rev_growth_rates", [])
        if growth_rates and max(growth_rates) - min(growth_rates) > 0.30:
            risks.append("Highly volatile revenue growth — forecast may be unreliable")

        dte = f.get("debt_to_equity", 0) or 0
        if dte > 150:
            risks.append(f"High debt-to-equity ({dte:.0f}) — interest costs could erode FCF")

        margins = historical.get("operating_margins", [])
        if margins and any(m < 0 for m in margins):
            risks.append("Company has had negative operating margins — profitability risk")

        beta = f.get("beta", 1.0) or 1.0
        if beta > 1.5:
            risks.append(f"High beta ({beta:.2f}) — WACC estimate very sensitive to market risk premium")

        if f.get("sector") in ("Technology", "Communication Services"):
            risks.append("Tech sector — disruption risk could invalidate long-term projections")

        if not risks:
            risks.append("Model relies on historical margins continuing — margin compression is main risk")

        return risks
