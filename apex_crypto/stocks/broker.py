"""Stock broker for the APEX Trading System.

Uses Alpaca Markets API for order execution (commission-free US stocks)
and yfinance for free market data and fundamentals.

Supports both paper and live trading modes via Alpaca's sandbox/live endpoints.

Usage::

    config = {
        "paper_trading": True,
        "paper_initial_balance": 100_000.0,
    }
    broker = StockBroker(config)
    await broker.initialize()
    order = await broker.place_market_order("AAPL", "buy", 10)
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

import pandas as pd

from apex_crypto.core.logging import get_logger, log_with_data

logger = get_logger("stocks.broker")

# Market hours (Eastern Time)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0


class StockBroker:
    """Broker for US stock trading via Alpaca + yfinance data.

    Args:
        config: Configuration dictionary with keys:
            - paper_trading (bool): Use paper trading mode.
            - paper_initial_balance (float): Starting balance for paper mode.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._paper_trading: bool = config.get("paper_trading", True)
        self._alpaca_client = None
        self._alpaca_data_client = None
        self._initialized = False

        # Paper trading state
        self._paper_balance: float = config.get("paper_initial_balance", 100_000.0)
        self._paper_positions: dict[str, dict] = {}
        self._paper_orders: list[dict] = []

        # yfinance data cache
        self._yf_cache: dict[str, dict] = {}
        self._yf_cache_ttl: int = 300  # 5 minutes

        log_with_data(logger, "info", "StockBroker created", {
            "paper_trading": self._paper_trading,
        })

    async def initialize(self) -> None:
        """Initialize Alpaca API connection."""
        api_key = os.environ.get("ALPACA_API_KEY", "")
        secret_key = os.environ.get("ALPACA_SECRET_KEY", "")

        if not api_key or not secret_key:
            logger.warning("Alpaca API keys not set — stock trading will use paper mode only")
            self._paper_trading = True
            self._initialized = True
            return

        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient

            self._alpaca_client = TradingClient(
                api_key, secret_key, paper=self._paper_trading
            )
            self._alpaca_data_client = StockHistoricalDataClient(
                api_key, secret_key
            )
            self._initialized = True

            account = self._alpaca_client.get_account()
            log_with_data(logger, "info", "Alpaca connected", {
                "equity": float(account.equity),
                "buying_power": float(account.buying_power),
                "paper": self._paper_trading,
            })
        except ImportError:
            logger.warning("alpaca-py not installed — using yfinance data + paper trading")
            self._paper_trading = True
            self._initialized = True
        except Exception as exc:
            logger.warning("Alpaca connection failed: %s — falling back to paper", exc)
            self._paper_trading = True
            self._initialized = True

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1d", limit: int = 500
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data for a stock using yfinance.

        Args:
            symbol: Stock ticker (e.g., "AAPL", "MSFT").
            timeframe: Timeframe string ("1d", "1h", "5m", etc.).
            limit: Number of bars to fetch.

        Returns:
            DataFrame with columns [open, high, low, close, volume].
        """
        try:
            import yfinance as yf

            # Map timeframe to yfinance period/interval
            interval_map = {
                "1m": ("5d", "1m"),
                "5m": ("60d", "5m"),
                "15m": ("60d", "15m"),
                "30m": ("60d", "30m"),
                "1h": ("730d", "1h"),
                "4h": ("730d", "1h"),  # yfinance doesn't support 4h, will resample
                "1d": ("5y", "1d"),
                "1w": ("10y", "1wk"),
            }

            period, interval = interval_map.get(timeframe, ("2y", "1d"))

            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                return None

            # Standardize column names
            df = df.rename(columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            })

            # Keep only OHLCV columns
            df = df[["open", "high", "low", "close", "volume"]]

            # Resample 4h from 1h if needed
            if timeframe == "4h" and interval == "1h":
                df = df.resample("4h").agg({
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }).dropna()

            # Limit to requested bars
            if len(df) > limit:
                df = df.tail(limit)

            return df

        except Exception as exc:
            logger.warning("Stock OHLCV fetch error %s %s: %s", symbol, timeframe, exc)
            return None

    async def fetch_fundamentals(self, symbol: str) -> dict[str, Any]:
        """Fetch fundamental data for a stock using yfinance.

        Returns dict with: P/E, EPS, revenue, market cap, dividend yield,
        sector, industry, analyst targets, financial statements, etc.
        """
        cache_key = f"fundamentals_{symbol}"
        cached = self._yf_cache.get(cache_key)
        if cached and time.time() - cached["ts"] < 3600:  # 1 hour cache
            return cached["data"]

        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info

            fundamentals = {
                "symbol": symbol,
                "name": info.get("longName", symbol),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "forward_pe": info.get("forwardPE", 0),
                "peg_ratio": info.get("pegRatio", 0),
                "eps": info.get("trailingEps", 0),
                "forward_eps": info.get("forwardEps", 0),
                "revenue": info.get("totalRevenue", 0),
                "revenue_growth": info.get("revenueGrowth", 0),
                "earnings_growth": info.get("earningsGrowth", 0),
                "profit_margin": info.get("profitMargins", 0),
                "operating_margin": info.get("operatingMargins", 0),
                "roe": info.get("returnOnEquity", 0),
                "roa": info.get("returnOnAssets", 0),
                "debt_to_equity": info.get("debtToEquity", 0),
                "current_ratio": info.get("currentRatio", 0),
                "book_value": info.get("bookValue", 0),
                "price_to_book": info.get("priceToBook", 0),
                "dividend_yield": info.get("dividendYield", 0),
                "payout_ratio": info.get("payoutRatio", 0),
                "beta": info.get("beta", 1.0),
                "52w_high": info.get("fiftyTwoWeekHigh", 0),
                "52w_low": info.get("fiftyTwoWeekLow", 0),
                "50d_avg": info.get("fiftyDayAverage", 0),
                "200d_avg": info.get("twoHundredDayAverage", 0),
                "avg_volume": info.get("averageVolume", 0),
                "target_mean": info.get("targetMeanPrice", 0),
                "target_high": info.get("targetHighPrice", 0),
                "target_low": info.get("targetLowPrice", 0),
                "recommendation": info.get("recommendationKey", "none"),
                "num_analysts": info.get("numberOfAnalystOpinions", 0),
                "current_price": info.get("currentPrice", 0),
            }

            # Fetch income statement
            try:
                income = ticker.income_stmt
                if income is not None and not income.empty:
                    fundamentals["income_statement"] = {
                        "periods": list(income.columns.strftime("%Y-%m-%d")),
                        "total_revenue": [float(x) if pd.notna(x) else 0 for x in income.loc["Total Revenue"]] if "Total Revenue" in income.index else [],
                        "net_income": [float(x) if pd.notna(x) else 0 for x in income.loc["Net Income"]] if "Net Income" in income.index else [],
                        "ebitda": [float(x) if pd.notna(x) else 0 for x in income.loc["EBITDA"]] if "EBITDA" in income.index else [],
                    }
            except Exception:
                pass

            # Fetch balance sheet
            try:
                balance = ticker.balance_sheet
                if balance is not None and not balance.empty:
                    fundamentals["balance_sheet"] = {
                        "total_assets": float(balance.loc["Total Assets"].iloc[0]) if "Total Assets" in balance.index else 0,
                        "total_debt": float(balance.loc["Total Debt"].iloc[0]) if "Total Debt" in balance.index else 0,
                        "total_equity": float(balance.loc["Stockholders Equity"].iloc[0]) if "Stockholders Equity" in balance.index else 0,
                        "cash": float(balance.loc["Cash And Cash Equivalents"].iloc[0]) if "Cash And Cash Equivalents" in balance.index else 0,
                    }
            except Exception:
                pass

            # Fetch cash flow
            try:
                cashflow = ticker.cashflow
                if cashflow is not None and not cashflow.empty:
                    fundamentals["cash_flow"] = {
                        "operating_cf": [float(x) if pd.notna(x) else 0 for x in cashflow.loc["Operating Cash Flow"]] if "Operating Cash Flow" in cashflow.index else [],
                        "free_cf": [float(x) if pd.notna(x) else 0 for x in cashflow.loc["Free Cash Flow"]] if "Free Cash Flow" in cashflow.index else [],
                        "capex": [float(x) if pd.notna(x) else 0 for x in cashflow.loc["Capital Expenditure"]] if "Capital Expenditure" in cashflow.index else [],
                    }
            except Exception:
                pass

            # Fetch earnings dates
            try:
                earnings_dates = ticker.earnings_dates
                if earnings_dates is not None and not earnings_dates.empty:
                    upcoming = earnings_dates[earnings_dates.index >= pd.Timestamp.now(tz="UTC")]
                    if not upcoming.empty:
                        fundamentals["next_earnings_date"] = str(upcoming.index[0])
                    fundamentals["earnings_history"] = []
                    past = earnings_dates[earnings_dates.index < pd.Timestamp.now(tz="UTC")].head(8)
                    for idx, row in past.iterrows():
                        fundamentals["earnings_history"].append({
                            "date": str(idx),
                            "eps_estimate": float(row.get("EPS Estimate", 0)) if pd.notna(row.get("EPS Estimate")) else None,
                            "eps_actual": float(row.get("Reported EPS", 0)) if pd.notna(row.get("Reported EPS")) else None,
                            "surprise_pct": float(row.get("Surprise(%)", 0)) if pd.notna(row.get("Surprise(%)")) else None,
                        })
            except Exception:
                pass

            self._yf_cache[cache_key] = {"data": fundamentals, "ts": time.time()}
            return fundamentals

        except Exception as exc:
            logger.warning("Fundamentals fetch error %s: %s", symbol, exc)
            return {"symbol": symbol, "error": str(exc)}

    async def get_balance(self) -> dict[str, Any]:
        """Get account balance."""
        if self._paper_trading or not self._alpaca_client:
            positions_value = sum(
                p.get("market_value", 0) for p in self._paper_positions.values()
            )
            return {
                "cash": self._paper_balance,
                "positions_value": positions_value,
                "total_equity": self._paper_balance + positions_value,
                "total_usdt": self._paper_balance + positions_value,  # compatibility
            }

        try:
            account = self._alpaca_client.get_account()
            return {
                "cash": float(account.cash),
                "positions_value": float(account.long_market_value),
                "total_equity": float(account.equity),
                "total_usdt": float(account.equity),  # compatibility
                "buying_power": float(account.buying_power),
            }
        except Exception as exc:
            logger.error("Balance fetch failed: %s", exc)
            return {"cash": 0, "positions_value": 0, "total_equity": 0, "total_usdt": 0}

    async def place_market_order(
        self, symbol: str, side: str, qty: float
    ) -> dict[str, Any]:
        """Place a market order.

        Args:
            symbol: Stock ticker.
            side: "buy" or "sell".
            qty: Number of shares.

        Returns:
            Order record dict.
        """
        if self._paper_trading or not self._alpaca_client:
            return await self._paper_market_order(symbol, side, qty)

        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            order = self._alpaca_client.submit_order(request)

            log_with_data(logger, "info", "Stock order placed", {
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "order_id": str(order.id),
            })

            return {
                "order_id": str(order.id),
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "status": order.status.value,
            }
        except Exception as exc:
            logger.error("Order failed %s %s %s: %s", symbol, side, qty, exc)
            raise

    async def execute_entry(self, signal: dict[str, Any]) -> dict[str, Any]:
        """Execute a trade entry from a signal dict (compatible with crypto engine)."""
        symbol = signal["symbol"]
        direction = signal["direction"]
        amount = signal.get("amount", 0)
        side = "buy" if direction == "long" else "sell"

        order = await self.place_market_order(symbol, side, amount)

        # Place stop loss if provided
        stop_loss = signal.get("stop_loss", 0)
        if stop_loss > 0 and not self._paper_trading and self._alpaca_client:
            try:
                from alpaca.trading.requests import StopOrderRequest
                from alpaca.trading.enums import OrderSide, TimeInForce

                stop_side = OrderSide.SELL if side == "buy" else OrderSide.BUY
                stop_request = StopOrderRequest(
                    symbol=symbol,
                    qty=amount,
                    side=stop_side,
                    time_in_force=TimeInForce.GTC,
                    stop_price=stop_loss,
                )
                self._alpaca_client.submit_order(stop_request)
            except Exception as exc:
                logger.warning("Stop loss order failed for %s: %s", symbol, exc)

        order["trade_id"] = order.get("order_id", str(uuid.uuid4()))
        return order

    async def close_position(self, symbol: str) -> dict[str, Any]:
        """Close all shares of a position."""
        if self._paper_trading or not self._alpaca_client:
            pos = self._paper_positions.pop(symbol, None)
            if pos:
                self._paper_balance += pos.get("market_value", 0)
            return {"symbol": symbol, "closed": True}

        try:
            self._alpaca_client.close_position(symbol)
            return {"symbol": symbol, "closed": True}
        except Exception as exc:
            logger.error("Close position failed %s: %s", symbol, exc)
            raise

    async def get_position(self, symbol: str) -> dict[str, Any]:
        """Get current position for a symbol."""
        if self._paper_trading:
            pos = self._paper_positions.get(symbol, {})
            return {"amount": pos.get("qty", 0.0), "symbol": symbol}

        try:
            pos = self._alpaca_client.get_open_position(symbol)
            return {
                "amount": float(pos.qty),
                "symbol": symbol,
                "market_value": float(pos.market_value),
                "unrealized_pl": float(pos.unrealized_pl),
            }
        except Exception:
            return {"amount": 0.0, "symbol": symbol}

    async def cancel_all_orders(self, symbol: str) -> list:
        """Cancel all open orders for a symbol."""
        if self._paper_trading:
            return []
        try:
            self._alpaca_client.cancel_orders()
            return []
        except Exception:
            return []

    async def close(self) -> None:
        """Cleanup resources."""
        pass

    # ------------------------------------------------------------------
    # Paper trading helpers
    # ------------------------------------------------------------------

    async def _paper_market_order(
        self, symbol: str, side: str, qty: float
    ) -> dict[str, Any]:
        """Simulate a market order in paper mode using yfinance price."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info
            price = info.get("currentPrice", 0) or info.get("regularMarketPrice", 0)
        except Exception:
            price = 0

        if price <= 0:
            # Try from cached OHLCV
            logger.warning("Paper order: could not get price for %s", symbol)
            price = 100.0  # fallback

        order_id = str(uuid.uuid4())[:8]
        cost = price * qty

        if side == "buy":
            if cost > self._paper_balance:
                raise ValueError(f"Insufficient paper balance: {self._paper_balance:.2f} < {cost:.2f}")
            self._paper_balance -= cost
            existing = self._paper_positions.get(symbol, {"qty": 0, "avg_price": 0, "market_value": 0})
            total_qty = existing["qty"] + qty
            avg_price = ((existing["avg_price"] * existing["qty"]) + (price * qty)) / total_qty if total_qty > 0 else price
            self._paper_positions[symbol] = {
                "qty": total_qty,
                "avg_price": avg_price,
                "market_value": total_qty * price,
            }
        else:
            pos = self._paper_positions.get(symbol, {})
            if pos.get("qty", 0) < qty:
                raise ValueError(f"Insufficient shares: {pos.get('qty', 0)} < {qty}")
            self._paper_balance += cost
            remaining = pos["qty"] - qty
            if remaining <= 0:
                self._paper_positions.pop(symbol, None)
            else:
                pos["qty"] = remaining
                pos["market_value"] = remaining * price

        log_with_data(logger, "info", "Paper stock order filled", {
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price,
            "cost": cost,
        })

        return {
            "order_id": order_id,
            "trade_id": order_id,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price,
            "status": "filled",
        }
