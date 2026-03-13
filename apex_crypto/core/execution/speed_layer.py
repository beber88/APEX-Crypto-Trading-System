"""Speed layer for sub-second trade execution.

Pre-computes position sizes, stop distances, and key levels every 5 minutes,
stores them in a cache for instant access when signals fire.
Implements order pre-staging and parallel execution.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from apex_crypto.core.logging import get_logger, log_with_data

logger = get_logger("execution.speed_layer")


class PreComputeCache:
    """In-memory cache of pre-computed trading parameters for all assets.

    Refreshed every 5 minutes. When a signal fires, all parameters
    are instantly available without computation.
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._cache: dict[str, dict[str, Any]] = {}
        self._last_refresh: float = 0.0
        self._refresh_interval: float = config.get("precompute_refresh_seconds", 300)
        self._pre_staged_orders: dict[str, dict[str, Any]] = {}

        log_with_data(logger, "info", "PreComputeCache initialized", {
            "refresh_interval": self._refresh_interval,
        })

    async def refresh(self, symbols: list[str], ohlcv_cache: dict,
                      indicator_engine: Any, risk_config: dict,
                      equity: float) -> None:
        """Pre-compute all trading parameters for all symbols.

        For each symbol, pre-calculates:
        - Current ATR value → stop distance
        - Position sizes for 0.5%, 1%, 1.5%, 2%, 3% risk
        - VWAP level
        - Key support/resistance zones
        - Bollinger Band levels
        """
        now = time.time()
        if now - self._last_refresh < self._refresh_interval:
            return

        refresh_start = time.time()

        for symbol in symbols:
            try:
                data = ohlcv_cache.get(symbol, {})
                entry = {}

                # Get ATR from 1h or 4h timeframe
                for tf in ["1h", "4h"]:
                    if tf in data and len(data[tf]) >= 14:
                        df = data[tf]
                        highs = df["high"].values
                        lows = df["low"].values
                        closes = df["close"].values

                        # ATR calculation
                        import numpy as np
                        tr = np.maximum(
                            highs[1:] - lows[1:],
                            np.maximum(
                                np.abs(highs[1:] - closes[:-1]),
                                np.abs(lows[1:] - closes[:-1])
                            )
                        )
                        atr = float(np.mean(tr[-14:])) if len(tr) >= 14 else float(np.mean(tr))
                        current_price = float(closes[-1])

                        entry["atr"] = atr
                        entry["atr_pct"] = atr / current_price * 100 if current_price > 0 else 0
                        entry["current_price"] = current_price
                        entry["timeframe"] = tf

                        # Pre-compute stop distances
                        atr_mult = risk_config.get("atr_stop_multiplier", 1.5)
                        entry["stop_distance"] = atr * atr_mult
                        entry["stop_long"] = current_price - entry["stop_distance"]
                        entry["stop_short"] = current_price + entry["stop_distance"]

                        # Pre-compute position sizes for various risk levels
                        entry["sizes"] = {}
                        for risk_pct in [0.5, 1.0, 1.5, 2.0, 3.0]:
                            if entry["stop_distance"] > 0:
                                risk_usdt = equity * (risk_pct / 100.0)
                                size_units = risk_usdt / entry["stop_distance"]
                                size_usdt = size_units * current_price
                                entry["sizes"][str(risk_pct)] = {
                                    "units": round(size_units, 8),
                                    "usdt": round(size_usdt, 2),
                                    "risk_usdt": round(risk_usdt, 2),
                                }

                        # VWAP (from 1h data - approximate daily VWAP)
                        if len(df) >= 24:
                            last_24 = df.tail(24)
                            typical_price = (last_24["high"] + last_24["low"] + last_24["close"]) / 3
                            vwap = float((typical_price * last_24["volume"]).sum() / last_24["volume"].sum())
                            entry["vwap"] = vwap
                            entry["vwap_deviation_pct"] = (current_price - vwap) / vwap * 100

                        # Bollinger Bands
                        if len(closes) >= 20:
                            sma20 = float(np.mean(closes[-20:]))
                            std20 = float(np.std(closes[-20:]))
                            entry["bb_upper"] = sma20 + 2 * std20
                            entry["bb_lower"] = sma20 - 2 * std20
                            entry["bb_middle"] = sma20

                        # Take profit levels
                        tp1_r = risk_config.get("tp1_r", 1.5)
                        tp2_r = risk_config.get("tp2_r", 2.5)
                        entry["tp_long"] = {
                            "tp1": current_price + entry["stop_distance"] * tp1_r,
                            "tp2": current_price + entry["stop_distance"] * tp2_r,
                        }
                        entry["tp_short"] = {
                            "tp1": current_price - entry["stop_distance"] * tp1_r,
                            "tp2": current_price - entry["stop_distance"] * tp2_r,
                        }

                        break

                if entry:
                    entry["computed_at"] = now
                    self._cache[symbol] = entry

            except Exception as exc:
                logger.debug("PreCompute error for %s: %s", symbol, exc)

        self._last_refresh = now
        elapsed_ms = (time.time() - refresh_start) * 1000

        log_with_data(logger, "info", "PreCompute cache refreshed", {
            "symbols": len(self._cache),
            "elapsed_ms": round(elapsed_ms, 1),
        })

    def get(self, symbol: str) -> Optional[dict[str, Any]]:
        """Get pre-computed data for a symbol."""
        entry = self._cache.get(symbol)
        if entry is None:
            return None
        # Check staleness (10 min max)
        if time.time() - entry.get("computed_at", 0) > 600:
            return None
        return entry

    def get_instant_order_params(self, symbol: str, direction: str,
                                  risk_pct: float = 1.0) -> Optional[dict[str, Any]]:
        """Get instant order parameters - everything needed to place an order NOW.

        Returns pre-computed entry, stop, TP, and size - ready to submit.
        """
        data = self.get(symbol)
        if data is None:
            return None

        size_key = str(risk_pct)
        size_data = data.get("sizes", {}).get(size_key)
        if size_data is None:
            return None

        if direction == "long":
            return {
                "symbol": symbol,
                "direction": "long",
                "entry_price": data["current_price"],
                "stop_loss": data["stop_long"],
                "take_profit_1": data["tp_long"]["tp1"],
                "take_profit_2": data["tp_long"]["tp2"],
                "size_units": size_data["units"],
                "size_usdt": size_data["usdt"],
                "risk_usdt": size_data["risk_usdt"],
                "atr": data["atr"],
                "precomputed": True,
            }
        else:
            return {
                "symbol": symbol,
                "direction": "short",
                "entry_price": data["current_price"],
                "stop_loss": data["stop_short"],
                "take_profit_1": data["tp_short"]["tp1"],
                "take_profit_2": data["tp_short"]["tp2"],
                "size_units": size_data["units"],
                "size_usdt": size_data["usdt"],
                "risk_usdt": size_data["risk_usdt"],
                "atr": data["atr"],
                "precomputed": True,
            }

    def pre_stage_order(self, symbol: str, direction: str,
                         risk_pct: float = 1.0) -> Optional[dict]:
        """Pre-stage an order for instant submission when confirmed."""
        params = self.get_instant_order_params(symbol, direction, risk_pct)
        if params:
            params["staged_at"] = time.time()
            self._pre_staged_orders[symbol] = params
            log_with_data(logger, "debug", "Order pre-staged", {
                "symbol": symbol, "direction": direction,
            })
        return params

    def get_staged_order(self, symbol: str) -> Optional[dict]:
        """Get a pre-staged order if still fresh (<60 seconds)."""
        order = self._pre_staged_orders.get(symbol)
        if order and time.time() - order.get("staged_at", 0) < 60:
            return order
        # Expired
        self._pre_staged_orders.pop(symbol, None)
        return None

    def get_all_cached(self) -> dict[str, dict[str, Any]]:
        """Return entire cache for dashboard display."""
        return dict(self._cache)


async def parallel_scan_symbols(
    symbols: list[str],
    scan_func,
    timeframes: list[str],
    max_concurrent: int = 10,
) -> list[dict[str, Any]]:
    """Scan all symbols in parallel using asyncio.gather().

    Instead of scanning 15 assets sequentially (15 x 200ms = 3s),
    scan all simultaneously (max 200ms total).

    Args:
        symbols: List of trading pairs to scan.
        scan_func: Async coroutine that scans a single symbol.
        timeframes: Timeframes to pass to scan_func.
        max_concurrent: Max parallel tasks (respect API limits).

    Returns:
        List of non-None scan results.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_scan(symbol: str):
        async with semaphore:
            try:
                return await scan_func(symbol, timeframes)
            except Exception as exc:
                logger.debug("Parallel scan error for %s: %s", symbol, exc)
                return None

    scan_start = time.time()
    tasks = [bounded_scan(sym) for sym in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    elapsed_ms = (time.time() - scan_start) * 1000
    valid_results = [r for r in results if r is not None and not isinstance(r, Exception)]

    log_with_data(logger, "info", "Parallel scan completed", {
        "total_symbols": len(symbols),
        "valid_results": len(valid_results),
        "elapsed_ms": round(elapsed_ms, 1),
    })

    return valid_results


async def parallel_order_execution(
    broker: Any,
    symbol: str,
    direction: str,
    entry_order: dict,
    sl_order: dict,
    tp_orders: list[dict],
) -> dict[str, Any]:
    """Execute entry then place SL/TP orders in parallel.

    Current: entry -> wait -> SL -> wait -> TP1 -> wait -> TP2 (sequential)
    Optimized: entry -> wait -> gather(SL, TP1, TP2) (parallel protective orders)
    """
    execution_start = time.time()

    # Step 1: Place entry order
    try:
        fill = await broker.execute_entry(entry_order)
    except Exception as exc:
        logger.error("Entry order failed for %s: %s", symbol, exc)
        return {"success": False, "error": str(exc)}

    # Step 2: Place protective orders in parallel
    protective_tasks = []

    if sl_order:
        protective_tasks.append(
            _safe_place_order(broker, "stop_loss", sl_order)
        )

    for tp in tp_orders:
        if tp:
            protective_tasks.append(
                _safe_place_order(broker, "take_profit", tp)
            )

    if protective_tasks:
        protective_results = await asyncio.gather(*protective_tasks, return_exceptions=True)
    else:
        protective_results = []

    elapsed_ms = (time.time() - execution_start) * 1000

    log_with_data(logger, "info", "Full trade execution completed", {
        "symbol": symbol,
        "direction": direction,
        "elapsed_ms": round(elapsed_ms, 1),
        "protective_orders": len(protective_tasks),
    })

    return {
        "success": True,
        "fill": fill,
        "protective_results": [r for r in protective_results if not isinstance(r, Exception)],
        "elapsed_ms": elapsed_ms,
    }


async def _safe_place_order(broker: Any, order_type: str, order: dict) -> dict:
    """Place an order with error handling."""
    try:
        result = await broker.place_order(order)
        return {"type": order_type, "success": True, "result": result}
    except Exception as exc:
        logger.warning("Failed to place %s order: %s", order_type, exc)
        return {"type": order_type, "success": False, "error": str(exc)}
