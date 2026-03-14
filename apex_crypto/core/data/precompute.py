"""Pre-compute indicator cache for the APEX Crypto Trading System.

Maintains a cache of pre-computed indicators for all symbols and timeframes
to avoid redundant computation during trading cycles.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import pandas as pd

from apex_crypto.core.logging import get_logger, log_with_data

logger = get_logger("data.precompute")


class PreComputeCache:
    """Caches pre-computed indicators for all symbols and timeframes.

    Refreshes when the underlying OHLCV data is updated.
    """

    def __init__(self, indicator_engine: Any, config: dict | None = None) -> None:
        self._indicator_engine = indicator_engine
        self._config = config or {}
        self._cache: dict[str, dict[str, pd.DataFrame]] = {}
        self._last_refresh: float = 0.0
        self._refresh_count: int = 0

        log_with_data(logger, "info", "PreComputeCache initialized", {
            "config": bool(config),
        })

    def refresh(
        self,
        ohlcv_cache: dict[str, dict[str, pd.DataFrame]],
    ) -> dict[str, Any]:
        """Re-compute indicators for all symbols with updated data.

        Args:
            ohlcv_cache: Symbol → timeframe → DataFrame mapping of OHLCV data.

        Returns:
            Summary of refresh results.
        """
        start = time.time()
        symbols_computed = 0
        errors: list[str] = []

        for symbol, tf_data in ohlcv_cache.items():
            if not tf_data:
                continue

            self._cache.setdefault(symbol, {})
            for tf, df in tf_data.items():
                if df is None or df.empty:
                    continue
                try:
                    ind_df = df.copy().reset_index()
                    computed = self._indicator_engine.compute_all(ind_df, tf)
                    self._cache[symbol][tf] = computed
                except Exception as exc:
                    errors.append(f"{symbol}/{tf}: {exc}")

            if self._cache.get(symbol):
                symbols_computed += 1

        self._last_refresh = time.time()
        self._refresh_count += 1
        elapsed = time.time() - start

        log_with_data(logger, "info", "PreCompute cache refreshed", {
            "symbols": symbols_computed,
            "cached_symbols": len(self._cache),
            "errors": len(errors),
            "elapsed_ms": round(elapsed * 1000),
        })

        return {
            "symbols_computed": symbols_computed,
            "cached_symbols": len(self._cache),
            "errors": errors,
            "elapsed_ms": round(elapsed * 1000),
        }

    def get(
        self, symbol: str, timeframe: str
    ) -> Optional[pd.DataFrame]:
        """Get cached indicators for a symbol and timeframe.

        Args:
            symbol: Trading pair symbol.
            timeframe: Candle timeframe.

        Returns:
            Computed indicator DataFrame or None if not cached.
        """
        return self._cache.get(symbol, {}).get(timeframe)

    def get_all(self, symbol: str) -> dict[str, pd.DataFrame]:
        """Get all cached timeframe indicators for a symbol."""
        return self._cache.get(symbol, {})

    @property
    def cached_symbols(self) -> int:
        """Number of symbols with cached data."""
        return len(self._cache)

    @property
    def last_refresh(self) -> float:
        """Timestamp of last cache refresh."""
        return self._last_refresh
