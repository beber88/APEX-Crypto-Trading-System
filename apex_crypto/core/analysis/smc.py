"""Smart Money Concepts (SMC/ICT) analysis engine for APEX crypto trading system.

Implements institutional trading concepts including order blocks, fair value gaps,
market structure breaks, liquidity sweeps, breaker blocks, change of character,
premium/discount zones, and automated support/resistance detection.

Input: pandas DataFrame with columns [timestamp, open, high, low, close, volume].
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from apex_crypto.core.logging import get_logger, log_with_data

logger = get_logger("smc")


class SMCEngine:
    """Smart Money Concepts analysis engine.

    Provides fractal-based swing detection, order block identification,
    fair value gap detection, market structure analysis, and other
    institutional trading concept tools.

    Attributes:
        swing_lookback: Number of bars on each side for swing point detection.
        ob_min_move_pct: Minimum impulsive move percentage for order blocks.
        fvg_min_size_pct: Minimum FVG size as percentage of price.
        sr_lookback: Number of bars for support/resistance calculation.
        sr_cluster_tolerance: Tolerance for clustering S/R levels.
    """

    def __init__(self, config: dict) -> None:
        """Initialize SMCEngine with configuration parameters.

        Args:
            config: SMC configuration section from config.yaml. Expected keys:
                swing_lookback (int): Lookback period for swing detection. Default 5.
                ob_min_move_pct (float): Min move % for order blocks. Default 0.5.
                fvg_min_size_pct (float): Min FVG size as % of price. Default 0.1.
                sr_lookback (int): Lookback for S/R calculation. Default 200.
                sr_cluster_tolerance (float): S/R clustering tolerance. Default 0.003.
        """
        self.swing_lookback: int = config.get("swing_lookback", 5)
        self.ob_min_move_pct: float = config.get("ob_min_move_pct", 0.5)
        self.fvg_min_size_pct: float = config.get("fvg_min_size_pct", 0.1)
        self.sr_lookback: int = config.get("sr_lookback", 200)
        self.sr_cluster_tolerance: float = config.get("sr_cluster_tolerance", 0.003)

        log_with_data(logger, "info", "SMCEngine initialized", {
            "swing_lookback": self.swing_lookback,
            "ob_min_move_pct": self.ob_min_move_pct,
            "fvg_min_size_pct": self.fvg_min_size_pct,
            "sr_lookback": self.sr_lookback,
            "sr_cluster_tolerance": self.sr_cluster_tolerance,
        })

    def detect_swing_points(
        self, df: pd.DataFrame, lookback: Optional[int] = None
    ) -> pd.DataFrame:
        """Detect fractal-based swing highs and swing lows.

        A swing high occurs when high[i] is greater than all highs within
        ``lookback`` bars on each side. A swing low occurs when low[i] is
        lower than all lows within ``lookback`` bars on each side.

        Args:
            df: OHLCV DataFrame with columns [timestamp, open, high, low, close, volume].
            lookback: Number of bars on each side to compare. Defaults to
                ``self.swing_lookback``.

        Returns:
            Copy of ``df`` with added columns: swing_high (bool), swing_low (bool),
            swing_high_price (float or NaN), swing_low_price (float or NaN).
        """
        lookback = lookback if lookback is not None else self.swing_lookback
        result = df.copy()
        n = len(result)

        swing_high = np.zeros(n, dtype=bool)
        swing_low = np.zeros(n, dtype=bool)

        highs = result["high"].values
        lows = result["low"].values

        if n < 2 * lookback + 1:
            result["swing_high"] = False
            result["swing_low"] = False
            result["swing_high_price"] = np.nan
            result["swing_low_price"] = np.nan
            log_with_data(logger, "warning", "Insufficient data for swing detection", {
                "rows": n, "required": 2 * lookback + 1,
            })
            return result

        for i in range(lookback, n - lookback):
            left_highs = highs[i - lookback: i]
            right_highs = highs[i + 1: i + lookback + 1]
            if highs[i] > np.max(left_highs) and highs[i] > np.max(right_highs):
                swing_high[i] = True

            left_lows = lows[i - lookback: i]
            right_lows = lows[i + 1: i + lookback + 1]
            if lows[i] < np.min(left_lows) and lows[i] < np.min(right_lows):
                swing_low[i] = True

        result["swing_high"] = swing_high
        result["swing_low"] = swing_low
        result["swing_high_price"] = np.where(swing_high, highs, np.nan)
        result["swing_low_price"] = np.where(swing_low, lows, np.nan)

        sh_count = int(np.sum(swing_high))
        sl_count = int(np.sum(swing_low))
        log_with_data(logger, "info", "Swing points detected", {
            "swing_highs": sh_count, "swing_lows": sl_count,
        })

        return result

    def detect_order_blocks(
        self, df: pd.DataFrame, min_move_pct: Optional[float] = None
    ) -> list[dict[str, Any]]:
        """Detect order blocks based on impulsive price moves.

        A bullish order block is the last bearish candle before an impulsive
        bullish move. A bearish order block is the last bullish candle before
        an impulsive bearish move. The impulsive move must exceed
        ``min_move_pct`` within 1-3 candles after the OB candle.

        Args:
            df: OHLCV DataFrame.
            min_move_pct: Minimum impulsive move as percentage of price.
                Defaults to ``self.ob_min_move_pct``.

        Returns:
            List of order block dicts with keys: type, index, high, low, mid,
            timestamp, valid.
        """
        min_move_pct = min_move_pct if min_move_pct is not None else self.ob_min_move_pct
        order_blocks: list[dict[str, Any]] = []
        n = len(df)

        if n < 4:
            log_with_data(logger, "warning", "Insufficient data for OB detection", {
                "rows": n,
            })
            return order_blocks

        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        timestamps = df["timestamp"].values

        for i in range(n - 2):
            is_bearish_candle = closes[i] < opens[i]
            is_bullish_candle = closes[i] > opens[i]

            if is_bearish_candle:
                # Check for impulsive bullish move in the next 1-3 candles
                max_reach = min(i + 4, n)
                for j in range(i + 1, max_reach):
                    move_pct = ((highs[j] - lows[i]) / lows[i]) * 100.0
                    if move_pct >= min_move_pct:
                        ob: dict[str, Any] = {
                            "type": "bullish",
                            "index": int(i),
                            "high": float(highs[i]),
                            "low": float(lows[i]),
                            "mid": float((highs[i] + lows[i]) / 2.0),
                            "timestamp": str(timestamps[i]),
                            "valid": True,
                        }
                        order_blocks.append(ob)
                        break

            if is_bullish_candle:
                # Check for impulsive bearish move in the next 1-3 candles
                max_reach = min(i + 4, n)
                for j in range(i + 1, max_reach):
                    move_pct = ((highs[i] - lows[j]) / highs[i]) * 100.0
                    if move_pct >= min_move_pct:
                        ob = {
                            "type": "bearish",
                            "index": int(i),
                            "high": float(highs[i]),
                            "low": float(lows[i]),
                            "mid": float((highs[i] + lows[i]) / 2.0),
                            "timestamp": str(timestamps[i]),
                            "valid": True,
                        }
                        order_blocks.append(ob)
                        break

        # Validate order blocks: invalidated if price closes beyond the OB
        for ob in order_blocks:
            ob_idx = ob["index"]
            subsequent_closes = closes[ob_idx + 1:]
            if ob["type"] == "bullish":
                # Bullish OB invalidated if price closes below its low
                if len(subsequent_closes) > 0 and np.any(subsequent_closes < ob["low"]):
                    ob["valid"] = False
            else:
                # Bearish OB invalidated if price closes above its high
                if len(subsequent_closes) > 0 and np.any(subsequent_closes > ob["high"]):
                    ob["valid"] = False

        valid_count = sum(1 for ob in order_blocks if ob["valid"])
        log_with_data(logger, "info", "Order blocks detected", {
            "total": len(order_blocks), "valid": valid_count,
            "bullish": sum(1 for ob in order_blocks if ob["type"] == "bullish"),
            "bearish": sum(1 for ob in order_blocks if ob["type"] == "bearish"),
        })

        return order_blocks

    def detect_fair_value_gaps(
        self, df: pd.DataFrame, min_size_pct: Optional[float] = None
    ) -> list[dict[str, Any]]:
        """Detect fair value gaps (FVGs) in price action.

        A bullish FVG exists when candle[i-2].high < candle[i].low, leaving
        a gap that price did not trade through. A bearish FVG exists when
        candle[i-2].low > candle[i].high.

        Args:
            df: OHLCV DataFrame.
            min_size_pct: Minimum gap size as percentage of price.
                Defaults to ``self.fvg_min_size_pct``.

        Returns:
            List of FVG dicts with keys: type, index, top, bottom, mid,
            timestamp, filled, fill_pct.
        """
        min_size_pct = min_size_pct if min_size_pct is not None else self.fvg_min_size_pct
        fvgs: list[dict[str, Any]] = []
        n = len(df)

        if n < 3:
            log_with_data(logger, "warning", "Insufficient data for FVG detection", {
                "rows": n,
            })
            return fvgs

        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        timestamps = df["timestamp"].values

        for i in range(2, n):
            ref_price = closes[i - 1]
            if ref_price == 0:
                continue

            # Bullish FVG: gap between candle[i-2].high and candle[i].low
            if lows[i] > highs[i - 2]:
                gap_size = lows[i] - highs[i - 2]
                gap_pct = (gap_size / ref_price) * 100.0
                if gap_pct >= min_size_pct:
                    top = float(lows[i])
                    bottom = float(highs[i - 2])
                    mid = (top + bottom) / 2.0

                    # Check fill status from subsequent candles
                    filled = False
                    fill_pct = 0.0
                    if i + 1 < n:
                        subsequent_lows = lows[i + 1:]
                        if len(subsequent_lows) > 0:
                            min_low = float(np.min(subsequent_lows))
                            if min_low <= bottom:
                                filled = True
                                fill_pct = 100.0
                            elif min_low < top:
                                fill_pct = float(
                                    ((top - min_low) / gap_size) * 100.0
                                )

                    fvgs.append({
                        "type": "bullish",
                        "index": int(i - 1),
                        "top": top,
                        "bottom": bottom,
                        "mid": mid,
                        "timestamp": str(timestamps[i - 1]),
                        "filled": filled,
                        "fill_pct": round(fill_pct, 2),
                    })

            # Bearish FVG: gap between candle[i-2].low and candle[i].high
            if highs[i] < lows[i - 2]:
                gap_size = lows[i - 2] - highs[i]
                gap_pct = (gap_size / ref_price) * 100.0
                if gap_pct >= min_size_pct:
                    top = float(lows[i - 2])
                    bottom = float(highs[i])
                    mid = (top + bottom) / 2.0

                    # Check fill status from subsequent candles
                    filled = False
                    fill_pct = 0.0
                    if i + 1 < n:
                        subsequent_highs = highs[i + 1:]
                        if len(subsequent_highs) > 0:
                            max_high = float(np.max(subsequent_highs))
                            if max_high >= top:
                                filled = True
                                fill_pct = 100.0
                            elif max_high > bottom:
                                fill_pct = float(
                                    ((max_high - bottom) / gap_size) * 100.0
                                )

                    fvgs.append({
                        "type": "bearish",
                        "index": int(i - 1),
                        "top": top,
                        "bottom": bottom,
                        "mid": mid,
                        "timestamp": str(timestamps[i - 1]),
                        "filled": filled,
                        "fill_pct": round(fill_pct, 2),
                    })

        log_with_data(logger, "info", "Fair value gaps detected", {
            "total": len(fvgs),
            "bullish": sum(1 for f in fvgs if f["type"] == "bullish"),
            "bearish": sum(1 for f in fvgs if f["type"] == "bearish"),
            "filled": sum(1 for f in fvgs if f["filled"]),
        })

        return fvgs

    def detect_breaker_blocks(
        self, df: pd.DataFrame, order_blocks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Detect breaker blocks from failed order blocks.

        A breaker block forms when an order block fails and price breaks
        through it, flipping its polarity. A failed bullish OB becomes a
        bearish breaker (resistance), and a failed bearish OB becomes a
        bullish breaker (support).

        Args:
            df: OHLCV DataFrame.
            order_blocks: List of order block dicts from ``detect_order_blocks``.

        Returns:
            List of breaker block dicts with keys: type, index, high, low,
            original_ob_type, timestamp.
        """
        breakers: list[dict[str, Any]] = []

        if not order_blocks or len(df) == 0:
            log_with_data(logger, "info", "No order blocks to evaluate for breakers", {})
            return breakers

        closes = df["close"].values
        timestamps = df["timestamp"].values
        n = len(closes)

        for ob in order_blocks:
            if ob["valid"]:
                continue

            ob_idx = ob["index"]
            if ob_idx + 1 >= n:
                continue

            # Find the bar where the OB was broken
            break_idx: Optional[int] = None
            if ob["type"] == "bullish":
                # Bullish OB broken when price closes below its low
                for k in range(ob_idx + 1, n):
                    if closes[k] < ob["low"]:
                        break_idx = k
                        break
            else:
                # Bearish OB broken when price closes above its high
                for k in range(ob_idx + 1, n):
                    if closes[k] > ob["high"]:
                        break_idx = k
                        break

            if break_idx is not None:
                # Polarity flip: bullish OB -> bearish breaker, vice versa
                breaker_type = "bearish" if ob["type"] == "bullish" else "bullish"
                breakers.append({
                    "type": breaker_type,
                    "index": int(break_idx),
                    "high": float(ob["high"]),
                    "low": float(ob["low"]),
                    "original_ob_type": ob["type"],
                    "timestamp": str(timestamps[break_idx]),
                })

        log_with_data(logger, "info", "Breaker blocks detected", {
            "total": len(breakers),
            "bullish": sum(1 for b in breakers if b["type"] == "bullish"),
            "bearish": sum(1 for b in breakers if b["type"] == "bearish"),
        })

        return breakers

    def detect_liquidity_sweeps(
        self, df: pd.DataFrame, swing_points_df: pd.DataFrame
    ) -> list[dict[str, Any]]:
        """Detect liquidity sweeps of equal highs and equal lows.

        Equal highs/lows are swing points within ``sr_cluster_tolerance`` of
        each other. A sweep occurs when price exceeds the level then closes
        back inside, indicating a stop hunt.

        Args:
            df: OHLCV DataFrame.
            swing_points_df: DataFrame with swing point columns from
                ``detect_swing_points``.

        Returns:
            List of sweep dicts with keys: type ('buy_side' or 'sell_side'),
            index, level, sweep_high or sweep_low, timestamp.
        """
        sweeps: list[dict[str, Any]] = []
        n = len(df)

        if n == 0:
            return sweeps

        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        timestamps = df["timestamp"].values
        tolerance = self.sr_cluster_tolerance

        # Collect swing high and swing low levels with their indices
        sh_mask = swing_points_df["swing_high"].values.astype(bool)
        sl_mask = swing_points_df["swing_low"].values.astype(bool)

        sh_indices = np.where(sh_mask)[0]
        sl_indices = np.where(sl_mask)[0]

        sh_prices = swing_points_df["swing_high_price"].values[sh_indices]
        sl_prices = swing_points_df["swing_low_price"].values[sl_indices]

        # Find clusters of equal highs (buy-side liquidity)
        buy_side_levels = self._cluster_levels(sh_prices, sh_indices, tolerance)

        # Find clusters of equal lows (sell-side liquidity)
        sell_side_levels = self._cluster_levels(sl_prices, sl_indices, tolerance)

        # Detect buy-side sweeps (price runs above equal highs then reverses)
        for level, member_indices in buy_side_levels:
            if len(member_indices) < 2:
                continue
            last_formation_idx = int(np.max(member_indices))
            for i in range(last_formation_idx + 1, n):
                if highs[i] > level and closes[i] < level:
                    sweeps.append({
                        "type": "buy_side",
                        "index": int(i),
                        "level": float(level),
                        "sweep_high": float(highs[i]),
                        "timestamp": str(timestamps[i]),
                    })
                    break

        # Detect sell-side sweeps (price runs below equal lows then reverses)
        for level, member_indices in sell_side_levels:
            if len(member_indices) < 2:
                continue
            last_formation_idx = int(np.max(member_indices))
            for i in range(last_formation_idx + 1, n):
                if lows[i] < level and closes[i] > level:
                    sweeps.append({
                        "type": "sell_side",
                        "index": int(i),
                        "level": float(level),
                        "sweep_low": float(lows[i]),
                        "timestamp": str(timestamps[i]),
                    })
                    break

        log_with_data(logger, "info", "Liquidity sweeps detected", {
            "total": len(sweeps),
            "buy_side": sum(1 for s in sweeps if s["type"] == "buy_side"),
            "sell_side": sum(1 for s in sweeps if s["type"] == "sell_side"),
        })

        return sweeps

    def _cluster_levels(
        self,
        prices: np.ndarray,
        indices: np.ndarray,
        tolerance: float,
    ) -> list[tuple[float, np.ndarray]]:
        """Cluster price levels that are within tolerance of each other.

        Args:
            prices: Array of price levels.
            indices: Corresponding bar indices.
            tolerance: Relative tolerance for clustering (e.g. 0.003 = 0.3%).

        Returns:
            List of (cluster_mean, member_indices) tuples.
        """
        if len(prices) == 0:
            return []

        sorted_order = np.argsort(prices)
        sorted_prices = prices[sorted_order]
        sorted_indices = indices[sorted_order]

        clusters: list[tuple[float, np.ndarray]] = []
        cluster_prices: list[float] = [sorted_prices[0]]
        cluster_idxs: list[int] = [sorted_indices[0]]

        for k in range(1, len(sorted_prices)):
            ref = np.mean(cluster_prices)
            if ref > 0 and abs(sorted_prices[k] - ref) / ref <= tolerance:
                cluster_prices.append(sorted_prices[k])
                cluster_idxs.append(sorted_indices[k])
            else:
                clusters.append((
                    float(np.mean(cluster_prices)),
                    np.array(cluster_idxs),
                ))
                cluster_prices = [sorted_prices[k]]
                cluster_idxs = [sorted_indices[k]]

        clusters.append((
            float(np.mean(cluster_prices)),
            np.array(cluster_idxs),
        ))

        return clusters

    def detect_market_structure_break(
        self, df: pd.DataFrame, swing_points_df: pd.DataFrame
    ) -> list[dict[str, Any]]:
        """Detect market structure breaks (MSB).

        A bullish MSB occurs when price breaks and closes above the last
        swing high. A bearish MSB occurs when price breaks and closes below
        the last swing low.

        Args:
            df: OHLCV DataFrame.
            swing_points_df: DataFrame with swing point columns from
                ``detect_swing_points``.

        Returns:
            List of MSB dicts with keys: type ('bullish' or 'bearish'),
            index, broken_level, close_price, timestamp.
        """
        msb_list: list[dict[str, Any]] = []
        n = len(df)

        if n == 0:
            return msb_list

        closes = df["close"].values
        timestamps = df["timestamp"].values

        sh_mask = swing_points_df["swing_high"].values.astype(bool)
        sl_mask = swing_points_df["swing_low"].values.astype(bool)
        sh_prices = swing_points_df["swing_high_price"].values
        sl_prices = swing_points_df["swing_low_price"].values

        last_swing_high: Optional[float] = None
        last_swing_low: Optional[float] = None
        last_sh_broken: Optional[float] = None
        last_sl_broken: Optional[float] = None

        for i in range(n):
            # Update last known swing points (only consider confirmed ones)
            if sh_mask[i]:
                last_swing_high = float(sh_prices[i])
                last_sh_broken = None
            if sl_mask[i]:
                last_swing_low = float(sl_prices[i])
                last_sl_broken = None

            # Check for bullish MSB: close above last swing high
            if (
                last_swing_high is not None
                and last_sh_broken != last_swing_high
                and closes[i] > last_swing_high
            ):
                msb_list.append({
                    "type": "bullish",
                    "index": int(i),
                    "broken_level": last_swing_high,
                    "close_price": float(closes[i]),
                    "timestamp": str(timestamps[i]),
                })
                last_sh_broken = last_swing_high

            # Check for bearish MSB: close below last swing low
            if (
                last_swing_low is not None
                and last_sl_broken != last_swing_low
                and closes[i] < last_swing_low
            ):
                msb_list.append({
                    "type": "bearish",
                    "index": int(i),
                    "broken_level": last_swing_low,
                    "close_price": float(closes[i]),
                    "timestamp": str(timestamps[i]),
                })
                last_sl_broken = last_swing_low

        log_with_data(logger, "info", "Market structure breaks detected", {
            "total": len(msb_list),
            "bullish": sum(1 for m in msb_list if m["type"] == "bullish"),
            "bearish": sum(1 for m in msb_list if m["type"] == "bearish"),
        })

        return msb_list

    def detect_change_of_character(
        self, df: pd.DataFrame, msb_list: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Detect Change of Character (CHoCH) events.

        A CHoCH is the first market structure break in the opposite direction
        of the prevailing trend. It signals a potential trend reversal.

        Args:
            df: OHLCV DataFrame (used for timestamp reference if needed).
            msb_list: List of MSB dicts from ``detect_market_structure_break``.

        Returns:
            List of CHoCH dicts with keys: type ('bullish' or 'bearish'),
            index, level, timestamp, previous_trend.
        """
        choch_list: list[dict[str, Any]] = []

        if not msb_list:
            log_with_data(logger, "info", "No MSBs to evaluate for CHoCH", {})
            return choch_list

        # Determine the prevailing trend from the first MSB
        current_trend: Optional[str] = None

        for msb in msb_list:
            msb_type = msb["type"]

            if current_trend is None:
                # Establish initial trend from first MSB
                current_trend = "bullish" if msb_type == "bullish" else "bearish"
                continue

            # CHoCH: MSB in the opposite direction of current trend
            if msb_type != current_trend:
                choch_list.append({
                    "type": msb_type,
                    "index": msb["index"],
                    "level": msb["broken_level"],
                    "timestamp": msb["timestamp"],
                    "previous_trend": current_trend,
                })
                # Update trend direction after CHoCH
                current_trend = msb_type
            # If same direction, trend continues (update trend to absorb
            # consecutive same-direction MSBs, which is a no-op here)

        log_with_data(logger, "info", "Change of character events detected", {
            "total": len(choch_list),
            "bullish": sum(1 for c in choch_list if c["type"] == "bullish"),
            "bearish": sum(1 for c in choch_list if c["type"] == "bearish"),
        })

        return choch_list

    def compute_premium_discount_zones(
        self, df: pd.DataFrame, swing_points_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute premium and discount zones relative to the last swing range.

        Premium zone is above the equilibrium (50% of range), discount zone
        is below it. Institutional traders look to sell in premium and buy
        in discount.

        Args:
            df: OHLCV DataFrame.
            swing_points_df: DataFrame with swing point columns from
                ``detect_swing_points``.

        Returns:
            Copy of ``df`` with added columns: equilibrium, premium_zone (bool),
            discount_zone (bool), zone_pct (0-100 where 0 = swing low,
            100 = swing high).
        """
        result = df.copy()
        n = len(result)

        # Defaults
        result["equilibrium"] = np.nan
        result["premium_zone"] = False
        result["discount_zone"] = False
        result["zone_pct"] = np.nan

        if n == 0:
            return result

        sh_mask = swing_points_df["swing_high"].values.astype(bool)
        sl_mask = swing_points_df["swing_low"].values.astype(bool)
        sh_prices = swing_points_df["swing_high_price"].values
        sl_prices = swing_points_df["swing_low_price"].values

        # Find the last significant swing high and swing low
        sh_indices = np.where(sh_mask)[0]
        sl_indices = np.where(sl_mask)[0]

        if len(sh_indices) == 0 or len(sl_indices) == 0:
            log_with_data(logger, "warning", "Insufficient swing points for zones", {
                "swing_highs": len(sh_indices), "swing_lows": len(sl_indices),
            })
            return result

        last_sh_idx = sh_indices[-1]
        last_sl_idx = sl_indices[-1]
        swing_high_val = float(sh_prices[last_sh_idx])
        swing_low_val = float(sl_prices[last_sl_idx])

        swing_range = swing_high_val - swing_low_val
        if swing_range <= 0:
            log_with_data(logger, "warning", "Invalid swing range for zones", {
                "swing_high": swing_high_val, "swing_low": swing_low_val,
            })
            return result

        equilibrium = swing_low_val + swing_range / 2.0

        closes = result["close"].values
        zone_pct = ((closes - swing_low_val) / swing_range) * 100.0

        result["equilibrium"] = equilibrium
        result["premium_zone"] = closes > equilibrium
        result["discount_zone"] = closes < equilibrium
        result["zone_pct"] = np.clip(zone_pct, 0.0, 100.0)

        log_with_data(logger, "info", "Premium/discount zones computed", {
            "swing_high": swing_high_val,
            "swing_low": swing_low_val,
            "equilibrium": equilibrium,
        })

        return result

    def compute_support_resistance(
        self,
        df: pd.DataFrame,
        lookback: Optional[int] = None,
        tolerance: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """Compute automatic support and resistance zones using KDE.

        Uses kernel density estimation on high and low prices to find
        clusters of price activity, then classifies each as support,
        resistance, or both.

        Args:
            df: OHLCV DataFrame.
            lookback: Number of recent bars to use. Defaults to
                ``self.sr_lookback``.
            tolerance: Relative tolerance for clustering levels. Defaults to
                ``self.sr_cluster_tolerance``.

        Returns:
            List of S/R zone dicts with keys: level, strength (number of
            touches), type ('support', 'resistance', or 'both').
        """
        lookback = lookback if lookback is not None else self.sr_lookback
        tolerance = tolerance if tolerance is not None else self.sr_cluster_tolerance
        zones: list[dict[str, Any]] = []

        if len(df) == 0:
            return zones

        # Use most recent lookback bars
        data = df.tail(lookback)
        highs = data["high"].values
        lows = data["low"].values
        closes = data["close"].values

        # Combine highs and lows for KDE
        price_points = np.concatenate([highs, lows])
        price_points = price_points[~np.isnan(price_points)]

        if len(price_points) < 3:
            log_with_data(logger, "warning", "Insufficient data for S/R KDE", {
                "points": len(price_points),
            })
            return zones

        price_range = np.max(price_points) - np.min(price_points)
        if price_range <= 0:
            return zones

        try:
            kde = gaussian_kde(price_points, bw_method="silverman")
        except (np.linalg.LinAlgError, ValueError) as exc:
            log_with_data(logger, "error", "KDE computation failed", {
                "error": str(exc),
            })
            return zones

        # Evaluate KDE over a fine grid
        grid_size = 500
        grid = np.linspace(
            np.min(price_points) - price_range * 0.05,
            np.max(price_points) + price_range * 0.05,
            grid_size,
        )
        density = kde(grid)

        # Find peaks in the density (local maxima)
        peak_indices: list[int] = []
        for i in range(1, grid_size - 1):
            if density[i] > density[i - 1] and density[i] > density[i + 1]:
                peak_indices.append(i)

        if not peak_indices:
            log_with_data(logger, "info", "No S/R peaks found in KDE", {})
            return zones

        peak_levels = grid[peak_indices]

        # Cluster nearby peaks
        clustered_levels: list[float] = []
        used = set()
        for i, level in enumerate(peak_levels):
            if i in used:
                continue
            cluster = [level]
            for j in range(i + 1, len(peak_levels)):
                if j in used:
                    continue
                if level > 0 and abs(peak_levels[j] - level) / level <= tolerance:
                    cluster.append(peak_levels[j])
                    used.add(j)
            clustered_levels.append(float(np.mean(cluster)))

        # Determine strength (number of touches) and type for each level
        last_close = closes[-1] if len(closes) > 0 else 0.0

        for level in clustered_levels:
            if level <= 0:
                continue

            tol_abs = level * tolerance

            # Count touches: how many bars had their high or low within tolerance
            high_touches = int(np.sum(np.abs(highs - level) <= tol_abs))
            low_touches = int(np.sum(np.abs(lows - level) <= tol_abs))
            strength = high_touches + low_touches

            if strength == 0:
                continue

            # Classify based on position relative to current price
            if level < last_close * (1.0 - tolerance):
                zone_type = "support"
            elif level > last_close * (1.0 + tolerance):
                zone_type = "resistance"
            else:
                zone_type = "both"

            zones.append({
                "level": round(level, 8),
                "strength": strength,
                "type": zone_type,
            })

        # Sort by strength descending
        zones.sort(key=lambda z: z["strength"], reverse=True)

        log_with_data(logger, "info", "Support/resistance zones computed", {
            "total": len(zones),
            "support": sum(1 for z in zones if z["type"] == "support"),
            "resistance": sum(1 for z in zones if z["type"] == "resistance"),
            "both": sum(1 for z in zones if z["type"] == "both"),
        })

        return zones

    def analyze_all(self, df: pd.DataFrame) -> dict[str, Any]:
        """Run all SMC analysis and return a comprehensive results dict.

        Executes every detection method in sequence, passing intermediate
        results as needed.

        Args:
            df: OHLCV DataFrame with columns [timestamp, open, high, low,
                close, volume].

        Returns:
            Dict with keys: swing_points (DataFrame), order_blocks,
            fair_value_gaps, breaker_blocks, liquidity_sweeps,
            market_structure_breaks, change_of_character,
            premium_discount_zones (DataFrame), support_resistance,
            and summary statistics.
        """
        log_with_data(logger, "info", "Starting full SMC analysis", {
            "rows": len(df),
        })

        # 1. Swing points (foundation for many other analyses)
        swing_points_df = self.detect_swing_points(df, self.swing_lookback)

        # 2. Order blocks
        order_blocks = self.detect_order_blocks(df, self.ob_min_move_pct)

        # 3. Fair value gaps
        fair_value_gaps = self.detect_fair_value_gaps(df, self.fvg_min_size_pct)

        # 4. Breaker blocks (depends on order blocks)
        breaker_blocks = self.detect_breaker_blocks(df, order_blocks)

        # 5. Liquidity sweeps (depends on swing points)
        liquidity_sweeps = self.detect_liquidity_sweeps(df, swing_points_df)

        # 6. Market structure breaks (depends on swing points)
        msb_list = self.detect_market_structure_break(df, swing_points_df)

        # 7. Change of character (depends on MSBs)
        choch_list = self.detect_change_of_character(df, msb_list)

        # 8. Premium/discount zones (depends on swing points)
        pd_zones_df = self.compute_premium_discount_zones(df, swing_points_df)

        # 9. Support/resistance
        support_resistance = self.compute_support_resistance(
            df, self.sr_lookback, self.sr_cluster_tolerance
        )

        results: dict[str, Any] = {
            "swing_points": swing_points_df,
            "order_blocks": order_blocks,
            "fair_value_gaps": fair_value_gaps,
            "breaker_blocks": breaker_blocks,
            "liquidity_sweeps": liquidity_sweeps,
            "market_structure_breaks": msb_list,
            "change_of_character": choch_list,
            "premium_discount_zones": pd_zones_df,
            "support_resistance": support_resistance,
            "summary": {
                "total_bars": len(df),
                "swing_highs": int(swing_points_df["swing_high"].sum()),
                "swing_lows": int(swing_points_df["swing_low"].sum()),
                "order_blocks_total": len(order_blocks),
                "order_blocks_valid": sum(
                    1 for ob in order_blocks if ob["valid"]
                ),
                "fvg_total": len(fair_value_gaps),
                "fvg_unfilled": sum(
                    1 for f in fair_value_gaps if not f["filled"]
                ),
                "breaker_blocks": len(breaker_blocks),
                "liquidity_sweeps": len(liquidity_sweeps),
                "msb_total": len(msb_list),
                "choch_total": len(choch_list),
                "sr_zones": len(support_resistance),
            },
        }

        log_with_data(logger, "info", "Full SMC analysis complete", results["summary"])

        return results
