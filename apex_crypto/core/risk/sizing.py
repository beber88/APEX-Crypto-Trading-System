"""Position sizing engine for APEX Crypto Trading System.

Implements fixed-fraction sizing, half-Kelly criterion, correlation-based
adjustments, and hard caps to ensure no single trade risks more than the
configured threshold of portfolio equity.
"""

import logging
from typing import Any, Optional

from apex_crypto.core.logging import get_logger, log_with_data

logger = get_logger("risk.sizing")


class PositionSizer:
    """Calculates position sizes using fixed-fraction and Kelly methods.

    Applies hard caps and correlation guards to keep portfolio risk bounded.

    Attributes:
        risk_per_trade_pct: Default percentage of portfolio risked per trade.
        max_position_pct: Maximum single-position size as % of portfolio.
        max_asset_concentration_pct: Maximum exposure to a single asset.
        correlation_reduce_threshold: Pearson r above which size is cut.
        correlation_reduce_factor: Fraction to reduce by when correlated.
    """

    def __init__(self, config: dict) -> None:
        """Initialise the position sizer from the risk config section.

        Args:
            config: The ``risk`` section of config.yaml.
        """
        self.risk_per_trade_pct: float = config.get("risk_per_trade_pct", 1.0)
        self.max_position_pct: float = config.get("max_position_pct", 5.0)
        self.max_asset_concentration_pct: float = config.get(
            "max_asset_concentration_pct", 10.0
        )
        self.correlation_reduce_threshold: float = config.get(
            "correlation_reduce_threshold", 0.75
        )
        self.correlation_reduce_factor: float = config.get(
            "correlation_reduce_factor", 0.4
        )

        log_with_data(
            logger,
            "info",
            "PositionSizer initialised",
            {
                "risk_per_trade_pct": self.risk_per_trade_pct,
                "max_position_pct": self.max_position_pct,
                "max_asset_concentration_pct": self.max_asset_concentration_pct,
                "correlation_reduce_threshold": self.correlation_reduce_threshold,
                "correlation_reduce_factor": self.correlation_reduce_factor,
            },
        )

    # ------------------------------------------------------------------
    # Fixed-fraction sizing
    # ------------------------------------------------------------------

    def calculate_position_size(
        self,
        portfolio_value: float,
        entry_price: float,
        stop_loss: float,
        risk_pct: Optional[float] = None,
    ) -> dict[str, Any]:
        """Calculate position size using fixed-fraction risk model.

        The number of units is derived so that the dollar distance from
        *entry_price* to *stop_loss* times the unit count equals the
        maximum acceptable loss (``risk_pct`` of portfolio value).

        Args:
            portfolio_value: Current total portfolio value in USDT.
            entry_price: Planned entry price for the asset.
            stop_loss: Stop-loss price for the trade.
            risk_pct: Percentage of portfolio to risk.  Falls back to the
                configured ``risk_per_trade_pct`` when not supplied.

        Returns:
            Dictionary with keys ``size_units``, ``size_usdt``,
            ``risk_usdt``, and ``risk_pct_actual``.

        Raises:
            ValueError: If ``entry_price`` equals ``stop_loss`` (zero
                distance) or any monetary value is non-positive.
        """
        if portfolio_value <= 0:
            raise ValueError(
                f"portfolio_value must be positive, got {portfolio_value}"
            )
        if entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {entry_price}")
        if stop_loss <= 0:
            raise ValueError(f"stop_loss must be positive, got {stop_loss}")

        distance = abs(entry_price - stop_loss)
        if distance == 0:
            raise ValueError(
                "entry_price and stop_loss must differ to compute risk distance"
            )

        effective_risk_pct = risk_pct if risk_pct is not None else self.risk_per_trade_pct
        risk_usdt = portfolio_value * (effective_risk_pct / 100.0)
        size_units = risk_usdt / distance
        size_usdt = size_units * entry_price
        risk_pct_actual = (risk_usdt / portfolio_value) * 100.0

        result: dict[str, Any] = {
            "size_units": round(size_units, 8),
            "size_usdt": round(size_usdt, 4),
            "risk_usdt": round(risk_usdt, 4),
            "risk_pct_actual": round(risk_pct_actual, 4),
        }

        log_with_data(
            logger,
            "debug",
            "Fixed-fraction position size calculated",
            {
                "portfolio_value": portfolio_value,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "distance": distance,
                "effective_risk_pct": effective_risk_pct,
                **result,
            },
        )

        return result

    # ------------------------------------------------------------------
    # Kelly criterion sizing
    # ------------------------------------------------------------------

    def calculate_kelly_size(
        self,
        portfolio_value: float,
        entry_price: float,
        stop_loss: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> dict[str, Any]:
        """Calculate position size using the half-Kelly criterion.

        The full Kelly fraction is::

            f* = (p * b - q) / b

        where *p* = ``win_rate``, *b* = ``avg_win / avg_loss``, and
        *q* = 1 - p.  This method applies **half-Kelly** (``f* * 0.5``)
        to reduce variance.

        Args:
            portfolio_value: Current total portfolio value in USDT.
            entry_price: Planned entry price for the asset.
            stop_loss: Stop-loss price for the trade.
            win_rate: Historical win rate (0.0 – 1.0).
            avg_win: Average winning trade return (absolute value).
            avg_loss: Average losing trade return (absolute value,
                expressed as a positive number).

        Returns:
            Dictionary with ``kelly_fraction``, ``half_kelly_fraction``,
            ``size_units``, and ``size_usdt``.

        Raises:
            ValueError: If ``avg_loss`` is zero, ``win_rate`` is outside
                [0, 1], or monetary values are non-positive.
        """
        if portfolio_value <= 0:
            raise ValueError(
                f"portfolio_value must be positive, got {portfolio_value}"
            )
        if entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {entry_price}")
        if stop_loss <= 0:
            raise ValueError(f"stop_loss must be positive, got {stop_loss}")
        if not 0.0 <= win_rate <= 1.0:
            raise ValueError(f"win_rate must be between 0 and 1, got {win_rate}")
        if avg_loss <= 0:
            raise ValueError(f"avg_loss must be positive, got {avg_loss}")
        if avg_win <= 0:
            raise ValueError(f"avg_win must be positive, got {avg_win}")

        p = win_rate
        q = 1.0 - p
        b = avg_win / avg_loss

        kelly_fraction = (p * b - q) / b
        # Clamp to [0, 1] — a negative Kelly means the edge is negative
        kelly_fraction = max(0.0, min(kelly_fraction, 1.0))
        half_kelly_fraction = kelly_fraction * 0.5

        risk_usdt = portfolio_value * half_kelly_fraction
        distance = abs(entry_price - stop_loss)

        if distance == 0:
            raise ValueError(
                "entry_price and stop_loss must differ to compute risk distance"
            )

        size_units = risk_usdt / distance
        size_usdt = size_units * entry_price

        result: dict[str, Any] = {
            "kelly_fraction": round(kelly_fraction, 6),
            "half_kelly_fraction": round(half_kelly_fraction, 6),
            "size_units": round(size_units, 8),
            "size_usdt": round(size_usdt, 4),
        }

        log_with_data(
            logger,
            "debug",
            "Kelly position size calculated",
            {
                "portfolio_value": portfolio_value,
                "win_rate": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "b_ratio": round(b, 4),
                **result,
            },
        )

        return result

    # ------------------------------------------------------------------
    # Final composite sizing
    # ------------------------------------------------------------------

    def final_position_size(
        self,
        portfolio_value: float,
        entry_price: float,
        stop_loss: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        signal_strength: str,
    ) -> dict[str, Any]:
        """Determine the final position size after all adjustments.

        Procedure:
        1. Compute fixed-fraction size.
        2. Compute half-Kelly size.
        3. Take the **minimum** of the two to be conservative.
        4. If ``signal_strength`` is ``"half"``, halve the size.
        5. Cap at ``max_position_pct`` (5 %) of portfolio per position.
        6. Cap at ``max_asset_concentration_pct`` (10 %) of portfolio
           per asset (applied downstream via ``check_correlation_guard``).

        Args:
            portfolio_value: Current total portfolio value in USDT.
            entry_price: Planned entry price for the asset.
            stop_loss: Stop-loss price for the trade.
            win_rate: Historical win rate (0.0 – 1.0).
            avg_win: Average winning trade return (positive float).
            avg_loss: Average losing trade return (positive float).
            signal_strength: ``"full"`` for a full-conviction signal,
                ``"half"`` for a reduced-conviction signal.

        Returns:
            Comprehensive sizing dictionary with keys:
            ``method``, ``size_units``, ``size_usdt``, ``risk_usdt``,
            ``risk_pct_actual``, ``kelly_fraction``,
            ``half_kelly_fraction``, ``signal_strength``,
            ``capped``, ``cap_reason``.
        """
        fixed = self.calculate_position_size(
            portfolio_value, entry_price, stop_loss
        )
        kelly = self.calculate_kelly_size(
            portfolio_value, entry_price, stop_loss, win_rate, avg_win, avg_loss
        )

        # Take the more conservative of the two methods
        if kelly["size_usdt"] <= fixed["size_usdt"]:
            chosen_method = "half_kelly"
            size_units = kelly["size_units"]
            size_usdt = kelly["size_usdt"]
        else:
            chosen_method = "fixed_fraction"
            size_units = fixed["size_units"]
            size_usdt = fixed["size_usdt"]

        # Halve for reduced-conviction signals
        if signal_strength == "half":
            size_units *= 0.5
            size_usdt *= 0.5

        # Apply hard cap: max_position_pct of portfolio
        cap_reason: Optional[str] = None
        capped = False
        max_position_usdt = portfolio_value * (self.max_position_pct / 100.0)
        if size_usdt > max_position_usdt:
            size_units = size_units * (max_position_usdt / size_usdt)
            size_usdt = max_position_usdt
            capped = True
            cap_reason = f"max_position_pct ({self.max_position_pct}%)"

        # Apply hard cap: max_asset_concentration_pct of portfolio
        max_asset_usdt = portfolio_value * (self.max_asset_concentration_pct / 100.0)
        if size_usdt > max_asset_usdt:
            size_units = size_units * (max_asset_usdt / size_usdt)
            size_usdt = max_asset_usdt
            capped = True
            cap_reason = f"max_asset_concentration_pct ({self.max_asset_concentration_pct}%)"

        distance = abs(entry_price - stop_loss)
        risk_usdt = size_units * distance
        risk_pct_actual = (risk_usdt / portfolio_value) * 100.0 if portfolio_value > 0 else 0.0

        result: dict[str, Any] = {
            "method": chosen_method,
            "size_units": round(size_units, 8),
            "size_usdt": round(size_usdt, 4),
            "risk_usdt": round(risk_usdt, 4),
            "risk_pct_actual": round(risk_pct_actual, 4),
            "kelly_fraction": kelly["kelly_fraction"],
            "half_kelly_fraction": kelly["half_kelly_fraction"],
            "signal_strength": signal_strength,
            "capped": capped,
            "cap_reason": cap_reason,
        }

        log_with_data(
            logger,
            "info",
            "Final position size determined",
            {
                "portfolio_value": portfolio_value,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "win_rate": win_rate,
                "fixed_size_usdt": fixed["size_usdt"],
                "kelly_size_usdt": kelly["size_usdt"],
                **result,
            },
        )

        return result

    # ------------------------------------------------------------------
    # Correlation guard
    # ------------------------------------------------------------------

    def check_correlation_guard(
        self,
        symbol: str,
        open_positions: list[dict[str, Any]],
        correlation_matrix: dict[str, dict[str, float]],
    ) -> float:
        """Check if a new position is highly correlated with existing ones.

        If the candidate ``symbol`` has a Pearson correlation coefficient
        greater than ``correlation_reduce_threshold`` (default 0.75) with
        **any** currently open position, the returned multiplier reduces
        the position size by ``correlation_reduce_factor`` (default 40 %).

        Args:
            symbol: Ticker of the asset being considered (e.g.
                ``"BTC/USDT"``).
            open_positions: List of dicts, each containing at least a
                ``"symbol"`` key for the position's ticker.
            correlation_matrix: Nested dict mapping
                ``symbol -> symbol -> float`` correlation.

        Returns:
            Size adjustment multiplier — ``1.0`` if no high-correlation
            conflict, or ``1.0 - correlation_reduce_factor`` (``0.6``)
            if a conflict is found.
        """
        if not open_positions or not correlation_matrix:
            log_with_data(
                logger,
                "debug",
                "Correlation guard: no open positions or empty matrix",
                {"symbol": symbol},
            )
            return 1.0

        symbol_correlations = correlation_matrix.get(symbol)
        if symbol_correlations is None:
            log_with_data(
                logger,
                "debug",
                "Correlation guard: symbol not in correlation matrix",
                {"symbol": symbol},
            )
            return 1.0

        for position in open_positions:
            pos_symbol = position.get("symbol", "")
            correlation = symbol_correlations.get(pos_symbol)

            if correlation is not None and abs(correlation) > self.correlation_reduce_threshold:
                multiplier = 1.0 - self.correlation_reduce_factor
                log_with_data(
                    logger,
                    "warning",
                    "Correlation guard triggered — reducing position size",
                    {
                        "new_symbol": symbol,
                        "correlated_symbol": pos_symbol,
                        "correlation": round(correlation, 4),
                        "threshold": self.correlation_reduce_threshold,
                        "size_multiplier": multiplier,
                    },
                )
                return multiplier

        log_with_data(
            logger,
            "debug",
            "Correlation guard: no high-correlation conflicts",
            {"symbol": symbol, "open_position_count": len(open_positions)},
        )
        return 1.0
