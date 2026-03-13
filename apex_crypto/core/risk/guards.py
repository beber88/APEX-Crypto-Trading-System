"""Market-condition guard rails for APEX Crypto Trading System.

Detects flash crashes, liquidation cascades, extreme funding rates,
wide spreads, thin liquidity, and regime-specific restrictions.  Each
guard returns structured results so the trading engine can decide
whether to block, reduce, or proceed with a trade.
"""

import time
import logging
from typing import Any, Optional

from apex_crypto.core.logging import get_logger, log_with_data

logger = get_logger("risk.guards")


class RiskGuards:
    """Real-time market-condition checks that protect against adverse events.

    Guards are lightweight, stateless checks (except for flash-crash
    timing) that run on every trade signal before order submission.

    Attributes:
        flash_crash_pct: Percentage drop in 2 minutes that triggers halt.
        flash_crash_halt_minutes: Minutes to halt after a flash crash.
        min_spread_pct: Maximum acceptable bid-ask spread as % of price.
        funding_rate_long_halt: Funding rate above which longs are closed.
        min_daily_volume_usd: Minimum 24 h volume for any traded asset.
        tier1_assets: List of Tier 1 asset tickers.
    """

    def __init__(self, config: dict) -> None:
        """Initialise risk guards from the full config dict.

        Args:
            config: The full system config (needs ``risk`` and ``assets``
                sections).
        """
        risk_cfg = config.get("risk", config)
        assets_cfg = config.get("assets", {})

        self.flash_crash_pct: float = risk_cfg.get("flash_crash_pct", 6.0)
        self.flash_crash_halt_minutes: int = int(
            risk_cfg.get("flash_crash_halt_minutes", 30)
        )
        self.min_spread_pct: float = risk_cfg.get("min_spread_pct", 0.15)
        self.funding_rate_long_halt: float = risk_cfg.get(
            "funding_rate_long_halt", 0.003
        )
        self.min_daily_volume_usd: float = risk_cfg.get(
            "min_daily_volume_usd",
            assets_cfg.get("min_daily_volume_usd", 8_000_000),
        )
        self.max_leverage: float = risk_cfg.get("max_leverage", 3.0)
        self.tier1_assets: list[str] = assets_cfg.get(
            "tier1",
            ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"],
        )

        # Rule 30 (Goodman): Black Swan kill switch
        self.kill_switch_drop_pct: float = risk_cfg.get("kill_switch_drop_pct", 20.0)
        self.kill_switch_window_minutes: int = int(
            risk_cfg.get("kill_switch_window_minutes", 60)
        )
        # Rule 15 (Eugene Ng): Break-even stop trigger
        self.breakeven_trigger_pct: float = risk_cfg.get("breakeven_trigger_pct", 1.5)
        # Rule 10 (GCR): Block shorts on low-cap coins
        self.min_short_market_cap_usd: float = risk_cfg.get(
            "min_short_market_cap_usd", 500_000_000
        )
        # Rule 7 (GCR): Volume vs market cap peak alert
        self.volume_mcap_ratio_alert: float = risk_cfg.get(
            "volume_mcap_ratio_alert", 1.0
        )

        # Flash-crash halt state
        self._flash_crash_halt_until: float = 0.0
        # Kill switch state
        self._kill_switch_triggered: bool = False

        log_with_data(
            logger,
            "info",
            "RiskGuards initialised",
            {
                "flash_crash_pct": self.flash_crash_pct,
                "flash_crash_halt_minutes": self.flash_crash_halt_minutes,
                "min_spread_pct": self.min_spread_pct,
                "funding_rate_long_halt": self.funding_rate_long_halt,
                "min_daily_volume_usd": self.min_daily_volume_usd,
                "tier1_count": len(self.tier1_assets),
            },
        )

    # ------------------------------------------------------------------
    # Individual guard checks
    # ------------------------------------------------------------------

    def check_flash_crash(
        self,
        current_price: float,
        price_2min_ago: float,
    ) -> tuple[bool, str]:
        """Detect a flash crash and halt new entries for a cooling period.

        A flash crash is defined as a price drop exceeding
        ``flash_crash_pct`` (default 6 %) within a 2-minute window.
        When triggered, new entries are blocked for
        ``flash_crash_halt_minutes`` (default 30 min).

        Args:
            current_price: Latest price of the asset.
            price_2min_ago: Price of the asset 2 minutes prior.

        Returns:
            ``(is_safe, message)``.  ``is_safe`` is ``False`` if a
            crash was just detected **or** a previous halt is still
            active.
        """
        now = time.time()

        # Check if a previous halt is still active
        if now < self._flash_crash_halt_until:
            remaining = (self._flash_crash_halt_until - now) / 60.0
            msg = (
                f"FLASH CRASH HALT ACTIVE: {remaining:.1f} min remaining "
                f"— no new entries"
            )
            log_with_data(
                logger,
                "warning",
                msg,
                {"remaining_minutes": round(remaining, 1)},
            )
            return False, msg

        if price_2min_ago <= 0:
            return True, "Cannot assess flash crash — previous price unavailable"

        drop_pct = ((price_2min_ago - current_price) / price_2min_ago) * 100.0

        if drop_pct >= self.flash_crash_pct:
            self._flash_crash_halt_until = now + (self.flash_crash_halt_minutes * 60)
            msg = (
                f"FLASH CRASH DETECTED: price dropped {drop_pct:.2f}% in 2 min "
                f"(threshold {self.flash_crash_pct:.1f}%) — halting entries for "
                f"{self.flash_crash_halt_minutes} min"
            )
            log_with_data(
                logger,
                "critical",
                msg,
                {
                    "current_price": current_price,
                    "price_2min_ago": price_2min_ago,
                    "drop_pct": round(drop_pct, 4),
                    "threshold_pct": self.flash_crash_pct,
                    "halt_until": self._flash_crash_halt_until,
                },
            )
            return False, msg

        msg = f"No flash crash detected (drop {drop_pct:.2f}%)"
        log_with_data(logger, "debug", msg, {"drop_pct": round(drop_pct, 4)})
        return True, msg

    def check_liquidation_cascade(
        self,
        liquidation_volume: float,
        avg_liquidation_volume: float,
    ) -> tuple[bool, float]:
        """Detect abnormal liquidation volume and reduce position size.

        When ``liquidation_volume`` exceeds twice the average, position
        sizes are reduced by 50 %.

        Args:
            liquidation_volume: Current liquidation volume in USDT over
                the recent window.
            avg_liquidation_volume: Average liquidation volume for the
                same window length.

        Returns:
            ``(is_safe, size_multiplier)``.  ``is_safe`` is ``False``
            (with ``size_multiplier = 0.5``) when a cascade is detected.
        """
        if avg_liquidation_volume <= 0:
            log_with_data(
                logger,
                "debug",
                "Liquidation cascade check skipped — no average data",
                {"liquidation_volume": liquidation_volume},
            )
            return True, 1.0

        ratio = liquidation_volume / avg_liquidation_volume

        if ratio >= 2.0:
            msg = (
                f"LIQUIDATION CASCADE: volume {liquidation_volume:,.0f} USDT "
                f"is {ratio:.1f}x average — reducing size by 50%"
            )
            log_with_data(
                logger,
                "warning",
                msg,
                {
                    "liquidation_volume": liquidation_volume,
                    "avg_liquidation_volume": avg_liquidation_volume,
                    "ratio": round(ratio, 2),
                    "size_multiplier": 0.5,
                },
            )
            return False, 0.5

        log_with_data(
            logger,
            "debug",
            "Liquidation volume normal",
            {"ratio": round(ratio, 2)},
        )
        return True, 1.0

    def check_funding_rate_extreme(
        self,
        funding_rate: float,
        direction: str,
    ) -> tuple[bool, str]:
        """Check for extreme funding rates that warrant closing longs.

        If ``funding_rate`` exceeds ``funding_rate_long_halt`` (default
        0.3 %) and the position direction is ``"long"``, the position
        should be closed.

        Args:
            funding_rate: Current funding rate as a decimal (e.g. 0.003
                for 0.3 %).
            direction: Position direction — ``"long"`` or ``"short"``.

        Returns:
            ``(is_safe, message)``.  ``is_safe`` is ``False`` when
            funding is extreme for longs.
        """
        if funding_rate > self.funding_rate_long_halt and direction.lower() == "long":
            msg = (
                f"EXTREME FUNDING RATE: {funding_rate * 100:.3f}% "
                f"(threshold {self.funding_rate_long_halt * 100:.3f}%) "
                f"— close long positions"
            )
            log_with_data(
                logger,
                "warning",
                msg,
                {
                    "funding_rate": funding_rate,
                    "funding_rate_pct": round(funding_rate * 100, 4),
                    "threshold": self.funding_rate_long_halt,
                    "direction": direction,
                },
            )
            return False, msg

        msg = (
            f"Funding rate acceptable: {funding_rate * 100:.3f}% "
            f"for {direction} position"
        )
        log_with_data(
            logger,
            "debug",
            msg,
            {"funding_rate": funding_rate, "direction": direction},
        )
        return True, msg

    def check_spread(
        self,
        bid: float,
        ask: float,
    ) -> tuple[bool, str]:
        """Check whether the bid-ask spread is acceptable for trading.

        The trade is skipped if the spread exceeds ``min_spread_pct``
        (default 0.15 %) of the mid-price.

        Args:
            bid: Current best bid price.
            ask: Current best ask price.

        Returns:
            ``(is_acceptable, message)``.
        """
        if bid <= 0 or ask <= 0:
            return False, "Invalid bid/ask prices — cannot assess spread"

        if ask < bid:
            return False, f"Crossed book: ask ({ask}) < bid ({bid}) — skipping trade"

        mid_price = (bid + ask) / 2.0
        spread = ask - bid
        spread_pct = (spread / mid_price) * 100.0

        if spread_pct > self.min_spread_pct:
            msg = (
                f"SPREAD TOO WIDE: {spread_pct:.4f}% "
                f"(limit {self.min_spread_pct:.2f}%) — skipping trade"
            )
            log_with_data(
                logger,
                "warning",
                msg,
                {
                    "bid": bid,
                    "ask": ask,
                    "mid_price": mid_price,
                    "spread": spread,
                    "spread_pct": round(spread_pct, 4),
                    "limit_pct": self.min_spread_pct,
                },
            )
            return False, msg

        msg = f"Spread acceptable: {spread_pct:.4f}% (limit {self.min_spread_pct:.2f}%)"
        log_with_data(logger, "debug", msg, {"spread_pct": round(spread_pct, 4)})
        return True, msg

    def check_liquidity(
        self,
        volume_24h: float,
    ) -> tuple[bool, str]:
        """Check that the asset has sufficient 24-hour trading volume.

        Assets with volume below ``min_daily_volume_usd`` (default
        $8 M) are rejected.

        Args:
            volume_24h: 24-hour trading volume in USD.

        Returns:
            ``(is_liquid, message)``.
        """
        if volume_24h >= self.min_daily_volume_usd:
            msg = (
                f"Liquidity sufficient: ${volume_24h:,.0f} "
                f"(min ${self.min_daily_volume_usd:,.0f})"
            )
            log_with_data(
                logger,
                "debug",
                msg,
                {"volume_24h": volume_24h, "min_required": self.min_daily_volume_usd},
            )
            return True, msg

        msg = (
            f"INSUFFICIENT LIQUIDITY: ${volume_24h:,.0f} 24h volume "
            f"(min ${self.min_daily_volume_usd:,.0f}) — skipping"
        )
        log_with_data(
            logger,
            "warning",
            msg,
            {"volume_24h": volume_24h, "min_required": self.min_daily_volume_usd},
        )
        return False, msg

    def check_regime_restrictions(
        self,
        regime: str,
    ) -> dict[str, Any]:
        """Return trading restrictions for the current market regime.

        In the ``CHAOS`` regime:
        - Max leverage is forced to 1x.
        - Position sizes are cut by 50 %.
        - Only Tier 1 assets are permitted.

        For all other regimes, no additional restrictions apply.

        Args:
            regime: Current market regime label (e.g. ``"TRENDING"``,
                ``"RANGING"``, ``"VOLATILE"``, ``"CHAOS"``).

        Returns:
            Dictionary with ``max_leverage`` (float),
            ``size_multiplier`` (float), and ``allowed_tiers``
            (list of str).
        """
        if regime.upper() == "CHAOS":
            restrictions: dict[str, Any] = {
                "max_leverage": 1.0,
                "size_multiplier": 0.5,
                "allowed_tiers": ["tier1"],
            }
            log_with_data(
                logger,
                "warning",
                "CHAOS regime — applying maximum restrictions",
                restrictions,
            )
            return restrictions

        restrictions = {
            "max_leverage": self.max_leverage,
            "size_multiplier": 1.0,
            "allowed_tiers": ["tier1", "tier2"],
        }
        log_with_data(
            logger,
            "debug",
            f"Regime {regime} — standard restrictions",
            restrictions,
        )
        return restrictions

    # ------------------------------------------------------------------
    # Rule 30: Black Swan Kill Switch (Goodman)
    # ------------------------------------------------------------------

    def check_kill_switch(
        self,
        btc_price_now: float,
        btc_price_1h_ago: float,
    ) -> tuple[bool, str]:
        """Detect a broad market crash and trigger emergency close-all.

        If BTC drops more than ``kill_switch_drop_pct`` (default 20%) in
        the configured window (default 60 min), the kill switch fires.
        All positions should be closed and converted to stablecoin.

        Args:
            btc_price_now: Current BTC/USDT price.
            btc_price_1h_ago: BTC/USDT price 1 hour ago.

        Returns:
            ``(is_safe, message)``.  ``is_safe`` is ``False`` when the
            kill switch has been triggered.
        """
        if self._kill_switch_triggered:
            return False, "KILL SWITCH ACTIVE — all trading halted, close all positions"

        if btc_price_1h_ago <= 0:
            return True, "Cannot assess kill switch — no historical BTC price"

        drop_pct = ((btc_price_1h_ago - btc_price_now) / btc_price_1h_ago) * 100.0

        if drop_pct >= self.kill_switch_drop_pct:
            self._kill_switch_triggered = True
            msg = (
                f"BLACK SWAN KILL SWITCH: BTC dropped {drop_pct:.2f}% in "
                f"{self.kill_switch_window_minutes} min "
                f"(threshold {self.kill_switch_drop_pct:.1f}%) — "
                f"EMERGENCY: close ALL positions, convert to USDC"
            )
            log_with_data(
                logger,
                "critical",
                msg,
                {
                    "btc_price_now": btc_price_now,
                    "btc_price_1h_ago": btc_price_1h_ago,
                    "drop_pct": round(drop_pct, 4),
                    "threshold_pct": self.kill_switch_drop_pct,
                },
            )
            return False, msg

        return True, f"Kill switch safe (BTC drop {drop_pct:.2f}%)"

    @property
    def is_kill_switch_triggered(self) -> bool:
        """Whether the kill switch has been triggered."""
        return self._kill_switch_triggered

    def reset_kill_switch(self) -> None:
        """Manually reset the kill switch after operator review."""
        self._kill_switch_triggered = False
        log_with_data(logger, "warning", "Kill switch manually reset by operator", {})

    # ------------------------------------------------------------------
    # Rule 10: Block shorts on low-cap coins (GCR)
    # ------------------------------------------------------------------

    def check_short_market_cap(
        self,
        market_cap_usd: float,
        direction: str,
    ) -> tuple[bool, str]:
        """Block short positions on coins with low market cap.

        Low-cap coins have extreme volatility that can liquidate short
        positions instantly. Only allow shorts on coins with market cap
        above ``min_short_market_cap_usd`` (default $500M).

        Args:
            market_cap_usd: Market capitalisation in USD.
            direction: ``"long"`` or ``"short"``.

        Returns:
            ``(is_safe, message)``.
        """
        if direction.lower() != "short":
            return True, "Long position — no market cap restriction"

        if market_cap_usd <= 0:
            # If we can't determine market cap, block shorts on unknown coins
            msg = "SHORT BLOCKED: unknown market cap — cannot short unverified assets"
            log_with_data(logger, "warning", msg, {"direction": direction})
            return False, msg

        if market_cap_usd < self.min_short_market_cap_usd:
            msg = (
                f"SHORT BLOCKED on low-cap: market cap ${market_cap_usd:,.0f} "
                f"below ${self.min_short_market_cap_usd:,.0f} minimum"
            )
            log_with_data(
                logger,
                "warning",
                msg,
                {
                    "market_cap_usd": market_cap_usd,
                    "min_required": self.min_short_market_cap_usd,
                },
            )
            return False, msg

        return True, f"Market cap ${market_cap_usd:,.0f} sufficient for short"

    # ------------------------------------------------------------------
    # Rule 7: Volume vs Market Cap peak alert (GCR)
    # ------------------------------------------------------------------

    def check_volume_mcap_ratio(
        self,
        volume_24h: float,
        market_cap_usd: float,
    ) -> tuple[bool, str]:
        """Alert when 24h trading volume exceeds market cap.

        When volume surpasses the total market cap, it signals a potential
        market top — the asset is likely being heavily traded by speculators
        and a reversal is imminent.

        Args:
            volume_24h: 24-hour trading volume in USD.
            market_cap_usd: Market capitalisation in USD.

        Returns:
            ``(is_warning, message)``.  ``is_warning`` is ``True`` when
            the ratio exceeds the threshold (peak signal detected).
        """
        if market_cap_usd <= 0:
            return False, "Cannot assess volume/mcap ratio — no market cap data"

        ratio = volume_24h / market_cap_usd

        if ratio >= self.volume_mcap_ratio_alert:
            msg = (
                f"PEAK WARNING: 24h volume ${volume_24h:,.0f} is {ratio:.2f}x "
                f"market cap ${market_cap_usd:,.0f} — potential top, look for exit"
            )
            log_with_data(
                logger,
                "warning",
                msg,
                {
                    "volume_24h": volume_24h,
                    "market_cap_usd": market_cap_usd,
                    "ratio": round(ratio, 4),
                    "threshold": self.volume_mcap_ratio_alert,
                },
            )
            return True, msg

        return False, f"Volume/mcap ratio {ratio:.2f} within normal range"

    # ------------------------------------------------------------------
    # Rule 15: Break-even stop check (Eugene Ng)
    # ------------------------------------------------------------------

    def should_move_to_breakeven(
        self,
        entry_price: float,
        current_price: float,
        direction: str,
    ) -> bool:
        """Check if a position's stop loss should be moved to break-even.

        After price moves ``breakeven_trigger_pct`` in the profitable
        direction, the stop loss should be moved to entry price to
        protect capital.

        Args:
            entry_price: Original entry price.
            current_price: Current market price.
            direction: ``"long"`` or ``"short"``.

        Returns:
            ``True`` if stop should be moved to break-even.
        """
        if entry_price <= 0:
            return False

        if direction.lower() == "long":
            profit_pct = ((current_price - entry_price) / entry_price) * 100.0
        elif direction.lower() == "short":
            profit_pct = ((entry_price - current_price) / entry_price) * 100.0
        else:
            return False

        should_move = profit_pct >= self.breakeven_trigger_pct

        if should_move:
            log_with_data(
                logger,
                "info",
                "Break-even stop triggered",
                {
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "direction": direction,
                    "profit_pct": round(profit_pct, 4),
                    "trigger_pct": self.breakeven_trigger_pct,
                },
            )

        return should_move

    # ------------------------------------------------------------------
    # Aggregate guard runner
    # ------------------------------------------------------------------

    def run_all_guards(
        self,
        market_data: dict[str, Any],
        regime: str,
    ) -> dict[str, Any]:
        """Run every guard check and return a combined result.

        This is the main entry point for the trading engine before
        placing any order.

        Args:
            market_data: Dictionary containing market state.  Expected
                keys:

                - ``current_price`` (float)
                - ``price_2min_ago`` (float)
                - ``liquidation_volume`` (float)
                - ``avg_liquidation_volume`` (float)
                - ``funding_rate`` (float)
                - ``direction`` (str): ``"long"`` or ``"short"``
                - ``bid`` (float)
                - ``ask`` (float)
                - ``volume_24h`` (float)

            regime: Current market regime label.

        Returns:
            Dictionary with:

            - ``can_trade`` (bool): ``False`` if any blocking guard fired.
            - ``size_multiplier`` (float): Combined multiplier from all
              size-reducing guards.
            - ``max_leverage`` (float): Effective leverage cap.
            - ``warnings`` (list[str]): Non-blocking advisory messages.
            - ``blocks`` (list[str]): Messages for guards that blocked
              the trade outright.
        """
        warnings: list[str] = []
        blocks: list[str] = []
        size_multiplier: float = 1.0

        # 1. Flash crash
        is_safe, msg = self.check_flash_crash(
            market_data.get("current_price", 0.0),
            market_data.get("price_2min_ago", 0.0),
        )
        if not is_safe:
            blocks.append(msg)

        # 2. Liquidation cascade
        liq_safe, liq_mult = self.check_liquidation_cascade(
            market_data.get("liquidation_volume", 0.0),
            market_data.get("avg_liquidation_volume", 0.0),
        )
        if not liq_safe:
            size_multiplier *= liq_mult
            warnings.append(
                f"Liquidation cascade — size reduced to {liq_mult * 100:.0f}%"
            )

        # 3. Funding rate
        direction = market_data.get("direction", "long")
        fund_safe, fund_msg = self.check_funding_rate_extreme(
            market_data.get("funding_rate", 0.0),
            direction,
        )
        if not fund_safe:
            blocks.append(fund_msg)

        # 4. Spread
        spread_ok, spread_msg = self.check_spread(
            market_data.get("bid", 0.0),
            market_data.get("ask", 0.0),
        )
        if not spread_ok:
            blocks.append(spread_msg)

        # 5. Liquidity
        liq_ok, liq_msg = self.check_liquidity(
            market_data.get("volume_24h", 0.0),
        )
        if not liq_ok:
            blocks.append(liq_msg)

        # 6. Regime restrictions
        regime_restrictions = self.check_regime_restrictions(regime)
        size_multiplier *= regime_restrictions["size_multiplier"]
        max_leverage = regime_restrictions["max_leverage"]

        if regime.upper() == "CHAOS":
            warnings.append(
                "CHAOS regime: leverage capped at 1x, size halved, Tier 1 only"
            )

        # 7. Kill switch (Rule 30)
        ks_safe, ks_msg = self.check_kill_switch(
            market_data.get("btc_price_now", 0.0),
            market_data.get("btc_price_1h_ago", 0.0),
        )
        if not ks_safe:
            blocks.append(ks_msg)

        # 8. Short market cap block (Rule 10)
        mcap = market_data.get("market_cap_usd", 0.0)
        if mcap > 0:
            mcap_safe, mcap_msg = self.check_short_market_cap(
                mcap, direction,
            )
            if not mcap_safe:
                blocks.append(mcap_msg)

        # 9. Volume vs market cap peak warning (Rule 7)
        if mcap > 0:
            is_peak, peak_msg = self.check_volume_mcap_ratio(
                market_data.get("volume_24h", 0.0), mcap,
            )
            if is_peak:
                warnings.append(peak_msg)

        can_trade = len(blocks) == 0

        result: dict[str, Any] = {
            "can_trade": can_trade,
            "size_multiplier": round(size_multiplier, 4),
            "max_leverage": max_leverage,
            "allowed_tiers": regime_restrictions["allowed_tiers"],
            "warnings": warnings,
            "blocks": blocks,
            "kill_switch": not ks_safe,
        }

        log_with_data(
            logger,
            "info" if can_trade else "warning",
            "All guards evaluated",
            {
                "can_trade": can_trade,
                "size_multiplier": result["size_multiplier"],
                "max_leverage": max_leverage,
                "warning_count": len(warnings),
                "block_count": len(blocks),
                "regime": regime,
            },
        )

        return result
