"""Weekly PDF report generator for the APEX Crypto Trading System.

Produces weekly summary reports covering 7-day P&L trends, strategy
win-rate evolution, top-performing assets, regime breakdowns, and
parameter review suggestions.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.charts.barcharts import VerticalBarChart

from apex_crypto.core.logging import get_logger, log_with_data

logger = get_logger("reporting.weekly")

# ---------------------------------------------------------------------------
# Colour palette (consistent with daily report)
# ---------------------------------------------------------------------------
_HEADER_BG = colors.HexColor("#1a1a2e")
_HEADER_FG = colors.HexColor("#e6e6e6")
_PROFIT_GREEN = colors.HexColor("#00c853")
_LOSS_RED = colors.HexColor("#ff1744")
_TABLE_HEADER_BG = colors.HexColor("#16213e")
_TABLE_ALT_ROW = colors.HexColor("#f5f5f5")
_BORDER_GREY = colors.HexColor("#cccccc")
_CHART_BLUE = colors.HexColor("#2196f3")
_CHART_ORANGE = colors.HexColor("#ff9800")
_CHART_TEAL = colors.HexColor("#009688")
_WARNING_AMBER = colors.HexColor("#ff8f00")

_PAGE_WIDTH, _PAGE_HEIGHT = letter
_LEFT_MARGIN = 0.75 * inch
_RIGHT_MARGIN = 0.75 * inch
_TOP_MARGIN = 0.75 * inch
_BOTTOM_MARGIN = 0.75 * inch
_USABLE_WIDTH = _PAGE_WIDTH - _LEFT_MARGIN - _RIGHT_MARGIN

# Win-rate decline threshold triggering a parameter review suggestion
_WIN_RATE_DECLINE_THRESHOLD = 5.0  # percentage points


class WeeklyReportGenerator:
    """Generates weekly PDF trading summary reports.

    Covers 7-day P&L curves, strategy win-rate trends, top-performing
    assets, regime breakdowns, and parameter review suggestions when
    strategy performance is declining.

    Attributes:
        output_dir: Directory where generated PDFs are saved.
    """

    def __init__(self, config: dict) -> None:
        """Initialize the weekly report generator.

        Args:
            config: Full application configuration dictionary.  The
                ``reporting.output_dir`` key determines PDF output
                location.
        """
        reporting_cfg = config.get("reporting", {})
        self.output_dir: str = reporting_cfg.get("output_dir", "./reports")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        log_with_data(
            logger,
            "info",
            "WeeklyReportGenerator initialised",
            {"output_dir": self.output_dir},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        week_start: str,
        week_end: str,
        stats: dict,
        daily_equity: list,
        strategy_stats: dict,
    ) -> str:
        """Generate a complete weekly PDF report.

        Args:
            week_start: Start date of the reporting week (e.g.
                ``"2026-03-02"``).
            week_end: End date of the reporting week (e.g.
                ``"2026-03-08"``).
            stats: Aggregate statistics dictionary with keys such as
                ``total_pnl``, ``total_trades``, ``wins``, ``losses``,
                ``avg_daily_pnl``, ``best_day``, ``worst_day``,
                ``sharpe``, ``max_drawdown``, ``asset_performance``
                (list of dicts with ``symbol``, ``pnl``, ``trades``),
                ``regimes`` (dict mapping regime names to percentage of
                time spent).
            daily_equity: List of daily equity snapshots (one per day,
                up to 7 entries).
            strategy_stats: Dictionary mapping strategy names to stat
                dicts.  Each dict should contain ``win_rate``,
                ``prev_win_rate``, ``avg_r``, ``total_trades``,
                ``wins``, ``losses``, ``total_pnl``.

        Returns:
            Absolute file path of the generated PDF.
        """
        filename = f"apex_weekly_report_{week_start}_to_{week_end}.pdf"
        filepath = os.path.join(self.output_dir, filename)

        doc = SimpleDocTemplate(
            filepath,
            pagesize=letter,
            leftMargin=_LEFT_MARGIN,
            rightMargin=_RIGHT_MARGIN,
            topMargin=_TOP_MARGIN,
            bottomMargin=_BOTTOM_MARGIN,
        )

        styles = getSampleStyleSheet()
        elements: list[Any] = []

        # Header
        elements.append(self._build_header(week_start, week_end, styles))
        elements.append(Spacer(1, 0.3 * inch))

        # Weekly summary stats
        elements.append(Paragraph("Weekly Summary", styles["Heading2"]))
        elements.append(Spacer(1, 0.1 * inch))
        elements.append(self._build_summary_table(stats, styles))
        elements.append(Spacer(1, 0.25 * inch))

        # 7-day P&L curve
        if daily_equity and len(daily_equity) >= 2:
            elements.append(Paragraph("7-Day Equity Curve", styles["Heading2"]))
            elements.append(Spacer(1, 0.1 * inch))
            elements.append(self._build_equity_curve(daily_equity))
            elements.append(Spacer(1, 0.25 * inch))

        # Strategy win-rate trends
        if strategy_stats:
            elements.append(
                Paragraph("Strategy Win Rate Trends", styles["Heading2"])
            )
            elements.append(Spacer(1, 0.1 * inch))
            elements.append(self._build_strategy_table(strategy_stats))
            elements.append(Spacer(1, 0.25 * inch))

        # Top 3 performing assets
        asset_performance = stats.get("asset_performance", [])
        if asset_performance:
            elements.append(
                Paragraph("Top Performing Assets", styles["Heading2"])
            )
            elements.append(Spacer(1, 0.1 * inch))
            elements.append(self._build_top_assets_table(asset_performance))
            elements.append(Spacer(1, 0.25 * inch))

        # Regime breakdown
        regimes = stats.get("regimes", {})
        if regimes:
            elements.append(
                Paragraph("Regime Breakdown", styles["Heading2"])
            )
            elements.append(Spacer(1, 0.1 * inch))
            elements.append(self._build_regime_breakdown(regimes))
            elements.append(Spacer(1, 0.25 * inch))

        # Parameter review suggestions
        suggestions = self._generate_parameter_suggestions(strategy_stats)
        if suggestions:
            elements.append(
                Paragraph("Parameter Review Suggestions", styles["Heading2"])
            )
            elements.append(Spacer(1, 0.1 * inch))
            elements.append(self._build_suggestions(suggestions, styles))
            elements.append(Spacer(1, 0.25 * inch))

        # Footer
        elements.append(Spacer(1, 0.2 * inch))
        footer_style = ParagraphStyle(
            "footer",
            parent=styles["Normal"],
            fontSize=8,
            textColor=colors.grey,
            alignment=1,
        )
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        elements.append(
            Paragraph(
                f"Generated by APEX Crypto Trading System at {generated_at}",
                footer_style,
            )
        )

        doc.build(elements)

        log_with_data(
            logger,
            "info",
            "Weekly report generated",
            {
                "week_start": week_start,
                "week_end": week_end,
                "filepath": filepath,
                "strategy_count": len(strategy_stats),
            },
        )

        return os.path.abspath(filepath)

    # ------------------------------------------------------------------
    # Builder methods
    # ------------------------------------------------------------------

    def _build_header(
        self, week_start: str, week_end: str, styles: Any
    ) -> Paragraph:
        """Build the report header paragraph.

        Args:
            week_start: Week start date string.
            week_end: Week end date string.
            styles: Reportlab stylesheet.

        Returns:
            A Paragraph element for the header.
        """
        header_style = ParagraphStyle(
            "report_header",
            parent=styles["Title"],
            fontSize=16,
            textColor=_HEADER_BG,
            spaceAfter=6,
            alignment=1,
        )
        return Paragraph(
            f"APEX Crypto Trading System &mdash; Weekly Report &mdash; "
            f"{week_start} to {week_end}",
            header_style,
        )

    def _build_summary_table(self, stats: dict, styles: Any) -> Table:
        """Build the weekly summary statistics table.

        Args:
            stats: Aggregate statistics dictionary.
            styles: Reportlab stylesheet.

        Returns:
            A Table element with weekly summary data.
        """
        total_pnl = stats.get("total_pnl", 0.0)
        total_trades = stats.get("total_trades", 0)
        wins = stats.get("wins", 0)
        losses = stats.get("losses", 0)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
        avg_daily_pnl = stats.get("avg_daily_pnl", 0.0)
        best_day = stats.get("best_day", {})
        worst_day = stats.get("worst_day", {})
        sharpe = stats.get("sharpe", 0.0)
        max_drawdown = stats.get("max_drawdown", 0.0)

        pnl_color = _PROFIT_GREEN if total_pnl >= 0 else _LOSS_RED

        best_day_str = (
            f"{best_day.get('date', 'N/A')}: ${best_day.get('pnl', 0.0):+,.2f}"
            if best_day else "N/A"
        )
        worst_day_str = (
            f"{worst_day.get('date', 'N/A')}: ${worst_day.get('pnl', 0.0):+,.2f}"
            if worst_day else "N/A"
        )

        data = [
            ["Metric", "Value"],
            ["Total P&L", f"${total_pnl:+,.2f}"],
            ["Avg Daily P&L", f"${avg_daily_pnl:+,.2f}"],
            ["Total Trades", str(total_trades)],
            ["Wins / Losses", f"{wins} / {losses}"],
            ["Win Rate", f"{win_rate:.1f}%"],
            ["Sharpe Ratio", f"{sharpe:.2f}"],
            ["Max Drawdown", f"{max_drawdown:.2f}%"],
            ["Best Day", best_day_str],
            ["Worst Day", worst_day_str],
        ]

        table = Table(data, colWidths=[2.5 * inch, 3.5 * inch])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), _TABLE_HEADER_BG),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("GRID", (0, 0), (-1, -1), 0.5, _BORDER_GREY),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _TABLE_ALT_ROW]),
            ("TEXTCOLOR", (1, 1), (1, 1), pnl_color),
            ("FONTNAME", (1, 1), (1, 1), "Helvetica-Bold"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ]))

        return table

    def _build_equity_curve(self, daily_equity: list) -> Drawing:
        """Build a 7-day equity curve chart.

        Args:
            daily_equity: List of daily equity values (up to 7).

        Returns:
            A Drawing containing the equity line chart.
        """
        width = _USABLE_WIDTH
        height = 160.0

        drawing = Drawing(width, height)

        chart_left = 65.0
        chart_bottom = 30.0
        chart_width = width - chart_left - 20
        chart_height = height - chart_bottom - 25

        # Background
        drawing.add(Rect(
            chart_left, chart_bottom, chart_width, chart_height,
            fillColor=colors.HexColor("#fafafa"),
            strokeColor=_BORDER_GREY,
            strokeWidth=0.5,
        ))

        equity_values = [
            float(v) if not isinstance(v, (int, float)) else v
            for v in daily_equity
        ]

        if len(equity_values) < 2:
            return drawing

        min_val = min(equity_values)
        max_val = max(equity_values)
        val_range = max_val - min_val if max_val != min_val else 1.0

        lp = LinePlot()
        lp.x = chart_left
        lp.y = chart_bottom
        lp.width = chart_width
        lp.height = chart_height

        plot_data = [(i, v) for i, v in enumerate(equity_values)]
        lp.data = [plot_data]

        lp.lines[0].strokeColor = _CHART_BLUE
        lp.lines[0].strokeWidth = 2.0

        lp.xValueAxis.valueMin = 0
        lp.xValueAxis.valueMax = len(equity_values) - 1
        lp.xValueAxis.labels.fontSize = 7
        lp.xValueAxis.labels.textColor = colors.grey

        lp.yValueAxis.valueMin = min_val - val_range * 0.05
        lp.yValueAxis.valueMax = max_val + val_range * 0.05
        lp.yValueAxis.labels.fontSize = 7
        lp.yValueAxis.labels.textColor = colors.grey

        drawing.add(lp)

        drawing.add(String(
            chart_left, height - 12,
            "7-Day Equity Curve",
            fontSize=9,
            fillColor=colors.HexColor("#333333"),
        ))

        return drawing

    def _build_strategy_table(self, strategy_stats: dict) -> Table:
        """Build the strategy win-rate trends table.

        Highlights strategies with declining win rates in amber.

        Args:
            strategy_stats: Dictionary mapping strategy names to stat
                dicts containing ``win_rate``, ``prev_win_rate``,
                ``avg_r``, ``total_trades``, ``wins``, ``losses``,
                ``total_pnl``.

        Returns:
            A Table element with strategy performance data.
        """
        headers = [
            "Strategy", "Win Rate", "Prev WR", "Change",
            "Avg R", "Trades", "P&L",
        ]
        data = [headers]

        declining_rows: list[int] = []

        for name, s_stats in sorted(strategy_stats.items()):
            win_rate = s_stats.get("win_rate", 0.0)
            prev_win_rate = s_stats.get("prev_win_rate", win_rate)
            wr_change = win_rate - prev_win_rate
            avg_r = s_stats.get("avg_r", 0.0)
            total_trades = s_stats.get("total_trades", 0)
            total_pnl = s_stats.get("total_pnl", 0.0)

            row_idx = len(data)
            if wr_change < -_WIN_RATE_DECLINE_THRESHOLD:
                declining_rows.append(row_idx)

            data.append([
                name,
                f"{win_rate:.1f}%",
                f"{prev_win_rate:.1f}%",
                f"{wr_change:+.1f}pp",
                f"{avg_r:.2f}R",
                str(total_trades),
                f"${total_pnl:+,.2f}",
            ])

        col_widths = [
            1.3 * inch, 0.7 * inch, 0.7 * inch, 0.7 * inch,
            0.6 * inch, 0.6 * inch, 1.0 * inch,
        ]
        table = Table(data, colWidths=col_widths, repeatRows=1)

        style_commands: list[Any] = [
            ("BACKGROUND", (0, 0), (-1, 0), _TABLE_HEADER_BG),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ("ALIGN", (0, 0), (0, -1), "LEFT"),
            ("GRID", (0, 0), (-1, -1), 0.5, _BORDER_GREY),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _TABLE_ALT_ROW]),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ]

        for row_idx in declining_rows:
            style_commands.append(
                ("TEXTCOLOR", (3, row_idx), (3, row_idx), _WARNING_AMBER)
            )
            style_commands.append(
                ("FONTNAME", (3, row_idx), (3, row_idx), "Helvetica-Bold")
            )

        table.setStyle(TableStyle(style_commands))
        return table

    def _build_top_assets_table(self, asset_performance: list) -> Table:
        """Build a table of the top 3 performing assets by P&L.

        Args:
            asset_performance: List of dicts with ``symbol``, ``pnl``,
                ``trades``, ``win_rate``.

        Returns:
            A Table element with the top 3 assets.
        """
        sorted_assets = sorted(
            asset_performance,
            key=lambda a: a.get("pnl", 0.0),
            reverse=True,
        )
        top_3 = sorted_assets[:3]

        headers = ["Rank", "Symbol", "P&L", "Trades", "Win Rate"]
        data = [headers]

        for rank, asset in enumerate(top_3, start=1):
            pnl = asset.get("pnl", 0.0)
            data.append([
                f"#{rank}",
                asset.get("symbol", ""),
                f"${pnl:+,.2f}",
                str(asset.get("trades", 0)),
                f"{asset.get('win_rate', 0.0):.1f}%",
            ])

        col_widths = [0.6 * inch, 1.2 * inch, 1.2 * inch, 0.8 * inch, 0.9 * inch]
        table = Table(data, colWidths=col_widths)

        style_commands: list[Any] = [
            ("BACKGROUND", (0, 0), (-1, 0), _TABLE_HEADER_BG),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (0, 0), (0, -1), "CENTER"),
            ("ALIGN", (2, 1), (-1, -1), "RIGHT"),
            ("GRID", (0, 0), (-1, -1), 0.5, _BORDER_GREY),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _TABLE_ALT_ROW]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ]

        # Colour P&L cells
        for row_idx in range(1, len(data)):
            pnl = top_3[row_idx - 1].get("pnl", 0.0)
            colour = _PROFIT_GREEN if pnl >= 0 else _LOSS_RED
            style_commands.append(
                ("TEXTCOLOR", (2, row_idx), (2, row_idx), colour)
            )
            style_commands.append(
                ("FONTNAME", (2, row_idx), (2, row_idx), "Helvetica-Bold")
            )

        table.setStyle(TableStyle(style_commands))
        return table

    def _build_regime_breakdown(self, regimes: dict) -> Table:
        """Build the regime time-distribution table.

        Args:
            regimes: Dictionary mapping regime names to the percentage
                of time spent in that regime during the week (e.g.
                ``{"STRONG_BULL": 35.2, "RANGING": 40.1, ...}``).

        Returns:
            A Table element with regime distribution.
        """
        headers = ["Regime", "% of Time", "Visual"]
        data = [headers]

        sorted_regimes = sorted(
            regimes.items(),
            key=lambda r: r[1],
            reverse=True,
        )

        for regime_name, pct in sorted_regimes:
            pct_val = float(pct)
            bar_chars = int(pct_val / 5)
            bar_visual = "|" * bar_chars
            data.append([
                regime_name,
                f"{pct_val:.1f}%",
                bar_visual,
            ])

        col_widths = [1.8 * inch, 1.0 * inch, 3.0 * inch]
        table = Table(data, colWidths=col_widths)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), _TABLE_HEADER_BG),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("FONTNAME", (2, 1), (2, -1), "Courier"),
            ("TEXTCOLOR", (2, 1), (2, -1), _CHART_BLUE),
            ("ALIGN", (1, 1), (1, -1), "RIGHT"),
            ("GRID", (0, 0), (-1, -1), 0.5, _BORDER_GREY),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _TABLE_ALT_ROW]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ]))

        return table

    def _build_suggestions(
        self, suggestions: list[dict], styles: Any
    ) -> Table:
        """Build the parameter review suggestions section.

        Args:
            suggestions: List of suggestion dicts with keys ``strategy``,
                ``issue``, ``recommendation``.
            styles: Reportlab stylesheet.

        Returns:
            A Table element with actionable suggestions.
        """
        headers = ["Strategy", "Issue", "Recommendation"]
        data = [headers]

        for suggestion in suggestions:
            data.append([
                suggestion.get("strategy", ""),
                suggestion.get("issue", ""),
                suggestion.get("recommendation", ""),
            ])

        col_widths = [1.3 * inch, 2.0 * inch, 2.7 * inch]
        table = Table(data, colWidths=col_widths, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), _WARNING_AMBER),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, _BORDER_GREY),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _TABLE_ALT_ROW]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ]))

        return table

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def _generate_parameter_suggestions(
        self, strategy_stats: dict
    ) -> list[dict]:
        """Analyse strategy stats and generate parameter review suggestions.

        A suggestion is generated for any strategy whose win rate has
        declined by more than ``_WIN_RATE_DECLINE_THRESHOLD`` percentage
        points compared to its previous period.

        Args:
            strategy_stats: Dictionary mapping strategy names to stat
                dicts with ``win_rate`` and ``prev_win_rate``.

        Returns:
            List of suggestion dictionaries with ``strategy``,
            ``issue``, and ``recommendation`` keys.
        """
        suggestions: list[dict] = []

        for name, s_stats in strategy_stats.items():
            win_rate = s_stats.get("win_rate", 0.0)
            prev_win_rate = s_stats.get("prev_win_rate", win_rate)
            wr_change = win_rate - prev_win_rate

            if wr_change < -_WIN_RATE_DECLINE_THRESHOLD:
                avg_r = s_stats.get("avg_r", 0.0)
                total_trades = s_stats.get("total_trades", 0)

                recommendation_parts: list[str] = []

                if win_rate < 40.0:
                    recommendation_parts.append(
                        "Win rate is critically low. Consider tightening "
                        "entry filters or increasing minimum signal score."
                    )
                elif win_rate < 50.0:
                    recommendation_parts.append(
                        "Win rate below 50%. Review entry conditions and "
                        "regime filters for this strategy."
                    )
                else:
                    recommendation_parts.append(
                        "Win rate declining but still above 50%. Monitor "
                        "closely over the next week."
                    )

                if avg_r < 1.0:
                    recommendation_parts.append(
                        "Average R-multiple below 1.0 -- review "
                        "take-profit placement and exit logic."
                    )

                if total_trades > 20:
                    recommendation_parts.append(
                        "High trade count suggests possible over-trading. "
                        "Consider raising conviction thresholds."
                    )

                suggestions.append({
                    "strategy": name,
                    "issue": (
                        f"Win rate declined {abs(wr_change):.1f}pp "
                        f"({prev_win_rate:.1f}% -> {win_rate:.1f}%)"
                    ),
                    "recommendation": " ".join(recommendation_parts),
                })

                log_with_data(
                    logger,
                    "warning",
                    "Strategy win rate declining — parameter review suggested",
                    {
                        "strategy": name,
                        "win_rate": win_rate,
                        "prev_win_rate": prev_win_rate,
                        "decline_pp": abs(wr_change),
                    },
                )

        return suggestions
