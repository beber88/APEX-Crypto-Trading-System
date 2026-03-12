"""Daily PDF report generator for the APEX Crypto Trading System.

Produces comprehensive daily trading reports using reportlab, including
portfolio summaries, trade tables, equity curve charts, strategy
performance breakdowns, regime summaries, and risk metrics.
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
from reportlab.graphics import renderPDF

from apex_crypto.core.logging import get_logger, log_with_data

logger = get_logger("reporting.daily")

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_HEADER_BG = colors.HexColor("#1a1a2e")
_HEADER_FG = colors.HexColor("#e6e6e6")
_PROFIT_GREEN = colors.HexColor("#00c853")
_LOSS_RED = colors.HexColor("#ff1744")
_TABLE_HEADER_BG = colors.HexColor("#16213e")
_TABLE_ALT_ROW = colors.HexColor("#f5f5f5")
_BORDER_GREY = colors.HexColor("#cccccc")

# Page dimensions
_PAGE_WIDTH, _PAGE_HEIGHT = letter
_LEFT_MARGIN = 0.75 * inch
_RIGHT_MARGIN = 0.75 * inch
_TOP_MARGIN = 0.75 * inch
_BOTTOM_MARGIN = 0.75 * inch
_USABLE_WIDTH = _PAGE_WIDTH - _LEFT_MARGIN - _RIGHT_MARGIN


class DailyReportGenerator:
    """Generates daily PDF trading reports.

    Produces a single-file PDF containing portfolio summary, trade
    details, equity curve, strategy breakdowns, regime summaries,
    and risk metrics for a given trading day.

    Attributes:
        output_dir: Directory where generated PDFs are saved.
    """

    def __init__(self, config: dict) -> None:
        """Initialize the daily report generator.

        Args:
            config: Full application configuration dictionary.  The
                ``reporting.output_dir`` key is used to determine where
                PDF files are written.
        """
        reporting_cfg = config.get("reporting", {})
        self.output_dir: str = reporting_cfg.get("output_dir", "./reports")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        log_with_data(
            logger,
            "info",
            "DailyReportGenerator initialised",
            {"output_dir": self.output_dir},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        date: str,
        stats: dict,
        trades: list[dict],
        equity_curve: list[float],
    ) -> str:
        """Generate a complete daily PDF report.

        Args:
            date: Report date string (e.g. ``"2026-03-11"``).
            stats: Portfolio statistics dictionary with keys such as
                ``equity``, ``prev_equity``, ``daily_pnl``, ``wins``,
                ``losses``, ``strategies`` (dict of strategy-level stats),
                ``regimes`` (dict of per-asset regime info),
                ``current_drawdown``, ``var_estimate``.
            trades: List of trade dictionaries, each containing
                ``symbol``, ``direction``, ``entry_price``,
                ``exit_price``, ``pnl``, ``r_multiple``, ``strategy``,
                ``entry_time``, ``exit_time``.
            equity_curve: List of equity values representing the
                intraday or trailing equity curve.

        Returns:
            Absolute file path of the generated PDF.
        """
        filename = f"apex_daily_report_{date}.pdf"
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

        # a) Header
        elements.append(self._build_header(date, styles))
        elements.append(Spacer(1, 0.3 * inch))

        # b) Portfolio summary
        elements.append(self._build_portfolio_summary(stats, styles))
        elements.append(Spacer(1, 0.25 * inch))

        # c) Trade table
        if trades:
            elements.append(
                Paragraph("Trade Log", styles["Heading2"])
            )
            elements.append(Spacer(1, 0.1 * inch))
            elements.append(self._build_trade_table_platypus(trades))
            elements.append(Spacer(1, 0.25 * inch))

            # d) Best and worst trade
            elements.append(self._build_best_worst_trades(trades, styles))
            elements.append(Spacer(1, 0.25 * inch))

        # e) Strategy performance table
        strategies = stats.get("strategies", {})
        if strategies:
            elements.append(
                Paragraph("Strategy Performance", styles["Heading2"])
            )
            elements.append(Spacer(1, 0.1 * inch))
            elements.append(self._build_strategy_table(strategies))
            elements.append(Spacer(1, 0.25 * inch))

        # f) Regime summary per asset
        regimes = stats.get("regimes", {})
        if regimes:
            elements.append(
                Paragraph("Regime Summary", styles["Heading2"])
            )
            elements.append(Spacer(1, 0.1 * inch))
            elements.append(self._build_regime_table(regimes))
            elements.append(Spacer(1, 0.25 * inch))

        # g) Risk metrics
        elements.append(
            Paragraph("Risk Metrics", styles["Heading2"])
        )
        elements.append(Spacer(1, 0.1 * inch))
        elements.append(self._build_risk_metrics(stats, styles))
        elements.append(Spacer(1, 0.25 * inch))

        # Equity curve chart
        if equity_curve and len(equity_curve) >= 2:
            elements.append(
                Paragraph("Equity Curve", styles["Heading2"])
            )
            elements.append(Spacer(1, 0.1 * inch))
            elements.append(self._build_equity_chart_drawing(equity_curve))
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
            "Daily report generated",
            {"date": date, "filepath": filepath, "trade_count": len(trades)},
        )

        return os.path.abspath(filepath)

    # ------------------------------------------------------------------
    # Canvas-based methods (for direct canvas drawing)
    # ------------------------------------------------------------------

    def _build_trade_table(
        self,
        canvas: Any,
        trades: list[dict],
        y_position: float,
    ) -> float:
        """Draw a formatted trade table directly on a reportlab canvas.

        Args:
            canvas: A reportlab Canvas instance.
            trades: List of trade dictionaries.
            y_position: Starting Y coordinate on the canvas.

        Returns:
            Updated Y position after drawing the table.
        """
        headers = ["Symbol", "Dir", "Entry", "Exit", "P&L", "R-Mult", "Strategy"]
        col_widths = [80, 40, 70, 70, 70, 55, 90]
        x_start = _LEFT_MARGIN
        row_height = 16

        # Draw header row
        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(_TABLE_HEADER_BG)
        total_width = sum(col_widths)
        canvas.rect(x_start, y_position - row_height, total_width, row_height, fill=1)

        canvas.setFillColor(colors.white)
        x_offset = x_start
        for i, header in enumerate(headers):
            canvas.drawString(x_offset + 3, y_position - row_height + 4, header)
            x_offset += col_widths[i]

        y_position -= row_height

        # Draw data rows
        canvas.setFont("Helvetica", 7)
        for row_idx, trade in enumerate(trades):
            y_position -= row_height

            if y_position < _BOTTOM_MARGIN + 20:
                canvas.showPage()
                y_position = _PAGE_HEIGHT - _TOP_MARGIN
                canvas.setFont("Helvetica", 7)

            if row_idx % 2 == 0:
                canvas.setFillColor(_TABLE_ALT_ROW)
                canvas.rect(x_start, y_position, total_width, row_height, fill=1)

            pnl = trade.get("pnl", 0.0)
            pnl_color = _PROFIT_GREEN if pnl >= 0 else _LOSS_RED

            row_data = [
                trade.get("symbol", ""),
                trade.get("direction", ""),
                f"{trade.get('entry_price', 0.0):.2f}",
                f"{trade.get('exit_price', 0.0):.2f}",
                f"${pnl:+.2f}",
                f"{trade.get('r_multiple', 0.0):.2f}R",
                trade.get("strategy", ""),
            ]

            x_offset = x_start
            for i, cell in enumerate(row_data):
                if i == 4:
                    canvas.setFillColor(pnl_color)
                else:
                    canvas.setFillColor(colors.black)
                canvas.drawString(x_offset + 3, y_position + 4, str(cell))
                x_offset += col_widths[i]

        canvas.setFillColor(colors.black)
        return y_position

    def _build_equity_chart(
        self,
        canvas: Any,
        equity_data: list[float],
        y_position: float,
    ) -> float:
        """Draw a mini equity curve chart on a reportlab canvas.

        Args:
            canvas: A reportlab Canvas instance.
            equity_data: List of equity values to plot.
            y_position: Starting Y coordinate on the canvas.

        Returns:
            Updated Y position after drawing the chart.
        """
        if not equity_data or len(equity_data) < 2:
            return y_position

        chart_width = _USABLE_WIDTH
        chart_height = 120
        chart_y = y_position - chart_height - 10

        drawing = self._create_equity_drawing(equity_data, chart_width, chart_height)
        renderPDF.draw(drawing, canvas, _LEFT_MARGIN, chart_y)

        return chart_y - 10

    # ------------------------------------------------------------------
    # Platypus-based helper builders (for doc.build flow)
    # ------------------------------------------------------------------

    def _build_header(self, date: str, styles: Any) -> Paragraph:
        """Build the report header paragraph.

        Args:
            date: Report date string.
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
            f"APEX Crypto Trading System &mdash; Daily Report &mdash; {date}",
            header_style,
        )

    def _build_portfolio_summary(self, stats: dict, styles: Any) -> Table:
        """Build the portfolio summary table.

        Args:
            stats: Portfolio statistics dictionary.
            styles: Reportlab stylesheet.

        Returns:
            A Table element with portfolio summary data.
        """
        equity = stats.get("equity", 0.0)
        prev_equity = stats.get("prev_equity", 0.0)
        daily_pnl = stats.get("daily_pnl", equity - prev_equity)
        pnl_pct = (daily_pnl / prev_equity * 100) if prev_equity != 0 else 0.0
        wins = stats.get("wins", 0)
        losses = stats.get("losses", 0)
        total_trades = wins + losses
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

        pnl_color = _PROFIT_GREEN if daily_pnl >= 0 else _LOSS_RED

        data = [
            ["Metric", "Value"],
            ["Current Equity", f"${equity:,.2f}"],
            ["Previous Equity", f"${prev_equity:,.2f}"],
            ["Daily P&L", f"${daily_pnl:+,.2f} ({pnl_pct:+.2f}%)"],
            ["Wins", str(wins)],
            ["Losses", str(losses)],
            ["Win Rate", f"{win_rate:.1f}%"],
            ["Total Trades", str(total_trades)],
        ]

        table = Table(data, colWidths=[2.5 * inch, 3 * inch])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), _TABLE_HEADER_BG),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("GRID", (0, 0), (-1, -1), 0.5, _BORDER_GREY),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _TABLE_ALT_ROW]),
            ("TEXTCOLOR", (1, 3), (1, 3), pnl_color),
            ("FONTNAME", (1, 3), (1, 3), "Helvetica-Bold"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ]))

        return table

    def _build_trade_table_platypus(self, trades: list[dict]) -> Table:
        """Build a trade log table as a Platypus Table flowable.

        Args:
            trades: List of trade dictionaries.

        Returns:
            A Table element with all trade data.
        """
        headers = ["Symbol", "Dir", "Entry", "Exit", "P&L", "R-Mult", "Strategy"]
        data = [headers]

        for trade in trades:
            pnl = trade.get("pnl", 0.0)
            data.append([
                trade.get("symbol", ""),
                trade.get("direction", ""),
                f"{trade.get('entry_price', 0.0):.2f}",
                f"{trade.get('exit_price', 0.0):.2f}",
                f"${pnl:+,.2f}",
                f"{trade.get('r_multiple', 0.0):.2f}R",
                trade.get("strategy", ""),
            ])

        col_widths = [
            1.1 * inch, 0.5 * inch, 0.85 * inch, 0.85 * inch,
            0.9 * inch, 0.7 * inch, 1.2 * inch,
        ]
        table = Table(data, colWidths=col_widths, repeatRows=1)

        style_commands: list[Any] = [
            ("BACKGROUND", (0, 0), (-1, 0), _TABLE_HEADER_BG),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (2, 1), (-1, -1), "RIGHT"),
            ("ALIGN", (0, 0), (1, -1), "LEFT"),
            ("GRID", (0, 0), (-1, -1), 0.5, _BORDER_GREY),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _TABLE_ALT_ROW]),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ]

        # Colour P&L cells
        for row_idx, trade in enumerate(trades, start=1):
            pnl = trade.get("pnl", 0.0)
            colour = _PROFIT_GREEN if pnl >= 0 else _LOSS_RED
            style_commands.append(("TEXTCOLOR", (4, row_idx), (4, row_idx), colour))
            style_commands.append(("FONTNAME", (4, row_idx), (4, row_idx), "Helvetica-Bold"))

        table.setStyle(TableStyle(style_commands))
        return table

    def _build_best_worst_trades(
        self, trades: list[dict], styles: Any
    ) -> Table:
        """Build a summary of the best and worst trades.

        Args:
            trades: List of trade dictionaries.
            styles: Reportlab stylesheet.

        Returns:
            A Table element highlighting best and worst trades.
        """
        sorted_by_pnl = sorted(trades, key=lambda t: t.get("pnl", 0.0))
        worst = sorted_by_pnl[0]
        best = sorted_by_pnl[-1]

        data = [
            ["", "Symbol", "Direction", "P&L", "R-Multiple", "Strategy"],
            [
                "Best Trade",
                best.get("symbol", ""),
                best.get("direction", ""),
                f"${best.get('pnl', 0.0):+,.2f}",
                f"{best.get('r_multiple', 0.0):.2f}R",
                best.get("strategy", ""),
            ],
            [
                "Worst Trade",
                worst.get("symbol", ""),
                worst.get("direction", ""),
                f"${worst.get('pnl', 0.0):+,.2f}",
                f"{worst.get('r_multiple', 0.0):.2f}R",
                worst.get("strategy", ""),
            ],
        ]

        table = Table(data, colWidths=[
            1.0 * inch, 1.0 * inch, 0.8 * inch, 0.9 * inch, 0.9 * inch, 1.2 * inch,
        ])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), _TABLE_HEADER_BG),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, _BORDER_GREY),
            ("BACKGROUND", (0, 1), (0, 1), _PROFIT_GREEN),
            ("TEXTCOLOR", (0, 1), (0, 1), colors.white),
            ("BACKGROUND", (0, 2), (0, 2), _LOSS_RED),
            ("TEXTCOLOR", (0, 2), (0, 2), colors.white),
            ("FONTNAME", (0, 1), (0, 2), "Helvetica-Bold"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ]))

        return table

    def _build_strategy_table(self, strategies: dict) -> Table:
        """Build the strategy performance summary table.

        Args:
            strategies: Dictionary mapping strategy names to stat dicts.
                Each stat dict should contain ``win_rate``, ``avg_r``,
                ``total_trades``, ``wins``, ``losses``, ``total_pnl``.

        Returns:
            A Table element with per-strategy performance.
        """
        headers = ["Strategy", "Win Rate", "Avg R", "Trades", "Wins", "Losses", "Total P&L"]
        data = [headers]

        for name, strat_stats in sorted(strategies.items()):
            win_rate = strat_stats.get("win_rate", 0.0)
            avg_r = strat_stats.get("avg_r", 0.0)
            total_trades = strat_stats.get("total_trades", 0)
            wins = strat_stats.get("wins", 0)
            losses = strat_stats.get("losses", 0)
            total_pnl = strat_stats.get("total_pnl", 0.0)

            data.append([
                name,
                f"{win_rate:.1f}%",
                f"{avg_r:.2f}R",
                str(total_trades),
                str(wins),
                str(losses),
                f"${total_pnl:+,.2f}",
            ])

        col_widths = [
            1.3 * inch, 0.8 * inch, 0.7 * inch, 0.65 * inch,
            0.55 * inch, 0.55 * inch, 1.0 * inch,
        ]
        table = Table(data, colWidths=col_widths, repeatRows=1)
        table.setStyle(TableStyle([
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
        ]))

        return table

    def _build_regime_table(self, regimes: dict) -> Table:
        """Build the regime summary table.

        Args:
            regimes: Dictionary mapping asset symbols to regime info dicts.
                Each dict should contain ``regime``, ``confidence``,
                and optionally ``trend_strength``.

        Returns:
            A Table element with per-asset regime data.
        """
        headers = ["Asset", "Current Regime", "Confidence", "Trend Strength"]
        data = [headers]

        for asset, regime_info in sorted(regimes.items()):
            if isinstance(regime_info, str):
                data.append([asset, regime_info, "N/A", "N/A"])
            else:
                data.append([
                    asset,
                    regime_info.get("regime", "UNKNOWN"),
                    f"{regime_info.get('confidence', 0.0):.1f}%",
                    f"{regime_info.get('trend_strength', 0.0):.2f}",
                ])

        col_widths = [1.3 * inch, 1.5 * inch, 1.0 * inch, 1.0 * inch]
        table = Table(data, colWidths=col_widths, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), _TABLE_HEADER_BG),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (2, 1), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, _BORDER_GREY),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _TABLE_ALT_ROW]),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ]))

        return table

    def _build_risk_metrics(self, stats: dict, styles: Any) -> Table:
        """Build the risk metrics summary table.

        Args:
            stats: Portfolio statistics dictionary containing
                ``current_drawdown`` and ``var_estimate``.
            styles: Reportlab stylesheet.

        Returns:
            A Table element with risk metric data.
        """
        current_drawdown = stats.get("current_drawdown", 0.0)
        var_estimate = stats.get("var_estimate", 0.0)
        max_drawdown = stats.get("max_drawdown", 0.0)
        sharpe = stats.get("sharpe_ratio", 0.0)
        daily_vol = stats.get("daily_volatility", 0.0)

        dd_color = _LOSS_RED if current_drawdown > 5.0 else colors.black

        data = [
            ["Metric", "Value"],
            ["Current Drawdown", f"{current_drawdown:.2f}%"],
            ["Max Drawdown", f"{max_drawdown:.2f}%"],
            ["Value at Risk (95%)", f"${var_estimate:,.2f}"],
            ["Sharpe Ratio", f"{sharpe:.2f}"],
            ["Daily Volatility", f"{daily_vol:.2f}%"],
        ]

        table = Table(data, colWidths=[2.5 * inch, 3 * inch])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), _TABLE_HEADER_BG),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, _BORDER_GREY),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _TABLE_ALT_ROW]),
            ("TEXTCOLOR", (1, 1), (1, 1), dd_color),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ]))

        return table

    # ------------------------------------------------------------------
    # Chart helpers
    # ------------------------------------------------------------------

    def _build_equity_chart_drawing(self, equity_data: list[float]) -> Drawing:
        """Build an equity curve chart as a reportlab Drawing flowable.

        Args:
            equity_data: List of equity values to plot.

        Returns:
            A Drawing object containing the line chart.
        """
        return self._create_equity_drawing(equity_data, _USABLE_WIDTH, 150)

    @staticmethod
    def _create_equity_drawing(
        equity_data: list[float],
        width: float,
        height: float,
    ) -> Drawing:
        """Create a Drawing containing a line plot of equity data.

        Args:
            equity_data: List of equity values.
            width: Drawing width in points.
            height: Drawing height in points.

        Returns:
            A reportlab Drawing with the equity line chart.
        """
        drawing = Drawing(width, height)

        chart_left = 60
        chart_bottom = 30
        chart_width = width - chart_left - 20
        chart_height = height - chart_bottom - 20

        # Background
        drawing.add(Rect(
            chart_left, chart_bottom, chart_width, chart_height,
            fillColor=colors.HexColor("#fafafa"),
            strokeColor=_BORDER_GREY,
            strokeWidth=0.5,
        ))

        if len(equity_data) < 2:
            return drawing

        min_val = min(equity_data)
        max_val = max(equity_data)
        val_range = max_val - min_val if max_val != min_val else 1.0

        # Build the line chart
        lp = LinePlot()
        lp.x = chart_left
        lp.y = chart_bottom
        lp.width = chart_width
        lp.height = chart_height

        plot_data = [(i, v) for i, v in enumerate(equity_data)]
        lp.data = [plot_data]

        lp.lines[0].strokeColor = colors.HexColor("#2196f3")
        lp.lines[0].strokeWidth = 1.5

        lp.xValueAxis.valueMin = 0
        lp.xValueAxis.valueMax = len(equity_data) - 1
        lp.xValueAxis.labels.fontSize = 7
        lp.xValueAxis.labels.textColor = colors.grey

        lp.yValueAxis.valueMin = min_val - val_range * 0.05
        lp.yValueAxis.valueMax = max_val + val_range * 0.05
        lp.yValueAxis.labels.fontSize = 7
        lp.yValueAxis.labels.textColor = colors.grey

        drawing.add(lp)

        # Title label
        drawing.add(String(
            chart_left, height - 12,
            "Equity Curve",
            fontSize=9,
            fillColor=colors.HexColor("#333333"),
        ))

        return drawing
