"""Dashboard application factory for the APEX Crypto Trading System.

Creates and configures the FastAPI app, wiring it to the live trading engine
so that dashboard endpoints can serve real-time data.
"""

from __future__ import annotations

import time
from typing import Any, Optional


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>APEX Crypto Trading System</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #0a0e17; color: #e0e6ed; }
.header { background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 100%); padding: 20px 30px; border-bottom: 1px solid #21262d; display: flex; justify-content: space-between; align-items: center; }
.header h1 { font-size: 22px; color: #58a6ff; }
.header .mode { padding: 6px 16px; border-radius: 20px; font-size: 13px; font-weight: 600; }
.mode-live { background: #1b4332; color: #40c057; border: 1px solid #2d6a4f; }
.mode-paper { background: #3d2800; color: #ffa94d; border: 1px solid #5c4000; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; padding: 20px; }
.card { background: #161b22; border: 1px solid #21262d; border-radius: 12px; padding: 20px; }
.card h2 { font-size: 14px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 16px; }
.metric { display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #21262d; }
.metric:last-child { border-bottom: none; }
.metric .label { color: #8b949e; font-size: 14px; }
.metric .value { font-size: 18px; font-weight: 600; }
.value-green { color: #40c057; }
.value-red { color: #f03e3e; }
.value-blue { color: #58a6ff; }
.value-yellow { color: #ffa94d; }
.strategies { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; }
.strat-badge { background: #0d1117; border: 1px solid #21262d; border-radius: 8px; padding: 8px 12px; font-size: 12px; text-align: center; }
.strat-active { border-color: #238636; color: #3fb950; }
.positions-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.positions-table th { color: #8b949e; text-align: left; padding: 8px; border-bottom: 1px solid #21262d; }
.positions-table td { padding: 8px; border-bottom: 1px solid #161b22; }
.signals-list { max-height: 300px; overflow-y: auto; }
.signal-item { background: #0d1117; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }
.regime-badge { display: inline-block; padding: 4px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; }
.regime-STRONG_BULL, .regime-WEAK_BULL { background: #1b4332; color: #40c057; }
.regime-STRONG_BEAR, .regime-WEAK_BEAR { background: #3d0000; color: #f03e3e; }
.regime-RANGING { background: #3d2800; color: #ffa94d; }
.regime-CHAOS { background: #2d0040; color: #c084fc; }
.refresh-bar { text-align: center; padding: 8px; color: #484f58; font-size: 12px; }
.no-data { color: #484f58; font-style: italic; padding: 20px; text-align: center; }
</style>
</head>
<body>
<div class="header">
  <h1>APEX Crypto Trading System</h1>
  <span class="mode" id="mode-badge">Loading...</span>
</div>

<div class="grid">
  <div class="card">
    <h2>Account</h2>
    <div class="metric"><span class="label">Equity</span><span class="value value-green" id="equity">--</span></div>
    <div class="metric"><span class="label">Peak Equity</span><span class="value" id="peak-equity">--</span></div>
    <div class="metric"><span class="label">Drawdown</span><span class="value value-red" id="drawdown">--</span></div>
    <div class="metric"><span class="label">Daily P&L</span><span class="value" id="daily-pnl">--</span></div>
  </div>

  <div class="card">
    <h2>Trading Status</h2>
    <div class="metric"><span class="label">Trades Today</span><span class="value value-blue" id="trades-today">0</span></div>
    <div class="metric"><span class="label">Open Positions</span><span class="value value-blue" id="open-positions">0</span></div>
    <div class="metric"><span class="label">Consecutive Losses</span><span class="value" id="consec-losses">0</span></div>
    <div class="metric"><span class="label">System Uptime</span><span class="value" id="uptime">--</span></div>
  </div>

  <div class="card">
    <h2>Strategies (10)</h2>
    <div class="strategies" id="strategies-grid"></div>
  </div>

  <div class="card" style="grid-column: span 2;">
    <h2>Market Regimes</h2>
    <div id="regimes-container" class="no-data">Waiting for data...</div>
  </div>

  <div class="card" style="grid-column: span 2;">
    <h2>Active Signals</h2>
    <div class="signals-list" id="signals-container">
      <div class="no-data">No active signals yet — engine is scanning...</div>
    </div>
  </div>

  <div class="card" style="grid-column: span 2;">
    <h2>Open Positions</h2>
    <div id="positions-container">
      <div class="no-data">No open positions</div>
    </div>
  </div>
</div>

<div class="refresh-bar" id="refresh-bar">Auto-refreshing every 10 seconds</div>

<script>
const strategies = [
  'trend_momentum', 'mean_reversion', 'breakout', 'smc', 'scalping',
  'funding_rate', 'swing', 'oi_divergence', 'quant_momentum', 'stat_arb'
];

const stratGrid = document.getElementById('strategies-grid');
strategies.forEach(s => {
  stratGrid.innerHTML += `<div class="strat-badge strat-active">${s.replace('_', ' ')}</div>`;
});

const startTime = Date.now();

async function fetchData(endpoint) {
  try {
    const resp = await fetch('/api/' + endpoint, {
      headers: { 'Authorization': 'Basic ' + btoa('admin:apex_dashboard_2024') }
    });
    if (!resp.ok) return null;
    return await resp.json();
  } catch(e) { return null; }
}

function fmt(n, d=2) { return n != null ? Number(n).toFixed(d) : '--'; }

async function refresh() {
  const [status, equity, risk, signals, positions, regimes] = await Promise.all([
    fetchData('status'), fetchData('equity'), fetchData('risk'),
    fetchData('signals'), fetchData('positions'), fetchData('regimes')
  ]);

  if (status) {
    const badge = document.getElementById('mode-badge');
    badge.textContent = (status.mode || 'paper').toUpperCase();
    badge.className = 'mode mode-' + (status.mode || 'paper');
    const up = Math.floor((Date.now() - startTime) / 1000);
    const h = Math.floor(up/3600), m = Math.floor((up%3600)/60), s = up%60;
    document.getElementById('uptime').textContent = `${h}h ${m}m ${s}s`;
  }

  if (equity) {
    document.getElementById('equity').textContent = '$' + fmt(equity.current_equity);
    document.getElementById('peak-equity').textContent = '$' + fmt(equity.peak_equity);
    document.getElementById('drawdown').textContent = fmt(equity.drawdown_pct) + '%';
  }

  if (risk) {
    document.getElementById('daily-pnl').textContent = fmt(risk.daily_loss_pct) + '%';
    document.getElementById('daily-pnl').className = 'value ' + (risk.daily_loss_pct >= 0 ? 'value-green' : 'value-red');
    document.getElementById('trades-today').textContent = risk.trades_today || 0;
    document.getElementById('open-positions').textContent = risk.positions_count || 0;
    document.getElementById('consec-losses').textContent = risk.consecutive_losses || 0;
  }

  if (signals && signals.signals && signals.signals.length > 0) {
    const cont = document.getElementById('signals-container');
    cont.innerHTML = signals.signals.map(s =>
      `<div class="signal-item">
        <span><strong>${s.symbol}</strong> &mdash; ${s.strategy}</span>
        <span class="${s.direction==='long'?'value-green':'value-red'}">${s.direction.toUpperCase()} (${s.score})</span>
      </div>`
    ).join('');
  }

  if (positions && positions.positions && positions.positions.length > 0) {
    const cont = document.getElementById('positions-container');
    cont.innerHTML = `<table class="positions-table">
      <tr><th>Symbol</th><th>Direction</th><th>Entry</th><th>Current</th><th>P&L</th></tr>
      ${positions.positions.map(p => {
        const pnl = p.direction === 'long'
          ? ((p.current_price - p.entry_price) / p.entry_price * 100)
          : ((p.entry_price - p.current_price) / p.entry_price * 100);
        return `<tr>
          <td>${p.symbol}</td>
          <td class="${p.direction==='long'?'value-green':'value-red'}">${p.direction}</td>
          <td>$${fmt(p.entry_price)}</td>
          <td>$${fmt(p.current_price)}</td>
          <td class="${pnl>=0?'value-green':'value-red'}">${fmt(pnl)}%</td>
        </tr>`;
      }).join('')}
    </table>`;
  }

  if (regimes && regimes.regimes) {
    const cont = document.getElementById('regimes-container');
    const entries = Object.entries(regimes.regimes);
    if (entries.length > 0) {
      cont.innerHTML = entries.map(([sym, data]) =>
        `<span style="margin:4px;display:inline-block">
          <strong>${sym}</strong> <span class="regime-badge regime-${data.regime || data}">${data.regime || data}</span>
        </span>`
      ).join(' ');
      cont.className = '';
    }
  }

  document.getElementById('refresh-bar').textContent =
    'Last updated: ' + new Date().toLocaleTimeString() + ' — Auto-refreshing every 10s';
}

refresh();
setInterval(refresh, 10000);
</script>
</body>
</html>"""


class EngineDataStore:
    """Adapter that exposes trading engine state as attributes
    expected by the dashboard routes."""

    def __init__(self, engine) -> None:
        self._engine = engine
        self._trading_paused = False

    @property
    def mode(self) -> str:
        cfg = self._engine._full_config.get("system", {})
        return cfg.get("mode", "paper")

    @property
    def current_equity(self) -> float:
        return self._engine._equity_stats.get("current_equity", 0.0)

    @property
    def peak_equity(self) -> float:
        return self._engine._equity_stats.get("peak_equity", 0.0)

    @property
    def equity_curve(self) -> list[dict[str, Any]]:
        return [{"timestamp": time.time(), "value": self.current_equity}]

    @property
    def open_positions(self) -> list[dict[str, Any]]:
        return list(self._engine._open_positions)

    @property
    def trade_history(self) -> list[dict[str, Any]]:
        return []

    @property
    def current_signals(self) -> list[dict[str, Any]]:
        return list(self._engine._current_signals)

    @property
    def current_regimes(self) -> dict[str, Any]:
        return dict(self._engine._current_regimes)

    @property
    def fear_greed(self) -> dict[str, Any]:
        return {"value": 50, "classification": "Neutral", "timestamp": time.time()}

    @property
    def funding_rates(self) -> dict[str, Any]:
        return {}

    @property
    def risk_metrics(self) -> dict[str, Any]:
        stats = self._engine._equity_stats
        daily = self._engine._daily_stats
        return {
            "drawdown_pct": stats.get("current_drawdown_pct", 0.0),
            "daily_loss_pct": daily.get("daily_pnl_pct", 0.0),
            "positions_count": len(self._engine._open_positions),
            "trades_today": daily.get("trades_today", 0),
            "consecutive_losses": daily.get("consecutive_losses", 0),
        }

    @property
    def performance_30d(self) -> dict[str, Any]:
        return {
            "sharpe_30d": 0.0,
            "win_rate_30d": 0.5,
            "profit_factor_30d": 0.0,
            "total_trades_30d": 0,
            "total_pnl_30d": 0.0,
        }

    def set_trading_paused(self, paused: bool) -> None:
        self._trading_paused = paused
        self._engine._running = not paused


def create_app(config: dict, engine) -> Any:
    """Create the FastAPI app with engine state injected."""
    from fastapi.responses import HTMLResponse
    from apex_crypto.dashboard.api.routes import app

    # Inject the data store so routes can access engine state
    app.state.data_store = EngineDataStore(engine)

    # Add root HTML dashboard page
    @app.get("/", response_class=HTMLResponse)
    async def dashboard_home():
        return DASHBOARD_HTML

    return app
