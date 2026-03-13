"""Dashboard application factory for the APEX Crypto Trading System.

Creates and configures the FastAPI app, wiring it to the live trading engine
so that dashboard endpoints can serve real-time data.
"""

from __future__ import annotations

import time
from typing import Any, Optional


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>APEX Bloomberg Terminal</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
<style>
:root {
  --bg-primary: #0a0e17; --bg-secondary: #111827; --bg-tertiary: #1f2937;
  --bg-card: #0f1629; --border: #1e293b; --text-primary: #e2e8f0;
  --text-secondary: #94a3b8; --text-muted: #64748b;
  --green: #10b981; --green-bg: rgba(16,185,129,0.12); --green-dim: #059669;
  --red: #ef4444; --red-bg: rgba(239,68,68,0.12); --red-dim: #dc2626;
  --blue: #3b82f6; --blue-bg: rgba(59,130,246,0.12);
  --yellow: #f59e0b; --yellow-bg: rgba(245,158,11,0.12);
  --purple: #8b5cf6; --purple-bg: rgba(139,92,246,0.12);
  --orange: #f97316;
}
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: var(--bg-primary); color: var(--text-primary); overflow-x: hidden; }

/* ── Top Bar ── */
.topbar { background: linear-gradient(180deg, #111827 0%, #0f172a 100%); border-bottom: 1px solid var(--border); padding: 0 24px; height: 48px; display: flex; align-items: center; justify-content: space-between; position: sticky; top: 0; z-index: 100; }
.topbar-left { display: flex; align-items: center; gap: 16px; }
.logo { font-size: 18px; font-weight: 800; color: var(--yellow); letter-spacing: 2px; font-family: 'Courier New', monospace; }
.logo span { color: var(--text-muted); font-weight: 400; font-size: 11px; margin-left: 8px; letter-spacing: 0; }
.mode-pill { padding: 3px 12px; border-radius: 3px; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }
.mode-live { background: var(--green-bg); color: var(--green); border: 1px solid var(--green-dim); }
.mode-paper { background: var(--yellow-bg); color: var(--yellow); border: 1px solid rgba(245,158,11,0.3); }
.topbar-center { display: flex; align-items: center; gap: 20px; font-size: 13px; font-variant-numeric: tabular-nums; }
.topbar-center .label { color: var(--text-muted); font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; }
.topbar-center .value { font-weight: 700; font-size: 14px; }
.topbar-right { display: flex; align-items: center; gap: 16px; font-size: 12px; color: var(--text-secondary); }
.pulse { width: 7px; height: 7px; border-radius: 50%; background: var(--green); display: inline-block; animation: pulse 2s infinite; box-shadow: 0 0 6px var(--green); }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }
@keyframes flash-green { 0% { background: rgba(16,185,129,0.3); } 100% { background: transparent; } }
@keyframes flash-red { 0% { background: rgba(239,68,68,0.3); } 100% { background: transparent; } }
.flash-up { animation: flash-green 0.5s ease-out; }
.flash-down { animation: flash-red 0.5s ease-out; }
.clock { font-variant-numeric: tabular-nums; font-family: 'Courier New', monospace; }

/* ── Price Ticker ── */
.ticker-bar { background: var(--bg-primary); border-bottom: 1px solid var(--border); padding: 6px 0; overflow: hidden; white-space: nowrap; }
.ticker-scroll { display: flex; gap: 32px; animation: ticker-scroll 30s linear infinite; padding: 0 16px; }
.ticker-item { display: inline-flex; align-items: center; gap: 8px; font-size: 12px; font-variant-numeric: tabular-nums; }
.ticker-sym { font-weight: 700; color: var(--text-primary); }
.ticker-price { font-weight: 600; }
.ticker-chg { font-size: 11px; font-weight: 600; }
@keyframes ticker-scroll { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }

/* ── Alerts Panel ── */
.alerts-panel { position: fixed; top: 60px; right: 16px; z-index: 200; display: flex; flex-direction: column; gap: 8px; }
.alert-toast { padding: 12px 18px; border-radius: 6px; font-size: 12px; font-weight: 600; animation: slide-in 0.3s ease-out, fade-out 0.5s ease-in 7.5s forwards; max-width: 320px; border-left: 3px solid; }
.alert-trade { background: var(--blue-bg); border-color: var(--blue); color: var(--blue); }
.alert-tp { background: var(--green-bg); border-color: var(--green); color: var(--green); }
.alert-sl { background: var(--red-bg); border-color: var(--red); color: var(--red); }
@keyframes slide-in { from { transform: translateX(100%); opacity: 0; } to { transform: translateX(0); opacity: 1; } }
@keyframes fade-out { from { opacity: 1; } to { opacity: 0; display: none; } }

/* ── KPI Strip ── */
.kpi-strip { display: grid; grid-template-columns: repeat(6, 1fr); gap: 1px; background: var(--border); border-bottom: 1px solid var(--border); }
.kpi { background: var(--bg-secondary); padding: 16px 20px; text-align: center; }
.kpi-label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
.kpi-value { font-size: 22px; font-weight: 700; font-variant-numeric: tabular-nums; }
.kpi-sub { font-size: 11px; color: var(--text-secondary); margin-top: 4px; }

/* ── Main Layout ── */
.main { display: grid; grid-template-columns: 1fr 360px; gap: 0; min-height: calc(100vh - 56px - 73px); }
.main-content { padding: 16px; display: flex; flex-direction: column; gap: 16px; }
.sidebar { background: var(--bg-secondary); border-left: 1px solid var(--border); overflow-y: auto; max-height: calc(100vh - 129px); }

/* ── Cards ── */
.card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
.card-header { padding: 14px 18px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; }
.card-title { font-size: 13px; font-weight: 600; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; }
.card-badge { font-size: 11px; padding: 2px 8px; border-radius: 4px; }
.card-body { padding: 16px 18px; }

/* ── Chart ── */
.chart-container { position: relative; height: 280px; }

/* ── Equity Sparkline Row ── */
.equity-row { display: grid; grid-template-columns: 2fr 1fr; gap: 16px; }

/* ── Positions Table ── */
.pos-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.pos-table th { color: var(--text-muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border); font-weight: 500; }
.pos-table td { padding: 12px; border-bottom: 1px solid rgba(43,49,57,0.5); font-variant-numeric: tabular-nums; }
.pos-table tr:hover { background: rgba(30,35,41,0.5); }
.pos-tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }
.pos-long { background: var(--green-bg); color: var(--green); }
.pos-short { background: var(--red-bg); color: var(--red); }
.pnl-pos { color: var(--green); }
.pnl-neg { color: var(--red); }

/* ── Sidebar Sections ── */
.sb-section { border-bottom: 1px solid var(--border); }
.sb-header { padding: 14px 18px; font-size: 12px; font-weight: 600; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1px; display: flex; justify-content: space-between; align-items: center; }
.sb-body { padding: 0 18px 16px; }

/* ── Strategy Cards ── */
.strat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
.strat-chip { background: var(--bg-primary); border: 1px solid var(--border); border-radius: 6px; padding: 8px 10px; font-size: 11px; text-align: center; transition: all 0.2s; }
.strat-chip.active { border-color: var(--green-dim); color: var(--green); }

/* ── Signal Feed ── */
.signal-feed { max-height: 400px; overflow-y: auto; }
.signal-item { display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid rgba(43,49,57,0.4); }
.signal-item:last-child { border-bottom: none; }
.signal-left { display: flex; flex-direction: column; gap: 2px; }
.signal-sym { font-weight: 600; font-size: 13px; }
.signal-strat { font-size: 11px; color: var(--text-muted); }
.signal-score { font-size: 14px; font-weight: 700; }

/* ── Regime Badges ── */
.regime-row { display: flex; flex-wrap: wrap; gap: 6px; }
.regime-chip { padding: 5px 10px; border-radius: 6px; font-size: 11px; font-weight: 600; display: flex; align-items: center; gap: 6px; }
.regime-chip .sym { color: var(--text-primary); }
.r-bull { background: var(--green-bg); color: var(--green); }
.r-bear { background: var(--red-bg); color: var(--red); }
.r-range { background: var(--yellow-bg); color: var(--yellow); }
.r-chaos { background: var(--purple-bg); color: var(--purple); }

/* ── Performance Bars ── */
.perf-bar { margin-bottom: 12px; }
.perf-label { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px; }
.perf-label span:first-child { color: var(--text-secondary); }
.perf-label span:last-child { font-weight: 600; }
.perf-track { height: 6px; background: var(--bg-primary); border-radius: 3px; overflow: hidden; }
.perf-fill { height: 100%; border-radius: 3px; transition: width 0.5s ease; }

/* ── Trade Log ── */
.trade-log { max-height: 250px; overflow-y: auto; font-size: 12px; }
.trade-row { display: grid; grid-template-columns: 90px 1fr 60px 70px; padding: 8px 0; border-bottom: 1px solid rgba(43,49,57,0.3); align-items: center; }
.trade-time { color: var(--text-muted); font-variant-numeric: tabular-nums; }

/* ── Activity Log ── */
.activity-log { max-height: 200px; overflow-y: auto; font-size: 12px; }
.log-entry { padding: 6px 0; border-bottom: 1px solid rgba(43,49,57,0.2); display: flex; gap: 8px; }
.log-time { color: var(--text-muted); min-width: 65px; font-variant-numeric: tabular-nums; }
.log-msg { color: var(--text-secondary); }

/* ── Empty State ── */
.empty { padding: 30px; text-align: center; color: var(--text-muted); font-size: 13px; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--bg-tertiary); border-radius: 3px; }

/* ── Responsive ── */
@media (max-width: 1024px) {
  .main { grid-template-columns: 1fr; }
  .sidebar { max-height: none; border-left: none; border-top: 1px solid var(--border); }
  .kpi-strip { grid-template-columns: repeat(3, 1fr); }
  .equity-row { grid-template-columns: 1fr; }
}
</style>
</head>
<body>

<!-- ── Top Bar ── -->
<div class="topbar">
  <div class="topbar-left">
    <div class="logo">APEX<span>Trading Terminal v2.0</span></div>
    <span class="mode-pill" id="mode-pill">--</span>
  </div>
  <div class="topbar-center">
    <div><div class="label">Portfolio</div><div class="value" id="k-equity" style="color:var(--green)">--</div></div>
    <div><div class="label">Daily P&L</div><div class="value" id="k-pnl">--</div></div>
    <div><div class="label">Open P&L</div><div class="value" id="k-open-pnl">--</div></div>
    <div><div class="label">Win Rate</div><div class="value" id="k-wr">--</div></div>
    <div><div class="label">Drawdown</div><div class="value" id="k-dd" style="color:var(--red)">--</div></div>
  </div>
  <div class="topbar-right">
    <span><span class="pulse"></span> <span id="engine-status">Running</span></span>
    <span class="clock" id="clock">--:--:-- UTC</span>
  </div>
</div>

<!-- ── Price Ticker ── -->
<div class="ticker-bar">
  <div class="ticker-scroll" id="ticker-scroll"></div>
</div>

<!-- ── Alerts Panel ── -->
<div class="alerts-panel" id="alerts-panel"></div>

<!-- ── KPI Strip ── -->
<div class="kpi-strip">
  <div class="kpi">
    <div class="kpi-label">Portfolio Value</div>
    <div class="kpi-value" id="k-equity2" style="color:var(--green)">--</div>
    <div class="kpi-sub" id="k-equity-sub">USDT</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Daily P&L</div>
    <div class="kpi-value" id="k-pnl2">--</div>
    <div class="kpi-sub" id="k-pnl-sub">today</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Drawdown</div>
    <div class="kpi-value" id="k-dd2" style="color:var(--red)">--</div>
    <div class="kpi-sub" id="k-dd-sub">from peak</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Open Positions</div>
    <div class="kpi-value" id="k-pos" style="color:var(--blue)">0</div>
    <div class="kpi-sub">of 12 max</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Trades Today</div>
    <div class="kpi-value" id="k-trades" style="color:var(--blue)">0</div>
    <div class="kpi-sub">of 50 max</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Strategies</div>
    <div class="kpi-value" id="k-strats" style="color:var(--purple)">15</div>
    <div class="kpi-sub">active</div>
  </div>
</div>

<!-- ── Main Layout ── -->
<div class="main">
  <div class="main-content">

    <!-- Equity Chart + Performance -->
    <div class="equity-row">
      <div class="card">
        <div class="card-header">
          <span class="card-title">Equity Curve</span>
          <span id="eq-change" style="font-size:13px;font-weight:600"></span>
        </div>
        <div class="card-body">
          <div class="chart-container"><canvas id="equityChart"></canvas></div>
        </div>
      </div>
      <div class="card">
        <div class="card-header"><span class="card-title">Performance</span></div>
        <div class="card-body">
          <div class="perf-bar">
            <div class="perf-label"><span>Win Rate</span><span id="p-wr">50%</span></div>
            <div class="perf-track"><div class="perf-fill" id="p-wr-bar" style="width:50%;background:var(--green)"></div></div>
          </div>
          <div class="perf-bar">
            <div class="perf-label"><span>Profit Factor</span><span id="p-pf">0.00</span></div>
            <div class="perf-track"><div class="perf-fill" id="p-pf-bar" style="width:0%;background:var(--blue)"></div></div>
          </div>
          <div class="perf-bar">
            <div class="perf-label"><span>Sharpe Ratio</span><span id="p-sr">0.00</span></div>
            <div class="perf-track"><div class="perf-fill" id="p-sr-bar" style="width:0%;background:var(--purple)"></div></div>
          </div>
          <div class="perf-bar">
            <div class="perf-label"><span>Max Drawdown</span><span id="p-mdd">0.00%</span></div>
            <div class="perf-track"><div class="perf-fill" id="p-mdd-bar" style="width:0%;background:var(--red)"></div></div>
          </div>
          <div style="margin-top:16px;padding-top:12px;border-top:1px solid var(--border)">
            <div class="perf-label" style="margin-bottom:8px"><span>Total P&L (30d)</span><span id="p-total-pnl" style="color:var(--green)">$0.00</span></div>
            <div class="perf-label"><span>Total Trades (30d)</span><span id="p-total-trades">0</span></div>
          </div>
        </div>
      </div>
    </div>

    <!-- Open Positions -->
    <div class="card">
      <div class="card-header">
        <span class="card-title">Open Positions</span>
        <span class="card-badge" style="background:var(--blue-bg);color:var(--blue)" id="pos-count">0 Active</span>
      </div>
      <div id="positions-body">
        <div class="empty">No open positions &mdash; bot is scanning for opportunities...</div>
      </div>
    </div>

    <!-- PnL Distribution Chart -->
    <div class="card">
      <div class="card-header"><span class="card-title">P&L Distribution</span></div>
      <div class="card-body"><div class="chart-container" style="height:180px"><canvas id="pnlChart"></canvas></div></div>
    </div>

    <!-- Activity Log -->
    <div class="card">
      <div class="card-header"><span class="card-title">Activity Log</span></div>
      <div class="card-body">
        <div class="activity-log" id="activity-log">
          <div class="log-entry"><span class="log-time">--:--</span><span class="log-msg">System starting up...</span></div>
        </div>
      </div>
    </div>

  </div>

  <!-- ── Sidebar ── -->
  <div class="sidebar">

    <!-- Market Regimes -->
    <div class="sb-section">
      <div class="sb-header">Market Regimes</div>
      <div class="sb-body"><div class="regime-row" id="regime-container"><span class="empty" style="padding:10px">Scanning...</span></div></div>
    </div>

    <!-- Active Signals -->
    <div class="sb-section">
      <div class="sb-header">Live Signals <span class="card-badge" style="background:var(--yellow-bg);color:var(--yellow)" id="sig-count">0</span></div>
      <div class="sb-body">
        <div class="signal-feed" id="signal-feed"><div class="empty">Waiting for signals...</div></div>
      </div>
    </div>

    <!-- Strategies -->
    <div class="sb-section">
      <div class="sb-header">Strategies <span style="color:var(--green)">10 Active</span></div>
      <div class="sb-body"><div class="strat-grid" id="strat-grid"></div></div>
    </div>

    <!-- Fear & Greed -->
    <div class="sb-section">
      <div class="sb-header">Fear & Greed Index</div>
      <div class="sb-body" style="text-align:center">
        <div id="fg-value" style="font-size:42px;font-weight:700;color:var(--yellow)">--</div>
        <div id="fg-label" style="font-size:13px;color:var(--text-secondary);margin-top:4px">Loading...</div>
      </div>
    </div>

    <!-- Funding Rates -->
    <div class="sb-section">
      <div class="sb-header">Funding Rates</div>
      <div class="sb-body"><div id="funding-container"><span class="empty" style="padding:5px">Loading...</span></div></div>
    </div>

  </div>
</div>

<script>
async function api(ep) {
  try {
    const r = await fetch('/api/' + ep);
    return r.ok ? await r.json() : null;
  } catch(e) { return null; }
}
function $(id) { return document.getElementById(id); }
function fmt(n,d=2) { return n!=null&&!isNaN(n) ? Number(n).toFixed(d) : '--'; }
function fmtK(n) { if(!n||isNaN(n)) return '--'; return n>=1e6?'$'+(n/1e6).toFixed(2)+'M':n>=1e3?'$'+(n/1e3).toFixed(1)+'K':'$'+n.toFixed(2); }

// ── Init Strategies ──
const STRATS = ['trend_momentum','mean_reversion','breakout','smc','scalping','funding_rate','swing','oi_divergence','quant_momentum','stat_arb'];
const sg = $('strat-grid');
STRATS.forEach(s => { sg.innerHTML += `<div class="strat-chip active">${s.replace(/_/g,' ')}</div>`; });

// ── Charts ──
const eqCtx = $('equityChart').getContext('2d');
const equityData = [];
const equityChart = new Chart(eqCtx, {
  type: 'line',
  data: { datasets: [{ data: equityData, borderColor: '#0ecb81', backgroundColor: 'rgba(14,203,129,0.08)', fill: true, borderWidth: 2, pointRadius: 0, tension: 0.3 }] },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { type: 'time', time: { unit: 'minute', displayFormats: { minute: 'HH:mm' } }, grid: { color: 'rgba(43,49,57,0.5)' }, ticks: { color: '#5e6673', maxTicksLimit: 8 } },
      y: { grid: { color: 'rgba(43,49,57,0.5)' }, ticks: { color: '#5e6673', callback: v => '$'+v.toFixed(0) } }
    },
    interaction: { intersect: false, mode: 'index' }
  }
});

const pnlCtx = $('pnlChart').getContext('2d');
const pnlData = { labels: [], datasets: [{ data: [], backgroundColor: [], borderRadius: 3, barPercentage: 0.6 }] };
const pnlChart = new Chart(pnlCtx, {
  type: 'bar',
  data: pnlData,
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { grid: { display: false }, ticks: { color: '#5e6673', font: { size: 10 } } },
      y: { grid: { color: 'rgba(43,49,57,0.5)' }, ticks: { color: '#5e6673', callback: v => v+'%' } }
    }
  }
});

// ── Activity Log ──
const logEntries = [];
function addLog(msg) {
  const t = new Date().toLocaleTimeString('en-US', {hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false});
  logEntries.unshift({time:t, msg});
  if(logEntries.length > 50) logEntries.pop();
  const el = $('activity-log');
  el.innerHTML = logEntries.map(e => `<div class="log-entry"><span class="log-time">${e.time}</span><span class="log-msg">${e.msg}</span></div>`).join('');
}

// ── Clock ──
setInterval(() => {
  $('clock').textContent = new Date().toLocaleTimeString('en-US', {hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false});
}, 1000);

// ── Main Refresh ──
let prevEquity = 0;
async function refresh() {
  const [status, equity, risk, signals, positions, regimes, perf, fg, funding] = await Promise.all([
    api('status'), api('equity'), api('risk'), api('signals'),
    api('positions'), api('regimes'), api('performance'), api('fear-greed'), api('funding-rates')
  ]);

  // Mode
  if(status) {
    const m = status.mode||'paper';
    const pill = $('mode-pill');
    pill.textContent = m.toUpperCase();
    pill.className = 'mode-pill mode-'+m;
  }

  // KPIs
  if(equity) {
    const eq = equity.current_equity||0;
    $('k-equity').textContent = fmtK(eq);
    $('k-dd').textContent = fmt(equity.drawdown_pct)+'%';
    // Update equity chart
    equityData.push({x: new Date(), y: eq});
    if(equityData.length > 360) equityData.shift();
    equityChart.update('none');
    // Change indicator
    if(prevEquity && eq !== prevEquity) {
      const chg = ((eq - prevEquity)/prevEquity*100);
      $('eq-change').textContent = (chg>=0?'+':'')+chg.toFixed(3)+'%';
      $('eq-change').style.color = chg>=0?'var(--green)':'var(--red)';
    }
    prevEquity = eq;
  }

  if(risk) {
    const pnl = risk.daily_loss_pct||0;
    $('k-pnl').textContent = (pnl>=0?'+':'')+fmt(pnl)+'%';
    $('k-pnl').style.color = pnl>=0?'var(--green)':'var(--red)';
    $('k-pos').textContent = risk.positions_count||0;
    $('k-trades').textContent = risk.trades_today||0;
  }

  // Performance
  if(perf) {
    const wr = (perf.win_rate_30d||0.5)*100;
    $('k-wr').textContent = fmt(wr,1)+'%';
    $('k-wr').style.color = wr>=50?'var(--green)':'var(--red)';
    $('p-wr').textContent = fmt(wr,1)+'%';
    $('p-wr-bar').style.width = wr+'%';
    const pf = perf.profit_factor_30d||0;
    $('p-pf').textContent = fmt(pf);
    $('p-pf-bar').style.width = Math.min(pf/3*100,100)+'%';
    const sr = perf.sharpe_30d||0;
    $('p-sr').textContent = fmt(sr);
    $('p-sr-bar').style.width = Math.min(Math.abs(sr)/3*100,100)+'%';
    $('p-sr-bar').style.background = sr>=0?'var(--purple)':'var(--red)';
    $('p-total-pnl').textContent = '$'+fmt(perf.total_pnl_30d||0);
    $('p-total-pnl').style.color = (perf.total_pnl_30d||0)>=0?'var(--green)':'var(--red)';
    $('p-total-trades').textContent = perf.total_trades_30d||0;
  }

  if(equity) {
    const dd = equity.drawdown_pct||0;
    $('p-mdd').textContent = fmt(dd)+'%';
    $('p-mdd-bar').style.width = Math.min(dd/12*100,100)+'%';
  }

  // Positions
  if(positions) {
    const pp = positions.positions||[];
    $('pos-count').textContent = pp.length+' Active';
    if(pp.length > 0) {
      const pnls = [];
      let html = `<table class="pos-table"><tr><th>Symbol</th><th>Side</th><th>Entry</th><th>Current</th><th>Size</th><th>P&L</th><th>Strategy</th></tr>`;
      pp.forEach(p => {
        const pnl = p.direction==='long'
          ? ((p.current_price-p.entry_price)/p.entry_price*100)
          : ((p.entry_price-p.current_price)/p.entry_price*100);
        pnls.push({sym:p.symbol, pnl});
        html += `<tr>
          <td style="font-weight:600">${p.symbol}</td>
          <td><span class="pos-tag ${p.direction==='long'?'pos-long':'pos-short'}">${p.direction.toUpperCase()}</span></td>
          <td>$${fmt(p.entry_price,4)}</td>
          <td>$${fmt(p.current_price,4)}</td>
          <td>${fmt(p.amount,4)}</td>
          <td class="${pnl>=0?'pnl-pos':'pnl-neg'}">${pnl>=0?'+':''}${fmt(pnl)}%</td>
          <td style="color:var(--text-muted)">${p.strategy||'--'}</td>
        </tr>`;
      });
      html += '</table>';
      $('positions-body').innerHTML = html;
      // Update PnL chart
      pnlData.labels = pnls.map(p=>p.sym.replace('/USDT',''));
      pnlData.datasets[0].data = pnls.map(p=>+p.pnl.toFixed(2));
      pnlData.datasets[0].backgroundColor = pnls.map(p=>p.pnl>=0?'rgba(14,203,129,0.7)':'rgba(246,70,93,0.7)');
      pnlChart.update();
    } else {
      $('positions-body').innerHTML = '<div class="empty">No open positions &mdash; bot is scanning for opportunities...</div>';
      pnlData.labels=[]; pnlData.datasets[0].data=[]; pnlChart.update();
    }
  }

  // Signals
  if(signals) {
    const ss = signals.signals||[];
    $('sig-count').textContent = ss.length;
    if(ss.length > 0) {
      $('signal-feed').innerHTML = ss.map(s => `
        <div class="signal-item">
          <div class="signal-left">
            <span class="signal-sym">${s.symbol}</span>
            <span class="signal-strat">${s.strategy} &middot; ${s.timeframe||'4h'}</span>
          </div>
          <span class="signal-score" style="color:${s.direction==='long'?'var(--green)':'var(--red)'}">
            ${s.direction==='long'?'LONG':'SHORT'} ${Math.abs(s.score)}
          </span>
        </div>`).join('');
    } else {
      $('signal-feed').innerHTML = '<div class="empty">Waiting for signals...</div>';
    }
  }

  // Regimes
  if(regimes) {
    const rr = regimes.regimes||{};
    const entries = Object.entries(rr);
    if(entries.length > 0) {
      $('regime-container').innerHTML = entries.map(([sym, data]) => {
        const regime = data.regime||data||'RANGING';
        let cls = 'r-range';
        if(regime.includes('BULL')) cls='r-bull';
        else if(regime.includes('BEAR')) cls='r-bear';
        else if(regime==='CHAOS') cls='r-chaos';
        return `<div class="regime-chip ${cls}"><span class="sym">${sym.replace('/USDT','')}</span>${regime}</div>`;
      }).join('');
    }
  }

  // Fear & Greed
  if(fg) {
    const val = fg.value||fg.fear_greed_value||50;
    $('fg-value').textContent = val;
    const cls = fg.classification||fg.value_classification||(val<25?'Extreme Fear':val<45?'Fear':val<55?'Neutral':val<75?'Greed':'Extreme Greed');
    $('fg-label').textContent = cls;
    $('fg-value').style.color = val<25?'var(--red)':val<45?'var(--yellow)':val<55?'var(--text-secondary)':val<75?'var(--green)':'var(--green)';
  }

  // Funding
  if(funding) {
    const rates = funding.rates||funding.funding_rates||{};
    const entries = Object.entries(rates);
    if(entries.length > 0) {
      $('funding-container').innerHTML = entries.slice(0,10).map(([sym,data]) => {
        const rate = typeof data==='number'?data:(data.rate||0);
        const pct = (rate*100).toFixed(4);
        const color = rate>0?'var(--green)':rate<0?'var(--red)':'var(--text-secondary)';
        return `<div style="display:flex;justify-content:space-between;padding:4px 0;font-size:12px"><span>${sym.replace('/USDT','')}</span><span style="color:${color}">${pct}%</span></div>`;
      }).join('');
    }
  }
}

addLog('APEX Terminal v2.0 initialized');
addLog('Connecting to trading engine...');

// ── Price Ticker ──
const TICKER_SYMBOLS = ['BTC/USDT','ETH/USDT','SOL/USDT','BNB/USDT','XRP/USDT','DOGE/USDT','AVAX/USDT','LINK/USDT','DOT/USDT','ADA/USDT'];
let prevPrices = {};
function updateTicker(positions) {
  const prices = {};
  if(positions && positions.positions) {
    positions.positions.forEach(p => { prices[p.symbol] = p.current_price; });
  }
  const container = $('ticker-scroll');
  if(!container) return;
  let html = '';
  TICKER_SYMBOLS.forEach(sym => {
    const price = prices[sym] || prevPrices[sym] || 0;
    const prev = prevPrices[sym] || price;
    const chg = prev > 0 ? ((price - prev) / prev * 100) : 0;
    const color = chg >= 0 ? 'var(--green)' : 'var(--red)';
    const arrow = chg >= 0 ? '&#9650;' : '&#9660;';
    html += '<span class="ticker-item"><span class="ticker-sym">'+sym.replace('/USDT','')+'</span><span class="ticker-price" style="color:'+color+'">$'+fmt(price,price>100?0:2)+'</span><span class="ticker-chg" style="color:'+color+'">'+arrow+' '+fmt(Math.abs(chg),2)+'%</span></span>';
    prevPrices[sym] = price;
  });
  // Duplicate for seamless scroll
  container.innerHTML = html + html;
}

// ── WebSocket with auto-reconnect ──
let ws = null;
let wsRetryDelay = 1000;
function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(proto + '//' + location.host + '/ws');
  ws.onopen = () => {
    wsRetryDelay = 1000;
    addLog('WebSocket connected (real-time mode)');
    $('engine-status').textContent = 'Live';
  };
  ws.onmessage = (evt) => {
    try {
      const msg = JSON.parse(evt.data);
      if(msg.type === 'update' && msg.data) {
        handleWSUpdate(msg.data);
      } else if(msg.type === 'trade_alert') {
        showAlert(msg.event, msg.symbol);
      }
    } catch(e) {}
  };
  ws.onclose = () => {
    $('engine-status').textContent = 'Reconnecting...';
    setTimeout(connectWS, Math.min(wsRetryDelay, 16000));
    wsRetryDelay *= 2;
  };
  ws.onerror = () => { ws.close(); };
}

function handleWSUpdate(data) {
  // Top bar quick stats
  const eq = data.portfolio_value || data.equity || 0;
  if($('k-equity')) $('k-equity').textContent = fmtK(eq);
  if($('k-equity2')) $('k-equity2').textContent = fmtK(eq);
  const pnl = data.daily_pnl_pct || 0;
  if($('k-pnl')) { $('k-pnl').textContent = (pnl>=0?'+':'')+fmt(pnl)+'%'; $('k-pnl').style.color = pnl>=0?'var(--green)':'var(--red)'; }
  if($('k-pnl2')) { $('k-pnl2').textContent = (pnl>=0?'+':'')+fmt(pnl)+'%'; $('k-pnl2').style.color = pnl>=0?'var(--green)':'var(--red)'; }
  if($('k-open-pnl')) { const op = data.open_pnl||0; $('k-open-pnl').textContent = (op>=0?'+':'')+fmt(op); $('k-open-pnl').style.color = op>=0?'var(--green)':'var(--red)'; }
  const dd = data.drawdown_pct || 0;
  if($('k-dd')) $('k-dd').textContent = fmt(dd)+'%';
  if($('k-dd2')) $('k-dd2').textContent = fmt(dd)+'%';
  const wr = (data.win_rate || 0.5) * 100;
  if($('k-wr')) { $('k-wr').textContent = fmt(wr,0)+'%'; $('k-wr').style.color = wr>=50?'var(--green)':'var(--red)'; }
}

// ── Alerts ──
function showAlert(type, symbol) {
  const panel = $('alerts-panel');
  if(!panel) return;
  const cls = type.includes('close') ? 'alert-sl' : type.includes('tp') ? 'alert-tp' : 'alert-trade';
  const msg = type === 'position_closed' ? 'Position closed: '+symbol : 'New signal: '+symbol;
  const div = document.createElement('div');
  div.className = 'alert-toast ' + cls;
  div.textContent = msg;
  panel.appendChild(div);
  setTimeout(() => { div.remove(); }, 8000);
}

// Start WebSocket connection
connectWS();

// Fallback: REST polling every 2 seconds
refresh().then(() => addLog('Initial data loaded'));
setInterval(() => { refresh().then(() => {}); }, 2000);
// Ticker update every 1 second
setInterval(() => { api('positions').then(p => updateTicker(p)); }, 1000);
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
