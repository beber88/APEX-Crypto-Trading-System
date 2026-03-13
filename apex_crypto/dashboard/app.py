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
  --bg-primary:#0a0a0f;--bg-secondary:#111119;--bg-tertiary:#1a1a26;
  --bg-card:#0e0e18;--bg-card-header:#131320;
  --border:#1e1e30;--border-bright:#2a2a45;
  --text-primary:#d4d4dc;--text-secondary:#8888a0;--text-muted:#555570;
  --green:#00d68f;--green-bg:rgba(0,214,143,0.08);--green-dim:rgba(0,214,143,0.3);
  --red:#ff4757;--red-bg:rgba(255,71,87,0.08);--red-dim:rgba(255,71,87,0.3);
  --blue:#3b82f6;--blue-bg:rgba(59,130,246,0.1);
  --yellow:#f59e0b;--yellow-bg:rgba(245,158,11,0.1);
  --orange:#ff6b35;--purple:#a78bfa;--cyan:#22d3ee;
  --font-mono:'Consolas','SF Mono','Fira Code','Courier New',monospace;
}
*{margin:0;padding:0;box-sizing:border-box}
html,body{height:100%;overflow:hidden}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:var(--bg-primary);color:var(--text-primary);font-size:13px;line-height:1.4}
.tnum{font-variant-numeric:tabular-nums;font-family:var(--font-mono)}
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border-bright);border-radius:3px}
@keyframes flash-green{0%{background-color:rgba(0,214,143,0.3)}100%{background-color:transparent}}
@keyframes flash-red{0%{background-color:rgba(255,71,87,0.3)}100%{background-color:transparent}}
@keyframes pulse-dot{0%,100%{opacity:1}50%{opacity:0.3}}
@keyframes slide-in-right{from{transform:translateX(100%);opacity:0}to{transform:translateX(0);opacity:1}}
@keyframes slide-out-right{from{transform:translateX(0);opacity:1}to{transform:translateX(100%);opacity:0}}
@keyframes ticker-scroll{0%{transform:translateX(0)}100%{transform:translateX(-50%)}}
.flash-green{animation:flash-green .6s ease-out}
.flash-red{animation:flash-red .6s ease-out}
.topbar{background:var(--bg-secondary);border-bottom:1px solid var(--border);padding:0 20px;height:52px;display:flex;align-items:center;justify-content:space-between;z-index:100}
.topbar-left{display:flex;align-items:center;gap:16px}
.logo{font-size:15px;font-weight:800;color:var(--yellow);letter-spacing:2px;text-transform:uppercase;font-family:var(--font-mono)}
.live-dot{width:8px;height:8px;border-radius:50%;background:var(--green);display:inline-block;animation:pulse-dot 1.5s infinite;margin-right:4px;vertical-align:middle}
.status-badge{padding:3px 10px;border-radius:3px;font-size:11px;font-weight:700;letter-spacing:1px;font-family:var(--font-mono)}
.status-live{background:var(--green-bg);color:var(--green);border:1px solid var(--green-dim)}
.status-paper{background:var(--yellow-bg);color:var(--yellow);border:1px solid rgba(245,158,11,0.3)}
.topbar-center{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--text-secondary)}
.topbar-right{display:flex;align-items:center;gap:14px;font-size:12px;color:var(--text-secondary);font-family:var(--font-mono)}
.summary-strip{background:var(--bg-secondary);border-bottom:1px solid var(--border);padding:6px 20px;display:flex;align-items:center;gap:24px;font-size:12px;font-family:var(--font-mono);color:var(--text-secondary)}
.summary-strip .sep{color:var(--border-bright)}
.ticker-wrap{background:var(--bg-primary);border-bottom:1px solid var(--border);overflow:hidden;height:30px}
.ticker-track{display:flex;align-items:center;height:30px;white-space:nowrap;animation:ticker-scroll 60s linear infinite}
.ticker-item{display:inline-flex;align-items:center;gap:6px;padding:0 20px;font-size:12px;font-family:var(--font-mono);font-weight:600}
.ticker-item .sym{color:var(--text-primary)}
.ticker-item .chg-up{color:var(--green)}
.ticker-item .chg-down{color:var(--red)}
.shell{display:flex;flex-direction:column;height:calc(100vh - 52px - 28px - 30px)}
.grid-top{display:grid;grid-template-columns:1fr 1.6fr 1fr;gap:1px;background:var(--border);flex:1;min-height:0}
.grid-bottom{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1px;background:var(--border);flex:1;min-height:0}
.panel{background:var(--bg-card);display:flex;flex-direction:column;min-height:0;overflow:hidden}
.panel-hdr{background:var(--bg-card-header);padding:8px 14px;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--border);flex-shrink:0}
.panel-title{font-size:11px;font-weight:700;color:var(--text-secondary);text-transform:uppercase;letter-spacing:1px}
.panel-badge{font-size:10px;padding:2px 7px;border-radius:3px;font-weight:600;font-family:var(--font-mono)}
.panel-body{flex:1;overflow:auto;padding:10px 14px;min-height:0}
.panel-body-np{flex:1;overflow:auto;min-height:0}
.pos-tbl{width:100%;border-collapse:collapse;font-size:11px;font-family:var(--font-mono)}
.pos-tbl th{color:var(--text-muted);font-size:10px;text-transform:uppercase;letter-spacing:.5px;padding:7px 8px;text-align:left;border-bottom:1px solid var(--border);font-weight:500;position:sticky;top:0;background:var(--bg-card-header);z-index:1}
.pos-tbl td{padding:6px 8px;border-bottom:1px solid rgba(30,30,48,0.5);font-variant-numeric:tabular-nums;white-space:nowrap}
.pos-tbl tr:hover{background:rgba(30,30,48,0.6)}
.pos-tbl .row-profit{border-left:2px solid var(--green)}
.pos-tbl .row-loss{border-left:2px solid var(--red)}
.dir-long{color:var(--green);font-weight:700}
.dir-short{color:var(--red);font-weight:700}
.pnl-pos{color:var(--green);font-weight:600}
.pnl-neg{color:var(--red);font-weight:600}
.btn-close{background:var(--red-bg);color:var(--red);border:1px solid var(--red-dim);padding:2px 8px;border-radius:3px;font-size:10px;cursor:pointer;font-family:var(--font-mono);font-weight:600}
.btn-close:hover{background:var(--red);color:#fff}
.radar-item{display:grid;grid-template-columns:50px 1fr 45px;align-items:center;gap:8px;padding:4px 0;font-size:11px;font-family:var(--font-mono);cursor:pointer}
.radar-item:hover{background:rgba(30,30,48,0.4)}
.radar-sym{font-weight:600;color:var(--text-primary)}
.radar-bar-wrap{height:12px;background:var(--bg-primary);border-radius:2px;position:relative;overflow:hidden}
.radar-bar-center{position:absolute;left:50%;top:0;bottom:0;width:1px;background:var(--border-bright)}
.radar-bar{position:absolute;top:1px;bottom:1px;border-radius:1px;transition:width .5s ease,left .5s ease}
.radar-score{text-align:right;font-weight:600}
.radar-expand{padding:6px 0 6px 58px;font-size:10px;color:var(--text-muted);display:none}
.radar-expand.open{display:block}
.trades-tbl{width:100%;border-collapse:collapse;font-size:10px;font-family:var(--font-mono)}
.trades-tbl th{color:var(--text-muted);font-size:9px;text-transform:uppercase;padding:6px;text-align:left;border-bottom:1px solid var(--border);font-weight:500;position:sticky;top:0;background:var(--bg-card-header);z-index:1}
.trades-tbl td{padding:5px 6px;border-bottom:1px solid rgba(30,30,48,0.3);white-space:nowrap}
.trades-tbl .total-row{font-weight:700;border-top:1px solid var(--border);background:var(--bg-card-header)}
.gauge-wrap{text-align:center;margin-bottom:12px}
.gauge-outer{width:120px;height:60px;margin:0 auto;border-radius:60px 60px 0 0;background:var(--bg-primary);position:relative;overflow:hidden;border:1px solid var(--border);border-bottom:none}
.gauge-fill{position:absolute;bottom:0;left:0;right:0;background:var(--green);transition:height .5s ease,background .5s ease}
.gauge-label{font-size:18px;font-weight:700;font-family:var(--font-mono);margin-top:4px}
.gauge-sub{font-size:10px;color:var(--text-muted)}
.risk-bar{margin-bottom:10px}
.risk-bar-label{display:flex;justify-content:space-between;font-size:10px;margin-bottom:3px;font-family:var(--font-mono)}
.risk-bar-label span:first-child{color:var(--text-secondary)}
.risk-bar-label span:last-child{font-weight:600}
.risk-bar-track{height:5px;background:var(--bg-primary);border-radius:3px;overflow:hidden}
.risk-bar-fill{height:100%;border-radius:3px;transition:width .5s ease}
.fg-display{text-align:center;margin-bottom:12px;padding:8px 0}
.fg-number{font-size:36px;font-weight:800;font-family:var(--font-mono);line-height:1}
.fg-label{font-size:11px;color:var(--text-secondary);margin-top:4px}
.regime-grid{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:10px}
.regime-tag{padding:3px 7px;border-radius:3px;font-size:10px;font-weight:700;font-family:var(--font-mono)}
.regime-bull{background:var(--green-bg);color:var(--green)}
.regime-bear{background:var(--red-bg);color:var(--red)}
.regime-range{background:var(--yellow-bg);color:var(--yellow)}
.regime-chaos{background:rgba(167,139,250,0.1);color:var(--purple)}
.funding-row{display:flex;justify-content:space-between;padding:3px 0;font-size:11px;font-family:var(--font-mono)}
.news-item{padding:5px 0;border-bottom:1px solid rgba(30,30,48,0.3);font-size:11px}
.news-item:last-child{border-bottom:none}
.news-time{color:var(--text-muted);font-size:10px;font-family:var(--font-mono)}
.section-label{font-size:10px;font-weight:600;color:var(--text-muted);text-transform:uppercase;letter-spacing:.5px;margin:10px 0 6px;padding-bottom:4px;border-bottom:1px solid var(--border)}
.section-label:first-child{margin-top:0}
.alerts-panel{position:fixed;top:90px;right:16px;z-index:200;display:flex;flex-direction:column;gap:6px;pointer-events:none;max-width:340px}
.alert-toast{background:var(--bg-secondary);border:1px solid var(--border);border-radius:6px;padding:10px 14px;font-size:12px;font-family:var(--font-mono);pointer-events:auto;animation:slide-in-right .3s ease-out;box-shadow:0 4px 20px rgba(0,0,0,0.5)}
.alert-toast.dismiss{animation:slide-out-right .3s ease-in forwards}
.alert-trade{border-left:3px solid var(--blue)}
.alert-tp{border-left:3px solid var(--green)}
.alert-sl{border-left:3px solid var(--red)}
.alert-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px}
.alert-type{font-weight:700;font-size:10px;text-transform:uppercase;letter-spacing:.5px}
.alert-time{color:var(--text-muted);font-size:10px}
.alert-msg{color:var(--text-secondary)}
.empty{padding:20px;text-align:center;color:var(--text-muted);font-size:12px}
.chart-wrap{position:relative;width:100%;height:100%}
@media(max-width:1200px){.grid-top{grid-template-columns:1fr 1fr}.grid-bottom{grid-template-columns:1fr 1fr}}
@media(max-width:768px){.grid-top,.grid-bottom{grid-template-columns:1fr}.topbar-center{display:none}}
</style>
</head>
<body>
<div class="topbar">
  <div class="topbar-left">
    <span class="logo">APEX TRADING SYSTEM</span>
    <span><span class="live-dot" id="live-dot"></span><span id="conn-status" style="font-size:11px;font-weight:700;color:var(--green)">LIVE</span></span>
    <span class="status-badge status-paper" id="mode-badge">PAPER</span>
  </div>
  <div class="topbar-center">
    <span>Portfolio: <strong class="tnum" id="tb-equity">$0.00</strong></span>
    <span id="tb-equity-chg" class="tnum" style="color:var(--green)">--</span>
  </div>
  <div class="topbar-right"><span class="tnum" id="utc-clock">00:00:00 UTC</span></div>
</div>
<div class="summary-strip">
  <span>Open P&amp;L: <strong class="tnum" id="ss-open-pnl">$0.00</strong></span><span class="sep">|</span>
  <span>Closed P&amp;L: <strong class="tnum" id="ss-closed-pnl">$0.00</strong></span><span class="sep">|</span>
  <span>Win Rate: <strong class="tnum" id="ss-winrate">0%</strong></span><span class="sep">|</span>
  <span>Drawdown: <strong class="tnum" id="ss-drawdown">0.00%</strong></span><span class="sep">|</span>
  <span>Positions: <strong class="tnum" id="ss-pos-count">0/8</strong></span><span class="sep">|</span>
  <span>Trades: <strong class="tnum" id="ss-trade-count">0/25</strong></span>
</div>
<div class="ticker-wrap"><div class="ticker-track" id="ticker-track"></div></div>
<div class="shell">
  <div class="grid-top">
    <div class="panel">
      <div class="panel-hdr"><span class="panel-title">Live Equity Curve</span><span class="panel-badge tnum" style="background:var(--blue-bg);color:var(--blue)" id="eq-value">$0.00</span></div>
      <div class="panel-body" style="padding:8px"><div class="chart-wrap"><canvas id="equityChart"></canvas></div></div>
    </div>
    <div class="panel">
      <div class="panel-hdr"><span class="panel-title">Open Positions</span><span class="panel-badge tnum" style="background:var(--green-bg);color:var(--green)" id="pos-badge">0 active</span></div>
      <div class="panel-body-np" id="positions-wrap"><div class="empty">No open positions -- scanning...</div></div>
    </div>
    <div class="panel">
      <div class="panel-hdr"><span class="panel-title">Signal Radar</span><span class="panel-badge tnum" style="background:var(--yellow-bg);color:var(--yellow)" id="sig-badge">0 signals</span></div>
      <div class="panel-body" id="radar-body"><div class="empty">Scanning assets...</div></div>
    </div>
  </div>
  <div class="grid-bottom">
    <div class="panel">
      <div class="panel-hdr"><span class="panel-title">Recent Trades</span><span class="panel-badge tnum" style="background:var(--blue-bg);color:var(--blue)" id="trades-badge">0 trades</span></div>
      <div class="panel-body-np" id="trades-wrap"><div class="empty">No closed trades yet</div></div>
    </div>
    <div class="panel">
      <div class="panel-hdr"><span class="panel-title">Risk Dashboard</span><span class="panel-badge" style="background:var(--green-bg);color:var(--green)" id="risk-status">NORMAL</span></div>
      <div class="panel-body" id="risk-body">
        <div class="gauge-wrap"><div class="gauge-outer"><div class="gauge-fill" id="dd-gauge-fill" style="height:0%"></div></div><div class="gauge-label tnum" id="dd-gauge-val">0.00%</div><div class="gauge-sub">Current Drawdown (max 12%)</div></div>
        <div style="height:100px;margin-bottom:10px"><canvas id="dailyPnlChart"></canvas></div>
        <div class="risk-bar"><div class="risk-bar-label"><span>Daily Loss</span><span class="tnum" id="rb-loss">$0 / $300</span></div><div class="risk-bar-track"><div class="risk-bar-fill" id="rb-loss-fill" style="width:0%;background:var(--red)"></div></div></div>
        <div class="risk-bar"><div class="risk-bar-label"><span>Positions</span><span class="tnum" id="rb-pos">0 / 8</span></div><div class="risk-bar-track"><div class="risk-bar-fill" id="rb-pos-fill" style="width:0%;background:var(--blue)"></div></div></div>
        <div class="risk-bar"><div class="risk-bar-label"><span>Trades Today</span><span class="tnum" id="rb-trades">0 / 25</span></div><div class="risk-bar-track"><div class="risk-bar-fill" id="rb-trades-fill" style="width:0%;background:var(--purple)"></div></div></div>
      </div>
    </div>
    <div class="panel">
      <div class="panel-hdr"><span class="panel-title">Market Intelligence</span></div>
      <div class="panel-body" id="intel-body">
        <div class="fg-display"><div class="fg-number tnum" id="fg-value" style="color:var(--yellow)">--</div><div class="fg-label" id="fg-label">Loading...</div></div>
        <div class="section-label">Asset Regimes</div>
        <div class="regime-grid" id="regime-grid"><span style="color:var(--text-muted);font-size:11px">Loading...</span></div>
        <div class="section-label">Extreme Funding Rates</div>
        <div id="funding-list"><span style="color:var(--text-muted);font-size:11px">Loading...</span></div>
        <div class="section-label">News Headlines</div>
        <div id="news-list"><span style="color:var(--text-muted);font-size:11px">Loading...</span></div>
      </div>
    </div>
  </div>
</div>
<div class="alerts-panel" id="alerts-panel"></div>

<script>
function $(id){return document.getElementById(id)}
function fmt(n,d){d=d==null?2:d;return n!=null&&!isNaN(n)?Number(n).toFixed(d):'--'}
function fmtUSD(n){if(n==null||isNaN(n))return'$0.00';var a=Math.abs(n);if(a>=1e6)return(n<0?'-':'')+'$'+(a/1e6).toFixed(2)+'M';if(a>=1e4)return(n<0?'-':'')+'$'+(a/1e3).toFixed(1)+'K';return(n<0?'-$':'$')+a.toFixed(2)}
function fmtPrice(n){if(n==null||isNaN(n))return'--';if(n>=1000)return n.toLocaleString('en-US',{minimumFractionDigits:0,maximumFractionDigits:0});if(n>=1)return n.toFixed(2);return n.toFixed(4)}
function pnlColor(v){return v>=0?'var(--green)':'var(--red)'}
function pnlSign(v){return v>=0?'+':''}
async function api(ep){try{var r=await fetch('/api/'+ep);return r.ok?await r.json():null}catch(e){return null}}
async function postApi(ep){try{var r=await fetch('/api/'+ep,{method:'POST'});return r.ok?await r.json():null}catch(e){return null}}

var prevValues={};
function flashCell(el,nv,key){if(!el)return;var old=prevValues[key];prevValues[key]=nv;if(old!=null&&old!==nv){el.classList.remove('flash-green','flash-red');void el.offsetWidth;el.classList.add(nv>old?'flash-green':'flash-red')}}

function updateClock(){var now=new Date();$('utc-clock').textContent=String(now.getUTCHours()).padStart(2,'0')+':'+String(now.getUTCMinutes()).padStart(2,'0')+':'+String(now.getUTCSeconds()).padStart(2,'0')+' UTC'}
setInterval(updateClock,1000);updateClock();

function showAlert(type,msg){var p=$('alerts-panel'),t=document.createElement('div'),cls=type==='tp'?'alert-tp':type==='sl'?'alert-sl':'alert-trade',label=type==='tp'?'TP HIT':type==='sl'?'SL HIT':'NEW TRADE',now=new Date(),ts=String(now.getUTCHours()).padStart(2,'0')+':'+String(now.getUTCMinutes()).padStart(2,'0')+':'+String(now.getUTCSeconds()).padStart(2,'0');t.className='alert-toast '+cls;t.innerHTML='<div class="alert-header"><span class="alert-type" style="color:'+(type==='tp'?'var(--green)':type==='sl'?'var(--red)':'var(--blue)')+'">'+label+'</span><span class="alert-time">'+ts+'</span></div><div class="alert-msg">'+msg+'</div>';p.prepend(t);setTimeout(function(){t.classList.add('dismiss');setTimeout(function(){t.remove()},300)},8000)}

var TICKER_ASSETS=['BTC/USDT','ETH/USDT','SOL/USDT','BNB/USDT','XRP/USDT','ADA/USDT','AVAX/USDT','DOGE/USDT','DOT/USDT','MATIC/USDT','LINK/USDT','UNI/USDT','ATOM/USDT','LTC/USDT','NEAR/USDT'];
var tickerPrices={},tickerChanges={};
function updateTicker(){var track=$('ticker-track'),items='';TICKER_ASSETS.forEach(function(a){var sym=a.replace('/USDT',''),price=tickerPrices[a],chg=tickerChanges[a],ps=price!=null?fmtPrice(price):'--',cs=chg!=null?(pnlSign(chg)+fmt(chg,2)+'%'):'--',cc=chg>=0?'chg-up':'chg-down',ar=chg>=0?'\u25B2':'\u25BC';items+='<span class="ticker-item"><span class="sym">'+sym+'</span> <span class="tnum">$'+ps+'</span> <span class="'+cc+' tnum">'+ar+cs+'</span></span>'});track.innerHTML=items+items}

var equityData=[],baselineValue=null,equityChart=null;
function initEquityChart(){var ctx=$('equityChart').getContext('2d');equityChart=new Chart(ctx,{type:'line',data:{datasets:[{label:'Portfolio',data:equityData,borderColor:'#00d68f',backgroundColor:'rgba(0,214,143,0.06)',fill:true,borderWidth:2,pointRadius:0,tension:0.3,order:1},{label:'Baseline',data:[],borderColor:'rgba(136,136,160,0.3)',borderWidth:1,borderDash:[5,5],pointRadius:0,fill:false,order:2},{label:'Zero',data:[],borderColor:'rgba(255,71,87,0.2)',borderWidth:1,borderDash:[3,3],pointRadius:0,fill:false,order:3}]},options:{responsive:true,maintainAspectRatio:false,animation:{duration:0},plugins:{legend:{display:false},tooltip:{mode:'index',intersect:false,backgroundColor:'rgba(14,14,24,0.9)',titleFont:{family:'Consolas,monospace',size:11},bodyFont:{family:'Consolas,monospace',size:11},callbacks:{label:function(c){return c.dataset.label+': $'+fmt(c.parsed.y,2)}}}},scales:{x:{type:'time',time:{unit:'hour',displayFormats:{hour:'HH:mm'}},grid:{color:'rgba(30,30,48,0.5)'},ticks:{color:'#555570',maxTicksLimit:8,font:{size:10,family:'Consolas,monospace'}}},y:{grid:{color:'rgba(30,30,48,0.5)'},ticks:{color:'#555570',font:{size:10,family:'Consolas,monospace'},callback:function(v){return'$'+v.toFixed(0)}}}},interaction:{intersect:false,mode:'index'}}})}

var dailyPnlChart=null;
function initDailyPnlChart(){var ctx=$('dailyPnlChart').getContext('2d');dailyPnlChart=new Chart(ctx,{type:'bar',data:{labels:[],datasets:[{data:[],backgroundColor:[],borderRadius:2,barPercentage:0.7}]},options:{responsive:true,maintainAspectRatio:false,animation:{duration:0},plugins:{legend:{display:false}},scales:{x:{display:false},y:{grid:{color:'rgba(30,30,48,0.4)'},ticks:{color:'#555570',font:{size:9,family:'Consolas,monospace'},callback:function(v){return'$'+v}}}}}})}

var ws=null,wsReconnectDelay=1000,wsMaxDelay=30000,wsConnected=false,usingPolling=false;
function connectWS(){var proto=location.protocol==='https:'?'wss:':'ws:';try{ws=new WebSocket(proto+'//'+location.host+'/ws')}catch(e){startPolling();return}ws.onopen=function(){wsConnected=true;wsReconnectDelay=1000;usingPolling=false;$('conn-status').textContent='LIVE';$('conn-status').style.color='var(--green)';$('live-dot').style.background='var(--green)'};ws.onmessage=function(e){try{handleWSData(JSON.parse(e.data))}catch(x){}};ws.onclose=function(){wsConnected=false;$('conn-status').textContent='RECONNECTING';$('conn-status').style.color='var(--yellow)';$('live-dot').style.background='var(--yellow)';setTimeout(function(){wsReconnectDelay=Math.min(wsReconnectDelay*2,wsMaxDelay);connectWS()},wsReconnectDelay);if(!usingPolling)startPolling()};ws.onerror=function(){if(ws)ws.close()}}
function handleWSData(d){if(d.type==='positions')updatePositions(d.payload);if(d.type==='equity')updateEquity(d.payload);if(d.type==='signals')updateSignals(d.payload);if(d.type==='risk')updateRisk(d.payload);if(d.type==='alert')showAlert(d.alert_type||'trade',d.message||'');if(d.type==='ticker')updateTickerData(d.payload)}

var pollFI=null,pollMI=null,pollSI=null;
function startPolling(){if(usingPolling)return;usingPolling=true;$('conn-status').textContent='POLLING';$('conn-status').style.color='var(--orange)';$('live-dot').style.background='var(--orange)';if(pollFI)clearInterval(pollFI);pollFI=setInterval(pollFast,1000);if(pollMI)clearInterval(pollMI);pollMI=setInterval(pollMedium,2000);if(pollSI)clearInterval(pollSI);pollSI=setInterval(pollSlow,30000);pollFast();pollMedium();pollSlow()}
async function pollFast(){var p=await api('positions');if(p)updatePositions(p);var e=await api('equity');if(e)updateEquity(e)}
async function pollMedium(){var r=await Promise.all([api('status'),api('risk')]);if(r[0])updateStatus(r[0]);if(r[1])updateRisk(r[1])}
async function pollSlow(){var r=await Promise.all([api('signals'),api('regimes'),api('fear-greed'),api('funding-rates')]);if(r[0])updateSignals(r[0]);if(r[1])updateRegimes(r[1]);if(r[2])updateFearGreed(r[2]);if(r[3])updateFunding(r[3])}

var startEquity=null;
function updateStatus(d){var m=(d.mode||'paper').toUpperCase();$('mode-badge').textContent=m;$('mode-badge').className='status-badge '+(m==='LIVE'?'status-live':'status-paper')}

function updateEquity(d){
  var eq=d.current_equity||0;
  if(baselineValue==null&&eq>0)baselineValue=eq;
  if(startEquity==null&&eq>0)startEquity=eq;
  $('tb-equity').textContent=fmtUSD(eq);flashCell($('tb-equity'),eq,'tb-eq');$('eq-value').textContent=fmtUSD(eq);
  if(startEquity&&startEquity>0){var ca=eq-startEquity,cp=(ca/startEquity)*100,ar=ca>=0?'\u25B2':'\u25BC',el=$('tb-equity-chg');el.textContent=ar+' '+pnlSign(ca)+fmtUSD(Math.abs(ca))+' ('+pnlSign(cp)+fmt(cp,2)+'% today)';el.style.color=pnlColor(ca)}
  var dd=d.drawdown_pct||0;$('ss-drawdown').textContent=fmt(dd,2)+'%';$('ss-drawdown').style.color=dd>5?'var(--red)':dd>2?'var(--yellow)':'var(--text-secondary)';
  equityData.push({x:new Date(),y:eq});if(equityData.length>2000)equityData.shift();
  if(equityChart){if(baselineValue!=null&&equityData.length>=2){var f=equityData[0].x,l=equityData[equityData.length-1].x;equityChart.data.datasets[1].data=[{x:f,y:baselineValue},{x:l,y:baselineValue}]}equityChart.update('none')}
}

function updatePositions(d){
  var pp=d.positions||[];$('pos-badge').textContent=pp.length+' active';$('ss-pos-count').textContent=pp.length+'/8';
  var openPnl=0;pp.forEach(function(p){openPnl+=(p.direction==='long'?(p.current_price-p.entry_price):(p.entry_price-p.current_price))*(p.amount||1)});
  $('ss-open-pnl').textContent=pnlSign(openPnl)+fmtUSD(Math.abs(openPnl));$('ss-open-pnl').style.color=pnlColor(openPnl);
  pp.forEach(function(p){if(p.symbol&&p.current_price){tickerPrices[p.symbol]=p.current_price;if(p.entry_price)tickerChanges[p.symbol]=((p.current_price-p.entry_price)/p.entry_price)*100}});updateTicker();
  if(!pp.length){$('positions-wrap').innerHTML='<div class="empty">No open positions -- scanning...</div>';return}
  pp.sort(function(a,b){var pA=a.direction==='long'?(a.current_price-a.entry_price)/a.entry_price:(a.entry_price-a.current_price)/a.entry_price;var pB=b.direction==='long'?(b.current_price-b.entry_price)/b.entry_price:(b.entry_price-b.current_price)/b.entry_price;return pB-pA});
  var h='<table class="pos-tbl"><thead><tr><th>Symbol</th><th>Dir</th><th>Entry</th><th>Current</th><th>P&amp;L$</th><th>P&amp;L%</th><th>SL</th><th>TP1</th><th>Age</th><th>Strategy</th><th></th></tr></thead><tbody>';
  pp.forEach(function(p){
    var pP=p.direction==='long'?((p.current_price-p.entry_price)/p.entry_price)*100:((p.entry_price-p.current_price)/p.entry_price)*100;
    var pA=p.direction==='long'?(p.current_price-p.entry_price)*(p.amount||1):(p.entry_price-p.current_price)*(p.amount||1);
    var rc=pP>=0?'row-profit':'row-loss',pc=pP>=0?'pnl-pos':'pnl-neg',dc=p.direction==='long'?'dir-long':'dir-short';
    var sym=p.symbol||'--',ss=sym.replace('/USDT',''),age='--';
    if(p.entry_time||p.opened_at){var ot=new Date(p.entry_time||p.opened_at),dm=Date.now()-ot.getTime(),dh=Math.floor(dm/3600000),dmin=Math.floor((dm%3600000)/60000);age=dh>0?dh+'h '+dmin+'m':dmin+'m'}
    var sl=p.stop_loss?fmtPrice(p.stop_loss):'--',tp=p.take_profit_1||p.take_profit||p.tp1;tp=tp?fmtPrice(tp):'--';var ci='pnl-'+ss;
    h+='<tr class="'+rc+'"><td style="font-weight:700">'+ss+'</td><td class="'+dc+'">'+(p.direction||'--').toUpperCase()+'</td><td class="tnum">'+fmtPrice(p.entry_price)+'</td><td class="tnum">'+fmtPrice(p.current_price)+'</td><td class="'+pc+' tnum" id="'+ci+'-a">'+pnlSign(pA)+fmtUSD(Math.abs(pA))+'</td><td class="'+pc+' tnum" id="'+ci+'-p">'+pnlSign(pP)+fmt(pP,2)+'%</td><td class="tnum" style="color:var(--red-dim)">'+sl+'</td><td class="tnum" style="color:var(--green-dim)">'+tp+'</td><td class="tnum" style="color:var(--text-muted)">'+age+'</td><td style="color:var(--text-muted);font-size:10px">'+(p.strategy||'--')+'</td><td><button class="btn-close" onclick="closePosition(\''+sym+'\')">Close</button></td></tr>';
    setTimeout(function(){flashCell($(ci+'-a'),pA,ci+'-a');flashCell($(ci+'-p'),pP,ci+'-p')},50);
  });
  h+='</tbody></table>';$('positions-wrap').innerHTML=h;
}

async function closePosition(s){var r=await postApi('close/'+encodeURIComponent(s));if(r)showAlert('trade','Closing: '+s)}

function updateSignals(d){
  var ss=d.signals||[];$('sig-badge').textContent=ss.length+' signals';
  if(!ss.length){$('radar-body').innerHTML='<div class="empty">No active signals</div>';return}
  var h='';ss.forEach(function(s,i){
    var sym=(s.symbol||'').replace('/USDT',''),score=s.score||0,dir=s.direction||(score>=0?'long':'short');
    var bc=dir==='long'?'var(--green)':'var(--red)',as=Math.min(Math.abs(score),100),bw=(as/100)*50;
    var bl=dir==='long'?'50%':(50-bw)+'%',sc=dir==='long'?'pnl-pos':'pnl-neg',eid='re-'+i;
    h+='<div class="radar-item" onclick="toggleRadar(\''+eid+'\')"><span class="radar-sym">'+sym+'</span><div class="radar-bar-wrap"><div class="radar-bar-center"></div><div class="radar-bar" style="left:'+bl+';width:'+bw+'%;background:'+bc+'"></div></div><span class="radar-score '+sc+'">'+pnlSign(score)+Math.abs(score)+'</span></div><div class="radar-expand" id="'+eid+'">Strategy: '+(s.strategy||'--')+' | TF: '+(s.timeframe||'4h')+' | Conf: '+(s.confidence||'--')+'</div>';
  });$('radar-body').innerHTML=h;
}
function toggleRadar(id){var el=$(id);if(el)el.classList.toggle('open')}

function updateRisk(d){
  var dd=d.drawdown_pct||d.daily_loss_pct||0,pc=d.positions_count||0,tt=d.trades_today||0,dl=Math.abs(d.daily_loss_pct||0);
  var dp=Math.min(Math.abs(dd),12),gh=(dp/12)*100,gf=$('dd-gauge-fill');
  gf.style.height=gh+'%';gf.style.background=dp>8?'var(--red)':dp>4?'var(--yellow)':'var(--green)';
  $('dd-gauge-val').textContent=fmt(Math.abs(dd),2)+'%';$('dd-gauge-val').style.color=dp>8?'var(--red)':dp>4?'var(--yellow)':'var(--green)';
  var re=$('risk-status');if(dp>8){re.textContent='CRITICAL';re.style.background='var(--red-bg)';re.style.color='var(--red)'}else if(dp>4){re.textContent='ELEVATED';re.style.background='var(--yellow-bg)';re.style.color='var(--yellow)'}else{re.textContent='NORMAL';re.style.background='var(--green-bg)';re.style.color='var(--green)'}
  var dla=dl*(startEquity||10000)/100;$('rb-loss').textContent='$'+fmt(dla,0)+' / $300';$('rb-loss-fill').style.width=Math.min(dla/300*100,100)+'%';
  $('rb-pos').textContent=pc+' / 8';$('rb-pos-fill').style.width=(pc/8*100)+'%';$('ss-pos-count').textContent=pc+'/8';
  $('rb-trades').textContent=tt+' / 25';$('rb-trades-fill').style.width=(tt/25*100)+'%';$('ss-trade-count').textContent=tt+'/25';
  var wr=d.win_rate||0;$('ss-winrate').textContent=fmt(wr*100,0)+'%';
}

function updateRegimes(d){var rr=d.regimes||{},ee=Object.entries(rr);if(!ee.length)return;$('regime-grid').innerHTML=ee.map(function(e){var sym=e[0].replace('/USDT',''),r=typeof e[1]==='string'?e[1]:(e[1].regime||'RANGING'),c='regime-range';if(r.indexOf('BULL')>=0)c='regime-bull';else if(r.indexOf('BEAR')>=0)c='regime-bear';else if(r==='CHAOS')c='regime-chaos';return'<span class="regime-tag '+c+'">'+sym+' '+r+'</span>'}).join('')}

function updateFearGreed(d){var v=d.value||d.fear_greed_value||50,c=d.classification||d.value_classification||(v<25?'Extreme Fear':v<45?'Fear':v<55?'Neutral':v<75?'Greed':'Extreme Greed');$('fg-value').textContent=v;$('fg-label').textContent=c;$('fg-value').style.color=v<25?'var(--red)':v<45?'var(--orange)':v<55?'var(--text-secondary)':v<75?'var(--green)':'var(--cyan)'}

function updateFunding(d){var rates=d.rates||d.funding_rates||{},ee=Object.entries(rates);if(!ee.length)return;ee.sort(function(a,b){var ra=typeof a[1]==='number'?Math.abs(a[1]):Math.abs(a[1].rate||0),rb=typeof b[1]==='number'?Math.abs(b[1]):Math.abs(b[1].rate||0);return rb-ra});$('funding-list').innerHTML=ee.slice(0,3).map(function(e){var sym=e[0].replace('/USDT',''),rate=typeof e[1]==='number'?e[1]:(e[1].rate||0),pct=(rate*100).toFixed(4),color=rate>0.0001?'var(--green)':rate<-0.0001?'var(--red)':'var(--text-muted)';return'<div class="funding-row"><span>'+sym+'</span><span style="color:'+color+'">'+pct+'%</span></div>'}).join('')}

function updateTickerData(d){if(!d)return;Object.keys(d).forEach(function(s){tickerPrices[s]=d[s].price;tickerChanges[s]=d[s].change||0});updateTicker()}

function updateTrades(d){
  var trades=d.trades||[];$('trades-badge').textContent=trades.length+' trades';
  if(!trades.length){$('trades-wrap').innerHTML='<div class="empty">No closed trades yet</div>';return}
  var tP=0,tR=0;trades.forEach(function(t){tP+=(t.pnl||0);tR+=(t.r_multiple||0)});
  $('ss-closed-pnl').textContent=pnlSign(tP)+fmtUSD(Math.abs(tP));$('ss-closed-pnl').style.color=pnlColor(tP);
  var h='<table class="trades-tbl"><thead><tr><th>Time</th><th>Symbol</th><th>Dir</th><th>Entry</th><th>Exit</th><th>P&amp;L</th><th>R-Mult</th><th>Strategy</th></tr></thead><tbody>';
  trades.slice(0,50).forEach(function(t){var pnl=t.pnl||0,pc=pnl>=0?'pnl-pos':'pnl-neg',ts=t.closed_at?new Date(t.closed_at).toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit',hour12:false}):'--',sym=(t.symbol||'--').replace('/USDT',''),dc=t.direction==='long'?'dir-long':'dir-short';h+='<tr><td style="color:var(--text-muted)">'+ts+'</td><td style="font-weight:600">'+sym+'</td><td class="'+dc+'">'+(t.direction||'--').toUpperCase()+'</td><td class="tnum">'+fmtPrice(t.entry_price)+'</td><td class="tnum">'+fmtPrice(t.exit_price)+'</td><td class="'+pc+' tnum">'+pnlSign(pnl)+fmtUSD(Math.abs(pnl))+'</td><td class="tnum" style="color:'+pnlColor(t.r_multiple||0)+'">'+pnlSign(t.r_multiple||0)+fmt(t.r_multiple||0,2)+'R</td><td style="color:var(--text-muted)">'+(t.strategy||'--')+'</td></tr>'});
  h+='<tr class="total-row"><td colspan="5" style="text-align:right">TOTAL</td><td class="'+(tP>=0?'pnl-pos':'pnl-neg')+' tnum">'+pnlSign(tP)+fmtUSD(Math.abs(tP))+'</td><td class="tnum" style="color:'+pnlColor(tR)+'">'+pnlSign(tR)+fmt(tR,2)+'R</td><td></td></tr></tbody></table>';
  $('trades-wrap').innerHTML=h;
  if(dailyPnlChart&&trades.length>0){var dm={};trades.forEach(function(t){var k=t.closed_at?t.closed_at.substring(0,10):'x';if(!dm[k])dm[k]=0;dm[k]+=(t.pnl||0)});var days=Object.keys(dm).sort().slice(-30);dailyPnlChart.data.labels=days.map(function(d){return d.substring(5)});dailyPnlChart.data.datasets[0].data=days.map(function(d){return+dm[d].toFixed(2)});dailyPnlChart.data.datasets[0].backgroundColor=days.map(function(d){return dm[d]>=0?'rgba(0,214,143,0.6)':'rgba(255,71,87,0.6)'});dailyPnlChart.update('none')}
}

initEquityChart();initDailyPnlChart();updateTicker();connectWS();
setTimeout(function(){if(!wsConnected)startPolling()},2000);
setInterval(async function(){if(wsConnected)return;var e=await api('equity');if(e)updateEquity(e)},5000);
setInterval(async function(){if(wsConnected)return;var p=await api('positions');if(p)updatePositions(p)},1000);
setInterval(async function(){if(wsConnected)return;var r=await Promise.all([api('signals'),api('regimes'),api('fear-greed'),api('funding-rates')]);if(r[0])updateSignals(r[0]);if(r[1])updateRegimes(r[1]);if(r[2])updateFearGreed(r[2]);if(r[3])updateFunding(r[3])},30000);
setInterval(async function(){var d=await api('trades');if(d)updateTrades(d)},10000);
setInterval(async function(){var r=await Promise.all([api('status'),api('risk'),api('performance')]);if(r[0])updateStatus(r[0]);if(r[1])updateRisk(r[1]);if(r[2]){var wr=r[2].win_rate_30d||0.5;$('ss-winrate').textContent=fmt(wr*100,0)+'%'}},5000);
(async function(){var r=await Promise.all([api('status'),api('equity'),api('risk'),api('signals'),api('positions'),api('regimes'),api('fear-greed'),api('funding-rates'),api('trades'),api('performance')]);if(r[0])updateStatus(r[0]);if(r[1])updateEquity(r[1]);if(r[2])updateRisk(r[2]);if(r[3])updateSignals(r[3]);if(r[4])updatePositions(r[4]);if(r[5])updateRegimes(r[5]);if(r[6])updateFearGreed(r[6]);if(r[7])updateFunding(r[7]);if(r[8])updateTrades(r[8]);if(r[9]){var wr=r[9].win_rate_30d||0.5;$('ss-winrate').textContent=fmt(wr*100,0)+'%'}})();
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
