# APEX Crypto Trading System

Autonomous cryptocurrency trading system built for MEXC exchange. Combines Smart Money Concepts (SMC), technical analysis, machine learning regime detection, and multi-strategy signal aggregation to execute trades across spot and futures markets.

## Architecture

```
apex_crypto/
├── config/           # YAML configuration & loader
├── core/
│   ├── analysis/     # SMC detection, indicators, patterns, regime classification
│   ├── data/         # OHLCV ingestion, WebSocket streaming, alt-data (funding, OI, sentiment)
│   ├── execution/    # MEXC broker, order manager, position tracker
│   ├── risk/         # Kelly sizing, portfolio limits, circuit-breaker guards
│   ├── signals/      # Multi-strategy aggregator & decision engine
│   └── strategies/   # 8 strategy modules (see below)
├── backtest/         # Vectorized backtester, walk-forward optimization, Monte Carlo
├── dashboard/        # FastAPI + WebSocket real-time dashboard
├── ml/               # XGBoost regime model, FinBERT sentiment
├── reporting/        # Daily & weekly PDF/Telegram reports
├── telegram/         # Alert system & command bot
└── main.py           # Entry point & orchestrator
```

## Strategies

| Strategy | Timeframe | Description |
|----------|-----------|-------------|
| **SMC** | 4h / 15m entry | Order Block retests, FVG fills, liquidity sweep fades |
| **Trend Momentum** | 4h / 1h entry | EMA alignment + ADX + RSI confirmation |
| **Swing** | 1d / 4h entry | Multi-day holds on strong setups |
| **Breakout** | 4h | Volume-confirmed breakouts with structure validation |
| **Mean Reversion** | 1h | Bollinger + RSI extremes with reversal confirmation |
| **Scalping** | 1m-3m | BTC/ETH microstructure trades |
| **Funding Rate** | 4h | Exploits funding rate extremes on perpetuals |
| **OI Divergence** | 4h | Open Interest vs price divergence signals |

## Risk Management

- **Kelly Criterion** position sizing with configurable max risk per trade (default 1%)
- **Portfolio limits**: max 8 open positions, 10% asset concentration cap
- **Circuit breakers**: 3% daily loss pause, 12% max drawdown halt
- **Flash crash guard**: 6% drop in 2 minutes triggers 30-minute halt
- **Correlation reduction**: correlated positions get reduced sizing

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (for TimescaleDB, Redis, Grafana, Prometheus)
- MEXC exchange account with API keys
- Telegram bot (for alerts)

### Setup

```bash
# Clone the repo
git clone https://github.com/beber88/apex-trading-system.git
cd apex-trading-system

# Create a Python virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and fill in environment variables
cp .env.template .env
# Edit .env with your API keys (see below)

# Start infrastructure (TimescaleDB, Redis, Grafana, Prometheus)
docker-compose up -d

# Initialize the database
python scripts/setup_db.py

# Download historical data (required for backtesting)
python scripts/download_history.py

# Run backtests first (recommended before live trading)
python scripts/run_backtest.py

# Start the system in paper mode (default)
python -m apex_crypto.main
```

### Environment Variables

Copy `.env.template` to `.env` and fill in:

| Variable | Where to get it |
|----------|----------------|
| `MEXC_API_KEY` / `MEXC_SECRET_KEY` | [MEXC API Management](https://www.mexc.com/user/openapi) — enable Spot + Futures trading |
| `TELEGRAM_BOT_TOKEN` | Telegram @BotFather → `/newbot` |
| `TELEGRAM_CHAT_ID` | Send message to your bot, then check `getUpdates` API |
| `CRYPTOPANIC_API_KEY` | [CryptoPanic Developers](https://cryptopanic.com/developers/api/) (free tier) |
| `POSTGRES_PASSWORD` | Choose a strong password for TimescaleDB |
| `DASHBOARD_USER` / `DASHBOARD_PASSWORD` | Choose credentials for the web dashboard |
| `GRAFANA_PASSWORD` | Choose a password for Grafana |

### Going Live

1. Run backtests and verify Sharpe > 1.5, win rate > 45%, max drawdown < 20%
2. Paper trade for at least 2 weeks
3. In `config/config.yaml`, set `system.mode: "live"` and `exchange.testnet: false`
4. Start with minimal capital and monitor via Telegram + Dashboard

## Dashboard

Access at `http://localhost:8000` after starting the system. Shows:
- Live P&L, open positions, and trade history
- Strategy performance breakdown
- Market regime classification
- Real-time WebSocket price feeds

## Monitoring

Grafana is available at `http://localhost:3000` with pre-configured Prometheus metrics for:
- Trade execution latency
- Strategy signal counts
- Risk utilization
- System health

## Configuration

All parameters are in `apex_crypto/config/config.yaml`. Key sections:
- `assets` — which pairs to trade and volume filters
- `risk` — position sizing, drawdown limits, circuit breakers
- `strategies` — enable/disable individual strategies and tune parameters
- `signals` — score thresholds for trade entry

## License

Private — all rights reserved.
