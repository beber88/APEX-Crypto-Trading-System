#!/bin/bash
# ═══════════════════════════════════════════════════════
# APEX Crypto Trading System — Startup Script
# ═══════════════════════════════════════════════════════
#
# Usage:
#   ./start.sh                    — Start in Paper mode (default, safe)
#   ./start.sh paper              — Start in Paper mode
#   ./start.sh paper 8h           — Paper mode, auto-stop after 8 hours
#   ./start.sh live               — Start in LIVE mode (real money!)
#   ./start.sh live 8h            — Live mode, auto-stop after 8 hours
#   ./start.sh report             — Generate status report and exit
#
# Dashboard: http://localhost:8000
# Login: admin / ApexTrader2026!
# ═══════════════════════════════════════════════════════

set -e
cd "$(dirname "$0")"

# Detect python command (macOS uses python3)
PYTHON="python3"
if ! command -v python3 &>/dev/null; then
    PYTHON="python"
fi
if ! command -v $PYTHON &>/dev/null; then
    echo "ERROR: Python not found. Install Python 3.9+ first."
    echo "  macOS: brew install python"
    exit 1
fi

MODE="${1:-paper}"
DURATION="${2:-}"

# Load environment
if [ -f .env ]; then
    export $(grep -v '^#' .env | grep -v '^$' | xargs)
    echo "✓ Environment loaded from .env"
elif [ -f .env.template ]; then
    echo "  .env file not found — creating from .env.template"
    cp .env.template .env
    echo ""
    echo "  IMPORTANT: Edit .env with your MEXC API keys before running live."
    echo "  For paper trading, we'll use placeholder keys (data from exchange only)."
    echo ""
    export $(grep -v '^#' .env | grep -v '^$' | xargs)
else
    echo "ERROR: Neither .env nor .env.template found!"
    exit 1
fi

# Handle report mode
if [ "$MODE" = "report" ]; then
    echo "Generating status report..."
    $PYTHON -m apex_crypto.main --report
    exit 0
fi

# Set mode in config
if [ "$MODE" = "paper" ]; then
    echo "  Mode: PAPER (simulation — no real trades)"
    sed -i'' -e 's/mode: "live"/mode: "paper"/' apex_crypto/config/config.yaml
elif [ "$MODE" = "live" ]; then
    echo "  ⚠️  Mode: LIVE TRADING — Real money!"
    echo "  Press Ctrl+C within 5 seconds to cancel..."
    sleep 5
    sed -i'' -e 's/mode: "paper"/mode: "live"/' apex_crypto/config/config.yaml
fi

DURATION_DISPLAY="unlimited"
DURATION_FLAG=""
if [ -n "$DURATION" ]; then
    DURATION_DISPLAY="$DURATION"
    DURATION_FLAG="--duration $DURATION"
fi

echo ""
echo "═══════════════════════════════════════════"
echo "  APEX CRYPTO TRADING SYSTEM"
echo "  Mode:      $MODE"
echo "  Duration:  $DURATION_DISPLAY"
echo "  Dashboard: http://localhost:8000"
echo "  Health:    http://localhost:8000/health"
echo "  Login:     admin / ApexTrader2026!"
echo "═══════════════════════════════════════════"
echo ""

# Start the system
$PYTHON -m apex_crypto.main --mode "$MODE" $DURATION_FLAG
