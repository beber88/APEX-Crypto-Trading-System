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

MODE="${1:-paper}"
DURATION="${2:-}"

# Load environment
if [ -f .env ]; then
    export $(grep -v '^#' .env | grep -v '^$' | xargs)
    echo "✓ Environment loaded from .env"
else
    echo "✗ ERROR: .env file not found! Copy .env.template to .env and fill in your keys."
    exit 1
fi

# Validate required keys
if [ -z "$MEXC_API_KEY" ] || [ -z "$MEXC_SECRET_KEY" ]; then
    echo "✗ ERROR: MEXC_API_KEY and MEXC_SECRET_KEY must be set in .env"
    exit 1
fi

# Handle report mode
if [ "$MODE" = "report" ]; then
    echo "Generating status report..."
    python -m apex_crypto.main --report
    exit 0
fi

# Set mode in config
if [ "$MODE" = "paper" ]; then
    echo "  Mode: PAPER (simulation — no real trades)"
    sed -i 's/mode: "live"/mode: "paper"/' apex_crypto/config/config.yaml
elif [ "$MODE" = "live" ]; then
    echo "  ⚠️  Mode: LIVE TRADING — Real money!"
    echo "  Press Ctrl+C within 5 seconds to cancel..."
    sleep 5
    sed -i 's/mode: "paper"/mode: "live"/' apex_crypto/config/config.yaml
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
python -m apex_crypto.main --mode "$MODE" $DURATION_FLAG
