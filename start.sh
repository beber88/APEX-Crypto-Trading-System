#!/bin/bash
# ═══════════════════════════════════════════════════════
# APEX Crypto Trading System — Startup Script
# ═══════════════════════════════════════════════════════
#
# Usage:
#   ./start.sh          — Start in LIVE mode
#   ./start.sh paper    — Start in Paper (simulation) mode
#
# Dashboard: http://localhost:8000
# Login: admin / ApexTrader2026!
# ═══════════════════════════════════════════════════════

set -e
cd "$(dirname "$0")"

MODE="${1:-live}"

# Load environment
if [ -f .env ]; then
    export $(grep -v '^#' .env | grep -v '^$' | xargs)
    echo "✓ Environment loaded from .env"
else
    echo "✗ ERROR: .env file not found! Copy .env.example and fill in your keys."
    exit 1
fi

# Validate required keys
if [ -z "$MEXC_API_KEY" ] || [ -z "$MEXC_SECRET_KEY" ]; then
    echo "✗ ERROR: MEXC_API_KEY and MEXC_SECRET_KEY must be set in .env"
    exit 1
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

echo ""
echo "═══════════════════════════════════════════"
echo "  APEX CRYPTO TRADING SYSTEM"
echo "  Mode: $MODE"
echo "  Dashboard: http://localhost:8000"
echo "  Health:    http://localhost:8000/health"
echo "  Login:     admin / ApexTrader2026!"
echo "═══════════════════════════════════════════"
echo ""

# Start the system
python -m apex_crypto.main
