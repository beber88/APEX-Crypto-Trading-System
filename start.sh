#!/bin/bash
# ═══════════════════════════════════════════════════════
# APEX Crypto Trading System — Quick Start Script
# ═══════════════════════════════════════════════════════
set -e

echo "============================================"
echo "  APEX CRYPTO TRADING SYSTEM — SETUP"
echo "============================================"

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "ERROR: Python 3 is not installed."
    echo "Install it: brew install python3 (Mac) or sudo apt install python3 (Linux)"
    exit 1
fi

echo "[1/4] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

echo "[2/4] Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "[3/4] Creating directories..."
mkdir -p data reports logs ml/models

echo "[4/4] Starting trading system (paper mode)..."
echo ""
echo "============================================"
echo "  Mode:      PAPER TRADING"
echo "  Dashboard: http://localhost:8000"
echo "  Logs:      logs/trading_session.log"
echo "  Stop:      Press Ctrl+C"
echo "============================================"
echo ""

export PYTHONPATH="$(pwd)"
python3 -m apex_crypto.main 2>&1 | tee logs/trading_session.log
