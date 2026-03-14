#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  APEX Crypto Trading System — One-Click Deployment
# ═══════════════════════════════════════════════════════════════
#
#  Usage:
#    ./deploy.sh              — Deploy with Docker (recommended)
#    ./deploy.sh --no-docker  — Run directly with Python
#    ./deploy.sh --paper      — Start in paper (simulation) mode
#
# ═══════════════════════════════════════════════════════════════

set -e
cd "$(dirname "$0")"

# ── Colors ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

echo ""
echo -e "${CYAN}${BOLD}═══════════════════════════════════════════════════${NC}"
echo -e "${CYAN}${BOLD}    APEX CRYPTO TRADING SYSTEM — DEPLOYMENT${NC}"
echo -e "${CYAN}${BOLD}═══════════════════════════════════════════════════${NC}"
echo ""

# ── Parse Arguments ──
USE_DOCKER=true
MODE="live"
for arg in "$@"; do
    case $arg in
        --no-docker) USE_DOCKER=false ;;
        --paper)     MODE="paper" ;;
    esac
done

# ── Step 1: Check .env ──
echo -e "${BOLD}[1/6] Checking environment...${NC}"
if [ ! -f .env ]; then
    echo -e "${RED}   ERROR: .env file not found!${NC}"
    echo ""
    echo "   Create .env with your keys:"
    echo "   ─────────────────────────────────────────"
    echo "   MEXC_API_KEY=your_api_key_here"
    echo "   MEXC_SECRET_KEY=your_secret_key_here"
    echo "   TELEGRAM_BOT_TOKEN=your_bot_token_here"
    echo "   TELEGRAM_CHAT_ID="
    echo "   DASHBOARD_USER=admin"
    echo "   DASHBOARD_PASSWORD=ApexTrader2026!"
    echo "   POSTGRES_PASSWORD=apex_secure_2026"
    echo "   GRAFANA_PASSWORD=apex_grafana_2026"
    echo "   CRYPTOPANIC_API_KEY="
    echo "   ─────────────────────────────────────────"
    exit 1
fi

# Load and validate keys
export $(grep -v '^#' .env | grep -v '^$' | xargs)

MISSING=false
if [ -z "$MEXC_API_KEY" ] || [ "$MEXC_API_KEY" = "your_api_key_here" ]; then
    echo -e "${RED}   MEXC_API_KEY is missing or not configured${NC}"
    MISSING=true
fi
if [ -z "$MEXC_SECRET_KEY" ] || [ "$MEXC_SECRET_KEY" = "your_secret_key_here" ]; then
    echo -e "${RED}   MEXC_SECRET_KEY is missing or not configured${NC}"
    MISSING=true
fi
if [ "$MISSING" = true ]; then
    echo -e "${RED}   Fix your .env file and try again.${NC}"
    exit 1
fi
echo -e "${GREEN}   .env loaded — MEXC keys present${NC}"

# ── Step 2: Set trading mode ──
echo -e "${BOLD}[2/6] Setting trading mode...${NC}"
if [ "$MODE" = "paper" ]; then
    sed -i 's/mode: "live"/mode: "paper"/' apex_crypto/config/config.yaml
    echo -e "${GREEN}   Mode: PAPER (simulation — no real money)${NC}"
else
    sed -i 's/mode: "paper"/mode: "live"/' apex_crypto/config/config.yaml
    echo -e "${YELLOW}   Mode: LIVE TRADING — Real money!${NC}"
fi

# ── Step 3: Setup Telegram (if chat_id missing) ──
echo -e "${BOLD}[3/6] Checking Telegram...${NC}"
if [ -n "$TELEGRAM_BOT_TOKEN" ] && [ -z "$TELEGRAM_CHAT_ID" ]; then
    echo -e "${YELLOW}   Telegram chat ID not set.${NC}"
    echo -e "${YELLOW}   Send a message to your bot on Telegram, then run:${NC}"
    echo -e "${YELLOW}   ./setup_telegram.sh${NC}"
    echo ""
elif [ -n "$TELEGRAM_BOT_TOKEN" ] && [ -n "$TELEGRAM_CHAT_ID" ]; then
    echo -e "${GREEN}   Telegram configured (chat: $TELEGRAM_CHAT_ID)${NC}"
else
    echo -e "${YELLOW}   Telegram not configured (optional)${NC}"
fi

# ── Deploy ──
if [ "$USE_DOCKER" = true ]; then

    # ── Step 4: Check Docker ──
    echo -e "${BOLD}[4/6] Checking Docker...${NC}"
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}   Docker not found! Install Docker first:${NC}"
        echo "   https://docs.docker.com/get-docker/"
        echo ""
        echo "   Or run without Docker:"
        echo "   ./deploy.sh --no-docker"
        exit 1
    fi
    if ! docker info &> /dev/null 2>&1; then
        echo -e "${RED}   Docker is not running! Start Docker Desktop and try again.${NC}"
        exit 1
    fi
    echo -e "${GREEN}   Docker is ready${NC}"

    # ── Step 5: Build ──
    echo -e "${BOLD}[5/6] Building containers...${NC}"
    echo ""
    docker compose build trading_bot
    echo ""
    echo -e "${GREEN}   Build complete${NC}"

    # ── Step 6: Launch ──
    echo -e "${BOLD}[6/6] Starting all services...${NC}"
    docker compose up -d
    echo ""

    # Wait for health
    echo -e "${BOLD}Waiting for services to start...${NC}"
    sleep 5

    # Check container status
    echo ""
    echo -e "${BOLD}Container Status:${NC}"
    docker compose ps
    echo ""

    # ── Summary ──
    echo -e "${CYAN}${BOLD}═══════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}${BOLD}   APEX TRADING SYSTEM IS RUNNING!${NC}"
    echo -e "${CYAN}${BOLD}═══════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "   ${BOLD}Dashboard:${NC}    http://localhost:8000"
    echo -e "   ${BOLD}Login:${NC}        ${DASHBOARD_USER:-admin} / ${DASHBOARD_PASSWORD:-ApexTrader2026!}"
    echo -e "   ${BOLD}Health:${NC}       http://localhost:8000/health"
    echo -e "   ${BOLD}Grafana:${NC}      http://localhost:3001"
    echo -e "   ${BOLD}Prometheus:${NC}   http://localhost:9090"
    echo -e "   ${BOLD}Mode:${NC}         ${MODE}"
    echo ""
    echo -e "   ${BOLD}View logs:${NC}    docker compose logs -f trading_bot"
    echo -e "   ${BOLD}Stop:${NC}         docker compose down"
    echo -e "   ${BOLD}Restart:${NC}      docker compose restart trading_bot"
    echo ""

else

    # ── No-Docker Path ──
    echo -e "${BOLD}[4/6] Checking Python...${NC}"
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}   Python 3 not found! Install Python 3.11+${NC}"
        exit 1
    fi
    PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo -e "${GREEN}   Python $PYVER${NC}"

    echo -e "${BOLD}[5/6] Installing dependencies...${NC}"
    # Install without heavy ML packages for faster startup
    grep -v -E "^(torch|transformers|celery|reportlab|prometheus)" requirements.txt > /tmp/apex-req-core.txt
    pip install -q -r /tmp/apex-req-core.txt
    rm /tmp/apex-req-core.txt
    echo -e "${GREEN}   Dependencies installed${NC}"

    # Create directories
    mkdir -p data reports logs ml/models

    echo -e "${BOLD}[6/6] Starting APEX Trading System...${NC}"
    echo ""
    echo -e "${CYAN}${BOLD}═══════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}${BOLD}   APEX TRADING SYSTEM STARTING!${NC}"
    echo -e "${CYAN}${BOLD}═══════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "   ${BOLD}Dashboard:${NC}    http://localhost:8000"
    echo -e "   ${BOLD}Login:${NC}        ${DASHBOARD_USER:-admin} / ${DASHBOARD_PASSWORD:-ApexTrader2026!}"
    echo -e "   ${BOLD}Health:${NC}       http://localhost:8000/health"
    echo -e "   ${BOLD}Mode:${NC}         ${MODE}"
    echo ""
    echo -e "   ${BOLD}Stop:${NC}         Ctrl+C"
    echo ""

    export PYTHONPATH="$(pwd)"
    python3 -m apex_crypto.main
fi
