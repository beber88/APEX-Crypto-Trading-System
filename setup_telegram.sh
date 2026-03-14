#!/bin/bash
# Get your Telegram Chat ID
# 1. Open Telegram and send ANY message to @BGAPEXBot
# 2. Run this script
# 3. It will update your .env automatically

set -e
cd "$(dirname "$0")"

source .env

if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
    echo "ERROR: TELEGRAM_BOT_TOKEN not set in .env"
    exit 1
fi

echo "Fetching chat ID from Telegram..."
echo "(Make sure you sent a message to @BGAPEXBot first!)"
echo ""

RESPONSE=$(curl -s "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/getUpdates")
CHAT_ID=$(echo "$RESPONSE" | python3 -c "
import sys, json
data = json.load(sys.stdin)
if data.get('result'):
    for update in data['result']:
        chat = update.get('message', {}).get('chat', {})
        if chat.get('id'):
            print(chat['id'])
            break
")

if [ -n "$CHAT_ID" ]; then
    sed -i "s/TELEGRAM_CHAT_ID=.*/TELEGRAM_CHAT_ID=$CHAT_ID/" .env
    echo "✓ Chat ID found: $CHAT_ID"
    echo "✓ Updated .env"
    echo ""
    echo "Sending test message..."
    curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        -d "chat_id=$CHAT_ID" \
        -d "text=🚀 APEX Trading System connected! Bot is ready." > /dev/null
    echo "✓ Test message sent! Check your Telegram."
else
    echo "✗ No messages found. Send a message to @BGAPEXBot and try again."
fi
