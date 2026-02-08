#!/bin/bash
# Quick test script to debug API endpoints

API_KEY="$1"
ENDPOINT="$2"

if [ -z "$API_KEY" ] || [ -z "$ENDPOINT" ]; then
    echo "Usage: ./test_api.sh <api_key> <endpoint>"
    echo ""
    echo "Example:"
    echo "  ./test_api.sh sk-ant-xxx https://api.z.ai/api/anthropic/v1/messages"
    exit 1
fi

echo "Testing endpoint: $ENDPOINT"
echo "API Key: ${API_KEY:0:10}..."
echo ""

curl -X POST "$ENDPOINT" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 100,
    "messages": [
      {"role": "user", "content": "Hi"}
    ]
  }' \
  -v \
  2>&1 | head -50
