#!/bin/bash
#
# Test Kimi API endpoint with curl to verify correct format.
#
# Usage:
#   ./scripts/test_api_curl.sh
#   # Or with custom API key:
#   MOONSHOT_API_KEY=your-key ./scripts/test_api_curl.sh
#
# This script tests:
# - Correct endpoint URL format
# - Request body format
# - Authentication header format
# - Response parsing
#

set -euo pipefail

# Load API key from environment or .env.local
if [ -z "${MOONSHOT_API_KEY:-}" ]; then
    if [ -z "${KIMI_API_KEY:-}" ]; then
        if [ -f ".env.local" ]; then
            export MOONSHOT_API_KEY=$(grep "^MOONSHOT_API_KEY=" .env.local | cut -d '=' -f2- | tr -d '"' | tr -d "'" | head -1)
            if [ -z "$MOONSHOT_API_KEY" ]; then
                export KIMI_API_KEY=$(grep "^KIMI_API_KEY=" .env.local | cut -d '=' -f2- | tr -d '"' | tr -d "'" | head -1)
            fi
        fi
    fi
fi

API_KEY="${MOONSHOT_API_KEY:-${KIMI_API_KEY:-}}"

if [ -z "$API_KEY" ]; then
    echo "ERROR: MOONSHOT_API_KEY or KIMI_API_KEY not found"
    echo "Set it with: export MOONSHOT_API_KEY='your-api-key'"
    echo "Or add to .env.local: echo 'MOONSHOT_API_KEY=your-api-key' >> .env.local"
    exit 1
fi

# API endpoint (official Moonshot AI endpoint)
ENDPOINT="https://api.moonshot.ai/v1/chat/completions"

echo "=========================================="
echo "Testing Kimi API with curl"
echo "=========================================="
echo "Endpoint: $ENDPOINT"
echo "API Key: ${API_KEY:0:10}...${API_KEY: -4}"
echo ""

# Test request
echo "Making API request..."
echo ""

RESPONSE=$(curl -s -w "\n%{http_code}" \
    -X POST "$ENDPOINT" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $API_KEY" \
    -d '{
        "model": "kimi-k2-thinking",
        "messages": [
            {"role": "user", "content": "What is 2+2? Answer briefly."}
        ],
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 50
    }')

# Extract HTTP status code (last line)
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
# Extract response body (all but last line)
BODY=$(echo "$RESPONSE" | sed '$d')

echo "HTTP Status Code: $HTTP_CODE"
echo ""

if [ "$HTTP_CODE" -eq 200 ]; then
    echo "✅ API request successful!"
    echo ""
    echo "Response body:"
    echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
    echo ""
    
    # Extract and display response text if available
    if command -v jq &> /dev/null; then
        echo "Response text:"
        echo "$BODY" | jq -r '.choices[0].message.content' 2>/dev/null || echo "Could not parse response"
    elif command -v python3 &> /dev/null; then
        echo "Response text:"
        echo "$BODY" | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "Could not parse response"
    fi
    
    echo ""
    echo "✅ All tests passed!"
    exit 0
    
elif [ "$HTTP_CODE" -eq 429 ]; then
    echo "⚠️ Rate limited (429)"
    echo "This is expected on free tier if requests are too frequent"
    echo ""
    echo "Response:"
    echo "$BODY"
    echo ""
    echo "💡 Tip: Wait 20 seconds between requests on free tier"
    exit 0
    
elif [ "$HTTP_CODE" -eq 401 ] || [ "$HTTP_CODE" -eq 403 ]; then
    echo "❌ Authentication failed ($HTTP_CODE)"
    echo "Check that your API key is correct"
    echo ""
    echo "Response:"
    echo "$BODY"
    exit 1
    
elif [ "$HTTP_CODE" -eq 404 ]; then
    echo "❌ Endpoint not found (404)"
    echo "Check that the endpoint URL is correct: $ENDPOINT"
    echo ""
    echo "Response:"
    echo "$BODY"
    exit 1
    
else
    echo "❌ API request failed with status $HTTP_CODE"
    echo ""
    echo "Response:"
    echo "$BODY"
    exit 1
fi

