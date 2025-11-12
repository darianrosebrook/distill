#!/bin/bash
#
# Daily KD dataset generation script for free tier
# Runs within daily token limits (1.5M tokens = ~1,200 samples/day)
#
# Usage:
#   ./scripts/make_kd_mix_daily.sh
#   # Or add to crontab:
#   # 0 2 * * * /path/to/distill/scripts/make_kd_mix_daily.sh >> /path/to/logs/kd_generation.log 2>&1
#
# Environment variables:
#   KIMI_API_KEY: API key for Kimi API (required)
#   KIMI_ENDPOINT: API endpoint (default: https://api.kimi.com/v1)
#   KD_OUTPUT_DIR: Output directory (default: data/)
#   KD_CACHE_DIR: Cache directory (default: data/kd_cache/)
#   SAMPLES_PER_DAY: Samples to generate per day (default: 1200)
#

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Defaults
KIMI_ENDPOINT="${KIMI_ENDPOINT:-https://api.kimi.com/v1}"
KD_OUTPUT_DIR="${KD_OUTPUT_DIR:-data}"
KD_CACHE_DIR="${KD_CACHE_DIR:-data/kd_cache}"
SAMPLES_PER_DAY="${SAMPLES_PER_DAY:-1200}"

# Free tier rate limits
RPM=3  # Requests per minute
DELAY=20  # Seconds between requests (60/3 = 20)

# Load API key from .env.local if not in environment
if [ -z "${KIMI_API_KEY:-}" ]; then
    if [ -f ".env.local" ]; then
        export KIMI_API_KEY=$(grep "^KIMI_API_KEY=" .env.local | cut -d '=' -f2- | tr -d '"' | tr -d "'")
    fi
fi

# Check for API key
if [ -z "${KIMI_API_KEY:-}" ]; then
    echo "ERROR: KIMI_API_KEY not found in environment or .env.local"
    echo "Set it with: export KIMI_API_KEY='your-api-key'"
    echo "Or add to .env.local: echo 'KIMI_API_KEY=your-api-key' >> .env.local"
    exit 1
fi

# Create output directory
mkdir -p "$KD_OUTPUT_DIR"
mkdir -p "$KD_CACHE_DIR"

# Generate date-based output filename
DATE=$(date +%Y%m%d)
OUTPUT_FILE="$KD_OUTPUT_DIR/kd_mix_${DATE}.jsonl"
COMBINED_FILE="$KD_OUTPUT_DIR/kd_mix_combined.jsonl"

# Log file
LOG_FILE="$KD_OUTPUT_DIR/kd_generation_${DATE}.log"

echo "==========================================" | tee -a "$LOG_FILE"
echo "KD Dataset Generation - $(date)" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "Endpoint: $KIMI_ENDPOINT" | tee -a "$LOG_FILE"
echo "Samples: $SAMPLES_PER_DAY" | tee -a "$LOG_FILE"
echo "Output: $OUTPUT_FILE" | tee -a "$LOG_FILE"
echo "Cache: $KD_CACHE_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run dataset generation
python -m scripts.make_kd_mix \
    --out "$OUTPUT_FILE" \
    --teacher "$KIMI_ENDPOINT" \
    --total "$SAMPLES_PER_DAY" \
    --cache-dir "$KD_CACHE_DIR" \
    --delay "$DELAY" \
    --temperature 1.5 \
    --top-p 0.95 \
    --max-tokens 1024 \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    # Count samples generated
    SAMPLE_COUNT=$(wc -l < "$OUTPUT_FILE" 2>/dev/null || echo "0")
    
    # Append to combined file
    if [ -f "$OUTPUT_FILE" ]; then
        cat "$OUTPUT_FILE" >> "$COMBINED_FILE"
        echo "" >> "$COMBINED_FILE"  # Add newline separator
    fi
    
    echo "" | tee -a "$LOG_FILE"
    echo "✅ Success: Generated $SAMPLE_COUNT samples" | tee -a "$LOG_FILE"
    echo "   Daily file: $OUTPUT_FILE" | tee -a "$LOG_FILE"
    echo "   Combined file: $COMBINED_FILE" | tee -a "$LOG_FILE"
    
    # Show combined file stats
    if [ -f "$COMBINED_FILE" ]; then
        TOTAL_SAMPLES=$(wc -l < "$COMBINED_FILE")
        echo "   Total samples in combined file: $TOTAL_SAMPLES" | tee -a "$LOG_FILE"
    fi
else
    echo "" | tee -a "$LOG_FILE"
    echo "❌ Error: Generation failed with exit code $EXIT_CODE" | tee -a "$LOG_FILE"
    exit $EXIT_CODE
fi

echo "==========================================" | tee -a "$LOG_FILE"
echo "Completed at $(date)" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

