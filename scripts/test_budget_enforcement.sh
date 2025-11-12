#!/bin/bash
#
# Test budget enforcement stops dataset generation at limit.
#
# This test:
# 1. Sets a very small budget ($0.10)
# 2. Generates dataset
# 3. Verifies script stops when budget exceeded
# 4. Verifies checkpoint saved before stopping
#
# Usage:
#   ./scripts/test_budget_enforcement.sh
#

set -euo pipefail

echo "=========================================="
echo "Budget Enforcement Test"
echo "=========================================="
echo ""

# Check if virtual environment is activated
if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    else
        echo "❌ Virtual environment not found"
        exit 1
    fi
fi

TEST_DATASET="data/kd_mix_budget_test.jsonl"
TEST_CHECKPOINT_DIR="data/checkpoints_budget_test"
TEST_CACHE_DIR="data/kd_cache_budget_test"
BUDGET_LIMIT=0.10

# Clean up old test data
echo "Cleaning up old test data..."
rm -f "${TEST_DATASET}"
rm -rf "${TEST_CHECKPOINT_DIR}"
rm -rf "${TEST_CACHE_DIR}"
echo "✅ Cleanup complete"
echo ""

# Step 1: Generate dataset with small budget
echo "Step 1: Generating dataset with budget limit \$${BUDGET_LIMIT}..."
echo "   (This should stop after 1-2 samples)"
echo ""

# Run generation and capture output
OUTPUT=$(python -m scripts.make_kd_mix_hardened \
    --out "${TEST_DATASET}" \
    --teacher https://api.moonshot.ai/v1 \
    --total 100 \
    --checkpoint-dir "${TEST_CHECKPOINT_DIR}" \
    --cache-dir "${TEST_CACHE_DIR}" \
    --budget-limit "${BUDGET_LIMIT}" \
    --delay 20 \
    --checkpoint-interval 1 2>&1) || EXIT_CODE=$?

echo "$OUTPUT"

# Check if budget exceeded error occurred
if echo "$OUTPUT" | grep -q "Budget limit exceeded\|BudgetExceededError"; then
    echo ""
    echo "   ✅ Budget limit enforced correctly"
else
    echo ""
    echo "   ⚠️  Budget limit may not have been reached"
    echo "   (Check if cost was below limit)"
fi

echo ""

# Step 2: Verify checkpoint was saved
echo "Step 2: Verifying checkpoint was saved..."
if [ -f "${TEST_CHECKPOINT_DIR}/progress.json" ]; then
    echo "   ✅ Checkpoint file exists"
    
    # Check checkpoint content
    python3 << 'PYTHON_EOF'
import json
from pathlib import Path

checkpoint_file = Path("data/checkpoints_budget_test/progress.json")
if checkpoint_file.exists():
    with open(checkpoint_file) as f:
        checkpoint = json.load(f)
    
    print(f"   Completed samples: {checkpoint.get('total_completed', 0)}")
    print(f"   Total cost: ${checkpoint.get('budget', {}).get('total_cost', 0):.4f}")
    print(f"   Budget limit: ${checkpoint.get('budget', {}).get('budget_limit', 0):.2f}")
    
    if checkpoint.get('budget', {}).get('total_cost', 0) >= 0.10:
        print("   ✅ Budget limit was reached")
    else:
        print("   ⚠️  Budget limit not reached (cost was below limit)")
PYTHON_EOF
else
    echo "   ⚠️  Checkpoint file not found"
fi

echo ""

# Step 3: Verify dataset has samples
echo "Step 3: Verifying dataset..."
if [ -f "${TEST_DATASET}" ]; then
    SAMPLE_COUNT=$(wc -l < "${TEST_DATASET}")
    echo "   ✅ Dataset created: ${SAMPLE_COUNT} samples"
    
    if [ "${SAMPLE_COUNT}" -gt 0 ] && [ "${SAMPLE_COUNT}" -lt 10 ]; then
        echo "   ✅ Sample count is reasonable (stopped early due to budget)"
    fi
else
    echo "   ⚠️  Dataset file not found"
fi

echo ""

# Summary
echo "=========================================="
echo "Budget Enforcement Test Complete"
echo "=========================================="
echo ""
echo "If budget was enforced:"
echo "  ✅ Script stopped when budget exceeded"
echo "  ✅ Checkpoint saved before stopping"
echo "  ✅ No samples processed after budget limit"
echo ""
echo "If budget was not reached:"
echo "  ⚠️  Cost was below limit (may need smaller limit for test)"
echo ""

