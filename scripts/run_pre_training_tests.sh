#!/bin/bash
#
# Run all pre-training tests in sequence.
#
# This runs the critical tests before spending money on training:
# 1. Toy training smoke test (no cost)
# 2. Dataset generation test (~$0.10)
# 3. Budget enforcement test (~$0.01)
# 4. Training execution test (no cost)
#
# Usage:
#   ./scripts/run_pre_training_tests.sh
#

set -euo pipefail

echo "=========================================="
echo "Pre-Training Test Suite"
echo "=========================================="
echo ""
echo "This will run all critical tests before spending money on training."
echo "Estimated cost: < $1"
echo "Estimated time: 15-30 minutes"
echo ""
read -p "Continue? (y/N): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled"
    exit 0
fi

echo ""

# Test 1: Toy training smoke test
echo "Test 1: Toy Training Smoke Test (no cost)"
echo "=========================================="
make smoke_training || {
    echo "❌ Toy training test failed"
    exit 1
}
echo ""

# Test 2: Dataset generation
echo "Test 2: Dataset Generation Test (~$0.10)"
echo "=========================================="
bash scripts/test_dataset_generation.sh || {
    echo "❌ Dataset generation test failed"
    exit 1
}
echo ""

# Test 3: Budget enforcement
echo "Test 3: Budget Enforcement Test (~$0.01)"
echo "=========================================="
bash scripts/test_budget_enforcement.sh || {
    echo "❌ Budget enforcement test failed"
    exit 1
}
echo ""

# Test 4: Training execution
echo "Test 4: Training Execution Test (no cost)"
echo "=========================================="
bash scripts/test_training_execution.sh || {
    echo "❌ Training execution test failed"
    exit 1
}
echo ""

# Summary
echo "=========================================="
echo "✅ All Pre-Training Tests PASSED"
echo "=========================================="
echo ""
echo "You're ready to generate the full dataset!"
echo ""
echo "Next steps:"
echo "1. Review test results above"
echo "2. Generate full dataset:"
echo "   python -m scripts.make_kd_mix_hardened \\"
echo "       --out data/kd_mix.jsonl \\"
echo "       --teacher https://api.moonshot.ai/v1 \\"
echo "       --total 10000 \\"
echo "       --budget-limit 50.0 \\"
echo "       --delay 20"
echo ""
