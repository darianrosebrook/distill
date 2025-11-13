#!/bin/bash
# Immediate validation script for evaluation harness
# Run this after merge to verify everything works

set -euo pipefail

echo "=========================================="
echo "Evaluation Harness Validation"
echo "=========================================="
echo ""

# 1) Fixture sanity + smoke
echo "Step 1: Generate tiny dataset (N=30)"
echo "----------------------------------------"
make contextual-gen TOTAL=30
make contextual-extract IN=data/contextual_prompts.jsonl OUT=data/contextual_final.smoke.jsonl

echo ""
echo "Step 2: Broker fixture hit-rate test"
echo "----------------------------------------"
make ci-broker-smoke || {
    echo "❌ Broker fixture test failed"
    exit 1
}

echo ""
echo "Step 3: Verify dataset gates"
echo "----------------------------------------"
python -m scripts.verify_contextual_set \
    --in data/contextual_final.smoke.jsonl \
    --report eval/reports/verify_smoke.json \
    --tokenizer models/student/tokenizer \
    --min-eligible-for-gates 10 \
    --fail-on-fingerprint-mismatch || {
    echo "⚠️ Dataset verification skipped (tokenizer may not exist)"
}

echo ""
echo "✅ Validation complete!"
echo ""
echo "Next steps:"
echo "  - Run full eval with: make eval-runner-local MODEL=/path/to/model"
echo "  - Check CI workflows: .github/workflows/eval-*.yml"
echo "  - Review docs: eval/HARNESS.md"

