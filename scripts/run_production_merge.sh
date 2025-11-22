#!/bin/bash
# Run production dataset merge after all components are ready
# Author: @darianrosebrook

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Activate venv if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

echo "=================================================================================="
echo "PRODUCTION DATASET MERGE"
echo "=================================================================================="
echo ""

# Check if CAWS tool examples are ready
if [ ! -f "data/caws_tool_examples_filled.jsonl" ]; then
    echo "WARNING: data/caws_tool_examples_filled.jsonl not found."
    echo "         CAWS tool examples may still be generating."
    echo "         Continuing with merge (CAWS examples will be skipped if not ready)..."
    echo ""
fi

# Run the merge script
python3 -m scripts.merge_production_datasets \
    --worker-out data/worker_production.jsonl \
    --judge-train-out data/judge/train_production.jsonl \
    --judge-val-out data/judge/val_production.jsonl \
    --drafter-out data/drafter/drafter_production.jsonl

echo ""
echo "=================================================================================="
echo "MERGE COMPLETE"
echo "=================================================================================="
echo ""
echo "Next steps:"
echo "  1. Review the merge summary above"
echo "  2. Run quality audit: python3 -m scripts.generate_quality_report \\"
echo "       --worker data/worker_production.jsonl \\"
echo "       --judge data/judge/train_production.jsonl \\"
echo "       --drafter data/drafter/drafter_production.jsonl \\"
echo "       --out docs/DATASET_QUALITY_REPORT_PRODUCTION.md"
echo ""


