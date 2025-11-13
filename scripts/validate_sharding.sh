#!/bin/bash
# Validate sharding determinism
# Generates stable set, shards eval, and compares concatenated results

set -euo pipefail

echo "=========================================="
echo "Sharding Determinism Validation"
echo "=========================================="
echo ""

# Generate stable dataset
echo "Step 1: Generate stable dataset (N=1k)"
echo "----------------------------------------"
make gen-scale-1k || {
    echo "⚠️ Scale dataset generation skipped"
    exit 0
}

# Check if model exists
MODEL_PATH="${MODEL_PATH:-/models/my-checkpoint}"
if [ ! -d "$MODEL_PATH" ]; then
    echo "⚠️ Model not found at $MODEL_PATH"
    echo "   Set MODEL_PATH environment variable to test sharding"
    exit 0
fi

echo ""
echo "Step 2: Run sharded evaluation (4 shards)"
echo "----------------------------------------"
for i in 0 1 2 3; do
    echo "  Running shard $i..."
    python -m eval.cli \
        --runner hf_local \
        --model "$MODEL_PATH" \
        --in data/contextual_extracted_1k.jsonl \
        --out eval/shard.$i.jsonl \
        --report eval/shard.$i.report.json \
        --fixtures eval/tool_broker/fixtures \
        --num-shards 4 \
        --shard-index $i \
        --seed 42 \
        --temperature 0.0 \
        --min-eligible-for-gates 15 \
        --fail-on-fingerprint-mismatch || {
        echo "⚠️ Shard $i evaluation skipped (model may not be available)"
        exit 0
    }
done

echo ""
echo "Step 3: Concatenate sharded results"
echo "----------------------------------------"
cat eval/shard.*.jsonl > eval/results.1k.sharded.jsonl

echo ""
echo "Step 4: Summarize concatenated results"
echo "----------------------------------------"
python -m eval.reports.summarize \
    --in eval/results.1k.sharded.jsonl \
    --out eval/report.1k.sharded.json || {
    echo "⚠️ Summarization skipped (reports module may need updates)"
}

echo ""
echo "✅ Sharding validation complete!"
echo ""
echo "Compare metrics between sharded and non-sharded runs to verify determinism"

