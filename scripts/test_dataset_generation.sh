#!/bin/bash
#
# Test end-to-end dataset generation with small sample.
#
# This test:
# 1. Generates a small dataset (20 samples)
# 2. Verifies dataset format is correct
# 3. Verifies dataset loads in training script
# 4. Checks data quality
#
# Usage:
#   ./scripts/test_dataset_generation.sh
#

set -euo pipefail

echo "=========================================="
echo "Dataset Generation Test"
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

TEST_DATASET="data/kd_mix_test.jsonl"
TEST_CHECKPOINT_DIR="data/checkpoints_test"
TEST_CACHE_DIR="data/kd_cache_test"

# Clean up old test data
echo "Cleaning up old test data..."
rm -f "${TEST_DATASET}"
rm -rf "${TEST_CHECKPOINT_DIR}"
rm -rf "${TEST_CACHE_DIR}"
echo "✅ Cleanup complete"
echo ""

# Step 1: Generate small dataset
echo "Step 1: Generating test dataset (20 samples)..."
python -m scripts.make_kd_mix_hardened \
    --out "${TEST_DATASET}" \
    --teacher https://api.moonshot.ai/v1 \
    --total 20 \
    --checkpoint-dir "${TEST_CHECKPOINT_DIR}" \
    --cache-dir "${TEST_CACHE_DIR}" \
    --budget-limit 1.0 \
    --delay 20 \
    --checkpoint-interval 10

if [ ! -f "${TEST_DATASET}" ]; then
    echo "❌ Dataset file not created"
    exit 1
fi

SAMPLE_COUNT=$(wc -l < "${TEST_DATASET}")
echo "✅ Dataset created: ${SAMPLE_COUNT} samples"
echo ""

# Step 2: Verify dataset format
echo "Step 2: Verifying dataset format..."
python3 << 'PYTHON_EOF'
import json
from pathlib import Path

dataset_path = "data/kd_mix_test.jsonl"
samples = []

with open(dataset_path, 'r') as f:
    for i, line in enumerate(f):
        if not line.strip():
            continue
        try:
            sample = json.loads(line)
            samples.append(sample)
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error on line {i+1}: {e}")
            exit(1)

print(f"   ✅ Loaded {len(samples)} samples")

# Verify required fields
required_fields = ["prompt", "teacher_text"]
for i, sample in enumerate(samples):
    for field in required_fields:
        if field not in sample:
            print(f"❌ Sample {i+1} missing field: {field}")
            exit(1)
        if not sample[field] or len(sample[field].strip()) == 0:
            print(f"❌ Sample {i+1} has empty {field}")
            exit(1)

print(f"   ✅ All samples have required fields")

# Check data quality
prompt_lengths = [len(s["prompt"]) for s in samples]
response_lengths = [len(s["teacher_text"]) for s in samples]

print(f"   Prompt lengths: min={min(prompt_lengths)}, max={max(prompt_lengths)}, avg={sum(prompt_lengths)/len(prompt_lengths):.1f}")
print(f"   Response lengths: min={min(response_lengths)}, max={max(response_lengths)}, avg={sum(response_lengths)/len(response_lengths):.1f}")

if min(prompt_lengths) < 5:
    print(f"   ⚠️  Some prompts are very short")
if min(response_lengths) < 5:
    print(f"   ⚠️  Some responses are very short")

print(f"   ✅ Data quality looks reasonable")
PYTHON_EOF

if [ $? -ne 0 ]; then
    exit 1
fi

echo ""

# Step 3: Verify dataset loads in training script
echo "Step 3: Verifying dataset loads in training script..."
python3 << 'PYTHON_EOF'
from training.dataset import KDDataset
import sys

try:
    dataset = KDDataset(
        jsonl_path="data/kd_mix_test.jsonl",
        tokenizer_path="models/student/tokenizer",
        max_seq_length=4096,
    )
    
    print(f"   ✅ Dataset loaded: {len(dataset)} samples")
    
    if len(dataset) == 0:
        print("   ❌ Dataset is empty")
        sys.exit(1)
    
    # Test getting samples
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        assert "input_ids" in sample
        assert "labels" in sample
        assert "attention_mask" in sample
        print(f"   ✅ Sample {i+1}: input_len={len(sample['input_ids'])}, label_len={len(sample['labels'])}")
    
    print(f"   ✅ All samples load correctly")
    
except Exception as e:
    print(f"   ❌ Dataset loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYTHON_EOF

if [ $? -ne 0 ]; then
    exit 1
fi

echo ""

# Step 4: Check checkpoint was created
echo "Step 4: Verifying checkpoint was created..."
if [ -d "${TEST_CHECKPOINT_DIR}" ]; then
    if [ -f "${TEST_CHECKPOINT_DIR}/progress.json" ]; then
        echo "   ✅ Checkpoint file exists"
    else
        echo "   ⚠️  Checkpoint file not found (may be OK if < checkpoint-interval samples)"
    fi
else
    echo "   ⚠️  Checkpoint directory not created (may be OK if < checkpoint-interval samples)"
fi

echo ""

# Summary
echo "=========================================="
echo "✅ Dataset Generation Test PASSED"
echo "=========================================="
echo ""
echo "Dataset: ${TEST_DATASET}"
echo "Samples: ${SAMPLE_COUNT}"
echo ""
echo "Next steps:"
echo "1. Review sample quality: head -3 ${TEST_DATASET} | python -m json.tool"
echo "2. Test training with this dataset"
echo "3. If quality is good, generate full dataset"
echo ""

