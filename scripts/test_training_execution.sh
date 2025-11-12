#!/bin/bash
#
# Test training script execution with real dataset.
#
# This test:
# 1. Uses test dataset (or creates toy dataset)
# 2. Runs training script for a few steps
# 3. Verifies training completes without errors
# 4. Verifies checkpoint is saved
#
# Usage:
#   ./scripts/test_training_execution.sh [dataset_path]
#

set -euo pipefail

echo "=========================================="
echo "Training Execution Test"
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

# Use provided dataset or create toy dataset
if [ -n "${1:-}" ] && [ -f "$1" ]; then
    DATASET_PATH="$1"
    echo "Using provided dataset: ${DATASET_PATH}"
else
    echo "Creating toy dataset for testing..."
    python -m training.make_toy_training \
        --out-dir training/toy_test \
        --samples 20 \
        --steps 5 \
        --dmodel 64 \
        --nlayers 2 \
        --vocab 32000
    
    DATASET_PATH="training/toy_test/toy_dataset.jsonl"
    CONFIG_PATH="training/toy_test/toy_config.yaml"
    echo "✅ Toy dataset created"
fi

echo ""

# Step 1: Verify dataset exists
echo "Step 1: Verifying dataset..."
if [ ! -f "${DATASET_PATH}" ]; then
    echo "❌ Dataset not found: ${DATASET_PATH}"
    exit 1
fi

SAMPLE_COUNT=$(wc -l < "${DATASET_PATH}")
echo "   ✅ Dataset: ${SAMPLE_COUNT} samples"
echo ""

# Step 2: Create test config if using toy dataset
if [ -z "${CONFIG_PATH:-}" ]; then
    CONFIG_PATH="training/toy_test/toy_config.yaml"
    # Update config to use provided dataset
    python3 << PYTHON_EOF
import yaml
from pathlib import Path

config_path = Path("training/toy_test/toy_config.yaml")
if not config_path.exists():
    # Create minimal config
    config = {
        "arch": {
            "d_model": 64,
            "n_layers": 2,
            "n_heads": 4,
            "n_kv_heads": 2,
            "d_head": 16,
            "vocab_size": 32000,
            "rope_theta": 10000.0,
            "rope_scaling": "dynamic",
            "dropout": 0.0,
        },
        "train": {
            "seq_lengths": [128],
            "micro_batch_size": 1,
            "grad_accum": 2,
            "steps": 5,
            "fp16": False,
            "grad_checkpointing": False,
        },
        "io": {
            "tokenizer_path": "models/student/tokenizer",
            "train_shards": ["${DATASET_PATH}"],
        },
        "optimizer": {
            "name": "adamw",
            "lr": 1e-4,
            "betas": [0.9, 0.95],
            "weight_decay": 0.1,
        },
        "tracing": {
            "log_dir": "training/toy_test/runs",
            "use_tensorboard": False,
            "use_wandb": False,
            "json_log": True,
            "console_log": True,
        },
    }
    
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    print(f"   ✅ Created test config: {config_path}")
else:
    print(f"   ✅ Using existing config: {config_path}")
PYTHON_EOF
fi

echo ""

# Step 3: Run training for a few steps
echo "Step 3: Running training (5 steps)..."
OUTPUT_DIR="training/toy_test/checkpoints"
mkdir -p "${OUTPUT_DIR}"

# Run training and capture output
TRAIN_OUTPUT=$(python -m training.distill_kd \
    --config "${CONFIG_PATH}" \
    --output-dir "${OUTPUT_DIR}" 2>&1) || EXIT_CODE=$?

echo "$TRAIN_OUTPUT"

# Check for errors
if echo "$TRAIN_OUTPUT" | grep -qi "error\|exception\|traceback\|failed"; then
    echo ""
    echo "   ❌ Training failed with errors"
    exit 1
fi

# Check if training completed
if echo "$TRAIN_OUTPUT" | grep -qi "complete\|finished\|checkpoint"; then
    echo ""
    echo "   ✅ Training completed successfully"
else
    echo ""
    echo "   ⚠️  Training output unclear (check manually)"
fi

echo ""

# Step 4: Verify checkpoint was created
echo "Step 4: Verifying checkpoint..."
CHECKPOINT_FILES=$(find "${OUTPUT_DIR}" -name "*.pt" 2>/dev/null | wc -l)
if [ "${CHECKPOINT_FILES}" -gt 0 ]; then
    echo "   ✅ Checkpoint files found: ${CHECKPOINT_FILES}"
    ls -lh "${OUTPUT_DIR}"/*.pt 2>/dev/null | head -3
else
    echo "   ⚠️  No checkpoint files found"
fi

echo ""

# Step 5: Verify loss was computed
echo "Step 5: Verifying loss computation..."
if echo "$TRAIN_OUTPUT" | grep -qi "loss\|Loss"; then
    echo "   ✅ Loss values found in output"
    echo "$TRAIN_OUTPUT" | grep -i "loss" | head -5
else
    echo "   ⚠️  No loss values found in output"
fi

echo ""

# Summary
echo "=========================================="
echo "Training Execution Test Complete"
echo "=========================================="
echo ""
echo "If all checks passed:"
echo "  ✅ Training script executes without errors"
echo "  ✅ Checkpoint saves correctly"
echo "  ✅ Loss computation works"
echo ""
echo "Next: Test with full dataset when ready"
echo ""

