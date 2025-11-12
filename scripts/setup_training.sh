#!/bin/bash
#
# Setup script for training environment.
#
# This script:
# 1. Installs missing dependencies
# 2. Downloads tokenizer
# 3. Verifies configuration
# 4. Tests dataset loading
#
# Usage:
#   ./scripts/setup_training.sh
#

set -euo pipefail

echo "=========================================="
echo "Training Environment Setup"
echo "=========================================="
echo ""

# Check if virtual environment is activated
if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "⚠️  Virtual environment not detected"
    echo "   Activating venv..."
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    else
        echo "❌ Virtual environment not found. Create one with: python3 -m venv venv"
        exit 1
    fi
fi

echo "✅ Virtual environment: $VIRTUAL_ENV"
echo ""

# Step 1: Install dependencies
echo "Step 1: Installing dependencies..."
pip install -q transformers tensorboard || {
    echo "❌ Failed to install dependencies"
    exit 1
}
echo "✅ Dependencies installed"
echo ""

# Step 2: Download tokenizer
echo "Step 2: Setting up tokenizer..."
if [ -d "models/student/tokenizer" ] && [ -f "models/student/tokenizer/tokenizer_config.json" ]; then
    echo "✅ Tokenizer already exists"
else
    echo "   Downloading Llama-2 tokenizer..."
    python3 << 'PYTHON_EOF'
from transformers import AutoTokenizer
import os

tokenizer_path = "models/student/tokenizer"
os.makedirs(tokenizer_path, exist_ok=True)

print(f"   Downloading to {tokenizer_path}...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.save_pretrained(tokenizer_path)
print(f"   ✅ Tokenizer saved (vocab_size={len(tokenizer)})")
PYTHON_EOF
fi
echo ""

# Step 3: Verify tokenizer
echo "Step 3: Verifying tokenizer..."
python3 << 'PYTHON_EOF'
from training.dataset import load_tokenizer

try:
    tokenizer = load_tokenizer("models/student/tokenizer")
    print(f"   ✅ Tokenizer loaded (vocab_size={len(tokenizer)})")
    
    # Test encoding
    test_text = "Hello, world!"
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    if decoded == test_text:
        print(f"   ✅ Encoding test passed ({len(tokens)} tokens)")
    else:
        print(f"   ⚠️  Encoding round-trip mismatch")
except Exception as e:
    print(f"   ❌ Tokenizer verification failed: {e}")
    exit(1)
PYTHON_EOF
echo ""

# Step 4: Verify config
echo "Step 4: Verifying training configuration..."
python3 << 'PYTHON_EOF'
from training.distill_kd import load_config, merge_configs

try:
    cfg = merge_configs(['configs/worker_9b.yaml'])
    print(f"   ✅ Config loaded")
    print(f"      Model: {cfg['arch']['d_model']}d_model, {cfg['arch']['n_layers']} layers")
    print(f"      Batch: {cfg['train']['micro_batch_size']} * {cfg['train']['grad_accum']} = {cfg['train']['micro_batch_size'] * cfg['train']['grad_accum']}")
    print(f"      Tokenizer: {cfg['io']['tokenizer_path']}")
    print(f"      Train shards: {cfg['io']['train_shards']}")
except Exception as e:
    print(f"   ❌ Config verification failed: {e}")
    exit(1)
PYTHON_EOF
echo ""

# Step 5: Test dataset loading
echo "Step 5: Testing dataset loading..."
if [ -f "data/kd_mix.jsonl" ]; then
    python3 << 'PYTHON_EOF'
from training.dataset import KDDataset

try:
    dataset = KDDataset(
        jsonl_path="data/kd_mix.jsonl",
        tokenizer_path="models/student/tokenizer",
        max_seq_length=4096,
    )
    print(f"   ✅ Dataset loaded: {len(dataset)} samples")
    
    if len(dataset) == 0:
        print(f"   ⚠️  Dataset is empty - need to generate training data")
    elif len(dataset) < 100:
        print(f"   ⚠️  Dataset is small ({len(dataset)} samples) - recommend 10k+ for training")
    else:
        print(f"   ✅ Dataset size looks good")
except Exception as e:
    print(f"   ⚠️  Dataset loading failed (may be expected if dataset is empty): {e}")
PYTHON_EOF
else
    echo "   ⚠️  Dataset file not found: data/kd_mix.jsonl"
    echo "      Generate with: python -m scripts.make_kd_mix_hardened --out data/kd_mix.jsonl --teacher https://api.moonshot.ai/v1 --total 10000"
fi
echo ""

# Summary
echo "=========================================="
echo "Setup Summary"
echo "=========================================="
echo "✅ Dependencies installed"
echo "✅ Tokenizer configured"
echo "✅ Configuration verified"
echo ""
echo "Next steps:"
echo "1. Generate training dataset (if not done):"
echo "   python -m scripts.make_kd_mix_hardened \\"
echo "       --out data/kd_mix.jsonl \\"
echo "       --teacher https://api.moonshot.ai/v1 \\"
echo "       --total 10000 \\"
echo "       --checkpoint-dir data/checkpoints/ \\"
echo "       --budget-limit 50.0 \\"
echo "       --delay 20"
echo ""
echo "2. Start training:"
echo "   python -m training.distill_kd --config configs/worker_9b.yaml"
echo ""
echo "3. Monitor training:"
echo "   tensorboard --logdir runs/"
echo ""

