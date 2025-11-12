#!/bin/bash
#
# Smoke test for training pipeline using toy model and dataset.
#
# This test:
# 1. Creates a tiny model and dataset
# 2. Runs a few training steps
# 3. Verifies checkpoint saving
# 4. Verifies loss computation
#
# Usage:
#   ./scripts/test_toy_training.sh
#

set -euo pipefail

echo "=========================================="
echo "Toy Training Smoke Test"
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

TOY_DIR="training/toy_test"
TOY_CONFIG="${TOY_DIR}/toy_config.yaml"

# Step 1: Create toy training setup
echo "Step 1: Creating toy training setup..."
python -m training.make_toy_training \
    --out-dir "${TOY_DIR}" \
    --samples 10 \
    --steps 5 \
    --dmodel 64 \
    --nlayers 2 \
    --vocab 32000 \
    --tokenizer models/student/tokenizer

if [ ! -f "${TOY_CONFIG}" ]; then
    echo "❌ Failed to create toy config"
    exit 1
fi

echo "✅ Toy setup created"
echo ""

# Step 2: Verify dataset loads
echo "Step 2: Verifying dataset loading..."
python3 << 'PYTHON_EOF'
from training.dataset import KDDataset
import sys

try:
    dataset = KDDataset(
        jsonl_path="training/toy_test/toy_dataset.jsonl",
        tokenizer_path="models/student/tokenizer",
        max_seq_length=128,
    )
    print(f"   ✅ Dataset loaded: {len(dataset)} samples")
    
    if len(dataset) == 0:
        print("   ❌ Dataset is empty")
        sys.exit(1)
    
    # Test getting a sample
    sample = dataset[0]
    assert "input_ids" in sample
    assert "labels" in sample
    print(f"   ✅ Sample structure correct")
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

# Step 3: Verify model creation
echo "Step 3: Verifying model creation..."
python3 << 'PYTHON_EOF'
from models.student.architectures.gqa_transformer import StudentLM, ModelCfg
import torch
import sys

try:
    cfg = ModelCfg(
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_head=16,
        vocab_size=32000,
        rope_theta=10000.0,
        rope_scaling="dynamic",
        dropout=0.0,
    )
    
    model = StudentLM(cfg)
    print(f"   ✅ Model created")
    
    # Test forward pass (use vocab_size from config)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 10))
    with torch.no_grad():
        logits = model(input_ids)
    
    assert logits.shape == (1, 10, cfg.vocab_size)
    print(f"   ✅ Forward pass works (output shape: {logits.shape})")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ Model parameters: {total_params:,}")
except Exception as e:
    print(f"   ❌ Model creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYTHON_EOF

if [ $? -ne 0 ]; then
    exit 1
fi

echo ""

# Step 4: Test training step (without full training loop)
echo "Step 4: Testing training step..."
python3 << 'PYTHON_EOF'
from models.student.architectures.gqa_transformer import StudentLM, ModelCfg
from training.dataset import KDDataset, collate_kd_batch
from training.losses import combined_kd_loss
from torch.utils.data import DataLoader
import torch
import sys

try:
    # Create tiny model
    cfg = ModelCfg(
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_head=16,
        vocab_size=32000,
        rope_theta=10000.0,
        rope_scaling="dynamic",
        dropout=0.0,
    )
    
    model = StudentLM(cfg)
    model.train()
    
    # Create dataset and dataloader
    dataset = KDDataset(
        jsonl_path="training/toy_test/toy_dataset.jsonl",
        tokenizer_path="models/student/tokenizer",
        max_seq_length=128,
    )
    
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_kd_batch)
    
    # Get one batch
    batch = next(iter(dataloader))
    
    # Forward pass
    input_ids = batch["input_ids"]
    student_logits = model(input_ids)
    
    # Cast to FP32 for loss computation (model outputs FP16)
    student_logits = student_logits.float()
    
    # Create dummy teacher logits
    teacher_logits = torch.randn_like(student_logits)
    
    # Compute loss
    loss_dict = combined_kd_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        teacher_targets=teacher_logits.argmax(dim=-1),
        ground_truth_targets=batch["labels"],
        kl_weight=0.5,
        ce_teacher_weight=0.3,
        ce_ground_truth_weight=0.2,
        kd_temperature=2.0,
    )
    
    print(f"   ✅ Training step works")
    print(f"      Loss: {loss_dict['total'].item():.4f}")
    print(f"      KL: {loss_dict['kl'].item():.4f}")
    print(f"      CE teacher: {loss_dict['ce_teacher'].item():.4f}")
    print(f"      CE GT: {loss_dict['ce_ground_truth'].item():.4f}")
    
    # Test backward pass
    loss_dict["total"].backward()
    print(f"   ✅ Backward pass works")
    
except Exception as e:
    print(f"   ❌ Training step failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYTHON_EOF

if [ $? -ne 0 ]; then
    exit 1
fi

echo ""

# Step 5: Test checkpoint save/load
echo "Step 5: Testing checkpoint save/load..."
python3 << 'PYTHON_EOF'
from models.student.architectures.gqa_transformer import StudentLM, ModelCfg
import torch
import sys
from pathlib import Path

try:
    cfg = ModelCfg(
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_head=16,
        vocab_size=32000,
        rope_theta=10000.0,
        rope_scaling="dynamic",
        dropout=0.0,
    )
    
    # Create and save model
    model1 = StudentLM(cfg)
    checkpoint_path = Path("training/toy_test/test_checkpoint.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        "model_state_dict": model1.state_dict(),
        "step": 5,
        "loss": 0.5,
    }, checkpoint_path)
    
    # Load into new model
    model2 = StudentLM(cfg)
    checkpoint = torch.load(checkpoint_path)
    model2.load_state_dict(checkpoint["model_state_dict"])
    
    # Verify weights match
    input_ids = torch.randint(0, cfg.vocab_size, (1, 10))
    with torch.no_grad():
        logits1 = model1(input_ids)
        logits2 = model2(input_ids)
    
    assert torch.allclose(logits1, logits2, atol=1e-5)
    print(f"   ✅ Checkpoint save/load works")
    print(f"      Step: {checkpoint['step']}")
    print(f"      Loss: {checkpoint['loss']}")
    
except Exception as e:
    print(f"   ❌ Checkpoint test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYTHON_EOF

if [ $? -ne 0 ]; then
    exit 1
fi

echo ""

# Summary
echo "=========================================="
echo "✅ Toy Training Smoke Test PASSED"
echo "=========================================="
echo ""
echo "All components verified:"
echo "  ✅ Dataset loading"
echo "  ✅ Model creation"
echo "  ✅ Forward pass"
echo "  ✅ Loss computation"
echo "  ✅ Backward pass"
echo "  ✅ Checkpoint save/load"
echo ""
echo "Training pipeline is ready!"
echo ""
echo "To run full toy training:"
echo "  python -m training.distill_kd --config ${TOY_CONFIG}"
echo ""

