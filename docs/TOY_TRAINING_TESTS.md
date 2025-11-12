# Toy Training Tests

## Overview

Similar to the conversion toy tests (`make_toy_block`, `make_toy_torch`, `make_toy_onnx`), we have toy training tests that verify the training pipeline works without requiring the full 9B model or 10k+ dataset.

## Purpose

**Why toy tests?**
- **Quick verification**: Test training pipeline in seconds vs hours
- **No dependencies**: Don't need full dataset or GPU
- **CI/CD friendly**: Can run in automated pipelines
- **Debugging**: Isolate issues before full training runs

## Components

### 1. `training/make_toy_training.py`

Creates a minimal training setup:
- **Tiny model**: 64d_model, 2 layers, 32000 vocab (matches tokenizer)
- **Tiny dataset**: 10 samples (enough for a few training steps)
- **Minimal config**: Optimized for quick testing

**Usage**:
```bash
python -m training.make_toy_training \
    --out-dir training/toy_test \
    --samples 10 \
    --steps 5 \
    --dmodel 64 \
    --nlayers 2 \
    --vocab 32000
```

**Outputs**:
- `training/toy_test/toy_dataset.jsonl` - Tiny KD dataset
- `training/toy_test/toy_config.yaml` - Training config
- `training/toy_test/toy_model_init.pt` - Initialized model checkpoint

### 2. `scripts/test_toy_training.sh`

Smoke test script that verifies:
1. ✅ Dataset loading works
2. ✅ Model creation works
3. ✅ Forward pass works
4. ✅ Loss computation works
5. ✅ Backward pass works
6. ✅ Checkpoint save/load works

**Usage**:
```bash
bash scripts/test_toy_training.sh
```

**What it tests**:
- Dataset loader can load JSONL and tokenize
- Model can be created with tiny config
- Forward pass produces correct output shapes
- Loss functions compute correctly
- Gradients can be computed
- Checkpoints can be saved and loaded

### 3. Makefile Integration

Added to Makefile for consistency with conversion tests:

```bash
# Create toy training setup
make toy-training

# Run full smoke test
make smoke_training
```

## Comparison with Conversion Toy Tests

| Aspect | Conversion Tests | Training Tests |
|--------|-----------------|----------------|
| **Purpose** | Test ONNX→CoreML conversion | Test training pipeline |
| **Model Size** | Tiny transformer block | Tiny full model (64d, 2 layers) |
| **What's Tested** | Export, conversion, probes | Dataset, model, loss, checkpoint |
| **Time** | Seconds | Seconds |
| **Dependencies** | ONNX, CoreML tools | PyTorch, transformers |

## Example Output

```
==========================================
Toy Training Smoke Test
==========================================

Step 1: Creating toy training setup...
✅ Toy setup created

Step 2: Verifying dataset loading...
   ✅ Dataset loaded: 10 samples
   ✅ Sample structure correct

Step 3: Verifying model creation...
   ✅ Model created
   ✅ Forward pass works (output shape: torch.Size([1, 10, 32000]))
   ✅ Model parameters: 4,219,200

Step 4: Testing training step...
   ✅ Training step works
      Loss: 5.2972
      Loss components: ['kl_div', 'ce_teacher', 'ce_ground_truth', 'total']
   ✅ Backward pass works

Step 5: Testing checkpoint save/load...
   ✅ Checkpoint save/load works

==========================================
✅ Toy Training Smoke Test PASSED
==========================================
```

## When to Use

### ✅ Use Toy Tests For:
- **CI/CD pipelines**: Quick verification before merge
- **Development**: Test changes to training code
- **Debugging**: Isolate training pipeline issues
- **Documentation**: Demonstrate training setup works

### ❌ Don't Use For:
- **Full training validation**: Need real dataset/model
- **Performance testing**: Too small to be representative
- **Accuracy validation**: Model too small to learn meaningfully

## Integration with Full Training

Toy tests verify the **pipeline works**, but full training requires:

1. **Real dataset**: 10k+ samples from `make_kd_mix_hardened.py`
2. **Full model**: 9B parameters (configs/worker_9b.yaml)
3. **GPU**: For reasonable training time
4. **Monitoring**: TensorBoard/WandB for progress

## Next Steps

After toy tests pass:

1. **Generate real dataset**:
   ```bash
   python -m scripts.make_kd_mix_hardened \
       --out data/kd_mix.jsonl \
       --teacher https://api.moonshot.ai/v1 \
       --total 10000
   ```

2. **Start full training**:
   ```bash
   python -m training.distill_kd --config configs/worker_9b.yaml
   ```

3. **Monitor progress**:
   ```bash
   tensorboard --logdir runs/
   ```

## Status

| Component | Status | Notes |
|-----------|--------|-------|
| `make_toy_training.py` | ✅ Complete | Creates tiny training setup |
| `test_toy_training.sh` | ✅ Complete | Smoke test script |
| Makefile integration | ✅ Complete | `make smoke_training` |
| Documentation | ✅ Complete | This document |

## Related Tests

- **Conversion toy tests**: `make smoke_toy`, `make smoke_torch`
- **Unit tests**: `tests/unit/` - Component-level tests
- **Integration tests**: `tests/integration/` - Full pipeline tests

