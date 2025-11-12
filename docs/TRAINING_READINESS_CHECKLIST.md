# Training Readiness Checklist

## Overview

This checklist ensures all components are ready for model training. Run through each section systematically before starting training.

## ✅ Completed Components

### 1. Training Infrastructure
- ✅ Training scripts implemented (`distill_kd.py`, `distill_process.py`, `distill_tool_select.py`, etc.)
- ✅ Dataset loaders implemented (`dataset.py`, `dataset_tool_select.py`, `dataset_post_tool.py`, `dataset_final.py`)
- ✅ Loss functions implemented (`losses.py`, `process_losses.py`)
- ✅ Training tracing/logging implemented (`tracing.py`)
- ✅ Model architecture implemented (`gqa_transformer.py`)

### 2. API Integration
- ✅ Teacher API client with error handling (`teacher_client.py`)
- ✅ Tier-aware rate limiting and backoff
- ✅ Dataset generation scripts (`make_kd_mix.py`, `make_kd_mix_hardened.py`)
- ✅ Proxy server for trace capture (`proxy_server.py`)

### 3. Constrained Decoding
- ✅ JSON constrained decoder (`constrained_decode.py`)
- ✅ Schema registry (`schema_registry.py`)
- ✅ Unit tests (28 tests passing)

### 4. Testing
- ✅ Unit tests for core components
- ✅ Integration tests for checkpoint/resume
- ✅ Integration tests for budget tracking
- ✅ API test scripts

## ⚠️ Required Before Training

### 1. Dataset Generation

**Status**: ⚠️ **REQUIRED**

**Action Items**:
- [ ] Generate KD dataset using `make_kd_mix_hardened.py`
- [ ] Verify dataset format matches expected JSONL structure
- [ ] Check dataset size (recommended: 10k+ samples for initial training)
- [ ] Validate dataset quality (sample a few examples)

**Commands**:
```bash
# Generate dataset (free tier: ~5 days for 10k samples)
python -m scripts.make_kd_mix_hardened \
    --out data/kd_mix.jsonl \
    --teacher https://api.moonshot.ai/v1 \
    --total 10000 \
    --checkpoint-dir data/checkpoints/ \
    --cache-dir data/kd_cache/ \
    --budget-limit 50.0 \
    --checkpoint-interval 100 \
    --delay 20  # Free tier: 3 RPM = 20s delay

# Verify dataset
head -5 data/kd_mix.jsonl | python -m json.tool
wc -l data/kd_mix.jsonl
```

**Expected Format**:
```json
{
  "prompt": "user question or instruction",
  "teacher_text": "teacher model response",
  "metadata": {
    "source": "general|domain|tool",
    "tokens": {"input": 200, "output": 1024}
  }
}
```

### 2. Tokenizer Setup

**Status**: ⚠️ **REQUIRED**

**Action Items**:
- [ ] Download or configure tokenizer
- [ ] Verify tokenizer path in config matches actual location
- [ ] Test tokenizer loading

**Options**:

**Option A: Use HuggingFace Tokenizer** (Recommended)
```bash
# Download tokenizer (e.g., Llama-2 tokenizer)
python3 << EOF
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.save_pretrained("models/student/tokenizer")
print("✅ Tokenizer saved to models/student/tokenizer")
EOF
```

**Option B: Use Existing Tokenizer**
- Update `configs/worker_9b.yaml`:
  ```yaml
  io:
    tokenizer_path: "path/to/your/tokenizer"  # HuggingFace format
  ```

**Verification**:
```bash
python3 << EOF
from training.dataset import load_tokenizer
tokenizer = load_tokenizer("models/student/tokenizer")
print(f"✅ Tokenizer loaded: vocab_size={len(tokenizer)}")
test_text = "Hello, world!"
tokens = tokenizer.encode(test_text)
print(f"Test encoding: {tokens}")
EOF
```

### 3. Model Checkpoint

**Status**: ⚠️ **REQUIRED**

**Action Items**:
- [ ] Initialize model from scratch OR load base checkpoint
- [ ] Verify model architecture matches config
- [ ] Check model size (9B params ≈ 18GB in FP16)

**Option A: Initialize from Scratch**
```python
# Model will be initialized randomly
# Set in config:
init:
  base_checkpoint: null
```

**Option B: Load Base Checkpoint**
```yaml
# In configs/worker_9b.yaml:
init:
  base_checkpoint: "path/to/base/checkpoint.pt"
```

**Verification**:
```bash
python3 << EOF
import torch
from models.student.architectures.gqa_transformer import StudentLM, ModelCfg
from configs.worker_9b import load_config

cfg = load_config("configs/worker_9b.yaml")
model_cfg = ModelCfg(**cfg["arch"])
model = StudentLM(model_cfg)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ Model created: {total_params/1e9:.2f}B total params")
print(f"   Trainable: {trainable_params/1e9:.2f}B")
EOF
```

### 4. Dependencies

**Status**: ⚠️ **VERIFY**

**Required Packages**:
- [ ] PyTorch (with CUDA if using GPU)
- [ ] transformers (for tokenizer)
- [ ] tensorboard (for visualization)
- [ ] yaml (for config loading)

**Check Installation**:
```bash
# Check PyTorch
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Check transformers
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Check tensorboard
python3 -c "import tensorboard; print('TensorBoard installed')" || pip install tensorboard

# Install missing dependencies
pip install transformers tensorboard pyyaml
```

### 5. Configuration

**Status**: ⚠️ **REVIEW**

**Action Items**:
- [ ] Review `configs/worker_9b.yaml` settings
- [ ] Adjust batch size based on GPU memory
- [ ] Set learning rate schedule
- [ ] Configure checkpoint saving frequency
- [ ] Set up tracing/logging

**Key Config Settings**:
```yaml
train:
  micro_batch_size: 2      # Adjust based on GPU memory
  grad_accum: 16           # Effective batch = 2 * 16 = 32
  steps: 200000            # Total training steps
  fp16: true               # Use FP16 for memory efficiency
  grad_checkpointing: true # Trade compute for memory

io:
  tokenizer_path: "models/student/tokenizer"  # Verify path
  train_shards: ["data/kd_mix.jsonl"]        # Verify dataset path

tracing:
  use_tensorboard: true    # Enable TensorBoard
  use_wandb: false         # Optional: enable for cloud tracking
```

### 6. Hardware Requirements

**Status**: ⚠️ **VERIFY**

**Minimum Requirements**:
- GPU: 24GB+ VRAM (for 9B model in FP16)
- RAM: 32GB+
- Storage: 100GB+ free (for checkpoints, datasets, logs)

**Check GPU**:
```bash
# NVIDIA GPU
nvidia-smi

# Apple Silicon (M1/M2/M3)
system_profiler SPDisplaysDataType
```

**Memory Estimation**:
- Model (FP16): ~18GB
- Gradients: ~18GB
- Optimizer states (AdamW): ~36GB
- Activations: ~4-8GB (depends on batch size)
- **Total**: ~80GB+ VRAM needed for full training

**Options for Limited Memory**:
- Use gradient checkpointing (already enabled)
- Reduce `micro_batch_size` to 1
- Increase `grad_accum` to maintain effective batch size
- Use CPU offloading (slower but works)

### 7. Training Scripts Integration

**Status**: ⚠️ **VERIFY**

**Action Items**:
- [ ] Verify `distill_kd.py` can load config
- [ ] Test dataset loading
- [ ] Test model creation
- [ ] Run a single training step (dry run)

**Dry Run Test**:
```bash
# Test config loading
python3 -c "
from training.distill_kd import load_config, merge_configs
cfg = merge_configs(['configs/worker_9b.yaml'])
print('✅ Config loaded')
print(f'Model: {cfg[\"arch\"][\"d_model\"]}d_model, {cfg[\"arch\"][\"n_layers\"]} layers')
print(f'Batch size: {cfg[\"train\"][\"micro_batch_size\"]} * {cfg[\"train\"][\"grad_accum\"]} = {cfg[\"train\"][\"micro_batch_size\"] * cfg[\"train\"][\"grad_accum\"]}')
"

# Test dataset loading (requires dataset and tokenizer)
python3 << EOF
from training.dataset import KDDataset
try:
    dataset = KDDataset(
        jsonl_path="data/kd_mix.jsonl",
        tokenizer_path="models/student/tokenizer",
        max_seq_length=4096,
    )
    print(f"✅ Dataset loaded: {len(dataset)} samples")
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"   Sample keys: {sample.keys()}")
except Exception as e:
    print(f"❌ Dataset loading failed: {e}")
EOF
```

## Training Workflow

### Stage 1: Knowledge Distillation (KD)
```bash
python -m training.distill_kd \
    --config configs/worker_9b.yaml \
    --output-dir models/student/checkpoints
```

**What it does**:
- Trains student model to mimic teacher outputs
- Uses KL divergence + cross-entropy losses
- Saves checkpoints periodically

**Monitor**:
```bash
# Terminal 1: Training
python -m training.distill_kd --config configs/worker_9b.yaml

# Terminal 2: TensorBoard
tensorboard --logdir runs/

# Terminal 3: Monitor logs
tail -f runs/*/metrics.jsonl | jq
```

### Stage 2: Process Supervision (Tool Selection)
```bash
python -m training.distill_tool_select \
    --config configs/worker_9b.yaml \
    --resume models/student/checkpoints/checkpoint_100000.pt
```

**What it does**:
- Trains model to select correct tools
- Generates valid JSON arguments
- Uses constrained decoding

### Stage 3: Post-Tool Integration
```bash
python -m training.distill_post_tool \
    --config configs/worker_9b.yaml \
    --resume models/student/checkpoints/checkpoint_150000.pt
```

**What it does**:
- Trains model to integrate tool results
- Generates follow-up reasoning/code

### Stage 4: Final Answer Generation
```bash
python -m training.distill_final \
    --config configs/worker_9b.yaml \
    --resume models/student/checkpoints/checkpoint_180000.pt
```

**What it does**:
- Trains model to generate final answers
- Completes tool-use workflows

## Quick Start Checklist

Before running training, ensure:

- [ ] **Dataset exists**: `data/kd_mix.jsonl` with 10k+ samples
- [ ] **Tokenizer configured**: `models/student/tokenizer` exists
- [ ] **Config reviewed**: `configs/worker_9b.yaml` settings verified
- [ ] **Dependencies installed**: PyTorch, transformers, tensorboard
- [ ] **GPU available**: 24GB+ VRAM or CPU offloading configured
- [ ] **Dry run passed**: Config and dataset loading work

## Troubleshooting

### Dataset Not Found
```bash
# Generate dataset first
python -m scripts.make_kd_mix_hardened --out data/kd_mix.jsonl --teacher https://api.moonshot.ai/v1 --total 1000
```

### Tokenizer Not Found
```bash
# Download tokenizer
python3 << EOF
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.save_pretrained("models/student/tokenizer")
EOF
```

### Out of Memory
- Reduce `micro_batch_size` to 1
- Increase `grad_accum` to maintain batch size
- Enable `grad_checkpointing: true`
- Use FP16: `fp16: true`

### Training Too Slow
- Check GPU utilization: `nvidia-smi`
- Verify data loading: check DataLoader `num_workers`
- Profile training loop: add timing logs

## Next Steps After Checklist Complete

1. **Start KD Training**: Run `distill_kd.py` with small dataset first
2. **Monitor Progress**: Watch TensorBoard and logs
3. **Validate Checkpoints**: Test model inference periodically
4. **Iterate**: Adjust hyperparameters based on validation metrics
5. **Proceed to Next Stage**: Move to process supervision after KD converges

## Status Summary

| Component | Status | Priority | Notes |
|-----------|--------|----------|-------|
| Training Scripts | ✅ Ready | - | All scripts implemented |
| Dataset Generation | ⚠️ Required | HIGH | Need to generate KD dataset |
| Tokenizer | ⚠️ Required | HIGH | Need to download/configure |
| Model Checkpoint | ⚠️ Required | HIGH | Initialize or load base |
| Dependencies | ⚠️ Verify | MEDIUM | Check PyTorch, transformers |
| Configuration | ⚠️ Review | MEDIUM | Review configs |
| Hardware | ⚠️ Verify | HIGH | Check GPU/RAM |
| Testing | ✅ Ready | - | Dry run before full training |

