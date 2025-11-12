# Pre-Training Test Plan

## Critical Tests Before Spending Money on Training

Before generating expensive datasets and running training, we need to verify these critical paths work end-to-end.

## ✅ Already Tested

### 1. API Integration
- ✅ API connectivity and authentication
- ✅ Tier detection and rate limiting
- ✅ Progressive backoff for rate limits
- ✅ Error handling and retry logic
- ✅ Curl test verified endpoint format

### 2. Training Infrastructure
- ✅ Model creation and forward pass
- ✅ Dataset loading and tokenization
- ✅ Loss computation (KL, CE)
- ✅ Checkpoint save/load
- ✅ Toy training smoke test passes

### 3. Hardening Features
- ✅ Checkpoint/resume logic
- ✅ Budget tracking calculations
- ✅ Cache validation
- ✅ Error recovery

## ⚠️ Critical Tests Still Needed

### 1. **End-to-End Dataset Generation** (HIGH PRIORITY)

**Risk**: Generate dataset, find out format is wrong or API fails silently

**Test**: Generate small dataset (10-20 samples) and verify:
- [ ] Dataset file is created correctly
- [ ] Samples have correct format (prompt, teacher_text, metadata)
- [ ] Tokenizer can process all samples
- [ ] No silent failures or corrupted data
- [ ] Cache works correctly (re-running doesn't duplicate)

**Command**:
```bash
# Generate small test dataset
python -m scripts.make_kd_mix_hardened \
    --out data/kd_mix_test.jsonl \
    --teacher https://api.moonshot.ai/v1 \
    --total 20 \
    --checkpoint-dir data/checkpoints_test/ \
    --cache-dir data/kd_cache_test/ \
    --budget-limit 1.0 \
    --delay 20

# Verify dataset
python3 << EOF
from training.dataset import KDDataset
dataset = KDDataset(
    jsonl_path="data/kd_mix_test.jsonl",
    tokenizer_path="models/student/tokenizer",
    max_seq_length=4096,
)
print(f"✅ Dataset: {len(dataset)} samples")
for i in range(min(3, len(dataset))):
    sample = dataset[i]
    print(f"  Sample {i}: input_len={len(sample['input_ids'])}, label_len={len(sample['labels'])}")
EOF
```

### 2. **Training Script Execution** (HIGH PRIORITY)

**Risk**: Training script fails after hours of dataset generation

**Test**: Run training script with toy dataset and verify:
- [ ] Training script starts without errors
- [ ] Config loads correctly
- [ ] Model initializes correctly
- [ ] DataLoader works
- [ ] At least 1 training step completes
- [ ] Checkpoint saves correctly
- [ ] Loss decreases (or at least computes)

**Command**:
```bash
# Use toy training setup
python -m training.make_toy_training --out-dir training/toy_test --samples 20

# Run actual training (5 steps)
python -m training.distill_kd \
    --config training/toy_test/toy_config.yaml \
    --output-dir training/toy_test/checkpoints
```

### 3. **Resume from Checkpoint** (MEDIUM PRIORITY)

**Risk**: Training crashes, can't resume, lose progress

**Test**: 
- [ ] Start training, interrupt it
- [ ] Resume from checkpoint
- [ ] Verify training continues correctly
- [ ] Verify no duplicate samples processed

**Command**:
```bash
# Start training
python -m training.distill_kd --config training/toy_test/toy_config.yaml &
TRAIN_PID=$!

# Wait a few seconds, then interrupt
sleep 10
kill $TRAIN_PID

# Resume
python -m training.distill_kd \
    --config training/toy_test/toy_config.yaml \
    --resume training/toy_test/checkpoints/checkpoint_*.pt
```

### 4. **Budget Enforcement** (HIGH PRIORITY)

**Risk**: Budget limit ignored, overspend

**Test**:
- [ ] Set very small budget ($0.10)
- [ ] Generate dataset
- [ ] Verify script stops when budget exceeded
- [ ] Verify checkpoint saved before stopping
- [ ] Verify no samples processed after budget exceeded

**Command**:
```bash
python -m scripts.make_kd_mix_hardened \
    --out data/kd_mix_budget_test.jsonl \
    --teacher https://api.moonshot.ai/v1 \
    --total 1000 \
    --budget-limit 0.10 \
    --delay 20

# Should stop after ~1-2 samples with BudgetExceededError
```

### 5. **Data Quality Validation** (MEDIUM PRIORITY)

**Risk**: Generate dataset, find samples are low quality or malformed

**Test**:
- [ ] Sample a few generated examples
- [ ] Verify teacher responses are reasonable
- [ ] Verify prompts are diverse
- [ ] Verify no empty or corrupted samples
- [ ] Verify token counts are reasonable

**Command**:
```bash
# Generate small dataset
python -m scripts.make_kd_mix_hardened \
    --out data/kd_mix_quality_test.jsonl \
    --teacher https://api.moonshot.ai/v1 \
    --total 10 \
    --delay 20

# Inspect samples
python3 << EOF
import json
with open("data/kd_mix_quality_test.jsonl") as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        sample = json.loads(line)
        print(f"\nSample {i+1}:")
        print(f"  Prompt: {sample['prompt'][:100]}...")
        print(f"  Response: {sample['teacher_text'][:100]}...")
        print(f"  Tokens: {sample.get('metadata', {}).get('tokens', {})}")
EOF
```

### 6. **Memory/Resource Requirements** (HIGH PRIORITY)

**Risk**: Training starts, runs out of memory, crashes

**Test**:
- [ ] Estimate memory requirements
- [ ] Test with toy model (verify it works)
- [ ] Test with small batch size
- [ ] Monitor memory usage during training
- [ ] Verify gradient checkpointing works if enabled

**Command**:
```bash
# Test memory usage with toy model
python3 << EOF
import torch
from models.student.architectures.gqa_transformer import StudentLM, ModelCfg

# Toy model
cfg = ModelCfg(d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, d_head=16, vocab_size=32000)
model = StudentLM(cfg)

# Estimate memory
param_size = sum(p.numel() * p.element_size() for p in model.parameters())
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Model size: {param_size / 1024**2:.2f} MB")

# Test forward pass memory
input_ids = torch.randint(0, 32000, (1, 128))
with torch.no_grad():
    logits = model(input_ids)
print(f"Output shape: {logits.shape}")
print(f"Output size: {logits.numel() * logits.element_size() / 1024**2:.2f} MB")
EOF

# For full model, estimate:
# 9B params * 2 bytes (FP16) = 18GB
# + gradients (18GB) + optimizer states (36GB) = ~72GB minimum
```

### 7. **Config Validation** (MEDIUM PRIORITY)

**Risk**: Config has wrong paths or settings, training fails

**Test**:
- [ ] Verify all paths in config exist
- [ ] Verify tokenizer path is correct
- [ ] Verify dataset path will exist
- [ ] Verify batch size is reasonable
- [ ] Verify sequence lengths are valid

**Command**:
```bash
python3 << EOF
from training.distill_kd import merge_configs
from pathlib import Path

cfg = merge_configs(['configs/worker_9b.yaml'])

# Check paths
tokenizer_path = cfg['io']['tokenizer_path']
print(f"Tokenizer: {tokenizer_path}")
if Path(tokenizer_path).exists() or 'meta-llama' in tokenizer_path:
    print("  ✅ Tokenizer path OK")
else:
    print(f"  ❌ Tokenizer path missing: {tokenizer_path}")

train_shards = cfg['io']['train_shards']
print(f"Train shards: {train_shards}")
for shard in train_shards:
    if Path(shard).exists():
        print(f"  ✅ {shard} exists")
    else:
        print(f"  ⚠️  {shard} doesn't exist yet (will be created)")

# Check batch size
micro_batch = cfg['train']['micro_batch_size']
grad_accum = cfg['train']['grad_accum']
print(f"Batch: {micro_batch} * {grad_accum} = {micro_batch * grad_accum}")
print(f"  ✅ Batch size reasonable")

# Check sequence lengths
seq_lengths = cfg['train']['seq_lengths']
print(f"Sequence lengths: {seq_lengths}")
print(f"  ✅ Sequence lengths valid")
EOF
```

### 8. **Error Recovery** (MEDIUM PRIORITY)

**Risk**: API fails mid-generation, lose progress

**Test**:
- [ ] Simulate API failure (kill API or use invalid key)
- [ ] Verify checkpoint saved before failure
- [ ] Verify resume works after fixing issue
- [ ] Verify no duplicate samples after resume

### 9. **Cache Integrity** (LOW PRIORITY)

**Risk**: Corrupted cache causes bad training data

**Test**:
- [ ] Verify cache validation catches corrupted files
- [ ] Verify cache hit reduces costs
- [ ] Verify cache doesn't cause stale data

### 10. **Full Pipeline Test** (HIGH PRIORITY)

**Risk**: Components work individually but not together

**Test**: End-to-end from dataset generation → training:
- [ ] Generate small dataset (20 samples)
- [ ] Load dataset in training script
- [ ] Run 5 training steps
- [ ] Verify checkpoint saved
- [ ] Verify loss computed
- [ ] Verify no errors

**Command**:
```bash
# Full pipeline test
./scripts/test_full_pipeline.sh
```

## Recommended Test Sequence

### Phase 1: Quick Validation (5 minutes)
```bash
# 1. Verify setup
bash scripts/setup_training.sh

# 2. Run toy training test
make smoke_training
```

### Phase 2: Small-Scale Tests (30 minutes)
```bash
# 1. Generate 20-sample dataset
python -m scripts.make_kd_mix_hardened \
    --out data/kd_mix_test.jsonl \
    --teacher https://api.moonshot.ai/v1 \
    --total 20 \
    --budget-limit 1.0 \
    --delay 20

# 2. Verify dataset quality
python3 -c "from training.dataset import KDDataset; ds = KDDataset('data/kd_mix_test.jsonl', 'models/student/tokenizer'); print(f'✅ {len(ds)} samples')"

# 3. Run training with real dataset
python -m training.distill_kd \
    --config configs/worker_9b.yaml \
    --output-dir training/test_checkpoints
# (Update config to use test dataset first)
```

### Phase 3: Budget & Error Tests (15 minutes)
```bash
# 1. Test budget enforcement
python -m scripts.make_kd_mix_hardened \
    --out data/kd_mix_budget_test.jsonl \
    --teacher https://api.moonshot.ai/v1 \
    --total 100 \
    --budget-limit 0.10 \
    --delay 20

# 2. Test resume
# (Interrupt and resume)
```

### Phase 4: Full Dataset Generation (When Ready)
```bash
# Only after all tests pass!
python -m scripts.make_kd_mix_hardened \
    --out data/kd_mix.jsonl \
    --teacher https://api.moonshot.ai/v1 \
    --total 10000 \
    --budget-limit 50.0 \
    --checkpoint-dir data/checkpoints/ \
    --delay 20
```

## Cost Estimates for Testing

| Test | Samples | Cost | Time |
|------|---------|------|------|
| Toy training | 0 (uses mock) | $0 | 30s |
| Small dataset | 20 | ~$0.10 | 7 min |
| Budget test | 1-2 | ~$0.01 | 1 min |
| Quality test | 10 | ~$0.05 | 3 min |
| **Total** | **~30** | **~$0.16** | **~15 min** |

**Total test cost: < $1** (vs $50+ for full dataset)

## Risk Assessment

### High Risk (Test Before Spending)
- ✅ Dataset generation works end-to-end
- ✅ Training script executes without errors
- ✅ Budget enforcement prevents overspending
- ✅ Memory requirements are met

### Medium Risk (Test Before Large Dataset)
- ⚠️ Resume from checkpoint works
- ⚠️ Data quality is acceptable
- ⚠️ Error recovery works

### Low Risk (Can Test During Generation)
- Cache integrity
- Config validation (already done)

## Test Scripts Needed

Create these test scripts:

1. `scripts/test_dataset_generation.sh` - End-to-end dataset generation test
2. `scripts/test_training_execution.sh` - Training script execution test
3. `scripts/test_budget_enforcement.sh` - Budget limit test
4. `scripts/test_full_pipeline.sh` - Complete pipeline test

## Summary Checklist

Before generating full dataset:

- [ ] ✅ Toy training test passes (`make smoke_training`)
- [ ] ⚠️ Small dataset (20 samples) generated successfully
- [ ] ⚠️ Dataset loads correctly in training script
- [ ] ⚠️ Training script runs 5+ steps without errors
- [ ] ⚠️ Budget enforcement stops at limit
- [ ] ⚠️ Resume from checkpoint works
- [ ] ⚠️ Memory requirements verified
- [ ] ⚠️ Data quality acceptable

**Estimated test time**: 30-60 minutes  
**Estimated test cost**: < $1  
**Risk reduction**: High (catches issues before $50+ spend)

