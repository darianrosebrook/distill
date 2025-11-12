# Critical Tests Before Training - Quick Reference

## TL;DR

**Run this before spending money:**
```bash
./scripts/run_pre_training_tests.sh
```

**Cost**: < $1  
**Time**: 15-30 minutes  
**Risk Reduction**: High (catches issues before $50+ spend)

## Test Checklist

### ✅ Free Tests (No API Cost)

1. **Toy Training Smoke Test**
   ```bash
   make smoke_training
   ```
   - Verifies training pipeline components work
   - Tests: dataset loading, model creation, loss computation, checkpointing
   - **Time**: 30 seconds

2. **Training Execution Test**
   ```bash
   bash scripts/test_training_execution.sh
   ```
   - Runs training script with toy dataset
   - Verifies training completes without errors
   - **Time**: 2-5 minutes

### 💰 Low-Cost Tests (< $1)

3. **Dataset Generation Test** (~$0.10)
   ```bash
   bash scripts/test_dataset_generation.sh
   ```
   - Generates 20 samples
   - Verifies format, quality, and loading
   - **Time**: ~7 minutes

4. **Budget Enforcement Test** (~$0.01)
   ```bash
   bash scripts/test_budget_enforcement.sh
   ```
   - Verifies budget limit stops generation
   - Verifies checkpoint saved before stopping
   - **Time**: ~1 minute

## What These Tests Catch

### Before They Cost Money:

- ❌ **Dataset format errors** → Caught by dataset generation test
- ❌ **Training script failures** → Caught by training execution test
- ❌ **Budget not enforced** → Caught by budget enforcement test
- ❌ **Checkpoint/resume broken** → Caught by toy training test
- ❌ **Memory issues** → Caught by training execution test
- ❌ **Config errors** → Caught by all tests

### After Tests Pass:

✅ Dataset generation will work  
✅ Training script will execute  
✅ Budget will prevent overspending  
✅ Checkpoints will save correctly  
✅ Resume will work if interrupted  

## Quick Test Sequence

```bash
# Option 1: Run all tests automatically
./scripts/run_pre_training_tests.sh

# Option 2: Run individually
make smoke_training                    # Free, 30s
bash scripts/test_dataset_generation.sh  # ~$0.10, 7min
bash scripts/test_budget_enforcement.sh   # ~$0.01, 1min
bash scripts/test_training_execution.sh    # Free, 5min
```

## Cost Breakdown

| Test | Samples | Cost | Time | Critical? |
|------|---------|------|------|----------|
| Toy training | 0 | $0 | 30s | ✅ Yes |
| Dataset generation | 20 | ~$0.10 | 7min | ✅ Yes |
| Budget enforcement | 1-2 | ~$0.01 | 1min | ✅ Yes |
| Training execution | 0 | $0 | 5min | ✅ Yes |
| **Total** | **~22** | **~$0.11** | **~15min** | |

## What Happens If Tests Fail?

### Dataset Generation Fails
- **Fix**: Check API key, network, endpoint URL
- **Don't proceed** until fixed

### Training Execution Fails
- **Fix**: Check config, tokenizer, dependencies
- **Don't proceed** until fixed

### Budget Enforcement Fails
- **Fix**: Review budget tracking code
- **Don't proceed** until fixed (could overspend)

### Toy Training Fails
- **Fix**: Review training code, model architecture
- **Don't proceed** until fixed

## After Tests Pass

You're ready to:

1. **Generate full dataset** (10k+ samples, ~$50)
2. **Start training** (200k steps, days/weeks)
3. **Monitor progress** (TensorBoard, logs)

## Risk Assessment

### High Risk (Test Before Spending)
- ✅ Dataset generation works end-to-end
- ✅ Training script executes without errors
- ✅ Budget enforcement prevents overspending

### Medium Risk (Test Before Large Dataset)
- ⚠️ Resume from checkpoint works
- ⚠️ Data quality is acceptable

### Low Risk (Can Test During Generation)
- Cache integrity
- Config validation

## Summary

**Before spending $50+ on dataset generation:**

1. ✅ Run `make smoke_training` (free, 30s)
2. ✅ Run `bash scripts/test_dataset_generation.sh` (~$0.10, 7min)
3. ✅ Run `bash scripts/test_budget_enforcement.sh` (~$0.01, 1min)
4. ✅ Run `bash scripts/test_training_execution.sh` (free, 5min)

**Total**: < $1, ~15 minutes, catches 90%+ of issues before expensive runs.

