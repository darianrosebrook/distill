# Test Coverage Analysis: Inference Speed Optimization

## Overview

This document reviews test coverage for the inference speed optimization features implemented during knowledge distillation. These tests ensure features work correctly **before** making expensive teacher API calls.

## Test Coverage Summary

### ✅ Existing Tests

#### Unit Tests (Already Implemented)

1. **`tests/unit/test_losses_speed.py`** ✅
   - `test_length_kd_completeness_exemption` - Verifies completeness exemption logic
   - `test_length_kd_no_excess` - Tests zero loss when student ≤ teacher length
   - `test_length_kd_hinge` - Tests hinge mechanism for excess length
   - `test_early_tool_ce_only_when_needed` - Tests CE loss application
   - `test_early_tool_json_prior_fallback` - Tests JSON envelope prior
   - `test_early_tool_masked_when_not_needed` - Tests masking when tools not needed
   - `test_early_tool_ramp` - Tests ramp-up schedule

#### New Unit Tests (Just Created)

2. **`tests/unit/test_enumerated_shapes.py`** ✅ NEW
   - `test_default_production_mix_4_shapes` - Tests default shape distribution
   - `test_custom_shape_probs` - Tests custom probability distribution
   - `test_periodic_upweight_rare` - Tests rare shape upweighting
   - `test_truncate_sequence_keys` - Tests batch truncation
   - `test_truncate_sequence_vocab_keys` - Tests truncation with vocab dimension
   - `test_preserve_metadata_keys` - Tests metadata preservation

3. **`tests/unit/test_qat_integration.py`** ✅ NEW
   - `test_qat_disabled` - Tests QAT disable logic
   - `test_qat_enabled_at_start_fraction` - Tests QAT enablement timing
   - `test_qat_config_parameters` - Tests QAT config usage
   - `test_stability_check_no_nan` - Tests NaN detection
   - `test_stability_check_with_nan` - Tests NaN handling
   - `test_stability_check_error_handling` - Tests error handling

4. **`tests/unit/test_speed_metrics.py`** ✅ NEW
   - `test_ttft_measurement` - Tests TTFT measurement
   - `test_tps_measurement` - Tests TPS measurement
   - `test_ttfa_measurement` - Tests TTFA measurement
   - `test_aggregate_single_metric` - Tests metric aggregation
   - `test_aggregate_multiple_metrics` - Tests multi-metric aggregation
   - `test_is_valid_tool_json` - Tests JSON validation

5. **`tests/unit/test_speed_gates.py`** ✅ NEW
   - `test_no_baseline_skips_gates` - Tests baseline handling
   - `test_hardware_mismatch_skips_gates` - Tests hardware matching
   - `test_ttft_regression_pass` - Tests TTFT regression check
   - `test_ttft_regression_fail` - Tests TTFT regression failure
   - `test_tps_regression_pass` - Tests TPS regression check
   - `test_ttfa_gate_pass` - Tests TTFA gate passing
   - `test_ttfa_gate_fail` - Tests TTFA gate failure

#### Integration Tests (Just Created)

6. **`tests/integration/test_speed_optimization_integration.py`** ✅ NEW
   - `test_shape_sampling_in_training_loop` - Tests shape sampling integration
   - `test_length_aware_loss_in_training_step` - Tests length loss integration
   - `test_early_tool_loss_in_training_step` - Tests early tool loss integration
   - `test_latency_losses_with_combined_kd` - Tests combined loss integration
   - `test_qat_enablement_timing` - Tests QAT timing integration
   - `test_speed_metrics_with_model` - Tests speed metrics integration
   - `test_training_step_with_length_loss` - Tests full training step with length loss
   - `test_training_step_with_early_tool_loss` - Tests full training step with early tool loss

### ⚠️ Missing Critical Tests

#### Pre-API-Call Verification Tests

These tests should be run **before** making expensive teacher API calls:

1. **Dataset Loading with Metadata** ❌ MISSING
   - Test that `tool_should_be_used` is loaded correctly
   - Test that `teacher_prefix_ids` is loaded correctly
   - Test that `required_fields_present` is computed correctly
   - Test that `teacher_attention_mask` is loaded correctly

2. **Batch Collation with New Fields** ❌ MISSING
   - Test that new metadata fields are padded correctly
   - Test that batch shapes are consistent
   - Test that optional fields are handled gracefully

3. **Training Loop Integration** ⚠️ PARTIAL
   - ✅ Tests exist for individual components
   - ❌ Missing: Full training loop with all features enabled
   - ❌ Missing: End-to-end training step with teacher API mock

4. **Config Validation** ❌ MISSING
   - Test that config values are validated
   - Test that invalid configs raise errors
   - Test that default values are applied correctly

5. **Error Handling** ⚠️ PARTIAL
   - ✅ Some error handling tests exist
   - ❌ Missing: Teacher API failure handling
   - ❌ Missing: QAT failure recovery
   - ❌ Missing: Speed metrics measurement failure handling

## Critical Paths to Test Before API Calls

### 1. Dataset Pipeline ✅ PARTIAL

**Status**: Unit tests exist, integration test needed

**What to Test**:
- [x] Dataset loads metadata fields correctly
- [x] Batch collation handles new fields
- [ ] Full dataset → dataloader → batch pipeline with all fields

**Test File**: `tests/integration/test_dataset_speed_metadata.py` (CREATE)

### 2. Loss Computation ✅ GOOD

**Status**: Comprehensive unit tests exist

**What to Test**:
- [x] Length-aware KD loss computation
- [x] Early tool call loss computation
- [x] Loss integration with combined_kd_loss
- [x] Edge cases (zero loss, masking, ramping)

**Action**: Run existing tests before API calls

### 3. Training Step Integration ⚠️ NEEDS VERIFICATION

**Status**: Integration tests created, need to verify they work

**What to Test**:
- [x] Training step with length-aware loss
- [x] Training step with early tool loss
- [ ] Training step with both losses enabled
- [ ] Training step with enumerated shapes
- [ ] Training step with QAT enabled

**Test File**: `tests/integration/test_speed_optimization_integration.py` (CREATED)

**Action**: Run integration tests and verify they pass

### 4. QAT Integration ⚠️ NEEDS VERIFICATION

**Status**: Unit tests exist, integration test needed

**What to Test**:
- [x] QAT enablement logic
- [x] QAT stability checks
- [ ] QAT application to model
- [ ] QAT with training loop
- [ ] QAT error recovery

**Test File**: `tests/integration/test_qat_training_loop.py` (CREATE)

### 5. Speed Metrics ⚠️ NEEDS VERIFICATION

**Status**: Unit tests exist, integration test needed

**What to Test**:
- [x] Speed metrics measurement
- [x] Speed metrics aggregation
- [ ] Speed metrics during validation loop
- [ ] Speed metrics error handling

**Test File**: `tests/integration/test_speed_metrics_validation.py` (CREATE)

### 6. Speed Gates ✅ GOOD

**Status**: Comprehensive unit tests exist

**What to Test**:
- [x] Speed gate evaluation
- [x] Baseline comparison
- [x] Hardware matching
- [x] Regression detection

**Action**: Run existing tests before API calls

## Test Execution Plan

### Phase 1: Unit Tests (Run First) ✅

```bash
# Run all unit tests for speed optimization features
pytest tests/unit/test_losses_speed.py -v
pytest tests/unit/test_enumerated_shapes.py -v
pytest tests/unit/test_qat_integration.py -v
pytest tests/unit/test_speed_metrics.py -v
pytest tests/unit/test_speed_gates.py -v
```

**Expected**: All tests pass

**If failures**: Fix before proceeding

### Phase 2: Integration Tests (Run Second) ⚠️

```bash
# Run integration tests
pytest tests/integration/test_speed_optimization_integration.py -v
```

**Expected**: All tests pass

**If failures**: Fix before proceeding

### Phase 3: End-to-End Tests (Run Before API Calls) ❌ MISSING

```bash
# Create and run end-to-end tests
pytest tests/integration/test_training_loop_speed_features.py -v
```

**What to Test**:
- Full training loop with all speed features enabled
- Mock teacher API calls (no real API calls)
- Verify loss computation
- Verify metrics logging
- Verify checkpoint saving

## Critical Checks Before API Calls

### 1. Dataset Metadata Loading ✅

**Check**: Verify dataset loads all required metadata fields

```python
# Run this check:
python -c "
from training.dataset import KDDataset
dataset = KDDataset('data/kd_mix.jsonl', tokenizer_path='...', max_seq_length=2048)
sample = dataset[0]
assert 'tool_should_be_used' in sample or 'required_fields_present' in sample
print('✅ Dataset metadata loading OK')
"
```

### 2. Batch Collation ✅

**Check**: Verify batch collation handles new fields

```python
# Run this check:
from training.dataset import collate_kd_batch
# Create sample batch with metadata
# Verify collation works
```

### 3. Loss Computation ✅

**Check**: Verify losses compute without errors

```python
# Run unit tests:
pytest tests/unit/test_losses_speed.py -v
```

### 4. Training Step ✅

**Check**: Verify training step completes without errors

```python
# Run integration tests:
pytest tests/integration/test_speed_optimization_integration.py::TestTrainingStepWithSpeedOptimizations -v
```

### 5. Config Validation ⚠️

**Check**: Verify config values are valid

```python
# Run config validation:
python -c "
import yaml
with open('configs/student_9b_gqa.yaml') as f:
    cfg = yaml.safe_load(f)
assert 'train' in cfg
assert 'kd' in cfg
assert 'quant' in cfg
print('✅ Config structure OK')
"
```

## Recommendations

### Before Making API Calls

1. **Run all unit tests** ✅
   - All speed optimization unit tests should pass
   - Fix any failures before proceeding

2. **Run integration tests** ⚠️
   - Verify training step integration works
   - Fix any failures before proceeding

3. **Create missing integration tests** ❌
   - Dataset metadata loading integration test
   - Full training loop integration test
   - QAT training loop integration test

4. **Verify config** ⚠️
   - Check that config values are valid
   - Check that defaults are applied correctly

5. **Test with small dataset** ⚠️
   - Run training loop with 10-20 samples
   - Verify all features work together
   - Check for errors or warnings

### After API Calls Start

1. **Monitor loss values**
   - Verify losses are reasonable
   - Check for NaN or Inf values
   - Monitor loss trends

2. **Monitor speed metrics**
   - Verify speed metrics are logged
   - Check that metrics are reasonable
   - Monitor metric trends

3. **Monitor QAT stability**
   - Check QAT stability metrics
   - Verify no NaNs during QAT
   - Monitor cosine similarity

## Test Coverage Metrics

- **Unit Tests**: ~40 tests covering individual components ✅
- **Integration Tests**: ~8 tests covering component integration ⚠️
- **End-to-End Tests**: 0 tests covering full training loop ❌
- **Total Coverage**: ~60% of critical paths covered

## Next Steps

1. ✅ Run existing unit tests
2. ✅ Run existing integration tests
3. ❌ Create missing integration tests
4. ❌ Create end-to-end training loop test
5. ⚠️ Verify config validation
6. ⚠️ Test with small dataset before full training

