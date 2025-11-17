# Test Baseline Fix Checklist

**Date Created**: November 16, 2025
**Author**: @darianrosebrook
**Status**: In Development

## Executive Summary

This checklist tracks the resolution of 133 failing unit tests and integration test issues identified in the baseline assessment. The goal is to achieve 100% pass rate with 80%+ code coverage.

---

## Priority 1: Focus on Core Training Issues (Not ONNX)

### Note on ONNX Export
- ONNX is **NOT used in the production pipeline** (uses PyTorch → CoreML instead)
- ONNX tests can be skipped for mutation testing baseline
- Focus efforts on core training module issues instead
- **Files**: `tests/conversion/test_judge_export_onnx.py` (can skip)

---

## Priority 1: High-Impact Fixes (8-10 hours)

### 1. Dataset Mocking Issues (47 failures)
- [ ] Create `tests/conftest_mock_utils.py` with reusable mock utilities
  - [ ] `create_mock_tokenizer()` - returns dict-like with `__getitem__`
  - [ ] `create_mock_tokenizer_with_len()` - includes `__len__`
  - [ ] `create_mock_tokenizer_subscriptable()` - full support
- [ ] Update `test_dataset.py` to use new mock utilities (23 failures)
  - [ ] Fix `setup_mock_tokenizer()` parameter passing
  - [ ] Ensure mock returns proper dict structures
  - [ ] Test: `TestKDDataset::test_kd_dataset_init_basic`
- [ ] Update `test_dataset_answer_generation.py` (8 failures)
  - [ ] Fix prompt_tokens mock to be subscriptable
  - [ ] Test: `TestAnswerGenerationDataset::test_answer_generation_dataset_getitem_basic`
- [ ] Update `test_dataset_post_tool.py` (9 failures)
  - [ ] Fix tokenizer mock subscriptable behavior
  - [ ] Fix max_target_length parameter handling
  - [ ] Test: `TestPostToolDataset::test_post_tool_dataset_init_basic`
- [ ] Update `test_dataset_tool_select.py` (7 failures)
  - [ ] Fix target_encoding mock subscriptability
  - [ ] Test: `TestToolSelectDataset::test_tool_select_dataset_getitem_basic`

**Subtask Status**: ⚠️ Not Started
**Expected Time**: 4-5 hours
**Tests Fixed**: 47

### 2. Monitoring Interface Issues (31 failures)
- [ ] Review `training/monitoring.py` class definitions
  - [ ] Verify `MetricsCollector` interface
  - [ ] Verify `HealthChecker` interface
  - [ ] Verify `SystemHealthChecks` interface
- [ ] Add missing methods to `MetricsCollector`
  - [ ] Implement `add_metric()`
  - [ ] Implement `get_metrics_by_name()`
  - [ ] Implement `get_metrics_by_tags()`
  - [ ] Implement `get_latest_metric()`
  - [ ] Implement `get_metric_statistics()`
  - [ ] Implement `clear_metrics()`
- [ ] Add missing methods to `HealthChecker`
  - [ ] Implement `add_check()`
  - [ ] Implement `run_check()`
  - [ ] Implement `run_all_checks()`
  - [ ] Implement `get_component_status()`
  - [ ] Implement `get_overall_health()`
- [ ] Update test files
  - [ ] `test_monitoring.py` (31 failures)
  - [ ] `test_monitoring_actual.py` (4 failures)
- [ ] Verify thread safety implementation
  - [ ] Test: `TestConcurrencyAndPerformance::test_concurrent_metric_logging`

**Subtask Status**: ⚠️ Not Started
**Expected Time**: 3-4 hours
**Tests Fixed**: 35

### 3. Distill KD Function Signature Mismatches (9 failures)
- [ ] Review `training/distill_kd.py` function signatures
  - [ ] Check `entropy_weighting()` parameters
  - [ ] Check `intermediate_layer_loss()` parameters
  - [ ] Document all parameter names and types
- [ ] Update test calls in `test_distill_kd.py`
  - [ ] Fix `test_train_step_with_entropy_scheduling`
  - [ ] Fix `test_train_step_with_intermediate_layers_and_teacher_states`
  - [ ] Fix `test_train_step_with_intermediate_layers_default_mapping`
  - [ ] Fix `test_train_step_with_intermediate_layers_projection_creation`
  - [ ] Fix `test_main_tokenizer_path_missing`
  - [ ] Fix `test_main_config_validation_failure`
  - [ ] Fix `test_main_config_provenance_logging`
- [ ] Update test calls in `test_distill_process.py`
  - [ ] Fix `test_main_success`
  - [ ] Fix `test_main_checkpoint_saving`

**Subtask Status**: ⚠️ Not Started
**Expected Time**: 2-3 hours
**Tests Fixed**: 9

### 4. Tracing Module Implementation (12 failures)
- [ ] Review `training/tracing.py` interface
  - [ ] Understand `TrainingTracer` class structure
  - [ ] Check all required methods
  - [ ] Identify incomplete implementations
- [ ] Complete tracer implementation
  - [ ] Implement missing methods
  - [ ] Fix decorator/mock issues
- [ ] Update `test_tracing.py` if needed
  - [ ] Test: `TestTrainingTracerInit::test_tracer_init_basic`
  - [ ] Test: `TestTrainingTracerLogMetrics::test_log_metrics_tensorboard`
  - [ ] (10 more tests)

**Subtask Status**: ⚠️ Not Started
**Expected Time**: 3-4 hours
**Tests Fixed**: 12

---

## Priority 2: Remaining Issues (2-3 hours)

### 5. JSON Repair Detection (5 failures)
- [ ] Review `training/json_repair.py` implementation
  - [ ] Check JSON repair detection logic
  - [ ] Verify jsonrepair integration
- [ ] Fix test cases
  - [ ] `test_repair_json_with_jsonrepair`
  - [ ] `test_check_json_repair_needed_invalid_json`
  - [ ] `test_check_json_repair_needed_repairable_json`
  - [ ] `test_batch_check_json_repair_mixed`
  - [ ] `test_repair_workflow`

**Subtask Status**: ⚠️ Not Started
**Expected Time**: 1-2 hours
**Tests Fixed**: 5

### 6. Progress Tracking Path Mock Issue (2 failures)
- [ ] Review `training/progress_integration.py` and `distill_process.py`
  - [ ] Understand how output_dir is used
  - [ ] Find why Mock object isn't converted to Path
- [ ] Fix mock Path in test fixtures
  - [ ] Ensure mock returns proper string/PathLike
  - [ ] Test: `test_main_success`
  - [ ] Test: `test_main_checkpoint_saving`

**Subtask Status**: ⚠️ Not Started
**Expected Time**: 1 hour
**Tests Fixed**: 2

### 7. Tokenizer Migration Mock Issues (4 failures)
- [ ] Review `training/tokenizer_migration.py`
- [ ] Fix mock tokenizer with proper len() support
  - [ ] Test: `TestVerifyTokenIDs::test_verify_token_ids_matching`
  - [ ] Test: `TestResizeModelEmbeddings::test_resize_model_embeddings_basic`
  - [ ] Test: `TestResizeModelEmbeddings::test_resize_model_embeddings_with_new_vocab_size`
  - [ ] Test: `TestResizeModelEmbeddings::test_resize_model_embeddings_tokenizer_len`

**Subtask Status**: ⚠️ Not Started
**Expected Time**: 1 hour
**Tests Fixed**: 4

### 8. Latent Curriculum Mock Issues (4 failures)
- [ ] Use improved mock tokenizer from Priority 2.1
  - [ ] Test: `test_curriculum_applies_latent_slots`
  - [ ] Test: `test_curriculum_creates_loss_mask`
  - [ ] Test: `test_loss_mask_masks_latent_spans`
  - [ ] Test: `test_loss_mask_excludes_latent_spans_from_supervision`

**Subtask Status**: ⚠️ Not Started
**Expected Time**: 1 hour
**Tests Fixed**: 4

### 9. Teacher Cache Initialization Errors (2 errors)
- [ ] Debug import/initialization issues
  - [ ] Check `training/teacher_cache.py`
  - [ ] Verify test setup
  - [ ] Test: `TestTeacherCacheIntegration::test_complete_cache_workflow`
  - [ ] Test: `TestTeacherCacheIntegration::test_cache_with_version_upgrade`

**Subtask Status**: ⚠️ Not Started
**Expected Time**: 1 hour
**Tests Fixed**: 2

### 10. Teacher Stub Toy Issue (1 failure)
- [ ] Fix mock issue in `test_teacher_stub_toy.py`
  - [ ] Test: `TestTeacherLogits::test_teacher_logits_hot_tokens`

**Subtask Status**: ⚠️ Not Started
**Expected Time**: 30 minutes
**Tests Fixed**: 1

---

## Verification Milestones

### Milestone 1: Fix Blocking Issue
- [ ] ONNX export timeout resolved or skipped appropriately
- [ ] Mutation testing baseline can be established
- **Completion**: ✅ Must complete before other fixes

### Milestone 2: Fix Priority 2 Issues (72 tests)
- [ ] Mock utilities module created and tested
- [ ] 47 dataset tests passing
- [ ] 31 monitoring tests passing
- [ ] 9 function signature tests passing
- [ ] 12 tracing tests passing
- **Coverage**: Should reach ~70%
- **Pass Rate**: Should reach ~96% (1442/1554)

### Milestone 3: Fix Priority 3 Issues (16 tests)
- [ ] All remaining issues resolved
- [ ] **Coverage**: Should reach 80%+
- [ ] **Pass Rate**: Should reach 100% (1554/1554)

### Milestone 4: Establish Mutation Baseline
- [ ] Run: `mutatest -s training/distill_kd.py -m sd -n 20`
- [ ] Document mutation score for distill_kd.py
- [ ] Establish baseline for future comparison

### Final Verification
- [ ] `pytest tests/training/ -q --timeout=120` = 100% pass
- [ ] Coverage report shows 80%+ coverage
- [ ] Mutation testing baseline established
- [ ] All integration tests passing (skip timeout test if necessary)

---

## Files to Modify

### New Files to Create
- [ ] `tests/conftest_mock_utils.py` - Mock utilities module
- [ ] Any helper modules needed by fixes

### Files to Edit
- [ ] `training/monitoring.py` - Add missing interface methods
- [ ] `training/distill_kd.py` - Verify/fix function signatures
- [ ] `training/tracing.py` - Complete implementation
- [ ] `training/progress_integration.py` - Review Path handling
- [ ] Multiple test files - Apply fixes

### Test Files to Fix
1. `tests/training/test_dataset.py` (23 fixes)
2. `tests/training/test_dataset_answer_generation.py` (8 fixes)
3. `tests/training/test_dataset_post_tool.py` (9 fixes)
4. `tests/training/test_dataset_tool_select.py` (7 fixes)
5. `tests/training/test_monitoring.py` (31 fixes)
6. `tests/training/test_monitoring_actual.py` (4 fixes)
7. `tests/training/test_distill_kd.py` (9 fixes)
8. `tests/training/test_distill_process.py` (2 fixes)
9. `tests/training/test_json_repair.py` (5 fixes)
10. `tests/training/test_latent_curriculum.py` (3 fixes)
11. `tests/training/test_loss_mask_correctness.py` (1 fix)
12. `tests/training/test_tokenizer_migration.py` (4 fixes)
13. `tests/training/test_tracing.py` (12 fixes)
14. `tests/training/test_teacher_cache.py` (2 fixes)
15. `tests/training/test_teacher_stub_toy.py` (1 fix)

---

## Success Criteria

### Unit Tests
- [ ] 100% pass rate (1554/1554 tests)
- [ ] 0 failures
- [ ] 0 errors
- [ ] 0 skipped (except intentional markers)

### Coverage
- [ ] 80%+ line coverage
- [ ] 90%+ branch coverage
- [ ] All core training modules >80%

### Integration Tests
- [ ] All conversion tests passing
- [ ] All evaluation tests passing
- [ ] All e2e tests passing

### Mutation Testing
- [ ] Clean trial passes (all tests pass)
- [ ] Mutation score 70%+ for distill_kd.py
- [ ] Baseline established for future regressions

---

## Notes

- Virtual environment already activated: `source venv/bin/activate`
- Baseline report available: `TEST_BASELINE_REPORT.md`
- Summary available: `BASELINE_SUMMARY.txt`
- Test logs available: `baseline_unit_tests.log`, `baseline_integration_tests.log`

---

## Progress Tracking

| Milestone | Status | Tests Fixed | Coverage | Date |
|-----------|--------|------------|----------|------|
| Baseline Established | ✅ | - | 50% | Nov 16 |
| Blocking Issues | ⚠️ | 0/0 | - | - |
| Priority 2 Fixes | ⚠️ | 0/72 | - | - |
| Priority 3 Fixes | ⚠️ | 0/16 | - | - |
| Mutation Baseline | ⚠️ | - | - | - |
| COMPLETE | ⚠️ | 0/133 | 0% | - |

---

**Last Updated**: November 16, 2025, 21:05 UTC
**Next Review**: Upon starting fixes

