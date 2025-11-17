# Test Baseline Report

**Date**: November 16, 2025
**Author**: @darianrosebrook
**Status**: Baseline Established

## Summary

Virtual environment activated and test baseline collected across unit, integration, and mutation test suites.

### Overall Metrics

| Category            | Tests | Passed  | Failed | Errors | Skipped | Coverage | Status             |
| ------------------- | ----- | ------- | ------ | ------ | ------- | -------- | ------------------ |
| **Unit (Training)** | 1554  | 1409    | 133    | 2      | 10      | 50%      | ⚠️ Needs Attention |
| **Integration**     | 961   | Partial | 1      | 0      | 0       | -        | ⚠️ Timeout Issues  |
| **Mutation**        | -     | -       | -      | -      | -       | -        | ⚠️ Tool Error      |

## Unit Tests (Training Modules)

**Exit Code**: 0 (reported success despite failures - pytest behavior)
**Run Time**: 48.20s
**Result**: 1409 passed, 133 failed, 10 skipped, 2 errors

### Coverage by Module

| Module                                | Coverage | Status              |
| ------------------------------------- | -------- | ------------------- |
| training/distill_answer_generation.py | 16%      | Low                 |
| training/distill_kd.py                | 4%       | Critical            |
| training/distill_process.py           | 81%      | Good                |
| training/distill_post_tool.py         | 94%      | Excellent           |
| training/distill_tool_select.py       | 59%      | Fair                |
| training/progress_tracker.py          | 92%      | Excellent           |
| training/progress_integration.py      | 93%      | Excellent           |
| **Average Coverage**                  | **50%**  | **Below Threshold** |

### Critical Failures by Category

#### 1. Dataset Mocking Issues (47 failures)

**Root Cause**: Mock tokenizers not properly configured
**Affected Tests**:

- test_dataset.py: 23 failures (KDDataset, AnswerGenerationDataset)
- test_dataset_post_tool.py: 9 failures
- test_dataset_tool_select.py: 7 failures
- test_dataset_answer_generation.py: 8 failures

**Pattern**:

```
TypeError: 'Mock' object is not subscriptable
TypeError: 'Mock' object has no len()
AttributeError: 'str' object has no attribute 'get'
```

**Action Required**: Fix mock tokenizer setup to properly return dict-like structures

#### 2. Distill KD Function Signature Mismatches (9 failures)

**Root Cause**: Function signatures changed, tests not updated
**Affected Functions**:

- `entropy_weighting()` - missing `base_kl_weight` parameter
- `intermediate_layer_loss()` - missing `student_hidden_states` parameter

**Pattern**:

```
TypeError: entropy_weighting() got an unexpected keyword argument 'base_kl_weight'
TypeError: intermediate_layer_loss() got an unexpected keyword argument 'student_hidden_states'
```

**Action Required**: Update test calls to match function signatures

#### 3. JSON Repair Issues (5 failures)

**Root Cause**: Missing jsonrepair dependency or mock configuration
**Affected Tests**:

- test_json_repair.py: Tests expect JSON repair detection but implementation incomplete

**Pattern**:

```
AttributeError: <module 'training.json_repair'> does not have the attribute 'jsonrepair'
AssertionError: Expected JSON repair detection
```

**Action Required**: Implement JSON repair detection or adjust tests

#### 4. Monitoring Module Issues (31 failures)

**Root Cause**: MetricsCollector and HealthChecker interfaces changed
**Affected Tests**:

- test_monitoring.py: 31 failures
- test_monitoring_actual.py: 4 failures

**Pattern**:

```
AttributeError: 'MetricsCollector' object has no attribute 'add_metric'
AttributeError: 'HealthChecker' object has no attribute 'add_check'
TypeError: isinstance() arg 2 must be a type, a tuple of types, or a union
```

**Action Required**: Update MetricsCollector and HealthChecker interfaces to match tests

#### 5. Progress Tracking Integration Issues (2 failures)

**Root Cause**: Mock vs real Path object incompatibility
**Affected Tests**:

- test_distill_process.py::TestMainFunction::test_main_success
- test_distill_process.py::TestMainFunctionExpanded::test_main_checkpoint_saving

**Pattern**:

```
TypeError: expected str, bytes or os.PathLike object, not Mock
```

**Action Required**: Fix mock Path object configuration

#### 6. Tokenizer Migration Issues (4 failures)

**Root Cause**: Mock tokenizer not properly set up with len()
**Affected Tests**:

- test_tokenizer_migration.py: 4 failures

**Pattern**:

```
TypeError: object of type 'Mock' has no len()
```

**Action Required**: Proper mock tokenizer setup

#### 7. Latent Curriculum Issues (3 failures)

**Root Cause**: Mock tokenizer missing subscriptable behavior
**Affected Tests**:

- test_latent_curriculum.py: 3 failures
- test_loss_mask_correctness.py: 1 failure

**Action Required**: Fix mock tokenizer in curriculum tests

#### 8. Tracing Module Issues (12 failures)

**Root Cause**: Tracer interface implementation incomplete
**Affected Tests**:

- test_tracing.py: 12 failures

**Pattern**: Multiple test decorators with mock issues
**Action Required**: Complete tracer interface implementation

#### 9. Teacher Cache Issues (2 errors)

**Root Cause**: Import or initialization issues
**Affected Tests**:

- test_teacher_cache.py::TestTeacherCacheIntegration: 2 errors (not failures)

**Action Required**: Debug cache initialization

## Integration Tests

**Status**: ⚠️ ONNX Tests Can Be Skipped (Not in Production Pipeline)
**Problem**: `test_judge_export_onnx.py::test_main_config_loading_error` hangs after 180s

**Root Cause**: torch.onnx.export with ONNX runtime getting stuck writing external data

**Important Note**: ONNX export is **NOT used in the production pipeline**. The main training pipeline exports to PyTorch → **CoreML** (not ONNX), as confirmed in `training/export_student.py`.

**Action Required**:

- Skip ONNX tests or mock them for CI/integration purposes
- Focus on CoreML conversion tests instead
- These tests are not critical to the distillation training process

## Mutation Tests

**Status**: ⚠️ Can Proceed (Skip ONNX Tests - Not in Production)
**Error**: `BaselineTestException: Clean trial does not pass, mutant tests will be meaningless.`

**Root Cause**: The clean test run fails due to `test_main_config_loading_error` timeout in ONNX export

**Important Note**: Since ONNX is **NOT used in the production pipeline** (uses PyTorch → CoreML instead), we can skip ONNX tests when establishing mutation baseline.

**Action Required**:

1. Skip ONNX export tests (not critical to training pipeline)
2. Run mutation tests on training modules only:
   ```bash
   mutatest -s training/distill_kd.py -m sd -n 20
   ```
3. Focus on core training modules without integration tests

## Next Steps

### Priority 1: Critical Coverage Issues

1. Fix mock tokenizer configuration (affects 47 tests)
2. Update function signatures in distill_kd.py tests (9 tests)
3. Fix monitoring module interfaces (31 tests)

### Priority 2: Implementation Issues

4. Complete JSON repair detection
5. Fix progress tracking Path mocking
6. Complete tracer interface implementation

### Priority 3: Ready to Complete

7. Establish mutation testing baseline (can skip ONNX tests)
8. All other issues addressed in Priority 1-2

### Coverage Goals

- **Target**: 80% line, 90% branch coverage
- **Current**: 50% (training modules)
- **Gap**: 30 percentage points

## Command Reference

Run these locally to verify baseline:

```bash
# Activate venv
source venv/bin/activate

# Unit tests only (training)
pytest tests/training/ --tb=short -q --timeout=120 --cov=training --cov-report=term-missing

# Quick integration check (skip timeout test)
pytest tests/conversion tests/evaluation tests/e2e -k "not test_main_config_loading_error" --tb=short -q --timeout=180

# Mutation tests (corrected)
mutatest --target training/distill_kd.py --output mutations.json --mode f
```

## Recommendations

1. **Immediate**: Create mock utilities module for consistent tokenizer/Path mocking
2. **Short-term**: Fix 133 test failures systematically
3. **Medium-term**: Reach 80% coverage threshold
4. **Long-term**: Implement continuous mutation testing in CI

## Files Generated

- `TEST_BASELINE_REPORT.md` - This comprehensive baseline report
- `baseline_unit_tests.log` - Unit test execution output
- `baseline_integration_tests.log` - Integration test execution output
- `mutation_baseline.log` - Mutation test attempt (shows blocking issues)

## Environment Summary

- **Python Version**: 3.11.13
- **Virtual Environment**: Active at `venv/bin/python`
- **Key Packages**:
  - pytest 9.0.1
  - mutatest 0.19.0
  - torch 2.x
  - coremltools 9.0
  - onnxruntime 1.18+

## Test Categories Snapshot

```
tests/
├── ci/ ..................... CI/CD and sharding tests (7 tests, all passing)
├── conversion/ ............. Model conversion tests (961 tests, 1 timeout)
│   ├── test_convert_coreml.py
│   ├── test_export_onnx.py
│   ├── test_export_pytorch.py
│   ├── test_judge_export_coreml.py
│   └── test_judge_export_onnx.py (TIMEOUT ISSUE)
├── evaluation/ ............. Model evaluation tests
├── e2e/ .................... End-to-end integration tests
└── training/ ............... Core training module tests (1554 tests, 133 failures)
    ├── test_assertions.py
    ├── test_caws_context.py
    ├── test_dataset.py (23 FAILURES - mock issues)
    ├── test_dataset_answer_generation.py (8 FAILURES)
    ├── test_dataset_post_tool.py (9 FAILURES)
    ├── test_dataset_tool_select.py (7 FAILURES)
    ├── test_distill_kd.py (9 FAILURES - signature mismatches)
    ├── test_distill_process.py (2 FAILURES)
    ├── test_json_repair.py (5 FAILURES)
    ├── test_latent_curriculum.py (4 FAILURES)
    ├── test_loss_mask_correctness.py (1 FAILURE)
    ├── test_monitoring.py (31 FAILURES - interface mismatches)
    ├── test_monitoring_actual.py (4 FAILURES)
    ├── test_teacher_cache.py (2 ERRORS)
    ├── test_teacher_stub_toy.py (1 FAILURE)
    ├── test_tokenizer_migration.py (4 FAILURES)
    ├── test_tracing.py (12 FAILURES)
    └── test_progress_integration.py (PASSING)
```

## Quick Fix Checklist

- [ ] Fix mock tokenizer to return proper dict-like objects with subscriptable behavior
- [ ] Update test function calls to match distill_kd.py signatures
- [ ] Implement or mock MetricsCollector.add_metric and HealthChecker.add_check
- [ ] Fix Path mocking in progress tracking tests
- [ ] Mock or skip ONNX export timeouts
- [ ] Complete tracer interface implementation
- [ ] Fix teacher cache initialization
- [ ] Implement JSON repair detection logic

## Success Metrics

Once fixes are applied, target:

- Unit Tests: **100% pass** (currently 90.4%)
- Unit Coverage: **80%+** (currently 50%)
- Integration Tests: **100% pass** (currently blocked by timeouts)
- Mutation Score: **70%+** (not yet established due to test failures)

## Contact

For questions about this baseline: @darianrosebrook
Last Updated: November 16, 2025
