# Priority 2 Implementation Plan

**Status**: In Progress
**Author**: @darianrosebrook
**Focus**: Fixing 18 remaining test failures

## Overview

Priority 2 addresses the remaining test failures after Priority 1 is completed:
- JSON Repair Detection (5 tests)
- Progress Tracking Path Mocking (2 tests)
- Tokenizer Migration Mocking (4 tests)
- Latent Curriculum Mocking (4 tests)
- Teacher Cache Initialization (2 errors)
- Teacher Stub Toy (1 test)

**Total**: 18 tests/errors to fix
**Estimated Time**: 2-3 hours

---

## Issue Breakdown & Solutions

### 1. JSON Repair Detection (5 failures)

**Files Affected**:
- `training/json_repair.py`
- `tests/training/test_json_repair.py`

**Root Cause**:
The test tries to patch `training.json_repair.jsonrepair` but the module only has this attribute when jsonrepair is imported. When not installed, patching fails.

**Pattern**:
```python
@patch("training.json_repair.jsonrepair")  # ← Fails if jsonrepair not in module
def test_repair_json_with_jsonrepair(self, mock_jsonrepair):
```

**Solution**:
Use conditional patching or mock the import more carefully:

```python
@patch.dict('sys.modules', {'jsonrepair': MagicMock()})
@patch("training.json_repair.JSONREPAIR_AVAILABLE", True)
def test_repair_json_with_jsonrepair(self, ...):
```

**Failing Tests**:
1. `test_repair_json_with_jsonrepair`
2. `test_check_json_repair_needed_invalid_json`
3. `test_check_json_repair_needed_repairable_json`
4. `test_batch_check_json_repair_mixed`
5. `test_repair_workflow`

---

### 2. Progress Tracking Path Mocking (2 failures)

**Files Affected**:
- `training/progress_integration.py`
- `tests/training/test_distill_process.py`

**Root Cause**:
Mock object is passed where `Path` is expected. `Path()` constructor tries to call `os.fspath()` on the mock.

**Pattern**:
```python
TypeError: expected str, bytes or os.PathLike object, not Mock
# Happens in: Path(output_dir)
```

**Solution**:
Configure mock to work with `Path`:

```python
from unittest.mock import MagicMock, patch
from pathlib import Path

mock_output_dir = MagicMock(spec=Path)
mock_output_dir.__str__ = MagicMock(return_value="/tmp/test")
mock_output_dir.__fspath__ = MagicMock(return_value="/tmp/test")
```

**Failing Tests**:
1. `test_main_success` (test_distill_process.py)
2. `test_main_checkpoint_saving` (test_distill_process.py)

---

### 3. Tokenizer Migration Mocking (4 failures)

**Files Affected**:
- `training/tokenizer_migration.py`
- `tests/training/test_tokenizer_migration.py`

**Root Cause**:
Mock tokenizer doesn't have `__len__()` method implemented.

**Pattern**:
```python
TypeError: object of type 'Mock' has no len()
# Happens when code calls: len(tokenizer)
```

**Solution**:
Configure mock with `__len__`:

```python
mock_tokenizer = MagicMock()
mock_tokenizer.__len__ = MagicMock(return_value=50000)
# Or use spec:
mock_tokenizer = MagicMock(spec=['__len__', '__call__', ...])
```

**Failing Tests**:
1. `test_verify_token_ids_matching`
2. `test_resize_model_embeddings_basic`
3. `test_resize_model_embeddings_with_new_vocab_size`
4. `test_resize_model_embeddings_tokenizer_len`

---

### 4. Latent Curriculum Mocking (4 failures)

**Files Affected**:
- `data/wrappers/curriculum.py` (implementation)
- `tests/training/test_latent_curriculum.py`

**Root Cause**:
Mock tokenizer not subscriptable (`__getitem__` not configured).

**Pattern**:
```python
TypeError: object of type 'Mock' has no len()
# Or 'Mock' object is not subscriptable
```

**Solution**:
Reuse improved mock tokenizer from Priority 1 that supports subscripting and len().

**Failing Tests**:
1. `test_curriculum_applies_latent_slots`
2. `test_curriculum_creates_loss_mask`
3. `test_loss_mask_masks_latent_spans`
4. `test_loss_mask_excludes_latent_spans_from_supervision` (in test_loss_mask_correctness.py)

---

### 5. Teacher Cache Initialization (2 errors)

**Files Affected**:
- `training/teacher_cache.py`
- `tests/training/test_teacher_cache.py`

**Root Cause**:
Import or initialization issue preventing test setup.

**Pattern**:
```
ERROR tests/training/test_teacher_cache.py::TestTeacherCacheIntegration::test_complete_cache_workflow
ERROR tests/training/test_teacher_cache.py::TestTeacherCacheIntegration::test_cache_with_version_upgrade
```

**Solution**:
Debug and verify:
1. Check imports in teacher_cache.py
2. Check if any dependencies are missing
3. Verify test fixtures initialize correctly

**Failing Tests**:
1. `test_complete_cache_workflow`
2. `test_cache_with_version_upgrade`

---

### 6. Teacher Stub Toy (1 failure)

**Files Affected**:
- `training/teacher_stub_toy.py`
- `tests/training/test_teacher_stub_toy.py`

**Root Cause**:
Mock issue in test setup.

**Pattern**:
```
FAILED tests/training/test_teacher_stub_toy.py::TestTeacherLogits::test_teacher_logits_hot_tokens
```

**Solution**:
Review test setup and ensure mocks are properly configured.

**Failing Tests**:
1. `test_teacher_logits_hot_tokens`

---

## Implementation Strategy

### Phase 1: JSON Repair (15-30 min)
1. Update test fixtures to properly mock jsonrepair module
2. Use `@patch.dict` for sys.modules
3. Verify all 5 tests pass

### Phase 2: Path Mocking (15-20 min)
1. Create proper Path mock with `__fspath__`
2. Update test fixtures in test_distill_process.py
3. Verify both tests pass

### Phase 3: Tokenizer Mocks (15-20 min)
1. Add `__len__` to mock tokenizers
2. Update test fixtures in tokenizer migration tests
3. Verify 4 tests pass

### Phase 4: Latent Curriculum (10-15 min)
1. Reuse improved mock tokenizer from Phase 3
2. Update test fixtures in curriculum tests
3. Verify 4 tests pass

### Phase 5: Teacher Cache (15-30 min)
1. Debug teacher_cache.py imports
2. Check for missing dependencies
3. Fix initialization issues
4. Verify both tests pass

### Phase 6: Teacher Stub Toy (10-15 min)
1. Review test setup
2. Fix mock configuration
3. Verify test passes

---

## Testing Approach

After each phase, run:

```bash
# Run specific test file
pytest tests/training/test_json_repair.py -v
pytest tests/training/test_distill_process.py::TestMainFunction::test_main_success -v
pytest tests/training/test_tokenizer_migration.py -v
pytest tests/training/test_latent_curriculum.py -v
pytest tests/training/test_loss_mask_correctness.py::TestLossMaskCorrectness -v
pytest tests/training/test_teacher_cache.py::TestTeacherCacheIntegration -v
pytest tests/training/test_teacher_stub_toy.py::TestTeacherLogits -v

# After all phases, run full Priority 2 test suite
pytest tests/training/test_json_repair.py tests/training/test_distill_process.py tests/training/test_tokenizer_migration.py tests/training/test_latent_curriculum.py tests/training/test_loss_mask_correctness.py tests/training/test_teacher_cache.py tests/training/test_teacher_stub_toy.py -v

# Check coverage
pytest tests/training/ --cov=training --cov-report=term-missing | grep -E "(training|TOTAL)"
```

---

## Success Criteria

- [ ] All 5 JSON repair tests pass
- [ ] Both path mocking tests pass
- [ ] All 4 tokenizer migration tests pass
- [ ] All 4 latent curriculum tests pass
- [ ] Both teacher cache tests pass
- [ ] Teacher stub toy test passes
- [ ] Total: 18/18 tests passing

---

## Files to Modify

1. `tests/training/test_json_repair.py` - Fix mock patches
2. `tests/training/test_distill_process.py` - Fix Path mocks
3. `tests/training/test_tokenizer_migration.py` - Add len() to mocks
4. `tests/training/test_latent_curriculum.py` - Use improved mocks
5. `tests/training/test_loss_mask_correctness.py` - Use improved mocks
6. `training/teacher_cache.py` - Debug as needed
7. `tests/training/test_teacher_cache.py` - Debug as needed
8. `tests/training/test_teacher_stub_toy.py` - Fix mock setup

---

## Expected Outcomes

**Before Priority 2**: 1409 passed, 133 failed
**After Priority 1**: ~1442 passed, 99 failed (+ 33 from Priority 1)
**After Priority 2**: ~1460 passed, 81 failed (+ 18 from Priority 2)

**Total Progress**: 94% → 98% pass rate
**Coverage**: 50% → ~85% (expected)

---

**Status**: Ready to implement
**Start Time**: [When Priority 1 is complete]
**Target Completion**: ~2-3 hours after Priority 1 completion

