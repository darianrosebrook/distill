# Priority 2 - Quick Fix Reference

## 1. JSON Repair Tests - Fix Mock Patching

**Issue**: `@patch("training.json_repair.jsonrepair")` fails when jsonrepair not installed

**Fix**: Change decorator to conditionally mock sys.modules

```python
from unittest.mock import patch, MagicMock
import sys

# For the failing test:
@patch.dict(sys.modules, {'jsonrepair': MagicMock()})
@patch("training.json_repair.JSONREPAIR_AVAILABLE", True)
def test_repair_json_with_jsonrepair(self, mock_available):
    # Test code...
```

**File**: `tests/training/test_json_repair.py`
**Tests**: 5 (test_repair_json_with_jsonrepair, test_check_json_repair_needed_invalid_json, etc.)

---

## 2. Progress Tracking Path Mocking

**Issue**: `TypeError: expected str, bytes or os.PathLike object, not Mock`

**Fix**: Properly configure Path mock

```python
from unittest.mock import MagicMock, patch
from pathlib import Path

# Create proper Path-like mock
mock_path = MagicMock()
mock_path.__str__ = MagicMock(return_value="/tmp/test_output")
mock_path.__fspath__ = MagicMock(return_value="/tmp/test_output")
mock_path.mkdir = MagicMock()
mock_path.parent = MagicMock()

# Use in test
with patch('training.progress_integration.Path', return_value=mock_path):
    # Test code...
```

**File**: `tests/training/test_distill_process.py`
**Tests**: 2 (test_main_success, test_main_checkpoint_saving)

---

## 3. Tokenizer Migration - Add len() Support

**Issue**: `TypeError: object of type 'Mock' has no len()`

**Fix**: Add `__len__` to mock tokenizer

```python
from unittest.mock import MagicMock

# Create mock with len support
mock_tokenizer = MagicMock()
mock_tokenizer.__len__ = MagicMock(return_value=50000)  # vocab size

# Or configure via side_effect
mock_tokenizer = MagicMock()
mock_tokenizer.__len__.return_value = 50000

# Test will now support: len(mock_tokenizer)
```

**File**: `tests/training/test_tokenizer_migration.py`
**Tests**: 4 (all TestResizeModelEmbeddings and TestVerifyTokenIDs tests)

---

## 4. Latent Curriculum - Use Improved Mock

**Issue**: Same as #3 - mock needs subscripting and len()

**Fix**: Reuse tokenizer mock from Priority 1

```python
# After Priority 1 creates the shared mock utility
from tests.conftest_mock_utils import create_mock_tokenizer_with_len

# In tests:
mock_tokenizer = create_mock_tokenizer_with_len()
mock_tokenizer['input_ids'] = MagicMock(return_value=torch.tensor(...))
```

**File**: `tests/training/test_latent_curriculum.py`, `tests/training/test_loss_mask_correctness.py`
**Tests**: 4 (curriculum applies, creates loss mask, masks latent spans, excludes from supervision)

---

## 5. Teacher Cache - Debug Initialization

**Issue**: ERROR in test collection (not a test failure)

**Steps**:
1. Check `training/teacher_cache.py` imports
2. Look for any missing dependencies
3. Check `tests/training/test_teacher_cache.py` fixtures
4. Verify setup methods work

**File**: `training/teacher_cache.py`, `tests/training/test_teacher_cache.py`
**Tests**: 2 (test_complete_cache_workflow, test_cache_with_version_upgrade)

---

## 6. Teacher Stub Toy - Fix Mock Setup

**Issue**: Test mock configuration issue

**Steps**:
1. Review `tests/training/test_teacher_stub_toy.py::TestTeacherLogits`
2. Check mock setup for teacher outputs
3. Ensure tensors are properly configured

**File**: `tests/training/test_teacher_stub_toy.py`
**Tests**: 1 (test_teacher_logits_hot_tokens)

---

## Order of Implementation

1. **JSON Repair** (15-30 min) - Straightforward patch fix
2. **Path Mocking** (15-20 min) - Simple PathLike mock
3. **Tokenizer Migration** (15-20 min) - Add len() support
4. **Latent Curriculum** (10-15 min) - Reuse from #3
5. **Teacher Cache** (15-30 min) - Debug initialization
6. **Teacher Stub Toy** (10-15 min) - Fix mock config

**Total**: 2-3 hours

---

## Testing After Each Fix

```bash
# JSON Repair
pytest tests/training/test_json_repair.py -v

# Path Mocking
pytest tests/training/test_distill_process.py::TestMainFunction -v

# Tokenizer Migration
pytest tests/training/test_tokenizer_migration.py -v

# Latent Curriculum
pytest tests/training/test_latent_curriculum.py tests/training/test_loss_mask_correctness.py -v

# Teacher Cache
pytest tests/training/test_teacher_cache.py::TestTeacherCacheIntegration -v

# Teacher Stub
pytest tests/training/test_teacher_stub_toy.py::TestTeacherLogits -v

# All Priority 2
pytest tests/training/test_json_repair.py tests/training/test_distill_process.py tests/training/test_tokenizer_migration.py tests/training/test_latent_curriculum.py tests/training/test_loss_mask_correctness.py tests/training/test_teacher_cache.py tests/training/test_teacher_stub_toy.py -q
```

---

## Success Criteria

- [ ] 5/5 JSON repair tests pass
- [ ] 2/2 path mock tests pass
- [ ] 4/4 tokenizer migration tests pass
- [ ] 4/4 latent curriculum tests pass
- [ ] 2/2 teacher cache tests pass
- [ ] 1/1 teacher stub test passes

**Total**: 18/18 Priority 2 tests passing ✓

---

**Ready to proceed after Priority 1 is complete!**

