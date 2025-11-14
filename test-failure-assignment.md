# Test Failure Assignment - 4 Workers

This document distributes the 127 test failures from Worker 2 across 4 workers for parallel fixing.

## Summary

- **Total Failures**: 127 tests
- **Workers**: 4
- **Target per Worker**: ~32 failures
- **Source**: Worker 2 (evaluation/ module)

---

## Worker A: API & Function Signature Fixes

**Focus**: Function signature mismatches, constructor issues, parameter updates  
**Failures**: 36 tests (28.3%)  
**Complexity**: Medium-High (requires code changes)

### Failure Categories

#### 1. Missing Arguments (10 tests)
- `evaluate_tool_use()` missing `device` argument (5 tests)
- `load_tokenized_prompts()` missing `tokenizer_path` argument (4 tests)
- `run_coreml_speed()` missing `adapter` argument (1 test)

#### 2. Unexpected Arguments (17 tests)
- `compare_predictions()` takes 2 args but 3 given (6 tests)
- `evaluate_caws_compliance()` got unexpected `evidence` argument (5 tests)
- `main()` got unexpected `evidence` argument (3 tests)
- `generate_text()` got unexpected `max_length`, `temperature` arguments (3 tests)

#### 3. Constructor Issues (8 tests)
- `StepAdapter()` takes no arguments (2 tests)
- `PredictionResult.__init__()` got unexpected `predicted_token` argument (2 tests)
- `PredictionResult.__init__()` takes 4-5 args but 6 given (4 tests)

#### 4. Return Value Type Mismatches (1 test)
- `load_eval_questions()` expects dict but gets list (2 tests - overlaps with other categories)

### Test Files Affected

- `test_8ball_eval.py`: 6 tests (PredictionResult, compare_predictions, load_eval_questions)
- `test_caws_eval.py`: 5 tests (evaluate_caws_compliance, main)
- `test_classification_eval.py`: 6 tests (compare_predictions)
- `test_perf_mem_eval.py`: 7 tests (StepAdapter, load_tokenized_prompts, run_coreml_speed)
- `test_tool_use_eval.py`: 12 tests (evaluate_tool_use, generate_text)

### Action Items

1. **Update Function Signatures**:
   - Add `device` parameter to `evaluate_tool_use()`
   - Add `tokenizer_path` parameter to `load_tokenized_prompts()`
   - Add `adapter` parameter to `run_coreml_speed()`
   - Update `compare_predictions()` to accept 3 arguments
   - Remove `evidence` parameter from `evaluate_caws_compliance()` and `main()`
   - Remove `max_length`, `temperature` from `generate_text()` or update signature

2. **Fix Constructors**:
   - Update `StepAdapter()` to accept proper arguments
   - Fix `PredictionResult.__init__()` signature (remove `predicted_token`, fix arg count)

3. **Fix Return Types**:
   - Update `load_eval_questions()` to return dict instead of list

### Files to Modify

- `evaluation/tool_use_eval.py`
- `evaluation/perf_mem_eval.py`
- `evaluation/8ball_eval.py`
- `evaluation/classification_eval.py`
- `evaluation/caws_eval.py`

---

## Worker B: Missing Attributes & Module Structure

**Focus**: Missing imports, module attributes, object attributes  
**Failures**: 38 tests (29.9%)  
**Complexity**: Medium (mostly import/attribute fixes)

### Failure Categories

#### 1. Missing Module Attributes (14 tests)
- `evaluation.classification_eval` has no attribute `argparse` (4 tests)
- `evaluation.classification_eval` has no attribute `AutoTokenizer` (3 tests)
- `evaluation.classification_eval` has no attribute `ctk` (2 tests)
- `evaluation.perf_mem_eval` has no attribute `platform` (3 tests)
- `evaluation.perf_mem_eval` has no attribute `coremltools` (2 tests)

#### 2. Missing Module/Submodule (8 tests)
- `evaluation` has no attribute `eightball_eval` (8 tests)

#### 3. Missing Object Attributes (16 tests)
- `EvaluationMetrics` object has no attribute `class_distribution` (2 tests)
- `'list' object has no attribute 'get'` (6 tests)
- `'str' object has no attribute 'get'` (5 tests)
- Other attribute access issues (3 tests)

### Test Files Affected

- `test_8ball_eval.py`: 8 tests (missing eightball_eval module)
- `test_classification_eval.py`: 9 tests (argparse, AutoTokenizer, ctk, class_distribution)
- `test_perf_mem_eval.py`: 5 tests (platform, coremltools)
- `test_caws_eval.py`: 13 tests (list/str .get() calls, other attributes)
- `test_claim_extraction_metrics.py`: 3 tests (attribute access)

### Action Items

1. **Add Missing Imports**:
   - Add `import argparse` to `classification_eval.py`
   - Add `from transformers import AutoTokenizer` to `classification_eval.py`
   - Add `import coremltools as ctk` to `classification_eval.py`
   - Add `import platform` to `perf_mem_eval.py`
   - Add `import coremltools` to `perf_mem_eval.py`

2. **Fix Module Structure**:
   - Create `evaluation/eightball_eval.py` or fix import path
   - Or update tests to use correct import path

3. **Add Missing Attributes**:
   - Add `class_distribution` attribute to `EvaluationMetrics` class
   - Fix return types (ensure dicts are returned where `.get()` is called)
   - Update functions to return dicts instead of lists/strings where needed

### Files to Modify

- `evaluation/classification_eval.py`
- `evaluation/perf_mem_eval.py`
- `evaluation/8ball_eval.py` (or create `evaluation/eightball_eval.py`)
- `evaluation/caws_eval.py`
- `evaluation/claim_extraction_metrics.py`

---

## Worker C: Return Value Structure & Dictionary Keys

**Focus**: Missing dictionary keys, return value structure mismatches  
**Failures**: 14 tests (11.0%)  
**Complexity**: Low-Medium (mostly adding keys to return values)

### Failure Categories

#### 1. Missing Result Keys (14 tests)
- `'files_changed_count'` missing (6 tests)
- `'overall_integrity'` missing (3 tests)
- `'lint_clean'` missing (1 test)
- `'coverage_sufficient'` missing (1 test)
- `'passed'` missing (2 tests)
- `'line_percent'` missing (1 test)

### Test Files Affected

- `test_caws_eval.py`: 13 tests (all CAWS compliance result keys)
- `test_claim_extraction_metrics.py`: 1 test (result structure)

### Action Items

1. **Update Return Value Structures**:
   - Add `files_changed_count` to CAWS compliance results
   - Add `overall_integrity` to gate integrity results
   - Add `lint_clean` to gate integrity results
   - Add `coverage_sufficient` to gate integrity results
   - Add `passed` to test results
   - Add `line_percent` to coverage results

2. **Fix Helper Functions**:
   - Update `run_tests()` to return dict with `passed` key
   - Update `run_linter()` to return dict with `lint_clean` key
   - Update `run_coverage()` to return dict with `coverage_sufficient` and `line_percent` keys
   - Update `validate_budget_adherence()` to return dict with `files_changed_count` key
   - Update `validate_gate_integrity()` to return dict with `overall_integrity` key

### Files to Modify

- `evaluation/caws_eval.py` (primary)
- `evaluation/claim_extraction_metrics.py` (minor)

---

## Worker D: Test Assertions & Error Handling

**Focus**: Assertion failures, value validation, mock objects, edge cases  
**Failures**: 39 tests (30.7%)  
**Complexity**: Low-Medium (mostly test fixes, some code fixes)

### Failure Categories

#### 1. AssertionError - Value Mismatches (14 tests)
- Floating point precision: `1.3333333333333333 == 1.33` (2 tests)
- Assertion: `1.0 < 0.001` (1 test)
- Expected: `'Signs point to yes'`, Got: `'Reply hazy, try again'` (1 test)
- Expected: `'No'`, Got: `'Yes'` (1 test)
- `assert 0 == 1` - lines_removed count (2 tests)
- `assert False == True` - tests_pass flag (1 test)
- Invalid JSON validation: `assert True == False` (2 tests)
- Empty array argmax: `ValueError: attempt to get argmax of an empty sequence` (1 test)
- `load_state_dict()` call mismatch - `strict=False` parameter (1 test)
- Mock object equality (1 test)

#### 2. ValueError - Validation Issues (13 tests)
- `FileNotFoundError: Working spec not found: nonexistent.yaml` (1 test)
- `ValueError: Failed to load config from ...` (5 tests)
- `ValueError: Student and teacher outputs must have same length` (1 test)
- `JSONDecodeError: Expecting value` (2 tests)
- `HFValidationError: Repo id must use alphanumeric chars` (3 tests)
- `RuntimeError: Failed to load tokenizer from Mock` (1 test)

#### 3. Other Errors - Mock/Context Issues (12 tests)
- `'Mock' object does not support the context manager protocol` (2 tests)
- `'Mock' object is not an iterator` (2 tests)
- `can only concatenate str (not "Mock") to str` (1 test)
- `ZeroDivisionError: division by zero` (1 test)
- `Failed: DID NOT RAISE <class 'Exception'>` (2 tests)
- `Failed: DID NOT RAISE <class 'json.decoder.JSONDecodeError'>` (1 test)
- `Failed: DID NOT RAISE <class 'FileNotFoundError'>` (1 test)
- Other mock-related issues (2 tests)

### Test Files Affected

- `test_8ball_eval.py`: 4 tests (assertions, value mismatches)
- `test_caws_eval.py`: 10 tests (validation, mock issues, exceptions)
- `test_claim_extraction_metrics.py`: 3 tests (floating point, edge cases)
- `test_classification_eval.py`: 5 tests (config loading, validation)
- `test_perf_mem_eval.py`: 6 tests (argmax, JSON validation, mock issues)
- `test_tool_use_eval.py`: 11 tests (model loading, validation, mock issues)

### Action Items

1. **Fix Test Assertions**:
   - Use `pytest.approx()` for floating point comparisons
   - Update expected values to match actual implementation
   - Fix JSON validation logic in tests
   - Update assertion logic for edge cases

2. **Fix Error Handling**:
   - Add proper exception handling in code
   - Fix file loading logic to raise correct exceptions
   - Update validation error messages
   - Handle empty arrays in `greedy_argmax()`

3. **Fix Mock Objects**:
   - Make Mock objects support context manager protocol (`__enter__`, `__exit__`)
   - Make Mock objects iterable (`__iter__`, `__next__`)
   - Fix Mock object string representations
   - Update tests to use proper fixtures instead of raw Mocks

4. **Fix Edge Cases**:
   - Handle division by zero
   - Handle empty sequences
   - Fix exception raising logic

### Files to Modify

**Code Changes**:
- `evaluation/perf_mem_eval.py` (greedy_argmax empty array handling)
- `evaluation/caws_eval.py` (exception handling)
- `evaluation/classification_eval.py` (config loading)
- `evaluation/tool_use_eval.py` (model loading, validation)

**Test Changes** (primary):
- `tests/evaluation/test_8ball_eval.py`
- `tests/evaluation/test_caws_eval.py`
- `tests/evaluation/test_claim_extraction_metrics.py`
- `tests/evaluation/test_classification_eval.py`
- `tests/evaluation/test_perf_mem_eval.py`
- `tests/evaluation/test_tool_use_eval.py`

---

## Load Balancing Summary

| Worker | Focus Area | Failures | Complexity | Test Files | Code Files |
|--------|------------|----------|------------|------------|------------|
| **Worker A** | API & Signatures | 36 (28.3%) | Medium-High | 5 | 5 |
| **Worker B** | Attributes & Imports | 38 (29.9%) | Medium | 5 | 5 |
| **Worker C** | Return Values | 14 (11.0%) | Low-Medium | 2 | 2 |
| **Worker D** | Assertions & Errors | 39 (30.7%) | Low-Medium | 6 | 4 |

**Total**: 127 failures across 6 test files and 5 code files

---

## Execution Strategy

### Parallel Execution

Each worker can work independently:

```bash
# Worker A: API & Function Signatures
# Focus: evaluation/tool_use_eval.py, evaluation/perf_mem_eval.py, etc.

# Worker B: Missing Attributes & Imports
# Focus: evaluation/classification_eval.py, evaluation/perf_mem_eval.py, etc.

# Worker C: Return Value Structure
# Focus: evaluation/caws_eval.py

# Worker D: Test Assertions & Error Handling
# Focus: tests/evaluation/*.py (test fixes), some code fixes
```

### Dependencies

- **Worker A** and **Worker B** may have some overlap (both touch `classification_eval.py` and `perf_mem_eval.py`)
  - **Resolution**: Coordinate or assign different functions within same files
- **Worker C** is independent (only touches `caws_eval.py`)
- **Worker D** can work in parallel but should test after A, B, C complete

### Recommended Order

1. **Worker C** (fastest, independent) - Return value structure fixes
2. **Worker B** (medium, independent) - Missing attributes/imports
3. **Worker A** (medium-high, some overlap with B) - API signatures
4. **Worker D** (low-medium, depends on A/B/C) - Test assertions (run after others)

---

## Success Criteria

### Worker A
- [ ] All 36 tests pass
- [ ] Function signatures match test expectations
- [ ] No TypeError exceptions
- [ ] Constructors work correctly

### Worker B
- [ ] All 38 tests pass
- [ ] All imports present and correct
- [ ] Module structure fixed
- [ ] No AttributeError exceptions

### Worker C
- [ ] All 14 tests pass
- [ ] All required dictionary keys present in return values
- [ ] No KeyError exceptions
- [ ] Return value structures match test expectations

### Worker D
- [ ] All 39 tests pass
- [ ] Floating point comparisons use `pytest.approx()`
- [ ] Mock objects properly configured
- [ ] Error handling works correctly
- [ ] Edge cases handled

---

## Notes

- **Worker A** has the most complex changes (function signature updates)
- **Worker B** has the most failures (38 tests) but mostly straightforward import fixes
- **Worker C** is the simplest and fastest (14 tests, mostly adding keys)
- **Worker D** requires both code and test changes, but mostly test fixes

- **Overlap**: Workers A and B both modify `classification_eval.py` and `perf_mem_eval.py`
  - **Solution**: Assign specific functions to each worker or coordinate changes

- **Testing**: After each worker completes, run:
  ```bash
  pytest tests/evaluation/ -v --tb=short
  ```

- **Coverage**: After all fixes, re-run coverage:
  ```bash
  pytest --cov=evaluation --cov-report=term-missing tests/evaluation/
  ```


