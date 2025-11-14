# Worker 2 Test Failure Analysis

## Summary

**Total Tests**: 189  
**Passed**: 62 (32.8%)  
**Failed**: 127 (67.2%)

## Failure Categories

### 1. TypeError - Function Signature Mismatches (Most Common)

#### Missing Arguments
- `evaluate_tool_use()` missing `device` argument (5 tests)
- `load_tokenized_prompts()` missing `tokenizer_path` argument (4 tests)
- `run_coreml_speed()` missing `adapter` argument (1 test)

#### Unexpected Arguments
- `compare_predictions()` takes 2 args but 3 given (6 tests)
- `evaluate_caws_compliance()` got unexpected `evidence` argument (5 tests)
- `main()` got unexpected `evidence` argument (3 tests)
- `generate_text()` got unexpected `max_length`, `temperature` arguments (3 tests)

#### Constructor Issues
- `StepAdapter()` takes no arguments (2 tests)
- `PredictionResult.__init__()` got unexpected `predicted_token` argument (2 tests)
- `PredictionResult.__init__()` takes 4-5 args but 6 given (4 tests)

**Impact**: 36 tests (28.3% of failures)

### 2. AttributeError - Missing Attributes/Modules (Second Most Common)

#### Missing Module Attributes
- `evaluation.classification_eval` has no attribute `argparse` (4 tests)
- `evaluation.classification_eval` has no attribute `AutoTokenizer` (3 tests)
- `evaluation.classification_eval` has no attribute `ctk` (2 tests)
- `evaluation.perf_mem_eval` has no attribute `platform` (3 tests)
- `evaluation.perf_mem_eval` has no attribute `coremltools` (2 tests)
- `evaluation` has no attribute `eightball_eval` (8 tests)

#### Missing Object Attributes
- `EvaluationMetrics` object has no attribute `class_distribution` (1 test)
- `'list' object has no attribute 'get'` (6 tests)
- `'str' object has no attribute 'get'` (5 tests)

**Impact**: 38 tests (29.9% of failures)

### 3. KeyError - Missing Dictionary Keys

#### Missing Result Keys
- `'files_changed_count'` missing (6 tests)
- `'overall_integrity'` missing (3 tests)
- `'lint_clean'` missing (1 test)
- `'coverage_sufficient'` missing (1 test)
- `'passed'` missing (2 tests)
- `'line_percent'` missing (1 test)

**Impact**: 14 tests (11.0% of failures)

### 4. AssertionError - Test Assertions Failing

#### Value Mismatches
- Floating point precision: `1.3333333333333333 == 1.33` (2 tests)
- Assertion: `1.0 < 0.001` (1 test)
- Expected: `'Signs point to yes'`, Got: `'Reply hazy, try again'` (1 test)
- Expected: `'No'`, Got: `'Yes'` (1 test)

#### Logic Failures
- `assert 0 == 1` - lines_removed count (2 tests)
- `assert False == True` - tests_pass flag (1 test)
- Invalid JSON validation: `assert True == False` (2 tests)
- Empty array argmax: `ValueError: attempt to get argmax of an empty sequence` (1 test)

#### Mock Assertions
- `load_state_dict()` call mismatch - `strict=False` parameter (1 test)
- Mock object equality (1 test)

**Impact**: 14 tests (11.0% of failures)

### 5. ValueError - Value Validation Issues

#### File/Module Not Found
- `FileNotFoundError: Working spec not found: nonexistent.yaml` (1 test)
- `ValueError: Failed to load config from ...` (5 tests)
- `ValueError: Student and teacher outputs must have same length` (1 test)

#### Validation Errors
- `JSONDecodeError: Expecting value` (2 tests)
- `HFValidationError: Repo id must use alphanumeric chars` (3 tests)
- `RuntimeError: Failed to load tokenizer from Mock` (1 test)

**Impact**: 13 tests (10.2% of failures)

### 6. Other Errors

#### Context Manager Protocol
- `'Mock' object does not support the context manager protocol` (2 tests)

#### Iterator Issues
- `'Mock' object is not an iterator` (2 tests)

#### String Concatenation
- `can only concatenate str (not "Mock") to str` (1 test)

#### Division by Zero
- `ZeroDivisionError: division by zero` (1 test)

#### Expected Exceptions Not Raised
- `Failed: DID NOT RAISE <class 'Exception'>` (2 tests)
- `Failed: DID NOT RAISE <class 'json.decoder.JSONDecodeError'>` (1 test)
- `Failed: DID NOT RAISE <class 'FileNotFoundError'>` (1 test)

**Impact**: 12 tests (9.4% of failures)

## Test File Breakdown

### test_8ball_eval.py
- **Total**: 43 tests
- **Passed**: 7
- **Failed**: 36
- **Main Issues**:
  - Missing `eightball_eval` module (8 tests)
  - `PredictionResult` signature mismatches (6 tests)
  - `load_eval_questions()` expects dict but gets list (2 tests)

### test_caws_eval.py
- **Total**: 50 tests
- **Passed**: 0
- **Failed**: 50
- **Main Issues**:
  - Missing result keys (`files_changed_count`, `overall_integrity`, etc.) (13 tests)
  - `evaluate_caws_compliance()` signature mismatch (5 tests)
  - `validate_provenance_clarity()` expects dict but gets str (5 tests)
  - Missing helper function return keys (4 tests)

### test_claim_extraction_metrics.py
- **Total**: 35 tests
- **Passed**: 32
- **Failed**: 3
- **Main Issues**:
  - Floating point precision in assertions (2 tests)
  - Edge case validation (1 test)

### test_classification_eval.py
- **Total**: 48 tests
- **Passed**: 12
- **Failed**: 36
- **Main Issues**:
  - Missing `argparse` attribute (4 tests)
  - Missing `AutoTokenizer` attribute (3 tests)
  - `compare_predictions()` signature mismatch (6 tests)
  - Config loading issues (5 tests)
  - Missing `class_distribution` attribute (2 tests)

### test_perf_mem_eval.py
- **Total**: 28 tests
- **Passed**: 11
- **Failed**: 17
- **Main Issues**:
  - Missing `platform` attribute (3 tests)
  - Missing `coremltools` attribute (2 tests)
  - `load_tokenized_prompts()` signature mismatch (4 tests)
  - `StepAdapter()` constructor issues (2 tests)
  - Mock context manager issues (2 tests)

### test_tool_use_eval.py
- **Total**: 21 tests
- **Passed**: 0
- **Failed**: 21
- **Main Issues**:
  - `evaluate_tool_use()` missing `device` argument (5 tests)
  - `generate_text()` signature mismatches (4 tests)
  - Mock object issues (3 tests)
  - HFValidationError with Mock objects (3 tests)

## Root Cause Analysis

### 1. API Evolution
Many tests were written for older API versions. The implementation has changed but tests haven't been updated.

### 2. Missing Test Infrastructure
- Tests try to patch modules/attributes that don't exist
- Mock objects don't properly simulate real objects
- Missing fixtures and test helpers

### 3. Type/Interface Mismatches
- Tests expect dictionaries but implementation returns strings/lists
- Function signatures changed (added/removed parameters)
- Return value structure changed

### 4. Test Data Issues
- Floating point precision in assertions
- Mock objects passed where real objects expected
- Invalid test data that doesn't match expected format

## Recommendations

### High Priority Fixes

1. **Update Function Signatures** (36 tests)
   - Add missing `device` parameter to `evaluate_tool_use()`
   - Add missing `tokenizer_path` parameter to `load_tokenized_prompts()`
   - Update `compare_predictions()` to accept 3 arguments
   - Remove `evidence` parameter from `evaluate_caws_compliance()`

2. **Fix Missing Attributes** (38 tests)
   - Add proper imports in modules (`argparse`, `platform`, `coremltools`)
   - Fix module structure (`eightball_eval` submodule)
   - Update return value structures to include expected keys

3. **Update Return Values** (14 tests)
   - Add missing keys to result dictionaries
   - Fix return value types (dict vs str vs list)

### Medium Priority Fixes

4. **Fix Test Assertions** (14 tests)
   - Use `pytest.approx()` for floating point comparisons
   - Update expected values to match actual implementation
   - Fix JSON validation logic

5. **Improve Error Handling** (13 tests)
   - Add proper exception handling in tests
   - Fix file loading logic
   - Update validation error messages

### Low Priority Fixes

6. **Fix Mock Objects** (12 tests)
   - Make Mock objects support context manager protocol
   - Make Mock objects iterable
   - Fix Mock object string representations

## Test Coverage Impact

Current coverage is **6.44%** with many test failures. Fixing these issues would:
- Increase test coverage significantly
- Improve test reliability
- Make tests more maintainable
- Enable better CI/CD integration

## Next Steps

1. Fix function signatures to match test expectations
2. Update return value structures
3. Add missing imports and attributes
4. Fix test assertions and mock objects
5. Re-run tests to verify fixes
6. Update coverage reports

