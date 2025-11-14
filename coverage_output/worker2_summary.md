# Worker 2 Coverage Report - Evaluation Module

## Summary

**Worker**: 2  
**Module**: `evaluation/`  
**Test Execution Date**: Generated from coverage run  
**Total Statements**: 8,661  
**Coverage**: 6.44%  
**Tests Run**: 189  
**Tests Passed**: 62  
**Tests Failed**: 127

## Coverage by File

### Well-Covered Files (>30% coverage)

| File | Coverage | Statements | Covered |
|------|----------|------------|---------|
| `claim_extraction_metrics.py` | 61.2% | 67 | 41/67 |
| `caws_eval.py` | 47.3% | 169 | 80/169 |
| `tool_use_eval.py` | 37.6% | 197 | 74/197 |
| `perf_mem_eval.py` | 35.2% | 429 | 151/429 |
| `8ball_eval.py` | 34.4% | 180 | 62/180 |
| `classification_eval.py` | 30.9% | 223 | 69/223 |

### Uncovered Files (0% coverage)

| File | Statements |
|------|------------|
| `compare_8ball_pipelines.py` | 77 |
| `long_ctx_eval.py` | 4 |
| `performance_benchmarks.py` | 89 |
| `pipeline_preservation_eval.py` | 100 |
| `reasoning_eval.py` | 106 |
| `toy/binary_classifier.py` | 23 |
| `toy/eight_ball.py` | 30 |
| `toy/eight_ball_config.py` | 22 |
| `toy/ternary_classifier.py` | 23 |
| `toy_contracts.py` | 151 |

## Test Results

### Test Files Executed

1. `tests/evaluation/test_8ball_eval.py` - 43 tests
2. `tests/evaluation/test_caws_eval.py` - 50 tests
3. `tests/evaluation/test_claim_extraction_metrics.py` - 35 tests
4. `tests/evaluation/test_classification_eval.py` - 48 tests
5. `tests/evaluation/test_perf_mem_eval.py` - 28 tests
6. `tests/evaluation/test_tool_use_eval.py` - 21 tests

### Common Failure Patterns

1. **API Mismatches**: Tests expect different function signatures than implemented
2. **Missing Attributes**: Tests reference attributes that don't exist in modules
3. **Type Errors**: Function calls with incorrect argument types
4. **Import Errors**: Tests trying to patch modules/attributes that don't exist
5. **Assertion Failures**: Floating point precision issues, expected values don't match

## Coverage Goals

**Current Coverage**: 6.44%  
**Target Coverage**: 80%+ line coverage, 90%+ branch coverage

### Priority Files for Improvement

1. **High Priority** (Large files with low coverage):
   - `perf_mem_eval.py` (429 stmts, 35.2% coverage)
   - `classification_eval.py` (223 stmts, 30.9% coverage)
   - `tool_use_eval.py` (197 stmts, 37.6% coverage)

2. **Medium Priority** (Uncovered files):
   - `toy_contracts.py` (151 stmts, 0% coverage)
   - `pipeline_preservation_eval.py` (100 stmts, 0% coverage)
   - `reasoning_eval.py` (106 stmts, 0% coverage)
   - `performance_benchmarks.py` (89 stmts, 0% coverage)

3. **Low Priority** (Small uncovered files):
   - `compare_8ball_pipelines.py` (77 stmts, 0% coverage)
   - `toy/*.py` files (98 stmts total, 0% coverage)
   - `long_ctx_eval.py` (4 stmts, 0% coverage)

## Coverage Artifacts

- **HTML Report**: `htmlcov/worker2/index.html`
- **JSON Report**: `coverage_output/worker2_coverage.json`
- **Test Output**: `coverage_output/worker2_test_output.txt`

## Next Steps

1. **Fix Test Failures**: Address the 127 failing tests to improve coverage
2. **Add Missing Tests**: Create tests for uncovered files (0% coverage)
3. **Improve Existing Tests**: Update tests to match current API signatures
4. **Increase Coverage**: Target 80%+ coverage for all evaluation modules
5. **Validate Coverage**: Ensure branch coverage meets 90%+ threshold

## Notes

- Worker 2 is responsible for testing the `evaluation/` module
- Coverage includes all files that import from `evaluation/` module
- Test failures indicate API mismatches between tests and implementation
- Some test failures are due to missing test infrastructure (mocks, fixtures)

