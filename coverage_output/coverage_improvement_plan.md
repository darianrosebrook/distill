# Coverage Improvement Plan

## Current Status

**Overall Coverage**: 24.05% (2190/9104 lines)

### Recent Improvements
- `training/input_validation.py`: 53.01% (88/166) - 13 new tests
- `training/assertions.py`: 29.76% (25/84) - 20+ new tests
- `training/claim_extraction.py`: 68.48% (63/92) - 25+ new edge case tests
- `training/quality_scoring.py`: 7.83% (9/115) - 15+ new edge case tests

**Total New Tests**: 73+ tests added

## Module Coverage Analysis

### Training Modules - High Priority

#### 1. `training/quality_scoring.py` - 7.83% (9/115)
**Status**: Has comprehensive tests (67 tests) but low coverage
**Issue**: Tests likely use mocks or don't execute actual code paths
**Action**: 
- Run coverage analysis to identify untested lines
- Ensure tests execute actual functions, not mocks
- Add integration tests that exercise real code paths

#### 2. `training/logging_utils.py` - 0% (0/58)
**Status**: Has comprehensive tests (30 tests) but 0% coverage
**Issue**: Tests heavily mock StructuredLogger, not executing actual code
**Action**:
- Refactor tests to use real logger instances
- Test actual file I/O operations
- Test JSON serialization with real data
- Verify log file creation and content

#### 3. `training/caws_context.py` - 19.12% (26/136)
**Status**: Has comprehensive tests (24 tests) but low coverage
**Issue**: Tests may not cover all code paths or edge cases
**Action**:
- Add tests for error handling paths
- Test file I/O operations with invalid inputs
- Test YAML parsing edge cases
- Test budget derivation edge cases

#### 4. `training/feature_flags.py` - 0% (0/127)
**Status**: Has comprehensive tests (31 tests) but 0% coverage
**Issue**: Tests likely use mocks or don't execute actual FeatureManager code
**Action**:
- Ensure tests use real FeatureManager instances
- Test actual feature flag state management
- Test environment variable loading with real os.environ
- Test configuration application with real configs

#### 5. `training/assertions.py` - 29.76% (25/84)
**Status**: Recently improved but still needs work
**Action**:
- Continue adding edge case tests
- Test error handling paths
- Test assertion message formatting

#### 6. `training/input_validation.py` - 53.01% (88/166)
**Status**: Recently improved but still needs work
**Action**:
- Add more edge case tests
- Test validation error messages
- Test type conversion edge cases

### Training Modules - Medium Priority

#### 7. `training/caws_structure.py` - 87.80% (36/41)
**Status**: Well covered, minor improvements needed
**Action**: Add edge case tests for remaining uncovered lines

#### 8. `training/utils.py` - 88.89% (24/27)
**Status**: Well covered, minor improvements needed
**Action**: Add edge case tests for remaining uncovered lines

#### 9. `training/halt_targets.py` - 19.57% (9/46)
**Status**: Low coverage, needs tests
**Action**: Create comprehensive test suite

#### 10. `training/monitoring.py` - 0% (0/191)
**Status**: No coverage, large file
**Action**: Create comprehensive test suite

#### 11. `training/performance_monitor.py` - 0% (0/91)
**Status**: No coverage
**Action**: Create comprehensive test suite

### Evaluation Modules - All 0% Coverage

#### Priority Evaluation Modules:
1. `evaluation/perf_mem_eval.py` - 0% (0/429) - Largest file
2. `evaluation/classification_eval.py` - 0% (0/223)
3. `evaluation/tool_use_eval.py` - 0% (0/197)
4. `evaluation/8ball_eval.py` - 0% (0/181)
5. `evaluation/caws_eval.py` - 0% (0/169)
6. `evaluation/toy_contracts.py` - 0% (0/151)
7. `evaluation/reasoning_eval.py` - 0% (0/106)
8. `evaluation/pipeline_preservation_eval.py` - 0% (0/100)
9. `evaluation/performance_benchmarks.py` - 0% (0/89)
10. `evaluation/compare_8ball_pipelines.py` - 0% (0/78)
11. `evaluation/claim_extraction_metrics.py` - 0% (0/67)

## Recommended Next Steps

### Option 1: Continue Training Modules (Recommended)
**Focus**: Complete coverage for training modules with existing tests but low coverage

1. **Fix test execution issues**:
   - `training/quality_scoring.py` - Ensure tests execute real functions
   - `training/logging_utils.py` - Use real logger instances, not mocks
   - `training/feature_flags.py` - Use real FeatureManager instances
   - `training/caws_context.py` - Add missing edge case tests

2. **Expected Impact**: 
   - `quality_scoring.py`: 7.83% → 80%+ (106 lines)
   - `logging_utils.py`: 0% → 80%+ (46 lines)
   - `feature_flags.py`: 0% → 80%+ (102 lines)
   - `caws_context.py`: 19.12% → 80%+ (109 lines)
   - **Total**: ~363 new lines covered

### Option 2: Start Evaluation Modules
**Focus**: Create comprehensive test suites for evaluation modules

1. **Start with smaller modules**:
   - `evaluation/claim_extraction_metrics.py` - 67 lines
   - `evaluation/compare_8ball_pipelines.py` - 78 lines
   - `evaluation/performance_benchmarks.py` - 89 lines

2. **Expected Impact**: 
   - 3 modules × 80% coverage = ~190 lines covered
   - Lower impact than fixing training modules

### Option 3: Mixed Approach
**Focus**: Fix 2 training modules, then start evaluation modules

1. **Fix highest-impact training modules**:
   - `training/logging_utils.py` - 0% → 80%+ (46 lines)
   - `training/feature_flags.py` - 0% → 80%+ (102 lines)

2. **Then start evaluation modules**:
   - `evaluation/claim_extraction_metrics.py` - 67 lines
   - `evaluation/compare_8ball_pipelines.py` - 78 lines

## Recommendation

**Option 1 is recommended** because:
1. Tests already exist, just need to fix execution
2. Higher impact (363 lines vs 190 lines)
3. Faster to fix existing tests than create new ones
4. Better ROI - existing tests are comprehensive, just not executing

## Action Plan

### Phase 1: Fix Test Execution (Priority)
1. Analyze why `quality_scoring.py` tests show low coverage
2. Fix `logging_utils.py` tests to use real logger instances
3. Fix `feature_flags.py` tests to use real FeatureManager
4. Add missing edge case tests for `caws_context.py`

### Phase 2: Continue Training Modules
1. Improve `assertions.py` coverage (29.76% → 80%+)
2. Improve `input_validation.py` coverage (53.01% → 80%+)
3. Add tests for `halt_targets.py` (19.57% → 80%+)

### Phase 3: Start Evaluation Modules
1. Create tests for `evaluation/claim_extraction_metrics.py`
2. Create tests for `evaluation/compare_8ball_pipelines.py`
3. Create tests for `evaluation/performance_benchmarks.py`

## Success Metrics

- **Target Coverage**: 80%+ line coverage, 90%+ branch coverage
- **Current**: 24.05% line coverage, 0% branch coverage
- **Goal**: Reach 40%+ line coverage in next milestone
- **Long-term**: Reach 80%+ line coverage for all production modules

