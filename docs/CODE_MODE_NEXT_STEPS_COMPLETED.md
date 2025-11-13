# Code-Mode Next Steps - Completion Summary

## ✅ Completed Items

### 1. Proper Span Target Computation (MET)

**Implementation:**

- Rewrote `compute_span_targets_from_tokenized()` to use tokenizer `offset_mapping` when available
- Falls back to character-based approximation if `offset_mapping` not available
- Integrated into dataset loading: `KDDataset.__getitem__()` computes spans from `teacher_text` if not pre-computed
- Uses precise token positions via `encode_plus(return_offsets_mapping=True)`

**Files Changed:**

- `data/generators/mcp_code_mode.py`: Complete rewrite with `offset_mapping` support
- `training/dataset.py`: Added span target computation during dataset loading

**Status:** ✅ Complete - Uses tokenizer offset_mapping for precise token positions

---

### 2. TorchScript Smoke Test (MET)

**Implementation:**

- Added `tests/unit/test_code_mode_loss_torchscript.py` with:
  - `test_code_mode_loss_torchscript()`: Verifies module can be scripted
  - `test_code_mode_loss_forward_method_torchscript()`: Tests vectorized eligibility computation (TorchScript-compatible)

**Files Changed:**

- `tests/unit/test_code_mode_loss_torchscript.py`: New test file

**Status:** ✅ Complete - TorchScript compatibility verified

---

### 3. Baseline Capture & Comparison (MET)

**Implementation:**

- Added `capture_baseline_ces_metrics()` function to capture baseline from direct-tool runner
- Updated `evaluate_code_mode_gates()` to compute:
  - `abs_ces`: Absolute CES tokens
  - `ces_delta_percent`: % change vs baseline
  - `ces_improvement_percent`: Improvement percentage (negative delta)
- Updated `summarize_results()` to include baseline comparison metrics in summary header
- Baseline loaded from `baseline_report_path` (same mechanism as speed gates)

**Files Changed:**

- `eval/scoring/scorer.py`: Added `capture_baseline_ces_metrics()` and enhanced gate evaluation
- `eval/reports/summarize.py`: Added baseline comparison metrics to summary

**Status:** ✅ Complete - Baseline capture and comparison implemented

---

### 4. Weight Scheduler Unit Test (MET)

**Implementation:**

- Added `tests/unit/test_code_mode_weight_scheduler.py` with:
  - `test_code_mode_weight_scheduler()`: Tests linear warmup from 0.1 to 0.3 over 5000 steps
  - `test_code_mode_weight_scheduler_zero_warmup()`: Tests zero warmup edge case
  - Verifies step=0 → 0.1, step=warmup → 0.3, linearity, and post-warmup hold

**Files Changed:**

- `tests/unit/test_code_mode_weight_scheduler.py`: New test file

**Status:** ✅ Complete - Weight scheduler fully tested

---

## Summary

All 4 next-step items have been completed:

1. ✅ **Span Target Computation**: Uses tokenizer `offset_mapping` for precise token positions
2. ✅ **TorchScript Test**: Module verified to be JIT-compatible
3. ✅ **Baseline Support**: Capture and comparison mechanism implemented
4. ✅ **Weight Scheduler Test**: Unit tests verify correct warmup behavior

## Remaining Work (Lower Priority)

### Sandbox CES Instrumentation

- Requires sandbox runtime integration
- Would instrument `fs.readFile` and `console.log` to measure actual bytes read/returned
- Currently uses metadata estimates (acceptable for now)

### Dataset Transform Assertion

- Could add assertion that ≥X% of eligible examples have non-empty spans
- Would catch data generation issues early

## Testing Status

All new tests pass:

- ✅ `test_code_mode_loss_torchscript`
- ✅ `test_code_mode_weight_scheduler`
- ✅ `test_single_small_tool_exempt` (from previous fixes)
- ✅ `test_adversarial_printing` (from previous fixes)
- ✅ `test_pii_binding_path_no_leak` (from previous fixes)

## Code Quality

- ✅ All changes pass linting
- ✅ Backward compatible (fallbacks for missing features)
- ✅ Proper error handling
- ✅ Documentation updated
