# Code-Mode Fixes Applied - Compliance Summary

## Status: Major Fixes Complete

### ✅ 1. Differentiable Loss (MET - Core)

**Fixed:**

- Loss now uses `log_softmax(student_logits)` for gradients
- Token-level log-probability shaping via `gather` operations
- Normalization by **count of scored token positions** (not just #eligible rows)
- Stores `num_scored_tokens` for logging

**Remaining:**

- TorchScript smoke test (see #11)

**Files Changed:**

- `training/losses.py`: Complete rewrite of forward() with proper normalization

---

### ⚠️ 2. Span Targets (PARTIAL)

**Fixed:**

- Data generator emits `span_targets` placeholder structure
- Helper function `compute_span_targets_from_text()` added

**Remaining:**

- Need proper tokenizer `offset_mapping` implementation for precise token positions
- Dataset transform to materialize spans at build time
- Assert ≥X% of eligible examples have non-empty spans

**Files Changed:**

- `data/generators/mcp_code_mode.py`: Added span_targets structure and helper

---

### ✅ 3. Config Single Source of Truth (MET)

**Fixed:**

- Removed env var checks from hot path (`train_step`)
- Config (`distill.code_mode.enabled`) is primary, env vars kept for backward compatibility only
- Checkpoint metadata **always persisted** (even when disabled)
- Eval config check updated (config first, then env var fallback)

**Files Changed:**

- `training/distill_kd.py`: Removed `TRAIN_CODE_MODE` from hot path
- `eval/reports/summarize.py`: Config-first check
- `training/distill_kd.py`: Checkpoint metadata always saved

---

### ✅ 4. Vectorized Eligibility (MET)

**Fixed:**

- `_compute_eligibility_mask()` handles both dict and list metadata
- Batch-safe boolean mask computation
- **NO train-time decoding** - removed all `tokenizer.decode` calls from code-mode path

**Files Changed:**

- `training/losses.py`: Added `_compute_eligibility_mask()` method
- `training/distill_kd.py`: Removed decoding from code-mode loss path

---

### ✅ 5. Hardened Leak Detection (MET)

**Fixed:**

- Compiled regex patterns (email, phone, SSN)
- `MIN_SNIPPET_LEN = 24` constant defined
- Only counts leaks when: `(has_binding_path && matched && snippet_len >= MIN_SNIPPET_LEN)`

**Files Changed:**

- `eval/scoring/scorer.py`: Hardened leak detection with compiled patterns and length threshold

---

### ⚠️ 6. CES Accounting (PARTIAL)

**Fixed:**

- Enhanced CES calculation with file_read_tokens and log_return_tokens
- Distinguishes code-mode (sandbox-isolated) vs direct-tool (echoed)

**Remaining:**

- Sandbox runtime hooks not implemented (would require sandbox integration)
- Still uses `len(split())` for token estimates (should use tokenizer or bytes→tokens heuristic)

**Files Changed:**

- `eval/scoring/scorer.py`: Enhanced CES calculation

---

### ✅ 7. Execution Correctness Gate (MET)

**Fixed:**

- E2E test structure exists
- Checks for side-effect (file write with hash)
- Verifies ≤200 chars of large blob in tokens

**Files Changed:**

- `tests/e2e/test_code_mode.py`: Added `TestCodeModeExecutionCorrectness`

---

### ✅ 8. Single-Tool Exemption (MET)

**Fixed:**

- Regression test added: `test_single_small_tool_exempt()`
- Verifies `eligibility_mask[b]==False` for single small tool
- Verifies loss contribution is exactly zero

**Files Changed:**

- `tests/e2e/test_code_mode.py`: Added `test_single_small_tool_exempt()`

---

### ✅ 9. Weight Scheduling (MET)

**Fixed:**

- Linear warmup implemented: `lerp(w_start, w_target, step / warmup_steps)`
- Weight added to `loss_dict` for logging
- Config-driven via `weight_schedule` section

**Remaining:**

- Unit test for scheduler (step=0 → 0.1, step=warmup → 0.3)

**Files Changed:**

- `training/distill_kd.py`: Implemented weight scheduler with logging

---

### ⚠️ 10. Baseline Support (MISSED)

**Remaining:**

- No `baselines.json` mechanism
- No absolute CES and %Δ reporting
- Need to capture direct-tool baseline for same task/seed

**Next Steps:**

- Add baseline capture mechanism
- Load baseline during eval
- Compute and report absolute CES and %Δ in summary header

---

### ⚠️ 11. Batch/Device Hygiene (PARTIAL)

**Fixed:**

- Module initialization moved to `main()` (before training loop)
- Module stored as `train_step._code_mode_loss_module` attribute
- All train-time string ops removed from code-mode path

**Remaining:**

- TorchScript smoke test not added
- Module still accessed via function attribute (could be cleaner with closure)

**Files Changed:**

- `training/distill_kd.py`: Module initialization in `main()`

---

### ✅ 12. Missing Tests (MET)

**Fixed:**

- ✅ Adversarial printing test: `test_adversarial_printing()`
- ✅ PII binding path test: `test_pii_binding_path_no_leak()`
- ✅ Single-tool exemption test: `test_single_small_tool_exempt()`

**Files Changed:**

- `tests/e2e/test_code_mode.py`: Added all three missing tests

---

## Summary

**Met (Core):** 7/12 items fully met
**Partial:** 4/12 items partially met (need follow-up work)
**Missed:** 1/12 items (baseline support)

**Critical Path Remaining:**

1. Proper span target computation with tokenizer offset_mapping
2. TorchScript smoke test
3. Baseline capture and comparison mechanism
4. Sandbox CES instrumentation (if sandbox integration available)

**High-Value Quick Wins:**

- Add unit test for weight scheduler
- Add TorchScript smoke test
- Implement baseline capture mechanism
