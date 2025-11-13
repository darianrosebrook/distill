# Toy Test Coverage Analysis: Milestones 1 & 2

## Overview

This document analyzes toy test coverage for:

- **Milestone 1**: Code-Mode MCP Distillation & Evaluation
- **Milestone 2**: Latent Reasoning + Outer-Loop Refinement

## Milestone 1 Coverage

### ✅ Covered by Toy Tests

**File**: `tests/e2e/test_toy_code_mode.py`

1. **`test_toy_training_without_code_mode()`**

   - ✅ Verifies backward compatibility (code-mode disabled by default)
   - ✅ Ensures training works without code-mode parameters
   - ✅ Validates `combined_kd_loss` handles missing code-mode gracefully

2. **`test_toy_training_with_code_mode_disabled()`**

   - ✅ Tests explicit code-mode disable (`code_mode_loss=None`, `code_mode_weight=0.0`)
   - ✅ Verifies no code-mode loss in output

3. **`test_toy_training_with_code_mode_enabled()`**

   - ✅ Tests code-mode loss module initialization with toy models
   - ✅ Verifies `CodeModePreferenceLoss` works with small vocabularies
   - ✅ Tests eligibility rules (min_tools, min_intermediate_chars, pii_tags_present)
   - ✅ Validates loss computation and gradient flow
   - ✅ Tests integration with `combined_kd_loss`

4. **`test_toy_training_env_var_doesnt_break()`**
   - ✅ Ensures `TRAIN_CODE_MODE` env var doesn't break when code-mode not configured
   - ✅ Tests backward compatibility with env vars

### ✅ Covered by Toy Tests

**File**: `tests/e2e/test_toy_code_mode.py`

5. **`test_toy_code_mode_with_span_targets()`**

   - ✅ Tests with actual span targets computed from tokenized text
   - ✅ Verifies token-level log-probability gathering works correctly
   - ✅ Tests `compute_span_targets_from_tokenized()` function
   - ✅ Verifies span targets structure and token positions

6. **`test_toy_code_mode_weight_scheduler_integration()`**

   - ✅ Tests linear warmup schedule for `code_mode_weight`
   - ✅ Verifies weight ramps from `start_weight` to `target_weight`
   - ✅ Tests weight at step 0, midpoint, warmup_steps, and after warmup
   - ✅ Verifies linear interpolation and weight affects total loss

7. **`test_mixed_batch_eligibility()`** (in `test_toy_combined_milestones.py`)

   - ✅ Tests vectorized eligibility mask with mixed batch
   - ✅ Verifies code-mode loss only applies to eligible samples
   - ✅ Tests some samples eligible, some not

8. **`test_toy_pipeline_with_code_mode()`** (in `test_toy_pipeline.py`)

   - ✅ End-to-end toy pipeline with code-mode enabled
   - ✅ Tests full flow: generate → train → export with code-mode
   - ✅ Verifies checkpoint structure and feature flags

### ✅ Covered by Full E2E Tests (Not Toy)

**File**: `tests/e2e/test_code_mode.py`

- Large blob scenarios
- Multi-tool chains
- PII scenarios
- Execution correctness gates
- Single-tool exemption
- Adversarial printing
- PII binding path

## Milestone 2 Coverage

### ✅ Covered by Toy Tests

**File**: `tests/e2e/test_latent_reasoning.py`

1. **`test_training_with_latent_curriculum()`**

   - ✅ Tests `LatentCurriculum` wrapper integration
   - ✅ Verifies sentinel insertion (`<bot>`, `<eot>`)
   - ✅ Tests loss masking over latent slots
   - ✅ Validates metadata tracking

2. **`test_inference_with_latent_spans()`**

   - ✅ Tests `LatentModeEngine` with sentinels
   - ✅ Verifies mode transitions (latent ↔ language)
   - ✅ Tests `forward_hidden` path (no LM head decode)

3. **`test_caws_budget_enforcement()`**

   - ✅ Tests max loops enforcement
   - ✅ Verifies halt conditions

4. **`test_latent_spans_respect_caws_tier()`**
   - ✅ Tests CAWS tier limits (Tier 1/2/3)
   - ✅ Verifies `max_latent_spans` per tier

**File**: `tests/e2e/test_token_reduction.py`

1. **`test_baseline_direct_cot()`**

   - ✅ Establishes baseline (no latent spans)
   - ✅ Measures baseline token count

2. **`test_latent_mode_token_reduction()`**

   - ✅ Tests latent mode token reduction
   - ✅ Measures efficiency metrics

3. **`test_token_reduction_at_equal_accuracy()`**

   - ✅ Verifies ≥25% token reduction at equal accuracy
   - ✅ Tests accuracy maintenance

4. **`test_efficiency_curves()`**
   - ✅ Tests efficiency curve computation
   - ✅ Validates accuracy vs tokens/time metrics

### ✅ Covered by Unit Tests (Not Toy)

- **`tests/models/test_halt_head.py`**: Halt head initialization and forward pass
- **`tests/runtime/test_refinement_controller.py`**: Refinement controller logic
- **`tests/training/test_latent_curriculum.py`**: Latent curriculum wrapper
- **`tests/runtime/test_latent_mode.py`**: Latent mode engine
- **`tests/runtime/test_latent_smoke.py`**: Smoke tests for latent mode

### ✅ Covered by Toy Tests

**File**: `tests/e2e/test_latent_reasoning.py`

5. **`test_toy_halt_head_integration()`**

   - ✅ Tests halt head logits influence on refinement controller
   - ✅ Tests halt probability threshold (0.7 default)
   - ✅ Verifies halt when probability > threshold
   - ✅ Verifies continue when probability < threshold
   - ✅ Tests halt head disabled mode

6. **`test_toy_training_inference_loop_mismatch()`**

   - ✅ Tests training with L=4-6 loops vs inference with L=1-3
   - ✅ Verifies training uses more loops than inference
   - ✅ Tests CAWS tier limits: Tier 1 (L≤1), Tier 2 (L≤2), Tier 3 (L≤3)
   - ✅ Verifies refinement controller respects inference loop limits

7. **`test_toy_progressive_curriculum()`**

   - ✅ Tests curriculum progression (c=1 → c=2)
   - ✅ Verifies 1 latent slot per replaced step with c=1
   - ✅ Verifies 2 latent slots per replaced step with c=2
   - ✅ Tests stability check structure
   - ✅ Verifies loss mask correctly masks all latent slots

8. **`test_toy_pipeline_with_latent_mode()`** (in `test_toy_pipeline.py`)

   - ✅ End-to-end toy pipeline with latent mode enabled
   - ✅ Tests full flow: generate → train → export with latent mode
   - ✅ Verifies checkpoint structure and feature flags

## Combined Milestone Coverage

### ✅ Covered by Toy Tests

**File**: `tests/e2e/test_toy_combined_milestones.py`

1. **`test_training_with_both_features()`**

   - ✅ Tests code-mode loss + latent curriculum together
   - ✅ Verifies loss mask from latent curriculum is applied correctly
   - ✅ Tests integration with `combined_kd_loss`
   - ✅ Verifies backward pass works with both features

2. **`test_code_mode_with_latent_spans()`**

   - ✅ Tests TypeScript orchestration within latent spans
   - ✅ Verifies code-mode can work with latent reasoning
   - ✅ Tests TS API calls can be generated within latent spans
   - ✅ Verifies code-mode loss applies to eligible scenarios even with latent curriculum

3. **`test_caws_budget_with_code_mode()`**

   - ✅ Tests CAWS tier limits with code-mode scenarios
   - ✅ Verifies refinement loops respect CAWS budgets
   - ✅ Tests code-mode doesn't bypass CAWS constraints
   - ✅ Verifies all three CAWS tiers (Tier 1/2/3)

4. **`test_mixed_batch_eligibility()`**

   - ✅ Tests vectorized eligibility mask with mixed batch
   - ✅ Verifies code-mode loss only applies to eligible samples
   - ✅ Tests latent curriculum can apply independently

5. **`test_full_pipeline_integration()`**

   - ✅ Tests full pipeline: generate → train (with both) → verify
   - ✅ Verifies integration points work together
   - ✅ Tests mixed eligibility in same batch

### ⚠️ Remaining Gaps

1. **Full Pipeline Export/Verify**: `test_toy_pipeline.py` still doesn't enable both features

   - Missing: End-to-end test with export → verify steps
   - Missing: Runtime verification of generated code

2. **CES Improvement Measurement**: No toy test for CES improvement with both features
   - Missing: Baseline vs combined milestone CES comparison
   - Missing: Token reduction measurement with both features

## Recommendations

### High Priority

1. ✅ **Add Combined Milestone Toy Test** (COMPLETED)

   - ✅ Created `test_toy_combined_milestones.py`
   - ✅ Tests code-mode loss + latent curriculum together
   - ✅ Verifies both features work in same training run
   - ✅ Tests TypeScript orchestration within latent spans
   - ✅ Tests CAWS budget enforcement with code-mode

2. **Enhance Toy Pipeline Test**

   - Update `test_toy_pipeline.py` to optionally enable milestones
   - Add flags: `--enable-code-mode`, `--enable-latent-mode`
   - Verify full pipeline with both features

3. **Add Span Targets Toy Test**

   - Test `CodeModePreferenceLoss` with actual span targets
   - Verify token-level log-probability gathering

4. **Add Weight Scheduler Toy Test**
   - Test linear warmup for `code_mode_weight`
   - Verify schedule: `start_weight` → `target_weight` over `warmup_steps`

### Medium Priority

5. **Add Halt Head Integration Test**

   - Test halt head logits in refinement controller
   - Verify halt probability threshold

6. **Add Progressive Curriculum Test**

   - Test curriculum progression (c=1 → c=2)
   - Verify training/inference loop mismatch

7. **Add Mixed Eligibility Batch Test**
   - Test vectorized eligibility mask with mixed batch
   - Verify some samples eligible, some not

### Low Priority

8. **Add CAWS + Code-Mode Integration Test**
   - Test CAWS tier limits with code-mode scenarios
   - Verify refinement loops with code-mode eligible tasks

## Test Execution Summary

### Run All Milestone 1 Toy Tests

```bash
pytest tests/e2e/test_toy_code_mode.py -v
```

### Run All Milestone 2 Toy Tests

```bash
pytest tests/e2e/test_latent_reasoning.py tests/e2e/test_token_reduction.py -v
```

### Run Combined Milestone Tests

```bash
pytest tests/e2e/test_toy_combined_milestones.py -v
```

### Run Full E2E Tests (Not Toy)

```bash
# Milestone 1
pytest tests/e2e/test_code_mode.py -v

# Milestone 2
pytest tests/runtime/test_latent_mode.py tests/runtime/test_latent_smoke.py -v
```

### Run Unit Tests

```bash
# Milestone 1
pytest tests/unit/test_code_mode_loss*.py -v

# Milestone 2
pytest tests/models/test_halt_head.py tests/runtime/test_refinement_controller.py tests/training/test_latent_curriculum.py -v
```

## Coverage Summary

| Component                | Milestone 1 | Milestone 2 | Combined   |
| ------------------------ | ----------- | ----------- | ---------- |
| **Toy Tests**            | ✅ 4 tests  | ✅ 8 tests  | ✅ 5 tests |
| **Full E2E**             | ✅ 7+ tests | ✅ Multiple | ❌ 0 tests |
| **Unit Tests**           | ✅ Multiple | ✅ Multiple | ❌ 0 tests |
| **Pipeline Integration** | ⚠️ Partial  | ⚠️ Partial  | ⚠️ Partial |

**Legend:**

- ✅ Fully covered
- ⚠️ Partially covered
- ❌ Not covered

## Conclusion

**Milestone 1** has good toy test coverage for basic functionality, but missing:

- Span targets integration
- Weight scheduling
- Full pipeline integration

**Milestone 2** has good toy test coverage for core features, but missing:

- Halt head integration
- Training/inference loop mismatch
- Full pipeline integration

**Combined milestones** now have **5 toy tests** covering:

- Training with both features
- TypeScript orchestration within latent spans
- CAWS budget enforcement
- Mixed batch eligibility
- Full pipeline integration

Remaining gaps:

- Full pipeline export/verify steps
- CES improvement measurement with both features
