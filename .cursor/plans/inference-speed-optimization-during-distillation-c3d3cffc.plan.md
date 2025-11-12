<!-- c3d3cffc-8780-4356-92e5-1f19933cb92d 71116018-6f5b-494d-af84-cfd1fb367ff5 -->
# Inference-Speed Optimization During Distillation (Replacement Plan)

## Purpose

Bake **latency and throughput** into the model *during* distillation, then validate on the **exported CoreML/ANE path** with the same governance you use for CAWS (strict/lax F1, privacy, controls). Treat speed and quality as co-equal gates.

---

## Invariants (must hold at all times)

- **No quality regressions:** `ΔF1_lax ≥ −0.01`, `ΔF1_strict ≥ −0.01`.
- **Zero privacy/control failures:** `privacy_ok_rate = 1.0`, `controls_with_integration = 0`.
- **Determinism:** sharded vs. baseline per-example equality passes with optimizations on.
- **Export path is source of truth:** measure TTFT/TPS/TTFA on **CoreML (mlprogram on ANE)**, not the PyTorch loop.
- **M-series hardware target:** All speed measurements on M1 Max 64GB or equivalent M2/M3 Pro hardware.
- **ANE placement:** All attention ops must run on ANE (0 CPU fallbacks, GPU fallback acceptable for embedding/logits only).

---

## Phase 0 — Foundations (one time)

**Files**

- `eval/cli.py`, `eval/scoring/scorer.py`
- `eval/reports/` (schema header)
- `coreml/runtime/*` (minimal harness)

**Actions**

1. **Report schema: add speed + hardware**
   ```json
   {
     "speed_metrics": {
       "ttft_ms": {"p50":0,"p90":0,"p95":0},
       "tps": {"p50":0,"p90":0,"p95":0},
       "ttfa_tokens": {"p50":0,"p95":0},
       "ttfa_ms": {"p50":0,"p95":0}
     },
     "hardware": {
       "soc":"<M-series>",
       "os":"macOS <ver>",
       "coremltools":"9.x",
       "export_path":"pytorch_exportedprogram_coreml"
     }
   }
   ```


Gate: refuse speed diffs across different `hardware.soc` unless `--allow-cross-hw`.

**M-series hardware detection:**

- Auto-detect M1 Max, M2 Pro, M3 Pro from `platform.processor()` or `sysctl -n machdep.cpu.brand_string`
- Normalize to chip family (M1-series, M2-series, M3-series) for comparison
- M1 Max 64GB is the baseline; M2/M3 Pro should match or exceed

2. **CoreML speed harness (minimal)**

   - Export → load `mlmodel` → run short fixed prompts.
   - Record TTFT (tokenizer + first step) & steady-state TPS.
   - Record TTFA (time/tokens until first valid tool JSON).

3. **Determinism mode for HTTP runners**

   - `--determinism-mode`: temp=0, top_p=1, **no retries**, capture `request_id`, fail if any retry occurs.

---

## Phase 1 — Latency-Aware Losses (with safety rails)

**Files**

- `training/losses.py`
- `training/distill_kd.py`
- `configs/kd_recipe.yaml`, `configs/student_9b_gqa.yaml`

**Add**

1. **Length-aware KD (hinged, completeness-aware)**

   - Penalize `(len_student − len_teacher)/len_teacher` *only if* student **omits** required tool arguments/evidence.
   - Exempt when per-field coverage (set-equivalence under normalization) is met.

2. **Early tool-call loss (gated + ramped)**

   - Apply **only** when `tool_should_be_used=True` (from teacher metadata).
   - Validate JSON using the **same schema** as the scorer.
   - Ramp `early_tool_weight: 0 → target` over first *K* epochs.

**Acceptance**

- `median_tokens_out_student ≤ 0.85 × teacher`.
- Tool-absent slice: no increase in false tool calls; TTFA remains ∞ (no tool).

---

## Phase 2 — Enumerated-Shape Training (export-aligned, M-series optimized)

**Files**

- `training/distill_kd.py`, `training/dataset.py`
- `configs/student_9b_gqa.yaml`

**Actions**

- Train at **ANE-optimal export shapes**: `[512, 1024, 2048]` (primary) with optional `4096` for long-context.
  - **M-series ANE characteristics:**
    - Optimal: 512, 1024, 2048 (best throughput/latency)
    - Acceptable: 4096 (long-context, slightly slower)
    - Avoid: >4096 (ANE efficiency drops, prefer chunking)
- Sampling = **Dirichlet schedule** approximating production mix:
  - **M1 Max / M2/M3 Pro**: `[0.4, 0.4, 0.15, 0.05]` (512, 1024, 2048, 4096)
  - Focus on 512/1024 for tool calls (most common)
  - 2048 for longer reasoning
  - 4096 for rare long-context cases
- Add **per-shape eval slices**; track drift.
- **M-series specific**: Validate ANE placement during export (no CPU fallbacks).

**Acceptance**

- `max_shape_drift(ΔF1_lax across shapes) ≤ 0.01`.
- **ANE validation**: Empirically verify ANE placement (Instruments → Core ML template)
  - Measure ANE residency: >80% steps on ANE (sample wall-times)
  - Fail CI if ANE residency drops >10% vs baseline
  - Note: CoreML decides placement at runtime; cannot hard-force via config
- Shape distribution matches production usage on M-series Macs (use `configs/hardware_profiles.yaml`).

---

## Phase 3 — QAT in Main Loop (late, guarded, M-series optimized)

**Files**

- `training/distill_kd.py`, `training/quant_qat_int8.py`
- `configs/student_9b_gqa.yaml` (`quant.enabled`, `quant.start_fraction`)

**Actions**

- Enable QAT in the **last 15–25%** of steps.
- Lower LR (e.g., 10×), optional EMA.
- **M-series ANE optimization**: Prioritize INT8 quantization (ANE's sweet spot)
  - Quantize: **per-channel INT8 weights** on linear/GEMM (ANE-optimized)
  - Keep LayerNorm/softmax FP16 (ANE supports mixed precision)
  - **KV cache**: FP16 (first pass), consider INT8 for M1 Max 64GB (more memory headroom)
  - **Attention ops**: Ensure INT8 weights → FP16 activations (ANE-optimal pattern)

**M-series specific considerations:**

- **M1 Max 64GB**: Can use larger KV cache (FP16), more aggressive quantization
- **M2/M3 Pro**: Similar to M1 Max, validate ANE placement explicitly
- **Batch size**: ANE prefers smaller batches (1-4), but 64GB unified memory allows larger micro-batches during training
- **Memory pressure**: Less concern with 64GB unified memory, but still optimize for ANE efficiency

**Acceptance**

- Probe cosine ≥ **0.999** (attention/MLP pre/post QAT).
- No NaNs. CoreML export shows **ANE-eligible** kernels.
- **ANE placement validation**: 
  - Use Instruments → Core ML template to verify attention/MLP ops on ANE
  - Measure ANE residency empirically (sample wall-times during inference)
  - Fail CI if ANE residency drops >10% vs baseline
  - Note: Cannot hard-force placement via config; verify empirically
- **Hardware-specific gates**: 
  - Use `eval/hw_profile.py` to match hardware profile
  - Gate on **relative regression** (≤5% vs last blessed on same profile)
  - See `configs/hardware_profiles.yaml` for chip-specific speed bands

---

## Phase 4 — Speed Metrics During Validation

**Files**

- `training/speed_metrics.py` (new)
- `training/distill_kd.py`, `training/tracing.py`

**Actions**

- Measure proxies (TTFT/TPS/TTFA) during **validation only**.
- Log to TB/W&B; tag with `hardware.soc` and `export=False` (these are proxies).

**Acceptance**

- Proxies are present every N steps; document expected offset vs. CoreML.

---

## Phase 5 — Speed Gates in Scorer (relative, per-HW)

**Files**

- `eval/scoring/scorer.py`, `eval/cli.py`
- `evaluation/performance_benchmarks.py`

**Actions**

- Enforce **relative** gates vs. last blessed report on **same hardware**:

  - TTFT p50/p95: **no regression > 5%**
  - TPS p50/p95: **no regression > 5%**
  - TTFA p95 tokens: **≤ 25**
- Co-gate with quality: a PR fails if **either** speed or quality regresses.

**Acceptance**

- PR and nightly runs fail on >5% regression; artifacts include speed/hardware header.

---

## Phase 6 — CoreML Runtime Measurement (authoritative, M-series hardware)

**Files**

- `evaluation/perf_mem_eval.py`
- `coreml/runtime/*`

**Actions**

- Replace placeholders with true CoreML eval on **M-series hardware**:

  - TTFT split: **Tokenizer** vs. **First CoreML step** (ANE latency).
  - TPS at steady state (ANE throughput).
  - **ANE-specific metrics:**
    - ANE utilization (% of ops on ANE vs GPU/CPU)
    - Memory bandwidth (unified memory advantage)
    - Per-shape performance (512/1024/2048/4096)
  - Optional: Energy (Instruments), micro-batch {1,2,4}.
  - **Hardware detection**: Auto-detect M1 Max, M2 Pro, M3 Pro and adjust expectations.

**Realistic performance bands** (for ~9B Worker, INT8-W/FP16-A, batch=1, good export):

- **M1 Max 64GB** (baseline):
  - TTFT p50: **300-500 ms** (context-dependent, ~2k tokens)
  - TTFT p95: **450-800 ms**
  - TPS p50: **25-35 tok/s** (2k context), **30-40 tok/s** (≤1k context)
  - TPS p95: **20-30 tok/s**
  - TTFA p95: **≤25 tokens** (absolute correctness gate)
- **M2/M3 Pro**: Similar or slightly better (see `configs/hardware_profiles.yaml`)

**Important**: These are **acceptance bands** for sanity checks. **CI gates are relative** (≤5% regression vs. last blessed report on same hardware profile).

**Acceptance**

- CoreML metrics within **5%** of Instruments; publish in report.
- **Hardware-specific gates**: 
  - Use `eval/hw_profile.py` to match hardware profile
  - Fail if measured on different hardware profile (unless `--allow-cross-hw`)
  - Gate on **relative regression** (≤5% vs last blessed on same profile), not absolute targets
- Report includes `hardware_profile_key` (e.g., "m1-max-64g", "m3-pro-36g") for reproducibility.
- **TTFT split**: Record `{tokenizer_ms, first_step_ms}` for diagnostics
- **KV size budget**: Log `(heads, head_dim, seq_len)` → KV bytes

---

## Governance & Determinism

**Always enforce**

- **Fingerprints:** `dataset_sha256`, `tool_registry_sha256`, `tokenizer_fingerprint`, `prompt_wrapper_sha256`, `runner_fingerprint`, `model_fingerprint`. Null → hard fail.
- **Sharding determinism:** per-example normalized equality, coverage = 100%, no dups/missing.
- **HTTP determinism-mode:** retries disabled; any retry → fail determinism run.

---

## Failure-Mode Cards

- **Hallucinated tools after Phase 1:** ensure `tool_should_be_used` gating; halve `early_tool_weight`; verify no-tool slice.
- **Strict F1 dips with length penalty:** enable completeness exemption; reduce hinge slope; cap penalty.
- **ANE fallback at export:** scan graph for `int64` tensors / unsupported activations; insert casts; re-export with enumerated shapes only.
- **QAT instability:** delay start, halve LR, freeze LayerNorm scale/bias last 10% steps.
- **M-series specific:**
  - **ANE not utilized:** 
    - Check `compute_units="all"` in export config
    - **Verify empirically**: Use Instruments → Core ML template (cannot hard-force via config)
    - Ensure INT8 weights + FP16 activations pattern
    - Avoid ops that force CPU (int64 tensors, unsupported activations)
    - Measure ANE residency: sample wall-times; fail if <80% steps on ANE
  - **Poor performance on M1 Max:** 
    - Validate shapes are ANE-optimal (512/1024/2048 primary)
    - **Use batch=1** for interactive (batch 2-4 helps TPS but hurts p95 latency)
    - Verify KV cache precision (FP16 for M1 Max)
    - **Use relative gates**: Don't chase absolute targets; gate on ≤5% regression vs baseline
    - Split TTFT: check `tokenizer_ms` vs `first_step_ms` for diagnostics
  - **Memory pressure (unlikely with 64GB):** 
    - Use unified memory to **avoid** pressure (larger KV, longer prompts)
    - If occurs, reduce batch size or KV cache precision
  - **Shape mismatch:** 
    - Ensure training shapes match export shapes exactly
    - ANE is sensitive to shape changes
    - Add periodic upweight for rare shapes (e.g., 4096 every N steps)

---

## Makefile Targets (add)

```makefile
# Authoritative CoreML speed run (same slice each time)
speed-coreml:
	python -m evaluation.perf_mem_eval \
	  --model coreml/artifacts/worker/model.mlpackage \
	  --dataset data/contextual_final.jsonl.head100 \
	  --out eval/reports/speed_coreml.json \
	  --hardware "$(HARDWARE)" \
	  --export-path pytorch_exportedprogram_coreml

# Train with latency-aware losses (ramped) + enumerated shapes
train-student-speed:
	python -m training.distill_kd \
	  --config configs/student_9b_gqa.yaml \
	  --loss.length_weight 0.2 \
	  --loss.early_tool_weight 0.0:0.2@warmup_k=3 \
	  --shapes 512,1024,2048,4096

# Enable QAT for final 20% of steps
train-student-qat:
	python -m training.distill_kd \
	  --config configs/student_9b_gqa.yaml \
	  --quant.enabled true \
	  --quant.start_fraction 0.8 \
	  --optimizer.lr 1e-4
```

---

## Acceptance Checklist (copy into CI doc)

- [ ] **Quality**: `ΔF1_lax ≥ −0.01`, `ΔF1_strict ≥ −0.01`; privacy=1.0; controls=0.
- [ ] **Determinism**: sharded==baseline per-example; fingerprints all present & equal.
- [ ] **Latency losses**: `median_tokens_out_student ≤ 0.85×teacher`; no-tool slice unchanged.
- [ ] **Enumerated shapes**: shape drift ≤ 0.01 F1.
- [ ] **QAT**: cosine ≥ 0.999; no NaNs; ANE kernels present; 0 CPU fallbacks.
- [ ] **Speed gates (same hardware)**: TTFT/TPS p50/p95 regressions ≤ 5%; TTFA p95 tokens ≤ 25.
- [ ] **CoreML parity**: CoreML speed within 5% of Instruments.

---

## Implementation Notes (drop-in stubs)

**`training/losses.py` signatures**

```python
def length_aware_kd_loss(student_ids, teacher_ids, required_fields_mask, hinge=0.15, slope=1.0):
    """
    Penalize extra length only when required_fields_mask indicates missing arguments/evidence.
    Returns scalar loss and diagnostics.
    """

def early_tool_call_loss(tokens, tool_should_be_used, json_validator, N=25, weight=0.2, ramp_t=0.0_1.0):
    """
    Encourage valid tool JSON within first N tokens when tool_should_be_used=True.
    Uses the same validator as eval scorer.
    """
```

**`training/speed_metrics.py`**

```python
def measure_proxy(model, batch, tokenizer) -> dict:
    # Return {'ttft_ms':..., 'tps':..., 'ttfa_tokens':..., 'ttfa_ms':...}
    # Called only in validation; tag export=False
```

**`evaluation/perf_mem_eval.py`**

```python
def run_coreml_speed(model_mlpackage, dataset_slice) -> dict:
    # ExportedProgram->CoreML path, measure TTFT/TPS/TTFA on ANE
    # Write eval/reports/speed_coreml.json with header.hardware + speed_metrics
```

---

Current progress.

### 1. Wired new losses into `training/distill_kd.py`

- Added imports for `length_aware_kd_loss` and `early_tool_call_loss`
- Created `compute_required_fields_present()` helper to check completeness
- Integrated length-aware KD loss:
  - Computes `required_fields_present` from batch data
  - Applies loss only when student length exceeds teacher and required fields are missing
  - Configurable via `use_length_aware_kd`, `length_kd_weight`, `length_kd_hinge`, `length_kd_slope`
- Integrated early tool call loss:
  - Uses `tool_should_be_used` from batch metadata
  - Supports `teacher_prefix_ids` for teacher-guided CE
  - Implements ramp schedule (0→1 over warmup epochs)
  - Configurable via `use_early_tool_call_loss`, `early_tool_weight`, `early_tool_N`, etc.

### 2. Updated CoreML speed harness (`evaluation/perf_mem_eval.py`)

- Added `load_tokenized_prompts()` to load and tokenize from `data/contextual_final.jsonl`
- Updated `main()` to use real dataset by default
- Added tokenizer support for TTFA detection
- Added `--max-samples` argument to limit evaluation size
- Improved TTFA detection with real tokenizer decoding

### 3. Added unit tests (`tests/unit/test_losses_speed.py`)

- `test_length_kd_completeness_exemption`: Verifies completeness exemption works
- `test_length_kd_no_excess`: Tests zero loss when student ≤ teacher length
- `test_length_kd_hinge`: Tests hinge mechanism
- `test_early_tool_ce_only_when_needed`: Tests CE loss application
- `test_early_tool_json_prior_fallback`: Tests JSON prior fallback
- `test_early_tool_masked_when_not_needed`: Tests safety masking
- `test_early_tool_ramp`: Tests ramp scaling

### Configuration example

Add to your training config YAML:

```yaml
distillation:
  # Length-aware KD loss
  use_length_aware_kd: true
  length_kd_weight: 0.05
  length_kd_hinge: 0.15
  length_kd_slope: 1.0
  
  # Early tool call loss
  use_early_tool_call_loss: true
  early_tool_weight: 0.05
  early_tool_N: 25
  early_tool_warmup_epochs: 5
  early_tool_json_prior_weight: 0.02
  early_tool_ce_weight: 0.2
```

### Usage

The losses are integrated and will activate when:

- `use_length_aware_kd: true` and batch contains `teacher_attention_mask`
- `use_early_tool_call_loss: true` and batch contains `tool_should_be_used`

Both losses return diagnostics that are logged automatically in the training loop.

## Continue:

Review the current progress and start filling in any gaps that arise with full integration and implementations.

## Additional M-Series Optimizations

See `docs/M_SERIES_ADVANCED_OPTIMIZATIONS.md` for:

- Prompt caching (30-50% TTFT reduction)
- Speculative decoding integration (25-40% TTFT improvement)
- ANE residency monitoring (validation)
- Tokenizer I/O optimization
- KV cache optimization
- Batch policy enforcement
- Energy efficiency tracking
- Memory layout optimization

### To-dos

- [x] Review training/losses.py
- [x] Review training/distill_kd.py
- [x] Review evaluation/perf_mem_eval.py
- [x] Review tests/unit/test_losses_speed.py
- [x] Continue with integrating the plan into our project.
- [x] Create comprehensive test coverage
- [ ] **Phase 7**: Implement prompt caching (see `docs/M_SERIES_ADVANCED_OPTIMIZATIONS.md`)
- [ ] **Phase 8**: Integrate speculative decoding (drafter + worker)
- [ ] **Phase 9**: Implement ANE residency monitoring
- [ ] **Phase 10-14**: Additional optimizations (tokenizer I/O, KV cache, batch policy, energy tracking, memory layout)