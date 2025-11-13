# M-Series Mac Optimization Guide

## Overview

This guide documents M-series Mac-specific optimizations for inference speed during distillation. Target hardware: **M1 Max 64GB MacBook Pro** and newer Pro-level chips (M2 Pro, M3 Pro).

**Important**: All speed gates are **relative** (≤5% regression vs. last blessed baseline on the same hardware), not absolute targets. Use hardware profiles (`configs/hardware_profiles.yaml`) for chip-specific expectations.

## Key M-Series Characteristics

### Apple Neural Engine (ANE)

- **Optimal shapes**: 512, 1024, 2048 tokens (highest throughput/latency ratio)
- **Supported**: 4096 tokens (long-context, slightly worse TPS due to tiling)
- **Avoid**: >4096 tokens (ANE efficiency drops significantly, prefer chunking)
- **Precision**: INT8 weights + FP16 activations (ANE's sweet spot)
- **Batch size**: **batch=1** for interactive inference (default); batch 2-4 can improve TPS but hurts p95 latency (tail tokens wait)

### Unified Memory Architecture

- **64GB unified memory**: Use to **avoid** memory pressure (larger KV cache, longer prompts), not to justify large interactive batches
- **Memory bandwidth**: High bandwidth enables larger KV caches
- **Training**: Can use larger micro-batches (4-8) during training
- **Inference**: Optimize for ANE efficiency (batch=1 interactive, batch>1 only for offline workloads)

## Plan Adjustments

### Phase 2: Enumerated-Shape Training

**Changes:**

1. **Shape selection**: Focus on ANE-optimal shapes `[512, 1024, 2048]` with optional `4096`
2. **Production mix**: `[0.4, 0.4, 0.15, 0.05]` for `[512, 1024, 2048, 4096]`
   - 512/1024: Most common (tool calls, short reasoning)
   - 2048: Longer reasoning
   - 4096: Rare long-context cases
3. **ANE validation**: Ensure all attention ops run on ANE (0 CPU fallbacks)

**Config updates:**

```yaml
train:
  seq_lengths: [512, 1024, 2048, 4096] # M-series ANE-optimal
  shape_probs: [0.4, 0.4, 0.15, 0.05] # Production mix
```

### Phase 3: QAT Optimization

**M-series specific:**

1. **INT8 priority**: ANE's sweet spot is INT8 weights + FP16 activations
2. **KV cache**: FP16 for M1 Max 64GB (can use larger cache with unified memory)
3. **Attention pattern**: Ensure INT8 weights → FP16 activations (ANE-optimal)
4. **Batch size**: ANE prefers 1-4 for inference, but training can use larger batches

**Config updates:**

```yaml
quant:
  enabled: true
  start_fraction: 0.8 # Last 20% of training
  weight_bits: 8 # INT8 (ANE-optimized)
  act_bits: 16 # FP16 activations
  kv_cache_precision: fp16 # M1 Max 64GB can handle larger FP16 cache
```

### Phase 6: Hardware-Specific Measurement

**Realistic performance bands** (for ~9B Worker, INT8-W/FP16-A, batch=1, good export):

- **M1 Max 64GB** (baseline):
  - TTFT p50: **300-500 ms** (context-dependent, ~2k tokens)
  - TTFT p95: **450-800 ms**
  - TPS p50: **25-35 tok/s** (2k context), **30-40 tok/s** (≤1k context)
  - TPS p95: **20-30 tok/s**
  - TTFA p95: **≤25 tokens** (absolute correctness gate)
- **M2/M3 Pro**: Similar or slightly better (see `configs/hardware_profiles.yaml`)

**Important**: These are **acceptance bands** for sanity checks. **CI gates are relative** (≤5% regression vs. last blessed report on same hardware profile).

**Hardware detection:**

- Auto-detect via `sysctl -n machdep.cpu.brand_string` (via `eval/hw_profile.py`)
- Match to hardware profile in `configs/hardware_profiles.yaml`
- **Never compare speed across different hardware profiles** (unless `--allow-cross-hw`)
- Use hardware profile's `speed_targets` for sanity-check ranges

## Configuration Changes

### `configs/student_9b_gqa.yaml`

```yaml
train:
  # M-series ANE-optimal shapes
  seq_lengths: [512, 1024, 2048, 4096]
  shape_probs: [0.4, 0.4, 0.15, 0.05] # Production mix
  micro_batch_size: 4 # ANE prefers smaller batches, but 64GB allows this
```

### `configs/convert_coreml.yaml`

```yaml
coreml:
  enumerate_shapes: [512, 1024, 2048, 4096] # M-series optimized
  mlprogram: true
  compute_units: all # Prefer ANE, fall back GPU

# Note: CoreML decides device placement at runtime; you cannot hard-force "must_place"
# Instead:
# 1. Export with mlprogram and enumerated shapes
# 2. Avoid ops that force CPU (int64 tensors, unsupported activations)
# 3. Verify ANE placement empirically (Instruments → Core ML template)
# 4. Fail CI if ANE residency drops >10% vs baseline (measure via profiling)
```

## Validation Checklist

- [ ] Exported **mlprogram** with enumerated shapes matches training shapes
- [ ] **No int64** leaks; LayerNorm/softmax in FP16
- [ ] **ANE residency** empirically good (attention/MLP on ANE; no large CPU hot spots)
  - Use Instruments → Core ML template to verify
  - Fail CI if ANE residency drops >10% vs baseline
- [ ] **Per-shape drift** ≤ 0.01 F1; QAT cosine ≥ 0.999; no NaNs
- [ ] **Speed gates** (TTFT/TPS p50 & p95) do **not** regress >5% vs. last-blessed on same `hardware.soc`
- [ ] **TTFA p95 tokens** ≤ 25 with early-tool loss enabled
- [ ] **Batch policy** documented: interactive batch=1; batch>1 only for offline tasks
- [ ] **TTFT split** recorded: `{tokenizer_ms, first_step_ms}` for diagnostics
- [ ] **KV size budget** logged: `(heads, head_dim, seq_len)` → KV bytes

## Troubleshooting

### ANE Not Utilized

- Check `compute_units="all"` in export config
- **Verify empirically**: Use Instruments → Core ML template to check ANE vs GPU time
- Ensure INT8 weights + FP16 activations pattern
- Validate shapes are ANE-optimal (512/1024/2048)
- Avoid ops that force CPU: no int64 tensors, no unsupported activations
- **Measure ANE residency**: Sample wall-times during inference; fail if <80% steps on ANE

### Poor Performance on M1 Max

- Validate shapes are ANE-optimal (512/1024/2048 primary)
- **Use batch=1** for interactive inference (batch 2-4 helps TPS but hurts p95 latency)
- Verify KV cache precision (FP16 for M1 Max)
- Check TTFT split: if `tokenizer_ms` dominates, optimize tokenizer; if `first_step_ms` dominates, check ANE placement
- **Use relative gates**: Don't chase absolute targets; gate on ≤5% regression vs baseline

### Shape Mismatch

- Ensure training shapes match export shapes exactly
- ANE is sensitive to shape changes
- Use enumerated shapes during training
- Add periodic upweight for rare shapes (e.g., 4096 every N steps) to prevent drift

### Speed Regression

- **Check hardware profile**: Ensure comparing against same `hardware.soc` (use `eval/hw_profile.py`)
- Use **relative gates** (≤5% regression), not absolute targets
- Split TTFT: `tokenizer_ms` vs `first_step_ms` to diagnose
- Check ANE residency: if dropped, investigate export changes
- Verify batch size: interactive should be batch=1

## Hardware Profiles

Use `configs/hardware_profiles.yaml` for chip-specific configuration:

- **Speed targets**: Realistic acceptance bands per chip tier
- **Shape mix**: Production distribution per hardware
- **Gating**: Relative regression limits (≤5% default)
- **Batch policy**: Interactive vs offline defaults

Load profiles via `eval/hw_profile.py`:

```python
from eval.hw_profile import load_profiles, match_profile
from pathlib import Path

profiles = load_profiles(Path("configs/hardware_profiles.yaml"))
current = match_profile(profiles)  # HWProfile with chip-specific config
```

## Scorer Integration

In your evaluator/scorer, use hardware profiles for relative gating:

```python
from eval.hw_profile import load_profiles, match_profile, require_same_profile
from pathlib import Path

# Load hardware profiles
profiles = load_profiles(Path("configs/hardware_profiles.yaml"))
current = match_profile(profiles)  # HWProfile

# Add hardware profile key to report
report_header["hardware_profile_key"] = current.key

# When comparing to blessed baseline
baseline_key = blessed_report.get("hardware_profile_key") or blessed_report.get("header", {}).get("hardware_profile_key")
if gates_cfg.get("require_same_profile", current.config["gating"].get("require_same_profile", True)):
    require_same_profile(current.key, baseline_key)

# Relative gates (from hardware profile)
limits = current.config["gating"]["max_regression_pct"]
# Fail if TTFT/TPS regresses >5% vs baseline (same hardware)
# TTFT: higher is worse; TPS: lower is worse
```

## Additional Diagnostics

### TTFT Split

Record TTFT as `{tokenizer_ms, first_step_ms}` for diagnostics:
[evidence: evaluation/perf_mem_eval.json#perf.ttft_ms]

- If `tokenizer_ms` dominates: optimize tokenizer
- If `first_step_ms` dominates: check ANE placement, model export

### KV Size Budget

Log KV cache size for memory pressure analysis:

```python
kv_bytes = (n_heads * head_dim * seq_len * 2) * 2  # FP16 = 2 bytes, 2 for K+V
# Log: {"kv_size_bytes": kv_bytes, "seq_len": seq_len, "heads": n_heads}
```

This helps reason about unified-memory pressure when raising 4096 usage.
[evidence: evaluation/perf_mem_eval.json#perf.memory_usage]

## References

- Plan: `.cursor/plans/inference-speed-optimization-during-distillation-c3d3cffc.plan.md`
- Hardware profiles: `configs/hardware_profiles.yaml`
- Profile loader: `eval/hw_profile.py`
- CoreML export: `configs/convert_coreml.yaml`
- Training config: `configs/student_9b_gqa.yaml`
- Speed harness: `evaluation/perf_mem_eval.py`
