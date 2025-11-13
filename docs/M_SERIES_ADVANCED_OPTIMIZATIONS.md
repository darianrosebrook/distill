# M-Series Advanced Optimizations

## Overview

This document outlines additional M-series Apple Silicon optimizations beyond the core inference speed optimization plan. These optimizations leverage Apple's unified memory architecture, Neural Engine (ANE), and CoreML capabilities for maximum performance.

## Current State

### ✅ Already Implemented

1. **Hardware Profiles** (`configs/hardware_profiles.yaml`)

   - Chip-specific speed targets and gating
   - Batch policies (interactive vs offline)
   - KV cache precision policies
   - Quantization policies

2. **Enumerated Shapes** (Phase 2)

   - ANE-optimal shapes: 512, 1024, 2048, 4096
   - Dirichlet sampling with production mix

3. **QAT Integration** (Phase 3)

   - INT8 weights + FP16 activations (ANE-optimal)
   - Per-channel quantization

4. **KV Cache Support**

   - `forward_decode()` methods in model architecture
   - FP16 KV cache precision

5. **Speed Metrics** (Phase 4)
   - TTFT/TPS/TTFA measurement
   - TTFT split (tokenizer_ms vs first_step_ms)

### ⚠️ Mentioned but Not Fully Implemented

1. **Speculative Decoding** (Drafter model)

   - Makefile target exists
   - Not integrated into inference pipeline

2. **Prompt Caching**

   - Mentioned in plan
   - Not implemented

3. **ANE Residency Monitoring**
   - Config mentions it
   - No actual measurement code

## Additional Optimizations

### 1. Prompt Caching (High Impact)

**Problem**: System prompts and policy prompts are identical across requests, but we recompute embeddings every time.

**Solution**: Cache prompt embeddings in unified memory between requests.

**Implementation**:

```python
# coreml/runtime/prompt_cache.py
class PromptCache:
    """Cache prompt embeddings for repeated system/policy prompts."""

    def __init__(self, model, max_cache_size_mb: int = 100):
        self.model = model
        self.cache: Dict[str, torch.Tensor] = {}
        self.max_size_bytes = max_cache_size_mb * 1024 * 1024

    def get_or_compute(self, prompt: str, prompt_hash: str) -> torch.Tensor:
        """Get cached embedding or compute and cache."""
        if prompt_hash in self.cache:
            return self.cache[prompt_hash]

        # Compute embedding
        embedding = self.model.encode_prompt(prompt)

        # Cache if under limit
        if self._cache_size() + embedding.numel() * 2 < self.max_size_bytes:
            self.cache[prompt_hash] = embedding

        return embedding

    def _cache_size(self) -> int:
        """Return cache size in bytes."""
        return sum(t.numel() * t.element_size() for t in self.cache.values())
```

**Benefits**:

- **30-50% TTFT reduction** for repeated system prompts
- Leverages unified memory (no memory pressure with 64GB)
- Zero quality impact (deterministic caching)

**Acceptance**:

- Repeated runs with identical system prompt reduce TTFT by ≥30%
- Cache hit rate ≥95% for system prompts
- No memory pressure (unified memory advantage)

**Files**:

- `coreml/runtime/prompt_cache.py` (new)
- `evaluation/perf_mem_eval.py` (integrate cache)
- `eval/cli.py` (optional cache flag)

---

### 2. Speculative Decoding Integration (High Impact)

**Problem**: TTFT is dominated by first token generation. We can use a smaller drafter model to generate tokens faster, then verify with the worker.

**Solution**: Integrate drafter model for speculative decoding.

**Implementation**:

```python
# coreml/runtime/speculative_decode.py
class SpeculativeDecoder:
    """Speculative decoding with drafter + worker verification."""

    def __init__(self, drafter_model, worker_model, k: int = 2):
        self.drafter = drafter_model  # ~4B model
        self.worker = worker_model    # ~9B model
        self.k = k  # Draft tokens per step

    def generate(self, prompt_ids: torch.Tensor, max_tokens: int) -> List[int]:
        """Generate with speculative decoding."""
        tokens = []
        kv_cache_worker = None

        while len(tokens) < max_tokens:
            # Drafter generates K tokens
            draft_tokens = self._draft_k_tokens(prompt_ids + tokens, k=self.k)

            # Worker verifies/accepts
            accepted = self._verify_tokens(draft_tokens, kv_cache_worker)
            tokens.extend(accepted)

            # If all rejected, generate one token normally
            if len(accepted) == 0:
                next_token = self._generate_one_token(prompt_ids + tokens, kv_cache_worker)
                tokens.append(next_token)

        return tokens

    def _draft_k_tokens(self, input_ids: torch.Tensor, k: int) -> List[int]:
        """Draft K tokens using smaller model."""
        # Fast generation with drafter
        pass

    def _verify_tokens(self, draft_tokens: List[int], kv_cache) -> List[int]:
        """Verify draft tokens with worker model."""
        # Accept/reject logic
        pass
```

**Benefits**:

- **25-40% TTFT improvement** (drafter is 2-3x faster)
- **10-20% TPS improvement** (fewer worker forward passes)
- Rollback rate ≤10% (acceptable)

**Acceptance**:

- TTFT improves ≥25% vs baseline
- p95 rollback rate ≤10%
- No CAWS gate regressions
- Drafter TTFT ≤120-200ms (vs worker 300-500ms)

**Files**:

- `coreml/runtime/speculative_decode.py` (new)
- `evaluation/perf_mem_eval.py` (integrate decoder)
- `configs/drafter_4b.yaml` (ensure exists)

---

### 3. ANE Residency Monitoring (Critical for Validation)

**Problem**: We can't hard-force ANE placement, but we need to verify it's working.

**Solution**: Measure ANE residency empirically using Instruments or wall-clock sampling.

**Implementation**:

```python
# coreml/runtime/ane_monitor.py
class ANEResidencyMonitor:
    """Monitor ANE residency during inference."""

    def __init__(self, model_mlpackage_path: str):
        self.model_path = model_mlpackage_path
        self.samples = []

    def measure_residency(self, num_samples: int = 100) -> Dict[str, float]:
        """
        Measure ANE residency by sampling wall-clock times.

        Uses Instruments → Core ML template if available,
        otherwise falls back to wall-clock sampling.
        """
        # Option 1: Use Instruments (if available)
        if self._instruments_available():
            return self._measure_with_instruments(num_samples)

        # Option 2: Wall-clock sampling (fallback)
        return self._measure_wall_clock(num_samples)

    def _measure_with_instruments(self, num_samples: int) -> Dict[str, float]:
        """Use Instruments → Core ML template."""
        # Requires Instruments.app integration
        # Returns: {"ane_time_pct": 0.85, "gpu_time_pct": 0.10, "cpu_time_pct": 0.05}
        pass

    def _measure_wall_clock(self, num_samples: int) -> Dict[str, float]:
        """Fallback: wall-clock sampling (less accurate)."""
        # Sample inference times, estimate ANE vs GPU/CPU
        # Less accurate but works without Instruments
        pass
```

**Benefits**:

- **Validation**: Ensure ANE is actually being used
- **CI gates**: Fail if ANE residency drops >10% vs baseline
- **Debugging**: Identify ops that force CPU fallback

**Acceptance**:

- ANE residency ≥80% for attention/MLP ops
- Fail CI if residency drops >10% vs baseline
- Report residency in speed report header

**Files**:

- `coreml/runtime/ane_monitor.py` (new)
- `evaluation/perf_mem_eval.py` (integrate monitor)
- `eval/scoring/scorer.py` (add residency gate)

---

### 4. Tokenizer I/O Optimization (Medium Impact)

**Problem**: Tokenizer I/O can dominate TTFT, especially for long prompts.

**Solution**: Pre-allocate CoreML I/O tensors and use ring buffers.

**Implementation**:

```python
# coreml/runtime/tokenizer_optimized.py
class OptimizedTokenizer:
    """Tokenizer with pre-allocated buffers and ring buffers."""

    def __init__(self, tokenizer, max_seq_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        # Pre-allocate CoreML I/O tensors
        self.input_buffer = torch.zeros(max_seq_length, dtype=torch.long)
        self.output_buffer = torch.zeros(max_seq_length, dtype=torch.long)

        # Ring buffer for streaming tokens
        self.ring_buffer = RingBuffer(size=max_seq_length)

    def encode_streaming(self, text: str) -> Iterator[int]:
        """Encode with streaming (ring buffer)."""
        # Stream tokens into ring buffer
        # Reduces memory allocations
        pass

    def encode_batch_optimized(self, texts: List[str]) -> torch.Tensor:
        """Encode batch with pre-allocated buffers."""
        # Use pre-allocated buffers
        # Avoid GC pressure
        pass
```

**Benefits**:

- **10-20% TTFT reduction** for long prompts
- Reduced GC pressure (no alloc spikes)
- Better memory locality

**Acceptance**:

- GC pressure flat (no alloc spikes in Instruments)
- TTFT reduction ≥10% for prompts >1k tokens
- No quality impact

**Files**:

- `coreml/runtime/tokenizer_optimized.py` (new)
- `evaluation/perf_mem_eval.py` (use optimized tokenizer)

---

### 5. KV Cache Optimization (Medium Impact)

**Problem**: KV cache size grows with sequence length. We can optimize for unified memory.

**Solution**: Optimize KV cache layout and sizing for ANE efficiency.

**Implementation**:

```python
# coreml/runtime/kv_cache_optimized.py
class OptimizedKVCache:
    """KV cache optimized for ANE and unified memory."""

    def __init__(self, n_heads: int, head_dim: int, max_seq_len: int, precision: str = "fp16"):
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.precision = precision

        # Pre-allocate KV cache (unified memory advantage)
        element_size = 2 if precision == "fp16" else 4
        kv_size_bytes = n_heads * head_dim * max_seq_len * 2 * element_size  # K+V

        # Use unified memory (no pressure with 64GB)
        self.k_cache = torch.zeros(n_heads, max_seq_len, head_dim, dtype=self._dtype())
        self.v_cache = torch.zeros(n_heads, max_seq_len, head_dim, dtype=self._dtype())

    def update(self, k: torch.Tensor, v: torch.Tensor, position: int):
        """Update cache at position (in-place for efficiency)."""
        # In-place update (ANE-friendly)
        self.k_cache[:, position:position+1, :] = k
        self.v_cache[:, position:position+1, :] = v

    def get_slice(self, start: int, end: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get KV cache slice (ANE-optimal layout)."""
        # Return contiguous slice (ANE-friendly)
        return self.k_cache[:, start:end, :], self.v_cache[:, start:end, :]
```

**Benefits**:

- **Reduced memory allocations** (pre-allocated cache)
- **ANE-friendly layout** (contiguous, aligned)
- **Unified memory advantage** (no pressure with 64GB)

**Acceptance**:

- KV cache size logged: `(heads, head_dim, seq_len) → bytes`
- No memory pressure (unified memory)
- ANE-friendly layout verified

**Files**:

- `coreml/runtime/kv_cache_optimized.py` (new)
- `models/student/architectures/gqa_transformer.py` (integrate)

---

### 6. Energy Efficiency Tracking (Low Priority)

**Problem**: Energy efficiency matters for laptops. We should track it.

**Solution**: Track energy per token using Instruments Energy Log.

**Implementation**:

```python
# coreml/runtime/energy_tracker.py
class EnergyTracker:
    """Track energy consumption during inference."""

    def measure_energy_per_token(self, num_tokens: int = 100) -> Dict[str, float]:
        """
        Measure energy per token using Instruments Energy Log.

        Returns:
            {
                "energy_per_100_tokens_j": 2.5,  # Joules per 100 tokens
                "power_watts": 15.0,  # Average power during inference
            }
        """
        # Requires Instruments.app integration
        # Or use system power APIs (less accurate)
        pass
```

**Benefits**:

- **Battery life optimization** (important for laptops)
- **Power efficiency metrics** (J/100 tokens)
- **Thermal management** (identify hot spots)

**Acceptance**:

- Energy per 100 tokens logged (optional metric)
- Power consumption tracked
- No thermal throttling during inference

**Files**:

- `coreml/runtime/energy_tracker.py` (new)
- `evaluation/perf_mem_eval.py` (optional integration)

---

### 7. Batch Policy Enforcement (Medium Impact)

**Problem**: Batch size affects latency vs throughput trade-off. We need automatic selection.

**Solution**: Enforce batch policy based on workload type (interactive vs offline).

**Implementation**:

```python
# coreml/runtime/batch_policy.py
class BatchPolicy:
    """Enforce batch policy based on workload type."""

    def __init__(self, hardware_profile: HWProfile):
        self.profile = hardware_profile
        self.interactive_batch = hardware_profile.config["batch_policy"]["interactive_default"]
        self.offline_batches = hardware_profile.config["batch_policy"]["offline_allowed"]

    def select_batch_size(self, workload_type: str = "interactive") -> int:
        """
        Select batch size based on workload type.

        Args:
            workload_type: "interactive" or "offline"

        Returns:
            Optimal batch size
        """
        if workload_type == "interactive":
            return self.interactive_batch  # Always 1 for interactive
        else:
            # Offline: use larger batch if it improves TPS without hurting p95 latency
            return self._optimize_offline_batch()

    def _optimize_offline_batch(self) -> int:
        """Optimize batch size for offline workloads."""
        # Test batches 2-4, select highest TPS with <10% p95 latency penalty
        # Default: return first allowed batch size
        return self.offline_batches[0] if self.offline_batches else 1
```

**Benefits**:

- **Automatic optimization** (no manual tuning)
- **Workload-aware** (interactive vs offline)
- **Hardware-specific** (uses hardware profile)

**Acceptance**:

- Interactive always uses batch=1
- Offline uses batch 2-4 if TPS improves ≥10% with <10% p95 latency penalty
- Policy enforced automatically

**Files**:

- `coreml/runtime/batch_policy.py` (new)
- `evaluation/perf_mem_eval.py` (integrate policy)
- `eval/cli.py` (add workload_type flag)

---

### 8. Memory Layout Optimization (Low Priority)

**Problem**: Tensor layout affects ANE efficiency.

**Solution**: Ensure tensors are ANE-aligned (contiguous, proper dtype).

**Implementation**:

```python
# coreml/runtime/memory_layout.py
def ensure_ane_aligned(tensor: torch.Tensor) -> torch.Tensor:
    """
    Ensure tensor is ANE-aligned (contiguous, proper dtype).

    ANE requirements:
    - Contiguous memory layout
    - FP16 or INT8 dtype
    - Proper alignment (16-byte aligned)
    """
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    # Ensure proper dtype (FP16 for activations, INT8 for weights)
    if tensor.dtype not in [torch.float16, torch.int8]:
        tensor = tensor.to(torch.float16)

    return tensor
```

**Benefits**:

- **ANE efficiency** (proper layout)
- **Reduced memory transfers** (contiguous)
- **Better fusion** (ANE can fuse operations)

**Acceptance**:

- All tensors verified ANE-aligned
- No layout-related performance regressions

**Files**:

- `coreml/runtime/memory_layout.py` (new)
- `conversion/convert_coreml.py` (verify during export)

---

## Implementation Priority

### Phase 7: Prompt Caching (High Priority)

**Impact**: 30-50% TTFT reduction for repeated prompts  
**Effort**: Medium  
**Risk**: Low (deterministic caching)

**Files**:

- `coreml/runtime/prompt_cache.py` (new)
- `evaluation/perf_mem_eval.py` (integrate)
- `eval/cli.py` (add cache flag)

### Phase 8: Speculative Decoding Integration (High Priority)

**Impact**: 25-40% TTFT improvement  
**Effort**: High  
**Risk**: Medium (rollback logic complexity)

**Files**:

- `coreml/runtime/speculative_decode.py` (new)
- `evaluation/perf_mem_eval.py` (integrate)
- `configs/drafter_4b.yaml` (verify exists)

### Phase 9: ANE Residency Monitoring (Critical for Validation)

**Impact**: Validation and debugging  
**Effort**: Medium  
**Risk**: Low (measurement only)

**Files**:

- `coreml/runtime/ane_monitor.py` (new)
- `evaluation/perf_mem_eval.py` (integrate)
- `eval/scoring/scorer.py` (add residency gate)

### Phase 10: Tokenizer I/O Optimization (Medium Priority)

**Impact**: 10-20% TTFT reduction for long prompts  
**Effort**: Low  
**Risk**: Low

**Files**:

- `coreml/runtime/tokenizer_optimized.py` (new)
- `evaluation/perf_mem_eval.py` (integrate)

### Phase 11: KV Cache Optimization (Medium Priority)

**Impact**: Reduced allocations, ANE efficiency  
**Effort**: Medium  
**Risk**: Low

**Files**:

- `coreml/runtime/kv_cache_optimized.py` (new)
- `models/student/architectures/gqa_transformer.py` (integrate)

### Phase 12: Batch Policy Enforcement (Medium Priority)

**Impact**: Automatic optimization  
**Effort**: Low  
**Risk**: Low

**Files**:

- `coreml/runtime/batch_policy.py` (new)
- `evaluation/perf_mem_eval.py` (integrate)

### Phase 13: Energy Tracking (Low Priority)

**Impact**: Battery life optimization  
**Effort**: Medium  
**Risk**: Low

**Files**:

- `coreml/runtime/energy_tracker.py` (new)
- `evaluation/perf_mem_eval.py` (optional integration)

### Phase 14: Memory Layout Optimization (Low Priority)

**Impact**: ANE efficiency  
**Effort**: Low  
**Risk**: Low

**Files**:

- `coreml/runtime/memory_layout.py` (new)
- `conversion/convert_coreml.py` (verify during export)

---

## Integration with Existing Plan

These optimizations complement the existing inference speed optimization plan:

- **Phase 1-6**: Core optimizations (already implemented)
- **Phase 7-14**: Advanced optimizations (this document)

**Order of Implementation**:

1. Phase 7 (Prompt Caching) - Quick win
2. Phase 9 (ANE Residency) - Critical for validation
3. Phase 8 (Speculative Decoding) - High impact
4. Phase 10-14 (Remaining optimizations)

---

## Acceptance Criteria

### Prompt Caching

- [ ] Repeated runs with identical system prompt reduce TTFT by ≥30%
- [ ] Cache hit rate ≥95% for system prompts
- [ ] No memory pressure (unified memory advantage)

### Speculative Decoding

- [ ] TTFT improves ≥25% vs baseline
- [ ] p95 rollback rate ≤10%
- [ ] No CAWS gate regressions
- [ ] Drafter TTFT ≤120-200ms

### ANE Residency Monitoring

- [ ] ANE residency ≥80% for attention/MLP ops
- [ ] Fail CI if residency drops >10% vs baseline
- [ ] Report residency in speed report header

### Tokenizer I/O Optimization

- [ ] GC pressure flat (no alloc spikes)
- [ ] TTFT reduction ≥10% for prompts >1k tokens

### KV Cache Optimization

- [ ] KV cache size logged
- [ ] No memory pressure
- [ ] ANE-friendly layout verified

### Batch Policy Enforcement

- [ ] Interactive always uses batch=1
- [ ] Offline uses batch 2-4 if TPS improves ≥10% with <10% p95 latency penalty

---

## References

- Plan: `.cursor/plans/inference-speed-optimization-during-distillation-c3d3cffc.plan.md`
- M-Series Guide: `docs/M_SERIES_OPTIMIZATION_GUIDE.md`
- Hardware Profiles: `configs/hardware_profiles.yaml`
- CoreML Export: `configs/convert_coreml.yaml`
