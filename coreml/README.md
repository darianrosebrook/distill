# CoreML Runtime and Optimization

CoreML runtime utilities, ANE optimization, and inference acceleration for Apple Silicon.

## Overview

This directory contains:
- **Runtime utilities**: CoreML model loading, generation, and optimization
- **ANE checks**: Validation that models run on Apple Neural Engine
- **Probes**: Parity validation between PyTorch and CoreML
- **Optimizations**: Prompt caching, KV cache optimization, speculative decoding

## Directory Structure

```
coreml/
├── runtime/          # CoreML runtime utilities
│   ├── generate_coreml.py      # Text generation with tool calls
│   ├── prompt_cache.py         # Prompt caching (30-50% TTFT reduction)
│   ├── kv_cache_optimized.py  # ANE-friendly KV cache layout
│   ├── speculative_decode.py  # Speculative decoding (25-40% TTFT improvement)
│   ├── tokenizer_optimized.py  # Optimized tokenizer (10-20% TTFT reduction)
│   ├── batch_policy.py         # Workload-aware batch sizing
│   ├── constrained_decode.py   # JSON-constrained decoding for tool calls
│   └── ane_monitor.py          # ANE residency monitoring
├── probes/           # Parity probes for validation
├── ane_checks.py     # ANE compatibility validation
└── artifacts/        # Compiled CoreML models (gitignored)
```

## Runtime Features

### Prompt Caching
- 30-50% TTFT reduction for repeated system prompts
- Leverages unified memory (no pressure with 64GB)
- Zero quality impact (deterministic caching)

### Speculative Decoding
- 25-40% TTFT improvement
- Uses drafter model for fast token generation
- Verified by worker/judge models

### KV Cache Optimization
- ANE-friendly memory layout
- Unified memory utilization
- Reduced memory bandwidth

### Tokenizer Optimization
- 10-20% TTFT reduction for long prompts
- Optimized batch encoding
- Pre-allocated buffers

## ANE Compatibility

All models are validated for ANE compatibility:
- MLProgram format (required for ANE)
- Enumerated shapes (512/1024/2048)
- INT8 weights + FP16 activations
- ANE operator support

**Check ANE compatibility**:
```bash
python -m coreml.ane_checks --mlpackage coreml/artifacts/worker/model.mlpackage
```

## Parity Validation

Probes validate that CoreML models match PyTorch outputs:

```bash
python -m coreml.probes.compare_probes \
  --pt models/student/checkpoints/latest.pt \
  --ml coreml/artifacts/worker/model.mlpackage
```

## Performance Targets

- **TTFT**: < 500ms for 8k context (with optimizations)
- **Throughput**: > 50 tokens/sec
- **ANE Residency**: > 80% of inference time on ANE

## See Also

- [`conversion/README.md`](../conversion/README.md) - Model conversion documentation
- [`docs/PIPELINE_CRITICAL_PATH_REVIEW.md`](../docs/PIPELINE_CRITICAL_PATH_REVIEW.md) - Complete pipeline review










