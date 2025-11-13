# Deployment Guide

Complete guide for deploying models with advanced features (latent reasoning, halt heads, refinement loops).

## Overview

This guide covers:
- Exporting models with halt head support
- Converting to CoreML with multiple outputs
- Generating runtime configuration files
- Running production inference with advanced features
- Evaluation with latent mode and halt heads

## Quick Start

### 1. Deploy Model with All Artifacts

```bash
# Generate all deployment artifacts
make deploy-full

# Or use the script directly
python -m scripts.deploy_model \
    --checkpoint models/student/checkpoints/latest.pt \
    --out-dir models/student/deployed/ \
    --export-pytorch \
    --export-coreml \
    --latent-mode \
    --halt-head \
    --caws-tier tier_2
```

This generates:
- `model_prefill_fp16.pt` - PyTorch prefill model
- `model_decode_fp16.pt` - PyTorch decode model
- `model.mlpackage` - CoreML model (if requested)
- `runtime_config.json` - Runtime configuration
- `deployment_manifest.json` - Model capabilities manifest

### 2. Run Production Inference

```bash
# Simple inference
python -m scripts.inference_production \
    --model models/student/deployed/model_prefill_fp16.pt \
    --prompt "Your prompt here" \
    --output results.json

# With refinement loops
python -m scripts.inference_production \
    --model models/student/deployed/model_prefill_fp16.pt \
    --prompt "Your prompt here" \
    --use-refinement \
    --config models/student/deployed/runtime_config.json \
    --track-efficiency
```

## Model Export

### PyTorch Export

Models are exported with architecture flags preserved:

```bash
python -m conversion.export_pytorch \
    --checkpoint models/student/checkpoints/latest.pt \
    --out models/student/exported/ \
    --mode both
```

The export process:
1. Loads checkpoint with `model_arch` flags
2. Recreates model with correct architecture (`use_halt_head`, etc.)
3. Exports prefill and decode models
4. Generates contract files with output specifications

### CoreML Conversion

CoreML conversion handles multiple outputs:

```bash
python -m conversion.convert_coreml \
    --backend pytorch \
    --in models/student/exported/student_fp16.pt \
    --out coreml/artifacts/worker/model.mlpackage \
    --contract models/student/exported/student_fp16_contract.json
```

The conversion:
- Reads `use_halt_head` from contract.json
- Preserves both `logits` and `halt_logits` outputs when enabled
- Names outputs correctly for runtime extraction

## Runtime Configuration

### Configuration File Format

Runtime config files are JSON with the following structure:

```json
{
  "latent_mode_enabled": true,
  "halt_head_enabled": true,
  "caws_tier": "tier_2",
  "max_refinement_loops": 5,
  "judge_score_threshold": 0.8,
  "halt_probability_threshold": 0.7,
  "curriculum_probability": 1.0,
  "curriculum_slots": 1,
  "max_latent_spans": 10,
  "max_latent_length": 100,
  "temperature": 1.0,
  "max_new_tokens": 256,
  "enable_efficiency_tracking": true
}
```

### Environment Variables

Runtime config can also be loaded from environment variables:

```bash
export LATENT_MODE=1
export HALT_HEAD=1
export CAWS_TIER=tier_2
export MAX_REFINEMENT_LOOPS=5
export JUDGE_THRESHOLD=0.8
export HALT_THRESHOLD=0.7
```

### CAWS Budget Tiers

| Tier | Max Loops | Max Latent Spans | Use Case |
|------|-----------|------------------|----------|
| Tier 1 | ≤1 | 0 | Strictest budget, no latent reasoning |
| Tier 2 | ≤2 | ≤1 | Balanced budget, minimal latent |
| Tier 3 | ≤3 | ≤3 | More permissive, allows more latent spans |

## Production Inference

### Using InferenceOrchestrator

The `InferenceOrchestrator` coordinates all advanced features:

```python
from runtime.orchestration.inference import create_inference_orchestrator_from_checkpoint
from runtime.config import RuntimeConfig

# Load runtime config
config = RuntimeConfig.from_file("runtime_config.json")

# Create orchestrator
orchestrator = create_inference_orchestrator_from_checkpoint(
    checkpoint_path="model.pt",
    tokenizer=tokenizer,
    latent_mode_enabled=config.latent_mode_enabled,
    halt_head_enabled=config.halt_head_enabled,
    caws_tier=config.caws_tier.value,
)

# Simple generation
output = orchestrator.generate_simple("Your prompt")

# With refinement
result = orchestrator.generate_with_refinement(
    prompt="Your prompt",
    judge_fn=lambda output: 0.9,  # Judge function
)
```

### CoreML Runtime

CoreML models support halt logits extraction:

```python
from coreml.runtime.generate_coreml import generate_tool_call

result = generate_tool_call(
    model=coreml_model,
    tokenizer=tokenizer,
    prompt="Your prompt",
    tools=tools,
    return_halt_logits=True,
)

halt_logits = result.get("halt_logits")
halt_probability = compute_halt_probability(halt_logits)
```

## Evaluation with Advanced Features

### Running Evaluation with Latent Mode

```bash
EVAL_LATENT=1 python -m eval.cli \
    --runner orchestrator \
    --model models/student/checkpoints/latest.pt \
    --in data/contextual_final.jsonl \
    --out eval/results/latent.jsonl \
    --report eval/reports/latent.json \
    --fixtures eval/tool_broker/fixtures
```

### Running Evaluation with Code-Mode

```bash
EVAL_CODE_MODE=1 python -m eval.cli \
    --runner hf_local \
    --model models/student/checkpoints/latest.pt \
    --in data/contextual_final.jsonl \
    --out eval/results/code_mode.jsonl \
    --report eval/reports/code_mode.json \
    --fixtures eval/tool_broker/fixtures
```

### Evaluation Metrics

The evaluation pipeline tracks:
- `latent_spans_used`: Number of latent spans in generation
- `refinement_loops`: Number of refinement iterations
- `halt_logits`: Halt head logits [continue, halt]
- `halt_probability`: Computed halt probability
- `ces_tokens_total`: Context Efficiency Score
- `ces_tokens_code_mode`: Code-mode specific CES

## Deployment Manifest

The deployment manifest (`deployment_manifest.json`) includes:

```json
{
  "model_info": {
    "checkpoint_step": 1000,
    "use_halt_head": true,
    "use_self_evaluation": false,
    "arch": { ... }
  },
  "artifacts": {
    "pytorch_export": "model_prefill_fp16.pt",
    "coreml_export": "model.mlpackage",
    "runtime_config": "runtime_config.json"
  },
  "capabilities": {
    "latent_reasoning": true,
    "halt_head": true,
    "code_mode": true
  }
}
```

## Backward Compatibility

All changes maintain backward compatibility:

- Models without halt heads continue to work
- CoreML conversion handles both single and multi-output models
- Evaluation without runtime configs uses standard generation
- Existing Makefile targets continue to function

## Troubleshooting

### Halt Head Not Working

1. Verify checkpoint has `model_arch.use_halt_head = true`
2. Check export contract includes `use_halt_head` flag
3. Ensure CoreML conversion reads contract correctly
4. Verify runtime config has `halt_head_enabled = true`

### Latent Mode Not Activating

1. Check `LATENT_MODE=1` env var or runtime config
2. Verify model output contains `<bot>` tokens
3. Check curriculum probability > 0
4. Ensure evaluation config loads correctly

### CoreML Multi-Output Issues

1. Verify contract.json includes `halt_logits` in outputs
2. Check CoreML model spec has 2 outputs
3. Ensure output names are `logits` and `halt_logits`
4. Verify conversion logs show both outputs renamed

## See Also

- [`runtime/config.py`](../runtime/config.py) - Runtime configuration system
- [`runtime/orchestration/inference.py`](../runtime/orchestration/inference.py) - Inference orchestrator
- [`conversion/export_pytorch.py`](../conversion/export_pytorch.py) - PyTorch export
- [`conversion/convert_coreml.py`](../conversion/convert_coreml.py) - CoreML conversion
- [`eval/cli.py`](../eval/cli.py) - Evaluation harness

