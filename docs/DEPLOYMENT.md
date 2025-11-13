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

## Environment Requirements

### Python Version Requirements

**Critical**: Export and conversion steps require Python 3.10 or 3.11.

- **Training**: Python 3.13+ is supported
- **Export/Conversion**: Python 3.10 or 3.11 required (version gates enforce this)
- **Toy Models**: Can bypass version checks with `--toy` flag

#### Installing Python 3.11 (macOS)

```bash
# Using Homebrew
brew install python@3.11

# Verify installation
python3.11 --version

# Use for export/conversion steps
python3.11 -m conversion.export_pytorch --checkpoint model.pt --out exported/
```

#### Multi-Python Environment Strategy

The project uses different Python versions for different steps:

1. **Training**: Use default Python (3.13+)
   ```bash
   python -m training.run_distill --checkpoint model.pt
   ```

2. **Export/Conversion**: Use Python 3.11 explicitly
   ```bash
   python3.11 -m conversion.export_pytorch --checkpoint model.pt --out exported/
   python3.11 -m conversion.convert_coreml --backend pytorch --in exported/model.pt --out model.mlpackage
   ```

3. **Toy Models**: Can use any Python version with `--toy` flag
   ```bash
   python -m conversion.export_pytorch --checkpoint toy.pt --out exported/ --toy
   ```

### PyTorch Version Compatibility

**Current Status**: PyTorch 2.9.1 shows compatibility warnings with coremltools but conversion still works.

- **PyTorch**: 2.9.1+ (training and export)
- **coremltools**: Latest version (conversion)
- **Known Issue**: Version mismatch warnings are non-fatal but should be monitored

#### Compatibility Notes

- PyTorch 2.9.1 has not been fully tested with coremltools
- Warnings appear during conversion but do not block functionality
- Monitor conversion logs for any actual errors vs warnings
- Consider pinning PyTorch version if issues arise: `pip install torch==2.8.0`

## Model Export

### PyTorch Export

Models are exported with architecture flags preserved:

```bash
python3.11 -m conversion.export_pytorch \
    --checkpoint models/student/checkpoints/latest.pt \
    --out models/student/exported/ \
    --mode both
```

The export process:
1. Loads checkpoint with `model_arch` flags
2. Recreates model with correct architecture (`use_halt_head`, etc.)
3. Exports prefill and decode models
4. Generates contract files with output specifications
5. Validates enumerated shapes (T64, T128, T256, T512, T1024, T2048, T4096)

### Shape Enumeration Validation

Export includes shape enumeration to verify model works across different sequence lengths:

- **Toy Models**: T64, T128, T256 (T128 is primary)
- **Production Models**: T512, T1024, T2048, T4096 (T1024 is primary)

**Known Issues**:
- Some shapes may fail with RuntimeError (e.g., T64/T256 in toy models)
- Primary shape (T128 for toy, T1024 for production) should always work
- Shape failures are logged but don't block export if primary shape succeeds

**Verification**:
```bash
# Check contract file for shape validation results
cat models/student/exported/model_contract.json | jq '.shape_validation'
```

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

### Python Version Errors

**Error**: `Version check failed: Python 3.13 detected. This project requires Python 3.10 or 3.11.`

**Solution**:
1. Install Python 3.11: `brew install python@3.11`
2. Use Python 3.11 for export/conversion: `python3.11 -m conversion.export_pytorch ...`
3. For toy models, use `--toy` flag to bypass: `python -m conversion.export_pytorch --toy ...`

**Error**: `python3.11: command not found`

**Solution**:
- Verify installation: `brew list python@3.11`
- Check PATH: `which python3.11`
- Reinstall if needed: `brew reinstall python@3.11`

### PyTorch Version Compatibility Warnings

**Warning**: `Torch version 2.9.1 has not been tested with coremltools.`

**Solution**:
- This is a non-fatal warning - conversion should still work
- Monitor conversion logs for actual errors (not warnings)
- If conversion fails, consider downgrading: `pip install torch==2.8.0`
- Check coremltools version: `pip show coremltools`

### Shape Enumeration Failures

**Error**: `RuntimeError` during shape validation for T64/T256

**Solution**:
- Primary shapes (T128 for toy, T1024 for production) should always work
- Secondary shape failures are logged but don't block export
- Check contract file for which shapes succeeded: `jq '.shape_validation' contract.json`
- If primary shape fails, investigate model architecture or export configuration

**Verification**:
```bash
# Check which shapes validated successfully
python -c "import json; print(json.load(open('contract.json'))['shape_validation'])"
```

### CoreML Conversion Errors

**Error**: `UnboundLocalError: cannot access local variable 'torch'`

**Solution**:
- This bug was fixed - ensure you have latest code
- If still occurring, check `conversion/convert_coreml.py` has `import torch` inside function
- Verify PyTorch is installed: `python -c "import torch; print(torch.__version__)"`

**Error**: CoreML conversion fails silently

**Solution**:
1. Check conversion logs for detailed error messages
2. Verify PyTorch model loads correctly: `torch.jit.load('model.pt')`
3. Check contract file exists and is valid JSON
4. Verify coremltools installation: `python -c "import coremltools; print(coremltools.__version__)"`
5. Try with `--allow-placeholder` flag to see if it's a non-critical issue

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

### Version Gate Strategy

**Understanding Version Gates**:

- **Production Models**: Version gates enforce Python 3.10/3.11 for export/conversion
- **Toy Models**: Can bypass gates with `--toy` flag for faster testing
- **Purpose**: Ensure compatibility with TorchScript export and CoreML conversion

**When to Bypass**:
- Only for toy/test models
- Never bypass for production models
- Use `--toy` flag explicitly when testing pipeline with toy checkpoints

## See Also

- [`runtime/config.py`](../runtime/config.py) - Runtime configuration system
- [`runtime/orchestration/inference.py`](../runtime/orchestration/inference.py) - Inference orchestrator
- [`conversion/export_pytorch.py`](../conversion/export_pytorch.py) - PyTorch export
- [`conversion/convert_coreml.py`](../conversion/convert_coreml.py) - CoreML conversion
- [`eval/cli.py`](../eval/cli.py) - Evaluation harness

