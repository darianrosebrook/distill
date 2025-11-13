# Runtime

**Why it exists**: Enables CAWS-compliant model inference with tool integration, budget enforcement, and multi-step reasoning - the operational runtime that makes distilled models actually useful.

**What's in it**: Core inference engine, refinement controllers, and orchestration utilities that handle the complex dance of model generation + tool execution + CAWS compliance.

**Key Features**:

- Inference Engine: Handles multi-turn conversations with tool integration
- Refinement Controllers: Enforces CAWS Tier 1/2/3 budgets and halt detection
- Multi-step Reasoning: Supports latent reasoning spans and progressive refinement
- Performance Monitoring: Built-in metrics collection and latency optimization

## Overview

The runtime directory bridges the gap between trained models and operational deployment, implementing the CAWS-compliant inference patterns that make tool-integrated reasoning possible.

## Directory Structure

```
runtime/
├── api_contract.py      # API contract definitions
├── config.py           # Runtime configuration
├── engine/             # Inference engine components
│   └── loop.py        # Main inference loop
└── orchestration/      # Orchestration utilities
    ├── inference.py   # Inference orchestration
    └── refine.py     # Refinement controller
```

## Key Components

### Inference Engine (`engine/`)

**Main inference loop** with support for:

- Multi-turn conversations
- Tool integration and execution
- Latent reasoning spans
- CAWS budget enforcement

**Usage**:

```python
from runtime.engine.loop import InferenceEngine

engine = InferenceEngine(model=model, tokenizer=tokenizer)
result = engine.generate(
    prompt="Process this data",
    max_steps=10,
    caws_budget_tier=CAWSBudgetTier.TIER_2
)
```

### Refinement Controller (`orchestration/refine.py`)

**CAWS-compliant refinement** with:

- Halt head integration
- Budget enforcement (Tier 1/2/3)
- Progressive reasoning
- Tool call validation

**Features**:

- **Halt Detection**: Learns when to stop refinement
- **Budget Limits**: Enforces CAWS tier constraints
- **Error Recovery**: Graceful handling of failures
- **Metrics Collection**: Performance monitoring

### API Contracts (`api_contract.py`)

**Standardized interfaces** for:

- Model inference APIs
- Tool execution contracts
- Response validation
- Error handling

## Runtime Configuration

Runtime behavior is configured via:

```python
from runtime.config import RuntimeConfig

config = RuntimeConfig(
    max_latent_spans=3,      # TIER_3 limit
    max_refinement_loops=3,  # TIER_3 limit
    enable_halt_head=True,   # Use halt head for early stopping
    tool_timeout_ms=5000,    # Tool execution timeout
)
```

## Integration Points

### With Training

- **Halt Head Training**: `training/halt_targets.py`
- **Latent Curriculum**: `training/dataset.py`
- **CAWS Compliance**: `training/caws_context.py`

### With Evaluation

- **Performance Monitoring**: `evaluation/perf_mem_eval.py`
- **Tool Integration**: `eval/tool_broker/`
- **CAWS Validation**: `evaluation/caws_eval.py`

## See Also

- [`coreml/runtime/`](../coreml/runtime/) - CoreML-specific runtime
- [`docs/DEPLOYMENT.md`](../docs/DEPLOYMENT.md) - Deployment configuration
- [`training/halt_targets.py`](../training/halt_targets.py) - Halt head training
- [`eval/tool_broker/`](../eval/tool_broker/) - Tool integration
