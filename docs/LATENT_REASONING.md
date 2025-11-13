# Latent Reasoning + Outer-Loop Refinement

## Overview

Latent reasoning enables the model to process hidden states without generating tokens, reducing token count while maintaining reasoning capability. This is achieved through:

- **Latent spans**: Marked with `<bot>` and `<eot>` sentinel tokens
- **Learned halting**: Halt head predicts when to stop refinement loops
- **Outer-loop refinement**: Multiple refinement passes with CAWS-budgeted limits
- **Progressive curriculum**: Training replaces CoT steps with latent slots

## Architecture Decision Record (ADR)

### Decision: Latent Spans vs Explicit CoT

**Context**: Traditional chain-of-thought (CoT) reasoning generates explicit reasoning tokens, increasing token count and latency. For long reasoning chains, this becomes expensive.

**Decision**: Use latent spans for early reasoning steps, processing hidden states without token generation. Only final reasoning steps and answers are generated as tokens.

**Rationale**:
- Reduces generated tokens by 25-40% on long chains
- Maintains reasoning capability through hidden state processing
- Enables faster inference with fewer token generations
- Compatible with existing training infrastructure

**Trade-offs**:
- Requires additional model components (halt head, forward_hidden)
- Less interpretable (hidden reasoning not visible)
- Requires careful curriculum design to train effectively

**Alternatives Considered**:
- Explicit CoT with compression: Still generates tokens, less efficient
- External reasoning module: Adds complexity, harder to train end-to-end
- Shorter CoT chains: Reduces reasoning quality

## Training Curriculum Progression

### Progressive Curriculum Stages

The latent curriculum progressively replaces CoT steps with latent slots:

**S0 (Baseline)**: Full CoT
```
Prompt: "Solve this problem"
Step 1: Analyze the problem
Step 2: Find solution
Step 3: Verify answer
Answer: 42
```

**S1 (Early Latent)**: Replace first m=2 steps with c=1 latent slot
```
Prompt: "Solve this problem"
<bot> ... <eot>  # Latent reasoning (2 steps compressed)
Step 3: Verify answer
Answer: 42
```

**S2 (More Latent)**: Replace more steps, increase latent slots
```
Prompt: "Solve this problem"
<bot> ... <eot>  # Latent reasoning (3 steps compressed)
Answer: 42
```

### Curriculum Parameters

- **m**: Number of CoT steps to replace (default: 2)
- **c**: Number of latent slots per replaced step (default: 1, can increase to 2 when stable)
- **p**: Probability of applying curriculum (default: 0.5)

### Loss Masking

Latent slots are masked in the loss function:
- No supervision on hidden steps (latent spans)
- Supervise remaining visible steps + final answer
- Enables model to learn reasoning in latent space

## Halting Logic

### Halting Conditions

Refinement halts when any of these conditions are met:

1. **Judge Score + Delta Shrinking**: Judge score ≥ τ AND deltas shrinking (diff-IoU↑, failing tests↓)
2. **Halt Head Threshold**: Learned halt head probability > threshold
3. **CAWS Tier Limit**: Hard-cap by tier (max loops reached)

### Halt Head

The halt head is a learned linear layer (hidden_dim → 2) that predicts:
- `[continue, halt]` logits
- Trained to predict when refinement should stop
- Uses pooled hidden state as input

### Delta Shrinking Detection

Deltas are considered shrinking when:
- **diff-IoU increasing**: Current diff-IoU ≥ previous diff-IoU
- **failing tests decreasing**: Current failing tests ≤ previous failing tests

This indicates convergence toward a solution.

## CAWS Budget Tiers

### Tier Limits

CAWS budget tiers enforce different limits:

| Tier | Max Latent Spans | Max Loops | Use Case |
|------|------------------|-----------|----------|
| Tier-1 | 0 | ≤1 | Critical systems, no latent reasoning |
| Tier-2 | ≤1 | ≤2 | Standard features, limited latent |
| Tier-3 | ≤3 | ≤3 | Complex tasks, more latent reasoning |

### Budget Enforcement

Budgets are enforced at runtime by `RefinementController`:
- Hard-cap on loops per tier
- Hard-cap on latent spans per tier
- Cannot exceed limits regardless of halting conditions

## Performance Improvements

### Token Reduction

Latent reasoning achieves:
- **25-40% token reduction** on long-chain tasks
- Maintains ≥ baseline accuracy
- Reduces context window usage

### Efficiency Metrics

Efficiency is measured by:
- **Token reduction percentage**: (baseline_tokens - current_tokens) / baseline_tokens
- **Accuracy delta**: current_accuracy - baseline_accuracy (must be ≥ -0.01)
- **Loop count**: Average refinement loops (must be ≤ baseline)
- **Latency**: Wall-clock time (must not be worse than baseline)

### Efficiency Gates

Gates ensure quality:
- ≥ baseline accuracy with ≥25-40% fewer generated tokens
- Average refinement loops ≤ current self-refine
- Tail latency not worse than baseline
- No CAWS regressions

## Runtime Flags

### Training Flags

- `TRAIN_LATENT=1`: Enable latent curriculum in training
- Config: `latent.enabled: true` in `configs/kd_recipe.yaml`

### Inference Flags

- `LATENT_MODE=1`: Enable latent mode processing
- `HALT_HEAD=1`: Enable learned halting
- `MAX_LOOPS=<n>`: Override max loops per CAWS tier

### Evaluation Flags

- `EVAL_LATENT=1`: Enable latent efficiency gates
- Config: `eval/configs/latent.yaml`

## Usage Examples

### Training with Latent Curriculum

```yaml
# configs/kd_recipe.yaml
latent:
  enabled: true
  m: 2  # Replace 2 CoT steps
  c: 1  # 1 latent slot per step
  p: 0.5  # 50% probability
  training_loops: 4  # More loops during training
```

```bash
TRAIN_LATENT=1 python -m training.distill_kd --config configs/worker_9b.yaml configs/kd_recipe.yaml
```

### Inference with Latent Mode

```python
from runtime.engine.loop import LatentModeEngine

engine = LatentModeEngine(
    model=model,
    tokenizer=tokenizer,
    latent_mode_enabled=True,
)

result = engine.generate_with_latent_mode(
    input_ids,
    max_new_tokens=256,
    return_halt_logits=True,
)
```

### Refinement Controller

```python
from runtime.orchestration.refine import RefinementController, CAWSBudgetTier

controller = RefinementController(
    judge_score_threshold=0.8,
    halt_probability_threshold=0.7,
    caws_tier=CAWSBudgetTier.TIER_2,
    halt_head_enabled=True,
)

loops = 0
while loops < controller.max_loops:
    output = worker.generate(inputs)
    score = judge.score(output)
    halt_logits = output.get("halt_logits")
    
    should_halt, metadata = controller.should_halt(
        output, score, halt_logits
    )
    
    if should_halt:
        break
    
    inputs = update_inputs(output)
    loops += 1
```

## Observability

### Latent Probe

Use `LatentProbe` to inspect latent spans:

```python
from eval.probes.latent_probe import LatentProbe

probe = LatentProbe(tokenizer, model)
spans = probe.extract_latent_spans(tokens, hidden_states)
visualization = probe.visualize_spans(tokens, spans)
```

### Debugging

- Check `mode_transitions` in engine output
- Inspect `latent_span_lengths` for span statistics
- Review `errors` list for safety violations
- Use `LatentProbe` to visualize spans

## Safety Checks

### Automatic Safety Measures

1. **Max Latent Length**: Latent spans cannot exceed `max_latent_length` (default: 100)
2. **Max Latent Spans**: Cannot exceed `max_latent_spans` per sequence (default: 10)
3. **Mode Transition Validation**: Unmatched sentinels trigger errors and fallback
4. **Error Handling**: Exceptions in `forward_hidden` trigger fallback to language mode
5. **Stuck Detection**: Generation ending in latent mode triggers automatic exit

### Fallback Behavior

On errors or safety violations:
- Exit latent mode immediately
- Continue in language mode
- Log errors for debugging
- Do not crash or corrupt state

## Rollback Plan

To disable latent reasoning:

1. **Training**: Set `TRAIN_LATENT=0` or `latent.enabled: false`
2. **Inference**: Set `LATENT_MODE=0`
3. **Halting**: Set `HALT_HEAD=0`
4. **Evaluation**: Set `EVAL_LATENT=0`

All components are flag-guarded and backwards-compatible. Disabling flags reverts to normal CoT behavior.

## References

- Plan document: `.cursor/plans/code-mode-latent-reasoning.md`
- Code-mode documentation: `docs/CODE_MODE_MCP_DISTILLATION.md`
- CAWS integration: `docs/CAWS_AGENT_GUIDE.md`

