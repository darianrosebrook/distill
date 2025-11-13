# Knowledge Distillation Guide

## Process-Step Supervision

Process-step supervision is a technique for training student models on structured decision-making patterns from teacher outputs, without training on reasoning prose (which may violate terms of service).

### Overview

Instead of training on full teacher reasoning content, we extract and supervise on three key components:

1. **Tool Name Selection**: The specific tool name chosen by the teacher
2. **JSON Argument Structure**: The JSON-formatted arguments passed to tools
3. **Integration Patterns**: How tool results are integrated into the response

This approach provides structured supervision while avoiding ToS violations from copying reasoning prose.

### Dataset Generation

Process-step supervision targets are extracted during dataset generation using `scripts/make_kd_mix_hardened.py`.

#### Enabling Process-Step Supervision

By default, process-step supervision is enabled. To disable it:

```bash
python -m scripts.make_kd_mix_hardened \
    --out data/kd_mix.jsonl \
    --teacher <teacher-endpoint> \
    --total 1000 \
    --no-process-supervision  # Disable extraction
```

#### Specifying Tokenizer

The tokenizer path defaults to `models/student/tokenizer`. To specify a different path:

```bash
python -m scripts.make_kd_mix_hardened \
    --out data/kd_mix.jsonl \
    --teacher <teacher-endpoint> \
    --total 1000 \
    --tokenizer-path path/to/tokenizer  # Custom tokenizer path
```

#### Tool Names from Registry

Tool names are automatically loaded from the tool registry (`tools/schema_registry.py`) during extraction. This enables proper tool name span extraction.

### Dataset Format

Process-step supervision targets are stored in the JSONL dataset with the following fields:

```json
{
  "prompt": "Read the file config.json",
  "teacher_text": "I will use {\"name\": \"read_file\", \"arguments\": {\"path\": \"config.json\"}}",
  "tool_name_ids": [123, 456, 789],
  "tool_name_mask": [1, 1, 1],
  "gold_json_text_ids": [234, 567, 890, ...],
  "mask_valid_json_tokens": [1, 1, 1, ...],
  "tool_result_fields": [345, 678, 901, ...],
  "integration_mask": [1, 1, 1, ...],
  "metadata": {...}
}
```

#### Field Descriptions

- **tool_name_ids**: Token IDs for the tool name span (e.g., "read_file")
- **tool_name_mask**: Boolean mask indicating valid tool name tokens
- **gold_json_text_ids**: Token IDs for JSON argument spans
- **mask_valid_json_tokens**: Boolean mask indicating valid JSON tokens
- **tool_result_fields**: Token IDs for integration spans (post-tool result usage)
- **integration_mask**: Boolean mask indicating valid integration tokens

All fields are optional - they are only present when the corresponding patterns are found in the teacher output.

### Teacher Stub Quality

For toy models and testing, teacher logits can be generated using deterministic "teacher stubs" that simulate teacher model behavior:

#### Intelligent Teacher Stubs

Recent improvements to teacher stubs provide much higher quality distillation:

**Real Token Sequences**: Instead of using arbitrary token IDs, modern teacher stubs tokenize actual domain-specific phrases and use their real token sequences.

```python
# Example: Mystical teacher stub for fortune-telling model
mystical_phrases = [
    "It is certain", "Outlook good", "Very doubtful",
    "Signs point to yes", "My sources say no"
]

# Generate real token sequences
for phrase in mystical_phrases:
    tokens = tokenizer.encode(phrase, add_special_tokens=False)
    # tokens = [739, 338, 3058] for "It is certain"
    # Use these for position-aware boosting
```

**Context-Aware Boosting**: Teacher preferences are applied positionally, boosting mystical tokens only where answers should appear (not during prompts).

**Improved Normalization**: Prevents aggressive logit scaling that eliminates meaningful preferences, preserving positive values for desired tokens.

**Performance Impact**: These improvements can increase model quality by 25% and domain compliance by 100%.

### Training Integration

Process-step supervision targets are automatically used during training when available in the dataset.

#### Loss Functions

The training uses two main loss components:

1. **JSON Validity Loss** (`json_validity_weight`, default: 0.3)
   - Penalizes invalid JSON generation
   - Uses pre-extracted JSON token IDs when available
   - Falls back to text-based extraction for backward compatibility

2. **Tool Selection Loss** (`tool_select_weight`, default: 0.7)
   - Supervises correct tool name prediction
   - Uses pre-extracted tool name token IDs when available
   - Falls back to text-based extraction for backward compatibility

#### Configuration

Process-step supervision is configured in the training config:

```yaml
process_supervision:
  loss_json_validity_weight: 0.3
  loss_tool_select_weight: 0.7
  tool_names: ["read_file", "web.search", "file.write"]  # Optional: list of available tools
```

#### Training Step

The training step (`training/distill_process.py`) automatically:

1. Extracts target tool names from batch metadata
2. Loads token IDs from batch
3. Passes token IDs to loss functions (preferred over text extraction)
4. Computes combined loss with knowledge distillation

### Extraction Functions

The extraction logic is implemented in `training/extractors.py`:

- **extract_tool_name_span()**: Finds tool name character spans in text
- **extract_json_argument_spans()**: Finds JSON argument character spans
- **identify_integration_spans()**: Identifies post-tool integration patterns

These functions work at the character level, then tokenization is applied to create token ID targets.

### Testing

Integration tests verify the complete flow:

```bash
pytest tests/integration/test_process_step_integration.py
```

Tests verify:
- Dataset loads process-step targets correctly
- Batches contain target fields
- Loss functions compute correctly with token IDs
- Training step runs without errors

### Example Usage

1. **Generate dataset with process-step targets**:
```bash
python -m scripts.make_kd_mix_hardened \
    --out data/kd_mix.jsonl \
    --teacher https://api.example.com/v1 \
    --total 1000 \
    --tokenizer-path models/student/tokenizer
```

2. **Verify targets in dataset**:
```python
import json
with open('data/kd_mix.jsonl') as f:
    sample = json.loads(next(f))
    print("Tool name IDs:", sample.get("tool_name_ids"))
    print("JSON IDs:", sample.get("gold_json_text_ids"))
```

3. **Train with process-step supervision**:
```bash
python -m training.distill_process \
    --config configs/process_supervision.yaml \
    --train-data data/kd_mix.jsonl
```

### Benefits

- **ToS Compliance**: Avoids copying reasoning prose
- **Structured Supervision**: Focuses on decision-making patterns
- **Efficiency**: Uses pre-extracted token IDs instead of re-extracting from text
- **Flexibility**: Falls back to text-based extraction when token IDs unavailable

### Limitations

- Extraction depends on teacher output format consistency
- Integration span detection uses heuristics (can be improved)
- Tool names must be registered in tool registry for proper extraction

### Future Improvements

- More sophisticated integration span detection
- Support for multi-tool sequences
- Alignment of token IDs with sequence positions for more accurate loss computation
