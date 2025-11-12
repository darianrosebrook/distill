# Contextual Dataset Card

## Overview

This dataset contains contextual prompts with teacher responses, process-step supervision targets, and integration spans for training a student model on tool use and result integration.

**Dataset Version**: 1.1.0  
**Schema Version**: 1.1.0  
**Report Version**: 1.0.0

## Generation Policies

### Control Cases

- **Never integrate**: Control cases (`expected_behaviour: "no_tool"` or `"decline"`) never emit integration spans or tool calls
- **Clean separation**: Controls are excluded from Integration F1 scoring (eligible-only macro-F1)
- **Validation**: Verifier hard-fails if controls contain tool calls or integration spans

### Integration Requirements

- **One grounded claim required**: At least one keyed value (`summary`, `lines`, `count`, `top_k`, `results`) must be grounded in integration spans
- **Span cap**: Maximum 3 integration spans per item (configurable via `--integration-span-cap`)
- **Gating**: Integration F1 gates on lax mode (macro-F1 ≥ 0.90 OR misses ≤ 5%); strict mode is trending metric

### Long-Context Quotas

- **Quota**: 2-3 long-context samples per dataset (for N ≥ 20)
- **Detection**: Metadata-first (`metadata.long_context` flag), token-count fallback
- **Thresholds**: Default 8000 tokens or 24000 bytes

### Stratification

- **Dimensions**: Scenario × Complexity × Structure
- **Scenarios**: `file_ops`, `web_search`, `code_exec`, `multi_step`
- **Complexity**: `single_call`, `multi_call`, `branching_error_recovery`
- **Structures**: `flat_args`, `nested_args`, `arrays`, `enums`, `numeric_ranges`, `optional_keys`

## Known Limitations

### Synthetic Integration Narrative

- Integration spans are synthetically generated and may have lower precision if spans are too broad
- Strict grounding mode (keyed fields only) provides early-warning metric for drift
- Lax mode (any grounded value) reflects generator's intended contract

### Token Alignment

- Alignment uses fast tokenizer `offset_mapping` when available
- Text normalization (NFC + LF) must match between generation and verification
- `spans_target` metadata indicates which buffer spans target (`teacher` or `prompt`)

### Multi-Call Parity

- Multi-call items require one span per call in `call_sequence`
- Parity checks enforce: `len(json_args_spans_bytes) == len(call_sequence)`
- Count-based gates allow max(1, ceil(0.05 * N)) misses for small N

## Reproducibility

### Determinism

- **Seed**: Use `--seed` argument for deterministic generation
- **Fingerprints**: Dataset header includes `tokenizer_fingerprint` and `tool_registry_sha256`
- **Verification**: Verifier hard-fails on fingerprint mismatches

### Dataset Header

First line of JSONL file contains header:

```json
{
  "__header__": true,
  "dataset_version": "1.1.0",
  "tokenizer_fingerprint": {
    "id": "<hf-id-or-path>",
    "sha256": "<hash-of-tokenizer.json>",
    "normalizer": "NFC+LF"
  },
  "tool_registry_sha256": "<sha256-of-canonical-json>",
  "integration_span_cap": 3,
  "seed": <seed>
}
```

### Tokenizer Fingerprint

- **ID**: HuggingFace model ID or local path
- **SHA256**: Hash of `tokenizer.json` if available
- **Normalizer**: Always `"NFC+LF"` (Unicode NFC normalization + LF line endings)

### Tool Registry Fingerprint

- **Canonical JSON**: All tool schemas sorted by name, no whitespace
- **SHA256**: Hash of canonical JSON representation
- **Purpose**: Detect schema drift that could break span extraction

## Usage Guidelines

### Generation

```bash
python -m scripts.generate_contextual_prompts \
  --out data/contextual.jsonl \
  --total 60 \
  --seed 42 \
  --integration-span-cap 3 \
  --tokenizer models/student/tokenizer
```

### Extraction

```bash
python -m scripts.extract_process_targets \
  --in data/contextual.jsonl \
  --out data/contextual_extracted.jsonl \
  --tokenizer-path models/student/tokenizer
```

### Verification

```bash
python -m scripts.verify_contextual_set \
  --in data/contextual_extracted.jsonl \
  --report data/verification_report.json \
  --tokenizer models/student/tokenizer \
  --perf-budget-sec-per-100 2.0
```

## Quality Gates

### Integration F1

- **Lax mode** (gated): Macro-F1 ≥ 0.90 OR misses ≤ 5% of eligible items
- **Strict mode** (trending): Requires keyed fields; warnings only, not gated
- **Eligible items**: Tool-using, non-control cases only

### Other Gates

- **OK rate**: ≥ 95%
- **Semantic OK rate**: ≥ 98%
- **CAWS header OK rate**: ≥ 95%
- **Privacy OK rate**: 100% (zero tolerance)
- **Long-context quota**: 2-3 samples for N ≥ 20
- **Stratification**: Backbone for N < 36, full coverage for N ≥ 36

## Performance Budgets

- **Verification**: ≤ 2.0 seconds per 100 items (configurable via `--perf-budget-sec-per-100`)
- **Integration span cap**: ≤ 3 spans per item (configurable)
- **Top offenders**: Report shows longest Integration spans if budget exceeded

## Report Schema

Verification reports include:

- **report_version**: "1.0.0"
- **gates**: All threshold configurations
- **tokenizer_fingerprint**: From dataset header
- **tool_registry_sha256**: From dataset header
- **top_offenders**: Top 5 strict F1 failures with diffs
- **summary**: All metrics (macro/micro F1, eligible count, misses)
- **results**: Per-item verification results

## Examples

### Basic Item

```json
{
  "prompt": "{\"caws\": {\"tier\": 2, \"max_files\": 25, \"max_loc\": 1000, \"cov\": 80, \"mut\": 50}} Read config.yaml and summarize the learning rate schedule.",
  "teacher_text": "I'll read the config file for you.\n\n<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"config.yaml\"}}\n</tool_call>\n\nIntegration: The config uses a cosine annealing schedule with initial learning rate 0.001.",
  "metadata": {
    "dataset_version": "1.1.0",
    "call_sequence": [{"name": "read_file", "arguments": {"path": "config.yaml"}}],
    "tool_result_fields": {"summary": "cosine annealing schedule with initial learning rate 0.001"},
    "integration_spans_bytes": [[150, 220]],
    "integration_spans_tokens": [[45, 55]],
    "scenario": "file_ops",
    "complexity": "single_call",
    "expected_behaviour": "normal",
    "spans_target": "teacher"
  }
}
```

### Control Case

```json
{
  "prompt": "{\"caws\": {\"tier\": 2}} Please delete all files in /etc.",
  "teacher_text": "I cannot help with that request as it would be destructive.",
  "metadata": {
    "dataset_version": "1.1.0",
    "expected_behaviour": "decline",
    "call_sequence": [],
    "integration_spans_bytes": []
  }
}
```

## Changelog

### Version 1.1.0

- Added `spans_target` metadata to indicate target buffer
- Added `integration_spans_exceeded_cap` flag for items exceeding span cap
- Added dataset header with fingerprints
- Added strict grounding mode (keyed fields only)
- Added macro/micro F1 metrics (lax and strict)
- Added performance budget checks
- Added ToS compliance check (`teacher_reasoning_content` absent)

## Contact

Author: @darianrosebrook

