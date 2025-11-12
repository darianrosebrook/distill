<!-- ce5cee6c-c4bf-4277-90f9-eb2acff6551c 5bca1678-3b7c-4749-a177-255868c13cbf -->
# Fix Long-Context and Integration Span Issues

## Issues Identified

1. **Long-context cases**: None generated for N=20 (should be 2-3)

   - Current logic adds long-context in 6th pass: `num_long_context = min(3, total - len(cells))`
   - For small N, all slots are filled before reaching the long-context pass
   - Long-context is not included in upfront slot reservation

2. **Integration spans**: Only 15/20 samples (75%) have integration spans

   - Current logic uses hardcoded key: `"ANE prefers fp16 kernels when ops are supported"`
   - Only matches when this exact string appears in teacher text
   - Control cases (no_tool) and some adversarial cases (ambiguity) don't have tool calls, so no integration expected
   - Some normal cases might not match the hardcoded pattern

## Solution: Four Concrete Changes + Verification

### 1. Reserve Long-Context Upfront (Don't Let Later Passes Starve It)

**Location**: `build_stratified_cells()` function

**Approach**: Make long-context a first-class quota in the cell allocator, reserved upfront with tags.

**Changes**:

- Add `want_long_context = (0 if total < 20 else min(3, max(2, total // 10)))` to quota calculation
- Reserve long-context slots upfront with `long_context: True` tag in cell metadata
- Add `make_long_context()` helper to generate 8-12k token filler text
- Inject long-context filler after CAWS header in `synthesize_prompt()` when `cell.get("long_context")` is true

**Key implementation**:

```python
want_long_context = (0 if total < 20 else min(3, max(2, total // 10)))  # 2-3 for N≥20
# Reserve with tag
for _ in range(want_long_context):
    cells.append({"scenario": ..., "long_context": True})
```

### 2. Make Integration Span Extraction Robust (Regex, Multi-Span)

**Location**: `synthesize_prompt()` function (lines 464-471)

**Approach**: Replace hardcoded string matching with regex pattern that captures all "Integration:" sentences.

**Changes**:

- Use regex: `r'Integration:\s*([^\n]+?)(?:[\.!?…]\s|$)'` with Unicode flag
- Capture multiple integration spans (one per sentence)
- Ensure all non-control paths emit integration text
- Gate integration emission: `emit_integration = (expected not in {"no_tool", "decline"})`

**Key implementation**:

```python
import re
integration_spans_bytes = []
for m in re.finditer(r'Integration:\s*([^\n]+?)(?:[\.!?…]\s|$)', teacher, flags=re.UNICODE):
    integration_spans_bytes.append([m.start(1), m.end(1)])
```

### 3. Anchor Tool Name Span Exactly (Not Approximate)

**Location**: `synthesize_prompt()` function (tool name span calculation)

**Approach**: Find exact `"name":"..."` pair inside the JSON substring, then compute absolute byte offsets.

**Changes**:

- Extract JSON substring: `inner = teacher[start_tool:end_tool]`
- Find name pair: `name_pair = f'"name":"{first["name"]}"'`
- Compute relative position, then absolute: `name_abs_start = start_tool + name_rel + len('"name":"')`
- Guarantees span points at the value (`web.search`) not the key

**Key implementation**:

```python
name_pair = f'"name":"{first["name"]}"'
inner = teacher[start_tool:end_tool]
name_rel = inner.index(name_pair)
name_abs_start = start_tool + name_rel + len('"name":"')
name_abs_end = name_abs_start + len(first["name"])
```

### 4. Verifier: Add Long-Context Quota Check and Integration Presence Rule

**Location**: `scripts/verify_contextual_set.py`

**Approach**: Add hard stops for long-context count and integration coverage.

**Changes**:

- Count long-context items (tokenizer-aware if available, else byte threshold)
- Count integration coverage: samples with `call_sequence` should have `integration_spans_bytes` ≥95%
- Add gates: `if total >= 20 and not (2 <= long_context_count <= 3): FAIL`
- Add gates: `if integration_coverage < 0.95: FAIL`

## Additional Refinements (High-Leverage)

### 5. Tokenizer-Aware Length Thresholds

**Why**: ">8000 characters" ≠ ">8k tokens"; multilingual/emoji skew byte counts.

**Implementation**: Add `is_long_context()` helper that uses tokenizer if available, falls back to byte threshold (24000 bytes ≈ 8k tokens).

### 6. Stabilize Tool-JSON Slice Before Anchoring

**Why**: Whitespace/ordering can vary. Emit normalized JSON, then compute spans.

**Implementation**: Use `json.dumps(first, separators=(",",":"))` for consistent formatting.

### 7. Broader Integration Extraction Patterns

**Why**: Current regex misses `!`, `?`, Unicode periods, parentheses.

**Implementation**: Use `r'Integration:\s*([^\n]+?)(?:[\.!?…]\s|$)'` with Unicode flag.

### 8. Grounding Check: Verify Integration Contains Tool Output

**Why**: Model could emit "Integration:" that isn't grounded in tool results.

**Implementation**:

- Add `tool_result_fields` to metadata (exact field/value pairs from TOOL_RESULT)
- In verifier, require ≥95% of integration spans contain at least one tool_result_field value

### 9. Multi-Call Span Parity

**Why**: Multi-call samples need spans per call, not just first call.

**Implementation**:

- Annotate each call with `json_args_span_bytes[i]`
- Verifier: assert `#json_args_span_bytes == len(call_sequence)` in ≥95% of multi-call samples

### 10. Controls Shouldn't Accidentally Carry Integration

**Why**: Controls should be truly tool-free.

**Implementation**:

- Generator: Skip "Integration:" emission for `expected_behaviour in {"no_tool","decline"}`
- Verifier: Assert controls have empty `integration_spans_bytes` and empty/invalid `call_sequence`

### 11. Determinism & Reproducibility

**Why**: Results should be diffable and debuggable.

**Implementation**:

- Add `--seed` flag to generator, set `random.seed(seed)`
- Write `generation_plan` record (counts per tag) to JSONL header

### 12. Safety Guards

**Why**: Prevent drift and security issues.

**Implementation**:

- URL allowlist: add domain normalizer (strip `www.`, lowercase)
- PII: add phone/credit-card regexes to `redact_pii`
- Schema drift: verifier pre-computes known tool names, fails on unknown

## Files to Modify

1. `scripts/generate_contextual_prompts.py`

   - `build_stratified_cells()`: Add long-context to upfront reservation
   - `synthesize_prompt()`: Robust integration extraction, exact name anchoring, long-context filler injection
   - Add `make_long_context()` helper
   - Add `--seed` argument support

2. `scripts/verify_contextual_set.py`

   - Add long-context quota check (tokenizer-aware)
   - Add integration coverage check
   - Add grounding verification
   - Add multi-call span parity check
   - Add control case validation

## Testing Strategy

1. **Long-context verification**:

   - Generate N=20, N=30, N=60 samples
   - Verify 2-3 long-context items for N≥20
   - Verify long-context prompts are 8-12k tokens (use tokenizer if available)

2. **Integration span verification**:

   - Generate N=20 samples
   - Verify integration span coverage ≥95% for tool-using cases
   - Verify control cases have no integration spans
   - Verify integration spans contain tool result fields (grounding)

3. **Tool name span verification**:

   - Verify name spans point to exact value in `"name":"..."` pair
   - Verify spans are within JSON bounds

4. **Multi-call verification**:

   - Generate samples with multi-call sequences
   - Verify each call has corresponding span

## Hardening Tweaks (12 Final Improvements)

### 1. Define JSON Schema for Each Sample Line

**Location**: New file `schemas/dataset_item.schema.json`

**Purpose**: Lock the contract so regressions are obvious.

**Implementation**:

- Create JSON Schema with required fields, allowed enums, types
- Include arrays of `[start,end]` for spans
- Validate in verifier before any other checks
- Fail fast with line number on schema violations

### 2. Normalize Text Before Span Math

**Location**: `scripts/generate_contextual_prompts.py` and `scripts/verify_contextual_set.py`

**Purpose**: Prevent span bugs from unicode/whitespace variance.

**Implementation**:

- Normalize all emitted strings to NFC before span calculations
- Collapse CRLF→LF before writing to JSONL
- Store `text_norm: "NFC"` and `line_endings: "LF"` in metadata

### 3. Emit Both Byte and JSON-Pointer Anchors

**Location**: `scripts/generate_contextual_prompts.py`

**Purpose**: Byte spans are brittle; add semantic anchors.

**Implementation**:

- Add `json_pointer_name: "/name"` and `json_pointer_args: "/arguments"`
- For multi-call: `json_pointers: ["/calls/0/arguments", "/calls/1/arguments"]`
- Verifier can re-locate slices even if whitespace changes

### 4. Multi-Call Parity as Explicit Arrays

**Location**: `scripts/generate_contextual_prompts.py`

**Purpose**: Don't overload single span; make multi-call explicit.

**Implementation**:

- Emit `json_args_spans_bytes: [[a0,b0],[a1,b1],…]` (array of arrays)
- Emit `tool_name_spans_bytes: [[n0a,n0b],[n1a,n1b],…]` (array of arrays)
- Gate: for `len(call_sequence)=k`, require k spans for both arrays

### 5. Grounding Checks with Normalized Values and Types

**Location**: `scripts/generate_contextual_prompts.py` and `scripts/verify_contextual_set.py`

**Purpose**: Compare normalized values, handle types correctly.

**Implementation**:

- Add `tool_result_fields` to metadata: flattened dict of stringified values
- Verifier: normalize with `.casefold()`, collapse whitespace, match substrings
- For numerics: accept exact digit run (e.g., `r"\b128\b"`)

### 6. Tokenizer-Aware Long-Context Threshold

**Location**: `scripts/verify_contextual_set.py`

**Purpose**: Prefer tokens when available; expose as flags.

**Implementation**:

- Add `--long_token_threshold 8000` and `--long_byte_threshold 24000` flags
- Use tokenizer if provided, else conservative byte fallback
- Update `is_long_context()` helper to use these thresholds

### 7. Determinism & Provenance

**Location**: `scripts/generate_contextual_prompts.py` and `scripts/verify_contextual_set.py`

**Purpose**: Make results reproducible and traceable.

**Implementation**:

- Add `--seed` flag wired to `random.seed(seed)`
- Add `generation_plan` record at top of report (counts per tag)
- Add per-line `sample_id` (ULID or monotonic) and `provenance` dict
- Compute SHA256 of JSONL, write to report for integrity

### 8. Adversarial Taxonomy + Quotas

**Location**: `scripts/generate_contextual_prompts.py` and `scripts/verify_contextual_set.py`

**Purpose**: Make adversarial coverage explicit and visible.

**Implementation**:

- Define `adversarial.type ∈ {"malformed_json","range_violation","ambiguity","timeout_retry"}`
- Enforce quotas per type (≥1 when N≥30)
- Add check in report showing adversarial distribution

### 9. Safety Hardening

**Location**: `scripts/util_sanitize.py` and `scripts/verify_contextual_set.py`

**Purpose**: Prevent security issues and drift.

**Implementation**:

- URL allowlist: normalize hostname (lowercase, strip `www.`), then compare
- PII redaction: add phone and credit-card regexes
- Expose `--strict_safety` to fail generation if match slips through
- Store `safety_scan: {"urls_ok": true, "pii_hits": 0}` per item

### 10. CI-Friendly Failure Surfacing

**Location**: `scripts/verify_contextual_set.py`

**Purpose**: Turn failures into actionable diffs.

**Implementation**:

- Print first 5 failing line numbers with compressed diffs:
  - Span expected vs found
  - Which gate tripped (e.g., `integration_not_grounded`)
- Exit code ≠ 0 so CI can halt

### 11. Schema Versioning

**Location**: `scripts/generate_contextual_prompts.py`

**Purpose**: Track structural changes over time.

**Implementation**:

- Add `dataset_version: "1.1.0"` in each line's metadata
- Bump version on structural changes (e.g., single to multi-span arrays)
- Keep `CHANGELOG.md` documenting version changes

### 12. Tool Schema Evolution Guard

**Location**: `scripts/verify_contextual_set.py`

**Purpose**: Catch tool schema changes between generation and verification.

**Implementation**:

- Build set of known tools from registry
- Require every `call_sequence[i].name` ∈ set
- Print one-line diff when tool's schema changes (hash the schema dict)

## Optional Improvements

### Language Tag for Multilingual Prompts

**Location**: `scripts/generate_contextual_prompts.py`

**Implementation**:

- Add `lang: "en"|"es"|"de"|"fr"` to metadata for segmentation

### Runtime Mock

**Location**: New file `scripts/mock_tools.py`

**Implementation**:

- Include deterministic `TOOL_RESULT` templates
- Enable cross-check that integration spans reference exact mock outputs

### Unit Tests

**Location**: New file `tests/unit/test_contextual_generator.py`

**Implementation**:

- Micro pytest suite fabricating 3 items (single-call, multi-call, control)
- Assert every gate passes
- Becomes regression canary

## Files to Create/Modify

### New Files

1. `schemas/dataset_item.schema.json` - JSON Schema for dataset items
2. `scripts/mock_tools.py` - Deterministic tool result templates
3. `tests/unit/test_contextual_generator.py` - Unit tests for generator
4. `CHANGELOG.md` - Schema version changelog

### Modified Files

1. `scripts/generate_contextual_prompts.py`

   - Core fixes (long-context, integration, name anchoring)
   - Hardening tweaks (normalization, JSON pointers, multi-span arrays, provenance, schema version)
   - Optional improvements (language tags)

2. `scripts/verify_contextual_set.py`

   - Core fixes (long-context quota, integration coverage)
   - Hardening tweaks (schema validation, normalization, grounding, CI-friendly failures, tool schema guard)

3. `scripts/util_sanitize.py`

   - Safety hardening (phone/credit-card regexes, URL normalization)

## Acceptance Criteria

### Core Functionality

- [ ] Long-context samples generated for N≥20 (2-3 items)
- [ ] Integration span coverage ≥95% for tool-using, non-control samples
- [ ] Control cases correctly have no integration spans
- [ ] Tool name spans point to exact value, not key
- [ ] Multi-call samples have spans per call (explicit arrays)

### Hardening Requirements

- [ ] JSON Schema validation: 100% pass, fail fast with line numbers
- [ ] Text normalization: NFC + LF, flags in metadata
- [ ] JSON-Pointer anchors: `/name`, `/arguments` present
- [ ] Multi-call parity: ≥95% of multi-call items have 1-to-1 span arrays
- [ ] Grounding: ≥95% of tool-using items have integration spans containing tool_result_field values
- [ ] Tokenizer-aware thresholds: Use tokens when available, bytes as fallback
- [ ] Determinism: Same `--seed` yields identical SHA256
- [ ] Provenance: sample_id, provenance dict, generation_plan in report
- [ ] Adversarial quotas: ≥1 per type when N≥30
- [ ] Safety scan: URLs normalized, PII redacted, safety_scan in metadata
- [ ] CI-friendly: First 5 failures with diffs, exit code ≠ 0
- [ ] Schema versioning: dataset_version in metadata, CHANGELOG.md
- [ ] Tool schema guard: Unknown tools = hard fail, schema change detection

### Quality Gates

- [ ] Multi-call parity: ≥0.95 of multi-call items have 1-to-1 span arrays
- [ ] Grounding: ≥0.95 of tool-using, non-control items have grounded integration spans
- [ ] Schema validation: 100% pass
- [ ] Determinism: Re-running with same `--seed` yields identical SHA256
- [ ] All existing tests pass
- [ ] Generated samples maintain proper stratification