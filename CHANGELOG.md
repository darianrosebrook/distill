# Dataset Schema Changelog

## Version 1.1.0

### Added
- `dataset_version` field in metadata (required)
- `sample_id` field for unique identification
- `provenance` object with seed, scenario, and tags
- `lang` field (defaults to "en", multilingual codes for non-English)
- `text_norm` and `line_endings` fields (NFC, LF)
- `safety_scan` object with `urls_ok` and `pii_hits`
- `json_args_spans_bytes` array for multi-call support
- `tool_name_spans_bytes` array for multi-call support
- `json_pointers` array for semantic anchoring
- `json_pointer_name` and `json_pointer_args` for single-call
- `tool_result_fields` for grounding verification

### Changed
- CAWS header: removed `spec` field (now ≤30 tokens)
- Integration span extraction: now uses regex pattern matching instead of hardcoded string
- Tool name spans: now use exact anchoring within JSON substring
- Text normalization: all text normalized to NFC before span calculations
- Multi-call support: explicit arrays for spans per call

### Fixed
- Long-context samples now reserved upfront (2-3 for N≥20)
- Integration spans extracted with robust regex (handles multiple sentences)
- Tool name spans point to exact value, not key
- Multi-call samples have spans per call

## Version 1.0.0

Initial schema version.

