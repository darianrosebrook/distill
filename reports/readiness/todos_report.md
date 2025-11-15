# Improved Hidden TODO Analysis Report (v2.0)
============================================================

## Summary
- Total files: 36009
- Non-ignored files: 307
- Ignored files: 35702
- Files with hidden TODOs: 40
- Total hidden TODOs found: 4507
- Code stub detections: 4
- High confidence TODOs (≥0.9): 4504
- Medium confidence TODOs (≥0.6): 3
- Low confidence TODOs (<0.6): 0
- Minimum confidence threshold: 0.7

## Files by Language
- **cpp**: 1 files
- **javascript**: 1 files
- **python**: 251 files
- **shell**: 18 files
- **typescript**: 4 files
- **yaml**: 32 files

## Pattern Statistics
- `\bfor\s+now\b(?!(_|\.|anal|\sanal|s))`: 3866 occurrences
- `\bsimplified\b(?!(_|\.|anal|\sanal|s))`: 504 occurrences
- `\bPLACEHOLDER\b`: 121 occurrences
- `\bPLACEHOLDER\b.*?:`: 100 occurrences
- `\bTODO\b(?!(_|\.|anal|\sanal|s))`: 9 occurrences
- `python_raise_not_implemented`: 4 occurrences
- `\bTODO\b.*?:`: 3 occurrences
- `\bsimplified\s+.*?\s+implementation\b`: 2 occurrences
- `\bpatch\b.*?(fix|solution)`: 2 occurrences
- `\bin\s+a\s+real\b(?!(_|\.|anal|\sanal|s))`: 2 occurrences
- `\bin\s+practice\b.*?(this\s+would|this\s+should|this\s+will)`: 2 occurrences
- `\bstub\s+implementation\b`: 1 occurrences
- `\bstub\s+implementation\s+for\b`: 1 occurrences
- `\bin\s+production\b.*?(implement|add|fix)`: 1 occurrences

## Files with High-Confidence Hidden TODOs
- `mutants/training/distill_kd.py` (python): 2006 high-confidence TODOs
- `mutants/training/run_toy_distill.py` (python): 1638 high-confidence TODOs
- `mutants/training/examples_priority3_integration.py` (python): 465 high-confidence TODOs
- `mutants/training/quant_qat_int8.py` (python): 174 high-confidence TODOs
- `mutants/training/claim_extraction.py` (python): 102 high-confidence TODOs
- `mutants/training/losses.py` (python): 29 high-confidence TODOs
- `scripts/analyze_todo_risks.py` (python): 12 high-confidence TODOs
- `evaluation/perf_mem_eval.py` (python): 9 high-confidence TODOs
- `scripts/assess_readiness.py` (python): 6 high-confidence TODOs
- `conversion/convert_coreml.py` (python): 6 high-confidence TODOs
- `training/examples_priority3_integration.py` (python): 5 high-confidence TODOs
- `eval/scenario_harness.py` (python): 5 high-confidence TODOs
- `scripts/monitor_watchdog.py` (python): 4 high-confidence TODOs
- `eval/scoring/scorer.py` (python): 4 high-confidence TODOs
- `eval/runners/orchestrator.py` (python): 4 high-confidence TODOs
- `arbiter/claims/pipeline.py` (python): 4 high-confidence TODOs
- `training/claim_extraction.py` (python): 3 high-confidence TODOs
- `scripts/convert_to_gguf.py` (python): 2 high-confidence TODOs
- `scripts/patch_mutatest.py` (python): 2 high-confidence TODOs
- `scripts/verify-code-quality.py` (python): 2 high-confidence TODOs
- `evaluation/8ball_eval.py` (python): 2 high-confidence TODOs
- `coreml/probes/compare_probes.py` (python): 2 high-confidence TODOs
- `coreml/runtime/ane_monitor.py` (python): 2 high-confidence TODOs
- `infra/version_gate.py` (python): 1 high-confidence TODOs
- `runtime/api_contract.py` (python): 1 high-confidence TODOs
- `coreml/ane_checks.py` (python): 1 high-confidence TODOs
- `scripts/test_8_ball_coreml.py` (python): 1 high-confidence TODOs
- `scripts/verify_contextual_set.py` (python): 1 high-confidence TODOs
- `scripts/inference_production.py` (python): 1 high-confidence TODOs
- `scripts/validate_data_provenance.py` (python): 1 high-confidence TODOs
- `scripts/check_readiness.py` (python): 1 high-confidence TODOs
- `scripts/smoke_test_pipeline.py` (python): 1 high-confidence TODOs
- `conversion/make_toy_block.py` (python): 1 high-confidence TODOs
- `conversion/judge_export_coreml.py` (python): 1 high-confidence TODOs
- `evaluation/toy_contracts.py` (python): 1 high-confidence TODOs
- `eval/halt_calibration.py` (python): 1 high-confidence TODOs
- `data/generators/mcp_code_mode.py` (python): 1 high-confidence TODOs
- `eval/runners/base.py` (python): 1 high-confidence TODOs
- `coreml/runtime/constrained_decode.py` (python): 1 high-confidence TODOs

## Engineering-Grade TODO Suggestions

The following TODOs should be upgraded to the engineering-grade format:

### `scripts/assess_readiness.py:403` (python)
**Original:** TODO Blocker Ratio (25%)...
**Suggested Tier:** 3
**Priority:** Medium
**Missing Elements:** completion_checklist, acceptance_criteria, dependencies, governance

**Suggested Template:**
```
# TODO: Blocker Ratio (25%)
#       <One-sentence context & why this exists>
#
# COMPLETION CHECKLIST:
# [ ] Primary functionality implemented
# [ ] API/data structures defined & stable
# [ ] Error handling + validation aligned with error taxonomy
# [ ] Tests: Unit ≥80% branch coverage (≥50% mutation if enabled)
# [ ] Integration tests for external systems/contracts
# [ ] Documentation: public API + system behavior
# [ ] Performance/profiled against SLA (CPU/mem/latency throughput)
# [ ] Security posture reviewed (inputs, authz, sandboxing)
# [ ] Observability: logs (debug), metrics (SLO-aligned), tracing
# [ ] Configurability and feature flags defined if relevant
# [ ] Failure-mode cards documented (degradation paths)
#
# ACCEPTANCE CRITERIA:
# - <User-facing measurable behavior>
# - <Invariant or schema contract requirements>
# - <Performance/statistical bounds>
# - <Interoperation requirements or protocol contract>
#
# DEPENDENCIES:
# - <System or feature this relies on> (Required/Optional)
# - <Interop/contract references>
# - File path(s)/module links to dependent code
#
# ESTIMATED EFFORT: <Number + confidence range>
# PRIORITY: Medium
# BLOCKING: {Yes/No} – If Yes: explicitly list what it blocks
#
# GOVERNANCE:
# - CAWS Tier: 3 (impacts rigor, provenance, review policy)
# - Change Budget: <LOC or file count> (if relevant)
# - Reviewer Requirements: <Roles or domain expertise>
```

### `scripts/verify-code-quality.py:54` (python)
**Original:** Only flag PLACEHOLDER/TODO/MOCK_DATA when they're clearly marked as such...
**Suggested Tier:** 3
**Priority:** Medium
**Missing Elements:** completion_checklist, acceptance_criteria, dependencies, governance

**Suggested Template:**
```
# TODO: /MOCK_DATA when they're clearly marked as such
#       <One-sentence context & why this exists>
#
# COMPLETION CHECKLIST:
# [ ] Primary functionality implemented
# [ ] API/data structures defined & stable
# [ ] Error handling + validation aligned with error taxonomy
# [ ] Tests: Unit ≥80% branch coverage (≥50% mutation if enabled)
# [ ] Integration tests for external systems/contracts
# [ ] Documentation: public API + system behavior
# [ ] Performance/profiled against SLA (CPU/mem/latency throughput)
# [ ] Security posture reviewed (inputs, authz, sandboxing)
# [ ] Observability: logs (debug), metrics (SLO-aligned), tracing
# [ ] Configurability and feature flags defined if relevant
# [ ] Failure-mode cards documented (degradation paths)
#
# ACCEPTANCE CRITERIA:
# - <User-facing measurable behavior>
# - <Invariant or schema contract requirements>
# - <Performance/statistical bounds>
# - <Interoperation requirements or protocol contract>
#
# DEPENDENCIES:
# - <System or feature this relies on> (Required/Optional)
# - <Interop/contract references>
# - File path(s)/module links to dependent code
#
# ESTIMATED EFFORT: <Number + confidence range>
# PRIORITY: Medium
# BLOCKING: {Yes/No} – If Yes: explicitly list what it blocks
#
# GOVERNANCE:
# - CAWS Tier: 3 (impacts rigor, provenance, review policy)
# - Change Budget: <LOC or file count> (if relevant)
# - Reviewer Requirements: <Roles or domain expertise>
```

### `scripts/verify-code-quality.py:295` (python)
**Original:** TODO/PLACEHOLDER check...
**Suggested Tier:** 3
**Priority:** Medium
**Missing Elements:** completion_checklist, acceptance_criteria, dependencies, governance

**Suggested Template:**
```
# TODO: /PLACEHOLDER check
#       <One-sentence context & why this exists>
#
# COMPLETION CHECKLIST:
# [ ] Primary functionality implemented
# [ ] API/data structures defined & stable
# [ ] Error handling + validation aligned with error taxonomy
# [ ] Tests: Unit ≥80% branch coverage (≥50% mutation if enabled)
# [ ] Integration tests for external systems/contracts
# [ ] Documentation: public API + system behavior
# [ ] Performance/profiled against SLA (CPU/mem/latency throughput)
# [ ] Security posture reviewed (inputs, authz, sandboxing)
# [ ] Observability: logs (debug), metrics (SLO-aligned), tracing
# [ ] Configurability and feature flags defined if relevant
# [ ] Failure-mode cards documented (degradation paths)
#
# ACCEPTANCE CRITERIA:
# - <User-facing measurable behavior>
# - <Invariant or schema contract requirements>
# - <Performance/statistical bounds>
# - <Interoperation requirements or protocol contract>
#
# DEPENDENCIES:
# - <System or feature this relies on> (Required/Optional)
# - <Interop/contract references>
# - File path(s)/module links to dependent code
#
# ESTIMATED EFFORT: <Number + confidence range>
# PRIORITY: Medium
# BLOCKING: {Yes/No} – If Yes: explicitly list what it blocks
#
# GOVERNANCE:
# - CAWS Tier: 3 (impacts rigor, provenance, review policy)
# - Change Budget: <LOC or file count> (if relevant)
# - Reviewer Requirements: <Roles or domain expertise>
```

### `scripts/analyze_todo_risks.py:11` (python)
**Original:** TODO Risk Analysis Script Analyzes TODO patterns from todos.json to identify production risks: 1. Cr...
**Suggested Tier:** 1
**Priority:** Critical
**Missing Elements:** completion_checklist, acceptance_criteria, dependencies, governance

**Suggested Template:**
```
# TODO: Risk Analysis Script Analyzes TODO patterns from todos.json to identify production risks: 1. Critical path TODOs (training/conversion) 2. "For now" implementation risk assessment 3. Generate actionable resolution plan @author: @darianrosebrook
#       <One-sentence context & why this exists>
#
# COMPLETION CHECKLIST:
# [ ] Primary functionality implemented
# [ ] API/data structures defined & stable
# [ ] Error handling + validation aligned with error taxonomy
# [ ] Tests: Unit ≥80% branch coverage (≥50% mutation if enabled)
# [ ] Integration tests for external systems/contracts
# [ ] Documentation: public API + system behavior
# [ ] Performance/profiled against SLA (CPU/mem/latency throughput)
# [ ] Security posture reviewed (inputs, authz, sandboxing)
# [ ] Observability: logs (debug), metrics (SLO-aligned), tracing
# [ ] Configurability and feature flags defined if relevant
# [ ] Failure-mode cards documented (degradation paths)
#
# ACCEPTANCE CRITERIA:
# - <User-facing measurable behavior>
# - <Invariant or schema contract requirements>
# - <Performance/statistical bounds>
# - <Interoperation requirements or protocol contract>
#
# DEPENDENCIES:
# - <System or feature this relies on> (Required/Optional)
# - <Interop/contract references>
# - File path(s)/module links to dependent code
#
# ESTIMATED EFFORT: <Number + confidence range>
# PRIORITY: Critical
# BLOCKING: {Yes/No} – If Yes: explicitly list what it blocks
#
# GOVERNANCE:
# - CAWS Tier: 1 (impacts rigor, provenance, review policy)
# - Change Budget: <LOC or file count> (if relevant)
# - Reviewer Requirements: <Roles or domain expertise>
```

### `scripts/analyze_todo_risks.py:22` (python)
**Original:** Load TODO data from JSON file....
**Suggested Tier:** 3
**Priority:** Medium
**Missing Elements:** completion_checklist, acceptance_criteria, dependencies, governance

**Suggested Template:**
```
# TODO: data from JSON file.
#       <One-sentence context & why this exists>
#
# COMPLETION CHECKLIST:
# [ ] Primary functionality implemented
# [ ] API/data structures defined & stable
# [ ] Error handling + validation aligned with error taxonomy
# [ ] Tests: Unit ≥80% branch coverage (≥50% mutation if enabled)
# [ ] Integration tests for external systems/contracts
# [ ] Documentation: public API + system behavior
# [ ] Performance/profiled against SLA (CPU/mem/latency throughput)
# [ ] Security posture reviewed (inputs, authz, sandboxing)
# [ ] Observability: logs (debug), metrics (SLO-aligned), tracing
# [ ] Configurability and feature flags defined if relevant
# [ ] Failure-mode cards documented (degradation paths)
#
# ACCEPTANCE CRITERIA:
# - <User-facing measurable behavior>
# - <Invariant or schema contract requirements>
# - <Performance/statistical bounds>
# - <Interoperation requirements or protocol contract>
#
# DEPENDENCIES:
# - <System or feature this relies on> (Required/Optional)
# - <Interop/contract references>
# - File path(s)/module links to dependent code
#
# ESTIMATED EFFORT: <Number + confidence range>
# PRIORITY: Medium
# BLOCKING: {Yes/No} – If Yes: explicitly list what it blocks
#
# GOVERNANCE:
# - CAWS Tier: 3 (impacts rigor, provenance, review policy)
# - Change Budget: <LOC or file count> (if relevant)
# - Reviewer Requirements: <Roles or domain expertise>
```

### `scripts/analyze_todo_risks.py:50` (python)
**Original:** Categorize TODO by risk level. Returns: CRITICAL, HIGH, MEDIUM, or LOW...
**Suggested Tier:** 1
**Priority:** Critical
**Missing Elements:** completion_checklist, acceptance_criteria, dependencies, governance

**Suggested Template:**
```
# TODO: by risk level. Returns: CRITICAL, HIGH, MEDIUM, or LOW
#       <One-sentence context & why this exists>
#
# COMPLETION CHECKLIST:
# [ ] Primary functionality implemented
# [ ] API/data structures defined & stable
# [ ] Error handling + validation aligned with error taxonomy
# [ ] Tests: Unit ≥80% branch coverage (≥50% mutation if enabled)
# [ ] Integration tests for external systems/contracts
# [ ] Documentation: public API + system behavior
# [ ] Performance/profiled against SLA (CPU/mem/latency throughput)
# [ ] Security posture reviewed (inputs, authz, sandboxing)
# [ ] Observability: logs (debug), metrics (SLO-aligned), tracing
# [ ] Configurability and feature flags defined if relevant
# [ ] Failure-mode cards documented (degradation paths)
#
# ACCEPTANCE CRITERIA:
# - <User-facing measurable behavior>
# - <Invariant or schema contract requirements>
# - <Performance/statistical bounds>
# - <Interoperation requirements or protocol contract>
#
# DEPENDENCIES:
# - <System or feature this relies on> (Required/Optional)
# - <Interop/contract references>
# - File path(s)/module links to dependent code
#
# ESTIMATED EFFORT: <Number + confidence range>
# PRIORITY: Critical
# BLOCKING: {Yes/No} – If Yes: explicitly list what it blocks
#
# GOVERNANCE:
# - CAWS Tier: 1 (impacts rigor, provenance, review policy)
# - Change Budget: <LOC or file count> (if relevant)
# - Reviewer Requirements: <Roles or domain expertise>
```

### `scripts/analyze_todo_risks.py:205` (python)
**Original:** Read code context around a TODO line....
**Suggested Tier:** 3
**Priority:** Medium
**Missing Elements:** completion_checklist, acceptance_criteria, dependencies, governance

**Suggested Template:**
```
# TODO: line.
#       <One-sentence context & why this exists>
#
# COMPLETION CHECKLIST:
# [ ] Primary functionality implemented
# [ ] API/data structures defined & stable
# [ ] Error handling + validation aligned with error taxonomy
# [ ] Tests: Unit ≥80% branch coverage (≥50% mutation if enabled)
# [ ] Integration tests for external systems/contracts
# [ ] Documentation: public API + system behavior
# [ ] Performance/profiled against SLA (CPU/mem/latency throughput)
# [ ] Security posture reviewed (inputs, authz, sandboxing)
# [ ] Observability: logs (debug), metrics (SLO-aligned), tracing
# [ ] Configurability and feature flags defined if relevant
# [ ] Failure-mode cards documented (degradation paths)
#
# ACCEPTANCE CRITERIA:
# - <User-facing measurable behavior>
# - <Invariant or schema contract requirements>
# - <Performance/statistical bounds>
# - <Interoperation requirements or protocol contract>
#
# DEPENDENCIES:
# - <System or feature this relies on> (Required/Optional)
# - <Interop/contract references>
# - File path(s)/module links to dependent code
#
# ESTIMATED EFFORT: <Number + confidence range>
# PRIORITY: Medium
# BLOCKING: {Yes/No} – If Yes: explicitly list what it blocks
#
# GOVERNANCE:
# - CAWS Tier: 3 (impacts rigor, provenance, review policy)
# - Change Budget: <LOC or file count> (if relevant)
# - Reviewer Requirements: <Roles or domain expertise>
```

### `scripts/analyze_todo_risks.py:263` (python)
**Original:** Critical Path TODO Inventory...
**Suggested Tier:** 3
**Priority:** Medium
**Missing Elements:** completion_checklist, acceptance_criteria, dependencies, governance

**Suggested Template:**
```
# TODO: Inventory
#       <One-sentence context & why this exists>
#
# COMPLETION CHECKLIST:
# [ ] Primary functionality implemented
# [ ] API/data structures defined & stable
# [ ] Error handling + validation aligned with error taxonomy
# [ ] Tests: Unit ≥80% branch coverage (≥50% mutation if enabled)
# [ ] Integration tests for external systems/contracts
# [ ] Documentation: public API + system behavior
# [ ] Performance/profiled against SLA (CPU/mem/latency throughput)
# [ ] Security posture reviewed (inputs, authz, sandboxing)
# [ ] Observability: logs (debug), metrics (SLO-aligned), tracing
# [ ] Configurability and feature flags defined if relevant
# [ ] Failure-mode cards documented (degradation paths)
#
# ACCEPTANCE CRITERIA:
# - <User-facing measurable behavior>
# - <Invariant or schema contract requirements>
# - <Performance/statistical bounds>
# - <Interoperation requirements or protocol contract>
#
# DEPENDENCIES:
# - <System or feature this relies on> (Required/Optional)
# - <Interop/contract references>
# - File path(s)/module links to dependent code
#
# ESTIMATED EFFORT: <Number + confidence range>
# PRIORITY: Medium
# BLOCKING: {Yes/No} – If Yes: explicitly list what it blocks
#
# GOVERNANCE:
# - CAWS Tier: 3 (impacts rigor, provenance, review policy)
# - Change Budget: <LOC or file count> (if relevant)
# - Reviewer Requirements: <Roles or domain expertise>
```

### `eval/runners/orchestrator.py:172` (python)
**Original:** TODO: Integrate with tool broker for actual tool call extraction...
**Suggested Tier:** 3
**Priority:** Medium
**Missing Elements:** completion_checklist, acceptance_criteria, dependencies, governance

**Suggested Template:**
```
# TODO: Integrate with tool broker for actual tool call extraction
#       <One-sentence context & why this exists>
#
# COMPLETION CHECKLIST:
# [ ] Primary functionality implemented
# [ ] API/data structures defined & stable
# [ ] Error handling + validation aligned with error taxonomy
# [ ] Tests: Unit ≥80% branch coverage (≥50% mutation if enabled)
# [ ] Integration tests for external systems/contracts
# [ ] Documentation: public API + system behavior
# [ ] Performance/profiled against SLA (CPU/mem/latency throughput)
# [ ] Security posture reviewed (inputs, authz, sandboxing)
# [ ] Observability: logs (debug), metrics (SLO-aligned), tracing
# [ ] Configurability and feature flags defined if relevant
# [ ] Failure-mode cards documented (degradation paths)
#
# ACCEPTANCE CRITERIA:
# - <User-facing measurable behavior>
# - <Invariant or schema contract requirements>
# - <Performance/statistical bounds>
# - <Interoperation requirements or protocol contract>
#
# DEPENDENCIES:
# - <System or feature this relies on> (Required/Optional)
# - <Interop/contract references>
# - File path(s)/module links to dependent code
#
# ESTIMATED EFFORT: <Number + confidence range>
# PRIORITY: Medium
# BLOCKING: {Yes/No} – If Yes: explicitly list what it blocks
#
# GOVERNANCE:
# - CAWS Tier: 3 (impacts rigor, provenance, review policy)
# - Change Budget: <LOC or file count> (if relevant)
# - Reviewer Requirements: <Roles or domain expertise>
```

## Pattern Categories by Confidence
### Placeholder Code (3 items)
#### High Confidence (2 items)
- `training/claim_extraction.py:46` (python, conf: 1.0 (context: 0.0)): Simplified claim extractor for training purposes. This is an intentionally simpl...
- `eval/runners/orchestrator.py:170` (python, conf: 1.0 (context: 0.0)): Extract tool calls from output (simplified - real implementation would parse)...
#### Medium Confidence (1 items)
- `data/resources/mcp_ts_api/callMCPTool.ts:15` (typescript, conf: 0.9 (context: -0.2)): This is a stub implementation for training data generation...

### Future Improvements (3 items)
#### High Confidence (3 items)
- `infra/version_gate.py:44` (python, conf: 1.0 (context: 0.3)): Version gates: Check Python, macOS, and dependency versions before proceeding. V...
- `eval/scenario_harness.py:216` (python, conf: 0.9 (context: 0.0)): In practice, this would execute a test script or unit test...
- `eval/scenario_harness.py:229` (python, conf: 0.9 (context: 0.0)): In practice, this would call a judge model to compare outputs...

### Explicit Todos (4497 items)
#### High Confidence (4495 items)
- `training/examples_priority3_integration.py:187` (python, conf: 1.0 (context: 0.0)): Compute quality score for teacher output. Supports multiple scoring methods: - "...
- `training/examples_priority3_integration.py:229` (python, conf: 0.9 (context: -0.2)): This is a simplified BLEU approximation without nltk...
- `training/examples_priority3_integration.py:236` (python, conf: 1.0 (context: 0.0)): Unigram precision (simplified BLEU-1)...
- ... and 4492 more high-confidence items
#### Medium Confidence (2 items)
- `training/claim_extraction.py:15` (python, conf: 0.8 (context: -0.5)): Claim extraction utilities for training dataset generation and loss computation....
- `mutants/training/claim_extraction.py:11` (python, conf: 0.8 (context: -0.5)): Claim extraction utilities for training dataset generation and loss computation....

### Temporary Solutions (2 items)
#### High Confidence (2 items)
- `scripts/patch_mutatest.py:11` (python, conf: 0.9 (context: 0.0)): Patch mutatest to fix the random.sample issue with sets in Python 3.11. This pat...
- `scripts/patch_mutatest.py:18` (python, conf: 0.9 (context: 0.0)): Patch mutatest run.py to fix random.sample(set) issue....

### Code Stubs (4 items)
#### High Confidence (4 items)
- `evaluation/perf_mem_eval.py:104` (python, conf: 0.9 (context: 0.2)): raise NotImplementedError(...
- `evaluation/perf_mem_eval.py:122` (python, conf: 0.9 (context: 0.2)): raise NotImplementedError(...
- `evaluation/perf_mem_eval.py:140` (python, conf: 0.9 (context: 0.2)): raise NotImplementedError(...
- ... and 1 more high-confidence items
