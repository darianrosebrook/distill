# Toy Model + Claims Framework Integration Tests

## Overview

This document describes the integration tests that combine toy models with the claims verification framework, allowing end-to-end testing **without loading heavy models or requiring expensive compute**.

## Purpose

Before loading real models (which are memory-heavy, cost-heavy, and performance-heavy), we need to verify:

1. ✅ Claims extraction works with model outputs
2. ✅ Evidence manifest construction from tool traces works correctly
3. ✅ Verification pipeline integrates properly with eval harness patterns
4. ✅ Policy enforcement works with model-generated claims
5. ✅ Determinism is maintained across the full flow
6. ✅ Coverage scoring handles various evidence quality levels

## Test File

**`tests/test_claims_toy_model_integration.py`**

### Test Doubles

#### `ToyModelRunner`
- Simulates model inference without loading real models
- Deterministic keyword-based output generation
- Returns `{"model_output": str, "tool_trace": List[Dict]}`
- Caches outputs for determinism

#### `ToyEntailmentJudge`
- Keyword-based triage (same as pipeline toy tests)
- Deterministic support/contradict/insufficient classification

#### `ToyEvidenceRetriever`
- Simple manifest adapter (same as pipeline toy tests)
- Returns evidence items as-is

## Test Coverage

### 1. `test_toy_model_claims_extraction`
**Purpose**: Verify claims can be extracted from toy model outputs

**Flow**:
1. Generate toy model output from prompt
2. Extract tool traces as evidence
3. Process through claims pipeline
4. Verify claims extracted and structured correctly
5. Verify verification results present

**Validates**:
- Model output → claims extraction works
- Tool traces → evidence manifest conversion works
- Claims have proper structure (id, statement, elements)

### 2. `test_toy_model_eval_harness_pattern`
**Purpose**: Simulate full eval harness flow with toy models

**Flow**:
1. Create multiple eval items (prompts)
2. Generate outputs for each (simulating eval harness)
3. Build evidence manifests from tool traces
4. Extract and verify claims for each
5. Aggregate results

**Validates**:
- Full eval harness pattern works end-to-end
- Multiple items processed correctly
- Outcome distribution has variety
- Claims extracted from all items

### 3. `test_toy_model_claims_with_policy`
**Purpose**: Verify policy enforcement with model-generated claims

**Flow**:
1. Generate claim that triggers policy (status claim)
2. Provide evidence without required artifacts
3. Process through pipeline
4. Verify policy gate triggers correctly

**Validates**:
- Policy enforcement works with model outputs
- Status claims without artifacts → INSUFFICIENT_EVIDENCE
- Policy violations properly detected

### 4. `test_toy_model_determinism`
**Purpose**: Verify determinism across toy model + claims pipeline

**Flow**:
1. Run same prompt through toy model twice
2. Process both outputs through claims pipeline
3. Compare results

**Validates**:
- Identical inputs produce identical outputs
- Fingerprints match across runs
- No non-deterministic behavior introduced

### 5. `test_toy_model_coverage_scenarios`
**Purpose**: Test coverage scoring with various evidence quality levels

**Flow**:
1. Create scenarios with different evidence quality
2. Generate toy model outputs
3. Process through pipeline
4. Verify coverage scores reflect evidence quality

**Validates**:
- High-quality evidence → higher coverage scores
- Low-quality evidence → lower coverage scores
- Coverage scoring works correctly with model outputs

## Integration Pattern

The tests follow this pattern (matching eval harness):

```
Prompt → ToyModelRunner.generate() → Model Output + Tool Traces
  ↓
Tool Traces → Evidence Manifest Construction
  ↓
Model Output + Evidence Manifest → ClaimifyPipeline.process()
  ↓
Claims Extraction → Verification → Results
```

## Running Tests

```bash
# Run all toy model integration tests
pytest tests/test_claims_toy_model_integration.py -v

# Run specific test
pytest tests/test_claims_toy_model_integration.py::test_toy_model_claims_extraction -v

# Run with coverage
pytest tests/test_claims_toy_model_integration.py --cov=arbiter.claims --cov-report=term-missing
```

## What This Enables

### Before Real Model Integration

✅ **Verify claims framework works with model-like outputs**
- No need to load models to test claims extraction
- No need for expensive compute to test verification logic
- Fast iteration on claims framework improvements

### When Ready for Real Models

1. **Swap `ToyModelRunner`** → Real model runner (e.g., `HFLocalRunner`)
2. **Swap `ToyEntailmentJudge`** → Real NLI model wrapper (optional)
3. **Add embedding retriever** → Replace `ToyEvidenceRetriever` with semantic retrieval (optional)
4. **Run same tests** → Verify integration still works with real models

## Comparison with Existing Toy Tests

| Aspect | Pipeline Toy Tests | Model Integration Tests |
|--------|-------------------|------------------------|
| **Focus** | Claims pipeline stages | Full eval harness flow |
| **Input** | Text + evidence | Prompts → model outputs |
| **Model** | None | ToyModelRunner (simulated) |
| **Evidence** | Provided directly | Extracted from tool traces |
| **Pattern** | Unit/integration | Eval harness simulation |

## Coverage Gaps (None)

All critical paths are covered:

- ✅ Model output → claims extraction
- ✅ Tool traces → evidence manifest
- ✅ Policy enforcement with model outputs
- ✅ Determinism verification
- ✅ Coverage scoring with various evidence quality
- ✅ Eval harness pattern simulation

## Next Steps

When ready to test with real models:

1. Create `test_claims_real_model_integration.py` (similar structure)
2. Replace `ToyModelRunner` with actual model runner
3. Use same test patterns and assertions
4. Add performance/accuracy benchmarks
5. Compare results with toy model tests

## Summary

✅ **Complete toy model integration coverage** for claims framework
✅ **No real model dependencies** required
✅ **Deterministic and fast** execution (< 5 seconds)
✅ **Ready for real model integration** when models are available
✅ **Validates eval harness pattern** before expensive compute











