# Judge Training Quick Start Guide

## 10-Minute Smoke Bring-Up

### 1. Seed Data

Generate smoke test data for quick validation:

```bash
make judge_smoke
```

This writes `data/judge/train.jsonl` and `data/judge/val.jsonl` with example pairwise comparisons.

### 2. Train the Judge

Train the judge model with pairwise ranking and clause labeling:

```bash
make judge_train
```

Watch logs for `val_pairwise_acc` climbing above 0.75 on the toy set. Training will:
- Use pairwise ranking loss (margin-based)
- Apply clause labeling loss (BCE multi-label)
- Save checkpoint to `arbiter/judge_training/artifacts/judge.pt`

### 3. Export & Convert

Export to ONNX with enumerated shapes:

```bash
make judge_onnx
```

This exports models for sequence lengths 256, 512, and 1024 to `arbiter/judge_training/artifacts/onnx/`.

Convert to CoreML:

```bash
make judge_coreml
```

This converts the ONNX model to CoreML format at `arbiter/judge_training/artifacts/coreml/judge.mlpackage`.

### 4. Sanity Check & Latency

Test the judge CLI:

```bash
make judge_cli PROMPT='Fix off-by-one' A='adds test + fix' B='rename only'
```

Benchmark latency on your M1 Max:

```bash
make judge_bench
```

This reports p50, p95, and mean decision latency in milliseconds.

## Promote to Real CAWS Data

Convert your CAWS adjudication logs to pairwise training format:

```bash
make judge_pairs_from_caws IN=data/caws_logs.jsonl OUT=data/judge/train.jsonl
```

Input format (JSONL):
```json
{
  "id": "job-123",
  "prompt": "...",
  "candidates": [
    {"text": "...", "clauses": ["EVIDENCE_COMPLETENESS"], "score": 0.78},
    {"text": "...", "clauses": ["BUDGET_ADHERENCE"], "score": 0.55}
  ],
  "winner_index": 0
}
```

Then retrain and export as above:

```bash
make judge_train
make judge_onnx
make judge_coreml
```

## Wire into the Arbiter Loop (Minimal)

Use the CoreML judge in your arbiter runtime:

```python
from arbiter.judge_training.runtime import CoreMLJudge

judge = CoreMLJudge(
    mlpackage_path="arbiter/judge_training/artifacts/coreml/judge.mlpackage",
    hf_name="microsoft/deberta-v3-small",
    clauses=["EVIDENCE_COMPLETENESS", "BUDGET_ADHERENCE", "GATE_INTEGRITY", 
             "PROVENANCE_CLARITY", "WAIVER_JUSTIFICATION"]
)

# Compare two candidates
result = judge.compare(prompt, candidate_a, candidate_b)

# Gate acceptance by:
# 1. Pairwise verdict (A/B/TIE)
if result["verdict"] == "A":
    # Candidate A preferred
    
# 2. Clause probabilities exceeding thresholds per CAWS article
for clause, prob in result["A"]["clauses"]:
    if prob > threshold_for_clause(clause):
        # Clause satisfied
        
# 3. Deterministic checks (schemas, tests) before final PASS
if passes_deterministic_checks(candidate_a):
    verdict = "PASS"
```

## Integration Checklist

- [ ] Smoke test data generated
- [ ] Judge model trained (val_pairwise_acc > 0.75)
- [ ] ONNX export successful
- [ ] CoreML conversion successful
- [ ] CLI test passes
- [ ] Latency benchmark acceptable (<50ms p95)
- [ ] CAWS logs converted to training pairs
- [ ] Retrained on real data
- [ ] Integrated into arbiter loop
- [ ] Clause thresholds configured
- [ ] Deterministic checks wired

## Next Steps

Once the basic workflow is validated:

1. **Add Calibration**: Temperature/Platt calibration + clause-specific thresholds
2. **REST/RPC Interface**: Surface a tiny REST/RPC around the CoreML judge for Rust/C++ callers
3. **Swift Bridge**: Use the Swift bridge (`arbiter/swift/JudgeBridge`) to host the `.mlpackage` directly and expose C FFI for the arbiter

See `arbiter/swift/JudgeBridge/README.md` for Swift bridge usage.

