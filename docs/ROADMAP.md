# ROADMAP.md (distillation-MVP vs arbiter-next)

## Objectives

- **MVP**: Train/convert a K2-distilled student (8–9B, GQA) to CoreML with enumerated shapes (4k/8k/16k), validated by probes and perf gates.

- **Arbiter-next**: Add judge council + CAWS gates, Swift bridge, and task-specific runners.

## Work Buckets

### ✅ Distillation-MVP (keep now)

- `models/student/architectures/gqa_transformer.py` (student core)
- `training/distill_kd.py` (KD loop) + process-supervision harness (tool JSON)
- `conversion/export_onnx.py` + `conversion/onnx_surgery.py`
- `coreml/probes/*` + `coreml/ane_checks.py`
- `evaluation/perf_mem_eval.py` (TTFA/tok-s/mem) + tool-use eval
- Quantization/QAT modules; long-context curriculum (2k→4k→8k→16k)

### ⬇ Arbiter-next (phase in)

- Judge training (pairwise ranker + clause tags)
- CoreML Judge Swift bridge (C FFI + HTTP)
- Claim extraction/CAWS gates & provenance ledger
- Speculative decoding (1–2B draft) + router

## Sequence (suggested)

1. **Spec lock**: Student 9B GQA (8 KV heads, d_head=128) with 4k/8k/16k enumerations.

2. **KD pilot (5–10k prompts)** → export→CoreML→probes→perf gates.

3. **Add process supervision** (tool JSON + constrained decoding) → re-export/probe.

4. **QAT to INT8** → re-export/probe; verify long-ctx.

5. **Prefill/decoder split exports** (optional) for 16k decode.

6. **Introduce judge** + Swift bridge; wire CAWS quality gates.

