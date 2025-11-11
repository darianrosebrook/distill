# Distillation Roadmap: K2-Thinking → ANE-Friendly Student

## Overview

This roadmap prioritizes components by their relevance to the core distillation objective: **distill K2-Thinking into an ANE-friendly student that runs well on M1 Max**.

## Component Prioritization

### Essential for Distillation → CoreML (Keep Now)

**Critical Path Components:**

- **Student Architecture & Export Toolchain**
  - `models/student/architectures/gqa_transformer.py` - StudentLM with GQA, RoPE, RMSNorm, SwiGLU
  - `conversion/export_onnx.py` - ONNX export with enumerated shapes
  - `conversion/onnx_surgery.py` - ONNX graph cleanup utilities
  - `conversion/convert_coreml.py` - CoreML conversion
  - **Status**: ✅ Complete and ready

- **Quantization & Enumerated Shapes Strategy**
  - INT8 (or W4A8) quantization with enumerated contexts (4k/8k/16k)
  - `training/quant_qat_int8.py` - QAT modules with per-channel observers
  - **Status**: ✅ Scaffold complete, ready for training integration

- **Long-Context Plan**
  - GQA + KV cache budgets + prefill/decoder split
  - Decoder-only export with KV cache support
  - **Status**: ✅ Architecture supports long-context, export ready

- **Probe/Parity + Perf Harnesses**
  - `coreml/probes/*` - PyTorch↔CoreML parity checks
  - `evaluation/perf_mem_eval.py` - TTFA/tok-s/memory evaluation
  - **Status**: ✅ Complete, ready for validation

### Highly Useful for Model Behavior (Bring In During Training)

- **Process Supervision/Tool-Use Harness**
  - Tool JSON schema validation
  - Constrained decoding for tool calls
  - `training/distill_process.py` - Process supervision training
  - `coreml/runtime/constrained_decode.py` - JSON-constrained decoder
  - **Status**: ✅ Scaffold complete, integrate during first training cycle

### Valuable but Not Required to Finish Distillation (Phase In Later)

- **Judge Training Scaffold**
  - Pairwise CAWS scoring
  - Clause labeling
  - **When**: Stage 2, after student compiles and meets perf gates
  - **Status**: ✅ Complete scaffold, ready for Stage 2

- **CoreML Judge Swift Bridge + HTTP Shim**
  - C FFI for Rust/C++ integration
  - HTTP server for cross-process access
  - **When**: Stage 2, if running judge on-device from arbiter runtime
  - **Status**: ✅ Complete, optional for initial bring-up

- **Claim Extraction/CAWS Audit Machinery**
  - 4-stage claim extraction pipeline
  - CAWS compliance evaluation
  - **When**: Stage 2, for full governance loop
  - **Status**: ✅ Complete scaffold, orthogonal to distillation

## Suggested Sequence

### Stage 1: Distillation MVP (Current Focus)

**Goal**: Get a working student model that compiles to CoreML and meets performance gates.

1. **Lock Student Spec**
   - Model: 8-9B GQA
   - Architecture: GQA=8 KV heads, d_head=128
   - Context: 4k/8k/16k enumerated shapes
   - Precision: FP16 → INT8 (via QAT)

2. **Run 5-10k Prompt KD Pilot**
   - Sample from K2-Thinking teacher
   - Include small tool-trace set
   - Train with standard KD loss
   - Export → CoreML → Run probes and perf gates
   - **Acceptance**: Probe parity passes, perf meets targets

3. **Fold In Process Supervision**
   - Add tool-calling supervision
   - Train with process supervision loss
   - Re-export and re-probe
   - **Acceptance**: JSON validity ≥98%, tool-select ≥90%

4. **Introduce QAT**
   - Apply INT8 quantization-aware training
   - Re-validate probes/perf
   - **Acceptance**: Accuracy maintained, perf improved

### Stage 2: Governance & Operation (After MVP)

**Goal**: Add CAWS governance and runtime enforcement.

5. **Layer On Judge**
   - Train pairwise ranker on CAWS adjudication data
   - Export judge to CoreML
   - Integrate into arbiter loop
   - **Acceptance**: Pairwise accuracy ≥85%, clause F1 ≥0.85

6. **Add Swift Bridge** (Optional)
   - Deploy CoreML judge with C FFI
   - HTTP server for remote access
   - **When**: If needed for Rust/C++ arbiter runtime

7. **CAWS Audit Machinery** (Optional)
   - Claim extraction pipeline
   - Evidence verification
   - **When**: Building full governance loop

## File Organization by Stage

### Stage 1 (Distillation MVP)

```
models/student/architectures/gqa_transformer.py  ✅
conversion/export_onnx.py                        ✅
conversion/onnx_surgery.py                      ✅
conversion/convert_coreml.py                    ✅
coreml/probes/*                                  ✅
evaluation/perf_mem_eval.py                      ✅
training/quant_qat_int8.py                       ✅
training/distill_kd.py                           ✅
training/distill_process.py                      ✅
coreml/runtime/constrained_decode.py             ✅
```

### Stage 2 (Governance)

```
arbiter/judge_training/*                         ✅ (scaffold ready)
arbiter/swift/JudgeBridge/*                      ✅ (scaffold ready)
arbiter/claimify/*                               ✅ (scaffold ready)
arbiter/eval/caws_metrics.py                     ✅ (scaffold ready)
evaluation/caws_eval.py                          ✅ (scaffold ready)
```

## Acceptance Gates by Stage

### Stage 1 Gates

- **Architecture**: Model exports to ONNX/CoreML without errors
- **Probe Parity**: PyTorch↔CoreML intermediate activations match (rel_err < 2%)
- **Performance**: TTFA < target, tok/s > target, memory < budget
- **Tool Use**: JSON validity ≥98%, tool-select accuracy ≥90%
- **Long Context**: Needle retrieval ≥90% @ 16k context
- **Quantization**: INT8 maintains accuracy within tolerance

### Stage 2 Gates

- **Judge Accuracy**: Pairwise accuracy ≥85% on CAWS adjudication set
- **Clause Mapping**: F1 score ≥0.85 for CAWS clause classification
- **Claim Verification**: Precision ≥0.9, Recall ≥0.85
- **Runtime Latency**: Judge decision latency <50ms p95

## Current Status

- ✅ **Stage 1 Scaffold**: Complete
- ✅ **Stage 2 Scaffold**: Complete (ready for Stage 2)
- ⏳ **Stage 1 Training**: Ready to begin
- ⏳ **Stage 2 Integration**: Pending Stage 1 completion

## Next Immediate Steps

1. **Finalize Student Spec** - Lock architecture parameters
2. **Prepare KD Dataset** - Sample from K2-Thinking teacher
3. **Run Pilot Training** - 5-10k prompts, validate export pipeline
4. **Run Acceptance Gates** - Probe parity, perf, tool-use
5. **Iterate** - Add process supervision, QAT, re-validate

