# TODO Resolution Plan

**Generated**: 2025-11-15 10:09:30

## Overview

This plan prioritizes TODO resolution by risk level and production impact.

## Resolution Milestones

### Milestone 1: Critical Blockers

**Priority**: IMMEDIATE
**Target**: Resolve all CRITICAL risk TODOs
**Estimated Effort**: 1-2 weeks

**TODOs to Resolve**:

1. **`conversion/convert_coreml.py`** (Line 473)
   - Risk: CRITICAL
   - Comment: ONNX is not a supported production path - always create placeholder...
   - Estimated Effort: 2-4 days
   - Dependencies: None
   - Acceptance Criteria: TODO resolved, tests pass, no production blockers

1. **`conversion/judge_export_coreml.py`** (Line 40)
   - Risk: CRITICAL
   - Comment: Convert Judge ONNX model to CoreML with INT8 quantization. Note: CoreMLTools does not natively support ONNX→CoreML conversion. For production, convert...
   - Estimated Effort: 2-4 days
   - Dependencies: None
   - Acceptance Criteria: TODO resolved, tests pass, no production blockers

1. **`coreml/runtime/ane_monitor.py`** (Line 189)
   - Risk: CRITICAL
   - Comment: This is a simplified heuristic - production would use more sophisticated analysis...
   - Estimated Effort: 2-4 days
   - Dependencies: None
   - Acceptance Criteria: TODO resolved, tests pass, no production blockers

### Milestone 2: High Priority TODOs

**Priority**: HIGH
**Target**: Resolve HIGH risk TODOs in critical paths
**Estimated Effort**: 2-3 weeks

**TODOs to Resolve**:

#### `conversion/convert_coreml.py` (3 TODOs)

1. **Line 11**
   - Comment: Convert ONNX → CoreML (mlprogram). Uses public MIL converter API. Usage: python -m conversion.convert_coreml \ --backend onnx \ --in onnx/toy.sanitize...
   - Estimated Effort: 1-3 days
   - Acceptance Criteria: TODO resolved, functionality verified

1. **Line 80**
   - Comment: Convert PyTorch model (TorchScript or ExportedProgram) to CoreML. Args: pytorch_model: TorchScript module or torch.export.ExportedProgram output_path:...
   - Estimated Effort: 1-3 days
   - Acceptance Criteria: TODO resolved, functionality verified

1. **Line 419**
   - Comment: Convert ONNX model to CoreML using public MIL converter API. Args: onnx_path: Path to ONNX model file output_path: Output path for .mlpackage compute_...
   - Estimated Effort: 1-3 days
   - Acceptance Criteria: TODO resolved, functionality verified

#### `coreml/ane_checks.py` (1 TODOs)

1. **Line 82**
   - Comment: Check for placeholder marker...
   - Estimated Effort: 1-3 days
   - Acceptance Criteria: TODO resolved, functionality verified

#### `coreml/probes/compare_probes.py` (2 TODOs)

1. **Line 36**
   - Comment: Run CoreML model inference. Assumes placeholder check already done in main()....
   - Estimated Effort: 1-3 days
   - Acceptance Criteria: TODO resolved, functionality verified

1. **Line 60**
   - Comment: Check for placeholder marker...
   - Estimated Effort: 1-3 days
   - Acceptance Criteria: TODO resolved, functionality verified

#### `coreml/runtime/ane_monitor.py` (1 TODOs)

1. **Line 131**
   - Comment: NOTE: This is an intentional fallback implementation, not a placeholder....
   - Estimated Effort: 1-3 days
   - Acceptance Criteria: TODO resolved, functionality verified

#### `training/examples_priority3_integration.py` (1 TODOs)

1. **Line 183**
   - Comment: Compute quality score for teacher output. This is a placeholder - in practice, you would use: - Human evaluation scores - Automated metrics (BLEU, ROU...
   - Estimated Effort: 1-3 days
   - Acceptance Criteria: TODO resolved, functionality verified

#### `training/losses.py` (1 TODOs)

1. **Line 892**
   - Comment: Combine penalties (sum with equal weights for now)...
   - Estimated Effort: 1-3 days
   - Acceptance Criteria: TODO resolved, functionality verified

#### `training/quant_qat_int8.py` (1 TODOs)

1. **Line 279**
   - Comment: For now, we'll use a simple approach: quantize the embedding weights...
   - Estimated Effort: 1-3 days
   - Acceptance Criteria: TODO resolved, functionality verified

#### `training/run_toy_distill.py` (1 TODOs)

1. **Line 313**
   - Comment: For now, weight all positions equally but this could be improved...
   - Estimated Effort: 1-3 days
   - Acceptance Criteria: TODO resolved, functionality verified

### Milestone 3: High-Risk "For Now" Implementations

**Priority**: MEDIUM
**Target**: Replace high-risk "for now" implementations with proper code
**Estimated Effort**: 3-4 weeks

**Total High-Risk "For Now" Instances**: 8

**Focus Areas**:

- **other**: 14 instances
- **data**: 3 instances
- **weights**: 2 instances
- **quantization**: 1 instances

**Action Items**:

1. Review each high-risk "for now" implementation
2. Document intended behavior vs current behavior
3. Implement proper solution
4. Add tests to verify correctness
5. Remove "for now" comment

## General Recommendations

### Immediate Actions

1. **Review CRITICAL TODOs**: Assess each CRITICAL TODO for actual production impact
2. **Document "For Now" Decisions**: For TODOs that will remain, document why and when they should be addressed
3. **Add Tests**: Ensure all TODO resolutions are covered by tests
4. **Track Progress**: Use this plan to track resolution progress

### Long-term Strategy

1. **Prevent Accumulation**: Establish code review practices to prevent new "for now" implementations
2. **Regular Audits**: Schedule quarterly TODO audits to track resolution progress
3. **Documentation**: Maintain documentation of temporary implementations and their intended replacements
4. **Automated Detection**: Consider adding linting rules to flag new "for now" patterns

