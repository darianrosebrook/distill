# Security Remediation Plan

**Date**: 2025-01-14  
**Project**: Distill (kimi-student)  
**Status**: In Development  
**Priority**: Critical

---

## Executive Summary

This plan addresses 94 security issues identified by Bandit SAST scanning:

- **9 HIGH severity** issues (critical priority)
- **85 MEDIUM severity** issues (high priority)
- **4,749 LOW severity** issues (review and improve)

**Target**: Resolve all HIGH and MEDIUM severity issues before production deployment.

---

## Issue Categories & Remediation Strategy

### Category 1: Unsafe Model Loading (HIGH Priority)

#### Issue: Unsafe PyTorch Load

**Risk**: Arbitrary code execution via pickle deserialization  
**Count**: 1 confirmed, ~35 potential locations

**Affected Files**:

- `arbiter/judge_training/export_onnx.py:21` (confirmed HIGH)
- `scripts/inference_production.py:56`
- `runtime/orchestration/inference.py:297`
- `training/distill_kd.py:374`
- `training/distill_process.py:56`
- `training/quant_qat_int8.py:322`
- Additional files in `training/`, `evaluation/`, `scripts/`

**Remediation**:

```python
# BEFORE (unsafe)
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# AFTER (safe)
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
```

**Action Plan**:

1. Audit all `torch.load()` calls (35 files identified)
2. Add `weights_only=True` parameter to all checkpoint loads
3. For legacy checkpoints that require full state, document exception and add validation
4. Add unit tests verifying safe loading behavior

**Estimated Effort**: 2-3 hours

---

#### Issue: Unsafe Hugging Face Hub Downloads

**Risk**: Model tampering, supply chain attacks  
**Count**: 5 confirmed in `arbiter/judge_training/`, ~26 total potential

**Affected Files**:

- `arbiter/judge_training/runtime.py:14`
- `arbiter/judge_training/model.py:25`
- `arbiter/judge_training/export_onnx.py:28`
- `arbiter/judge_training/dataset.py:38`
- Additional files in `evaluation/`, `models/`, `scripts/`

**Remediation**:

```python
# BEFORE (unsafe - no revision pinning)
tokenizer = AutoTokenizer.from_pretrained(hf_name, use_fast=True)
model = AutoModel.from_pretrained(hf_name)

# AFTER (safe - with revision pinning)
# Option 1: Pin to specific commit SHA
tokenizer = AutoTokenizer.from_pretrained(
    hf_name,
    revision="abc123def456...",  # Specific commit SHA
    use_fast=True
)

# Option 2: Pin to specific tag (if available)
tokenizer = AutoTokenizer.from_pretrained(
    hf_name,
    revision="v1.0.0",  # Specific version tag
    use_fast=True
)

# Option 3: Use trust_remote_code=False explicitly (default, but be explicit)
tokenizer = AutoTokenizer.from_pretrained(
    hf_name,
    revision="main",  # At minimum, pin to branch
    trust_remote_code=False,
    use_fast=True
)
```

**Action Plan**:

1. Identify all model names used in `from_pretrained()` calls
2. For each model, determine appropriate revision strategy:
   - Production models: Pin to specific commit SHA
   - Development models: Pin to specific tag or branch
3. Create configuration file mapping model names to revisions
4. Update all `from_pretrained()` calls to use revision parameter
5. Add validation to ensure revisions are specified

**Estimated Effort**: 4-6 hours

---

### Category 2: Network Security (HIGH Priority)

#### Issue: HTTP Request Without Timeout

**Risk**: Denial of service, resource exhaustion  
**Count**: 1 confirmed, potential others

**Affected Files**:

- `capture/proxy_server.py:39` (confirmed)

**Remediation**:

```python
# BEFORE (unsafe)
async with httpx.AsyncClient(timeout=None) as client:
    upstream = await client.stream(method, url, headers=headers, content=body)

# AFTER (safe)
# For streaming requests, use reasonable timeouts
timeout = httpx.Timeout(
    connect=10.0,  # Connection timeout
    read=300.0,    # Read timeout (5 minutes for streaming)
    write=10.0,    # Write timeout
    pool=10.0      # Pool timeout
)
async with httpx.AsyncClient(timeout=timeout) as client:
    upstream = await client.stream(method, url, headers=headers, content=body)
```

**Action Plan**:

1. Audit all HTTP client usage (httpx, requests, aiohttp)
2. Add explicit timeouts to all HTTP requests
3. Configure timeouts based on operation type:
   - API calls: 30-60 seconds
   - File downloads: 300-600 seconds
   - Streaming: Connection timeout + read timeout per chunk
4. Add timeout configuration to environment variables

**Estimated Effort**: 2-3 hours

---

### Category 3: Error Handling (MEDIUM Priority)

#### Issue: Bare Except/Pass Blocks

**Risk**: Silent failures, security issues hidden  
**Count**: Majority of 4,749 LOW severity issues

**Remediation Strategy**:

```python
# BEFORE (problematic)
try:
    risky_operation()
except:
    pass  # Silent failure

# AFTER (improved)
try:
    risky_operation()
except SpecificException as e:
    logger.warning(f"Operation failed: {e}", exc_info=True)
    # Handle specific case or re-raise if critical
except Exception as e:
    logger.error(f"Unexpected error: {e}", exc_info=True)
    raise  # Re-raise if unexpected
```

**Action Plan**:

1. Categorize except/pass blocks by risk level:
   - **Critical**: Security-sensitive operations (input validation, auth)
   - **High**: Business logic operations (model loading, data processing)
   - **Medium**: Utility operations (file I/O, logging)
   - **Low**: Non-critical operations (optional features)
2. Prioritize fixing critical and high-risk blocks
3. For each block:
   - Identify specific exception types
   - Add appropriate logging
   - Implement proper error handling or re-raising
4. Create linting rule to prevent new bare except/pass blocks

**Estimated Effort**: 8-12 hours (prioritized approach)

---

## Implementation Phases

### Phase 1: Critical Fixes (Week 1)

**Goal**: Fix all HIGH severity issues

**Tasks**:

1. ✅ Fix unsafe `torch.load()` in `arbiter/judge_training/export_onnx.py`
2. ✅ Add `weights_only=True` to all checkpoint loads (35 files)
3. ✅ Fix HTTP timeout in `capture/proxy_server.py`
4. ✅ Pin Hugging Face model revisions (5 confirmed + audit others)
5. ✅ Add unit tests for safe loading patterns

**Acceptance Criteria**:

- All HIGH severity Bandit issues resolved
- Zero `torch.load()` calls without `weights_only=True`
- Zero HTTP clients without explicit timeouts
- All `from_pretrained()` calls use revision pinning

---

### Phase 2: Medium Priority Fixes (Week 2)

**Goal**: Address all MEDIUM severity issues

**Tasks**:

1. Audit and categorize all MEDIUM severity issues
2. Fix critical error handling patterns
3. Add input validation where missing
4. Review and improve exception handling
5. Add security-focused unit tests

**Acceptance Criteria**:

- All MEDIUM severity Bandit issues resolved or documented
- Critical error handling improved
- Input validation in place for security-sensitive operations

---

### Phase 3: Code Quality Improvements (Week 3-4)

**Goal**: Improve error handling patterns

**Tasks**:

1. Review and improve high-risk except/pass blocks
2. Add comprehensive logging for error conditions
3. Implement proper exception propagation
4. Create linting rules to prevent regressions

**Acceptance Criteria**:

- Critical and high-risk except/pass blocks improved
- Linting rules prevent new problematic patterns
- Error handling follows best practices

---

## Verification & Testing

### Security Testing Requirements

1. **Unit Tests**:

   - Test safe model loading with `weights_only=True`
   - Test revision pinning for Hugging Face models
   - Test HTTP timeout behavior
   - Test error handling paths

2. **Integration Tests**:

   - Test model loading from checkpoints
   - Test model downloads with revision pinning
   - Test HTTP client timeout behavior

3. **Security Scanning**:
   - Re-run Bandit after fixes
   - Verify HIGH and MEDIUM issues resolved
   - Track LOW severity issue reduction

---

## Configuration Management

### Model Revision Configuration

Create `configs/model_revisions.yaml`:

```yaml
model_revisions:
  # Production models - pin to specific commit SHA
  "microsoft/DialoGPT-medium":
    revision: "abc123def456..."
    trust_remote_code: false

  # Development models - pin to tag
  "bert-base-uncased":
    revision: "v1.0.0"
    trust_remote_code: false

  # Default fallback
  default:
    revision: "main"
    trust_remote_code: false
```

### Timeout Configuration

Add to environment configuration:

```python
# configs/timeouts.yaml or environment variables
HTTP_TIMEOUTS:
  connect: 10.0
  read: 60.0
  write: 10.0
  pool: 10.0

STREAMING_TIMEOUTS:
  connect: 10.0
  read: 300.0  # Longer for streaming
  write: 10.0
  pool: 10.0
```

---

## Risk Assessment

### High Risk Issues (Immediate Action Required)

- Unsafe PyTorch loading → Arbitrary code execution
- Unsafe Hugging Face downloads → Supply chain attacks
- HTTP requests without timeout → DoS vulnerability

### Medium Risk Issues (Address Before Production)

- Poor error handling → Silent failures, security issues hidden
- Missing input validation → Potential injection attacks
- Insecure configuration → Potential misconfiguration attacks

### Low Risk Issues (Continuous Improvement)

- Bare except/pass blocks → Code quality, maintainability
- Missing type hints → Code clarity, potential bugs
- Inconsistent error handling → Code quality

---

## Success Metrics

### Immediate Goals

- [ ] Zero HIGH severity Bandit issues
- [ ] Zero MEDIUM severity Bandit issues
- [ ] 100% of `torch.load()` calls use `weights_only=True`
- [ ] 100% of `from_pretrained()` calls use revision pinning
- [ ] 100% of HTTP clients have explicit timeouts

### Quality Metrics

- [ ] Security test coverage > 80% for critical paths
- [ ] All security fixes have corresponding unit tests
- [ ] Linting rules prevent security regressions
- [ ] Documentation updated with security best practices

---

## Maintenance & Prevention

### Pre-Commit Hooks

Add security checks to pre-commit hooks:

```bash
# Check for unsafe torch.load
grep -r "torch\.load(" --include="*.py" | grep -v "weights_only=True"

# Check for from_pretrained without revision
grep -r "from_pretrained(" --include="*.py" | grep -v "revision="

# Check for HTTP clients without timeout
grep -r "httpx\|requests\|aiohttp" --include="*.py" | grep -v "timeout="
```

### CI/CD Integration

- Run Bandit on every PR
- Block merges with HIGH or MEDIUM severity issues
- Track security issue trends over time

### Documentation

- Update development guidelines with security best practices
- Document model revision management process
- Create security checklist for new code

---

## Appendix: File Inventory

### Files Requiring torch.load() Fixes

- `arbiter/judge_training/export_onnx.py` (HIGH priority)
- `scripts/inference_production.py`
- `runtime/orchestration/inference.py`
- `training/distill_kd.py`
- `training/distill_process.py`
- `training/quant_qat_int8.py`
- `training/distill_tool_select.py`
- `training/distill_post_tool.py`
- `training/distill_answer_generation.py`
- `evaluation/tool_use_eval.py`
- `evaluation/reasoning_eval.py`
- Additional files in `scripts/`, `coreml/`, `tests/`

### Files Requiring from_pretrained() Fixes

- `arbiter/judge_training/runtime.py` (HIGH priority)
- `arbiter/judge_training/model.py` (HIGH priority)
- `arbiter/judge_training/export_onnx.py` (HIGH priority)
- `arbiter/judge_training/dataset.py` (HIGH priority)
- `evaluation/8ball_eval.py`
- `evaluation/tool_use_eval.py`
- `evaluation/classification_eval.py`
- `models/teacher/teacher_client.py`
- `scripts/inference_production.py`
- Additional files in `evaluation/`, `models/`, `scripts/`

### Files Requiring HTTP Timeout Fixes

- `capture/proxy_server.py` (HIGH priority)
- `eval/runners/openai_http.py` (has timeout, verify)
- `models/teacher/teacher_client.py` (has timeout, verify)
- Additional files using HTTP clients

---

**Next Steps**: Begin Phase 1 implementation, starting with HIGH severity issues.

**Report Generated**: 2025-01-14  
**Last Updated**: 2025-01-14
