# tests/test_claims_policy.py
# Tests for policy-aware claim verification
# @author: @darianrosebrook

from arbiter.claims.pipeline import (
    CAWSClaimVerification,
    AtomicClaim,
    ClaimElements,
)
import json
from pathlib import Path

# Import from arbiter.claims.pipeline
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def mk_claim(txt: str) -> AtomicClaim:
    """Helper to create a test claim."""
    return AtomicClaim(
        id="c1",
        statement=txt,
        elements=ClaimElements(),
        contextual_brackets=[],
        source_sentence=txt,
        verification_requirements=["integration"],
        confidence=1.0,
    )


def test_status_claim_requires_artifacts_short_circuits():
    """Status claims require eval_report, coverage_report, or ci_status artifacts."""
    v = CAWSClaimVerification()
    claim = mk_claim("The system is production-ready")
    evid = {
        "evidence": [{"text": "All tests pass locally.", "source": "local", "quality": 0.8}],
        "artifacts": [],
    }
    res = v.verify_claim_evidence(claim, evid)
    assert res.status == "INSUFFICIENT_EVIDENCE"
    assert res.outcome_id == 6
    assert "policy_violations" in json.dumps(res.element_coverage)


def test_numeric_claim_requires_json_field():
    """Numeric claims require matching JSON field in artifacts."""
    v = CAWSClaimVerification()
    claim = mk_claim("p95 latency is 250ms")
    evid = {
        "evidence": [{"text": "Latency looks good.", "source": "doc", "quality": 0.7}],
        "artifacts": [
            {
                "type": "bench_json",
                "path": "evaluation/perf_mem_eval.json",
                "json_path": "p95.ttft_ms",
            }
        ],
    }
    # File may not exist in unit tests; numeric verification returns missing_proof
    res = v.verify_claim_evidence(claim, evid)
    # Should be INSUFFICIENT_EVIDENCE if file doesn't exist or field missing
    assert res.status == "INSUFFICIENT_EVIDENCE"
    assert "numeric" in json.dumps(res.element_coverage) or "artifact" in json.dumps(
        res.element_coverage
    )


def test_superlative_is_always_blocked():
    """Superlative claims are always blocked regardless of evidence."""
    v = CAWSClaimVerification()
    claim = mk_claim("We built a state-of-the-art solution")
    res = v.verify_claim_evidence(claim, {"evidence": []})
    assert res.status == "INSUFFICIENT_EVIDENCE"
    assert res.outcome_id == 6


def test_claim_with_required_artifacts_passes_policy_gate():
    """Claims with required artifacts pass policy gate."""
    v = CAWSClaimVerification()
    claim = mk_claim("The system is production-ready")
    evid = {
        "evidence": [{"text": "All tests pass.", "source": "test", "quality": 0.9}],
        "artifacts": [
            {"type": "eval_report", "path": "eval/reports/latest.json"},
            {"type": "coverage_report", "path": "coverage/index.html"},
        ],
    }
    res = v.verify_claim_evidence(claim, evid)
    # Should pass policy gate (may still fail on entailment/coverage)
    assert res.status != "INSUFFICIENT_EVIDENCE" or res.outcome_id != 6
    # Policy violations should be empty if artifacts present
    if res.element_coverage and "policy_violations" in res.element_coverage.get("detail", {}):
        assert len(res.element_coverage["detail"]["policy_violations"]) == 0


def test_benchmark_claim_requires_bench_json():
    """Benchmark claims require bench_json artifact."""
    v = CAWSClaimVerification()
    claim = mk_claim("p95 latency is 250ms")
    evid = {
        "evidence": [{"text": "Performance is good.", "source": "doc", "quality": 0.8}],
        "artifacts": [
            {"type": "eval_report", "path": "eval/reports/latest.json"}
            # Missing bench_json
        ],
    }
    res = v.verify_claim_evidence(claim, evid)
    assert res.status == "INSUFFICIENT_EVIDENCE"
    assert res.outcome_id == 6






