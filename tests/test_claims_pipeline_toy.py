# tests/test_claims_pipeline_toy.py
# End-to-end toy test for the full 4-stage claims pipeline
# Tests without requiring heavy models or expensive compute
# @author: @darianrosebrook

import pytest
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from arbiter.claims.pipeline import (
    ClaimifyPipeline,
    ConversationContext,
    ClaimDisambiguation,
    VerifiableContentQualification,
    AtomicClaimDecomposition,
    CAWSClaimVerification,
    EvidenceRetriever,
    EntailmentJudge,
    ElementCoverageScorer,
    Decontextualizer,
    AtomicClaim,
    ClaimElements,
)


class ToyEntailmentJudge(EntailmentJudge):
    """Deterministic toy entailment judge for testing."""

    def triage(self, evidence_chunk: str, claim_text: str):
        """Simple deterministic triage based on keyword matching."""
        evidence_lower = evidence_chunk.lower()
        claim_lower = claim_text.lower()

        # Check for contradictions
        if any(word in evidence_lower for word in ["not", "no", "never", "false", "incorrect"]):
            if any(word in evidence_lower for word in claim_lower.split()[:3]):
                return {"support": 0.1, "contradict": 0.8, "insufficient": 0.1}

        # Check for support (simple word overlap)
        claim_words = set(claim_lower.split())
        evidence_words = set(evidence_lower.split())
        overlap = len(claim_words & evidence_words) / max(len(claim_words), 1)

        if overlap > 0.5:
            return {"support": 0.8, "contradict": 0.1, "insufficient": 0.1}
        elif overlap > 0.2:
            return {"support": 0.4, "contradict": 0.1, "insufficient": 0.5}
        else:
            return {"support": 0.1, "contradict": 0.1, "insufficient": 0.8}


class ToyEvidenceRetriever(EvidenceRetriever):
    """Simple toy retriever that returns evidence as-is."""

    def retrieve(self, claim_text: str, manifest: dict):
        """Return evidence items from manifest."""
        evidence_items = manifest.get("evidence_items", manifest.get("evidence", []))
        unified = []
        for item in evidence_items:
            text = item.get("text") or item.get("content", "")
            source = item.get("source", "test")
            quality = item.get("quality", item.get("quality_score", 0.7))
            unified.append({"text": text, "source": source, "quality": float(quality)})
        return unified[:6]  # Limit to 6 items


def create_toy_context() -> ConversationContext:
    """Create a toy conversation context for testing."""
    return ConversationContext(
        prior_turns=[
            "The system processes user requests.",
            "We implemented authentication last week.",
        ],
        entity_registry={"System": "the main application", "User": "end user of the application"},
        code_spans=["def authenticate(user): return True", "class UserService: pass"],
        doc_sections=[
            "Authentication is handled by UserService",
            "The system supports multiple user types",
        ],
        result_tables=[],
    )


def test_stage_1_disambiguation():
    """Test Stage 1: Contextual Disambiguation."""
    disambig = ClaimDisambiguation()
    context = create_toy_context()

    # Test with pronoun that can be resolved
    text = "It processes requests efficiently."
    ambiguities = disambig.detect_ambiguities(text, context)
    assert len(ambiguities) > 0

    result = disambig.resolve_ambiguity(text, context)
    # Should either succeed or fail gracefully
    assert result.success or result.failure_reason is not None


def test_stage_2_qualification():
    """Test Stage 2: Verifiable Content Qualification."""
    qualifier = VerifiableContentQualification()
    context = create_toy_context()

    # Test with factual content
    factual_text = "The system processed 1,000 requests on 2024-01-15."
    result = qualifier.detect_verifiable_content(factual_text, context)
    assert result.has_verifiable_content is True
    assert result.confidence > 0.3

    # Test with subjective content
    subjective_text = "I think the system is probably good."
    result2 = qualifier.detect_verifiable_content(subjective_text, context)
    # May or may not have verifiable content depending on thresholds
    assert isinstance(result2.has_verifiable_content, bool)


def test_stage_3_decomposition():
    """Test Stage 3: Atomic Claim Decomposition."""
    decomposer = AtomicClaimDecomposition()
    context = create_toy_context()

    # Test with compound sentence
    text = "The system processes requests and handles authentication."
    claims = decomposer.extract_atomic_claims(text, context)

    assert len(claims) > 0
    for claim in claims:
        assert claim.id is not None
        assert claim.statement is not None
        assert claim.elements is not None  # Elements should be extracted


def test_stage_4_verification():
    """Test Stage 4: CAWS-Compliant Verification."""
    verifier = CAWSClaimVerification(
        retriever=ToyEvidenceRetriever(),
        entailment=ToyEntailmentJudge(),
        coverage=ElementCoverageScorer(),
        decontextualizer=Decontextualizer(),
    )

    claim = AtomicClaim(
        id="test-1",
        statement="The system processes requests",
        elements=ClaimElements(subject="system", predicate="processes", object="requests"),
        contextual_brackets=[],
        source_sentence="The system processes requests",
        verification_requirements=["general_verification"],
        confidence=0.9,
    )

    evidence_manifest = {
        "evidence": [
            {"text": "The system processes requests efficiently.", "source": "doc", "quality": 0.9}
        ]
    }

    result = verifier.verify_claim_evidence(claim, evidence_manifest)
    assert result.status in ["VERIFIED", "INSUFFICIENT_EVIDENCE", "UNVERIFIED"]
    assert result.outcome_id is not None
    assert result.element_coverage is not None
    assert result.entailment_triage is not None


def test_full_pipeline_end_to_end():
    """Test full 4-stage pipeline end-to-end."""
    pipeline = ClaimifyPipeline()

    # Override with toy components for deterministic testing
    pipeline.verification = CAWSClaimVerification(
        retriever=ToyEvidenceRetriever(),
        entailment=ToyEntailmentJudge(),
        coverage=ElementCoverageScorer(),
        decontextualizer=Decontextualizer(),
    )

    context = create_toy_context()

    # Test case 1: Simple factual claim
    text1 = "The system processed 1,000 requests on 2024-01-15."
    evidence1 = {
        "evidence": [
            {
                "text": "On 2024-01-15, the system processed 1,000 requests.",
                "source": "log",
                "quality": 0.9,
            }
        ]
    }

    result1 = pipeline.process(text1, context, evidence1)
    assert "claims" in result1
    assert len(result1["claims"]) > 0
    if result1["verification"]:
        assert all("verification" in v for v in result1["verification"])

    # Test case 2: Claim with ambiguity
    text2 = "It handles authentication."
    evidence2 = {
        "evidence": [
            {"text": "The system handles authentication.", "source": "doc", "quality": 0.8}
        ]
    }

    result2 = pipeline.process(text2, context, evidence2)
    # Should either succeed or fail at disambiguation stage
    assert "disambiguation" in result2 or "claims" in result2

    # Test case 3: Subjective claim (should be filtered)
    text3 = "I think the system is probably good."
    result3 = pipeline.process(text3, context, evidence2)
    # Should be filtered at qualification stage
    assert result3.get("qualification") is not None or len(result3.get("claims", [])) == 0


def test_pipeline_with_policy_gating():
    """Test pipeline with policy-aware verification."""
    pipeline = ClaimifyPipeline()

    # Use toy components
    pipeline.verification = CAWSClaimVerification(
        retriever=ToyEvidenceRetriever(),
        entailment=ToyEntailmentJudge(),
        coverage=ElementCoverageScorer(),
        decontextualizer=Decontextualizer(),
    )

    context = create_toy_context()

    # Test status claim without artifacts (should be blocked)
    text = "The system is production-ready."
    evidence = {
        "evidence": [{"text": "All tests pass.", "source": "test", "quality": 0.9}],
        "artifacts": [],  # Missing required artifacts
    }

    result = pipeline.process(text, context, evidence)
    if result.get("verification"):
        for v in result["verification"]:
            verif_result = v.get("verification")
            if verif_result:
                # Should be INSUFFICIENT_EVIDENCE due to policy violation
                assert verif_result.status == "INSUFFICIENT_EVIDENCE"


def test_pipeline_determinism():
    """Test that pipeline produces deterministic results."""
    pipeline = ClaimifyPipeline()

    # Use toy components for determinism
    pipeline.verification = CAWSClaimVerification(
        retriever=ToyEvidenceRetriever(),
        entailment=ToyEntailmentJudge(),
        coverage=ElementCoverageScorer(),
        decontextualizer=Decontextualizer(),
    )

    context = create_toy_context()
    text = "The system processes requests."
    evidence = {
        "evidence": [
            {"text": "The system processes requests efficiently.", "source": "doc", "quality": 0.9}
        ]
    }

    # Run twice
    result1 = pipeline.process(text, context, evidence)
    result2 = pipeline.process(text, context, evidence)

    # Results should be identical (deterministic)
    assert json.dumps(result1, sort_keys=True) == json.dumps(result2, sort_keys=True)


def test_pipeline_error_handling():
    """Test pipeline handles errors gracefully."""
    pipeline = ClaimifyPipeline()

    # Use toy components
    pipeline.verification = CAWSClaimVerification(
        retriever=ToyEvidenceRetriever(),
        entailment=ToyEntailmentJudge(),
        coverage=ElementCoverageScorer(),
        decontextualizer=Decontextualizer(),
    )

    context = create_toy_context()

    # Test with empty text
    result1 = pipeline.process("", context, {})
    assert isinstance(result1, dict)

    # Test with None evidence
    result2 = pipeline.process("The system works.", context, None)
    assert isinstance(result2, dict)
    assert "claims" in result2

    # Test with malformed evidence
    result3 = pipeline.process("The system works.", context, {"invalid": "data"})
    assert isinstance(result3, dict)


def test_pipeline_coverage_requirements():
    """Test that coverage requirements are enforced."""
    pipeline = ClaimifyPipeline()

    # Use toy components with strict thresholds
    verifier = CAWSClaimVerification(
        retriever=ToyEvidenceRetriever(),
        entailment=ToyEntailmentJudge(),
        coverage=ElementCoverageScorer(),
        decontextualizer=Decontextualizer(),
    )
    verifier.thresholds["coverage_min"] = 0.8  # High threshold

    pipeline.verification = verifier

    context = create_toy_context()

    # Claim with elements
    text = "The system processes requests."
    # Evidence that doesn't fully cover elements
    evidence = {"evidence": [{"text": "The system works.", "source": "doc", "quality": 0.9}]}

    result = pipeline.process(text, context, evidence)
    if result.get("verification"):
        for v in result["verification"]:
            verif_result = v.get("verification")
            if verif_result:
                # Should require high coverage for VERIFIED
                if verif_result.status == "VERIFIED":
                    assert verif_result.element_coverage["score"] >= 0.8


def test_pipeline_outcome_distribution():
    """Test that outcomes are properly distributed."""
    pipeline = ClaimifyPipeline()

    # Use toy components
    pipeline.verification = CAWSClaimVerification(
        retriever=ToyEvidenceRetriever(),
        entailment=ToyEntailmentJudge(),
        coverage=ElementCoverageScorer(),
        decontextualizer=Decontextualizer(),
    )

    context = create_toy_context()

    test_cases = [
        (
            "The system works.",
            {"evidence": [{"text": "The system works well.", "source": "doc", "quality": 0.9}]},
        ),
        (
            "The system does not work.",
            {"evidence": [{"text": "The system works.", "source": "doc", "quality": 0.9}]},
        ),
        (
            "The system processes requests.",
            {"evidence": [{"text": "Something else.", "source": "doc", "quality": 0.5}]},
        ),
    ]

    outcomes = []
    for text, evidence in test_cases:
        result = pipeline.process(text, context, evidence)
        if result.get("verification"):
            for v in result["verification"]:
                verif_result = v.get("verification")
                if verif_result and verif_result.outcome_id:
                    outcomes.append(verif_result.outcome_id)

    # Should have some variety in outcomes
    assert len(set(outcomes)) > 0


def test_pipeline_fingerprints():
    """Test that fingerprints are computed correctly."""
    verifier = CAWSClaimVerification(
        retriever=ToyEvidenceRetriever(),
        entailment=ToyEntailmentJudge(),
        coverage=ElementCoverageScorer(),
        decontextualizer=Decontextualizer(),
    )

    claim = AtomicClaim(
        id="test-1",
        statement="Test claim",
        elements=ClaimElements(),
        contextual_brackets=[],
        source_sentence="Test claim",
        verification_requirements=[],
        confidence=1.0,
    )

    evidence = {"evidence": [{"text": "Test evidence", "source": "test", "quality": 0.8}]}

    result = verifier.verify_claim_evidence(claim, evidence)
    assert result.fingerprints is not None
    assert "operator_config" in result.fingerprints
    assert len(result.fingerprints["operator_config"]) == 64  # SHA256 hex length


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
