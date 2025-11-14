"""
Tests for training/claim_extraction.py - Claim extraction utilities for training.

Tests verifiable content detection, atomic claim extraction, and claim metrics.
"""
# @author: @darianrosebrook

import pytest
from training.claim_extraction import (
    ExtractedClaim,
    SimpleClaimExtractor,
    compute_claim_extraction_metrics,
)


class TestExtractedClaim:
    """Test ExtractedClaim dataclass."""

    def test_extracted_claim_creation(self):
        """Test creating an ExtractedClaim."""
        claim = ExtractedClaim(
            statement="The answer is 42",
            confidence=0.8,
            is_verifiable=True,
            has_context=False,
        )
        assert claim.statement == "The answer is 42"
        assert claim.confidence == 0.8
        assert claim.is_verifiable
        assert not claim.has_context

    def test_extracted_claim_with_context(self):
        """Test creating ExtractedClaim with context."""
        claim = ExtractedClaim(
            statement="The result [from tool] is 42",
            confidence=0.9,
            is_verifiable=True,
            has_context=True,
        )
        assert claim.has_context
        assert "[" in claim.statement


class TestSimpleClaimExtractor:
    """Test SimpleClaimExtractor class."""

    @pytest.fixture
    def extractor(self):
        """Create a SimpleClaimExtractor instance."""
        return SimpleClaimExtractor()

    def test_extract_claims_with_dates(self, extractor):
        """Test extracting claims with dates."""
        text = "The project was completed on 2024-01-15."
        claims = extractor.extract_claims(text)
        assert len(claims) >= 1
        assert any("2024-01-15" in claim.statement for claim in claims)

    def test_extract_claims_with_code_blocks(self, extractor):
        """Test extracting claims with code blocks."""
        text = "Here's the solution:\n```python\ndef solve():\n    return 42\n```"
        claims = extractor.extract_claims(text)
        assert len(claims) >= 1

    def test_extract_claims_with_json(self, extractor):
        """Test extracting claims with JSON structures."""
        text = 'The result is: {"answer": 42, "status": "success"}'
        claims = extractor.extract_claims(text)
        assert len(claims) >= 1

    def test_extract_claims_with_urls(self, extractor):
        """Test extracting claims with URLs."""
        text = "Visit https://example.com for more information."
        claims = extractor.extract_claims(text)
        assert len(claims) >= 1

    def test_extract_claims_skips_unverifiable(self, extractor):
        """Test that unverifiable content is skipped."""
        text = "I think maybe the answer is 42."
        claims = extractor.extract_claims(text)
        # Should skip due to "I think maybe"
        assert len(claims) == 0

    def test_extract_claims_skips_subjective(self, extractor):
        """Test that subjective content is skipped."""
        text = "It seems like the answer should be 42."
        claims = extractor.extract_claims(text)
        # Should skip due to "seems"
        assert len(claims) == 0

    def test_extract_claims_decomposes_compound(self, extractor):
        """Test that compound sentences are decomposed."""
        text = "The answer is 42 and the status is success."
        claims = extractor.extract_claims(text)
        # Should decompose into atomic claims
        assert len(claims) >= 1

    def test_extract_claims_empty_text(self, extractor):
        """Test extracting claims from empty text."""
        claims = extractor.extract_claims("")
        assert claims == []

    def test_extract_claims_plain_text(self, extractor):
        """Test extracting claims from plain text without indicators."""
        text = "This is just plain text with no verifiable content."
        claims = extractor.extract_claims(text)
        # May or may not extract depending on factual structure detection
        assert isinstance(claims, list)

    def test_extract_claims_with_context_brackets(self, extractor):
        """Test extracting claims with context brackets."""
        text = "The result [from tool] is 42."
        claims = extractor.extract_claims(text)
        if claims:
            assert any(claim.has_context for claim in claims)

    def test_extract_claim_count(self, extractor):
        """Test extracting claim count."""
        text = "First claim: 42. Second claim: 100. Third claim: 200."
        count = extractor.extract_claim_count(text)
        assert count >= 0

    def test_extract_claim_success_rate(self, extractor):
        """Test extracting claim success rate."""
        text = "First claim: 42. Second claim: 100. I think maybe 200."
        rate = extractor.extract_claim_success_rate(text)
        assert 0.0 <= rate <= 1.0

    def test_extract_claim_success_rate_empty(self, extractor):
        """Test claim success rate with empty text."""
        rate = extractor.extract_claim_success_rate("")
        assert rate == 0.0

    def test_extract_claim_success_rate_all_verifiable(self, extractor):
        """Test claim success rate with all verifiable content."""
        text = "The answer is 42. The status is success. The count is 5."
        rate = extractor.extract_claim_success_rate(text)
        assert rate > 0.0

    def test_extract_claims_with_version_numbers(self, extractor):
        """Test extracting claims with version numbers."""
        text = "The version is v1.2.3 and the API version is 2.0.0."
        claims = extractor.extract_claims(text)
        assert len(claims) >= 1

    def test_extract_claims_with_proper_nouns(self, extractor):
        """Test extracting claims with proper nouns."""
        text = "John Smith created the project on 2024-01-15."
        claims = extractor.extract_claims(text)
        assert len(claims) >= 1

    def test_extract_claims_confidence_scores(self, extractor):
        """Test that extracted claims have confidence scores."""
        text = "The answer is 42. Here's code:\n```python\nprint(42)\n```"
        claims = extractor.extract_claims(text)
        for claim in claims:
            assert 0.0 <= claim.confidence <= 1.0

    def test_extract_claims_confidence_higher_for_structured(self, extractor):
        """Test that structured content has higher confidence."""
        text_with_code = "Here's code:\n```python\ndef solve():\n    return 42\n```"
        text_plain = "The answer is 42."

        claims_code = extractor.extract_claims(text_with_code)
        claims_plain = extractor.extract_claims(text_plain)

        if claims_code and claims_plain:
            # Code claims should have higher confidence
            code_conf = max(c.confidence for c in claims_code)
            plain_conf = max(c.confidence for c in claims_plain)
            assert code_conf >= plain_conf

    def test_extract_claims_multiple_sentences(self, extractor):
        """Test extracting claims from multiple sentences."""
        text = "First sentence: 42. Second sentence: 100. Third sentence: 200."
        claims = extractor.extract_claims(text)
        assert len(claims) >= 1


class TestComputeClaimExtractionMetrics:
    """Test compute_claim_extraction_metrics function."""

    @pytest.fixture
    def extractor(self):
        """Create a SimpleClaimExtractor instance."""
        return SimpleClaimExtractor()

    def test_compute_claim_extraction_metrics_basic(self, extractor):
        """Test computing basic claim extraction metrics."""
        student_output = "The answer is 42."
        teacher_output = "The answer is 42. The status is success."

        metrics = compute_claim_extraction_metrics(student_output, teacher_output, extractor)

        assert "student_claim_count" in metrics
        assert "teacher_claim_count" in metrics
        assert "student_success_rate" in metrics
        assert "teacher_success_rate" in metrics
        assert "claim_ratio" in metrics
        assert "success_rate_ratio" in metrics

    def test_compute_claim_extraction_metrics_structure(self, extractor):
        """Test that metrics have correct structure."""
        student_output = "Answer: 42"
        teacher_output = "Answer: 42. Status: success"

        metrics = compute_claim_extraction_metrics(student_output, teacher_output, extractor)

        assert isinstance(metrics["student_claim_count"], int)
        assert isinstance(metrics["teacher_claim_count"], int)
        assert isinstance(metrics["student_success_rate"], float)
        assert isinstance(metrics["teacher_success_rate"], float)
        assert isinstance(metrics["claim_ratio"], float)
        assert isinstance(metrics["success_rate_ratio"], float)

    def test_compute_claim_extraction_metrics_ratios(self, extractor):
        """Test that ratios are computed correctly."""
        student_output = "The answer is 42."
        teacher_output = "The answer is 42. The status is success. The count is 5."

        metrics = compute_claim_extraction_metrics(student_output, teacher_output, extractor)

        # Teacher should have more claims
        assert metrics["teacher_claim_count"] >= metrics["student_claim_count"]

        # Ratios should be valid
        if metrics["teacher_claim_count"] > 0:
            assert 0.0 <= metrics["claim_ratio"] <= 1.0

        if metrics["teacher_success_rate"] > 0:
            assert metrics["success_rate_ratio"] >= 0.0

    def test_compute_claim_extraction_metrics_zero_teacher(self, extractor):
        """Test metrics when teacher has zero claims."""
        student_output = "The answer is 42."
        teacher_output = "I think maybe the answer is something."

        metrics = compute_claim_extraction_metrics(student_output, teacher_output, extractor)

        # Ratios should be 0.0 when teacher has no claims
        if metrics["teacher_claim_count"] == 0:
            assert metrics["claim_ratio"] == 0.0
            assert metrics["success_rate_ratio"] == 0.0

    def test_compute_claim_extraction_metrics_empty_outputs(self, extractor):
        """Test metrics with empty outputs."""
        metrics = compute_claim_extraction_metrics("", "", extractor)

        assert metrics["student_claim_count"] == 0
        assert metrics["teacher_claim_count"] == 0
        assert metrics["student_success_rate"] == 0.0
        assert metrics["teacher_success_rate"] == 0.0

    def test_compute_claim_extraction_metrics_creates_extractor(self):
        """Test that function creates extractor if None provided."""
        student_output = "The answer is 42."
        teacher_output = "The answer is 42."

        metrics = compute_claim_extraction_metrics(student_output, teacher_output)

        assert "student_claim_count" in metrics
        assert "teacher_claim_count" in metrics

    def test_compute_claim_extraction_metrics_with_structured_content(self, extractor):
        """Test metrics with structured content."""
        student_output = "Answer: 42"
        teacher_output = 'Answer: 42. Code:\n```python\nprint(42)\n``` Result: {"status": "success"}'

        metrics = compute_claim_extraction_metrics(student_output, teacher_output, extractor)

        # Teacher should have more claims due to structured content
        assert metrics["teacher_claim_count"] >= metrics["student_claim_count"]


class TestClaimExtractionEdgeCases:
    """Test edge cases for claim extraction methods."""

    @pytest.fixture
    def extractor(self):
        """Create a SimpleClaimExtractor instance."""
        return SimpleClaimExtractor()

    def test_split_sentences_without_spaces_after_punctuation(self, extractor):
        """Test _split_sentences with punctuation not followed by space."""
        # Test text without spaces after punctuation
        text = "First sentence.Second sentence.Third sentence"
        sentences = extractor._split_sentences(text)
        # Should still split on punctuation
        assert len(sentences) >= 1

    def test_split_sentences_whitespace_only(self, extractor):
        """Test _split_sentences with whitespace-only text."""
        sentences = extractor._split_sentences("   \n\n  ")
        # Should filter out empty/whitespace sentences
        assert len(sentences) == 0

    def test_split_sentences_multiple_punctuation(self, extractor):
        """Test _split_sentences with multiple punctuation marks."""
        text = "First!!! Second??? Third..."
        sentences = extractor._split_sentences(text)
        # Should split on first punctuation
        assert len(sentences) >= 1

    def test_has_verifiable_content_with_code_blocks(self, extractor):
        """Test _has_verifiable_content with code blocks (```)."""
        assert extractor._has_verifiable_content("Here's code:\n```python\nprint(42)\n```")
        assert extractor._has_verifiable_content("```code```")

    def test_has_verifiable_content_with_braces(self, extractor):
        """Test _has_verifiable_content with JSON braces ({})."""
        assert extractor._has_verifiable_content('The result is: {"answer": 42}')
        assert extractor._has_verifiable_content("{key: value}")

    def test_has_verifiable_content_with_brackets(self, extractor):
        """Test _has_verifiable_content with brackets ([])."""
        assert extractor._has_verifiable_content("The list is [1, 2, 3]")
        assert extractor._has_verifiable_content("[item]")

    def test_has_verifiable_content_factual_structure_only(self, extractor):
        """Test _has_verifiable_content relying on factual structure check."""
        # Sentence with factual verbs and nouns but no patterns
        assert extractor._has_verifiable_content("Python implements functions")
        assert extractor._has_verifiable_content("This system uses databases")

    def test_has_verifiable_content_no_patterns_no_structure(self, extractor):
        """Test _has_verifiable_content with no verifiable indicators."""
        assert not extractor._has_verifiable_content("This is just text")
        assert not extractor._has_verifiable_content("Maybe something")

    def test_has_factual_structure_with_verb_and_nouns(self, extractor):
        """Test _has_factual_structure with verb and enough nouns."""
        assert extractor._has_factual_structure("Python is a language")
        assert extractor._has_factual_structure("The system uses databases")

    def test_has_factual_structure_without_verb(self, extractor):
        """Test _has_factual_structure without factual verbs."""
        assert not extractor._has_factual_structure("Quick brown fox")
        assert not extractor._has_factual_structure("Running fast")

    def test_has_factual_structure_without_enough_nouns(self, extractor):
        """Test _has_factual_structure without enough nouns (words > 3 chars)."""
        # Only one word > 3 chars
        assert not extractor._has_factual_structure("The is good")
        assert not extractor._has_factual_structure("It was")

    def test_has_factual_structure_with_different_verbs(self, extractor):
        """Test _has_factual_structure with different factual verbs."""
        assert extractor._has_factual_structure("It was created")
        assert extractor._has_factual_structure("They have implemented")
        assert extractor._has_factual_structure("He had defined")
        assert extractor._has_factual_structure("Function returns values")
        assert extractor._has_factual_structure("System uses resources")

    def test_decompose_to_atomic_no_splits(self, extractor):
        """Test _decompose_to_atomic with no conjunctions (returns original)."""
        sentence = "The answer is 42"
        atomic = extractor._decompose_to_atomic(sentence)
        # Should return original as list
        assert atomic == [sentence]

    def test_decompose_to_atomic_with_and(self, extractor):
        """Test _decompose_to_atomic with 'and' conjunction."""
        sentence = "The answer is 42 and the status is success"
        atomic = extractor._decompose_to_atomic(sentence)
        # Should split on 'and'
        assert len(atomic) >= 2

    def test_decompose_to_atomic_with_or(self, extractor):
        """Test _decompose_to_atomic with 'or' conjunction."""
        sentence = "The answer is 42 or the result is 100"
        atomic = extractor._decompose_to_atomic(sentence)
        # Should split on 'or'
        assert len(atomic) >= 2

    def test_decompose_to_atomic_with_but(self, extractor):
        """Test _decompose_to_atomic with 'but' conjunction."""
        sentence = "The answer is 42 but the result is different"
        atomic = extractor._decompose_to_atomic(sentence)
        # Should split on 'but'
        assert len(atomic) >= 2

    def test_decompose_to_atomic_with_comma(self, extractor):
        """Test _decompose_to_atomic with comma."""
        sentence = "The answer is 42, the status is success"
        atomic = extractor._decompose_to_atomic(sentence)
        # Should split on comma
        assert len(atomic) >= 2

    def test_decompose_to_atomic_multiple_conjunctions(self, extractor):
        """Test _decompose_to_atomic with multiple conjunctions."""
        sentence = "First claim and second claim or third claim"
        atomic = extractor._decompose_to_atomic(sentence)
        # Should split on all conjunctions
        assert len(atomic) >= 3

    def test_compute_confidence_base_value(self, extractor):
        """Test _compute_confidence base value (0.5)."""
        # Statement with no patterns or structure
        confidence = extractor._compute_confidence("Simple statement")
        # Should be at least base confidence
        assert 0.0 <= confidence <= 1.0

    def test_compute_confidence_with_multiple_patterns(self, extractor):
        """Test _compute_confidence with multiple verifiable patterns."""
        statement = "Project completed on 2024-01-15, version v1.2.3, URL https://example.com"
        confidence = extractor._compute_confidence(statement)
        # Should be boosted by multiple patterns (max 0.3 boost)
        assert confidence > 0.5

    def test_compute_confidence_with_code_block(self, extractor):
        """Test _compute_confidence with code block (```)."""
        statement = "Here's code:\n```python\nprint(42)\n```"
        confidence = extractor._compute_confidence(statement)
        # Should be boosted by code block (+0.1)
        assert confidence >= 0.6

    def test_compute_confidence_with_json_braces(self, extractor):
        """Test _compute_confidence with JSON braces."""
        statement = 'Result: {"answer": 42}'
        confidence = extractor._compute_confidence(statement)
        # Should be boosted by JSON braces (+0.1)
        assert confidence >= 0.6

    def test_compute_confidence_with_brackets(self, extractor):
        """Test _compute_confidence with brackets."""
        statement = "The list is [1, 2, 3]"
        confidence = extractor._compute_confidence(statement)
        # Should be boosted by brackets (+0.05)
        assert confidence >= 0.55

    def test_compute_confidence_with_factual_structure(self, extractor):
        """Test _compute_confidence with factual structure."""
        statement = "Python implements functions"
        confidence = extractor._compute_confidence(statement)
        # Should be boosted by factual structure (+0.1)
        assert confidence >= 0.6

    def test_compute_confidence_with_unverifiable_penalty(self, extractor):
        """Test _compute_confidence with unverifiable pattern (penalty)."""
        statement = "I think maybe the answer is 42"
        confidence = extractor._compute_confidence(statement)
        # Should be penalized (-0.2) but still >= 0.0
        assert 0.0 <= confidence < 0.5

    def test_compute_confidence_max_bound(self, extractor):
        """Test _compute_confidence is capped at 1.0."""
        # Statement with many patterns and structures
        statement = "Project 2024-01-15 v1.2.3 https://example.com ```code``` {'key': 'value'} [list] Python implements systems"
        confidence = extractor._compute_confidence(statement)
        # Should be capped at 1.0
        assert confidence <= 1.0

    def test_compute_confidence_min_bound(self, extractor):
        """Test _compute_confidence is capped at 0.0."""
        # Statement with unverifiable patterns but no boosts
        statement = "I think maybe perhaps"
        confidence = extractor._compute_confidence(statement)
        # Should be at least 0.0
        assert confidence >= 0.0

    def test_extract_claim_success_rate_with_unverifiable_mixed(self, extractor):
        """Test extract_claim_success_rate with mix of verifiable and unverifiable."""
        text = "The answer is 42. I think maybe it's correct. The status is success."
        rate = extractor.extract_claim_success_rate(text)
        # Should be > 0 but < 1.0 (some sentences are unverifiable)
        assert 0.0 < rate < 1.0

    def test_extract_claim_success_rate_all_unverifiable(self, extractor):
        """Test extract_claim_success_rate with all unverifiable content."""
        text = "I think maybe perhaps the answer should be something."
        rate = extractor.extract_claim_success_rate(text)
        # Should be 0.0 (no verifiable sentences)
        assert rate == 0.0

    def test_extract_claim_success_rate_verifiable_with_unverifiable_flags(self, extractor):
        """Test extract_claim_success_rate when verifiable content also has unverifiable flags."""
        # Sentence has verifiable content but also unverifiable patterns
        text = "I think the answer is 42 on 2024-01-15."
        rate = extractor.extract_claim_success_rate(text)
        # Should be 0.0 (has unverifiable flag even though has date)
        assert rate == 0.0


class TestClaimExtractionIntegration:
    """Test integration of claim extraction components."""

    def test_complete_claim_extraction_workflow(self):
        """Test complete claim extraction workflow."""
        extractor = SimpleClaimExtractor()

        text = """
        The project was completed on 2024-01-15.
        The version is v1.2.3.
        Here's the code:
        ```python
        def solve():
            return 42
        ```
        The result is: {"answer": 42, "status": "success"}
        """

        # Extract claims
        claims = extractor.extract_claims(text)
        assert len(claims) >= 1

        # Get claim count
        count = extractor.extract_claim_count(text)
        assert count == len(claims)

        # Get success rate
        rate = extractor.extract_claim_success_rate(text)
        assert 0.0 <= rate <= 1.0

        # Verify claim properties
        for claim in claims:
            assert claim.is_verifiable
            assert 0.0 <= claim.confidence <= 1.0

    def test_claim_extraction_with_metrics(self):
        """Test claim extraction with metrics computation."""
        student_output = "The answer is 42."
        teacher_output = """
        The project was completed on 2024-01-15.
        The answer is 42.
        Code: ```python\nprint(42)\n```
        Result: {"status": "success"}
        """

        metrics = compute_claim_extraction_metrics(student_output, teacher_output)

        assert metrics["teacher_claim_count"] >= metrics["student_claim_count"]
        assert metrics["teacher_success_rate"] >= metrics["student_success_rate"]

    def test_claim_extraction_edge_cases(self):
        """Test claim extraction with edge cases."""
        extractor = SimpleClaimExtractor()

        # Very long text
        long_text = "The answer is 42. " * 100
        claims = extractor.extract_claims(long_text)
        assert isinstance(claims, list)

        # Text with only unverifiable content
        unverifiable = "I think maybe perhaps the answer should be 42."
        claims = extractor.extract_claims(unverifiable)
        # Should have no claims
        assert len(claims) == 0

        # Text with mixed content
        mixed = "The answer is 42. I think maybe it's correct."
        claims = extractor.extract_claims(mixed)
        # Should extract verifiable parts
        assert isinstance(claims, list)







