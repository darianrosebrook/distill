"""
Claim extraction utilities for training dataset generation and loss computation.

Provides simplified claim extraction for training purposes, focusing on:
- Verifiable content detection
- Atomic claim extraction
- Claim extraction success rate measurement

Reference: CLAIM_EXTRACTION_SKEPTICISM_GUARD_RAILS.md
@author: @darianrosebrook
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ExtractedClaim:
    """Simplified claim representation for training."""

    statement: str
    confidence: float
    is_verifiable: bool
    has_context: bool


class SimpleClaimExtractor:
    """
    Simplified claim extractor for training purposes.

    Detects verifiable claims using heuristics:
    - Factual indicators (dates, quantities, code references)
    - Structured content (code blocks, lists, JSON)
    - Atomic statements (single facts, not compound)
    """

    # Patterns indicating verifiable content
    VERIFIABLE_PATTERNS = [
        r"\d{4}-\d{2}-\d{2}",  # Dates
        r"\d+\.\d+",  # Versions/numbers
        r"v\d+",  # Version numbers
        r"#[A-Za-z0-9]+",  # Code references
        r"```[\s\S]*?```",  # Code blocks
        r"\{[\s\S]*?\}",  # JSON-like structures
        r"\[[\s\S]*?\]",  # Array-like structures
        r"https?://[^\s]+",  # URLs
        r"[A-Z][a-z]+ [A-Z][a-z]+",  # Proper nouns
    ]

    # Patterns indicating unverifiable/subjective content
    UNVERIFIABLE_PATTERNS = [
        r"\b(I think|I believe|I feel|in my opinion)\b",
        r"\b(probably|maybe|perhaps|might|could)\b",
        r"\b(seems|appears|looks like)\b",
        r"\b(should|ought to|must)\b",  # Prescriptive without evidence
    ]

    def extract_claims(self, text: str) -> List[ExtractedClaim]:
        """
        Extract atomic claims from text.

        Args:
            text: Input text to extract claims from

        Returns:
            List of extracted claims with confidence scores
        """
        claims = []

        # Split into sentences
        sentences = self._split_sentences(text)

        for sentence in sentences:
            # Check if sentence has verifiable content
            is_verifiable = self._has_verifiable_content(sentence)

            if not is_verifiable:
                continue

            # Check for unverifiable patterns (skip if found)
            if self._has_unverifiable_content(sentence):
                continue

            # Extract atomic claims (split on conjunctions, etc.)
            atomic_statements = self._decompose_to_atomic(sentence)

            for statement in atomic_statements:
                # Check if statement has context brackets
                has_context = "[" in statement and "]" in statement

                # Compute confidence based on indicators
                confidence = self._compute_confidence(statement)

                claims.append(
                    ExtractedClaim(
                        statement=statement.strip(),
                        confidence=confidence,
                        is_verifiable=True,
                        has_context=has_context,
                    )
                )

        return claims

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting (can be improved)
        sentences = re.split(r"[.!?]\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _has_verifiable_content(self, sentence: str) -> bool:
        """Check if sentence contains verifiable content."""
        # Check for verifiable patterns
        for pattern in self.VERIFIABLE_PATTERNS:
            if re.search(pattern, sentence, re.IGNORECASE):
                return True

        # Check for structured content
        if "```" in sentence or "{" in sentence or "[" in sentence:
            return True

        # Check for factual statements (has subject + verb + object)
        if self._has_factual_structure(sentence):
            return True

        return False

    def _has_unverifiable_content(self, sentence: str) -> bool:
        """Check if sentence contains unverifiable/subjective content."""
        for pattern in self.UNVERIFIABLE_PATTERNS:
            if re.search(pattern, sentence, re.IGNORECASE):
                return True
        return False

    def _has_factual_structure(self, sentence: str) -> bool:
        """Check if sentence has factual structure (simplified)."""
        # Check for common factual patterns
        factual_verbs = [
            "is",
            "was",
            "are",
            "were",
            "has",
            "have",
            "had",
            "implements",
            "defines",
            "creates",
            "returns",
            "uses",
        ]

        words = sentence.lower().split()
        has_verb = any(verb in words for verb in factual_verbs)
        has_noun = len([w for w in words if len(w) > 3]) >= 2

        return has_verb and has_noun

    def _decompose_to_atomic(self, sentence: str) -> List[str]:
        """
        Decompose sentence into atomic claims.

        Splits on conjunctions, commas, etc.
        """
        # Split on common conjunctions
        parts = re.split(r"\s+(and|or|but|,)\s+", sentence)

        # Filter out conjunctions themselves
        atomic = [
            p.strip() for p in parts if p.strip() and p.lower() not in ["and", "or", "but", ","]
        ]

        # If no splits, return original
        if len(atomic) <= 1:
            return [sentence]

        return atomic

    def _compute_confidence(self, statement: str) -> float:
        """
        Compute confidence score for claim (0.0 to 1.0).

        Higher confidence for:
        - More verifiable indicators
        - Structured content
        - Clear factual statements
        """
        confidence = 0.5  # Base confidence

        # Boost for verifiable patterns
        pattern_count = sum(
            1 for p in self.VERIFIABLE_PATTERNS if re.search(p, statement, re.IGNORECASE)
        )
        confidence += min(pattern_count * 0.1, 0.3)

        # Boost for structured content
        if "```" in statement:
            confidence += 0.1
        if "{" in statement and "}" in statement:
            confidence += 0.1
        if "[" in statement and "]" in statement:
            confidence += 0.05

        # Boost for factual structure
        if self._has_factual_structure(statement):
            confidence += 0.1

        # Penalize for unverifiable patterns
        if self._has_unverifiable_content(statement):
            confidence -= 0.2

        return max(0.0, min(1.0, confidence))

    def extract_claim_count(self, text: str) -> int:
        """Get count of extractable claims."""
        claims = self.extract_claims(text)
        return len(claims)

    def extract_claim_success_rate(self, text: str) -> float:
        """
        Compute claim extraction success rate.

        Returns ratio of verifiable sentences to total sentences.
        """
        sentences = self._split_sentences(text)
        if not sentences:
            return 0.0

        verifiable_count = sum(
            1
            for s in sentences
            if self._has_verifiable_content(s) and not self._has_unverifiable_content(s)
        )
        return verifiable_count / len(sentences) if sentences else 0.0


def compute_claim_extraction_metrics(
    student_output: str, teacher_output: str, extractor: Optional[SimpleClaimExtractor] = None
) -> Dict[str, Any]:
    """
    Compute claim extraction metrics comparing student and teacher outputs.

    Args:
        student_output: Student model output text
        teacher_output: Teacher model output text
        extractor: Optional claim extractor (creates new if None)

    Returns:
        Dictionary with metrics:
        - student_claim_count: Number of claims extracted from student output
        - teacher_claim_count: Number of claims extracted from teacher output
        - student_success_rate: Claim extraction success rate for student
        - teacher_success_rate: Claim extraction success rate for teacher
        - claim_ratio: student_claim_count / teacher_claim_count (if teacher > 0)
        - success_rate_ratio: student_success_rate / teacher_success_rate (if teacher > 0)
    """
    if extractor is None:
        extractor = SimpleClaimExtractor()

    student_claims = extractor.extract_claims(student_output)
    teacher_claims = extractor.extract_claims(teacher_output)

    student_count = len(student_claims)
    teacher_count = len(teacher_claims)

    student_success_rate = extractor.extract_claim_success_rate(student_output)
    teacher_success_rate = extractor.extract_claim_success_rate(teacher_output)

    claim_ratio = student_count / teacher_count if teacher_count > 0 else 0.0
    success_rate_ratio = (
        student_success_rate / teacher_success_rate if teacher_success_rate > 0 else 0.0
    )

    return {
        "student_claim_count": student_count,
        "teacher_claim_count": teacher_count,
        "student_success_rate": student_success_rate,
        "teacher_success_rate": teacher_success_rate,
        "claim_ratio": claim_ratio,
        "success_rate_ratio": success_rate_ratio,
    }
