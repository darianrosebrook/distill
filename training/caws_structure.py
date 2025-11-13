"""
CAWS structure extraction and scoring.

Provides utilities for computing CAWS structure scores to measure
how well outputs match expected CAWS-compliant structure patterns.

Author: @darianrosebrook
"""

import re
from typing import List, Dict


def caws_structure_score(text: str) -> float:
    """
    Compute CAWS structure score.

    Checks:
    1. Required headers: "Working Spec", "Invariants", "Acceptance"
    2. Code blocks present (bonus)
    3. Fields non-empty (simple heuristic)

    Args:
        text: Text to score

    Returns:
        Structure score between 0.0 and 1.0
    """
    if not text or not text.strip():
        return 0.0

    text_lower = text.lower()

    # Check for required headers
    # These are common CAWS structure markers
    headers = ["working spec", "invariants", "acceptance"]
    header_count = sum(1 for h in headers if h in text_lower)
    header_score = header_count / len(headers) if len(headers) > 0 else 0.0

    # Check for code blocks (bonus for structured content)
    has_code = bool(re.search(r"```", text))
    code_bonus = 0.5 if has_code else 0.0

    # Check for structured content indicators
    # JSON-like structures
    has_json = bool(re.search(r"\{[^{}]*\}", text))
    # Lists (bullet or numbered)
    has_lists = bool(re.search(r"^[\s]*[-*]|\d+\.", text, re.MULTILINE))
    # Markdown headers
    has_markdown = bool(re.search(r"^#+\s", text, re.MULTILINE))

    structure_bonus = 0.0
    if has_json:
        structure_bonus += 0.15
    if has_lists:
        structure_bonus += 0.1
    if has_markdown:
        structure_bonus += 0.1

    # Simple field non-empty check
    # Check if text has substantial content (not just headers)
    word_count = len(text.split())
    field_score = 0.5  # Base score for having content
    if word_count > 50:
        field_score = 0.7  # More content = better
    elif word_count > 100:
        field_score = 0.9  # Substantial content
    elif word_count < 10:
        field_score = 0.2  # Too little content

    # Weighted combination
    total_score = (
        header_score * 0.4  # Headers are most important
        + code_bonus * 0.2  # Code blocks indicate structure
        + structure_bonus * 0.2  # Other structured content
        + field_score * 0.2  # Content completeness
    )

    return min(1.0, total_score)


def extract_caws_structure_elements(text: str) -> Dict[str, bool]:
    """
    Extract CAWS structure elements from text.

    Args:
        text: Text to analyze

    Returns:
        Dictionary with structure element flags
    """
    if not text:
        return {
            "has_working_spec": False,
            "has_invariants": False,
            "has_acceptance": False,
            "has_code_blocks": False,
            "has_json": False,
            "has_lists": False,
            "has_markdown": False,
        }

    text_lower = text.lower()

    return {
        "has_working_spec": "working spec" in text_lower,
        "has_invariants": "invariants" in text_lower,
        "has_acceptance": "acceptance" in text_lower,
        "has_code_blocks": bool(re.search(r"```", text)),
        "has_json": bool(re.search(r"\{[^{}]*\}", text)),
        "has_lists": bool(re.search(r"^[\s]*[-*]|\d+\.", text, re.MULTILINE)),
        "has_markdown": bool(re.search(r"^#+\s", text, re.MULTILINE)),
    }


def batch_caws_structure_score(texts: List[str]) -> Dict[str, float]:
    """
    Compute CAWS structure scores for a batch of texts.

    Args:
        texts: List of text strings

    Returns:
        Dictionary with:
        - mean_score: Average structure score
        - min_score: Minimum score
        - max_score: Maximum score
        - scores: List of individual scores
    """
    if not texts:
        return {
            "mean_score": 0.0,
            "min_score": 0.0,
            "max_score": 0.0,
            "scores": [],
        }

    scores = [caws_structure_score(text) for text in texts]

    return {
        "mean_score": sum(scores) / len(scores) if scores else 0.0,
        "min_score": min(scores) if scores else 0.0,
        "max_score": max(scores) if scores else 0.0,
        "scores": scores,
    }
