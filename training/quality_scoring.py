"""
Quality Scoring Utilities for Teacher Outputs

Provides utilities for computing quality scores for teacher model outputs.
These scores are used for self-evaluation head training.

Author: @darianrosebrook
"""

import re
from typing import Optional, Dict, List
import json


def compute_heuristic_quality_score(
    teacher_output: str,
    ground_truth: Optional[str] = None,
    prompt: Optional[str] = None,
) -> float:
    """
    Compute heuristic quality score for teacher output.

    Simple heuristic-based scoring that checks:
    - Structure (code blocks, lists, JSON)
    - Length appropriateness
    - Word overlap with ground truth (if provided)
    - Basic coherence indicators

    Args:
        teacher_output: Teacher model generated text
        ground_truth: Optional ground truth text for comparison
        prompt: Optional prompt for context

    Returns:
        Quality score between 0.0 and 1.0
    """
    if not teacher_output or not teacher_output.strip():
        return 0.0

    score = 0.5  # Base score

    # Check for structured content (code blocks, lists, JSON)
    has_code_blocks = "```" in teacher_output
    has_lists = "- " in teacher_output[:200] or "1. " in teacher_output[:200]
    has_json = "{" in teacher_output and "}" in teacher_output
    has_markdown = "##" in teacher_output or "###" in teacher_output

    if has_code_blocks or has_json:
        score += 0.2  # Structured content is good
    elif has_lists or has_markdown:
        score += 0.1

    # Check length appropriateness
    word_count = len(teacher_output.split())
    len(teacher_output)

    # Reward reasonable length (not too short, not too long)
    if 50 <= word_count <= 500:
        score += 0.15
    elif 20 <= word_count < 50:
        score += 0.05
    elif word_count < 10:
        score -= 0.3  # Too short
    elif word_count > 2000:
        score -= 0.1  # Possibly too verbose

    # Check for coherence indicators
    # Avoid repetitive patterns
    words = teacher_output.split()
    if len(words) > 10:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            score -= 0.2  # Too repetitive
        elif unique_ratio > 0.7:
            score += 0.05  # Good diversity

    # Check for common error patterns
    error_patterns = [
        r"undefined",
        r"error:",
        r"exception:",
        r"failed",
        r"cannot",
    ]
    error_count = sum(
        1 for pattern in error_patterns if re.search(pattern, teacher_output, re.IGNORECASE)
    )
    if error_count > 2:
        score -= 0.2

    # Ground truth comparison (if provided)
    if ground_truth:
        teacher_words = set(teacher_output.lower().split())
        gt_words = set(ground_truth.lower().split())

        if gt_words:
            # Word overlap ratio
            overlap = len(teacher_words & gt_words) / len(gt_words)
            score += overlap * 0.15

            # Length similarity
            length_ratio = min(len(teacher_output), len(ground_truth)) / max(
                len(teacher_output), len(ground_truth)
            )
            score += length_ratio * 0.05

    # Prompt relevance (if prompt provided)
    if prompt:
        prompt_words = set(prompt.lower().split())
        output_words = set(teacher_output.lower().split())

        # Check if output addresses prompt keywords
        if prompt_words:
            keyword_coverage = len(output_words & prompt_words) / len(prompt_words)
            score += keyword_coverage * 0.1

    return max(0.0, min(1.0, score))


def compute_json_validity_score(text: str) -> float:
    """
    Compute JSON validity score.

    Checks if text contains valid JSON and how well-formed it is.

    Args:
        text: Text to check for JSON

    Returns:
        Score between 0.0 and 1.0 (1.0 = valid JSON, 0.0 = no JSON or invalid)
    """
    # Try to find JSON in text
    json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
    matches = re.findall(json_pattern, text)

    if not matches:
        return 0.0

    # Try to parse each match
    valid_count = 0
    for match in matches:
        try:
            json.loads(match)
            valid_count += 1
        except json.JSONDecodeError:
            pass

    if not matches:
        return 0.0

    return valid_count / len(matches)


def compute_code_block_score(text: str) -> float:
    """
    Compute code block quality score.

    Checks for presence and quality of code blocks.

    Args:
        text: Text to check for code blocks

    Returns:
        Score between 0.0 and 1.0
    """
    code_block_pattern = r"```(\w+)?\n(.*?)```"
    matches = re.findall(code_block_pattern, text, re.DOTALL)

    if not matches:
        return 0.0

    score = 0.5  # Base score for having code blocks

    # Check for language specification
    languages = [lang for lang, _ in matches if lang]
    if languages:
        score += 0.2  # Language specified

    # Check code block length (not too short, not too long)
    for _, code in matches:
        lines = code.split("\n")
        if 5 <= len(lines) <= 100:
            score += 0.1
        elif len(lines) < 2:
            score -= 0.1

    return min(1.0, score / len(matches))


def compute_composite_quality_score(
    teacher_output: str,
    ground_truth: Optional[str] = None,
    prompt: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute composite quality score using multiple metrics.

    Args:
        teacher_output: Teacher model generated text
        ground_truth: Optional ground truth text
        prompt: Optional prompt for context
        weights: Optional weights for different metrics

    Returns:
        Composite quality score between 0.0 and 1.0
    """
    if weights is None:
        weights = {
            "heuristic": 0.6,
            "json_validity": 0.2,
            "code_blocks": 0.2,
        }

    scores = {}

    # Heuristic score
    scores["heuristic"] = compute_heuristic_quality_score(teacher_output, ground_truth, prompt)

    # JSON validity score
    scores["json_validity"] = compute_json_validity_score(teacher_output)

    # Code block score
    scores["code_blocks"] = compute_code_block_score(teacher_output)

    # Weighted average
    total_score = sum(
        scores.get(metric, 0.0) * weights.get(metric, 0.0) for metric in weights.keys()
    )

    total_weight = sum(weights.values())

    return total_score / total_weight if total_weight > 0 else 0.0


def batch_compute_quality_scores(
    teacher_outputs: List[str],
    ground_truths: Optional[List[str]] = None,
    prompts: Optional[List[str]] = None,
    method: str = "composite",
) -> List[float]:
    """
    Compute quality scores for a batch of teacher outputs.

    Args:
        teacher_outputs: List of teacher model outputs
        ground_truths: Optional list of ground truth texts
        prompts: Optional list of prompts
        method: Scoring method ("heuristic", "composite", "json_validity", "code_blocks")

    Returns:
        List of quality scores
    """
    scores = []

    for i, output in enumerate(teacher_outputs):
        ground_truth = ground_truths[i] if ground_truths and i < len(ground_truths) else None
        prompt = prompts[i] if prompts and i < len(prompts) else None

        if method == "heuristic":
            score = compute_heuristic_quality_score(output, ground_truth, prompt)
        elif method == "composite":
            score = compute_composite_quality_score(output, ground_truth, prompt)
        elif method == "json_validity":
            score = compute_json_validity_score(output)
        elif method == "code_blocks":
            score = compute_code_block_score(output)
        else:
            score = compute_heuristic_quality_score(output, ground_truth, prompt)

        scores.append(score)

    return scores


# Example usage
if __name__ == "__main__":
    # Test quality scoring
    test_output = """
    Here's a Python function to solve this:
    
    ```python
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    ```
    
    This function computes the nth Fibonacci number recursively.
    """

    score = compute_composite_quality_score(test_output)
    print(f"Quality score: {score:.2f}")

    # Test with ground truth
    ground_truth = "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"
    score_with_gt = compute_composite_quality_score(test_output, ground_truth=ground_truth)
    print(f"Quality score (with GT): {score_with_gt:.2f}")
