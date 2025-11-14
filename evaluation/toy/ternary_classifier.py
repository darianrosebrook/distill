#!/usr/bin/env python3
"""
Ternary classifier toy model evaluation configuration.

Token-based ternary classifier that outputs YES/NO/UNCERTAIN decisions for
"should we proceed?" questions based on provided evidence.

- Token ID 400: "YES" (should proceed)
- Token ID 401: "NO" (should not proceed)
- Token ID 402: "UNCERTAIN" (insufficient evidence)
"""

from typing import Dict, List
from pathlib import Path
import json
from evaluation.classification_eval import ClassificationConfig


# Ternary classifier answers
TERNARY_ANSWERS = [
    "YES",  # Token ID 400 - should proceed
    "NO",  # Token ID 401 - should not proceed
    "UNCERTAIN",  # Token ID 402 - insufficient evidence
]

# Token ID range for ternary classifier answers
TERNARY_TOKEN_START = 400
TERNARY_TOKEN_END = 402
TERNARY_TOKEN_IDS = list(range(TERNARY_TOKEN_START, TERNARY_TOKEN_END + 1))

# Create mappings
ID_TO_TERNARY_ANSWER: Dict[int, str] = {
    token_id: answer for token_id, answer in zip(TERNARY_TOKEN_IDS, TERNARY_ANSWERS)
}

TERNARY_ANSWER_TO_ID: Dict[str, int] = {
    answer: token_id for token_id, answer in ID_TO_TERNARY_ANSWER.items()
}


# Ternary classification configuration
TERNARY_CLASSIFIER_CONFIG = ClassificationConfig(
    name="ternary-classifier",
    class_names=TERNARY_ANSWERS,
    token_ids=TERNARY_TOKEN_IDS,
    id_to_name=ID_TO_TERNARY_ANSWER,
    name_to_id=TERNARY_ANSWER_TO_ID,
)


def load_ternary_eval_questions(eval_file: Path) -> List[str]:
    """Load evaluation questions for ternary classification."""
    if not eval_file.exists():
        # Create a default set for ternary classification with evidence
        default_questions = [
            "EVIDENCE: The code passes all unit tests with 100% coverage. All integration tests pass. No critical bugs reported. QUESTION: Should we proceed? ANSWER (YES or NO or UNCERTAIN):",
            "EVIDENCE: Multiple critical security vulnerabilities found in dependency. No fix available from vendor. Production system at risk. QUESTION: Should we proceed? ANSWER (YES or NO or UNCERTAIN):",
            "EVIDENCE: Performance benchmarks show 30% improvement. Memory usage optimized. All SLAs met. QUESTION: Should we proceed? ANSWER (YES or NO or UNCERTAIN):",
            "EVIDENCE: Only 40% test coverage. Several edge cases untested. Documentation incomplete. QUESTION: Should we proceed? ANSWER (YES or NO or UNCERTAIN):",
            "EVIDENCE: Feature implemented successfully. User acceptance testing passed. Ready for production. QUESTION: Should we proceed? ANSWER (YES or NO or UNCERTAIN):",
            "EVIDENCE: Database migration script tested successfully. Rollback plan documented. Data integrity verified. QUESTION: Should we proceed? ANSWER (YES or NO or UNCERTAIN):",
            "EVIDENCE: API response times increased by 150%. Memory leak detected. Performance regression identified. QUESTION: Should we proceed? ANSWER (YES or NO or UNCERTAIN):",
            "EVIDENCE: Code review completed with minor style issues. All functionality working correctly. QUESTION: Should we proceed? ANSWER (YES or NO or UNCERTAIN):",
            "EVIDENCE: Breaking API changes introduced. All clients require updates. Migration guide incomplete. QUESTION: Should we proceed? ANSWER (YES or NO or UNCERTAIN):",
            "EVIDENCE: Load testing completed successfully. System handles 5x expected load. Scalability verified. QUESTION: Should we proceed? ANSWER (YES or NO or UNCERTAIN):",
            "EVIDENCE: Unit tests failing. Build broken. Cannot proceed with deployment. QUESTION: Should we proceed? ANSWER (YES or NO or UNCERTAIN):",
            "EVIDENCE: All integration tests pass. End-to-end workflows verified. Quality gates passed. QUESTION: Should we proceed? ANSWER (YES or NO or UNCERTAIN):",
            "EVIDENCE: Third-party service dependency down. No fallback implemented. High availability risk. QUESTION: Should we proceed? ANSWER (YES or NO or UNCERTAIN):",
            "EVIDENCE: Code changes minimal and focused. Impact analysis completed. Safe to deploy. QUESTION: Should we proceed? ANSWER (YES or NO or UNCERTAIN):",
            "EVIDENCE: Database schema changes untested. Migration script incomplete. Data loss risk. QUESTION: Should we proceed? ANSWER (YES or NO or UNCERTAIN):",
            "EVIDENCE: Performance monitoring implemented. Alerting configured. Observability complete. QUESTION: Should we proceed? ANSWER (YES or NO or UNCERTAIN):",
            "EVIDENCE: API contract changes undocumented. Client applications may break. QUESTION: Should we proceed? ANSWER (YES or NO or UNCERTAIN):",
            "EVIDENCE: Security review completed. No vulnerabilities found. Compliance verified. QUESTION: Should we proceed? ANSWER (YES or NO or UNCERTAIN):",
            "EVIDENCE: Test environment unstable. Flaky tests detected. Reliability concerns. QUESTION: Should we proceed? ANSWER (YES or NO or UNCERTAIN):",
            "EVIDENCE: Feature flags implemented. Gradual rollout plan ready. Risk mitigation in place. QUESTION: Should we proceed? ANSWER (YES or NO or UNCERTAIN):",
        ]
        eval_file.parent.mkdir(parents=True, exist_ok=True)
        with open(eval_file, "w") as f:
            json.dump({"questions": default_questions}, f, indent=2)
        return default_questions

    with open(eval_file) as f:
        data = json.load(f)
    return data.get("questions", [])


def get_ternary_questions(
    eval_file: Path = Path("evaluation/ternary_eval_questions.json"),
) -> List[str]:
    """Get ternary classifier evaluation questions."""
    return load_ternary_eval_questions(eval_file)
