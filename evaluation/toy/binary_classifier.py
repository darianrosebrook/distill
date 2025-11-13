#!/usr/bin/env python3
"""
Binary classifier toy model evaluation configuration.

Token-based binary classifier that outputs YES/NO decisions for "should we proceed?" questions.

- Token ID 300: "YES" (should proceed)
- Token ID 301: "NO" (should not proceed)
"""

from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path
import json
from evaluation.classification_eval import ClassificationConfig


# Binary classifier answers
BINARY_ANSWERS = [
    "YES",  # Token ID 300 - should proceed
    "NO",   # Token ID 301 - should not proceed
]

# Token ID range for binary classifier answers
BINARY_TOKEN_START = 300
BINARY_TOKEN_END = 301
BINARY_TOKEN_IDS = list(range(BINARY_TOKEN_START, BINARY_TOKEN_END + 1))

# Create mappings
ID_TO_BINARY_ANSWER: Dict[int, str] = {
    token_id: answer for token_id, answer in zip(BINARY_TOKEN_IDS, BINARY_ANSWERS)
}

BINARY_ANSWER_TO_ID: Dict[str, int] = {
    answer: token_id for token_id, answer in ID_TO_BINARY_ANSWER.items()
}


# Binary classification configuration
BINARY_CLASSIFIER_CONFIG = ClassificationConfig(
    name="binary-classifier",
    class_names=BINARY_ANSWERS,
    token_ids=BINARY_TOKEN_IDS,
    id_to_name=ID_TO_BINARY_ANSWER,
    name_to_id=BINARY_ANSWER_TO_ID,
)


def load_binary_eval_questions(eval_file: Path) -> List[str]:
    """Load evaluation questions for binary classification."""
    if not eval_file.exists():
        # Create a default set for binary classification with evidence
        default_questions = [
            "EVIDENCE: The code passes all unit tests with 100% coverage. All integration tests pass. No critical bugs reported. QUESTION: Should we proceed? ANSWER (YES or NO):",
            "EVIDENCE: Multiple critical security vulnerabilities found in dependency. No fix available from vendor. Production system at risk. QUESTION: Should we proceed? ANSWER (YES or NO):",
            "EVIDENCE: Performance benchmarks show 50% improvement over previous version. Memory usage reduced by 30%. All SLAs met. QUESTION: Should we proceed? ANSWER (YES or NO):",
            "EVIDENCE: Only 60% test coverage. Several edge cases not tested. Documentation incomplete. QUESTION: Should we proceed? ANSWER (YES or NO):",
            "EVIDENCE: Feature implemented successfully. User acceptance testing passed. Ready for production deployment. QUESTION: Should we proceed? ANSWER (YES or NO):",
            "EVIDENCE: Database migration script tested successfully. Rollback plan documented. Data integrity verified. QUESTION: Should we proceed? ANSWER (YES or NO):",
            "EVIDENCE: API response times increased by 200%. Memory leak detected. Performance regression. QUESTION: Should we proceed? ANSWER (YES or NO):",
            "EVIDENCE: Code review completed with minor style issues. All functionality implemented correctly. Tests passing. QUESTION: Should we proceed? ANSWER (YES or NO):",
            "EVIDENCE: Breaking API changes introduced. All clients need updates. Migration guide incomplete. QUESTION: Should we proceed? ANSWER (YES or NO):",
            "EVIDENCE: Load testing completed successfully. System handles 10x expected load. Scalability verified. QUESTION: Should we proceed? ANSWER (YES or NO):",
        ]
        eval_file.parent.mkdir(parents=True, exist_ok=True)
        with open(eval_file, "w") as f:
            json.dump({"questions": default_questions}, f, indent=2)
        return default_questions

    with open(eval_file) as f:
        data = json.load(f)
    return data.get("questions", [])


def get_binary_questions(eval_file: Path = Path("evaluation/binary_eval_questions.json")) -> List[str]:
    """Get binary classifier evaluation questions."""
    return load_binary_eval_questions(eval_file)
