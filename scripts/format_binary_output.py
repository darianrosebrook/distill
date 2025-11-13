#!/usr/bin/env python3
"""
Format binary classifier Ollama output by mapping YES/NO tokens to human-readable answers.

Usage:
    ollama run binary-classifier "EVIDENCE: ... QUESTION: Should we proceed? ANSWER (YES or NO):" | python scripts/format_binary_output.py
"""

import sys
import re

# Import binary classifier configuration
from evaluation.toy.binary_classifier import (
    BINARY_ANSWERS,
    ID_TO_BINARY_ANSWER,
    BINARY_TOKEN_IDS,
)

# Also map the DECISION token names (from tokenizer)
DECISION_TOKEN_NAMES = [
    f"<DECISION_{answer}>"
    for answer in BINARY_ANSWERS
]

NAME_TO_ANSWER = {
    name: answer
    for name, answer in zip(DECISION_TOKEN_NAMES, BINARY_ANSWERS)
}


def format_output(text: str) -> str:
    """Format Ollama output by replacing token IDs/names with answers."""
    output = text

    # Replace DECISION token names with answers
    for token_name, answer in NAME_TO_ANSWER.items():
        output = output.replace(token_name, answer)

    # Replace <token_XXX> patterns with answers
    for token_id in BINARY_TOKEN_IDS:
        token_pattern = f"<token_{token_id}>"
        if token_pattern in output:
            answer = ID_TO_BINARY_ANSWER[token_id]
            output = output.replace(token_pattern, answer)

    # Clean up any remaining artifacts
    output = re.sub(r"<token_\d+>", "", output)
    output = re.sub(r"\s+", " ", output).strip()

    return output


def main():
    if len(sys.argv) > 1:
        # Read from argument
        text = " ".join(sys.argv[1:])
    else:
        # Read from stdin
        text = sys.stdin.read()

    formatted = format_output(text)

    # Add human-readable interpretation
    if "YES" in formatted:
        result = "YES - You should proceed"
    elif "NO" in formatted:
        result = "NO - You should not proceed"
    else:
        result = formatted

    print(result)


if __name__ == "__main__":
    main()
