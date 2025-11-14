#!/usr/bin/env python3
"""
Format ternary classifier Ollama output by mapping YES/NO/UNCERTAIN tokens to human-readable answers.

Usage:
    ollama run ternary-classifier "EVIDENCE: ... QUESTION: Should we proceed? ANSWER (YES or NO or UNCERTAIN):" | python scripts/format_ternary_output.py
"""

import sys
import re

# Import ternary classifier configuration
from evaluation.toy.ternary_classifier import (
    TERNARY_ANSWERS,
    ID_TO_TERNARY_ANSWER,
    TERNARY_TOKEN_IDS,
)

# Also map the TERNARY token names (from tokenizer)
TERNARY_TOKEN_NAMES = [f"<TERNARY_{answer}>" for answer in TERNARY_ANSWERS]

NAME_TO_ANSWER = {name: answer for name, answer in zip(TERNARY_TOKEN_NAMES, TERNARY_ANSWERS)}


def format_output(text: str) -> str:
    """Format Ollama output by replacing token IDs/names with answers."""
    output = text

    # Replace TERNARY token names with answers
    for token_name, answer in NAME_TO_ANSWER.items():
        output = output.replace(token_name, answer)

    # Replace <token_XXX> patterns with answers
    for token_id in TERNARY_TOKEN_IDS:
        token_pattern = f"<token_{token_id}>"
        if token_pattern in output:
            answer = ID_TO_TERNARY_ANSWER[token_id]
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
    elif "UNCERTAIN" in formatted:
        result = "UNCERTAIN - Insufficient evidence to decide"
    else:
        result = formatted

    print(result)


if __name__ == "__main__":
    main()
