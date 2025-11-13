#!/usr/bin/env python3
"""
Format 8-ball Ollama output by mapping token IDs to human-readable answers.

Usage:
    ollama run 8-ball "Should I go to the doctor?" | python scripts/format_8ball_output.py
"""

import sys
import re

# Import 8-ball configuration
from evaluation.toy.eight_ball import (
    EIGHT_BALL_ANSWERS,
    ID_TO_ANSWER,
    ANSWER_TO_ID,
    EIGHT_BALL_TOKEN_IDS,
)

# Also map the BALL token names (from tokenizer)
BALL_TOKEN_NAMES = [
    "<BALL_IT_IS_CERTAIN>",
    "<BALL_IT_IS_DECIDEDLY_SO>",
    "<BALL_WITHOUT_A_DOUBT>",
    "<BALL_YES_DEFINITELY>",
    "<BALL_YOU_MAY_RELY_ON_IT>",
    "<BALL_AS_I_SEE_IT_YES>",
    "<BALL_MOST_LIKELY>",
    "<BALL_OUTLOOK_GOOD>",
    "<BALL_YES>",
    "<BALL_SIGNS_POINT_TO_YES>",
    "<BALL_REPLY_HAZY_TRY_AGAIN>",
    "<BALL_ASK_AGAIN_LATER>",
    "<BALL_BETTER_NOT_TELL_YOU_NOW>",
    "<BALL_CANNOT_PREDICT_NOW>",
    "<BALL_CONCENTRATE_AND_ASK_AGAIN>",
    "<BALL_DONT_COUNT_ON_IT>",
    "<BALL_MY_REPLY_IS_NO>",
    "<BALL_MY_SOURCES_SAY_NO>",
    "<BALL_OUTLOOK_NOT_SO_GOOD>",
    "<BALL_VERY_DOUBTFUL>",
]

NAME_TO_ANSWER = {
    name: answer
    for name, answer in zip(BALL_TOKEN_NAMES, EIGHT_BALL_ANSWERS)
}


def format_output(text: str) -> str:
    """Format Ollama output by replacing token IDs/names with answers."""
    output = text

    # Replace BALL token names with answers
    for token_name, answer in NAME_TO_ANSWER.items():
        output = output.replace(token_name, answer)

    # Replace <token_XXX> patterns with answers
    for token_id in EIGHT_BALL_TOKEN_IDS:
        token_pattern = f"<token_{token_id}>"
        if token_pattern in output:
            answer = ID_TO_ANSWER[token_id]
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
    print(formatted)


if __name__ == "__main__":
    main()

