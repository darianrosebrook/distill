#!/usr/bin/env python3
"""
8-Ball toy model evaluation configuration.

Provides the 8-ball specific configuration for use with the classification
evaluation framework. This treats the 8-ball model as a 20-class classifier.
"""

from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path
import json


# 8-Ball answer mapping: token IDs 200-219
EIGHT_BALL_ANSWERS = [
    "It is certain",
    "It is decidedly so",
    "Without a doubt",
    "Yes definitely",
    "You may rely on it",
    "As I see it, yes",
    "Most likely",
    "Outlook good",
    "Yes",
    "Signs point to yes",
    "Reply hazy, try again",
    "Ask again later",
    "Better not tell you now",
    "Cannot predict now",
    "Concentrate and ask again",
    "Don't count on it",
    "My reply is no",
    "My sources say no",
    "Outlook not so good",
    "Very doubtful",
]

# Token ID range for 8-ball answers
EIGHT_BALL_TOKEN_START = 200
EIGHT_BALL_TOKEN_END = 219
EIGHT_BALL_TOKEN_IDS = list(range(EIGHT_BALL_TOKEN_START, EIGHT_BALL_TOKEN_END + 1))

# Create mapping from token ID to answer
ID_TO_ANSWER: Dict[int, str] = {
    token_id: answer for token_id, answer in zip(EIGHT_BALL_TOKEN_IDS, EIGHT_BALL_ANSWERS)
}

# Reverse mapping
ANSWER_TO_ID: Dict[str, int] = {answer: token_id for token_id, answer in ID_TO_ANSWER.items()}


@dataclass
class ClassificationConfig:
    """Configuration for a classification evaluation task."""
    name: str
    class_names: List[str]
    token_ids: List[int]
    id_to_name: Dict[int, str]
    name_to_id: Dict[str, int]


# 8-Ball classification configuration
EIGHT_BALL_CONFIG = ClassificationConfig(
    name="8-ball",
    class_names=EIGHT_BALL_ANSWERS,
    token_ids=EIGHT_BALL_TOKEN_IDS,
    id_to_name=ID_TO_ANSWER,
    name_to_id=ANSWER_TO_ID,
)


def load_eval_questions(eval_file: Path) -> List[str]:
    """Load evaluation questions from JSON file."""
    if not eval_file.exists():
        # Create a default set if file doesn't exist
        default_questions = [
            "Should I go to the doctor?",
            "Will I get the promotion?",
            "Is it the right time to change careers?",
            "Will my cat learn quantum mechanics?",
            "Should I take this job?",
            "Will this work?",
            "Is this the right path?",
            "Should I proceed?",
            "Will it succeed?",
            "Can I trust this?",
            "Should I invest in this?",
            "Will the weather be good?",
            "Should I move to a new city?",
            "Will I find love?",
            "Should I start my own business?",
            "Will this relationship last?",
            "Should I go back to school?",
            "Will I be successful?",
            "Should I follow my dreams?",
            "Will everything be okay?",
        ]
        eval_file.parent.mkdir(parents=True, exist_ok=True)
        with open(eval_file, "w") as f:
            json.dump({"questions": default_questions}, f, indent=2)
        return default_questions

    with open(eval_file) as f:
        data = json.load(f)
    return data.get("questions", [])


def get_eight_ball_questions(eval_file: Path = Path("evaluation/8ball_eval_questions.json")) -> List[str]:
    """Get 8-ball evaluation questions."""
    return load_eval_questions(eval_file)
