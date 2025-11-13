"""
Generate toy KD dataset for end-to-end pipeline testing.

Creates N KD pairs (prompt → teacher target) with tool-like spans for verification.
30% of samples include simple tool spans (tool.call{...}) to test claims extraction.

Can also generate Magic 8 Ball training data for mystical model outputs.

Usage:
    python -m data.make_toy_kd --out toy_kd.jsonl --n 128
    python -m data.make_toy_kd --out magic_8_ball.jsonl --n 128 --magic-8-ball
    python -m data.make_toy_kd --demo  # Show sample Magic 8 Ball data
"""

import argparse
import json
import random
import hashlib
import sys
from pathlib import Path


def mk_item(i: int, vocab_size: int = 512, magic_8_ball: bool = False) -> dict:
    """Create a single KD sample."""
    if magic_8_ball:
        return mk_magic_8_ball_item(i, vocab_size)

    # Original toy KD logic
    prompt = f"Q{i}: do a tiny action then call a tool"

    # 30% include a simple tool span
    if i % 3 == 0:
        target = 'ok tool.call{"name":"sum","args":{"a":1,"b":2}}'
    elif i % 3 == 1:
        target = 'ok tool.call{"name":"read_file","args":{"path":"test.txt"}}'
    else:
        target = "ok"

    return {
        "id": i,
        "prompt": prompt,
        "teacher_text": target,
        "metadata": {
            "source": "toy_kd",
            "has_tool_span": "tool.call{" in target,
        },
    }


def mk_magic_8_ball_item(i: int, vocab_size: int = 512) -> dict:
    """Create a Magic 8 Ball KD sample."""
    # Magic 8 Ball classic responses
    magic_answers = [
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

    # Mystical flair options
    flair_options = [
        "",  # No extra flair
        " 🔮",
        " ✨",
        " 🌟",
        " The spirits say:",
        " The crystal ball reveals:",
    ]

    # Create yes/no questions that should get Magic 8 Ball answers
    question_templates = [
        "Will this work?",
        "Is this the right path?",
        "Should I proceed?",
        "Will it succeed?",
        "Is the answer yes?",
        "Will this be successful?",
        "Should I continue?",
        "Is this a good idea?",
        "Will it work out?",
        "Is this correct?",
    ]

    # Deterministic selection based on index
    question_idx = i % len(question_templates)
    # Pseudo-random but deterministic
    answer_idx = (i * 7) % len(magic_answers)
    # Different multiplier for variety
    flair_idx = (i * 13) % len(flair_options)

    prompt = f"🎱 {question_templates[question_idx]}"
    mystical_answer = magic_answers[answer_idx]
    flair = flair_options[flair_idx]

    # Sometimes add technical claims (like the original toy model)
    if i % 5 == 0:  # 20% of samples include technical claims
        target = f"The system achieves p95 latency of 250ms. {mystical_answer}{flair}! [tool:read_file(path='perf.json')]"
    elif i % 5 == 1:
        target = f"The system processed 1,000 requests on 2024-01-15. {mystical_answer}{flair}! [tool:read_file(path='logs.json')]"
    elif i % 5 == 2:
        target = f"The system is production-ready. All tests pass. {mystical_answer}{flair}! [tool:read_file(path='test_results.json')]"
    else:
        target = f"{mystical_answer}{flair}. The mystical realm responds to your inquiry."

    return {
        "id": i,
        "prompt": prompt,
        "teacher_text": target,
        "metadata": {
            "source": "magic_8_ball_kd",
            "has_tool_span": "tool:" in target,
            "mystical_answer": mystical_answer,
        },
    }


def main():
    ap = argparse.ArgumentParser(description="Generate toy KD dataset")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--n", type=int, default=128, help="Number of samples")
    ap.add_argument("--vocab", type=int, default=512, help="Vocabulary size (for compatibility)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument(
        "--magic-8-ball",
        action="store_true",
        help="Generate Magic 8 Ball mystical training data instead of tool data",
    )
    args = ap.parse_args()

    # Set seed for reproducibility
    random.seed(args.seed)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate samples
    samples = []
    for i in range(args.n):
        samples.append(mk_item(i, args.vocab, args.magic_8_ball))

    # Write JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Compute dataset hash
    dataset_content = "\n".join([json.dumps(s, ensure_ascii=False) for s in samples])
    dataset_sha256 = hashlib.sha256(dataset_content.encode("utf-8")).hexdigest()

    tool_span_count = sum(1 for s in samples if s["metadata"]["has_tool_span"])

    dataset_type = "Magic 8 Ball" if args.magic_8_ball else "toy"
    print(f"[make_toy_kd] Created {dataset_type} dataset: {output_path}")
    print(f"  Samples: {len(samples)}")
    print(f"  Tool spans: {tool_span_count} ({100 * tool_span_count / len(samples):.1f}%)")
    if args.magic_8_ball:
        mystical_answers = sum(1 for s in samples if "mystical_answer" in s["metadata"])
        print(
            f"  Mystical answers: {mystical_answers} ({100 * mystical_answers / len(samples):.1f}%)"
        )
    print(f"  Dataset SHA256: {dataset_sha256[:16]}...")


if __name__ == "__main__":
    # For testing Magic 8 Ball generation
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        print("🎱 Magic 8 Ball Dataset Demo:")
        print("=" * 50)
        for i in range(5):
            sample = mk_magic_8_ball_item(i, 512)
            print(f"Prompt: {sample['prompt']}")
            print(f"Answer: {sample['teacher_text']}")
            print()
    else:
        main()
