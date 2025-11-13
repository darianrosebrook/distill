"""
Generate toy KD dataset for end-to-end pipeline testing.

Creates N KD pairs (prompt → teacher target) with tool-like spans for verification.
30% of samples include simple tool spans (tool.call{...}) to test claims extraction.

Can also generate 8-Ball training data for mystical model outputs.

Usage:
    python -m data.make_toy_kd --out toy_kd.jsonl --n 128
    python -m data.make_toy_kd --out 8_ball.jsonl --n 128 --eight-ball
    python -m data.make_toy_kd --out binary.jsonl --n 128 --binary-classifier
    python -m data.make_toy_kd --out ternary.jsonl --n 128 --ternary-classifier
    python -m data.make_toy_kd --demo  # Show sample data
"""

import argparse
import json
import random
import hashlib
import sys
from pathlib import Path


def mk_item(i: int, vocab_size: int = 512, eight_ball: bool = False, binary_classifier: bool = False, ternary_classifier: bool = False) -> dict:
    """Create a single KD sample."""
    if eight_ball:
        return mk_eight_ball_item(i, vocab_size)
    elif binary_classifier:
        return mk_binary_classifier_item(i, vocab_size)
    elif ternary_classifier:
        return mk_ternary_classifier_item(i, vocab_size)

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


def mk_eight_ball_item(i: int, vocab_size: int = 512) -> dict:
    """Create an 8-ball KD sample."""
    # 8-ball classic responses
    eight_ball_answers = [
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

    # Create yes/no questions that should get 8-Ball answers
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
    answer_idx = (i * 7) % len(eight_ball_answers)
    # Different multiplier for variety
    flair_idx = (i * 13) % len(flair_options)

    prompt = f"🎱 {question_templates[question_idx]}"
    mystical_answer = eight_ball_answers[answer_idx]
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
            "source": "8_ball_kd",
            "has_tool_span": "tool:" in target,
            "mystical_answer": mystical_answer,
        },
    }


def mk_binary_classifier_item(i: int, vocab_size: int = 512) -> dict:
    """Create a binary classifier KD sample."""
    # Evidence scenarios for binary classification
    evidence_scenarios = [
        "All tests pass with 100% coverage. Code review completed successfully. No critical issues found.",
        "Security vulnerability detected in dependency. No patch available. Production deployment blocked.",
        "Performance benchmarks show 30% improvement. Memory usage optimized. All SLAs met.",
        "Only 40% test coverage. Multiple edge cases untested. Documentation incomplete.",
        "Feature fully implemented. User acceptance testing passed. Ready for production.",
        "Database migration tested successfully. Rollback plan documented. Data integrity verified.",
        "API response times increased by 150%. Memory leak detected. Performance regression identified.",
        "Code review completed with minor style issues. All functionality working correctly.",
        "Breaking API changes introduced. All clients require updates. Migration guide needed.",
        "Load testing passed with 5x expected load. Scalability requirements met.",
        "Unit tests failing. Build broken. Cannot proceed with deployment.",
        "All integration tests pass. End-to-end workflows verified. Quality gates passed.",
        "Third-party service dependency down. No fallback implemented. High availability risk.",
        "Code changes minimal and focused. Impact analysis completed. Safe to deploy.",
        "Database schema changes untested. Migration script incomplete. Data loss risk.",
        "Performance monitoring implemented. Alerting configured. Observability complete.",
        "API contract changes undocumented. Client applications may break.",
        "Security review completed. No vulnerabilities found. Compliance verified.",
        "Test environment unstable. Flaky tests detected. Reliability concerns.",
        "Feature flags implemented. Gradual rollout plan ready. Risk mitigation in place.",
    ]

    # Select evidence deterministically based on index
    evidence_idx = i % len(evidence_scenarios)
    evidence = evidence_scenarios[evidence_idx]

    # Create prompt with evidence format
    prompt = f"EVIDENCE: {evidence} QUESTION: Should we proceed? ANSWER (YES or NO):"

    # Deterministically assign YES/NO based on evidence content hash
    # This ensures consistent labeling for the same evidence
    evidence_hash = hashlib.sha256(evidence.encode()).hexdigest()
    hash_int = int(evidence_hash[:8], 16)  # Use first 8 hex chars as int

    # Classify based on evidence content (deterministic but varied)
    if "vulnerability" in evidence or "blocked" in evidence or "failing" in evidence or "broken" in evidence:
        # Clear NO cases
        label = "NO"
    elif "100% coverage" in evidence or "successful" in evidence or "verified" in evidence or "passed" in evidence:
        # Clear YES cases
        label = "YES"
    elif "improvement" in evidence or "optimized" in evidence or "met" in evidence:
        # Generally positive but check hash for variety
        label = "YES" if hash_int % 3 != 0 else "NO"  # 2/3 YES, 1/3 NO
    elif "incomplete" in evidence or "untested" in evidence or "regression" in evidence:
        # Generally negative but check hash for variety
        label = "NO" if hash_int % 3 != 0 else "YES"  # 2/3 NO, 1/3 YES
    else:
        # Borderline cases - use hash for deterministic variety
        label = "YES" if hash_int % 2 == 0 else "NO"

    return {
        "id": i,
        "prompt": prompt,
        "teacher_text": label,  # Will be tokenized to YES/NO tokens during training
        "metadata": {
            "source": "binary_classifier",
            "evidence": evidence,
            "label": label,
            "evidence_hash": evidence_hash[:8],  # For debugging consistency
        },
    }


def mk_ternary_classifier_item(i: int, vocab_size: int = 512) -> dict:
    """Create a ternary classifier KD sample."""
    # Evidence scenarios for ternary classification (YES/NO/UNCERTAIN)
    evidence_scenarios = [
        "All tests pass with 100% coverage. All integration tests pass. No critical bugs reported.",
        "Multiple critical security vulnerabilities found in dependency. No fix available from vendor. Production system at risk.",
        "Performance benchmarks show 30% improvement. Memory usage optimized. All SLAs met.",
        "Only 40% test coverage. Several edge cases untested. Documentation incomplete.",
        "Feature implemented successfully. User acceptance testing passed. Ready for production.",
        "Database migration script tested successfully. Rollback plan documented. Data integrity verified.",
        "API response times increased by 150%. Memory leak detected. Performance regression identified.",
        "Code review completed with minor style issues. All functionality working correctly.",
        "Breaking API changes introduced. All clients require updates. Migration guide incomplete.",
        "Load testing completed successfully. System handles 5x expected load. Scalability verified.",
        "Unit tests failing. Build broken. Cannot proceed with deployment.",
        "All integration tests pass. End-to-end workflows verified. Quality gates passed.",
        "Third-party service dependency down. No fallback implemented. High availability risk.",
        "Code changes minimal and focused. Impact analysis completed. Safe to deploy.",
        "Database schema changes untested. Migration script incomplete. Data loss risk.",
        "Performance monitoring implemented. Alerting configured. Observability complete.",
        "API contract changes undocumented. Client applications may break.",
        "Security review completed. No vulnerabilities found. Compliance verified.",
        "Test environment unstable. Flaky tests detected. Reliability concerns.",
        "Feature flags implemented. Gradual rollout plan ready. Risk mitigation in place.",
        "Partial test coverage. Some functionality not validated. Risk assessment incomplete.",
        "Code changes affect core business logic. Extensive testing required but not performed.",
        "Third-party API changes may impact functionality. Integration testing pending.",
        "Database performance not benchmarked. Potential scalability issues unknown.",
        "Security implications not fully assessed. Compliance requirements unclear.",
        "User experience changes significant. Usability testing not conducted.",
        "Monitoring and alerting not configured. Operational visibility limited.",
        "Documentation not updated to reflect changes. Knowledge transfer incomplete.",
        "Rollback procedures not tested. Recovery plan unverified.",
        "Stakeholder communication incomplete. Change management process unclear.",
    ]

    # Select evidence deterministically based on index
    evidence_idx = i % len(evidence_scenarios)
    evidence = evidence_scenarios[evidence_idx]

    # Create prompt with evidence format
    prompt = f"EVIDENCE: {evidence} QUESTION: Should we proceed? ANSWER (YES or NO or UNCERTAIN):"

    # Deterministically assign YES/NO/UNCERTAIN based on evidence content hash
    # This ensures consistent labeling for the same evidence
    evidence_hash = hashlib.sha256(evidence.encode()).hexdigest()
    hash_int = int(evidence_hash[:8], 16)  # Use first 8 hex chars as int

    # Classify based on evidence content (deterministic but varied)
    if "vulnerability" in evidence or "blocked" in evidence or "failing" in evidence or "broken" in evidence:
        # Clear NO cases
        label = "NO"
    elif "100% coverage" in evidence or "successful" in evidence or "verified" in evidence or "passed" in evidence:
        # Clear YES cases
        label = "YES"
    elif "improvement" in evidence or "optimized" in evidence or "met" in evidence:
        # Generally positive but check hash for variety
        if hash_int % 4 == 0:
            label = "UNCERTAIN"
        else:
            label = "YES"
    elif "incomplete" in evidence or "untested" in evidence or "regression" in evidence:
        # Generally negative but check hash for variety
        if hash_int % 4 == 0:
            label = "UNCERTAIN"
        else:
            label = "NO"
    elif "pending" in evidence or "unclear" in evidence or "unknown" in evidence or "limited" in evidence:
        # Borderline/uncertain cases
        if hash_int % 3 == 0:
            label = "YES"
        elif hash_int % 3 == 1:
            label = "NO"
        else:
            label = "UNCERTAIN"
    else:
        # Mixed cases - use hash for deterministic variety
        hash_mod = hash_int % 5
        if hash_mod == 0:
            label = "YES"
        elif hash_mod == 1:
            label = "NO"
        else:
            label = "UNCERTAIN"  # Higher chance of UNCERTAIN for mixed cases

    return {
        "id": i,
        "prompt": prompt,
        "teacher_text": label,  # Will be tokenized to YES/NO/UNCERTAIN tokens during training
        "metadata": {
            "source": "ternary_classifier",
            "evidence": evidence,
            "label": label,
            "evidence_hash": evidence_hash[:8],  # For debugging consistency
        },
    }


def main():
    ap = argparse.ArgumentParser(description="Generate toy KD dataset")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--n", type=int, default=128, help="Number of samples")
    ap.add_argument("--vocab", type=int, default=512, help="Vocabulary size (for compatibility)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument(
        "--eight-ball",
        dest="eight_ball",
        action="store_true",
        help="Generate 8-ball mystical training data instead of tool data",
    )
    ap.add_argument(
        "--binary-classifier",
        dest="binary_classifier",
        action="store_true",
        help="Generate binary classifier YES/NO training data",
    )
    ap.add_argument(
        "--ternary-classifier",
        dest="ternary_classifier",
        action="store_true",
        help="Generate ternary classifier YES/NO/UNCERTAIN training data",
    )
    args = ap.parse_args()

    # Set seed for reproducibility
    random.seed(args.seed)

    # Auto-organize outputs into toys directories based on data type
    if args.eight_ball:
        toy_type = "8ball"
    elif args.binary_classifier:
        toy_type = "binary"
    elif args.ternary_classifier:
        toy_type = "ternary"
    else:
        toy_type = "pipeline"

    # If output path doesn't already include toys/, prepend it
    output_path_str = args.out
    if not output_path_str.startswith("toys/"):
        output_path_str = f"toys/{toy_type}/{output_path_str}"
    args.out = output_path_str

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate samples
    samples = []
    for i in range(args.n):
        samples.append(mk_item(i, args.vocab, args.eight_ball, args.binary_classifier, args.ternary_classifier))

    # Write JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Compute dataset hash
    dataset_content = "\n".join([json.dumps(s, ensure_ascii=False) for s in samples])
    dataset_sha256 = hashlib.sha256(dataset_content.encode("utf-8")).hexdigest()

    if args.ternary_classifier:
        dataset_type = "ternary-classifier"
        yes_count = sum(1 for s in samples if s["metadata"]["label"] == "YES")
        no_count = sum(1 for s in samples if s["metadata"]["label"] == "NO")
        uncertain_count = sum(1 for s in samples if s["metadata"]["label"] == "UNCERTAIN")
        print(f"[make_toy_kd] Created {dataset_type} dataset: {output_path}")
        print(f"  Samples: {len(samples)}")
        print(f"  YES labels: {yes_count} ({100 * yes_count / len(samples):.1f}%)")
        print(f"  NO labels: {no_count} ({100 * no_count / len(samples):.1f}%)")
        print(f"  UNCERTAIN labels: {uncertain_count} ({100 * uncertain_count / len(samples):.1f}%)")
    elif args.binary_classifier:
        dataset_type = "binary-classifier"
        yes_count = sum(1 for s in samples if s["metadata"]["label"] == "YES")
        no_count = sum(1 for s in samples if s["metadata"]["label"] == "NO")
        print(f"[make_toy_kd] Created {dataset_type} dataset: {output_path}")
        print(f"  Samples: {len(samples)}")
        print(f"  YES labels: {yes_count} ({100 * yes_count / len(samples):.1f}%)")
        print(f"  NO labels: {no_count} ({100 * no_count / len(samples):.1f}%)")
    elif args.eight_ball:
        dataset_type = "8-ball"
        tool_span_count = sum(1 for s in samples if s["metadata"]["has_tool_span"])
        mystical_answers = sum(1 for s in samples if "mystical_answer" in s["metadata"])
        print(f"[make_toy_kd] Created {dataset_type} dataset: {output_path}")
        print(f"  Samples: {len(samples)}")
        print(f"  Tool spans: {tool_span_count} ({100 * tool_span_count / len(samples):.1f}%)")
        print(
            f"  Mystical answers: {mystical_answers} ({100 * mystical_answers / len(samples):.1f}%)"
        )
    else:
        dataset_type = "toy"
        tool_span_count = sum(1 for s in samples if s["metadata"]["has_tool_span"])
        print(f"[make_toy_kd] Created {dataset_type} dataset: {output_path}")
        print(f"  Samples: {len(samples)}")
        print(f"  Tool spans: {tool_span_count} ({100 * tool_span_count / len(samples):.1f}%)")
    print(f"  Dataset SHA256: {dataset_sha256[:16]}...")


if __name__ == "__main__":
    # For testing data generation
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        print("🎱 8-ball Dataset Demo:")
        print("=" * 50)
        for i in range(3):
            sample = mk_eight_ball_item(i, 512)
            print(f"Prompt: {sample['prompt']}")
            print(f"Answer: {sample['teacher_text']}")
            print()

        print("🤖 Binary Classifier Dataset Demo:")
        print("=" * 50)
        for i in range(2):
            sample = mk_binary_classifier_item(i, 512)
            print(f"Prompt: {sample['prompt']}")
            print(f"Answer: {sample['teacher_text']}")
            print()

        print("🔺 Ternary Classifier Dataset Demo:")
        print("=" * 50)
        for i in range(2):
            sample = mk_ternary_classifier_item(i, 512)
            print(f"Prompt: {sample['prompt']}")
            print(f"Answer: {sample['teacher_text']}")
            print()
    else:
        main()
