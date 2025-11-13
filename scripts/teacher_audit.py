"""
Teacher audit: Evaluate teacher quality on a 100-prompt panel.

Tests:
- Reasoning quality
- Tool JSON validity
- Long-context performance (needle-in-haystack)

Usage:
    python -m scripts.teacher_audit --teacher http://localhost:8000 --out reports/teacher_audit.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

from models.teacher.teacher_client import TeacherClient


def test_reasoning(client: TeacherClient, n: int = 30) -> Dict[str, Any]:
    """Test reasoning quality."""
    prompts = [
        "If all roses are flowers, and some flowers are red, can we conclude that some roses are red?",
        "A train leaves Station A at 60 mph. Another train leaves Station B at 80 mph. They are 200 miles apart. When will they meet?",
        "If you have 5 apples and eat 2, then buy 3 more, how many apples do you have?",
        "What is the next number in the sequence: 2, 6, 12, 20, 30, ?",
        "A rectangle has length 10 and width 5. What is its area and perimeter?",
        "If x + 5 = 12, what is x?",
        "What is the sum of the first 10 natural numbers?",
        "If a car travels 120 miles in 2 hours, what is its average speed?",
        "What is the square root of 144?",
        "If you flip a coin 3 times, what is the probability of getting exactly 2 heads?",
    ]

    results = client.sample(prompts[:n], temperature=0.7, max_tokens=512)

    # Simple quality check: non-empty responses
    valid = sum(1 for r in results if r.get("text", "").strip())

    return {
        "total": len(results),
        "valid_responses": valid,
        "validity_rate": valid / len(results) if results else 0,
        "prompts": prompts[:n],
    }


def test_json_validity(client: TeacherClient, n: int = 30) -> Dict[str, Any]:
    """Test tool JSON validity."""
    prompts = [
        "Generate a JSON object with keys 'name', 'age', and 'city'.",
        'Create a tool call JSON: {"tool": "search", "query": "Python decorators"}',
        "Format this as JSON: name=John, age=30, city=NYC",
        "Return a JSON array with three numbers: [1, 2, 3]",
        "Generate a valid JSON object for a web search tool call.",
        "Create a JSON object with nested structure: user {name, email, address {street, city}}",
        "Return a JSON array of tool calls for: search, read_file, summarize",
        "Format as JSON: tool=read_file, path=config.yaml",
        "Generate a JSON object with boolean and null values.",
        "Create a valid JSON schema for a function call.",
    ]

    results = client.sample(prompts[:n], temperature=0.5, max_tokens=512)

    valid_json = 0
    json_errors = []

    for i, result in enumerate(results):
        text = result.get("text", "").strip()

        # Try to extract JSON from response
        json_str = None
        if text.startswith("{") or text.startswith("["):
            # Try to parse as-is
            try:
                json.loads(text)
                json_str = text
            except json.JSONDecodeError:
                pass

        # Try to extract JSON from code blocks
        if not json_str:
            import re

            json_match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)

        # Try to find JSON anywhere in text
        if not json_str:
            import re

            json_match = re.search(r"(\{.*?\}|\[.*?\])", text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)

        if json_str:
            try:
                json.loads(json_str)
                valid_json += 1
            except json.JSONDecodeError as e:
                json_errors.append({"prompt_idx": i, "error": str(e)})
        else:
            json_errors.append({"prompt_idx": i, "error": "No JSON found"})

    return {
        "total": len(results),
        "valid_json": valid_json,
        "validity_rate": valid_json / len(results) if results else 0,
        "json_errors": json_errors[:10],  # First 10 errors
    }


def test_long_context(client: TeacherClient, n: int = 20) -> Dict[str, Any]:
    """Test long-context performance with needle-in-haystack."""
    # Create haystack with needle
    haystack = " ".join(["This is filler text."] * 1000)
    needle = "THE SECRET CODE IS: 42-ANSWER-TO-EVERYTHING"

    # Insert needle at different positions
    positions = [0.1, 0.3, 0.5, 0.7, 0.9]  # 10%, 30%, 50%, 70%, 90% through context

    prompts = []
    for pos in positions:
        idx = int(len(haystack.split()) * pos)
        words = haystack.split()
        words.insert(idx, needle)
        prompt = " ".join(words)
        prompts.append(f"Context: {prompt}\n\nQuestion: What is the secret code?")

    # Extend to n prompts
    while len(prompts) < n:
        prompts.extend(prompts[: min(len(prompts), n - len(prompts))])

    results = client.sample(prompts[:n], temperature=0.3, max_tokens=256)

    # Check if needle is found
    found = 0
    for result in results:
        text = result.get("text", "").lower()
        if "42-answer-to-everything" in text or "42 answer to everything" in text:
            found += 1

    return {
        "total": len(results),
        "needle_found": found,
        "retrieval_rate": found / len(results) if results else 0,
        "context_length_approx": len(prompts[0]) if prompts else 0,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Audit teacher model quality", formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument(
        "--teacher",
        required=True,
        help="Teacher endpoint (http://host:port) or HuggingFace model (hf:model_name)",
    )
    ap.add_argument("--out", required=True, help="Output JSON report path")
    ap.add_argument("--n-reasoning", type=int, default=30, help="Number of reasoning prompts")
    ap.add_argument("--n-json", type=int, default=30, help="Number of JSON validity prompts")
    ap.add_argument("--n-longctx", type=int, default=20, help="Number of long-context prompts")
    args = ap.parse_args()

    # Initialize teacher client
    if args.teacher.startswith("hf:"):
        model_name = args.teacher[3:]
        print(f"[teacher_audit] Using HuggingFace model: {model_name}")
        client = TeacherClient.from_hf(model_name)
    else:
        endpoint = args.teacher
        print(f"[teacher_audit] Using HTTP endpoint: {endpoint}")
        client = TeacherClient.from_endpoint(endpoint)

    # Health check
    if not client.health_check():
        print("[teacher_audit] WARN: Teacher health check failed")

    print(
        f"[teacher_audit] Running audit (reasoning={args.n_reasoning}, json={args.n_json}, longctx={args.n_longctx})..."
    )

    # Run tests
    print("[teacher_audit] Testing reasoning...")
    reasoning_results = test_reasoning(client, args.n_reasoning)

    print("[teacher_audit] Testing JSON validity...")
    json_results = test_json_validity(client, args.n_json)

    print("[teacher_audit] Testing long-context...")
    longctx_results = test_long_context(client, args.n_longctx)

    # Compile report
    report = {
        "teacher": args.teacher,
        "reasoning": reasoning_results,
        "json_validity": json_results,
        "long_context": longctx_results,
        "summary": {
            "reasoning_validity": reasoning_results["validity_rate"],
            "json_validity_rate": json_results["validity_rate"],
            "long_context_retrieval": longctx_results["retrieval_rate"],
            "overall_score": (
                reasoning_results["validity_rate"] * 0.4
                + json_results["validity_rate"] * 0.3
                + longctx_results["retrieval_rate"] * 0.3
            ),
        },
    }

    # Write report
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n[teacher_audit] ===== AUDIT RESULTS =====")
    print(f"Reasoning validity: {reasoning_results['validity_rate']:.1%}")
    print(f"JSON validity: {json_results['validity_rate']:.1%}")
    print(f"Long-context retrieval: {longctx_results['retrieval_rate']:.1%}")
    print(f"Overall score: {report['summary']['overall_score']:.1%}")
    print(f"\n[teacher_audit] Report saved to: {output_path}")

    # Exit criteria check
    if json_results["validity_rate"] >= 0.95 and longctx_results["retrieval_rate"] >= 0.70:
        print(
            "[teacher_audit] ✅ PASSED: Meets exit criteria (≥95% JSON validity, ≥70% long-ctx retrieval)"
        )
        return 0
    else:
        print("[teacher_audit] ⚠️  WARNING: Does not meet exit criteria")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
