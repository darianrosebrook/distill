"""
Generate toy KD dataset for end-to-end pipeline testing.

Creates N KD pairs (prompt → teacher target) with tool-like spans for verification.
30% of samples include simple tool spans (tool.call{...}) to test claims extraction.

Usage:
    python -m data.make_toy_kd --out toy_kd.jsonl --n 128
"""
import argparse
import json
import random
import hashlib
from pathlib import Path


def mk_item(i: int, vocab_size: int = 512) -> dict:
    """Create a single KD sample."""
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
        }
    }


def main():
    ap = argparse.ArgumentParser(description="Generate toy KD dataset")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--n", type=int, default=128, help="Number of samples")
    ap.add_argument("--vocab", type=int, default=512,
                    help="Vocabulary size (for compatibility)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    # Set seed for reproducibility
    random.seed(args.seed)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate samples
    samples = []
    for i in range(args.n):
        samples.append(mk_item(i, args.vocab))

    # Write JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    # Compute dataset hash
    dataset_content = '\n'.join(
        [json.dumps(s, ensure_ascii=False) for s in samples])
    dataset_sha256 = hashlib.sha256(
        dataset_content.encode('utf-8')).hexdigest()

    tool_span_count = sum(1 for s in samples if s["metadata"]["has_tool_span"])

    print(f"[make_toy_kd] Created toy dataset: {output_path}")
    print(f"  Samples: {len(samples)}")
    print(
        f"  Tool spans: {tool_span_count} ({100*tool_span_count/len(samples):.1f}%)")
    print(f"  Dataset SHA256: {dataset_sha256[:16]}...")


if __name__ == '__main__':
    main()
