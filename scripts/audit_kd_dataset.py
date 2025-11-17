"""
Audit KD dataset and generate human-readable summary report.

Usage:
    python -m scripts.audit_kd_dataset \
      --input data/kd_mix_1500.jsonl \
      --output reports/kd_mix_1500_audit.md \
      --samples 20
"""

import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


def detect_truncation(text: str) -> bool:
    """Detect if text appears truncated (ends mid-sentence)."""
    if not text:
        return False
    
    # Check for common truncation patterns
    text = text.strip()
    if not text:
        return False
    
    # Ends with incomplete sentence (no period/exclamation/question mark)
    last_char = text[-1]
    if last_char not in '.!?':
        # Check if it's mid-word or mid-sentence
        if ' ' in text[-50:]:  # Has spaces in last 50 chars (not mid-word)
            return True
    
    # Ends with ellipsis or cutoff markers
    if text.endswith('...') or text.endswith('…'):
        return True
    
    # Ends with markdown formatting that suggests truncation
    if text.endswith('**') or text.endswith('*'):
        return True
    
    return False


def analyze_dataset(input_path: Path, num_samples: int = 20) -> Dict[str, Any]:
    """Analyze KD dataset and return statistics."""
    records = []
    prompt_hashes = Counter()
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                records.append(record)
                # Hash prompt for duplicate detection
                prompt_hash = hash(record.get('prompt', ''))
                prompt_hashes[prompt_hash] += 1
            except json.JSONDecodeError as e:
                print(f"WARN: Failed to parse line: {e}")
                continue
    
    if not records:
        return {"error": "No valid records found"}
    
    # Compute statistics
    teacher_text_lengths = [len(r.get('teacher_text', '')) for r in records]
    prompt_lengths = [len(r.get('prompt', '')) for r in records]
    
    non_empty_teacher = [l for l in teacher_text_lengths if l > 0]
    empty_count = len(teacher_text_lengths) - len(non_empty_teacher)
    
    # Token estimates
    teacher_token_estimates = [estimate_tokens(r.get('teacher_text', '')) for r in records if r.get('teacher_text')]
    
    # Truncation detection
    truncated_count = sum(1 for r in records if detect_truncation(r.get('teacher_text', '')))
    
    # Duplicate detection
    duplicate_prompts = sum(1 for count in prompt_hashes.values() if count > 1)
    unique_prompts = len(prompt_hashes)
    
    # Sample random examples
    sample_indices = random.sample(range(len(records)), min(num_samples, len(records)))
    samples = [records[i] for i in sample_indices]
    
    stats = {
        'total_samples': len(records),
        'non_empty_teacher': len(non_empty_teacher),
        'empty_teacher': empty_count,
        'truncated_responses': truncated_count,
        'unique_prompts': unique_prompts,
        'duplicate_prompts': duplicate_prompts,
        'teacher_text_lengths': {
            'min': min(teacher_text_lengths) if teacher_text_lengths else 0,
            'max': max(teacher_text_lengths) if teacher_text_lengths else 0,
            'mean': statistics.mean(teacher_text_lengths) if teacher_text_lengths else 0,
            'median': statistics.median(teacher_text_lengths) if teacher_text_lengths else 0,
            'p10': statistics.quantiles(teacher_text_lengths, n=10)[0] if len(teacher_text_lengths) >= 10 else 0,
            'p25': statistics.quantiles(teacher_text_lengths, n=4)[0] if len(teacher_text_lengths) >= 4 else 0,
            'p75': statistics.quantiles(teacher_text_lengths, n=4)[2] if len(teacher_text_lengths) >= 4 else 0,
            'p90': statistics.quantiles(teacher_text_lengths, n=10)[8] if len(teacher_text_lengths) >= 10 else 0,
        },
        'prompt_lengths': {
            'min': min(prompt_lengths) if prompt_lengths else 0,
            'max': max(prompt_lengths) if prompt_lengths else 0,
            'mean': statistics.mean(prompt_lengths) if prompt_lengths else 0,
            'median': statistics.median(prompt_lengths) if prompt_lengths else 0,
        },
        'teacher_token_estimates': {
            'min': min(teacher_token_estimates) if teacher_token_estimates else 0,
            'max': max(teacher_token_estimates) if teacher_token_estimates else 0,
            'mean': statistics.mean(teacher_token_estimates) if teacher_token_estimates else 0,
            'median': statistics.median(teacher_token_estimates) if teacher_token_estimates else 0,
        },
        'samples': samples,
        'has_teacher_logits': any(r.get('teacher_logits') is not None for r in records),
        'has_reasoning': any('reasoning' in r or 'reasoning_content' in r for r in records),
    }
    
    return stats


def generate_report(stats: Dict[str, Any], output_path: Path):
    """Generate markdown report from statistics."""
    lines = []
    
    lines.append("# KD Dataset Audit Report\n")
    lines.append(f"**Dataset**: `{stats.get('input_file', 'unknown')}`\n")
    lines.append(f"**Generated**: {stats.get('timestamp', 'unknown')}\n\n")
    
    # Summary
    lines.append("## Summary\n\n")
    lines.append(f"- **Total Samples**: {stats['total_samples']}\n")
    lines.append(f"- **Non-Empty Teacher Text**: {stats['non_empty_teacher']}\n")
    lines.append(f"- **Empty Teacher Text**: {stats['empty_teacher']}\n")
    lines.append(f"- **Truncated Responses**: {stats['truncated_responses']}\n")
    lines.append(f"- **Unique Prompts**: {stats['unique_prompts']}\n")
    lines.append(f"- **Duplicate Prompts**: {stats['duplicate_prompts']}\n")
    lines.append(f"- **Has Teacher Logits**: {stats['has_teacher_logits']}\n")
    lines.append(f"- **Has Reasoning Field**: {stats['has_reasoning']}\n\n")
    
    # Teacher Text Length Distribution
    lines.append("## Teacher Text Length Distribution\n\n")
    ttl = stats['teacher_text_lengths']
    lines.append(f"- **Min**: {ttl['min']:,} chars\n")
    lines.append(f"- **Max**: {ttl['max']:,} chars\n")
    lines.append(f"- **Mean**: {ttl['mean']:.1f} chars\n")
    lines.append(f"- **Median**: {ttl['median']:.1f} chars\n")
    lines.append(f"- **10th Percentile**: {ttl['p10']:.1f} chars\n")
    lines.append(f"- **25th Percentile**: {ttl['p25']:.1f} chars\n")
    lines.append(f"- **75th Percentile**: {ttl['p75']:.1f} chars\n")
    lines.append(f"- **90th Percentile**: {ttl['p90']:.1f} chars\n\n")
    
    # Token Estimates
    lines.append("## Token Estimates (Approximate)\n\n")
    tte = stats['teacher_token_estimates']
    lines.append(f"- **Min**: {tte['min']:,} tokens\n")
    lines.append(f"- **Max**: {tte['max']:,} tokens\n")
    lines.append(f"- **Mean**: {tte['mean']:.1f} tokens\n")
    lines.append(f"- **Median**: {tte['median']:.1f} tokens\n\n")
    
    # Prompt Length Distribution
    lines.append("## Prompt Length Distribution\n\n")
    pl = stats['prompt_lengths']
    lines.append(f"- **Min**: {pl['min']:,} chars\n")
    lines.append(f"- **Max**: {pl['max']:,} chars\n")
    lines.append(f"- **Mean**: {pl['mean']:.1f} chars\n")
    lines.append(f"- **Median**: {pl['median']:.1f} chars\n\n")
    
    # Quality Issues
    lines.append("## Quality Issues\n\n")
    issues = []
    if stats['empty_teacher'] > 0:
        issues.append(f"- {stats['empty_teacher']} samples have empty teacher_text")
    if stats['truncated_responses'] > 0:
        issues.append(f"- {stats['truncated_responses']} samples appear truncated")
    if stats['duplicate_prompts'] > 0:
        issues.append(f"- {stats['duplicate_prompts']} prompts appear multiple times")
    if not stats['has_teacher_logits']:
        issues.append("- No teacher_logits available (CE-only KD)")
    
    if issues:
        lines.append("\n".join(issues))
    else:
        lines.append("- No major issues detected")
    lines.append("\n\n")
    
    # Sample Examples
    lines.append("## Sample Examples\n\n")
    for i, sample in enumerate(stats['samples'], 1):
        lines.append(f"### Example {i}\n\n")
        lines.append(f"**Prompt**: {sample.get('prompt', 'N/A')}\n\n")
        teacher_text = sample.get('teacher_text', '')
        preview = teacher_text[:400] + ('...' if len(teacher_text) > 400 else '')
        lines.append(f"**Teacher Text** ({len(teacher_text)} chars):\n```\n{preview}\n```\n\n")
        if sample.get('metadata'):
            lines.append(f"**Metadata**: {json.dumps(sample['metadata'], indent=2)}\n\n")
        lines.append("---\n\n")
    
    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(''.join(lines))
    
    print(f"Audit report written to: {output_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Audit KD dataset and generate summary report",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--input", required=True, help="Input JSONL file path")
    ap.add_argument("--output", required=True, help="Output markdown report path")
    ap.add_argument("--samples", type=int, default=20, help="Number of random samples to include")
    args = ap.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return 1
    
    print(f"[audit_kd_dataset] Analyzing dataset: {input_path}")
    stats = analyze_dataset(input_path, num_samples=args.samples)
    
    if 'error' in stats:
        print(f"ERROR: {stats['error']}")
        return 1
    
    # Add metadata
    stats['input_file'] = str(input_path)
    from datetime import datetime
    stats['timestamp'] = datetime.now().isoformat()
    
    output_path = Path(args.output)
    generate_report(stats, output_path)
    
    print(f"[audit_kd_dataset] Analysis complete:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Non-empty teacher_text: {stats['non_empty_teacher']}")
    print(f"  Empty teacher_text: {stats['empty_teacher']}")
    print(f"  Truncated: {stats['truncated_responses']}")
    print(f"  Unique prompts: {stats['unique_prompts']}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

