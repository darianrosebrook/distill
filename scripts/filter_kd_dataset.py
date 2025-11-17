"""
Filter KD dataset by quality criteria.

Usage:
    python -m scripts.filter_kd_dataset \
      --input data/kd_mix_1500.jsonl \
      --output data/kd_mix_1500.filtered.jsonl \
      --min-length 100 \
      --max-length 5000 \
      --dedupe
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Set


def hash_prompt(prompt: str) -> str:
    """Generate hash for prompt deduplication."""
    return hashlib.sha256(prompt.encode()).hexdigest()


def detect_truncation(text: str) -> bool:
    """Detect if text appears truncated."""
    if not text:
        return False
    
    text = text.strip()
    if not text:
        return False
    
    # Ends without sentence-ending punctuation
    last_char = text[-1]
    if last_char not in '.!?':
        if ' ' in text[-50:]:  # Has spaces (not mid-word)
            return True
    
    # Ends with ellipsis or cutoff markers
    if text.endswith('...') or text.endswith('…'):
        return True
    
    # Ends with incomplete markdown
    if text.endswith('**') or text.endswith('*'):
        return True
    
    return False


def filter_dataset(
    input_path: Path,
    output_path: Path,
    min_length: int = 0,
    max_length: int = None,
    dedupe: bool = False,
    remove_truncated: bool = False,
) -> Dict[str, Any]:
    """Filter dataset and return statistics."""
    seen_prompts: Set[str] = set()
    stats = {
        'total': 0,
        'passed': 0,
        'filtered_min_length': 0,
        'filtered_max_length': 0,
        'filtered_truncated': 0,
        'filtered_duplicate': 0,
        'filtered_empty': 0,
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            if not line.strip():
                continue
            
            try:
                record = json.loads(line)
                stats['total'] += 1
            except json.JSONDecodeError:
                continue
            
            # Check empty teacher_text
            teacher_text = record.get('teacher_text', '')
            if not teacher_text or not teacher_text.strip():
                stats['filtered_empty'] += 1
                continue
            
            # Check length bounds
            text_len = len(teacher_text)
            if text_len < min_length:
                stats['filtered_min_length'] += 1
                continue
            
            if max_length and text_len > max_length:
                stats['filtered_max_length'] += 1
                continue
            
            # Check truncation
            if remove_truncated and detect_truncation(teacher_text):
                stats['filtered_truncated'] += 1
                continue
            
            # Check duplicates
            if dedupe:
                prompt = record.get('prompt', '')
                prompt_hash = hash_prompt(prompt)
                if prompt_hash in seen_prompts:
                    stats['filtered_duplicate'] += 1
                    continue
                seen_prompts.add(prompt_hash)
            
            # Passed all filters
            fout.write(json.dumps(record, ensure_ascii=False) + '\n')
            stats['passed'] += 1
    
    return stats


def generate_manifest(stats: Dict[str, Any], output_path: Path):
    """Generate manifest file with filtering statistics."""
    manifest_path = output_path.with_suffix('.manifest.json')
    
    manifest = {
        'filtering_stats': stats,
        'filters_applied': {
            'min_length': stats.get('min_length', 0),
            'max_length': stats.get('max_length'),
            'dedupe': stats.get('dedupe', False),
            'remove_truncated': stats.get('remove_truncated', False),
        },
    }
    
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Manifest written to: {manifest_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Filter KD dataset by quality criteria",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--input", required=True, help="Input JSONL file path")
    ap.add_argument("--output", required=True, help="Output JSONL file path")
    ap.add_argument("--min-length", type=int, default=0, help="Minimum teacher_text length (chars)")
    ap.add_argument("--max-length", type=int, default=None, help="Maximum teacher_text length (chars)")
    ap.add_argument("--dedupe", action="store_true", help="Remove duplicate prompts")
    ap.add_argument("--remove-truncated", action="store_true", help="Remove truncated responses")
    args = ap.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return 1
    
    output_path = Path(args.output)
    
    print(f"[filter_kd_dataset] Filtering dataset:")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Min length: {args.min_length}")
    print(f"  Max length: {args.max_length or 'unlimited'}")
    print(f"  Dedupe: {args.dedupe}")
    print(f"  Remove truncated: {args.remove_truncated}")
    
    stats = filter_dataset(
        input_path=input_path,
        output_path=output_path,
        min_length=args.min_length,
        max_length=args.max_length,
        dedupe=args.dedupe,
        remove_truncated=args.remove_truncated,
    )
    
    # Add filter parameters to stats
    stats['min_length'] = args.min_length
    stats['max_length'] = args.max_length
    stats['dedupe'] = args.dedupe
    stats['remove_truncated'] = args.remove_truncated
    
    generate_manifest(stats, output_path)
    
    print(f"\n[filter_kd_dataset] Filtering complete:")
    print(f"  Total records: {stats['total']}")
    print(f"  Passed filters: {stats['passed']}")
    print(f"  Filtered (empty): {stats['filtered_empty']}")
    print(f"  Filtered (min_length): {stats['filtered_min_length']}")
    print(f"  Filtered (max_length): {stats['filtered_max_length']}")
    print(f"  Filtered (truncated): {stats['filtered_truncated']}")
    print(f"  Filtered (duplicate): {stats['filtered_duplicate']}")
    print(f"  Output: {output_path}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

