"""
Add latent curriculum tags to high-quality, structured Worker samples.

Marks samples suitable for latent-space reasoning experiments with:
- latent_curriculum: true
- ir_ready: true (if parseable into structured steps)

Author: @darianrosebrook
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any


def is_ir_ready(sample: Dict[str, Any]) -> bool:
    """
    Check if sample is ready for IR/latent-space experiments.
    
    Criteria:
    - Has clean tool-use with JSON arguments
    - Has process-step supervision
    - Has structured reasoning (teacher_reasoning or parseable steps)
    - Has CAWS context (for structured decision-making)
    """
    # Check for tool-use with process supervision
    has_tool_use = (
        sample.get("task_type") in ["tool_use", "caws_tool"]
        and (sample.get("tool_name_ids") or sample.get("gold_json_text_ids"))
    )
    
    # Check for structured reasoning
    has_reasoning = bool(sample.get("teacher_reasoning"))
    
    # Check for CAWS context (structured decision-making)
    has_caws = bool(sample.get("caws_context") or sample.get("caws_level", 0) > 0)
    
    # Check for clean structure (not too noisy)
    teacher_text = sample.get("teacher_text", "")
    is_clean = (
        len(teacher_text) > 100  # Not too short
        and len(teacher_text) < 8000  # Not too long (for IR experiments)
        and "TODO" not in teacher_text.upper()
        and "PLACEHOLDER" not in teacher_text.upper()
    )
    
    # IR-ready if it has structured elements and is clean
    return (has_tool_use or has_caws) and has_reasoning and is_clean


def should_tag_latent_curriculum(sample: Dict[str, Any]) -> bool:
    """
    Determine if sample should be tagged for latent curriculum.
    
    Criteria:
    - High quality (has reasoning, process supervision, or CAWS context)
    - Clean structure
    - Representative of key task families
    """
    # Must have at least one of: reasoning, process supervision, or CAWS
    has_quality_signal = (
        bool(sample.get("teacher_reasoning"))
        or bool(sample.get("tool_name_ids") or sample.get("gold_json_text_ids"))
        or bool(sample.get("caws_context") or sample.get("caws_level", 0) > 0)
    )
    
    # Must be clean
    teacher_text = sample.get("teacher_text", "")
    is_clean = (
        len(teacher_text) > 50
        and "TODO" not in teacher_text.upper()
        and "PLACEHOLDER" not in teacher_text.upper()
        and "MOCK" not in teacher_text.upper()
    )
    
    return has_quality_signal and is_clean


def tag_samples(samples: List[Dict[str, Any]], target_count: int = 1000) -> List[Dict[str, Any]]:
    """
    Tag samples for latent curriculum.
    
    Prioritizes:
    1. Tool-use with process supervision
    2. CAWS context samples
    3. Long-context samples (for context decay experiments)
    4. Samples with reasoning
    """
    tagged_samples = []
    tagged_count = 0
    
    # Priority 1: Tool-use with process supervision
    for sample in samples:
        if tagged_count >= target_count:
            break
        if (
            sample.get("task_type") in ["tool_use", "caws_tool"]
            and (sample.get("tool_name_ids") or sample.get("gold_json_text_ids"))
            and should_tag_latent_curriculum(sample)
            and not sample.get("latent_curriculum")
        ):
            sample["latent_curriculum"] = True
            sample["ir_ready"] = is_ir_ready(sample)
            tagged_samples.append(sample)
            tagged_count += 1
    
    # Priority 2: CAWS context samples
    for sample in samples:
        if tagged_count >= target_count:
            break
        if (
            sample.get("caws_context") or sample.get("caws_level", 0) > 0
            and should_tag_latent_curriculum(sample)
            and not sample.get("latent_curriculum")
        ):
            sample["latent_curriculum"] = True
            sample["ir_ready"] = is_ir_ready(sample)
            tagged_samples.append(sample)
            tagged_count += 1
    
    # Priority 3: Long-context samples
    for sample in samples:
        if tagged_count >= target_count:
            break
        if (
            sample.get("task_type") == "long_context"
            or (len(sample.get("prompt", "")) + len(sample.get("teacher_text", ""))) > 8000
        ) and should_tag_latent_curriculum(sample) and not sample.get("latent_curriculum"):
            sample["latent_curriculum"] = True
            sample["ir_ready"] = is_ir_ready(sample)
            tagged_samples.append(sample)
            tagged_count += 1
    
    # Priority 4: Samples with reasoning
    for sample in samples:
        if tagged_count >= target_count:
            break
        if (
            sample.get("teacher_reasoning")
            and should_tag_latent_curriculum(sample)
            and not sample.get("latent_curriculum")
        ):
            sample["latent_curriculum"] = True
            sample["ir_ready"] = is_ir_ready(sample)
            tagged_samples.append(sample)
            tagged_count += 1
    
    # Update all samples (not just tagged ones)
    for sample in samples:
        if sample not in tagged_samples:
            # Ensure latent_curriculum is False if not tagged
            if "latent_curriculum" not in sample:
                sample["latent_curriculum"] = False
            if "ir_ready" not in sample:
                sample["ir_ready"] = False
    
    return samples


def main():
    ap = argparse.ArgumentParser(
        description="Add latent curriculum tags to Worker samples",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--input", required=True, help="Input Worker dataset JSONL")
    ap.add_argument("--output", required=True, help="Output JSONL file with tags")
    ap.add_argument(
        "--target-count",
        type=int,
        default=1000,
        help="Target number of samples to tag (default: 1000)",
    )
    args = ap.parse_args()
    
    input_file = Path(args.input)
    output_file = Path(args.output)
    
    # Load samples
    print(f"[add_latent_curriculum_tags] Loading samples from {input_file}")
    samples = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
                if "__header__" in sample:
                    continue
                samples.append(sample)
            except json.JSONDecodeError as e:
                print(f"[add_latent_curriculum_tags] WARN: Invalid JSON: {e}")
                continue
    
    print(f"[add_latent_curriculum_tags] Loaded {len(samples)} samples")
    
    # Tag samples
    tagged_samples = tag_samples(samples, target_count=args.target_count)
    
    # Count tags
    latent_curriculum_count = sum(1 for s in tagged_samples if s.get("latent_curriculum"))
    ir_ready_count = sum(1 for s in tagged_samples if s.get("ir_ready"))
    
    print(f"\n[add_latent_curriculum_tags] Tagging Statistics:")
    print(f"  latent_curriculum: {latent_curriculum_count} samples")
    print(f"  ir_ready: {ir_ready_count} samples")
    
    # Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in tagged_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    print(f"\n[add_latent_curriculum_tags] Wrote {len(tagged_samples)} samples to {output_file}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

