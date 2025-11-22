"""
Generate long-context KD samples (8-16k tokens) for Worker dataset.

These samples help train the Worker model to handle long contexts with
CAWS specs, multi-step tool chains, and complex reasoning.

Author: @darianrosebrook
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from datetime import datetime

from scripts.prompt_sources import get_prompt_mix
from models.teacher.teacher_client import TeacherClient, APITier, TIER_LIMITS
from training.caws_context import extract_caws_context_dict
from training.extractors import (
    extract_tool_name_span,
    extract_json_argument_spans,
    identify_integration_spans,
)
from tools.schema_registry import get_registry


def load_tokenizer(tokenizer_path: str):
    """Load tokenizer from path."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        return tokenizer
    except Exception as e:
        print(
            f"[generate_long_context_kd] WARN: Failed to load tokenizer: {e}")
        return None


def extract_process_step_targets(
    teacher_text: str,
    tokenizer,
    tool_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Extract process-step supervision targets from teacher output."""
    if tokenizer is None:
        return {}

    targets = {}

    # Extract tool name span
    tool_name_span = extract_tool_name_span(teacher_text, tool_names)
    if tool_name_span:
        start_char, end_char = tool_name_span
        tool_name_text = teacher_text[start_char:end_char]
        tool_name_ids = tokenizer.encode(
            tool_name_text, add_special_tokens=False)
        targets["tool_name_ids"] = tool_name_ids
        targets["tool_name_mask"] = [1] * len(tool_name_ids)

    # Extract JSON argument spans
    json_spans = extract_json_argument_spans(teacher_text)
    if json_spans:
        all_json_ids = []
        all_json_mask = []
        for start_char, end_char in json_spans:
            json_text = teacher_text[start_char:end_char]
            json_ids = tokenizer.encode(json_text, add_special_tokens=False)
            all_json_ids.extend(json_ids)
            all_json_mask.extend([1] * len(json_ids))
        if all_json_ids:
            targets["gold_json_text_ids"] = all_json_ids
            targets["mask_valid_json_tokens"] = all_json_mask

    # Extract integration spans
    integration_spans = identify_integration_spans(teacher_text)
    if integration_spans:
        all_integration_ids = []
        all_integration_mask = []
        for start_char, end_char in integration_spans:
            integration_text = teacher_text[start_char:end_char]
            integration_ids = tokenizer.encode(
                integration_text, add_special_tokens=False)
            all_integration_ids.extend(integration_ids)
            all_integration_mask.extend([1] * len(integration_ids))
        if all_integration_ids:
            targets["integration_mask"] = all_integration_mask

    return targets


def generate_long_context_prompt(base_prompt: str, caws_context: Dict[str, Any]) -> str:
    """Generate a long-context prompt by adding CAWS context and multi-step instructions."""
    # Format CAWS context compactly
    caws_json = json.dumps(caws_context, separators=(",", ":"))

    # Create long-context prompt with multiple steps
    long_prompt = f"""{caws_json}

Task: {base_prompt}

This task requires multiple steps:
1. Analyze the current codebase structure
2. Identify files that need modification
3. Make the necessary changes while respecting CAWS budgets and scope
4. Verify changes with tests
5. Document the changes with evidence and provenance

Please provide a comprehensive solution that addresses all steps."""

    return long_prompt


def main():
    ap = argparse.ArgumentParser(
        description="Generate long-context KD samples (8-16k tokens)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--out", required=True, help="Output JSONL file path")
    ap.add_argument(
        "--teacher",
        required=True,
        help="Teacher endpoint (http://host:port) or HuggingFace model (hf:model_name)",
    )
    ap.add_argument("--total", type=int, default=250,
                    help="Total number of samples (default: 250)")
    ap.add_argument(
        "--tokenizer-path",
        type=str,
        default="models/student/tokenizer",
        help="Path to tokenizer (default: models/student/tokenizer)",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Maximum tokens to generate (default: 16384)",
    )
    ap.add_argument(
        "--caws-spec-id",
        help="CAWS spec ID to use for context extraction",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    ap.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay between requests in seconds (default: 0.1)",
    )
    ap.add_argument(
        "--tier",
        choices=["free", "tier1", "tier2", "tier3", "tier4", "tier5"],
        help="Manually specify API tier (overrides auto-detection)",
    )
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # Extract CAWS context
    caws_context_dict = None
    if args.caws_spec_id:
        try:
            caws_context_dict = extract_caws_context_dict(
                ".", spec_id=args.caws_spec_id)
        except Exception as e:
            print(
                f"[generate_long_context_kd] WARN: Failed to extract CAWS context: {e}")

    if not caws_context_dict:
        # Default CAWS context
        caws_context_dict = {
            "spec_id": "PROJ-695",
            "title": "distill",
            "risk_tier": 2,
            "budget": {"max_files": 25, "max_loc": 1000},
            "scope": {"in": ["arbiter/", "training/", "evaluation/"], "out": ["venv/", "__pycache__/"]},
        }

    # Initialize teacher client
    if args.teacher.startswith("hf:"):
        model_name = args.teacher[3:]
        teacher_client = TeacherClient.from_hf(model_name)
    else:
        teacher_client = TeacherClient.from_endpoint(
            args.teacher, max_retries=5)

    # Set tier if manually specified
    if args.tier:
        manual_tier = APITier[args.tier.upper()]
        teacher_client._tier = manual_tier
        teacher_client._tier_limits = TIER_LIMITS[manual_tier]
        print(
            f"[generate_long_context_kd] Using manually specified tier: {manual_tier.value}")

    # Display tier info and get concurrency limit
    max_workers = 1  # Default to sequential for free tier
    if hasattr(teacher_client, "get_tier"):
        tier = teacher_client.get_tier()
        tier_limits = teacher_client.get_tier_limits()
        print(f"[generate_long_context_kd] API Tier: {tier.value}")
        print(
            f"[generate_long_context_kd] Rate limits: {tier_limits.rpm} RPM, {tier_limits.tpm:,} TPM")
        if tier_limits.tpd:
            print(
                f"[generate_long_context_kd] Daily limit: {tier_limits.tpd:,} tokens")
        print(
            f"[generate_long_context_kd] Recommended delay: {tier_limits.delay}s")
        print(
            f"[generate_long_context_kd] Concurrency limit: {tier_limits.concurrency}")
        # Use tier concurrency limit, but cap at reasonable number for stability
        # Tier 2 allows 100 concurrent requests, but we use 90 to leave headroom
        # for other processes and avoid hitting the exact limit
        max_workers = min(tier_limits.concurrency - 10,
                          90) if tier_limits.concurrency else 1
        if max_workers > 1:
            print(
                f"[generate_long_context_kd] Using {max_workers} concurrent workers for faster generation")

        # Warn if delay doesn't match tier
        if args.delay > 0 and abs(args.delay - tier_limits.delay) > tier_limits.delay * 0.5:
            print(
                f"[generate_long_context_kd] WARN: Delay ({args.delay}s) doesn't match tier recommendation ({tier_limits.delay}s)")
            if not args.tier:
                print(
                    "[generate_long_context_kd] WARN: If you're on a higher tier, specify with --tier tier1 (or tier2/tier3/etc)")
            print(
                f"[generate_long_context_kd] WARN: Consider using --delay {tier_limits.delay} for optimal rate limit compliance")

    # Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer_path)

    # Get base prompts (use tool-use and code editing prompts)
    prompts = get_prompt_mix(
        total=args.total,
        general_ratio=0.2,
        domain_ratio=0.4,
        tool_ratio=0.4,
    )

    # Generate samples with concurrent processing
    samples = []
    samples_lock = Lock()
    errors = 0

    def process_single_prompt(i: int, base_prompt: str) -> Optional[Dict[str, Any]]:
        """Process a single prompt and return sample or None if error."""
        nonlocal errors

        try:
            # Generate long-context prompt
            long_prompt = generate_long_context_prompt(
                base_prompt, caws_context_dict)

            # Generate teacher response
            results = teacher_client.sample(
                [long_prompt],
                temperature=1.0,
                top_p=0.95,
                max_tokens=args.max_tokens,
            )

            if not results or results[0].get("error"):
                error_msg = results[0].get(
                    "error", "Unknown error") if results else "No results"
                print(
                    f"[generate_long_context_kd] ERROR: Sample {i+1}: {error_msg}")
                with samples_lock:
                    errors += 1
                return None

            result = results[0]
            teacher_text = result.get("text", "")
            reasoning_content = result.get("reasoning_content")

            # Combine reasoning_content and text if reasoning_content exists
            if reasoning_content:
                teacher_text = f"{reasoning_content}\n\n{teacher_text}"

            if not teacher_text:
                print(
                    f"[generate_long_context_kd] WARN: Sample {i+1}: Empty teacher response")
                with samples_lock:
                    errors += 1
                return None

            # Estimate token count (rough heuristic: ~4 chars per token)
            # For long-context samples, we want substantial responses with reasoning
            # Samples with reasoning_content are valuable even if shorter, so use lenient thresholds
            estimated_tokens = len(teacher_text) // 4
            estimated_prompt_tokens = len(long_prompt) // 4
            estimated_total_tokens = estimated_prompt_tokens + estimated_tokens

            # Acceptance criteria:
            # - If reasoning_content exists: accept if total >= 2000 tokens (reasoning is valuable)
            # - If no reasoning_content: require total >= 4000 tokens (need substantial content)
            if reasoning_content:
                # Samples with reasoning_content are valuable for training even if shorter
                if estimated_total_tokens < 2000:
                    print(
                        f"[generate_long_context_kd] WARN: Sample {i+1}: Total context too short ({estimated_total_tokens} tokens) despite reasoning, skipping")
                    return None
            else:
                # Without reasoning_content, need more substantial responses
                if estimated_tokens < 2000:
                    print(
                        f"[generate_long_context_kd] WARN: Sample {i+1}: Response too short ({estimated_tokens} tokens, no reasoning), skipping")
                    return None
                if estimated_total_tokens < 4000:
                    print(
                        f"[generate_long_context_kd] WARN: Sample {i+1}: Total context too short ({estimated_total_tokens} tokens), skipping")
                    return None

            # Create sample
            sample = {
                "id": f"kd-long-{i+1:06d}",
                "role": "worker",
                "task_type": "long_context",
                "caws_level": 2,
                "source": "teacher_kd",
                "prompt": long_prompt,
                "teacher_text": teacher_text,
                "caws_context": {
                    "working_spec": {
                        "id": caws_context_dict.get("spec_id", "unknown"),
                        "title": caws_context_dict.get("title", "Unknown"),
                        "risk_tier": caws_context_dict.get("risk_tier", 2),
                        "budget": caws_context_dict.get("budget", {}),
                        "scope": caws_context_dict.get("scope", {}),
                    }
                },
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "estimated_output_tokens": estimated_tokens,
                    "estimated_prompt_tokens": estimated_prompt_tokens,
                    "estimated_total_tokens": estimated_total_tokens,
                    "has_reasoning_content": reasoning_content is not None,
                    "long_context": True,
                },
            }

            # Extract process-step supervision targets
            if tokenizer:
                try:
                    tool_names = get_registry().list_tools()
                    process_targets = extract_process_step_targets(
                        teacher_text=teacher_text,
                        tokenizer=tokenizer,
                        tool_names=tool_names,
                    )
                    sample.update(process_targets)
                except Exception as e:
                    print(
                        f"[generate_long_context_kd] WARN: Sample {i+1}: Failed to extract process-step targets: {e}")

            print(
                f"[generate_long_context_kd] Completed sample {i+1}/{len(prompts)}")
            return sample

        except Exception as e:
            print(f"[generate_long_context_kd] ERROR: Sample {i+1}: {e}")
            with samples_lock:
                errors += 1
            return None

    # Prepare output file for incremental writing
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Also create a temp file to preserve all samples (including extras)
    temp_output_path = Path(str(output_path) + ".tmp")
    
    # Open files for incremental writing
    output_file = open(output_path, "w", encoding="utf-8")
    temp_file = open(temp_output_path, "w", encoding="utf-8")

    # Process prompts concurrently or sequentially based on tier
    print(
        f"[generate_long_context_kd] Processing {len(prompts)} samples with {max_workers} worker(s)...")
    print(f"[generate_long_context_kd] Writing samples incrementally to {output_path} and {temp_output_path}")

    def write_sample(sample: Dict[str, Any]):
        """Write sample to both output files."""
        sample_json = json.dumps(sample, ensure_ascii=False) + "\n"
        output_file.write(sample_json)
        temp_file.write(sample_json)
        output_file.flush()
        temp_file.flush()

    if max_workers > 1:
        # Use concurrent processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(process_single_prompt, i, prompt): i
                for i, prompt in enumerate(prompts)
            }

            # Collect results as they complete
            for future in as_completed(future_to_index):
                i = future_to_index[future]
                try:
                    sample = future.result()
                    if sample:
                        with samples_lock:
                            samples.append(sample)
                            # Write incrementally
                            write_sample(sample)
                except Exception as e:
                    print(
                        f"[generate_long_context_kd] ERROR: Sample {i+1} future exception: {e}")
                    with samples_lock:
                        errors += 1

                # Rate limiting delay between completions
                if args.delay > 0:
                    time.sleep(args.delay)
    else:
        # Sequential processing
        for i, base_prompt in enumerate(prompts):
            sample = process_single_prompt(i, base_prompt)
            if sample:
                samples.append(sample)
                # Write incrementally
                write_sample(sample)

            # Rate limiting delay
            if args.delay > 0:
                time.sleep(args.delay)

    # Close files
    output_file.close()
    temp_file.close()
    
    # Rename temp file to indicate it contains all samples
    final_temp_path = Path(str(output_path) + ".all_samples")
    temp_output_path.rename(final_temp_path)
    print(f"[generate_long_context_kd] All samples (including extras) saved to {final_temp_path}")

    print(
        f"\n[generate_long_context_kd] Generated {len(samples)} long-context samples")
    if errors > 0:
        print(
            f"[generate_long_context_kd] Encountered {errors} errors during generation")
    print(f"  Output: {output_path}")
    print(f"  All samples (including extras): {final_temp_path}")


if __name__ == "__main__":
    main()
