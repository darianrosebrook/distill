"""
Fill CAWS tool examples with teacher responses and process-step supervision.

Extracts prompts from existing CAWS tool examples, generates teacher responses
via KD pipeline, and extracts process-step supervision targets.

Author: @darianrosebrook
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from models.teacher.teacher_client import TeacherClient
from training.extractors import (
    extract_tool_name_span,
    extract_json_argument_spans,
    identify_integration_spans,
)
from tools.schema_registry import get_registry


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
        tool_name_ids = tokenizer.encode(tool_name_text, add_special_tokens=False)
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
            integration_ids = tokenizer.encode(integration_text, add_special_tokens=False)
            all_integration_ids.extend(integration_ids)
            all_integration_mask.extend([1] * len(integration_ids))
        if all_integration_ids:
            targets["integration_mask"] = all_integration_mask

    return targets


def load_tokenizer(tokenizer_path: str):
    """Load tokenizer from path."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        return tokenizer
    except Exception as e:
        print(f"[fill_caws_tool_examples] WARN: Failed to load tokenizer: {e}")
        return None


def fill_sample(
    sample: Dict[str, Any],
    teacher_client: TeacherClient,
    tokenizer,
    max_tokens: int = 16384,
) -> Optional[Dict[str, Any]]:
    """Fill a single CAWS tool example with teacher response."""
    prompt = sample.get("prompt", "")
    if not prompt:
        print(f"[fill_caws_tool_examples] WARN: Sample {sample.get('id')} has no prompt")
        return None

    # Generate teacher response
    try:
        results = teacher_client.sample(
            [prompt],
            temperature=1.0,
            top_p=0.95,
            max_tokens=max_tokens,
        )
        if not results or results[0].get("error"):
            error_msg = results[0].get("error", "Unknown error") if results else "No results"
            print(f"[fill_caws_tool_examples] ERROR: Failed to generate response for {sample.get('id')}: {error_msg}")
            return None
        
        result = results[0]
        teacher_text = result.get("text", "")
        reasoning_content = result.get("reasoning_content")
        
        # Combine reasoning_content and text if reasoning_content exists
        if reasoning_content:
            teacher_text = f"{reasoning_content}\n\n{teacher_text}"
    except Exception as e:
        print(f"[fill_caws_tool_examples] ERROR: Failed to generate response for {sample.get('id')}: {e}")
        return None

    if not teacher_text:
        print(f"[fill_caws_tool_examples] WARN: Empty teacher response for {sample.get('id')}")
        return None

    # Create filled sample
    filled_sample = sample.copy()
    filled_sample["teacher_text"] = teacher_text
    filled_sample["source"] = "teacher_kd"

    # Extract process-step supervision targets
    if tokenizer:
        try:
            tool_names = get_registry().list_tools()
            process_targets = extract_process_step_targets(
                teacher_text=teacher_text,
                tokenizer=tokenizer,
                tool_names=tool_names,
            )
            filled_sample.update(process_targets)
        except Exception as e:
            print(f"[fill_caws_tool_examples] WARN: Failed to extract process-step targets: {e}")

    return filled_sample


def main():
    ap = argparse.ArgumentParser(
        description="Fill CAWS tool examples with teacher responses",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--in", dest="input_file", required=True, help="Input CAWS tool examples JSONL")
    ap.add_argument("--out", required=True, help="Output JSONL file path")
    ap.add_argument(
        "--teacher",
        required=True,
        help="Teacher endpoint (http://host:port) or HuggingFace model (hf:model_name)",
    )
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
        "--delay",
        type=float,
        default=0.12,
        help="Delay between requests in seconds (default: 0.12 for tier2)",
    )
    ap.add_argument(
        "--tier",
        choices=["free", "tier1", "tier2", "tier3", "tier4", "tier5"],
        help="Manually specify API tier (overrides auto-detection)",
    )
    args = ap.parse_args()

    input_file = Path(args.input_file)
    output_file = Path(args.out)

    # Load input samples
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
                print(f"[fill_caws_tool_examples] WARN: Invalid JSON: {e}")
                continue

    print(f"[fill_caws_tool_examples] Loaded {len(samples)} samples from {input_file}")

    # Initialize teacher client
    if args.teacher.startswith("hf:"):
        model_name = args.teacher[3:]
        teacher_client = TeacherClient.from_hf(model_name)
    else:
        teacher_client = TeacherClient.from_endpoint(args.teacher, max_retries=5)
    
    # Set tier if manually specified
    if args.tier:
        from models.teacher.teacher_client import APITier, TIER_LIMITS
        manual_tier = APITier[args.tier.upper()]
        teacher_client._tier = manual_tier
        teacher_client._tier_limits = TIER_LIMITS[manual_tier]
        print(f"[fill_caws_tool_examples] Using manually specified tier: {manual_tier.value}")

    # Display tier info and get concurrency limit
    max_workers = 1  # Default to sequential for free tier
    if hasattr(teacher_client, "get_tier"):
        tier = teacher_client.get_tier()
        tier_limits = teacher_client.get_tier_limits()
        print(f"[fill_caws_tool_examples] API Tier: {tier.value}")
        print(f"[fill_caws_tool_examples] Rate limits: {tier_limits.rpm} RPM, {tier_limits.tpm:,} TPM")
        if tier_limits.tpd:
            print(f"[fill_caws_tool_examples] Daily limit: {tier_limits.tpd:,} tokens")
        print(f"[fill_caws_tool_examples] Recommended delay: {tier_limits.delay}s")
        print(f"[fill_caws_tool_examples] Concurrency limit: {tier_limits.concurrency}")
        # Use tier concurrency limit, but cap at reasonable number for stability
        # Tier 2 allows 100 concurrent requests, but we use 90 to leave headroom
        max_workers = min(tier_limits.concurrency - 10, 90) if tier_limits.concurrency else 1
        if max_workers > 1:
            print(f"[fill_caws_tool_examples] Using {max_workers} concurrent workers for faster generation")

        # Warn if delay doesn't match tier
        if args.delay > 0 and abs(args.delay - tier_limits.delay) > tier_limits.delay * 0.5:
            print(f"[fill_caws_tool_examples] WARN: Delay ({args.delay}s) doesn't match tier recommendation ({tier_limits.delay}s)")

    # Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer_path)

    # Prepare output file for incremental writing
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_f = open(output_file, "w", encoding="utf-8")
    filled_samples = []
    filled_samples_lock = Lock()
    errors = 0

    def process_single_sample(i: int, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single sample and return filled sample or None if error."""
        nonlocal errors
        
        try:
            filled = fill_sample(sample, teacher_client, tokenizer, max_tokens=args.max_tokens)
            if filled:
                # Write incrementally
                sample_json = json.dumps(filled, ensure_ascii=False) + "\n"
                with filled_samples_lock:
                    filled_samples.append(filled)
                    output_f.write(sample_json)
                    output_f.flush()
                print(f"[fill_caws_tool_examples] Completed sample {i+1}/{len(samples)}: {sample.get('id')}")
                return filled
            else:
                with filled_samples_lock:
                    errors += 1
                return None
        except Exception as e:
            print(f"[fill_caws_tool_examples] ERROR: Sample {i+1} ({sample.get('id')}): {e}")
            with filled_samples_lock:
                errors += 1
            return None

    # Process samples concurrently or sequentially based on tier
    print(f"[fill_caws_tool_examples] Processing {len(samples)} samples with {max_workers} worker(s)...")

    if max_workers > 1:
        # Use concurrent processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(process_single_sample, i, sample): i
                for i, sample in enumerate(samples)
            }

            # Collect results as they complete
            for future in as_completed(future_to_index):
                i = future_to_index[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"[fill_caws_tool_examples] ERROR: Sample {i+1} future exception: {e}")
                    with filled_samples_lock:
                        errors += 1

                # Rate limiting delay between completions
                if args.delay > 0:
                    time.sleep(args.delay)
    else:
        # Sequential processing
        for i, sample in enumerate(samples):
            process_single_sample(i, sample)
            # Rate limiting delay
            if args.delay > 0:
                time.sleep(args.delay)

    # Close output file
    output_f.close()

    print(f"\n[fill_caws_tool_examples] Wrote {len(filled_samples)} filled samples to {output_file}")
    if errors > 0:
        print(f"[fill_caws_tool_examples] Encountered {errors} errors during generation")


if __name__ == "__main__":
    main()


