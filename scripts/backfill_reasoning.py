"""
Backfill teacher_reasoning for KD samples that are missing it.

This script takes samples that were generated without teacher_reasoning,
re-queries the teacher API with the same prompt/parameters, and updates
the samples with the reasoning content.
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.teacher.teacher_client import TeacherClient, APITier, TIER_LIMITS
from training.caws_context import format_caws_context_for_prompt


class ReasoningBackfiller:
    """Backfill reasoning content for KD samples."""

    def __init__(
        self,
        teacher_endpoint: str,
        max_tokens: int = 16384,
        temperature: float = 1.0,
        top_p: float = 0.95,
        delay: float = 0.3,
        tier: Optional[str] = None,
    ):
        """Initialize the backfiller with teacher client configuration."""
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.delay = delay

        # Initialize teacher client
        if teacher_endpoint.startswith("hf:"):
            model_name = teacher_endpoint[3:]
            self.client = TeacherClient.from_hf(model_name)
        else:
            self.client = TeacherClient.from_endpoint(
                teacher_endpoint, max_retries=5, retry_backoff_factor=2.0
            )

        # Override tier if specified
        if tier:
            manual_tier = APITier[tier.upper()]
            self.client._tier = manual_tier
            self.client._tier_limits = TIER_LIMITS[manual_tier]
            print(f"[ReasoningBackfiller] Using tier: {manual_tier.value}")

        # Get concurrency limit from tier
        self.max_workers = 1  # Default to sequential
        if hasattr(self.client, "get_tier"):
            tier_obj = self.client.get_tier()
            tier_limits = self.client.get_tier_limits()
            # Use tier-aware concurrency (90% of max, leaves 10% headroom for safety)
            # For Tier 2: 100 * 0.9 = 90 concurrent requests
            self.max_workers = min(tier_limits.concurrency - 10, 90) if tier_limits.concurrency else 1
            if self.max_workers > 1:
                print(f"[ReasoningBackfiller] Using {self.max_workers} concurrent workers")

    def reconstruct_augmented_prompt(self, sample: Dict[str, Any]) -> str:
        """
        Reconstruct the augmented prompt from sample.

        Uses:
        - sample['prompt'] (original prompt)
        - sample.get('augmented_prompt') (if available)
        - sample.get('caws_context') (if available, to rebuild CAWS formatting)
        """
        # If augmented_prompt is available, use it
        if sample.get("augmented_prompt"):
            return sample["augmented_prompt"]

        # Otherwise, try to reconstruct from prompt + CAWS context
        prompt = sample.get("prompt", "")
        caws_context_dict = sample.get("caws_context") or sample.get("metadata", {}).get("caws_context")

        if caws_context_dict:
            # Try to reconstruct CAWS context string
            try:
                from training.caws_context import CAWSContext

                # Extract all required fields with defaults
                working_spec = caws_context_dict.get("working_spec", {})
                spec_id = caws_context_dict.get("spec_id") or working_spec.get("id", "unknown")
                title = caws_context_dict.get("title") or working_spec.get("title", "Unknown")
                risk_tier = caws_context_dict.get("risk_tier") or working_spec.get("risk_tier", 2)
                mode = caws_context_dict.get("mode") or working_spec.get("mode", "feature")
                budget = caws_context_dict.get("budget") or working_spec.get("budget", {})
                scope = caws_context_dict.get("scope") or working_spec.get("scope", {})
                quality = caws_context_dict.get("quality", {})
                acceptance_summary = caws_context_dict.get("acceptance_summary", [])
                invariants = caws_context_dict.get("invariants", [])

                # Create CAWSContext object with all required fields
                caws_context = CAWSContext(
                    spec_id=spec_id,
                    title=title,
                    risk_tier=risk_tier,
                    mode=mode,
                    budget=budget,
                    scope=scope,
                    quality=quality,
                    acceptance_summary=acceptance_summary,
                    invariants=invariants,
                )
                caws_context_str = format_caws_context_for_prompt(caws_context)
                if caws_context_str:
                    return f"{prompt}\n\n{caws_context_str}"
            except Exception as e:
                print(f"WARN: Failed to reconstruct CAWS context: {e}, using original prompt")

        return prompt

    def backfill_sample(self, sample: Dict[str, Any]) -> Optional[str]:
        """
        Backfill reasoning for a single sample.

        Returns:
            Reasoning content if successful, None otherwise.
        """
        # Reconstruct the augmented prompt
        augmented_prompt = self.reconstruct_augmented_prompt(sample)

        # Get original parameters from sample metadata if available
        metadata = sample.get("metadata", {})
        temperature = metadata.get("temperature", self.temperature)
        top_p = metadata.get("top_p", self.top_p)
        max_tokens = metadata.get("max_tokens", self.max_tokens)

        try:
            # Query teacher with same parameters
            teacher_results = self.client.sample(
                [augmented_prompt],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                return_logits=False,  # Don't need logits for backfill
            )

            if teacher_results and not teacher_results[0].get("error"):
                reasoning_content = teacher_results[0].get("reasoning_content")
                if reasoning_content:
                    return reasoning_content
                else:
                    print(f"WARN: No reasoning_content in response for sample {sample.get('id', 'unknown')}")
                    return None
            else:
                error_msg = teacher_results[0].get("error", "Unknown error") if teacher_results else "No results"
                print(f"ERROR: Teacher API error for sample {sample.get('id', 'unknown')}: {error_msg}")
                return None

        except Exception as e:
            print(f"ERROR: Exception while backfilling sample {sample.get('id', 'unknown')}: {e}")
            return None

    def backfill_dataset(
        self,
        input_file: Path,
        output_file: Path,
        checkpoint_interval: int = 50,
        resume: bool = False,
    ) -> Dict[str, Any]:
        """
        Backfill reasoning for all samples in a dataset.

        Returns:
            Statistics dictionary with counts and costs.
        """
        # Load samples
        samples: List[Dict[str, Any]] = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('{"__header__'):
                    continue
                try:
                    sample = json.loads(line)
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    print(f"ERROR: Invalid JSON: {e}")
                    continue

        print(f"Loaded {len(samples)} samples to backfill")

        # Check for checkpoint if resuming
        checkpoint_dir = output_file.parent / "checkpoints" / output_file.stem
        completed_indices: set = set()
        if resume and checkpoint_dir.exists():
            checkpoint_file = checkpoint_dir / "latest.json"
            if checkpoint_file.exists():
                try:
                    with open(checkpoint_file, "r") as f:
                        checkpoint_data = json.load(f)
                    completed_indices = set(checkpoint_data.get("completed_indices", []))
                    print(f"Resuming: {len(completed_indices)} samples already completed")
                except Exception as e:
                    print(f"WARN: Failed to load checkpoint: {e}")

        # Statistics
        stats = {
            "total": len(samples),
            "completed": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0,
        }

        # Create checkpoint directory
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Thread-safe locks
        results_lock = Lock()
        checkpoint_lock = Lock()

        # Collect results (index -> updated sample)
        results_dict: Dict[int, Dict[str, Any]] = {}

        # First, write all skipped samples (already completed)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as out_f:
            for i, sample in enumerate(samples):
                if i in completed_indices:
                    stats["skipped"] += 1
                    results_dict[i] = sample
                    out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        # Process pending samples
        pending_samples = [
            (i, sample) for i, sample in enumerate(samples) if i not in completed_indices
        ]

        if self.max_workers > 1 and len(pending_samples) > 1:
            # Use concurrent processing
            print(f"[ReasoningBackfiller] Processing {len(pending_samples)} samples with {self.max_workers} concurrent workers...")

            def process_single_sample(i: int, sample: Dict[str, Any]) -> tuple[int, Optional[str]]:
                """Process a single sample and return (index, reasoning_content)."""
                sample_id = sample.get("id", f"sample-{i}")
                print(f"[{i+1}/{len(samples)}] Backfilling reasoning for {sample_id}...")
                reasoning_content = self.backfill_sample(sample)
                return (i, reasoning_content)

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_index = {
                    executor.submit(process_single_sample, i, sample): i
                    for i, sample in pending_samples
                }

                # Process completed tasks as they finish
                for future in as_completed(future_to_index):
                    i = future_to_index[future]
                    try:
                        sample_idx, reasoning_content = future.result()
                        sample = samples[sample_idx]

                        if reasoning_content:
                            sample["teacher_reasoning"] = reasoning_content
                            sample["metadata"]["reasoning_backfilled_at"] = datetime.now().isoformat()
                            with results_lock:
                                stats["success"] += 1
                            print(f"  Success: {len(reasoning_content)} chars of reasoning")
                        else:
                            with results_lock:
                                stats["failed"] += 1
                            print(f"  Failed: No reasoning content obtained")

                        # Store result
                        with results_lock:
                            results_dict[sample_idx] = sample
                            stats["completed"] += 1

                        # Periodic checkpointing
                        if stats["completed"] % checkpoint_interval == 0:
                            with checkpoint_lock:
                                completed_indices.add(sample_idx)
                                checkpoint_data = {
                                    "completed_indices": list(completed_indices),
                                    "timestamp": datetime.now().isoformat(),
                                    "stats": dict(stats),
                                }
                                checkpoint_file = checkpoint_dir / "latest.json"
                                with open(checkpoint_file, "w") as f:
                                    json.dump(checkpoint_data, f, indent=2)
                                print(f"  Checkpoint saved: {stats['completed']}/{len(samples)} completed")

                    except Exception as e:
                        print(f"ERROR: Exception processing sample {i+1}: {e}")
                        with results_lock:
                            stats["failed"] += 1
                            stats["completed"] += 1
                            # Store original sample without reasoning
                            results_dict[i] = samples[i]

        else:
            # Sequential processing (original logic)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "a", encoding="utf-8") as out_f:
                for i, sample in pending_samples:
                    sample_id = sample.get("id", f"sample-{i}")

                    # Backfill reasoning
                    print(f"[{i+1}/{len(samples)}] Backfilling reasoning for {sample_id}...")
                    reasoning_content = self.backfill_sample(sample)

                    if reasoning_content:
                        sample["teacher_reasoning"] = reasoning_content
                        sample["metadata"]["reasoning_backfilled_at"] = datetime.now().isoformat()
                        stats["success"] += 1
                        print(f"  Success: {len(reasoning_content)} chars of reasoning")
                    else:
                        stats["failed"] += 1
                        print(f"  Failed: No reasoning content obtained")

                    # Write updated sample
                    out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    results_dict[i] = sample
                    stats["completed"] += 1

                    # Save checkpoint
                    if stats["completed"] % checkpoint_interval == 0:
                        completed_indices.add(i)
                        checkpoint_data = {
                            "completed_indices": list(completed_indices),
                            "timestamp": datetime.now().isoformat(),
                            "stats": dict(stats),
                        }
                        checkpoint_file = checkpoint_dir / "latest.json"
                        with open(checkpoint_file, "w") as f:
                            json.dump(checkpoint_data, f, indent=2)
                        print(f"  Checkpoint saved: {stats['completed']}/{len(samples)} completed")

                    # Rate limiting delay
                    if i < len(samples) - 1:  # Don't delay after last sample
                        time.sleep(self.delay)

        # Write all results in order (for concurrent mode)
        if self.max_workers > 1:
            with open(output_file, "w", encoding="utf-8") as out_f:
                for i in range(len(samples)):
                    if i in results_dict:
                        out_f.write(json.dumps(results_dict[i], ensure_ascii=False) + "\n")

        # Final checkpoint
        checkpoint_data = {
            "completed_indices": list(range(len(samples))),
            "timestamp": datetime.now().isoformat(),
            "stats": stats,
        }
        checkpoint_file = checkpoint_dir / "latest.json"
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill teacher_reasoning for KD samples"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL file with samples missing reasoning",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL file with reasoning backfilled",
    )
    parser.add_argument(
        "--teacher",
        required=True,
        help="Teacher endpoint (HTTP or hf:model_name)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Maximum tokens for teacher response",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for teacher sampling",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p for teacher sampling",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="Delay between API requests (seconds)",
    )
    parser.add_argument(
        "--tier",
        help="API tier override (tier1, tier2, etc.)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50,
        help="Save checkpoint every N samples",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: Input file does not exist: {args.input}")
        return 1

    print(f"Backfilling reasoning for samples in {args.input}...")
    print(f"Output: {args.output}")
    print(f"Teacher: {args.teacher}")

    backfiller = ReasoningBackfiller(
        teacher_endpoint=args.teacher,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        delay=args.delay,
        tier=args.tier,
    )

    stats = backfiller.backfill_dataset(
        input_file=args.input,
        output_file=args.output,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume,
    )

    print("\n" + "=" * 80)
    print("BACKFILL COMPLETE")
    print("=" * 80)
    print(f"Total samples: {stats['total']}")
    print(f"Successfully backfilled: {stats['success']}")
    print(f"Failed: {stats['failed']}")
    print(f"Skipped (already completed): {stats['skipped']}")
    print(f"Output written to: {args.output}")

    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
