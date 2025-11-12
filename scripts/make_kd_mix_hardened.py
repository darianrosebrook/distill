"""
Hardened KD dataset generator with resume, checkpointing, and robust error handling.

Key improvements:
- Resume from checkpoint (survives crashes/interruptions)
- Progress persistence (saves state every N samples)
- Budget tracking (monitors API costs)
- Validation of cached data
- Partial result saving
- Better error recovery
- Cost estimation before starting

Usage:
    python -m scripts.make_kd_mix_hardened \
        --out data/kd_mix.jsonl \
        --teacher https://api.kimi.com/v1 \
        --total 1000 \
        --checkpoint-dir data/checkpoints/ \
        --cache-dir data/kd_cache/ \
        --budget-limit 10.0
"""
import argparse
import json
import os
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import sys

from models.teacher.teacher_client import TeacherClient, APITier
from scripts.prompt_sources import get_prompt_mix, load_prompts_from_file


# Cost constants (from Kimi API pricing)
INPUT_COST_PER_MILLION = 0.60  # Cache-miss
INPUT_COST_CACHE_HIT_PER_MILLION = 0.15
OUTPUT_COST_PER_MILLION = 2.50


class BudgetTracker:
    """Track API costs and enforce budget limits."""
    
    def __init__(self, budget_limit: Optional[float] = None):
        self.budget_limit = budget_limit
        self.total_cost = 0.0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached_samples = 0
        self.api_samples = 0
        
    def add_sample(self, input_tokens: int, output_tokens: int, cached: bool = False):
        """Add a sample and calculate cost."""
        if cached:
            self.cached_samples += 1
            # Cache hit cost
            cost = (input_tokens / 1_000_000) * INPUT_COST_CACHE_HIT_PER_MILLION
        else:
            self.api_samples += 1
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            # Cache miss cost
            cost = (input_tokens / 1_000_000) * INPUT_COST_PER_MILLION
        
        cost += (output_tokens / 1_000_000) * OUTPUT_COST_PER_MILLION
        self.total_cost += cost
        
        # Check budget limit
        if self.budget_limit and self.total_cost > self.budget_limit:
            raise BudgetExceededError(
                f"Budget limit exceeded: ${self.total_cost:.2f} > ${self.budget_limit:.2f}"
            )
    
    def get_estimate(self, total_samples: int, avg_input_tokens: int = 200, avg_output_tokens: int = 1024) -> float:
        """Estimate total cost for dataset generation."""
        total_input = total_samples * avg_input_tokens
        total_output = total_samples * avg_output_tokens
        
        # Assume some cache hits (conservative: 0% for estimate)
        input_cost = (total_input / 1_000_000) * INPUT_COST_PER_MILLION
        output_cost = (total_output / 1_000_000) * OUTPUT_COST_PER_MILLION
        
        return input_cost + output_cost
    
    def get_status(self) -> Dict[str, Any]:
        """Get current budget status."""
        return {
            "total_cost": self.total_cost,
            "budget_limit": self.budget_limit,
            "remaining": self.budget_limit - self.total_cost if self.budget_limit else None,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_samples": self.cached_samples,
            "api_samples": self.api_samples,
        }


class BudgetExceededError(Exception):
    """Raised when budget limit is exceeded."""
    pass


class CheckpointManager:
    """Manage checkpoints for resuming interrupted runs."""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = checkpoint_dir / "progress.json"
        self.results_file = checkpoint_dir / "results.jsonl"
        
    def save_checkpoint(self, completed_indices: List[int], results: List[Dict[str, Any]], 
                       budget_tracker: BudgetTracker, start_time: float):
        """Save checkpoint state."""
        checkpoint_data = {
            "completed_indices": completed_indices,
            "total_completed": len(completed_indices),
            "budget": budget_tracker.get_status(),
            "start_time": start_time,
            "last_update": time.time(),
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        # Save partial results
        with open(self.results_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"[Checkpoint] Saved: {len(completed_indices)} samples completed")
    
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint state if exists."""
        if not self.checkpoint_file.exists():
            return None
        
        with open(self.checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
        
        # Load partial results
        results = []
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))
        
        checkpoint_data["results"] = results
        return checkpoint_data
    
    def clear_checkpoint(self):
        """Clear checkpoint files."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        if self.results_file.exists():
            self.results_file.unlink()


def load_cache(cache_dir: Path, prompt: str) -> Optional[Dict[str, Any]]:
    """Load cached result for a prompt with validation."""
    if not cache_dir.exists():
        return None
    
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    cache_file = cache_dir / f"{prompt_hash}.json"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            
            # Validate cached data
            if not isinstance(cached_data, dict):
                print(f"[Cache] WARN: Invalid cache format for {prompt_hash[:8]}, ignoring")
                return None
            
            if "prompt" not in cached_data or "teacher_text" not in cached_data:
                print(f"[Cache] WARN: Missing required fields in cache {prompt_hash[:8]}, ignoring")
                return None
            
            # Verify prompt matches (sanity check)
            if cached_data["prompt"] != prompt:
                print(f"[Cache] WARN: Prompt mismatch in cache {prompt_hash[:8]}, ignoring")
                return None
            
            return cached_data
        except json.JSONDecodeError:
            print(f"[Cache] WARN: Corrupted cache file {prompt_hash[:8]}, ignoring")
            return None
        except Exception as e:
            print(f"[Cache] WARN: Error loading cache {prompt_hash[:8]}: {e}, ignoring")
            return None
    
    return None


def save_cache(cache_dir: Path, prompt: str, result: Dict[str, Any]):
    """Save result to cache with atomic write."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    cache_file = cache_dir / f"{prompt_hash}.json"
    temp_file = cache_file.with_suffix('.tmp')
    
    try:
        # Atomic write: write to temp, then rename
        with open(temp_file, 'w') as f:
            json.dump(result, f, indent=2)
        temp_file.replace(cache_file)
    except Exception as e:
        print(f"[Cache] WARN: Failed to save cache {prompt_hash[:8]}: {e}")
        if temp_file.exists():
            temp_file.unlink()


def estimate_cost(total_samples: int, avg_input: int = 200, avg_output: int = 1024) -> Dict[str, Any]:
    """Estimate API costs for dataset generation."""
    tracker = BudgetTracker()
    estimated_cost = tracker.get_estimate(total_samples, avg_input, avg_output)
    
    return {
        "total_samples": total_samples,
        "estimated_cost": estimated_cost,
        "estimated_input_tokens": total_samples * avg_input,
        "estimated_output_tokens": total_samples * avg_output,
        "cost_per_sample": estimated_cost / total_samples if total_samples > 0 else 0,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Hardened KD dataset generator with resume and budget tracking",
        formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument('--out', required=True, help='Output JSONL file path')
    ap.add_argument('--teacher', required=True,
                    help='Teacher endpoint (http://host:port) or HuggingFace model (hf:model_name)')
    ap.add_argument('--total', type=int, default=1000, help='Total number of prompts to generate')
    ap.add_argument('--general-ratio', type=float, default=0.5, help='Ratio of general prompts')
    ap.add_argument('--domain-ratio', type=float, default=0.3, help='Ratio of domain-specific prompts')
    ap.add_argument('--tool-ratio', type=float, default=0.2, help='Ratio of tool trace prompts')
    ap.add_argument('--prompts-file', help='Load prompts from JSONL file instead of generating')
    ap.add_argument('--cache-dir', help='Directory to cache teacher responses')
    ap.add_argument('--checkpoint-dir', help='Directory for checkpoint files (enables resume)')
    ap.add_argument('--checkpoint-interval', type=int, default=50, help='Save checkpoint every N samples')
    ap.add_argument('--budget-limit', type=float, help='Maximum budget in USD (stops if exceeded)')
    ap.add_argument('--resume', action='store_true', help='Resume from checkpoint if available')
    ap.add_argument('--clear-checkpoint', action='store_true', help='Clear existing checkpoint before starting')
    ap.add_argument('--temperature', type=float, default=1.5, help='Sampling temperature')
    ap.add_argument('--top-p', type=float, default=0.95, help='Top-p sampling')
    ap.add_argument('--max-tokens', type=int, default=1024, help='Maximum tokens to generate')
    ap.add_argument('--return-logits', action='store_true', help='Request logits from teacher (if supported)')
    ap.add_argument('--batch-size', type=int, default=1, help='Batch size for teacher queries')
    ap.add_argument('--delay', type=float, default=0.1, help='Delay between requests (seconds)')
    args = ap.parse_args()
    
    # Estimate costs before starting
    cost_estimate = estimate_cost(args.total)
    print(f"[make_kd_mix_hardened] Cost estimate for {args.total} samples:")
    print(f"  Estimated cost: ${cost_estimate['estimated_cost']:.2f}")
    print(f"  Cost per sample: ${cost_estimate['cost_per_sample']:.4f}")
    print(f"  Estimated tokens: {cost_estimate['estimated_input_tokens']:,} input + {cost_estimate['estimated_output_tokens']:,} output")
    
    if args.budget_limit:
        if cost_estimate['estimated_cost'] > args.budget_limit:
            print(f"[make_kd_mix_hardened] WARN: Estimated cost (${cost_estimate['estimated_cost']:.2f}) exceeds budget limit (${args.budget_limit:.2f})")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                print("[make_kd_mix_hardened] Aborted by user")
                sys.exit(1)
    
    # Setup checkpoint manager
    checkpoint_manager = None
    if args.checkpoint_dir:
        checkpoint_manager = CheckpointManager(Path(args.checkpoint_dir))
        if args.clear_checkpoint:
            checkpoint_manager.clear_checkpoint()
            print("[make_kd_mix_hardened] Cleared existing checkpoint")
    
    # Load prompts
    if args.prompts_file:
        prompts = load_prompts_from_file(args.prompts_file)
    else:
        prompts = get_prompt_mix(
            total=args.total,
            general_ratio=args.general_ratio,
            domain_ratio=args.domain_ratio,
            tool_ratio=args.tool_ratio,
        )
    
    # Resume from checkpoint if requested
    completed_indices = set()
    results = []
    budget_tracker = BudgetTracker(budget_limit=args.budget_limit)
    start_time = time.time()
    
    if args.resume and checkpoint_manager:
        checkpoint_data = checkpoint_manager.load_checkpoint()
        if checkpoint_data:
            completed_indices = set(checkpoint_data["completed_indices"])
            results = checkpoint_data.get("results", [])
            budget_status = checkpoint_data.get("budget", {})
            budget_tracker.total_cost = budget_status.get("total_cost", 0.0)
            budget_tracker.input_tokens = budget_status.get("input_tokens", 0)
            budget_tracker.output_tokens = budget_status.get("output_tokens", 0)
            budget_tracker.cached_samples = budget_status.get("cached_samples", 0)
            budget_tracker.api_samples = budget_status.get("api_samples", 0)
            start_time = checkpoint_data.get("start_time", time.time())
            
            print(f"[make_kd_mix_hardened] Resumed from checkpoint: {len(completed_indices)}/{len(prompts)} samples completed")
            print(f"[make_kd_mix_hardened] Budget status: ${budget_tracker.total_cost:.2f} spent")
    
    # Initialize teacher client
    if args.teacher.startswith("hf:"):
        model_name = args.teacher[3:]
        client = TeacherClient.from_hf(model_name)
    else:
        client = TeacherClient.from_endpoint(
            args.teacher,
            max_retries=5,
            retry_backoff_factor=2.0
        )
    
    # Display tier info
    if hasattr(client, 'get_tier'):
        tier = client.get_tier()
        tier_limits = client.get_tier_limits()
        print(f"[make_kd_mix_hardened] API Tier: {tier.value}")
        print(f"[make_kd_mix_hardened] Rate limits: {tier_limits.rpm} RPM, {tier_limits.tpm:,} TPM")
        if tier_limits.tpd:
            print(f"[make_kd_mix_hardened] Daily limit: {tier_limits.tpd:,} tokens")
        print(f"[make_kd_mix_hardened] Recommended delay: {tier_limits.delay}s")
        
        # Warn if delay doesn't match tier
        if args.delay > 0 and abs(args.delay - tier_limits.delay) > tier_limits.delay * 0.5:
            print(f"[make_kd_mix_hardened] WARN: Delay ({args.delay}s) doesn't match tier recommendation ({tier_limits.delay}s)")
            print(f"[make_kd_mix_hardened] WARN: Consider using --delay {tier_limits.delay} for optimal rate limit compliance")
    
    # Setup cache
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    
    # Create output directory
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Sample from teacher
    cached = 0
    errors = 0
    consecutive_errors = 0
    max_consecutive_errors = 10
    
    print(f"[make_kd_mix_hardened] Sampling from teacher (batch_size={args.batch_size}, delay={args.delay}s)...")
    print(f"[make_kd_mix_hardened] Starting from sample {len(completed_indices) + 1}/{len(prompts)}")
    
    try:
        for i, prompt in enumerate(prompts):
            # Skip if already completed
            if i in completed_indices:
                continue
            
            # Check cache
            cached_result = None
            if cache_dir:
                cached_result = load_cache(cache_dir, prompt)
            
            if cached_result:
                results.append(cached_result)
                completed_indices.add(i)
                cached += 1
                
                # Estimate tokens for budget tracking
                teacher_text = cached_result.get("teacher_text", "")
                # Rough estimate: ~4 chars per token
                input_tokens = len(prompt) // 4
                output_tokens = len(teacher_text) // 4
                budget_tracker.add_sample(input_tokens, output_tokens, cached=True)
                
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = len(completed_indices) / elapsed if elapsed > 0 else 0
                    print(f"[make_kd_mix_hardened] Progress: {len(completed_indices)}/{len(prompts)} "
                          f"(cached: {cached}, errors: {errors}, rate: {rate:.1f} samples/s, "
                          f"cost: ${budget_tracker.total_cost:.2f})")
                continue
            
            # Query teacher (with built-in retry logic)
            try:
                teacher_results = client.sample(
                    [prompt],
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    return_logits=args.return_logits,
                )
                
                if teacher_results and not teacher_results[0].get("error"):
                    result = {
                        "prompt": prompt,
                        "teacher_text": teacher_results[0]["text"],
                        "teacher_logits": teacher_results[0].get("logits"),
                        "metadata": {
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "max_tokens": args.max_tokens,
                            "timestamp": datetime.now().isoformat(),
                        }
                    }
                    
                    results.append(result)
                    completed_indices.add(i)
                    consecutive_errors = 0
                    
                    # Track budget
                    teacher_text = result["teacher_text"]
                    input_tokens = len(prompt) // 4  # Rough estimate
                    output_tokens = len(teacher_text) // 4
                    budget_tracker.add_sample(input_tokens, output_tokens, cached=False)
                    
                    # Save to cache
                    if cache_dir:
                        save_cache(cache_dir, prompt, result)
                else:
                    errors += 1
                    consecutive_errors += 1
                    error_msg = teacher_results[0].get("error", "Unknown error") if teacher_results else "No response"
                    print(f"[make_kd_mix_hardened] WARN: Failed to get response for prompt {i+1}: {error_msg}")
                    
                    # Check for specific error types
                    if "rate limit" in error_msg.lower() or "429" in error_msg:
                        print(f"[make_kd_mix_hardened] Rate limit hit, consider increasing --delay or upgrading tier")
                    elif "token" in error_msg.lower() and "limit" in error_msg.lower():
                        print(f"[make_kd_mix_hardened] Token limit exceeded, consider reducing --max-tokens")
                    elif "connection" in error_msg.lower() or "network" in error_msg.lower():
                        print(f"[make_kd_mix_hardened] Network error, will retry on next attempt")
                    
                    # Stop if too many consecutive errors
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"[make_kd_mix_hardened] ERROR: {max_consecutive_errors} consecutive errors, stopping")
                        break
                        
            except BudgetExceededError as e:
                print(f"[make_kd_mix_hardened] ERROR: {e}")
                break
            except Exception as e:
                errors += 1
                consecutive_errors += 1
                print(f"[make_kd_mix_hardened] ERROR: Unexpected error for prompt {i+1}: {e}")
                if consecutive_errors >= max_consecutive_errors:
                    print(f"[make_kd_mix_hardened] ERROR: {max_consecutive_errors} consecutive errors, stopping")
                    break
            
            # Save checkpoint periodically
            if checkpoint_manager and len(completed_indices) % args.checkpoint_interval == 0:
                checkpoint_manager.save_checkpoint(
                    sorted(completed_indices),
                    results,
                    budget_tracker,
                    start_time
                )
            
            # Progress update
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = len(completed_indices) / elapsed if elapsed > 0 else 0
                remaining = len(prompts) - len(completed_indices)
                eta = remaining / rate if rate > 0 else 0
                print(f"[make_kd_mix_hardened] Progress: {len(completed_indices)}/{len(prompts)} "
                      f"(cached: {cached}, errors: {errors}, rate: {rate:.1f} samples/s, "
                      f"cost: ${budget_tracker.total_cost:.2f}, ETA: {eta/60:.1f} min)")
            
            # Rate limiting
            if args.delay > 0:
                time.sleep(args.delay)
        
        # Final checkpoint
        if checkpoint_manager:
            checkpoint_manager.save_checkpoint(
                sorted(completed_indices),
                results,
                budget_tracker,
                start_time
            )
        
        # Write results
        print(f"[make_kd_mix_hardened] Writing {len(results)} results to {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        # Final summary
        elapsed = time.time() - start_time
        budget_status = budget_tracker.get_status()
        print(f"[make_kd_mix_hardened] ✅ Complete:")
        print(f"  Samples: {len(results)}/{len(prompts)}")
        print(f"  Cached: {cached}")
        print(f"  Errors: {errors}")
        print(f"  Time: {elapsed/60:.1f} minutes")
        print(f"  Rate: {len(results)/elapsed:.2f} samples/s")
        print(f"  Budget: ${budget_status['total_cost']:.2f} spent")
        if budget_status['remaining']:
            print(f"  Remaining: ${budget_status['remaining']:.2f}")
        print(f"  Tokens: {budget_status['input_tokens']:,} input + {budget_status['output_tokens']:,} output")
        print(f"[make_kd_mix_hardened] Output: {output_path}")
        
    except KeyboardInterrupt:
        print(f"\n[make_kd_mix_hardened] Interrupted by user")
        if checkpoint_manager:
            checkpoint_manager.save_checkpoint(
                sorted(completed_indices),
                results,
                budget_tracker,
                start_time
            )
            print(f"[make_kd_mix_hardened] Checkpoint saved. Resume with --resume flag")
        sys.exit(1)


if __name__ == '__main__':
    main()


