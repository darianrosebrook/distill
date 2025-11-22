#!/usr/bin/env python3
"""
LR Schedule Debug Harness

Instruments and visualizes learning rate schedule behavior to verify:
- Warmup region is finite and correct
- LR never increases after warmup
- Max LR never exceeds configured optimizer.lr
- Schedule matches expected behavior for short runs

Usage:
    python scripts/debug_lr_schedule.py --config configs/student_3b_kd_test.yaml --total-steps 1500 --output lr_schedule_debug.csv
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.distill_kd import create_optimizer


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_scheduler_from_config(
    optimizer: torch.optim.Optimizer,
    cfg: Dict[str, Any],
    total_steps: int,
) -> tuple[torch.optim.lr_scheduler.LambdaLR, int]:
    """
    Create LR scheduler exactly as distill_kd.py does.

    Returns:
        (scheduler, warmup_steps)
    """
    import math

    train_cfg = cfg.get("train", {})
    
    # Adaptive warmup calculation (from distill_kd.py lines 2681-2687)
    # Create a dummy model to compute parameter count
    # For debug purposes, we'll use a simple estimate or create minimal model
    d_model = cfg.get("arch", {}).get("d_model", 2560)
    n_layers = cfg.get("arch", {}).get("n_layers", 20)
    n_heads = cfg.get("arch", {}).get("n_heads", 20)
    vocab_size = cfg.get("arch", {}).get("vocab_size", 32000)
    
    # Rough parameter count estimate for 3B model
    # This is approximate but sufficient for warmup calculation
    model_params = d_model * vocab_size + n_layers * (4 * d_model * d_model)
    model_size_factor = math.log(max(1, model_params)) / 10
    
    base_warmup_pct = 0.1  # 10% of training
    adaptive_warmup_pct = min(0.2, max(0.05, base_warmup_pct * model_size_factor))
    warmup_steps = int(total_steps * adaptive_warmup_pct)
    
    # LR lambda function (from distill_kd.py lines 2705-2716)
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return step / max(1, warmup_steps)
        else:
            # Cosine annealing with final LR floor
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine_decay = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
            min_lr_factor = 0.01  # Don't decay below 1% of peak LR
            return max(min_lr_factor, cosine_decay)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler, warmup_steps


def simulate_training_steps(
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
) -> list[Dict[str, Any]]:
    """
    Simulate training steps and log LR at each step.
    
    Returns:
        List of dicts with step, lr, phase info
    """
    results = []
    
    for step in range(1, total_steps + 1):
        # Determine phase
        if step < warmup_steps:
            phase = "warmup"
        else:
            phase = "decay"
        
        # Get current LR
        current_lr = optimizer.param_groups[0]["lr"]
        
        # Step scheduler (as training loop does)
        scheduler.step()
        
        results.append({
            "step": step,
            "lr": current_lr,
            "phase": phase,
            "warmup_steps": warmup_steps,
            "base_lr": base_lr,
        })
    
    return results


def verify_schedule_properties(results: list[Dict[str, Any]], base_lr: float) -> Dict[str, Any]:
    """
    Verify schedule properties and return diagnostics.
    
    Enforces strict invariants:
    - LR never exceeds base_lr
    - LR never increases after warmup (monotonic decay)
    - Warmup is finite
    
    Returns:
        Dict with verification results
    """
    diagnostics = {
        "warmup_finite": False,
        "warmup_steps": 0,
        "lr_never_increases_after_warmup": True,
        "max_lr_never_exceeds_base": True,
        "final_lr_reasonable": True,
        "violations": [],
    }
    
    if not results:
        diagnostics["violations"].append("No results to verify")
        return diagnostics
    
    # Find warmup region
    warmup_steps = results[0]["warmup_steps"]
    diagnostics["warmup_steps"] = warmup_steps
    
    # Check warmup is finite
    if warmup_steps < len(results):
        diagnostics["warmup_finite"] = True
    else:
        diagnostics["violations"].append(f"Warmup steps ({warmup_steps}) >= total steps ({len(results)})")
    
    # Extract LR list for invariant checks
    lr_list = [r["lr"] for r in results]
    
    # Strict invariant: LR never exceeds base_lr (with small tolerance for floating point)
    tolerance = 1e-12
    for i, lr in enumerate(lr_list):
        if lr > base_lr + tolerance:
            diagnostics["max_lr_never_exceeds_base"] = False
            diagnostics["violations"].append(
                f"LR at step {results[i]['step']} ({lr:.2e}) exceeds base LR ({base_lr:.2e})"
            )
    
    # Strict invariant: LR never increases after warmup (monotonic decay)
    decay_region = [r for r in results if r["step"] > warmup_steps]
    if len(decay_region) > 1:
        for i in range(len(decay_region) - 1):
            curr_lr = decay_region[i]["lr"]
            next_lr = decay_region[i + 1]["lr"]
            # Allow tiny tolerance for floating point errors
            if next_lr > curr_lr + tolerance:
                diagnostics["lr_never_increases_after_warmup"] = False
                diagnostics["violations"].append(
                    f"LR increased after warmup at step {decay_region[i+1]['step']}: "
                    f"{curr_lr:.2e} -> {next_lr:.2e}"
                )
    
    # Check final LR is reasonable (should be decaying, not increasing)
    if len(results) > 1:
        final_lr = results[-1]["lr"]
        mid_lr = results[len(results) // 2]["lr"]
        # For short-run schedules, final LR should be <= mid LR
        # Allow 1% tolerance for floating point
        if final_lr > mid_lr * 1.01:
            diagnostics["final_lr_reasonable"] = False
            diagnostics["violations"].append(
                f"Final LR ({final_lr:.2e}) is higher than mid-run LR ({mid_lr:.2e})"
            )
    
    return diagnostics


def main():
    parser = argparse.ArgumentParser(description="Debug LR schedule behavior")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=1500,
        help="Total training steps to simulate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="lr_schedule_debug.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Optional JSON output file for diagnostics",
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)
    
    cfg = load_config(str(config_path))
    
    # Get optimizer config
    opt_cfg = cfg.get("optimizer", {})
    base_lr = opt_cfg.get("lr", 2.0e-4)
    
    # Create dummy optimizer (we only need it for scheduler)
    # Create minimal model for optimizer
    dummy_model = nn.Linear(10, 10)
    optimizer = create_optimizer(dummy_model, cfg)
    
    # Create scheduler
    scheduler, warmup_steps = create_scheduler_from_config(
        optimizer, cfg, args.total_steps
    )
    
    print(f"Config: {config_path}")
    print(f"Base LR: {base_lr:.2e}")
    print(f"Total steps: {args.total_steps}")
    print(f"Warmup steps: {warmup_steps} ({warmup_steps/args.total_steps*100:.1f}%)")
    print(f"Decay steps: {args.total_steps - warmup_steps}")
    print()
    
    # Simulate training
    print("Simulating training steps...")
    results = simulate_training_steps(scheduler, optimizer, args.total_steps, warmup_steps, base_lr)
    
    # Verify properties with strict invariants
    print("Verifying schedule properties...")
    diagnostics = verify_schedule_properties(results, base_lr)
    
    # Print diagnostics
    print("\n=== Schedule Diagnostics ===")
    print(f"Warmup finite: {diagnostics['warmup_finite']}")
    print(f"Warmup steps: {diagnostics['warmup_steps']}")
    print(f"LR never increases after warmup: {diagnostics['lr_never_increases_after_warmup']}")
    print(f"Max LR never exceeds base: {diagnostics['max_lr_never_exceeds_base']}")
    print(f"Final LR reasonable: {diagnostics['final_lr_reasonable']}")
    
    # Strict invariant checks (exit with error if violated)
    all_passed = (
        diagnostics['warmup_finite'] and
        diagnostics['lr_never_increases_after_warmup'] and
        diagnostics['max_lr_never_exceeds_base'] and
        diagnostics['final_lr_reasonable']
    )
    
    if diagnostics["violations"]:
        print("\n⚠️  VIOLATIONS DETECTED:")
        for violation in diagnostics["violations"]:
            print(f"  - {violation}")
        print("\n❌ Schedule invariants FAILED - do not use this config for training!")
        sys.exit(1)
    else:
        print("\n✅ All schedule properties verified!")
    
    # Print key checkpoints
    print("\n=== Key Checkpoints ===")
    key_steps = [750, 1000, 1125, 1200, 1300, 1400, 1500]
    for step in key_steps:
        if step <= len(results):
            r = results[step - 1]  # 0-indexed
            print(f"Step {step:4d}: LR = {r['lr']:.2e}, Phase = {r['phase']}")
    
    # Write CSV output
    output_path = Path(args.output)
    print(f"\nWriting results to {output_path}...")
    with open(output_path, "w") as f:
        f.write("step,lr,phase,warmup_steps,base_lr\n")
        for r in results:
            f.write(f"{r['step']},{r['lr']:.10e},{r['phase']},{r['warmup_steps']},{r['base_lr']:.10e}\n")
    print(f"✅ Wrote {len(results)} rows to {output_path}")
    
    # Write JSON diagnostics if requested
    if args.json_output:
        json_path = Path(args.json_output)
        with open(json_path, "w") as f:
            json.dump({
                "config_path": str(config_path),
                "total_steps": args.total_steps,
                "base_lr": base_lr,
                "warmup_steps": warmup_steps,
                "diagnostics": diagnostics,
                "key_checkpoints": {
                    step: {
                        "lr": results[step - 1]["lr"],
                        "phase": results[step - 1]["phase"],
                    }
                    for step in key_steps
                    if step <= len(results)
                },
            }, f, indent=2)
        print(f"✅ Wrote diagnostics to {json_path}")
    
    # Exit with error code if violations found
    if diagnostics["violations"]:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

