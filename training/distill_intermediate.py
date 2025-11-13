"""
Intermediate layer distillation training (DEPRECATED).

This module is deprecated. Intermediate layer loss is now integrated into
training.distill_kd. Use the main KD training script instead.

Usage:
    # OLD (deprecated):
    python -m training.distill_intermediate --config configs/student_7b_gqa.yaml
    
    # NEW (use this instead):
    python -m training.distill_kd --config configs/worker_9b.yaml configs/kd_recipe.yaml
    # Intermediate layer loss is enabled via kd_recipe.yaml:
    # losses:
    #   intermediate_layer_weight: 0.1
"""
import sys
import typer


def main(config: str = typer.Argument(...)):
    """
    DEPRECATED: Intermediate distillation training entry point.
    
    This script is deprecated. Intermediate layer loss is now integrated
    into training.distill_kd. Use the main KD training script instead.
    
    Args:
        config: Configuration file path (ignored, script is deprecated)
    """
    print("=" * 80)
    print("ERROR: training.distill_intermediate is DEPRECATED")
    print("=" * 80)
    print()
    print("Intermediate layer distillation is now integrated into training.distill_kd.")
    print("Please use the main KD training script instead:")
    print()
    print("  python -m training.distill_kd --config configs/worker_9b.yaml configs/kd_recipe.yaml")
    print()
    print("To enable intermediate layer loss, configure in kd_recipe.yaml:")
    print()
    print("  losses:")
    print("    intermediate_layer_weight: 0.1  # Enable intermediate layer matching")
    print()
    print("The 'inter' Makefile target is also deprecated.")
    print("Use 'make worker' or 'make kd' instead.")
    print()
    print("=" * 80)
    sys.exit(1)


if __name__ == "__main__":
    typer.run(main)
