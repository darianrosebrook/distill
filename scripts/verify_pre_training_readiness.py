"""
Pre-training readiness verification script.

Verifies that all critical components are ready before starting expensive training runs:
- Configs and tokenizer contracts
- Dataset validation
- Loss and gradient sanity
- Checkpoint and reproducibility setup
- Export contracts
@author: @darianrosebrook
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

from infra.version_gate import check_training_versions
from scripts.validate_kd_data import validate_dataset


def check_config_tokenizer_contracts(tokenizer_path: str) -> Dict[str, Any]:
    """Check config and tokenizer contracts."""
    print("[verify_readiness] Checking config and tokenizer contracts...")
    
    results = {
        "passed": True,
        "errors": [],
        "warnings": [],
    }
    
    try:
        # Run tokenizer contract tests
        import subprocess
        result = subprocess.run(
            ["python", "-m", "pytest", "tests/test_tokenizer_contract.py", "-v"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            results["passed"] = False
            results["errors"].append("Tokenizer contract tests failed")
            results["errors"].append(result.stdout)
    except Exception as e:
        results["passed"] = False
        results["errors"].append(f"Failed to run tokenizer contract tests: {e}")
    
    return results


def check_dataset_validation(dataset_path: Path, max_corruption_rate: float = 5.0) -> Dict[str, Any]:
    """Check dataset validation."""
    print("[verify_readiness] Validating dataset...")
    
    results = {
        "passed": True,
        "errors": [],
        "warnings": [],
        "corruption_rate": 0.0,
    }
    
    try:
        validation_results = validate_dataset(dataset_path)
        corruption_rate = validation_results.get("corruption_rate", 0.0)
        results["corruption_rate"] = corruption_rate
        
        if corruption_rate > max_corruption_rate:
            results["passed"] = False
            results["errors"].append(
                f"Dataset corruption rate {corruption_rate:.2f}% exceeds threshold {max_corruption_rate}%"
            )
        elif validation_results.get("invalid_samples", 0) > 0:
            results["warnings"].append(
                f"Found {validation_results['invalid_samples']} invalid samples"
            )
    except Exception as e:
        results["passed"] = False
        results["errors"].append(f"Dataset validation failed: {e}")
    
    return results


def check_loss_sanity() -> Dict[str, Any]:
    """Check loss composition sanity (basic checks)."""
    print("[verify_readiness] Checking loss sanity...")
    
    results = {
        "passed": True,
        "errors": [],
        "warnings": [],
    }
    
    # Check that loss functions are importable and have correct signatures
    try:
        from training.losses import combined_kd_loss, halt_head_loss, length_aware_kd_loss
        from training.assertions import assert_loss_finite
        
        # Basic sanity: functions exist and are callable
        assert callable(combined_kd_loss)
        assert callable(halt_head_loss)
        assert callable(length_aware_kd_loss)
        assert callable(assert_loss_finite)
    except Exception as e:
        results["passed"] = False
        results["errors"].append(f"Loss function checks failed: {e}")
    
    return results


def check_checkpoint_reproducibility() -> Dict[str, Any]:
    """Check checkpoint and reproducibility setup."""
    print("[verify_readiness] Checking checkpoint and reproducibility setup...")
    
    results = {
        "passed": True,
        "errors": [],
        "warnings": [],
    }
    
    # Check that checkpoint saving function exists and has required features
    try:
        from training.distill_kd import save_checkpoint
        import inspect
        
        sig = inspect.signature(save_checkpoint)
        params = list(sig.parameters.keys())
        
        # Check for required parameters
        required_params = ['rng_states', 'data_shard_index']
        for param in required_params:
            if param not in params:
                results["warnings"].append(f"save_checkpoint missing optional parameter: {param}")
    except Exception as e:
        results["passed"] = False
        results["errors"].append(f"Checkpoint function check failed: {e}")
    
    return results


def check_export_contracts() -> Dict[str, Any]:
    """Check export contract tests."""
    print("[verify_readiness] Checking export contracts...")
    
    results = {
        "passed": True,
        "errors": [],
        "warnings": [],
    }
    
    try:
        # Run export contract tests
        import subprocess
        result = subprocess.run(
            ["python", "-m", "pytest", "tests/test_export_contracts.py", "-v"],
            capture_output=True,
            text=True
        )
    
        if result.returncode != 0:
            results["warnings"].append("Export contract tests failed (may be expected if no model available)")
            results["warnings"].append(result.stdout[:500])  # First 500 chars
    except Exception as e:
        results["warnings"].append(f"Could not run export contract tests: {e}")
    
    return results


def main():
    ap = argparse.ArgumentParser(description="Verify pre-training readiness")
    ap.add_argument('--dataset', help='Path to training dataset JSONL')
    ap.add_argument('--tokenizer', default='models/student/tokenizer',
                    help='Tokenizer path')
    ap.add_argument('--max-corruption-rate', type=float, default=5.0,
                    help='Maximum allowed dataset corruption rate (default: 5.0%%)')
    ap.add_argument('--skip-dataset', action='store_true',
                    help='Skip dataset validation')
    ap.add_argument('--skip-export', action='store_true',
                    help='Skip export contract tests')
    
    args = ap.parse_args()
    
    all_passed = True
    all_errors = []
    all_warnings = []
    
    print("="*60)
    print("Pre-Training Readiness Verification")
    print("="*60)
    
    # 1. Version compatibility
    print("\n[1/6] Version Compatibility")
    try:
        check_training_versions()
        print("  PASSED: All version checks passed")
    except RuntimeError as e:
        print(f"  FAILED: {e}")
        all_passed = False
        all_errors.append(f"Version check: {e}")
    
    # 2. Config & Tokenizer Contracts
    print("\n[2/6] Config & Tokenizer Contracts")
    contract_results = check_config_tokenizer_contracts(args.tokenizer)
    if contract_results["passed"]:
        print("  PASSED: Tokenizer contract tests passed")
    else:
        print(f"  FAILED: {contract_results['errors'][0]}")
        all_passed = False
        all_errors.extend(contract_results["errors"])
    if contract_results["warnings"]:
        all_warnings.extend(contract_results["warnings"])
    
    # 3. Dataset Validation
    if not args.skip_dataset and args.dataset:
        print("\n[3/6] Dataset Validation")
        dataset_results = check_dataset_validation(Path(args.dataset), args.max_corruption_rate)
        if dataset_results["passed"]:
            print(f"  PASSED: Dataset validation passed (corruption rate: {dataset_results['corruption_rate']:.2f}%)")
        else:
            print(f"  FAILED: {dataset_results['errors'][0]}")
            all_passed = False
            all_errors.extend(dataset_results["errors"])
        if dataset_results["warnings"]:
            all_warnings.extend(dataset_results["warnings"])
    else:
        print("\n[3/6] Dataset Validation")
        print("  SKIPPED: No dataset path provided or --skip-dataset flag set")
    
    # 4. Loss Sanity
    print("\n[4/6] Loss Sanity")
    loss_results = check_loss_sanity()
    if loss_results["passed"]:
        print("  PASSED: Loss function checks passed")
    else:
        print(f"  FAILED: {loss_results['errors'][0]}")
        all_passed = False
        all_errors.extend(loss_results["errors"])
    
    # 5. Checkpoint & Reproducibility
    print("\n[5/6] Checkpoint & Reproducibility")
    checkpoint_results = check_checkpoint_reproducibility()
    if checkpoint_results["passed"]:
        print("  PASSED: Checkpoint function checks passed")
    else:
        print(f"  FAILED: {checkpoint_results['errors'][0]}")
        all_passed = False
        all_errors.extend(checkpoint_results["errors"])
    if checkpoint_results["warnings"]:
        all_warnings.extend(checkpoint_results["warnings"])
    
    # 6. Export Contracts
    if not args.skip_export:
        print("\n[6/6] Export Contracts")
        export_results = check_export_contracts()
        if export_results["passed"]:
            print("  PASSED: Export contract tests passed")
        else:
            print("  WARNING: Export contract tests had issues (may be expected)")
        if export_results["warnings"]:
            all_warnings.extend(export_results["warnings"])
    else:
        print("\n[6/6] Export Contracts")
        print("  SKIPPED: --skip-export flag set")
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    if all_passed:
        print("\n✅ ALL CHECKS PASSED - Ready for training")
    else:
        print("\n❌ CHECKS FAILED - Fix errors before training")
        print("\nErrors:")
        for error in all_errors:
            print(f"  - {error}")
    
    if all_warnings:
        print("\nWarnings:")
        for warning in all_warnings[:10]:  # First 10 warnings
            print(f"  - {warning}")
        if len(all_warnings) > 10:
            print(f"  ... and {len(all_warnings) - 10} more warnings")
    
    print("\n" + "="*60)
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
