#!/usr/bin/env python3
"""
Mutation testing runner for the distill project.

Runs mutatest on specified modules and generates reports.

Usage:
    python scripts/run_mutation_testing.py --module training/losses.py
    python scripts/run_mutation_testing.py --module training/distill_kd.py --mode f
    python scripts/run_mutation_testing.py --all-critical --mode s
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


# Critical modules that should have high mutation scores (70%+)
CRITICAL_MODULES = [
    "training/distill_kd.py",
    "training/losses.py",
    "training/input_validation.py",
    "conversion/export_onnx.py",
    "conversion/convert_coreml.py",
]

# Standard modules (target: 60%+)
STANDARD_MODULES = [
    "training/dataset.py",
    "training/process_losses.py",
    "models/teacher/teacher_client.py",
]

# Utility modules (target: 50%+)
UTILITY_MODULES = [
    "training/utils.py",
    "training/assertions.py",
]


def run_mutation_test(
    module: str,
    mode: str = "s",
    n_locations: int = 10,
    output_file: Optional[str] = None,
    exception_threshold: Optional[int] = None,
    test_cmd: Optional[str] = None,
    timeout_factor: float = 2.0,
) -> int:
    """
    Run mutation testing on a module.

    Args:
        module: Path to module to test
        mode: Mutation testing mode (f=full, s=break on survivor, d=break on detection, sd=both)
        n_locations: Number of locations to mutate (0 = all)
        output_file: Output RST file for results
        exception_threshold: Raise exception if this many survivors found
        test_cmd: Custom test command (default: pytest tests for module)
        timeout_factor: Factor for timeout detection

    Returns:
        Exit code (0 = success, non-zero = failure)
    """
    module_path = Path(module)
    if not module_path.exists():
        print(f"❌ Module not found: {module}")
        return 1

    # Determine test file path
    if test_cmd is None:
        # Try to find corresponding test file
        module_name = module_path.stem
        test_file = Path("tests") / module_path.parent / f"test_{module_name}.py"
        
        if test_file.exists():
            test_cmd = f"pytest {test_file} -x"
        else:
            # Fall back to running all tests (slower but works)
            test_cmd = "pytest tests/ -x"

    print(f"\n🧬 Running mutation testing on: {module}")
    print(f"   Mode: {mode}")
    print(f"   Test command: {test_cmd}")
    print(f"   Locations: {n_locations if n_locations > 0 else 'all'}")

    cmd = [
        "mutatest",
        "-s", str(module_path),
        "-t", test_cmd,
        "-m", mode,
        "--timeout_factor", str(timeout_factor),
        "--nocov",  # Ignore coverage files to avoid compatibility issues
    ]

    if n_locations > 0:
        cmd.extend(["-n", str(n_locations)])

    if output_file:
        cmd.extend(["-o", output_file])
        print(f"   Output: {output_file}")

    if exception_threshold is not None:
        cmd.extend(["--exception", str(exception_threshold)])

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\n⚠️  Mutation testing interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Error running mutation testing: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Run mutation testing on modules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test a single module
  python scripts/run_mutation_testing.py --module training/losses.py

  # Test with full mode (all mutations)
  python scripts/run_mutation_testing.py --module training/losses.py --mode f

  # Test all critical modules
  python scripts/run_mutation_testing.py --all-critical

  # Test with custom output and exception threshold
  python scripts/run_mutation_testing.py --module training/losses.py \\
      --output mutation_report.rst --exception 5
        """,
    )

    parser.add_argument(
        "--module",
        type=str,
        help="Module to test (e.g., training/losses.py)",
    )
    parser.add_argument(
        "--all-critical",
        action="store_true",
        help="Test all critical modules",
    )
    parser.add_argument(
        "--all-standard",
        action="store_true",
        help="Test all standard modules",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["f", "s", "d", "sd"],
        default="s",
        help="Mutation testing mode (f=full, s=survivor, d=detection, sd=both)",
    )
    parser.add_argument(
        "-n",
        "--n-locations",
        type=int,
        default=10,
        help="Number of locations to mutate (0 = all, default: 10)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output RST file for results",
    )
    parser.add_argument(
        "-x",
        "--exception",
        type=int,
        help="Raise exception if this many survivors found",
    )
    parser.add_argument(
        "-t",
        "--test-cmd",
        type=str,
        help="Custom test command (default: auto-detect from module)",
    )
    parser.add_argument(
        "--timeout-factor",
        type=float,
        default=2.0,
        help="Timeout factor for mutation trials (default: 2.0)",
    )

    args = parser.parse_args()

    # Determine modules to test
    modules: List[str] = []
    if args.module:
        modules = [args.module]
    elif args.all_critical:
        modules = CRITICAL_MODULES
    elif args.all_standard:
        modules = STANDARD_MODULES
    else:
        parser.print_help()
        print("\n❌ Must specify --module, --all-critical, or --all-standard")
        return 1

    # Run mutation testing on each module
    results = []
    for module in modules:
        output_file = args.output
        if output_file and len(modules) > 1:
            # Add module name to output file if testing multiple
            output_path = Path(output_file)
            output_file = str(
                output_path.parent / f"{output_path.stem}_{Path(module).stem}{output_path.suffix}"
            )

        exit_code = run_mutation_test(
            module=module,
            mode=args.mode,
            n_locations=args.n_locations,
            output_file=output_file,
            exception_threshold=args.exception,
            test_cmd=args.test_cmd,
            timeout_factor=args.timeout_factor,
        )
        results.append((module, exit_code))

    # Summary
    print("\n" + "=" * 60)
    print("Mutation Testing Summary")
    print("=" * 60)
    for module, exit_code in results:
        status = "✅ PASSED" if exit_code == 0 else "❌ FAILED"
        print(f"{status}: {module}")

    # Return non-zero if any failed
    return 0 if all(ec == 0 for _, ec in results) else 1


if __name__ == "__main__":
    sys.exit(main())

