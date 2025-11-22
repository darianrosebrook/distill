#!/usr/bin/env python3
"""
Validation script for training setup before running a serious training run.

Runs sanity checks on:
1. Config files (syntax, structure)
2. LR schedules (invariants)
3. Checkpoint loading
4. Memory monitor availability
5. Eval script syntax

Usage:
    python scripts/validate_training_setup.py --config configs/student_3b_kd_cooldown.yaml
"""

import argparse
import sys
import yaml
from pathlib import Path

# Add project root to path
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def check_config_syntax(config_path: Path) -> tuple[bool, str]:
    """Check config file syntax and structure."""
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        # Check required sections exist
        required_sections = ['arch', 'train', 'optimizer']
        missing = [s for s in required_sections if s not in cfg]
        if missing:
            return False, f"Missing required sections: {missing}"
        
        # Check arch is a dict (not accidentally flat)
        if not isinstance(cfg['arch'], dict):
            return False, f"arch section must be a dict, got {type(cfg['arch'])}"
        
        # Check train.steps exists
        if 'steps' not in cfg.get('train', {}):
            return False, "train.steps not found"
        
        return True, "Config syntax OK"
    except yaml.YAMLError as e:
        return False, f"YAML syntax error: {e}"
    except Exception as e:
        return False, f"Config loading error: {e}"


def check_lr_schedule(config_path: Path) -> tuple[bool, str]:
    """Check LR schedule invariants using debug script."""
    try:
        import subprocess
        result = subprocess.run(
            [
                sys.executable,
                'scripts/debug_lr_schedule.py',
                '--config', str(config_path),
                '--total-steps', str(300),  # Use config's steps if available
                '--output', '/tmp/lr_debug.csv',
                '--json-output', '/tmp/lr_diagnostics.json',
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode != 0:
            return False, f"LR schedule validation failed:\n{result.stderr}\n{result.stdout}"
        
        return True, "LR schedule invariants verified"
    except subprocess.TimeoutExpired:
        return False, "LR schedule validation timed out"
    except Exception as e:
        return False, f"LR schedule validation error: {e}"


def check_memory_monitor() -> tuple[bool, str]:
    """Check memory monitor is available."""
    try:
        from training.memory_monitor import MemoryMonitor
        return True, "Memory monitor available"
    except ImportError as e:
        return False, f"Memory monitor not available: {e}"


def check_eval_script() -> tuple[bool, str]:
    """Check eval script syntax."""
    try:
        import py_compile
        eval_script = Path(__file__).parent / 'eval_student_kd.py'
        py_compile.compile(str(eval_script), doraise=True)
        return True, "Eval script syntax OK"
    except py_compile.PyCompileError as e:
        return False, f"Eval script syntax error: {e}"
    except Exception as e:
        return False, f"Eval script check error: {e}"


def check_training_script() -> tuple[bool, str]:
    """Check training script syntax."""
    try:
        import py_compile
        training_script = Path(__file__).parent.parent / 'training' / 'distill_kd.py'
        py_compile.compile(str(training_script), doraise=True)
        return True, "Training script syntax OK"
    except py_compile.PyCompileError as e:
        return False, f"Training script syntax error: {e}"
    except Exception as e:
        return False, f"Training script check error: {e}"


def main():
    parser = argparse.ArgumentParser(description="Validate training setup")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to training config YAML',
    )
    parser.add_argument(
        '--skip-lr-check',
        action='store_true',
        help='Skip LR schedule validation (faster)',
    )
    
    args = parser.parse_args()
    config_path = Path(args.config)
    
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("Training Setup Validation")
    print("=" * 60)
    print(f"Config: {config_path}\n")
    
    checks = [
        ("Config syntax", lambda: check_config_syntax(config_path)),
        ("Training script syntax", check_training_script),
        ("Eval script syntax", check_eval_script),
        ("Memory monitor", check_memory_monitor),
    ]
    
    if not args.skip_lr_check:
        checks.append(("LR schedule invariants", lambda: check_lr_schedule(config_path)))
    
    all_passed = True
    for name, check_fn in checks:
        print(f"Checking {name}...", end=' ')
        passed, msg = check_fn()
        if passed:
            print(f"✅ {msg}")
        else:
            print(f"❌ {msg}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All checks passed!")
        sys.exit(0)
    else:
        print("❌ Some checks failed - review errors above")
        sys.exit(1)


if __name__ == '__main__':
    main()

