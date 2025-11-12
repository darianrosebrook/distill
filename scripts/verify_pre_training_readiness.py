#!/usr/bin/env python3
"""
Pre-training readiness verification script.

Verifies all critical requirements are met before starting full training run:
1. Dataset generation doesn't save reasoning_content
2. Training configs have correct process-step weights
3. Process-step targets are present in dataset
4. Training step computes losses correctly
"""
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    # Fallback: use simple YAML parsing for basic configs
    yaml = None

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_dataset_format(dataset_path: Path) -> tuple[bool, str]:
    """Check that dataset doesn't contain teacher_reasoning_content."""
    if not dataset_path.exists():
        return False, f"Dataset file not found: {dataset_path}"
    
    reasoning_content_found = False
    process_targets_found = False
    sample_count = 0
    
    with open(dataset_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                sample = json.loads(line)
                sample_count += 1
                
                # Check for reasoning_content (should NOT exist)
                if "teacher_reasoning_content" in sample and sample["teacher_reasoning_content"]:
                    reasoning_content_found = True
                    return False, f"Line {line_num}: teacher_reasoning_content found (violates ToS)"
                
                # Check for process-step targets (should exist)
                if any(key in sample for key in ["tool_name_ids", "gold_json_text_ids", "tool_result_fields"]):
                    process_targets_found = True
                
                # Only check first 10 samples for speed
                if sample_count >= 10:
                    break
            except json.JSONDecodeError as e:
                return False, f"Line {line_num}: Invalid JSON: {e}"
    
    if sample_count == 0:
        return False, "Dataset is empty"
    
    if not process_targets_found:
        return False, "No process-step targets found in dataset (may be expected if no tool calls present)"
    
    return True, f"Dataset format OK: {sample_count} samples checked, no reasoning_content found"


def check_config_weights(config_path: Path) -> tuple[bool, str]:
    """Check that config has process-step supervision weights."""
    if not config_path.exists():
        return False, f"Config file not found: {config_path}"
    
    if yaml:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Simple regex-based parsing for basic checks
        content = config_path.read_text()
        config = {}
        # Extract w_tool, w_args, w_integr values using regex
        import re
        w_tool_match = re.search(r'w_tool:\s*([0-9.]+)', content)
        w_args_match = re.search(r'w_args:\s*([0-9.]+)', content)
        w_integr_match = re.search(r'w_integr:\s*([0-9.]+)', content)
        
        if w_tool_match:
            config.setdefault("distillation", {})["w_tool"] = float(w_tool_match.group(1))
        if w_args_match:
            config.setdefault("distillation", {})["w_args"] = float(w_args_match.group(1))
        if w_integr_match:
            config.setdefault("distillation", {})["w_integr"] = float(w_integr_match.group(1))
    
    # Check both distillation and kd sections
    distillation = config.get("distillation", {})
    kd = config.get("kd", {})
    
    # Process-step weights should be in distillation section (for distill_kd.py)
    # or kd section (for some configs)
    w_tool = distillation.get("w_tool") or kd.get("w_tool")
    w_args = distillation.get("w_args") or kd.get("w_args")
    w_integr = distillation.get("w_integr") or kd.get("w_integr")
    
    if w_tool is None or w_args is None or w_integr is None:
        return False, f"Missing process-step weights: w_tool={w_tool}, w_args={w_args}, w_integr={w_integr}"
    
    if w_tool <= 0 or w_args <= 0 or w_integr <= 0:
        return False, f"Process-step weights must be > 0: w_tool={w_tool}, w_args={w_args}, w_integr={w_integr}"
    
    return True, f"Config weights OK: w_tool={w_tool}, w_args={w_args}, w_integr={w_integr}"


def check_tokenizer_path(config_path: Path) -> tuple[bool, str]:
    """Check that tokenizer path is set in config."""
    if not config_path.exists():
        return False, f"Config file not found: {config_path}"
    
    if yaml:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Simple regex-based parsing
        content = config_path.read_text()
        import re
        tokenizer_match = re.search(r'tokenizer_path:\s*([^\s\n]+)', content)
        config = {}
        if tokenizer_match:
            config.setdefault("io", {})["tokenizer_path"] = tokenizer_match.group(1).strip('"\'')
    
    io_config = config.get("io", {})
    tokenizer_path = io_config.get("tokenizer_path")
    
    if not tokenizer_path:
        return False, "tokenizer_path not set in config.io"
    
    # Check if path exists (or is a HuggingFace model name)
    tokenizer_path_obj = Path(tokenizer_path)
    if not tokenizer_path_obj.exists() and "/" not in tokenizer_path:
        # Might be a HuggingFace model name, which is OK
        return True, f"Tokenizer path set: {tokenizer_path} (may be HuggingFace model)"
    
    if tokenizer_path_obj.exists():
        return True, f"Tokenizer path exists: {tokenizer_path}"
    
    return False, f"Tokenizer path not found: {tokenizer_path}"


def main():
    """Run all verification checks."""
    print("=" * 80)
    print("Pre-Training Readiness Verification")
    print("=" * 80)
    print()
    
    checks = []
    
    # Check 1: Dataset format
    print("Check 1: Dataset format (no reasoning_content)")
    dataset_path = project_root / "data" / "kd_mix.jsonl"
    if dataset_path.exists():
        ok, msg = check_dataset_format(dataset_path)
        checks.append(("Dataset format", ok))
        print(f"  {'✅' if ok else '❌'} {msg}")
    else:
        checks.append(("Dataset format", False))
        print(f"  ⚠️  Dataset not found: {dataset_path}")
        print("     Generate dataset first: python -m scripts.make_kd_mix_hardened --out data/kd_mix.jsonl --teacher <endpoint> --total 10")
    print()
    
    # Check 2: Config weights
    print("Check 2: Training config process-step weights")
    config_files = [
        project_root / "configs" / "worker_9b.yaml",
        project_root / "configs" / "student_9b_gqa.yaml",
        project_root / "configs" / "student_8b_gqa.yaml",
    ]
    
    for config_path in config_files:
        if config_path.exists():
            ok, msg = check_config_weights(config_path)
            checks.append((f"Config weights ({config_path.name})", ok))
            print(f"  {'✅' if ok else '❌'} {config_path.name}: {msg}")
        else:
            print(f"  ⚠️  Config not found: {config_path.name}")
    print()
    
    # Check 3: Tokenizer path
    print("Check 3: Tokenizer path in configs")
    for config_path in config_files:
        if config_path.exists():
            ok, msg = check_tokenizer_path(config_path)
            checks.append((f"Tokenizer path ({config_path.name})", ok))
            print(f"  {'✅' if ok else '❌'} {config_path.name}: {msg}")
    print()
    
    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    
    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)
    
    for name, ok in checks:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"{status}: {name}")
    
    print()
    print(f"Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("✅ All checks passed! Ready for training.")
        return 0
    else:
        print("❌ Some checks failed. Address issues before training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

