#!/usr/bin/env python3
"""
Verification script for Priority 1: Remove reasoning_content loss, add process-step supervision.

Checks:
1. No reasoning_content_loss() function exists
2. Process-step loss functions exist
3. Combined KD loss uses process-step losses
4. CoT-free validation works
5. Dataset generation doesn't save reasoning_content
6. Extractors module exists and works
"""
import sys
import importlib.util
from pathlib import Path
import ast
import re

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_file_for_pattern(file_path: Path, pattern: str, should_exist: bool = True, description: str = ""):
    """Check if pattern exists in file."""
    if not file_path.exists():
        return False, f"File not found: {file_path}"
    
    content = file_path.read_text()
    exists = bool(re.search(pattern, content))
    
    if should_exist and not exists:
        return False, f"Pattern not found: {description or pattern}"
    elif not should_exist and exists:
        return False, f"Pattern should not exist: {description or pattern}"
    
    return True, f"Pattern check passed: {description or pattern}"


def check_function_exists(file_path: Path, function_name: str, should_exist: bool = True):
    """Check if function exists in Python file."""
    if not file_path.exists():
        return False, f"File not found: {file_path}"
    
    try:
        tree = ast.parse(file_path.read_text())
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        exists = function_name in functions
        
        if should_exist and not exists:
            return False, f"Function not found: {function_name}"
        elif not should_exist and exists:
            return False, f"Function should not exist: {function_name}"
        
        return True, f"Function check passed: {function_name}"
    except SyntaxError as e:
        return False, f"Syntax error in {file_path}: {e}"


def check_imports(file_path: Path, module_name: str):
    """Check if module can be imported."""
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            return False, f"Could not create spec for {file_path}"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return True, f"Successfully imported {module_name}"
    except Exception as e:
        return False, f"Import error: {e}"


def main():
    """Run all verification checks."""
    print("=" * 80)
    print("Priority 1 Verification: Remove reasoning_content, Add Process-Step Supervision")
    print("=" * 80)
    print()
    
    checks = []
    
    # Check 1: reasoning_content_loss() should NOT exist
    print("Check 1: reasoning_content_loss() should NOT exist")
    losses_py = project_root / "training" / "losses.py"
    ok, msg = check_function_exists(losses_py, "reasoning_content_loss", should_exist=False)
    checks.append(("reasoning_content_loss removed", ok))
    print(f"  {'✅' if ok else '❌'} {msg}")
    print()
    
    # Check 2: Process-step loss functions SHOULD exist
    print("Check 2: Process-step loss functions should exist")
    for func_name in ["tool_name_loss", "json_argument_loss", "integration_copy_loss"]:
        ok, msg = check_function_exists(losses_py, func_name, should_exist=True)
        checks.append((f"{func_name} exists", ok))
        print(f"  {'✅' if ok else '❌'} {msg}")
    print()
    
    # Check 3: combined_kd_loss should NOT have teacher_reasoning_content parameter
    print("Check 3: combined_kd_loss should NOT have teacher_reasoning_content parameter")
    ok, msg = check_file_for_pattern(
        losses_py,
        r"teacher_reasoning_content.*:.*Optional",
        should_exist=False,
        description="teacher_reasoning_content parameter in combined_kd_loss"
    )
    checks.append(("combined_kd_loss no reasoning_content param", ok))
    print(f"  {'✅' if ok else '❌'} {msg}")
    print()
    
    # Check 4: combined_kd_loss should have process-step parameters
    print("Check 4: combined_kd_loss should have process-step parameters")
    for param in ["tool_name_ids", "gold_json_text_ids", "tool_result_fields"]:
        ok, msg = check_file_for_pattern(
            losses_py,
            rf"{param}.*:.*Optional",
            should_exist=True,
            description=f"{param} parameter in combined_kd_loss"
        )
        checks.append((f"combined_kd_loss has {param}", ok))
        print(f"  {'✅' if ok else '❌'} {msg}")
    print()
    
    # Check 5: CoT-free validation in dataset.py
    print("Check 5: CoT-free validation in dataset.py")
    dataset_py = project_root / "training" / "dataset.py"
    ok, msg = check_file_for_pattern(
        dataset_py,
        r"CoT-free.*teacher_reasoning_content",
        should_exist=True,
        description="CoT-free validation raises ValueError"
    )
    checks.append(("CoT-free validation in dataset", ok))
    print(f"  {'✅' if ok else '❌'} {msg}")
    print()
    
    # Check 6: CoT-free validation in distill_kd.py
    print("Check 6: CoT-free validation in distill_kd.py")
    distill_kd_py = project_root / "training" / "distill_kd.py"
    ok, msg = check_file_for_pattern(
        distill_kd_py,
        r"CoT-free.*teacher_reasoning_content.*detected",
        should_exist=True,
        description="CoT-free validation in training loop"
    )
    checks.append(("CoT-free validation in distill_kd", ok))
    print(f"  {'✅' if ok else '❌'} {msg}")
    print()
    
    # Check 7: Dataset generation doesn't save reasoning_content
    print("Check 7: Dataset generation doesn't save reasoning_content")
    make_kd_script = project_root / "scripts" / "make_kd_mix_hardened.py"
    ok, msg = check_file_for_pattern(
        make_kd_script,
        r'result\["teacher_reasoning_content"\]',
        should_exist=False,
        description="Saving teacher_reasoning_content to result"
    )
    checks.append(("Dataset generation no reasoning_content save", ok))
    print(f"  {'✅' if ok else '❌'} {msg}")
    print()
    
    # Check 8: Extractors module exists
    print("Check 8: Extractors module exists and can be imported")
    extractors_py = project_root / "training" / "extractors.py"
    if extractors_py.exists():
        ok, msg = check_imports(extractors_py, "training.extractors")
        checks.append(("extractors module importable", ok))
        print(f"  {'✅' if ok else '❌'} {msg}")
        
        # Check for key functions
        for func_name in ["extract_tool_name_span", "extract_json_argument_spans", "identify_integration_spans"]:
            ok, msg = check_function_exists(extractors_py, func_name, should_exist=True)
            checks.append((f"{func_name} exists", ok))
            print(f"  {'✅' if ok else '❌'} {msg}")
    else:
        checks.append(("extractors module exists", False))
        print(f"  ❌ Extractors module not found: {extractors_py}")
    print()
    
    # Check 9: Process-step targets in collate function
    print("Check 9: Process-step targets handled in collate_kd_batch")
    ok, msg = check_file_for_pattern(
        dataset_py,
        r"tool_name_ids_list",
        should_exist=True,
        description="Process-step targets in collate function"
    )
    checks.append(("collate handles process-step targets", ok))
    print(f"  {'✅' if ok else '❌'} {msg}")
    print()
    
    # Check 10: Process-step targets extracted in distill_kd.py
    print("Check 10: Process-step targets extracted in distill_kd.py")
    ok, msg = check_file_for_pattern(
        distill_kd_py,
        r"tool_name_ids.*=.*batch\.get",
        should_exist=True,
        description="Extracting tool_name_ids from batch"
    )
    checks.append(("distill_kd extracts process-step targets", ok))
    print(f"  {'✅' if ok else '❌'} {msg}")
    print()
    
    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    
    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)
    
    for check_name, ok in checks:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"{status}: {check_name}")
    
    print()
    print(f"Total: {passed}/{total} checks passed")
    
    if passed == total:
        print("✅ All checks passed! Priority 1 implementation verified.")
        return 0
    else:
        print(f"❌ {total - passed} check(s) failed. Please review and fix.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

