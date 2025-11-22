"""
Promote dataset version to canonical training file.

Creates symlink from canonical training file to versioned dataset and
updates dataset README with version information.

Author: @darianrosebrook
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_report(report_path: Path) -> Optional[dict]:
    """Load acceptance criteria from coverage report."""
    if not report_path.exists():
        return None
    
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Extract acceptance criteria status from markdown
        all_pass = "**Overall Status**: ✅ **ALL CRITERIA PASSED**" in content
        
        return {"all_pass": all_pass}
    except Exception:
        return None


def update_readme_file(readme_path: Path, version: str, dataset_path: Path, report_path: Optional[Path] = None):
    """Update dataset README with version information."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Read existing README or create new
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        content = "# Dataset Versions\n\n"
    
    # Add or update version entry
    report_status = False
    if report_path:
        report = load_report(report_path)
        if report:
            report_status = report.get("all_pass", False)
    
    version_section = f"""
## Version {version}

**Promoted**: {timestamp}  
**Dataset**: `{dataset_path.name}`  
**Status**: {'✅ Ready for training' if report_status else '⚠️ Check coverage report'}

"""
    
    # Check if version section already exists
    if f"## Version {version}" in content:
        # Replace existing section
        import re
        pattern = rf"## Version {re.escape(version)}.*?(?=## Version|\Z)"
        content = re.sub(pattern, version_section, content, flags=re.DOTALL)
    else:
        # Append new section
        content += version_section
    
    # Write updated README
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)


def promote_dataset(
    input_dataset: Path,
    canonical_path: Path,
    version: str,
    update_readme: bool = True,
    report_path: Optional[Path] = None,
) -> int:
    """
    Promote dataset version to canonical training file.
    
    Args:
        input_dataset: Versioned dataset file (e.g., worker_production_tools_v1.jsonl)
        canonical_path: Canonical training file path (e.g., worker_production_train.jsonl)
        version: Version string (e.g., "v1.1")
        update_readme: Whether to update README
        report_path: Optional path to coverage report for validation
        
    Returns:
        0 on success, 1 on failure
    """
    if not input_dataset.exists():
        print(f"[promote_dataset] ERROR: Input dataset not found: {input_dataset}")
        return 1
    
    # Validate report if provided
    if report_path and report_path.exists():
        report = load_report(report_path)
        if report and not report.get("all_pass"):
            print(f"[promote_dataset] WARN: Coverage report indicates criteria not passed")
            print(f"[promote_dataset] Proceeding anyway (use --require-pass to enforce)")
    
    # Create canonical directory if needed
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove existing symlink/file if it exists
    if canonical_path.exists() or canonical_path.is_symlink():
        if canonical_path.is_symlink():
            canonical_path.unlink()
        else:
            print(f"[promote_dataset] WARN: Canonical path exists as file, not symlink: {canonical_path}")
            response = input(f"Overwrite? (y/N): ")
            if response.lower() != "y":
                print(f"[promote_dataset] Aborted")
                return 1
            canonical_path.unlink()
    
    # Create symlink
    try:
        # Use absolute paths for symlink
        input_abs = input_dataset.resolve()
        canonical_abs = canonical_path.resolve()
        
        # Create relative symlink if possible (more portable)
        try:
            relative_path = Path(input_abs).relative_to(canonical_abs.parent)
            canonical_path.symlink_to(relative_path)
        except ValueError:
            # If not in same directory tree, use absolute path
            canonical_path.symlink_to(input_abs)
        
        print(f"[promote_dataset] Created symlink: {canonical_path} -> {input_dataset}")
    except Exception as e:
        print(f"[promote_dataset] ERROR: Failed to create symlink: {e}")
        return 1
    
    # Update README if requested
    if update_readme:
        readme_path = canonical_path.parent / "README.md"
        update_readme_file(readme_path, version, input_dataset, report_path)
        print(f"[promote_dataset] Updated README: {readme_path}")
    
    print(f"[promote_dataset] SUCCESS: Promoted {version} to canonical training file")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="Promote dataset version to canonical training file",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument(
        "--input",
        required=True,
        help="Input versioned dataset (e.g., worker_production_tools_v1.jsonl)",
    )
    ap.add_argument(
        "--canonical",
        required=True,
        help="Canonical training file path (e.g., worker_production_train.jsonl)",
    )
    ap.add_argument(
        "--version",
        required=True,
        help="Version string (e.g., v1.1)",
    )
    ap.add_argument(
        "--update-readme",
        action="store_true",
        default=True,
        help="Update dataset README with version info (default: True)",
    )
    ap.add_argument(
        "--no-update-readme",
        dest="update_readme",
        action="store_false",
        help="Skip README update",
    )
    ap.add_argument(
        "--report",
        help="Path to coverage report for validation",
    )
    ap.add_argument(
        "--require-pass",
        action="store_true",
        help="Require coverage report to pass before promotion",
    )
    
    args = ap.parse_args()
    
    input_file = Path(args.input)
    canonical_file = Path(args.canonical)
    report_file = Path(args.report) if args.report else None
    
    # Validate report if required
    if args.require_pass and report_file:
        if not report_file.exists():
            print(f"[promote_dataset] ERROR: Coverage report not found: {report_file}")
            return 1
        
        report = load_report(report_file)
        if not report or not report.get("all_pass"):
            print(f"[promote_dataset] ERROR: Coverage report indicates criteria not passed")
            return 1
    
    return promote_dataset(
        input_file,
        canonical_file,
        args.version,
        update_readme=args.update_readme,
        report_path=report_file,
    )


if __name__ == "__main__":
    sys.exit(main())

