"""
Data provenance validation script.

Validates dataset provenance, licensing, and PII handling.
Ensures all samples have proper source, license, and PII flags.
@author: @darianrosebrook
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict

import typer

app = typer.Typer()

# Whitelist of allowed sources
ALLOWED_SOURCES = {
    "teacher_api",  # Teacher API responses
    "synthetic_generation",  # Synthetically generated
    "open_dataset",  # Open datasets (with proper license)
}

# Whitelist of allowed licenses
ALLOWED_LICENSES = {
    "apache-2.0",
    "mit",
    "bsd-3-clause",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "public-domain",
    "custom_tos",  # Custom ToS (e.g., teacher API ToS)
}


def validate_provenance(
    sample: Dict[str, Any],
    line_num: int,
) -> List[str]:
    """
    Validate provenance fields for a single sample.

    Args:
        sample: Sample dictionary
        line_num: Line number for error reporting

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Check required fields
    if "source" not in sample:
        errors.append(f"Line {line_num}: Missing 'source' field")
    elif sample["source"] not in ALLOWED_SOURCES:
        errors.append(
            f"Line {line_num}: Unknown source '{sample['source']}'. "
            f"Allowed sources: {', '.join(ALLOWED_SOURCES)}"
        )

    if "license" not in sample:
        errors.append(f"Line {line_num}: Missing 'license' field")
    elif sample["license"] not in ALLOWED_LICENSES:
        errors.append(
            f"Line {line_num}: Unknown license '{sample['license']}'. "
            f"Allowed licenses: {', '.join(ALLOWED_LICENSES)}"
        )

    if "pii_flag" not in sample:
        errors.append(f"Line {line_num}: Missing 'pii_flag' field")
    elif sample["pii_flag"] is True:
        # Check if PII is properly redacted
        prompt = sample.get("prompt", "")
        teacher_text = sample.get("teacher_text", "")

        # Check for common PII patterns (simplified check - production would use proper PII detection library)
        pii_patterns = [
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
            r"\b\d{3}\.\d{3}\.\d{4}\b",  # Phone
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
        ]

        import re

        has_pii = False
        for pattern in pii_patterns:
            if re.search(pattern, prompt) or re.search(pattern, teacher_text):
                has_pii = True
                break

        if has_pii:
            errors.append(
                f"Line {line_num}: PII detected but pii_flag=True. "
                "PII must be redacted before being written to disk."
            )

    return errors


def validate_dataset_provenance(
    jsonl_path: Path,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Validate provenance for entire dataset.

    Args:
        jsonl_path: Path to dataset JSONL file
        strict: If True, fail on any provenance error

    Returns:
        Dictionary with validation results
    """
    total_samples = 0
    valid_samples = 0
    invalid_samples = 0
    total_errors = 0
    error_details = defaultdict(int)

    with open(jsonl_path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                data = json.loads(line)

                # Skip header lines
                if data.get("__header__", False):
                    continue

                total_samples += 1

                # Validate provenance
                errors = validate_provenance(data, line_num)

                if errors:
                    invalid_samples += 1
                    total_errors += len(errors)
                    for error in errors:
                        error_details[error] += 1
                else:
                    valid_samples += 1

            except json.JSONDecodeError as e:
                invalid_samples += 1
                total_errors += 1
                error_details[f"JSONDecodeError: {e}"] += 1
            except Exception as e:
                invalid_samples += 1
                total_errors += 1
                error_details[f"Unhandled error: {e}"] += 1

    results = {
        "total_samples": total_samples,
        "valid_samples": valid_samples,
        "invalid_samples": invalid_samples,
        "total_errors": total_errors,
        "error_summary": dict(error_details),
        "pass": invalid_samples == 0,
    }

    return results


@app.command()
def main(
    input_path: str = typer.Argument(..., help="Path to dataset JSONL file"),
    strict: bool = typer.Option(True, help="Fail on any provenance error"),
    output_path: Optional[str] = typer.Option(None, help="Output path for validation report"),
):
    """
    Validate data provenance for a dataset.
    """
    jsonl_path = Path(input_path)

    if not jsonl_path.exists():
        print(f"[validate_provenance] ERROR: Dataset file not found: {jsonl_path}")
        sys.exit(1)

    print(f"[validate_provenance] Validating provenance for: {jsonl_path}")

    results = validate_dataset_provenance(jsonl_path, strict=strict)

    print("\n=== Provenance Validation Results ===")
    print(f"Total samples: {results['total_samples']}")
    print(f"Valid samples: {results['valid_samples']}")
    print(f"Invalid samples: {results['invalid_samples']}")
    print(f"Total errors: {results['total_errors']}")

    if results["error_summary"]:
        print("\nError Details:")
        for error, count in results["error_summary"].items():
            print(f"  - {error}: {count} occurrences")

    # Save report if requested
    if output_path:
        report_path = Path(output_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[validate_provenance] Report saved to: {report_path}")

    if results["pass"]:
        print("\n[validate_provenance] ✅ Provenance validation PASSED")
        sys.exit(0)
    else:
        print("\n[validate_provenance] ❌ Provenance validation FAILED")
        sys.exit(1)


if __name__ == "__main__":
    app()
