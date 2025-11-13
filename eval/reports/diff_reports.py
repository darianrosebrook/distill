"""Compare evaluation reports to detect regressions."""

from __future__ import annotations

import json
import sys
from typing import Any, Dict


def pick(report: Dict[str, Any], key: str) -> float | None:
    """Extract metric from report summary."""
    return report.get("summary", {}).get(key)


def diff_reports(report_a_path: str, report_b_path: str, threshold: float = 0.01) -> Dict[str, Any]:
    """
    Compare two evaluation reports and detect regressions.

    Args:
        report_a_path: Path to baseline (previous) report
        report_b_path: Path to current report
        threshold: Minimum delta to consider a regression (default: 0.01 = 1%)

    Returns:
        Dictionary with regression analysis
    """
    with open(report_a_path, "r", encoding="utf-8") as f:
        report_a = json.load(f)

    with open(report_b_path, "r", encoding="utf-8") as f:
        report_b = json.load(f)

    # Handle merged reports (multiple shards)
    if "reports" in report_a:
        # Use first shard for comparison (or aggregate if needed)
        report_a = report_a["reports"][0] if report_a["reports"] else report_a

    if "reports" in report_b:
        report_b = report_b["reports"][0] if report_b["reports"] else report_b

    fields = [
        "avg_integration_f1_macro_lax",
        "avg_integration_f1_macro_strict",
        "integration_f1_eligible_count",
        "controls_with_integration",
        "privacy_ok_rate",
        "integration_coverage",
    ]

    regressions = []
    improvements = []
    unchanged = []

    for key in fields:
        val_a = pick(report_a, key)
        val_b = pick(report_b, key)

        if val_a is None or val_b is None:
            continue

        delta = val_b - val_a

        # For fields where higher is better
        if key in [
            "avg_integration_f1_macro_lax",
            "avg_integration_f1_macro_strict",
            "integration_f1_eligible_count",
            "privacy_ok_rate",
            "integration_coverage",
        ]:
            if delta < -threshold:
                regressions.append(
                    {
                        "metric": key,
                        "baseline": val_a,
                        "current": val_b,
                        "delta": delta,
                        "delta_pct": (delta / val_a * 100) if val_a != 0 else 0,
                    }
                )
            elif delta > threshold:
                improvements.append(
                    {
                        "metric": key,
                        "baseline": val_a,
                        "current": val_b,
                        "delta": delta,
                        "delta_pct": (delta / val_a * 100) if val_a != 0 else 0,
                    }
                )
            else:
                unchanged.append(
                    {
                        "metric": key,
                        "value": val_a,
                    }
                )

        # For fields where lower is better (controls_with_integration)
        elif key == "controls_with_integration":
            if delta > threshold:
                regressions.append(
                    {
                        "metric": key,
                        "baseline": val_a,
                        "current": val_b,
                        "delta": delta,
                    }
                )
            elif delta < -threshold:
                improvements.append(
                    {
                        "metric": key,
                        "baseline": val_a,
                        "current": val_b,
                        "delta": delta,
                    }
                )
            else:
                unchanged.append(
                    {
                        "metric": key,
                        "value": val_a,
                    }
                )

    result = {
        "baseline": report_a_path,
        "current": report_b_path,
        "regressions": regressions,
        "improvements": improvements,
        "unchanged": unchanged,
        "gates_ok_baseline": report_a.get("gates_ok", False),
        "gates_ok_current": report_b.get("gates_ok", False),
    }

    return result


def main():
    """CLI entry point."""
    if len(sys.argv) < 3:
        print("Usage: python -m eval.reports.diff_reports <baseline.json> <current.json>")
        sys.exit(1)

    baseline_path = sys.argv[1]
    current_path = sys.argv[2]
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01

    result = diff_reports(baseline_path, current_path, threshold)

    print(json.dumps(result, indent=2))

    # Exit with error code if regressions found
    if result["regressions"]:
        print(f"\n❌ Found {len(result['regressions'])} regressions:", file=sys.stderr)
        for r in result["regressions"]:
            print(
                f"  - {r['metric']}: {r['baseline']:.4f} → {r['current']:.4f} (Δ {r['delta']:.4f})",
                file=sys.stderr,
            )
        sys.exit(2)

    if not result["gates_ok_current"]:
        print("\n❌ Current report gates failed", file=sys.stderr)
        sys.exit(1)

    print("\n✅ No regressions detected")
    sys.exit(0)


if __name__ == "__main__":
    main()
