#!/usr/bin/env python3
"""
Inspect Run Script

Loads a run manifest and prints compact summary:
- Pass/fail on Phase 0/1/2 gates
- Key metrics with thresholds
- Direct links to failure logs

Usage:
    python scripts/inspect_run.py <manifest_path>
    python scripts/inspect_run.py <manifest_path> --json
    python scripts/inspect_run.py <manifest_path> --phase phase0
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.run_manifest import RunManifest, GateStatus


def print_summary(
    manifest: RunManifest, phase_filter: Optional[str] = None, json_output: bool = False
):
    """Print compact summary of run manifest."""
    if json_output:
        import json

        print(json.dumps(manifest.to_dict(), indent=2))
        return

    print("=" * 60)
    print(f"Run Manifest: {manifest.run_id}")
    print("=" * 60)
    print(f"Created: {manifest.created_at}")
    print(f"Schema Version: {manifest.schema_version}")
    print()

    # Config & Data
    print("Config & Data:")
    print(f"  Config Fingerprint: {manifest.config_fingerprint[:16]}...")
    print(f"  Dataset Fingerprints: {len(manifest.dataset_fingerprints)} dataset(s)")
    for i, fp in enumerate(manifest.dataset_fingerprints):
        print(f"    [{i + 1}] {fp[:16]}...")
    print(f"  Code Commit: {manifest.code_commit_sha[:8] if manifest.code_commit_sha else 'N/A'}")
    print()

    # Environment
    if manifest.environment_versions:
        print("Environment:")
        for key, value in manifest.environment_versions.items():
            print(f"  {key}: {value}")
        print()

    # Phase Gates
    print("Phase Gates:")
    phases_to_show = manifest.phase_gates
    if phase_filter:
        phases_to_show = [pg for pg in phases_to_show if pg.phase == phase_filter]

    if not phases_to_show:
        print("  No phase gates recorded")
    else:
        for pg in phases_to_show:
            status_icon = (
                "✅"
                if pg.status == GateStatus.PASS
                else "❌"
                if pg.status == GateStatus.FAIL
                else "⏳"
            )
            print(f"  {status_icon} {pg.phase.upper()}: {pg.status.value.upper()}")

            if pg.gates:
                for gate_name, gate_status in pg.gates.items():
                    gate_icon = (
                        "✅"
                        if gate_status == GateStatus.PASS
                        else "❌"
                        if gate_status == GateStatus.FAIL
                        else "⏳"
                    )
                    print(f"      {gate_icon} {gate_name}: {gate_status.value}")

            if pg.notes:
                print(f"      Notes: {pg.notes}")
    print()

    # Key Metrics
    if manifest.key_metrics:
        print("Key Metrics:")
        for metric in manifest.key_metrics:
            status_icon = "✅" if metric.passed else "❌"
            print(
                f"  {status_icon} {metric.name}: {metric.value} {metric.unit} (threshold: {metric.threshold})"
            )
        print()

    # Artifacts
    print("Artifacts:")
    if manifest.training_logs_path:
        print(f"  Training Logs: {manifest.training_logs_path}")
    if manifest.evaluation_results_path:
        print(f"  Evaluation Results: {manifest.evaluation_results_path}")
    if manifest.export_artifacts_path:
        print(f"  Export Artifacts: {manifest.export_artifacts_path}")
    if manifest.coreml_benchmarks_path:
        print(f"  CoreML Benchmarks: {manifest.coreml_benchmarks_path}")
    if manifest.checkpoint_paths:
        print(f"  Checkpoints: {len(manifest.checkpoint_paths)} checkpoint(s)")
        for i, cp_path in enumerate(manifest.checkpoint_paths[:5]):  # Show first 5
            print(f"    [{i + 1}] {cp_path}")
        if len(manifest.checkpoint_paths) > 5:
            print(f"    ... and {len(manifest.checkpoint_paths) - 5} more")
    print()

    # Overall Status
    if manifest.all_phases_passed():
        print("✅ All phases passed")
    else:
        failed_gates = manifest.get_failed_gates()
        if failed_gates:
            print("❌ Failed Gates:")
            for gate in failed_gates:
                print(f"  - {gate}")
        else:
            print("⏳ Some gates pending")


def main():
    parser = argparse.ArgumentParser(description="Inspect run manifest")
    parser.add_argument("manifest_path", type=Path, help="Path to run manifest (JSON or YAML)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--phase", help="Filter by phase (e.g., phase0, phase1, phase2)")
    args = parser.parse_args()

    if not args.manifest_path.exists():
        print(f"Error: Manifest file not found: {args.manifest_path}", file=sys.stderr)
        sys.exit(1)

    # Load manifest
    try:
        if args.manifest_path.suffix == ".yaml" or args.manifest_path.suffix == ".yml":
            manifest = RunManifest.load_yaml(args.manifest_path)
        else:
            manifest = RunManifest.load_json(args.manifest_path)
    except Exception as e:
        print(f"Error: Failed to load manifest: {e}", file=sys.stderr)
        sys.exit(1)

    # Print summary
    print_summary(manifest, phase_filter=args.phase, json_output=args.json)


if __name__ == "__main__":
    main()
