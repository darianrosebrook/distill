"""
Comprehensive dataset audit script for Worker, Judge, and Drafter datasets.

Checks:
- Schema compliance
- Required fields
- Data quality
- Distribution balance
- Potential poisoning issues
- Format consistency

Author: @darianrosebrook
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Set, Optional
from collections import Counter, defaultdict
import sys


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load samples from JSONL file."""
    samples = []
    if not file_path.exists():
        return samples

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                # Skip header/metadata lines
                if sample.get("__header__"):
                    continue
                sample["_line_num"] = line_num
                samples.append(sample)
            except json.JSONDecodeError as e:
                print(f"  ❌ ERROR: Invalid JSON at line {line_num}: {e}")
                return []

    return samples


def audit_worker_dataset(samples: List[Dict[str, Any]], dataset_name: str) -> Dict[str, Any]:
    """Audit Worker dataset."""
    print(f"\n{'='*80}")
    print(f"AUDITING WORKER DATASET: {dataset_name}")
    print(f"{'='*80}")

    issues = []
    warnings = []

    # Required fields
    required_fields = ["id", "prompt", "role"]
    optional_fields = ["task_type", "caws_level",
                       "source", "teacher_text", "metadata"]

    # Check each sample
    field_counts = Counter()
    task_types = Counter()
    caws_levels = Counter()
    sources = Counter()
    roles = Counter()

    for i, sample in enumerate(samples):
        # Check required fields
        for field in required_fields:
            if field not in sample:
                issues.append(
                    f"Sample {i+1} (line {sample.get('_line_num', '?')}): Missing required field '{field}'")

        # Track field presence
        for field in required_fields + optional_fields:
            if field in sample:
                field_counts[field] += 1

        # Track distributions
        if "task_type" in sample:
            task_types[sample["task_type"]] += 1
        if "caws_level" in sample:
            caws_levels[sample["caws_level"]] += 1
        if "source" in sample:
            sources[sample["source"]] += 1
        if "role" in sample:
            roles[sample["role"]] += 1

        # Check prompt quality
        prompt = sample.get("prompt", "")
        if len(prompt) < 10:
            warnings.append(
                f"Sample {i+1}: Very short prompt ({len(prompt)} chars)")
        if len(prompt) > 50000:
            warnings.append(
                f"Sample {i+1}: Very long prompt ({len(prompt)} chars)")

        # Check teacher_text quality
        teacher_text = sample.get("teacher_text", "")
        if teacher_text and len(teacher_text) < 10:
            warnings.append(
                f"Sample {i+1}: Very short teacher_text ({len(teacher_text)} chars)")

        # Check CAWS level consistency
        caws_level = sample.get("caws_level", 0)
        has_caws_context = "caws_context" in sample or "working_spec" in sample
        has_evidence = "evidence_manifest" in sample
        has_provenance = "provenance_chain" in sample

        if caws_level == 0 and (has_caws_context or has_evidence or has_provenance):
            warnings.append(f"Sample {i+1}: caws_level=0 but has CAWS fields")
        if caws_level == 1 and not has_caws_context:
            warnings.append(
                f"Sample {i+1}: caws_level=1 but missing caws_context")
        if caws_level == 2 and not (has_evidence and has_provenance):
            warnings.append(
                f"Sample {i+1}: caws_level=2 but missing evidence/provenance")

        # Check for placeholder/mock data (but allow legitimate TODO mentions)
        prompt_lower = prompt.lower()
        teacher_text_lower = teacher_text.lower()
        if "placeholder" in prompt_lower or "placeholder" in teacher_text_lower:
            issues.append(f"Sample {i+1}: Contains 'placeholder' text")
        if "mock" in prompt_lower or "mock" in teacher_text_lower:
            warnings.append(f"Sample {i+1}: Contains 'mock' text")
        # Only flag TODO if it looks like placeholder code, not legitimate task descriptions
        if ("// todo" in prompt_lower or "// todo" in teacher_text_lower or
            "# todo" in prompt_lower or "# todo" in teacher_text_lower or
                "todo:" in prompt_lower or "todo:" in teacher_text_lower):
            warnings.append(f"Sample {i+1}: Contains 'TODO' placeholder code")

    # Report
    print(f"\nTotal samples: {len(samples)}")

    print(f"\n✅ Field Coverage:")
    for field in required_fields + optional_fields:
        count = field_counts.get(field, 0)
        pct = (count / len(samples) * 100) if samples else 0
        status = "✅" if field in required_fields and count == len(
            samples) else "⚠️" if count < len(samples) else "✅"
        print(f"  {status} {field}: {count}/{len(samples)} ({pct:.1f}%)")

    if task_types:
        print(f"\n📊 Task Type Distribution:")
        for task_type, count in task_types.most_common():
            pct = (count / len(samples) * 100) if samples else 0
            print(f"  {task_type}: {count} ({pct:.1f}%)")

    if caws_levels:
        print(f"\n📊 CAWS Level Distribution:")
        for level, count in sorted(caws_levels.items()):
            pct = (count / len(samples) * 100) if samples else 0
            print(f"  Level {level}: {count} ({pct:.1f}%)")

    if sources:
        print(f"\n📊 Source Distribution:")
        for source, count in sources.most_common():
            pct = (count / len(samples) * 100) if samples else 0
            print(f"  {source}: {count} ({pct:.1f}%)")

    if roles:
        print(f"\n📊 Role Distribution:")
        for role, count in roles.most_common():
            pct = (count / len(samples) * 100) if samples else 0
            print(f"  {role}: {count} ({pct:.1f}%)")

    # Report issues
    if issues:
        print(f"\n❌ CRITICAL ISSUES ({len(issues)}):")
        for issue in issues[:10]:  # Show first 10
            print(f"  {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")

    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for warning in warnings[:10]:  # Show first 10
            print(f"  {warning}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more")

    if not issues and not warnings:
        print(f"\n✅ No issues found!")

    return {
        "total": len(samples),
        "issues": len(issues),
        "warnings": len(warnings),
        "task_types": dict(task_types),
        "caws_levels": dict(caws_levels),
        "sources": dict(sources),
    }


def audit_judge_dataset(samples: List[Dict[str, Any]], dataset_name: str) -> Dict[str, Any]:
    """Audit Judge dataset."""
    print(f"\n{'='*80}")
    print(f"AUDITING JUDGE DATASET: {dataset_name}")
    print(f"{'='*80}")

    issues = []
    warnings = []

    # Required fields
    required_fields = ["id", "prompt", "a", "b", "winner"]

    # Check each sample
    winners = Counter()
    clause_counts = Counter()
    a_clauses = Counter()
    b_clauses = Counter()

    for i, sample in enumerate(samples):
        # Check required fields
        for field in required_fields:
            if field not in sample:
                issues.append(
                    f"Sample {i+1} (line {sample.get('_line_num', '?')}): Missing required field '{field}'")

        # Check a and b structure
        if "a" not in sample:
            issues.append(f"Sample {i+1}: Missing 'a' field")
        else:
            if "text" not in sample["a"]:
                issues.append(f"Sample {i+1}: Missing 'a.text' field")
            if "clauses" not in sample["a"]:
                warnings.append(
                    f"Sample {i+1}: Missing 'a.clauses' field (will default to empty)")
            else:
                for clause in sample["a"]["clauses"]:
                    a_clauses[clause] += 1
                    clause_counts[clause] += 1

        if "b" not in sample:
            issues.append(f"Sample {i+1}: Missing 'b' field")
        else:
            if "text" not in sample["b"]:
                issues.append(f"Sample {i+1}: Missing 'b.text' field")
            if "clauses" not in sample["b"]:
                warnings.append(
                    f"Sample {i+1}: Missing 'b.clauses' field (will default to empty)")
            else:
                for clause in sample["b"]["clauses"]:
                    b_clauses[clause] += 1
                    clause_counts[clause] += 1

        # Check winner
        winner = sample.get("winner", "")
        if winner not in ["a", "b", "tie"]:
            issues.append(
                f"Sample {i+1}: Invalid winner '{winner}' (must be 'a', 'b', or 'tie')")
        else:
            winners[winner] += 1

        # Check for valid CAWS clauses
        valid_clauses = {
            "EVIDENCE_COMPLETENESS",
            "BUDGET_ADHERENCE",
            "GATE_INTEGRITY",
            "PROVENANCE_CLARITY",
            "WAIVER_JUSTIFICATION",
        }

        all_clauses = sample.get("a", {}).get(
            "clauses", []) + sample.get("b", {}).get("clauses", [])
        for clause in all_clauses:
            if clause not in valid_clauses:
                issues.append(
                    f"Sample {i+1}: Invalid clause '{clause}' (not in valid CAWS clauses)")

        # Check prompt quality
        prompt = sample.get("prompt", "")
        if len(prompt) < 10:
            warnings.append(
                f"Sample {i+1}: Very short prompt ({len(prompt)} chars)")

        # Check text quality
        a_text = sample.get("a", {}).get("text", "")
        b_text = sample.get("b", {}).get("text", "")
        if len(a_text) < 10:
            warnings.append(
                f"Sample {i+1}: Very short a.text ({len(a_text)} chars)")
        if len(b_text) < 10:
            warnings.append(
                f"Sample {i+1}: Very short b.text ({len(b_text)} chars)")

        # Check for placeholder/mock data (but allow legitimate TODO mentions)
        prompt_lower = prompt.lower()
        a_text_lower = a_text.lower()
        b_text_lower = b_text.lower()
        if "placeholder" in prompt_lower or "placeholder" in a_text_lower or "placeholder" in b_text_lower:
            issues.append(f"Sample {i+1}: Contains 'placeholder' text")
        if "mock" in prompt_lower or "mock" in a_text_lower or "mock" in b_text_lower:
            warnings.append(f"Sample {i+1}: Contains 'mock' text")
        # Only flag TODO if it looks like placeholder code
        if ("// todo" in prompt_lower or "// todo" in a_text_lower or "// todo" in b_text_lower or
            "# todo" in prompt_lower or "# todo" in a_text_lower or "# todo" in b_text_lower or
                "todo:" in prompt_lower or "todo:" in a_text_lower or "todo:" in b_text_lower):
            warnings.append(f"Sample {i+1}: Contains 'TODO' placeholder code")

    # Report
    print(f"\nTotal samples: {len(samples)}")

    print(f"\n📊 Winner Distribution:")
    for winner, count in winners.most_common():
        pct = (count / len(samples) * 100) if samples else 0
        print(f"  {winner}: {count} ({pct:.1f}%)")

    if clause_counts:
        print(f"\n📊 Clause Distribution (all):")
        for clause, count in clause_counts.most_common():
            pct = (count / (len(samples) * 2) * 100) if samples else 0
            print(f"  {clause}: {count} ({pct:.1f}%)")

    if a_clauses:
        print(f"\n📊 Clause Distribution (a only):")
        for clause, count in a_clauses.most_common():
            pct = (count / len(samples) * 100) if samples else 0
            print(f"  {clause}: {count} ({pct:.1f}%)")

    if b_clauses:
        print(f"\n📊 Clause Distribution (b only):")
        for clause, count in b_clauses.most_common():
            pct = (count / len(samples) * 100) if samples else 0
            print(f"  {clause}: {count} ({pct:.1f}%)")

    # Check balance
    if winners:
        max_winner = max(winners.values())
        min_winner = min(winners.values())
        imbalance = (max_winner - min_winner) / \
            len(samples) * 100 if samples else 0
        if imbalance > 50:
            warnings.append(
                f"Significant winner imbalance: {imbalance:.1f}% difference")

    # Report issues
    if issues:
        print(f"\n❌ CRITICAL ISSUES ({len(issues)}):")
        for issue in issues[:10]:  # Show first 10
            print(f"  {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")

    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for warning in warnings[:10]:  # Show first 10
            print(f"  {warning}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more")

    if not issues and not warnings:
        print(f"\n✅ No issues found!")

    return {
        "total": len(samples),
        "issues": len(issues),
        "warnings": len(warnings),
        "winners": dict(winners),
        "clauses": dict(clause_counts),
    }


def audit_drafter_dataset(samples: List[Dict[str, Any]], dataset_name: str) -> Dict[str, Any]:
    """Audit Drafter dataset."""
    print(f"\n{'='*80}")
    print(f"AUDITING DRAFTER DATASET: {dataset_name}")
    print(f"{'='*80}")

    issues = []
    warnings = []

    # Check each sample
    lengths = []
    task_types = Counter()

    for i, sample in enumerate(samples):
        # Check required fields (same as Worker)
        if "prompt" not in sample:
            issues.append(f"Sample {i+1}: Missing 'prompt' field")
        if "teacher_text" not in sample and "text" not in sample:
            warnings.append(
                f"Sample {i+1}: Missing 'teacher_text' or 'text' field")

        # Check length (Drafter should have short responses)
        text = sample.get("teacher_text", sample.get("text", ""))
        length = len(text)
        lengths.append(length)

        if length > 2000:
            warnings.append(
                f"Sample {i+1}: Very long response ({length} chars, Drafter should be <2000)")

        if "task_type" in sample:
            task_types[sample["task_type"]] += 1

    # Report
    print(f"\nTotal samples: {len(samples)}")

    if lengths:
        avg_length = sum(lengths) / len(lengths)
        max_length = max(lengths)
        min_length = min(lengths)
        print(f"\n📊 Length Statistics:")
        print(f"  Average: {avg_length:.1f} chars")
        print(f"  Min: {min_length} chars")
        print(f"  Max: {max_length} chars")

        # Length buckets
        buckets = {"<200": 0, "200-400": 0,
                   "400-800": 0, "800-2000": 0, ">2000": 0}
        for length in lengths:
            if length < 200:
                buckets["<200"] += 1
            elif length < 400:
                buckets["200-400"] += 1
            elif length < 800:
                buckets["400-800"] += 1
            elif length < 2000:
                buckets["800-2000"] += 1
            else:
                buckets[">2000"] += 1

        print(f"\n📊 Length Distribution:")
        for bucket, count in buckets.items():
            pct = (count / len(samples) * 100) if samples else 0
            print(f"  {bucket}: {count} ({pct:.1f}%)")

    if task_types:
        print(f"\n📊 Task Type Distribution:")
        for task_type, count in task_types.most_common():
            pct = (count / len(samples) * 100) if samples else 0
            print(f"  {task_type}: {count} ({pct:.1f}%)")

    # Report issues
    if issues:
        print(f"\n❌ CRITICAL ISSUES ({len(issues)}):")
        for issue in issues[:10]:
            print(f"  {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")

    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for warning in warnings[:10]:
            print(f"  {warning}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more")

    if not issues and not warnings:
        print(f"\n✅ No issues found!")

    return {
        "total": len(samples),
        "issues": len(issues),
        "warnings": len(warnings),
        "avg_length": sum(lengths) / len(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Audit generated datasets for quality and compliance",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument(
        "--worker",
        nargs="+",
        help="Worker dataset files to audit",
    )
    ap.add_argument(
        "--judge",
        nargs="+",
        help="Judge dataset files to audit",
    )
    ap.add_argument(
        "--drafter",
        nargs="+",
        help="Drafter dataset files to audit",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="Audit all datasets in standard locations",
    )
    args = ap.parse_args()

    results = {}

    if args.all:
        # Audit all standard datasets
        worker_files = [
            "data/worker_combined.jsonl",
            "data/caws_tool_examples.jsonl",
            "data/contextual_prompts_new.jsonl",
        ]
        judge_files = [
            "data/judge/train.jsonl",
            "data/judge/val.jsonl",
            "data/judge/caws_scenarios.jsonl",
            "data/judge/adjudication_cycle.jsonl",
            "data/judge/worker_pairs.jsonl",
        ]
        drafter_files = [
            "data/drafter/drafter_dataset.jsonl",
        ]
    else:
        worker_files = args.worker or []
        judge_files = args.judge or []
        drafter_files = args.drafter or []

    # Audit Worker datasets
    for worker_file in worker_files:
        file_path = Path(worker_file)
        if not file_path.exists():
            print(f"⚠️  File not found: {worker_file}")
            continue

        samples = load_jsonl(file_path)
        if samples:
            result = audit_worker_dataset(samples, worker_file)
            results[worker_file] = result
        else:
            print(f"⚠️  No samples loaded from {worker_file}")

    # Audit Judge datasets
    for judge_file in judge_files:
        file_path = Path(judge_file)
        if not file_path.exists():
            print(f"⚠️  File not found: {judge_file}")
            continue

        samples = load_jsonl(file_path)
        if samples:
            result = audit_judge_dataset(samples, judge_file)
            results[judge_file] = result
        else:
            print(f"⚠️  No samples loaded from {judge_file}")

    # Audit Drafter datasets
    for drafter_file in drafter_files:
        file_path = Path(drafter_file)
        if not file_path.exists():
            print(f"⚠️  File not found: {drafter_file}")
            continue

        samples = load_jsonl(file_path)
        if samples:
            result = audit_drafter_dataset(samples, drafter_file)
            results[drafter_file] = result
        else:
            print(f"⚠️  No samples loaded from {drafter_file}")

    # Summary
    print(f"\n{'='*80}")
    print("AUDIT SUMMARY")
    print(f"{'='*80}")

    total_issues = sum(r.get("issues", 0) for r in results.values())
    total_warnings = sum(r.get("warnings", 0) for r in results.values())
    total_samples = sum(r.get("total", 0) for r in results.values())

    print(f"\nTotal datasets audited: {len(results)}")
    print(f"Total samples: {total_samples}")
    print(f"Total critical issues: {total_issues}")
    print(f"Total warnings: {total_warnings}")

    if total_issues == 0 and total_warnings == 0:
        print(f"\n✅ All datasets pass audit!")
        sys.exit(0)
    elif total_issues == 0:
        print(f"\n⚠️  Datasets have warnings but no critical issues")
        sys.exit(0)
    else:
        print(f"\n❌ Datasets have critical issues that must be fixed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
