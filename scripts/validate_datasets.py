#!/usr/bin/env python3
"""
Validate datasets against DATASET_STANDARDS_V1.md invariants.

This script performs programmatic validation of all invariants specified
in the dataset standards document, ensuring production datasets meet
quality gates before training.

Author: @darianrosebrook
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load samples from JSONL file."""
    samples = []
    if not file_path.exists():
        print(f"  ❌ ERROR: File not found: {file_path}")
        return samples

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith('{"__header__'):
                continue
            try:
                sample = json.loads(line)
                sample["_line_num"] = line_num
                samples.append(sample)
            except json.JSONDecodeError as e:
                print(f"  ❌ ERROR: Invalid JSON at line {line_num}: {e}")
                return []

    return samples


class ValidationResult:
    """Tracks validation results."""

    def __init__(self, dataset_name: str, role: str):
        self.dataset_name = dataset_name
        self.role = role
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.stats: Dict[str, Any] = {}

    def error(self, message: str, sample_idx: Optional[int] = None) -> None:
        """Record a validation error."""
        if sample_idx is not None:
            self.errors.append(f"Sample {sample_idx + 1}: {message}")
        else:
            self.errors.append(message)

    def warn(self, message: str, sample_idx: Optional[int] = None) -> None:
        """Record a validation warning."""
        if sample_idx is not None:
            self.warnings.append(f"Sample {sample_idx + 1}: {message}")
        else:
            self.warnings.append(message)

    def is_valid(self) -> bool:
        """Check if validation passed (no errors)."""
        return len(self.errors) == 0

    def print_summary(self) -> None:
        """Print validation summary."""
        print(f"\n{'='*80}")
        print(f"VALIDATION SUMMARY: {self.dataset_name} ({self.role})")
        print(f"{'='*80}")
        print(f"Errors: {len(self.errors)}")
        print(f"Warnings: {len(self.warnings)}")
        
        if self.stats:
            print("\nStatistics:")
            for key, value in self.stats.items():
                print(f"  {key}: {value}")
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors[:20]:  # Show first 20
                print(f"  - {error}")
            if len(self.errors) > 20:
                print(f"  ... and {len(self.errors) - 20} more errors")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings[:10]:  # Show first 10
                print(f"  - {warning}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more warnings")


def validate_common_fields(sample: Dict[str, Any], result: ValidationResult, idx: int) -> None:
    """Validate common required fields per DATASET_STANDARDS_V1.md section 1.2."""
    # Required top-level fields
    required_fields = ["id", "role", "metadata", "caws_context", "evidence_manifest", "provenance_chain"]
    
    for field in required_fields:
        if field not in sample:
            result.error(f"Missing required field: {field}", idx)
    
    # Validate metadata structure
    metadata = sample.get("metadata")
    if metadata is None:
        result.error("metadata is None (must be object)", idx)
    elif not isinstance(metadata, dict):
        result.error("metadata must be a dict", idx)
    else:
        required_metadata = ["task_type", "caws_level"]
        for field in required_metadata:
            if field not in metadata:
                result.warn(f"metadata.{field} missing (recommended)", idx)
    
    # Validate caws_context structure
    caws_context = sample.get("caws_context")
    if caws_context is None:
        result.warn("caws_context is None (allowed for Drafter)", idx)
    elif isinstance(caws_context, dict):
        working_spec = caws_context.get("working_spec")
        if working_spec is None:
            result.warn("caws_context.working_spec missing", idx)
        elif isinstance(working_spec, dict):
            required_ws_fields = ["id", "title", "risk_tier", "budget", "scope"]
            for field in required_ws_fields:
                if field not in working_spec:
                    result.error(f"caws_context.working_spec.{field} missing", idx)
            
            # Validate budget structure
            budget = working_spec.get("budget")
            if budget is not None and isinstance(budget, dict):
                if "max_files" not in budget or "max_loc" not in budget:
                    result.error("caws_context.working_spec.budget missing max_files or max_loc", idx)
            
            # Validate scope structure
            scope = working_spec.get("scope")
            if scope is not None and isinstance(scope, dict):
                if "in" not in scope or "out" not in scope:
                    result.error("caws_context.working_spec.scope missing 'in' or 'out'", idx)
    
    # Validate evidence_manifest structure
    evidence_manifest = sample.get("evidence_manifest")
    if evidence_manifest is None:
        result.error("evidence_manifest is None (must be object)", idx)
    elif isinstance(evidence_manifest, dict):
        required_em_fields = ["claims", "verification_status", "evidence_references"]
        for field in required_em_fields:
            if field not in evidence_manifest:
                result.error(f"evidence_manifest.{field} missing", idx)
    
    # Validate provenance_chain structure
    provenance_chain = sample.get("provenance_chain")
    if provenance_chain is None:
        result.error("provenance_chain is None (must be object)", idx)
    elif isinstance(provenance_chain, dict):
        if "steps" not in provenance_chain or "audit_trail" not in provenance_chain:
            result.error("provenance_chain missing 'steps' or 'audit_trail'", idx)


def validate_worker_dataset(samples: List[Dict[str, Any]], dataset_name: str) -> ValidationResult:
    """Validate Worker dataset per DATASET_STANDARDS_V1.md section 2."""
    result = ValidationResult(dataset_name, "worker")
    
    if not samples:
        result.error("No samples found")
        return result
    
    result.stats["total_samples"] = len(samples)
    
    task_types = Counter()
    caws_levels = Counter()
    has_process_supervision = 0
    has_tool_use = 0
    has_caws_context = 0
    teacher_text_empty = 0
    
    for idx, sample in enumerate(samples):
        # Common fields
        validate_common_fields(sample, result, idx)
        
        # Worker-specific required fields (section 2.1)
        required_fields = ["prompt", "teacher_text", "student_target_ids"]
        for field in required_fields:
            if field not in sample:
                result.error(f"Missing required Worker field: {field}", idx)
        
        # Validate teacher_text (section 2.2 invariant)
        teacher_text = sample.get("teacher_text", "")
        if not isinstance(teacher_text, str):
            result.error("teacher_text must be a string", idx)
        elif len(teacher_text.strip()) == 0:
            result.error("teacher_text is empty (violates invariant)", idx)
            teacher_text_empty += 1
        
        # Track task_type
        task_type = sample.get("task_type") or sample.get("metadata", {}).get("task_type")
        if task_type:
            task_types[task_type] += 1
            
            # Validate process supervision for tool_use (section 2.2 invariant)
            if task_type == "tool_use":
                has_tool_use += 1
                if "process_supervision" not in sample:
                    result.error("task_type='tool_use' but process_supervision missing", idx)
                elif isinstance(sample.get("process_supervision"), dict):
                    ps = sample["process_supervision"]
                    required_ps_fields = ["tool_name_ids", "gold_json_text_ids", "integration_mask"]
                    for field in required_ps_fields:
                        if field not in ps:
                            result.error(f"process_supervision.{field} missing for tool_use", idx)
                        elif not isinstance(ps[field], list):
                            result.error(f"process_supervision.{field} must be a list", idx)
                    if all(field in ps for field in required_ps_fields):
                        has_process_supervision += 1
                else:
                    result.error("process_supervision must be a dict for tool_use", idx)
            
            # Validate plain_kd doesn't have tool_calls (section 2.2 invariant)
            if task_type == "plain_kd":
                tool_calls = sample.get("tool_calls")
                if tool_calls and len(tool_calls) > 0:
                    result.warn("task_type='plain_kd' but tool_calls present", idx)
        
        # Validate CAWS level -> caws_context (section 2.2 invariant)
        caws_level = sample.get("caws_level") or sample.get("metadata", {}).get("caws_level")
        if caws_level:
            caws_levels[caws_level] += 1
            if caws_level != "none" and caws_level != 0:
                caws_ctx = sample.get("caws_context")
                if not isinstance(caws_ctx, dict) or not caws_ctx.get("working_spec"):
                    result.error(f"caws_level={caws_level} but caws_context.working_spec not populated", idx)
                else:
                    has_caws_context += 1
    
    # Distribution checks (section 2.3)
    result.stats["task_type_distribution"] = dict(task_types)
    result.stats["caws_level_distribution"] = dict(caws_levels)
    result.stats["tool_use_samples"] = has_tool_use
    result.stats["process_supervision_coverage"] = f"{has_process_supervision}/{has_tool_use}" if has_tool_use > 0 else "N/A"
    result.stats["caws_context_samples"] = has_caws_context
    result.stats["empty_teacher_text"] = teacher_text_empty
    
    # Check distribution targets
    total = len(samples)
    if total < 1500:
        result.warn(f"Sample count ({total}) below target (1500-2000+)")
    
    if task_types:
        plain_kd_pct = (task_types.get("plain_kd", 0) / total * 100) if total > 0 else 0
        tool_use_pct = (task_types.get("tool_use", 0) / total * 100) if total > 0 else 0
        
        if plain_kd_pct < 30 or plain_kd_pct > 40:
            result.warn(f"plain_kd distribution ({plain_kd_pct:.1f}%) outside target (30-40%)")
        if tool_use_pct < 30 or tool_use_pct > 40:
            result.warn(f"tool_use distribution ({tool_use_pct:.1f}%) outside target (30-40%)")
    
    return result


def validate_judge_dataset(samples: List[Dict[str, Any]], dataset_name: str) -> ValidationResult:
    """Validate Judge dataset per DATASET_STANDARDS_V1.md section 3."""
    result = ValidationResult(dataset_name, "judge")
    
    if not samples:
        result.error("No samples found")
        return result
    
    result.stats["total_samples"] = len(samples)
    
    winners = Counter()
    stages = Counter()
    has_debate_scores = 0
    invalid_debate_scores = 0
    
    for idx, sample in enumerate(samples):
        # Common fields (caws_context can be null for Judge)
        validate_common_fields(sample, result, idx)
        
        # Judge-specific required fields (section 3.1)
        required_fields = ["prompt", "working_spec", "a", "b", "winner", "adjudication_stage"]
        for field in required_fields:
            if field not in sample:
                result.error(f"Missing required Judge field: {field}", idx)
        
        # Validate winner (section 3.2 invariant)
        winner = sample.get("winner")
        if winner not in ("a", "b", "tie"):
            result.error(f"Invalid winner value: {winner!r} (must be 'a', 'b', or 'tie')", idx)
        else:
            winners[winner] += 1
        
        # Validate adjudication_stage (section 3.2 invariant)
        stage = sample.get("adjudication_stage")
        valid_stages = ["pleading", "examination", "deliberation", "verdict", "publication"]
        if stage not in valid_stages:
            result.error(f"Invalid adjudication_stage: {stage!r} (must be one of {valid_stages})", idx)
        else:
            stages[stage] += 1
        
        # Validate debate_scores structure (section 3.1)
        debate_scores = sample.get("debate_scores")
        if debate_scores is None:
            result.warn("debate_scores missing (optional in v1 but recommended)", idx)
        elif isinstance(debate_scores, dict):
            has_debate_scores += 1
            for side in ["a", "b"]:
                if side not in debate_scores:
                    result.error(f"debate_scores.{side} missing", idx)
                elif isinstance(debate_scores[side], dict):
                    side_scores = debate_scores[side]
                    required_scores = ["E", "B", "G", "P", "total"]
                    for score_key in required_scores:
                        if score_key not in side_scores:
                            result.error(f"debate_scores.{side}.{score_key} missing", idx)
                    
                    # Validate total calculation (section 3.2 invariant)
                    if all(k in side_scores for k in required_scores):
                        calculated_total = (
                            0.4 * side_scores["E"] +
                            0.3 * side_scores["B"] +
                            0.2 * side_scores["G"] +
                            0.1 * side_scores["P"]
                        )
                        actual_total = side_scores["total"]
                        if abs(calculated_total - actual_total) > 0.01:  # Tolerance
                            result.error(
                                f"debate_scores.{side}.total mismatch: "
                                f"calculated={calculated_total:.3f}, actual={actual_total:.3f}",
                                idx
                            )
                            invalid_debate_scores += 1
        
        # Validate sides a and b (section 3.1)
        for side in ["a", "b"]:
            side_data = sample.get(side)
            if side_data is None:
                result.error(f"Side {side} missing", idx)
            elif isinstance(side_data, dict):
                if "text" not in side_data:
                    result.error(f"{side}.text missing", idx)
                if "clauses" not in side_data:
                    result.warn(f"{side}.clauses missing (recommended)", idx)
                if "evidence_manifest" not in side_data:
                    result.warn(f"{side}.evidence_manifest missing (recommended)", idx)
    
    # Distribution checks (section 3.3)
    result.stats["winner_distribution"] = dict(winners)
    result.stats["adjudication_stage_distribution"] = dict(stages)
    result.stats["debate_scores_coverage"] = f"{has_debate_scores}/{len(samples)}"
    result.stats["invalid_debate_scores"] = invalid_debate_scores
    
    # Check distribution targets
    total = len(samples)
    if total < 5000:
        result.warn(f"Sample count ({total}) below target (5000-10000)")
    
    if winners:
        tie_pct = (winners.get("tie", 0) / total * 100) if total > 0 else 0
        if tie_pct >= 10:
            result.warn(f"Tie percentage ({tie_pct:.1f}%) above target (<10%)")
    
    return result


def validate_drafter_dataset(samples: List[Dict[str, Any]], dataset_name: str) -> ValidationResult:
    """Validate Drafter dataset per DATASET_STANDARDS_V1.md section 4."""
    result = ValidationResult(dataset_name, "drafter")
    
    if not samples:
        result.error("No samples found")
        return result
    
    result.stats["total_samples"] = len(samples)
    
    length_buckets = Counter()
    exceeds_length_limit = 0
    
    for idx, sample in enumerate(samples):
        # Drafter uses same base structure as Worker
        validate_common_fields(sample, result, idx)
        
        # Drafter-specific fields (section 4.1)
        if "draft_segment" not in sample:
            result.warn("draft_segment missing (recommended)", idx)
        if "draft_context_window" not in sample:
            result.warn("draft_context_window missing (recommended)", idx)
        
        # Length bucket (section 4.2)
        metadata = sample.get("metadata", {})
        length_bucket = metadata.get("length_bucket")
        if length_bucket:
            length_buckets[length_bucket] += 1
        
        # Check output length (section 4.2 constraint)
        teacher_text = sample.get("teacher_text", "")
        if teacher_text:
            # Rough token estimate: 4 chars per token
            estimated_tokens = len(teacher_text) // 4
            if estimated_tokens > 2000:
                result.error(f"Output exceeds 2k token limit (estimated {estimated_tokens} tokens)", idx)
                exceeds_length_limit += 1
    
    result.stats["length_bucket_distribution"] = dict(length_buckets)
    result.stats["exceeds_length_limit"] = exceeds_length_limit
    
    # Check distribution targets
    total = len(samples)
    if total < 1000:
        result.warn(f"Sample count ({total}) below target (1000-1500)")
    
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate datasets against DATASET_STANDARDS_V1.md invariants"
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to dataset JSONL file",
    )
    parser.add_argument(
        "--role",
        choices=["worker", "judge", "drafter"],
        required=True,
        help="Dataset role",
    )
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Exit with error code if warnings are present",
    )
    
    args = parser.parse_args()
    
    if not args.dataset.exists():
        print(f"❌ ERROR: Dataset file not found: {args.dataset}")
        return 1
    
    samples = load_jsonl(args.dataset)
    
    if args.role == "worker":
        result = validate_worker_dataset(samples, str(args.dataset))
    elif args.role == "judge":
        result = validate_judge_dataset(samples, str(args.dataset))
    elif args.role == "drafter":
        result = validate_drafter_dataset(samples, str(args.dataset))
    else:
        print(f"❌ ERROR: Unknown role: {args.role}")
        return 1
    
    result.print_summary()
    
    if not result.is_valid():
        print("\n❌ VALIDATION FAILED")
        return 1
    
    if args.fail_on_warnings and result.warnings:
        print("\n⚠️  VALIDATION PASSED WITH WARNINGS (--fail-on-warnings enabled)")
        return 1
    
    print("\n✅ VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())


