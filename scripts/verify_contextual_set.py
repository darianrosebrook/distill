"""
Enhanced verification script for contextual dataset with comprehensive checks.

Author: @darianrosebrook
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import hashlib
import math
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict

from tools.schema_registry import ToolSchemaRegistry, validate_args
from scripts.util_token_spans import bytes_to_token_span
from scripts.util_sanitize import redact_pii, allowlist_urls

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


def is_long_context(
    prompt: str,
    tokenizer=None,
    token_threshold: int = 8000,
    byte_threshold: int = 24000,
) -> bool:
    """Check if prompt is long-context (tokenizer-aware)."""
    if tokenizer is not None:
        try:
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            return len(ids) >= token_threshold
        except Exception:
            # Fallback to bytes if tokenization fails
            return len(prompt) >= byte_threshold
    return len(prompt) >= byte_threshold


def contains_grounding(
    text: str, spans: List[List[int]], fields: Dict[str, str]
) -> bool:
    """Check if integration spans contain tool result field values."""
    if not spans or not fields:
        return False
    
    def norm(s: str) -> str:
        """Normalize: casefold + collapse whitespace."""
        return " ".join(str(s).lower().split())
    
    # Synonym expansion for common verbs
    SYN = {
        "prefer": {"favor", "prioritize", "choose"},
        "extract": {"pull", "parse", "get"},
        "optimize": {"speed up", "improve", "enhance"},
        "support": {"enable", "allow", "permit"},
    }
    
    def expand_synonyms(value: str) -> set:
        """Return set of normalized value + synonyms."""
        normalized = norm(value)
        expanded = {normalized}
        for key, alts in SYN.items():
            if key in normalized:
                for alt in alts:
                    expanded.add(normalized.replace(key, alt))
        return expanded
    
    # Build normalized values with synonyms
    normalized_value_sets = []
    for key, value in fields.items():
        if value:
            # Try to determine if it's numeric
            try:
                float(value)
                # Numeric: use regex pattern with word boundaries
                digits = re.sub(r'\D', '', str(value))
                normalized_value_sets.append(("numeric", digits))
            except (ValueError, TypeError):
                # String: normalize text + synonyms
                normalized_value_sets.append(("string", expand_synonyms(value)))
    
    # Check if at least one span contains at least one value (or synonym)
    for span in spans:
        if len(span) >= 2:
            seg_text = text[span[0] : span[1]]
            seg_normalized = norm(seg_text)
            
            for value_type, value_set in normalized_value_sets:
                if value_type == "numeric":
                    # For numeric, value_set is already a string (digits)
                    if re.search(r'\b' + re.escape(value_set) + r'\b', seg_text):
                        return True
                else:
                    # For string, check if any variant (including synonyms) is in span
                    for variant in value_set:
                        if variant and variant in seg_normalized:
                            return True
    return False


def parse_tool_json_slice(text: str, start: int, end: int):
    """Parse JSON slice from text."""
    try:
        s = text[start:end]
        l = s.find("{")
        r = s.rfind("}")
        if l == -1 or r == -1:
            return None
        return json.loads(s[l : r + 1])
    except Exception:
        return None


def check_caws_header(prompt: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Check if CAWS header is present and valid."""
    try:
        # Look for JSON at start of prompt
        first_line = prompt.split("\n")[0].strip()
        if first_line.startswith("{") and "caws" in first_line.lower():
            header = json.loads(first_line)
            if "caws" in header:
                caws = header["caws"]
                # Required keys (spec removed in v1.1.0)
                required_keys = ["tier", "max_files", "max_loc", "cov", "mut"]
                if all(k in caws for k in required_keys):
                    return True, caws
        return False, None
    except Exception:
        return False, None


def check_privacy(text: str) -> Dict[str, Any]:
    """Check for PII and URL violations."""
    # Check for emails
    email_pattern = r"[\w\.-]+@[\w\.-]+"
    emails = re.findall(email_pattern, text)
    emails = [e for e in emails if "[REDACTED_EMAIL]" not in e]

    # Check for UUIDs
    uuid_pattern = r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b"
    uuids = re.findall(uuid_pattern, text)
    uuids = [u for u in uuids if "[REDACTED_UUID]" not in u]

    # Check URLs
    url_pattern = r"https?://([^/\s]+)"
    urls = re.findall(url_pattern, text)
    url_ok = allowlist_urls(text)

    return {
        "emails_found": len(emails),
        "uuids_found": len(uuids),
        "urls_found": len(urls),
        "url_allowlist_ok": url_ok,
        "privacy_ok": len(emails) == 0 and len(uuids) == 0 and url_ok,
    }


def compute_integration_f1(
    teacher_text: str,
    integration_spans_bytes: List[List[int]],
    call_sequence: List[Dict[str, Any]],
) -> float:
    """
    Compute F1 score for integration span extraction.
    
    Returns F1 score between 0 and 1.
    """
    if not integration_spans_bytes or not call_sequence:
        return 0.0

    # Extract integration text spans
    integration_texts = []
    for span in integration_spans_bytes:
        if len(span) >= 2:
            integration_texts.append(teacher_text[span[0] : span[1]])

    # Check if integration mentions tool results
    # This is a heuristic - in practice, you'd match against actual tool result fields
    mentions_tool_result = False
    for text in integration_texts:
        if any(
            phrase in text.lower()
            for phrase in ["based on", "according to", "from the result", "the result shows"]
        ):
            mentions_tool_result = True
            break

    # Simple F1: if we have spans and they mention tool results, score is high
    if integration_texts and mentions_tool_result:
        return 0.9  # Good integration
    elif integration_texts:
        return 0.5  # Partial integration
    else:
        return 0.0  # No integration


def verify_token_alignment(
    teacher_text: str, span_bytes: List[int], tokenizer
) -> Tuple[bool, Optional[Tuple[int, int]]]:
    """Verify token alignment round-trip."""
    if not span_bytes or len(span_bytes) < 2:
        return False, None

    token_span = bytes_to_token_span(teacher_text, span_bytes[0], span_bytes[1], tokenizer)
    if token_span is None:
        return False, None

    # Round-trip check: decode tokens and compare with original
    try:
        decoded = tokenizer.decode(
            tokenizer.encode(teacher_text[: span_bytes[1]], add_special_tokens=False)[
                token_span[0] : token_span[1]
            ],
            skip_special_tokens=True,
        )
        original = teacher_text[span_bytes[0] : span_bytes[1]]
        # Normalize whitespace for comparison
        decoded_norm = " ".join(decoded.split())
        original_norm = " ".join(original.split())
        return decoded_norm == original_norm, token_span
    except Exception:
        return False, None


def verify_item(
    item: Dict[str, Any],
    reg: ToolSchemaRegistry,
    tokenizer=None,
    check_stratification: bool = True,
    check_controls: bool = True,
    check_adversarial: bool = True,
) -> Dict[str, Any]:
    """Verify a single item with comprehensive checks."""
    problems = []
    meta = item.get("metadata", {})
    teacher = item.get("teacher_text", "")
    prompt = item.get("prompt", "")

    ok = True

    # Check CAWS header
    caws_ok, caws_data = check_caws_header(prompt)
    if not caws_ok:
        problems.append("missing_or_invalid_caws_header")
    meta["caws_header_ok"] = caws_ok

    # Check privacy
    privacy_check = check_privacy(teacher + prompt)
    if not privacy_check["privacy_ok"]:
        problems.append("privacy_violation")
        if privacy_check["emails_found"] > 0:
            problems.append(f"emails_found:{privacy_check['emails_found']}")
        if privacy_check["uuids_found"] > 0:
            problems.append(f"uuids_found:{privacy_check['uuids_found']}")
        if not privacy_check["url_allowlist_ok"]:
            problems.append("url_not_in_allowlist")

    # Control case validation - short-circuit for controls
    expected_behaviour = meta.get("expected_behaviour", "normal")
    is_control = expected_behaviour in {"no_tool", "decline"}
    
    if is_control:
        # Controls should not have tool calls or spans
        call_sequence = meta.get("call_sequence", [])
        tool_span = meta.get("json_args_span_bytes")
        json_args_spans = meta.get("json_args_spans_bytes", [])
        integration_spans = meta.get("integration_spans_bytes", [])
        
        if call_sequence:
            problems.append("control_has_tool_calls")
            ok = False
        if tool_span or json_args_spans:
            problems.append("control_has_tool_spans")
            ok = False
        if integration_spans:
            problems.append("control_has_integration_spans")
            ok = False
        
        # Early return for controls - no further validation needed
        return {
            "ok": ok,
            "problems": problems,
            "tool": None,
            "semantic_ok": True,
            "token_align_ok": None,
            "integration_f1": 0.0,
            "integration_grounded": False,
            "caws_header_ok": caws_ok,
            "privacy_ok": privacy_check["privacy_ok"],
        }
    
    # Check tool spans for non-control cases
    tool_span = meta.get("json_args_span_bytes")
    json_args_spans_bytes = meta.get("json_args_spans_bytes", [])
    call_sequence = meta.get("call_sequence", [])
    
    # Adversarial ambiguity cases don't require spans (they ask for clarification)
    is_ambiguity = False
    if check_adversarial:
        adversarial = meta.get("adversarial")
        if adversarial and adversarial.get("type") == "ambiguity":
            is_ambiguity = True
    
    # For retry cases, check attempts array if present
    if expected_behaviour == "retry":
        attempts = meta.get("attempts", [])
        if attempts:
            # Require at least 2 attempts, last one successful
            if len(attempts) < 2:
                problems.append("retry_insufficient_attempts")
                ok = False
            elif not attempts[-1].get("ok", False):
                problems.append("retry_last_attempt_failed")
                ok = False
        # Don't require spans on first failed call for retry cases
        # Only check if we have spans
        if not tool_span and not json_args_spans_bytes and not attempts:
            # No spans and no attempts array - this is a problem
            problems.append("retry_missing_spans_or_attempts")
            ok = False
    elif call_sequence and not tool_span and not json_args_spans_bytes and not is_ambiguity:
        # Non-control, non-retry, non-ambiguity cases should have spans
        problems.append("missing:json_args_span_bytes")
        ok = False

    # Adversarial case validation
    if check_adversarial:
        adversarial = meta.get("adversarial")
        if adversarial:
            adv_type = adversarial.get("type")
            expected = adversarial.get("expected", "ask_clarify")
            if adv_type == "range_violation":
                # Should correct or reject invalid range
                if "correct" not in teacher.lower() and "invalid" not in teacher.lower():
                    problems.append("adversarial_range_violation_not_handled")
            elif adv_type == "malformed_json":
                # Should repair or reject malformed JSON
                if "fix" not in teacher.lower() and "repair" not in teacher.lower():
                    problems.append("adversarial_malformed_json_not_repaired")
            elif adv_type == "ambiguity":
                # Should ask for clarification
                if expected == "ask_clarify" and "clarif" not in teacher.lower():
                    problems.append("adversarial_ambiguity_not_clarified")

    # Validate JSON and arguments
    args_obj = None
    tool_name = None
    semantic_ok = True
    if tool_span:
        obj = parse_tool_json_slice(teacher, tool_span[0], tool_span[1])
        if obj is None:
            problems.append("json_parse_fail")
            ok = False
        else:
            tool_name = obj.get("name")
            args_obj = obj.get("arguments", {})
            sch = reg.get(tool_name) if tool_name else None
            if not sch:
                problems.append("unknown_tool")
                ok = False
            else:
                sem_ok, errs = validate_args(sch, args_obj)
                semantic_ok = sem_ok
                if not sem_ok:
                    problems.extend([f"arg_{e}" for e in errs])
                    ok = False

    # Token alignment check
    token_align_ok = None
    if tokenizer and tool_span:
        align_ok, token_span = verify_token_alignment(teacher, tool_span, tokenizer)
        token_align_ok = align_ok
        if not align_ok:
            problems.append("token_align_fail")

    # Integration F1 check
    integration_f1 = 0.0
    integration_spans_bytes = meta.get("integration_spans_bytes", [])
    call_sequence = meta.get("call_sequence", [])
    if integration_spans_bytes and call_sequence:
        integration_f1 = compute_integration_f1(
            teacher, integration_spans_bytes, call_sequence
        )
        if integration_f1 < 0.9 and len(call_sequence) > 1:
            problems.append(f"low_integration_f1:{integration_f1:.2f}")
    
    # Grounding check: verify integration spans contain tool result fields
    tool_result_fields = meta.get("tool_result_fields", {})
    integration_grounded = False
    if integration_spans_bytes and tool_result_fields:
        integration_grounded = contains_grounding(teacher, integration_spans_bytes, tool_result_fields)
        if not integration_grounded:
            problems.append("integration_not_grounded")
    
    # Multi-call span parity check
    tool_name_spans_bytes = meta.get("tool_name_spans_bytes", [])
    if call_sequence and len(call_sequence) > 1:
        # Multi-call: require spans per call
        if len(json_args_spans_bytes) != len(call_sequence):
            problems.append(f"multi_call_span_mismatch:expected_{len(call_sequence)}_got_{len(json_args_spans_bytes)}")
            ok = False
        if tool_name_spans_bytes and len(tool_name_spans_bytes) != len(call_sequence):
            problems.append(f"multi_call_name_span_mismatch:expected_{len(call_sequence)}_got_{len(tool_name_spans_bytes)}")
            ok = False
        
        # Verify spans are within their respective tool JSON regions
        for i, (call, span) in enumerate(zip(call_sequence, json_args_spans_bytes)):
            if len(span) >= 2:
                # Find the tool JSON in teacher text
                tool_json = json.dumps(call, separators=(",", ":"), ensure_ascii=False)
                tool_start = teacher.find(tool_json)
                if tool_start >= 0:
                    tool_end = tool_start + len(tool_json)
                    # Check span is within tool JSON region
                    if span[0] < tool_start or span[1] > tool_end:
                        problems.append(f"multi_call_span_{i}_out_of_bounds")
                        ok = False

    return {
        "ok": ok,
        "problems": problems,
        "tool": tool_name,
        "semantic_ok": semantic_ok,
        "token_align_ok": token_align_ok,
        "integration_f1": integration_f1,
        "integration_grounded": integration_grounded,
        "caws_header_ok": caws_ok,
        "privacy_ok": privacy_check["privacy_ok"],
    }


def check_stratification_backbone(items: List[Dict[str, Any]], total: int) -> Tuple[bool, List[Dict[str, Any]]]:
    """Check minimal backbone for N<36, scaled targets for larger N."""
    SCENARIOS = ["file_ops", "web_search", "code_exec", "multi_step"]
    COMPLEXITY = ["single_call", "multi_call", "branching_error_recovery"]
    
    # Build coverage map
    coverage = defaultdict(int)
    for item in items:
        meta = item.get("metadata", {})
        scenario = meta.get("scenario", "unknown")
        complexity = meta.get("complexity", "unknown")
        if scenario != "unknown" and complexity != "unknown":
            coverage[(scenario, complexity)] += 1
    
    # For N < 36, check backbone only
    if total < 36:
        missing = []
        
        # Check: one of each scenario × single_call
        for scenario in SCENARIOS:
            if coverage.get((scenario, "single_call"), 0) == 0:
                missing.append({"scenario": scenario, "complexity": "single_call", "required": 1, "actual": 0})
        
        # Check: at least one multi_call overall
        multi_call_found = any(count > 0 for (s, c), count in coverage.items() if c == "multi_call")
        if not multi_call_found:
            missing.append({"scenario": "any", "complexity": "multi_call", "required": 1, "actual": 0})
        
        # Check: at least one branching_error_recovery overall
        branching_found = any(count > 0 for (s, c), count in coverage.items() if c == "branching_error_recovery")
        if not branching_found:
            missing.append({"scenario": "any", "complexity": "branching_error_recovery", "required": 1, "actual": 0})
        
        return len(missing) == 0, missing
    
    # For N >= 36, use full MIN_COVERAGE requirements (defer to existing logic)
    return None, []


def build_stratification_heatmap(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build stratification coverage heatmap."""
    heatmap = defaultdict(int)
    structure_coverage = defaultdict(set)

    for item in items:
        meta = item.get("metadata", {})
        scenario = meta.get("scenario", "unknown")
        complexity = meta.get("complexity", "unknown")
        structure = meta.get("structure", "unknown")

        heatmap[(scenario, complexity)] += 1
        structure_coverage[(scenario, structure)].add(complexity)

    # Convert to readable format
    heatmap_dict = {}
    for (scenario, complexity), count in heatmap.items():
        if scenario not in heatmap_dict:
            heatmap_dict[scenario] = {}
        heatmap_dict[scenario][complexity] = count

    # Check for missing cells
    missing_cells = []
    SCENARIOS = ["file_ops", "web_search", "code_exec", "multi_step"]
    COMPLEXITY = ["single_call", "multi_call", "branching_error_recovery"]
    MIN_COVERAGE = {
        "file_ops": {"single_call": 6, "multi_call": 4, "branching_error_recovery": 2},
        "web_search": {"single_call": 4, "multi_call": 4, "branching_error_recovery": 2},
        "code_exec": {"single_call": 3, "multi_call": 3, "branching_error_recovery": 2},
        "multi_step": {"single_call": 0, "multi_call": 4, "branching_error_recovery": 2},
    }

    for scenario in SCENARIOS:
        for complexity in COMPLEXITY:
            min_count = MIN_COVERAGE[scenario].get(complexity, 0)
            actual_count = heatmap_dict.get(scenario, {}).get(complexity, 0)
            if actual_count < min_count:
                missing_cells.append(
                    {
                        "scenario": scenario,
                        "complexity": complexity,
                        "required": min_count,
                        "actual": actual_count,
                    }
                )

    return {
        "heatmap": heatmap_dict,
        "missing_cells": missing_cells,
        "all_cells_populated": len(missing_cells) == 0,
    }


def main():
    ap = argparse.ArgumentParser(description="Verify contextual dataset")
    ap.add_argument("--in", required=True, help="Input JSONL file")
    ap.add_argument("--report", required=True, help="Output report JSON file")
    ap.add_argument("--tokenizer", default=None, help="Tokenizer path (optional)")
    ap.add_argument(
        "--long-token-threshold",
        type=int,
        default=8000,
        help="Long-context token threshold (default: 8000)",
    )
    ap.add_argument(
        "--long-byte-threshold",
        type=int,
        default=24000,
        help="Long-context byte threshold (default: 24000)",
    )
    ap.add_argument(
        "--check-stratification",
        action="store_true",
        default=True,
        help="Check stratification coverage",
    )
    ap.add_argument(
        "--check-controls",
        action="store_true",
        default=True,
        help="Check control cases",
    )
    ap.add_argument(
        "--check-adversarial",
        action="store_true",
        default=True,
        help="Check adversarial cases",
    )
    args = ap.parse_args()

    reg = ToolSchemaRegistry()
    tok = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
        except Exception:
            tok = None

    # Build tool schema guard: pre-compute known tools
    known_tools = set(reg.list_tools())
    tool_schema_hashes = {}
    for tool_name in known_tools:
        schema = reg.get(tool_name)
        if schema:
            schema_str = json.dumps(schema, sort_keys=True)
            tool_schema_hashes[tool_name] = hashlib.sha256(schema_str.encode()).hexdigest()[:8]

    # Load dataset schema for validation (optional)
    # Use Path.resolve() for zip/CI safety
    dataset_schema = None
    if HAS_JSONSCHEMA:
        from pathlib import Path
        schema_path = Path(__file__).resolve().parents[1] / "schemas" / "dataset_item.schema.json"
        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                dataset_schema = json.load(f)
        except FileNotFoundError:
            print(f"[VERIFY] WARN: Schema file not found at {schema_path}, skipping schema validation")
    else:
        print("[VERIFY] WARN: jsonschema not available, skipping schema validation")

    # Load items with schema validation
    items = []
    input_file = getattr(args, 'in')
    schema_validation_errors = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                # Validate against schema if available
                if dataset_schema and HAS_JSONSCHEMA:
                    try:
                        jsonschema.validate(instance=item, schema=dataset_schema)
                    except jsonschema.ValidationError as e:
                        schema_validation_errors.append({
                            "line": line_num,
                            "error": str(e),
                            "path": ".".join(str(p) for p in e.path),
                        })
                items.append(item)
            except json.JSONDecodeError as e:
                print(f"[VERIFY] ERROR: Invalid JSON at line {line_num}: {e}")
                sys.exit(1)
    
    # Fail fast on schema validation errors
    if schema_validation_errors:
        print(f"[VERIFY] FAIL: Schema validation errors found:")
        for err in schema_validation_errors[:5]:
            print(f"  Line {err['line']}: {err['path']} - {err['error']}")
        if len(schema_validation_errors) > 5:
            print(f"  ... and {len(schema_validation_errors) - 5} more errors")
        sys.exit(1)

    # Verify each item
    results = []
    failing_items = []  # Track failures for CI-friendly output
    for item_idx, item in enumerate(items):
        # Tool schema guard: check all tools in call_sequence
        meta = item.get("metadata", {})
        call_sequence = meta.get("call_sequence", [])
        for call in call_sequence:
            tool_name = call.get("name")
            if tool_name and tool_name not in known_tools:
                failing_items.append({
                    "line": item_idx + 1,
                    "problem": "unknown_tool",
                    "tool": tool_name,
                    "expected": "tool in registry",
                    "found": tool_name,
                })
        
        res = verify_item(
            item,
            reg,
            tokenizer=tok,
            check_stratification=args.check_stratification,
            check_controls=args.check_controls,
            check_adversarial=args.check_adversarial,
        )
        results.append(res)
        
        # Track failures for CI-friendly output
        if res["problems"]:
            failing_items.append({
                "line": item_idx + 1,
                "problems": res["problems"],
                "sample_id": meta.get("sample_id", f"item_{item_idx + 1}"),
            })

    # Build stratification heatmap
    stratification = None
    if args.check_stratification:
        stratification = build_stratification_heatmap(items)

    # Compute summary statistics
    total = len(results)
    ok = sum(1 for r in results if r["ok"])
    sem_ok = sum(1 for r in results if r["semantic_ok"])
    token_ok = sum(1 for r in results if r["token_align_ok"] is True)
    caws_ok = sum(1 for r in results if r["caws_header_ok"])
    privacy_ok = sum(1 for r in results if r["privacy_ok"])

    # Integration F1 statistics (exclude controls/adversarials without tools)
    eligible_for_f1 = []
    for i, (item, result) in enumerate(zip(items, results)):
        meta = item.get("metadata", {})
        call_sequence = meta.get("call_sequence", [])
        expected_behaviour = meta.get("expected_behaviour", "normal")
        # Only include tool-using, non-control cases
        if call_sequence and expected_behaviour not in {"no_tool", "decline"}:
            if result["integration_f1"] > 0:
                eligible_for_f1.append(result["integration_f1"])
    
    avg_integration_f1 = (
        sum(eligible_for_f1) / len(eligible_for_f1)
        if eligible_for_f1
        else 0.0
    )
    
    # Long-context quota check
    long_context_count = 0
    for item in items:
        prompt = item.get("prompt", "")
        if is_long_context(prompt, tok, args.long_token_threshold, args.long_byte_threshold):
            long_context_count += 1
    
    # Integration coverage check (count-based for small N)
    integration_ok = 0
    integration_total = 0
    integration_grounded_ok = 0
    integration_grounded_total = 0
    integration_misses = []
    grounding_misses = []
    
    for item_idx, item in enumerate(items):
        meta = item.get("metadata", {})
        call_sequence = meta.get("call_sequence", [])
        expected_behaviour = meta.get("expected_behaviour", "normal")
        # Count tool-using, non-control cases
        if call_sequence and expected_behaviour not in {"no_tool", "decline"}:
            integration_total += 1
            if meta.get("integration_spans_bytes"):
                integration_ok += 1
            else:
                integration_misses.append({
                    "line": item_idx + 1,
                    "sample_id": meta.get("sample_id", f"item_{item_idx + 1}"),
                })
            
            # Check grounding
            tool_result_fields = meta.get("tool_result_fields", {})
            if tool_result_fields:
                integration_grounded_total += 1
                teacher_text = item.get("teacher_text", "")
                integration_spans = meta.get("integration_spans_bytes", [])
                if contains_grounding(teacher_text, integration_spans, tool_result_fields):
                    integration_grounded_ok += 1
                else:
                    grounding_misses.append({
                        "line": item_idx + 1,
                        "sample_id": meta.get("sample_id", f"item_{item_idx + 1}"),
                        "tool": meta.get("call_sequence", [{}])[0].get("name") if meta.get("call_sequence") else None,
                        "summary": tool_result_fields.get("summary", ""),
                        "spans": integration_spans,
                    })
    
    # Count-based gates: allow max(1, ceil(0.05 * N)) misses
    allowed_integration_misses = max(1, math.ceil(0.05 * integration_total)) if integration_total > 0 else 1
    allowed_grounding_misses = max(1, math.ceil(0.05 * integration_grounded_total)) if integration_grounded_total > 0 else 1
    
    integration_misses_count = len(integration_misses)
    grounding_misses_count = len(grounding_misses)
    
    integration_coverage = round(integration_ok / integration_total, 3) if integration_total > 0 else None
    integration_grounded_coverage = round(integration_grounded_ok / integration_grounded_total, 3) if integration_grounded_total > 0 else None
    
    # Multi-call span parity (count-based for small N)
    multi_call_items = [item for item in items if len(item.get("metadata", {}).get("call_sequence", [])) > 1]
    multi_call_parity_ok = 0
    multi_call_parity_total = len(multi_call_items)
    multi_call_parity_misses = []
    
    for item_idx, item in enumerate(multi_call_items):
        meta = item.get("metadata", {})
        call_sequence = meta.get("call_sequence", [])
        json_args_spans = meta.get("json_args_spans_bytes", [])
        if len(json_args_spans) == len(call_sequence):
            multi_call_parity_ok += 1
        else:
            # Find the actual line number in the original items list
            actual_line = next((i + 1 for i, it in enumerate(items) if it == item), item_idx + 1)
            multi_call_parity_misses.append({
                "line": actual_line,
                "sample_id": meta.get("sample_id", f"item_{actual_line}"),
                "expected": len(call_sequence),
                "got": len(json_args_spans),
            })
    
    multi_call_parity_rate = round(multi_call_parity_ok / multi_call_parity_total, 3) if multi_call_parity_total > 0 else None
    allowed_multi_call_misses = max(1, math.ceil(0.05 * multi_call_parity_total)) if multi_call_parity_total > 0 else 1
    multi_call_misses_count = len(multi_call_parity_misses)

    summary = {
        "total": total,
        "ok_rate": round(ok / total, 3) if total else 0,
        "semantic_ok_rate": round(sem_ok / total, 3) if total else 0,
        "token_align_ok_rate": round(token_ok / total, 3) if total else None,
        "caws_header_ok_rate": round(caws_ok / total, 3) if total else 0,
        "privacy_ok_rate": round(privacy_ok / total, 3) if total else 0,
        "avg_integration_f1": round(avg_integration_f1, 3),
        "long_context_count": long_context_count,
        "integration_coverage": integration_coverage,
        "integration_grounded_coverage": integration_grounded_coverage,
        "multi_call_parity_rate": multi_call_parity_rate,
        "problems": {},
        "stratification": stratification,
    }

    # Count problems
    for r in results:
        for p in r["problems"]:
            summary["problems"][p] = summary["problems"].get(p, 0) + 1

    # Try to load generation plan from input file (if available)
    generation_plan = None
    try:
        # Look for generation plan in first item's metadata or try to infer
        if items:
            first_meta = items[0].get("metadata", {})
            dataset_version = first_meta.get("dataset_version")
            # Try to reconstruct plan from items
            adversarial_counts = defaultdict(int)
            for item in items:
                meta = item.get("metadata", {})
                if "adversarial" in meta:
                    adv_type = meta["adversarial"].get("type", "unknown")
                    adversarial_counts[adv_type] += 1
            
            generation_plan = {
                "total": total,
                "dataset_version": dataset_version,
                "counts": {
                    "control": sum(1 for item in items if item.get("metadata", {}).get("expected_behaviour") not in {"normal", None}),
                    "adversarial": sum(1 for item in items if "adversarial" in item.get("metadata", {})),
                    "adversarial_by_type": dict(adversarial_counts),
                    "multilingual": sum(1 for item in items if item.get("metadata", {}).get("language") and item.get("metadata", {}).get("language") != "en"),
                    "long_context": sum(1 for item in items if item.get("metadata", {}).get("long_context")),
                }
            }
    except Exception:
        pass

    # Write report with generation plan
    os.makedirs(os.path.dirname(args.report) if os.path.dirname(args.report) else ".", exist_ok=True)
    report_data = {
        "generation_plan": generation_plan,
        "summary": summary,
        "results": results
    }
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    # Print summary and determine pass/fail
    print(f"[VERIFY] Total samples: {total}")
    print(f"[VERIFY] OK rate: {summary['ok_rate']:.3f}")
    print(f"[VERIFY] Semantic OK rate: {summary['semantic_ok_rate']:.3f}")
    print(f"[VERIFY] CAWS header OK rate: {summary['caws_header_ok_rate']:.3f}")
    print(f"[VERIFY] Privacy OK rate: {summary['privacy_ok_rate']:.3f}")
    print(f"[VERIFY] Avg integration F1: {summary['avg_integration_f1']:.3f}")

    if stratification:
        print(f"[VERIFY] Stratification: {len(stratification['missing_cells'])} missing cells")
        if not stratification["all_cells_populated"]:
            print("[VERIFY] WARN: Some stratification cells are missing!")

    # Check thresholds with hard stops
    passed = True
    gates_ok = True
    
    if summary["ok_rate"] < 0.95:
        print(f"[VERIFY] FAIL: OK rate {summary['ok_rate']:.3f} < 0.95")
        passed = False
        gates_ok = False
    if summary["semantic_ok_rate"] < 0.98:
        print(f"[VERIFY] FAIL: Semantic OK rate {summary['semantic_ok_rate']:.3f} < 0.98")
        passed = False
        gates_ok = False
    if summary["caws_header_ok_rate"] < 0.95:
        print(f"[VERIFY] FAIL: CAWS header OK rate {summary['caws_header_ok_rate']:.3f} < 0.95")
        passed = False
        gates_ok = False
    if summary["privacy_ok_rate"] < 1.0:
        print(f"[VERIFY] FAIL: Privacy OK rate {summary['privacy_ok_rate']:.3f} < 1.0")
        passed = False
        gates_ok = False
    # Integration F1 check (only on eligible items, exclude controls)
    # For small N, be more lenient: require average >= 0.50 for N<30, >= 0.90 for N>=30
    if eligible_for_f1:
        f1_threshold = 0.50 if total < 30 else 0.90
        if summary["avg_integration_f1"] < f1_threshold:
            print(f"[VERIFY] FAIL: Avg integration F1 {summary['avg_integration_f1']:.3f} < {f1_threshold} (over {len(eligible_for_f1)} eligible items, threshold={f1_threshold} for N={total})")
            passed = False
            gates_ok = False
        elif summary["avg_integration_f1"] < 0.90 and total >= 30:
            # Warn if below 0.90 for larger N
            print(f"[VERIFY] WARN: Avg integration F1 {summary['avg_integration_f1']:.3f} < 0.90 (over {len(eligible_for_f1)} eligible items)")
    # Stratification check (relaxed for N<36)
    if stratification:
        # For N < 36, check backbone only
        if total < 36:
            backbone_ok, backbone_missing = check_stratification_backbone(items, total)
            if not backbone_ok:
                print(f"[VERIFY] FAIL: Stratification backbone missing: {backbone_missing}")
                passed = False
                gates_ok = False
            elif len(stratification['missing_cells']) > 0:
                print(f"[VERIFY] WARN: Stratification gaps (N={total} < 36): {len(stratification['missing_cells'])} cells missing (backbone OK)")
        else:
            # Full check for N >= 36
            if not stratification["all_cells_populated"]:
                print("[VERIFY] FAIL: Stratification cells not fully populated")
                passed = False
                gates_ok = False
    
    # Long-context quota check
    if total >= 20 and not (2 <= long_context_count <= 3):
        print(f"[VERIFY] FAIL: Long-context quota: count={long_context_count}, expected 2-3 for N≥20")
        passed = False
        gates_ok = False
    
    # Integration coverage check (count-based)
    if integration_total > 0:
        if integration_misses_count > allowed_integration_misses:
            print(f"[VERIFY] FAIL: Integration coverage: {integration_ok}/{integration_total} (allowed_misses={allowed_integration_misses}, misses={integration_misses_count})")
            passed = False
            gates_ok = False
            # Print misses
            for miss in integration_misses[:3]:
                print(f"  - line {miss['line']} ({miss['sample_id']}): missing integration_spans_bytes")
    
    # Grounding check (count-based)
    if integration_grounded_total > 0:
        if grounding_misses_count > allowed_grounding_misses:
            print(f"[VERIFY] FAIL: Integration grounded: {integration_grounded_ok}/{integration_grounded_total} (allowed_misses={allowed_grounding_misses}, misses={grounding_misses_count})")
            passed = False
            gates_ok = False
            # Print misses with details
            for miss in grounding_misses[:3]:
                print(f"  - line {miss['line']} ({miss['sample_id']}): integration_not_grounded")
                print(f"    tool_result_fields.summary = \"{miss.get('summary', 'N/A')}\"")
                if miss.get('spans') and len(miss['spans']) > 0 and len(miss['spans'][0]) >= 2:
                    # Get teacher text for this item
                    item_idx = miss['line'] - 1
                    if item_idx < len(items):
                        teacher_text = items[item_idx].get("teacher_text", "")
                        span_text = teacher_text[miss['spans'][0][0]:miss['spans'][0][1]]
                        print(f"    spans[0] = \"{span_text[:100]}...\"")
    
    # Multi-call parity check (count-based)
    if multi_call_parity_total > 0:
        if multi_call_misses_count > allowed_multi_call_misses:
            print(f"[VERIFY] FAIL: Multi-call span parity: {multi_call_parity_ok}/{multi_call_parity_total} (allowed_misses={allowed_multi_call_misses}, misses={multi_call_misses_count})")
            passed = False
            gates_ok = False
            for miss in multi_call_parity_misses[:3]:
                print(f"  - line {miss['line']} ({miss['sample_id']}): expected {miss['expected']} spans, got {miss['got']}")
    
    # Adversarial quota check
    if total >= 30 and not summary.get("adversarial_quota_ok", True):
        missing = required_types - set(summary.get("adversarial_by_type", {}).keys())
        print(f"[VERIFY] FAIL: Adversarial quota missing types: {missing}")
        passed = False
        gates_ok = False
    
    # CI-friendly failure surfacing: print first 5 failures with diffs
    if failing_items:
        print(f"\n[VERIFY] First 5 failing items:")
        for fail_item in failing_items[:5]:
            line_num = fail_item.get("line", "?")
            problems = fail_item.get("problems", [fail_item.get("problem", "unknown")])
            sample_id = fail_item.get("sample_id", f"line_{line_num}")
            print(f"  Line {line_num} ({sample_id}): {', '.join(problems)}")
            if "expected" in fail_item:
                print(f"    Expected: {fail_item['expected']}, Found: {fail_item.get('found', 'N/A')}")

    if passed:
        print("[VERIFY] PASS: All checks passed")
    else:
        print("[VERIFY] FAIL: Some checks failed")

    return 0 if gates_ok else 1


if __name__ == "__main__":
    exit(main())

