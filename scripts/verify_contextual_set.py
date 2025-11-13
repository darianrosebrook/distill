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
import time
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict

from tools.schema_registry import ToolSchemaRegistry, validate_args
from scripts.util_token_spans import (
    bytes_to_token_span,
    normalize_text_for_alignment,
)
from scripts.util_sanitize import allowlist_urls

# ---- Small helpers for macro/micro over verify_item results ----


def _eligible(r: Dict[str, Any]) -> bool:
    """Check if result is eligible for integration F1."""
    return r.get("integration_f1_eligible", False) or (
        r.get("integration_f1_lax") is not None or r.get(
            "integration_f1_strict") is not None
    )


def _f1(r: Dict[str, Any], mode: str) -> float:
    """Extract F1 score for given mode."""
    if mode == "lax":
        return r.get("integration_f1_lax", r.get("integration_f1", 0.0))
    else:  # strict
        return r.get("integration_f1_strict", 0.0)


def _prec(r: Dict[str, Any], mode: str) -> float:
    """Extract precision for given mode."""
    if mode == "lax":
        return r.get("integration_precision_lax", r.get("integration_precision", 0.0))
    else:  # strict
        return r.get("integration_precision_strict", 0.0)


def _rec(r: Dict[str, Any], mode: str) -> float:
    """Extract recall for given mode."""
    if mode == "lax":
        return r.get("integration_recall_lax", r.get("integration_recall", 0.0))
    else:  # strict
        return r.get("integration_recall_strict", 0.0)


def macro_f1(results: List[Dict[str, Any]], mode: str = "lax") -> float:
    """Compute macro-averaged F1 over eligible items."""
    elig = [r for r in results if _eligible(r)]
    if not elig:
        return 0.0
    return sum(_f1(r, mode) for r in elig) / len(elig)


def macro_precision(results: List[Dict[str, Any]], mode: str = "lax") -> float:
    """Compute macro-averaged precision over eligible items."""
    elig = [r for r in results if _eligible(r)]
    if not elig:
        return 0.0
    return sum(_prec(r, mode) for r in elig) / len(elig)


def macro_recall(results: List[Dict[str, Any]], mode: str = "lax") -> float:
    """Compute macro-averaged recall over eligible items."""
    elig = [r for r in results if _eligible(r)]
    if not elig:
        return 0.0
    return sum(_rec(r, mode) for r in elig) / len(elig)


def micro_f1(results: List[Dict[str, Any]], mode: str = "lax") -> float:
    """Compute micro-averaged F1 over eligible items."""
    elig = [r for r in results if _eligible(r)]
    if not elig:
        return 0.0
    # Micro over items: compute global P/R, then F1
    prec = sum(_prec(r, mode) for r in elig) / len(elig) if elig else 0.0
    rec = sum(_rec(r, mode) for r in elig) / len(elig) if elig else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def per_tool_deltas(items: List[Dict[str, Any]], results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate per-tool lax/strict macro-F1 and compute delta = lax - strict.
    A sample contributes to tools present in its call_sequence.
    """
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for item, res in zip(items, results):
        meta = item.get("metadata", {})
        calls = meta.get("call_sequence", []) or []
        tool_names = {c.get("name")
                      for c in calls if isinstance(c, dict) and c.get("name")}
        for t in tool_names:
            buckets.setdefault(t, []).append(res)

    out: Dict[str, Dict[str, float]] = {}
    for tool, bucket in buckets.items():
        if not bucket:
            continue
        # Filter to eligible items only
        elig_bucket = [r for r in bucket if _eligible(r)]
        if not elig_bucket:
            continue
        lax = macro_f1(elig_bucket, mode="lax")
        strict = macro_f1(elig_bucket, mode="strict")
        out[tool] = {
            "f1_lax_macro": round(lax, 3),
            "f1_strict_macro": round(strict, 3),
            "delta_lax_minus_strict": round(lax - strict, 3),
            "sample_count": len(elig_bucket),
        }
    return out


try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


def is_long_context_item(
    item: Dict[str, Any],
    tokenizer=None,
    token_threshold: int = 8000,
    byte_threshold: int = 24000,
) -> bool:
    """
    Check if item is long-context.

    Prefers metadata flag, then computes from prompt if flag missing.
    """
    meta = item.get("metadata", {})
    if "long_context" in meta:
        return bool(meta["long_context"])

    # Fallback: compute from prompt
    prompt = item.get("prompt", "")
    if tokenizer is not None:
        try:
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            return len(ids) >= token_threshold
        except Exception:
            # Fallback to bytes if tokenization fails
            return len(prompt) >= byte_threshold
    return len(prompt) >= byte_threshold


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
                normalized_value_sets.append(
                    ("string", expand_synonyms(value)))

    # Check if at least one span contains at least one value (or synonym)
    for span in spans:
        if len(span) >= 2:
            seg_text = text[span[0]: span[1]]
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
        left_brace = s.find("{")
        r = s.rfind("}")
        if left_brace == -1 or r == -1:
            return None
        return json.loads(s[left_brace: r + 1])
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
    """Check for PII and URL violations with URL-context heuristics."""
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

    # URL-context heuristics: downgrade/upgrade severity based on context
    high_severity_pii = 0
    medium_severity_pii = 0
    contextual_pii = 0

    # Check emails/phones near contextual keywords (downgrade to contextual)
    contextual_keywords = {"contact", "support",
                           "careers", "press", "help", "info"}
    high_severity_keywords = {"invoice", "ssn",
                              "patient", "claim", "medical", "financial"}

    for email in emails:
        # Find email position in text
        email_pos = text.find(email)
        if email_pos >= 0:
            # Check ±30 chars around email
            context_start = max(0, email_pos - 30)
            context_end = min(len(text), email_pos + len(email) + 30)
            context = text[context_start:context_end].lower()

            # Check for high-severity keywords
            if any(kw in context for kw in high_severity_keywords):
                high_severity_pii += 1
            # Check for contextual keywords (downgrade)
            elif any(kw in context for kw in contextual_keywords):
                contextual_pii += 1
            else:
                medium_severity_pii += 1

    # Check phones (similar pattern)
    phone_pattern = r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"
    phones = re.findall(phone_pattern, text)
    for phone in phones:
        phone_pos = text.find(phone)
        if phone_pos >= 0:
            context_start = max(0, phone_pos - 30)
            context_end = min(len(text), phone_pos + len(phone) + 30)
            context = text[context_start:context_end].lower()

            if any(kw in context for kw in high_severity_keywords):
                high_severity_pii += 1
            elif any(kw in context for kw in contextual_keywords):
                contextual_pii += 1
            else:
                medium_severity_pii += 1

    # Privacy OK if no high-severity PII and URL allowlist passes
    privacy_ok = high_severity_pii == 0 and url_ok

    return {
        "emails_found": len(emails),
        "uuids_found": len(uuids),
        "urls_found": len(urls),
        "url_allowlist_ok": url_ok,
        "high_severity_pii": high_severity_pii,
        "medium_severity_pii": medium_severity_pii,
        "contextual_pii": contextual_pii,
        "privacy_ok": privacy_ok,
    }


def normalize_text_for_grounding(s: str) -> str:
    """Normalize text for grounding comparison: casefold + collapse whitespace."""
    return " ".join(str(s).casefold().split())


def grounded_values_in_span(seg: str, fields: Dict[str, str]) -> bool:
    """
    Check if span contains any grounded value from tool_result_fields (lax mode).

    Returns True if span contains:
    - Any string value (normalized)
    - Any numeric value (with word boundaries)
    - URL host + first path segment
    """
    if not seg or not fields:
        return False

    seg_norm = normalize_text_for_grounding(seg)

    # Check string values
    for v in fields.values():
        if isinstance(v, str) and v:
            v_norm = normalize_text_for_grounding(v)
            if v_norm in seg_norm:
                return True

    # Check numeric values (with word boundaries)
    import re
    for v in fields.values():
        if isinstance(v, (int, float)):
            # Match the number with word boundaries
            num_str = str(int(v))
            if re.search(rf"\b{re.escape(num_str)}\b", seg_norm):
                return True

    # Check URLs (host + first path segment)
    for v in fields.values():
        if isinstance(v, str) and "://" in v:
            try:
                from urllib.parse import urlparse
                u = urlparse(v)
                host = (u.netloc or "").lower().lstrip("www.")
                path = (u.path or "/").strip("/").split("/", 1)[0]
                if host and host in seg_norm:
                    if not path or path in seg_norm:
                        return True
            except Exception:
                pass

    return False


def grounded_values_in_span_strict(seg: str, fields: Dict[str, str]) -> bool:
    """
    Check if span contains at least one keyed value from tool_result_fields (strict mode).

    Strict mode requires at least one of these keyed fields to be grounded:
    - summary
    - lines
    - count
    - top_k
    - results (if list with items)

    Returns True if span contains any keyed value, False otherwise.
    """
    if not seg or not fields:
        return False

    # Keyed fields that must be grounded in strict mode
    keyed_fields = ["summary", "lines", "count", "top_k", "results"]

    seg_norm = normalize_text_for_grounding(seg)
    import re

    # Check keyed string values
    for key in keyed_fields:
        if key in fields:
            v = fields[key]
            if isinstance(v, str) and v:
                v_norm = normalize_text_for_grounding(v)
                if v_norm in seg_norm:
                    return True

    # Check keyed numeric values (with word boundaries and tolerance)
    for key in keyed_fields:
        if key in fields:
            v = fields[key]
            if isinstance(v, (int, float)):
                # Try exact match first
                num_str = str(int(v))
                if re.search(rf"\b{re.escape(num_str)}\b", seg_norm):
                    return True
                # Try with tolerance: strip commas, handle percentages, allow 2 ULP difference
                # Normalize: remove commas, handle % -> divide by 100, handle locale formatting
                # Locale-aware: handle different decimal separators (1.234,56 vs 1,234.56)
                # Try to detect locale format: if comma appears after period (or vice versa), it's likely locale-formatted
                # For now, normalize by removing all non-digit characters except one decimal point
                seg_digits = re.sub(r'[^\d.]', '', seg_norm)
                # Also try replacing comma with period (for European formats like 1.234,56)
                seg_digits_alt = re.sub(
                    r'[^\d,]', '', seg_norm).replace(',', '.')
                try:
                    seg_num = float(seg_digits) if seg_digits else None
                    seg_num_alt = float(
                        seg_digits_alt) if seg_digits_alt and seg_digits_alt != seg_digits else None
                    if seg_num is not None:
                        # Check absolute difference <= 1e-6 or <= 2 ULP
                        abs_diff = abs(float(v) - seg_num)
                        if abs_diff <= 1e-6:
                            return True
                        # Check percentage format (e.g., "50%" vs 0.5)
                        if "%" in seg_norm or "percent" in seg_norm.lower():
                            seg_pct = seg_num / 100.0
                            if abs(float(v) - seg_pct) <= 1e-6:
                                return True
                    # Try alternative format (comma as decimal separator)
                    if seg_num_alt is not None:
                        abs_diff_alt = abs(float(v) - seg_num_alt)
                        if abs_diff_alt <= 1e-6:
                            return True
                        if "%" in seg_norm or "percent" in seg_norm.lower():
                            seg_pct_alt = seg_num_alt / 100.0
                            if abs(float(v) - seg_pct_alt) <= 1e-6:
                                return True
                except (ValueError, TypeError):
                    pass

    # Check keyed URLs (host + first path segment)
    for key in keyed_fields:
        if key in fields:
            v = fields[key]
            if isinstance(v, str) and "://" in v:
                try:
                    from urllib.parse import urlparse
                    u = urlparse(v)
                    host = (u.netloc or "").lower().lstrip("www.")
                    path = (u.path or "/").strip("/").split("/", 1)[0]
                    if host and host in seg_norm:
                        if not path or path in seg_norm:
                            return True
                except Exception:
                    pass

    # Check results list (if present and non-empty)
    if "results" in fields:
        results = fields["results"]
        if isinstance(results, list) and len(results) > 0:
            # Check if any result item is mentioned
            for item in results[:3]:  # Check first 3 items
                if isinstance(item, str):
                    item_norm = normalize_text_for_grounding(item)
                    if item_norm in seg_norm:
                        return True

    return False


def compute_integration_f1(
    teacher_text: str,
    integration_spans_bytes: List[List[int]],
    tool_result_fields: Dict[str, str],
) -> Tuple[float, float, float]:
    """
    Compute per-item Integration F1 score (lax mode).

    Args:
        teacher_text: Full teacher text
        integration_spans_bytes: List of [start, end] byte spans
        tool_result_fields: Dict of tool result fields for grounding

    Returns:
        (precision, recall, f1) tuple
    """
    if not integration_spans_bytes or not tool_result_fields:
        return (0.0, 0.0, 0.0)

    # Count grounded spans
    grounded_count = 0
    for span in integration_spans_bytes:
        if len(span) >= 2:
            seg = teacher_text[span[0]: span[1]]
            if grounded_values_in_span(seg, tool_result_fields):
                grounded_count += 1

    # Precision: grounded spans / total spans
    total_spans = max(1, len(integration_spans_bytes))
    precision = grounded_count / total_spans

    # Recall: 1 if at least one grounded span, else 0
    recall = 1.0 if grounded_count >= 1 else 0.0

    # F1: harmonic mean
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return (precision, recall, f1)


def compute_integration_f1_strict(
    teacher_text: str,
    integration_spans_bytes: List[List[int]],
    tool_result_fields: Dict[str, str],
) -> Tuple[float, float, float]:
    """
    Compute per-item Integration F1 score (strict mode).

    Strict mode requires at least one keyed value (summary, lines, count, top_k) to be grounded.

    Args:
        teacher_text: Full teacher text
        integration_spans_bytes: List of [start, end] byte spans
        tool_result_fields: Dict of tool result fields for grounding

    Returns:
        (precision, recall, f1) tuple
    """
    if not integration_spans_bytes or not tool_result_fields:
        return (0.0, 0.0, 0.0)

    # Count grounded spans (strict mode)
    grounded_count = 0
    for span in integration_spans_bytes:
        if len(span) >= 2:
            seg = teacher_text[span[0]: span[1]]
            if grounded_values_in_span_strict(seg, tool_result_fields):
                grounded_count += 1

    # Precision: grounded spans / total spans
    total_spans = max(1, len(integration_spans_bytes))
    precision = grounded_count / total_spans

    # Recall: 1 if at least one grounded span, else 0
    recall = 1.0 if grounded_count >= 1 else 0.0

    # F1: harmonic mean
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return (precision, recall, f1)


def verify_token_alignment(
    text: str,
    span_bytes: List[int],
    tokenizer,
    text_norm: Optional[str] = None,
    line_endings: Optional[str] = None,
) -> Tuple[bool, Optional[Tuple[int, int]]]:
    """
    Verify token alignment round-trip.

    Args:
        text: Text containing the span
        span_bytes: [start, end] byte offsets
        tokenizer: Tokenizer instance
        text_norm: Normalization format ("NFC" or None)
        line_endings: Line ending format ("LF" or None)
    """
    if not span_bytes or len(span_bytes) < 2:
        return False, None

    # Normalize text to match format used when computing byte spans
    text_normalized = normalize_text_for_alignment(
        text, text_norm=text_norm, line_endings=line_endings)

    token_span = bytes_to_token_span(
        text_normalized, span_bytes[0], span_bytes[1], tokenizer)
    if token_span is None:
        return False, None

    # Round-trip check: decode tokens and compare with original
    try:
        decoded = tokenizer.decode(
            tokenizer.encode(text_normalized[: span_bytes[1]], add_special_tokens=False)[
                token_span[0]: token_span[1]
            ],
            skip_special_tokens=True,
        )
        original = text_normalized[span_bytes[0]: span_bytes[1]]
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
        if privacy_check["high_severity_pii"] > 0:
            problems.append(
                f"high_severity_pii:{privacy_check['high_severity_pii']}")
        if privacy_check["emails_found"] > 0:
            problems.append(f"emails_found:{privacy_check['emails_found']}")
        if privacy_check["uuids_found"] > 0:
            problems.append(f"uuids_found:{privacy_check['uuids_found']}")
        if not privacy_check["url_allowlist_ok"]:
            problems.append("url_not_in_allowlist")
    # Warn on medium-severity PII (but don't fail)
    if privacy_check.get("medium_severity_pii", 0) > 0:
        problems.append(
            f"medium_severity_pii_warning:{privacy_check['medium_severity_pii']}")

    # ToS compliance check: assert teacher_reasoning_content field is absent
    if "teacher_reasoning_content" in item:
        problems.append("tos_violation:teacher_reasoning_content_present")
        ok = False

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
            "integration_f1": 0.0,  # Keep for backward compatibility
            "integration_precision": 0.0,
            "integration_recall": 0.0,
            "integration_f1_lax": 0.0,
            "integration_precision_lax": 0.0,
            "integration_recall_lax": 0.0,
            "integration_f1_strict": 0.0,
            "integration_precision_strict": 0.0,
            "integration_recall_strict": 0.0,
            "integration_grounded": False,
            "caws_header_ok": caws_ok,
            "privacy_ok": privacy_check["privacy_ok"],
        }

    # Get normalization metadata for alignment
    text_norm = meta.get("text_norm")
    line_endings = meta.get("line_endings")
    spans_target = meta.get("spans_target", "teacher")

    # Choose the correct buffer for span alignment
    if spans_target == "teacher":
        target_text = teacher
    else:
        target_text = prompt

    # Normalize target text for alignment
    target_text_normalized = normalize_text_for_alignment(
        target_text, text_norm=text_norm, line_endings=line_endings
    )

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

    # Token alignment check (use normalized target text)
    token_align_ok = None
    if tokenizer and tool_span:
        align_ok, token_span = verify_token_alignment(
            target_text_normalized, tool_span, tokenizer,
            text_norm=text_norm, line_endings=line_endings
        )
        token_align_ok = align_ok
        if not align_ok:
            problems.append("token_align_fail")

    # Multi-call parity check
    if call_sequence and len(call_sequence) > 1:
        k = len(call_sequence)
        json_spans_count = len(json_args_spans_bytes)
        tool_name_spans_bytes = meta.get("tool_name_spans_bytes", [])
        tool_name_spans_count = len(tool_name_spans_bytes)

        if json_spans_count != k:
            problems.append(
                f"json_args_spans_mismatch:expected_{k}_got_{json_spans_count}")
        if tool_name_spans_count != k and tool_name_spans_count > 0:
            problems.append(
                f"tool_name_spans_mismatch:expected_{k}_got_{tool_name_spans_count}")

    # Integration F1 check (per-item) - compute both lax and strict
    integration_f1_lax = 0.0
    integration_precision_lax = 0.0
    integration_recall_lax = 0.0
    integration_f1_strict = 0.0
    integration_precision_strict = 0.0
    integration_recall_strict = 0.0
    integration_spans_bytes = meta.get("integration_spans_bytes", [])
    tool_result_fields = meta.get("tool_result_fields", {})
    call_sequence = meta.get("call_sequence", [])

    if integration_spans_bytes and tool_result_fields and call_sequence:
        # Compute lax F1
        integration_precision_lax, integration_recall_lax, integration_f1_lax = compute_integration_f1(
            teacher, integration_spans_bytes, tool_result_fields
        )
        # Compute strict F1
        integration_precision_strict, integration_recall_strict, integration_f1_strict = compute_integration_f1_strict(
            teacher, integration_spans_bytes, tool_result_fields
        )
        # Gate on lax F1 (keep existing threshold)
        if integration_f1_lax < 0.75:
            problems.append(f"low_integration_f1_lax:{integration_f1_lax:.2f}")
        # Warn on strict failures (but don't gate)
        if integration_f1_strict < 0.75:
            problems.append(
                f"low_integration_f1_strict:{integration_f1_strict:.2f}")

    # Grounding check: verify integration spans contain tool result fields (lax mode)
    tool_result_fields = meta.get("tool_result_fields", {})
    negative_control = meta.get("negative_control", False)
    integration_grounded = False
    if integration_spans_bytes and tool_result_fields:
        integration_grounded = contains_grounding(
            teacher, integration_spans_bytes, tool_result_fields)

    # Negative control check: require grounded=false when negative_control=true
    if negative_control and integration_grounded:
        problems.append(
            "negative_control_grounded:integration_spans_grounded_when_should_be_empty")
        ok = False
    elif not negative_control and not integration_grounded and integration_spans_bytes and tool_result_fields:
        # Normal case: integration spans should be grounded
        problems.append("integration_not_grounded")

    # Multi-call span parity check
    tool_name_spans_bytes = meta.get("tool_name_spans_bytes", [])
    if call_sequence and len(call_sequence) > 1:
        # Multi-call: require spans per call
        if len(json_args_spans_bytes) != len(call_sequence):
            problems.append(
                f"multi_call_span_mismatch:expected_{len(call_sequence)}_got_{len(json_args_spans_bytes)}")
            ok = False
        if tool_name_spans_bytes and len(tool_name_spans_bytes) != len(call_sequence):
            problems.append(
                f"multi_call_name_span_mismatch:expected_{len(call_sequence)}_got_{len(tool_name_spans_bytes)}")
            ok = False

        # Verify spans are within their respective tool JSON regions
        for i, (call, span) in enumerate(zip(call_sequence, json_args_spans_bytes)):
            if len(span) >= 2:
                # Find the tool JSON in teacher text
                tool_json = json.dumps(call, separators=(
                    ",", ":"), ensure_ascii=False)
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
        "integration_f1": integration_f1_lax,  # Keep for backward compatibility
        "integration_precision": integration_precision_lax,
        "integration_recall": integration_recall_lax,
        "integration_f1_lax": integration_f1_lax,
        "integration_precision_lax": integration_precision_lax,
        "integration_recall_lax": integration_recall_lax,
        "integration_f1_strict": integration_f1_strict,
        "integration_precision_strict": integration_precision_strict,
        "integration_recall_strict": integration_recall_strict,
        "integration_grounded": integration_grounded,
        "caws_header_ok": caws_ok,
        "privacy_ok": privacy_check["privacy_ok"],
        "privacy_check": privacy_check,  # Include full privacy check details
    }


def check_stratification_backbone(items: List[Dict[str, Any]], total: int) -> Tuple[bool, List[Dict[str, Any]]]:
    """Check minimal backbone for N<36, scaled targets for larger N."""
    SCENARIOS = ["file_ops", "web_search", "code_exec", "multi_step"]

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
        # Note: multi_step scenarios don't have single_call (MIN_COVERAGE says 0)
        for scenario in SCENARIOS:
            if scenario == "multi_step":
                continue  # Skip multi_step - it doesn't have single_call samples
            if coverage.get((scenario, "single_call"), 0) == 0:
                missing.append(
                    {"scenario": scenario, "complexity": "single_call", "required": 1, "actual": 0})

        # Check: at least one multi_call overall
        multi_call_found = any(count > 0 for (
            s, c), count in coverage.items() if c == "multi_call")
        if not multi_call_found:
            missing.append(
                {"scenario": "any", "complexity": "multi_call", "required": 1, "actual": 0})

        # Check: at least one branching_error_recovery overall
        branching_found = any(count > 0 for (
            s, c), count in coverage.items() if c == "branching_error_recovery")
        if not branching_found:
            missing.append(
                {"scenario": "any", "complexity": "branching_error_recovery", "required": 1, "actual": 0})

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
    ap.add_argument("--tokenizer", default=None,
                    help="Tokenizer path (optional)")
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
    ap.add_argument(
        "--perf-budget-sec-per-100",
        type=float,
        default=2.0,
        help="Performance budget: seconds per 100 items (default: 2.0)",
    )
    ap.add_argument(
        "--min-eligible-for-gates",
        type=int,
        default=15,
        help="Minimum eligible items required for F1 gates (default: 15)",
    )
    ap.add_argument(
        "--fail-on-fingerprint-mismatch",
        action="store_true",
        default=True,
        help="Fail verification on fingerprint mismatch (default: True)",
    )
    ap.add_argument(
        "--no-fail-on-fingerprint-mismatch",
        dest="fail_on_fingerprint_mismatch",
        action="store_false",
        help="Do not fail on fingerprint mismatch (overrides default)",
    )
    ap.add_argument(
        "--secondary-tokenizer",
        required=False,
        help="Optional secondary tokenizer path for cross-tokenizer verification",
    )
    ap.add_argument(
        "--next-registry",
        required=False,
        help="Optional path to next registry version for forward-compat check",
    )
    args = ap.parse_args()

    # Start timing for performance budget
    verification_start_time = time.time()

    reg = ToolSchemaRegistry()
    tok = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
        except Exception:
            tok = None

    # Load secondary tokenizer if provided
    secondary_tok = None
    if args.secondary_tokenizer:
        try:
            from transformers import AutoTokenizer
            secondary_tok = AutoTokenizer.from_pretrained(
                args.secondary_tokenizer, use_fast=True)
        except Exception as e:
            print(
                f"[VERIFY] WARN: Could not load secondary tokenizer from {args.secondary_tokenizer}: {e}")
            secondary_tok = None

    # Load next registry if provided
    next_reg = None
    if args.next_registry:
        try:
            # Try to load as a Python module path or file path
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "next_registry", args.next_registry)
            if spec and spec.loader:
                next_reg_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(next_reg_module)
                next_reg = next_reg_module.ToolSchemaRegistry() if hasattr(
                    next_reg_module, 'ToolSchemaRegistry') else None
            if next_reg is None:
                print(
                    f"[VERIFY] WARN: Could not load next registry from {args.next_registry}")
        except Exception as e:
            print(
                f"[VERIFY] WARN: Could not load next registry from {args.next_registry}: {e}")
            next_reg = None

    # Build tool schema guard: pre-compute known tools
    known_tools = set(reg.list_tools())
    tool_schema_hashes = {}
    for tool_name in known_tools:
        schema = reg.get(tool_name)
        if schema:
            schema_str = json.dumps(schema, sort_keys=True)
            tool_schema_hashes[tool_name] = hashlib.sha256(
                schema_str.encode()).hexdigest()[:8]

    # Load dataset schema for validation (optional)
    # Use Path.resolve() for zip/CI safety
    dataset_schema = None
    if HAS_JSONSCHEMA:
        from pathlib import Path
        schema_path = Path(__file__).resolve(
        ).parents[1] / "schemas" / "dataset_item.schema.json"
        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                dataset_schema = json.load(f)
        except FileNotFoundError:
            print(
                f"[VERIFY] WARN: Schema file not found at {schema_path}, skipping schema validation")
    else:
        print("[VERIFY] WARN: jsonschema not available, skipping schema validation")

    # Load items with schema validation and header detection
    items = []
    input_file = getattr(args, 'in')
    schema_validation_errors = []
    dataset_header = None
    header_line_num = None

    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)

                # Check if this is the header line
                if item.get("__header__") is True:
                    dataset_header = item
                    header_line_num = line_num
                    # Skip header line, don't add to items
                    continue

                # Validate against schema if available
                if dataset_schema and HAS_JSONSCHEMA:
                    try:
                        jsonschema.validate(
                            instance=item, schema=dataset_schema)
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
        print("[VERIFY] FAIL: Schema validation errors found:")
        for err in schema_validation_errors[:5]:
            print(f"  Line {err['line']}: {err['path']} - {err['error']}")
        if len(schema_validation_errors) > 5:
            print(f"  ... and {len(schema_validation_errors) - 5} more errors")
        sys.exit(1)

    # Verify fingerprints from header and initialize integration_span_cap
    fingerprint_warnings = []
    fingerprint_errors = []
    integration_span_cap = 3  # Default cap

    if dataset_header:
        # Extract fingerprints from header
        header_tokenizer_fp = dataset_header.get("tokenizer_fingerprint")
        header_registry_sha256 = dataset_header.get("tool_registry_sha256")
        header_integration_span_cap = dataset_header.get(
            "integration_span_cap", 3)

        # Update integration span cap from header
        integration_span_cap = header_integration_span_cap

        # Compute current fingerprints
        current_tokenizer_fp = None
        if args.tokenizer:
            # Import compute_tokenizer_fingerprint from generate_contextual_prompts
            try:
                from scripts.generate_contextual_prompts import compute_tokenizer_fingerprint
                current_tokenizer_fp = compute_tokenizer_fingerprint(
                    args.tokenizer)
            except Exception as e:
                fingerprint_warnings.append(
                    f"Could not compute current tokenizer fingerprint: {e}")

        # Compute current registry fingerprint
        try:
            from scripts.generate_contextual_prompts import compute_registry_fingerprint
            current_registry_sha256 = compute_registry_fingerprint(reg)
        except Exception as e:
            fingerprint_warnings.append(
                f"Could not compute current registry fingerprint: {e}")
            current_registry_sha256 = None

        # Verify tokenizer fingerprint
        if header_tokenizer_fp and current_tokenizer_fp:
            header_sha256 = header_tokenizer_fp.get("sha256")
            current_sha256 = current_tokenizer_fp.get("sha256")
            if header_sha256 and current_sha256 and header_sha256 != current_sha256:
                fingerprint_errors.append(
                    f"Tokenizer fingerprint mismatch: header={header_sha256[:16]}..., current={current_sha256[:16]}...")
            elif header_tokenizer_fp.get("id") != current_tokenizer_fp.get("id"):
                fingerprint_warnings.append(
                    f"Tokenizer ID mismatch: header={header_tokenizer_fp.get('id')}, current={current_tokenizer_fp.get('id')}")

        # Verify registry fingerprint
        if header_registry_sha256 and current_registry_sha256:
            if header_registry_sha256 != current_registry_sha256:
                fingerprint_errors.append(
                    f"Tool registry fingerprint mismatch: header={header_registry_sha256[:16]}..., current={current_registry_sha256[:16]}...")
    else:
        fingerprint_warnings.append(
            "Dataset header missing (expected first line with __header__: true)")

    # Print fingerprint warnings/errors
    if fingerprint_warnings:
        for warn in fingerprint_warnings:
            print(f"[VERIFY] WARN: {warn}")

    if fingerprint_errors:
        print("[VERIFY] FAIL: Fingerprint verification errors:")
        for err in fingerprint_errors:
            print(f"  {err}")
        if args.fail_on_fingerprint_mismatch:
            sys.exit(1)
        else:
            print(
                "[VERIFY] WARN: Continuing despite fingerprint mismatch (--no-fail-on-fingerprint-mismatch)")

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

        # Forward-compat check: verify against next registry if provided
        if next_reg:
            call_sequence = meta.get("call_sequence", [])
            for call in call_sequence:
                tool_name = call.get("name")
                if tool_name:
                    next_schema = next_reg.get(tool_name)
                    if not next_schema:
                        res["problems"].append(
                            f"unknown_tool_next_registry:{tool_name}")
                    else:
                        # Check arg semantics against next registry
                        args_obj = call.get("arguments", {})
                        sem_ok, errs = validate_args(next_schema, args_obj)
                        if not sem_ok:
                            res["problems"].extend(
                                [f"arg_next_registry_{e}" for e in errs])

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
    # Aggregate privacy statistics
    total_high_severity_pii = sum(r.get("privacy_check", {}).get(
        "high_severity_pii", 0) if isinstance(r.get("privacy_check"), dict) else 0 for r in results)
    total_medium_severity_pii = sum(r.get("privacy_check", {}).get(
        "medium_severity_pii", 0) if isinstance(r.get("privacy_check"), dict) else 0 for r in results)
    total_contextual_pii = sum(r.get("privacy_check", {}).get(
        "contextual_pii", 0) if isinstance(r.get("privacy_check"), dict) else 0 for r in results)

    # Integration F1 statistics (macro-averaged over eligible items only)
    eligible_items = []
    eligible_results = []
    for i, (item, result) in enumerate(zip(items, results)):
        meta = item.get("metadata", {})
        call_sequence = meta.get("call_sequence", [])
        expected_behaviour = meta.get("expected_behaviour", "normal")
        # Only include tool-using, non-control cases
        if call_sequence and expected_behaviour not in {"no_tool", "decline"}:
            eligible_items.append(item)
            eligible_results.append(result)

    # Macro-average F1, precision, recall over eligible items (lax and strict)
    if eligible_results:
        # Lax mode metrics
        f1_scores_lax = [r.get("integration_f1_lax", r.get(
            "integration_f1", 0.0)) for r in eligible_results]
        precisions_lax = [r.get("integration_precision_lax", r.get(
            "integration_precision", 0.0)) for r in eligible_results]
        recalls_lax = [r.get("integration_recall_lax", r.get(
            "integration_recall", 0.0)) for r in eligible_results]
        avg_integration_f1_macro_lax = sum(f1_scores_lax) / len(f1_scores_lax)
        avg_integration_precision_macro_lax = sum(
            precisions_lax) / len(precisions_lax)
        avg_integration_recall_macro_lax = sum(recalls_lax) / len(recalls_lax)

        # Strict mode metrics
        f1_scores_strict = [r.get("integration_f1_strict", 0.0)
                            for r in eligible_results]
        precisions_strict = [
            r.get("integration_precision_strict", 0.0) for r in eligible_results]
        recalls_strict = [r.get("integration_recall_strict", 0.0)
                          for r in eligible_results]
        avg_integration_f1_macro_strict = sum(
            f1_scores_strict) / len(f1_scores_strict)
        avg_integration_precision_macro_strict = sum(
            precisions_strict) / len(precisions_strict)
        avg_integration_recall_macro_strict = sum(
            recalls_strict) / len(recalls_strict)

        # Micro-F1: aggregate TP/FP/FN across all items, then compute F1
        # For micro-F1, we sum all grounded spans and total spans across items
        total_grounded_spans_lax = sum(precisions_lax[i] * len(item.get("metadata", {}).get("integration_spans_bytes", []))
                                       for i, item in enumerate(eligible_items)
                                       if len(item.get("metadata", {}).get("integration_spans_bytes", [])) > 0)
        total_spans_lax = sum(len(item.get("metadata", {}).get("integration_spans_bytes", []))
                              for item in eligible_items)
        micro_precision_lax = total_grounded_spans_lax / \
            total_spans_lax if total_spans_lax > 0 else 0.0
        # At least one item has recall=1
        micro_recall_lax = 1.0 if sum(recalls_lax) > 0 else 0.0
        if micro_precision_lax + micro_recall_lax == 0:
            avg_integration_f1_micro_lax = 0.0
        else:
            avg_integration_f1_micro_lax = 2 * micro_precision_lax * \
                micro_recall_lax / (micro_precision_lax + micro_recall_lax)

        # Micro-F1 strict
        total_grounded_spans_strict = sum(precisions_strict[i] * len(item.get("metadata", {}).get("integration_spans_bytes", []))
                                          for i, item in enumerate(eligible_items)
                                          if len(item.get("metadata", {}).get("integration_spans_bytes", [])) > 0)
        micro_precision_strict = total_grounded_spans_strict / \
            total_spans_lax if total_spans_lax > 0 else 0.0
        micro_recall_strict = 1.0 if sum(recalls_strict) > 0 else 0.0
        if micro_precision_strict + micro_recall_strict == 0:
            avg_integration_f1_micro_strict = 0.0
        else:
            avg_integration_f1_micro_strict = 2 * micro_precision_strict * \
                micro_recall_strict / \
                (micro_precision_strict + micro_recall_strict)

        # Count misses (recall = 0) for both modes
        integration_misses_count_lax = sum(1 for r in recalls_lax if r == 0.0)
        integration_misses_count_strict = sum(
            1 for r in recalls_strict if r == 0.0)
        allowed_integration_f1_misses = max(
            1, math.ceil(0.05 * len(eligible_items)))

        # Backward compatibility: use lax metrics for old field names
        avg_integration_f1 = avg_integration_f1_macro_lax
        avg_integration_precision = avg_integration_precision_macro_lax
        avg_integration_recall = avg_integration_recall_macro_lax
        integration_misses_count = integration_misses_count_lax
    else:
        avg_integration_f1 = 1.0
        avg_integration_precision = 1.0
        avg_integration_recall = 1.0
        avg_integration_f1_macro_lax = 1.0
        avg_integration_precision_macro_lax = 1.0
        avg_integration_recall_macro_lax = 1.0
        avg_integration_f1_macro_strict = 1.0
        avg_integration_precision_macro_strict = 1.0
        avg_integration_recall_macro_strict = 1.0
        avg_integration_f1_micro_lax = 1.0
        avg_integration_f1_micro_strict = 1.0
        integration_misses_count = 0
        integration_misses_count_lax = 0
        integration_misses_count_strict = 0
        allowed_integration_f1_misses = 1

    # Long-context quota check (use metadata flag when available)
    long_context_count = 0
    for item in items:
        if is_long_context_item(item, tok, args.long_token_threshold, args.long_byte_threshold):
            long_context_count += 1

    # Integration span cap check (already set from header if available, otherwise default to 3)
    integration_spans_over_cap_count = 0
    integration_spans_over_cap_items = []
    integration_span_count_histogram = defaultdict(
        int)  # Count of items with N spans
    for item_idx, item in enumerate(items):
        meta = item.get("metadata", {})
        integration_spans = meta.get("integration_spans_bytes", [])
        span_count = len(integration_spans)
        # Update histogram
        integration_span_count_histogram[span_count] += 1
        # Check if over cap
        if span_count > integration_span_cap:
            integration_spans_over_cap_count += 1
            integration_spans_over_cap_items.append({
                "line": item_idx + 1,
                "sample_id": meta.get("sample_id", f"item_{item_idx + 1}"),
                "span_count": span_count,
                "cap": integration_span_cap,
            })

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
    allowed_integration_misses = max(1, math.ceil(
        0.05 * integration_total)) if integration_total > 0 else 1
    allowed_grounding_misses = max(1, math.ceil(
        0.05 * integration_grounded_total)) if integration_grounded_total > 0 else 1

    integration_misses_count = len(integration_misses)
    grounding_misses_count = len(grounding_misses)

    integration_coverage = round(
        integration_ok / integration_total, 3) if integration_total > 0 else None
    integration_grounded_coverage = round(
        integration_grounded_ok / integration_grounded_total, 3) if integration_grounded_total > 0 else None

    # Multi-call span parity (count-based for small N)
    multi_call_items = [item for item in items if len(
        item.get("metadata", {}).get("call_sequence", [])) > 1]
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
            actual_line = next(
                (i + 1 for i, it in enumerate(items) if it == item), item_idx + 1)
            multi_call_parity_misses.append({
                "line": actual_line,
                "sample_id": meta.get("sample_id", f"item_{actual_line}"),
                "expected": len(call_sequence),
                "got": len(json_args_spans),
            })

    multi_call_parity_rate = round(
        multi_call_parity_ok / multi_call_parity_total, 3) if multi_call_parity_total > 0 else None
    # Initialize variables that may be referenced in summary
    num_eligible = len(eligible_items)
    num_controls = sum(1 for item in items if item.get("metadata", {}).get(
        "expected_behaviour") in {"no_tool", "decline"})
    num_negative_controls = sum(1 for item in items if item.get(
        "metadata", {}).get("negative_control", False))
    controls_with_integration = 0

    allowed_multi_call_misses = max(1, math.ceil(
        0.05 * multi_call_parity_total)) if multi_call_parity_total > 0 else 1
    multi_call_misses_count = len(multi_call_parity_misses)

    summary = {
        "total": total,
        "num_eligible": num_eligible,
        "num_controls": num_controls,
        "num_negative_controls": num_negative_controls,
        "controls_with_integration": controls_with_integration,
        "ok_rate": round(ok / total, 3) if total else 0,
        "semantic_ok_rate": round(sem_ok / total, 3) if total else 0,
        "token_align_ok_rate": round(token_ok / total, 3) if total else None,
        "caws_header_ok_rate": round(caws_ok / total, 3) if total else 0,
        "privacy_ok_rate": round(privacy_ok / total, 3) if total else 0,
        "high_severity_pii_count": total_high_severity_pii,
        "medium_severity_pii_count": total_medium_severity_pii,
        "contextual_pii_count": total_contextual_pii,
        # Backward compatibility (lax mode)
        "avg_integration_f1": round(avg_integration_f1, 3),
        "avg_integration_precision": round(avg_integration_precision, 3) if eligible_results else None,
        "avg_integration_recall": round(avg_integration_recall, 3) if eligible_results else None,
        # New metrics: macro-F1 (lax and strict)
        "avg_integration_f1_macro_lax": round(avg_integration_f1_macro_lax, 3) if eligible_results else 1.0,
        "avg_integration_precision_macro_lax": round(avg_integration_precision_macro_lax, 3) if eligible_results else None,
        "avg_integration_recall_macro_lax": round(avg_integration_recall_macro_lax, 3) if eligible_results else None,
        "avg_integration_f1_macro_strict": round(avg_integration_f1_macro_strict, 3) if eligible_results else 1.0,
        "avg_integration_precision_macro_strict": round(avg_integration_precision_macro_strict, 3) if eligible_results else None,
        "avg_integration_recall_macro_strict": round(avg_integration_recall_macro_strict, 3) if eligible_results else None,
        # New metrics: micro-F1 (lax and strict)
        "avg_integration_f1_micro_lax": round(avg_integration_f1_micro_lax, 3) if eligible_results else 1.0,
        "avg_integration_f1_micro_strict": round(avg_integration_f1_micro_strict, 3) if eligible_results else 1.0,
        # Eligible count and misses tracking
        "integration_f1_eligible_count": len(eligible_items),
        # Backward compatibility (lax)
        "integration_f1_misses": integration_misses_count,
        "integration_f1_misses_lax": integration_misses_count_lax if eligible_results else 0,
        "integration_f1_misses_strict": integration_misses_count_strict if eligible_results else 0,
        "integration_f1_allowed_misses": allowed_integration_f1_misses,
        "long_context_count": long_context_count,
        "integration_coverage": integration_coverage,
        "integration_grounded_coverage": integration_grounded_coverage,
        "multi_call_parity_rate": multi_call_parity_rate,
        "integration_span_cap": integration_span_cap,
        "integration_spans_over_cap_count": integration_spans_over_cap_count,
        # Top 5 offenders
        "integration_spans_over_cap_items": integration_spans_over_cap_items[:5],
        # Histogram: count of items with N spans
        "integration_span_count_histogram": dict(sorted(integration_span_count_histogram.items())),
        "adversarial_taxonomy_ok": True,  # Will be updated below
        "adversarial_taxonomy_missing": [],  # Will be updated below
        "duplicate_sample_ids": [],  # Will be updated below
        "sharding": None,  # Will be updated below
        "problems": {},
        "stratification": stratification,
    }

    # Add per-tool deltas (computed after summary dict creation)
    summary["per_tool_deltas"] = per_tool_deltas(items, results)

    # Update summary with adversarial taxonomy and sharding info (will be computed below)
    # These are initialized above, will be updated after computation

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

    # Adversarial taxonomy enforcement: For N >= 30, require presence of key types
    adversarial_taxonomy_ok = True
    adversarial_taxonomy_missing = []
    if total >= 30 and args.check_adversarial:
        required_types = ["ambiguity", "malformed_json", "range_violation"]
        min_per_type = 2
        for adv_type in required_types:
            count = adversarial_counts.get(adv_type, 0)
            if count < min_per_type:
                adversarial_taxonomy_ok = False
                adversarial_taxonomy_missing.append({
                    "type": adv_type,
                    "required": min_per_type,
                    "actual": count,
                    "needed": min_per_type - count,
                })

    # Check for duplicate sample_ids (sharding validation)
    sample_ids_seen = {}
    duplicate_sample_ids = []
    for item_idx, item in enumerate(items):
        meta = item.get("metadata", {})
        sample_id = meta.get("sample_id")
        if sample_id:
            if sample_id in sample_ids_seen:
                duplicate_sample_ids.append({
                    "sample_id": sample_id,
                    "line_1": sample_ids_seen[sample_id],
                    "line_2": item_idx + 1,
                })
            else:
                sample_ids_seen[sample_id] = item_idx + 1

    if duplicate_sample_ids:
        print(
            f"[VERIFY] FAIL: Found {len(duplicate_sample_ids)} duplicate sample_ids (sharding violation)")
        for dup in duplicate_sample_ids[:5]:
            print(
                f"  - {dup['sample_id']}: lines {dup['line_1']} and {dup['line_2']}")
        gates_ok = False
        passed = False

    # Extract sharding info from dataset header
    sharding_info = None
    if dataset_header:
        sharding_info = {
            "num_shards": dataset_header.get("num_shards", 1),
            "shard_index": dataset_header.get("shard_index", 0),
        }

    # Update summary with computed values (adversarial taxonomy, sharding, duplicates)
    summary["adversarial_taxonomy_ok"] = adversarial_taxonomy_ok if 'adversarial_taxonomy_ok' in locals() else True
    summary["adversarial_taxonomy_missing"] = adversarial_taxonomy_missing if 'adversarial_taxonomy_missing' in locals() else [
    ]
    summary["duplicate_sample_ids"] = duplicate_sample_ids if 'duplicate_sample_ids' in locals() else [
    ]
    summary["sharding"] = sharding_info if 'sharding_info' in locals() else None

    # Compute dataset SHA256 (excluding header)
    dataset_sha256 = None
    try:
        with open(input_file, "rb") as f:
            content_lines = []
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line.decode("utf-8"))
                        # Skip header line
                        if item.get("__header__") is True:
                            continue
                        content_lines.append(line)
                    except json.JSONDecodeError:
                        pass
            content = b"".join(content_lines)
            dataset_sha256 = hashlib.sha256(content).hexdigest()
    except Exception as e:
        print(f"[VERIFY] WARN: Could not compute dataset SHA256: {e}")

    # Control contamination check: hard fail if controls have integration spans
    # (controls_with_integration already initialized above)
    control_contamination_items = []
    for item_idx, item in enumerate(items):
        meta = item.get("metadata", {})
        expected_behaviour = meta.get("expected_behaviour", "normal")
        if expected_behaviour in {"no_tool", "decline"}:
            integration_spans = meta.get("integration_spans_bytes", [])
            call_sequence = meta.get("call_sequence", [])
            if integration_spans or call_sequence:
                controls_with_integration += 1
                control_contamination_items.append({
                    "line": item_idx + 1,
                    "sample_id": meta.get("sample_id", f"item_{item_idx + 1}"),
                    "expected_behaviour": expected_behaviour,
                    "integration_spans_count": len(integration_spans),
                    "call_sequence_count": len(call_sequence),
                })

    # Define gates verbatim (thresholds & policies)
    gates = {
        "ok_rate": {"threshold": 0.95, "policy": "hard_fail"},
        "semantic_ok_rate": {"threshold": 0.98, "policy": "hard_fail"},
        "caws_header_ok_rate": {"threshold": 0.95, "policy": "hard_fail"},
        "privacy_ok_rate": {"threshold": 1.0, "policy": "hard_fail"},
        "integration_f1_macro_lax": {"threshold": 0.90, "policy": "count_based_misses", "misses_allowed_pct": 0.05, "min_eligible": args.min_eligible_for_gates},
        "integration_f1_macro_strict": {"threshold": 0.75, "policy": "warning_only"},
        "token_align_ok_rate": {"threshold": 0.98, "policy": "hard_fail"},
        "integration_coverage": {"threshold": 0.98, "policy": "count_based_misses", "misses_allowed_pct": 0.05},
        "integration_grounded_coverage": {"threshold": 0.97, "policy": "count_based_misses", "misses_allowed_pct": 0.05},
        "multi_call_parity_rate": {"threshold": 0.95, "policy": "count_based_misses", "misses_allowed_pct": 0.05},
        "long_context_quota": {"min_pct": 0.02, "max_pct": 0.05, "policy": "warning_only", "min_n": 20},
        "controls_with_integration": {"threshold": 0, "policy": "hard_fail"},
    }

    # Secondary tokenizer verification (if provided)
    results_t2 = None
    elig_t2 = None
    if secondary_tok:
        print("[VERIFY] Running verification with secondary tokenizer...")
        results_t2 = []
        for item in items:
            res_t2 = verify_item(
                item,
                reg,
                tokenizer=secondary_tok,
                check_stratification=args.check_stratification,
                check_controls=args.check_controls,
                check_adversarial=args.check_adversarial,
            )
            results_t2.append(res_t2)

        # Compute eligible items for secondary tokenizer
        elig_t2 = []
        for item, res_t2 in zip(items, results_t2):
            meta = item.get("metadata", {})
            call_sequence = meta.get("call_sequence", [])
            expected_behaviour = meta.get("expected_behaviour", "normal")
            if call_sequence and expected_behaviour not in {"no_tool", "decline"}:
                elig_t2.append(res_t2)

        # Add secondary tokenizer metrics to summary
        if elig_t2:
            summary["token_align_ok_rate_t2"] = round(
                sum(1 for r in results_t2 if r.get("token_align_ok") is True) /
                max(1, sum(1 for r in results_t2 if r.get(
                    "token_align_ok") is not None)), 3
            ) if any(r.get("token_align_ok") is not None for r in results_t2) else None
            summary["avg_integration_f1_macro_lax_t2"] = round(
                macro_f1(results_t2, mode="lax"), 3)
            summary["avg_integration_f1_micro_lax_t2"] = round(
                micro_f1(results_t2, mode="lax"), 3)
            summary["avg_integration_f1_macro_strict_t2"] = round(
                macro_f1(results_t2, mode="strict"), 3)
            summary["avg_integration_f1_micro_strict_t2"] = round(
                micro_f1(results_t2, mode="strict"), 3)
        else:
            summary["token_align_ok_rate_t2"] = None
            summary["avg_integration_f1_macro_lax_t2"] = None
            summary["avg_integration_f1_micro_lax_t2"] = None
            summary["avg_integration_f1_macro_strict_t2"] = None
            summary["avg_integration_f1_micro_strict_t2"] = None

    # Warn on large per-tool deltas
    per_tool = summary.get("per_tool_deltas", {})
    large_deltas = {t: v for t, v in per_tool.items() if v.get(
        "delta_lax_minus_strict", 0.0) > 0.2}
    if large_deltas:
        print("[VERIFY] WARN: Large lax→strict deltas detected per tool:")
        for t, v in sorted(large_deltas.items(), key=lambda kv: kv[1].get("delta_lax_minus_strict", 0.0), reverse=True):
            print(f"  - {t}: Δ={v.get('delta_lax_minus_strict', 0.0):.3f} (lax={v.get('f1_lax_macro', 0.0):.3f}, strict={v.get('f1_strict_macro', 0.0):.3f}, samples={v.get('sample_count', 0)})")

    # Write report with generation plan
    os.makedirs(os.path.dirname(args.report) if os.path.dirname(
        args.report) else ".", exist_ok=True)

    # Report header with hardening
    REPORT_VERSION = "1.0.0"
    report_header = {
        "report_version": REPORT_VERSION,
        "dataset_sha256": dataset_sha256,
        "num_items": total,
        "num_eligible": num_eligible,
        "num_controls": num_controls,
        "num_negative_controls": num_negative_controls if 'num_negative_controls' in locals() else 0,
        "gates": gates,
        "tokenizer_fingerprint": current_tokenizer_fp if 'current_tokenizer_fp' in locals() else None,
        "tool_registry_sha256": current_registry_sha256 if 'current_registry_sha256' in locals() else None,
        "integration_span_cap": integration_span_cap,
        "min_eligible_for_gates": args.min_eligible_for_gates,
        "fail_on_fingerprint_mismatch": args.fail_on_fingerprint_mismatch,
    }

    # Add secondary tokenizer fingerprint if provided
    if args.secondary_tokenizer:
        try:
            from scripts.generate_contextual_prompts import compute_tokenizer_fingerprint
            report_header["secondary_tokenizer_fingerprint"] = compute_tokenizer_fingerprint(
                args.secondary_tokenizer)
        except Exception as e:
            print(
                f"[VERIFY] WARN: Could not compute secondary tokenizer fingerprint: {e}")

    # Add next registry SHA256 if provided
    if next_reg:
        try:
            from scripts.generate_contextual_prompts import compute_registry_fingerprint
            report_header["next_registry_sha256"] = compute_registry_fingerprint(
                next_reg)
        except Exception as e:
            print(
                f"[VERIFY] WARN: Could not compute next registry fingerprint: {e}")

    report_data = {
        "header": report_header,
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
    print(
        f"[VERIFY] CAWS header OK rate: {summary['caws_header_ok_rate']:.3f}")
    print(f"[VERIFY] Privacy OK rate: {summary['privacy_ok_rate']:.3f}")
    if eligible_results:
        print(
            f"[VERIFY] Integration F1 (lax): macro={summary['avg_integration_f1_macro_lax']:.3f}, micro={summary['avg_integration_f1_micro_lax']:.3f}")
        print(
            f"[VERIFY] Integration F1 (strict): macro={summary['avg_integration_f1_macro_strict']:.3f}, micro={summary['avg_integration_f1_micro_strict']:.3f}")
        print(f"[VERIFY] Eligible items: {summary['integration_f1_eligible_count']}, misses (lax/strict): {summary['integration_f1_misses_lax']}/{summary['integration_f1_misses_strict']}, allowed: {summary['integration_f1_allowed_misses']}")
    else:
        print(
            f"[VERIFY] Avg integration F1: {summary['avg_integration_f1']:.3f}")

    if stratification:
        print(
            f"[VERIFY] Stratification: {len(stratification['missing_cells'])} missing cells")
        if not stratification["all_cells_populated"]:
            print("[VERIFY] WARN: Some stratification cells are missing!")

    # Adversarial taxonomy check
    if total >= 30 and args.check_adversarial:
        if not summary.get("adversarial_taxonomy_ok", True):
            print("[VERIFY] FAIL: Adversarial taxonomy minimums not met:")
            for missing in summary.get("adversarial_taxonomy_missing", []):
                print(
                    f"  - {missing['type']}: {missing['actual']}/{missing['required']} (need {missing['needed']} more)")
            gates_ok = False
            passed = False
        else:
            print("[VERIFY] Adversarial taxonomy: All required types present")

    # Sharding info
    if summary.get("sharding"):
        shard_info = summary["sharding"]
        print(
            f"[VERIFY] Sharding: {shard_info['num_shards']} shards, index {shard_info['shard_index']}")

    # Duplicate sample_id check (already printed above if found)

    # Check thresholds with hard stops
    passed = True
    gates_ok = True

    if summary["ok_rate"] < 0.95:
        print(f"[VERIFY] FAIL: OK rate {summary['ok_rate']:.3f} < 0.95")
        passed = False
        gates_ok = False
    if summary["semantic_ok_rate"] < 0.98:
        print(
            f"[VERIFY] FAIL: Semantic OK rate {summary['semantic_ok_rate']:.3f} < 0.98")
        passed = False
        gates_ok = False
    if summary["caws_header_ok_rate"] < 0.95:
        print(
            f"[VERIFY] FAIL: CAWS header OK rate {summary['caws_header_ok_rate']:.3f} < 0.95")
        passed = False
        gates_ok = False
    if summary["privacy_ok_rate"] < 1.0:
        print(
            f"[VERIFY] FAIL: Privacy OK rate {summary['privacy_ok_rate']:.3f} < 1.0")
        passed = False
        gates_ok = False
    # Control contamination check: hard fail if controls have integration spans (must run before gates)
    if controls_with_integration > 0:
        print(
            f"[VERIFY] FAIL: Control contamination detected: {controls_with_integration} control items have integration spans or tool calls")
        for item in control_contamination_items[:5]:
            print(
                f"  Line {item['line']} ({item['sample_id']}): {item['expected_behaviour']}, spans={item['integration_spans_count']}, calls={item['call_sequence_count']}")
        if len(control_contamination_items) > 5:
            print(f"  ... and {len(control_contamination_items) - 5} more")
        gates_ok = False
        passed = False

    # Integration F1 check (macro-averaged, eligible-only, count-based misses) - gate on lax mode
    # Small-N gate policy: require min_eligible_for_gates items
    if eligible_results:
        f1_threshold = 0.90
        # Check min-eligible floor
        if num_eligible < args.min_eligible_for_gates:
            print(
                f"[VERIFY] INCONCLUSIVE: Eligible items ({num_eligible}) < min_eligible_for_gates ({args.min_eligible_for_gates})")
            print("[VERIFY] Skipping F1 gates due to insufficient eligible items")
            # Mark as inconclusive but don't fail
            summary["inconclusive"] = True
            summary["inconclusive_reason"] = f"eligible_items ({num_eligible}) < min_eligible_for_gates ({args.min_eligible_for_gates})"
        # Gate on lax macro-F1 OR misses ≤ allowed (only if we have enough eligible items)
        elif avg_integration_f1_macro_lax < f1_threshold and integration_misses_count_lax > allowed_integration_f1_misses:
            print(
                f"[VERIFY] FAIL: Integration F1 (lax): macro={avg_integration_f1_macro_lax:.3f} < {f1_threshold} (over {len(eligible_items)} eligible items, misses={integration_misses_count_lax}, allowed={allowed_integration_f1_misses})")
            passed = False
            gates_ok = False

        # Secondary tokenizer gates (optional)
        if secondary_tok and elig_t2 and summary.get("avg_integration_f1_macro_lax_t2") is not None:
            if summary["avg_integration_f1_macro_lax_t2"] < f1_threshold:
                print(
                    f"[VERIFY] FAIL: Secondary tokenizer lax F1 below threshold: {summary['avg_integration_f1_macro_lax_t2']:.3f} < {f1_threshold}")
                gates_ok = False
            if summary.get("token_align_ok_rate_t2") is not None and summary["token_align_ok_rate_t2"] < 0.98:
                print(
                    f"[VERIFY] FAIL: Secondary tokenizer token alignment rate below threshold: {summary['token_align_ok_rate_t2']:.3f} < 0.98")
                gates_ok = False
            # Print top offenders (lax mode)
            low_f1_items = [(i, r) for i, r in zip(eligible_items, eligible_results) if r.get(
                "integration_f1_lax", r.get("integration_f1", 0.0)) < 0.75]
            for item, result in low_f1_items[:3]:
                meta = item.get("metadata", {})
                sample_id = meta.get("sample_id", "unknown")
                fields = meta.get("tool_result_fields", {})
                spans = meta.get("integration_spans_bytes", [])
                f1_lax = result.get("integration_f1_lax",
                                    result.get("integration_f1", 0.0))
                print(
                    f"  - {sample_id}: F1_lax={f1_lax:.2f}, fields={list(fields.keys())}, spans={len(spans)}")
                if spans and len(spans[0]) >= 2:
                    teacher_text = item.get("teacher_text", "")
                    span_text = teacher_text[spans[0][0]:spans[0][1]]
                    print(f"    First span: \"{span_text[:80]}...\"")

        # Warn on strict failures (but don't gate)
        if avg_integration_f1_macro_strict < 0.75 or integration_misses_count_strict > allowed_integration_f1_misses:
            print(
                f"[VERIFY] WARN: Integration F1 (strict): macro={avg_integration_f1_macro_strict:.3f}, misses={integration_misses_count_strict} (strict mode is trending metric, not gated)")
            # Print top strict offenders
            low_f1_strict_items = [(i, r) for i, r in zip(
                eligible_items, eligible_results) if r.get("integration_f1_strict", 0.0) < 0.75]
            for item, result in low_f1_strict_items[:5]:
                meta = item.get("metadata", {})
                sample_id = meta.get("sample_id", "unknown")
                fields = meta.get("tool_result_fields", {})
                spans = meta.get("integration_spans_bytes", [])
                f1_strict = result.get("integration_f1_strict", 0.0)
                f1_lax = result.get("integration_f1_lax",
                                    result.get("integration_f1", 0.0))
                print(
                    f"  - {sample_id}: F1_strict={f1_strict:.2f} (F1_lax={f1_lax:.2f}), fields={list(fields.keys())}, spans={len(spans)}")
                # Show diff: expected fields vs found text
                if spans and len(spans[0]) >= 2:
                    teacher_text = item.get("teacher_text", "")
                    span_text = teacher_text[spans[0][0]:spans[0][1]]
                    expected_keys = [k for k in ["summary",
                                                 "lines", "count", "top_k"] if k in fields]
                    print(f"    Expected keyed fields: {expected_keys}")
                    print(f"    Found span text: \"{span_text[:100]}...\"")
    # Stratification check (relaxed for N<36)
    if stratification:
        # For N < 36, check backbone only
        if total < 36:
            backbone_ok, backbone_missing = check_stratification_backbone(
                items, total)
            if not backbone_ok:
                print(
                    f"[VERIFY] FAIL: Stratification backbone missing: {backbone_missing}")
                passed = False
                gates_ok = False
            elif len(stratification['missing_cells']) > 0:
                print(
                    f"[VERIFY] WARN: Stratification gaps (N={total} < 36): {len(stratification['missing_cells'])} cells missing (backbone OK)")
        else:
            # Full check for N >= 36
            if not stratification["all_cells_populated"]:
                print("[VERIFY] FAIL: Stratification cells not fully populated")
                passed = False
                gates_ok = False

    # Long-context quota check
    if total >= 20 and not (2 <= long_context_count <= 3):
        print(
            f"[VERIFY] FAIL: Long-context quota: count={long_context_count}, expected 2-3 for N≥20")
        passed = False
        gates_ok = False

    # Integration span cap check (soft gate - warning unless it harms F1)
    if integration_spans_over_cap_count > 0:
        print(
            f"[VERIFY] WARN: Integration span cap exceeded: {integration_spans_over_cap_count} items exceed cap of {integration_span_cap}")
        for offender in integration_spans_over_cap_items[:5]:
            print(
                f"  - Line {offender['line']} ({offender['sample_id']}): {offender['span_count']} spans (cap: {offender['cap']})")
        # Only fail if it significantly harms F1 (more than 10% of eligible items)
        if eligible_results and integration_spans_over_cap_count > len(eligible_items) * 0.1:
            print(
                f"[VERIFY] FAIL: Too many items exceed integration span cap ({integration_spans_over_cap_count} > {len(eligible_items) * 0.1:.0f})")
            passed = False
            gates_ok = False

    # Integration coverage check (count-based)
    if integration_total > 0:
        if integration_misses_count > allowed_integration_misses:
            print(
                f"[VERIFY] FAIL: Integration coverage: {integration_ok}/{integration_total} (allowed_misses={allowed_integration_misses}, misses={integration_misses_count})")
            passed = False
            gates_ok = False
            # Print misses
            for miss in integration_misses[:3]:
                print(
                    f"  - line {miss['line']} ({miss['sample_id']}): missing integration_spans_bytes")

    # Grounding check (count-based)
    if integration_grounded_total > 0:
        if grounding_misses_count > allowed_grounding_misses:
            print(
                f"[VERIFY] FAIL: Integration grounded: {integration_grounded_ok}/{integration_grounded_total} (allowed_misses={allowed_grounding_misses}, misses={grounding_misses_count})")
            passed = False
            gates_ok = False
            # Print misses with details
            for miss in grounding_misses[:3]:
                print(
                    f"  - line {miss['line']} ({miss['sample_id']}): integration_not_grounded")
                print(
                    f"    tool_result_fields.summary = \"{miss.get('summary', 'N/A')}\"")
                if miss.get('spans') and len(miss['spans']) > 0 and len(miss['spans'][0]) >= 2:
                    # Get teacher text for this item
                    item_idx = miss['line'] - 1
                    if item_idx < len(items):
                        teacher_text = items[item_idx].get("teacher_text", "")
                        span_text = teacher_text[miss['spans']
                                                 [0][0]:miss['spans'][0][1]]
                        print(f"    spans[0] = \"{span_text[:100]}...\"")

    # Multi-call parity check (count-based)
    if multi_call_parity_total > 0:
        if multi_call_misses_count > allowed_multi_call_misses:
            print(
                f"[VERIFY] FAIL: Multi-call span parity: {multi_call_parity_ok}/{multi_call_parity_total} (allowed_misses={allowed_multi_call_misses}, misses={multi_call_misses_count})")
            passed = False
            gates_ok = False
            for miss in multi_call_parity_misses[:3]:
                print(
                    f"  - line {miss['line']} ({miss['sample_id']}): expected {miss['expected']} spans, got {miss['got']}")

    # Adversarial quota check
    required_adversarial_types = {
        "range_violation", "malformed_json", "ambiguity"}
    if total >= 30 and not summary.get("adversarial_quota_ok", True):
        missing = required_adversarial_types - \
            set(summary.get("adversarial_by_type", {}).keys())
        print(f"[VERIFY] FAIL: Adversarial quota missing types: {missing}")
        passed = False
        gates_ok = False

    # CI-friendly failure surfacing: print first 5 failures with diffs
    if failing_items:
        print("\n[VERIFY] First 5 failing items:")
        for fail_item in failing_items[:5]:
            line_num = fail_item.get("line", "?")
            problems = fail_item.get(
                "problems", [fail_item.get("problem", "unknown")])
            sample_id = fail_item.get("sample_id", f"line_{line_num}")
            print(f"  Line {line_num} ({sample_id}): {', '.join(problems)}")
            if "expected" in fail_item:
                print(
                    f"    Expected: {fail_item['expected']}, Found: {fail_item.get('found', 'N/A')}")

    # Performance budget check
    verification_time_sec = time.time() - verification_start_time
    time_per_100_items = (verification_time_sec / total) * \
        100 if total > 0 else 0.0

    if time_per_100_items > args.perf_budget_sec_per_100:
        print(
            f"[VERIFY] WARN: Performance budget exceeded: {time_per_100_items:.2f}s per 100 items (budget: {args.perf_budget_sec_per_100}s)")
        # Find top regex offenders (longest Integration spans)
        top_offenders = []
        for item in items:
            meta = item.get("metadata", {})
            integration_spans = meta.get("integration_spans_bytes", [])
            if integration_spans:
                teacher_text = item.get("teacher_text", "")
                max_span_length = 0
                for span in integration_spans:
                    if len(span) >= 2:
                        span_length = span[1] - span[0]
                        max_span_length = max(max_span_length, span_length)
                if max_span_length > 0:
                    top_offenders.append({
                        "sample_id": meta.get("sample_id", "unknown"),
                        "max_span_length": max_span_length,
                        "span_count": len(integration_spans),
                    })
        top_offenders.sort(key=lambda x: x["max_span_length"], reverse=True)
        print("[VERIFY] Top 5 regex offenders (longest Integration spans):")
        for offender in top_offenders[:5]:
            print(
                f"  - {offender['sample_id']}: {offender['max_span_length']} bytes, {offender['span_count']} spans")

    # Add performance metrics to summary
    summary["verification_time_sec"] = round(verification_time_sec, 3)
    summary["time_per_100_items"] = round(time_per_100_items, 3)
    summary["perf_budget_sec_per_100"] = args.perf_budget_sec_per_100
    summary["perf_budget_exceeded"] = time_per_100_items > args.perf_budget_sec_per_100

    if passed:
        print("[VERIFY] PASS: All checks passed")
    else:
        print("[VERIFY] FAIL: Some checks failed")

    return 0 if gates_ok else 1


if __name__ == "__main__":
    exit(main())
