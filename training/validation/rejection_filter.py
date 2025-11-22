"""Rejection filter for dataset samples.

This module provides rules-based filtering to detect and reject problematic
samples before training, including:
- Hallucinated tool names
- Malformed JSON
- Inconsistent CAWS clauses
- Reasoning contamination in supervision targets

Author: @darianrosebrook
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple


class RejectionFilter:
    """Rules-based filter for detecting problematic samples."""
    
    def __init__(
        self,
        reject_hallucinated_tools: bool = True,
        reject_malformed_json: bool = True,
        reject_inconsistent_caws: bool = True,
        reject_contaminated_supervision: bool = True,
        known_tool_names: Optional[Set[str]] = None,
    ):
        """
        Initialize rejection filter.
        
        Args:
            reject_hallucinated_tools: Reject samples with unknown tool names
            reject_malformed_json: Reject samples with invalid JSON in tool calls
            reject_inconsistent_caws: Reject samples with inconsistent CAWS clauses
            reject_contaminated_supervision: Reject samples with contaminated supervision
            known_tool_names: Set of known valid tool names (if None, uses default set)
        """
        self.reject_hallucinated_tools = reject_hallucinated_tools
        self.reject_malformed_json = reject_malformed_json
        self.reject_inconsistent_caws = reject_inconsistent_caws
        self.reject_contaminated_supervision = reject_contaminated_supervision
        
        # Default known tool names (can be extended)
        if known_tool_names is None:
            self.known_tool_names = {
                "codebase_search",
                "read_file",
                "write",
                "search_replace",
                "grep",
                "run_terminal_cmd",
                "list_dir",
                "glob_file_search",
                "delete_file",
                "mcp_cursor-ide-browser_browser_navigate",
                "mcp_cursor-ide-browser_browser_snapshot",
                "mcp_cursor-ide-browser_browser_click",
                "mcp_cursor-ide-browser_browser_type",
            }
        else:
            self.known_tool_names = known_tool_names
    
    def check_hallucinated_tools(self, sample: Dict[str, Any]) -> Optional[str]:
        """
        Check for hallucinated tool names.
        
        Returns:
            Rejection reason if found, None otherwise
        """
        if not self.reject_hallucinated_tools:
            return None
        
        tool_calls = sample.get("tool_calls", [])
        if not tool_calls:
            return None
        
        if not isinstance(tool_calls, list):
            return "tool_calls is not a list"
        
        for i, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                return f"tool_calls[{i}] is not a dict"
            
            tool_name = tool_call.get("name")
            if not tool_name:
                continue
            
            if tool_name not in self.known_tool_names:
                return f"Unknown tool name: {tool_name}"
        
        return None
    
    def check_malformed_json(self, sample: Dict[str, Any]) -> Optional[str]:
        """
        Check for malformed JSON in tool call arguments.
        
        Returns:
            Rejection reason if found, None otherwise
        """
        if not self.reject_malformed_json:
            return None
        
        tool_calls = sample.get("tool_calls", [])
        if not tool_calls:
            return None
        
        if not isinstance(tool_calls, list):
            return None
        
        for i, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                continue
            
            arguments = tool_call.get("arguments")
            if arguments is None:
                continue
            
            # If arguments is a string, try to parse as JSON
            if isinstance(arguments, str):
                try:
                    json.loads(arguments)
                except json.JSONDecodeError:
                    return f"Malformed JSON in tool_calls[{i}].arguments"
            
            # If arguments is a dict, validate it's well-formed
            elif isinstance(arguments, dict):
                try:
                    json.dumps(arguments)
                except (TypeError, ValueError):
                    return f"Invalid arguments dict in tool_calls[{i}]"
        
        # Check process supervision JSON arrays
        process_supervision = sample.get("process_supervision")
        if isinstance(process_supervision, dict):
            for field in ["tool_name_ids", "gold_json_text_ids", "integration_mask"]:
                value = process_supervision.get(field)
                if value is not None and not isinstance(value, list):
                    return f"process_supervision.{field} is not a list"
                if isinstance(value, list):
                    # Check all elements are valid types
                    if field in ["tool_name_ids", "gold_json_text_ids"]:
                        if not all(isinstance(x, int) for x in value):
                            return f"process_supervision.{field} contains non-integer values"
                    elif field == "integration_mask":
                        if not all(isinstance(x, (int, bool)) and (x == 0 or x == 1) for x in value):
                            return f"process_supervision.{field} contains invalid mask values"
        
        return None
    
    def check_inconsistent_caws(self, sample: Dict[str, Any]) -> Optional[str]:
        """
        Check for inconsistent CAWS clauses or context.
        
        Returns:
            Rejection reason if found, None otherwise
        """
        if not self.reject_inconsistent_caws:
            return None
        
        caws_level = sample.get("metadata", {}).get("caws_level")
        if not isinstance(caws_level, str):
            caws_level = sample.get("caws_level")
        
        caws_context = sample.get("caws_context")
        working_spec = sample.get("working_spec")
        
        # If caws_level is not "none", should have CAWS context
        if caws_level and caws_level != "none" and caws_level != 0:
            if not caws_context and not working_spec:
                return f"caws_level={caws_level} but no CAWS context present"
        
        # If CAWS context exists, should have valid working_spec
        if caws_context:
            if not isinstance(caws_context, dict):
                return "caws_context is not a dict"
            
            ws = caws_context.get("working_spec")
            if ws and isinstance(ws, dict):
                # Check required fields
                if "risk_tier" in ws:
                    risk_tier = ws["risk_tier"]
                    if not isinstance(risk_tier, int) or risk_tier < 1 or risk_tier > 3:
                        return f"Invalid risk_tier: {risk_tier} (must be 1-3)"
        
        # For Judge samples, check clause consistency
        if "a" in sample and "b" in sample:
            a_clauses = sample.get("a", {}).get("clauses", [])
            b_clauses = sample.get("b", {}).get("clauses", [])
            
            if isinstance(a_clauses, list) and isinstance(b_clauses, list):
                # Check for obviously inconsistent clause sets
                # (e.g., both have BUDGET_VIOLATION but winner is "a")
                winner = sample.get("winner")
                if winner == "a" and isinstance(a_clauses, list):
                    # Winner "a" should not have obvious violations
                    violation_clauses = {"BUDGET_VIOLATION", "SCOPE_VIOLATION", "GATE_FAILURE"}
                    a_clause_set = set(a_clauses) if isinstance(a_clauses, list) else set()
                    if a_clause_set & violation_clauses:
                        # This might be okay if "b" has worse violations
                        # But flag for manual review
                        pass
        
        return None
    
    def check_contaminated_supervision(self, sample: Dict[str, Any]) -> Optional[str]:
        """
        Check for reasoning contamination in supervision targets.
        
        This checks if process supervision targets contain reasoning
        that shouldn't be there (e.g., explanations mixed with tool calls).
        
        Returns:
            Rejection reason if found, None otherwise
        """
        if not self.reject_contaminated_supervision:
            return None
        
        process_supervision = sample.get("process_supervision")
        if not isinstance(process_supervision, dict):
            return None
        
        teacher_text = sample.get("teacher_text", "")
        if not teacher_text:
            return None
        
        # Check if tool_name_ids point to actual tool names
        tool_name_ids = process_supervision.get("tool_name_ids", [])
        if tool_name_ids:
            # This is a heuristic: if tool_name_ids exist but teacher_text
            # doesn't contain obvious tool call patterns, might be contaminated
            tool_call_patterns = [
                r'"name"\s*:\s*"[^"]+"',
                r"tool_name",
                r"function_call",
            ]
            
            has_tool_pattern = any(re.search(pattern, teacher_text) for pattern in tool_call_patterns)
            if not has_tool_pattern and len(tool_name_ids) > 0:
                # Might be contaminated, but not necessarily wrong
                # Only flag if very suspicious
                pass
        
        # Check if integration_mask length matches gold_json_text_ids
        gold_json_ids = process_supervision.get("gold_json_text_ids", [])
        integration_mask = process_supervision.get("integration_mask", [])
        
        if gold_json_ids and integration_mask:
            if len(gold_json_ids) != len(integration_mask):
                return f"integration_mask length ({len(integration_mask)}) != gold_json_text_ids length ({len(gold_json_ids)})"
        
        return None
    
    def filter_sample(self, sample: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Filter a single sample.
        
        Returns:
            Tuple of (should_reject, reason_if_rejected)
        """
        # Check hallucinated tools
        reason = self.check_hallucinated_tools(sample)
        if reason:
            return True, f"hallucinated_tools: {reason}"
        
        # Check malformed JSON
        reason = self.check_malformed_json(sample)
        if reason:
            return True, f"malformed_json: {reason}"
        
        # Check inconsistent CAWS
        reason = self.check_inconsistent_caws(sample)
        if reason:
            return True, f"inconsistent_caws: {reason}"
        
        # Check contaminated supervision
        reason = self.check_contaminated_supervision(sample)
        if reason:
            return True, f"contaminated_supervision: {reason}"
        
        return False, None
    
    def filter_samples(
        self,
        samples: List[Dict[str, Any]],
        mark_rejected: bool = True,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Filter a list of samples.
        
        Args:
            samples: List of sample dictionaries
            mark_rejected: If True, mark rejected samples with rejection_reason instead of removing
            
        Returns:
            Tuple of (accepted_samples, rejected_samples)
        """
        accepted = []
        rejected = []
        
        for sample in samples:
            should_reject, reason = self.filter_sample(sample)
            
            if should_reject:
                if mark_rejected:
                    # Mark with rejection reason
                    if "metadata" not in sample:
                        sample["metadata"] = {}
                    sample["metadata"]["rejection_reason"] = reason
                    sample["metadata"]["rejected"] = True
                    # Still add to rejected list for tracking
                    rejected.append(sample)
                else:
                    # Remove entirely
                    rejected.append(sample)
            else:
                accepted.append(sample)
        
        return accepted, rejected


def filter_dataset(
    input_path: str,
    output_path: str,
    role: str = "worker",
    mark_rejected: bool = True,
    known_tool_names: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Filter a dataset file, removing or marking rejected samples.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        role: Dataset role
        mark_rejected: If True, mark rejected samples instead of removing
        known_tool_names: Set of known valid tool names
        
    Returns:
        Statistics dictionary with counts
    """
    import json
    from pathlib import Path
    
    filter_obj = RejectionFilter(known_tool_names=known_tool_names)
    
    input_p = Path(input_path)
    output_p = Path(output_path)
    
    accepted_samples = []
    rejected_samples = []
    
    with input_p.open() as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            if line.startswith('{"__header__'):
                # Preserve headers
                if not mark_rejected:
                    accepted_samples.append(line)
                continue
            
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                # Invalid JSON - reject
                rejected_samples.append({"raw_line": line, "reason": "invalid_json"})
                continue
            
            should_reject, reason = filter_obj.filter_sample(sample)
            
            if should_reject:
                if mark_rejected:
                    # Mark and keep
                    if "metadata" not in sample:
                        sample["metadata"] = {}
                    sample["metadata"]["rejection_reason"] = reason
                    sample["metadata"]["rejected"] = True
                    rejected_samples.append(sample)
                    # Still write to output if marking
                    accepted_samples.append(sample)
                else:
                    rejected_samples.append(sample)
            else:
                accepted_samples.append(sample)
    
    # Write output
    with output_p.open("w") as fout:
        for sample in accepted_samples:
            if isinstance(sample, str):
                fout.write(sample + "\n")
            else:
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    stats = {
        "total": len(accepted_samples) + len(rejected_samples),
        "accepted": len(accepted_samples),
        "rejected": len(rejected_samples),
        "rejection_rate": len(rejected_samples) / (len(accepted_samples) + len(rejected_samples)) if (accepted_samples or rejected_samples) else 0.0,
    }
    
    return stats


