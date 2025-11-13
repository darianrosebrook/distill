"""
MCP Code-Mode Data Generator

Generates training examples that prefer TypeScript API orchestration
over direct MCP tool calls for eligible scenarios.

Reference: code-mode-latent-reasoning.md Milestone 1

Author: @darianrosebrook
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
import random
import re


def make_code_mode_example(
    servers: List[str],
    task: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate a code-mode example with TypeScript orchestration.

    Args:
        servers: List of server names (e.g., ["google-drive", "salesforce"])
        task: Task dict with "instruction" and optional metadata

    Returns:
        Example dict with prompt, teacher_targets, and metadata
    """
    instruction = task.get("instruction", "")
    tool_count = len(servers)

    # Generate TypeScript imports
    imports = []
    for server in servers:
        imports.append(f"import * as {server.replace('-', '_')} from './servers/{server}';")

    # Generate progressive discovery pattern (search before use)
    # Example: First discover available documents, then fetch specific one
    ts_code_lines = []

    # Progressive discovery: search/list before accessing specific resource
    if "google-drive" in servers:
        ts_code_lines.append("// Discover available documents")
        ts_code_lines.append("const files = await google_drive.listFiles({ folderId: 'root' });")
        ts_code_lines.append("const docId = files.items[0].id;")
        ts_code_lines.append("const doc = await google_drive.getDocument({ documentId: docId });")

    if "salesforce" in servers:
        ts_code_lines.append("// Query records before updating")
        ts_code_lines.append(
            "const records = await salesforce.queryRecords({ objectType: 'SalesMeeting', limit: 10 });"
        )
        ts_code_lines.append("const recordId = records[0].Id;")

    # Process large data in sandbox (not echoed to tokens)
    if tool_count >= 2:
        ts_code_lines.append("// Process large data in sandbox")
        ts_code_lines.append("const notes = summarize(doc.content, 5);")
        ts_code_lines.append("// Large content stays in sandbox, only summary returned")

    # Update records with processed data
    if "salesforce" in servers:
        ts_code_lines.append("await salesforce.updateRecord({")
        ts_code_lines.append("  objectType: 'SalesMeeting',")
        ts_code_lines.append("  recordId: recordId,")
        ts_code_lines.append("  data: { Notes: notes }")
        ts_code_lines.append("});")

    # Log summary only (not full payloads)
    ts_code_lines.append("console.log('Task completed successfully');")

    # Combine into teacher targets
    teacher_targets = "\n".join(imports + [""] + ts_code_lines)

    # Calculate intermediate sizes (simulate large payloads)
    intermediate_sizes = []
    if "google-drive" in servers:
        # Simulate large document (50k chars)
        intermediate_sizes.append(50000)
    if tool_count >= 2:
        # Additional intermediate processing
        intermediate_sizes.append(15000)

    # Check for PII (from task metadata if available)
    pii_tags_present = task.get("metadata", {}).get("pii_tags_present", False)

    # Determine eligibility
    eligible_for_code_mode = (
        tool_count >= 2 or max(intermediate_sizes) >= 10000
        if intermediate_sizes
        else False or pii_tags_present
    )

    # Compute span targets for token-level loss (requires tokenizer at generation time)
    # For now, emit placeholder structure; actual spans computed during dataset loading
    span_targets = {
        "ts_mode_spans": [],  # Will be populated with token positions for TS orchestration markers
        "direct_tool_spans": [],  # Will be populated with token positions for direct tool calls (if any)
    }

    return {
        "prompt": instruction,
        "teacher_targets": [teacher_targets],
        "metadata": {
            "tool_count": tool_count,
            "intermediate_sizes": intermediate_sizes,
            "pii_tags_present": pii_tags_present,
            "eligible_for_code_mode": eligible_for_code_mode,
            "servers": servers,
            "span_targets": span_targets,  # Placeholder for token-level spans
        },
    }


def compute_span_targets_from_tokenized(
    text: str,
    tokenizer,
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Compute span targets from tokenized text using tokenizer offset_mapping.

    Identifies token positions for TS orchestration markers and direct tool calls.
    Uses precise tokenizer offset_mapping when available for accurate token positions.

    Args:
        text: Teacher target text
        tokenizer: Tokenizer instance (should support encode_plus with return_offsets_mapping)

    Returns:
        Dict with 'ts_mode_spans' and 'direct_tool_spans' lists of (start, end) token positions
    """
    ts_mode_spans = []
    direct_tool_spans = []

    # TS orchestration markers to find (exact patterns)
    ts_markers = [
        r"\bimport\b",
        r"\bfrom\b",
        r"\bcallMCPTool\b",
        r"\bawait\b",
        r"['\"]\.\/servers",
    ]

    # Direct tool call markers
    direct_markers = [
        r"<\|tool_call\|>",
        r"<\|tool_result\|>",
        r"\btool_call\b",
        r"\btool_result\b",
    ]

    # Try to use offset_mapping for precise token positions
    try:
        # Use encode_plus with return_offsets_mapping if available
        if hasattr(tokenizer, "encode_plus"):
            encoding = tokenizer.encode_plus(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
                return_tensors=None,
            )
            encoding["input_ids"]
            offset_mapping = encoding.get("offset_mapping", None)

            if offset_mapping:
                # Use precise offset mapping to find token positions
                text_lower = text.lower()

                # Find TS mode spans
                for marker_pattern in ts_markers:
                    for match in re.finditer(marker_pattern, text_lower, re.IGNORECASE):
                        start_char = match.start()
                        end_char = match.end()

                        # Find tokens that overlap with this character range
                        start_token = None
                        end_token = None

                        for token_idx, (tok_start, tok_end) in enumerate(offset_mapping):
                            if tok_start is None or tok_end is None:
                                continue

                            # Token overlaps with marker span
                            if tok_start <= end_char and tok_end >= start_char:
                                if start_token is None:
                                    start_token = token_idx
                                end_token = token_idx + 1

                        if start_token is not None and end_token is not None:
                            ts_mode_spans.append((start_token, end_token))

                # Find direct tool call spans
                for marker_pattern in direct_markers:
                    for match in re.finditer(marker_pattern, text_lower, re.IGNORECASE):
                        start_char = match.start()
                        end_char = match.end()

                        start_token = None
                        end_token = None

                        for token_idx, (tok_start, tok_end) in enumerate(offset_mapping):
                            if tok_start is None or tok_end is None:
                                continue

                            if tok_start <= end_char and tok_end >= start_char:
                                if start_token is None:
                                    start_token = token_idx
                                end_token = token_idx + 1

                        if start_token is not None and end_token is not None:
                            direct_tool_spans.append((start_token, end_token))

                return {
                    "ts_mode_spans": ts_mode_spans,
                    "direct_tool_spans": direct_tool_spans,
                }
    except Exception:
        pass

    # Fallback: approximate using character positions
    try:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        num_tokens = len(encoded)
        text_len = len(text)

        # Rough approximation: assume ~4 chars per token
        chars_per_token = text_len / max(1, num_tokens)

        text_lower = text.lower()

        # Find TS mode spans (approximate)
        for marker_pattern in ts_markers:
            for match in re.finditer(marker_pattern, text_lower, re.IGNORECASE):
                start_char = match.start()
                end_char = match.end()

                start_token = max(0, int(start_char / chars_per_token))
                end_token = min(num_tokens, int(end_char / chars_per_token) + 1)

                if end_token > start_token:
                    ts_mode_spans.append((start_token, end_token))

        # Find direct tool call spans (approximate)
        for marker_pattern in direct_markers:
            for match in re.finditer(marker_pattern, text_lower, re.IGNORECASE):
                start_char = match.start()
                end_char = match.end()

                start_token = max(0, int(start_char / chars_per_token))
                end_token = min(num_tokens, int(end_char / chars_per_token) + 1)

                if end_token > start_token:
                    direct_tool_spans.append((start_token, end_token))
    except Exception:
        # Final fallback: return empty spans
        pass

    return {
        "ts_mode_spans": ts_mode_spans,
        "direct_tool_spans": direct_tool_spans,
    }


# Alias for backward compatibility
compute_span_targets_from_text = compute_span_targets_from_tokenized


def generate_code_mode_dataset(
    num_examples: int = 100,
    servers: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate a dataset of code-mode examples.

    Args:
        num_examples: Number of examples to generate
        servers: Optional list of server names (defaults to common ones)

    Returns:
        List of example dicts
    """
    if servers is None:
        servers = ["google-drive", "salesforce", "slack"]

    examples = []

    # Task templates for different scenarios
    task_templates = [
        {
            "instruction": "Fetch a document from Google Drive, summarize it, and update a Salesforce record with the summary.",
            "metadata": {"pii_tags_present": False},
        },
        {
            "instruction": "List files from Google Drive, process the largest document, and send a summary to Slack.",
            "metadata": {"pii_tags_present": False},
        },
        {
            "instruction": "Query Salesforce records, fetch related documents from Google Drive, and update records with processed information.",
            "metadata": {"pii_tags_present": True},  # Simulate PII scenario
        },
    ]

    for i in range(num_examples):
        # Select random servers (2-3 for multi-tool scenarios)
        num_servers = random.randint(2, min(3, len(servers)))
        selected_servers = random.sample(servers, num_servers)

        # Select random task template
        task_template = random.choice(task_templates)

        # Generate example
        example = make_code_mode_example(
            servers=selected_servers,
            task=task_template,
        )

        examples.append(example)

    return examples
