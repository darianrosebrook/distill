"""
Extractors for process-step supervision targets.

Extracts structured targets from teacher outputs for process-step supervision:
- Tool name spans
- JSON argument spans
- Post-tool integration spans

These extractors avoid training on reasoning_content prose while still
supervising the decision-making process.
"""
import json
import re
from typing import Optional, List, Dict, Tuple, Any


def extract_tool_call(text: str, tool_names: Optional[List[str]] = None) -> Optional[Dict]:
    """
    Extract tool call from text.

    Args:
        text: Text containing tool call
        tool_names: Optional list of valid tool names

    Returns:
        Dict with 'name' and 'arguments' if found, None otherwise
    """
    # Look for JSON tool call
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text)

    for match in matches:
        try:
            obj = json.loads(match)
            if isinstance(obj, dict) and 'name' in obj:
                return obj
        except:
            continue

    # Try parsing entire text
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict) and 'name' in obj:
            return obj
    except:
        pass

    return None


def extract_tool_name_span(
    teacher_text: str,
    tool_names: Optional[List[str]] = None
) -> Optional[Tuple[int, int]]:
    """
    Extract tool name token span from teacher output.

    Args:
        teacher_text: Teacher's output text
        tool_names: Optional list of valid tool names

    Returns:
        (start_char_idx, end_char_idx) if found, None otherwise
    """
    tool_call = extract_tool_call(teacher_text, tool_names)
    if tool_call:
        tool_name = tool_call.get('name', '')
        if tool_name:
            # Find tool name in text
            # Look for pattern: "name": "tool_name"
            pattern = rf'"name"\s*:\s*"{re.escape(tool_name)}"'
            match = re.search(pattern, teacher_text)
            if match:
                # Find the actual tool name span (inside quotes)
                tool_name_pattern = rf'"{re.escape(tool_name)}"'
                tool_match = re.search(tool_name_pattern, teacher_text)
                if tool_match:
                    return (tool_match.start(), tool_match.end())

    return None


def extract_json_argument_spans(teacher_text: str) -> List[Tuple[int, int]]:
    """
    Extract JSON argument token spans from teacher output.

    Args:
        teacher_text: Teacher's output text

    Returns:
        List of (start_char_idx, end_char_idx) tuples for JSON spans
    """
    spans = []

    # Find JSON objects
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    for match in re.finditer(json_pattern, teacher_text):
        try:
            # Validate it's valid JSON
            json.loads(match.group())
            spans.append((match.start(), match.end()))
        except:
            continue

    return spans


def identify_integration_spans(
    teacher_text: str,
    tool_results: Optional[List[Dict]] = None
) -> List[Tuple[int, int]]:
    """
    Identify post-tool integration spans where teacher integrates tool results.

    Strategy:
    1. Look for citations/references to tool outputs
    2. Identify spans that copy tool result fields
    3. Use simple heuristics for now (can be improved)

    Args:
        teacher_text: Teacher's output text
        tool_results: Optional list of tool result dictionaries

    Returns:
        List of (start_char_idx, end_char_idx) tuples for integration spans
    """
    spans = []

    # Pattern 1: Citations like "According to the search results..."
    citation_patterns = [
        r'According to [^.]*\.',
        r'Based on [^.]*\.',
        r'The [^.]* shows [^.]*\.',
        r'From [^.]*:',
    ]

    for pattern in citation_patterns:
        for match in re.finditer(pattern, teacher_text, re.IGNORECASE):
            spans.append((match.start(), match.end()))

    # Pattern 2: Direct copying of tool result fields
    # Look for quoted strings that might be from tool results
    if tool_results:
        for result in tool_results:
            if isinstance(result, dict):
                # Look for result field values in text
                for key, value in result.items():
                    if isinstance(value, str) and len(value) > 10:
                        # Search for value in text
                        escaped_value = re.escape(value[:50])  # First 50 chars
                        for match in re.finditer(escaped_value, teacher_text):
                            spans.append((match.start(), match.end()))

    # Remove overlapping spans (keep longest)
    if spans:
        spans = _merge_overlapping_spans(spans)

    return spans


def _merge_overlapping_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping spans, keeping the longest."""
    if not spans:
        return []

    # Sort by start position
    sorted_spans = sorted(spans, key=lambda x: x[0])
    merged = [sorted_spans[0]]

    for current in sorted_spans[1:]:
        last = merged[-1]
        # If overlapping, merge
        if current[0] <= last[1]:
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)

    return merged
