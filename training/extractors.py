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

    Enhanced strategy:
    1. Look for citations/references to tool outputs (expanded patterns)
    2. Identify spans that copy tool result fields (with fuzzy matching)
    3. Detect structured data integration (numbers, lists, formatted data)
    4. Context-aware detection (spans after tool call markers)

    Args:
        teacher_text: Teacher's output text
        tool_results: Optional list of tool result dictionaries

    Returns:
        List of (start_char_idx, end_char_idx) tuples for integration spans
    """
    spans = []

    # Pattern 1: Enhanced citation patterns
    citation_patterns = [
        # Explicit citations
        r'According to [^.]*\.',
        r'Based on [^.]*\.',
        r'The [^.]* shows [^.]*\.',
        r'From [^.]*:',
        # Implicit references
        r'The result[s]? [^.]*\.',
        r'Tool output [^.]*\.',
        r'Search result[s]? [^.]*\.',
        r'Found [^.]*\.',
        r'Retrieved [^.]*\.',
        # Integration markers
        r'Integration: [^.]*\.',
        r'Summary: [^.]*\.',
        r'Insight[s]?: [^.]*\.',
    ]

    for pattern in citation_patterns:
        for match in re.finditer(pattern, teacher_text, re.IGNORECASE):
            # Extend span to include following sentence if it contains data
            end_pos = match.end()
            # Look for following sentence (up to next period or newline)
            next_sentence_match = re.search(
                r'\.\s+[A-Z][^.]*\.', teacher_text[end_pos:end_pos+200])
            if next_sentence_match:
                end_pos = end_pos + next_sentence_match.end()
            spans.append((match.start(), end_pos))

    # Pattern 2: Direct copying of tool result fields with fuzzy matching
    if tool_results:
        for result in tool_results:
            if isinstance(result, dict):
                # Look for result field values in text
                for key, value in result.items():
                    if isinstance(value, str) and len(value) > 10:
                        # Try exact match first
                        # First 100 chars for better matching
                        escaped_value = re.escape(value[:100])
                        for match in re.finditer(escaped_value, teacher_text):
                            spans.append((match.start(), match.end()))

                        # Try fuzzy match: look for key phrases from value
                        # Split value into words and look for sequences
                        words = value.split()[:20]  # First 20 words
                        if len(words) >= 3:
                            # Look for 3+ consecutive words from value
                            phrase = ' '.join(words[:3])
                            escaped_phrase = re.escape(phrase)
                            for match in re.finditer(escaped_phrase, teacher_text, re.IGNORECASE):
                                # Extend to include surrounding context
                                start = max(0, match.start() - 20)
                                end = min(len(teacher_text), match.end() + 50)
                                spans.append((start, end))

                    elif isinstance(value, (int, float)):
                        # Look for numeric values from tool results
                        value_str = str(value)
                        # Match number with context (e.g., "count: 5" or "Found 5 results")
                        num_pattern = rf'\b{re.escape(value_str)}\b'
                        for match in re.finditer(num_pattern, teacher_text):
                            # Include surrounding context
                            start = max(0, match.start() - 30)
                            end = min(len(teacher_text), match.end() + 30)
                            spans.append((start, end))

                    elif isinstance(value, list) and len(value) > 0:
                        # Look for list length mentions (e.g., "Found 5 items")
                        list_len = len(value)
                        len_pattern = rf'\b{list_len}\b.*?(?:item|result|entry|element)'
                        for match in re.finditer(len_pattern, teacher_text, re.IGNORECASE):
                            spans.append((match.start(), match.end()))

                        # Look for first item if it's a string
                        if isinstance(value[0], str) and len(value[0]) > 5:
                            first_item = value[0][:50]
                            escaped_item = re.escape(first_item)
                            for match in re.finditer(escaped_item, teacher_text):
                                spans.append((match.start(), match.end()))

    # Pattern 3: Structured data integration (formatted lists, tables, etc.)
    # Look for patterns that suggest structured data from tools
    structured_patterns = [
        r'Lines? \d+(?:-\d+)?(?:,\s*\d+(?:-\d+)?)*:',  # Line numbers
        r'Count:?\s*\d+',  # Count indicators
        r'Top \d+:',  # Top-K results
        r'Results?:?\s*\d+',  # Result counts
    ]

    for pattern in structured_patterns:
        for match in re.finditer(pattern, teacher_text, re.IGNORECASE):
            # Extend to include following content
            end_pos = match.end()
            # Look for following content up to next sentence or newline
            next_content = re.search(
                r'[^\n.]{10,100}', teacher_text[end_pos:end_pos+150])
            if next_content:
                end_pos = end_pos + next_content.end()
            spans.append((match.start(), end_pos))

    # Pattern 4: Context-aware detection - look for integration after tool call markers
    # Find positions where tool calls might have occurred (JSON patterns)
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    tool_call_positions = []
    for match in re.finditer(json_pattern, teacher_text):
        # Tool call likely ends here, integration follows
        tool_call_positions.append(match.end())

    # Look for integration text after tool calls
    for pos in tool_call_positions:
        # Look for integration patterns in next 500 chars after tool call
        following_text = teacher_text[pos:pos+500]
        integration_markers = [
            r'Integration:',
            r'Summary:',
            r'Based on',
            r'According to',
            r'Found',
            r'Retrieved',
        ]
        for marker in integration_markers:
            marker_match = re.search(marker, following_text, re.IGNORECASE)
            if marker_match:
                # Include from marker to end of sentence/paragraph
                start = pos + marker_match.start()
                # Find end of sentence or paragraph
                end_match = re.search(
                    r'\.\s+(?=[A-Z]|\n|$)', teacher_text[start:start+300])
                if end_match:
                    end = start + end_match.end()
                else:
                    end = min(len(teacher_text), start + 200)
                spans.append((start, end))
                break

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
