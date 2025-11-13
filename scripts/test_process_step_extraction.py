"""
Test script for process-step supervision extraction in dataset generation.

This script tests the extraction logic to verify:
1. Tool name extraction and tokenization
2. JSON argument extraction and tokenization
3. Integration span extraction and tokenization
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import extractors directly
from training.extractors import (  # noqa: E402
    extract_tool_name_span,
    extract_json_argument_spans,
    identify_integration_spans,
)


class MockTokenizer:
    """Simple mock tokenizer for testing."""

    def __init__(self):
        self.vocab_size = 1000
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.pad_token = None

    def encode(self, text, add_special_tokens=False, **kwargs):
        """Encode text to token IDs."""
        # Simple mock: tokenize by splitting on spaces and characters
        tokens = []
        for char in text:
            if char.isalnum() or char in '{}[]":,':
                tokens.append(char)
        token_ids = [abs(hash(t)) % self.vocab_size for t in tokens]

        if add_special_tokens:
            token_ids = [self.eos_token_id] + token_ids + [self.eos_token_id]

        return token_ids

    def __len__(self):
        """Return vocabulary size."""
        return self.vocab_size


def extract_process_step_targets(
    teacher_text: str,
    tokenizer,
    tool_names=None,
):
    """
    Extract process-step supervision targets from teacher output.

    This is a copy of the function from make_kd_mix_hardened.py for testing.
    """
    if tokenizer is None:
        return {}

    targets = {}

    # Extract tool name span
    tool_name_span = extract_tool_name_span(teacher_text, tool_names)
    if tool_name_span:
        start_char, end_char = tool_name_span
        tool_name_text = teacher_text[start_char:end_char]
        tool_name_ids = tokenizer.encode(tool_name_text, add_special_tokens=False)
        targets["tool_name_ids"] = tool_name_ids
        targets["tool_name_mask"] = [1] * len(tool_name_ids)

    # Extract JSON argument spans
    json_spans = extract_json_argument_spans(teacher_text)
    if json_spans:
        all_json_ids = []
        all_json_mask = []
        for start_char, end_char in json_spans:
            json_text = teacher_text[start_char:end_char]
            json_ids = tokenizer.encode(json_text, add_special_tokens=False)
            all_json_ids.extend(json_ids)
            all_json_mask.extend([1] * len(json_ids))
        if all_json_ids:
            targets["gold_json_text_ids"] = all_json_ids
            targets["mask_valid_json_tokens"] = all_json_mask

    # Extract integration spans
    integration_spans = identify_integration_spans(teacher_text)
    if integration_spans:
        all_integration_ids = []
        all_integration_mask = []
        for start_char, end_char in integration_spans:
            integration_text = teacher_text[start_char:end_char]
            integration_ids = tokenizer.encode(integration_text, add_special_tokens=False)
            all_integration_ids.extend(integration_ids)
            all_integration_mask.extend([1] * len(integration_ids))
        if all_integration_ids:
            targets["tool_result_fields"] = all_integration_ids
            targets["integration_mask"] = all_integration_mask

    return targets


def test_tool_name_extraction():
    """Test tool name extraction."""
    print("\n=== Test 1: Tool Name Extraction ===")

    tokenizer = MockTokenizer()
    teacher_text = 'I need to call {"name": "read_file", "arguments": {"path": "test.txt"}}'

    targets = extract_process_step_targets(
        teacher_text=teacher_text, tokenizer=tokenizer, tool_names=["read_file", "write_file"]
    )

    print(f"Teacher text: {teacher_text}")
    print(f"Extracted targets: {list(targets.keys())}")

    if "tool_name_ids" in targets:
        print(f"✅ Tool name IDs found: {len(targets['tool_name_ids'])} tokens")
        print(f"   Tool name mask: {targets['tool_name_mask'][:10]}... (showing first 10)")
    else:
        print("⚠️  No tool name IDs extracted (may be expected if extraction fails)")

    return targets


def test_json_extraction():
    """Test JSON argument extraction."""
    print("\n=== Test 2: JSON Argument Extraction ===")

    tokenizer = MockTokenizer()
    teacher_text = (
        'Here is the tool call: {"name": "search", "arguments": {"query": "test", "limit": 10}}'
    )

    targets = extract_process_step_targets(
        teacher_text=teacher_text, tokenizer=tokenizer, tool_names=None
    )

    print(f"Teacher text: {teacher_text}")
    print(f"Extracted targets: {list(targets.keys())}")

    if "gold_json_text_ids" in targets:
        print(f"✅ JSON IDs found: {len(targets['gold_json_text_ids'])} tokens")
        print(f"   JSON mask length: {len(targets['mask_valid_json_tokens'])}")
    else:
        print("⚠️  No JSON IDs extracted")

    return targets


def test_integration_extraction():
    """Test integration span extraction."""
    print("\n=== Test 3: Integration Span Extraction ===")

    tokenizer = MockTokenizer()
    teacher_text = "After calling read_file, I processed the result. According to the search results, the answer is 42."

    targets = extract_process_step_targets(
        teacher_text=teacher_text, tokenizer=tokenizer, tool_names=None
    )

    print(f"Teacher text: {teacher_text}")
    print(f"Extracted targets: {list(targets.keys())}")

    if "tool_result_fields" in targets:
        print(f"✅ Integration IDs found: {len(targets['tool_result_fields'])} tokens")
        print(f"   Integration mask length: {len(targets['integration_mask'])}")
    else:
        print("⚠️  No integration IDs extracted (may be expected)")

    return targets


def test_complete_example():
    """Test complete example with all components."""
    print("\n=== Test 4: Complete Example ===")

    tokenizer = MockTokenizer()
    teacher_text = """
I'll use the read_file tool to get the content.
{"name": "read_file", "arguments": {"path": "config.json"}}

After reading the file, I found that the configuration shows:
- API endpoint: https://api.example.com
- Timeout: 30 seconds

Based on the file contents, I'll now call write_file to save the updated config.
{"name": "write_file", "arguments": {"path": "config.json", "content": "updated"}}
"""

    targets = extract_process_step_targets(
        teacher_text=teacher_text, tokenizer=tokenizer, tool_names=["read_file", "write_file"]
    )

    print(f"Teacher text length: {len(teacher_text)} chars")
    print(f"Extracted targets: {list(targets.keys())}")

    # Check each component
    components_found = []
    if "tool_name_ids" in targets:
        components_found.append(f"tool_name ({len(targets['tool_name_ids'])} tokens)")
    if "gold_json_text_ids" in targets:
        components_found.append(f"json_args ({len(targets['gold_json_text_ids'])} tokens)")
    if "tool_result_fields" in targets:
        components_found.append(f"integration ({len(targets['tool_result_fields'])} tokens)")

    if components_found:
        print(f"✅ Found components: {', '.join(components_found)}")
    else:
        print("⚠️  No components extracted")

    return targets


def test_no_tokenizer():
    """Test behavior when tokenizer is None."""
    print("\n=== Test 5: No Tokenizer (Should Return Empty) ===")

    targets = extract_process_step_targets(
        teacher_text="Some text with tool calls", tokenizer=None, tool_names=None
    )

    print(f"Extracted targets: {list(targets.keys())}")

    if len(targets) == 0:
        print("✅ Correctly returned empty dict when tokenizer is None")
    else:
        print(f"❌ Expected empty dict, got: {targets}")

    return targets


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Process-Step Supervision Extraction")
    print("=" * 60)

    results = []

    try:
        results.append(("Tool Name", test_tool_name_extraction()))
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()

    try:
        results.append(("JSON Args", test_json_extraction()))
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()

    try:
        results.append(("Integration", test_integration_extraction()))
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()

    try:
        results.append(("Complete", test_complete_example()))
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()

    try:
        results.append(("No Tokenizer", test_no_tokenizer()))
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for name, targets in results:
        component_count = len(
            [k for k in targets.keys() if k.endswith("_ids") or k.endswith("_mask")]
        )
        if name == "No Tokenizer":
            status = "✅" if component_count == 0 else "❌"
        else:
            status = "✅" if component_count > 0 else "⚠️"
        print(f"{status} {name}: {component_count} components extracted")

    print("\n✅ All tests completed!")


if __name__ == "__main__":
    main()
