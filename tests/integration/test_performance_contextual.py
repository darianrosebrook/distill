"""
Performance tests for contextual dataset generation pipeline.

Tests:
1. Generation performance
2. Extraction performance
3. Verification performance
4. Memory usage
"""

import pytest
import time
import sys
from unittest.mock import Mock


@pytest.fixture
def mock_tokenizer():
    """Create mock tokenizer for performance tests."""
    tokenizer = Mock()
    tokenizer.is_fast = True

    def tokenize_side_effect(text, **kwargs):
        return {
            "input_ids": list(range(len(text))),
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }

    tokenizer.side_effect = tokenize_side_effect
    tokenizer.encode = Mock(side_effect=lambda text, **kwargs: list(range(len(text))))
    tokenizer.decode = Mock(
        side_effect=lambda ids, **kwargs: "".join(chr(65 + i % 26) for i in ids)
    )
    return tokenizer


def test_generation_performance():
    """Test generation time per sample."""
    from scripts.generate_contextual_prompts import synthesize_prompt
    from tools.schema_registry import ToolSchemaRegistry

    reg = ToolSchemaRegistry()

    times = []
    for i in range(10):
        cell = {
            "scenario": "file_ops",
            "complexity": "single_call",
            "structure": "flat_args",
        }
        start = time.time()
        prompt, history, meta = synthesize_prompt(cell, reg)
        elapsed = time.time() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    # Should generate in < 0.1 seconds per sample
    assert avg_time < 0.1


def test_extraction_performance(mock_tokenizer):
    """Test token span computation time."""
    from scripts.extract_process_targets import extract_process_step_targets

    teacher_text = 'I will call {"name": "read_file", "arguments": {"path": "test.txt"}}'

    times = []
    for i in range(10):
        start = time.time()
        extract_process_step_targets(teacher_text, mock_tokenizer)
        elapsed = time.time() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    # Should extract in < 0.05 seconds per sample
    assert avg_time < 0.05


def test_verification_performance(mock_tokenizer):
    """Test verification time per sample."""
    from scripts.verify_contextual_set import verify_item
    from tools.schema_registry import ToolSchemaRegistry

    reg = ToolSchemaRegistry()
    item = {
        "prompt": '{"caws": {"tier": 2}}',
        "teacher_text": '{"name": "read_file", "arguments": {"path": "test.txt"}}',
        "metadata": {
            "dataset_version": "1.1.0",
            "call_sequence": [{"name": "read_file", "arguments": {"path": "test.txt"}}],
            "json_args_span_bytes": [0, 50],
            "expected_behaviour": "normal",
        },
    }

    times = []
    for i in range(10):
        start = time.time()
        verify_item(item, reg, tokenizer=mock_tokenizer)
        elapsed = time.time() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    # Should verify in < 0.1 seconds per sample
    assert avg_time < 0.1


def test_memory_usage_large_dataset(mock_tokenizer):
    """Test memory usage for large dataset."""
    import json
    from scripts.generate_contextual_prompts import synthesize_prompt
    from scripts.extract_process_targets import process_sample
    from tools.schema_registry import ToolSchemaRegistry

    reg = ToolSchemaRegistry()

    # Generate 100 samples
    samples = []
    for i in range(100):
        cell = {
            "scenario": "file_ops",
            "complexity": "single_call",
            "structure": "flat_args",
        }
        prompt, history, meta = synthesize_prompt(cell, reg)

        item = {
            "prompt": prompt,
            "teacher_text": history[0]["content"] if history else "No response",
            "metadata": meta,
        }

        processed = process_sample(item, mock_tokenizer, reg)
        samples.append(processed)

    # Check memory usage
    total_size = sum(sys.getsizeof(json.dumps(s)) for s in samples)
    avg_size = total_size / len(samples)

    # Average sample size should be reasonable (< 50KB)
    assert avg_size < 50000

    # Total should be reasonable (< 10MB for 100 samples)
    assert total_size < 10000000
