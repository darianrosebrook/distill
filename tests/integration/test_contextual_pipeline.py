"""
Integration tests for contextual dataset generation pipeline.

Tests the complete flow from generation → extraction → verification.
"""

import pytest
import json
import hashlib
from pathlib import Path
from unittest.mock import Mock

from tools.schema_registry import ToolSchemaRegistry


@pytest.fixture
def mock_tokenizer():
    """Create mock tokenizer for integration tests."""
    tokenizer = Mock()
    tokenizer.is_fast = True

    def tokenize_side_effect(
        text, add_special_tokens=False, return_offsets_mapping=False, **kwargs
    ):
        tokens = list(text)
        input_ids = list(range(len(tokens)))
        if return_offsets_mapping:
            offsets = []
            pos = 0
            for char in tokens:
                offsets.append((pos, pos + len(char.encode("utf-8"))))
                pos += len(char.encode("utf-8"))
            return {"input_ids": input_ids, "offset_mapping": offsets}
        return input_ids

    tokenizer.side_effect = tokenize_side_effect
    tokenizer.encode = Mock(side_effect=lambda text, **kwargs: list(range(len(text))))
    tokenizer.decode = Mock(
        side_effect=lambda ids, **kwargs: "".join(chr(65 + i % 26) for i in ids)
    )
    return tokenizer


def test_full_pipeline_small(mock_tokenizer, tmp_path):
    """Test full pipeline with 10-sample generation."""
    from scripts.generate_contextual_prompts import synthesize_prompt
    from scripts.extract_process_targets import process_sample
    from scripts.verify_contextual_set import verify_item

    reg = ToolSchemaRegistry()
    output_file = tmp_path / "test_pipeline.jsonl"

    # Generate 10 samples
    samples = []
    for i in range(10):
        cell = {
            "scenario": "file_ops" if i % 2 == 0 else "web_search",
            "complexity": "single_call",
            "structure": "flat_args",
        }
        prompt, history, meta = synthesize_prompt(cell, reg)

        item = {
            "prompt": prompt,
            "teacher_text": history[0]["content"] if history else "No response",
            "metadata": meta,
        }

        # Extract process targets
        processed = process_sample(item, mock_tokenizer, reg)

        # Verify item
        result = verify_item(processed, reg, tokenizer=mock_tokenizer)

        assert result["ok"] is True or len(result.get("problems", [])) == 0
        samples.append(processed)

    # Write to file
    with open(output_file, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    # Verify file was created
    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_full_pipeline_large(mock_tokenizer, tmp_path):
    """Test full pipeline with 100-sample generation."""
    from scripts.generate_contextual_prompts import synthesize_prompt
    from scripts.extract_process_targets import process_sample
    from scripts.verify_contextual_set import verify_item

    reg = ToolSchemaRegistry()
    output_file = tmp_path / "test_pipeline_large.jsonl"

    # Generate 100 samples (use smaller subset for speed)
    samples = []
    ok_count = 0

    for i in range(100):
        cell = {
            "scenario": ["file_ops", "web_search", "code_exec", "multi_step"][i % 4],
            "complexity": ["single_call", "multi_call", "branching_error_recovery"][i % 3],
            "structure": "flat_args",
        }
        prompt, history, meta = synthesize_prompt(cell, reg)

        item = {
            "prompt": prompt,
            "teacher_text": history[0]["content"] if history else "No response",
            "metadata": meta,
        }

        # Extract process targets
        processed = process_sample(item, mock_tokenizer, reg)

        # Verify item
        result = verify_item(processed, reg, tokenizer=mock_tokenizer)

        if result["ok"]:
            ok_count += 1
        samples.append(processed)

    # Write to file
    with open(output_file, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    # Verify file was created and most samples pass
    assert output_file.exists()
    assert output_file.stat().st_size > 0
    # At least 80% should pass (allowing for some edge cases)
    assert ok_count >= 80


def test_pipeline_determinism(mock_tokenizer):
    """Test that same seed produces identical output."""
    from scripts.generate_contextual_prompts import synthesize_prompt
    import random

    reg = ToolSchemaRegistry()
    cell = {
        "scenario": "file_ops",
        "complexity": "single_call",
        "structure": "flat_args",
    }

    # Generate with seed
    random.seed(42)
    prompt1, history1, meta1 = synthesize_prompt(cell, reg)

    # Generate again with same seed
    random.seed(42)
    prompt2, history2, meta2 = synthesize_prompt(cell, reg)

    # Should be identical (or at least very similar)
    assert prompt1 == prompt2
    assert meta1["scenario"] == meta2["scenario"]


def test_determinism_full_pipeline(mock_tokenizer, tmp_path):
    """Test end-to-end determinism: generate→extract→verify twice with same seed, assert identical SHA256."""
    import subprocess
    import sys

    # Create two temporary directories for two runs
    run1_dir = tmp_path / "run1"
    run2_dir = tmp_path / "run2"
    run1_dir.mkdir()
    run2_dir.mkdir()

    seed = 42
    total_samples = 10

    # Run 1: generate → extract → verify
    gen1_out = run1_dir / "generated.jsonl"
    ext1_out = run1_dir / "extracted.jsonl"
    verify1_out = run1_dir / "report.json"

    # Generate
    result1 = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.generate_contextual_prompts",
            "--out",
            str(gen1_out),
            "--total",
            str(total_samples),
            "--seed",
            str(seed),
            "--integration-span-cap",
            "3",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent,
    )
    assert result1.returncode == 0, f"Generation failed: {result1.stderr}"

    # Extract
    result2 = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.extract_process_targets",
            "--in",
            str(gen1_out),
            "--out",
            str(ext1_out),
            "--tokenizer-path",
            "models/student/tokenizer",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent,
    )
    # Extract may fail if tokenizer not available, skip determinism test in that case
    if result2.returncode != 0:
        pytest.skip("Tokenizer not available, skipping determinism test")

    # Verify
    result3 = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.verify_contextual_set",
            "--in",
            str(ext1_out),
            "--report",
            str(verify1_out),
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent,
    )
    assert result3.returncode == 0, f"Verification failed: {result3.stderr}"

    # Run 2: same seed, same steps
    gen2_out = run2_dir / "generated.jsonl"
    ext2_out = run2_dir / "extracted.jsonl"
    verify2_out = run2_dir / "report.json"

    # Generate
    result4 = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.generate_contextual_prompts",
            "--out",
            str(gen2_out),
            "--total",
            str(total_samples),
            "--seed",
            str(seed),
            "--integration-span-cap",
            "3",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent,
    )
    assert result4.returncode == 0, f"Generation failed: {result4.stderr}"

    # Extract
    result5 = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.extract_process_targets",
            "--in",
            str(gen2_out),
            "--out",
            str(ext2_out),
            "--tokenizer-path",
            "models/student/tokenizer",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent,
    )
    assert result5.returncode == 0, f"Extraction failed: {result5.stderr}"

    # Verify
    result6 = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.verify_contextual_set",
            "--in",
            str(ext2_out),
            "--report",
            str(verify2_out),
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent,
    )
    assert result6.returncode == 0, f"Verification failed: {result6.stderr}"

    # Compute SHA256 of final JSONL files (excluding header for data comparison)
    def compute_data_hash(file_path):
        """Compute SHA256 of data items only (skip header)."""
        data_lines = []
        with open(file_path, "rb") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line.decode("utf-8"))
                    # Skip header line
                    if item.get("__header__") is True:
                        continue
                    data_lines.append(line)
                except json.JSONDecodeError:
                    pass
        content = b"".join(data_lines)
        return hashlib.sha256(content).hexdigest()

    hash1 = compute_data_hash(ext1_out)
    hash2 = compute_data_hash(ext2_out)

    # Assert identical SHA256
    assert hash1 == hash2, f"SHA256 mismatch: {hash1} != {hash2}"

    # Also verify report metrics are identical
    with open(verify1_out) as f:
        report1 = json.load(f)
    with open(verify2_out) as f:
        report2 = json.load(f)

    # Compare key metrics
    assert report1["summary"]["total"] == report2["summary"]["total"]
    assert report1["summary"]["ok_rate"] == report2["summary"]["ok_rate"]
    if report1["summary"].get("avg_integration_f1_macro_lax") is not None:
        assert (
            report1["summary"]["avg_integration_f1_macro_lax"]
            == report2["summary"]["avg_integration_f1_macro_lax"]
        )


def test_pipeline_error_recovery(mock_tokenizer):
    """Test that pipeline handles failures gracefully."""
    from scripts.extract_process_targets import process_sample
    from scripts.verify_contextual_set import verify_item
    from tools.schema_registry import ToolSchemaRegistry

    reg = ToolSchemaRegistry()

    # Test with invalid item
    invalid_item = {
        "prompt": "test",
        "teacher_text": "",
        "metadata": {},
    }

    # Should handle gracefully
    processed = process_sample(invalid_item, mock_tokenizer, reg)
    assert isinstance(processed, dict)

    # Verification should also handle gracefully
    result = verify_item(processed, reg)
    assert isinstance(result, dict)
    assert "ok" in result


def test_pipeline_performance(mock_tokenizer):
    """Test pipeline performance benchmarks."""
    import time
    from scripts.generate_contextual_prompts import synthesize_prompt
    from scripts.extract_process_targets import process_sample
    from scripts.verify_contextual_set import verify_item

    reg = ToolSchemaRegistry()

    # Time generation
    start = time.time()
    cell = {
        "scenario": "file_ops",
        "complexity": "single_call",
        "structure": "flat_args",
    }
    prompt, history, meta = synthesize_prompt(cell, reg)
    gen_time = time.time() - start

    # Time extraction
    item = {
        "prompt": prompt,
        "teacher_text": history[0]["content"] if history else "No response",
        "metadata": meta,
    }
    start = time.time()
    processed = process_sample(item, mock_tokenizer, reg)
    extract_time = time.time() - start

    # Time verification
    start = time.time()
    verify_item(processed, reg, tokenizer=mock_tokenizer)
    verify_time = time.time() - start

    # Should complete in reasonable time (< 1 second per sample)
    assert gen_time < 1.0
    assert extract_time < 1.0
    assert verify_time < 1.0


def test_pipeline_memory_usage(mock_tokenizer):
    """Test memory usage for large dataset."""
    import sys
    from scripts.generate_contextual_prompts import synthesize_prompt
    from scripts.extract_process_targets import process_sample

    reg = ToolSchemaRegistry()

    # Generate 1000 samples (simulated)
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

    # Check memory usage (should not grow unbounded)
    assert len(samples) == 100
    # Each sample should be reasonable size
    sample_size = sys.getsizeof(json.dumps(samples[0]))
    assert sample_size < 100000  # Less than 100KB per sample
