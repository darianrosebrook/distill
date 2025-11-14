"""
Tests for training/dataloader.py - JSONL data loading utilities.

Tests load_jsonl function with valid/invalid JSONL files, encoding issues,
and edge cases.
"""
# @author: @darianrosebrook

import json
import tempfile
from pathlib import Path

import pytest

from training.dataloader import load_jsonl


class TestLoadJSONL:
    """Test load_jsonl function functionality."""

    def test_load_jsonl_valid_file(self):
        """Test loading valid JSONL file."""
        # Create temporary JSONL file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"key1": "value1", "key2": 42}\n')
            f.write('{"key3": "value3", "key4": [1, 2, 3]}\n')
            f.write('{"key5": true, "key6": null}\n')
            temp_path = f.name

        try:
            # Load and verify
            results = list(load_jsonl(temp_path))

            assert len(results) == 3
            assert results[0] == {"key1": "value1", "key2": 42}
            assert results[1] == {"key3": "value3", "key4": [1, 2, 3]}
            assert results[2] == {"key5": True, "key6": None}
        finally:
            Path(temp_path).unlink()

    def test_load_jsonl_empty_file(self):
        """Test loading empty JSONL file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            temp_path = f.name

        try:
            results = list(load_jsonl(temp_path))
            assert len(results) == 0
        finally:
            Path(temp_path).unlink()

    def test_load_jsonl_single_line(self):
        """Test loading JSONL file with single line."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"single": "line"}\n')
            temp_path = f.name

        try:
            results = list(load_jsonl(temp_path))
            assert len(results) == 1
            assert results[0] == {"single": "line"}
        finally:
            Path(temp_path).unlink()

    def test_load_jsonl_with_whitespace(self):
        """Test loading JSONL file with whitespace lines raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"key1": "value1"}\n')
            f.write('\n')  # Empty line - invalid JSON
            f.write('{"key2": "value2"}\n')
            temp_path = f.name

        try:
            # Empty lines cause JSON decode errors
            with pytest.raises(json.JSONDecodeError):
                list(load_jsonl(temp_path))
        finally:
            Path(temp_path).unlink()

    def test_load_jsonl_invalid_json(self):
        """Test loading JSONL file with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"valid": "json"}\n')
            f.write('{"invalid": json}\n')  # Invalid JSON
            temp_path = f.name

        try:
            # Should raise JSONDecodeError
            with pytest.raises(json.JSONDecodeError):
                list(load_jsonl(temp_path))
        finally:
            Path(temp_path).unlink()

    def test_load_jsonl_unicode_content(self):
        """Test loading JSONL file with unicode content."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", encoding="utf-8", delete=False) as f:
            f.write('{"unicode": "🚀 αβγ 中文"}\n')
            f.write('{"emoji": "✅❌⚠️"}\n')
            temp_path = f.name

        try:
            results = list(load_jsonl(temp_path))
            assert len(results) == 2
            assert results[0]["unicode"] == "🚀 αβγ 中文"
            assert results[1]["emoji"] == "✅❌⚠️"
        finally:
            Path(temp_path).unlink()

    def test_load_jsonl_encoding_utf8(self):
        """Test loading JSONL file with UTF-8 encoding."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", encoding="utf-8", delete=False) as f:
            f.write('{"text": "café résumé"}\n')
            temp_path = f.name

        try:
            results = list(load_jsonl(temp_path))
            assert results[0]["text"] == "café résumé"
        finally:
            Path(temp_path).unlink()

    def test_load_jsonl_iterator_behavior(self):
        """Test that load_jsonl returns an iterator."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"key1": "value1"}\n')
            f.write('{"key2": "value2"}\n')
            temp_path = f.name

        try:
            iterator = load_jsonl(temp_path)
            assert hasattr(iterator, "__iter__")
            assert hasattr(iterator, "__next__")

            # Should be able to iterate
            first = next(iterator)
            assert first == {"key1": "value1"}

            second = next(iterator)
            assert second == {"key2": "value2"}

            # Should raise StopIteration when done
            with pytest.raises(StopIteration):
                next(iterator)
        finally:
            Path(temp_path).unlink()

    def test_load_jsonl_complex_nested_structures(self):
        """Test loading JSONL file with complex nested structures."""
        complex_data = {
            "nested": {"level1": {"level2": {"level3": "deep"}}},
            "array": [1, 2, [3, 4, [5, 6]]],
            "mixed": {"string": "value", "number": 42, "boolean": True, "null": None},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(complex_data) + "\n")
            temp_path = f.name

        try:
            results = list(load_jsonl(temp_path))
            assert len(results) == 1
            assert results[0] == complex_data
        finally:
            Path(temp_path).unlink()

    def test_load_jsonl_file_not_found(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            list(load_jsonl("/nonexistent/path/file.jsonl"))

    def test_load_jsonl_large_file(self):
        """Test loading large JSONL file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Write 1000 lines
            for i in range(1000):
                f.write(json.dumps({"index": i, "data": f"value_{i}"}) + "\n")
            temp_path = f.name

        try:
            results = list(load_jsonl(temp_path))
            assert len(results) == 1000
            assert results[0]["index"] == 0
            assert results[999]["index"] == 999
        finally:
            Path(temp_path).unlink()

    def test_load_jsonl_no_newline_at_end(self):
        """Test loading JSONL file without newline at end."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"key1": "value1"}\n')
            f.write('{"key2": "value2"}')  # No newline at end
            temp_path = f.name

        try:
            results = list(load_jsonl(temp_path))
            assert len(results) == 2
            assert results[0] == {"key1": "value1"}
            assert results[1] == {"key2": "value2"}
        finally:
            Path(temp_path).unlink()

    def test_load_jsonl_numeric_types(self):
        """Test loading JSONL file with various numeric types."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"int": 42, "float": 3.14, "negative": -10, "scientific": 1e10}\n')
            temp_path = f.name

        try:
            results = list(load_jsonl(temp_path))
            assert results[0]["int"] == 42
            assert results[0]["float"] == 3.14
            assert results[0]["negative"] == -10
            assert results[0]["scientific"] == 1e10
        finally:
            Path(temp_path).unlink()

    def test_load_jsonl_boolean_and_null(self):
        """Test loading JSONL file with boolean and null values."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"true_val": true, "false_val": false, "null_val": null}\n')
            temp_path = f.name

        try:
            results = list(load_jsonl(temp_path))
            assert results[0]["true_val"] is True
            assert results[0]["false_val"] is False
            assert results[0]["null_val"] is None
        finally:
            Path(temp_path).unlink()
