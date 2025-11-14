"""
Tests for training/dataloader.py - JSONL data loading utility.

Tests load_jsonl function for reading JSONL files.
"""
# @author: @darianrosebrook

import json
import tempfile
from pathlib import Path

import pytest

from training.dataloader import load_jsonl


class TestLoadJsonl:
    """Test load_jsonl function."""

    def test_load_jsonl_basic(self, tmp_path):
        """Test basic JSONL loading."""
        data_file = tmp_path / "test.jsonl"
        with open(data_file, "w") as f:
            f.write('{"key1": "value1", "key2": 42}\n')
            f.write('{"key1": "value2", "key2": 43}\n')

        items = list(load_jsonl(str(data_file)))

        assert len(items) == 2
        assert items[0]["key1"] == "value1"
        assert items[0]["key2"] == 42
        assert items[1]["key1"] == "value2"
        assert items[1]["key2"] == 43

    def test_load_jsonl_empty_file(self, tmp_path):
        """Test loading empty JSONL file."""
        data_file = tmp_path / "empty.jsonl"
        data_file.touch()

        items = list(load_jsonl(str(data_file)))

        assert len(items) == 0

    def test_load_jsonl_single_line(self, tmp_path):
        """Test loading JSONL with single line."""
        data_file = tmp_path / "single.jsonl"
        with open(data_file, "w") as f:
            f.write('{"id": 1, "text": "hello"}\n')

        items = list(load_jsonl(str(data_file)))

        assert len(items) == 1
        assert items[0]["id"] == 1
        assert items[0]["text"] == "hello"

    def test_load_jsonl_whitespace_lines(self, tmp_path):
        """Test loading JSONL with whitespace-only lines."""
        data_file = tmp_path / "whitespace.jsonl"
        with open(data_file, "w") as f:
            f.write('{"id": 1}\n')
            f.write("\n")
            f.write("   \n")
            f.write('{"id": 2}\n')

        items = list(load_jsonl(str(data_file)))

        # Should skip empty/whitespace lines
        assert len(items) == 2
        assert items[0]["id"] == 1
        assert items[1]["id"] == 2

    def test_load_jsonl_complex_objects(self, tmp_path):
        """Test loading JSONL with complex nested objects."""
        data_file = tmp_path / "complex.jsonl"
        complex_obj = {
            "id": 1,
            "metadata": {"tags": ["tag1", "tag2"], "score": 0.95},
            "data": {"nested": {"deep": "value"}},
        }
        with open(data_file, "w") as f:
            f.write(json.dumps(complex_obj) + "\n")

        items = list(load_jsonl(str(data_file)))

        assert len(items) == 1
        assert items[0]["id"] == 1
        assert items[0]["metadata"]["tags"] == ["tag1", "tag2"]
        assert items[0]["metadata"]["score"] == 0.95
        assert items[0]["data"]["nested"]["deep"] == "value"

    def test_load_jsonl_iterator_behavior(self, tmp_path):
        """Test that load_jsonl returns an iterator."""
        data_file = tmp_path / "test.jsonl"
        with open(data_file, "w") as f:
            f.write('{"id": 1}\n')
            f.write('{"id": 2}\n')
            f.write('{"id": 3}\n')

        iterator = load_jsonl(str(data_file))

        # Should be an iterator
        assert hasattr(iterator, "__iter__")
        assert hasattr(iterator, "__next__")

        # Should be able to iterate
        items = list(iterator)
        assert len(items) == 3

    def test_load_jsonl_multiple_iterations(self, tmp_path):
        """Test that iterator can be consumed multiple times."""
        data_file = tmp_path / "test.jsonl"
        with open(data_file, "w") as f:
            f.write('{"id": 1}\n')
            f.write('{"id": 2}\n')

        iterator = load_jsonl(str(data_file))

        # First iteration
        items1 = list(iterator)
        assert len(items1) == 2

        # Second iteration (should work, but iterator is consumed)
        items2 = list(iterator)
        assert len(items2) == 0  # Iterator is exhausted

    def test_load_jsonl_file_not_found(self):
        """Test loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            list(load_jsonl("nonexistent.jsonl"))

    def test_load_jsonl_invalid_json(self, tmp_path):
        """Test loading JSONL with invalid JSON line."""
        data_file = tmp_path / "invalid.jsonl"
        with open(data_file, "w") as f:
            f.write('{"valid": "json"}\n')
            f.write("invalid json line\n")
            f.write('{"another": "valid"}\n')

        # Should raise JSONDecodeError for invalid line
        with pytest.raises(json.JSONDecodeError):
            list(load_jsonl(str(data_file)))

    def test_load_jsonl_unicode_content(self, tmp_path):
        """Test loading JSONL with unicode content."""
        data_file = tmp_path / "unicode.jsonl"
        with open(data_file, "w", encoding="utf-8") as f:
            f.write('{"text": "Hello 世界", "emoji": "🚀"}\n')

        items = list(load_jsonl(str(data_file)))

        assert len(items) == 1
        assert items[0]["text"] == "Hello 世界"
        assert items[0]["emoji"] == "🚀"

    def test_load_jsonl_large_file(self, tmp_path):
        """Test loading large JSONL file."""
        data_file = tmp_path / "large.jsonl"
        num_lines = 1000

        with open(data_file, "w") as f:
            for i in range(num_lines):
                f.write(json.dumps({"id": i, "data": f"item_{i}"}) + "\n")

        items = list(load_jsonl(str(data_file)))

        assert len(items) == num_lines
        assert items[0]["id"] == 0
        assert items[-1]["id"] == num_lines - 1

    def test_load_jsonl_path_object(self, tmp_path):
        """Test loading JSONL with Path object instead of string."""
        data_file = tmp_path / "test.jsonl"
        with open(data_file, "w") as f:
            f.write('{"id": 1}\n')

        # Should work with Path object
        items = list(load_jsonl(data_file))

        assert len(items) == 1
        assert items[0]["id"] == 1

    def test_load_jsonl_encoding(self, tmp_path):
        """Test that JSONL loading handles UTF-8 encoding correctly."""
        data_file = tmp_path / "encoding.jsonl"
        with open(data_file, "w", encoding="utf-8") as f:
            f.write('{"text": "Café résumé"}\n')

        items = list(load_jsonl(str(data_file)))

        assert len(items) == 1
        assert items[0]["text"] == "Café résumé"

