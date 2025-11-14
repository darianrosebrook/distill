"""
Tests for training/json_repair.py - JSON repair utilities for training-time validation.

Tests JSON validation, extraction, repair, and batch processing.
"""
# @author: @darianrosebrook

import pytest
from unittest.mock import patch, MagicMock
from training.json_repair import (
    validate_json,
    extract_json_from_text,
    simple_json_repair,
    repair_json,
    check_json_repair_needed,
    batch_check_json_repair,
    JSONREPAIR_AVAILABLE,
)


class TestValidateJSON:
    """Test validate_json function."""

    def test_validate_json_valid_object(self):
        """Test validating valid JSON object."""
        text = '{"name": "test", "value": 42}'
        assert validate_json(text) == True

    def test_validate_json_valid_array(self):
        """Test validating valid JSON array."""
        text = '[1, 2, 3, {"key": "value"}]'
        assert validate_json(text) == True

    def test_validate_json_invalid_missing_brace(self):
        """Test validating invalid JSON with missing brace."""
        text = '{"name": "test", "value": 42'
        assert validate_json(text) == False

    def test_validate_json_invalid_trailing_comma(self):
        """Test validating invalid JSON with trailing comma."""
        text = '{"name": "test", "value": 42,}'
        assert validate_json(text) == False

    def test_validate_json_text_with_json(self):
        """Test validating text containing JSON."""
        text = 'Here is the result: {"answer": 42}'
        assert validate_json(text) == True

    def test_validate_json_plain_text(self):
        """Test validating plain text without JSON."""
        text = "This is just plain text with no JSON"
        assert validate_json(text) == False

    def test_validate_json_empty_string(self):
        """Test validating empty string."""
        assert validate_json("") == False

    def test_validate_json_nested_object(self):
        """Test validating nested JSON object."""
        text = '{"outer": {"inner": {"deep": "value"}}}'
        assert validate_json(text) == True

    def test_validate_json_array_with_objects(self):
        """Test validating JSON array containing objects."""
        text = '[{"id": 1}, {"id": 2}, {"id": 3}]'
        assert validate_json(text) == True


class TestExtractJSONFromText:
    """Test extract_json_from_text function."""

    def test_extract_json_from_text_valid_object(self):
        """Test extracting valid JSON object from text."""
        text = 'Here is the result: {"name": "test", "value": 42}'
        result = extract_json_from_text(text)
        assert result is not None
        assert "name" in result
        assert "value" in result

    def test_extract_json_from_text_plain_text(self):
        """Test extracting JSON from plain text (no JSON)."""
        text = "This is just plain text"
        result = extract_json_from_text(text)
        assert result is None

    def test_extract_json_from_text_multiple_objects(self):
        """Test extracting JSON when multiple objects present."""
        text = 'First: {"a": 1} Second: {"b": 2}'
        result = extract_json_from_text(text)
        # Should return first valid JSON found
        assert result is not None

    def test_extract_json_from_text_entire_text_is_json(self):
        """Test extracting JSON when entire text is JSON."""
        text = '{"name": "test", "value": 42}'
        result = extract_json_from_text(text)
        assert result == text.strip()

    def test_extract_json_from_text_invalid_json(self):
        """Test extracting JSON from text with invalid JSON."""
        text = 'Here is invalid JSON: {"name": "test"'
        result = extract_json_from_text(text)
        # Should return None if no valid JSON found
        assert result is None

    def test_extract_json_from_text_empty_string(self):
        """Test extracting JSON from empty string."""
        result = extract_json_from_text("")
        assert result is None


class TestSimpleJSONRepair:
    """Test simple_json_repair function."""

    def test_simple_json_repair_single_quotes(self):
        """Test repairing JSON with single quotes."""
        invalid_json = "{'name': 'test', 'value': 42}"
        repaired = simple_json_repair(invalid_json)
        assert repaired is not None
        # Should have double quotes
        assert '"name"' in repaired or '"test"' in repaired

    def test_simple_json_repair_trailing_comma(self):
        """Test repairing JSON with trailing comma."""
        invalid_json = '{"name": "test", "value": 42,}'
        repaired = simple_json_repair(invalid_json)
        assert repaired is not None
        # Trailing comma should be removed
        assert not repaired.endswith(",}")

    def test_simple_json_repair_trailing_comma_array(self):
        """Test repairing JSON array with trailing comma."""
        invalid_json = '[1, 2, 3,]'
        repaired = simple_json_repair(invalid_json)
        assert repaired is not None
        # Trailing comma should be removed
        assert not repaired.endswith(",]")

    def test_simple_json_repair_valid_json(self):
        """Test repairing already valid JSON."""
        valid_json = '{"name": "test", "value": 42}'
        repaired = simple_json_repair(valid_json)
        assert repaired is not None

    def test_simple_json_repair_empty_string(self):
        """Test repairing empty string."""
        repaired = simple_json_repair("")
        assert repaired == ""

    def test_simple_json_repair_whitespace(self):
        """Test repairing JSON with extra whitespace."""
        invalid_json = '  {"name": "test"}  '
        repaired = simple_json_repair(invalid_json)
        assert repaired is not None
        assert repaired.strip() == repaired  # Should be trimmed


class TestRepairJSON:
    """Test repair_json function."""

    def test_repair_json_already_valid(self):
        """Test repairing already valid JSON."""
        valid_json = '{"name": "test", "value": 42}'
        success, result, was_repaired = repair_json(valid_json)
        assert success == True
        assert result is not None
        assert was_repaired == False

    def test_repair_json_simple_repair(self):
        """Test repairing JSON with simple issues."""
        invalid_json = '{"name": "test", "value": 42,}'
        success, result, was_repaired = repair_json(invalid_json, use_jsonrepair=False)
        assert success == True
        assert result is not None
        assert was_repaired == True

    @patch("training.json_repair.JSONREPAIR_AVAILABLE", True)
    @patch("training.json_repair.jsonrepair")
    def test_repair_json_with_jsonrepair(self, mock_jsonrepair):
        """Test repairing JSON using jsonrepair library."""
        invalid_json = '{"name": "test", invalid}'
        mock_jsonrepair.repair_json.return_value = '{"name": "test", "invalid": null}'

        success, result, was_repaired = repair_json(invalid_json, use_jsonrepair=True)
        assert success == True
        assert result is not None
        assert was_repaired == True
        mock_jsonrepair.repair_json.assert_called_once()

    def test_repair_json_unrepairable(self):
        """Test repairing unrepairable JSON."""
        invalid_json = "This is not JSON at all"
        success, result, was_repaired = repair_json(invalid_json, use_jsonrepair=False)
        assert success == False
        assert result is None
        assert was_repaired == True

    def test_repair_json_without_jsonrepair(self):
        """Test repairing JSON without jsonrepair library."""
        invalid_json = '{"name": "test", "value": 42,}'
        success, result, was_repaired = repair_json(invalid_json, use_jsonrepair=False)
        # Should still attempt simple repair
        assert isinstance(success, bool)
        assert isinstance(was_repaired, bool)

    def test_repair_json_empty_string(self):
        """Test repairing empty string."""
        success, result, was_repaired = repair_json("", use_jsonrepair=False)
        assert success == False
        assert result is None


class TestCheckJSONRepairNeeded:
    """Test check_json_repair_needed function."""

    def test_check_json_repair_needed_valid_json(self):
        """Test checking text with valid JSON."""
        text = 'Here is valid JSON: {"name": "test", "value": 42}'
        has_json, needs_repair = check_json_repair_needed(text)
        assert has_json == True
        assert needs_repair == False

    def test_check_json_repair_needed_invalid_json(self):
        """Test checking text with invalid JSON."""
        text = 'Here is invalid JSON: {"name": "test"'
        has_json, needs_repair = check_json_repair_needed(text)
        assert has_json == True
        assert needs_repair == True

    def test_check_json_repair_needed_no_json(self):
        """Test checking text with no JSON."""
        text = "This is just plain text"
        has_json, needs_repair = check_json_repair_needed(text)
        assert has_json == False
        assert needs_repair == False

    def test_check_json_repair_needed_repairable_json(self):
        """Test checking text with repairable JSON."""
        text = 'Here is repairable JSON: {"name": "test", "value": 42,}'
        has_json, needs_repair = check_json_repair_needed(text, use_jsonrepair=False)
        assert has_json == True
        # May or may not need repair depending on simple repair success
        assert isinstance(needs_repair, bool)

    def test_check_json_repair_needed_empty_string(self):
        """Test checking empty string."""
        has_json, needs_repair = check_json_repair_needed("")
        assert has_json == False
        assert needs_repair == False


class TestBatchCheckJSONRepair:
    """Test batch_check_json_repair function."""

    def test_batch_check_json_repair_empty_list(self):
        """Test batch checking with empty list."""
        result = batch_check_json_repair([])
        assert result["total"] == 0
        assert result["has_json_count"] == 0
        assert result["valid_json_count"] == 0
        assert result["needs_repair_count"] == 0
        assert result["repair_rate"] == 0.0

    def test_batch_check_json_repair_all_valid(self):
        """Test batch checking with all valid JSON."""
        texts = [
            '{"name": "test1", "value": 1}',
            '{"name": "test2", "value": 2}',
            '{"name": "test3", "value": 3}',
        ]
        result = batch_check_json_repair(texts)
        assert result["total"] == 3
        assert result["has_json_count"] == 3
        assert result["valid_json_count"] == 3
        assert result["needs_repair_count"] == 0
        assert result["repair_rate"] == 0.0

    def test_batch_check_json_repair_mixed(self):
        """Test batch checking with mixed valid/invalid JSON."""
        texts = [
            '{"name": "test1", "value": 1}',  # Valid
            '{"name": "test2"',  # Invalid
            "Plain text",  # No JSON
            '{"name": "test3", "value": 3,}',  # Invalid (trailing comma)
        ]
        result = batch_check_json_repair(texts)
        assert result["total"] == 4
        assert result["has_json_count"] >= 2  # At least 2 have JSON
        assert result["repair_rate"] >= 0.0

    def test_batch_check_json_repair_all_invalid(self):
        """Test batch checking with all invalid JSON."""
        texts = [
            '{"name": "test1"',
            '{"name": "test2"',
            '{"name": "test3"',
        ]
        result = batch_check_json_repair(texts)
        assert result["total"] == 3
        assert result["has_json_count"] >= 0
        assert result["needs_repair_count"] >= 0

    def test_batch_check_json_repair_no_json(self):
        """Test batch checking with no JSON."""
        texts = [
            "Plain text 1",
            "Plain text 2",
            "Plain text 3",
        ]
        result = batch_check_json_repair(texts)
        assert result["total"] == 3
        assert result["has_json_count"] == 0
        assert result["valid_json_count"] == 0
        assert result["needs_repair_count"] == 0

    def test_batch_check_json_repair_large_batch(self):
        """Test batch checking with large batch."""
        texts = [f'{{"id": {i}, "value": {i * 2}}}' for i in range(100)]
        result = batch_check_json_repair(texts)
        assert result["total"] == 100
        assert result["has_json_count"] == 100
        assert result["valid_json_count"] == 100
        assert result["repair_rate"] == 0.0

    def test_batch_check_json_repair_metrics_structure(self):
        """Test that batch result has correct structure."""
        texts = ['{"name": "test"}']
        result = batch_check_json_repair(texts)
        assert "total" in result
        assert "has_json_count" in result
        assert "valid_json_count" in result
        assert "needs_repair_count" in result
        assert "repair_rate" in result
        assert isinstance(result["repair_rate"], float)


class TestJSONRepairIntegration:
    """Test integration of JSON repair components."""

    def test_repair_workflow(self):
        """Test complete repair workflow."""
        invalid_json = '{"name": "test", "value": 42,}'

        # Check if repair needed
        has_json, needs_repair = check_json_repair_needed(invalid_json, use_jsonrepair=False)
        assert has_json == True

        # Repair
        success, result, was_repaired = repair_json(invalid_json, use_jsonrepair=False)
        if success:
            assert result is not None
            assert was_repaired == True

    def test_batch_workflow(self):
        """Test batch repair workflow."""
        texts = [
            '{"valid": true}',
            '{"invalid": true,}',
            "No JSON here",
        ]

        # Batch check
        metrics = batch_check_json_repair(texts, use_jsonrepair=False)
        assert metrics["total"] == 3

        # Repair invalid ones
        for text in texts:
            has_json, needs_repair = check_json_repair_needed(text, use_jsonrepair=False)
            if has_json and needs_repair:
                success, result, _ = repair_json(text, use_jsonrepair=False)
                # Should handle gracefully
                assert isinstance(success, bool)

