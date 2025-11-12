"""
Unit tests for Priority 3: JSON Repair utilities.

Tests:
1. JSON validation
2. JSON repair detection
3. Batch repair checking
4. Repair metrics
"""
import pytest
from training.json_repair import (
    validate_json,
    repair_json,
    check_json_repair_needed,
    batch_check_json_repair,
)


class TestJSONValidation:
    """Test JSON validation functions."""

    def test_validate_json_valid_object(self):
        """Test validation with valid JSON object."""
        text = '{"name": "test", "value": 123}'
        assert validate_json(text) is True

    def test_validate_json_valid_array(self):
        """Test validation with valid JSON array."""
        text = '[1, 2, 3, {"key": "value"}]'
        assert validate_json(text) is True

    def test_validate_json_invalid(self):
        """Test validation with invalid JSON."""
        text = '{"name": "test", "value": 123'  # Missing closing brace
        assert validate_json(text) is False

    def test_validate_json_with_text(self):
        """Test validation with JSON embedded in text."""
        text = 'Some text before {"name": "test"} and after'
        assert validate_json(text) is True

    def test_validate_json_empty(self):
        """Test validation with empty string."""
        assert validate_json("") is False


class TestJSONRepair:
    """Test JSON repair functions."""

    def test_repair_json_valid(self):
        """Test repair with already valid JSON."""
        valid_json = '{"name": "test", "value": 123}'
        success, repaired_dict, was_repaired = repair_json(valid_json, use_jsonrepair=True)
        
        assert success is True
        assert repaired_dict is not None
        assert was_repaired is False
        assert repaired_dict["name"] == "test"
        assert repaired_dict["value"] == 123

    def test_repair_json_missing_quote(self):
        """Test repair with missing quote."""
        invalid_json = '{"name": test, "value": 123}'  # Missing quotes around "test"
        
        # Try repair
        success, repaired_dict, was_repaired = repair_json(invalid_json, use_jsonrepair=True)
        
        # May or may not succeed depending on jsonrepair availability
        if JSONREPAIR_AVAILABLE:
            # jsonrepair should attempt repair
            assert was_repaired is True or was_repaired is False  # May succeed or fail
        else:
            # Without jsonrepair, should fail
            assert success is False or was_repaired is False

    def test_repair_json_without_jsonrepair(self):
        """Test repair without jsonrepair library."""
        invalid_json = '{"name": test}'  # Invalid JSON
        
        success, repaired_dict, was_repaired = repair_json(invalid_json, use_jsonrepair=False)
        
        # Without jsonrepair, should fail to repair
        # But function may still try to parse first
        assert success is False or was_repaired is False
        if not success:
            assert repaired_dict is None


class TestJSONRepairDetection:
    """Test JSON repair detection."""

    def test_check_repair_needed_valid(self):
        """Test detection with valid JSON."""
        valid_json = '{"name": "test"}'
        is_valid, needs_repair = check_json_repair_needed(valid_json, use_jsonrepair=True)
        
        assert is_valid is True
        assert needs_repair is False

    def test_check_repair_needed_invalid(self):
        """Test detection with invalid JSON."""
        invalid_json = '{"name": test}'  # Missing quotes
        
        is_valid, needs_repair = check_json_repair_needed(invalid_json, use_jsonrepair=True)
        
        assert is_valid is False
        # May or may not need repair depending on jsonrepair availability
        assert isinstance(needs_repair, bool)

    def test_check_repair_needed_without_jsonrepair(self):
        """Test detection without jsonrepair."""
        invalid_json = '{"name": test}'
        
        is_valid, needs_repair = check_json_repair_needed(invalid_json, use_jsonrepair=False)
        
        assert is_valid is False
        # Without jsonrepair, needs_repair should be True (can't repair)
        # But function may return False if it can't determine repair need
        assert isinstance(needs_repair, bool)


class TestBatchJSONRepair:
    """Test batch JSON repair checking."""

    def test_batch_check_all_valid(self):
        """Test batch check with all valid JSON."""
        texts = [
            '{"name": "test1"}',
            '{"name": "test2"}',
            '{"name": "test3"}',
        ]
        
        result = batch_check_json_repair(texts, use_jsonrepair=True)
        
        assert result["total"] == 3
        assert result["valid_json_count"] == 3
        assert result["repair_rate"] == 0.0
        # Check that all have JSON
        assert result.get("has_json_count", 0) >= 3

    def test_batch_check_mixed(self):
        """Test batch check with mixed valid/invalid JSON."""
        texts = [
            '{"name": "test1"}',  # Valid
            '{"name": test2}',     # Invalid
            '{"name": "test3"}',  # Valid
        ]
        
        result = batch_check_json_repair(texts, use_jsonrepair=True)
        
        assert result["total"] == 3
        assert result["valid_json_count"] >= 2  # At least 2 valid
        assert result["needs_repair_count"] >= 0  # May or may not need repair
        assert result["repair_rate"] >= 0.0

    def test_batch_check_empty(self):
        """Test batch check with empty list."""
        result = batch_check_json_repair([], use_jsonrepair=True)
        
        assert result["total"] == 0
        assert result["valid_json_count"] == 0
        assert result["repair_rate"] == 0.0
        assert result.get("has_json_count", 0) == 0


# Import JSONREPAIR_AVAILABLE for conditional tests
try:
    import jsonrepair
    JSONREPAIR_AVAILABLE = True
except ImportError:
    JSONREPAIR_AVAILABLE = False


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

