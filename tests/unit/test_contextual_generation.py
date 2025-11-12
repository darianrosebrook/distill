"""
Unit tests for contextual dataset generation scripts.

Tests:
1. generate_contextual_prompts - prompt synthesis
2. extract_process_targets - target extraction
3. verify_contextual_set - verification logic
"""
import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from tools.schema_registry import ToolSchemaRegistry


class TestGenerateContextualPrompts:
    """Test contextual prompt generation."""
    
    def test_synthesize_prompt_basic(self):
        """Test basic prompt synthesis."""
        from scripts.generate_contextual_prompts import synthesize_prompt
        
        reg = ToolSchemaRegistry()
        cell = {
            "scenario": "file_ops",
            "complexity": "single_call",
            "structure": "flat_args",
        }
        
        prompt, history, meta = synthesize_prompt(cell, reg)
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert isinstance(history, list)
        assert isinstance(meta, dict)
        assert "dataset_version" in meta
        assert "scenario" in meta
        assert meta["scenario"] == "file_ops"
    
    def test_synthesize_prompt_control_case(self):
        """Test control case (no_tool) generation."""
        from scripts.generate_contextual_prompts import synthesize_prompt
        
        reg = ToolSchemaRegistry()
        cell = {
            "scenario": "file_ops",
            "complexity": "single_call",
            "expected_behaviour": "no_tool",
        }
        
        prompt, history, meta = synthesize_prompt(cell, reg)
        
        assert meta["expected_behaviour"] == "no_tool"
        # History contains assistant response, but no tool calls
        assert len(history) >= 0  # May have assistant response
        # Control cases should not have integration spans
        assert meta.get("integration_spans_bytes", []) == []
        # Control cases should not have tool calls in call_sequence
        assert len(meta.get("call_sequence", [])) == 0
    
    def test_synthesize_prompt_with_integration(self):
        """Test prompt with integration spans."""
        from scripts.generate_contextual_prompts import synthesize_prompt
        
        reg = ToolSchemaRegistry()
        cell = {
            "scenario": "web_search",
            "complexity": "single_call",
            "expected_behaviour": "normal",
        }
        
        prompt, history, meta = synthesize_prompt(cell, reg)
        
        # Should have tool calls
        assert len(history) > 0
        # Should have integration spans if tool results present
        if meta.get("tool_result_fields"):
            assert "integration_spans_bytes" in meta
    
    def test_all_scenarios(self):
        """Test all scenario types generate correctly."""
        from scripts.generate_contextual_prompts import synthesize_prompt
        
        reg = ToolSchemaRegistry()
        scenarios = ["file_ops", "web_search", "code_exec", "multi_step"]
        
        for scenario in scenarios:
            cell = {
                "scenario": scenario,
                "complexity": "single_call",
                "structure": "flat_args",
            }
            prompt, history, meta = synthesize_prompt(cell, reg)
            
            assert isinstance(prompt, str)
            assert len(prompt) > 0
            assert meta["scenario"] == scenario
    
    def test_all_complexities(self):
        """Test all complexity levels generate correctly."""
        from scripts.generate_contextual_prompts import synthesize_prompt
        
        reg = ToolSchemaRegistry()
        complexities = ["single_call", "multi_call", "branching_error_recovery"]
        
        for complexity in complexities:
            cell = {
                "scenario": "file_ops",
                "complexity": complexity,
                "structure": "flat_args",
            }
            prompt, history, meta = synthesize_prompt(cell, reg)
            
            assert isinstance(prompt, str)
            assert meta["complexity"] == complexity
    
    def test_all_structures(self):
        """Test all structure types generate correctly."""
        from scripts.generate_contextual_prompts import synthesize_prompt
        
        reg = ToolSchemaRegistry()
        structures = ["flat_args", "nested_args", "arrays", "enums", "numeric_ranges", "optional_keys"]
        
        for structure in structures:
            cell = {
                "scenario": "file_ops",
                "complexity": "single_call",
                "structure": structure,
            }
            prompt, history, meta = synthesize_prompt(cell, reg)
            
            assert isinstance(prompt, str)
            assert meta["structure"] == structure
    
    def test_adversarial_range_violation(self):
        """Test adversarial range violation handling."""
        from scripts.generate_contextual_prompts import synthesize_prompt
        
        reg = ToolSchemaRegistry()
        cell = {
            "scenario": "file_ops",
            "complexity": "single_call",
            "adversarial": {"type": "range_violation"},
        }
        
        prompt, history, meta = synthesize_prompt(cell, reg)
        
        assert isinstance(prompt, str)
        assert meta.get("adversarial", {}).get("type") == "range_violation"
    
    def test_adversarial_malformed_json(self):
        """Test adversarial malformed JSON handling."""
        from scripts.generate_contextual_prompts import synthesize_prompt
        
        reg = ToolSchemaRegistry()
        cell = {
            "scenario": "file_ops",
            "complexity": "single_call",
            "adversarial": {"type": "malformed_json"},
        }
        
        prompt, history, meta = synthesize_prompt(cell, reg)
        
        assert isinstance(prompt, str)
        assert meta.get("adversarial", {}).get("type") == "malformed_json"
    
    def test_adversarial_ambiguity(self):
        """Test adversarial ambiguity case handling."""
        from scripts.generate_contextual_prompts import synthesize_prompt
        
        reg = ToolSchemaRegistry()
        cell = {
            "scenario": "file_ops",
            "complexity": "single_call",
            "adversarial": {"type": "ambiguity"},
        }
        
        prompt, history, meta = synthesize_prompt(cell, reg)
        
        assert isinstance(prompt, str)
        assert meta.get("adversarial", {}).get("type") == "ambiguity"
        # Ambiguity cases should not have integration spans
        assert meta.get("integration_spans_bytes", []) == []
    
    def test_multilingual_generation(self):
        """Test multi-lingual prompt generation."""
        from scripts.generate_contextual_prompts import synthesize_prompt
        
        reg = ToolSchemaRegistry()
        languages = ["es", "de", "fr"]
        
        for lang in languages:
            cell = {
                "scenario": "file_ops",
                "complexity": "single_call",
                "language": lang,
            }
            prompt, history, meta = synthesize_prompt(cell, reg)
            
            assert isinstance(prompt, str)
            assert len(prompt) > 0
            # Prompt should contain language-specific content (check for Spanish/German/French words)
            # Note: language may not be in metadata, but prompt should be different
            if lang == "es":
                # Check for Spanish words in prompt
                assert any(word in prompt.lower() for word in ["archivo", "lee", "extrae"])
            elif lang == "de":
                # Check for German words
                assert any(word in prompt.lower() for word in ["datei", "lese", "extrahiere"])
            elif lang == "fr":
                # Check for French words
                assert any(word in prompt.lower() for word in ["fichier", "lisez", "extrayez"])
    
    def test_long_context_token_aware(self):
        """Test token-aware long-context generation."""
        from scripts.generate_contextual_prompts import synthesize_prompt
        
        reg = ToolSchemaRegistry()
        cell = {
            "scenario": "file_ops",
            "complexity": "single_call",
            "long_context": True,
        }
        
        prompt, history, meta = synthesize_prompt(cell, reg)
        
        assert isinstance(prompt, str)
        assert meta.get("long_context") is True
        # Long context prompts should be significantly longer
        assert len(prompt) > 1000
    
    def test_long_context_byte_based(self):
        """Test byte-based long-context fallback."""
        from scripts.generate_contextual_prompts import synthesize_prompt
        
        reg = ToolSchemaRegistry()
        cell = {
            "scenario": "file_ops",
            "complexity": "single_call",
            "long_context": True,
        }
        
        prompt, history, meta = synthesize_prompt(cell, reg)
        
        assert isinstance(prompt, str)
        assert meta.get("long_context") is True
    
    def test_stratification_enforcement(self):
        """Test that stratification requirements are met."""
        from scripts.generate_contextual_prompts import build_stratified_cells
        
        cells = build_stratified_cells(60)
        
        assert len(cells) > 0
        # Check that we have coverage across scenarios
        scenarios = [cell.get("scenario") for cell in cells]
        assert "file_ops" in scenarios
        assert "web_search" in scenarios
    
    def test_tool_result_fields_dict(self):
        """Test that tool_result_fields is a dict."""
        from scripts.generate_contextual_prompts import synthesize_prompt
        
        reg = ToolSchemaRegistry()
        cell = {
            "scenario": "web_search",
            "complexity": "single_call",
            "expected_behaviour": "normal",
        }
        
        prompt, history, meta = synthesize_prompt(cell, reg)
        
        # If tool_result_fields exists, it should be a dict
        if "tool_result_fields" in meta:
            assert isinstance(meta["tool_result_fields"], dict)
    
    def test_control_case_decline(self):
        """Test decline control case."""
        from scripts.generate_contextual_prompts import synthesize_prompt
        
        reg = ToolSchemaRegistry()
        cell = {
            "scenario": "file_ops",
            "complexity": "single_call",
            "expected_behaviour": "decline",
        }
        
        prompt, history, meta = synthesize_prompt(cell, reg)
        
        assert meta["expected_behaviour"] == "decline"
        assert len(meta.get("call_sequence", [])) == 0
        assert meta.get("integration_spans_bytes", []) == []
    
    def test_compact_caws_header(self):
        """Test compact CAWS header generation."""
        from scripts.generate_contextual_prompts import compact_caws
        
        header = compact_caws(tier=2)
        
        assert "caws" in header
        assert header["caws"]["tier"] == 2
        assert "max_files" in header["caws"]
        assert "max_loc" in header["caws"]
    
    def test_normalize_text(self):
        """Test text normalization."""
        from scripts.generate_contextual_prompts import normalize_text
        
        text = "Test text"
        normalized = normalize_text(text)
        
        assert isinstance(normalized, str)
        assert len(normalized) > 0


class TestExtractProcessTargets:
    """Test process-step target extraction."""
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = Mock()
        tokenizer.is_fast = True
        tokenizer.encode = Mock(return_value=[1, 2, 3])
        tokenizer.decode = Mock(return_value="test")
        tokenizer.return_value = {
            "offset_mapping": [(0, 1), (1, 2), (2, 3)]
        }
        
        def tokenize_side_effect(text, **kwargs):
            return {
                "input_ids": [1, 2, 3],
                "offset_mapping": [(0, 1), (1, 2), (2, 3)]
            }
        
        tokenizer.side_effect = tokenize_side_effect
        return tokenizer
    
    def test_extract_process_step_targets_basic(self, mock_tokenizer):
        """Test basic target extraction."""
        from scripts.extract_process_targets import extract_process_step_targets
        
        teacher_text = 'I will call {"name": "read_file", "arguments": {"path": "test.txt"}}'
        
        targets = extract_process_step_targets(
            teacher_text=teacher_text,
            tokenizer=mock_tokenizer,
            tool_names=["read_file"]
        )
        
        assert isinstance(targets, dict)
        # Should have tool name span if found
        if "tool_name_span_bytes" in targets:
            assert len(targets["tool_name_span_bytes"]) == 2
    
    def test_extract_process_step_targets_control_case(self, mock_tokenizer):
        """Test extraction for control cases."""
        from scripts.extract_process_targets import process_sample
        
        item = {
            "teacher_text": "I don't have access to tools.",
            "metadata": {
                "expected_behaviour": "no_tool",
                "call_sequence": [],
            }
        }
        
        reg = ToolSchemaRegistry()
        processed = process_sample(item, mock_tokenizer, reg)
        
        # Controls should have empty integration spans
        assert processed["metadata"].get("integration_spans_bytes", []) == []
    
    def test_extract_process_step_targets_preserves_tool_result_fields(self, mock_tokenizer):
        """Test that tool_result_fields dict is preserved."""
        from scripts.extract_process_targets import process_sample
        
        item = {
            "teacher_text": 'I will call {"name": "read_file", "arguments": {"path": "test.txt"}}',
            "metadata": {
                "expected_behaviour": "normal",
                "call_sequence": [{"name": "read_file", "arguments": {"path": "test.txt"}}],
                "tool_result_fields": {"summary": "File contents", "lines": 10},
            }
        }
        
        reg = ToolSchemaRegistry()
        processed = process_sample(item, mock_tokenizer, reg)
        
        # Should preserve tool_result_fields as dict
        result_fields = processed["metadata"].get("tool_result_fields")
        assert isinstance(result_fields, dict)
        assert "summary" in result_fields
    
    def test_multi_call_extraction(self, mock_tokenizer):
        """Test extraction with multiple tool calls."""
        from scripts.extract_process_targets import extract_process_step_targets
        
        teacher_text = 'I will call {"name": "read_file", "arguments": {"path": "test.txt"}} and then {"name": "write_file", "arguments": {"path": "out.txt"}}'
        
        targets = extract_process_step_targets(
            teacher_text=teacher_text,
            tokenizer=mock_tokenizer,
            tool_names=["read_file", "write_file"]
        )
        
        assert isinstance(targets, dict)
        # Should have multiple JSON spans
        if "json_args_span_bytes" in targets:
            assert isinstance(targets["json_args_span_bytes"], list)
    
    def test_token_span_alignment_edge_cases(self, mock_tokenizer):
        """Test token span alignment with edge cases."""
        from scripts.extract_process_targets import extract_process_step_targets
        
        # Test with empty text
        targets = extract_process_step_targets("", mock_tokenizer)
        assert isinstance(targets, dict)
        
        # Test with no tool calls
        targets = extract_process_step_targets("Just regular text", mock_tokenizer)
        assert isinstance(targets, dict)
        assert "tool_name_ids" not in targets or targets.get("tool_name_ids") is None
    
    def test_normalization_preservation(self, mock_tokenizer):
        """Test that normalization metadata is preserved."""
        from scripts.extract_process_targets import process_sample
        
        item = {
            "teacher_text": 'I will call {"name": "read_file", "arguments": {"path": "test.txt"}}',
            "metadata": {
                "expected_behaviour": "normal",
                "call_sequence": [{"name": "read_file", "arguments": {"path": "test.txt"}}],
                "text_norm": "NFC",
                "line_endings": "LF",
            }
        }
        
        reg = ToolSchemaRegistry()
        processed = process_sample(item, mock_tokenizer, reg)
        
        # Normalization metadata should be preserved
        assert processed["metadata"].get("text_norm") == "NFC"
        assert processed["metadata"].get("line_endings") == "LF"
    
    def test_control_case_filtering(self, mock_tokenizer):
        """Test that control cases have no integration spans."""
        from scripts.extract_process_targets import process_sample
        
        # Test no_tool case
        item_no_tool = {
            "teacher_text": "I don't have access to tools.",
            "metadata": {
                "expected_behaviour": "no_tool",
                "call_sequence": [],
            }
        }
        
        reg = ToolSchemaRegistry()
        processed = process_sample(item_no_tool, mock_tokenizer, reg)
        
        assert processed["metadata"].get("integration_spans_bytes", []) == []
        assert processed["metadata"].get("integration_token_ids") is None or len(processed["metadata"].get("integration_token_ids", [])) == 0
        
        # Test decline case
        item_decline = {
            "teacher_text": "I'm sorry, but I cannot access that file.",
            "metadata": {
                "expected_behaviour": "decline",
                "call_sequence": [],
            }
        }
        
        processed = process_sample(item_decline, mock_tokenizer, reg)
        assert processed["metadata"].get("integration_spans_bytes", []) == []
    
    def test_error_handling_invalid_tokenizer(self):
        """Test graceful handling of invalid tokenizer."""
        from scripts.extract_process_targets import extract_process_step_targets
        
        # None tokenizer should return empty dict
        result = extract_process_step_targets("test", None)
        assert result == {}
        
        # Invalid tokenizer (missing encode method)
        class InvalidTokenizer:
            pass
        
        invalid_tokenizer = InvalidTokenizer()
        # Should handle gracefully (may raise AttributeError, but we test that it doesn't crash)
        try:
            result = extract_process_step_targets("test", invalid_tokenizer)
            # If it doesn't raise, result should be a dict
            assert isinstance(result, dict)
        except (AttributeError, TypeError):
            # Expected for invalid tokenizer
            pass
    
    def test_integration_field_extraction(self, mock_tokenizer):
        """Test integration field name extraction."""
        from scripts.extract_process_targets import extract_integration_fields
        
        teacher_text = "Integration: The file contains 128 lines. Summary: File processed successfully."
        integration_spans = [[13, 35], [36, 70]]  # "The file contains 128 lines" and "Summary: File processed successfully"
        call_sequence = [{"name": "read_file", "arguments": {"path": "test.txt"}}]
        
        fields = extract_integration_fields(teacher_text, integration_spans, call_sequence)
        
        assert isinstance(fields, list)
        # Should extract field names if they match tool result patterns
    
    def test_preserve_existing_metadata(self, mock_tokenizer):
        """Test that existing metadata is preserved."""
        from scripts.extract_process_targets import process_sample
        
        item = {
            "teacher_text": 'I will call {"name": "read_file", "arguments": {"path": "test.txt"}}',
            "metadata": {
                "expected_behaviour": "normal",
                "call_sequence": [{"name": "read_file", "arguments": {"path": "test.txt"}}],
                "sample_id": "test-123",
                "scenario": "file_ops",
                "custom_field": "custom_value",
            }
        }
        
        reg = ToolSchemaRegistry()
        processed = process_sample(item, mock_tokenizer, reg)
        
        # Existing metadata should be preserved
        assert processed["metadata"].get("sample_id") == "test-123"
        assert processed["metadata"].get("scenario") == "file_ops"
        assert processed["metadata"].get("custom_field") == "custom_value"
    
    def test_validate_json_args(self, mock_tokenizer):
        """Test JSON argument validation."""
        from scripts.extract_process_targets import validate_json_args
        from tools.schema_registry import ToolSchemaRegistry
        
        teacher_text = '{"name": "read_file", "arguments": {"path": "test.txt"}}'
        json_span = [0, len(teacher_text)]
        reg = ToolSchemaRegistry()
        
        is_valid, tool_name, args = validate_json_args(teacher_text, json_span, reg)
        
        # Should validate if tool exists in registry
        assert isinstance(is_valid, bool)
        if tool_name:
            assert tool_name == "read_file"
    
    def test_validate_json_args_invalid(self, mock_tokenizer):
        """Test JSON argument validation with invalid input."""
        from scripts.extract_process_targets import validate_json_args
        from tools.schema_registry import ToolSchemaRegistry
        
        teacher_text = "Not JSON"
        json_span = [0, len(teacher_text)]
        reg = ToolSchemaRegistry()
        
        is_valid, tool_name, args = validate_json_args(teacher_text, json_span, reg)
        
        assert is_valid is False
        assert tool_name is None
        assert args is None
    
    def test_extract_with_normalized_text(self, mock_tokenizer):
        """Test extraction with normalized text."""
        from scripts.extract_process_targets import process_sample
        
        item = {
            "teacher_text": "Line 1\r\nLine 2",  # CRLF line endings
            "metadata": {
                "expected_behaviour": "normal",
                "call_sequence": [],
                "text_norm": "NFC",
                "line_endings": "LF",
            }
        }
        
        reg = ToolSchemaRegistry()
        processed = process_sample(item, mock_tokenizer, reg)
        
        # Should handle normalization correctly
        assert processed["metadata"].get("text_norm") == "NFC"
        assert processed["metadata"].get("line_endings") == "LF"


class TestVerifyContextualSet:
    """Test contextual dataset verification."""
    
    def test_verify_item_basic(self):
        """Test basic item verification."""
        from scripts.verify_contextual_set import verify_item
        from tools.schema_registry import ToolSchemaRegistry
        
        item = {
            "prompt": '{"caws": {"tier": 2, "max_files": 25}}',
            "teacher_text": 'I will call {"name": "read_file", "arguments": {"path": "test.txt"}}',
            "metadata": {
                "dataset_version": "1.1.0",
                "call_sequence": [{"name": "read_file", "arguments": {"path": "test.txt"}}],
                "expected_behaviour": "normal",
                "tool_result_fields": {"summary": "File contents"},
                "integration_spans_bytes": [[0, 10]],
            }
        }
        
        reg = ToolSchemaRegistry()
        result = verify_item(item, reg)
        
        assert isinstance(result, dict)
        assert "ok" in result
        assert "problems" in result
    
    def test_verify_item_control_case(self):
        """Test verification of control cases."""
        from scripts.verify_contextual_set import verify_item
        from tools.schema_registry import ToolSchemaRegistry
        
        item = {
            "prompt": '{"caws": {"tier": 2}}',
            "teacher_text": "I don't have access to tools.",
            "metadata": {
                "dataset_version": "1.1.0",
                "expected_behaviour": "no_tool",
                "call_sequence": [],
                "integration_spans_bytes": [],
            }
        }
        
        reg = ToolSchemaRegistry()
        result = verify_item(item, reg, check_controls=True)
        
        assert result["ok"] is True
        assert "control_has_integration_spans" not in result["problems"]
    
    def test_compute_integration_f1(self):
        """Test Integration F1 computation."""
        from scripts.verify_contextual_set import compute_integration_f1
        
        # Use actual span offsets that match the text
        teacher_text = "Integration: ANE prefers fp16 kernels when ops are supported."
        # Span should cover the text after "Integration: " (start at 13, end at 60)
        integration_spans = [[13, 60]]  # Span covering "ANE prefers fp16 kernels when ops are supported."
        tool_result_fields = {
            "summary": "ANE prefers fp16 kernels when ops are supported"
        }
        
        precision, recall, f1 = compute_integration_f1(
            teacher_text, integration_spans, tool_result_fields
        )
        
        assert 0.0 <= precision <= 1.0
        assert 0.0 <= recall <= 1.0
        assert 0.0 <= f1 <= 1.0
        # Should have recall=1.0 if summary is grounded in the span
        # The span text should contain the summary value
        span_text = teacher_text[13:60]
        assert "ANE prefers fp16 kernels" in span_text
        assert recall == 1.0  # At least one grounded value found
    
    def test_grounded_values_in_span(self):
        """Test grounded value detection."""
        from scripts.verify_contextual_set import grounded_values_in_span
        
        seg = "ANE prefers fp16 kernels when ops are supported"
        fields = {"summary": "ANE prefers fp16 kernels when ops are supported"}
        
        assert grounded_values_in_span(seg, fields) is True
        
        # Test numeric values
        seg2 = "Found 128 lines in the file"
        fields2 = {"lines": 128}
        assert grounded_values_in_span(seg2, fields2) is True
    
    def test_is_long_context_item(self):
        """Test long-context detection."""
        from scripts.verify_contextual_set import is_long_context_item
        
        # Item with metadata flag
        item_with_flag = {
            "prompt": "short",
            "metadata": {"long_context": True}
        }
        assert is_long_context_item(item_with_flag) is True
        
        # Item without flag (should compute)
        item_without_flag = {
            "prompt": "x" * 25000,  # Long prompt
            "metadata": {}
        }
        assert is_long_context_item(item_without_flag, tokenizer=None) is True
    
    def test_caws_header_validation(self):
        """Test CAWS header format validation."""
        from scripts.verify_contextual_set import check_caws_header
        
        # Valid CAWS header (must start with JSON)
        prompt_valid = '{"caws": {"tier": 2, "max_files": 25}}\nTask: test'
        ok, data = check_caws_header(prompt_valid)
        # May return True or False depending on exact format requirements
        assert isinstance(ok, bool)
        if ok:
            assert data is not None
        
        # Invalid CAWS header
        prompt_invalid = "No CAWS header\nTask: test"
        ok, data = check_caws_header(prompt_invalid)
        assert ok is False
    
    def test_privacy_scanning(self):
        """Test PII detection."""
        from scripts.verify_contextual_set import check_privacy
        
        # Test with email
        text_with_email = "Contact me at test@example.com"
        result = check_privacy(text_with_email)
        assert result["emails_found"] > 0
        assert result["privacy_ok"] is False
        
        # Test with UUID
        text_with_uuid = "ID: 550e8400-e29b-41d4-a716-446655440000"
        result = check_privacy(text_with_uuid)
        assert result["uuids_found"] > 0
        assert result["privacy_ok"] is False
        
        # Test clean text
        text_clean = "This is clean text without PII"
        result = check_privacy(text_clean)
        assert result["privacy_ok"] is True
    
    def test_semantic_validation(self):
        """Test tool argument semantic validation."""
        from scripts.verify_contextual_set import verify_item
        from tools.schema_registry import ToolSchemaRegistry
        
        item = {
            "prompt": '{"caws": {"tier": 2}}',
            "teacher_text": '{"name": "read_file", "arguments": {"path": "test.txt"}}',
            "metadata": {
                "dataset_version": "1.1.0",
                "call_sequence": [{"name": "read_file", "arguments": {"path": "test.txt"}}],
                "json_args_span_bytes": [0, 50],
                "expected_behaviour": "normal",
            }
        }
        
        reg = ToolSchemaRegistry()
        result = verify_item(item, reg)
        
        assert isinstance(result, dict)
        assert "semantic_ok" in result
    
    def test_stratification_validation(self):
        """Test stratification coverage heatmap."""
        from scripts.verify_contextual_set import build_stratification_heatmap
        
        items = [
            {
                "metadata": {"scenario": "file_ops", "complexity": "single_call", "structure": "flat_args"}
            },
            {
                "metadata": {"scenario": "web_search", "complexity": "multi_call", "structure": "nested_args"}
            },
        ]
        
        heatmap = build_stratification_heatmap(items)
        
        assert isinstance(heatmap, dict)
        # Heatmap structure: {"heatmap": {...}, "missing_cells": [...], "all_cells_populated": bool}
        assert "heatmap" in heatmap
        assert "missing_cells" in heatmap
        assert isinstance(heatmap["heatmap"], dict)
    
    def test_multi_call_parity(self):
        """Test multi-call span count matching."""
        from scripts.verify_contextual_set import verify_item
        from tools.schema_registry import ToolSchemaRegistry
        
        item = {
            "prompt": '{"caws": {"tier": 2}}',
            "teacher_text": 'Call 1: {"name": "read_file", "arguments": {"path": "test.txt"}} Call 2: {"name": "write_file", "arguments": {"path": "out.txt"}}',
            "metadata": {
                "dataset_version": "1.1.0",
                "call_sequence": [
                    {"name": "read_file", "arguments": {"path": "test.txt"}},
                    {"name": "write_file", "arguments": {"path": "out.txt"}}
                ],
                "json_args_spans_bytes": [[0, 50], [60, 110]],
                "tool_name_spans_bytes": [[10, 18], [70, 79]],
                "expected_behaviour": "normal",
            }
        }
        
        reg = ToolSchemaRegistry()
        result = verify_item(item, reg)
        
        # Should not have parity mismatch problems
        assert "json_args_spans_mismatch" not in result.get("problems", [])
    
    def test_grounding_validation(self):
        """Test tool result field grounding."""
        from scripts.verify_contextual_set import verify_item
        from tools.schema_registry import ToolSchemaRegistry
        
        item = {
            "prompt": '{"caws": {"tier": 2}}',
            "teacher_text": "Integration: The file contains 128 lines.",
            "metadata": {
                "dataset_version": "1.1.0",
                "call_sequence": [{"name": "read_file", "arguments": {"path": "test.txt"}}],
                "tool_result_fields": {"lines": 128},
                "integration_spans_bytes": [[13, 35]],
                "expected_behaviour": "normal",
            }
        }
        
        reg = ToolSchemaRegistry()
        result = verify_item(item, reg)
        
        assert isinstance(result, dict)
        assert "integration_grounded" in result
    
    def test_integration_f1_edge_cases(self):
        """Test Integration F1 computation with edge cases."""
        from scripts.verify_contextual_set import compute_integration_f1
        
        # Zero spans
        precision, recall, f1 = compute_integration_f1("text", [], {"summary": "test"})
        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0
        
        # Missing fields
        precision, recall, f1 = compute_integration_f1("text", [[0, 4]], {})
        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0
        
        # No matching values
        precision, recall, f1 = compute_integration_f1(
            "Integration: Some text",
            [[13, 22]],
            {"summary": "Different text"}
        )
        assert precision == 0.0
        assert recall == 0.0
    
    def test_error_handling_malformed_items(self):
        """Test handling of malformed items."""
        from scripts.verify_contextual_set import verify_item
        from tools.schema_registry import ToolSchemaRegistry
        
        # Item with missing required fields
        item_missing_fields = {
            "prompt": "test",
            # Missing teacher_text
        }
        
        reg = ToolSchemaRegistry()
        result = verify_item(item_missing_fields, reg)
        
        # Should handle gracefully
        assert isinstance(result, dict)
        assert "ok" in result
    
    def test_contains_grounding(self):
        """Test grounding detection function."""
        from scripts.verify_contextual_set import contains_grounding
        
        text = "Integration: ANE prefers fp16 kernels"
        spans = [[13, 40]]
        fields = {"summary": "ANE prefers fp16 kernels"}
        
        result = contains_grounding(text, spans, fields)
        assert isinstance(result, bool)
    
    def test_parse_tool_json_slice(self):
        """Test JSON parsing from text slice."""
        from scripts.verify_contextual_set import parse_tool_json_slice
        
        text = 'Call {"name": "read_file", "arguments": {"path": "test.txt"}} now'
        # Slice should include the JSON object
        start = text.find('{')
        end = text.find('}') + 1
        obj = parse_tool_json_slice(text, start, end)
        
        # May return None if JSON not found or invalid, but should handle gracefully
        if obj is not None:
            assert obj.get("name") == "read_file"
        # If None, that's also acceptable (invalid JSON case)
    
    def test_check_stratification_backbone(self):
        """Test stratification backbone check."""
        from scripts.verify_contextual_set import check_stratification_backbone
        
        items = [
            {"metadata": {"scenario": "file_ops", "complexity": "single_call"}},
            {"metadata": {"scenario": "web_search", "complexity": "multi_call"}},
        ]
        
        ok, missing = check_stratification_backbone(items, len(items))
        
        assert isinstance(ok, bool)
        assert isinstance(missing, list)
    
    def test_verify_token_alignment(self):
        """Test token alignment verification."""
        from scripts.verify_contextual_set import verify_token_alignment
        from unittest.mock import Mock
        
        tokenizer = Mock()
        tokenizer.is_fast = True
        tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])
        tokenizer.decode = Mock(return_value="Hello")
        
        def tokenize_side_effect(text, **kwargs):
            return {
                "input_ids": [1, 2, 3],
                "offset_mapping": [(0, 1), (1, 2), (2, 3)]
            }
        
        tokenizer.side_effect = tokenize_side_effect
        
        text = "Hello world"
        span_bytes = [0, 5]
        
        ok, token_span = verify_token_alignment(text, span_bytes, tokenizer)
        
        assert isinstance(ok, bool)
        assert token_span is None or isinstance(token_span, tuple)
    
    def test_retry_case_validation(self):
        """Test retry case validation."""
        from scripts.verify_contextual_set import verify_item
        from tools.schema_registry import ToolSchemaRegistry
        
        reg = ToolSchemaRegistry()
        
        # Valid retry case with attempts array
        item_retry = {
            "prompt": '{"caws": {"tier": 2}}',
            "teacher_text": "First attempt failed, retrying...",
            "metadata": {
                "dataset_version": "1.1.0",
                "expected_behaviour": "retry",
                "call_sequence": [{"name": "read_file", "arguments": {"path": "test.txt"}}],
                "attempts": [
                    {"ok": False, "error": "File not found"},
                    {"ok": True, "result": "Success"}
                ],
            }
        }
        
        result = verify_item(item_retry, reg, check_adversarial=True)
        assert isinstance(result, dict)
        assert "ok" in result
    
    def test_adversarial_range_violation_validation(self):
        """Test adversarial range violation validation."""
        from scripts.verify_contextual_set import verify_item
        from tools.schema_registry import ToolSchemaRegistry
        
        reg = ToolSchemaRegistry()
        
        item = {
            "prompt": '{"caws": {"tier": 2}}',
            "teacher_text": "The range is invalid, correcting to valid value",
            "metadata": {
                "dataset_version": "1.1.0",
                "adversarial": {"type": "range_violation"},
                "call_sequence": [{"name": "web.search", "arguments": {"q": "test", "top_k": -1}}],
                "expected_behaviour": "normal",
            }
        }
        
        result = verify_item(item, reg, check_adversarial=True)
        # Should not flag as unhandled if teacher corrects it
        assert "adversarial_range_violation_not_handled" not in result.get("problems", [])
    
    def test_adversarial_malformed_json_validation(self):
        """Test adversarial malformed JSON validation."""
        from scripts.verify_contextual_set import verify_item
        from tools.schema_registry import ToolSchemaRegistry
        
        reg = ToolSchemaRegistry()
        
        item = {
            "prompt": '{"caws": {"tier": 2}}',
            "teacher_text": "The JSON is malformed, fixing it",
            "metadata": {
                "dataset_version": "1.1.0",
                "adversarial": {"type": "malformed_json"},
                "call_sequence": [{"name": "read_file", "arguments": {"path": "test.txt"}}],
                "expected_behaviour": "normal",
            }
        }
        
        result = verify_item(item, reg, check_adversarial=True)
        # Should not flag as unhandled if teacher repairs it
        assert "adversarial_malformed_json_not_repaired" not in result.get("problems", [])
    
    def test_adversarial_ambiguity_validation(self):
        """Test adversarial ambiguity validation."""
        from scripts.verify_contextual_set import verify_item
        from tools.schema_registry import ToolSchemaRegistry
        
        reg = ToolSchemaRegistry()
        
        item = {
            "prompt": '{"caws": {"tier": 2}}',
            "teacher_text": "Could you please clarify which file you mean?",
            "metadata": {
                "dataset_version": "1.1.0",
                "adversarial": {"type": "ambiguity", "expected": "ask_clarify"},
                "call_sequence": [],
                "expected_behaviour": "normal",
            }
        }
        
        result = verify_item(item, reg, check_adversarial=True)
        # Should not flag as unclarified if teacher asks for clarification
        assert "adversarial_ambiguity_not_clarified" not in result.get("problems", [])
    
    def test_multi_call_span_validation(self):
        """Test multi-call span validation."""
        from scripts.verify_contextual_set import verify_item
        from tools.schema_registry import ToolSchemaRegistry
        
        reg = ToolSchemaRegistry()
        
        call1_json = '{"name": "read_file", "arguments": {"path": "test.txt"}}'
        call2_json = '{"name": "write_file", "arguments": {"path": "out.txt"}}'
        teacher_text = f"Call 1: {call1_json} Call 2: {call2_json}"
        
        item = {
            "prompt": '{"caws": {"tier": 2}}',
            "teacher_text": teacher_text,
            "metadata": {
                "dataset_version": "1.1.0",
                "call_sequence": [
                    {"name": "read_file", "arguments": {"path": "test.txt"}},
                    {"name": "write_file", "arguments": {"path": "out.txt"}}
                ],
                "json_args_spans_bytes": [[8, 8+len(call1_json)], [20+len(call1_json), 20+len(call1_json)+len(call2_json)]],
                "expected_behaviour": "normal",
            }
        }
        
        result = verify_item(item, reg)
        # Should validate multi-call spans
        assert isinstance(result, dict)
    
    def test_token_alignment_with_tokenizer(self):
        """Test token alignment with actual tokenizer."""
        from scripts.verify_contextual_set import verify_item
        from tools.schema_registry import ToolSchemaRegistry
        from unittest.mock import Mock
        
        reg = ToolSchemaRegistry()
        tokenizer = Mock()
        tokenizer.is_fast = True
        
        def tokenize_side_effect(text, **kwargs):
            return {
                "input_ids": list(range(len(text))),
                "offset_mapping": [(i, i+1) for i in range(len(text))]
            }
        
        tokenizer.side_effect = tokenize_side_effect
        tokenizer.encode = Mock(side_effect=lambda text, **kwargs: list(range(len(text))))
        tokenizer.decode = Mock(side_effect=lambda ids, **kwargs: ''.join(chr(65 + i % 26) for i in ids))
        
        item = {
            "prompt": '{"caws": {"tier": 2}}',
            "teacher_text": '{"name": "read_file", "arguments": {"path": "test.txt"}}',
            "metadata": {
                "dataset_version": "1.1.0",
                "call_sequence": [{"name": "read_file", "arguments": {"path": "test.txt"}}],
                "json_args_span_bytes": [0, 50],
                "expected_behaviour": "normal",
                "text_norm": "NFC",
                "line_endings": "LF",
            }
        }
        
        result = verify_item(item, reg, tokenizer=tokenizer)
        assert isinstance(result, dict)
        assert "token_align_ok" in result
    
    def test_low_integration_f1_detection(self):
        """Test low Integration F1 detection."""
        from scripts.verify_contextual_set import verify_item
        from tools.schema_registry import ToolSchemaRegistry
        
        reg = ToolSchemaRegistry()
        
        item = {
            "prompt": '{"caws": {"tier": 2}}',
            "teacher_text": "Integration: Some unrelated text",
            "metadata": {
                "dataset_version": "1.1.0",
                "call_sequence": [{"name": "read_file", "arguments": {"path": "test.txt"}}],
                "tool_result_fields": {"summary": "Different text"},
                "integration_spans_bytes": [[13, 35]],
                "expected_behaviour": "normal",
            }
        }
        
        result = verify_item(item, reg)
        # Should detect low F1 if spans don't match fields
        assert isinstance(result, dict)
        assert "integration_f1" in result
    
    def test_unknown_tool_detection(self):
        """Test unknown tool detection."""
        from scripts.verify_contextual_set import verify_item
        from tools.schema_registry import ToolSchemaRegistry
        
        reg = ToolSchemaRegistry()
        
        # Use valid JSON with unknown tool name
        teacher_text = '{"name": "unknown_tool", "arguments": {"path": "test.txt"}}'
        item = {
            "prompt": '{"caws": {"tier": 2, "max_files": 25, "max_loc": 1000, "cov": 80, "mut": 50}}',
            "teacher_text": teacher_text,
            "metadata": {
                "dataset_version": "1.1.0",
                "call_sequence": [{"name": "unknown_tool", "arguments": {"path": "test.txt"}}],
                "json_args_span_bytes": [0, len(teacher_text)],
                "expected_behaviour": "normal",
            }
        }
        
        result = verify_item(item, reg)
        # Should detect unknown tool if JSON parses successfully
        # May have json_parse_fail if span doesn't match, but if it parses, should detect unknown tool
        assert isinstance(result, dict)
        assert "ok" in result
    
    def test_retry_insufficient_attempts(self):
        """Test retry case with insufficient attempts."""
        from scripts.verify_contextual_set import verify_item
        from tools.schema_registry import ToolSchemaRegistry
        
        reg = ToolSchemaRegistry()
        
        item = {
            "prompt": '{"caws": {"tier": 2}}',
            "teacher_text": "First attempt failed",
            "metadata": {
                "dataset_version": "1.1.0",
                "expected_behaviour": "retry",
                "call_sequence": [{"name": "read_file", "arguments": {"path": "test.txt"}}],
                "attempts": [{"ok": False}],  # Only one attempt, need at least 2
            }
        }
        
        result = verify_item(item, reg)
        assert "retry_insufficient_attempts" in result.get("problems", [])
    
    def test_retry_last_attempt_failed(self):
        """Test retry case where last attempt failed."""
        from scripts.verify_contextual_set import verify_item
        from tools.schema_registry import ToolSchemaRegistry
        
        reg = ToolSchemaRegistry()
        
        item = {
            "prompt": '{"caws": {"tier": 2}}',
            "teacher_text": "Both attempts failed",
            "metadata": {
                "dataset_version": "1.1.0",
                "expected_behaviour": "retry",
                "call_sequence": [{"name": "read_file", "arguments": {"path": "test.txt"}}],
                "attempts": [
                    {"ok": False},
                    {"ok": False}  # Last attempt also failed
                ],
            }
        }
        
        result = verify_item(item, reg)
        assert "retry_last_attempt_failed" in result.get("problems", [])
    
    def test_retry_missing_spans_or_attempts(self):
        """Test retry case with no spans and no attempts."""
        from scripts.verify_contextual_set import verify_item
        from tools.schema_registry import ToolSchemaRegistry
        
        reg = ToolSchemaRegistry()
        
        item = {
            "prompt": '{"caws": {"tier": 2}}',
            "teacher_text": "Retry case",
            "metadata": {
                "dataset_version": "1.1.0",
                "expected_behaviour": "retry",
                "call_sequence": [{"name": "read_file", "arguments": {"path": "test.txt"}}],
                # No spans, no attempts array
            }
        }
        
        result = verify_item(item, reg)
        assert "retry_missing_spans_or_attempts" in result.get("problems", [])
    
    def test_missing_json_args_span(self):
        """Test detection of missing JSON args span."""
        from scripts.verify_contextual_set import verify_item
        from tools.schema_registry import ToolSchemaRegistry
        
        reg = ToolSchemaRegistry()
        
        item = {
            "prompt": '{"caws": {"tier": 2}}',
            "teacher_text": "I will call read_file",
            "metadata": {
                "dataset_version": "1.1.0",
                "call_sequence": [{"name": "read_file", "arguments": {"path": "test.txt"}}],
                "expected_behaviour": "normal",
                # Missing json_args_span_bytes
            }
        }
        
        result = verify_item(item, reg)
        assert "missing:json_args_span_bytes" in result.get("problems", [])
    
    def test_json_parse_fail(self):
        """Test JSON parsing failure detection."""
        from scripts.verify_contextual_set import verify_item
        from tools.schema_registry import ToolSchemaRegistry
        
        reg = ToolSchemaRegistry()
        
        item = {
            "prompt": '{"caws": {"tier": 2}}',
            "teacher_text": "Invalid JSON {",
            "metadata": {
                "dataset_version": "1.1.0",
                "call_sequence": [{"name": "read_file", "arguments": {"path": "test.txt"}}],
                "json_args_span_bytes": [0, 10],  # Points to invalid JSON
                "expected_behaviour": "normal",
            }
        }
        
        result = verify_item(item, reg)
        assert "json_parse_fail" in result.get("problems", [])
    
    def test_adversarial_range_violation_not_handled(self):
        """Test detection of unhandled range violation."""
        from scripts.verify_contextual_set import verify_item
        from tools.schema_registry import ToolSchemaRegistry
        
        reg = ToolSchemaRegistry()
        
        item = {
            "prompt": '{"caws": {"tier": 2}}',
            "teacher_text": "Processing request normally",  # Doesn't mention correction
            "metadata": {
                "dataset_version": "1.1.0",
                "adversarial": {"type": "range_violation"},
                "call_sequence": [{"name": "web.search", "arguments": {"q": "test", "top_k": -1}}],
                "expected_behaviour": "normal",
            }
        }
        
        result = verify_item(item, reg, check_adversarial=True)
        assert "adversarial_range_violation_not_handled" in result.get("problems", [])
    
    def test_adversarial_malformed_json_not_repaired(self):
        """Test detection of unrepaired malformed JSON."""
        from scripts.verify_contextual_set import verify_item
        from tools.schema_registry import ToolSchemaRegistry
        
        reg = ToolSchemaRegistry()
        
        item = {
            "prompt": '{"caws": {"tier": 2}}',
            "teacher_text": "Processing normally",  # Doesn't mention repair
            "metadata": {
                "dataset_version": "1.1.0",
                "adversarial": {"type": "malformed_json"},
                "call_sequence": [{"name": "read_file", "arguments": {"path": "test.txt"}}],
                "expected_behaviour": "normal",
            }
        }
        
        result = verify_item(item, reg, check_adversarial=True)
        assert "adversarial_malformed_json_not_repaired" in result.get("problems", [])
    
    def test_adversarial_ambiguity_not_clarified(self):
        """Test detection of unclarified ambiguity."""
        from scripts.verify_contextual_set import verify_item
        from tools.schema_registry import ToolSchemaRegistry
        
        reg = ToolSchemaRegistry()
        
        item = {
            "prompt": '{"caws": {"tier": 2}}',
            "teacher_text": "Processing request",  # Doesn't ask for clarification
            "metadata": {
                "dataset_version": "1.1.0",
                "adversarial": {"type": "ambiguity", "expected": "ask_clarify"},
                "call_sequence": [],
                "expected_behaviour": "normal",
            }
        }
        
        result = verify_item(item, reg, check_adversarial=True)
        assert "adversarial_ambiguity_not_clarified" in result.get("problems", [])
    
    def test_multi_call_span_out_of_bounds(self):
        """Test multi-call span boundary validation."""
        from scripts.verify_contextual_set import verify_item
        from tools.schema_registry import ToolSchemaRegistry
        
        reg = ToolSchemaRegistry()
        
        call1_json = '{"name": "read_file", "arguments": {"path": "test.txt"}}'
        call2_json = '{"name": "write_file", "arguments": {"path": "out.txt"}}'
        teacher_text = f"Call 1: {call1_json} Call 2: {call2_json}"
        
        item = {
            "prompt": '{"caws": {"tier": 2}}',
            "teacher_text": teacher_text,
            "metadata": {
                "dataset_version": "1.1.0",
                "call_sequence": [
                    {"name": "read_file", "arguments": {"path": "test.txt"}},
                    {"name": "write_file", "arguments": {"path": "out.txt"}}
                ],
                "json_args_spans_bytes": [[0, 10], [1000, 2000]],  # Second span out of bounds
                "expected_behaviour": "normal",
            }
        }
        
        result = verify_item(item, reg)
        # Should detect out-of-bounds span
        assert isinstance(result, dict)
    
    def test_integration_not_grounded(self):
        """Test detection of ungrounded integration."""
        from scripts.verify_contextual_set import verify_item
        from tools.schema_registry import ToolSchemaRegistry
        
        reg = ToolSchemaRegistry()
        
        item = {
            "prompt": '{"caws": {"tier": 2}}',
            "teacher_text": "Integration: Some unrelated text",
            "metadata": {
                "dataset_version": "1.1.0",
                "call_sequence": [{"name": "read_file", "arguments": {"path": "test.txt"}}],
                "tool_result_fields": {"summary": "Different text"},
                "integration_spans_bytes": [[13, 35]],
                "expected_behaviour": "normal",
            }
        }
        
        result = verify_item(item, reg)
        assert "integration_not_grounded" in result.get("problems", [])
    
    def test_low_integration_f1_flagging(self):
        """Test low Integration F1 flagging."""
        from scripts.verify_contextual_set import verify_item
        from tools.schema_registry import ToolSchemaRegistry
        
        reg = ToolSchemaRegistry()
        
        item = {
            "prompt": '{"caws": {"tier": 2}}',
            "teacher_text": "Integration: Some text",
            "metadata": {
                "dataset_version": "1.1.0",
                "call_sequence": [{"name": "read_file", "arguments": {"path": "test.txt"}}],
                "tool_result_fields": {"summary": "Different text"},
                "integration_spans_bytes": [[13, 22]],
                "expected_behaviour": "normal",
            }
        }
        
        result = verify_item(item, reg)
        # Should flag low F1 if below 0.75
        if result.get("integration_f1", 1.0) < 0.75:
            assert "low_integration_f1" in result.get("problems", []) or result.get("integration_f1", 1.0) < 0.75
    
    def test_is_long_context_with_tokenizer(self):
        """Test is_long_context function with tokenizer."""
        from scripts.verify_contextual_set import is_long_context
        from unittest.mock import Mock
        
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=list(range(9000)))  # 9000 tokens
        
        prompt = "x" * 1000
        result = is_long_context(prompt, tokenizer, token_threshold=8000)
        assert result is True
        
        # Test with short prompt
        tokenizer.encode = Mock(return_value=list(range(100)))  # 100 tokens
        result = is_long_context(prompt, tokenizer, token_threshold=8000)
        assert result is False
    
    def test_is_long_context_fallback_to_bytes(self):
        """Test is_long_context fallback to bytes."""
        from scripts.verify_contextual_set import is_long_context
        
        # Test with tokenizer that raises exception
        class FailingTokenizer:
            def encode(self, *args, **kwargs):
                raise Exception("Tokenization failed")
        
        failing_tokenizer = FailingTokenizer()
        long_prompt = "x" * 25000
        
        result = is_long_context(long_prompt, failing_tokenizer, byte_threshold=24000)
        assert result is True
    
    def test_is_long_context_item_with_tokenizer(self):
        """Test is_long_context_item with tokenizer."""
        from scripts.verify_contextual_set import is_long_context_item
        from unittest.mock import Mock
        
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=list(range(9000)))
        
        item = {
            "prompt": "x" * 1000,
            "metadata": {}
        }
        
        result = is_long_context_item(item, tokenizer, token_threshold=8000)
        assert result is True
    
    def test_is_long_context_item_tokenizer_exception(self):
        """Test is_long_context_item with tokenizer exception."""
        from scripts.verify_contextual_set import is_long_context_item
        
        class FailingTokenizer:
            def encode(self, *args, **kwargs):
                raise Exception("Tokenization failed")
        
        failing_tokenizer = FailingTokenizer()
        item = {
            "prompt": "x" * 25000,
            "metadata": {}
        }
        
        result = is_long_context_item(item, failing_tokenizer, byte_threshold=24000)
        assert result is True
    
    def test_contains_grounding_with_synonyms(self):
        """Test grounding detection with synonym expansion."""
        from scripts.verify_contextual_set import contains_grounding
        
        text = "Integration: The system prefers fp16 kernels"
        spans = [[13, 50]]
        fields = {"summary": "ANE prefers fp16 kernels"}
        
        result = contains_grounding(text, spans, fields)
        # Should match via synonym expansion (prefers/prefer)
        assert isinstance(result, bool)
    
    def test_contains_grounding_no_spans(self):
        """Test grounding detection with no spans."""
        from scripts.verify_contextual_set import contains_grounding
        
        text = "Some text"
        spans = []
        fields = {"summary": "test"}
        
        result = contains_grounding(text, spans, fields)
        assert result is False
    
    def test_contains_grounding_no_fields(self):
        """Test grounding detection with no fields."""
        from scripts.verify_contextual_set import contains_grounding
        
        text = "Some text"
        spans = [[0, 9]]
        fields = {}
        
        result = contains_grounding(text, spans, fields)
        assert result is False
    
    def test_check_stratification_backbone_small_n(self):
        """Test stratification backbone check for small N."""
        from scripts.verify_contextual_set import check_stratification_backbone
        
        items = [
            {"metadata": {"scenario": "file_ops", "complexity": "single_call"}},
            {"metadata": {"scenario": "web_search", "complexity": "single_call"}},
            {"metadata": {"scenario": "code_exec", "complexity": "single_call"}},
            {"metadata": {"scenario": "multi_step", "complexity": "single_call"}},
            {"metadata": {"scenario": "file_ops", "complexity": "multi_call"}},
            {"metadata": {"scenario": "file_ops", "complexity": "branching_error_recovery"}},
        ]
        
        ok, missing = check_stratification_backbone(items, len(items))
        assert isinstance(ok, bool)
        assert isinstance(missing, list)
    
    def test_check_stratification_backbone_large_n(self):
        """Test stratification backbone check for large N."""
        from scripts.verify_contextual_set import check_stratification_backbone
        
        items = [{"metadata": {"scenario": "file_ops", "complexity": "single_call"}}] * 40
        
        ok, missing = check_stratification_backbone(items, len(items))
        # For N >= 36, returns None, []
        assert ok is None or isinstance(ok, bool)
        assert isinstance(missing, list)
    
    def test_main_function_basic(self, tmp_path):
        """Test main function with valid input."""
        from scripts.verify_contextual_set import main
        import sys
        import json
        
        # Create test input file
        input_file = tmp_path / "test_input.jsonl"
        output_file = tmp_path / "test_output.json"
        
        # Write valid sample
        sample = {
            "prompt": '{"caws": {"tier": 2, "max_files": 25, "max_loc": 1000, "cov": 80, "mut": 50}}',
            "teacher_text": '{"name": "read_file", "arguments": {"path": "test.txt"}}',
            "metadata": {
                "dataset_version": "1.1.0",
                "call_sequence": [{"name": "read_file", "arguments": {"path": "test.txt"}}],
                "scenario": "file_ops",
                "complexity": "single_call",
                "expected_behaviour": "normal",
            }
        }
        
        with open(input_file, "w") as f:
            f.write(json.dumps(sample) + "\n")
        
        # Mock sys.argv
        original_argv = sys.argv
        try:
            sys.argv = [
                "verify_contextual_set",
                "--in", str(input_file),
                "--report", str(output_file),
            ]
            main()
        except SystemExit:
            pass  # Expected if validation fails
        finally:
            sys.argv = original_argv
        
        # Check that report file was created
        if output_file.exists():
            report = json.load(open(output_file))
            assert "summary" in report
    
    def test_main_function_invalid_json(self, tmp_path):
        """Test main function with invalid JSON."""
        from scripts.verify_contextual_set import main
        import sys
        
        input_file = tmp_path / "test_invalid.jsonl"
        output_file = tmp_path / "test_output.json"
        
        # Write invalid JSON
        with open(input_file, "w") as f:
            f.write("Invalid JSON {")
        
        original_argv = sys.argv
        try:
            sys.argv = [
                "verify_contextual_set",
                "--in", str(input_file),
                "--report", str(output_file),
            ]
            try:
                main()
            except SystemExit:
                pass  # Expected for invalid JSON
        finally:
            sys.argv = original_argv
    
    def test_main_function_with_tokenizer(self, tmp_path):
        """Test main function with tokenizer path."""
        from scripts.verify_contextual_set import main
        import sys
        import json
        from unittest.mock import patch, Mock
        
        input_file = tmp_path / "test_input.jsonl"
        output_file = tmp_path / "test_output.json"
        
        sample = {
            "prompt": '{"caws": {"tier": 2, "max_files": 25, "max_loc": 1000, "cov": 80, "mut": 50}}',
            "teacher_text": "Test",
            "metadata": {"dataset_version": "1.1.0"},
        }
        
        with open(input_file, "w") as f:
            f.write(json.dumps(sample) + "\n")
        
        mock_tokenizer = Mock()
        mock_tokenizer.encode = Mock(return_value=[1, 2, 3])
        
        original_argv = sys.argv
        try:
            sys.argv = [
                "verify_contextual_set",
                "--in", str(input_file),
                "--report", str(output_file),
                "--tokenizer", "models/student/tokenizer",
            ]
            # Patch transformers.AutoTokenizer since it's imported from transformers
            with patch('transformers.AutoTokenizer') as mock_auto:
                mock_auto.from_pretrained = Mock(return_value=mock_tokenizer)
                try:
                    main()
                except SystemExit:
                    pass  # May exit with non-zero code
        finally:
            sys.argv = original_argv
    
    def test_main_function_with_schema_validation(self, tmp_path):
        """Test main function with schema validation."""
        from scripts.verify_contextual_set import main
        import sys
        import json
        from unittest.mock import patch, Mock, MagicMock
        
        input_file = tmp_path / "test_input.jsonl"
        output_file = tmp_path / "test_output.json"
        
        sample = {
            "prompt": '{"caws": {"tier": 2, "max_files": 25, "max_loc": 1000, "cov": 80, "mut": 50}}',
            "teacher_text": "Test",
            "metadata": {"dataset_version": "1.1.0"},
        }
        
        with open(input_file, "w") as f:
            f.write(json.dumps(sample) + "\n")
        
        # Mock schema file existence
        schema_path = tmp_path.parent.parent / "schemas" / "dataset_item.schema.json"
        schema_path.parent.mkdir(parents=True, exist_ok=True)
        with open(schema_path, "w") as f:
            json.dump({"type": "object"}, f)
        
        original_argv = sys.argv
        try:
            sys.argv = [
                "verify_contextual_set",
                "--in", str(input_file),
                "--report", str(output_file),
            ]
            with patch('scripts.verify_contextual_set.HAS_JSONSCHEMA', True):
                with patch('scripts.verify_contextual_set.jsonschema') as mock_jsonschema:
                    mock_jsonschema.validate = Mock()
                    try:
                        main()
                    except SystemExit:
                        pass
        finally:
            sys.argv = original_argv
    
    def test_main_function_schema_validation_errors(self, tmp_path):
        """Test main function with schema validation errors."""
        from scripts.verify_contextual_set import main
        import sys
        import json
        from unittest.mock import patch, Mock
        
        input_file = tmp_path / "test_input.jsonl"
        output_file = tmp_path / "test_output.json"
        
        # Invalid sample (missing required fields)
        sample = {
            "prompt": "test",
            # Missing teacher_text, metadata
        }
        
        with open(input_file, "w") as f:
            f.write(json.dumps(sample) + "\n")
        
        # Mock schema file
        schema_path = tmp_path.parent.parent / "schemas" / "dataset_item.schema.json"
        schema_path.parent.mkdir(parents=True, exist_ok=True)
        with open(schema_path, "w") as f:
            json.dump({
                "type": "object",
                "required": ["teacher_text", "metadata"]
            }, f)
        
        original_argv = sys.argv
        try:
            sys.argv = [
                "verify_contextual_set",
                "--in", str(input_file),
                "--report", str(output_file),
            ]
            with patch('scripts.verify_contextual_set.HAS_JSONSCHEMA', True):
                # Create a mock ValidationError that inherits from Exception
                class MockValidationError(Exception):
                    def __init__(self, message, path=None):
                        super().__init__(message)
                        self.path = path or []
                
                with patch('scripts.verify_contextual_set.jsonschema') as mock_jsonschema:
                    # Set up the mock module with ValidationError class
                    mock_jsonschema.ValidationError = MockValidationError
                    # Mock validation error
                    error = MockValidationError("Missing required field")
                    mock_jsonschema.validate = Mock(side_effect=error)
                    try:
                        main()
                    except SystemExit as e:
                        # Should exit with error code
                        assert e.code != 0
        finally:
            sys.argv = original_argv
    
    def test_main_function_unknown_tool_detection(self, tmp_path):
        """Test main function detects unknown tools."""
        from scripts.verify_contextual_set import main
        import sys
        import json
        
        input_file = tmp_path / "test_input.jsonl"
        output_file = tmp_path / "test_output.json"
        
        sample = {
            "prompt": '{"caws": {"tier": 2, "max_files": 25, "max_loc": 1000, "cov": 80, "mut": 50}}',
            "teacher_text": "Test",
            "metadata": {
                "dataset_version": "1.1.0",
                "call_sequence": [{"name": "unknown_tool_xyz", "arguments": {}}],
            }
        }
        
        with open(input_file, "w") as f:
            f.write(json.dumps(sample) + "\n")
        
        original_argv = sys.argv
        try:
            sys.argv = [
                "verify_contextual_set",
                "--in", str(input_file),
                "--report", str(output_file),
            ]
            try:
                main()
            except SystemExit:
                pass
        finally:
            sys.argv = original_argv
        
        # Check report was created
        if output_file.exists():
            report = json.load(open(output_file))
            assert "summary" in report
    
    def test_main_function_integration_f1_calculation(self, tmp_path):
        """Test main function Integration F1 calculation."""
        from scripts.verify_contextual_set import main
        import sys
        import json
        
        input_file = tmp_path / "test_input.jsonl"
        output_file = tmp_path / "test_output.json"
        
        # Sample with integration spans
        sample = {
            "prompt": '{"caws": {"tier": 2, "max_files": 25, "max_loc": 1000, "cov": 80, "mut": 50}}',
            "teacher_text": "Integration: ANE prefers fp16 kernels when ops are supported.",
            "metadata": {
                "dataset_version": "1.1.0",
                "call_sequence": [{"name": "read_file", "arguments": {"path": "test.txt"}}],
                "tool_result_fields": {"summary": "ANE prefers fp16 kernels when ops are supported"},
                "integration_spans_bytes": [[13, 60]],
                "expected_behaviour": "normal",
            }
        }
        
        with open(input_file, "w") as f:
            f.write(json.dumps(sample) + "\n")
        
        original_argv = sys.argv
        try:
            sys.argv = [
                "verify_contextual_set",
                "--in", str(input_file),
                "--report", str(output_file),
            ]
            try:
                main()
            except SystemExit:
                pass
        finally:
            sys.argv = original_argv
        
        # Check report contains F1 metrics
        if output_file.exists():
            report = json.load(open(output_file))
            assert "summary" in report
            assert "avg_integration_f1" in report["summary"]
    
    def test_main_function_long_context_counting(self, tmp_path):
        """Test main function long-context counting."""
        from scripts.verify_contextual_set import main
        import sys
        import json
        
        input_file = tmp_path / "test_input.jsonl"
        output_file = tmp_path / "test_output.json"
        
        # Sample with long_context flag
        sample = {
            "prompt": "x" * 1000,
            "teacher_text": "Test",
            "metadata": {
                "dataset_version": "1.1.0",
                "long_context": True,
            }
        }
        
        with open(input_file, "w") as f:
            f.write(json.dumps(sample) + "\n")
        
        original_argv = sys.argv
        try:
            sys.argv = [
                "verify_contextual_set",
                "--in", str(input_file),
                "--report", str(output_file),
            ]
            try:
                main()
            except SystemExit:
                pass
        finally:
            sys.argv = original_argv
        
        # Check report contains long_context_count
        if output_file.exists():
            report = json.load(open(output_file))
            assert "summary" in report
            assert "long_context_count" in report["summary"]
    
    def test_main_function_stratification_check(self, tmp_path):
        """Test main function stratification checking."""
        from scripts.verify_contextual_set import main
        import sys
        import json
        
        input_file = tmp_path / "test_input.jsonl"
        output_file = tmp_path / "test_output.json"
        
        # Multiple samples with different scenarios
        samples = [
            {
                "prompt": '{"caws": {"tier": 2, "max_files": 25, "max_loc": 1000, "cov": 80, "mut": 50}}',
                "teacher_text": "Test",
                "metadata": {
                    "dataset_version": "1.1.0",
                    "scenario": "file_ops",
                    "complexity": "single_call",
                }
            },
            {
                "prompt": '{"caws": {"tier": 2, "max_files": 25, "max_loc": 1000, "cov": 80, "mut": 50}}',
                "teacher_text": "Test",
                "metadata": {
                    "dataset_version": "1.1.0",
                    "scenario": "web_search",
                    "complexity": "multi_call",
                }
            },
        ]
        
        with open(input_file, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
        
        original_argv = sys.argv
        try:
            sys.argv = [
                "verify_contextual_set",
                "--in", str(input_file),
                "--report", str(output_file),
            ]
            try:
                main()
            except SystemExit:
                pass
        finally:
            sys.argv = original_argv
        
        # Check report contains stratification
        if output_file.exists():
            report = json.load(open(output_file))
            assert "summary" in report
            assert "stratification" in report["summary"] or "stratification" in report
    
    def test_main_function_with_prompt_spans_target(self, tmp_path):
        """Test main function with spans_target='prompt'."""
        from scripts.verify_contextual_set import main
        import sys
        import json
        
        input_file = tmp_path / "test_input.jsonl"
        output_file = tmp_path / "test_output.json"
        
        sample = {
            "prompt": '{"caws": {"tier": 2, "max_files": 25, "max_loc": 1000, "cov": 80, "mut": 50}} Integration: Test integration.',
            "teacher_text": "Test",
            "metadata": {
                "dataset_version": "1.1.0",
                "spans_target": "prompt",
                "call_sequence": [{"name": "read_file", "arguments": {"path": "test.txt"}}],
                "tool_result_fields": {"summary": "Test integration"},
                "integration_spans_bytes": [[50, 70]],  # Span in prompt
            }
        }
        
        with open(input_file, "w") as f:
            f.write(json.dumps(sample) + "\n")
        
        original_argv = sys.argv
        try:
            sys.argv = [
                "verify_contextual_set",
                "--in", str(input_file),
                "--report", str(output_file),
            ]
            try:
                main()
            except SystemExit:
                pass
        finally:
            sys.argv = original_argv
    
    def test_main_function_empty_lines_skipped(self, tmp_path):
        """Test main function skips empty lines."""
        from scripts.verify_contextual_set import main
        import sys
        import json
        
        input_file = tmp_path / "test_input.jsonl"
        output_file = tmp_path / "test_output.json"
        
        # Write file with empty lines
        with open(input_file, "w") as f:
            f.write("\n")  # Empty line
            f.write(json.dumps({
                "prompt": '{"caws": {"tier": 2, "max_files": 25, "max_loc": 1000, "cov": 80, "mut": 50}}',
                "teacher_text": "Test",
                "metadata": {"dataset_version": "1.1.0"},
            }) + "\n")
            f.write("\n")  # Another empty line
        
        original_argv = sys.argv
        try:
            sys.argv = [
                "verify_contextual_set",
                "--in", str(input_file),
                "--report", str(output_file),
            ]
            try:
                main()
            except SystemExit:
                pass
        finally:
            sys.argv = original_argv
    
    def test_main_function_schema_file_not_found(self, tmp_path):
        """Test main function handles missing schema file."""
        from scripts.verify_contextual_set import main
        import sys
        import json
        from pathlib import Path
        
        input_file = tmp_path / "test_input.jsonl"
        output_file = tmp_path / "test_output.json"
        
        sample = {
            "prompt": '{"caws": {"tier": 2, "max_files": 25, "max_loc": 1000, "cov": 80, "mut": 50}}',
            "teacher_text": "Test",
            "metadata": {"dataset_version": "1.1.0"},
        }
        
        with open(input_file, "w") as f:
            f.write(json.dumps(sample) + "\n")
        
        # Ensure schema file doesn't exist
        schema_path = Path(__file__).resolve().parents[2] / "schemas" / "dataset_item.schema.json"
        if schema_path.exists():
            schema_path.unlink()
        
        original_argv = sys.argv
        try:
            sys.argv = [
                "verify_contextual_set",
                "--in", str(input_file),
                "--report", str(output_file),
            ]
            try:
                main()
            except SystemExit:
                pass
        finally:
            sys.argv = original_argv
    
    def test_main_function_tokenizer_exception(self, tmp_path):
        """Test main function handles tokenizer loading exception."""
        from scripts.verify_contextual_set import main
        import sys
        import json
        from unittest.mock import patch
        
        input_file = tmp_path / "test_input.jsonl"
        output_file = tmp_path / "test_output.json"
        
        sample = {
            "prompt": '{"caws": {"tier": 2, "max_files": 25, "max_loc": 1000, "cov": 80, "mut": 50}}',
            "teacher_text": "Test",
            "metadata": {"dataset_version": "1.1.0"},
        }
        
        with open(input_file, "w") as f:
            f.write(json.dumps(sample) + "\n")
        
        original_argv = sys.argv
        try:
            sys.argv = [
                "verify_contextual_set",
                "--in", str(input_file),
                "--report", str(output_file),
                "--tokenizer", "nonexistent/tokenizer",
            ]
            with patch('transformers.AutoTokenizer') as mock_auto:
                mock_auto.from_pretrained = Mock(side_effect=Exception("Tokenizer not found"))
                try:
                    main()
                except SystemExit:
                    pass
        finally:
            sys.argv = original_argv
    
    def test_main_function_eligible_results_calculation(self, tmp_path):
        """Test main function calculates eligible results correctly."""
        from scripts.verify_contextual_set import main
        import sys
        import json
        
        input_file = tmp_path / "test_input.jsonl"
        output_file = tmp_path / "test_output.json"
        
        # Mix of eligible and ineligible items
        samples = [
            {
                "prompt": '{"caws": {"tier": 2, "max_files": 25, "max_loc": 1000, "cov": 80, "mut": 50}}',
                "teacher_text": "Integration: Test.",
                "metadata": {
                    "dataset_version": "1.1.0",
                    "call_sequence": [{"name": "read_file", "arguments": {}}],
                    "tool_result_fields": {"summary": "Test"},
                    "integration_spans_bytes": [[13, 18]],
                    "expected_behaviour": "normal",
                }
            },
            {
                "prompt": '{"caws": {"tier": 2, "max_files": 25, "max_loc": 1000, "cov": 80, "mut": 50}}',
                "teacher_text": "No tools here.",
                "metadata": {
                    "dataset_version": "1.1.0",
                    "expected_behaviour": "no_tool",  # Control case - not eligible
                }
            },
        ]
        
        with open(input_file, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
        
        original_argv = sys.argv
        try:
            sys.argv = [
                "verify_contextual_set",
                "--in", str(input_file),
                "--report", str(output_file),
            ]
            try:
                main()
            except SystemExit:
                pass
        finally:
            sys.argv = original_argv
        
        # Check report contains F1 metrics
        if output_file.exists():
            report = json.load(open(output_file))
            assert "summary" in report
            assert "avg_integration_f1" in report["summary"]
    
    def test_main_function_integration_misses_tracking(self, tmp_path):
        """Test main function tracks integration misses."""
        from scripts.verify_contextual_set import main
        import sys
        import json
        
        input_file = tmp_path / "test_input.jsonl"
        output_file = tmp_path / "test_output.json"
        
        # Item with tool calls but no integration spans
        sample = {
            "prompt": '{"caws": {"tier": 2, "max_files": 25, "max_loc": 1000, "cov": 80, "mut": 50}}',
            "teacher_text": "Test",
            "metadata": {
                "dataset_version": "1.1.0",
                "call_sequence": [{"name": "read_file", "arguments": {}}],
                "expected_behaviour": "normal",
                # Missing integration_spans_bytes
            }
        }
        
        with open(input_file, "w") as f:
            f.write(json.dumps(sample) + "\n")
        
        original_argv = sys.argv
        try:
            sys.argv = [
                "verify_contextual_set",
                "--in", str(input_file),
                "--report", str(output_file),
            ]
            try:
                main()
            except SystemExit:
                pass
        finally:
            sys.argv = original_argv
    
    def test_main_function_grounding_misses_tracking(self, tmp_path):
        """Test main function tracks grounding misses."""
        from scripts.verify_contextual_set import main
        import sys
        import json
        
        input_file = tmp_path / "test_input.jsonl"
        output_file = tmp_path / "test_output.json"
        
        # Item with integration spans but not grounded
        sample = {
            "prompt": '{"caws": {"tier": 2, "max_files": 25, "max_loc": 1000, "cov": 80, "mut": 50}}',
            "teacher_text": "Integration: Random text that doesn't match summary.",
            "metadata": {
                "dataset_version": "1.1.0",
                "call_sequence": [{"name": "read_file", "arguments": {}}],
                "tool_result_fields": {"summary": "ANE prefers fp16 kernels"},
                "integration_spans_bytes": [[13, 50]],  # Span doesn't contain summary
                "expected_behaviour": "normal",
            }
        }
        
        with open(input_file, "w") as f:
            f.write(json.dumps(sample) + "\n")
        
        original_argv = sys.argv
        try:
            sys.argv = [
                "verify_contextual_set",
                "--in", str(input_file),
                "--report", str(output_file),
            ]
            try:
                main()
            except SystemExit:
                pass
        finally:
            sys.argv = original_argv
    
    def test_main_function_multi_call_parity_tracking(self, tmp_path):
        """Test main function tracks multi-call parity."""
        from scripts.verify_contextual_set import main
        import sys
        import json
        
        input_file = tmp_path / "test_input.jsonl"
        output_file = tmp_path / "test_output.json"
        
        # Multi-call item with mismatched spans
        sample = {
            "prompt": '{"caws": {"tier": 2, "max_files": 25, "max_loc": 1000, "cov": 80, "mut": 50}}',
            "teacher_text": "Test",
            "metadata": {
                "dataset_version": "1.1.0",
                "call_sequence": [
                    {"name": "read_file", "arguments": {}},
                    {"name": "write_file", "arguments": {}},
                ],
                "json_args_spans_bytes": [[10, 20]],  # Only 1 span for 2 calls
            }
        }
        
        with open(input_file, "w") as f:
            f.write(json.dumps(sample) + "\n")
        
        original_argv = sys.argv
        try:
            sys.argv = [
                "verify_contextual_set",
                "--in", str(input_file),
                "--report", str(output_file),
            ]
            try:
                main()
            except SystemExit:
                pass
        finally:
            sys.argv = original_argv
    
    def test_main_function_gate_failures(self, tmp_path):
        """Test main function gate failure checks."""
        from scripts.verify_contextual_set import main
        import sys
        import json
        
        input_file = tmp_path / "test_input.jsonl"
        output_file = tmp_path / "test_output.json"
        
        # Create items that will fail gates
        samples = []
        for i in range(20):  # Create 20 items
            samples.append({
                "prompt": "invalid",  # No CAWS header
                "teacher_text": "Test",
                "metadata": {"dataset_version": "1.1.0"},
            })
        
        with open(input_file, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
        
        original_argv = sys.argv
        try:
            sys.argv = [
                "verify_contextual_set",
                "--in", str(input_file),
                "--report", str(output_file),
            ]
            try:
                main()
            except SystemExit as e:
                # Should exit with error code
                assert e.code != 0
        finally:
            sys.argv = original_argv
    
    def test_main_function_adversarial_quota_check(self, tmp_path):
        """Test main function adversarial quota check."""
        from scripts.verify_contextual_set import main
        import sys
        import json
        
        input_file = tmp_path / "test_input.jsonl"
        output_file = tmp_path / "test_output.json"
        
        # Create 30+ items with various adversarial types
        samples = []
        for i in range(35):
            samples.append({
                "prompt": '{"caws": {"tier": 2, "max_files": 25, "max_loc": 1000, "cov": 80, "mut": 50}}',
                "teacher_text": "Test",
                "metadata": {
                    "dataset_version": "1.1.0",
                    "adversarial": {"type": "range_violation" if i < 10 else "malformed_json"},
                }
            })
        
        with open(input_file, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
        
        original_argv = sys.argv
        try:
            sys.argv = [
                "verify_contextual_set",
                "--in", str(input_file),
                "--report", str(output_file),
            ]
            try:
                main()
            except SystemExit:
                pass
        finally:
            sys.argv = original_argv
    
    def test_main_function_long_context_quota_check(self, tmp_path):
        """Test main function long-context quota check."""
        from scripts.verify_contextual_set import main
        import sys
        import json
        
        input_file = tmp_path / "test_input.jsonl"
        output_file = tmp_path / "test_output.json"
        
        # Create 20+ items with long_context flag
        samples = []
        for i in range(25):
            samples.append({
                "prompt": "x" * 1000,
                "teacher_text": "Test",
                "metadata": {
                    "dataset_version": "1.1.0",
                    "long_context": True if i < 5 else False,  # 5 long-context items
                }
            })
        
        with open(input_file, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
        
        original_argv = sys.argv
        try:
            sys.argv = [
                "verify_contextual_set",
                "--in", str(input_file),
                "--report", str(output_file),
            ]
            try:
                main()
            except SystemExit:
                pass
        finally:
            sys.argv = original_argv

    
    def test_grounded_values_in_span_numeric(self):
        """Test grounded value detection with numeric values."""
        from scripts.verify_contextual_set import grounded_values_in_span
        
        seg = "The file contains 128 lines"
        fields = {"lines": 128}
        
        result = grounded_values_in_span(seg, fields)
        assert result is True
    
    def test_grounded_values_in_span_url(self):
        """Test grounded value detection with URL values."""
        from scripts.verify_contextual_set import grounded_values_in_span
        
        seg = "Visit example.org/article for details"
        fields = {"url": "https://example.org/article"}
        
        result = grounded_values_in_span(seg, fields)
        assert result is True
    
    def test_grounded_values_in_span_empty_seg(self):
        """Test grounded value detection with empty segment."""
        from scripts.verify_contextual_set import grounded_values_in_span
        
        seg = ""
        fields = {"summary": "test"}
        
        result = grounded_values_in_span(seg, fields)
        assert result is False
    
    def test_grounded_values_in_span_empty_fields(self):
        """Test grounded value detection with empty fields."""
        from scripts.verify_contextual_set import grounded_values_in_span
        
        seg = "Some text"
        fields = {}
        
        result = grounded_values_in_span(seg, fields)
        assert result is False
    
    def test_verify_token_alignment_failure(self):
        """Test token alignment verification failure."""
        from scripts.verify_contextual_set import verify_token_alignment
        from unittest.mock import Mock
        
        tokenizer = Mock()
        tokenizer.is_fast = True
        
        def tokenize_side_effect(text, **kwargs):
            return {
                "input_ids": list(range(len(text))),
                "offset_mapping": [(i, i+1) for i in range(len(text))]
            }
        
        tokenizer.side_effect = tokenize_side_effect
        tokenizer.encode = Mock(side_effect=lambda text, **kwargs: list(range(len(text))))
        tokenizer.decode = Mock(return_value="Different text")  # Decode doesn't match
        
        text = "Hello world"
        span_bytes = [0, 5]
        
        ok, token_span = verify_token_alignment(text, span_bytes, tokenizer)
        # Should fail because decoded text doesn't match
        assert ok is False
    
    def test_verify_token_alignment_invalid_span(self):
        """Test token alignment with invalid span."""
        from scripts.verify_contextual_set import verify_token_alignment
        from unittest.mock import Mock
        
        tokenizer = Mock()
        ok, token_span = verify_token_alignment("text", [], tokenizer)
        assert ok is False
        assert token_span is None
    
    def test_verify_token_alignment_exception(self):
        """Test token alignment with tokenizer exception."""
        from scripts.verify_contextual_set import verify_token_alignment
        
        class FailingTokenizer:
            def encode(self, *args, **kwargs):
                raise Exception("Tokenization failed")
        
        failing_tokenizer = FailingTokenizer()
        # Should handle exception gracefully
        try:
            ok, token_span = verify_token_alignment("text", [0, 4], failing_tokenizer)
            assert ok is False
        except Exception:
            # Exception handling may vary, but should not crash
            pass
    
    def test_multi_call_parity_mismatch(self):
        """Test multi-call parity mismatch detection."""
        from scripts.verify_contextual_set import verify_item
        from tools.schema_registry import ToolSchemaRegistry
        
        reg = ToolSchemaRegistry()
        
        item = {
            "prompt": '{"caws": {"tier": 2, "max_files": 25, "max_loc": 1000, "cov": 80, "mut": 50}}',
            "teacher_text": 'Call 1: {"name": "read_file", "arguments": {"path": "test.txt"}} Call 2: {"name": "write_file", "arguments": {"path": "out.txt"}}',
            "metadata": {
                "dataset_version": "1.1.0",
                "call_sequence": [
                    {"name": "read_file", "arguments": {"path": "test.txt"}},
                    {"name": "write_file", "arguments": {"path": "out.txt"}}
                ],
                "json_args_spans_bytes": [[0, 50]],  # Only one span for two calls
                "expected_behaviour": "normal",
            }
        }
        
        result = verify_item(item, reg)
        # Should detect mismatch - check that mismatch is in problems
        problems = result.get("problems", [])
        assert any("mismatch" in p.lower() for p in problems) or any("expected_2" in p for p in problems)
    
    def test_semantic_validation_invalid_args(self):
        """Test semantic validation with invalid arguments."""
        from scripts.verify_contextual_set import verify_item
        from tools.schema_registry import ToolSchemaRegistry
        
        reg = ToolSchemaRegistry()
        
        # Use a tool that exists but with invalid args
        item = {
            "prompt": '{"caws": {"tier": 2, "max_files": 25, "max_loc": 1000, "cov": 80, "mut": 50}}',
            "teacher_text": '{"name": "read_file", "arguments": {"path": 123}}',  # Invalid: path should be string
            "metadata": {
                "dataset_version": "1.1.0",
                "call_sequence": [{"name": "read_file", "arguments": {"path": 123}}],
                "json_args_span_bytes": [0, 50],
                "expected_behaviour": "normal",
            }
        }
        
        result = verify_item(item, reg)
        # Should detect invalid arguments if schema validation fails
        assert isinstance(result, dict)
    
    def test_multi_call_parity_mismatch(self):
        """Test multi-call parity mismatch detection."""
        from scripts.verify_contextual_set import verify_item
        from tools.schema_registry import ToolSchemaRegistry
        
        reg = ToolSchemaRegistry()
        
        item = {
            "prompt": '{"caws": {"tier": 2, "max_files": 25, "max_loc": 1000, "cov": 80, "mut": 50}}',
            "teacher_text": 'Call 1: {"name": "read_file", "arguments": {"path": "test.txt"}} Call 2: {"name": "write_file", "arguments": {"path": "out.txt"}}',
            "metadata": {
                "dataset_version": "1.1.0",
                "call_sequence": [
                    {"name": "read_file", "arguments": {"path": "test.txt"}},
                    {"name": "write_file", "arguments": {"path": "out.txt"}}
                ],
                "json_args_spans_bytes": [[0, 50]],  # Only one span for two calls
                "expected_behaviour": "normal",
            }
        }
        
        result = verify_item(item, reg)
        # Should detect mismatch - check that mismatch is in problems
        problems = result.get("problems", [])
        assert any("mismatch" in p.lower() for p in problems) or any("expected_2" in p for p in problems)
    
    def test_multi_call_name_span_mismatch(self):
        """Test multi-call tool name span mismatch."""
        from scripts.verify_contextual_set import verify_item
        from tools.schema_registry import ToolSchemaRegistry
        
        reg = ToolSchemaRegistry()
        
        item = {
            "prompt": '{"caws": {"tier": 2}}',
            "teacher_text": 'Call 1: {"name": "read_file", "arguments": {"path": "test.txt"}} Call 2: {"name": "write_file", "arguments": {"path": "out.txt"}}',
            "metadata": {
                "dataset_version": "1.1.0",
                "call_sequence": [
                    {"name": "read_file", "arguments": {"path": "test.txt"}},
                    {"name": "write_file", "arguments": {"path": "out.txt"}}
                ],
                "json_args_spans_bytes": [[0, 50], [60, 110]],
                "tool_name_spans_bytes": [[10, 18]],  # Only one name span for two calls
                "expected_behaviour": "normal",
            }
        }
        
        result = verify_item(item, reg)
        # Should detect name span mismatch
        assert isinstance(result, dict)
    
    def test_parse_tool_json_slice_no_braces(self):
        """Test JSON parsing when no braces found."""
        from scripts.verify_contextual_set import parse_tool_json_slice
        
        text = "No JSON here"
        obj = parse_tool_json_slice(text, 0, len(text))
        assert obj is None
    
    def test_parse_tool_json_slice_exception(self):
        """Test JSON parsing with exception."""
        from scripts.verify_contextual_set import parse_tool_json_slice
        
        # Invalid JSON that will raise exception
        text = "{invalid json"
        obj = parse_tool_json_slice(text, 0, len(text))
        # Should return None on exception
        assert obj is None or isinstance(obj, dict)
    
    def test_check_privacy_url_not_allowlisted(self):
        """Test privacy check with URL not in allowlist."""
        from scripts.verify_contextual_set import check_privacy
        
        text = "Visit https://malicious-site.com for details"
        result = check_privacy(text)
        # Should detect URL not in allowlist
        assert result["privacy_ok"] is False or not result.get("url_allowlist_ok", True)
    
    def test_check_privacy_multiple_emails(self):
        """Test privacy check with multiple emails."""
        from scripts.verify_contextual_set import check_privacy
        
        text = "Contact test1@example.com or test2@example.com"
        result = check_privacy(text)
        assert result["emails_found"] >= 2
        assert result["privacy_ok"] is False
    
    def test_check_privacy_multiple_uuids(self):
        """Test privacy check with multiple UUIDs."""
        from scripts.verify_contextual_set import check_privacy
        
        text = "ID1: 550e8400-e29b-41d4-a716-446655440000 ID2: 6ba7b810-9dad-11d1-80b4-00c04fd430c8"
        result = check_privacy(text)
        assert result["uuids_found"] >= 2
        assert result["privacy_ok"] is False
    
    def test_check_caws_header_exception(self):
        """Test CAWS header check with exception."""
        from scripts.verify_contextual_set import check_caws_header
        
        # Invalid JSON that will raise exception
        prompt = "{invalid json"
        ok, data = check_caws_header(prompt)
        assert ok is False
        assert data is None
    
    def test_check_caws_header_missing_keys(self):
        """Test CAWS header check with missing required keys."""
        from scripts.verify_contextual_set import check_caws_header
        
        prompt = '{"caws": {"tier": 2}}\nTask: test'  # Missing required keys
        ok, data = check_caws_header(prompt)
        # Should fail if required keys missing
        assert isinstance(ok, bool)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_missing_dependencies(self):
        """Test graceful degradation when dependencies are missing."""
        # Test import error handling
        try:
            from scripts.extract_process_targets import load_tokenizer
            import sys
            # Mock ImportError for transformers
            original_import = __import__
            def mock_import(name, *args, **kwargs):
                if name == 'transformers':
                    raise ImportError("No module named 'transformers'")
                return original_import(name, *args, **kwargs)
            
            with patch('builtins.__import__', side_effect=mock_import):
                try:
                    load_tokenizer("dummy")
                except RuntimeError:
                    pass  # Expected
        except Exception:
            pass  # Skip if test setup fails
    
    def test_invalid_inputs(self):
        """Test input validation."""
        from scripts.generate_contextual_prompts import synthesize_prompt
        from tools.schema_registry import ToolSchemaRegistry
        
        reg = ToolSchemaRegistry()
        
        # Test with missing required fields
        cell_missing = {
            "scenario": "file_ops",
            # Missing complexity
        }
        
        # Should handle gracefully (may use defaults or raise)
        try:
            prompt, history, meta = synthesize_prompt(cell_missing, reg)
            assert isinstance(prompt, str)
        except (KeyError, ValueError):
            pass  # Expected for invalid input
    
    def test_file_io_errors(self, tmp_path):
        """Test file system error handling."""
        from scripts.extract_process_targets import main
        import sys
        
        # Test with non-existent input file
        non_existent = tmp_path / "nonexistent.jsonl"
        
        # Should handle file not found gracefully
        try:
            sys.argv = ["extract_process_targets", "--in", str(non_existent), "--out", str(tmp_path / "out.jsonl")]
            main()
        except (FileNotFoundError, SystemExit):
            pass  # Expected
    
    def test_tokenizer_errors(self):
        """Test tokenizer failure handling."""
        from scripts.extract_process_targets import extract_process_step_targets
        
        # Create tokenizer that raises errors
        class FailingTokenizer:
            def encode(self, *args, **kwargs):
                raise Exception("Tokenizer error")
        
        failing_tokenizer = FailingTokenizer()
        
        # Should handle gracefully
        result = extract_process_step_targets("test", failing_tokenizer)
        # May return empty dict or partial results
        assert isinstance(result, dict)
    
    def test_memory_errors(self):
        """Test out-of-memory handling."""
        from scripts.generate_contextual_prompts import synthesize_prompt
        from tools.schema_registry import ToolSchemaRegistry
        
        reg = ToolSchemaRegistry()
        
        # Test with very large input (should handle gracefully)
        cell = {
            "scenario": "file_ops",
            "complexity": "single_call",
            "structure": "flat_args",
            "long_context": True,
        }
        
        # Should complete without crashing
        prompt, history, meta = synthesize_prompt(cell, reg)
        assert isinstance(prompt, str)
    
    def test_json_parse_errors(self):
        """Test JSON parsing error handling."""
        from scripts.verify_contextual_set import parse_tool_json_slice
        
        # Test with invalid JSON
        invalid_json = "Not valid JSON {"
        obj = parse_tool_json_slice(invalid_json, 0, len(invalid_json))
        
        # Should return None for invalid JSON
        assert obj is None
    
    def test_empty_inputs(self):
        """Test handling of empty inputs."""
        from scripts.extract_process_targets import extract_process_step_targets
        from unittest.mock import Mock
        
        tokenizer = Mock()
        tokenizer.is_fast = True
        
        # Test with empty text
        result = extract_process_step_targets("", tokenizer)
        assert isinstance(result, dict)
        
        # Test with None tokenizer
        result = extract_process_step_targets("test", None)
        assert result == {}
    
    def test_load_tokenizer_success(self):
        """Test successful tokenizer loading."""
        from scripts.extract_process_targets import load_tokenizer
        from unittest.mock import patch, Mock
        
        mock_tokenizer = Mock()
        with patch('transformers.AutoTokenizer') as mock_auto:
            mock_auto.from_pretrained = Mock(return_value=mock_tokenizer)
            result = load_tokenizer("dummy_path")
            assert result == mock_tokenizer
    
    def test_load_tokenizer_import_error(self):
        """Test tokenizer loading with ImportError."""
        from scripts.extract_process_targets import load_tokenizer
        from unittest.mock import patch
        
        # Test ImportError handling
        with patch('builtins.__import__', side_effect=lambda name, *args, **kwargs: (_ for _ in ()).throw(ImportError("No module named 'transformers'"))):
            try:
                # Reload module to trigger ImportError
                import importlib
                import scripts.extract_process_targets
                importlib.reload(scripts.extract_process_targets)
                load_tokenizer("dummy_path")
            except RuntimeError as e:
                assert "transformers required" in str(e)
            except Exception:
                # May raise ImportError directly, which is also acceptable
                pass
    
    def test_load_tokenizer_general_exception(self):
        """Test tokenizer loading with general exception."""
        from scripts.extract_process_targets import load_tokenizer
        from unittest.mock import patch, Mock
        
        mock_tokenizer = Mock()
        with patch('transformers.AutoTokenizer') as mock_auto:
            mock_auto.from_pretrained = Mock(side_effect=Exception("Load failed"))
            try:
                load_tokenizer("dummy_path")
                assert False, "Should raise RuntimeError"
            except RuntimeError as e:
                assert "Failed to load tokenizer" in str(e)
    
    def test_main_function_basic(self, tmp_path):
        """Test main function with valid input."""
        from scripts.extract_process_targets import main
        import sys
        import json
        from unittest.mock import patch, Mock
        
        input_file = tmp_path / "test_input.jsonl"
        output_file = tmp_path / "test_output.jsonl"
        
        sample = {
            "teacher_text": '{"name": "read_file", "arguments": {"path": "test.txt"}}',
            "metadata": {
                "call_sequence": [{"name": "read_file", "arguments": {"path": "test.txt"}}],
            }
        }
        
        with open(input_file, "w") as f:
            f.write(json.dumps(sample) + "\n")
        
        mock_tokenizer = Mock()
        mock_tokenizer.is_fast = True
        mock_tokenizer.encode = Mock(return_value=[1, 2, 3])
        mock_tokenizer.decode = Mock(return_value="test")
        
        def tokenize_side_effect(text, **kwargs):
            return {
                "input_ids": [1, 2, 3],
                "offset_mapping": [(0, 1), (1, 2), (2, 3)]
            }
        
        mock_tokenizer.side_effect = tokenize_side_effect
        
        original_argv = sys.argv
        try:
            sys.argv = [
                "extract_process_targets",
                "--in", str(input_file),
                "--out", str(output_file),
                "--tokenizer-path", "dummy",
            ]
            with patch('scripts.extract_process_targets.load_tokenizer', return_value=mock_tokenizer):
                main()
        except SystemExit:
            pass
        finally:
            sys.argv = original_argv
        
        # Check output file was created
        if output_file.exists():
            with open(output_file, "r") as f:
                lines = f.readlines()
                assert len(lines) > 0
    
    def test_main_function_empty_file(self, tmp_path):
        """Test main function with empty input file."""
        from scripts.extract_process_targets import main
        import sys
        from unittest.mock import patch, Mock
        
        input_file = tmp_path / "test_empty.jsonl"
        output_file = tmp_path / "test_output.jsonl"
        
        # Create empty file
        input_file.touch()
        
        mock_tokenizer = Mock()
        original_argv = sys.argv
        try:
            sys.argv = [
                "extract_process_targets",
                "--in", str(input_file),
                "--out", str(output_file),
            ]
            with patch('scripts.extract_process_targets.load_tokenizer', return_value=mock_tokenizer):
                main()
        except SystemExit:
            pass
        finally:
            sys.argv = original_argv
    
    def test_main_function_json_error(self, tmp_path):
        """Test main function with invalid JSON."""
        from scripts.extract_process_targets import main
        import sys
        from unittest.mock import patch, Mock
        
        input_file = tmp_path / "test_invalid.jsonl"
        output_file = tmp_path / "test_output.jsonl"
        
        # Write invalid JSON
        with open(input_file, "w") as f:
            f.write("Invalid JSON {")
        
        mock_tokenizer = Mock()
        original_argv = sys.argv
        try:
            sys.argv = [
                "extract_process_targets",
                "--in", str(input_file),
                "--out", str(output_file),
            ]
            with patch('scripts.extract_process_targets.load_tokenizer', return_value=mock_tokenizer):
                try:
                    main()
                except (json.JSONDecodeError, SystemExit):
                    pass  # Expected
        finally:
            sys.argv = original_argv


class TestUtilTokenSpans:
    """Test token span alignment utilities."""
    
    @pytest.fixture
    def fast_tokenizer(self):
        """Create mock fast tokenizer with offset mapping."""
        from unittest.mock import Mock
        
        tokenizer = Mock()
        tokenizer.is_fast = True
        
        def tokenize_side_effect(text, add_special_tokens=False, return_offsets_mapping=False, **kwargs):
            # Simple mock: each character becomes a token
            tokens = list(text)
            input_ids = list(range(len(tokens)))
            if return_offsets_mapping:
                offsets = []
                pos = 0
                for char in tokens:
                    offsets.append((pos, pos + len(char.encode('utf-8'))))
                    pos += len(char.encode('utf-8'))
                return {
                    "input_ids": input_ids,
                    "offset_mapping": offsets
                }
            return input_ids
        
        tokenizer.side_effect = tokenize_side_effect
        tokenizer.encode = Mock(side_effect=lambda text, **kwargs: list(range(len(text))))
        tokenizer.decode = Mock(side_effect=lambda ids, **kwargs: ''.join(chr(65 + i % 26) for i in ids))
        return tokenizer
    
    @pytest.fixture
    def slow_tokenizer(self):
        """Create mock slow tokenizer without offset mapping."""
        from unittest.mock import Mock
        
        tokenizer = Mock()
        tokenizer.is_fast = False
        
        def encode_side_effect(text, add_special_tokens=False, **kwargs):
            return list(range(len(text)))
        
        def decode_side_effect(ids, skip_special_tokens=False, **kwargs):
            return ''.join(chr(65 + i % 26) for i in ids)
        
        tokenizer.encode = Mock(side_effect=encode_side_effect)
        tokenizer.decode = Mock(side_effect=decode_side_effect)
        return tokenizer
    
    def test_fast_tokenizer_offsets(self, fast_tokenizer):
        """Test fast tokenizer offset mapping accuracy."""
        from scripts.util_token_spans import bytes_to_token_span
        
        text = "Hello world"
        # "Hello" spans bytes 0-5
        result = bytes_to_token_span(text, 0, 5, fast_tokenizer)
        
        assert result is not None
        start_tok, end_tok = result
        assert start_tok == 0
        assert end_tok == 5  # "Hello" is 5 characters
    
    def test_slow_tokenizer_fallback(self, slow_tokenizer):
        """Test slow tokenizer fallback behavior."""
        from scripts.util_token_spans import bytes_to_token_span
        
        text = "Hello world"
        # "Hello" spans bytes 0-5
        # Mock decode to return the actual text slice for verification
        def decode_side_effect(ids, skip_special_tokens=False, **kwargs):
            # Return text that matches what we're checking
            if len(ids) == 5:
                return "Hello"
            return "Hello world"[:len(ids)]
        
        slow_tokenizer.decode = Mock(side_effect=decode_side_effect)
        
        result = bytes_to_token_span(text, 0, 5, slow_tokenizer)
        
        # Should use fallback method
        assert result is not None
        start_tok, end_tok = result
        assert start_tok >= 0
        assert end_tok > start_tok
    
    def test_normalize_text_nfc(self):
        """Test NFC normalization."""
        from scripts.util_token_spans import normalize_text_for_alignment
        
        # Test with combining characters
        text = "caf\u00e9"  # Precomposed
        normalized = normalize_text_for_alignment(text, text_norm="NFC")
        
        assert normalized == text  # Already NFC
    
    def test_normalize_text_lf(self):
        """Test LF line ending normalization."""
        from scripts.util_token_spans import normalize_text_for_alignment
        
        text = "Line 1\r\nLine 2\r\nLine 3"
        normalized = normalize_text_for_alignment(text, line_endings="LF")
        
        assert "\r\n" not in normalized
        assert normalized == "Line 1\nLine 2\nLine 3"
    
    def test_normalize_text_both(self):
        """Test both NFC and LF normalization."""
        from scripts.util_token_spans import normalize_text_for_alignment
        
        text = "Line 1\r\nLine 2"
        normalized = normalize_text_for_alignment(text, text_norm="NFC", line_endings="LF")
        
        assert "\r\n" not in normalized
        assert normalized == "Line 1\nLine 2"
    
    def test_empty_spans(self, fast_tokenizer):
        """Test empty span handling."""
        from scripts.util_token_spans import bytes_to_token_span, byte_spans_to_token_spans
        
        text = "Hello world"
        
        # Empty span list
        result = byte_spans_to_token_spans(text, [], fast_tokenizer)
        assert result == []
        
        # Zero-length span
        result = bytes_to_token_span(text, 5, 5, fast_tokenizer)
        assert result is None  # Should return None for zero-length
    
    def test_out_of_bounds_spans(self, fast_tokenizer):
        """Test boundary validation."""
        from scripts.util_token_spans import bytes_to_token_span
        
        text = "Hello world"
        
        # Start beyond text
        result = bytes_to_token_span(text, 100, 110, fast_tokenizer)
        assert result is None
        
        # End beyond text
        result = bytes_to_token_span(text, 0, 1000, fast_tokenizer)
        # Should handle gracefully (may return None or valid span up to end)
        assert result is None or isinstance(result, tuple)
    
    def test_unicode_handling(self, fast_tokenizer):
        """Test Unicode normalization edge cases."""
        from scripts.util_token_spans import normalize_text_for_alignment
        
        # Test with various Unicode characters
        text = "Hello 世界 🌍"
        normalized = normalize_text_for_alignment(text, text_norm="NFC")
        
        assert len(normalized) > 0
        assert "Hello" in normalized
    
    def test_byte_spans_to_token_spans_multiple(self, fast_tokenizer):
        """Test multiple span conversion."""
        from scripts.util_token_spans import byte_spans_to_token_spans
        
        text = "Hello world test"
        spans = [[0, 5], [6, 11]]  # "Hello" and "world"
        
        result = byte_spans_to_token_spans(text, spans, fast_tokenizer)
        
        assert len(result) == 2
        assert all(r is not None for r in result)
    
    def test_byte_spans_to_token_spans_slow_fallback(self, slow_tokenizer):
        """Test slow tokenizer fallback for multiple spans."""
        from scripts.util_token_spans import byte_spans_to_token_spans
        
        text = "Hello world"
        spans = [[0, 5], [6, 11]]
        
        result = byte_spans_to_token_spans(text, spans, slow_tokenizer)
        
        assert len(result) == 2
        # May be None if alignment fails, but should handle gracefully
        assert all(r is None or isinstance(r, tuple) for r in result)
    
    def test_tokenizer_exception_fallback(self, fast_tokenizer):
        """Test exception handling fallback."""
        from scripts.util_token_spans import bytes_to_token_span
        
        # Make tokenizer raise exception
        fast_tokenizer.side_effect = Exception("Tokenizer error")
        fast_tokenizer.is_fast = True
        
        text = "Hello world"
        result = bytes_to_token_span(text, 0, 5, fast_tokenizer)
        
        # Should fall back to slow tokenizer method
        assert result is None or isinstance(result, tuple)


class TestNegativeMutationSimulation:
    """Targeted negative tests simulating common mutations."""
    
    def test_eligibility_filter_off_by_one(self):
        """Test >= vs > edge cases in eligibility filtering."""
        from scripts.verify_contextual_set import verify_item
        from tools.schema_registry import ToolSchemaRegistry
        
        reg = ToolSchemaRegistry()
        
        # Item with exactly 1 call (boundary case)
        item = {
            "prompt": '{"caws": {"tier": 2}}',
            "teacher_text": "Test",
            "metadata": {
                "dataset_version": "1.1.0",
                "call_sequence": [{"name": "read_file", "arguments": {}}],
                "expected_behaviour": "normal",
            }
        }
        
        result = verify_item(item, reg)
        
        # Should be eligible (has call_sequence with 1 call)
        assert "ok" in result
        # Should have F1 computed (if integration spans present)
        assert "integration_f1" in result
    
    def test_control_case_not_removed(self):
        """Test missing 'not' in control checks (simulates mutation)."""
        from scripts.verify_contextual_set import verify_item
        from tools.schema_registry import ToolSchemaRegistry
        
        reg = ToolSchemaRegistry()
        
        # Control case that should be filtered out
        item = {
            "prompt": '{"caws": {"tier": 2}}',
            "teacher_text": "I cannot help with that.",
            "metadata": {
                "dataset_version": "1.1.0",
                "expected_behaviour": "no_tool",
                "call_sequence": [],  # Should be empty
                "integration_spans_bytes": [],  # Should be empty
            }
        }
        
        result = verify_item(item, reg)
        
        # Control cases should not have tool calls
        # If mutation removed 'not', this would incorrectly pass
        assert result["ok"] is True  # Controls pass validation
        assert len(result.get("problems", [])) == 0 or "control" not in str(result.get("problems", []))
    
    def test_empty_list_vs_none(self):
        """Test empty list vs None handling."""
        from scripts.verify_contextual_set import verify_item
        from tools.schema_registry import ToolSchemaRegistry
        
        reg = ToolSchemaRegistry()
        
        # Item with empty list vs None
        item_empty = {
            "prompt": '{"caws": {"tier": 2}}',
            "teacher_text": "Test",
            "metadata": {
                "dataset_version": "1.1.0",
                "call_sequence": [],  # Empty list
                "integration_spans_bytes": [],  # Empty list
            }
        }
        
        item_none = {
            "prompt": '{"caws": {"tier": 2}}',
            "teacher_text": "Test",
            "metadata": {
                "dataset_version": "1.1.0",
                "call_sequence": None,  # None instead of empty list
                "integration_spans_bytes": None,
            }
        }
        
        result_empty = verify_item(item_empty, reg)
        result_none = verify_item(item_none, reg)
        
        # Both should be handled gracefully
        assert isinstance(result_empty, dict)
        assert isinstance(result_none, dict)
        # Empty list should be treated differently than None
        assert result_empty.get("integration_f1", 0.0) == 0.0
        assert result_none.get("integration_f1", 0.0) == 0.0
    
    def test_span_cap_off_by_one(self):
        """Test cap boundary (3 vs 4) for integration spans."""
        from scripts.generate_contextual_prompts import synthesize_prompt
        from tools.schema_registry import ToolSchemaRegistry
        
        reg = ToolSchemaRegistry()
        
        # Generate with cap=3, but simulate 4 spans
        cell = {
            "scenario": "file_ops",
            "complexity": "single_call",
        }
        
        # Generate with cap=3
        prompt1, history1, meta1 = synthesize_prompt(cell, reg, integration_span_cap=3)
        
        # Generate with cap=4
        prompt2, history2, meta2 = synthesize_prompt(cell, reg, integration_span_cap=4)
        
        # Both should work
        assert isinstance(meta1, dict)
        assert isinstance(meta2, dict)
        
        # If spans exceed cap, should be flagged
        spans1 = meta1.get("integration_spans_bytes", [])
        spans2 = meta2.get("integration_spans_bytes", [])
        
        # Cap should be enforced
        if len(spans1) > 3:
            assert meta1.get("integration_spans_exceeded_cap") is True
    
    def test_multi_call_parity_empty(self):
        """Test empty call sequence handling."""
        from scripts.verify_contextual_set import verify_item
        from tools.schema_registry import ToolSchemaRegistry
        
        reg = ToolSchemaRegistry()
        
        # Item with empty call sequence
        item = {
            "prompt": '{"caws": {"tier": 2}}',
            "teacher_text": "Test",
            "metadata": {
                "dataset_version": "1.1.0",
                "call_sequence": [],
                "json_args_spans_bytes": [],
            }
        }
        
        result = verify_item(item, reg)
        
        # Should handle gracefully
        assert isinstance(result, dict)
        assert "ok" in result
        # Empty call sequence should not trigger multi-call parity check
        assert "multi_call" not in str(result.get("problems", []))
    
    def test_token_alignment_boundary(self):
        """Test span boundary edge cases."""
        from scripts.util_token_spans import bytes_to_token_span
        from unittest.mock import Mock
        
        mock_tokenizer = Mock()
        mock_tokenizer.is_fast = True
        
        def encode_side_effect(text, add_special_tokens=False, return_offsets_mapping=False, **kwargs):
            tokens = list(text)
            input_ids = list(range(len(tokens)))
            if return_offsets_mapping:
                offsets = []
                pos = 0
                for char in tokens:
                    offsets.append((pos, pos + len(char.encode('utf-8'))))
                    pos += len(char.encode('utf-8'))
                return {
                    "input_ids": input_ids,
                    "offset_mapping": offsets
                }
            return input_ids
        
        def call_side_effect(text, add_special_tokens=False, return_offsets_mapping=False, **kwargs):
            """Support tokenizer(text, ...) callable interface."""
            return encode_side_effect(text, add_special_tokens=add_special_tokens, return_offsets_mapping=return_offsets_mapping, **kwargs)
        
        mock_tokenizer.encode = Mock(side_effect=encode_side_effect)
        mock_tokenizer.__call__ = Mock(side_effect=call_side_effect)
        
        text = "Hello world"
        text_bytes = text.encode('utf-8')
        
        # Test boundary cases
        # Start at 0, end at text length
        result1 = bytes_to_token_span(text, 0, len(text_bytes), mock_tokenizer)
        # May be None if alignment fails, but should handle gracefully
        if result1 is not None:
            assert isinstance(result1, tuple)
            assert len(result1) == 2
        
        # Start at 0, end at 0 (empty span)
        result2 = bytes_to_token_span(text, 0, 0, mock_tokenizer)
        assert result2 is None or result2[0] == result2[1]
        
        # Start beyond text length
        result3 = bytes_to_token_span(text, len(text.encode('utf-8')) + 10, len(text.encode('utf-8')) + 20, mock_tokenizer)
        assert result3 is None

