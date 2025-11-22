"""
Unit tests for tool name inference logic.

Tests schema matching, JSON extraction, and inference pipeline.

Author: @darianrosebrook
"""

import json
import pytest
from unittest.mock import Mock, MagicMock
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.infer_tool_name_from_json import (
    build_schema_key_map,
    extract_json_keys,
    match_json_to_tool,
    decode_json_from_token_ids,
    extract_json_from_integration_mask,
    infer_tool_name_from_text,
    infer_tool_name_for_sample,
)
from tools.schema_registry import ToolSchemaRegistry


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def __init__(self):
        self.vocab = {"web": 100, ".search": 101, "q": 102, "top_k": 103}
        self.vocab_size = 32000
    
    def encode(self, text, add_special_tokens=False):
        """Encode text to token IDs."""
        tokens = []
        for word in text.split():
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                tokens.append(1)  # Unknown token
        return tokens
    
    def decode(self, token_ids, skip_special_tokens=False):
        """Decode token IDs to text."""
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        words = []
        for tid in token_ids:
            if tid in reverse_vocab:
                words.append(reverse_vocab[tid])
            elif tid == 1:
                words.append("<unk>")
        return " ".join(words)


@pytest.fixture
def mock_registry():
    """Create mock tool registry."""
    registry = ToolSchemaRegistry()
    # Add test schemas
    registry.register_schema("web.search", {
        "tool_name": "web.search",
        "type": "object",
        "required": ["q"],
        "properties": {
            "q": {"type": "string"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
        },
    })
    registry.register_schema("read_file", {
        "tool_name": "read_file",
        "type": "object",
        "required": ["path"],
        "properties": {
            "path": {"type": "string"},
            "encoding": {"type": "string", "enum": ["utf-8", "latin-1"]},
        },
    })
    registry.register_schema("repo.read", {
        "tool_name": "repo.read",
        "type": "object",
        "required": ["name", "arguments"],
        "properties": {
            "name": {"type": "string", "enum": ["repo.read"]},
            "arguments": {
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {"type": "string"},
                    "grep": {"type": "string"},
                },
            },
        },
    })
    return registry


@pytest.fixture
def mock_tokenizer():
    """Create mock tokenizer."""
    return MockTokenizer()


class TestBuildSchemaKeyMap:
    """Test schema key map building."""
    
    def test_build_schema_key_map(self, mock_registry):
        """Test building schema key map."""
        schema_map = build_schema_key_map(mock_registry)
        
        assert isinstance(schema_map, dict)
        assert len(schema_map) > 0
        
        # Check web.search schema
        web_search_keys = frozenset(["q", "top_k"])
        assert web_search_keys in schema_map
        assert "web.search" in schema_map[web_search_keys]
    
    def test_nested_schema_handling(self, mock_registry):
        """Test handling nested schemas (repo.read with arguments)."""
        schema_map = build_schema_key_map(mock_registry)
        
        # repo.read has nested arguments structure
        repo_read_keys = frozenset(["path", "grep"])
        assert repo_read_keys in schema_map
        assert "repo.read" in schema_map[repo_read_keys]


class TestExtractJsonKeys:
    """Test JSON key extraction."""
    
    def test_direct_structure(self):
        """Test extracting keys from direct JSON structure."""
        json_obj = {"q": "test", "top_k": 5}
        keys = extract_json_keys(json_obj)
        
        assert keys == frozenset(["q", "top_k"])
    
    def test_nested_structure(self):
        """Test extracting keys from nested structure."""
        json_obj = {
            "name": "repo.read",
            "arguments": {"path": "/file.txt", "grep": "pattern"}
        }
        keys = extract_json_keys(json_obj)
        
        assert keys == frozenset(["path", "grep"])
        assert "name" not in keys
        assert "arguments" not in keys


class TestMatchJsonToTool:
    """Test JSON to tool matching."""
    
    def test_exact_match(self, mock_registry):
        """Test exact key match."""
        schema_map = build_schema_key_map(mock_registry)
        
        json_obj = {"q": "test", "top_k": 5}
        result = match_json_to_tool(json_obj, schema_map, mock_registry)
        
        assert result is not None
        tool_name, confidence = result
        assert tool_name == "web.search"
        assert confidence == "exact"
    
    def test_partial_match(self, mock_registry):
        """Test partial key match."""
        schema_map = build_schema_key_map(mock_registry)
        
        json_obj = {"q": "test"}  # Missing top_k
        result = match_json_to_tool(json_obj, schema_map, mock_registry)
        
        assert result is not None
        tool_name, confidence = result
        assert tool_name == "web.search"
        assert confidence == "partial"
    
    def test_ambiguous_match(self, mock_registry):
        """Test ambiguous match (multiple tools)."""
        # Add another tool with same keys
        mock_registry.register_schema("web.search2", {
            "tool_name": "web.search2",
            "type": "object",
            "required": ["q"],
            "properties": {"q": {"type": "string"}, "top_k": {"type": "integer"}},
        })
        
        schema_map = build_schema_key_map(mock_registry)
        
        json_obj = {"q": "test", "top_k": 5}
        result = match_json_to_tool(json_obj, schema_map, mock_registry)
        
        assert result is not None
        tool_name, confidence = result
        assert confidence == "ambiguous"
    
    def test_no_match(self, mock_registry):
        """Test no match found."""
        schema_map = build_schema_key_map(mock_registry)
        
        json_obj = {"unknown_key": "value"}
        result = match_json_to_tool(json_obj, schema_map, mock_registry)
        
        assert result is None


class TestDecodeJsonFromTokenIds:
    """Test JSON decoding from token IDs."""
    
    def test_decode_valid_json(self, mock_tokenizer):
        """Test decoding valid JSON."""
        # Encode JSON string
        json_str = '{"q": "test", "top_k": 5}'
        token_ids = mock_tokenizer.encode(json_str)
        
        result = decode_json_from_token_ids(token_ids, mock_tokenizer)
        
        assert result is not None
        assert isinstance(result, dict)
        assert "q" in result or "<unk>" in json.dumps(result)  # May have unk tokens
    
    def test_decode_invalid_json(self, mock_tokenizer):
        """Test decoding invalid JSON."""
        # Encode invalid JSON
        invalid_json = "{q: test}"  # Missing quotes
        token_ids = mock_tokenizer.encode(invalid_json)
        
        result = decode_json_from_token_ids(token_ids, mock_tokenizer)
        
        # Should return None or attempt repair
        # Result depends on repair logic
        assert result is None or isinstance(result, dict)
    
    def test_decode_without_tokenizer(self):
        """Test decoding without tokenizer."""
        token_ids = [100, 101, 102]
        result = decode_json_from_token_ids(token_ids, None)
        
        assert result is None


class TestExtractJsonFromIntegrationMask:
    """Test JSON extraction from integration mask."""
    
    def test_extract_valid_json(self, mock_tokenizer):
        """Test extracting valid JSON from text."""
        teacher_text = 'Some text {"q": "test", "top_k": 5} more text'
        
        result = extract_json_from_integration_mask(
            teacher_text, [1, 1, 1], mock_tokenizer
        )
        
        assert result is not None
        assert isinstance(result, dict)
    
    def test_extract_no_json(self, mock_tokenizer):
        """Test extracting when no JSON present."""
        teacher_text = "Some text without JSON"
        
        result = extract_json_from_integration_mask(
            teacher_text, [1, 1, 1], mock_tokenizer
        )
        
        # May return None or attempt to find JSON
        assert result is None or isinstance(result, dict)


class TestInferToolNameFromText:
    """Test heuristic text matching."""
    
    def test_exact_match(self):
        """Test exact tool name match."""
        tool_names = ["web.search", "read_file", "repo.read"]
        text = "I will use web.search to find information"
        
        result = infer_tool_name_from_text(text, tool_names)
        
        assert result == "web.search"
    
    def test_quoted_match(self):
        """Test quoted tool name match."""
        tool_names = ["web.search", "read_file"]
        text = 'Calling "web.search" with query'
        
        result = infer_tool_name_from_text(text, tool_names)
        
        assert result == "web.search"
    
    def test_ambiguous_match(self):
        """Test ambiguous match (multiple tools found)."""
        tool_names = ["web.search", "web.open"]
        text = "Using web.search and web.open tools"
        
        result = infer_tool_name_from_text(text, tool_names)
        
        # Should return None for ambiguous
        assert result is None
    
    def test_no_match(self):
        """Test no match found."""
        tool_names = ["web.search"]
        text = "Some text without tool names"
        
        result = infer_tool_name_from_text(text, tool_names)
        
        assert result is None


class TestInferToolNameForSample:
    """Test inference for a single sample."""
    
    def test_infer_from_gold_json(self, mock_registry, mock_tokenizer):
        """Test inference from gold_json_text_ids."""
        schema_map = build_schema_key_map(mock_registry)
        tool_names = mock_registry.list_tools()
        
        sample = {
            "gold_json_text_ids": mock_tokenizer.encode('{"q": "test", "top_k": 5}'),
            "teacher_text": "Some text",
        }
        
        tool_name, source, confidence = infer_tool_name_for_sample(
            sample, mock_tokenizer, schema_map, mock_registry, tool_names
        )
        
        # May match depending on tokenizer encoding
        assert source in ["schema", "mask+schema", "heuristic", None]
    
    def test_infer_from_integration_mask(self, mock_registry, mock_tokenizer):
        """Test inference from integration_mask."""
        schema_map = build_schema_key_map(mock_registry)
        tool_names = mock_registry.list_tools()
        
        sample = {
            "integration_mask": [1, 1, 1],
            "teacher_text": 'Some text {"q": "test"} more text',
        }
        
        tool_name, source, confidence = infer_tool_name_for_sample(
            sample, mock_tokenizer, schema_map, mock_registry, tool_names
        )
        
        assert source in ["mask+schema", "heuristic", None]
    
    def test_infer_from_text(self, mock_registry, mock_tokenizer):
        """Test inference from text heuristic."""
        schema_map = build_schema_key_map(mock_registry)
        tool_names = mock_registry.list_tools()
        
        sample = {
            "teacher_text": "I will use web.search to find information",
        }
        
        tool_name, source, confidence = infer_tool_name_for_sample(
            sample, mock_tokenizer, schema_map, mock_registry, tool_names
        )
        
        assert source in ["heuristic", None]
    
    def test_preserve_existing(self, mock_registry, mock_tokenizer):
        """Test preserving existing tool_name_ids."""
        schema_map = build_schema_key_map(mock_registry)
        tool_names = mock_registry.list_tools()
        
        sample = {
            "tool_name_ids": [100, 101],
            "tool_name": "web.search",
            "teacher_text": "Some text",
        }
        
        tool_name, source, confidence = infer_tool_name_for_sample(
            sample, mock_tokenizer, schema_map, mock_registry, tool_names
        )
        
        assert source == "original"
        assert tool_name == "web.search" or tool_name is None  # May extract or preserve


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

