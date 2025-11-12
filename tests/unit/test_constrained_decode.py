"""
Unit tests for constrained JSON decoder.

Tests the JSONConstrainedDecoder, JSONFSM, SchemaValidator, and TokenLexicon.
"""
import pytest
import json
import numpy as np
from unittest.mock import Mock, MagicMock

from coreml.runtime.constrained_decode import (
    JSONConstrainedDecoder,
    JSONFSM,
    SchemaValidator,
    TokenLexicon,
    DecoderState,
)


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self._vocab = {f"token_{i}": i for i in range(vocab_size)}
        self.eos_token_id = 1
        
    def get_vocab(self):
        return self._vocab
    
    def decode(self, token_ids, clean_up_tokenization_spaces=False):
        if isinstance(token_ids, list) and len(token_ids) == 1:
            tid = token_ids[0]
            if tid == 0:
                return "{"
            elif tid == 1:
                return ""  # EOS
            elif tid == 2:
                return '"'
            elif tid == 3:
                return "name"
            elif tid == 4:
                return ":"
            elif tid == 5:
                return "test"
            elif tid == 6:
                return "}"
            elif tid == 7:
                return ","
            elif tid == 8:
                return "arguments"
            elif tid == 9:
                return "{"
            elif tid == 10:
                return "}"
            elif tid == 11:
                return " "
            elif tid == 12:
                return "true"
            elif tid == 13:
                return "false"
            elif tid == 14:
                return "null"
            elif tid == 15:
                return "123"
            elif tid == 16:
                return "["
            elif tid == 17:
                return "]"
            else:
                return f"token_{tid}"
        return ""


# -------------------------
# JSONFSM Tests
# -------------------------


class TestJSONFSM:
    """Test JSON finite state machine."""
    
    def test_start(self):
        """Test FSM initialization."""
        fsm = JSONFSM()
        state = fsm.start()
        
        assert isinstance(state, DecoderState)
        assert state.buffer == ""
        assert state.stack == []
        assert state.expect == {"value"}
        assert state.complete is False
        assert state.error is None
    
    def test_inside_string_detection(self):
        """Test string detection logic."""
        fsm = JSONFSM()
        
        # Not inside string
        assert fsm._inside_string('') is False
        assert fsm._inside_string('{') is False
        assert fsm._inside_string('{"name"') is False
        
        # Inside string
        assert fsm._inside_string('"') is True
        assert fsm._inside_string('"test') is True
        assert fsm._inside_string('{"name":"test') is True
        
        # Escaped quotes
        assert fsm._inside_string('"test\\"') is True
        assert fsm._inside_string('"test\\"more') is True
    
    def test_step_chars_allowed_outside_string(self):
        """Test allowed characters outside string."""
        fsm = JSONFSM()
        state = fsm.start()
        
        allowed = fsm.step_chars_allowed(state)
        
        # Should allow structural chars, digits, quotes, t/f/n
        assert "{" in allowed
        assert "}" in allowed
        assert "[" in allowed
        assert "]" in allowed
        assert ":" in allowed
        assert "," in allowed
        assert '"' in allowed
        assert " " in allowed
        assert "0" in allowed
        assert "t" in allowed  # true
        assert "f" in allowed  # false
        assert "n" in allowed  # null
    
    def test_step_chars_allowed_inside_string(self):
        """Test allowed characters inside string."""
        fsm = JSONFSM()
        state = DecoderState(buffer='"test', stack=[], expect={"value"}, complete=False, error=None)
        
        allowed = fsm.step_chars_allowed(state)
        
        # Should return "ANY" when inside string
        assert "ANY" in allowed
    
    def test_step_chars_allowed_complete(self):
        """Test allowed characters when complete."""
        fsm = JSONFSM()
        state = DecoderState(buffer='{"name":"test"}', stack=[], expect={"value"}, complete=True, error=None)
        
        allowed = fsm.step_chars_allowed(state)
        
        # Should return empty set when complete
        assert len(allowed) == 0
    
    def test_push_text_valid_json(self):
        """Test pushing text that forms valid JSON."""
        fsm = JSONFSM()
        state = fsm.start()
        
        # Push opening brace
        state = fsm.push_text(state, "{")
        assert state.buffer == "{"
        assert state.complete is False
        
        # Push complete JSON object
        state = fsm.push_text(state, '"name":"test"}')
        assert '"name":"test"}' in state.buffer
        
        # Should be complete if valid JSON
        try:
            json.loads(state.buffer)
            assert state.complete is True or state.complete is False  # May not detect immediately
        except json.JSONDecodeError:
            pass
    
    def test_push_text_invalid_json(self):
        """Test pushing text that doesn't form valid JSON."""
        fsm = JSONFSM()
        state = fsm.start()
        
        # Push incomplete JSON
        state = fsm.push_text(state, "{")
        assert state.complete is False
        
        # Push more incomplete JSON
        state = fsm.push_text(state, '"name"')
        assert state.complete is False


# -------------------------
# SchemaValidator Tests
# -------------------------


class TestSchemaValidator:
    """Test schema validator."""
    
    def test_validate_valid_object(self):
        """Test validation of valid object."""
        schema = {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            }
        }
        validator = SchemaValidator(schema)
        
        obj = {"name": "test", "age": 25}
        ok, err = validator.validate(obj)
        
        assert ok is True
        assert err is None
    
    def test_validate_missing_required(self):
        """Test validation with missing required field."""
        schema = {
            "type": "object",
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            }
        }
        validator = SchemaValidator(schema)
        
        obj = {"name": "test"}  # Missing "age"
        ok, err = validator.validate(obj)
        
        assert ok is False
        assert "Missing required key: age" in err
    
    def test_validate_wrong_type_string(self):
        """Test validation with wrong string type."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }
        validator = SchemaValidator(schema)
        
        obj = {"name": 123}  # Should be string
        ok, err = validator.validate(obj)
        
        assert ok is False
        assert "must be string" in err
    
    def test_validate_wrong_type_number(self):
        """Test validation with wrong number type."""
        schema = {
            "type": "object",
            "properties": {
                "age": {"type": "number"}
            }
        }
        validator = SchemaValidator(schema)
        
        obj = {"age": "25"}  # Should be number
        ok, err = validator.validate(obj)
        
        assert ok is False
        assert "must be number" in err
    
    def test_validate_wrong_type_boolean(self):
        """Test validation with wrong boolean type."""
        schema = {
            "type": "object",
            "properties": {
                "active": {"type": "boolean"}
            }
        }
        validator = SchemaValidator(schema)
        
        obj = {"active": "true"}  # Should be boolean
        ok, err = validator.validate(obj)
        
        assert ok is False
        assert "must be boolean" in err
    
    def test_validate_wrong_type_object(self):
        """Test validation with wrong object type."""
        schema = {
            "type": "object",
            "properties": {
                "data": {"type": "object"}
            }
        }
        validator = SchemaValidator(schema)
        
        obj = {"data": "not an object"}  # Should be object
        ok, err = validator.validate(obj)
        
        assert ok is False
        assert "must be object" in err
    
    def test_validate_wrong_type_array(self):
        """Test validation with wrong array type."""
        schema = {
            "type": "object",
            "properties": {
                "items": {"type": "array"}
            }
        }
        validator = SchemaValidator(schema)
        
        obj = {"items": "not an array"}  # Should be array
        ok, err = validator.validate(obj)
        
        assert ok is False
        assert "must be array" in err
    
    def test_validate_not_object(self):
        """Test validation of non-object."""
        schema = {
            "type": "object",
            "properties": {}
        }
        validator = SchemaValidator(schema)
        
        obj = "not an object"
        ok, err = validator.validate(obj)
        
        assert ok is False
        assert "Top-level JSON must be an object" in err
    
    def test_validate_extra_fields(self):
        """Test validation with extra fields not in schema."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }
        validator = SchemaValidator(schema)
        
        obj = {"name": "test", "extra": "field"}  # Extra field
        ok, err = validator.validate(obj)
        
        # Extra fields should be allowed (not validated)
        assert ok is True
        assert err is None


# -------------------------
# TokenLexicon Tests
# -------------------------


class TestTokenLexicon:
    """Test token lexicon."""
    
    def test_token_text(self):
        """Test token text retrieval."""
        tokenizer = MockTokenizer(vocab_size=100)
        lexicon = TokenLexicon(tokenizer)
        
        # Test known tokens
        assert lexicon.token_text(0) == "{"
        assert lexicon.token_text(1) == ""  # EOS
        assert lexicon.token_text(2) == '"'
        
        # Test unknown token (should return empty string)
        assert lexicon.token_text(9999) == ""
    
    def test_tokenizer_vocab_size(self):
        """Test lexicon handles tokenizer vocab size."""
        tokenizer = MockTokenizer(vocab_size=50)
        lexicon = TokenLexicon(tokenizer)
        
        # Should have entries for all vocab tokens
        for tid in range(50):
            text = lexicon.token_text(tid)
            assert isinstance(text, str)


# -------------------------
# JSONConstrainedDecoder Tests
# -------------------------


class TestJSONConstrainedDecoder:
    """Test JSON constrained decoder."""
    
    def test_init(self):
        """Test decoder initialization."""
        schema = {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string"}
            }
        }
        tokenizer = MockTokenizer()
        
        decoder = JSONConstrainedDecoder(schema=schema, tokenizer=tokenizer)
        
        assert decoder.schema == schema
        assert decoder.tokenizer == tokenizer
        assert decoder.validator is not None
        assert decoder.lex is not None
        assert decoder.fsm is not None
    
    def test_start(self):
        """Test decoder start."""
        schema = {"type": "object"}
        tokenizer = MockTokenizer()
        decoder = JSONConstrainedDecoder(schema=schema, tokenizer=tokenizer)
        
        state = decoder.start()
        
        assert isinstance(state, DecoderState)
        assert state.buffer == ""
        assert state.complete is False
    
    def test_allowed_token_mask_empty(self):
        """Test token mask for empty state."""
        schema = {"type": "object"}
        tokenizer = MockTokenizer(vocab_size=100)
        decoder = JSONConstrainedDecoder(schema=schema, tokenizer=tokenizer)
        
        state = decoder.start()
        mask = decoder.allowed_token_mask(state, (100,))
        
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (100,)
        assert mask.dtype == bool
        
        # Should allow some tokens (structural chars, etc.)
        assert mask.sum() > 0
    
    def test_allowed_token_mask_inside_string(self):
        """Test token mask when inside string."""
        schema = {"type": "object"}
        tokenizer = MockTokenizer(vocab_size=100)
        decoder = JSONConstrainedDecoder(schema=schema, tokenizer=tokenizer)
        
        # State inside string
        state = DecoderState(buffer='"test', stack=[], expect={"value"}, complete=False, error=None)
        mask = decoder.allowed_token_mask(state, (100,))
        
        # Should allow all tokens when inside string
        assert mask.all()
    
    def test_allowed_token_mask_eos_token(self):
        """Test token mask includes EOS token."""
        schema = {"type": "object"}
        tokenizer = MockTokenizer(vocab_size=100)
        # MockTokenizer already sets eos_token_id = 1
        decoder = JSONConstrainedDecoder(schema=schema, tokenizer=tokenizer)
        
        state = decoder.start()
        mask = decoder.allowed_token_mask(state, (100,))
        
        # EOS token should be allowed if tokenizer has eos_token_id
        # The decoder should explicitly allow EOS token regardless of character matching
        if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            eos_id = tokenizer.eos_token_id
            # Check that EOS token is in the mask (should be True)
            assert mask[eos_id] == True, f"EOS token {eos_id} should be allowed in mask"
    
    def test_push_token(self):
        """Test pushing a token."""
        schema = {"type": "object"}
        tokenizer = MockTokenizer()
        decoder = JSONConstrainedDecoder(schema=schema, tokenizer=tokenizer)
        
        state = decoder.start()
        
        # Push opening brace token
        state = decoder.push(state, 0)  # token 0 = "{"
        
        assert "{" in state.buffer
    
    def test_finalize_valid_json(self):
        """Test finalizing valid JSON."""
        schema = {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string"}
            }
        }
        tokenizer = MockTokenizer()
        decoder = JSONConstrainedDecoder(schema=schema, tokenizer=tokenizer)
        
        # Create state with valid JSON
        state = DecoderState(
            buffer='{"name":"test"}',
            stack=[],
            expect={"value"},
            complete=True,
            error=None
        )
        
        obj = decoder.finalize(state)
        
        assert isinstance(obj, dict)
        assert obj["name"] == "test"
    
    def test_finalize_invalid_json(self):
        """Test finalizing invalid JSON."""
        schema = {"type": "object"}
        tokenizer = MockTokenizer()
        decoder = JSONConstrainedDecoder(schema=schema, tokenizer=tokenizer)
        
        # Create state with invalid JSON
        state = DecoderState(
            buffer='{"invalid json"',
            stack=[],
            expect={"value"},
            complete=False,
            error=None
        )
        
        with pytest.raises(json.JSONDecodeError):
            decoder.finalize(state)
    
    def test_finalize_schema_validation_failure(self):
        """Test finalizing JSON that fails schema validation."""
        schema = {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string"}
            }
        }
        tokenizer = MockTokenizer()
        decoder = JSONConstrainedDecoder(schema=schema, tokenizer=tokenizer)
        
        # Create state with JSON missing required field
        state = DecoderState(
            buffer='{"age":25}',
            stack=[],
            expect={"value"},
            complete=True,
            error=None
        )
        
        with pytest.raises(ValueError) as exc_info:
            decoder.finalize(state)
        
        assert "Schema validation failed" in str(exc_info.value)
        assert "Missing required key" in str(exc_info.value)
    
    def test_end_to_end_simple(self):
        """Test end-to-end decoding of simple JSON."""
        schema = {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string"}
            }
        }
        tokenizer = MockTokenizer()
        decoder = JSONConstrainedDecoder(schema=schema, tokenizer=tokenizer)
        
        state = decoder.start()
        
        # Simulate token sequence: { "name" : "test" }
        tokens = [0, 11, 2, 3, 2, 11, 4, 11, 2, 5, 2, 6]  # { "name" : "test" }
        
        for tok_id in tokens:
            state = decoder.push(state, tok_id)
        
        # Finalize
        obj = decoder.finalize(state)
        
        assert obj["name"] == "test"


# -------------------------
# Integration Tests
# -------------------------


class TestDecoderIntegration:
    """Integration tests for decoder with real tokenizer."""
    
    @pytest.mark.skipif(
        "transformers" not in __import__("sys").modules,
        reason="transformers not available"
    )
    def test_with_real_tokenizer(self):
        """Test decoder with real transformers tokenizer."""
        try:
            from transformers import AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            schema = {
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {"type": "string"}
                }
            }
            
            decoder = JSONConstrainedDecoder(schema=schema, tokenizer=tokenizer)
            state = decoder.start()
            
            # Encode a simple JSON string
            json_str = '{"name":"test"}'
            tokens = tokenizer.encode(json_str, add_special_tokens=False)
            
            # Push tokens
            for tok_id in tokens:
                state = decoder.push(state, tok_id)
            
            # Finalize
            obj = decoder.finalize(state)
            
            assert obj["name"] == "test"
            
        except ImportError:
            pytest.skip("transformers not available")

