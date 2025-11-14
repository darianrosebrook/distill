"""
Unit tests for sentinel token encoding/decoding.
"""
# @author: @darianrosebrook

import json
from pathlib import Path

from models.student.tokenizer.constants import BOT_TOKEN_ID, EOT_TOKEN_ID, BOT_TOKEN, EOT_TOKEN


class TestSentinelTokens:
    """Test sentinel token functionality."""

    def test_token_ids_defined(self):
        """Test that token IDs are defined."""
        # Note: Actual tokenizer encoding returns 32000 and 32001,
        # not 3 and 4 as shown in tokenizer_config.json added_tokens_decoder
        assert BOT_TOKEN_ID == 32000  # Actual tokenizer encoding
        assert EOT_TOKEN_ID == 32001  # Actual tokenizer encoding

    def test_token_strings_defined(self):
        """Test that token strings are defined."""
        assert BOT_TOKEN == "<bot>"
        assert EOT_TOKEN == "<eot>"

    def test_tokens_in_special_tokens_map(self):
        """Test that tokens are in special_tokens_map.json."""
        map_path = Path("models/student/tokenizer/special_tokens_map.json")
        if map_path.exists():
            with open(map_path) as f:
                special_tokens = json.load(f)

            assert "bot_token" in special_tokens
            assert "eot_token" in special_tokens
            assert special_tokens["bot_token"]["content"] == BOT_TOKEN
            assert special_tokens["eot_token"]["content"] == EOT_TOKEN

    def test_tokens_in_tokenizer_config(self):
        """Test that tokens are in tokenizer_config.json."""
        config_path = Path("models/student/tokenizer/tokenizer_config.json")
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

            added_tokens = config.get("added_tokens_decoder", {})
            assert "3" in added_tokens
            assert "4" in added_tokens
            assert added_tokens["3"]["content"] == BOT_TOKEN
            assert added_tokens["4"]["content"] == EOT_TOKEN
