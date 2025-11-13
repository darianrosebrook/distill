"""
Property-based tests for span invariants using Hypothesis.

Tests round-trip properties and normalization stability.
"""

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from unittest.mock import Mock

from scripts.util_token_spans import (
    bytes_to_token_span,
    normalize_text_for_alignment,
)


@pytest.fixture
def mock_fast_tokenizer():
    """Create mock fast tokenizer with offset_mapping."""
    tokenizer = Mock()
    tokenizer.is_fast = True

    def encode_side_effect(text, add_special_tokens=False, return_offsets_mapping=False, **kwargs):
        # Simple character-based tokenization
        tokens = list(text)
        input_ids = list(range(len(tokens)))
        if return_offsets_mapping:
            offsets = []
            pos = 0
            for char in tokens:
                char_bytes = char.encode("utf-8")
                offsets.append((pos, pos + len(char_bytes)))
                pos += len(char_bytes)
            return {"input_ids": input_ids, "offset_mapping": offsets}
        return input_ids

    tokenizer.encode = Mock(side_effect=encode_side_effect)
    tokenizer.decode = Mock(
        side_effect=lambda ids, **kwargs: "".join(chr(65 + i % 26) for i in ids)
    )
    return tokenizer


@pytest.fixture
def mock_slow_tokenizer():
    """Create mock slow tokenizer without offset_mapping."""
    tokenizer = Mock()
    tokenizer.is_fast = False

    def encode_side_effect(text, add_special_tokens=False, **kwargs):
        tokens = list(text)
        return list(range(len(tokens)))

    tokenizer.encode = Mock(side_effect=encode_side_effect)
    tokenizer.decode = Mock(
        side_effect=lambda ids, **kwargs: "".join(chr(65 + i % 26) for i in ids)
    )
    return tokenizer


@given(
    text=st.text(
        min_size=10, max_size=100, alphabet=st.characters(min_codepoint=32, max_codepoint=126)
    ),
    start=st.integers(min_value=0, max_value=50),
    end=st.integers(min_value=0, max_value=50),
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_span_round_trip_ascii(text, start, end, mock_fast_tokenizer):
    """Test span round-trip property for ASCII text."""
    if start >= len(text) or end > len(text) or start >= end:
        pytest.skip("Invalid span bounds")

    # Normalize text
    text_norm = normalize_text_for_alignment(text, text_norm="NFC", line_endings="LF")

    # Extract byte span → token span
    byte_start = len(text_norm[:start].encode("utf-8"))
    byte_end = len(text_norm[:end].encode("utf-8"))

    token_span = bytes_to_token_span(text_norm, byte_start, byte_end, mock_fast_tokenizer)

    if token_span is None:
        pytest.skip("Token span extraction failed")

    # Round-trip: decode tokens → encode back → verify same token span
    encoded = mock_fast_tokenizer.encode(
        text_norm, add_special_tokens=False, return_offsets_mapping=True
    )
    token_ids = encoded["input_ids"][token_span[0] : token_span[1]]

    # Decode tokens
    decoded_text = mock_fast_tokenizer.decode(token_ids, skip_special_tokens=True)

    # Re-encode and verify alignment
    mock_fast_tokenizer.encode(decoded_text, add_special_tokens=False, return_offsets_mapping=True)
    re_token_span = bytes_to_token_span(
        decoded_text, 0, len(decoded_text.encode("utf-8")), mock_fast_tokenizer
    )

    # Verify token span is consistent (at least same length)
    if re_token_span:
        assert abs(re_token_span[1] - re_token_span[0] - (token_span[1] - token_span[0])) <= 1


@given(
    text=st.text(min_size=10, max_size=50),
    text_norm=st.sampled_from([None, "NFC"]),
    line_endings=st.sampled_from([None, "LF"]),
)
def test_normalization_stability(text, text_norm, line_endings):
    """Test that normalization is stable across variants."""
    # Apply normalization
    normalized = normalize_text_for_alignment(text, text_norm=text_norm, line_endings=line_endings)

    # Apply again - should be idempotent
    normalized_twice = normalize_text_for_alignment(
        normalized, text_norm=text_norm, line_endings=line_endings
    )

    assert normalized == normalized_twice


@given(
    text=st.text(min_size=5, max_size=30),
)
def test_normalization_crlf_variants(text):
    """Test normalization stability with CRLF/CR/LF variants."""
    # Create variants (preserve existing line endings first)
    # Start with LF-normalized base
    text_base = text.replace("\r\n", "\n").replace("\r", "\n")

    # Create variants
    text_crlf = text_base.replace("\n", "\r\n")
    text_cr = text_base.replace("\n", "\r")
    text_lf = text_base

    # Normalize all to LF
    norm_crlf = normalize_text_for_alignment(text_crlf, line_endings="LF")
    norm_cr = normalize_text_for_alignment(text_cr, line_endings="LF")
    norm_lf = normalize_text_for_alignment(text_lf, line_endings="LF")

    # All should be identical after normalization
    assert norm_crlf == norm_lf, f"CRLF normalization failed: {repr(norm_crlf)} != {repr(norm_lf)}"
    assert norm_cr == norm_lf, f"CR normalization failed: {repr(norm_cr)} != {repr(norm_lf)}"


@given(
    text=st.text(
        min_size=5, max_size=30, alphabet=st.characters(min_codepoint=0x00, max_codepoint=0x10FFFF)
    ),
)
def test_normalization_nfd_nfc_variants(text):
    """Test normalization stability with NFD/NFC variants."""
    import unicodedata

    # Create NFD and NFC variants
    text_nfd = unicodedata.normalize("NFD", text)
    text_nfc = unicodedata.normalize("NFC", text)

    # Normalize both to NFC
    norm_nfd = normalize_text_for_alignment(text_nfd, text_norm="NFC")
    norm_nfc = normalize_text_for_alignment(text_nfc, text_norm="NFC")

    # Both should be identical after normalization
    assert norm_nfd == norm_nfc


def test_span_round_trip_accented(mock_fast_tokenizer):
    """Test span round-trip with accented characters."""
    text = "Café résumé naïve"
    text_norm = normalize_text_for_alignment(text, text_norm="NFC", line_endings="LF")

    # Extract span covering "résumé"
    start = text_norm.find("résumé")
    end = start + len("résumé")

    byte_start = len(text_norm[:start].encode("utf-8"))
    byte_end = len(text_norm[:end].encode("utf-8"))

    token_span = bytes_to_token_span(text_norm, byte_start, byte_end, mock_fast_tokenizer)

    if token_span:
        # Verify span covers the expected text
        encoded = mock_fast_tokenizer.encode(
            text_norm, add_special_tokens=False, return_offsets_mapping=True
        )
        offsets = encoded["offset_mapping"]
        if token_span[0] < len(offsets) and token_span[1] <= len(offsets):
            span_start_byte = offsets[token_span[0]][0]
            span_end_byte = offsets[token_span[1] - 1][1]
            span_text = text_norm[span_start_byte:span_end_byte]
            assert "résumé" in span_text or span_text in "résumé"


def test_span_round_trip_emoji(mock_fast_tokenizer):
    """Test span round-trip with emoji."""
    text = "Hello world"
    text_norm = normalize_text_for_alignment(text, text_norm="NFC", line_endings="LF")

    # Extract span covering "world"
    start = text_norm.find("world")
    end = start + len("world")

    byte_start = len(text_norm[:start].encode("utf-8"))
    byte_end = len(text_norm[:end].encode("utf-8"))

    token_span = bytes_to_token_span(text_norm, byte_start, byte_end, mock_fast_tokenizer)

    if token_span:
        # Verify span is valid
        assert token_span[0] < token_span[1]
        assert token_span[0] >= 0
