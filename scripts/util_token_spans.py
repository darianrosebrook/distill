"""
Token span alignment utilities for byte-to-token mapping.

Author: @darianrosebrook
"""

from typing import Tuple, Optional, List
import unicodedata


def bytes_to_token_span(text: str, start: int, end: int, tokenizer) -> Optional[Tuple[int, int]]:
    """
    Map byte span to token span using fast tokenizer offsets.

    Args:
        text: Full text (must match normalization used for byte spans)
        start: Start byte offset
        end: End byte offset
        tokenizer: Fast tokenizer instance with offset_mapping support

    Returns:
        (start_token_idx, end_token_idx) or None if alignment fails
    """
    if not hasattr(tokenizer, "is_fast") or not tokenizer.is_fast:
        # Fallback for slow tokenizers
        pre = text[:start]
        ids_pre = tokenizer.encode(pre, add_special_tokens=False)
        ids_all = tokenizer.encode(text[:end], add_special_tokens=False)
        i = len(ids_pre)
        j = len(ids_all)
        # Basic verification
        if tokenizer.decode(ids_all[i:j]).strip() == text[start:end].strip():
            return (i, j)
        return None

    # Use fast tokenizer offset mapping for robust alignment
    try:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]  # [(start, end), ...] per token

        # Find first token whose span intersects [start, end)
        start_tok = None
        for i, (s, e) in enumerate(offsets):
            if not (e <= start) and not (s >= end):
                start_tok = i
                break

        if start_tok is None:
            return None

        # Find last token intersecting [start, end)
        end_tok = start_tok
        for j in range(start_tok, len(offsets)):
            s, e = offsets[j]
            if s >= end:
                break
            end_tok = j + 1

        if end_tok <= start_tok:
            return None

        return (start_tok, end_tok)
    except Exception:
        # Fallback on error
        pre = text[:start]
        ids_pre = tokenizer.encode(pre, add_special_tokens=False)
        ids_all = tokenizer.encode(text[:end], add_special_tokens=False)
        i = len(ids_pre)
        j = len(ids_all)
        if tokenizer.decode(ids_all[i:j]).strip() == text[start:end].strip():
            return (i, j)
        return None


def byte_spans_to_token_spans(
    text: str, spans: List[List[int]], tokenizer
) -> List[Optional[Tuple[int, int]]]:
    """
    Map multiple byte spans to token spans using fast tokenizer offsets.

    Args:
        text: Full text (must match normalization used for byte spans)
        spans: List of [start, end] byte offset pairs
        tokenizer: Fast tokenizer instance

    Returns:
        List of (start_token_idx, end_token_idx) tuples or None for failed alignments
    """
    if not spans:
        return []

    if not hasattr(tokenizer, "is_fast") or not tokenizer.is_fast:
        # Fallback: map individually
        return [bytes_to_token_span(text, s[0], s[1], tokenizer) for s in spans]

    try:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]

        token_spans = []
        for a, b in spans:
            # Find first token intersecting [a, b)
            start_tok = None
            for i, (s, e) in enumerate(offsets):
                if not (e <= a) and not (s >= b):
                    start_tok = i
                    break

            if start_tok is None:
                token_spans.append(None)
                continue

            # Find last token intersecting [a, b)
            end_tok = start_tok
            for j in range(start_tok, len(offsets)):
                s, e = offsets[j]
                if s >= b:
                    break
                end_tok = j + 1

            if end_tok <= start_tok:
                token_spans.append(None)
            else:
                token_spans.append((start_tok, end_tok))

        return token_spans
    except Exception:
        # Fallback on error
        return [bytes_to_token_span(text, s[0], s[1], tokenizer) for s in spans]


def normalize_text_for_alignment(
    text: str, text_norm: Optional[str] = None, line_endings: Optional[str] = None
) -> str:
    """
    Normalize text to match the format used when computing byte spans.

    Args:
        text: Original text
        text_norm: Normalization format ("NFC" or None)
        line_endings: Line ending format ("LF" or None)

    Returns:
        Normalized text
    """
    if text_norm == "NFC":
        text = unicodedata.normalize("NFC", text)

    if line_endings == "LF":
        text = text.replace("\r\n", "\n").replace("\r", "\n")

    return text
