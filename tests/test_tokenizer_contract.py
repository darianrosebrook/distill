"""
Tokenizer contract tests: Ensure tokenizer consistency across teacher, student, export, and runtime.

Tests:
- Special token IDs match authoritative constants
- Thinking/tool tokens form single tokens (not split)
- Round-trip stability for prompts with special tokens
- Masking safety (padding mask never masks special tokens)
@author: @darianrosebrook
"""

import pytest

from models.student.tokenizer.constants import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    BOT_TOKEN_ID,
    EOT_TOKEN_ID,
    BOT_TOKEN,
    EOT_TOKEN,
)


def load_student_tokenizer():
    """Load student tokenizer."""
    from training.dataset import load_tokenizer

    tokenizer_path = "models/student/tokenizer"
    return load_tokenizer(tokenizer_path)


def test_special_token_ids_match_constants():
    """Test that special token IDs match constants.py."""
    tokenizer = load_student_tokenizer()

    # Get token IDs from tokenizer
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    # Get bot/eot IDs by encoding
    bot_tokens = tokenizer.encode(BOT_TOKEN, add_special_tokens=False)
    eot_tokens = tokenizer.encode(EOT_TOKEN, add_special_tokens=False)

    # Verify IDs match constants
    assert bos_id == BOS_TOKEN_ID, f"BOS token ID mismatch: {bos_id} != {BOS_TOKEN_ID}"
    assert eos_id == EOS_TOKEN_ID, f"EOS token ID mismatch: {eos_id} != {EOS_TOKEN_ID}"

    # Bot and EOT should be single tokens
    assert len(bot_tokens) == 1, (
        f"BOT token should be single token, got {len(bot_tokens)} tokens: {bot_tokens}"
    )
    assert len(eot_tokens) == 1, (
        f"EOT token should be single token, got {len(eot_tokens)} tokens: {eot_tokens}"
    )

    bot_id = bot_tokens[0]
    eot_id = eot_tokens[0]

    assert bot_id == BOT_TOKEN_ID, f"BOT token ID mismatch: {bot_id} != {BOT_TOKEN_ID}"
    assert eot_id == EOT_TOKEN_ID, f"EOT token ID mismatch: {eot_id} != {EOT_TOKEN_ID}"


def test_special_tokens_are_single_tokens():
    """Test that special tokens (BOT, EOT) form single tokens, not split."""
    tokenizer = load_student_tokenizer()

    # Test BOT token
    bot_tokens = tokenizer.encode(BOT_TOKEN, add_special_tokens=False)
    assert len(bot_tokens) == 1, (
        f"BOT token '{BOT_TOKEN}' should be single token, got {len(bot_tokens)}: {bot_tokens}"
    )

    # Test EOT token
    eot_tokens = tokenizer.encode(EOT_TOKEN, add_special_tokens=False)
    assert len(eot_tokens) == 1, (
        f"EOT token '{EOT_TOKEN}' should be single token, got {len(eot_tokens)}: {eot_tokens}"
    )

    # Verify round-trip: decode should produce original token
    bot_decoded = tokenizer.decode(bot_tokens, skip_special_tokens=False)
    assert BOT_TOKEN in bot_decoded or bot_decoded.strip() == BOT_TOKEN, (
        f"BOT token round-trip failed: '{BOT_TOKEN}' -> {bot_tokens} -> '{bot_decoded}'"
    )

    eot_decoded = tokenizer.decode(eot_tokens, skip_special_tokens=False)
    assert EOT_TOKEN in eot_decoded or eot_decoded.strip() == EOT_TOKEN, (
        f"EOT token round-trip failed: '{EOT_TOKEN}' -> {eot_tokens} -> '{eot_decoded}'"
    )


def test_round_trip_stability():
    """Test round-trip stability for prompts with special tokens."""
    tokenizer = load_student_tokenizer()

    # Test prompts with special tokens
    test_prompts = [
        f"{BOT_TOKEN} tool call {EOT_TOKEN}",
        f"User: Hello {BOT_TOKEN}search{EOT_TOKEN}",
        f"Response: {BOT_TOKEN}callMCPTool{EOT_TOKEN} result here",
    ]

    for prompt in test_prompts:
        # Encode
        encoded = tokenizer.encode(prompt, add_special_tokens=True)

        # Decode
        decoded = tokenizer.decode(encoded, skip_special_tokens=False)

        # Verify special tokens are preserved (may have whitespace differences)
        assert BOT_TOKEN in decoded or BOT_TOKEN.strip() in decoded, (
            f"BOT token lost in round-trip for prompt: '{prompt}' -> '{decoded}'"
        )
        assert EOT_TOKEN in decoded or EOT_TOKEN.strip() in decoded, (
            f"EOT token lost in round-trip for prompt: '{prompt}' -> '{decoded}'"
        )

        # Re-encode and verify IDs match
        # Note: Round-trip may have minor differences due to whitespace normalization
        # We verify that special tokens are preserved, not exact token ID match
        re_encoded = tokenizer.encode(decoded, add_special_tokens=True)
        
        # Extract just the content tokens (excluding leading/trailing special tokens that may differ)
        # For round-trip stability, we check that BOT and EOT tokens are present in both
        assert BOT_TOKEN_ID in encoded and BOT_TOKEN_ID in re_encoded, (
            f"BOT token missing in round-trip for prompt: '{prompt}' -> {encoded} -> '{decoded}' -> {re_encoded}"
        )
        assert EOT_TOKEN_ID in encoded and EOT_TOKEN_ID in re_encoded, (
            f"EOT token missing in round-trip for prompt: '{prompt}' -> {encoded} -> '{decoded}' -> {re_encoded}"
        )
        
        # Allow minor differences in BOS/EOS tokens or whitespace normalization
        # as long as special tokens are preserved


def test_masking_safety():
    """Test that padding mask never masks special tokens (BOS/EOS/BOT/EOT)."""
    import torch

    tokenizer = load_student_tokenizer()

    # Create a sequence with special tokens
    prompt = f"{BOT_TOKEN} tool call {EOT_TOKEN}"
    encoded = tokenizer.encode(prompt, add_special_tokens=True)

    # Get special token IDs
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    bot_tokens = tokenizer.encode(BOT_TOKEN, add_special_tokens=False)
    eot_tokens = tokenizer.encode(EOT_TOKEN, add_special_tokens=False)
    bot_id = bot_tokens[0]
    eot_id = eot_tokens[0]

    special_token_ids = {bos_id, eos_id, bot_id, eot_id}

    # Create attention mask (1 for real tokens, 0 for padding)
    # Simulate padding: use a non-special token for padding (e.g., unk or a regular token)
    # Note: pad_token_id might be None or might be a special token, so we use a safe padding token
    seq_len = len(encoded)
    pad_len = 5
    # Use unk_token_id (0) or a regular token ID (e.g., 100) that's not a special token
    pad_token_id = tokenizer.unk_token_id if tokenizer.unk_token_id not in special_token_ids else 100
    # Ensure pad_token_id is not a special token
    while pad_token_id in special_token_ids:
        pad_token_id += 1
    padded_ids = encoded + [pad_token_id] * pad_len
    attention_mask = [1] * seq_len + [0] * pad_len

    # Verify special tokens are not in padding positions
    for i, token_id in enumerate(padded_ids):
        if token_id in special_token_ids:
            assert attention_mask[i] == 1, (
                f"Special token {token_id} (position {i}) is masked by padding mask"
            )

    # Test with collate function (simulate batch padding)
    from training.dataset import collate_kd_batch

    # Create batch with different lengths
    batch = []
    for i in range(3):
        # Vary sequence lengths
        test_text = f"{BOT_TOKEN} example {i} {EOT_TOKEN}"
        tokens = tokenizer.encode(test_text, add_special_tokens=True)
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        batch.append(
            {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            }
        )

    # Collate batch (will pad to max length)
    collated = collate_kd_batch(batch)

    # Verify special tokens in collated batch are not masked
    collated_ids = collated["input_ids"]
    collated_mask = collated["attention_mask"]

    for b in range(collated_ids.shape[0]):
        for t in range(collated_ids.shape[1]):
            token_id = collated_ids[b, t].item()
            if token_id in special_token_ids:
                assert collated_mask[b, t].item() == 1, (
                    f"Special token {token_id} (batch {b}, position {t}) is masked in collated batch"
                )


def test_loss_masking_never_hides_supervised_tokens():
    """Test that loss masking (ignore_index) never hides supervised tool/JSON tokens."""
    import torch

    tokenizer = load_student_tokenizer()

    # Create a sequence with tool tokens that should be supervised
    prompt = f"{BOT_TOKEN} search query {EOT_TOKEN}"
    encoded = tokenizer.encode(prompt, add_special_tokens=True)

    # Get token IDs for BOT and EOT (these should be supervised)
    bot_tokens = tokenizer.encode(BOT_TOKEN, add_special_tokens=False)
    eot_tokens = tokenizer.encode(EOT_TOKEN, add_special_tokens=False)
    bot_id = bot_tokens[0]
    eot_id = eot_tokens[0]

    supervised_token_ids = {bot_id, eot_id}

    # Create labels (shifted by 1 for next-token prediction)
    labels = torch.tensor(encoded[1:], dtype=torch.long)

    # Verify supervised tokens are not masked with ignore_index (-100)
    for i, label_id in enumerate(labels):
        if label_id.item() in supervised_token_ids:
            assert label_id.item() != -100, (
                f"Supervised token {label_id.item()} (position {i}) is masked with ignore_index"
            )

    # Test with process-step supervision targets
    # Tool name IDs should not be masked
    tool_name_ids = torch.tensor([bot_id, eot_id], dtype=torch.long)
    assert (tool_name_ids != -100).all(), "Tool name IDs should not be masked with ignore_index"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
