"""
Deterministic teacher stub for toy distillation training.

Generates teacher logits without model weights - CPU-only, seeded for reproducibility.
Used by run_toy_distill.py to generate teacher logits during training.

Usage:
    from training.teacher_stub_toy import teacher_logits
    logits = teacher_logits(token_ids, vocab_size=512)
"""
import torch


def teacher_logits(token_ids: torch.Tensor, vocab_size: int = 512) -> torch.Tensor:
    """
    Generate deterministic teacher logits for toy training.

    Creates a "good" distribution peaked at common tokens like "ok", "tool", "call", braces.
    This simulates a teacher model that prefers tool-calling patterns.

    Args:
        token_ids: Input token IDs [B, T]
        vocab_size: Vocabulary size

    Returns:
        Teacher logits [B, T, V]
    """
    torch.manual_seed(123)  # Deterministic

    batch_size, seq_len = token_ids.shape
    device = token_ids.device

    # Create logits tensor
    logits = torch.zeros((batch_size, seq_len, vocab_size),
                         device=device, dtype=torch.float32)

    # Deterministic "good" distribution: peaked at pretend token IDs
    # These represent tokens like "ok", "tool", "call", "{", "}"
    hot_tokens = [5, 17, 33, 7, 8]  # Arbitrary but fixed token IDs

    # Set high logits for "good" tokens
    for hot_token in hot_tokens:
        if hot_token < vocab_size:
            logits[..., hot_token] = 3.0

    # Add some variation based on input tokens (but keep it deterministic)
    # This makes the teacher respond to input content
    for b in range(batch_size):
        for t in range(seq_len):
            input_token = token_ids[b, t].item()
            # Create a simple pattern: if input token is even, prefer even outputs
            if input_token % 2 == 0:
                logits[b, t, ::2] += 0.5
            else:
                logits[b, t, 1::2] += 0.5

    # Apply softmax-like normalization (but keep logits, not probabilities)
    # This creates a reasonable distribution without being too peaked
    logits = logits - logits.max(dim=-1, keepdim=True)[0]
    logits = logits * 2.0  # Temperature-like scaling

    return logits
