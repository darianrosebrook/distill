"""
Deterministic teacher stub for toy distillation training.

Generates teacher logits without model weights - CPU-only, seeded for reproducibility.
Used by run_toy_distill.py to generate teacher logits during training.

Supports both regular toy training and Magic 8 Ball mystical training.

Usage:
    from training.teacher_stub_toy import teacher_logits, magic_8_ball_teacher_logits
    logits = teacher_logits(token_ids, vocab_size=512)
    mystical_logits = magic_8_ball_teacher_logits(token_ids, vocab_size=512)
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


def magic_8_ball_teacher_logits(token_ids: torch.Tensor, vocab_size: int = 512) -> torch.Tensor:
    """
    Generate deterministic Magic 8 Ball teacher logits for mystical training.

    Creates a distribution peaked at tokens that form Magic 8 Ball responses like:
    "It is certain", "Outlook good", "Reply hazy, try again", etc.

    This simulates a teacher model that prefers mystical, fortune-telling patterns.

    Args:
        token_ids: Input token IDs [B, T]
        vocab_size: Vocabulary size

    Returns:
        Teacher logits [B, T, V] preferring mystical token patterns
    """
    torch.manual_seed(888)  # Mystical seed (8 Ball!)

    batch_size, seq_len = token_ids.shape
    device = token_ids.device

    # Create logits tensor
    logits = torch.zeros((batch_size, seq_len, vocab_size),
                         device=device, dtype=torch.float32)

    # Magic 8 Ball mystical token preferences
    # These represent tokens for words like: it, is, certain, outlook, good, reply, hazy, etc.
    mystical_tokens = [
        10, 15, 25, 30, 35, 40, 45, 50, 55, 60,  # "it", "is", "certain", etc.
        65, 70, 75, 80, 85, 90, 95, 100, 105, 110,  # mystical words
        115, 120, 125, 130, 135, 140, 145, 150, 155, 160,  # more mystical tokens
        7, 8, 12, 13, 17, 18, 22, 23  # punctuation and special chars
    ]

    # Filter to valid vocab range
    mystical_tokens = [t for t in mystical_tokens if t < vocab_size]

    # Set high logits for mystical tokens (even higher than regular teacher)
    for mystical_token in mystical_tokens:
        # Higher preference than regular teacher
        logits[..., mystical_token] = 4.0

    # Add mystical patterns based on position in sequence
    # Early tokens prefer "It is", middle prefer answers, later prefer flair
    for b in range(batch_size):
        for t in range(seq_len):
            if t < 3:  # Early in sequence - prefer "It is" patterns
                early_tokens = [10, 15, 25]  # it, is, certain-ish
                for token in early_tokens:
                    if token < vocab_size:
                        logits[b, t, token] += 2.0
            elif t < seq_len // 2:  # Middle - prefer main answers
                # outlook, good, reply, etc.
                mid_tokens = [35, 40, 45, 50, 55, 60]
                for token in mid_tokens:
                    if token < vocab_size:
                        logits[b, t, token] += 1.5
            else:  # Later - prefer flair and punctuation
                late_tokens = [7, 8, 12, 13, 17, 18]  # ! ? : ; etc.
                for token in late_tokens:
                    if token < vocab_size:
                        logits[b, t, token] += 1.0

    # Add some variation based on input tokens (mystical influence)
    for b in range(batch_size):
        for t in range(seq_len):
            input_token = token_ids[b, t].item()
            # Mystical patterns: primes and factors have special meaning
            if input_token > 1 and all(input_token % i != 0 for i in range(2, int(input_token**0.5) + 1)):
                # Prime numbers get mystical bonus
                logits[b, t, mystical_tokens[:5]] += 1.0
            elif input_token % 8 == 0:
                # Multiples of 8 (Magic 8 Ball!) get extra mystical energy
                logits[b, t, mystical_tokens[5:10]] += 1.5

    # Apply mystical normalization (temperature-like scaling for fortune telling)
    logits = logits - logits.max(dim=-1, keepdim=True)[0]
    logits = logits * 1.5  # Mystical temperature scaling

    return logits
