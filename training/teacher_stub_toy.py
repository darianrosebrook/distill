"""
Deterministic teacher stub for toy distillation training.

Generates teacher logits without model weights - CPU-only, seeded for reproducibility.
Used by run_toy_distill.py to generate teacher logits during training.

Supports both regular toy training and 8-Ball mystical training.

Usage:
    from training.teacher_stub_toy import teacher_logits, 8_ball_teacher_logits
    logits = teacher_logits(token_ids, vocab_size=512)
    mystical_logits = 8_ball_teacher_logits(token_ids, vocab_size=512)
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


def magic_8_ball_teacher_logits(token_ids: torch.Tensor, vocab_size: int = 512, tokenizer=None) -> torch.Tensor:
    """
    Generate context-aware 8-Ball teacher logits using real mystical token sequences.

    Instead of arbitrary token preferences, this uses actual token IDs from real mystical
    phrases like "It is certain", "Outlook good", etc., making the teacher/student
    relationship much more aligned.

    Args:
        token_ids: Input token IDs [B, T]
        vocab_size: Vocabulary size
        tokenizer: Tokenizer for encoding mystical phrases (if None, fallback to old method)

    Returns:
        Teacher logits [B, T, V] preferring real mystical token sequences
    """
    torch.manual_seed(888)  # Mystical seed (8 Ball!)

    batch_size, seq_len = token_ids.shape
    device = token_ids.device

    # Create logits tensor
    logits = torch.zeros((batch_size, seq_len, vocab_size),
                         device=device, dtype=torch.float32)

    if tokenizer is not None:

        # NEW: Use real mystical phrases and their token sequences
        mystical_phrases = [
            "It is certain",
            "It is decidedly so",
            "Without a doubt",
            "Yes definitely",
            "You may rely on it",
            "As I see it, yes",
            "Most likely",
            "Outlook good",
            "Yes",
            "Signs point to yes",
            "Reply hazy, try again",
            "Ask again later",
            "Better not tell you now",
            "Cannot predict now",
            "Concentrate and ask again",
            "Don't count on it",
            "My reply is no",
            "My sources say no",
            "Outlook not so good",
            "Very doubtful"
        ]

        try:
            # Get token sequences for each mystical phrase
            mystical_token_sequences = []
            for phrase in mystical_phrases:
                try:
                    tokens = tokenizer.encode(phrase, add_special_tokens=False)
                    if tokens:  # Only add non-empty sequences
                        mystical_token_sequences.append(tokens)
                except Exception:
                    continue

            # Add baseline preferences for mystical tokens everywhere (small boost)
            for seq in mystical_token_sequences:
                for token_id in seq:
                    if token_id < vocab_size:
                        # Small baseline preference for mystical tokens
                        logits[..., token_id] += 0.1
        except Exception:
            mystical_token_sequences = []

        # Context-aware: only boost mystical tokens where answers should appear
        # Based on training data, mystical answers appear after ~10-15 tokens of prompt
        for b in range(batch_size):
            for t in range(seq_len):
                # Only boost mystical tokens in positions where answers typically appear
                if t >= 10:  # Assume prompts are ~10 tokens, answers come after
                    # Boost tokens that appear in mystical sequences at this relative position
                    relative_pos = t - 10  # Position within the answer
                    for seq in mystical_token_sequences:
                        if relative_pos < len(seq):
                            token_id = seq[relative_pos]
                            if token_id < vocab_size:
                                # Higher score for exact sequence matches in answer positions
                                logits[b, t, token_id] += 3.0

                    # Additional boost for mystical words in answer positions
                    if relative_pos < 5:  # First few tokens of answer
                        mystical_words = ["it", "outlook",
                                          "yes", "cannot", "very"]
                        for word in mystical_words:
                            try:
                                word_tokens = tokenizer.encode(
                                    word, add_special_tokens=False)
                                for token_id in word_tokens:
                                    if token_id < vocab_size:
                                        logits[b, t, token_id] += 1.0
                            except Exception:
                                continue
    else:
        # FALLBACK: Original arbitrary token method (if no tokenizer)
        mystical_tokens = [
            # "it", "is", "certain", etc.
            10, 15, 25, 30, 35, 40, 45, 50, 55, 60,
            65, 70, 75, 80, 85, 90, 95, 100, 105, 110,  # mystical words
            115, 120, 125, 130, 135, 140, 145, 150, 155, 160,  # more mystical tokens
            7, 8, 12, 13, 17, 18, 22, 23  # punctuation and special chars
        ]

        mystical_tokens = [t for t in mystical_tokens if t < vocab_size]

        for mystical_token in mystical_tokens:
            logits[..., mystical_token] = 4.0

    # Apply gentle normalization (keep positive values for preferred tokens)
    # Only subtract a small baseline, don't make everything non-positive
    max_logits = logits.max(dim=-1, keepdim=True)[0]
    # Dampen negative but keep positive
    logits = torch.where(logits > 0, logits, logits * 0.1)
    logits = logits * 1.5  # Mystical temperature scaling

    return logits
