"""
Deterministic teacher stub for toy distillation training.

Generates teacher logits without model weights - CPU-only, seeded for reproducibility.
Used by run_toy_distill.py to generate teacher logits during training.

Supports both regular toy training and 8-Ball mystical training.

Usage:
    from training.teacher_stub_toy import teacher_logits, eight_ball_teacher_logits
    logits = teacher_logits(token_ids, vocab_size=512)
    mystical_logits = eight_ball_teacher_logits(token_ids, vocab_size=512)
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
    logits = torch.zeros((batch_size, seq_len, vocab_size), device=device, dtype=torch.float32)

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


def eight_ball_teacher_logits(
    token_ids: torch.Tensor, vocab_size: int = 512, tokenizer=None
) -> torch.Tensor:
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
    logits = torch.zeros((batch_size, seq_len, vocab_size), device=device, dtype=torch.float32)

    # 8-ball answer token IDs (200-219) - use these for classification approach
    eight_ball_token_ids = list(range(200, 220)) if vocab_size >= 220 else []

    # Binary classifier token IDs (300-301) - YES/NO decisions
    binary_token_ids = list(range(300, 302)) if vocab_size >= 302 else []

    # Ternary classifier token IDs (400-402) - YES/NO/UNCERTAIN decisions
    ternary_token_ids = list(range(400, 403)) if vocab_size >= 403 else []

    # CLASSIFICATION APPROACH: Check for ternary classifier first (highest priority)
    if ternary_token_ids and vocab_size >= 403:
        # Ternary classifier: Boost YES/NO/UNCERTAIN tokens at answer position
        for token_id in ternary_token_ids:
            logits[..., token_id] += 2.0  # Baseline boost

        # Very strong boost at the final position (where answer should appear)
        for b in range(batch_size):
            question_hash = hash(tuple(token_ids[b].cpu().numpy())) % 3
            preferred_answer_id = ternary_token_ids[question_hash]

            # Strong boost for the preferred answer at final position
            logits[b, -1, preferred_answer_id] += 15.0

            # Moderate boost for other ternary answers
            for token_id in ternary_token_ids:
                if token_id != preferred_answer_id:
                    logits[b, -1, token_id] += 5.0

        # Normalize to prevent overflow
        logits = logits / (logits.abs().max() + 1e-8) * 10.0
        return logits

    # CLASSIFICATION APPROACH: Check for binary classifier second
    elif binary_token_ids and vocab_size >= 302:
        # Binary classifier: Boost YES/NO tokens at answer position
        for token_id in binary_token_ids:
            logits[..., token_id] += 2.0  # Baseline boost

        # Very strong boost at the final position (where answer should appear)
        for b in range(batch_size):
            question_hash = hash(tuple(token_ids[b].cpu().numpy())) % 2
            preferred_answer_id = binary_token_ids[question_hash]

            # Strong boost for the preferred answer at final position
            logits[b, -1, preferred_answer_id] += 15.0

            # Moderate boost for other binary answer
            for token_id in binary_token_ids:
                if token_id != preferred_answer_id:
                    logits[b, -1, token_id] += 5.0

        # Normalize to prevent overflow
        logits = logits / (logits.abs().max() + 1e-8) * 10.0
        return logits

    # CLASSIFICATION APPROACH: If vocab_size supports it, use 8-ball token IDs
    elif eight_ball_token_ids and vocab_size >= 220:
        # Boost all 8-ball answer tokens everywhere (baseline preference)
        for token_id in eight_ball_token_ids:
            logits[..., token_id] += 2.0  # Baseline boost

        # Very strong boost at the final position (where answer should appear)
        # Use question hash to deterministically pick which answer
        for b in range(batch_size):
            question_hash = hash(tuple(token_ids[b].cpu().numpy())) % 20
            preferred_answer_id = eight_ball_token_ids[question_hash]

            # Strong boost for the preferred answer at final position
            logits[b, -1, preferred_answer_id] += 15.0

            # Moderate boost for other 8-ball answers
            for token_id in eight_ball_token_ids:
                if token_id != preferred_answer_id:
                    logits[b, -1, token_id] += 5.0

        # Normalize to prevent overflow
        logits = logits / (logits.abs().max() + 1e-8) * 10.0
        return logits

    # FALLBACK: Original approach using real mystical phrases
    if tokenizer is not None:
        # Use real mystical phrases and boost their actual token IDs
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
            "Very doubtful",
        ]

        # Collect all unique tokens that appear in mystical phrases
        mystical_tokens = set()
        for phrase in mystical_phrases:
            try:
                tokens = tokenizer.encode(phrase, add_special_tokens=False)
                mystical_tokens.update(tokens)
            except Exception:
                continue

        mystical_tokens = [t for t in mystical_tokens if t < vocab_size]

        # Boost mystical tokens everywhere (strong baseline preference)
        for token_id in mystical_tokens:
            logits[..., token_id] += 3.0  # Strong baseline boost

        # Very strong boost in answer positions (where mystical responses should appear)
        # Based on training data analysis, answers appear after ~5-10 tokens
        for b in range(batch_size):
            for t in range(seq_len):
                if t >= 5:  # Answers appear after prompt
                    relative_pos = t - 5

                    # Boost mystical tokens very strongly in answer positions
                    for token_id in mystical_tokens:
                        logits[b, t, token_id] += 8.0

                    # Extra boost for common starting tokens in mystical answers
                    if relative_pos == 0:  # First token of answer
                        # Common mystical starters: "It", "Yes", "Without", "Outlook", etc.
                        starter_tokens = []
                        for phrase in ["It", "Yes", "Without", "Outlook", "My"]:
                            try:
                                tokens = tokenizer.encode(phrase, add_special_tokens=False)
                                starter_tokens.extend(tokens[:1])  # First token only
                            except Exception:
                                continue
                        for token_id in starter_tokens:
                            if token_id < vocab_size:
                                logits[b, t, token_id] += 15.0  # Very strong boost for starters

    else:
        # FALLBACK: Use actual token IDs from mystical phrases (hardcoded for reliability)
        # These are the actual token IDs from the tokenizer for mystical words
        mystical_tokens = [
            # From "It is certain": 739(It), 338(is), 3058(certain)
            739,
            338,
            3058,
            # From "Yes": 3869
            3869,
            # From "Outlook good": 4451(Out), 6914(look), 1781(good), 451(not), 577(so)
            4451,
            6914,
            1781,
            451,
            577,
            # From "My reply is no": 1619(My), 8908(reply), 694(no)
            1619,
            8908,
            694,
            # From "Very doubtful": 18064(Very), 7404(doubt), 1319(ful)
            18064,
            7404,
            1319,
            # Common mystical punctuation and special tokens
            373,
            372,
            368,
            447,
            322,  # "on", "it", "ly", "ha", "and"
        ]

        mystical_tokens = [t for t in mystical_tokens if t < vocab_size]

        # Boost mystical tokens everywhere
        for token_id in mystical_tokens:
            logits[..., token_id] += 1.0

    # Apply normalization that preserves strong preferences
    # Keep positive values, dampen negative ones
    logits = torch.where(logits > 0, logits, logits * 0.1)
    # Scale up significantly to provide very strong teacher signal
    logits = logits * 5.0

    return logits
