"""
Tokenizer/vocab migration for special tokens.

Handles:
- Embedding resizing when new tokens are added
- Special token initialization
- ID stability verification
- Loss exclusion for special tokens
"""
# @author: @darianrosebrook

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn

from models.student.tokenizer.constants import BOT_TOKEN_ID, EOT_TOKEN_ID, BOT_TOKEN, EOT_TOKEN


def verify_token_ids(tokenizer) -> Dict[str, Any]:
    """
    Verify token ID stability across tokenizer configs.
    
    Args:
        tokenizer: Tokenizer instance
    
    Returns:
        Dict with verification results:
        - bot_token_id: int
        - eot_token_id: int
        - ids_match: bool
        - errors: List[str]
    """
    errors = []
    
    # Get token IDs from tokenizer
    bot_id_tokenizer = None
    eot_id_tokenizer = None
    
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        bot_id_tokenizer = tokenizer.convert_tokens_to_ids(BOT_TOKEN)
        eot_id_tokenizer = tokenizer.convert_tokens_to_ids(EOT_TOKEN)
    
    # Compare with constants
    ids_match = (
        bot_id_tokenizer == BOT_TOKEN_ID and
        eot_id_tokenizer == EOT_TOKEN_ID
    )
    
    if bot_id_tokenizer != BOT_TOKEN_ID:
        errors.append(
            f"BOT token ID mismatch: tokenizer={bot_id_tokenizer}, constant={BOT_TOKEN_ID}"
        )
    
    if eot_id_tokenizer != EOT_TOKEN_ID:
        errors.append(
            f"EOT token ID mismatch: tokenizer={eot_id_tokenizer}, constant={EOT_TOKEN_ID}"
        )
    
    return {
        "bot_token_id": bot_id_tokenizer,
        "eot_token_id": eot_id_tokenizer,
        "ids_match": ids_match,
        "errors": errors,
    }


def resize_model_embeddings(
    model: nn.Module,
    tokenizer,
    new_vocab_size: Optional[int] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Resize model embeddings to match tokenizer vocabulary.
    
    Handles:
    - Resizing embedding layer
    - Resizing LM head (if tied, handles both)
    - Initializing new token embeddings
    - Preserving existing embeddings
    
    Args:
        model: StudentLM model
        tokenizer: Tokenizer instance
        new_vocab_size: Optional new vocab size (if None, uses tokenizer.vocab_size)
    
    Returns:
        Tuple of (resized_model, metadata_dict)
    """
    metadata = {
        "original_vocab_size": None,
        "new_vocab_size": None,
        "tokens_added": [],
        "embedding_resized": False,
        "lm_head_resized": False,
    }
    
    # Get vocab sizes
    if hasattr(tokenizer, "vocab_size"):
        tokenizer_vocab_size = tokenizer.vocab_size
    elif hasattr(tokenizer, "__len__"):
        tokenizer_vocab_size = len(tokenizer)
    else:
        raise ValueError("Cannot determine tokenizer vocabulary size")
    
    if new_vocab_size is None:
        new_vocab_size = tokenizer_vocab_size
    
    # Get current model vocab size
    if hasattr(model, "embed"):
        original_vocab_size = model.embed.num_embeddings
    elif hasattr(model, "module") and hasattr(model.module, "embed"):
        original_vocab_size = model.module.embed.num_embeddings
    else:
        raise ValueError("Cannot determine model vocabulary size")
    
    metadata["original_vocab_size"] = original_vocab_size
    metadata["new_vocab_size"] = new_vocab_size
    
    # Check if resizing is needed
    if original_vocab_size >= new_vocab_size:
        # No resizing needed (tokenizer vocab <= model vocab)
        return model, metadata
    
    # Resize embedding layer
    if hasattr(model, "embed"):
        old_embed = model.embed
        new_embed = nn.Embedding(new_vocab_size, old_embed.embedding_dim)
        # Copy existing embeddings
        new_embed.weight.data[:original_vocab_size] = old_embed.weight.data
        # Initialize new token embeddings with small random values
        nn.init.normal_(
            new_embed.weight.data[original_vocab_size:],
            mean=0.0,
            std=0.02,
        )
        model.embed = new_embed
        metadata["embedding_resized"] = True
        metadata["tokens_added"] = list(range(original_vocab_size, new_vocab_size))
    
    # Resize LM head
    if hasattr(model, "lm_head"):
        old_lm_head = model.lm_head
        new_lm_head = nn.Linear(old_lm_head.in_features, new_vocab_size, bias=False)
        # Copy existing weights
        new_lm_head.weight.data[:original_vocab_size] = old_lm_head.weight.data
        # Initialize new token weights with small random values
        nn.init.normal_(
            new_lm_head.weight.data[original_vocab_size:],
            mean=0.0,
            std=0.02,
        )
        model.lm_head = new_lm_head
        metadata["lm_head_resized"] = True
    
    # Update model config if it exists
    if hasattr(model, "cfg"):
        model.cfg.vocab_size = new_vocab_size
    
    return model, metadata


def initialize_special_token_embeddings(
    model: nn.Module,
    tokenizer,
    special_token_ids: Optional[list] = None,
    init_method: str = "small_norm",
) -> Dict[str, Any]:
    """
    Initialize special token embeddings with sensible defaults.
    
    Args:
        model: StudentLM model
        tokenizer: Tokenizer instance
        special_token_ids: List of special token IDs (if None, uses BOT_TOKEN_ID, EOT_TOKEN_ID)
        init_method: Initialization method ("small_norm", "zeros", "mean")
    
    Returns:
        Dict with initialization metadata
    """
    if special_token_ids is None:
        special_token_ids = [BOT_TOKEN_ID, EOT_TOKEN_ID]
    
    metadata = {
        "special_tokens_initialized": [],
        "init_method": init_method,
    }
    
    if not hasattr(model, "embed"):
        return metadata
    
    embed = model.embed
    d_model = embed.embedding_dim
    
    for token_id in special_token_ids:
        if token_id >= embed.num_embeddings:
            continue
        
        if init_method == "small_norm":
            # Small random initialization with controlled norm
            init_vec = torch.randn(d_model) * 0.01
            embed.weight.data[token_id] = init_vec
        elif init_method == "zeros":
            embed.weight.data[token_id].zero_()
        elif init_method == "mean":
            # Initialize to mean of existing embeddings
            mean_embed = embed.weight.data[:token_id].mean(dim=0)
            embed.weight.data[token_id] = mean_embed
        
        metadata["special_tokens_initialized"].append(token_id)
    
    return metadata


def create_special_token_loss_mask(
    labels: torch.Tensor,
    tokenizer,
    special_token_ids: Optional[list] = None,
) -> torch.Tensor:
    """
    Create loss mask that excludes special tokens from cross-entropy loss.
    
    Special tokens (<bot>, <eot>) should not be predicted as next tokens
    in normal language modeling, so we mask them from the loss.
    
    Args:
        labels: [B, T] label tensor
        tokenizer: Tokenizer instance
        special_token_ids: List of special token IDs to mask (if None, uses BOT_TOKEN_ID, EOT_TOKEN_ID)
    
    Returns:
        loss_mask: [B, T] boolean tensor (True = supervise, False = mask)
    """
    if special_token_ids is None:
        special_token_ids = [BOT_TOKEN_ID, EOT_TOKEN_ID]
    
    # Create mask: True for tokens to supervise, False for special tokens
    loss_mask = torch.ones_like(labels, dtype=torch.bool)
    
    for token_id in special_token_ids:
        loss_mask[labels == token_id] = False
    
    return loss_mask


def migrate_tokenizer_and_model(
    model: nn.Module,
    tokenizer,
    tokenizer_path: str,
    verify_ids: bool = True,
    resize_embeddings: bool = True,
    init_special_tokens: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Complete tokenizer/model migration for new special tokens.
    
    Args:
        model: StudentLM model
        tokenizer: Tokenizer instance
        tokenizer_path: Path to tokenizer config (for verification)
        verify_ids: Whether to verify token ID stability
        resize_embeddings: Whether to resize embeddings
        init_special_tokens: Whether to initialize special token embeddings
    
    Returns:
        Tuple of (migrated_model, migration_metadata)
    """
    migration_metadata = {
        "tokenizer_path": tokenizer_path,
        "verification": {},
        "resize": {},
        "initialization": {},
    }
    
    # Step 1: Verify token IDs
    if verify_ids:
        verification = verify_token_ids(tokenizer)
        migration_metadata["verification"] = verification
        
        if not verification["ids_match"]:
            raise ValueError(
                f"Token ID mismatch detected: {verification['errors']}"
            )
    
    # Step 2: Resize embeddings if needed
    if resize_embeddings:
        model, resize_metadata = resize_model_embeddings(model, tokenizer)
        migration_metadata["resize"] = resize_metadata
    
    # Step 3: Initialize special token embeddings
    if init_special_tokens:
        init_metadata = initialize_special_token_embeddings(
            model, tokenizer, init_method="small_norm"
        )
        migration_metadata["initialization"] = init_metadata
    
    return model, migration_metadata

