# training/safe_model_loading.py
# Safe Hugging Face model loading utilities with revision pinning
# @author: @darianrosebrook

from typing import Optional, Dict, Any
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import warnings


def get_model_revision(model_name: str) -> Optional[str]:
    """
    Get pinned revision for a model name.
    
    For production models, this should return a specific commit SHA.
    For development, can return a tag or branch name.
    
    Args:
        model_name: Hugging Face model identifier
        
    Returns:
        Revision (commit SHA, tag, or branch) or None if not configured
    """
    # Model revision mapping - pin to specific commits for security
    # TODO: Move to config file for easier management
    revision_map: Dict[str, str] = {
        # Add model revisions here as they are identified
        # Example:
        # "microsoft/deberta-v3-small": "abc123def456...",
    }
    
    return revision_map.get(model_name)


def safe_from_pretrained_tokenizer(
    model_name: str,
    revision: Optional[str] = None,
    use_fast: bool = True,
    trust_remote_code: bool = False,
    **kwargs
) -> AutoTokenizer:
    """
    Safely load tokenizer with revision pinning.
    
    Args:
        model_name: Hugging Face model identifier
        revision: Explicit revision (commit SHA, tag, or branch). If None, uses get_model_revision()
        use_fast: Use fast tokenizer if available
        trust_remote_code: Whether to trust remote code (default: False for security)
        **kwargs: Additional arguments to pass to from_pretrained
        
    Returns:
        Loaded tokenizer
        
    Raises:
        ValueError: If model_name is invalid
    """
    if not model_name or not isinstance(model_name, str):
        raise ValueError(f"model_name must be a non-empty string, got {model_name}")
    
    # Get revision if not explicitly provided
    if revision is None:
        revision = get_model_revision(model_name)
    
    # If still no revision, use 'main' as minimum security measure
    # This at least pins to a branch rather than allowing arbitrary changes
    if revision is None:
        revision = "main"
        warnings.warn(
            f"Model '{model_name}' not in revision map. Using 'main' branch. "
            "For production, pin to specific commit SHA. See get_model_revision()",
            UserWarning
        )
    
    return AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        use_fast=use_fast,
        trust_remote_code=trust_remote_code,
        **kwargs
    )


def safe_from_pretrained_model(
    model_name: str,
    revision: Optional[str] = None,
    trust_remote_code: bool = False,
    **kwargs
) -> AutoModel:
    """
    Safely load model with revision pinning.
    
    Args:
        model_name: Hugging Face model identifier
        revision: Explicit revision (commit SHA, tag, or branch). If None, uses get_model_revision()
        trust_remote_code: Whether to trust remote code (default: False for security)
        **kwargs: Additional arguments to pass to from_pretrained
        
    Returns:
        Loaded model
        
    Raises:
        ValueError: If model_name is invalid
    """
    if not model_name or not isinstance(model_name, str):
        raise ValueError(f"model_name must be a non-empty string, got {model_name}")
    
    # Get revision if not explicitly provided
    if revision is None:
        revision = get_model_revision(model_name)
    
    # If still no revision, use 'main' as minimum security measure
    if revision is None:
        revision = "main"
        warnings.warn(
            f"Model '{model_name}' not in revision map. Using 'main' branch. "
            "For production, pin to specific commit SHA. See get_model_revision()",
            UserWarning
        )
    
    return AutoModel.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=trust_remote_code,
        **kwargs
    )


def safe_from_pretrained_causal_lm(
    model_name: str,
    revision: Optional[str] = None,
    trust_remote_code: bool = False,
    **kwargs
) -> AutoModelForCausalLM:
    """
    Safely load causal language model with revision pinning.
    
    Args:
        model_name: Hugging Face model identifier
        revision: Explicit revision (commit SHA, tag, or branch). If None, uses get_model_revision()
        trust_remote_code: Whether to trust remote code (default: False for security)
        **kwargs: Additional arguments to pass to from_pretrained
        
    Returns:
        Loaded model
        
    Raises:
        ValueError: If model_name is invalid
    """
    if not model_name or not isinstance(model_name, str):
        raise ValueError(f"model_name must be a non-empty string, got {model_name}")
    
    # Get revision if not explicitly provided
    if revision is None:
        revision = get_model_revision(model_name)
    
    # If still no revision, use 'main' as minimum security measure
    if revision is None:
        revision = "main"
        warnings.warn(
            f"Model '{model_name}' not in revision map. Using 'main' branch. "
            "For production, pin to specific commit SHA. See get_model_revision()",
            UserWarning
        )
    
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=trust_remote_code,
        **kwargs
    )

