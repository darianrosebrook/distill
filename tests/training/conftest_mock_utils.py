"""
Mock utilities for training tests.

Provides reusable mock objects for consistent test setup across dataset, 
tokenizer, and training tests.

Author: @darianrosebrook
"""

from unittest.mock import Mock, MagicMock
import torch


def create_mock_tokenizer(
    pad_token="[PAD]",
    eos_token="[EOS]",
    vocab_size=50257,
    encode_return_value=None,
):
    """
    Create a properly configured mock tokenizer with dict-like behavior.
    
    This tokenizer properly supports:
    - Dict-like access via __getitem__
    - len() operation
    - encode() and encode_plus() methods
    - Token properties (pad_token, eos_token)
    
    Args:
        pad_token: Padding token string
        eos_token: End-of-sequence token string
        vocab_size: Size of vocabulary for vocab_size property
        encode_return_value: Return value for encode() (default: [1, 2, 3, 4, 5])
    
    Returns:
        Mock tokenizer object with full dict-like interface
    """
    if encode_return_value is None:
        encode_return_value = [1, 2, 3, 4, 5]
    
    tokenizer = Mock()
    tokenizer.pad_token = pad_token
    tokenizer.eos_token = eos_token
    tokenizer.vocab_size = vocab_size
    
    # Make tokenizer callable and return dict-like object from encode_plus
    def encode_plus_impl(text, **kwargs):
        """Encode text and return dict-like object."""
        result = {
            "input_ids": encode_return_value.copy() if isinstance(encode_return_value, list) else [1, 2, 3],
            "attention_mask": [1] * len(encode_return_value) if isinstance(encode_return_value, list) else [1, 1, 1],
        }
        result_mock = Mock()
        result_mock.__getitem__ = lambda self, key: result[key]
        result_mock.keys = lambda: result.keys()
        result_mock.values = lambda: result.values()
        result_mock.items = lambda: result.items()
        result_mock.get = lambda key, default=None: result.get(key, default)
        # Also set as direct attributes
        for k, v in result.items():
            setattr(result_mock, k, v)
        return result_mock
    
    tokenizer.encode_plus = encode_plus_impl
    tokenizer.encode = Mock(return_value=encode_return_value.copy() if isinstance(encode_return_value, list) else [1, 2, 3])
    
    return tokenizer


def create_mock_tokenizer_with_len(
    pad_token="[PAD]",
    eos_token="[EOS]",
    vocab_size=50257,
    encode_return_value=None,
):
    """
    Create mock tokenizer with len() support.
    
    Extends create_mock_tokenizer with:
    - __len__() method support
    - Proper list-like behavior for tokenized outputs
    
    Args:
        pad_token: Padding token string
        eos_token: End-of-sequence token string
        vocab_size: Size of vocabulary
        encode_return_value: Return value for encode()
    
    Returns:
        Mock tokenizer with len() support
    """
    tokenizer = create_mock_tokenizer(
        pad_token=pad_token,
        eos_token=eos_token,
        vocab_size=vocab_size,
        encode_return_value=encode_return_value,
    )
    
    # Add __len__ support
    tokenizer.__len__ = Mock(return_value=vocab_size)
    
    return tokenizer


def create_mock_tokenizer_subscriptable(
    pad_token="[PAD]",
    eos_token="[EOS]",
    vocab_size=50257,
    encode_return_value=None,
):
    """
    Create mock tokenizer with full subscriptable/dict support.
    
    Extends create_mock_tokenizer with:
    - Full dict-like subscriptable behavior
    - len() support
    - Tensor output support
    - Proper list wrapping for tensor operations
    
    Args:
        pad_token: Padding token string
        eos_token: End-of-sequence token string
        vocab_size: Size of vocabulary
        encode_return_value: Return value for encode()
    
    Returns:
        Fully subscriptable mock tokenizer
    """
    if encode_return_value is None:
        encode_return_value = [1, 2, 3, 4, 5, 6]  # 6 tokens so that after [:-1] we get 5
    
    tokenizer = create_mock_tokenizer_with_len(
        pad_token=pad_token,
        eos_token=eos_token,
        vocab_size=vocab_size,
        encode_return_value=encode_return_value,
    )
    
    # Make encode_plus return a subscriptable dict-like object
    def encode_plus_impl(text, **kwargs):
        """Encode text with full dict support."""
        tokens = encode_return_value.copy() if isinstance(encode_return_value, list) else [1, 2, 3]
        result = {
            "input_ids": torch.tensor(tokens, dtype=torch.int64),  # 1D tensor
            "attention_mask": torch.tensor([1] * len(tokens), dtype=torch.int64),  # 1D tensor
        }
        
        # Create a dict-like mock that supports subscripting
        result_dict = MagicMock()
        result_dict.__getitem__ = lambda self, key: result[key]
        result_dict.__setitem__ = lambda self, key, val: result.update({key: val})
        result_dict.keys = lambda self=None: result.keys()
        result_dict.values = lambda self=None: result.values()
        result_dict.items = lambda self=None: result.items()
        result_dict.get = lambda self=None, key=None, default=None: result.get(key, default) if key is not None else None
        
        # Set direct attributes too
        for k, v in result.items():
            setattr(result_dict, k, v)
        
        return result_dict
    
    tokenizer.encode_plus = encode_plus_impl
    
    # Make encode return a 1D tensor
    def encode_impl(text, **kwargs):
        """Encode text to 1D tensor."""
        tokens = encode_return_value.copy() if isinstance(encode_return_value, list) else [1, 2, 3]
        result = torch.tensor(tokens, dtype=torch.int64)
        return result
    
    tokenizer.encode = encode_impl
    
    return tokenizer


def create_mock_encoded_output(
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
):
    """
    Create properly formatted mock encoded output.
    
    Args:
        input_ids: List or tensor of input IDs (default: [1, 2, 3])
        attention_mask: List or tensor of attention mask (default: all 1s)
        token_type_ids: List or tensor of token type IDs (optional)
    
    Returns:
        Dict-like mock object with subscriptable behavior
    """
    if input_ids is None:
        input_ids = [1, 2, 3]
    if attention_mask is None:
        attention_mask = [1] * len(input_ids)
    
    # Convert to tensors if needed
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor([input_ids], dtype=torch.int64) if isinstance(input_ids[0], int) else torch.tensor(input_ids, dtype=torch.int64)
    
    if not isinstance(attention_mask, torch.Tensor):
        attention_mask = torch.tensor([attention_mask], dtype=torch.int64) if isinstance(attention_mask[0], int) else torch.tensor(attention_mask, dtype=torch.int64)
    
    result_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    
    if token_type_ids is not None:
        if not isinstance(token_type_ids, torch.Tensor):
            token_type_ids = torch.tensor([token_type_ids], dtype=torch.int64) if isinstance(token_type_ids[0], int) else torch.tensor(token_type_ids, dtype=torch.int64)
        result_dict["token_type_ids"] = token_type_ids
    
    # Create subscriptable mock
    result_mock = MagicMock()
    result_mock.__getitem__ = lambda key: result_dict[key]
    result_mock.keys = lambda: result_dict.keys()
    result_mock.values = lambda: result_dict.values()
    result_mock.items = lambda: result_dict.items()
    result_mock.get = lambda key, default=None: result_dict.get(key, default)
    
    for k, v in result_dict.items():
        setattr(result_mock, k, v)
    
    return result_mock


def create_mock_path(path_string="/tmp/test"):
    """
    Create properly configured mock Path object.
    
    Args:
        path_string: String representation of path
    
    Returns:
        Mock Path that converts to string properly
    """
    mock_path = Mock()
    mock_path.__str__ = Mock(return_value=path_string)
    mock_path.__fspath__ = Mock(return_value=path_string)
    mock_path.as_posix = Mock(return_value=path_string)
    mock_path.parent = Mock()
    mock_path.parent.mkdir = Mock()
    mock_path.mkdir = Mock()
    mock_path.suffix = ".pt"
    mock_path.stem = "test"
    return mock_path


__all__ = [
    "create_mock_tokenizer",
    "create_mock_tokenizer_with_len",
    "create_mock_tokenizer_subscriptable",
    "create_mock_encoded_output",
    "create_mock_path",
]

