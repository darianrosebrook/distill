"""
Pytest configuration and shared fixtures.
"""
import json
import tempfile
from pathlib import Path
from typing import Generator

import pytest
import torch
import torch.nn as nn

from models.student.architectures.gqa_transformer import StudentLM, ModelCfg


@pytest.fixture
def device():
    """Get device for testing."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


@pytest.fixture
def small_model_cfg():
    """Small model config for faster tests."""
    return ModelCfg(
        d_model=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_head=32,
        vocab_size=1000,
        rope_theta=10000.0,
        rope_scaling='none',
        dropout=0.0,
    )


@pytest.fixture
def small_model(small_model_cfg, device):
    """Create a small model for testing."""
    model = StudentLM(small_model_cfg)
    model = model.to(device)
    model.eval()
    return model


@pytest.fixture
def temp_jsonl_file() -> Generator[Path, None, None]:
    """Create a temporary JSONL file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        # Write some test data
        test_data = [
            {
                "prompt": "Hello world",
                "teacher_text": "Hello world, this is a test",
                "teacher_logits": None,
            },
            {
                "prompt": "Test prompt 2",
                "teacher_text": "Test response 2",
                "teacher_logits": None,
            },
        ]
        for item in test_data:
            f.write(json.dumps(item) + '\n')
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def sample_batch():
    """Create a sample batch for testing."""
    batch_size = 2
    seq_len = 10
    vocab_size = 1000
    
    return {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len),
        "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "teacher_logits": torch.randn(batch_size, seq_len, vocab_size),
    }


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    class MockTokenizer:
        def __init__(self):
            self.vocab_size = 1000
            self.pad_token_id = 0
            self.eos_token_id = 1
        
        def __call__(self, text, return_tensors=None, padding=None, truncation=None, max_length=None, **kwargs):
            # Simple mock: tokenize by splitting on spaces
            tokens = text.split()
            token_ids = [hash(t) % self.vocab_size for t in tokens]
            
            if max_length:
                token_ids = token_ids[:max_length]
            
            if padding == "max_length" and max_length:
                while len(token_ids) < max_length:
                    token_ids.append(self.pad_token_id)
            
            result = {"input_ids": torch.tensor([token_ids])}
            
            if return_tensors == "pt":
                return result
            return result
        
        def encode(self, text, add_special_tokens=False, **kwargs):
            tokens = text.split()
            return [hash(t) % self.vocab_size for t in tokens]
        
        def decode(self, token_ids, skip_special_tokens=False, **kwargs):
            # Simple mock decode
            return " ".join([f"token_{i}" for i in token_ids])
    
    return MockTokenizer()

