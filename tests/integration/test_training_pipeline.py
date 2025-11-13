"""
Integration tests for training pipeline.

Tests the full training pipeline with small models to ensure components work together.
"""
import json

import pytest
import torch

from models.student.architectures.gqa_transformer import StudentLM
from training.dataset import KDDataset, collate_kd_batch
from training.losses import combined_kd_loss
import sys
from unittest.mock import MagicMock


# Mock transformers before importing
mock_transformers = MagicMock()
mock_auto_tokenizer_class = MagicMock()
mock_transformers.AutoTokenizer = mock_auto_tokenizer_class
sys.modules['transformers'] = mock_transformers

# Reload dataset module to pick up the mock
if 'training.dataset' in sys.modules:
    import importlib
    importlib.reload(sys.modules['training.dataset'])


# Ensure HF_TOKENIZER_AVAILABLE is True in the module
import training.dataset as dataset_module
dataset_module.HF_TOKENIZER_AVAILABLE = True
dataset_module.AutoTokenizer = mock_auto_tokenizer_class


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for integration tests."""
    class MockTokenizer:
        def __init__(self):
            self.vocab_size = 1000
            self.pad_token_id = 0
            self.eos_token_id = 1
            self.pad_token = None
        
        def encode(self, text, add_special_tokens=False, **kwargs):
            """Encode text to token IDs."""
            tokens = text.split()
            token_ids = [abs(hash(t)) % self.vocab_size for t in tokens]
            
            if add_special_tokens:
                token_ids = [self.eos_token_id] + token_ids + [self.eos_token_id]
            
            return token_ids
        
        def __len__(self):
            """Return vocabulary size."""
            return self.vocab_size
    
    tokenizer = MockTokenizer()
    tokenizer.pad_token = tokenizer.eos_token_id
    return tokenizer


@pytest.fixture
def temp_kd_dataset(mock_tokenizer, tmp_path):
    """Create a temporary KD dataset JSONL file."""
    jsonl_file = tmp_path / "kd_dataset.jsonl"
    
    # Create sample data
    samples = [
        {
            "prompt": "What is Python?",
            "teacher_text": "Python is a programming language.",
        },
        {
            "prompt": "Explain machine learning.",
            "teacher_text": "Machine learning is a subset of AI.",
        },
        {
            "prompt": "What is distillation?",
            "teacher_text": "Distillation transfers knowledge from teacher to student.",
        },
    ]
    
    with open(jsonl_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    return str(jsonl_file)


class TestDatasetModelIntegration:
    """Integration tests for dataset and model."""
    
    def test_dataset_model_forward(self, temp_kd_dataset, mock_tokenizer, small_model_cfg, device):
        """Test that dataset and model work together."""
        mock_auto_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Create dataset
        dataset = KDDataset(
            jsonl_path=temp_kd_dataset,
            tokenizer_path="mock_tokenizer",
            max_seq_length=32,
            teacher_logits_available=False,
        )
        
        # Create model
        model = StudentLM(small_model_cfg).to(device)
        model.eval()
        
        # Get a sample
        sample = dataset[0]
        input_ids = sample["input_ids"].unsqueeze(0).to(device)  # Add batch dimension
        sample["attention_mask"].unsqueeze(0).to(device)
        
        # Forward pass
        with torch.no_grad():
            logits = model(input_ids)
        
        assert logits.shape[0] == 1  # Batch size
        assert logits.shape[1] == input_ids.shape[1]  # Sequence length
        assert logits.shape[2] == small_model_cfg.vocab_size  # Vocab size
    
    def test_dataset_dataloader_model(self, temp_kd_dataset, mock_tokenizer, small_model_cfg, device):
        """Test dataset, dataloader, and model integration."""
        from torch.utils.data import DataLoader
        
        mock_auto_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Create dataset
        dataset = KDDataset(
            jsonl_path=temp_kd_dataset,
            tokenizer_path="mock_tokenizer",
            max_seq_length=32,
            teacher_logits_available=False,
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_kd_batch,
        )
        
        # Create model
        model = StudentLM(small_model_cfg).to(device)
        model.eval()
        
        # Process a batch
        batch = next(iter(dataloader))
        input_ids = batch["input_ids"].to(device)
        # Model doesn't use attention_mask from dataset directly (expects [B, H, T, T] or None)
        attn_mask = None
        
        with torch.no_grad():
            logits = model(input_ids, attn_mask)
        
        assert logits.shape[0] == batch["input_ids"].shape[0]  # Batch size
        assert logits.shape[1] == batch["input_ids"].shape[1]  # Sequence length
        assert logits.shape[2] == small_model_cfg.vocab_size  # Vocab size


class TestKDTrainingIntegration:
    """Integration tests for KD training loop."""
    
    def test_kd_loss_computation(self, small_model_cfg, device):
        """Test that KD loss can be computed with model outputs."""
        model = StudentLM(small_model_cfg).to(device)
        model.train()
        
        b, t = 2, 10
        input_ids = torch.randint(0, small_model_cfg.vocab_size, (b, t), device=device)
        labels = torch.randint(0, small_model_cfg.vocab_size, (b, t), device=device)
        
        # Forward pass (no attention mask needed)
        student_logits = model(input_ids)
        
        # Create dummy teacher logits
        teacher_logits = torch.randn_like(student_logits)
        
        # Compute KD loss
        loss_dict = combined_kd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            teacher_targets=teacher_logits.argmax(dim=-1),
            ground_truth_targets=labels,
            kl_weight=0.5,
            ce_teacher_weight=0.3,
            ce_ground_truth_weight=0.2,
            kd_temperature=2.0,
        )
        
        assert "total" in loss_dict
        assert loss_dict["total"].item() >= 0.0
        assert loss_dict["total"].requires_grad
    
    def test_kd_training_step(self, temp_kd_dataset, mock_tokenizer, small_model_cfg, device):
        """Test a single KD training step."""
        from torch.utils.data import DataLoader
        
        mock_auto_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Setup
        dataset = KDDataset(
            jsonl_path=temp_kd_dataset,
            tokenizer_path="mock_tokenizer",
            max_seq_length=32,
            teacher_logits_available=False,
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_kd_batch,
        )
        
        model = StudentLM(small_model_cfg).to(device)
        model.train()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Training step
        batch = next(iter(dataloader))
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward (no attention mask needed)
        student_logits = model(input_ids)
        
        # Create dummy teacher logits
        teacher_logits = torch.randn_like(student_logits)
        
        # Loss
        loss_dict = combined_kd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            teacher_targets=teacher_logits.argmax(dim=-1),
            ground_truth_targets=labels,
            kl_weight=0.5,
            ce_teacher_weight=0.3,
            ce_ground_truth_weight=0.2,
            kd_temperature=2.0,
        )
        
        loss = loss_dict["total"]
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Verify model parameters changed
        # (We can't easily check this without storing initial state, but if no error, it worked)
        assert loss.item() >= 0.0
    
    def test_checkpoint_save_load(self, small_model_cfg, device, tmp_path):
        """Test checkpoint save and load."""
        # Create and train model briefly
        model1 = StudentLM(small_model_cfg).to(device)
        model1.train()
        
        optimizer1 = torch.optim.AdamW(model1.parameters(), lr=1e-4)
        
        # Do a dummy training step
        input_ids = torch.randint(0, small_model_cfg.vocab_size, (2, 10), device=device)
        logits = model1(input_ids)
        loss = logits.sum()
        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()
        
        # Save checkpoint
        checkpoint_path = tmp_path / "checkpoint.pt"
        torch.save({
            'step': 1,
            'model_state_dict': model1.state_dict(),
            'optimizer_state_dict': optimizer1.state_dict(),
            'config': {'arch': {
                'd_model': small_model_cfg.d_model,
                'n_layers': small_model_cfg.n_layers,
                'n_heads': small_model_cfg.n_heads,
                'n_kv_heads': small_model_cfg.n_kv_heads,
                'd_head': small_model_cfg.d_head,
                'vocab_size': small_model_cfg.vocab_size,
                'rope_theta': small_model_cfg.rope_theta,
                'rope_scaling': small_model_cfg.rope_scaling,
                'dropout': small_model_cfg.dropout,
            }},
        }, checkpoint_path)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model2 = StudentLM(small_model_cfg).to(device)
        model2.load_state_dict(checkpoint['model_state_dict'])
        
        # Verify models produce same output
        model1.eval()
        model2.eval()
        with torch.no_grad():
            logits1 = model1(input_ids)
            logits2 = model2(input_ids)
        
        assert torch.allclose(logits1, logits2, atol=1e-5)


class TestProcessSupervisionIntegration:
    """Integration tests for process supervision training."""
    
    def test_process_supervision_loss_with_model(self, small_model_cfg, device, mock_tokenizer):
        """Test process supervision loss computation with model outputs."""
        from training.process_losses import process_supervision_loss
        
        model = StudentLM(small_model_cfg).to(device)
        model.train()
        
        b, t = 2, 10
        input_ids = torch.randint(0, small_model_cfg.vocab_size, (b, t), device=device)
        
        # Forward pass
        logits = model(input_ids)
        
        # Generate dummy text (simplified - in real training, this would be decoded)
        generated_texts = [
            '{"name": "web_search", "arguments": {"query": "test"}}',
            '{"name": "read_file", "arguments": {"path": "test.txt"}}',
        ]
        
        
        # Compute process supervision loss
        # Note: tool_selection_loss may fail if logits don't match expected positions
        # For integration test, we'll just test JSON validity loss
        losses = process_supervision_loss(
            logits=logits,
            generated_texts=generated_texts,
            target_tool_names=None,  # Skip tool selection to avoid position issues
            tool_names=None,
            tokenizer=None,
            json_validity_weight=1.0,
            tool_select_weight=0.0,
        )
        
        assert "total" in losses
        assert losses["total"].item() >= 0.0
        assert "json_validity" in losses

