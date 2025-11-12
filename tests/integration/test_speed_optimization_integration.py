"""
Integration tests for inference speed optimization features.

Tests the full integration of speed optimizations into training pipeline.
These tests verify that features work together before expensive API calls.
"""
import json
import tempfile
from pathlib import Path

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock

from models.student.architectures.gqa_transformer import StudentLM, ModelCfg
from training.dataset import KDDataset, collate_kd_batch
from training.distill_kd import (
    sample_enumerated_shape,
    truncate_batch_to_shape,
    should_enable_qat,
    train_step,
)
from training.losses import (
    length_aware_kd_loss,
    early_tool_call_loss,
    combined_kd_loss,
)
from training.speed_metrics import measure_proxy, aggregate_speed_metrics


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = Mock()
    tokenizer.vocab_size = 1000
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.pad_token = tokenizer.eos_token_id
    
    def encode(text, add_special_tokens=False, **kwargs):
        tokens = text.split()
        return [abs(hash(t)) % tokenizer.vocab_size for t in tokens]
    
    def decode(token_ids, skip_special_tokens=False, **kwargs):
        return '{"name": "test_tool", "arguments": {"key": "value"}}'
    
    def convert_tokens_to_ids(tokens):
        return {"{": 5, "[": 6, '"': 7}.get(tokens, None)
    
    tokenizer.encode = encode
    tokenizer.decode = decode
    tokenizer.convert_tokens_to_ids = convert_tokens_to_ids
    return tokenizer


@pytest.fixture
def temp_dataset_with_metadata(mock_tokenizer, tmp_path):
    """Create dataset with metadata for speed optimization features."""
    jsonl_file = tmp_path / "kd_dataset.jsonl"
    
    samples = [
        {
            "prompt": "Use web_search to find information about Python.",
            "teacher_text": '{"name": "web_search", "arguments": {"q": "Python"}}',
            "tool_should_be_used": True,
            "teacher_prefix_ids": [5, 7, 110, 101, 98],  # Mock JSON start tokens
            "required_fields_present": False,
        },
        {
            "prompt": "Just answer this question.",
            "teacher_text": "Python is a programming language.",
            "tool_should_be_used": False,
            "required_fields_present": True,
        },
    ]
    
    with open(jsonl_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    return str(jsonl_file)


class TestEnumeratedShapeTrainingIntegration:
    """Integration tests for enumerated shape training."""
    
    def test_shape_sampling_in_training_loop(self, temp_dataset_with_metadata, mock_tokenizer, small_model_cfg, device):
        """Test that shape sampling works in training context."""
        from torch.utils.data import DataLoader
        
        # Mock tokenizer loading
        with patch('training.dataset.AutoTokenizer') as mock_auto_tokenizer:
            mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
            
            dataset = KDDataset(
                jsonl_path=temp_dataset_with_metadata,
                tokenizer_path="mock_tokenizer",
                max_seq_length=4096,  # Max length
                teacher_logits_available=False,
            )
            
            dataloader = DataLoader(
                dataset,
                batch_size=2,
                shuffle=False,
                collate_fn=collate_kd_batch,
            )
            
            batch = next(iter(dataloader))
            
            # Sample a shape and truncate
            sampled_shape = sample_enumerated_shape([512, 1024, 2048, 4096], step=0)
            truncated_batch = truncate_batch_to_shape(batch, sampled_shape)
            
            # Verify truncation
            assert truncated_batch["input_ids"].shape[1] <= sampled_shape
            assert truncated_batch["attention_mask"].shape[1] <= sampled_shape
    
    def test_multiple_shapes_in_sequence(self):
        """Test that different shapes are sampled over multiple steps."""
        seq_lengths = [512, 1024, 2048]
        samples = [sample_enumerated_shape(seq_lengths, step=i) for i in range(20)]
        
        # Should see multiple different shapes
        unique_shapes = set(samples)
        assert len(unique_shapes) > 1  # Should sample different shapes


class TestLatencyAwareLossesIntegration:
    """Integration tests for latency-aware losses."""
    
    def test_length_aware_loss_in_training_step(self, small_model_cfg, device):
        """Test length-aware loss computation in training context."""
        model = StudentLM(small_model_cfg).to(device)
        model.train()
        
        b, t = 2, 20
        input_ids = torch.randint(0, small_model_cfg.vocab_size, (b, t), device=device)
        
        # Forward pass
        student_logits = model(input_ids)
        
        # Create attention masks
        student_attn_mask = torch.ones(b, t, device=device)
        teacher_attn_mask = torch.ones(b, t // 2, device=device)
        teacher_attn_mask = torch.nn.functional.pad(teacher_attn_mask, (0, t - teacher_attn_mask.size(1)))
        
        # Required fields present
        required_fields = torch.tensor([False, True], device=device)
        
        # Compute length-aware loss
        loss, diags = length_aware_kd_loss(
            student_attn_mask=student_attn_mask,
            teacher_attn_mask=teacher_attn_mask,
            required_fields_present=required_fields,
            hinge=0.15,
            slope=1.0,
        )
        
        assert loss.item() >= 0.0
        assert "len_kd.median_rel_excess" in diags
    
    def test_early_tool_loss_in_training_step(self, small_model_cfg, device, mock_tokenizer):
        """Test early tool call loss computation in training context."""
        model = StudentLM(small_model_cfg).to(device)
        model.train()
        
        b, t = 2, 20
        input_ids = torch.randint(0, small_model_cfg.vocab_size, (b, t), device=device)
        
        # Forward pass
        logits = model(input_ids)
        
        # Tool should be used
        tool_should_be_used = torch.tensor([True, False], device=device)
        teacher_prefix_ids = torch.full((b, 25), fill_value=-100, device=device)
        teacher_prefix_ids[0, :5] = torch.tensor([5, 7, 110, 101, 98], device=device)  # Mock JSON start
        
        # Compute early tool loss
        loss, diags = early_tool_call_loss(
            logits=logits,
            input_ids=input_ids,
            tool_should_be_used=tool_should_be_used,
            tokenizer=mock_tokenizer,
            teacher_prefix_ids=teacher_prefix_ids,
            N=25,
            ce_weight=0.2,
            ramp_t=1.0,
        )
        
        assert loss.item() >= 0.0
        assert "early_tool.frac_should_use" in diags
    
    def test_latency_losses_with_combined_kd(self, small_model_cfg, device, mock_tokenizer):
        """Test that latency-aware losses integrate with combined_kd_loss."""
        model = StudentLM(small_model_cfg).to(device)
        model.train()
        
        b, t = 2, 20
        input_ids = torch.randint(0, small_model_cfg.vocab_size, (b, t), device=device)
        labels = torch.randint(0, small_model_cfg.vocab_size, (b, t), device=device)
        
        # Forward pass
        student_logits = model(input_ids)
        teacher_logits = torch.randn_like(student_logits)
        
        # Standard KD loss
        kd_loss_dict = combined_kd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            teacher_targets=teacher_logits.argmax(dim=-1),
            ground_truth_targets=labels,
            kl_weight=0.5,
            ce_teacher_weight=0.3,
            ce_ground_truth_weight=0.2,
        )
        
        # Length-aware loss
        student_attn_mask = torch.ones(b, t, device=device)
        teacher_attn_mask = torch.ones(b, t // 2, device=device)
        teacher_attn_mask = torch.nn.functional.pad(teacher_attn_mask, (0, t - teacher_attn_mask.size(1)))
        required_fields = torch.tensor([False, False], device=device)
        
        length_loss, _ = length_aware_kd_loss(
            student_attn_mask=student_attn_mask,
            teacher_attn_mask=teacher_attn_mask,
            required_fields_present=required_fields,
        )
        
        # Combined loss should be computable
        total_loss = kd_loss_dict["total"] + 0.05 * length_loss
        assert total_loss.item() >= 0.0
        assert total_loss.requires_grad


class TestQATIntegration:
    """Integration tests for QAT in training loop."""
    
    def test_qat_enablement_timing(self):
        """Test QAT enablement at correct step."""
        qat_cfg = {"enabled": True, "start_fraction": 0.8}
        total_steps = 1000
        
        # Before QAT start
        assert not should_enable_qat(step=799, total_steps=total_steps, qat_cfg=qat_cfg)
        
        # At QAT start
        assert should_enable_qat(step=800, total_steps=total_steps, qat_cfg=qat_cfg)
        
        # After QAT start
        assert should_enable_qat(step=900, total_steps=total_steps, qat_cfg=qat_cfg)
    
    def test_qat_disabled_by_default(self):
        """Test that QAT is disabled by default."""
        qat_cfg = {}  # Empty config
        assert not should_enable_qat(step=1000, total_steps=1000, qat_cfg=qat_cfg)


class TestSpeedMetricsIntegration:
    """Integration tests for speed metrics during validation."""
    
    def test_speed_metrics_with_model(self, small_model_cfg, device, mock_tokenizer):
        """Test speed metrics measurement with real model."""
        model = StudentLM(small_model_cfg).to(device)
        model.eval()
        
        batch = {
            "input_ids": torch.randint(0, small_model_cfg.vocab_size, (1, 10), device=device),
            "attention_mask": torch.ones(1, 10, device=device),
        }
        
        metrics = measure_proxy(
            model=model,
            batch=batch,
            tokenizer=mock_tokenizer,
            device=device,
            max_new_tokens=5,
        )
        
        assert "ttft_ms" in metrics
        assert "tps" in metrics
        assert "ttfa_tokens" in metrics
        assert metrics["ttft_ms"] > 0.0
        assert metrics["tps"] > 0.0
    
    def test_speed_metrics_aggregation(self):
        """Test speed metrics aggregation."""
        metrics_list = [
            {"ttft_ms": 100.0, "tps": 50.0, "ttfa_tokens": 10.0, "ttfa_ms": 200.0},
            {"ttft_ms": 150.0, "tps": 60.0, "ttfa_tokens": 15.0, "ttfa_ms": 250.0},
            {"ttft_ms": 200.0, "tps": 70.0, "ttfa_tokens": 20.0, "ttfa_ms": 300.0},
        ]
        
        aggregated = aggregate_speed_metrics(metrics_list)
        
        assert aggregated["ttft_ms"]["p50"] == 150.0
        assert aggregated["ttft_ms"]["p95"] == 200.0
        assert aggregated["tps"]["p50"] == 60.0
        assert aggregated["ttfa_tokens"]["p50"] == 15.0


class TestTrainingStepWithSpeedOptimizations:
    """Integration tests for training step with speed optimizations."""
    
    def test_training_step_with_length_loss(self, small_model_cfg, device, mock_tokenizer):
        """Test training step with length-aware loss enabled."""
        model = StudentLM(small_model_cfg).to(device)
        model.train()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        batch = {
            "input_ids": torch.randint(0, small_model_cfg.vocab_size, (2, 20), device=device),
            "attention_mask": torch.ones(2, 20, device=device),
            "labels": torch.randint(0, small_model_cfg.vocab_size, (2, 20), device=device),
            "teacher_target_ids": torch.randint(0, small_model_cfg.vocab_size, (2, 20), device=device),
            "teacher_attention_mask": torch.ones(2, 10, device=device),
            "required_fields_present": torch.tensor([False, False], device=device),
        }
        
        cfg = {
            "kd": {
                "use_length_aware_kd": True,
                "length_kd_weight": 0.05,
                "length_kd_hinge": 0.15,
                "kl_weight": 0.5,
                "ce_teacher_weight": 0.3,
                "ce_ground_truth_weight": 0.2,
            },
        }
        
        # Training step should complete without errors
        loss_dict = train_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            scaler=None,
            cfg=cfg,
            device=device,
            grad_accum_steps=1,
            grad_accum_counter=0,
            current_step=0,
        )
        
        assert "total" in loss_dict
        assert "length_kd" in loss_dict
        assert loss_dict["total"] >= 0.0
    
    def test_training_step_with_early_tool_loss(self, small_model_cfg, device, mock_tokenizer):
        """Test training step with early tool call loss enabled."""
        model = StudentLM(small_model_cfg).to(device)
        model.train()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        batch = {
            "input_ids": torch.randint(0, small_model_cfg.vocab_size, (2, 20), device=device),
            "attention_mask": torch.ones(2, 20, device=device),
            "labels": torch.randint(0, small_model_cfg.vocab_size, (2, 20), device=device),
            "teacher_target_ids": torch.randint(0, small_model_cfg.vocab_size, (2, 20), device=device),
            "tool_should_be_used": torch.tensor([True, False], device=device),
            "teacher_prefix_ids": torch.full((2, 25), fill_value=-100, device=device),
        }
        batch["teacher_prefix_ids"][0, :5] = torch.tensor([5, 7, 110, 101, 98], device=device)
        
        cfg = {
            "kd": {
                "use_early_tool_call_loss": True,
                "early_tool_weight": 0.05,
                "early_tool_N": 25,
                "early_tool_ce_weight": 0.2,
                "kl_weight": 0.5,
                "ce_teacher_weight": 0.3,
                "ce_ground_truth_weight": 0.2,
            },
            "tokenizer_path": "mock_tokenizer",  # Provide tokenizer path for train_step
        }
        
        # Attach tokenizer to model for train_step to find it
        model.tokenizer = mock_tokenizer
        
        # Training step should complete without errors
        loss_dict = train_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            scaler=None,
            cfg=cfg,
            device=device,
            grad_accum_steps=1,
            grad_accum_counter=0,
            current_step=0,
        )
        
        assert "total" in loss_dict
        assert "early_tool" in loss_dict
        assert loss_dict["total"] >= 0.0

