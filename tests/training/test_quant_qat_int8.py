"""
Tests for training/quant_qat_int8.py - QAT (Quantization-Aware Training) modules for INT8 quantization.

Tests min-max observers, fake quantization, quantized layers, and model quantization.
"""
# @author: @darianrosebrook

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, mock_open, ANY
from training.quant_qat_int8 import (
    MinMaxObserver,
    FakeQuantize,
    QuantizedLinear,
    QuantizedAttention,
    QuantizedEmbedding,
    quantize_linear,
    quantize_attention,
    quantize_swiglu,
    quantize_model,
    load_model_from_checkpoint,
    main,
)


class TestMinMaxObserver:
    """Test MinMaxObserver class."""

    def test_min_max_observer_initialization(self):
        """Test MinMaxObserver initialization."""
        observer = MinMaxObserver(num_channels=64)
        assert observer.min_val.shape == (64,)
        assert observer.max_val.shape == (64,)
        assert observer.num_observations.item() == 0

    def test_min_max_observer_forward_2d(self):
        """Test observing 2D tensor with channel_dim=-1 (last dim is channels)."""
        observer = MinMaxObserver(num_channels=4, channel_dim=-1)
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

        observer(x)

        # For channel_dim=-1, each column represents a channel
        # x has shape [2, 4], so channels are along dim 1
        assert torch.allclose(
            observer.min_val, torch.tensor([1.0, 2.0, 3.0, 4.0]))
        assert torch.allclose(
            observer.max_val, torch.tensor([5.0, 6.0, 7.0, 8.0]))
        assert observer.num_observations.item() == 1

    def test_min_max_observer_forward_multiple_observations(self):
        """Test observing multiple times."""
        observer = MinMaxObserver(num_channels=3, channel_dim=-1)
        x1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        x2 = torch.tensor([[0.5, 1.5, 2.5], [7.0, 8.0, 9.0]])

        observer(x1)
        observer(x2)

        assert torch.allclose(observer.min_val, torch.tensor([0.5, 1.5, 2.5]))
        assert torch.allclose(observer.max_val, torch.tensor([7.0, 8.0, 9.0]))
        assert observer.num_observations.item() == 2

    def test_min_max_observer_get_scale_zero_point_signed(self):
        """Test getting scale and zero point for signed quantization."""
        observer = MinMaxObserver(num_channels=2)
        observer.min_val = torch.tensor([-10.0, -5.0])
        observer.max_val = torch.tensor([10.0, 5.0])

        scale, zero_point = observer.get_scale_zero_point(
            num_bits=8, signed=True)

        assert scale.shape == (2,)
        assert zero_point.shape == (2,)
        assert torch.all(scale > 0)
        assert torch.all(zero_point >= -128)
        assert torch.all(zero_point <= 127)

    def test_min_max_observer_get_scale_zero_point_unsigned(self):
        """Test getting scale and zero point for unsigned quantization."""
        observer = MinMaxObserver(num_channels=2)
        observer.min_val = torch.tensor([0.0, 1.0])
        observer.max_val = torch.tensor([10.0, 5.0])

        scale, zero_point = observer.get_scale_zero_point(
            num_bits=8, signed=False)

        assert scale.shape == (2,)
        assert zero_point.shape == (2,)
        assert torch.all(scale > 0)
        assert torch.all(zero_point >= 0)
        assert torch.all(zero_point <= 255)

    def test_min_max_observer_1d_tensor(self):
        """Test observing 1D tensor."""
        observer = MinMaxObserver(num_channels=3, channel_dim=0)
        x = torch.tensor([1.0, 2.0, 3.0])

        observer(x)

        # Should handle 1D tensors with channel_dim=0
        assert observer.num_observations.item() == 1
        assert torch.allclose(observer.min_val, torch.tensor([1.0, 2.0, 3.0]))
        assert torch.allclose(observer.max_val, torch.tensor([1.0, 2.0, 3.0]))


class TestFakeQuantize:
    """Test FakeQuantize class."""

    @pytest.fixture
    def observer(self):
        """Create a MinMaxObserver instance."""
        return MinMaxObserver(num_channels=4, channel_dim=-1)

    def test_fake_quantize_initialization(self, observer):
        """Test FakeQuantize initialization."""
        fake_quant = FakeQuantize(observer, num_bits=8, signed=True)
        assert fake_quant.observer == observer
        assert fake_quant.num_bits == 8
        assert fake_quant.signed
        assert fake_quant.scale.shape == (4,)
        assert fake_quant.zero_point.shape == (4,)

    def test_fake_quantize_forward_training(self, observer):
        """Test fake quantization in training mode."""
        fake_quant = FakeQuantize(observer, num_bits=8, signed=True)
        fake_quant.train()

        x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        result = fake_quant(x)

        # Observer should be updated
        assert observer.num_observations.item() == 1
        # Result should be quantized and dequantized
        assert result.shape == x.shape

    def test_fake_quantize_forward_eval(self, observer):
        """Test fake quantization in eval mode."""
        fake_quant = FakeQuantize(observer, num_bits=8, signed=True)
        fake_quant.eval()

        # Pre-populate observer
        observer.min_val = torch.tensor([0.0, 0.0, 0.0, 0.0])
        observer.max_val = torch.tensor([10.0, 10.0, 10.0, 10.0])
        observer.num_observations = torch.tensor(1)

        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        result = fake_quant(x)

        # Observer should not be updated in eval mode
        assert observer.num_observations.item() == 1
        assert result.shape == x.shape

    def test_fake_quantize_signed_quantization(self, observer):
        """Test signed fake quantization."""
        fake_quant = FakeQuantize(observer, num_bits=8, signed=True)
        fake_quant.train()

        x = torch.tensor([[-5.0, -2.0, 2.0, 5.0]])
        result = fake_quant(x)

        # Should quantize to range [-128, 127]
        assert result.shape == x.shape

    def test_fake_quantize_unsigned_quantization(self, observer):
        """Test unsigned fake quantization."""
        fake_quant = FakeQuantize(observer, num_bits=8, signed=False)
        fake_quant.train()

        x = torch.tensor([[0.0, 2.0, 5.0, 10.0]])
        result = fake_quant(x)

        # Should quantize to range [0, 255]
        assert result.shape == x.shape

    def test_fake_quantize_scalar_scale(self):
        """Test fake quantization with scalar scale."""
        observer = MinMaxObserver(num_channels=1)
        fake_quant = FakeQuantize(observer, num_bits=8, signed=True)
        fake_quant.train()

        x = torch.tensor([[-5.0, 0.0, 5.0]])
        result = fake_quant(x)

        assert result.shape == x.shape


class TestQuantizedLinear:
    """Test QuantizedLinear class."""

    @pytest.fixture
    def linear_layer(self):
        """Create a linear layer."""
        return nn.Linear(in_features=10, out_features=5, bias=True)

    def test_quantized_linear_initialization(self, linear_layer):
        """Test QuantizedLinear initialization."""
        quantized = QuantizedLinear(linear_layer, weight_bits=8, act_bits=8)

        assert quantized.weight_bits == 8
        assert quantized.act_bits == 8
        assert quantized.weight.shape == linear_layer.weight.shape
        assert quantized.bias is not None

    def test_quantized_linear_forward(self, linear_layer):
        """Test QuantizedLinear forward pass."""
        quantized = QuantizedLinear(linear_layer, weight_bits=8, act_bits=8)
        quantized.train()

        x = torch.randn(2, 10)
        result = quantized(x)

        assert result.shape == (2, 5)

    def test_quantized_linear_no_bias(self):
        """Test QuantizedLinear without bias."""
        linear_layer = nn.Linear(in_features=10, out_features=5, bias=False)
        quantized = QuantizedLinear(linear_layer, weight_bits=8, act_bits=8)

        assert quantized.bias is None

    def test_quantized_linear_different_bits(self, linear_layer):
        """Test QuantizedLinear with different bit widths."""
        quantized = QuantizedLinear(linear_layer, weight_bits=4, act_bits=8)

        assert quantized.weight_bits == 4
        assert quantized.act_bits == 8

        x = torch.randn(2, 10)
        result = quantized(x)

        assert result.shape == (2, 5)


class TestQuantizedAttention:
    """Test QuantizedAttention class."""

    @pytest.fixture
    def mock_attention(self):
        """Create a mock attention module."""
        from models.student.architectures.gqa_transformer import MHA_GQA

        # Create minimal config for attention
        class MockRoPE:
            def apply(self, q, k):
                return q, k

        attn = Mock(spec=MHA_GQA)
        attn.wq = nn.Linear(128, 128)
        attn.wk = nn.Linear(128, 64)
        attn.wv = nn.Linear(128, 64)
        attn.wo = nn.Linear(128, 128)
        attn.n_heads = 8
        attn.n_kv_heads = 4
        attn.d_head = 16
        attn.head_groups = 2
        attn.rope = MockRoPE()
        attn.attn_dropout = nn.Dropout(0.0)

        return attn

    def test_quantized_attention_initialization(self, mock_attention):
        """Test QuantizedAttention initialization."""
        quantized = QuantizedAttention(
            mock_attention, weight_bits=8, act_bits=8)

        assert quantized.n_heads == 8
        assert quantized.n_kv_heads == 4
        assert quantized.d_head == 16
        assert quantized.clamp_pre_softmax

    def test_quantized_attention_forward(self, mock_attention):
        """Test QuantizedAttention forward pass."""
        quantized = QuantizedAttention(
            mock_attention, weight_bits=8, act_bits=8)
        quantized.train()

        x = torch.randn(2, 10, 128)
        result = quantized(x)

        assert result.shape == (2, 10, 128)

    def test_quantized_attention_with_mask(self, mock_attention):
        """Test QuantizedAttention with attention mask."""
        quantized = QuantizedAttention(
            mock_attention, weight_bits=8, act_bits=8)
        quantized.train()

        x = torch.randn(2, 10, 128)
        attn_mask = torch.zeros(2, 10, 10)
        result = quantized(x, attn_mask=attn_mask)

        assert result.shape == (2, 10, 128)

    def test_quantized_attention_clamp_pre_softmax(self, mock_attention):
        """Test QuantizedAttention with pre-softmax clamping."""
        quantized = QuantizedAttention(
            mock_attention, weight_bits=8, act_bits=8, clamp_pre_softmax=True
        )

        x = torch.randn(2, 10, 128)
        result = quantized(x)

        assert result.shape == (2, 10, 128)

    def test_quantized_attention_no_clamp(self, mock_attention):
        """Test QuantizedAttention without pre-softmax clamping."""
        quantized = QuantizedAttention(
            mock_attention, weight_bits=8, act_bits=8, clamp_pre_softmax=False
        )

        x = torch.randn(2, 10, 128)
        result = quantized(x)

        assert result.shape == (2, 10, 128)


class TestQuantizedEmbedding:
    """Test QuantizedEmbedding class."""

    def test_quantized_embedding_initialization(self):
        """Test QuantizedEmbedding initialization."""
        embedding = nn.Embedding(1000, 128)
        quantized = QuantizedEmbedding(embedding, weight_bits=8)

        assert quantized.num_embeddings == 1000
        assert quantized.embedding_dim == 128
        assert quantized.weight_bits == 8
        assert quantized.weight.shape == embedding.weight.shape

    def test_quantized_embedding_forward(self):
        """Test QuantizedEmbedding forward pass."""
        embedding = nn.Embedding(1000, 128)
        quantized = QuantizedEmbedding(embedding, weight_bits=8)
        quantized.train()

        input_ids = torch.randint(0, 1000, (2, 10))
        result = quantized(input_ids)

        assert result.shape == (2, 10, 128)

    def test_quantized_embedding_extra_repr(self):
        """Test QuantizedEmbedding extra_repr."""
        embedding = nn.Embedding(1000, 128)
        quantized = QuantizedEmbedding(embedding, weight_bits=8)

        repr_str = quantized.extra_repr()
        assert "num_embeddings=1000" in repr_str
        assert "embedding_dim=128" in repr_str
        assert "weight_bits=8" in repr_str


class TestQuantizeFunctions:
    """Test quantization helper functions."""

    def test_quantize_linear(self):
        """Test quantize_linear function."""
        linear = nn.Linear(10, 5)
        quantized = quantize_linear(linear, weight_bits=8, act_bits=8)

        assert isinstance(quantized, QuantizedLinear)
        assert quantized.weight.shape == linear.weight.shape

    def test_quantize_attention(self, tmp_path):
        """Test quantize_attention function."""
        from models.student.architectures.gqa_transformer import MHA_GQA

        # Create minimal attention module
        class MockRoPE:
            def apply(self, q, k):
                return q, k

        attn = Mock(spec=MHA_GQA)
        attn.wq = nn.Linear(128, 128)
        attn.wk = nn.Linear(128, 64)
        attn.wv = nn.Linear(128, 64)
        attn.wo = nn.Linear(128, 128)
        attn.n_heads = 8
        attn.n_kv_heads = 4
        attn.d_head = 16
        attn.head_groups = 2
        attn.rope = MockRoPE()
        attn.attn_dropout = nn.Dropout(0.0)

        quantized = quantize_attention(attn, weight_bits=8, act_bits=8)

        assert isinstance(quantized, QuantizedAttention)

    def test_quantize_swiglu(self):
        """Test quantize_swiglu function."""
        from models.student.architectures.gqa_transformer import SwiGLU

        swiglu = SwiGLU(128, 256)
        quantized = quantize_swiglu(swiglu, weight_bits=8, act_bits=8)

        assert isinstance(quantized, SwiGLU)
        # Should have quantized linear layers
        assert hasattr(quantized, "w1")
        assert hasattr(quantized, "w2")
        assert hasattr(quantized, "w3")


class TestQuantizeModel:
    """Test quantize_model function."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock StudentLM model."""
        from models.student.architectures.gqa_transformer import StudentLM, ModelCfg

        cfg = ModelCfg(
            d_model=128,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            d_head=32,
            vocab_size=512,
        )

        model = StudentLM(cfg)
        return model

    def test_quantize_model_basic(self, mock_model):
        """Test basic model quantization."""
        quantized = quantize_model(mock_model, weight_bits=8, act_bits=8)

        assert quantized is not None
        # Should have quantized layers
        assert hasattr(quantized, "lm_head")

    def test_quantize_model_with_embeddings(self, mock_model):
        """Test model quantization with embedding quantization."""
        quantized = quantize_model(
            mock_model, weight_bits=8, act_bits=8, quantize_embeddings=True
        )

        assert quantized is not None

    def test_quantize_model_no_fake_quant_attention(self, mock_model):
        """Test model quantization without fake quant in attention."""
        quantized = quantize_model(
            mock_model, weight_bits=8, act_bits=8, fake_quant_in_attention=False
        )

        assert quantized is not None

    def test_quantize_model_no_clamp_pre_softmax(self, mock_model):
        """Test model quantization without pre-softmax clamping."""
        quantized = quantize_model(
            mock_model, weight_bits=8, act_bits=8, clamp_pre_softmax=False
        )

        assert quantized is not None

    def test_quantize_model_different_bits(self, mock_model):
        """Test model quantization with different bit widths."""
        quantized = quantize_model(mock_model, weight_bits=4, act_bits=8)

        assert quantized is not None


class TestLoadModelFromCheckpoint:
    """Test load_model_from_checkpoint function."""

    @patch("training.safe_checkpoint_loading.safe_load_checkpoint")
    @patch("training.quant_qat_int8.StudentLM")
    @patch("training.quant_qat_int8.ModelCfg")
    def test_load_model_from_checkpoint_success(
        self, mock_model_cfg, mock_student_lm, mock_safe_load
    ):
        """Test successful model loading from checkpoint."""
        mock_checkpoint = {
            "model_state_dict": {"layer.weight": torch.randn(10, 10)},
            "config": {
                "arch": {
                    "d_model": 512,
                    "n_layers": 8,
                    "n_heads": 8,
                    "n_kv_heads": 4,
                    "d_head": 64,
                    "vocab_size": 32000,
                }
            },
        }
        mock_safe_load.return_value = mock_checkpoint

        mock_config = Mock()
        mock_model_cfg.return_value = mock_config

        mock_model = Mock()
        mock_model.to = Mock(return_value=mock_model)
        mock_student_lm.return_value = mock_model

        device = torch.device("cpu")
        result = load_model_from_checkpoint("dummy_path.pt", device)

        assert result == mock_model
        mock_model.load_state_dict.assert_called_once()

    @patch("training.safe_checkpoint_loading.safe_load_checkpoint")
    @patch("training.quant_qat_int8.StudentLM")
    @patch("training.quant_qat_int8.ModelCfg")
    def test_load_model_from_checkpoint_no_config(
        self, mock_model_cfg, mock_student_lm, mock_safe_load
    ):
        """Test model loading without config in checkpoint."""
        mock_checkpoint = {"model_state_dict": {"weight": torch.randn(5, 5)}}
        mock_safe_load.return_value = mock_checkpoint

        mock_config = Mock()
        mock_model_cfg.return_value = mock_config

        mock_model = Mock()
        mock_model.to = Mock(return_value=mock_model)
        mock_student_lm.return_value = mock_model

        device = torch.device("cpu")
        result = load_model_from_checkpoint("dummy_path.pt", device)

        # Should use default config
        mock_model_cfg.assert_called_once_with()
        assert result == mock_model


class TestMainFunction:
    """Test main function and training utilities."""

    @patch("training.quant_qat_int8.torch.save")
    @patch("training.quant_qat_int8.DataLoader")
    @patch("training.quant_qat_int8.KDDataset")
    @patch("training.quant_qat_int8.load_model_from_checkpoint")
    @patch("training.quant_qat_int8.quantize_model")
    @patch("training.quant_qat_int8.torch.optim.AdamW")
    @patch("training.quant_qat_int8.argparse.ArgumentParser.parse_args")
    @patch("training.quant_qat_int8.yaml.safe_load")
    @patch("builtins.open", new_callable=mock_open)
    def test_main_qat_disabled(self, mock_file, mock_yaml, mock_parse_args,
                               mock_optimizer, mock_quantize, mock_load_checkpoint,
                               mock_dataset, mock_dataloader, mock_save):
        """Test main function when QAT is disabled."""
        # Mock config with QAT disabled
        mock_yaml.return_value = {"qat": {"enable": False}}

        # Mock arguments
        mock_args = Mock()
        mock_args.config = ["config.yaml"]
        mock_parse_args.return_value = mock_args

        # Call main
        main()

        # Should exit early without doing anything
        mock_load_checkpoint.assert_not_called()

    @patch("training.quant_qat_int8.torch.save")
    @patch("training.quant_qat_int8.DataLoader")
    @patch("training.quant_qat_int8.KDDataset")
    @patch("training.quant_qat_int8.load_model_from_checkpoint")
    @patch("training.quant_qat_int8.quantize_model")
    @patch("training.quant_qat_int8.torch.optim.AdamW")
    @patch("training.quant_qat_int8.argparse.ArgumentParser.parse_args")
    @patch("training.quant_qat_int8.yaml.safe_load")
    @patch("builtins.open", new_callable=mock_open)
    def test_main_training_loop(self, mock_file, mock_yaml, mock_parse_args,
                                mock_optimizer, mock_quantize, mock_load_checkpoint,
                                mock_dataset, mock_dataloader, mock_save):
        """Test main function training loop setup."""
        # Mock config with QAT enabled
        mock_yaml.return_value = {
            "qat": {"enable": True, "weight_bits": 8, "act_bits": 8},
            "optimizer": {"lr": 1e-4, "betas": [0.9, 0.95], "weight_decay": 0.1},
            "train": {"micro_batch_size": 2, "fp16": False, "grad_clip": 1.0},
            "io": {"train_shards": ["data.jsonl"], "tokenizer_path": "tokenizer"},
            "distillation": {"kl_weight": 0.5, "ce_teacher_weight": 0.3, "ce_ground_truth_weight": 0.2},
            "kd": {"teacher_logits_available": False, "kd_temperature": 2.0}
        }

        # Mock arguments
        mock_args = Mock()
        mock_args.config = ["config.yaml"]
        mock_args.checkpoint = "checkpoint.pt"
        mock_args.output_dir = "output"
        mock_args.steps = 10
        mock_args.save_every = 5
        mock_args.log_every = 1
        mock_parse_args.return_value = mock_args

        # Mock model and components
        mock_model = Mock()
        mock_load_checkpoint.return_value = mock_model

        mock_quantized_model = Mock()
        mock_quantize.return_value = mock_quantized_model

        mock_optimizer_instance = Mock()
        mock_optimizer.return_value = mock_optimizer_instance

        mock_data_loader = Mock()
        mock_dataloader.return_value = mock_data_loader

        # Mock empty dataloader (no batches)
        mock_data_loader.__iter__ = Mock(return_value=iter([]))

        # Call main
        main()

        # Verify components were called
        mock_load_checkpoint.assert_called_once_with("checkpoint.pt", ANY)
        mock_quantize.assert_called_once()
        mock_optimizer.assert_called_once()
        mock_save.assert_called_once()  # Final checkpoint


class TestQuantizationIntegration:
    """Test integration of quantization components."""

    def test_observer_to_fake_quant_workflow(self):
        """Test complete observer to fake quant workflow."""
        observer = MinMaxObserver(num_channels=4, channel_dim=-1)
        fake_quant = FakeQuantize(observer, num_bits=8, signed=True)
        fake_quant.train()

        # Observe some data
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        fake_quant(x)

        # Observer should be updated
        assert observer.num_observations.item() == 1
        # Scale and zero point should be computed
        assert fake_quant.scale.shape == (4,)
        assert fake_quant.zero_point.shape == (4,)

    def test_quantized_linear_workflow(self):
        """Test complete quantized linear workflow."""
        linear = nn.Linear(10, 5)
        quantized = QuantizedLinear(linear, weight_bits=8, act_bits=8)
        quantized.train()

        x = torch.randn(2, 10)
        result = quantized(x)

        # Should quantize both weights and activations
        assert result.shape == (2, 5)

    def test_quantized_model_forward(self, tmp_path):
        """Test quantized model forward pass."""
        from models.student.architectures.gqa_transformer import StudentLM, ModelCfg

        cfg = ModelCfg(
            d_model=128,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            d_head=32,
            vocab_size=512,
        )

        model = StudentLM(cfg)
        quantized = quantize_model(model, weight_bits=8, act_bits=8)
        quantized.train()

        x = torch.randint(0, 512, (2, 10))
        result = quantized(x)

        assert result.shape == (2, 10, 512)
