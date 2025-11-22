"""
Tests for multi-device training utilities.

Tests CPU offloading optimizer constraints, device config validation,
and checkpoint metadata handling.

Author: @darianrosebrook
"""

import pytest
import torch
import torch.nn as nn

from training.multi_device_utils import (
    CPUOffloadOptimizer,
    create_multi_device_optimizer,
    split_model_across_devices,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(100, 64)
        self.blocks = nn.ModuleList([nn.Linear(64, 64) for _ in range(num_layers)])
        self.output = nn.Linear(64, 100)

    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.output(x)


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return SimpleModel(num_layers=2)


@pytest.fixture
def mps_device():
    """Get MPS device if available, otherwise CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TestCPUOffloadOptimizer:
    """Test CPUOffloadOptimizer constraints."""

    def test_cpu_offload_single_device(self, simple_model, mps_device):
        """Test CPU offload works with single-device model."""
        model = simple_model.to(mps_device)
        optimizer = create_multi_device_optimizer(
            model,
            torch.optim.AdamW,
            {"lr": 1e-3},
            use_cpu_offload=True,
        )

        # Should create CPUOffloadOptimizer wrapper
        assert isinstance(optimizer, CPUOffloadOptimizer)

        # Optimizer state should be on CPU
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                if param in optimizer.state:
                    state = optimizer.state[param]
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor):
                            assert value.device.type == "cpu"

    def test_cpu_offload_rejects_split_model(self, simple_model, mps_device):
        """Test CPU offload raises error for split models."""
        # Split model across devices
        model = simple_model
        model.blocks[0].to(mps_device)
        model.blocks[1].to(torch.device("cpu"))
        model.embed.to(mps_device)
        model.output.to(mps_device)

        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="CPUOffloadOptimizer does not support model-parallel"):
            create_multi_device_optimizer(
                model,
                torch.optim.AdamW,
                {"lr": 1e-3},
                use_cpu_offload=True,
            )

    def test_cpu_offload_optimizer_step(self, simple_model, mps_device):
        """Test CPUOffloadOptimizer step() works correctly."""
        model = simple_model.to(mps_device)
        optimizer = create_multi_device_optimizer(
            model,
            torch.optim.AdamW,
            {"lr": 1e-3},
            use_cpu_offload=True,
        )

        # Do a training step
        input_ids = torch.randint(0, 100, (2, 10), device=mps_device)
        output = model(input_ids)
        loss = output.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify optimizer state is back on CPU after step
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                if param in optimizer.state:
                    state = optimizer.state[param]
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor):
                            assert value.device.type == "cpu"

    def test_cpu_offload_state_dict(self, simple_model, mps_device):
        """Test CPUOffloadOptimizer state_dict() returns CPU tensors."""
        model = simple_model.to(mps_device)
        optimizer = create_multi_device_optimizer(
            model,
            torch.optim.AdamW,
            {"lr": 1e-3},
            use_cpu_offload=True,
        )

        # Do a step to populate state
        input_ids = torch.randint(0, 100, (2, 10), device=mps_device)
        output = model(input_ids)
        loss = output.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Get state dict
        state_dict = optimizer.state_dict()

        # Verify all tensors in state dict are on CPU
        for param_group_state in state_dict.get("state", {}).values():
            for key, value in param_group_state.items():
                if isinstance(value, torch.Tensor):
                    assert value.device.type == "cpu"

    def test_cpu_offload_load_state_dict(self, simple_model, mps_device):
        """Test CPUOffloadOptimizer load_state_dict() moves tensors to CPU."""
        model = simple_model.to(mps_device)
        optimizer1 = create_multi_device_optimizer(
            model,
            torch.optim.AdamW,
            {"lr": 1e-3},
            use_cpu_offload=True,
        )

        # Train optimizer1
        input_ids = torch.randint(0, 100, (2, 10), device=mps_device)
        output = model(input_ids)
        loss = output.sum()
        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()

        # Save state dict
        state_dict = optimizer1.state_dict()

        # Create new optimizer and load state
        model2 = SimpleModel().to(mps_device)
        optimizer2 = create_multi_device_optimizer(
            model2,
            torch.optim.AdamW,
            {"lr": 1e-3},
            use_cpu_offload=True,
        )

        # Load state dict (should move to CPU)
        optimizer2.load_state_dict(state_dict)

        # Verify state is on CPU
        for param_group in optimizer2.param_groups:
            for param in param_group["params"]:
                if param in optimizer2.state:
                    state = optimizer2.state[param]
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor):
                            assert value.device.type == "cpu"


class TestModelSplitting:
    """Test model splitting utilities."""

    def test_split_model_alternate(self, simple_model, mps_device):
        """Test splitting model with alternate strategy."""
        model = simple_model
        split_model = split_model_across_devices(
            model,
            {
                "strategy": "alternate",
                "mps_device": mps_device,
                "cpu_device": torch.device("cpu"),
            },
        )

        # Verify blocks are on different devices
        assert split_model.blocks[0].weight.device == mps_device
        assert split_model.blocks[1].weight.device.type == "cpu"

    def test_split_model_first_half_mps(self, simple_model, mps_device):
        """Test splitting model with first_half_mps strategy."""
        model = SimpleModel(num_layers=4)
        split_model = split_model_across_devices(
            model,
            {
                "strategy": "first_half_mps",
                "mps_device": mps_device,
                "cpu_device": torch.device("cpu"),
            },
        )

        # First half on MPS, second half on CPU
        assert split_model.blocks[0].weight.device == mps_device
        assert split_model.blocks[1].weight.device == mps_device
        assert split_model.blocks[2].weight.device.type == "cpu"
        assert split_model.blocks[3].weight.device.type == "cpu"

    def test_split_model_creates_multi_device(self, simple_model, mps_device):
        """Test split model has parameters on multiple devices."""
        model = simple_model
        split_model = split_model_across_devices(
            model,
            {
                "strategy": "alternate",
                "mps_device": mps_device,
                "cpu_device": torch.device("cpu"),
            },
        )

        # Check that parameters are on different devices
        devices = {p.device for p in split_model.parameters()}
        assert len(devices) > 1


class TestDeviceConfigValidation:
    """Test device configuration validation."""

    def test_create_optimizer_allows_multi_device_without_offload(self, simple_model, mps_device):
        """Test optimizer creation allows multi-device without CPU offload."""
        # Split model
        model = simple_model
        model.blocks[0].to(mps_device)
        model.blocks[1].to(torch.device("cpu"))
        model.embed.to(mps_device)
        model.output.to(mps_device)

        # Should work without CPU offload (experimental)
        optimizer = create_multi_device_optimizer(
            model,
            torch.optim.AdamW,
            {"lr": 1e-3},
            use_cpu_offload=False,
        )

        # Should NOT be CPUOffloadOptimizer
        assert not isinstance(optimizer, CPUOffloadOptimizer)


class TestCheckpointDeviceConfig:
    """Test checkpoint device config metadata."""

    def test_checkpoint_includes_device_config(self, simple_model, mps_device, tmp_path):
        """Test checkpoint includes device config metadata."""
        from training.distill_kd import save_checkpoint

        model = simple_model.to(mps_device)
        optimizer = create_multi_device_optimizer(
            model,
            torch.optim.AdamW,
            {"lr": 1e-3},
            use_cpu_offload=True,
        )

        output_dir = tmp_path / "checkpoints"
        output_dir.mkdir()

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            step=100,
            loss=0.5,
            output_dir=output_dir,
            config={"test": "config"},
        )

        # Load checkpoint
        from training.safe_checkpoint_loading import safe_load_checkpoint
        checkpoint_path = output_dir / "checkpoint_step_100.pt"
        checkpoint = safe_load_checkpoint(checkpoint_path)

        # Verify device config is present
        assert "meta" in checkpoint
        assert "device_config" in checkpoint["meta"]
        device_config = checkpoint["meta"]["device_config"]

        assert "model_parallel" in device_config
        assert "cpu_offload" in device_config
        assert "primary_device" in device_config
        assert "all_devices" in device_config

        # Verify values
        assert device_config["model_parallel"] is False
        assert device_config["cpu_offload"] is True
        assert device_config["primary_device"] == str(mps_device)

    def test_checkpoint_includes_model_arch_hash(self, simple_model, mps_device, tmp_path):
        """Test checkpoint includes model architecture hash."""
        from training.distill_kd import save_checkpoint

        model = simple_model.to(mps_device)
        optimizer = create_multi_device_optimizer(
            model,
            torch.optim.AdamW,
            {"lr": 1e-3},
            use_cpu_offload=True,
        )

        output_dir = tmp_path / "checkpoints"
        output_dir.mkdir()

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            step=100,
            loss=0.5,
            output_dir=output_dir,
            config={"test": "config"},
        )

        # Load checkpoint
        from training.safe_checkpoint_loading import safe_load_checkpoint
        checkpoint_path = output_dir / "checkpoint_step_100.pt"
        checkpoint = safe_load_checkpoint(checkpoint_path)

        # Verify model arch hash is present
        assert "meta" in checkpoint
        assert "model_arch_hash" in checkpoint["meta"]
        assert checkpoint["meta"]["model_arch_hash"] is not None

    def test_checkpoint_model_state_on_cpu(self, simple_model, mps_device, tmp_path):
        """Test checkpoint model state dict is saved on CPU."""
        from training.distill_kd import save_checkpoint

        model = simple_model.to(mps_device)
        optimizer = create_multi_device_optimizer(
            model,
            torch.optim.AdamW,
            {"lr": 1e-3},
            use_cpu_offload=True,
        )

        output_dir = tmp_path / "checkpoints"
        output_dir.mkdir()

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            step=100,
            loss=0.5,
            output_dir=output_dir,
            config={"test": "config"},
        )

        # Load checkpoint
        from training.safe_checkpoint_loading import safe_load_checkpoint
        checkpoint_path = output_dir / "checkpoint_step_100.pt"
        checkpoint = safe_load_checkpoint(checkpoint_path, map_location="cpu")

        # Verify model state dict tensors are on CPU
        model_state = checkpoint["model_state_dict"]
        for key, value in model_state.items():
            if isinstance(value, torch.Tensor):
                assert value.device.type == "cpu"

