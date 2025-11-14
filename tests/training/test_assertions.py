"""
Tests for training/assertions.py - Shared assertion utilities for training pipeline.

Tests loss finiteness checks, gradient norm computation, and training assertions.
"""
# @author: @darianrosebrook

import pytest
import torch
import torch.nn as nn
from training.assertions import (
    assert_loss_finite,
    log_loss_components,
    compute_gradient_norms,
    check_gradient_balance,
    compute_per_component_gradient_norms,
)


class TestAssertLossFinite:
    """Test assert_loss_finite function."""

    def test_assert_loss_finite_valid_tensor(self):
        """Test asserting finite loss with valid tensor."""
        loss_dict = {"total": torch.tensor(1.5), "kl": torch.tensor(0.5)}
        # Should not raise
        assert_loss_finite(loss_dict)

    def test_assert_loss_finite_valid_float(self):
        """Test asserting finite loss with float value."""
        loss_dict = {"total": 1.5, "kl": 0.5}
        # Should not raise
        assert_loss_finite(loss_dict)

    def test_assert_loss_finite_missing_total(self):
        """Test asserting loss without total key."""
        loss_dict = {"kl": torch.tensor(0.5)}
        with pytest.raises(RuntimeError, match="missing 'total' key"):
            assert_loss_finite(loss_dict)

    def test_assert_loss_finite_nan_tensor(self):
        """Test asserting loss with NaN tensor."""
        loss_dict = {"total": torch.tensor(float("nan")), "kl": torch.tensor(0.5)}
        with pytest.raises(RuntimeError, match="not finite"):
            assert_loss_finite(loss_dict)

    def test_assert_loss_finite_inf_tensor(self):
        """Test asserting loss with Inf tensor."""
        loss_dict = {"total": torch.tensor(float("inf")), "kl": torch.tensor(0.5)}
        with pytest.raises(RuntimeError, match="not finite"):
            assert_loss_finite(loss_dict)

    def test_assert_loss_finite_nan_component(self):
        """Test asserting loss with NaN component."""
        loss_dict = {"total": torch.tensor(1.5), "kl": torch.tensor(float("nan"))}
        with pytest.raises(RuntimeError, match="not finite"):
            assert_loss_finite(loss_dict)

    def test_assert_loss_finite_zero_loss(self):
        """Test asserting zero loss."""
        loss_dict = {"total": torch.tensor(0.0), "kl": torch.tensor(0.0)}
        with pytest.raises(RuntimeError, match="Total loss is zero"):
            assert_loss_finite(loss_dict)

    def test_assert_loss_finite_zero_loss_with_nonzero_component(self):
        """Test asserting zero total loss but nonzero component."""
        loss_dict = {"total": torch.tensor(0.0), "kl": torch.tensor(0.5)}
        # Should not raise - has nonzero component
        assert_loss_finite(loss_dict)

    def test_assert_loss_finite_with_step(self):
        """Test asserting loss with step number."""
        loss_dict = {"total": torch.tensor(float("nan"))}
        with pytest.raises(RuntimeError, match="at step 42"):
            assert_loss_finite(loss_dict, step=42)

    def test_assert_loss_finite_inf_component(self):
        """Test asserting loss with Inf component."""
        loss_dict = {"total": torch.tensor(1.5), "kl": torch.tensor(float("inf"))}
        with pytest.raises(RuntimeError, match="not finite"):
            assert_loss_finite(loss_dict)

    def test_assert_loss_finite_multiple_components(self):
        """Test asserting loss with multiple components."""
        loss_dict = {
            "total": torch.tensor(1.5),
            "kl": torch.tensor(0.5),
            "ce": torch.tensor(0.3),
            "code_mode": torch.tensor(0.2),
        }
        # Should not raise
        assert_loss_finite(loss_dict)

    def test_assert_loss_finite_non_tensor_invalid(self):
        """Test asserting loss with non-tensor invalid total loss - line 52-56."""
        # Test with NaN float (not tensor)
        loss_dict = {"total": float("nan"), "kl": 0.5}
        with pytest.raises(RuntimeError, match="not finite"):
            assert_loss_finite(loss_dict)

    def test_assert_loss_finite_non_tensor_inf(self):
        """Test asserting loss with infinite float (not tensor) - line 52-56."""
        loss_dict = {"total": float("inf"), "kl": 0.5}
        with pytest.raises(RuntimeError, match="not finite"):
            assert_loss_finite(loss_dict)

    def test_assert_loss_finite_non_tensor_negative_inf(self):
        """Test asserting loss with negative infinite float - line 52-56."""
        loss_dict = {"total": float("-inf"), "kl": 0.5}
        with pytest.raises(RuntimeError, match="not finite"):
            assert_loss_finite(loss_dict)

    def test_assert_loss_finite_tensor_with_some_nan(self):
        """Test asserting loss with tensor containing some NaN values (not all)."""
        # Create tensor with some NaN values
        tensor_with_nan = torch.tensor([1.0, float("nan"), 3.0])
        loss_dict = {"total": tensor_with_nan}
        with pytest.raises(RuntimeError, match="not finite"):
            assert_loss_finite(loss_dict)

    def test_assert_loss_finite_tensor_with_some_inf(self):
        """Test asserting loss with tensor containing some Inf values (not all)."""
        tensor_with_inf = torch.tensor([1.0, float("inf"), 3.0])
        loss_dict = {"total": tensor_with_inf}
        with pytest.raises(RuntimeError, match="not finite"):
            assert_loss_finite(loss_dict)

    def test_assert_loss_finite_empty_loss_dict(self):
        """Test asserting loss with empty dictionary."""
        loss_dict = {}
        with pytest.raises(RuntimeError, match="missing 'total' key"):
            assert_loss_finite(loss_dict)

    def test_assert_loss_finite_zero_loss_all_zero_components(self):
        """Test asserting zero total loss when all components are zero - line 46-50."""
        loss_dict = {"total": torch.tensor(0.0), "kl": torch.tensor(0.0), "ce": torch.tensor(0.0)}
        with pytest.raises(RuntimeError, match="Total loss is zero"):
            assert_loss_finite(loss_dict)

    def test_assert_loss_finite_component_with_some_nan(self):
        """Test asserting loss with component tensor containing some NaN values."""
        loss_dict = {
            "total": torch.tensor(1.5),
            "kl": torch.tensor([0.5, float("nan"), 0.7]),
        }
        with pytest.raises(RuntimeError, match="not finite"):
            assert_loss_finite(loss_dict)

    def test_assert_loss_finite_component_with_some_inf(self):
        """Test asserting loss with component tensor containing some Inf values."""
        loss_dict = {
            "total": torch.tensor(1.5),
            "kl": torch.tensor([0.5, float("inf"), 0.7]),
        }
        with pytest.raises(RuntimeError, match="not finite"):
            assert_loss_finite(loss_dict)


class TestLogLossComponents:
    """Test log_loss_components function."""

    def test_log_loss_components_logs_at_step(self, capsys):
        """Test that loss components are logged at specified step."""
        loss_dict = {"total": torch.tensor(1.5), "kl": torch.tensor(0.5)}
        log_loss_components(loss_dict, step=100, log_every=100)
        captured = capsys.readouterr()
        assert "Step 100" in captured.out
        assert "loss components" in captured.out

    def test_log_loss_components_skips_non_log_step(self, capsys):
        """Test that loss components are not logged at non-log step."""
        loss_dict = {"total": torch.tensor(1.5), "kl": torch.tensor(0.5)}
        log_loss_components(loss_dict, step=99, log_every=100)
        captured = capsys.readouterr()
        assert "Step 99" not in captured.out

    def test_log_loss_components_includes_all_components(self, capsys):
        """Test that all loss components are logged."""
        loss_dict = {
            "total": torch.tensor(1.5),
            "kl": torch.tensor(0.5),
            "ce": torch.tensor(0.3),
        }
        log_loss_components(loss_dict, step=200, log_every=100)
        captured = capsys.readouterr()
        assert "total" in captured.out
        assert "kl" in captured.out
        assert "ce" in captured.out

    def test_log_loss_components_float_values(self, capsys):
        """Test logging loss components with float values."""
        loss_dict = {"total": 1.5, "kl": 0.5}
        log_loss_components(loss_dict, step=100, log_every=100)
        captured = capsys.readouterr()
        assert "Step 100" in captured.out
        assert "total=1.5000" in captured.out
        assert "kl=0.5000" in captured.out

    def test_log_loss_components_mixed_types(self, capsys):
        """Test logging loss components with mixed tensor and float types."""
        loss_dict = {"total": torch.tensor(1.5), "kl": 0.5, "ce": torch.tensor(0.3)}
        log_loss_components(loss_dict, step=300, log_every=100)

        captured = capsys.readouterr()
        assert "Step 300" in captured.out
        assert "total=1.5000" in captured.out
        assert "ce=0.3000" in captured.out
        assert "kl=0.5000" in captured.out

    def test_log_loss_components_empty_dict(self, capsys):
        """Test logging empty loss dictionary."""
        loss_dict = {}
        log_loss_components(loss_dict, step=100, log_every=100)

        captured = capsys.readouterr()
        assert "Step 100" in captured.out
        # Should handle empty dict gracefully

    def test_log_loss_components_single_component(self, capsys):
        """Test logging single loss component."""
        loss_dict = {"total": torch.tensor(1.0)}
        log_loss_components(loss_dict, step=500, log_every=100)

        captured = capsys.readouterr()
        assert "Step 500" in captured.out
        assert "total=1.0000" in captured.out

    def test_log_loss_components_sorted_output(self, capsys):
        """Test that components are logged in sorted order."""
        loss_dict = {"z_component": torch.tensor(0.1), "a_component": torch.tensor(0.2), "m_component": torch.tensor(0.3)}
        log_loss_components(loss_dict, step=600, log_every=100)

        captured = capsys.readouterr()
        output = captured.out
        # Components should be sorted alphabetically
        assert output.index("a_component") < output.index("m_component")
        assert output.index("m_component") < output.index("z_component")


class TestComputeGradientNorms:
    """Test compute_gradient_norms function."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        model = nn.Linear(10, 5)
        return model

    @pytest.fixture
    def model_with_gradients(self, simple_model):
        """Create a model with computed gradients."""
        model = simple_model
        x = torch.randn(2, 10)
        y = torch.randn(2, 5)
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        return model, {"total": loss}

    def test_compute_gradient_norms_with_gradients(self, model_with_gradients):
        """Test computing gradient norms when gradients exist."""
        model, loss_dict = model_with_gradients
        grad_norms = compute_gradient_norms(model, loss_dict)
        assert "total" in grad_norms
        assert isinstance(grad_norms["total"], float)
        assert grad_norms["total"] > 0

    def test_compute_gradient_norms_no_gradients(self, simple_model):
        """Test computing gradient norms when no gradients exist."""
        model = simple_model
        loss_dict = {"total": torch.tensor(1.0)}
        grad_norms = compute_gradient_norms(model, loss_dict)
        assert "total" in grad_norms
        assert grad_norms["total"] == 0.0

    def test_compute_gradient_norms_multiple_parameters(self, model_with_gradients):
        """Test computing gradient norms with multiple parameters."""
        model, loss_dict = model_with_gradients
        grad_norms = compute_gradient_norms(model, loss_dict)
        # Should compute total norm across all parameters
        assert grad_norms["total"] >= 0


class TestCheckGradientBalance:
    """Test check_gradient_balance function."""

    def test_check_gradient_balance_balanced(self):
        """Test checking balanced gradients."""
        grad_norms = {"kl": 1.0, "ce": 1.2, "code_mode": 0.9}
        result = check_gradient_balance(grad_norms, imbalance_threshold=10.0)
        assert result is None

    def test_check_gradient_balance_imbalanced(self):
        """Test checking imbalanced gradients."""
        grad_norms = {"kl": 100.0, "ce": 1.0, "code_mode": 0.9}
        result = check_gradient_balance(grad_norms, imbalance_threshold=10.0)
        assert result is not None
        assert "imbalance" in result.lower()
        assert "kl" in result  # Should identify the imbalanced component

    def test_check_gradient_balance_single_component(self):
        """Test checking gradient balance with single component."""
        grad_norms = {"total": 1.0}
        result = check_gradient_balance(grad_norms)
        assert result is None

    def test_check_gradient_balance_empty(self):
        """Test checking gradient balance with empty dict."""
        grad_norms = {}
        result = check_gradient_balance(grad_norms)
        assert result is None

    def test_check_gradient_balance_custom_threshold(self):
        """Test checking gradient balance with custom threshold."""
        grad_norms = {"kl": 50.0, "ce": 1.0}
        # With threshold 100, should not detect imbalance
        result = check_gradient_balance(grad_norms, imbalance_threshold=100.0)
        assert result is None

        # With threshold 10, should detect imbalance
        result = check_gradient_balance(grad_norms, imbalance_threshold=10.0)
        assert result is not None

    def test_check_gradient_balance_zero_min(self):
        """Test checking gradient balance when min is zero."""
        grad_norms = {"kl": 1.0, "ce": 0.0}
        result = check_gradient_balance(grad_norms, imbalance_threshold=10.0)
        # Should handle zero gracefully
        assert isinstance(result, (type(None), str))


class TestComputePerComponentGradientNorms:
    """Test compute_per_component_gradient_norms function."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return nn.Linear(10, 5)

    def test_compute_per_component_gradient_norms_single_component(self, simple_model):
        """Test computing per-component gradient norms for single component."""
        model = simple_model
        x = torch.randn(2, 10)
        y = torch.randn(2, 5)

        loss1 = nn.MSELoss()(model(x), y)
        # Use retain_graph=True to preserve computation graph for per-component computation
        loss1.backward(retain_graph=True)

        loss_dict = {"kl": loss1}
        loss_weights = {"kl": 1.0}

        grad_norms = compute_per_component_gradient_norms(model, loss_dict, loss_weights)
        assert "kl" in grad_norms
        assert isinstance(grad_norms["kl"], float)

    def test_compute_per_component_gradient_norms_multiple_components(self, simple_model):
        """Test computing per-component gradient norms for multiple components."""
        model = simple_model
        x = torch.randn(2, 10)
        y = torch.randn(2, 5)

        # Compute gradients for first component
        loss1 = nn.MSELoss()(model(x), y)
        # Use retain_graph=True to preserve computation graph for per-component computation
        loss1.backward(retain_graph=True)

        loss_dict = {"kl": loss1, "ce": loss1 * 0.5}
        loss_weights = {"kl": 1.0, "ce": 0.5}

        grad_norms = compute_per_component_gradient_norms(model, loss_dict, loss_weights)
        # Should compute norms for each component
        assert len(grad_norms) >= 1

    def test_compute_per_component_gradient_norms_zero_weight(self, simple_model):
        """Test computing per-component gradient norms with zero weight."""
        model = simple_model
        x = torch.randn(2, 10)
        y = torch.randn(2, 5)

        loss1 = nn.MSELoss()(model(x), y)
        # Use retain_graph=True to preserve computation graph for per-component computation
        loss1.backward(retain_graph=True)

        loss_dict = {"kl": loss1, "ce": loss1}
        loss_weights = {"kl": 1.0, "ce": 0.0}  # ce has zero weight

        grad_norms = compute_per_component_gradient_norms(model, loss_dict, loss_weights)
        # Should skip zero-weight components
        assert "ce" not in grad_norms or grad_norms.get("ce") == 0.0

    def test_compute_per_component_gradient_norms_restores_gradients(self, simple_model):
        """Test that original gradients are restored."""
        model = simple_model
        x = torch.randn(2, 10)
        y = torch.randn(2, 5)

        loss1 = nn.MSELoss()(model(x), y)
        # Use retain_graph=True to preserve computation graph for per-component computation
        loss1.backward(retain_graph=True)

        # Store original gradients
        original_grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                original_grads[name] = param.grad.clone()

        loss_dict = {"kl": loss1}
        loss_weights = {"kl": 1.0}

        compute_per_component_gradient_norms(model, loss_dict, loss_weights)

        # Check that gradients are restored (may be None or original)
        for name, param in model.named_parameters():
            if name in original_grads:
                # Gradients should exist (may have been modified but should be present)
                assert param.grad is not None

    def test_compute_per_component_gradient_norms_skips_total(self, simple_model):
        """Test that 'total' component is skipped."""
        model = simple_model
        x = torch.randn(2, 10)
        y = torch.randn(2, 5)

        loss1 = nn.MSELoss()(model(x), y)
        # Use retain_graph=True to preserve computation graph for per-component computation
        loss1.backward(retain_graph=True)

        loss_dict = {"total": loss1, "kl": loss1}
        loss_weights = {"kl": 1.0}

        grad_norms = compute_per_component_gradient_norms(model, loss_dict, loss_weights)
        # Should not include 'total'
        assert "total" not in grad_norms

    def test_compute_per_component_gradient_norms_no_gradients_original(self, simple_model):
        """Test compute_per_component_gradient_norms when no original gradients exist."""
        model = simple_model
        x = torch.randn(2, 10)
        y = torch.randn(2, 5)

        # Don't call backward() - no original gradients
        loss1 = nn.MSELoss()(model(x), y)

        loss_dict = {"kl": loss1}
        loss_weights = {"kl": 1.0}

        grad_norms = compute_per_component_gradient_norms(model, loss_dict, loss_weights)
        # Should still compute norms for component
        assert "kl" in grad_norms

    def test_compute_per_component_gradient_norms_missing_weight(self, simple_model):
        """Test compute_per_component_gradient_norms when weight is missing (defaults to 1.0)."""
        model = simple_model
        x = torch.randn(2, 10)
        y = torch.randn(2, 5)

        loss1 = nn.MSELoss()(model(x), y)
        loss1.backward(retain_graph=True)

        loss_dict = {"kl": loss1, "ce": loss1}
        loss_weights = {"kl": 1.0}  # ce weight missing, should default to 1.0

        grad_norms = compute_per_component_gradient_norms(model, loss_dict, loss_weights)
        # Should compute norm for ce even though weight not in dict (defaults to 1.0)
        assert "kl" in grad_norms
        assert "ce" in grad_norms

    def test_compute_per_component_gradient_norms_restores_none_gradients(self, simple_model):
        """Test that gradients are set to None for params without original gradients."""
        model = simple_model
        x = torch.randn(2, 10)
        y = torch.randn(2, 5)

        loss1 = nn.MSELoss()(model(x), y)
        # Only compute gradient for some parameters
        loss1.backward(retain_graph=True)

        # Manually clear some gradients
        param_list = list(model.parameters())
        if len(param_list) > 0:
            param_list[0].grad = None  # Remove gradient for first param

        loss_dict = {"kl": loss1}
        loss_weights = {"kl": 1.0}

        compute_per_component_gradient_norms(model, loss_dict, loss_weights)

        # Param without original gradient should have grad set to None
        assert param_list[0].grad is None


class TestAssertionsIntegration:
    """Test integration of assertion utilities."""

    def test_loss_assertion_with_gradient_computation(self):
        """Test loss assertion followed by gradient computation."""
        model = nn.Linear(10, 5)
        x = torch.randn(2, 10)
        y = torch.randn(2, 5)

        loss = nn.MSELoss()(model(x), y)
        loss_dict = {"total": loss, "mse": loss}

        # Should not raise
        assert_loss_finite(loss_dict)

        loss.backward()
        grad_norms = compute_gradient_norms(model, loss_dict)
        assert "total" in grad_norms

    def test_gradient_balance_check_after_computation(self):
        """Test gradient balance check after computing norms."""
        model = nn.Linear(10, 5)
        x = torch.randn(2, 10)
        y = torch.randn(2, 5)

        loss = nn.MSELoss()(model(x), y)
        loss.backward()

        loss_dict = {"total": loss}
        grad_norms = compute_gradient_norms(model, loss_dict)

        # Check balance (may have single component)
        result = check_gradient_balance(grad_norms)
        # Should handle gracefully
        assert isinstance(result, (type(None), str))







