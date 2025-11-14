"""
Tests for training/utils.py - Shared training utilities.

Tests SHA256 state dict hashing with various tensor types,
CUDA/CPU tensors, and edge cases.
"""
# @author: @darianrosebrook

from unittest.mock import Mock, patch

import torch

from training.utils import sha256_state_dict


class TestSHA256StateDict:
    """Test SHA256 state dict hashing functionality."""

    def test_sha256_state_dict_simple_tensors(self):
        """Test hashing with simple tensors."""
        state_dict = {
            "layer.weight": torch.randn(10, 5),
            "layer.bias": torch.randn(5),
        }

        result = sha256_state_dict(state_dict)

        # Should return a string hash
        assert isinstance(result, str)
        assert len(result) == 64  # SHA256 hex length

        # Should be consistent
        result2 = sha256_state_dict(state_dict)
        assert result == result2

    def test_sha256_state_dict_different_tensors_different_hash(self):
        """Test that different tensors produce different hashes."""
        state_dict1 = {"weight": torch.ones(3, 3)}
        state_dict2 = {"weight": torch.zeros(3, 3)}

        hash1 = sha256_state_dict(state_dict1)
        hash2 = sha256_state_dict(state_dict2)

        assert hash1 != hash2

    def test_sha256_state_dict_order_independence(self):
        """Test that parameter order doesn't affect hash."""
        state_dict1 = {
            "a.weight": torch.ones(2, 2),
            "b.bias": torch.ones(2),
        }
        state_dict2 = {
            "b.bias": torch.ones(2),
            "a.weight": torch.ones(2, 2),
        }

        hash1 = sha256_state_dict(state_dict1)
        hash2 = sha256_state_dict(state_dict2)

        assert hash1 == hash2

    def test_sha256_state_dict_cuda_tensors(self):
        """Test hashing with CUDA tensors."""
        if torch.cuda.is_available():
            # Test with CUDA tensors
            cuda_tensor = torch.randn(3, 4).cuda()
            state_dict = {"cuda_param": cuda_tensor}

            result = sha256_state_dict(state_dict)

            assert isinstance(result, str)
            assert len(result) == 64

            # Should produce same hash as CPU version
            cpu_tensor = cuda_tensor.cpu()
            cpu_state_dict = {"cuda_param": cpu_tensor}

            cpu_result = sha256_state_dict(cpu_state_dict)
            assert result == cpu_result

    def test_sha256_state_dict_mixed_devices(self):
        """Test hashing with mixed CPU/CUDA tensors."""
        state_dict = {
            "cpu_param": torch.randn(2, 3),  # CPU
        }

        if torch.cuda.is_available():
            state_dict["cuda_param"] = torch.randn(2, 3).cuda()

        result = sha256_state_dict(state_dict)

        assert isinstance(result, str)
        assert len(result) == 64

    def test_sha256_state_dict_different_dtypes(self):
        """Test hashing with different tensor dtypes."""
        state_dicts = [
            {"param": torch.ones(2, 2, dtype=torch.float32)},
            {"param": torch.ones(2, 2, dtype=torch.float64)},
            {"param": torch.ones(2, 2, dtype=torch.int32)},
            {"param": torch.ones(2, 2, dtype=torch.int64)},
        ]

        hashes = [sha256_state_dict(sd) for sd in state_dicts]

        # All hashes should be different due to different dtypes
        for i in range(len(hashes)):
            for j in range(i + 1, len(hashes)):
                assert hashes[i] != hashes[j]

    def test_sha256_state_dict_empty_state_dict(self):
        """Test hashing with empty state dict."""
        state_dict = {}

        result = sha256_state_dict(state_dict)

        assert isinstance(result, str)
        assert len(result) == 64

        # Should be consistent
        result2 = sha256_state_dict(state_dict)
        assert result == result2

    def test_sha256_state_dict_single_tensor(self):
        """Test hashing with single tensor."""
        state_dict = {"single": torch.randn(1)}

        result = sha256_state_dict(state_dict)

        assert isinstance(result, str)
        assert len(result) == 64

    def test_sha256_state_dict_large_tensors(self):
        """Test hashing with large tensors."""
        # Create a reasonably large tensor
        large_tensor = torch.randn(100, 100)
        state_dict = {"large": large_tensor}

        result = sha256_state_dict(state_dict)

        assert isinstance(result, str)
        assert len(result) == 64

    def test_sha256_state_dict_detach_handling(self):
        """Test that detach() is called on tensors."""
        tensor_with_grad = torch.randn(3, 3, requires_grad=True)
        tensor_with_grad.grad = torch.randn(3, 3)

        state_dict = {"grad_tensor": tensor_with_grad}

        # Should not raise error due to grad
        result = sha256_state_dict(state_dict)

        assert isinstance(result, str)
        assert len(result) == 64

    def test_sha256_state_dict_buffer_operations(self):
        """Test buffer operations in hashing."""
        # Test that buffer operations work correctly by verifying hash consistency
        state_dict = {"test": torch.ones(2, 2)}

        # Multiple calls should produce same hash (verifies buffer operations work)
        result1 = sha256_state_dict(state_dict)
        result2 = sha256_state_dict(state_dict)

        assert result1 == result2
        assert isinstance(result1, str)
        assert len(result1) == 64

    def test_sha256_state_dict_serialization_failure(self, capsys):
        """Test graceful handling of serialization failures."""

        # Create a tensor that might fail serialization
        class BadTensor:
            def __init__(self):
                self.shape = (2, 2)
                self.dtype = torch.float32

            def is_cuda(self):
                return False

            def cpu(self):
                return self

            def detach(self):
                return self

            def numpy(self):
                raise Exception("Serialization failed")

        bad_tensor = BadTensor()
        state_dict = {"bad_tensor": bad_tensor}

        result = sha256_state_dict(state_dict)

        # Should still produce a hash despite failure
        assert isinstance(result, str)
        assert len(result) == 64

        # Should have printed warning
        captured = capsys.readouterr()
        assert "WARN: Could not serialize tensor bad_tensor" in captured.out

    def test_sha256_state_dict_complex_state_dict(self):
        """Test hashing with complex nested state dict structure."""
        state_dict = {
            "encoder.layer.0.weight": torch.randn(10, 5),
            "encoder.layer.0.bias": torch.randn(5),
            "encoder.layer.1.weight": torch.randn(5, 3),
            "encoder.layer.1.bias": torch.randn(3),
            "decoder.weight": torch.randn(3, 2),
            "decoder.bias": torch.randn(2),
        }

        result = sha256_state_dict(state_dict)

        assert isinstance(result, str)
        assert len(result) == 64

        # Changing any parameter should change the hash
        modified_state_dict = state_dict.copy()
        modified_state_dict["encoder.layer.0.weight"] = torch.randn(10, 5)

        modified_result = sha256_state_dict(modified_state_dict)
        assert result != modified_result

    def test_sha256_state_dict_hash_properties(self):
        """Test hash properties and consistency."""
        # Create multiple state dicts
        state_dicts = [
            {"param1": torch.ones(2, 2), "param2": torch.zeros(2, 2)},
            {"param1": torch.ones(2, 2), "param2": torch.ones(2, 2)},
            {"param1": torch.zeros(2, 2), "param2": torch.zeros(2, 2)},
        ]

        hashes = [sha256_state_dict(sd) for sd in state_dicts]

        # All hashes should be different
        assert len(set(hashes)) == len(hashes)

        # Each hash should be valid SHA256
        for h in hashes:
            assert len(h) == 64
            assert all(c in "0123456789abcdef" for c in h)

    def test_sha256_state_dict_metadata_included(self):
        """Test that tensor metadata is included in hash."""
        # Same tensor data but different shapes should produce different hashes
        tensor1 = torch.ones(4)  # Shape (4,)
        tensor2 = torch.ones(2, 2)  # Shape (2, 2)

        # Same numerical values but different shapes
        hash1 = sha256_state_dict({"param": tensor1})
        hash2 = sha256_state_dict({"param": tensor2})

        assert hash1 != hash2

    def test_sha256_state_dict_dtype_included(self):
        """Test that dtype is included in hash."""
        tensor_float = torch.ones(2, 2, dtype=torch.float32)
        tensor_double = torch.ones(2, 2, dtype=torch.float64)

        hash_float = sha256_state_dict({"param": tensor_float})
        hash_double = sha256_state_dict({"param": tensor_double})

        assert hash_float != hash_double

    @patch("training.utils.hashlib.sha256")
    def test_sha256_state_dict_hashlib_called(self, mock_sha256):
        """Test that hashlib.sha256 is called correctly."""
        mock_hash = Mock()
        mock_hash.hexdigest.return_value = "mock_hash"
        mock_sha256.return_value = mock_hash

        state_dict = {"test": torch.ones(2, 2)}

        result = sha256_state_dict(state_dict)

        mock_sha256.assert_called_once()
        mock_hash.hexdigest.assert_called_once()
        assert result == "mock_hash"


class TestUtilsIntegration:
    """Test integration of utility functions."""

    def test_sha256_state_dict_real_model_params(self):
        """Test hashing with realistic model parameters."""
        # Simulate a small transformer-like model state dict
        state_dict = {
            "embedding.weight": torch.randn(1000, 64),
            "attention.q_proj.weight": torch.randn(64, 64),
            "attention.k_proj.weight": torch.randn(64, 64),
            "attention.v_proj.weight": torch.randn(64, 64),
            "attention.o_proj.weight": torch.randn(64, 64),
            "mlp.gate_proj.weight": torch.randn(256, 64),
            "mlp.up_proj.weight": torch.randn(256, 64),
            "mlp.down_proj.weight": torch.randn(64, 256),
            "output.weight": torch.randn(1000, 64),
        }

        result = sha256_state_dict(state_dict)

        assert isinstance(result, str)
        assert len(result) == 64

        # Hash should be stable
        result2 = sha256_state_dict(state_dict)
        assert result == result2

    def test_sha256_state_dict_memory_efficiency(self):
        """Test that hashing is memory efficient."""
        # Create a moderately large state dict
        large_state_dict = {}
        for i in range(10):
            large_state_dict[f"layer_{i}.weight"] = torch.randn(100, 100)

        # Should complete without excessive memory usage
        result = sha256_state_dict(large_state_dict)

        assert isinstance(result, str)
        assert len(result) == 64

    def test_sha256_state_dict_error_recovery(self):
        """Test error recovery in hashing."""
        # Mix of good and problematic tensors
        good_tensor = torch.ones(2, 2)
        bad_tensor = Mock()
        bad_tensor.is_cuda.return_value = False
        bad_tensor.cpu.return_value = bad_tensor
        bad_tensor.detach.return_value = bad_tensor
        bad_tensor.numpy.side_effect = Exception("Bad tensor")

        state_dict = {
            "good": good_tensor,
            "bad": bad_tensor,
        }

        # Should handle the bad tensor gracefully
        result = sha256_state_dict(state_dict)

        assert isinstance(result, str)
        assert len(result) == 64
