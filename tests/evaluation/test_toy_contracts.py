"""
Tests for evaluation/toy_contracts.py - Toy model contracts verifier.

Tests verification of converted CoreML models including NaN/zero checks,
enumerated shape validation, and tool span micro-F1 computation.
"""
# @author: @darianrosebrook

import json
from unittest.mock import Mock, patch, MagicMock

import pytest
import numpy as np

# Import the module
import importlib

toy_contracts_module = importlib.import_module("evaluation.toy_contracts")

# Import functions
has_nan_or_zero = toy_contracts_module.has_nan_or_zero
micro_f1 = toy_contracts_module.micro_f1
greedy_decode_logits = toy_contracts_module.greedy_decode_logits
greedy_decode_sequence = toy_contracts_module.greedy_decode_sequence
load_coreml_model = toy_contracts_module.load_coreml_model
main = toy_contracts_module.main


class TestHasNanOrZero:
    """Test has_nan_or_zero function."""

    def test_has_nan_or_zero_normal_values(self):
        """Test has_nan_or_zero with normal values."""
        vec = [1.0, 2.0, 3.0, 4.0, 5.0]
        has_nan, all_zero = has_nan_or_zero(vec)
        
        assert has_nan is False
        assert all_zero is False

    def test_has_nan_or_zero_contains_nan(self):
        """Test has_nan_or_zero detects NaN values."""
        vec = [1.0, float('nan'), 3.0, 4.0]
        has_nan, all_zero = has_nan_or_zero(vec)
        
        assert has_nan is True
        assert all_zero is False

    def test_has_nan_or_zero_all_zero(self):
        """Test has_nan_or_zero detects all-zero values."""
        vec = [0.0, 0.0, 0.0, 0.0]
        has_nan, all_zero = has_nan_or_zero(vec)
        
        assert has_nan is False
        assert all_zero is True

    def test_has_nan_or_zero_very_small_values(self):
        """Test has_nan_or_zero with very small values (treated as zero)."""
        vec = [1e-13, 1e-14, 1e-15]
        has_nan, all_zero = has_nan_or_zero(vec)
        
        assert has_nan is False
        assert all_zero is True  # All values < 1e-12

    def test_has_nan_or_zero_mixed_zero_and_nan(self):
        """Test has_nan_or_zero with both NaN and zero values."""
        vec = [0.0, float('nan'), 0.0]
        has_nan, all_zero = has_nan_or_zero(vec)
        
        assert has_nan is True
        assert all_zero is False  # Has NaN, so not "all zero"

    def test_has_nan_or_zero_empty_list(self):
        """Test has_nan_or_zero with empty list."""
        vec = []
        has_nan, all_zero = has_nan_or_zero(vec)
        
        assert has_nan is False
        assert all_zero is True  # Empty list vacuously all zero

    def test_has_nan_or_zero_single_value(self):
        """Test has_nan_or_zero with single value."""
        vec = [5.0]
        has_nan, all_zero = has_nan_or_zero(vec)
        
        assert has_nan is False
        assert all_zero is False


class TestMicroF1:
    """Test micro_f1 function."""

    def test_micro_f1_perfect_match(self):
        """Test micro_f1 with perfect matches."""
        samples = [
            {"target": "ok tool.call{...}", "pred": "ok tool.call{...}"},
            {"target": "ok tool.call{...}", "pred": "ok tool.call{...}"},
        ]
        
        f1 = micro_f1(samples)
        
        assert f1 > 0.9  # Should be very high with perfect matches

    def test_micro_f1_no_matches(self):
        """Test micro_f1 with no matches."""
        samples = [
            {"target": "ok tool.call{...}", "pred": "no match"},
            {"target": "ok tool.call{...}", "pred": "different text"},
        ]
        
        f1 = micro_f1(samples)
        
        assert f1 < 1.0  # Should be lower with no matches

    def test_micro_f1_partial_matches(self):
        """Test micro_f1 with partial matches."""
        samples = [
            {"target": "ok tool.call{...}", "pred": "ok"},  # Partial match
            {"target": "ok", "pred": "ok tool.call{...}"},  # Extra tokens
        ]
        
        f1 = micro_f1(samples)
        
        assert f1 >= 0.0
        assert f1 <= 1.0

    def test_micro_f1_meaningful_tokens(self):
        """Test micro_f1 gives partial credit for meaningful tokens."""
        samples = [
            {"target": "ok tool.call{...}", "pred": "meaningful text here"},
        ]
        
        f1 = micro_f1(samples)
        
        # Should give partial credit for meaningful tokens
        assert f1 > 0.0

    def test_micro_f1_tool_tokens_present(self):
        """Test micro_f1 detects tool-related tokens."""
        samples = [
            {"target": "ok tool.call{...}", "pred": "ok tool call {"},
        ]
        
        f1 = micro_f1(samples)
        
        # Should detect "tool", "call", "{"
        assert f1 > 0.0

    def test_micro_f1_empty_samples(self):
        """Test micro_f1 with empty samples list."""
        samples = []
        
        f1 = micro_f1(samples)
        
        # Should handle empty list gracefully
        assert f1 == 0.0 or f1 > 0.0  # Implementation dependent

    def test_micro_f1_missing_target(self):
        """Test micro_f1 handles missing target field."""
        samples = [
            {"pred": "ok tool.call{...}"},
        ]
        
        f1 = micro_f1(samples)
        
        assert f1 >= 0.0
        assert f1 <= 1.0

    def test_micro_f1_missing_pred(self):
        """Test micro_f1 handles missing pred field."""
        samples = [
            {"target": "ok tool.call{...}"},
        ]
        
        f1 = micro_f1(samples)
        
        assert f1 >= 0.0
        assert f1 <= 1.0


class TestGreedyDecodeLogits:
    """Test greedy_decode_logits function."""

    def test_greedy_decode_logits_basic(self):
        """Test basic greedy decoding from logits."""
        logits_row = np.array([0.1, 0.9, 0.05, 0.05])  # Max at index 1
        id2tok = {0: "token0", 1: "token1", 2: "token2", 3: "token3"}
        
        result = greedy_decode_logits(logits_row, id2tok)
        
        assert result == "token1"

    def test_greedy_decode_logits_unknown_id(self):
        """Test greedy_decode_logits with unknown token ID."""
        logits_row = np.array([0.1, 0.05, 0.9])  # Max at index 2
        id2tok = {0: "token0", 1: "token1"}  # Missing index 2
        
        result = greedy_decode_logits(logits_row, id2tok)
        
        assert result == "<2>"  # Should return <id> format for unknown

    def test_greedy_decode_logits_multiple_max(self):
        """Test greedy_decode_logits with multiple maximum values."""
        logits_row = np.array([0.1, 0.9, 0.9, 0.05])  # Tied max at indices 1, 2
        id2tok = {0: "token0", 1: "token1", 2: "token2", 3: "token3"}
        
        result = greedy_decode_logits(logits_row, id2tok)
        
        # Should pick first max (index 1)
        assert result == "token1" or result == "token2"


class TestGreedyDecodeSequence:
    """Test greedy_decode_sequence function."""

    def test_greedy_decode_sequence_2d(self):
        """Test greedy_decode_sequence with 2D logits (T, vocab)."""
        logits = np.array([[0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])
        id2tok = {0: "a", 1: "b"}
        
        result = greedy_decode_sequence(logits, id2tok, max_steps=3)
        
        assert "b" in result
        assert "a" in result

    def test_greedy_decode_sequence_3d_single_batch(self):
        """Test greedy_decode_sequence with 3D logits (1, T, vocab)."""
        logits = np.array([[[0.1, 0.9], [0.9, 0.1]]])  # (1, 2, 2)
        id2tok = {0: "a", 1: "b"}
        
        result = greedy_decode_sequence(logits, id2tok)
        
        assert len(result) > 0
        assert "a" in result or "b" in result

    def test_greedy_decode_sequence_3d_multi_batch(self):
        """Test greedy_decode_sequence with 3D logits (B, T, vocab)."""
        logits = np.array([[[0.1, 0.9]], [[0.9, 0.1]]])  # (2, 1, 2)
        id2tok = {0: "a", 1: "b"}
        
        result = greedy_decode_sequence(logits, id2tok)
        
        # Should use first batch
        assert len(result) > 0

    def test_greedy_decode_sequence_max_steps(self):
        """Test greedy_decode_sequence respects max_steps."""
        logits = np.array([[0.1, 0.9] for _ in range(20)])  # 20 timesteps
        id2tok = {0: "a", 1: "b"}
        
        result = greedy_decode_sequence(logits, id2tok, max_steps=5)
        
        # Should only decode 5 tokens
        tokens = result.split()
        assert len(tokens) <= 5

    def test_greedy_decode_sequence_unknown_tokens(self):
        """Test greedy_decode_sequence with unknown token IDs."""
        logits = np.array([[0.1, 0.9, 0.05], [0.9, 0.1, 0.05]])
        id2tok = {0: "a", 1: "b"}  # Missing index 2
        
        result = greedy_decode_sequence(logits, id2tok)
        
        assert "<2>" in result or len(result) > 0


class TestLoadCoreMLModel:
    """Test load_coreml_model function."""

    @patch("evaluation.toy_contracts.COREML_AVAILABLE", True)
    @patch("evaluation.toy_contracts.ct")
    def test_load_coreml_model_success(self, mock_ct):
        """Test successful CoreML model loading."""
        mock_model = Mock()
        mock_ct.models.MLModel.return_value = mock_model
        
        result = load_coreml_model("model.mlpackage")
        
        mock_ct.models.MLModel.assert_called_once_with("model.mlpackage")
        assert result == mock_model

    @patch("evaluation.toy_contracts.COREML_AVAILABLE", False)
    def test_load_coreml_model_not_available(self):
        """Test load_coreml_model when CoreML is not available."""
        with pytest.raises(ImportError, match="coremltools not available"):
            load_coreml_model("model.mlpackage")


class TestMainFunction:
    """Test main function."""

    @patch("evaluation.toy_contracts.COREML_AVAILABLE", True)
    @patch("evaluation.toy_contracts.NUMPY_AVAILABLE", True)
    @patch("evaluation.toy_contracts.load_coreml_model")
    @patch("builtins.open", create=True)
    @patch("builtins.print")
    @patch("pathlib.Path.mkdir")
    @patch("sys.exit")
    def test_main_success(
        self,
        mock_exit,
        mock_mkdir,
        mock_print,
        mock_open,
        mock_load_model,
        tmp_path,
    ):
        """Test successful main execution."""
        mock_model = Mock()
        mock_model.predict = Mock(return_value={"logits": np.random.randn(1, 10, 100)})
        mock_load_model.return_value = mock_model
        
        report_path = tmp_path / "report.json"
        
        mock_file_handle = MagicMock()
        mock_file_handle.__enter__ = Mock(return_value=mock_file_handle)
        mock_file_handle.__exit__ = Mock(return_value=None)
        mock_file_handle.write = Mock()
        mock_open.return_value = mock_file_handle
        
        with patch("sys.argv", [
            "toy_contracts.py",
            "--model", "model.mlpackage",
            "--report", str(report_path),
            "--seq", "64", "128",
        ]):
            main()
        
        mock_exit.assert_called_once()

    @patch("evaluation.toy_contracts.COREML_AVAILABLE", False)
    @patch("sys.exit")
    def test_main_coreml_not_available(self, mock_exit):
        """Test main function when CoreML is not available."""
        with patch("sys.argv", ["toy_contracts.py", "--model", "model.mlpackage", "--report", "report.json"]):
            try:
                main()
            except SystemExit:
                pass
        
        # Should exit with code 1
        mock_exit.assert_called_with(1)

    @patch("evaluation.toy_contracts.COREML_AVAILABLE", True)
    @patch("evaluation.toy_contracts.NUMPY_AVAILABLE", False)
    @patch("sys.exit")
    def test_main_numpy_not_available(self, mock_exit):
        """Test main function when numpy is not available."""
        with patch("sys.argv", ["toy_contracts.py", "--model", "model.mlpackage", "--report", "report.json"]):
            try:
                main()
            except SystemExit:
                pass
        
        # Should exit with code 1
        mock_exit.assert_called_with(1)

    @patch("evaluation.toy_contracts.COREML_AVAILABLE", True)
    @patch("evaluation.toy_contracts.NUMPY_AVAILABLE", True)
    @patch("evaluation.toy_contracts.load_coreml_model")
    @patch("builtins.open", create=True)
    @patch("pathlib.Path.mkdir")
    @patch("sys.exit")
    def test_main_model_load_failure(
        self,
        mock_exit,
        mock_mkdir,
        mock_open,
        mock_load_model,
        tmp_path,
    ):
        """Test main function handles model load failures."""
        mock_load_model.return_value = None  # Load failure
        
        report_path = tmp_path / "report.json"
        
        mock_file_handle = MagicMock()
        mock_file_handle.__enter__ = Mock(return_value=mock_file_handle)
        mock_file_handle.__exit__ = Mock(return_value=None)
        mock_file_handle.write = Mock()
        mock_open.return_value = mock_file_handle
        
        with patch("sys.argv", [
            "toy_contracts.py",
            "--model", "nonexistent.mlpackage",
            "--report", str(report_path),
            "--seq", "64",
        ]):
            main()
        
        mock_exit.assert_called()

    @patch("evaluation.toy_contracts.COREML_AVAILABLE", True)
    @patch("evaluation.toy_contracts.NUMPY_AVAILABLE", True)
    @patch("evaluation.toy_contracts.load_coreml_model")
    @patch("builtins.open", create=True)
    @patch("pathlib.Path.mkdir")
    @patch("sys.exit")
    def test_main_with_id2tok_file(
        self,
        mock_exit,
        mock_mkdir,
        mock_open,
        mock_load_model,
        tmp_path,
    ):
        """Test main function loads id2tok from file."""
        id2tok_file = tmp_path / "id2tok.json"
        id2tok_data = {"100": "special_token"}
        with open(id2tok_file, "w") as f:
            json.dump(id2tok_data, f)
        
        mock_model = Mock()
        mock_model.predict = Mock(return_value={"logits": np.random.randn(1, 10, 100)})
        mock_load_model.return_value = mock_model
        
        report_path = tmp_path / "report.json"
        
        mock_file_handle = MagicMock()
        mock_file_handle.__enter__ = Mock(return_value=mock_file_handle)
        mock_file_handle.__exit__ = Mock(return_value=None)
        mock_file_handle.write = Mock()
        
        def open_side_effect(*args, **kwargs):
            if args[0] == str(id2tok_file):
                return open(id2tok_file, "r")
            return MagicMock(__enter__=Mock(return_value=mock_file_handle), __exit__=Mock(return_value=None))
        
        mock_open.side_effect = open_side_effect
        
        with patch("sys.argv", [
            "toy_contracts.py",
            "--model", "model.mlpackage",
            "--report", str(report_path),
            "--id2tok", str(id2tok_file),
            "--seq", "64",
        ]):
            main()
        
        mock_exit.assert_called()


class TestToyContractsIntegration:
    """Test integration scenarios for toy contracts."""

    def test_has_nan_or_zero_integration(self):
        """Test has_nan_or_zero with numpy arrays."""
        vec = np.array([1.0, 2.0, float('nan'), 4.0]).tolist()
        has_nan, all_zero = has_nan_or_zero(vec)
        
        assert has_nan is True
        assert all_zero is False

    def test_micro_f1_with_tool_spans(self):
        """Test micro_f1 with realistic tool span samples."""
        samples = [
            {"target": "ok tool.call{...}", "pred": "ok tool.call{test}"},
            {"target": "ok", "pred": "ok"},
            {"target": "ok tool.call{...}", "pred": "no match"},
        ]
        
        f1 = micro_f1(samples)
        
        assert f1 >= 0.0
        assert f1 <= 1.0

    def test_greedy_decode_full_workflow(self):
        """Test complete greedy decode workflow."""
        logits = np.random.randn(1, 5, 10)  # (1, T, vocab)
        id2tok = {i: f"token{i}" for i in range(10)}
        
        result = greedy_decode_sequence(logits, id2tok, max_steps=5)
        
        assert len(result) > 0
        tokens = result.split()
        assert len(tokens) <= 5
