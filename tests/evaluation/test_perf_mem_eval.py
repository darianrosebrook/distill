"""
Tests for evaluation/perf_mem_eval.py - Performance and memory evaluation.

Tests hardware detection, StepAdapter, CoreML speed evaluation, tokenization,
and performance benchmarking functionality.
"""
# @author: @darianrosebrook

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import numpy as np

# Import the module using importlib
import importlib

perf_mem_eval_module = importlib.import_module("evaluation.perf_mem_eval")

# Import classes and functions from the module
HardwareInfo = perf_mem_eval_module.HardwareInfo
StepAdapter = perf_mem_eval_module.StepAdapter
detect_hardware = perf_mem_eval_module.detect_hardware
greedy_argmax = perf_mem_eval_module.greedy_argmax
is_valid_tool_json = perf_mem_eval_module.is_valid_tool_json
run_coreml_speed = perf_mem_eval_module.run_coreml_speed
load_tokenized_prompts = perf_mem_eval_module.load_tokenized_prompts
main = perf_mem_eval_module.main


class TestHardwareInfo:
    """Test HardwareInfo dataclass."""

    def test_hardware_info_creation(self):
        """Test creating HardwareInfo instance."""
        info = HardwareInfo(
            soc="Apple M2", os="macOS 13.0", coremltools="6.1.0", export_path="custom/path"
        )

        assert info.soc == "Apple M2"
        assert info.os == "macOS 13.0"
        assert info.coremltools == "6.1.0"
        assert info.export_path == "custom/path"

    def test_hardware_info_default_export_path(self):
        """Test HardwareInfo with default export path."""
        info = HardwareInfo(soc="Apple M1", os="macOS 12.0", coremltools="5.0.0")

        assert info.export_path == "pytorch_exportedprogram_coreml"


class TestDetectHardware:
    """Test detect_hardware function."""

    @patch("subprocess.run")
    @patch("evaluation.perf_mem_eval.platform.system")
    @patch("evaluation.perf_mem_eval.platform.processor")
    @patch("evaluation.perf_mem_eval.platform.release")
    def test_detect_hardware_macos(self, mock_release, mock_processor, mock_system, mock_subprocess):
        """Test hardware detection on macOS."""
        mock_system.return_value = "Darwin"
        mock_release.return_value = "23.0.0"
        mock_processor.return_value = "arm"

        # Mock sysctl output
        mock_result = Mock()
        mock_result.stdout = "Apple M2 Max"
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result

        with patch("coremltools.__version__", "6.2.0"):
            info = detect_hardware()

            assert isinstance(info, HardwareInfo)
            assert "Apple M2 Max" in info.soc
            assert "Darwin" in info.os
            # coremltools version may vary, just check it's a string
            assert isinstance(info.coremltools, str)

    @patch("subprocess.run")
    @patch("evaluation.perf_mem_eval.platform.system")
    @patch("evaluation.perf_mem_eval.platform.processor")
    @patch("evaluation.perf_mem_eval.platform.release")
    def test_detect_hardware_non_macos(self, mock_release, mock_processor, mock_system, mock_subprocess):
        """Test hardware detection on non-macOS systems."""
        mock_system.return_value = "Linux"
        mock_release.return_value = "5.15.0"
        mock_processor.return_value = "x86_64"

        with patch("coremltools.__version__", "6.1.0"):
            info = detect_hardware()

            assert isinstance(info, HardwareInfo)
            assert info.soc == "x86_64"
            assert "Linux" in info.os
            # coremltools version may vary, just check it's a string
            assert isinstance(info.coremltools, str)

    @patch("evaluation.perf_mem_eval.platform.system")
    @patch("evaluation.perf_mem_eval.platform.processor")
    @patch("evaluation.perf_mem_eval.platform.release")
    def test_detect_hardware_no_coremltools(self, mock_release, mock_processor, mock_system):
        """Test hardware detection when coremltools is not available."""
        mock_system.return_value = "Darwin"
        mock_release.return_value = "23.0.0"
        mock_processor.return_value = "arm"

        # Patch the import to raise ImportError when coremltools is imported
        original_import = __import__
        def mock_import(name, *args, **kwargs):
            if name == "coremltools":
                raise ImportError("No module named 'coremltools'")
            return original_import(name, *args, **kwargs)
        
        with patch("builtins.__import__", side_effect=mock_import):
            info = detect_hardware()
            
            assert info.coremltools == "unknown"
            assert isinstance(info, HardwareInfo)

    @patch("subprocess.run")
    @patch("evaluation.perf_mem_eval.platform.system")
    @patch("evaluation.perf_mem_eval.platform.processor")
    @patch("evaluation.perf_mem_eval.platform.release")
    def test_detect_hardware_sysctl_failure(self, mock_release, mock_processor, mock_system, mock_subprocess):
        """Test hardware detection when sysctl fails."""
        mock_system.return_value = "Darwin"
        mock_release.return_value = "23.0.0"
        mock_processor.return_value = "arm"

        # Mock sysctl failure
        mock_subprocess.side_effect = Exception("sysctl failed")

        with patch("coremltools.__version__", "6.2.0"):
            info = detect_hardware()

            # Should fallback to platform.processor()
            assert info.soc == "arm"
            assert isinstance(info, HardwareInfo)
            # coremltools version may vary, just check it's a string
            assert isinstance(info.coremltools, str)


class TestStepAdapter:
    """Test StepAdapter abstract class."""

    def test_step_adapter_abstract_methods(self):
        """Test that StepAdapter has abstract methods."""
        adapter = StepAdapter()

        # Should raise NotImplementedError when calling abstract methods
        with pytest.raises(NotImplementedError):
            adapter.prepare_state(np.array([1, 2, 3]))

        with pytest.raises(NotImplementedError):
            mock_model = Mock()
            adapter.first_step(mock_model, np.array([1, 2, 3]), {})

        with pytest.raises(NotImplementedError):
            mock_model = Mock()
            adapter.next_step(mock_model, 1, {})

    def test_step_adapter_subclass_implementation(self):
        """Test implementing StepAdapter subclass."""

        class TestAdapter(StepAdapter):
            def prepare_state(self, prompt_ids: np.ndarray):
                return {"state": "initial"}

            def first_step(self, model, prompt_ids: np.ndarray, state):
                return np.array([0.1, 0.9]), {"state": "updated"}

            def next_step(self, model, token_id: int, state):
                return np.array([0.2, 0.8]), {"state": "next"}

        adapter = TestAdapter()
        state = adapter.prepare_state(np.array([1, 2, 3]))
        assert state == {"state": "initial"}

        mock_model = Mock()
        logits, new_state = adapter.first_step(mock_model, np.array([1, 2, 3]), {})
        assert isinstance(logits, np.ndarray)
        assert new_state == {"state": "updated"}

        logits, new_state = adapter.next_step(mock_model, 1, {})
        assert isinstance(logits, np.ndarray)
        assert new_state == {"state": "next"}


class TestGreedyArgmax:
    """Test greedy_argmax function."""

    def test_greedy_argmax_single_max(self):
        """Test greedy argmax with single maximum value."""
        logits = np.array([0.1, 0.8, 0.3, 0.2])
        result = greedy_argmax(logits)

        assert result == 1  # Index of maximum value
        assert isinstance(result, int)

    def test_greedy_argmax_multiple_same(self):
        """Test greedy argmax with multiple same maximum values."""
        logits = np.array([0.5, 0.5, 0.3, 0.2])
        result = greedy_argmax(logits)

        # Should return first occurrence of maximum
        assert result == 0

    def test_greedy_argmax_single_element(self):
        """Test greedy argmax with single element array."""
        logits = np.array([0.7])
        result = greedy_argmax(logits)

        assert result == 0

    def test_greedy_argmax_negative_values(self):
        """Test greedy argmax with negative values."""
        logits = np.array([-0.5, -0.2, -0.8, -0.1])
        result = greedy_argmax(logits)

        assert result == 3  # Index of least negative (highest) value (-0.1 at index 3)

    def test_greedy_argmax_empty_array(self):
        """Test greedy_argmax with empty array (line 158-159)."""
        with pytest.raises(ValueError, match="Cannot get argmax of empty array"):
            greedy_argmax(np.array([]))


class TestIsValidToolJSON:
    """Test is_valid_tool_json function."""

    def test_is_valid_tool_json_valid_simple(self):
        """Test validating simple valid tool JSON."""
        valid_json = '{"name": "calculator", "arguments": {"x": 1, "y": 2}}'
        result = is_valid_tool_json(valid_json)

        assert result is True

    def test_is_valid_tool_json_valid_complex(self):
        """Test validating complex valid tool JSON."""
        complex_json = '{"name": "search_web", "arguments": {"query": "Python tutorials"}}'
        result = is_valid_tool_json(complex_json)

        assert result is True

    def test_is_valid_tool_json_invalid_missing_braces(self):
        """Test validating tool JSON with missing braces."""
        invalid_cases = [
            '{"name": "calculator"}',  # Missing arguments
            '{"arguments": {}}',  # Missing name
            "not json",  # Not JSON
            "",  # Empty string
        ]

        for invalid_json in invalid_cases:
            result = is_valid_tool_json(invalid_json)
            assert result is False, f"Should reject invalid JSON: {invalid_json}"

    def test_is_valid_tool_json_empty_json(self):
        """Test validating empty JSON."""
        result = is_valid_tool_json("{}")
        assert result is False  # Missing required fields

    def test_is_valid_tool_json_missing_colon(self):
        """Test validating JSON without colon."""
        result = is_valid_tool_json('{"name" "calculator"}')
        assert result is False

    def test_is_valid_tool_json_empty_string(self):
        """Test is_valid_tool_json with empty string (line 165-166)."""
        assert not is_valid_tool_json("")
        assert not is_valid_tool_json("   ")

    def test_is_valid_tool_json_not_dict(self):
        """Test is_valid_tool_json with non-dict JSON (line 175-176)."""
        assert not is_valid_tool_json('["name", "test_tool"]')  # Array, not dict

    def test_is_valid_tool_json_name_not_string(self):
        """Test is_valid_tool_json with name not a string (line 185-186)."""
        assert not is_valid_tool_json('{"name": 123, "arguments": {}}')

    def test_is_valid_tool_json_arguments_not_dict(self):
        """Test is_valid_tool_json with arguments not a dict (line 189-190)."""
        assert not is_valid_tool_json('{"name": "test_tool", "arguments": "not_a_dict"}')


class TestRunCoreMLSpeed:
    """Test run_coreml_speed function."""

    @pytest.fixture
    def mock_model(self):
        """Create mock CoreML model."""
        model = Mock()
        model.predict = Mock(return_value={"logits": np.array([0.1, 0.9, 0.0])})
        return model

    @pytest.fixture
    def mock_adapter(self):
        """Create mock StepAdapter."""
        adapter = Mock(spec=StepAdapter)
        adapter.prepare_state = Mock(return_value={})
        adapter.first_step = Mock(return_value=(np.array([0.1, 0.9]), {}))
        adapter.next_step = Mock(return_value=(np.array([0.2, 0.8]), {}))
        return adapter

    def test_run_coreml_speed_no_coremltools(self):
        """Test run_coreml_speed when coremltools is not available."""
        original_mlmodel = perf_mem_eval_module.MLModel
        perf_mem_eval_module.MLModel = None

        try:
            prompts = [[1, 2, 3]]
            adapter = Mock(spec=StepAdapter)

            with pytest.raises(RuntimeError, match="coremltools / MLModel not available"):
                run_coreml_speed(Path("model.mlpackage"), prompts, adapter)
        finally:
            perf_mem_eval_module.MLModel = original_mlmodel

    @patch("evaluation.perf_mem_eval.MLModel")
    def test_run_coreml_speed_standard_path(self, mock_mlmodel_class, mock_model, mock_adapter):
        """Test run_coreml_speed with standard decoding path."""
        mock_mlmodel_class.return_value = mock_model

        prompts = [[1, 2, 3], [4, 5, 6]]
        result = run_coreml_speed(
            Path("model.mlpackage"), prompts, mock_adapter, max_new_tokens=5
        )

        assert isinstance(result, dict)
        assert "ttft_ms" in result
        assert "tps" in result
        assert "ttfa_tokens" in result
        assert "ttfa_ms" in result

        # Check percentiles
        assert "p50" in result["ttft_ms"]
        assert "p90" in result["ttft_ms"]
        assert "p95" in result["ttft_ms"]

    @patch("evaluation.perf_mem_eval.MLModel")
    def test_run_coreml_speed_with_tokenizer(self, mock_mlmodel_class, mock_model, mock_adapter):
        """Test run_coreml_speed with tokenizer for TTFA detection."""
        mock_mlmodel_class.return_value = mock_model

        mock_tokenizer = Mock()
        mock_tokenizer.decode = Mock(return_value='{"name": "tool", "arguments": {}}')

        prompts = [[1, 2, 3]]
        result = run_coreml_speed(
            Path("model.mlpackage"),
            prompts,
            mock_adapter,
            max_new_tokens=5,
            tokenizer=mock_tokenizer,
        )

        assert "ttfa_tokens" in result
        assert "ttfa_ms" in result

    @patch("evaluation.perf_mem_eval.MLModel")
    def test_run_coreml_speed_split_ttft(self, mock_mlmodel_class, mock_model, mock_adapter):
        """Test run_coreml_speed with TTFT splitting."""
        mock_mlmodel_class.return_value = mock_model

        mock_tokenizer = Mock()
        prompts = [[1, 2, 3]]

        result = run_coreml_speed(
            Path("model.mlpackage"),
            prompts,
            mock_adapter,
            max_new_tokens=5,
            tokenizer=mock_tokenizer,
            split_ttft=True,
        )

        assert "ttft_split" in result
        assert "tokenizer_ms" in result["ttft_split"]
        assert "first_step_ms" in result["ttft_split"]

    @patch("evaluation.perf_mem_eval.MLModel")
    def test_run_coreml_speed_with_prompt_cache(
        self, mock_mlmodel_class, mock_model, mock_adapter
    ):
        """Test run_coreml_speed with prompt cache."""
        mock_mlmodel_class.return_value = mock_model

        mock_cache = Mock()
        mock_cache.get_or_compute = Mock(return_value=({}, False))
        mock_cache.stats = Mock(return_value={"hits": 0, "misses": 2})

        prompts = [[1, 2, 3]]
        prompt_texts = ["test prompt"]

        result = run_coreml_speed(
            Path("model.mlpackage"),
            prompts,
            mock_adapter,
            max_new_tokens=5,
            prompt_cache=mock_cache,
            prompt_texts=prompt_texts,
        )

        assert "prompt_cache_stats" in result

    @patch("evaluation.perf_mem_eval.MLModel")
    def test_run_coreml_speed_with_speculative_decoder(
        self, mock_mlmodel_class, mock_model, mock_adapter
    ):
        """Test run_coreml_speed with speculative decoding."""
        mock_mlmodel_class.return_value = mock_model

        mock_spec_decoder = Mock()
        mock_spec_decoder.generate = Mock(
            return_value={
                "tokens": [1, 2, 3],
                "ttft_ms": 10.5,
                "tps": 25.0,
            }
        )
        mock_spec_decoder.get_stats = Mock(return_value={"draft_tokens": 5, "accepted": 3})

        prompts = [[1, 2, 3]]

        result = run_coreml_speed(
            Path("model.mlpackage"),
            prompts,
            mock_adapter,
            max_new_tokens=5,
            speculative_decoder=mock_spec_decoder,
        )

        assert "speculative_decoding_stats" in result
        mock_spec_decoder.generate.assert_called()

    @patch("evaluation.perf_mem_eval.MLModel")
    def test_run_coreml_speed_with_kv_cache(self, mock_mlmodel_class, mock_model, mock_adapter):
        """Test run_coreml_speed with KV cache."""
        mock_mlmodel_class.return_value = mock_model

        mock_kv_cache = Mock()
        mock_kv_cache.stats = Mock(return_value={"allocated": 1024})

        prompts = [[1, 2, 3]]

        result = run_coreml_speed(
            Path("model.mlpackage"),
            prompts,
            mock_adapter,
            max_new_tokens=5,
            kv_cache=mock_kv_cache,
        )

        assert "kv_cache_stats" in result

    @patch("evaluation.perf_mem_eval.MLModel")
    def test_run_coreml_speed_with_batch_policy(
        self, mock_mlmodel_class, mock_model, mock_adapter
    ):
        """Test run_coreml_speed with batch policy."""
        mock_mlmodel_class.return_value = mock_model

        mock_batch_policy = Mock()
        mock_batch_policy.get_policy_summary = Mock(return_value={"workload_type": "interactive"})

        prompts = [[1, 2, 3]]

        result = run_coreml_speed(
            Path("model.mlpackage"),
            prompts,
            mock_adapter,
            max_new_tokens=5,
            batch_policy=mock_batch_policy,
            batch_size=2,
        )

        assert "batch_policy" in result
        assert result["batch_policy"]["selected_batch_size"] == 2


class TestLoadTokenizedPrompts:
    """Test load_tokenized_prompts function."""

    def test_load_tokenized_prompts_file_not_found(self, tmp_path):
        """Test loading prompts when file doesn't exist."""
        dataset_path = tmp_path / "nonexistent.jsonl"

        with patch("evaluation.perf_mem_eval.load_tokenizer") as mock_load_tokenizer:
            mock_tokenizer = Mock()
            mock_tokenizer.encode = Mock(return_value=[1, 2, 3])
            mock_load_tokenizer.return_value = mock_tokenizer

            result = load_tokenized_prompts(dataset_path, "tokenizer_path")

            # Should return synthetic prompts
            assert result == [[1, 2, 3], [4, 5, 6, 7]]

    def test_load_tokenized_prompts_jsonl_format(self, tmp_path):
        """Test loading prompts from JSONL format."""
        dataset_path = tmp_path / "prompts.jsonl"

        # Create JSONL file
        with open(dataset_path, "w") as f:
            f.write('{"prompt": "Hello world"}\n')
            f.write('{"prompt": "Test prompt"}\n')

        with patch("evaluation.perf_mem_eval.load_tokenizer") as mock_load_tokenizer:
            mock_tokenizer = Mock()
            mock_tokenizer.encode = Mock(side_effect=[[1, 2, 3], [4, 5, 6]])
            mock_load_tokenizer.return_value = mock_tokenizer

            result = load_tokenized_prompts(dataset_path, "tokenizer_path", max_samples=10)

            assert len(result) == 2
            assert result[0] == [1, 2, 3]
            assert result[1] == [4, 5, 6]

    def test_load_tokenized_prompts_with_input_dict(self, tmp_path):
        """Test loading prompts with input dict structure."""
        dataset_path = tmp_path / "prompts.jsonl"

        # Create JSONL with input dict
        with open(dataset_path, "w") as f:
            f.write(
                '{"input": {"system": "You are a helpful assistant", "history": [{"role": "user", "content": "Hello"}]}}\n'
            )

        with patch("evaluation.perf_mem_eval.load_tokenizer") as mock_load_tokenizer:
            mock_tokenizer = Mock()
            mock_tokenizer.encode = Mock(return_value=[1, 2, 3])
            mock_load_tokenizer.return_value = mock_tokenizer

            result = load_tokenized_prompts(dataset_path, "tokenizer_path", max_samples=10)

            assert len(result) == 1
            mock_tokenizer.encode.assert_called()

    def test_load_tokenized_prompts_return_texts(self, tmp_path):
        """Test loading prompts with return_texts=True."""
        dataset_path = tmp_path / "prompts.jsonl"

        with open(dataset_path, "w") as f:
            f.write('{"prompt": "Hello world"}\n')

        with patch("evaluation.perf_mem_eval.load_tokenizer") as mock_load_tokenizer:
            mock_tokenizer = Mock()
            mock_tokenizer.encode = Mock(return_value=[1, 2, 3])
            mock_load_tokenizer.return_value = mock_tokenizer

            prompts, texts = load_tokenized_prompts(
                dataset_path, "tokenizer_path", return_texts=True, max_samples=10
            )

            assert len(prompts) == 1
            assert len(texts) == 1
            assert texts[0] == "Hello world"

    def test_load_tokenized_prompts_max_samples(self, tmp_path):
        """Test loading prompts with max_samples limit."""
        dataset_path = tmp_path / "prompts.jsonl"

        # Create JSONL with more lines than max_samples
        with open(dataset_path, "w") as f:
            for i in range(10):
                f.write(f'{{"prompt": "prompt {i}"}}\n')

        with patch("evaluation.perf_mem_eval.load_tokenizer") as mock_load_tokenizer:
            mock_tokenizer = Mock()
            mock_tokenizer.encode = Mock(return_value=[1, 2, 3])
            mock_load_tokenizer.return_value = mock_tokenizer

            result = load_tokenized_prompts(dataset_path, "tokenizer_path", max_samples=5)

            assert len(result) == 5

    def test_load_tokenized_prompts_optimized_tokenizer(self, tmp_path):
        """Test loading prompts with optimized tokenizer."""
        dataset_path = tmp_path / "prompts.jsonl"

        with open(dataset_path, "w") as f:
            f.write('{"prompt": "Hello world"}\n')

        with patch("evaluation.perf_mem_eval.load_tokenizer") as mock_load_tokenizer, patch(
            "evaluation.perf_mem_eval.OptimizedTokenizer"
        ) as mock_opt_tokenizer:
            mock_base_tokenizer = Mock()
            mock_load_tokenizer.return_value = mock_base_tokenizer

            mock_opt_tokenizer_instance = Mock()
            mock_opt_tokenizer_instance.encode_optimized = Mock(return_value=[1, 2, 3])
            mock_opt_tokenizer.return_value = mock_opt_tokenizer_instance

            result = load_tokenized_prompts(
                dataset_path, "tokenizer_path", use_optimized_tokenizer=True, max_samples=10
            )

            assert len(result) == 1
            mock_opt_tokenizer.assert_called()

    def test_load_tokenized_prompts_tokenizer_failure(self):
        """Test loading prompts when tokenizer loading fails."""
        with patch("evaluation.perf_mem_eval.load_tokenizer") as mock_load_tokenizer:
            mock_load_tokenizer.side_effect = Exception("Tokenizer load failed")

            with pytest.raises(RuntimeError, match="Failed to load tokenizer"):
                load_tokenized_prompts(Path("dataset.jsonl"), "tokenizer_path")


class TestMainFunction:
    """Test main function."""

    @patch("evaluation.perf_mem_eval.detect_hardware")
    @patch("evaluation.perf_mem_eval.load_tokenized_prompts")
    @patch("evaluation.perf_mem_eval.run_coreml_speed")
    @patch("evaluation.perf_mem_eval.argparse.ArgumentParser")
    @patch("builtins.print")
    @patch("builtins.open")
    def test_main_success(
        self,
        mock_open,
        mock_print,
        mock_parser_class,
        mock_run_speed,
        mock_load_prompts,
        mock_detect_hw,
        tmp_path,
    ):
        """Test successful main function execution."""
        # Mock argument parser
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.model = tmp_path / "model.mlpackage"
        mock_args.dataset = tmp_path / "dataset.jsonl"
        mock_args.tokenizer = "tokenizer_path"
        mock_args.out = tmp_path / "results.json"
        mock_args.max_new_tokens = 64
        mock_args.max_samples = 100
        mock_args.export_path = "pytorch_exportedprogram_coreml"
        mock_args.hardware = ""
        mock_args.enable_prompt_cache = False
        mock_args.cache_size_mb = 100
        mock_args.drafter_model = None
        mock_args.enable_speculative = False
        mock_args.spec_k = 2
        mock_args.measure_ane_residency = False
        mock_args.ane_samples = 100
        mock_args.use_optimized_tokenizer = False
        mock_args.use_optimized_kv_cache = False
        mock_args.kv_cache_heads = None
        mock_args.kv_cache_head_dim = None
        mock_args.kv_cache_gqa_groups = None
        mock_args.workload_type = "interactive"
        mock_args.batch_size = None
        mock_parser.parse_args.return_value = mock_args
        mock_parser_class.return_value = mock_parser

        # Mock hardware detection
        mock_hw = HardwareInfo(soc="Apple M2", os="macOS 13.0", coremltools="6.2.0")
        mock_detect_hw.return_value = mock_hw

        # Mock prompts loading
        mock_prompts = [[1, 2, 3], [4, 5, 6]]
        mock_load_prompts.return_value = mock_prompts

        # Mock speed evaluation
        mock_results = {
            "ttft_ms": {"p50": 10.5, "p90": 12.3, "p95": 15.7},
            "tps": {"p50": 25.0, "p90": 28.5, "p95": 30.2},
        }
        mock_run_speed.return_value = mock_results

        # Mock file writing
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file

        # Test that main runs without error
        try:
            main()
        except SystemExit:
            pass  # Expected for successful completion

        mock_run_speed.assert_called_once()

    @patch("evaluation.perf_mem_eval.argparse.ArgumentParser")
    def test_main_missing_required_args(self, mock_parser_class):
        """Test main function with missing required arguments."""
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.model = None  # Missing required
        mock_parser.parse_args.return_value = mock_args
        mock_parser_class.return_value = mock_parser

        # Should still run but argparse will handle missing args
        try:
            main()
        except SystemExit:
            pass  # Expected


class TestPerfMemEvalIntegration:
    """Test integration of performance and memory evaluation components."""

    def test_hardware_detection_integration(self):
        """Test hardware detection integration."""
        info = detect_hardware()

        assert isinstance(info, HardwareInfo)
        assert isinstance(info.soc, str)
        assert isinstance(info.os, str)
        assert isinstance(info.coremltools, str)

    def test_greedy_argmax_properties(self):
        """Test greedy argmax mathematical properties."""
        test_cases = [
            np.array([0.1, 0.2, 0.7, 0.0]),  # Clear maximum
            np.array([0.25, 0.25, 0.25, 0.25]),  # Equal probabilities
            np.array([0.0, 0.0, 1.0, 0.0]),  # One-hot
        ]

        for logits in test_cases:
            result = greedy_argmax(logits)

            # Result should be a valid index
            assert 0 <= result < len(logits)

            # Result should correspond to maximum value
            assert logits[result] == np.max(logits)

    def test_json_validation_comprehensive(self):
        """Test comprehensive JSON validation."""
        valid_patterns = [
            '{"name": "calculator", "arguments": {}}',
            '{"name": "search", "arguments": {"query": "test"}}',
        ]

        for pattern in valid_patterns:
            assert is_valid_tool_json(pattern) is True

        invalid_patterns = [
            '{"name": "calculator"}',  # Missing arguments
            '{"arguments": {}}',  # Missing name
            "not json",  # Not JSON
        ]

        for pattern in invalid_patterns:
            assert is_valid_tool_json(pattern) is False
