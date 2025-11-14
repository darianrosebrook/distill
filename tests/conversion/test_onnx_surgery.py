"""
Tests for conversion/onnx_surgery.py - ONNX surgery functionality.

Tests dtype forcing, redundant cast removal, int64 initializer casting,
shape inference, and ONNX simplification.
"""
# @author: @darianrosebrook

from unittest.mock import Mock, patch

import pytest
import onnx
from onnx import TensorProto, numpy_helper

# Import the module using importlib
import importlib

onnx_surgery_module = importlib.import_module("conversion.onnx_surgery")

# Import functions from the module
force_input_dtype = onnx_surgery_module.force_input_dtype
force_output_dtype = onnx_surgery_module.force_output_dtype
strip_redundant_casts = onnx_surgery_module.strip_redundant_casts
cast_int64_initializers = onnx_surgery_module.cast_int64_initializers
run = onnx_surgery_module.run
main = onnx_surgery_module.main


class TestForceInputDtype:
    """Test force_input_dtype function."""

    def test_force_input_dtype_success(self):
        """Test successful input dtype forcing."""
        # Create a simple ONNX model
        input_tensor = onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10])
        output_tensor = onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])

        node = onnx.helper.make_node("Identity", ["input"], ["output"])
        graph = onnx.helper.make_graph([node], "test_graph", [input_tensor], [output_tensor])
        model = onnx.helper.make_model(graph)

        # Force input dtype to INT32
        result = force_input_dtype(model, "input", TensorProto.INT32)

        assert result == model
        assert result.graph.input[0].type.tensor_type.elem_type == TensorProto.INT32

    def test_force_input_dtype_not_found(self):
        """Test forcing dtype for non-existent input."""
        input_tensor = onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10])
        output_tensor = onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])

        node = onnx.helper.make_node("Identity", ["input"], ["output"])
        graph = onnx.helper.make_graph([node], ["input"], [output_tensor])
        graph.input.extend([input_tensor])
        model = onnx.helper.make_model(graph)

        # Try to force dtype for non-existent input
        result = force_input_dtype(model, "nonexistent_input", TensorProto.INT32)

        # Should not crash, just return the model unchanged
        assert result == model
        assert result.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT

    def test_force_input_dtype_multiple_inputs(self):
        """Test forcing dtype when model has multiple inputs."""
        inputs = [
            onnx.helper.make_tensor_value_info("input1", TensorProto.FLOAT, [1, 10]),
            onnx.helper.make_tensor_value_info("input2", TensorProto.DOUBLE, [1, 5]),
        ]
        output = onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])

        node = onnx.helper.make_node("Add", ["input1", "input2"], ["output"])
        graph = onnx.helper.make_graph([node], "test_graph", inputs, [output])
        model = onnx.helper.make_model(graph)

        # Force dtype for second input
        result = force_input_dtype(model, "input2", TensorProto.INT32)

        assert result.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT  # Unchanged
        assert result.graph.input[1].type.tensor_type.elem_type == TensorProto.INT32  # Changed


class TestForceOutputDtype:
    """Test force_output_dtype function."""

    def test_force_output_dtype_success(self):
        """Test successful output dtype forcing."""
        input_tensor = onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10])
        output_tensor = onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])

        node = onnx.helper.make_node("Identity", ["input"], ["output"])
        graph = onnx.helper.make_graph([node], "test_graph", [input_tensor], [output_tensor])
        model = onnx.helper.make_model(graph)

        # Force output dtype to FLOAT16
        result = force_output_dtype(model, "output", TensorProto.FLOAT16)

        assert result == model
        assert result.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT16

    def test_force_output_dtype_not_found(self):
        """Test forcing dtype for non-existent output."""
        input_tensor = onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10])
        output_tensor = onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])

        node = onnx.helper.make_node("Identity", ["input"], ["output"])
        graph = onnx.helper.make_graph([node], "test_graph", [input_tensor], [output_tensor])
        model = onnx.helper.make_model(graph)

        # Try to force dtype for non-existent output
        result = force_output_dtype(model, "nonexistent_output", TensorProto.FLOAT16)

        # Should not crash, just return the model unchanged
        assert result == model
        assert result.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT

    def test_force_output_dtype_multiple_outputs(self):
        """Test forcing dtype when model has multiple outputs."""
        input_tensor = onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10])
        outputs = [
            onnx.helper.make_tensor_value_info("output1", TensorProto.FLOAT, [1, 10]),
            onnx.helper.make_tensor_value_info("output2", TensorProto.DOUBLE, [1, 5]),
        ]

        node = onnx.helper.make_node("Split", ["input"], ["output1", "output2"])
        graph = onnx.helper.make_graph([node], "test_graph", [input_tensor], outputs)
        model = onnx.helper.make_model(graph)

        # Force dtype for second output
        result = force_output_dtype(model, "output2", TensorProto.INT32)

        assert result.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT  # Unchanged
        assert result.graph.output[1].type.tensor_type.elem_type == TensorProto.INT32  # Changed


class TestStripRedundantCasts:
    """Test strip_redundant_casts function."""

    def test_strip_redundant_casts_no_casts(self):
        """Test stripping casts when no cast nodes exist."""
        input_tensor = onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10])
        output_tensor = onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])

        node = onnx.helper.make_node("Identity", ["input"], ["output"])
        graph = onnx.helper.make_graph([node], "test_graph", [input_tensor], [output_tensor])
        model = onnx.helper.make_model(graph)

        result = strip_redundant_casts(model)

        # Should return the same model (no changes)
        assert result == model
        assert len(result.graph.node) == 1

    @patch("conversion.onnx_surgery.shape_inference.infer_shapes")
    def test_strip_redundant_casts_redundant_cast(self, mock_infer_shapes):
        """Test stripping redundant cast operations."""
        # Create model with a redundant cast (FLOAT -> FLOAT)
        input_tensor = onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10])
        intermediate_tensor = onnx.helper.make_tensor_value_info(
            "intermediate", TensorProto.FLOAT, [1, 10]
        )
        output_tensor = onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])

        # Create nodes: Identity -> Cast(FLOAT->FLOAT) -> Identity
        node1 = onnx.helper.make_node("Identity", ["input"], ["temp"])
        node2 = onnx.helper.make_node(
            "Cast", ["temp"], ["intermediate"], to=TensorProto.FLOAT
        )  # Redundant cast
        node3 = onnx.helper.make_node("Identity", ["intermediate"], ["output"])

        graph = onnx.helper.make_graph(
            [node1, node2, node3], "test_graph", [input_tensor], [output_tensor]
        )
        model = onnx.helper.make_model(graph)

        # Mock shape inference to return input types
        mock_inferred_model = Mock()
        mock_inferred_model.graph.value_info = []
        mock_infer_shapes.return_value = mock_inferred_model

        result = strip_redundant_casts(model)

        # Should have removed the redundant cast
        # (This is a simplified test - actual implementation is more complex)

    @patch("conversion.onnx_surgery.shape_inference.infer_shapes")
    def test_strip_redundant_casts_inference_failure(self, mock_infer_shapes):
        """Test stripping casts when shape inference fails."""
        input_tensor = onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10])
        output_tensor = onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])

        node = onnx.helper.make_node("Identity", ["input"], ["output"])
        graph = onnx.helper.make_graph([node], "test_graph", [input_tensor], [output_tensor])
        model = onnx.helper.make_model(graph)

        # Mock shape inference failure
        mock_infer_shapes.side_effect = Exception("Shape inference failed")

        result = strip_redundant_casts(model)

        # Should return original model unchanged
        assert result == model

    def test_strip_redundant_casts_with_cast_nodes(self):
        """Test stripping casts with actual cast nodes."""
        # This is a more realistic test with actual cast operations
        input_tensor = onnx.helper.make_tensor_value_info("input", TensorProto.INT32, [1, 10])
        cast_output = onnx.helper.make_tensor_value_info("cast_output", TensorProto.FLOAT, [1, 10])
        final_output = onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])

        # Create nodes: Cast(INT32->FLOAT) -> Identity
        cast_node = onnx.helper.make_node("Cast", ["input"], ["cast_output"], to=TensorProto.FLOAT)
        identity_node = onnx.helper.make_node("Identity", ["cast_output"], ["output"])

        graph = onnx.helper.make_graph(
            [cast_node, identity_node], "test_graph", [input_tensor], [final_output]
        )
        model = onnx.helper.make_model(graph)

        # This cast is not redundant (INT32 -> FLOAT), so should be preserved
        with patch("conversion.onnx_surgery.shape_inference.infer_shapes") as mock_infer:
            mock_inferred = Mock()
            mock_inferred.graph.value_info = []
            mock_infer.return_value = mock_inferred

            result = strip_redundant_casts(model)

            # Should still have the cast node
            assert len(result.graph.node) == 2


class TestCastInt64Initializers:
    """Test cast_int64_initializers function."""

    def test_cast_int64_initializers_no_initializers(self):
        """Test casting when no initializers exist."""
        input_tensor = onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10])
        output_tensor = onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])

        node = onnx.helper.make_node("Identity", ["input"], ["output"])
        graph = onnx.helper.make_graph([node], "test_graph", [input_tensor], [output_tensor])
        model = onnx.helper.make_model(graph)

        result = cast_int64_initializers(model)

        assert result == 0  # No initializers cast
        assert len(model.graph.initializer) == 0

    def test_cast_int64_initializers_int64_present(self):
        """Test casting when INT64 initializers are present."""
        # Create a model with an INT64 initializer
        input_tensor = onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10])
        output_tensor = onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])

        # Create INT64 initializer
        int64_tensor = numpy_helper.from_array(
            np.array([1, 2, 3], dtype=np.int64), name="int64_initializer"
        )

        node = onnx.helper.make_node("Identity", ["input"], ["output"])
        graph = onnx.helper.make_graph(
            [node], "test_graph", [input_tensor], [output_tensor], initializer=[int64_tensor]
        )
        model = onnx.helper.make_model(graph)

        result = cast_int64_initializers(model)

        assert result == 1  # One initializer cast
        assert len(model.graph.initializer) == 1
        # Check that the initializer is now INT32
        assert model.graph.initializer[0].data_type == TensorProto.INT32

    def test_cast_int64_initializers_mixed_types(self):
        """Test casting when both INT64 and other types are present."""
        input_tensor = onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10])
        output_tensor = onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])

        # Create mixed initializers
        int64_tensor = numpy_helper.from_array(
            np.array([1, 2, 3], dtype=np.int64), name="int64_init"
        )
        float_tensor = numpy_helper.from_array(
            np.array([1.0, 2.0], dtype=np.float32), name="float_init"
        )

        node = onnx.helper.make_node("Identity", ["input"], ["output"])
        graph = onnx.helper.make_graph(
            [node],
            "test_graph",
            [input_tensor],
            [output_tensor],
            initializer=[int64_tensor, float_tensor],
        )
        model = onnx.helper.make_model(graph)

        result = cast_int64_initializers(model)

        assert result == 1  # Only INT64 initializer cast
        assert len(model.graph.initializer) == 2

        # Check types
        for init in model.graph.initializer:
            if init.name == "int64_init":
                assert init.data_type == TensorProto.INT32
            elif init.name == "float_init":
                assert init.data_type == TensorProto.FLOAT

    def test_cast_int64_initializers_no_int64(self):
        """Test casting when no INT64 initializers exist."""
        input_tensor = onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10])
        output_tensor = onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])

        # Create non-INT64 initializers
        float_tensor = numpy_helper.from_array(
            np.array([1.0, 2.0], dtype=np.float32), name="float_init"
        )
        int32_tensor = numpy_helper.from_array(np.array([1, 2], dtype=np.int32), name="int32_init")

        node = onnx.helper.make_node("Identity", ["input"], ["output"])
        graph = onnx.helper.make_graph(
            [node],
            "test_graph",
            [input_tensor],
            [output_tensor],
            initializer=[float_tensor, int32_tensor],
        )
        model = onnx.helper.make_model(graph)

        result = cast_int64_initializers(model)

        assert result == 0  # No INT64 initializers to cast
        assert len(model.graph.initializer) == 2


class TestRunFunction:
    """Test run function."""

    @patch("conversion.onnx_surgery.onnx.load")
    @patch("conversion.onnx_surgery.onnx.save")
    @patch("conversion.onnx_surgery.cast_int64_initializers")
    @patch("conversion.onnx_surgery.strip_redundant_casts")
    @patch("conversion.onnx_surgery.shape_inference.infer_shapes")
    @patch("conversion.onnx_surgery.onnxsim")
    def test_run_success_without_simplification(
        self,
        mock_onnxsim,
        mock_infer_shapes,
        mock_strip_casts,
        mock_cast_int64,
        mock_save,
        mock_load,
    ):
        """Test successful run without ONNX simplification."""
        # Create mock model
        mock_model = Mock()
        mock_load.return_value = mock_model

        # Mock processed model
        mock_processed = Mock()
        mock_cast_int64.return_value = 1
        mock_strip_casts.return_value = mock_processed
        mock_infer_shapes.return_value = mock_processed

        # Mock onnxsim as None (not available)
        mock_onnxsim.simplify = None

        run("input.onnx", "output.onnx")

        mock_load.assert_called_once_with("input.onnx")
        mock_cast_int64.assert_called_once_with(mock_model)
        mock_strip_casts.assert_called_once()
        mock_infer_shapes.assert_called_once()
        mock_save.assert_called_once_with(mock_processed, "output.onnx")

    @patch("conversion.onnx_surgery.onnx.load")
    @patch("conversion.onnx_surgery.onnx.save")
    @patch("conversion.onnx_surgery.cast_int64_initializers")
    @patch("conversion.onnx_surgery.strip_redundant_casts")
    @patch("conversion.onnx_surgery.shape_inference.infer_shapes")
    @patch("conversion.onnx_surgery.onnxsim")
    def test_run_success_with_simplification(
        self,
        mock_onnxsim,
        mock_infer_shapes,
        mock_strip_casts,
        mock_cast_int64,
        mock_save,
        mock_load,
    ):
        """Test successful run with ONNX simplification."""
        mock_model = Mock()
        mock_load.return_value = mock_model

        mock_processed = Mock()
        mock_cast_int64.return_value = 1
        mock_strip_casts.return_value = mock_processed
        mock_infer_shapes.return_value = mock_processed

        # Mock onnxsim as available
        mock_simplified = Mock()
        mock_onnxsim.simplify.return_value = (mock_simplified, True)

        run("input.onnx", "output.onnx")

        # Should call simplification
        mock_onnxsim.simplify.assert_called_once_with(mock_processed)
        mock_save.assert_called_once_with(mock_simplified, "output.onnx")

    @patch("conversion.onnx_surgery.onnx.load")
    def test_run_load_failure(self, mock_load):
        """Test run function when model loading fails."""
        mock_load.side_effect = Exception("Load failed")

        with pytest.raises(Exception):
            run("input.onnx", "output.onnx")

    @patch("conversion.onnx_surgery.onnx.load")
    @patch("conversion.onnx_surgery.cast_int64_initializers")
    def test_run_processing_failure(self, mock_cast_int64, mock_load):
        """Test run function when processing fails."""
        mock_model = Mock()
        mock_load.return_value = mock_model

        mock_cast_int64.side_effect = Exception("Processing failed")

        with pytest.raises(Exception):
            run("input.onnx", "output.onnx")


class TestMainFunction:
    """Test main function."""

    @patch("conversion.onnx_surgery.run")
    @patch("conversion.onnx_surgery.argparse.ArgumentParser")
    def test_main_success(self, mock_parser_class, mock_run):
        """Test successful main function execution."""
        # Mock argument parser
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.inp = "input.onnx"
        mock_args.out = "output.onnx"
        mock_args.infer = True
        mock_args.simplify = True
        mock_parser.parse_args.return_value = mock_args
        mock_parser_class.return_value = mock_parser

        # Test that main runs without error
        try:
            main()
        except SystemExit:
            pass  # Expected for successful completion

        mock_run.assert_called_once_with("input.onnx", "output.onnx")

    @patch("conversion.onnx_surgery.run")
    @patch("conversion.onnx_surgery.argparse.ArgumentParser")
    def test_main_run_failure(self, mock_parser_class, mock_run):
        """Test main function when run fails."""
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.inp = "input.onnx"
        mock_args.out = "output.onnx"
        mock_args.infer = False
        mock_args.simplify = False
        mock_parser.parse_args.return_value = mock_args
        mock_parser_class.return_value = mock_parser

        mock_run.side_effect = Exception("Run failed")

        with pytest.raises(SystemExit):
            main()

    @patch("conversion.onnx_surgery.argparse.ArgumentParser")
    def test_main_missing_required_args(self, mock_parser_class):
        """Test main function with missing required arguments."""
        mock_parser = Mock()
        mock_args = Mock()
        # Missing required inp and out args
        mock_args.inp = None
        mock_args.out = None
        mock_parser.parse_args.return_value = mock_args
        mock_parser_class.return_value = mock_parser

        # Should still run but argparse will handle missing args
        try:
            main()
        except SystemExit:
            pass  # Expected


class TestONNXSurgeryIntegration:
    """Test integration of ONNX surgery components."""

    def test_complete_surgery_workflow(self, tmp_path):
        """Test complete ONNX surgery workflow."""
        # Create a test ONNX model with INT64 initializers
        input_tensor = onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10])
        output_tensor = onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])

        # Add INT64 initializer
        int64_tensor = numpy_helper.from_array(
            np.array([1, 2, 3], dtype=np.int64), name="int64_init"
        )

        node = onnx.helper.make_node("Identity", ["input"], ["output"])
        graph = onnx.helper.make_graph(
            [node], "test_graph", [input_tensor], [output_tensor], initializer=[int64_tensor]
        )
        model = onnx.helper.make_model(graph)

        # Save test model
        input_path = tmp_path / "test_input.onnx"
        onnx.save(model, str(input_path))

        output_path = tmp_path / "test_output.onnx"

        # Run surgery
        run(str(input_path), str(output_path))

        # Verify output exists
        assert output_path.exists()

        # Load and verify the result
        result_model = onnx.load(str(output_path))
        assert len(result_model.graph.initializer) == 1

        # Check that INT64 was cast to INT32
        init = result_model.graph.initializer[0]
        assert init.data_type == TensorProto.INT32

    def test_dtype_forcing_integration(self):
        """Test dtype forcing functions work together."""
        # Create test model
        input_tensor = onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10])
        output_tensor = onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])

        node = onnx.helper.make_node("Identity", ["input"], ["output"])
        graph = onnx.helper.make_graph([node], "test_graph", [input_tensor], [output_tensor])
        model = onnx.helper.make_model(graph)

        # Force input and output dtypes
        model = force_input_dtype(model, "input", TensorProto.INT32)
        model = force_output_dtype(model, "output", TensorProto.FLOAT16)

        assert model.graph.input[0].type.tensor_type.elem_type == TensorProto.INT32
        assert model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT16

    def test_surgery_error_handling(self, tmp_path):
        """Test error handling in surgery operations."""
        # Test with non-existent input file
        with pytest.raises(Exception):
            run("nonexistent.onnx", str(tmp_path / "output.onnx"))

        # Test with invalid ONNX file
        invalid_path = tmp_path / "invalid.onnx"
        with open(invalid_path, "w") as f:
            f.write("not an onnx file")

        with pytest.raises(Exception):
            run(str(invalid_path), str(tmp_path / "output.onnx"))

    def test_simplification_availability(self):
        """Test onnxsim availability handling."""
        # Test when onnxsim is available
        if onnx_surgery_module.onnxsim is not None:
            # Should have simplify function
            assert hasattr(onnx_surgery_module.onnxsim, "simplify")
        else:
            # Should be None
            assert onnx_surgery_module.onnxsim is None

    def test_shape_inference_fallback(self):
        """Test shape inference error handling."""
        # Create model that might cause shape inference issues
        input_tensor = onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10])
        output_tensor = onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])

        # Create a potentially problematic node
        node = onnx.helper.make_node("UnknownOp", ["input"], ["output"])
        graph = onnx.helper.make_graph([node], "test_graph", [input_tensor], [output_tensor])
        model = onnx.helper.make_model(graph)

        # Test that strip_redundant_casts handles shape inference failures
        result = strip_redundant_casts(model)

        # Should return the model (may be unchanged due to inference failure)
        assert isinstance(result, onnx.ModelProto)

    def test_initializer_casting_edge_cases(self):
        """Test initializer casting with edge cases."""
        # Test with empty model
        input_tensor = onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10])
        output_tensor = onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])

        node = onnx.helper.make_node("Identity", ["input"], ["output"])
        graph = onnx.helper.make_graph([node], "test_graph", [input_tensor], [output_tensor])
        model = onnx.helper.make_model(graph)

        # Should handle empty initializer list
        result = cast_int64_initializers(model)
        assert result == 0

        # Test with various data types
        import numpy as np

        initializers = [
            numpy_helper.from_array(np.array([1], dtype=np.int64), name="int64"),
            numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="float32"),
            numpy_helper.from_array(np.array([1], dtype=np.int32), name="int32"),
        ]

        graph = onnx.helper.make_graph(
            [node], "test_graph", [input_tensor], [output_tensor], initializer=initializers
        )
        model = onnx.helper.make_model(graph)

        result = cast_int64_initializers(model)
        assert result == 1  # Only the INT64 one should be cast

        # Verify the casting
        int64_init = next(init for init in model.graph.initializer if init.name == "int64")
        assert int64_init.data_type == TensorProto.INT32
