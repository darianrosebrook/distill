"""
Tests for conversion/make_toy_onnx.py - Toy ONNX model creation.

Tests ONNX graph creation with Gather, MatMul, Add, and Reshape operations.
"""
# @author: @darianrosebrook

import onnx
from unittest.mock import patch


# Import the module
import importlib
make_toy_onnx_module = importlib.import_module("conversion.make_toy_onnx")
main = make_toy_onnx_module.main


class TestMakeToyONNX:
    """Test toy ONNX model creation."""

    def test_main_success(self, tmp_path):
        """Test successful toy ONNX creation."""
        output_path = tmp_path / "toy.onnx"

        with patch("sys.argv", ["make_toy_onnx", "--out", str(output_path)]):
            main()

        assert output_path.exists()
        # Verify it's a valid ONNX file
        model = onnx.load(str(output_path))
        assert model is not None
        assert len(model.graph.input) == 2  # input_ids and attention_mask
        assert len(model.graph.output) == 1  # logits

    def test_main_default_args(self, tmp_path):
        """Test main function with default arguments."""
        output_path = tmp_path / "default_toy.onnx"

        with patch("sys.argv", ["make_toy_onnx", "--out", str(output_path)]):
            main()

        model = onnx.load(str(output_path))
        # Check default dimensions
        input_shape = model.graph.input[0].type.tensor_type.shape.dim
        assert input_shape[1].dim_value == 128  # Default seq

    def test_main_custom_args(self, tmp_path):
        """Test main function with custom arguments."""
        output_path = tmp_path / "custom_toy.onnx"

        with patch(
            "sys.argv",
            [
                "make_toy_onnx",
                "--seq",
                "256",
                "--vocab",
                "512",
                "--dmodel",
                "128",
                "--out",
                str(output_path),
            ],
        ):
            main()

        model = onnx.load(str(output_path))
        # Check custom dimensions
        input_shape = model.graph.input[0].type.tensor_type.shape.dim
        assert input_shape[1].dim_value == 256  # Custom seq

        output_shape = model.graph.output[0].type.tensor_type.shape.dim
        assert output_shape[1].dim_value == 256  # Custom seq
        assert output_shape[2].dim_value == 128  # Custom dmodel

    def test_onnx_model_structure(self, tmp_path):
        """Test ONNX model has correct structure."""
        output_path = tmp_path / "toy.onnx"

        with patch("sys.argv", ["make_toy_onnx", "--out", str(output_path)]):
            main()

        model = onnx.load(str(output_path))

        # Check inputs
        assert len(model.graph.input) == 2
        input_names = [inp.name for inp in model.graph.input]
        assert "input_ids" in input_names
        assert "attention_mask" in input_names

        # Check outputs
        assert len(model.graph.output) == 1
        assert model.graph.output[0].name == "logits"

        # Check nodes exist
        node_names = [node.op_type for node in model.graph.node]
        assert "Gather" in node_names
        assert "MatMul" in node_names
        assert "Add" in node_names
        assert "Reshape" in node_names

    def test_onnx_initializers(self, tmp_path):
        """Test ONNX model has correct initializers."""
        output_path = tmp_path / "toy.onnx"

        with patch("sys.argv", ["make_toy_onnx", "--out", str(output_path)]):
            main()

        model = onnx.load(str(output_path))

        # Check initializers exist
        initializer_names = [init.name for init in model.graph.initializer]
        assert "emb_w" in initializer_names
        assert "w1" in initializer_names
        assert "b1" in initializer_names
        assert "shape_td" in initializer_names
        assert "shape_1td" in initializer_names

    def test_onnx_opset_version(self, tmp_path):
        """Test ONNX model has correct opset version."""
        output_path = tmp_path / "toy.onnx"

        with patch("sys.argv", ["make_toy_onnx", "--out", str(output_path)]):
            main()

        model = onnx.load(str(output_path))
        assert model.opset_import[0].version == 19

    def test_onnx_graph_connectivity(self, tmp_path):
        """Test ONNX graph nodes are properly connected."""
        output_path = tmp_path / "toy.onnx"

        with patch("sys.argv", ["make_toy_onnx", "--out", str(output_path)]):
            main()

        model = onnx.load(str(output_path))

        # Verify graph structure: Gather -> Reshape -> MatMul -> Add -> Reshape
        nodes = model.graph.node
        assert len(nodes) >= 5

        # Check first node is Gather
        assert nodes[0].op_type == "Gather"
        assert "input_ids" in nodes[0].input
        assert "emb_w" in nodes[0].input

    def test_onnx_different_sizes(self, tmp_path):
        """Test creating ONNX models with different sizes."""
        sizes = [
            (64, 128, 32),  # Small
            (128, 256, 64),  # Medium
            (256, 512, 128),  # Large
        ]

        for seq, vocab, dmodel in sizes:
            output_path = tmp_path / f"toy_{seq}_{vocab}_{dmodel}.onnx"

            with patch(
                "sys.argv",
                [
                    "make_toy_onnx",
                    "--seq",
                    str(seq),
                    "--vocab",
                    str(vocab),
                    "--dmodel",
                    str(dmodel),
                    "--out",
                    str(output_path),
                ],
            ):
                main()

            model = onnx.load(str(output_path))
            # Verify dimensions
            input_shape = model.graph.input[0].type.tensor_type.shape.dim
            assert input_shape[1].dim_value == seq

    def test_onnx_output_directory_creation(self, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        output_path = tmp_path / "subdir" / "toy.onnx"

        with patch("sys.argv", ["make_toy_onnx", "--out", str(output_path)]):
            main()

        assert output_path.exists()
        assert output_path.parent.exists()

    def test_onnx_model_validity(self, tmp_path):
        """Test that created ONNX model is valid."""
        output_path = tmp_path / "toy.onnx"

        with patch("sys.argv", ["make_toy_onnx", "--out", str(output_path)]):
            main()

        # onnx.load should not raise an exception
        model = onnx.load(str(output_path))

        # Basic validation
        assert model.graph is not None
        assert len(model.graph.input) > 0
        assert len(model.graph.output) > 0
        assert len(model.graph.node) > 0







