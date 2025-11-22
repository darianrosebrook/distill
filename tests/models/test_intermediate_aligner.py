#!/usr/bin/env python3
"""
Tests for IntermediateAligner module and related functionality.

Tests the intermediate layer knowledge distillation components.
"""

import torch
import torch.nn as nn
from models.student.architectures.intermediate_aligner import IntermediateAligner
from training.losses import intermediate_layer_loss


class TestIntermediateAligner:
    """Test IntermediateAligner functionality."""

    def test_aligner_initialization(self):
        """Test that aligner initializes with correct projection layers."""
        mapping = [(0, 0), (4, 8), (8, 16)]
        student_d_model = 512
        teacher_d_model = 1024

        aligner = IntermediateAligner(
            mapping=mapping,
            student_d_model=student_d_model,
            teacher_d_model=teacher_d_model,
        )

        # Check mapping is stored
        assert aligner.mapping == mapping

        # Check projections are created
        assert len(aligner.projections) == len(mapping)
        for si, ti in mapping:
            key = f"s{si}_t{ti}"
            assert key in aligner.projections
            proj = aligner.projections[key]
            assert isinstance(proj, nn.Linear)
            assert proj.in_features == student_d_model
            assert proj.out_features == teacher_d_model

    def test_aligner_forward_pass(self):
        """Test forward pass produces correctly aligned pairs."""
        mapping = [(0, 0), (2, 4)]
        student_d_model = 256
        teacher_d_model = 512

        aligner = IntermediateAligner(
            mapping=mapping,
            student_d_model=student_d_model,
            teacher_d_model=teacher_d_model,
        )

        # Create mock hidden states
        batch_size, seq_len = 2, 10
        n_layers = 5

        student_hidden = [
            torch.randn(batch_size, seq_len, student_d_model)
            for _ in range(n_layers)
        ]
        teacher_hidden = [
            torch.randn(batch_size, seq_len, teacher_d_model)
            for _ in range(n_layers)
        ]

        # Forward pass
        aligned_pairs = aligner(student_hidden, teacher_hidden)

        # Check results
        assert len(aligned_pairs) == len(mapping)
        for (si, ti), (proj_student, teacher) in aligned_pairs.items():
            assert proj_student.shape == (batch_size, seq_len, teacher_d_model)
            assert teacher.shape == (batch_size, seq_len, teacher_d_model)
            # Verify it's the correct layer indices
            assert si in [0, 2]
            assert ti in [0, 4]


class TestLayerMapping:
    """Test layer mapping computation."""

    def test_layer_mapping_in_bounds(self):
        """Test that computed layer mappings stay within teacher bounds."""
        # Test various student/teacher layer combinations
        test_cases = [
            (8, 32),   # Small student, large teacher
            (24, 64),  # Medium sizes
            (32, 32),  # Equal sizes
            (1, 10),   # Edge case: single student layer
        ]

        for student_layers, teacher_layers in test_cases:
            mapping = []

            for si in range(student_layers):
                if student_layers == 1:
                    teacher_idx = 0
                else:
                    ratio = si / (student_layers - 1)
                    teacher_idx = int(round(ratio * (teacher_layers - 1)))

                # Clamp to valid range
                teacher_idx = max(0, min(teacher_layers - 1, teacher_idx))
                mapping.append((si, teacher_idx))

            # Verify all mappings are valid
            for si, ti in mapping:
                assert 0 <= si < student_layers
                assert 0 <= ti < teacher_layers

            # Verify endpoints are preserved
            if student_layers > 1:
                first_si, first_ti = mapping[0]
                last_si, last_ti = mapping[-1]
                assert first_ti == 0  # First student layer maps to first teacher
                assert last_ti == teacher_layers - 1  # Last student maps to last teacher


class TestIntermediateLayerLoss:
    """Test intermediate layer loss computation."""

    def test_loss_with_valid_pairs(self):
        """Test loss computation with valid aligned pairs."""
        batch_size, seq_len, d_model = 2, 8, 256

        # Create mock aligned pairs
        aligned_pairs = {
            (0, 0): (
                torch.randn(batch_size, seq_len, d_model),  # projected student
                torch.randn(batch_size, seq_len, d_model),  # teacher
            ),
            (4, 8): (
                torch.randn(batch_size, seq_len, d_model),
                torch.randn(batch_size, seq_len, d_model),
            ),
        }

        # Compute loss
        loss_dict = intermediate_layer_loss(
            aligned_pairs,
            use_layer_norm=True,
            mse_weight=0.1,
            cosine_weight=0.1,
        )

        # Check results
        assert "total" in loss_dict
        assert loss_dict["total"] >= 0

        # Check per-layer losses exist
        expected_keys = [
            "layer_s0_t0_mse", "layer_s0_t0_cosine", "layer_s0_t0_total",
            "layer_s4_t8_mse", "layer_s4_t8_cosine", "layer_s4_t8_total",
        ]
        for key in expected_keys:
            assert key in loss_dict
            assert loss_dict[key] >= 0

    def test_loss_with_empty_pairs(self):
        """Test loss computation with no aligned pairs returns zero."""
        loss_dict = intermediate_layer_loss(
            None,
            use_layer_norm=True,
            mse_weight=0.1,
            cosine_weight=0.1,
        )

        assert "total" in loss_dict
        assert loss_dict["total"] == 0.0

    def test_loss_without_layer_norm(self):
        """Test loss computation without layer normalization."""
        batch_size, seq_len, d_model = 2, 8, 256

        aligned_pairs = {
            (0, 0): (
                torch.randn(batch_size, seq_len, d_model),
                torch.randn(batch_size, seq_len, d_model),
            ),
        }

        loss_dict = intermediate_layer_loss(
            aligned_pairs,
            use_layer_norm=False,  # Disable layer norm
            mse_weight=0.5,
            cosine_weight=0.5,
        )

        assert "total" in loss_dict
        assert loss_dict["total"] >= 0


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_aligner_and_loss_integration(self):
        """Test that aligner output works with loss function."""
        mapping = [(0, 0), (2, 4)]
        student_d_model = 128
        teacher_d_model = 256

        aligner = IntermediateAligner(
            mapping=mapping,
            student_d_model=student_d_model,
            teacher_d_model=teacher_d_model,
        )

        # Create hidden states
        batch_size, seq_len = 2, 10
        n_layers = 5

        student_hidden = [
            torch.randn(batch_size, seq_len, student_d_model)
            for _ in range(n_layers)
        ]
        teacher_hidden = [
            torch.randn(batch_size, seq_len, teacher_d_model)
            for _ in range(n_layers)
        ]

        # Get aligned pairs
        aligned_pairs = aligner(student_hidden, teacher_hidden)

        # Compute loss
        loss_dict = intermediate_layer_loss(
            aligned_pairs,
            use_layer_norm=True,
            mse_weight=0.1,
            cosine_weight=0.1,
        )

        # Verify everything works together
        assert "total" in loss_dict
        assert loss_dict["total"] >= 0
        assert len([k for k in loss_dict.keys() if k.startswith("layer_")]) == 6  # 2 layers × 3 metrics each










