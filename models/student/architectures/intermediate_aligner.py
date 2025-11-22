#!/usr/bin/env python3
"""
Intermediate Layer Knowledge Distillation Aligner

Projects student hidden states into teacher hidden state space for
intermediate layer knowledge distillation.

This module is training-only and should be stripped from state dict
before inference export.

Author: @darianrosebrook
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple


class IntermediateAligner(nn.Module):
    """
    Projects student hidden states into teacher hidden state space
    for a set of (student_layer, teacher_layer) mappings.

    Encapsulated API: forward() returns aligned pairs directly,
    so losses.py doesn't need to touch internal projections.
    """

    def __init__(
        self,
        mapping: List[Tuple[int, int]],
        student_d_model: int,
        teacher_d_model: int,
    ) -> None:
        super().__init__()
        self.mapping: List[Tuple[int, int]] = mapping
        self.student_d_model = student_d_model
        self.teacher_d_model = teacher_d_model

        self.projections = nn.ModuleDict()
        for si, ti in mapping:
            key = f"s{si}_t{ti}"
            self.projections[key] = nn.Linear(
                student_d_model, teacher_d_model, bias=False
            )
        # Leave init as PyTorch default (or use Xavier if you prefer)

    def forward(
        self,
        student_hidden_states: List[torch.Tensor],  # each [B, T, Ds]
        teacher_hidden_states: List[torch.Tensor],  # each [B, T, Dt]
    ) -> Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns a dict mapping (si, ti) -> (projected_student, teacher_detached),
        both shaped [B, T, Dt].
        """
        aligned: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        for si, ti in self.mapping:
            key = f"s{si}_t{ti}"
            h_s = student_hidden_states[si]          # [B, T, Ds]
            h_t = teacher_hidden_states[ti].detach() # [B, T, Dt]
            proj = self.projections[key](h_s)        # [B, T, Dt]
            aligned[(si, ti)] = (proj, h_t)
        return aligned










