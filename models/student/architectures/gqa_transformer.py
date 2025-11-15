# models/student/architectures/gqa_transformer.py
# Minimal, CoreML-friendly student LM with GQA, RMSNorm, SwiGLU, and RoPE.
# Notes:
# - Keep ops "boring": matmul, add, mul, gelu/swiglu, layernorm/rmsnorm.
# - No dynamic control flow; shapes must be static during export.
# - RoPE is applied on Q/K only. GQA reduces KV cache footprint.
# @author: @darianrosebrook

from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelCfg:
    d_model: int = 3584
    n_layers: int = 32
    n_heads: int = 28
    n_kv_heads: int = 8
    d_head: int = 128
    vocab_size: int = 32000
    rope_theta: float = 10000.0
    rope_scaling: str = "dynamic"  # ["none","linear","dynamic"]
    dropout: float = 0.0

    # Intermediate layer KD config (distillation-only, ignored at inference)
    use_intermediate_layers: bool = False
    teacher_d_model: Optional[int] = None
    teacher_n_layers: Optional[int] = None
    layer_mapping: Optional[List[Tuple[int, int]]] = None


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CoreML-friendly: avoid fancy reductions beyond mean of squares.
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x * norm


class SwiGLU(nn.Module):
    def __init__(self, d_in: int, d_hidden: int):
        super().__init__()
        self.w1 = nn.Linear(d_in, d_hidden, bias=False)
        self.w3 = nn.Linear(d_in, d_hidden, bias=False)
        self.w2 = nn.Linear(d_hidden, d_in, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.w1(x)
        b = self.w3(x)
        return self.w2(F.silu(a) * b)


class RotaryEmbedding(nn.Module):
    """RoPE: rotate Q/K with base theta. Implements static, linear, or dynamic scaling.
    Export note: precompute cos/sin for enumerated T on exporter side if needed.
    """

    def __init__(self, d_head: int, theta: float = 10000.0, scaling: str = "dynamic"):
        super().__init__()
        self.d_head = d_head
        self.theta = theta
        self.scaling = scaling

    def _rope_angles(self, t: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        half = self.d_head // 2
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
        )
        # Positions 0..t-1
        pos = torch.arange(t, device=device, dtype=torch.float32)
        # Dynamic scaling: simple heuristic (can be replaced with NTK-aware scaling)
        if self.scaling == "dynamic":
            scale = max(1.0, t / 2048.0)
            inv_freq = inv_freq / scale
        angles = torch.einsum("p,d->pd", pos, inv_freq)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        return cos, sin

    def apply(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # q,k: [B, H, T, Dh]
        b, h, t, dh = q.shape
        device = q.device
        cos, sin = self._rope_angles(t, device)  # [T, Dh/2]
        # Interleave dims: (x1, x2) → rotate

        def rope(x: torch.Tensor) -> torch.Tensor:
            x1, x2 = x[..., : dh // 2], x[..., dh // 2 :]
            cos_t = cos.unsqueeze(0).unsqueeze(0)  # [1,1,T, Dh/2]
            sin_t = sin.unsqueeze(0).unsqueeze(0)
            xr = torch.cat([x1 * cos_t - x2 * sin_t, x2 * cos_t + x1 * sin_t], dim=-1)
            return xr

        return rope(q), rope(k)

    def apply_single(
        self, q: torch.Tensor, k: torch.Tensor, pos: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE for a single position (decode mode)."""
        # q,k: [B, H, 1, Dh]
        b, h, t, dh = q.shape
        assert t == 1, "Single position mode expects t=1"
        device = q.device

        half = self.d_head // 2
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
        )

        # Dynamic scaling
        if self.scaling == "dynamic":
            scale = max(1.0, (pos + 1) / 2048.0)
            inv_freq = inv_freq / scale

        angle = pos * inv_freq  # [Dh/2]
        cos = torch.cos(angle)  # [Dh/2]
        sin = torch.sin(angle)  # [Dh/2]

        def rope(x: torch.Tensor) -> torch.Tensor:
            x1, x2 = x[..., : dh // 2], x[..., dh // 2 :]
            cos_t = cos.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1,1,1, Dh/2]
            sin_t = sin.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            xr = torch.cat([x1 * cos_t - x2 * sin_t, x2 * cos_t + x1 * sin_t], dim=-1)
            return xr

        return rope(q), rope(k)


class MHA_GQA(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_head: int,
        rope: RotaryEmbedding,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert n_heads % n_kv_heads == 0, "GQA requires head groups"
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.head_groups = n_heads // n_kv_heads
        self.rope = rope

        self.wq = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * d_head, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * d_head, bias=False)
        self.wo = nn.Linear(n_heads * d_head, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t, d = x.shape
        q = self.wq(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)  # [B,H,T,Dh]
        k = self.wk(x).view(b, t, self.n_kv_heads, self.d_head).transpose(1, 2)  # [B,Hk,T,Dh]
        v = self.wv(x).view(b, t, self.n_kv_heads, self.d_head).transpose(1, 2)  # [B,Hk,T,Dh]

        # RoPE on Q/K
        q, k = self.rope.apply(q, k)

        # GQA: expand K,V across head groups
        if self.head_groups > 1:
            k = k.repeat_interleave(self.head_groups, dim=1)
            v = v.repeat_interleave(self.head_groups, dim=1)

        # Attention
        scale = 1.0 / math.sqrt(self.d_head)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B,H,T,T]
        if attn_mask is not None:
            mask_dtype = attn_scores.dtype
            mask_shape = attn_mask.shape
            
            if len(mask_shape) == 2:
                # Binary mask [B, T] (1=real token, 0=pad) - convert to additive mask
                # Padding positions get -1e4, real tokens get 0
                additive_mask = (1.0 - attn_mask.to(mask_dtype)).unsqueeze(1).unsqueeze(2) * -1e4
                attn_scores = attn_scores + additive_mask
            elif len(mask_shape) == 4:
                # Additive mask [B, n_heads, T, T] or [B, 1, T, T] - use directly
                # If mask has n_heads dimension, ensure it matches or can broadcast
                if mask_shape[1] == 1:
                    # [B, 1, T, T] - broadcast to [B, H, T, T]
                    additive_mask = attn_mask.to(mask_dtype).expand(-1, self.n_heads, -1, -1)
                elif mask_shape[1] == self.n_heads:
                    # [B, H, T, T] - use directly
                    additive_mask = attn_mask.to(mask_dtype)
                else:
                    raise ValueError(f"Mask shape {mask_shape} incompatible with {self.n_heads} heads")
                attn_scores = attn_scores + additive_mask
            else:
                raise ValueError(f"Unsupported mask shape: {mask_shape}")
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.attn_dropout(attn)
        y = torch.matmul(attn, v)  # [B,H,T,Dh]
        y = y.transpose(1, 2).contiguous().view(b, t, self.n_heads * self.d_head)
        return self.wo(y)

    def forward_decode(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        pos: int = 0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Decoder-only forward: single token with KV cache.

        Args:
            x: [B, 1, D] single token embedding
            kv_cache: Optional tuple of (k_cache, v_cache) each [B, Hk, T_cache, Dh]
            pos: Position index for RoPE

        Returns:
            output: [B, 1, D] attention output
            kv_cache: Updated (k_cache, v_cache) for next step
        """
        b, t, d = x.shape
        assert t == 1, "Decode mode expects single token"

        q = self.wq(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)  # [B,H,1,Dh]
        k_new = self.wk(x).view(b, t, self.n_kv_heads, self.d_head).transpose(1, 2)  # [B,Hk,1,Dh]
        v_new = self.wv(x).view(b, t, self.n_kv_heads, self.d_head).transpose(1, 2)  # [B,Hk,1,Dh]

        # RoPE on Q/K (single position)
        q_rope, k_rope = self.rope.apply_single(q, k_new, pos)

        # Update KV cache
        if kv_cache is None:
            k_cache = k_rope  # [B,Hk,1,Dh]
            v_cache = v_new
        else:
            k_cache_prev, v_cache_prev = kv_cache
            k_cache = torch.cat([k_cache_prev, k_rope], dim=2)  # [B,Hk,T+1,Dh]
            v_cache = torch.cat([v_cache_prev, v_new], dim=2)

        # GQA: expand K,V across head groups
        if self.head_groups > 1:
            k_expanded = k_cache.repeat_interleave(self.head_groups, dim=1)  # [B,H,T+1,Dh]
            v_expanded = v_cache.repeat_interleave(self.head_groups, dim=1)
        else:
            k_expanded = k_cache
            v_expanded = v_cache

        # Attention: q [B,H,1,Dh] @ k_expanded [B,H,T+1,Dh]^T -> [B,H,1,T+1]
        scale = 1.0 / math.sqrt(self.d_head)
        attn_scores = torch.matmul(q_rope, k_expanded.transpose(-2, -1)) * scale  # [B,H,1,T+1]
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.attn_dropout(attn)
        y = torch.matmul(attn, v_expanded)  # [B,H,1,Dh]
        y = y.transpose(1, 2).contiguous().view(b, t, self.n_heads * self.d_head)
        return self.wo(y), (k_cache, v_cache)


class Block(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = MHA_GQA(
            cfg.d_model,
            cfg.n_heads,
            cfg.n_kv_heads,
            cfg.d_head,
            rope=RotaryEmbedding(cfg.d_head, cfg.rope_theta, cfg.rope_scaling),
            dropout=cfg.dropout,
        )
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, 4 * cfg.d_model)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x

    def forward_decode(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        pos: int = 0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Decoder-only forward for single token with KV cache."""
        x_normed = self.norm1(x)
        attn_out, kv_new = self.attn.forward_decode(x_normed, kv_cache, pos)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, kv_new


class StudentLM(nn.Module):
    def __init__(
        self,
        cfg: Optional[ModelCfg] = None,
        use_self_evaluation: bool = False,
        use_halt_head: bool = False,
    ):
        super().__init__()
        self.cfg = cfg or ModelCfg()
        self.embed = nn.Embedding(self.cfg.vocab_size, self.cfg.d_model)
        self.blocks = nn.ModuleList([Block(self.cfg) for _ in range(self.cfg.n_layers)])
        self.norm_f = RMSNorm(self.cfg.d_model)
        self.lm_head = nn.Linear(self.cfg.d_model, self.cfg.vocab_size, bias=False)

        # Gradient checkpointing flag
        self.checkpointing = False

        # Self-evaluation head (optional)
        self.use_self_evaluation = use_self_evaluation
        if use_self_evaluation:
            self.eval_head = nn.Sequential(
                nn.Linear(self.cfg.d_model, self.cfg.d_model // 2),
                nn.ReLU(),
                nn.Linear(self.cfg.d_model // 2, 1),  # Single score
                nn.Sigmoid(),  # Output 0-1 confidence
            )

        # Halt head for learned halting (optional)
        self.use_halt_head = use_halt_head
        if use_halt_head:
            # Halt head outputs 2 logits: [continue, halt]
            self.halt_head = nn.Linear(self.cfg.d_model, 2)

        # Intermediate layer aligner (optional, distillation-only)
        self.intermediate_aligner = None
        if self.cfg.use_intermediate_layers and self.cfg.teacher_d_model is not None and self.cfg.teacher_n_layers is not None:
            from .intermediate_aligner import IntermediateAligner

            student_n_layers = self.cfg.n_layers
            teacher_n_layers = self.cfg.teacher_n_layers

            # Use provided mapping or compute ratio-based mapping with clamping
            if self.cfg.layer_mapping is not None:
                mapping = self.cfg.layer_mapping
            else:
                mapping = []
                # map [0..Ls-1] → [0..Lt-1] using ratio, with clamping
                for si in range(student_n_layers):
                    if student_n_layers == 1:
                        teacher_idx = 0
                    else:
                        ratio = si / (student_n_layers - 1)
                        teacher_idx = int(round(ratio * (teacher_n_layers - 1)))
                    teacher_idx = max(0, min(teacher_n_layers - 1, teacher_idx))
                    mapping.append((si, teacher_idx))

            self.intermediate_aligner = IntermediateAligner(
                mapping=mapping,
                student_d_model=self.cfg.d_model,
                teacher_d_model=self.cfg.teacher_d_model,
            )

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing to reduce memory usage during training."""
        self.checkpointing = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
        return_eval_score: bool = False,
        return_halt_logits: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, List[torch.Tensor]],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        """
        Forward pass with optional hidden states, evaluation score, and halt logits.

        Args:
            input_ids: [B, T] token IDs
            attn_mask: Optional attention mask
            return_hidden_states: If True, return list of hidden states per layer
            return_eval_score: If True, return self-evaluation score
            return_halt_logits: If True, return halt head logits [B, 2] (if use_halt_head=True)

        Returns:
            If return_hidden_states=False and return_eval_score=False and return_halt_logits=False:
                logits: [B, T, V]
            If return_halt_logits=True:
                logits: [B, T, V], halt_logits: [B, 2] (if use_halt_head=True)
            If return_hidden_states=True:
                logits: [B, T, V], hidden_states: List[[B, T, D]]
            If return_eval_score=True:
                logits: [B, T, V], eval_score: [B, 1] (if use_self_evaluation=True)
            Combinations return tuples accordingly
        """
        # input_ids: [B,T] int32/int64
        x = self.embed(input_ids)

        hidden_states = []
        if return_hidden_states:
            hidden_states.append(x)  # Include embedding output

        # Use gradient checkpointing if enabled (trades compute for memory)
        if self.checkpointing:
            for blk in self.blocks:
                x = torch.utils.checkpoint.checkpoint(blk, x, attn_mask, use_reentrant=False)
                if return_hidden_states:
                    hidden_states.append(x)
        else:
            for blk in self.blocks:
                x = blk(x, attn_mask)
                if return_hidden_states:
                    hidden_states.append(x)

        x = self.norm_f(x)

        # Self-evaluation score (if enabled)
        eval_score = None
        if return_eval_score and self.use_self_evaluation:
            # Use last token's hidden state for evaluation
            eval_score = self.eval_head(x[:, -1, :])  # [B, 1]

        # Halt head logits (if enabled)
        halt_logits = None
        if return_halt_logits and self.use_halt_head:
            # Pool hidden state (mean over sequence length) for halt head
            pooled = x.mean(dim=1)  # [B, D]
            halt_logits = self.halt_head(pooled)  # [B, 2]

        logits = self.lm_head(x).to(dtype=torch.float16)

        # Return based on flags
        result = [logits]
        if return_hidden_states:
            result.append(hidden_states)
        if return_eval_score and eval_score is not None:
            result.append(eval_score)
        if return_halt_logits and halt_logits is not None:
            result.append(halt_logits)

        if len(result) == 1:
            return result[0]
        return tuple(result)

    def forward_decode(
        self,
        input_ids: torch.Tensor,
        kv_caches: Optional[list] = None,
        pos: int = 0,
        return_halt_logits: bool = False,
    ) -> Union[Tuple[torch.Tensor, list], Tuple[torch.Tensor, list, torch.Tensor]]:
        """Decoder-only forward: single token with KV cache.

        Args:
            input_ids: [B, 1] single token
            kv_caches: Optional list of per-layer KV caches, each (k_cache, v_cache)
            pos: Position index for RoPE
            return_halt_logits: If True, return halt head logits [B, 2] (if use_halt_head=True)

        Returns:
            logits: [B, 1, V] logits for next token
            kv_caches: Updated list of KV caches
            halt_logits: [B, 2] halt head logits (if return_halt_logits=True and use_halt_head=True)
        """
        if kv_caches is None:
            kv_caches = [None] * self.cfg.n_layers

        x = self.embed(input_ids)  # [B, 1, D]
        updated_caches = []

        for i, blk in enumerate(self.blocks):
            x, kv_new = blk.forward_decode(x, kv_caches[i], pos)
            updated_caches.append(kv_new)

        x = self.norm_f(x)

        # Halt head logits (if enabled)
        halt_logits = None
        if return_halt_logits and self.use_halt_head:
            # Use current hidden state for halt head
            pooled = x.squeeze(1)  # [B, D] (remove sequence dim)
            halt_logits = self.halt_head(pooled)  # [B, 2]

        logits = self.lm_head(x).to(dtype=torch.float16)

        if return_halt_logits and halt_logits is not None:
            return logits, updated_caches, halt_logits
        return logits, updated_caches  # [B,1,V], list of (k,v) tuples

    def forward_hidden(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer blocks using hidden state (no embedding, no LM head).
        Used for latent mode processing where we process hidden states without token generation.

        Args:
            hidden_state: [B, T, D] hidden state tensor (already embedded)

        Returns:
            updated_hidden: [B, T, D] updated hidden state after transformer blocks
        """
        x = hidden_state
        for blk in self.blocks:
            x = blk(x, attn_mask=None)
        x = self.norm_f(x)
        return x
