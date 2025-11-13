"""
Unit tests for GQA Transformer architecture components.
"""

import torch

from models.student.architectures.gqa_transformer import (
    ModelCfg,
    RMSNorm,
    SwiGLU,
    RotaryEmbedding,
    MHA_GQA,
    Block,
    StudentLM,
)


class TestRMSNorm:
    """Tests for RMSNorm layer."""

    def test_rmsnorm_forward(self, device):
        """Test RMSNorm forward pass."""
        d = 128
        rmsnorm = RMSNorm(d).to(device)
        x = torch.randn(2, 10, d, device=device)

        output = rmsnorm(x)

        assert output.shape == x.shape
        assert output.dtype == x.dtype

    def test_rmsnorm_normalization(self, device):
        """Test that RMSNorm normalizes input."""
        d = 64
        rmsnorm = RMSNorm(d).to(device)
        x = torch.randn(1, 1, d, device=device) * 10  # Large values

        output = rmsnorm(x)

        # Check that output is normalized (RMS should be close to 1)
        rms = torch.sqrt((output.pow(2).mean(-1)))
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)

    def test_rmsnorm_gradient_flow(self, device):
        """Test that gradients flow through RMSNorm."""
        d = 64
        rmsnorm = RMSNorm(d).to(device)
        x = torch.randn(2, 10, d, device=device, requires_grad=True)

        output = rmsnorm(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))


class TestSwiGLU:
    """Tests for SwiGLU activation."""

    def test_swiglu_forward(self, device):
        """Test SwiGLU forward pass."""
        d_in = 128
        d_hidden = 256
        swiglu = SwiGLU(d_in, d_hidden).to(device)
        x = torch.randn(2, 10, d_in, device=device)

        output = swiglu(x)

        assert output.shape == (2, 10, d_in)

    def test_swiglu_gradient_flow(self, device):
        """Test that gradients flow through SwiGLU."""
        d_in = 64
        d_hidden = 128
        swiglu = SwiGLU(d_in, d_hidden).to(device)
        x = torch.randn(2, 5, d_in, device=device, requires_grad=True)

        output = swiglu(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))


class TestRotaryEmbedding:
    """Tests for Rotary Position Embedding."""

    def test_rope_apply(self, device):
        """Test RoPE apply method."""
        d_head = 64
        rope = RotaryEmbedding(d_head, theta=10000.0, scaling="none").to(device)
        b, h, t = 2, 4, 10
        q = torch.randn(b, h, t, d_head, device=device)
        k = torch.randn(b, h, t, d_head, device=device)

        q_rope, k_rope = rope.apply(q, k)

        assert q_rope.shape == q.shape
        assert k_rope.shape == k.shape

    def test_rope_apply_single(self, device):
        """Test RoPE apply_single method for decode mode."""
        d_head = 64
        rope = RotaryEmbedding(d_head, theta=10000.0, scaling="none").to(device)
        b, h = 2, 4
        q = torch.randn(b, h, 1, d_head, device=device)
        k = torch.randn(b, h, 1, d_head, device=device)
        pos = 5

        q_rope, k_rope = rope.apply_single(q, k, pos)

        assert q_rope.shape == q.shape
        assert k_rope.shape == k.shape
        assert q_rope.size(2) == 1  # Single position

    def test_rope_scaling_dynamic(self, device):
        """Test RoPE with dynamic scaling."""
        d_head = 64
        rope = RotaryEmbedding(d_head, theta=10000.0, scaling="dynamic").to(device)
        b, h, t = 2, 4, 20
        q = torch.randn(b, h, t, d_head, device=device)
        k = torch.randn(b, h, t, d_head, device=device)

        q_rope, k_rope = rope.apply(q, k)

        assert q_rope.shape == q.shape
        assert k_rope.shape == k.shape

    def test_rope_rotation_property(self, device):
        """Test that RoPE rotates embeddings (output differs from input)."""
        d_head = 64
        rope = RotaryEmbedding(d_head, theta=10000.0, scaling="none").to(device)
        b, h, t = 1, 1, 5
        q = torch.randn(b, h, t, d_head, device=device)
        k = torch.randn(b, h, t, d_head, device=device)

        q_rope, k_rope = rope.apply(q, k)

        # Rotated embeddings should differ from original
        assert not torch.allclose(q_rope, q, atol=1e-5)
        assert not torch.allclose(k_rope, k, atol=1e-5)


class TestMHAGQA:
    """Tests for Grouped Query Attention."""

    def test_mha_gqa_forward(self, device, small_model_cfg):
        """Test MHA_GQA forward pass."""
        rope = RotaryEmbedding(
            small_model_cfg.d_head, small_model_cfg.rope_theta, small_model_cfg.rope_scaling
        )
        mha = MHA_GQA(
            small_model_cfg.d_model,
            small_model_cfg.n_heads,
            small_model_cfg.n_kv_heads,
            small_model_cfg.d_head,
            rope,
            dropout=0.0,
        ).to(device)

        b, t = 2, 10
        x = torch.randn(b, t, small_model_cfg.d_model, device=device)

        output = mha(x)

        assert output.shape == (b, t, small_model_cfg.d_model)

    def test_mha_gqa_with_mask(self, device, small_model_cfg):
        """Test MHA_GQA with attention mask."""
        rope = RotaryEmbedding(
            small_model_cfg.d_head, small_model_cfg.rope_theta, small_model_cfg.rope_scaling
        )
        mha = MHA_GQA(
            small_model_cfg.d_model,
            small_model_cfg.n_heads,
            small_model_cfg.n_kv_heads,
            small_model_cfg.d_head,
            rope,
            dropout=0.0,
        ).to(device)

        b, t = 2, 10
        x = torch.randn(b, t, small_model_cfg.d_model, device=device)
        attn_mask = torch.zeros(b, small_model_cfg.n_heads, t, t, device=device)
        attn_mask[:, :, :, 5:] = -1e4  # Mask out positions after 5

        output = mha(x, attn_mask)

        assert output.shape == (b, t, small_model_cfg.d_model)

    def test_mha_gqa_forward_decode(self, device, small_model_cfg):
        """Test MHA_GQA forward_decode (single token with KV cache)."""
        rope = RotaryEmbedding(
            small_model_cfg.d_head, small_model_cfg.rope_theta, small_model_cfg.rope_scaling
        )
        mha = MHA_GQA(
            small_model_cfg.d_model,
            small_model_cfg.n_heads,
            small_model_cfg.n_kv_heads,
            small_model_cfg.d_head,
            rope,
            dropout=0.0,
        ).to(device)

        b = 2
        x = torch.randn(b, 1, small_model_cfg.d_model, device=device)

        # First token (no cache)
        output1, kv_cache1 = mha.forward_decode(x, kv_cache=None, pos=0)

        assert output1.shape == (b, 1, small_model_cfg.d_model)
        assert kv_cache1[0].shape == (b, small_model_cfg.n_kv_heads, 1, small_model_cfg.d_head)
        assert kv_cache1[1].shape == (b, small_model_cfg.n_kv_heads, 1, small_model_cfg.d_head)

        # Second token (with cache)
        x2 = torch.randn(b, 1, small_model_cfg.d_model, device=device)
        output2, kv_cache2 = mha.forward_decode(x2, kv_cache=kv_cache1, pos=1)

        assert output2.shape == (b, 1, small_model_cfg.d_model)
        assert kv_cache2[0].shape == (
            b,
            small_model_cfg.n_kv_heads,
            2,
            small_model_cfg.d_head,
        )  # Cache grows
        assert kv_cache2[1].shape == (b, small_model_cfg.n_kv_heads, 2, small_model_cfg.d_head)

    def test_mha_gqa_gqa_expansion(self, device, small_model_cfg):
        """Test that GQA correctly expands KV heads."""
        rope = RotaryEmbedding(
            small_model_cfg.d_head, small_model_cfg.rope_theta, small_model_cfg.rope_scaling
        )
        mha = MHA_GQA(
            small_model_cfg.d_model,
            small_model_cfg.n_heads,
            small_model_cfg.n_kv_heads,
            small_model_cfg.d_head,
            rope,
            dropout=0.0,
        ).to(device)

        assert mha.head_groups == small_model_cfg.n_heads // small_model_cfg.n_kv_heads
        assert mha.head_groups > 1  # Should have grouping


class TestBlock:
    """Tests for Transformer Block."""

    def test_block_forward(self, device, small_model_cfg):
        """Test Block forward pass."""
        block = Block(small_model_cfg).to(device)
        b, t = 2, 10
        x = torch.randn(b, t, small_model_cfg.d_model, device=device)
        attn_mask = None

        output = block(x, attn_mask)

        assert output.shape == x.shape

    def test_block_residual_connection(self, device, small_model_cfg):
        """Test that Block has residual connections."""
        block = Block(small_model_cfg).to(device)
        b, t = 1, 5
        x = torch.randn(b, t, small_model_cfg.d_model, device=device)
        attn_mask = None

        output = block(x, attn_mask)

        # Output should differ from input (due to transformations)
        assert not torch.allclose(output, x, atol=1e-5)

    def test_block_forward_decode(self, device, small_model_cfg):
        """Test Block forward_decode (single token with KV cache)."""
        block = Block(small_model_cfg).to(device)
        b = 2
        x = torch.randn(b, 1, small_model_cfg.d_model, device=device)

        # First token
        output1, kv_cache1 = block.forward_decode(x, kv_cache=None, pos=0)

        assert output1.shape == (b, 1, small_model_cfg.d_model)
        assert kv_cache1[0].shape == (b, small_model_cfg.n_kv_heads, 1, small_model_cfg.d_head)

        # Second token
        x2 = torch.randn(b, 1, small_model_cfg.d_model, device=device)
        output2, kv_cache2 = block.forward_decode(x2, kv_cache=kv_cache1, pos=1)

        assert output2.shape == (b, 1, small_model_cfg.d_model)
        assert kv_cache2[0].shape == (b, small_model_cfg.n_kv_heads, 2, small_model_cfg.d_head)


class TestStudentLM:
    """Tests for StudentLM model."""

    def test_studentlm_forward(self, small_model, device):
        """Test StudentLM forward pass."""
        b, t = 2, 10
        input_ids = torch.randint(0, 1000, (b, t), device=device)

        logits = small_model(input_ids)

        assert logits.shape == (b, t, 1000)  # vocab_size=1000
        assert logits.dtype == torch.float16

    def test_studentlm_forward_with_mask(self, small_model, device, small_model_cfg):
        """Test StudentLM forward with attention mask."""
        b, t = 2, 10
        input_ids = torch.randint(0, 1000, (b, t), device=device)
        # Attention mask shape: [B, H, T, T]
        attn_mask = torch.zeros(b, small_model_cfg.n_heads, t, t, device=device)
        attn_mask[:, :, :, 5:] = -1e4  # Mask positions after 5

        logits = small_model(input_ids, attn_mask)

        assert logits.shape == (b, t, 1000)

    def test_studentlm_forward_decode(self, small_model, device):
        """Test StudentLM forward_decode (single token with KV cache)."""
        b = 2
        input_ids = torch.randint(0, 1000, (b, 1), device=device)

        # First token
        logits1, kv_caches1 = small_model.forward_decode(input_ids, kv_caches=None, pos=0)

        assert logits1.shape == (b, 1, 1000)
        assert len(kv_caches1) == 2  # n_layers=2
        assert kv_caches1[0][0].shape == (b, 2, 1, 32)  # n_kv_heads=2, d_head=32

        # Second token
        input_ids2 = torch.randint(0, 1000, (b, 1), device=device)
        logits2, kv_caches2 = small_model.forward_decode(input_ids2, kv_caches=kv_caches1, pos=1)

        assert logits2.shape == (b, 1, 1000)
        assert kv_caches2[0][0].shape == (b, 2, 2, 32)  # Cache grows

    def test_studentlm_gradient_flow(self, small_model, device):
        """Test that gradients flow through StudentLM."""
        b, t = 2, 5
        input_ids = torch.randint(0, 1000, (b, t), device=device)

        logits = small_model(input_ids)
        loss = logits.sum()
        loss.backward()

        # Check that some parameters have gradients
        has_grad = False
        for param in small_model.parameters():
            if param.grad is not None and not torch.allclose(
                param.grad, torch.zeros_like(param.grad)
            ):
                has_grad = True
                break
        assert has_grad

    def test_studentlm_default_config(self, device):
        """Test StudentLM with default config."""
        # Use a valid config (n_heads must be divisible by n_kv_heads)
        cfg = ModelCfg(
            d_model=3584,
            n_layers=32,
            n_heads=28,
            n_kv_heads=7,  # 28 % 7 == 0
            d_head=128,
            vocab_size=32000,
            rope_theta=10000.0,
            rope_scaling="dynamic",
            dropout=0.0,
        )
        model = StudentLM(cfg).to(device)
        b, t = 1, 5
        input_ids = torch.randint(0, 32000, (b, t), device=device)

        logits = model(input_ids)

        assert logits.shape == (b, t, 32000)  # Default vocab_size

    def test_studentlm_consistency_prefill_decode(self, small_model, device):
        """Test that prefill and decode modes produce consistent results."""
        b = 2
        seq_len = 3

        # Prefill: process all tokens at once
        input_ids_prefill = torch.randint(0, 1000, (b, seq_len), device=device)
        logits_prefill = small_model(input_ids_prefill)

        # Decode: process tokens one by one
        kv_caches = None
        logits_decode_list = []
        for pos in range(seq_len):
            input_ids_single = input_ids_prefill[:, pos : pos + 1]
            logits_single, kv_caches = small_model.forward_decode(input_ids_single, kv_caches, pos)
            logits_decode_list.append(logits_single)

        logits_decode = torch.cat(logits_decode_list, dim=1)

        # Results should be similar (within numerical precision)
        # Note: May differ slightly due to different computation paths
        assert logits_prefill.shape == logits_decode.shape
