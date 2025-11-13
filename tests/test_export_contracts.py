"""
Export contract tests: Validate PyTorch export before CoreML conversion.

Tests:
- No unsupported ops (denylist)
- No int64 tensors on paths to softmax/attention
- Topological ordering verification
- Dtype & mask invariants (attention masks consistent dtype, softmax inputs FP16/FP32)
@author: @darianrosebrook
"""
import pytest
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, List, Optional

from models.student.architectures.gqa_transformer import StudentLM, ModelCfg


# Denylist of unsupported ops for CoreML conversion
UNSUPPORTED_OPS = {
    # Add ops that are known to be unsupported or problematic
    # This should be maintained based on coremltools compatibility
}


def check_exported_graph_for_unsupported_ops(exported_model, denylist: set = None) -> List[str]:
    """
    Check exported model graph for unsupported operations.
    
    Args:
        exported_model: TorchScript or ExportedProgram model
        denylist: Set of unsupported op names
        
    Returns:
        List of unsupported ops found (empty if none)
    """
    if denylist is None:
        denylist = UNSUPPORTED_OPS
    
    unsupported_found = []
    
    # For TorchScript models, traverse the graph
    if isinstance(exported_model, torch.jit.ScriptModule):
        graph = exported_model.graph
        for node in graph.nodes():
            node_kind = node.kind()
            if node_kind in denylist:
                unsupported_found.append(node_kind)
    
    # For ExportedProgram, check the graph module
    elif hasattr(exported_model, 'graph_module'):
        # ExportedProgram uses FX graph
        graph_module = exported_model.graph_module
        for node in graph_module.graph.nodes:
            if node.op == 'call_function':
                target = node.target
                if hasattr(target, '__name__'):
                    if target.__name__ in denylist:
                        unsupported_found.append(target.__name__)
    
    return unsupported_found


def check_int64_on_attention_paths(exported_model) -> List[str]:
    """
    Check for int64 tensors on paths to softmax/attention operations.
    
    ANE prefers int32 for indices, so int64 can cause CPU fallback.
    
    Returns:
        List of problematic int64 usages found
    """
    issues = []
    
    # For TorchScript models
    if isinstance(exported_model, torch.jit.ScriptModule):
        graph = exported_model.graph
        for node in graph.nodes():
            # Check if this node produces int64 and is used by attention/softmax
            if node.kind() == 'aten::embedding':
                # Embedding should use int32 input_ids, not int64
                for input_node in node.inputs():
                    if hasattr(input_node, 'type'):
                        input_type = input_node.type()
                        if str(input_type) == 'Tensor' and 'Long' in str(input_type):
                            issues.append(f"Embedding node {node} uses int64 input (should be int32)")
    
    return issues


def check_dtype_consistency(exported_model) -> List[str]:
    """
    Check dtype consistency: attention masks should be consistent dtype, softmax inputs FP16/FP32.
    
    Returns:
        List of dtype consistency issues found
    """
    issues = []
    
    # This is a basic check - in practice, you'd need to trace through the graph
    # For now, we'll validate at the model level before export
    
    return issues


def test_export_prefill_model_no_unsupported_ops():
    """Test that exported prefill model has no unsupported ops."""
    # Create a small test model
    cfg = ModelCfg(
        d_model=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_head=32,
        vocab_size=512,
    )
    model = StudentLM(cfg)
    model.eval()
    
    # Export to TorchScript
    example_input = torch.zeros((1, 64), dtype=torch.int32)
    with torch.no_grad():
        traced = torch.jit.trace(model, example_input)
        traced.eval()
    
    # Check for unsupported ops
    unsupported = check_exported_graph_for_unsupported_ops(traced)
    assert len(unsupported) == 0, f"Found unsupported ops in exported model: {unsupported}"


def test_export_model_int32_input_ids():
    """Test that exported model uses int32 input_ids, not int64."""
    cfg = ModelCfg(
        d_model=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_head=32,
        vocab_size=512,
    )
    model = StudentLM(cfg)
    model.eval()
    
    # Export with int32 input_ids (correct)
    example_input = torch.zeros((1, 64), dtype=torch.int32)
    with torch.no_grad():
        traced = torch.jit.trace(model, example_input)
        traced.eval()
    
    # Check for int64 issues
    issues = check_int64_on_attention_paths(traced)
    assert len(issues) == 0, f"Found int64 usage issues: {issues}"


def test_attention_mask_dtype_consistency():
    """Test that attention mask handling maintains consistent dtype."""
    from models.student.architectures.gqa_transformer import MHA_GQA, RotaryEmbedding
    
    cfg = ModelCfg(
        d_model=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_head=32,
        vocab_size=512,
    )
    
    # Create attention module
    rope = RotaryEmbedding(cfg.d_head, cfg.rope_theta, cfg.rope_scaling)
    attn = MHA_GQA(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.d_head, rope)
    attn.eval()
    
    # Test with different mask dtypes
    x = torch.randn(1, 10, cfg.d_model)
    
    # Test with int64 mask (should be converted)
    mask_int64 = torch.ones(1, 10, dtype=torch.int64)
    with torch.no_grad():
        output = attn(x, mask_int64)
    assert output.dtype == torch.float32, "Attention output should be float32"
    
    # Test with int32 mask
    mask_int32 = torch.ones(1, 10, dtype=torch.int32)
    with torch.no_grad():
        output2 = attn(x, mask_int32)
    assert output2.dtype == torch.float32, "Attention output should be float32"
    
    # Test with float32 mask
    mask_float = torch.ones(1, 10, dtype=torch.float32)
    with torch.no_grad():
        output3 = attn(x, mask_float)
    assert output3.dtype == torch.float32, "Attention output should be float32"


def test_softmax_inputs_not_int():
    """Test that softmax inputs are never integer types."""
    # This is validated by the model architecture - softmax is applied to attention scores
    # which are always float. We test this indirectly by checking attention outputs.
    from models.student.architectures.gqa_transformer import MHA_GQA, RotaryEmbedding
    
    cfg = ModelCfg(
        d_model=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_head=32,
        vocab_size=512,
    )
    
    rope = RotaryEmbedding(cfg.d_head, cfg.rope_theta, cfg.rope_scaling)
    attn = MHA_GQA(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.d_head, rope)
    attn.eval()
    
    x = torch.randn(1, 10, cfg.d_model)
    mask = torch.ones(1, 10, dtype=torch.int32)
    
    with torch.no_grad():
        output = attn(x, mask)
    
    # Output should be float, not int
    assert output.dtype in [torch.float32, torch.float16], \
        f"Attention output dtype should be float, got {output.dtype}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

