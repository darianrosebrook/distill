"""
CoreML runtime generator with streaming and constrained decoding for tool calls.

Supports:
- Streaming text generation
- Constrained JSON decoding for tool calls
- Tool selection stage
- Post-tool integration
- Final answer generation
"""
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False

from coreml.runtime.constrained_decode import JSONConstrainedDecoder


def load_coreml_model(mlpackage_path: str):
    """
    Load CoreML model from .mlpackage file.
    
    Args:
        mlpackage_path: Path to .mlpackage file
        
    Returns:
        Loaded CoreML model
    """
    if not COREML_AVAILABLE:
        raise ImportError("coremltools not available. Install with: pip install coremltools")
    
    model = ct.models.MLModel(mlpackage_path)
    return model


def generate_tool_call(
    model,
    tokenizer,
    prompt: str,
    tools: List[Dict[str, Any]],
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a tool call using constrained decoding.
    
    Args:
        model: CoreML model or PyTorch model
        tokenizer: Tokenizer for encoding/decoding
        prompt: Input prompt
        tools: List of available tools with schemas
        device: Device to run on (for PyTorch models)
        
    Returns:
        Dictionary with tool call: {"name": "...", "arguments": {...}}
    """
    # Create generic tool call schema
    tool_schema = {
        "type": "object",
        "required": ["name", "arguments"],
        "properties": {
            "name": {"type": "string"},
            "arguments": {"type": "object"}
        }
    }
    
    # Create constrained decoder
    decoder = JSONConstrainedDecoder(schema=tool_schema, tokenizer=tokenizer)
    state = decoder.start()
    
    # Encode prompt
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for tool call generation. Install with: pip install torch")
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    if device and hasattr(input_ids, "to"):
        input_ids = input_ids.to(device)
    
    generated_tokens = []
    max_new_tokens = 512
    
    # Determine if this is CoreML or PyTorch model
    is_coreml = hasattr(model, "prediction")
    
    for _ in range(max_new_tokens):
        if is_coreml:
            # CoreML inference
            # Note: This is a simplified version - actual CoreML inference
            # would need proper input/output handling
            raise NotImplementedError("CoreML inference not yet implemented. Use PyTorch model for now.")
        else:
            # PyTorch inference
            model.eval()
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                next_token_logits = logits[0, -1, :].cpu().numpy()  # [V]
        
        # Apply constrained decoding mask
        mask = decoder.allowed_token_mask(state, next_token_logits.shape)
        next_token_logits[~mask] = -float("inf")
        
        # Sample token (greedy)
        tok_id = int(next_token_logits.argmax())
        generated_tokens.append(tok_id)
        
        # Update decoder state
        state = decoder.push(state, tok_id)
        
        # Update input_ids for next iteration
        if not is_coreml and TORCH_AVAILABLE:
            input_ids = torch.cat([input_ids, torch.tensor([[tok_id]], device=input_ids.device)], dim=1)
        
        # Stop if decoder says we're complete
        if state.complete:
            break
        
        # Stop on EOS token
        if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            if tok_id == tokenizer.eos_token_id:
                break
    
    # Finalize and validate
    try:
        tool_call = decoder.finalize(state)
        return tool_call
    except ValueError as e:
        # Invalid JSON - try to extract anyway
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        tool_call = extract_tool_call_fallback(generated_text)
        if tool_call:
            return tool_call
        raise ValueError(f"Failed to generate valid tool call: {e}")


def extract_tool_call_fallback(text: str) -> Optional[Dict[str, Any]]:
    """Fallback extraction if constrained decoder fails."""
    import re
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text)
    
    for match in matches:
        try:
            obj = json.loads(match)
            if isinstance(obj, dict) and 'name' in obj:
                return obj
        except:
            continue
    
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict) and 'name' in obj:
            return obj
    except:
        pass
    
    return None


def generate_text_streaming(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 1024,
    device: Optional[str] = None,
    use_constrained_decoding: bool = False,
    schema: Optional[Dict[str, Any]] = None,
):
    """
    Generate text with streaming support.
    
    Args:
        model: CoreML or PyTorch model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        device: Device (for PyTorch)
        use_constrained_decoding: Whether to use constrained decoding
        schema: JSON schema for constrained decoding (if use_constrained_decoding=True)
        
    Yields:
        Generated text chunks
    """
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    if device and hasattr(input_ids, "to"):
        input_ids = input_ids.to(device)
    
    decoder = None
    state = None
    if use_constrained_decoding and schema:
        decoder = JSONConstrainedDecoder(schema=schema, tokenizer=tokenizer)
        state = decoder.start()
    
    generated_tokens = []
    
    # Determine if this is CoreML or PyTorch model
    is_coreml = hasattr(model, "prediction")
    
    for _ in range(max_new_tokens):
        if is_coreml:
            # CoreML inference (stub)
            raise NotImplementedError("CoreML streaming not yet implemented")
        else:
            # PyTorch inference
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch required for text generation. Install with: pip install torch")
            model.eval()
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                next_token_logits = logits[0, -1, :].cpu().numpy()  # [V]
        
        # Apply constrained decoding mask if enabled
        if decoder and state:
            mask = decoder.allowed_token_mask(state, next_token_logits.shape)
            next_token_logits[~mask] = -float("inf")
        
        # Sample token (greedy)
        tok_id = int(next_token_logits.argmax())
        generated_tokens.append(tok_id)
        
        # Update decoder state if using constrained decoding
        if decoder and state:
            state = decoder.push(state, tok_id)
            if state.complete:
                break
        
        # Update input_ids for next iteration
        if not is_coreml and TORCH_AVAILABLE:
            input_ids = torch.cat([input_ids, torch.tensor([[tok_id]], device=input_ids.device)], dim=1)
        
        # Decode and yield chunk
        chunk = tokenizer.decode([tok_id], skip_special_tokens=True)
        yield chunk
        
        # Stop on EOS token
        if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            if tok_id == tokenizer.eos_token_id:
                break


def main():
    """Example usage."""
    import argparse
    
    ap = argparse.ArgumentParser(description="CoreML runtime generator")
    ap.add_argument("--model", required=True, help="Path to .mlpackage or PyTorch checkpoint")
    ap.add_argument("--tokenizer", required=True, help="Path to tokenizer")
    ap.add_argument("--prompt", required=True, help="Input prompt")
    ap.add_argument("--tools", help="Path to tools manifest JSON")
    ap.add_argument("--max-tokens", type=int, default=512, help="Max tokens to generate")
    ap.add_argument("--use-constrained", action="store_true", help="Use constrained decoding for tool calls")
    
    args = ap.parse_args()
    
    # Load tokenizer
    from training.dataset import load_tokenizer
    tokenizer = load_tokenizer(args.tokenizer)
    
    # Load model
    if args.model.endswith(".mlpackage"):
        model = load_coreml_model(args.model)
    else:
        # PyTorch model
        import torch
        from models.student.architectures.gqa_transformer import StudentLM, ModelCfg
        # Load checkpoint and create model (simplified)
        checkpoint = torch.load(args.model, map_location="cpu")
        # ... model loading logic ...
        model = None  # Placeholder
    
    # Load tools if provided
    tools = []
    if args.tools:
        with open(args.tools, 'r') as f:
            tools = json.load(f)
    
    # Generate
    if args.use_constrained and tools:
        # Generate tool call with constrained decoding
        tool_call = generate_tool_call(model, tokenizer, args.prompt, tools)
        print(json.dumps(tool_call, indent=2))
    else:
        # Generate text normally
        for chunk in generate_text_streaming(model, tokenizer, args.prompt, args.max_tokens):
            print(chunk, end="", flush=True)
        print()


if __name__ == "__main__":
    main()
