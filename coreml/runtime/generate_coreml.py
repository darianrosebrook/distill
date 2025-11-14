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
import json
import numpy as np

try:
    import coremltools as ct

    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

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
    return_halt_logits: bool = False,
) -> Dict[str, Any]:
    """
    Generate a tool call using constrained decoding.

    Args:
        model: CoreML model or PyTorch model
        tokenizer: Tokenizer for encoding/decoding
        prompt: Input prompt
        tools: List of available tools with schemas
        device: Device to run on (for PyTorch models)
        return_halt_logits: If True, include halt logits in return value

    Returns:
        Dictionary with tool call: {"name": "...", "arguments": {...}}
        If return_halt_logits=True, also includes "halt_logits" key
    """
    # Create generic tool call schema
    tool_schema = {
        "type": "object",
        "required": ["name", "arguments"],
        "properties": {"name": {"type": "string"}, "arguments": {"type": "object"}},
    }

    # Create constrained decoder
    decoder = JSONConstrainedDecoder(schema=tool_schema, tokenizer=tokenizer)
    state = decoder.start()

    # Encode prompt
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch required for tool call generation. Install with: pip install torch"
        )

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    if device and hasattr(input_ids, "to"):
        input_ids = input_ids.to(device)

    generated_tokens = []
    max_new_tokens = 512
    halt_logits = None  # Track halt logits throughout generation

    # Determine if this is CoreML or PyTorch model
    is_coreml = hasattr(model, "prediction") or (
        COREML_AVAILABLE and isinstance(model, ct.models.MLModel)
    )

    # For CoreML, we need to track input_ids as numpy array and handle KV cache
    if is_coreml:
        # Convert input_ids to numpy for CoreML
        if TORCH_AVAILABLE and isinstance(input_ids, torch.Tensor):
            input_ids_np = input_ids.cpu().numpy().astype(np.int32)
        else:
            input_ids_np = np.array(input_ids, dtype=np.int32)
        # Track current sequence length for CoreML
        current_seq_len = input_ids_np.shape[1]
        # Initialize attention mask (all ones for prompt)
        attention_mask = np.ones((1, current_seq_len), dtype=np.int32)
    else:
        input_ids_np = None
        current_seq_len = None
        attention_mask = None

    for _ in range(max_new_tokens):
        if is_coreml:
            # CoreML inference
            # Prepare input dict for CoreML model
            inputs = {
                "input_ids": input_ids_np.astype(np.int32),
                "attention_mask": attention_mask.astype(np.int32),
            }

            # Run inference
            try:
                outputs = model.predict(inputs)
            except Exception as e:
                raise RuntimeError(f"CoreML inference failed: {e}")

            # Extract logits from output
            # Try common output key names
            if "logits" in outputs:
                logits = outputs["logits"]
            else:
                # Try to find logits key (case-insensitive)
                logits_key = None
                for key in outputs.keys():
                    if "logit" in key.lower() and "halt" not in key.lower():
                        logits_key = key
                        break
                if logits_key:
                    logits = outputs[logits_key]
                else:
                    raise ValueError(
                        f"Could not find logits in CoreML model output. Available keys: {list(outputs.keys())}"
                    )

            # Extract halt logits if available (update on each iteration)
            if "halt_logits" in outputs:
                halt_logits = outputs["halt_logits"]
            else:
                # Try to find halt logits key (case-insensitive)
                halt_key = None
                for key in outputs.keys():
                    if "halt" in key.lower() and "logit" in key.lower():
                        halt_key = key
                        break
                if halt_key:
                    halt_logits = outputs[halt_key]

            # Convert to numpy if needed and extract last token logits
            if isinstance(logits, np.ndarray):
                # Handle different output shapes: [B, T, V] or [T, V] or [V]
                if len(logits.shape) == 3:
                    # [B, T, V] -> take first batch and last token
                    next_token_logits = logits[0, -1, :]
                elif len(logits.shape) == 2:
                    # [T, V] -> take last token
                    next_token_logits = logits[-1, :]
                elif len(logits.shape) == 1:
                    # [V] -> already single token
                    next_token_logits = logits
                else:
                    raise ValueError(f"Unexpected logits shape: {logits.shape}")
            else:
                # Convert to numpy if it's a different type
                next_token_logits = np.array(logits)
                if len(next_token_logits.shape) > 1:
                    # Flatten and take last vocab_size elements if shape is wrong
                    next_token_logits = next_token_logits.flatten()
                    if (
                        hasattr(tokenizer, "vocab_size")
                        and len(next_token_logits) > tokenizer.vocab_size
                    ):
                        next_token_logits = next_token_logits[-tokenizer.vocab_size :]
        else:
            # PyTorch inference
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch required for PyTorch model inference")
            model.eval()
            with torch.no_grad():
                # Check if model supports halt head
                if hasattr(model, "use_halt_head") and model.use_halt_head and return_halt_logits:
                    outputs = model(input_ids, return_halt_logits=True)
                    if isinstance(outputs, tuple) and len(outputs) == 2:
                        logits, halt_logits_tensor = outputs
                        # Extract halt logits from tensor
                        if isinstance(halt_logits_tensor, torch.Tensor):
                            halt_logits = halt_logits_tensor.cpu().numpy()
                        else:
                            halt_logits = halt_logits_tensor
                    else:
                        logits = outputs[0] if isinstance(outputs, tuple) else outputs
                else:
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
        if is_coreml:
            # For CoreML, append token to input_ids_np and update attention_mask
            new_token = np.array([[tok_id]], dtype=np.int32)
            input_ids_np = np.concatenate([input_ids_np, new_token], axis=1)
            attention_mask = np.concatenate(
                [attention_mask, np.ones((1, 1), dtype=np.int32)], axis=1
            )
            current_seq_len += 1
        elif TORCH_AVAILABLE:
            input_ids = torch.cat(
                [input_ids, torch.tensor([[tok_id]], device=input_ids.device)], dim=1
            )

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

        # Add halt logits to return value if requested and available
        if return_halt_logits and halt_logits is not None:
            # Convert halt logits to appropriate format
            if isinstance(halt_logits, np.ndarray):
                # Take the last halt logits if shape is [B, 2] or [T, 2]
                if len(halt_logits.shape) == 2:
                    if halt_logits.shape[0] > 1:
                        # [B, 2] or [T, 2] -> take last
                        halt_logits_final = halt_logits[-1]
                    else:
                        halt_logits_final = halt_logits[0]
                else:
                    halt_logits_final = halt_logits
                tool_call["halt_logits"] = halt_logits_final.tolist()
            else:
                tool_call["halt_logits"] = halt_logits

        return tool_call
    except ValueError as e:
        # Invalid JSON - try to extract anyway
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        tool_call = extract_tool_call_fallback(generated_text)
        if tool_call:
            # Add halt logits if available
            if return_halt_logits and halt_logits is not None:
                if isinstance(halt_logits, np.ndarray):
                    if len(halt_logits.shape) == 2:
                        halt_logits_final = (
                            halt_logits[-1] if halt_logits.shape[0] > 1 else halt_logits[0]
                        )
                    else:
                        halt_logits_final = halt_logits
                    tool_call["halt_logits"] = halt_logits_final.tolist()
                else:
                    tool_call["halt_logits"] = halt_logits
            return tool_call
        raise ValueError(f"Failed to generate valid tool call: {e}")


def extract_tool_call_fallback(text: str) -> Optional[Dict[str, Any]]:
    """Fallback extraction if constrained decoder fails."""
    import re

    json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
    matches = re.findall(json_pattern, text)

    for match in matches:
        try:
            obj = json.loads(match)
            if isinstance(obj, dict) and "name" in obj:
                return obj
        except json.JSONDecodeError:
            continue

    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict) and "name" in obj:
            return obj
    except json.JSONDecodeError:
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
    is_coreml = hasattr(model, "prediction") or (
        COREML_AVAILABLE and isinstance(model, ct.models.MLModel)
    )

    # For CoreML, we need to track input_ids as numpy array
    if is_coreml:
        # Convert input_ids to numpy for CoreML
        if TORCH_AVAILABLE and isinstance(input_ids, torch.Tensor):
            input_ids_np = input_ids.cpu().numpy().astype(np.int32)
        else:
            input_ids_np = np.array(input_ids, dtype=np.int32)
        # Track current sequence length for CoreML
        current_seq_len = input_ids_np.shape[1]
        # Initialize attention mask (all ones for prompt)
        attention_mask = np.ones((1, current_seq_len), dtype=np.int32)
    else:
        input_ids_np = None
        current_seq_len = None
        attention_mask = None

    for _ in range(max_new_tokens):
        if is_coreml:
            # CoreML inference
            # Prepare input dict for CoreML model
            inputs = {
                "input_ids": input_ids_np.astype(np.int32),
                "attention_mask": attention_mask.astype(np.int32),
            }

            # Run inference
            try:
                outputs = model.predict(inputs)
            except Exception as e:
                raise RuntimeError(f"CoreML inference failed: {e}")

            # Extract logits from output
            if "logits" in outputs:
                logits = outputs["logits"]
            else:
                # Try to find logits key (case-insensitive)
                logits_key = None
                for key in outputs.keys():
                    if "logit" in key.lower():
                        logits_key = key
                        break
                if logits_key:
                    logits = outputs[logits_key]
                else:
                    raise ValueError(
                        f"Could not find logits in CoreML model output. Available keys: {list(outputs.keys())}"
                    )

            # Convert to numpy if needed and extract last token logits
            if isinstance(logits, np.ndarray):
                # Handle different output shapes: [B, T, V] or [T, V] or [V]
                if len(logits.shape) == 3:
                    next_token_logits = logits[0, -1, :]
                elif len(logits.shape) == 2:
                    next_token_logits = logits[-1, :]
                elif len(logits.shape) == 1:
                    next_token_logits = logits
                else:
                    raise ValueError(f"Unexpected logits shape: {logits.shape}")
            else:
                # Convert to numpy if it's a different type
                next_token_logits = np.array(logits)
                if len(next_token_logits.shape) > 1:
                    next_token_logits = next_token_logits.flatten()
                    if (
                        hasattr(tokenizer, "vocab_size")
                        and len(next_token_logits) > tokenizer.vocab_size
                    ):
                        next_token_logits = next_token_logits[-tokenizer.vocab_size :]
        else:
            # PyTorch inference
            if not TORCH_AVAILABLE:
                raise ImportError(
                    "PyTorch required for text generation. Install with: pip install torch"
                )
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
        if is_coreml:
            # For CoreML, append token to input_ids_np and update attention_mask
            new_token = np.array([[tok_id]], dtype=np.int32)
            input_ids_np = np.concatenate([input_ids_np, new_token], axis=1)
            attention_mask = np.concatenate(
                [attention_mask, np.ones((1, 1), dtype=np.int32)], axis=1
            )
            current_seq_len += 1
        elif TORCH_AVAILABLE:
            input_ids = torch.cat(
                [input_ids, torch.tensor([[tok_id]], device=input_ids.device)], dim=1
            )

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
    ap.add_argument(
        "--use-constrained", action="store_true", help="Use constrained decoding for tool calls"
    )

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
        from training.safe_checkpoint_loading import safe_load_checkpoint

        # Load checkpoint and create model
        checkpoint = safe_load_checkpoint(args.model, map_location="cpu")

        # Load config from checkpoint if available
        cfg = None
        if "config" in checkpoint:
            config_data = checkpoint["config"]
            arch_cfg = config_data.get("arch", {})
            cfg = ModelCfg(
                d_model=arch_cfg.get("d_model", 4096),
                n_layers=arch_cfg.get("n_layers", 32),
                n_heads=arch_cfg.get("n_heads", 32),
                n_kv_heads=arch_cfg.get("n_kv_heads", 8),
                d_head=arch_cfg.get("d_head", 128),
                vocab_size=arch_cfg.get("vocab_size", 32000),
                rope_theta=arch_cfg.get("rope_theta", 10000.0),
                rope_scaling=arch_cfg.get("rope_scaling", "dynamic"),
                dropout=arch_cfg.get("dropout", 0.0),
            )

        if cfg is None:
            cfg = ModelCfg()

        model = StudentLM(cfg)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

        model.eval()

        # Set device for PyTorch model (not CoreML)
        if not isinstance(model, str):  # CoreML models are loaded as file paths
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            device_str = str(device)  # For passing to generate functions
        else:
            # CoreML model - no device needed
            device_str = None

    # Load tools if provided
    tools = []
    if args.tools:
        with open(args.tools, "r") as f:
            tools = json.load(f)

    # Generate
    if args.use_constrained and tools:
        # Generate tool call with constrained decoding
        tool_call = generate_tool_call(model, tokenizer, args.prompt, tools, device=device_str)
        print(json.dumps(tool_call, indent=2))
    else:
        # Generate text normally
        for chunk in generate_text_streaming(
            model, tokenizer, args.prompt, args.max_tokens, device=device_str
        ):
            print(chunk, end="", flush=True)
        print()


if __name__ == "__main__":
    main()
