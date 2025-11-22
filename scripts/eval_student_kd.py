#!/usr/bin/env python3
"""
Minimal evaluation harness for student KD checkpoints.

Compares student model outputs vs teacher on a held-out evaluation set.
Computes simple metrics: perplexity, exact match rate, length ratio.

Usage:
    python scripts/eval_student_kd.py \
        --checkpoint models/student/checkpoints_3b_kd_test/checkpoint_step_1125.pt \
        --eval-jsonl data/test_data/kd_mix_test.jsonl \
        --tokenizer models/student/tokenizer \
        --output eval_results_step_1125.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import torch.nn.functional as F
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.student.architectures.gqa_transformer import StudentLM, ModelCfg
from training.dataset import load_tokenizer
from models.teacher.teacher_client import TeacherClient


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
    """Load checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def load_model_from_checkpoint(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
) -> tuple[StudentLM, ModelCfg]:
    """
    Load student model from checkpoint.
    
    Returns:
        (model, model_cfg)
    """
    checkpoint = load_checkpoint(checkpoint_path, device)
    
    # Get model config from checkpoint or config file
    if "model_cfg" in checkpoint:
        model_cfg_dict = checkpoint["model_cfg"]
    elif "config" in checkpoint:
        model_cfg_dict = checkpoint["config"]
    elif config_path:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        arch_cfg = cfg.get("arch", {})
        model_cfg_dict = arch_cfg
    else:
        raise ValueError("Could not find model config in checkpoint or config file")
    
    # Create ModelCfg
    model_cfg = ModelCfg(**model_cfg_dict)
    
    # Create model
    model = StudentLM(model_cfg)
    
    # Load state dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    
    return model, model_cfg


def load_eval_dataset(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load evaluation dataset from JSONL."""
    samples = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                # Skip header lines
                if "__header__" in data:
                    continue
                if "prompt" in data and "teacher_text" in data:
                    samples.append(data)
            except json.JSONDecodeError:
                continue
    return samples


def compute_perplexity(
    model: StudentLM,
    tokenizer: Any,
    text: str,
    device: torch.device,
    max_length: int = 1024,
) -> float:
    """
    Compute perplexity of model on given text.
    
    Returns:
        Perplexity (exp of cross-entropy loss)
    """
    # Tokenize
    tokens = tokenizer.encode(text, add_special_tokens=True, max_length=max_length, truncation=True)
    if len(tokens) < 2:
        return float("inf")
    
    # Convert to tensor
    input_ids = torch.tensor([tokens], device=device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # [1, T, V]
        
        # Compute cross-entropy loss
        # Shift labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        # Flatten for loss computation
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        # Cross-entropy loss
        loss = F.cross_entropy(shift_logits, shift_labels, reduction="mean")
        
        # Perplexity = exp(loss)
        perplexity = torch.exp(loss).item()
    
    return perplexity


def generate_text(
    model: StudentLM,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
) -> str:
    """
    Generate text from model given prompt using simple greedy/random sampling.
    
    Returns:
        Generated text (without prompt)
    """
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt").to(device)
    prompt_length = input_ids.size(1)
    
    # Generate tokens one at a time
    model.eval()
    generated_ids = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass
            outputs = model(generated_ids)
            logits = outputs.logits[:, -1, :]  # [1, V] - last token logits
            
            # Sample next token
            if temperature > 0:
                # Apply temperature
                logits = logits / temperature
                # Sample from distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = logits.argmax(dim=-1, keepdim=True)
            
            # Append to sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Stop if EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode only the new tokens
    new_tokens = generated_ids[0, prompt_length:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    return generated_text


def evaluate_checkpoint(
    checkpoint_path: str,
    eval_jsonl_path: str,
    tokenizer_path: str,
    device: torch.device,
    config_path: Optional[str] = None,
    teacher_cache_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate checkpoint on evaluation set.
    
    Returns:
        Dict with evaluation metrics
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    model, model_cfg = load_model_from_checkpoint(checkpoint_path, config_path, device)
    
    print(f"Loading tokenizer: {tokenizer_path}")
    tokenizer = load_tokenizer(tokenizer_path)
    
    print(f"Loading evaluation set: {eval_jsonl_path}")
    eval_samples = load_eval_dataset(eval_jsonl_path)
    if max_samples:
        eval_samples = eval_samples[:max_samples]
    print(f"Evaluating on {len(eval_samples)} samples")
    
    # Metrics
    perplexities = []
    exact_matches = 0
    length_ratios = []
    teacher_perplexities = []
    
    print("\nEvaluating samples...")
    for i, sample in enumerate(eval_samples):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(eval_samples)} samples")
        
        prompt = sample["prompt"]
        teacher_text = sample.get("teacher_text", "")
        
        if not teacher_text:
            continue
        
        # Compute perplexity on teacher text
        try:
            ppl = compute_perplexity(model, tokenizer, teacher_text, device)
            perplexities.append(ppl)
        except Exception as e:
            print(f"  WARN: Failed to compute perplexity for sample {i}: {e}")
            continue
        
        # Generate from student
        try:
            student_text = generate_text(model, tokenizer, prompt, device)
        except Exception as e:
            print(f"  WARN: Failed to generate for sample {i}: {e}")
            student_text = ""
        
        # Exact match
        if student_text.strip() == teacher_text.strip():
            exact_matches += 1
        
        # Length ratio
        student_len = len(student_text.split())
        teacher_len = len(teacher_text.split())
        if teacher_len > 0:
            length_ratios.append(student_len / teacher_len)
    
    # Compute aggregate metrics
    results = {
        "checkpoint_path": checkpoint_path,
        "eval_set_path": eval_jsonl_path,
        "num_samples": len(eval_samples),
        "num_evaluated": len(perplexities),
        "metrics": {
            "perplexity": {
                "mean": sum(perplexities) / len(perplexities) if perplexities else float("inf"),
                "median": sorted(perplexities)[len(perplexities) // 2] if perplexities else float("inf"),
                "min": min(perplexities) if perplexities else float("inf"),
                "max": max(perplexities) if perplexities else float("inf"),
            },
            "exact_match_rate": exact_matches / len(eval_samples) if eval_samples else 0.0,
            "length_ratio": {
                "mean": sum(length_ratios) / len(length_ratios) if length_ratios else 0.0,
                "median": sorted(length_ratios)[len(length_ratios) // 2] if length_ratios else 0.0,
            },
        },
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate student KD checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--eval-jsonl",
        type=str,
        required=True,
        help="Path to evaluation JSONL file",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="models/student/tokenizer",
        help="Path to tokenizer",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML (optional, uses checkpoint config if not provided)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (prints to stdout if not provided)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to use for evaluation",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for quick testing)",
    )
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Evaluate
    results = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        eval_jsonl_path=args.eval_jsonl,
        tokenizer_path=args.tokenizer,
        device=device,
        config_path=args.config,
        max_samples=args.max_samples,
    )
    
    # Print results
    print("\n=== Evaluation Results ===")
    print(f"Checkpoint: {results['checkpoint_path']}")
    print(f"Samples evaluated: {results['num_evaluated']}/{results['num_samples']}")
    print(f"\nPerplexity:")
    print(f"  Mean: {results['metrics']['perplexity']['mean']:.2f}")
    print(f"  Median: {results['metrics']['perplexity']['median']:.2f}")
    print(f"  Range: [{results['metrics']['perplexity']['min']:.2f}, {results['metrics']['perplexity']['max']:.2f}]")
    print(f"\nExact Match Rate: {results['metrics']['exact_match_rate']:.2%}")
    print(f"\nLength Ratio (student/teacher):")
    print(f"  Mean: {results['metrics']['length_ratio']['mean']:.2f}")
    print(f"  Median: {results['metrics']['length_ratio']['median']:.2f}")
    
    # Write output
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to: {args.output}")
    else:
        print("\n=== Full Results (JSON) ===")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

