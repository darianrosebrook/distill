"""
Creates a tiny training setup for smoke testing the training pipeline.

This creates:
1. A tiny student model (small enough to train quickly)
2. A tiny dataset (just enough samples for a few training steps)
3. A minimal config for testing

Usage:
    python -m training.make_toy_training \
        --out-dir training/toy_test \
        --samples 10 \
        --steps 5
"""
import argparse
import json
from pathlib import Path

import torch

from models.student.architectures.gqa_transformer import StudentLM, ModelCfg


def create_toy_model_config(d_model: int = 64, n_layers: int = 2, vocab_size: int = 32000) -> ModelCfg:
    """Create a tiny model configuration for testing."""
    return ModelCfg(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=4,
        n_kv_heads=2,  # GQA with 2:1 ratio
        d_head=d_model // 4,
        vocab_size=vocab_size,
        rope_theta=10000.0,
        rope_scaling="dynamic",
        dropout=0.0,
    )


def create_toy_dataset(output_path: Path, num_samples: int = 10, vocab_size: int = 32000):
    """Create a tiny KD dataset for testing."""
    samples = []
    
    # Simple prompts and responses
    prompts = [
        "What is 2+2?",
        "Write a hello world program.",
        "Explain recursion.",
        "What is a variable?",
        "How does a loop work?",
        "What is a function?",
        "Explain OOP.",
        "What is an API?",
        "How does HTTP work?",
        "What is JSON?",
    ]
    
    responses = [
        "2 + 2 equals 4.",
        "print('Hello, world!')",
        "Recursion is when a function calls itself.",
        "A variable stores a value.",
        "A loop repeats code multiple times.",
        "A function is reusable code.",
        "OOP organizes code into objects.",
        "An API is an interface for programs.",
        "HTTP is a protocol for web communication.",
        "JSON is a data format.",
    ]
    
    for i in range(num_samples):
        prompt_idx = i % len(prompts)
        sample = {
            "prompt": prompts[prompt_idx],
            "teacher_text": responses[prompt_idx],
            "metadata": {
                "source": "toy_test",
                "tokens": {
                    "input": len(prompts[prompt_idx].split()),
                    "output": len(responses[prompt_idx].split()),
                }
            }
        }
        samples.append(sample)
    
    # Write JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"[make_toy_training] Created toy dataset: {output_path} ({num_samples} samples)")


def create_toy_config(
    output_path: Path,
    model_cfg: ModelCfg,
    dataset_path: Path,
    tokenizer_path: str,
    num_steps: int = 5,
):
    """Create a minimal training config for testing."""
    config = {
        "arch": {
            "d_model": model_cfg.d_model,
            "n_layers": model_cfg.n_layers,
            "n_heads": model_cfg.n_heads,
            "n_kv_heads": model_cfg.n_kv_heads,
            "d_head": model_cfg.d_head,
            "vocab_size": model_cfg.vocab_size,
            "rope_theta": model_cfg.rope_theta,
            "rope_scaling": model_cfg.rope_scaling,
            "dropout": model_cfg.dropout,
        },
        "init": {
            "base_checkpoint": None,
        },
        "optimizer": {
            "name": "adamw",
            "lr": 1e-4,  # Smaller LR for toy test
            "betas": [0.9, 0.95],
            "weight_decay": 0.1,
        },
        "train": {
            "seq_lengths": [128],  # Small sequence length
            "micro_batch_size": 1,  # Small batch
            "grad_accum": 2,  # Effective batch = 2
            "steps": num_steps,
            "fp16": False,  # Use FP32 for simplicity
            "grad_checkpointing": False,  # Not needed for tiny model
        },
        "io": {
            "tokenizer_path": tokenizer_path,
            "train_shards": [str(dataset_path)],
            "val_shards": [],
        },
        "role": "worker",
        "distillation": {
            "type": "standard_kd",
            "kl_weight": 0.5,
            "ce_teacher_weight": 0.3,
            "ce_ground_truth_weight": 0.2,
        },
        "kd": {
            "teacher_endpoint": "mock",  # Won't be used in toy test
            "kd_temperature": 2.0,
            "teacher_logits_available": False,
        },
        "tracing": {
            "log_dir": str(output_path / "runs"),
            "use_tensorboard": False,  # Disable for quick tests
            "use_wandb": False,
            "json_log": True,
            "console_log": True,
        },
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import yaml
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"[make_toy_training] Created toy config: {output_path}")


def main():
    ap = argparse.ArgumentParser(description="Create toy training setup for smoke testing")
    ap.add_argument('--out-dir', type=str, default='training/toy_test', help='Output directory')
    ap.add_argument('--samples', type=int, default=10, help='Number of dataset samples')
    ap.add_argument('--steps', type=int, default=5, help='Number of training steps')
    ap.add_argument('--dmodel', type=int, default=64, help='Model dimension')
    ap.add_argument('--nlayers', type=int, default=2, help='Number of layers')
    ap.add_argument('--vocab', type=int, default=32000, help='Vocabulary size (must match tokenizer)')
    ap.add_argument('--tokenizer', type=str, default='models/student/tokenizer', help='Tokenizer path')
    args = ap.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model config
    model_cfg = create_toy_model_config(
        d_model=args.dmodel,
        n_layers=args.nlayers,
        vocab_size=args.vocab,
    )
    
    # Create dataset
    dataset_path = out_dir / "toy_dataset.jsonl"
    create_toy_dataset(dataset_path, num_samples=args.samples, vocab_size=args.vocab)
    
    # Create config
    config_path = out_dir / "toy_config.yaml"
    create_toy_config(
        config_path,
        model_cfg,
        dataset_path,
        args.tokenizer,
        num_steps=args.steps,
    )
    
    # Create model checkpoint (initialized randomly)
    model = StudentLM(model_cfg)
    checkpoint_path = out_dir / "toy_model_init.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": model_cfg.__dict__,
    }, checkpoint_path)
    print(f"[make_toy_training] Created toy model: {checkpoint_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("Toy Training Setup Created")
    print("="*60)
    print(f"Output directory: {out_dir}")
    print(f"Dataset: {dataset_path} ({args.samples} samples)")
    print(f"Config: {config_path}")
    print(f"Model: {checkpoint_path}")
    print("\nModel specs:")
    print(f"  d_model: {args.dmodel}")
    print(f"  n_layers: {args.nlayers}")
    print(f"  vocab_size: {args.vocab}")
    print("\nTraining specs:")
    print(f"  Steps: {args.steps}")
    print("  Batch size: 1 * 2 = 2")
    print("  Sequence length: 128")
    print("\nTo test training:")
    print(f"  python -m training.distill_kd --config {config_path}")
    print("="*60)


if __name__ == '__main__':
    main()

