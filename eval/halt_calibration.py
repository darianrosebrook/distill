"""
Halt head calibration evaluation.

Evaluates halt head performance on calibration dataset:
- ROC curve for halt decisions vs ground truth
- Expected effective length vs reference length
- Halt probability distribution across positions
@author: @darianrosebrook
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import roc_curve, auc

import torch

from models.student.architectures.gqa_transformer import StudentLM, ModelCfg
from training.dataset import load_tokenizer


def load_calibration_dataset(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Load halt calibration dataset."""
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def evaluate_halt_head(
    model: StudentLM, samples: List[Dict[str, Any]], tokenizer, device: torch.device
) -> Dict[str, Any]:
    """
    Evaluate halt head on calibration dataset.

    Returns:
        Dictionary with evaluation metrics
    """
    if not model.use_halt_head:
        raise ValueError("Model does not have halt head enabled")

    model.eval()

    halt_predictions = []
    halt_targets = []
    sequence_lengths = []
    reference_lengths = []

    with torch.no_grad():
        for sample in samples:
            # Get input text
            prompt = sample.get("prompt", "")
            reference_text = sample.get("reference_text", "")
            reference_length = sample.get("reference_length", len(reference_text.split()))

            # Tokenize
            full_text = prompt + reference_text
            encoded = tokenizer.encode(full_text, add_special_tokens=True)
            input_ids = torch.tensor([encoded], dtype=torch.long).to(device)

            # Get halt target (1 = should halt at end, 0 = continue)
            should_halt = sample.get("should_halt", True)  # Default: halt at end

            # Forward pass with halt head
            outputs = model(input_ids, return_halt_logits=True)
            if isinstance(outputs, tuple):
                logits, halt_logits = outputs
            else:
                halt_logits = None

            if halt_logits is not None:
                # Halt logits: [B, 2] where [0] = continue, [1] = halt
                halt_probs = torch.softmax(halt_logits, dim=-1)
                halt_prob = halt_probs[0, 1].item()  # Probability of halting

                halt_predictions.append(halt_prob)
                halt_targets.append(1 if should_halt else 0)
                sequence_lengths.append(len(encoded))
                reference_lengths.append(reference_length)

    # Compute metrics
    halt_predictions = np.array(halt_predictions)
    halt_targets = np.array(halt_targets)

    # ROC curve
    if len(np.unique(halt_targets)) > 1:  # Need both classes
        fpr, tpr, thresholds = roc_curve(halt_targets, halt_predictions)
        roc_auc = auc(fpr, tpr)
    else:
        fpr, tpr, thresholds = None, None, None
        roc_auc = None

    # Expected effective length (using halt probability as stopping probability)
    # This is a simplified metric - in practice, you'd simulate generation
    expected_lengths = []
    for i, (pred, ref_len) in enumerate(zip(halt_predictions, reference_lengths)):
        # Use halt probability to estimate expected length
        # Higher halt prob → shorter expected length
        expected_len = ref_len * (1 - pred) + sequence_lengths[i] * pred
        expected_lengths.append(expected_len)

    avg_expected_length = np.mean(expected_lengths)
    avg_reference_length = np.mean(reference_lengths)
    length_ratio = avg_expected_length / avg_reference_length if avg_reference_length > 0 else 0.0

    # Halt probability distribution
    halt_prob_mean = np.mean(halt_predictions)
    halt_prob_std = np.std(halt_predictions)

    results = {
        "roc_auc": roc_auc,
        "fpr": fpr.tolist() if fpr is not None else None,
        "tpr": tpr.tolist() if tpr is not None else None,
        "thresholds": thresholds.tolist() if thresholds is not None else None,
        "avg_expected_length": avg_expected_length,
        "avg_reference_length": avg_reference_length,
        "length_ratio": length_ratio,
        "halt_prob_mean": halt_prob_mean,
        "halt_prob_std": halt_prob_std,
        "num_samples": len(samples),
    }

    return results


def print_evaluation_report(results: Dict[str, Any]):
    """Print evaluation report."""
    print("\n" + "=" * 60)
    print("Halt Head Calibration Evaluation Report")
    print("=" * 60)
    print(f"\nDataset: {results['num_samples']} samples")

    if results["roc_auc"] is not None:
        print(f"\nROC AUC: {results['roc_auc']:.4f}")
        if results["roc_auc"] >= 0.7:
            print("  Status: GOOD (≥0.7)")
        elif results["roc_auc"] >= 0.5:
            print("  Status: FAIR (≥0.5)")
        else:
            print("  Status: POOR (<0.5)")
    else:
        print("\nROC AUC: Cannot compute (need both classes)")

    print("\nLength Metrics:")
    print(f"  Average expected length: {results['avg_expected_length']:.2f}")
    print(f"  Average reference length: {results['avg_reference_length']:.2f}")
    print(f"  Length ratio: {results['length_ratio']:.3f}")

    if results["length_ratio"] > 1.15:
        print("  WARNING: Expected length exceeds reference by >15%")
    elif results["length_ratio"] < 0.85:
        print("  WARNING: Expected length is <85% of reference (may stop too early)")
    else:
        print("  Status: Length ratio within acceptable range (0.85-1.15)")

    print("\nHalt Probability Distribution:")
    print(f"  Mean: {results['halt_prob_mean']:.4f}")
    print(f"  Std: {results['halt_prob_std']:.4f}")

    print("\n" + "=" * 60)


def main():
    ap = argparse.ArgumentParser(description="Evaluate halt head calibration")
    ap.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    ap.add_argument("--data", required=True, help="Calibration dataset JSONL path")
    ap.add_argument("--tokenizer", default="models/student/tokenizer", help="Tokenizer path")
    ap.add_argument("--output", help="Output JSON report path")

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "config" not in checkpoint:
        print("[halt_calibration] ERROR: Checkpoint missing config")
        sys.exit(1)

    config = checkpoint["config"]
    arch_cfg = config.get("arch", {})
    cfg = ModelCfg(
        d_model=arch_cfg.get("d_model", 4096),
        n_layers=arch_cfg.get("n_layers", 32),
        n_heads=arch_cfg.get("n_heads", 32),
        n_kv_heads=arch_cfg.get("n_kv_heads", 8),
        d_head=arch_cfg.get("d_head", 128),
        vocab_size=arch_cfg.get("vocab_size", 32000),
    )

    use_halt_head = config.get("use_halt_head", False)
    model = StudentLM(cfg, use_halt_head=use_halt_head)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.to(device)

    # Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer)

    # Load calibration dataset
    calibration_data = load_calibration_dataset(Path(args.data))

    # Evaluate
    results = evaluate_halt_head(model, calibration_data, tokenizer, device)

    # Print report
    print_evaluation_report(results)

    # Save results if output path specified
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[halt_calibration] Saved results to: {args.output}")

    # Exit with error if metrics are poor
    if results["roc_auc"] is not None and results["roc_auc"] < 0.5:
        print("\n[halt_calibration] FAILED: ROC AUC < 0.5")
        sys.exit(1)
    elif results["length_ratio"] > 1.15 or results["length_ratio"] < 0.85:
        print("\n[halt_calibration] WARNING: Length ratio outside acceptable range")
        sys.exit(1)
    else:
        print("\n[halt_calibration] PASSED: Calibration metrics acceptable")
        sys.exit(0)


if __name__ == "__main__":
    main()
