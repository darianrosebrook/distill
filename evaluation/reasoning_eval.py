"""
Reasoning evaluation: Test model reasoning capabilities.

Evaluates model on reasoning tasks:
- Mathematical reasoning
- Logical reasoning
- Step-by-step problem solving

Usage:
    python -m evaluation.reasoning_eval --checkpoint models/student/checkpoints/latest.pt --config configs/worker_9b.yaml
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn as nn

from models.student.architectures.gqa_transformer import StudentLM, ModelCfg


def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load config from checkpoint
    cfg = None
    if 'config' in checkpoint:
        config_data = checkpoint['config']
        arch_cfg = config_data.get('arch', {})
        cfg = ModelCfg(
            d_model=arch_cfg.get('d_model', 4096),
            n_layers=arch_cfg.get('n_layers', 32),
            n_heads=arch_cfg.get('n_heads', 32),
            n_kv_heads=arch_cfg.get('n_kv_heads', 8),
            d_head=arch_cfg.get('d_head', 128),
            vocab_size=arch_cfg.get('vocab_size', 32000),
            rope_theta=arch_cfg.get('rope_theta', 10000.0),
            rope_scaling=arch_cfg.get('rope_scaling', 'dynamic'),
            dropout=arch_cfg.get('dropout', 0.0),
        )
    
    if cfg is None:
        cfg = ModelCfg()
    
    model = StudentLM(cfg)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(device)
    model.eval()
    return model


def generate_text(model: nn.Module, tokenizer, prompt: str, max_new_tokens: int = 512,
                  device: torch.device = None) -> str:
    """Generate text from model using greedy decoding."""
    if device is None:
        device = next(model.parameters()).device
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=False)
    input_ids = inputs["input_ids"].to(device)
    
    # Generate tokens
    generated_ids = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass
            logits = model(input_ids=generated_ids, attn_mask=None)
            
            # Get next token (greedy)
            next_token_id = logits[0, -1, :].argmax(dim=-1).unsqueeze(0).unsqueeze(0)
            
            # Append to sequence
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            
            # Check for EOS token
            if tokenizer.eos_token_id and next_token_id.item() == tokenizer.eos_token_id:
                break
    
    # Decode
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # Remove prompt from output
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()
    
    return generated_text


def get_reasoning_prompts() -> List[Dict[str, Any]]:
    """Get default reasoning test prompts."""
    return [
        {
            'prompt': 'If all roses are flowers, and some flowers are red, can we conclude that some roses are red?',
            'category': 'logical',
        },
        {
            'prompt': 'A train leaves Station A at 60 mph. Another train leaves Station B at 80 mph. They are 200 miles apart. When will they meet?',
            'category': 'mathematical',
        },
        {
            'prompt': 'If you have 5 apples and eat 2, then buy 3 more, how many apples do you have?',
            'category': 'mathematical',
        },
        {
            'prompt': 'What is the next number in the sequence: 2, 6, 12, 20, 30, ?',
            'category': 'mathematical',
        },
        {
            'prompt': 'A rectangle has length 10 and width 5. What is its area and perimeter?',
            'category': 'mathematical',
        },
        {
            'prompt': 'If x + 5 = 12, what is x?',
            'category': 'mathematical',
        },
        {
            'prompt': 'What is the sum of the first 10 natural numbers?',
            'category': 'mathematical',
        },
        {
            'prompt': 'If a car travels 120 miles in 2 hours, what is its average speed?',
            'category': 'mathematical',
        },
        {
            'prompt': 'What is the square root of 144?',
            'category': 'mathematical',
        },
        {
            'prompt': 'If you flip a coin 3 times, what is the probability of getting exactly 2 heads?',
            'category': 'mathematical',
        },
    ]


def evaluate_reasoning(model: nn.Module, tokenizer, test_prompts: List[Dict[str, Any]],
                      device: torch.device) -> Dict[str, Any]:
    """
    Evaluate reasoning capabilities.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer for encoding/decoding
        test_prompts: List of dicts with 'prompt' and 'category'
        device: Device to run on
        
    Returns:
        Dictionary with metrics
    """
    total = len(test_prompts)
    results = []
    
    for i, test_case in enumerate(test_prompts):
        prompt = test_case.get('prompt', '')
        category = test_case.get('category', 'unknown')
        
        # Generate text
        generated_text = generate_text(model, tokenizer, prompt, max_new_tokens=512, device=device)
        
        # Simple quality check: non-empty response
        is_valid = len(generated_text.strip()) > 0
        
        results.append({
            'prompt': prompt,
            'category': category,
            'generated': generated_text,
            'valid': is_valid,
        })
        
        if (i + 1) % 10 == 0:
            print(f"[reasoning_eval] Processed {i + 1}/{total}")
    
    valid_count = sum(1 for r in results if r['valid'])
    validity_rate = valid_count / total if total > 0 else 0.0
    
    # Group by category
    category_stats = {}
    for result in results:
        cat = result['category']
        if cat not in category_stats:
            category_stats[cat] = {'total': 0, 'valid': 0}
        category_stats[cat]['total'] += 1
        if result['valid']:
            category_stats[cat]['valid'] += 1
    
    for cat in category_stats:
        stats = category_stats[cat]
        stats['validity_rate'] = stats['valid'] / stats['total'] if stats['total'] > 0 else 0.0
    
    return {
        'validity_rate': validity_rate,
        'valid_count': valid_count,
        'total': total,
        'category_stats': category_stats,
        'results': results,
    }


def main():
    ap = argparse.ArgumentParser(description="Reasoning evaluation")
    ap.add_argument('--checkpoint', required=True, help='Model checkpoint path')
    ap.add_argument('--config', nargs='+', help='Config file(s)')
    ap.add_argument('--test-data', help='Test data JSONL (optional)')
    ap.add_argument('--output', default='reports/reasoning_eval.json', help='Output report path')
    ap.add_argument('--tokenizer', default='models/student/tokenizer', help='Tokenizer path')
    args = ap.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load tokenizer
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    except ImportError:
        raise RuntimeError("transformers required for evaluation")
    
    # Load model
    print(f"[reasoning_eval] Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    
    # Load test data
    if args.test_data and Path(args.test_data).exists():
        test_prompts = []
        with open(args.test_data, 'r') as f:
            for line in f:
                if line.strip():
                    test_prompts.append(json.loads(line))
    else:
        test_prompts = get_reasoning_prompts()
        print(f"[reasoning_eval] Using default reasoning prompts")
    
    # Run evaluation
    print(f"[reasoning_eval] Evaluating on {len(test_prompts)} test cases...")
    results = evaluate_reasoning(model, tokenizer, test_prompts, device)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        'checkpoint': args.checkpoint,
        'metrics': {
            'validity_rate': results['validity_rate'],
            'valid_count': results['valid_count'],
            'total': results['total'],
            'category_stats': results['category_stats'],
        },
        'results': results['results'],
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"[reasoning_eval] ✅ Evaluation complete:")
    print(f"  Validity rate: {results['validity_rate']:.2%}")
    print(f"  Category breakdown:")
    for cat, stats in results['category_stats'].items():
        print(f"    {cat}: {stats['validity_rate']:.2%} ({stats['valid']}/{stats['total']})")
    print(f"  Report saved to: {output_path}")


if __name__ == '__main__':
    main()
