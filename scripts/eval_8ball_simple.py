#!/usr/bin/env python3
"""Simple 8-ball evaluation script"""

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.dataset import load_tokenizer

def main():
    # Load model
    checkpoint = torch.load('outputs/8ball_demo/model.pt', map_location='cpu')
    model = checkpoint['model']
    model.eval()
    
    tokenizer = load_tokenizer()
    
    # Test a simple prompt
    prompt = "Will this work?"
    print(f"Testing prompt: {prompt}")
    
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids']
    
    with torch.no_grad():
        outputs = model(input_ids)
        next_token_logits = outputs[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1)
        
        response_token = tokenizer.decode([next_token_id.item()])
        print(f"Model response: {response_token}")
        
        # Check if it's a reasonable 8-ball answer
        mystical_answers = ["yes", "no", "maybe", "certain", "doubt", "outlook"]
        is_reasonable = any(ans in response_token.lower() for ans in mystical_answers)
        print(f"Is reasonable 8-ball answer: {is_reasonable}")

if __name__ == "__main__":
    main()
EOF && echo "✅ Created simple evaluation script" && echo "" && echo "🔬 RUNNING SIMPLE EVALUATION..." && source venv/bin/activate && PYTHONPATH=/Users/darianrosebrook/Desktop/Projects/distill python scripts/eval_8ball_simple.py