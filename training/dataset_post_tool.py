"""
Dataset loader for post-tool integration training stage.

Loads post_tool.jsonl format:
{
    "input": {
        "system": "...",
        "tools": [...],
        "history": [...],
        "tool_result": {...}
    },
    "target": {
        "text": "assistant continuation after tool result"
    }
}
"""
from typing import Dict, Any, List
from pathlib import Path
import json
import torch
from torch.utils.data import Dataset

from training.dataset import load_tokenizer


class PostToolDataset(Dataset):
    """
    Dataset for post-tool integration training.
    
    Each example contains:
    - Input: system prompt, tools manifest, history, tool result
    - Target: assistant continuation that uses tool result
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer_path: str,
        max_seq_length: int = 4096,
    ):
        """
        Initialize post-tool dataset.
        
        Args:
            data_path: Path to post_tool.jsonl file
            tokenizer_path: Path to tokenizer
            max_seq_length: Maximum sequence length
        """
        self.data_path = Path(data_path)
        self.tokenizer = load_tokenizer(tokenizer_path)
        self.max_seq_length = max_seq_length
        
        # Load examples
        self.examples = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                example = json.loads(line)
                self.examples.append(example)
        
        print(f"[PostToolDataset] Loaded {len(self.examples)} examples from {data_path}")
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example.
        
        Returns:
            Dictionary with:
            - input_ids: [T] tokenized input
            - attention_mask: [T] attention mask
            - labels: [T] labels for next-token prediction
        """
        example = self.examples[idx]
        
        # Build input prompt
        input_data = example["input"]
        system = input_data.get("system", "")
        tools = input_data.get("tools", [])
        history = input_data.get("history", [])
        tool_result = input_data.get("tool_result", {})
        
        # Format tools manifest
        tools_text = "Available tools:\n"
        for tool in tools:
            name = tool.get("name", "")
            schema = tool.get("schema", {})
            tools_text += f"- {name}: {json.dumps(schema, indent=2)}\n"
        
        # Format history
        history_text = ""
        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            history_text += f"{role}: {content}\n"
        
        # Format tool result
        tool_result_text = f"Tool result: {json.dumps(tool_result, ensure_ascii=False, indent=2)}"
        
        # Build full prompt
        prompt = f"{system}\n\n{tools_text}\n\n{history_text}\n\n{tool_result_text}\n\nContinue:"
        
        # Get target text
        target_text = example["target"].get("text", "")
        
        # Combine prompt + target for next-token prediction
        full_text = prompt + " " + target_text
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        # Create labels (only for target portion)
        prompt_tokens = self.tokenizer(
            prompt,
            max_length=self.max_seq_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_len = prompt_tokens["input_ids"].shape[1]
        
        labels = encoding["input_ids"].squeeze(0).clone()
        # Mask prompt tokens
        labels[:prompt_len] = -100
        # Mask padding
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels,
            "target_text": target_text,
        }


def collate_post_tool_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for post-tool dataset.
    
    Args:
        batch: List of examples from dataset
        
    Returns:
        Batched tensors
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    
    # Metadata
    target_texts = [item["target_text"] for item in batch]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "target_texts": target_texts,
    }



