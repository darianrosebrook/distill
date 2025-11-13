"""
Dataset loader for tool selection training stage.

Loads tool_select.jsonl format:
{
    "input": {
        "system": "...",
        "tools": [...],
        "history": [...]
    },
    "target": {
        "name": "tool_name",
        "arguments": {...}
    }
}
"""
from typing import Dict, Any, List
from pathlib import Path
import json
import torch
from torch.utils.data import Dataset

from training.dataset import load_tokenizer


class ToolSelectDataset(Dataset):
    """
    Dataset for tool selection and argument synthesis training.
    
    Each example contains:
    - Input: system prompt, tools manifest, conversation history
    - Target: tool name and arguments JSON
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer_path: str,
        max_seq_length: int = 2048,
        max_target_length: int = 512,
    ):
        """
        Initialize tool selection dataset.
        
        Args:
            data_path: Path to tool_select.jsonl file
            tokenizer_path: Path to tokenizer
            max_seq_length: Maximum input sequence length
            max_target_length: Maximum target (tool call JSON) length
        """
        self.data_path = Path(data_path)
        self.tokenizer = load_tokenizer(tokenizer_path)
        self.max_seq_length = max_seq_length
        self.max_target_length = max_target_length
        
        # Load examples
        self.examples = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                example = json.loads(line)
                self.examples.append(example)
        
        print(f"[ToolSelectDataset] Loaded {len(self.examples)} examples from {data_path}")
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example.
        
        Returns:
            Dictionary with:
            - input_ids: [T] tokenized input
            - attention_mask: [T] attention mask
            - labels: [T_target] tokenized target (tool call JSON)
            - tool_name: str (for loss computation)
            - tool_arguments: dict (for validation)
        """
        example = self.examples[idx]
        
        # Build input prompt
        input_data = example["input"]
        system = input_data.get("system", "")
        tools = input_data.get("tools", [])
        history = input_data.get("history", [])
        
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
        
        # Build full prompt
        prompt = f"{system}\n\n{tools_text}\n\n{history_text}\n\nGenerate a tool call:"
        
        # Tokenize input
        input_encoding = self.tokenizer(
            prompt,
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        # Build target (tool call JSON)
        target_data = example["target"]
        tool_name = target_data.get("name", "")
        tool_arguments = target_data.get("arguments", {})
        
        # Format as JSON string
        tool_call_json = json.dumps({
            "name": tool_name,
            "arguments": tool_arguments
        }, ensure_ascii=False)
        
        # Tokenize target
        target_encoding = self.tokenizer(
            tool_call_json,
            max_length=self.max_target_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        # Create labels (shifted for next-token prediction)
        labels = target_encoding["input_ids"].squeeze(0).clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_encoding["input_ids"].squeeze(0),
            "attention_mask": input_encoding["attention_mask"].squeeze(0),
            "labels": labels,
            "tool_name": tool_name,
            "tool_arguments": tool_arguments,
            "tool_call_json": tool_call_json,
        }


def collate_tool_select_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for tool selection dataset.
    
    Args:
        batch: List of examples from dataset
        
    Returns:
        Batched tensors
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    
    # Metadata (for evaluation)
    tool_names = [item["tool_name"] for item in batch]
    tool_arguments = [item["tool_arguments"] for item in batch]
    tool_call_jsons = [item["tool_call_json"] for item in batch]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "tool_names": tool_names,
        "tool_arguments": tool_arguments,
        "tool_call_jsons": tool_call_jsons,
    }



