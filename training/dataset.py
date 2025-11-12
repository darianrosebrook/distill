"""
Dataset loader for knowledge distillation training.

Loads JSONL format from make_kd_mix.py and prepares batches for training.
"""
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

try:
    from transformers import AutoTokenizer
    HF_TOKENIZER_AVAILABLE = True
except ImportError:
    HF_TOKENIZER_AVAILABLE = False


def load_tokenizer(tokenizer_path: str):
    """Load tokenizer from path."""
    if not HF_TOKENIZER_AVAILABLE:
        raise RuntimeError(
            "transformers library required for tokenizer. Install with: pip install transformers")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class KDDataset(Dataset):
    """
    Dataset for knowledge distillation from JSONL format.

    Expected JSONL format:
    {
        "prompt": "...",
        "teacher_text": "...",
        # Process-step supervision targets (replaces teacher_reasoning_content):
        "tool_name_ids": [...],  # Optional: tool name token IDs
        "tool_name_mask": [...],  # Optional: mask for tool name tokens
        "gold_json_text_ids": [...],  # Optional: JSON token IDs
        "mask_valid_json_tokens": [...],  # Optional: mask for valid JSON tokens
        "tool_result_fields": [...],  # Optional: tool result field token IDs
        "integration_mask": [...],  # Optional: mask for integration spans
        "teacher_logits": [...],  # Optional
        "metadata": {...}
    }

    Note: teacher_reasoning_content is NOT supported (ToS violation risk).
    Use process-step supervision targets instead.
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer_path: str,
        max_seq_length: int = 4096,
        teacher_logits_available: bool = False,
    ):
        """
        Initialize KD dataset.

        Args:
            jsonl_path: Path to JSONL file from make_kd_mix.py
            tokenizer_path: Path to tokenizer (HuggingFace format)
            max_seq_length: Maximum sequence length for truncation
            teacher_logits_available: Whether teacher logits are in the dataset
        """
        self.jsonl_path = Path(jsonl_path)
        self.tokenizer_path = tokenizer_path
        self.max_seq_length = max_seq_length
        self.teacher_logits_available = teacher_logits_available

        # Load tokenizer
        if not HF_TOKENIZER_AVAILABLE:
            raise RuntimeError(
                "transformers library required for tokenizer. Install with: pip install transformers")

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load data
        self.samples = []
        self._load_data()

    def _load_data(self):
        """Load samples from JSONL file."""
        if not self.jsonl_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {self.jsonl_path}")

        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if "prompt" not in data or "teacher_text" not in data:
                        print(
                            f"[KDDataset] WARN: Skipping line {line_num+1}: missing required fields")
                        continue
                    # CoT-free validation: Fail if reasoning_content detected
                    if "teacher_reasoning_content" in data and data["teacher_reasoning_content"]:
                        raise ValueError(
                            f"CoT-free training: teacher_reasoning_content detected in line {line_num+1}. "
                            "Training on reasoning_content violates ToS. Use process-step supervision instead."
                        )
                    self.samples.append(data)
                except json.JSONDecodeError as e:
                    print(
                        f"[KDDataset] WARN: Skipping line {line_num+1}: JSON decode error: {e}")
                    continue

        print(
            f"[KDDataset] Loaded {len(self.samples)} samples from {self.jsonl_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary with:
            - input_ids: [T] token IDs
            - attention_mask: [T] attention mask
            - teacher_target_ids: [T] teacher's predicted tokens (if available)
            - teacher_logits: [T, V] teacher logits (if available)
        """
        sample = self.samples[idx]
        prompt = sample["prompt"]
        teacher_text = sample["teacher_text"]

        # Tokenize prompt
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)

        # Tokenize teacher response
        teacher_tokens = self.tokenizer.encode(
            teacher_text, add_special_tokens=False)

        # Combine: prompt + teacher response
        # For next-token prediction, we want to predict teacher tokens given prompt
        full_text = prompt + teacher_text
        full_tokens = self.tokenizer.encode(full_text, add_special_tokens=True)

        # Truncate if needed
        if len(full_tokens) > self.max_seq_length:
            full_tokens = full_tokens[:self.max_seq_length]

        # Create input_ids and labels
        # Labels are shifted by 1 for next-token prediction
        input_ids = torch.tensor(full_tokens[:-1], dtype=torch.long)
        labels = torch.tensor(full_tokens[1:], dtype=torch.long)

        # Attention mask (all ones for now, can be extended for padding)
        attention_mask = torch.ones_like(input_ids)

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,  # Ground truth labels
        }

        # Add process-step supervision targets if available
        if "tool_name_ids" in sample:
            result["tool_name_ids"] = torch.tensor(
                sample["tool_name_ids"], dtype=torch.long)
        if "tool_name_mask" in sample:
            result["tool_name_mask"] = torch.tensor(
                sample["tool_name_mask"], dtype=torch.bool)
        if "gold_json_text_ids" in sample:
            result["gold_json_text_ids"] = torch.tensor(
                sample["gold_json_text_ids"], dtype=torch.long)
        if "mask_valid_json_tokens" in sample:
            result["mask_valid_json_tokens"] = torch.tensor(
                sample["mask_valid_json_tokens"], dtype=torch.bool)
        if "tool_result_fields" in sample:
            result["tool_result_fields"] = torch.tensor(
                sample["tool_result_fields"], dtype=torch.long)
        if "integration_mask" in sample:
            result["integration_mask"] = torch.tensor(
                sample["integration_mask"], dtype=torch.bool)

        # Add teacher targets (use teacher tokens as targets)
        if teacher_tokens:
            teacher_target_ids = torch.tensor(
                teacher_tokens[:len(labels)], dtype=torch.long)
            # Pad or truncate to match labels length
            if len(teacher_target_ids) < len(labels):
                # Pad with -100 (ignore index)
                padding = torch.full(
                    (len(labels) - len(teacher_target_ids),), -100, dtype=torch.long)
                teacher_target_ids = torch.cat([teacher_target_ids, padding])
            elif len(teacher_target_ids) > len(labels):
                teacher_target_ids = teacher_target_ids[:len(labels)]
            result["teacher_target_ids"] = teacher_target_ids

        # Add teacher logits if available
        if self.teacher_logits_available and "teacher_logits" in sample and sample["teacher_logits"]:
            teacher_logits = torch.tensor(
                sample["teacher_logits"], dtype=torch.float32)
            # Reshape if needed: [T*V] -> [T, V] or [T, V] already
            vocab_size = len(self.tokenizer)
            if teacher_logits.dim() == 1:
                # Assume flattened [T*V]
                seq_len = len(labels)
                teacher_logits = teacher_logits[:seq_len *
                                                vocab_size].view(seq_len, vocab_size)
            elif teacher_logits.dim() == 2:
                # Already [T, V]
                seq_len = min(teacher_logits.size(0), len(labels))
                teacher_logits = teacher_logits[:seq_len]
                # Pad if needed
                if seq_len < len(labels):
                    padding = torch.zeros(
                        len(labels) - seq_len, vocab_size, dtype=torch.float32)
                    teacher_logits = torch.cat(
                        [teacher_logits, padding], dim=0)
            result["teacher_logits"] = teacher_logits

        # Add teacher quality score if available (for self-evaluation head training)
        if "teacher_quality_score" in sample:
            quality_score = sample["teacher_quality_score"]
            if isinstance(quality_score, (int, float)):
                result["teacher_quality_score"] = float(quality_score)
            elif isinstance(quality_score, str):
                try:
                    result["teacher_quality_score"] = float(quality_score)
                except ValueError:
                    pass  # Skip invalid quality scores

        # Add teacher hidden states if available (for intermediate layer matching)
        # NOTE: Hidden states are stored as lists of tensors in JSONL
        # They need to be loaded separately or extracted during training
        # For now, we just pass through the metadata indicating they exist
        if "teacher_hidden_states" in sample:
            # Store metadata flag (actual hidden states should be loaded separately
            # or extracted from teacher model during training)
            result["has_teacher_hidden_states"] = True

        return result


def collate_kd_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for KD batches.

    Pads sequences to the same length within the batch.
    Handles process-step supervision targets.

    Note: teacher_reasoning_content is NOT supported (ToS violation risk).
    """
    # Find max length in batch
    max_len = max(item["input_ids"].size(0) for item in batch)

    # Pad all sequences
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    teacher_target_ids_list = []
    teacher_logits_list = []
    # Process-step supervision targets
    tool_name_ids_list = []
    tool_name_mask_list = []
    gold_json_text_ids_list = []
    mask_valid_json_tokens_list = []
    tool_result_fields_list = []
    integration_mask_list = []

    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len

        # Pad input_ids
        input_ids = item["input_ids"]
        if pad_len > 0:
            input_ids = torch.cat(
                [input_ids, torch.zeros(pad_len, dtype=torch.long)])
        input_ids_list.append(input_ids)

        # Pad attention_mask
        attention_mask = item["attention_mask"]
        if pad_len > 0:
            attention_mask = torch.cat(
                [attention_mask, torch.zeros(pad_len, dtype=torch.long)])
        attention_mask_list.append(attention_mask)

        # Pad labels (with -100 for ignore index)
        labels = item["labels"]
        if pad_len > 0:
            labels = torch.cat(
                [labels, torch.full((pad_len,), -100, dtype=torch.long)])
        labels_list.append(labels)

        # Pad teacher_target_ids if present
        if "teacher_target_ids" in item:
            teacher_target_ids = item["teacher_target_ids"]
            if pad_len > 0:
                teacher_target_ids = torch.cat(
                    [teacher_target_ids, torch.full((pad_len,), -100, dtype=torch.long)])
            teacher_target_ids_list.append(teacher_target_ids)

        # Pad teacher_logits if present
        if "teacher_logits" in item:
            teacher_logits = item["teacher_logits"]
            vocab_size = teacher_logits.size(-1)
            if pad_len > 0:
                padding = torch.zeros(pad_len, vocab_size, dtype=torch.float32)
                teacher_logits = torch.cat([teacher_logits, padding], dim=0)
            teacher_logits_list.append(teacher_logits)

        # Collect process-step supervision targets
        if "tool_name_ids" in item:
            tool_name_ids = item["tool_name_ids"]
            if pad_len > 0:
                tool_name_ids = torch.cat(
                    [tool_name_ids, torch.full((pad_len,), -100, dtype=torch.long)])
            tool_name_ids_list.append(tool_name_ids)

        if "tool_name_mask" in item:
            tool_name_mask = item["tool_name_mask"]
            if pad_len > 0:
                tool_name_mask = torch.cat(
                    [tool_name_mask, torch.zeros(pad_len, dtype=torch.bool)])
            tool_name_mask_list.append(tool_name_mask)

        if "gold_json_text_ids" in item:
            gold_json_text_ids = item["gold_json_text_ids"]
            if pad_len > 0:
                gold_json_text_ids = torch.cat(
                    [gold_json_text_ids, torch.full((pad_len,), -100, dtype=torch.long)])
            gold_json_text_ids_list.append(gold_json_text_ids)

        if "mask_valid_json_tokens" in item:
            mask_valid_json_tokens = item["mask_valid_json_tokens"]
            if pad_len > 0:
                mask_valid_json_tokens = torch.cat(
                    [mask_valid_json_tokens, torch.zeros(pad_len, dtype=torch.bool)])
            mask_valid_json_tokens_list.append(mask_valid_json_tokens)

        if "tool_result_fields" in item:
            tool_result_fields = item["tool_result_fields"]
            if pad_len > 0:
                tool_result_fields = torch.cat(
                    [tool_result_fields, torch.full((pad_len,), -100, dtype=torch.long)])
            tool_result_fields_list.append(tool_result_fields)

        if "integration_mask" in item:
            integration_mask = item["integration_mask"]
            if pad_len > 0:
                integration_mask = torch.cat(
                    [integration_mask, torch.zeros(pad_len, dtype=torch.bool)])
            integration_mask_list.append(integration_mask)

    # Stack into batches
    result = {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "labels": torch.stack(labels_list),
    }

    if teacher_target_ids_list:
        result["teacher_target_ids"] = torch.stack(teacher_target_ids_list)

    if teacher_logits_list:
        result["teacher_logits"] = torch.stack(teacher_logits_list)

    # Add process-step supervision targets
    if tool_name_ids_list:
        result["tool_name_ids"] = torch.stack(tool_name_ids_list)
    if tool_name_mask_list:
        result["tool_name_mask"] = torch.stack(tool_name_mask_list)
    if gold_json_text_ids_list:
        result["gold_json_text_ids"] = torch.stack(gold_json_text_ids_list)
    if mask_valid_json_tokens_list:
        result["mask_valid_json_tokens"] = torch.stack(
            mask_valid_json_tokens_list)
    if tool_result_fields_list:
        result["tool_result_fields"] = torch.stack(tool_result_fields_list)
    if integration_mask_list:
        result["integration_mask"] = torch.stack(integration_mask_list)

    return result
