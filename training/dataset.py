"""
Dataset loader for knowledge distillation training.

Loads JSONL format from make_kd_mix.py and prepares batches for training.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset

try:
    import importlib.util
    spec = importlib.util.find_spec("transformers")
    HF_TOKENIZER_AVAILABLE = spec is not None
except (ValueError, AttributeError, ImportError):
    # Module might already be imported or not available
    try:
        import transformers  # noqa: F401
        HF_TOKENIZER_AVAILABLE = True
    except ImportError:
        HF_TOKENIZER_AVAILABLE = False


def load_tokenizer(tokenizer_path: str):
    """Load tokenizer from path."""
    if not HF_TOKENIZER_AVAILABLE:
        raise RuntimeError(
            "transformers library required for tokenizer. Install with: pip install transformers"
        )

    from training.safe_model_loading import safe_from_pretrained_tokenizer
    tokenizer = safe_from_pretrained_tokenizer(tokenizer_path)
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
        latent_curriculum: Optional[Any] = None,
    ):
        """
        Initialize KD dataset.

        Args:
            jsonl_path: Path to JSONL file from make_kd_mix.py
            tokenizer_path: Path to tokenizer (HuggingFace format)
            max_seq_length: Maximum sequence length for truncation
            teacher_logits_available: Whether teacher logits are in the dataset
            latent_curriculum: Optional LatentCurriculum wrapper instance
        """
        self.jsonl_path = Path(jsonl_path)
        self.tokenizer_path = tokenizer_path
        self.max_seq_length = max_seq_length
        self.teacher_logits_available = teacher_logits_available
        self.latent_curriculum = latent_curriculum

        # Load tokenizer
        if not HF_TOKENIZER_AVAILABLE:
            raise RuntimeError(
                "transformers library required for tokenizer. Install with: pip install transformers"
            )

        from training.safe_model_loading import safe_from_pretrained_tokenizer
        self.tokenizer = safe_from_pretrained_tokenizer(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load data and extract fingerprint
        self.samples = []
        self.dataset_fingerprint = None
        self.dataset_header = None
        self._load_data()

    def _load_data(self):
        """Load samples from JSONL file."""
        if not self.jsonl_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {self.jsonl_path}")

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            first_line = True
            valid_samples = 0
            skipped_samples = 0
            empty_teacher_text_count = 0  # Track empty teacher_text separately for warning suppression

            for line_num, line in enumerate(f):
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)

                    # Check for dataset header (first line with __header__ flag)
                    if first_line and "__header__" in data:
                        # Header can be either:
                        # 1. {"__header__": {...header_data...}} - header data nested under __header__
                        # 2. {"__header__": true, "dataset_fingerprint": "...", ...} - header data at top level
                        if isinstance(data["__header__"], dict):
                            # Case 1: header data is nested
                            self.dataset_header = data["__header__"]
                            # Extract fingerprint if available
                            if "dataset_fingerprint" in data["__header__"]:
                                self.dataset_fingerprint = data["__header__"]["dataset_fingerprint"]
                            elif "dataset_sha256" in data["__header__"]:
                                self.dataset_fingerprint = data["__header__"]["dataset_sha256"]
                        else:
                            # Case 2: header data at top level
                            self.dataset_header = data
                            # Extract fingerprint if available
                            if "dataset_fingerprint" in data:
                                self.dataset_fingerprint = data["dataset_fingerprint"]
                            elif "dataset_sha256" in data:
                                self.dataset_fingerprint = data["dataset_sha256"]
                        first_line = False
                        continue

                    first_line = False

                    # Validate required fields
                    if "prompt" not in data or "teacher_text" not in data:
                        print(
                            f"[KDDataset] WARN: Skipping line {line_num + 1}: missing required fields 'prompt' and/or 'teacher_text'"
                        )
                        skipped_samples += 1
                        continue

                    # Validate data types
                    if not isinstance(data["prompt"], str) or not isinstance(data["teacher_text"], str):
                        print(
                            f"[KDDataset] WARN: Skipping line {line_num + 1}: prompt and teacher_text must be strings"
                        )
                        skipped_samples += 1
                        continue

                    # Validate string lengths
                    if len(data["prompt"].strip()) == 0:
                        print(
                            f"[KDDataset] WARN: Skipping line {line_num + 1}: prompt is empty"
                        )
                        skipped_samples += 1
                        continue

                    if len(data["teacher_text"].strip()) == 0:
                        # Count empty teacher_text silently (will be reported in summary)
                        # Only warn for first few to indicate the issue exists
                        empty_teacher_text_count += 1
                        if empty_teacher_text_count <= 3:
                            print(
                                f"[KDDataset] WARN: Skipping line {line_num + 1}: teacher_text is empty"
                            )
                        skipped_samples += 1
                        continue

                    # CoT-free validation: Fail if reasoning_content detected
                    if "teacher_reasoning_content" in data and data["teacher_reasoning_content"]:
                        raise ValueError(
                            f"CoT-free training: teacher_reasoning_content detected in line {line_num + 1}. "
                            "Training on reasoning_content violates ToS. Use process-step supervision instead."
                        )

                    # Optional field validation
                    # Only warn about teacher_logits if we expect them to be available
                    if "teacher_logits" in data and self.teacher_logits_available:
                        logits = data["teacher_logits"]
                        if not isinstance(logits, list) or not all(isinstance(x, (int, float)) for x in logits):
                            print(
                                f"[KDDataset] WARN: Line {line_num + 1}: teacher_logits should be list of numbers, got {type(logits)}"
                            )

                    self.samples.append(data)
                    valid_samples += 1

                except json.JSONDecodeError as e:
                    print(
                        f"[KDDataset] WARN: Skipping line {line_num + 1}: JSON decode error: {e}")
                    skipped_samples += 1
                    continue
                except UnicodeDecodeError as e:
                    print(
                        f"[KDDataset] WARN: Skipping line {line_num + 1}: Unicode decode error: {e}")
                    skipped_samples += 1
                    continue

        print(f"[KDDataset] Loaded {valid_samples} valid samples, skipped {skipped_samples} invalid samples")

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
            - loss_mask: [T] boolean mask (if latent curriculum applied)
        """
        sample = self.samples[idx].copy()

        # Apply latent curriculum if enabled
        if self.latent_curriculum is not None:
            sample = self.latent_curriculum.apply(sample, self.tokenizer)

        prompt = sample.get(
            "prompt",
            sample.get("training_text", "").split("\n\n")[
                0] if "training_text" in sample else "",
        )
        teacher_text = sample.get("teacher_text", "")

        # Use training_text if available (from latent curriculum)
        if "training_text" in sample:
            full_text = sample["training_text"]
        else:
            full_text = prompt + teacher_text

        # Tokenize prompt
        self.tokenizer.encode(prompt, add_special_tokens=False)

        # Tokenize teacher response
        teacher_tokens = self.tokenizer.encode(
            teacher_text, add_special_tokens=False)

        # Compute span targets for code-mode loss if metadata contains teacher_targets
        span_targets = None
        if "metadata" in sample and "span_targets" in sample["metadata"]:
            # Check if span_targets need to be computed from teacher_text
            metadata_span_targets = sample["metadata"].get("span_targets", {})
            if not metadata_span_targets.get("ts_mode_spans") and not metadata_span_targets.get(
                "direct_tool_spans"
            ):
                # Compute spans from teacher_text using tokenizer
                try:
                    from data.generators.mcp_code_mode import compute_span_targets_from_tokenized

                    span_targets = compute_span_targets_from_tokenized(
                        teacher_text, self.tokenizer)
                except Exception:
                    span_targets = None
            else:
                span_targets = metadata_span_targets

        # Combine: prompt + teacher response
        # For next-token prediction, we want to predict teacher tokens given prompt
        full_tokens = self.tokenizer.encode(full_text, add_special_tokens=True)

        # Truncate if needed
        # Need max_seq_length + 1 tokens to get max_seq_length tokens after shifting
        if len(full_tokens) > self.max_seq_length + 1:
            full_tokens = full_tokens[: self.max_seq_length + 1]

        # Create input_ids and labels
        # Labels are shifted by 1 for next-token prediction
        input_ids = torch.tensor(full_tokens[:-1], dtype=torch.long)
        labels = torch.tensor(full_tokens[1:], dtype=torch.long)

        # Attention mask: 1 for real tokens, 0 for padding
        # Padding is handled in collate_kd_batch, so all tokens are valid here
        attention_mask = torch.ones_like(input_ids)

        result = {
            "input_ids": input_ids,
            "span_targets": span_targets,  # Token-level spans for code-mode loss
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
                sample["gold_json_text_ids"], dtype=torch.long
            )
        if "mask_valid_json_tokens" in sample:
            result["mask_valid_json_tokens"] = torch.tensor(
                sample["mask_valid_json_tokens"], dtype=torch.bool
            )
        if "tool_result_fields" in sample:
            result["tool_result_fields"] = torch.tensor(
                sample["tool_result_fields"], dtype=torch.long
            )
        if "integration_mask" in sample:
            result["integration_mask"] = torch.tensor(
                sample["integration_mask"], dtype=torch.bool)

        # Add teacher targets (use teacher tokens as targets)
        if teacher_tokens is not None and len(teacher_tokens) > 0:
            teacher_target_ids = torch.tensor(
                teacher_tokens[: len(labels)], dtype=torch.long)
            # Pad or truncate to match labels length
            if len(teacher_target_ids) < len(labels):
                # Pad with -100 (ignore index)
                padding = torch.full(
                    (len(labels) - len(teacher_target_ids),), -100, dtype=torch.long
                )
                teacher_target_ids = torch.cat([teacher_target_ids, padding])
            elif len(teacher_target_ids) > len(labels):
                teacher_target_ids = teacher_target_ids[: len(labels)]
            result["teacher_target_ids"] = teacher_target_ids

        # Add loss mask if latent curriculum was applied
        if "loss_mask" in sample and sample["loss_mask"] is not None:
            loss_mask = sample["loss_mask"]
            # Convert to tensor if it's a list
            if not isinstance(loss_mask, torch.Tensor):
                loss_mask = torch.tensor(loss_mask, dtype=torch.bool)
            # Ensure loss_mask matches labels length
            if len(loss_mask) < len(labels):
                # Pad with True (supervise padding)
                padding = torch.ones(
                    len(labels) - len(loss_mask), dtype=torch.bool)
                loss_mask = torch.cat([loss_mask, padding])
            elif len(loss_mask) > len(labels):
                loss_mask = loss_mask[: len(labels)]
            result["loss_mask"] = loss_mask

        # Add teacher logits if available
        if (
            self.teacher_logits_available
            and "teacher_logits" in sample
            and sample["teacher_logits"]
        ):
            teacher_logits = torch.tensor(
                sample["teacher_logits"], dtype=torch.float32)
            # Reshape if needed: [T*V] -> [T, V] or [T, V] already
            vocab_size = len(self.tokenizer)
            if teacher_logits.dim() == 1:
                # Assume flattened [T*V]
                seq_len = len(labels)
                teacher_logits = teacher_logits[: seq_len *
                                                vocab_size].view(seq_len, vocab_size)
            elif teacher_logits.dim() == 2:
                # Already [T, V]
                # Truncate to labels length if longer, but don't pad if shorter
                # Padding will be handled by collate_kd_batch
                seq_len = min(teacher_logits.size(0), len(labels))
                teacher_logits = teacher_logits[:seq_len]
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

        # Preserve metadata if present (for debugging, logging, etc.)
        if "metadata" in sample:
            result["metadata"] = sample["metadata"]

        # Add teacher hidden states if available (for intermediate layer matching)
        # Hidden states can be provided in two ways:
        # 1. Pre-computed and stored in JSONL (as lists of lists/floats)
        # 2. Extracted on-the-fly from teacher model during training
        if "teacher_hidden_states" in sample:
            teacher_hidden_states = sample["teacher_hidden_states"]

            # Try to load pre-computed hidden states if available
            if isinstance(teacher_hidden_states, list) and len(teacher_hidden_states) > 0:
                try:
                    # Convert list of lists to tensor format
                    # Expected format: List[List[List[float]]] -> List[torch.Tensor]
                    # Each inner list is [B, T, D] hidden state for one layer
                    hidden_states_tensors = []
                    for layer_states in teacher_hidden_states:
                        if isinstance(layer_states, list):
                            # Convert to tensor: [T, D] or [B, T, D]
                            layer_tensor = torch.tensor(
                                layer_states, dtype=torch.float32)
                            hidden_states_tensors.append(layer_tensor)
                        else:
                            # Already a tensor or unsupported format
                            break

                    if hidden_states_tensors:
                        result["teacher_hidden_states"] = hidden_states_tensors
                    else:
                        # Fallback: store metadata flag
                        result["has_teacher_hidden_states"] = True
                except Exception:
                    # If loading fails, store metadata flag
                    # Hidden states will need to be extracted from teacher model during training
                    result["has_teacher_hidden_states"] = True
            else:
                # Store metadata flag indicating hidden states should be extracted
                # from teacher model during training
                result["has_teacher_hidden_states"] = True

        return result


def collate_kd_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for KD batches.

    Pads sequences to the same length within the batch.
    Handles process-step supervision targets.

    Note: teacher_reasoning_content is NOT supported (ToS violation risk).
    """
    # Find max length in batch for input sequences
    max_len = max(item["input_ids"].size(0) for item in batch)

    # Find max lengths for process-step supervision targets (if present)
    # These are independent sequences that may not align with input_ids
    max_tool_name_ids_len = 0
    max_gold_json_text_ids_len = 0
    max_tool_result_fields_len = 0
    
    # Find max vocab_size for teacher_logits (if present)
    # Different items may have different vocab sizes, need to pad to max
    max_vocab_size = 0
    has_teacher_logits = False
    all_have_teacher_logits = True
    teacher_logits_count = 0
    for item in batch:
        if "tool_name_ids" in item:
            max_tool_name_ids_len = max(max_tool_name_ids_len, item["tool_name_ids"].size(0))
        if "gold_json_text_ids" in item:
            max_gold_json_text_ids_len = max(max_gold_json_text_ids_len, item["gold_json_text_ids"].size(0))
        if "tool_result_fields" in item:
            max_tool_result_fields_len = max(max_tool_result_fields_len, item["tool_result_fields"].size(0))
        if "teacher_logits" in item:
            has_teacher_logits = True
            teacher_logits_count += 1
            vocab_size = item["teacher_logits"].size(-1)
            max_vocab_size = max(max_vocab_size, vocab_size)
    
    # Check if all items have teacher_logits
    if has_teacher_logits:
        all_have_teacher_logits = (teacher_logits_count == len(batch))

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
    # Latent curriculum loss mask
    loss_mask_list = []

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
                    [teacher_target_ids, torch.full(
                        (pad_len,), -100, dtype=torch.long)]
                )
            teacher_target_ids_list.append(teacher_target_ids)

        # Pad teacher_logits if present
        if has_teacher_logits:
            if "teacher_logits" in item:
                teacher_logits = item["teacher_logits"]
                vocab_size = teacher_logits.size(-1)
                teacher_logits_seq_len = teacher_logits.size(0)
                
                # Pad sequence length to match max_len (not just input_ids length)
                teacher_logits_pad_len = max_len - teacher_logits_seq_len
                if teacher_logits_pad_len > 0:
                    padding = torch.zeros(teacher_logits_pad_len, vocab_size, dtype=torch.float32)
                    teacher_logits = torch.cat([teacher_logits, padding], dim=0)
                elif teacher_logits_seq_len > max_len:
                    # Truncate if longer than max_len
                    teacher_logits = teacher_logits[:max_len]
                
                # Pad vocab_size dimension if needed (to match max_vocab_size in batch)
                if max_vocab_size > 0 and vocab_size < max_vocab_size:
                    vocab_pad_len = max_vocab_size - vocab_size
                    vocab_padding = torch.zeros(teacher_logits.size(0), vocab_pad_len, dtype=torch.float32)
                    teacher_logits = torch.cat([teacher_logits, vocab_padding], dim=-1)
                
                teacher_logits_list.append(teacher_logits)
            else:
                # Missing teacher_logits - append None for this item
                teacher_logits_list.append(None)

        # Collect and pad process-step supervision targets
        # These are padded independently based on their own max lengths
        # If any item has a field, all items must have it (create empty tensors for missing ones)
        if max_tool_name_ids_len > 0:
            if "tool_name_ids" in item:
                tool_name_ids = item["tool_name_ids"]
                tool_pad_len = max_tool_name_ids_len - tool_name_ids.size(0)
                if tool_pad_len > 0:
                    tool_name_ids = torch.cat(
                        [tool_name_ids, torch.zeros(tool_pad_len, dtype=torch.long)]
                    )
            else:
                # Create empty tensor if missing
                tool_name_ids = torch.zeros(max_tool_name_ids_len, dtype=torch.long)
            tool_name_ids_list.append(tool_name_ids)

        if max_tool_name_ids_len > 0:
            if "tool_name_mask" in item:
                tool_name_mask = item["tool_name_mask"]
                # Pad to match tool_name_ids length
                tool_pad_len = max_tool_name_ids_len - tool_name_mask.size(0)
                if tool_pad_len > 0:
                    tool_name_mask = torch.cat(
                        [tool_name_mask, torch.zeros(tool_pad_len, dtype=torch.bool)])
            else:
                # Create empty tensor if missing
                tool_name_mask = torch.zeros(max_tool_name_ids_len, dtype=torch.bool)
            tool_name_mask_list.append(tool_name_mask)

        if max_gold_json_text_ids_len > 0:
            if "gold_json_text_ids" in item:
                gold_json_text_ids = item["gold_json_text_ids"]
                json_pad_len = max_gold_json_text_ids_len - gold_json_text_ids.size(0)
                if json_pad_len > 0:
                    gold_json_text_ids = torch.cat(
                        [gold_json_text_ids, torch.full(
                            (json_pad_len,), -100, dtype=torch.long)]
                    )
            else:
                # Create empty tensor if missing
                gold_json_text_ids = torch.full((max_gold_json_text_ids_len,), -100, dtype=torch.long)
            gold_json_text_ids_list.append(gold_json_text_ids)

        if max_gold_json_text_ids_len > 0:
            if "mask_valid_json_tokens" in item:
                mask_valid_json_tokens = item["mask_valid_json_tokens"]
                # Pad to match gold_json_text_ids length
                json_pad_len = max_gold_json_text_ids_len - mask_valid_json_tokens.size(0)
                if json_pad_len > 0:
                    mask_valid_json_tokens = torch.cat(
                        [mask_valid_json_tokens, torch.zeros(
                            json_pad_len, dtype=torch.bool)]
                    )
            else:
                # Create empty tensor if missing
                mask_valid_json_tokens = torch.zeros(max_gold_json_text_ids_len, dtype=torch.bool)
            mask_valid_json_tokens_list.append(mask_valid_json_tokens)

        if max_tool_result_fields_len > 0:
            if "tool_result_fields" in item:
                tool_result_fields = item["tool_result_fields"]
                result_pad_len = max_tool_result_fields_len - tool_result_fields.size(0)
                if result_pad_len > 0:
                    tool_result_fields = torch.cat(
                        [tool_result_fields, torch.full(
                            (result_pad_len,), -100, dtype=torch.long)]
                    )
            else:
                # Create empty tensor if missing
                tool_result_fields = torch.full((max_tool_result_fields_len,), -100, dtype=torch.long)
            tool_result_fields_list.append(tool_result_fields)

        if "integration_mask" in item:
            integration_mask = item["integration_mask"]
            # Integration mask should match input_ids length
            if pad_len > 0:
                integration_mask = torch.cat(
                    [integration_mask, torch.zeros(pad_len, dtype=torch.bool)]
                )
            integration_mask_list.append(integration_mask)

        # Handle loss mask (from latent curriculum)
        if "loss_mask" in item:
            loss_mask = item["loss_mask"]
            # Loss mask should match input_ids/labels length
            if pad_len > 0:
                # Pad with True (supervise padding tokens)
                loss_mask = torch.cat(
                    [loss_mask, torch.ones(pad_len, dtype=torch.bool)])
            loss_mask_list.append(loss_mask)

    # Stack into batches
    result = {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "labels": torch.stack(labels_list),
    }

    if teacher_target_ids_list:
        result["teacher_target_ids"] = torch.stack(teacher_target_ids_list)

    # Handle teacher_logits: stack if all items have them, otherwise return as list
    if teacher_logits_list and len(teacher_logits_list) > 0:
        if all_have_teacher_logits:
            # All items have teacher_logits - stack into batch tensor
            result["teacher_logits"] = torch.stack(teacher_logits_list)
        else:
            # Not all items have teacher_logits - return as list with None for missing items
            # This allows tests to check for None/empty in missing items
            result["teacher_logits"] = teacher_logits_list

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

    # Add loss mask if available
    if loss_mask_list:
        result["loss_mask"] = torch.stack(loss_mask_list)

    return result
