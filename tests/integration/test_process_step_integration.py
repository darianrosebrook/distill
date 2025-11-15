"""
Integration test for process-step supervision targets.

Tests the complete flow from dataset generation to training:
1. Dataset contains process-step supervision targets
2. Dataset loads targets correctly
3. Batches include target fields
4. Training step uses targets in loss computation
"""

import json

import pytest
import torch

from models.student.architectures.gqa_transformer import StudentLM
from training.dataset import KDDataset
from training.distill_process import train_step_process
from training.process_losses import process_supervision_loss


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for integration tests."""

    class MockTokenizer:
        def __init__(self):
            self.vocab_size = 1000
            self.pad_token_id = 0
            self.eos_token_id = 1
            self.pad_token = None

        def encode(self, text, add_special_tokens=False, **kwargs):
            """Encode text to token IDs."""
            # Simple mock: tokenize by splitting on spaces and characters
            tokens = []
            for char in text:
                if char.isalnum() or char in '{}[]":,':
                    tokens.append(char)
            token_ids = [abs(hash(t)) % self.vocab_size for t in tokens]

            if add_special_tokens:
                token_ids = [self.eos_token_id] + token_ids + [self.eos_token_id]

            return token_ids

        def decode(self, token_ids, skip_special_tokens=False, **kwargs):
            """Decode token IDs to text."""
            # Simple mock: reverse the hash (approximate)
            # For testing, just return a simple string
            if skip_special_tokens:
                token_ids = [
                    t for t in token_ids if t != self.eos_token_id and t != self.pad_token_id
                ]
            return "read_file" if token_ids else ""

        def __len__(self):
            """Return vocabulary size."""
            return self.vocab_size

    tokenizer = MockTokenizer()
    tokenizer.pad_token = tokenizer.eos_token_id
    return tokenizer


@pytest.fixture
def temp_process_step_dataset(mock_tokenizer, tmp_path):
    """Create a temporary dataset with process-step supervision targets."""
    dataset_file = tmp_path / "test_process_step.jsonl"

    # Create test samples with process-step targets
    samples = [
        {
            "prompt": "Read the file config.json",
            "teacher_text": 'I will use {"name": "read_file", "arguments": {"path": "config.json"}}',
            "tool_name_ids": mock_tokenizer.encode("read_file", add_special_tokens=False),
            "tool_name_mask": [1]
            * len(mock_tokenizer.encode("read_file", add_special_tokens=False)),
            "gold_json_text_ids": mock_tokenizer.encode(
                '{"name": "read_file", "arguments": {"path": "config.json"}}',
                add_special_tokens=False,
            ),
            "mask_valid_json_tokens": [1]
            * len(
                mock_tokenizer.encode(
                    '{"name": "read_file", "arguments": {"path": "config.json"}}',
                    add_special_tokens=False,
                )
            ),
        },
        {
            "prompt": "Search for Python tutorials",
            "teacher_text": 'I will search {"name": "web.search", "arguments": {"q": "Python tutorials"}}',
            "tool_name_ids": mock_tokenizer.encode("web.search", add_special_tokens=False),
            "tool_name_mask": [1]
            * len(mock_tokenizer.encode("web.search", add_special_tokens=False)),
            "gold_json_text_ids": mock_tokenizer.encode(
                '{"name": "web.search", "arguments": {"q": "Python tutorials"}}',
                add_special_tokens=False,
            ),
            "mask_valid_json_tokens": [1]
            * len(
                mock_tokenizer.encode(
                    '{"name": "web.search", "arguments": {"q": "Python tutorials"}}',
                    add_special_tokens=False,
                )
            ),
        },
    ]

    with open(dataset_file, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    return dataset_file


def test_dataset_loads_process_step_targets(temp_process_step_dataset, mock_tokenizer):
    """Test that dataset loads process-step supervision targets correctly."""
    # Mock the tokenizer loading
    import sys
    from unittest.mock import MagicMock

    mock_transformers = MagicMock()
    mock_auto_tokenizer_class = MagicMock()
    mock_auto_tokenizer_class.from_pretrained = MagicMock(return_value=mock_tokenizer)
    mock_transformers.AutoTokenizer = mock_auto_tokenizer_class
    sys.modules["transformers"] = mock_transformers

    # Reload dataset module if needed
    if "training.dataset" in sys.modules:
        import importlib

        importlib.reload(sys.modules["training.dataset"])

    import training.dataset as dataset_module

    dataset_module.HF_TOKENIZER_AVAILABLE = True
    dataset_module.AutoTokenizer = mock_auto_tokenizer_class

    # Create dataset
    dataset = KDDataset(
        jsonl_path=str(temp_process_step_dataset),
        tokenizer_path="dummy",  # Not used due to mock
        max_seq_length=512,
    )

    assert len(dataset) == 2, "Dataset should have 2 samples"

    # Get first sample
    sample = dataset[0]

    # Verify process-step targets are present
    assert "tool_name_ids" in sample, "tool_name_ids should be in sample"
    assert "tool_name_mask" in sample, "tool_name_mask should be in sample"
    assert "gold_json_text_ids" in sample, "gold_json_text_ids should be in sample"
    assert "mask_valid_json_tokens" in sample, "mask_valid_json_tokens should be in sample"

    # Verify they are tensors
    assert isinstance(sample["tool_name_ids"], torch.Tensor)
    assert isinstance(sample["tool_name_mask"], torch.Tensor)
    assert isinstance(sample["gold_json_text_ids"], torch.Tensor)
    assert isinstance(sample["mask_valid_json_tokens"], torch.Tensor)

    # Verify masks are boolean
    assert (
        sample["tool_name_mask"].dtype == torch.bool
        or sample["tool_name_mask"].dtype == torch.int64
    )
    assert (
        sample["mask_valid_json_tokens"].dtype == torch.bool
        or sample["mask_valid_json_tokens"].dtype == torch.int64
    )


def test_batch_contains_process_step_targets(temp_process_step_dataset, mock_tokenizer, device):
    """Test that batches contain process-step supervision targets."""
    import sys
    from unittest.mock import MagicMock
    from torch.utils.data import DataLoader

    mock_transformers = MagicMock()
    mock_auto_tokenizer_class = MagicMock()
    mock_auto_tokenizer_class.from_pretrained = MagicMock(return_value=mock_tokenizer)
    mock_transformers.AutoTokenizer = mock_auto_tokenizer_class
    sys.modules["transformers"] = mock_transformers

    if "training.dataset" in sys.modules:
        import importlib

        importlib.reload(sys.modules["training.dataset"])

    from training.dataset import collate_kd_batch
    import training.dataset as dataset_module

    dataset_module.HF_TOKENIZER_AVAILABLE = True
    dataset_module.AutoTokenizer = mock_auto_tokenizer_class

    # Create dataset and dataloader
    dataset = KDDataset(
        jsonl_path=str(temp_process_step_dataset),
        tokenizer_path="dummy",
        max_seq_length=512,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=collate_kd_batch,
    )

    # Get a batch
    batch = next(iter(dataloader))

    # Verify batch contains process-step targets
    assert "tool_name_ids" in batch, "tool_name_ids should be in batch"
    assert "tool_name_mask" in batch, "tool_name_mask should be in batch"
    assert "gold_json_text_ids" in batch, "gold_json_text_ids should be in batch"
    assert "mask_valid_json_tokens" in batch, "mask_valid_json_tokens should be in batch"


def test_process_supervision_loss_with_token_ids(mock_tokenizer, device):
    """Test that process supervision loss works with token IDs."""
    batch_size = 2
    seq_len = 20
    vocab_size = 1000

    # Create mock logits
    logits = torch.randn(batch_size, seq_len, vocab_size, device=device)

    # Create mock token IDs
    # Encode token IDs - these may have different lengths
    tool_name_ids_list = [
        torch.tensor(mock_tokenizer.encode("read_file", add_special_tokens=False), device=device),
        torch.tensor(mock_tokenizer.encode("web.search", add_special_tokens=False), device=device),
    ]
    # Pad to same length
    max_tool_len = max(len(ids) for ids in tool_name_ids_list)
    tool_name_ids = torch.stack([
        torch.cat([ids, torch.full((max_tool_len - len(ids),), -100, dtype=torch.long, device=device)])
        for ids in tool_name_ids_list
    ])

    tool_name_mask = torch.ones_like(tool_name_ids, dtype=torch.bool, device=device)

    # Encode JSON token IDs - these may have different lengths
    gold_json_text_ids_list = [
        torch.tensor(mock_tokenizer.encode('{"name": "read_file"}', add_special_tokens=False), device=device),
        torch.tensor(mock_tokenizer.encode('{"name": "web.search"}', add_special_tokens=False), device=device),
    ]
    # Pad to same length
    max_json_len = max(len(ids) for ids in gold_json_text_ids_list)
    gold_json_text_ids = torch.stack([
        torch.cat([ids, torch.full((max_json_len - len(ids),), -100, dtype=torch.long, device=device)])
        for ids in gold_json_text_ids_list
    ])

    mask_valid_json_tokens = torch.ones_like(gold_json_text_ids, dtype=torch.bool, device=device)

    # Compute loss
    loss_dict = process_supervision_loss(
        logits=logits,
        generated_texts=["dummy"] * batch_size,  # For backward compatibility
        tokenizer=mock_tokenizer,
        json_validity_weight=0.3,
        tool_select_weight=0.7,
        tool_name_ids=tool_name_ids,
        tool_name_mask=tool_name_mask,
        gold_json_text_ids=gold_json_text_ids,
        mask_valid_json_tokens=mask_valid_json_tokens,
    )

    # Verify loss dict structure
    assert "total" in loss_dict, "Loss dict should have 'total' key"
    assert "json_validity" in loss_dict, "Loss dict should have 'json_validity' key"
    assert "tool_selection" in loss_dict, "Loss dict should have 'tool_selection' key"

    # Verify losses are tensors and not NaN/inf
    assert isinstance(loss_dict["total"], torch.Tensor)
    assert not torch.isnan(loss_dict["total"]), "Total loss should not be NaN"
    assert not torch.isinf(loss_dict["total"]), "Total loss should not be Inf"
    assert loss_dict["total"].item() >= 0, "Total loss should be non-negative"


def test_training_step_with_process_step_targets(
    temp_process_step_dataset, mock_tokenizer, device, small_model_cfg
):
    """Test that training step works with process-step supervision targets."""
    from unittest.mock import patch
    from torch.utils.data import DataLoader

    # Patch safe_from_pretrained_tokenizer to return mock_tokenizer
    with patch("training.safe_model_loading.safe_from_pretrained_tokenizer", return_value=mock_tokenizer):
        from training.dataset import collate_kd_batch

        # Create model
        model = StudentLM(small_model_cfg)
        model = model.to(device)
        model.train()

        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Create dataset and dataloader
        dataset = KDDataset(
            jsonl_path=str(temp_process_step_dataset),
            tokenizer_path="dummy",
            max_seq_length=512,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=collate_kd_batch,
        )

        # Get a batch
        batch = next(iter(dataloader))

        # Move batch to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

        # Training config
        cfg = {
            "distillation": {
                "kl_weight": 0.5,
                "ce_teacher_weight": 0.3,
                "ce_ground_truth_weight": 0.2,
                "process_supervision_weight": 0.7,
            },
            "kd": {
                "kd_temperature": 2.0,
            },
            "optimizer": {
                "grad_clip": 1.0,
            },
        }

        proc_cfg = {
            "loss_json_validity_weight": 0.3,
            "loss_tool_select_weight": 0.7,
            "tool_names": ["read_file", "web.search"],
        }

        # Run training step
        try:
            loss_dict = train_step_process(
                model=model,
                batch=batch,
                optimizer=optimizer,
                scaler=None,
                cfg=cfg,
                device=device,
                tokenizer=mock_tokenizer,
                proc_cfg=proc_cfg,
            )

            # Verify loss dict structure
            assert "total" in loss_dict, "Loss dict should have 'total' key"
            assert "proc_total" in loss_dict or "proc_json_validity" in loss_dict, (
                "Loss dict should have process supervision losses"
            )

            # Verify losses are reasonable
            assert isinstance(loss_dict["total"], float)
            assert not (loss_dict["total"] != loss_dict["total"]), "Total loss should not be NaN"
            assert loss_dict["total"] >= 0, "Total loss should be non-negative"
        except Exception as e:
            # If training step fails, provide helpful error message
            pytest.fail(f"Training step failed: {e}")
