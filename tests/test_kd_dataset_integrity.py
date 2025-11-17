"""
Automated integrity tests for KD datasets.

Validates that KD dataset JSONL files meet quality standards:
- All records parse as valid JSON
- Required fields present and non-empty
- Reasonable length bounds
- Metadata structure correct
"""

import json
import pytest
from pathlib import Path
from typing import Dict, Any, List


def load_dataset_samples(jsonl_path: Path, max_samples: int = 100) -> List[Dict[str, Any]]:
    """Load random samples from JSONL file."""
    records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                records.append(record)
                if len(records) >= max_samples:
                    break
            except json.JSONDecodeError:
                continue
    return records


@pytest.fixture
def kd_dataset_path():
    """Fixture providing path to KD dataset."""
    # Default to the main dataset, but can be overridden
    path = Path("data/kd_mix_1500.jsonl")
    if not path.exists():
        pytest.skip(f"KD dataset not found: {path}")
    return path


def test_dataset_exists(kd_dataset_path):
    """Test that dataset file exists."""
    assert kd_dataset_path.exists(), f"Dataset file not found: {kd_dataset_path}"


def test_dataset_parseable(kd_dataset_path):
    """Test that all lines parse as valid JSON."""
    errors = []
    with open(kd_dataset_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: {e}")
    
    assert not errors, f"JSON parsing errors:\n" + "\n".join(errors)


def test_required_fields_present(kd_dataset_path):
    """Test that all records have required fields."""
    required_fields = ['prompt', 'teacher_text', 'metadata']
    records = load_dataset_samples(kd_dataset_path, max_samples=50)
    
    assert len(records) > 0, "No records found in dataset"
    
    missing_fields = []
    for i, record in enumerate(records):
        for field in required_fields:
            if field not in record:
                missing_fields.append(f"Record {i}: missing field '{field}'")
    
    assert not missing_fields, f"Missing required fields:\n" + "\n".join(missing_fields)


def test_prompt_non_empty(kd_dataset_path):
    """Test that all prompts are non-empty strings."""
    records = load_dataset_samples(kd_dataset_path, max_samples=50)
    
    empty_prompts = []
    for i, record in enumerate(records):
        prompt = record.get('prompt', '')
        if not isinstance(prompt, str) or not prompt.strip():
            empty_prompts.append(f"Record {i}: empty or invalid prompt")
    
    assert not empty_prompts, f"Empty prompts found:\n" + "\n".join(empty_prompts)


def test_teacher_text_non_empty(kd_dataset_path):
    """Test that teacher_text is non-empty and reasonable length."""
    records = load_dataset_samples(kd_dataset_path, max_samples=50)
    
    MIN_LENGTH = 50  # Minimum reasonable length
    MAX_LENGTH = 10000  # Maximum reasonable length
    
    issues = []
    for i, record in enumerate(records):
        teacher_text = record.get('teacher_text', '')
        
        if not isinstance(teacher_text, str):
            issues.append(f"Record {i}: teacher_text is not a string")
            continue
        
        text_len = len(teacher_text.strip())
        
        if text_len == 0:
            issues.append(f"Record {i}: teacher_text is empty")
        elif text_len < MIN_LENGTH:
            issues.append(f"Record {i}: teacher_text too short ({text_len} chars, min {MIN_LENGTH})")
        elif text_len > MAX_LENGTH:
            issues.append(f"Record {i}: teacher_text too long ({text_len} chars, max {MAX_LENGTH})")
    
    assert not issues, f"Teacher text issues:\n" + "\n".join(issues)


def test_metadata_structure(kd_dataset_path):
    """Test that metadata has expected structure."""
    records = load_dataset_samples(kd_dataset_path, max_samples=50)
    
    expected_metadata_keys = {'temperature', 'top_p', 'max_tokens'}
    
    issues = []
    for i, record in enumerate(records):
        metadata = record.get('metadata')
        
        if metadata is None:
            issues.append(f"Record {i}: metadata is None")
            continue
        
        if not isinstance(metadata, dict):
            issues.append(f"Record {i}: metadata is not a dict")
            continue
        
        for key in expected_metadata_keys:
            if key not in metadata:
                issues.append(f"Record {i}: metadata missing key '{key}'")
    
    assert not issues, f"Metadata structure issues:\n" + "\n".join(issues)


def test_metadata_values_valid(kd_dataset_path):
    """Test that metadata values are reasonable."""
    records = load_dataset_samples(kd_dataset_path, max_samples=50)
    
    issues = []
    for i, record in enumerate(records):
        metadata = record.get('metadata', {})
        
        # Check temperature
        temp = metadata.get('temperature')
        if temp is not None:
            if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                issues.append(f"Record {i}: invalid temperature {temp}")
        
        # Check top_p
        top_p = metadata.get('top_p')
        if top_p is not None:
            if not isinstance(top_p, (int, float)) or top_p < 0 or top_p > 1:
                issues.append(f"Record {i}: invalid top_p {top_p}")
        
        # Check max_tokens
        max_tokens = metadata.get('max_tokens')
        if max_tokens is not None:
            if not isinstance(max_tokens, int) or max_tokens < 1 or max_tokens > 100000:
                issues.append(f"Record {i}: invalid max_tokens {max_tokens}")
    
    assert not issues, f"Metadata value issues:\n" + "\n".join(issues)


def test_teacher_logits_optional(kd_dataset_path):
    """Test that teacher_logits field is optional but correctly typed when present."""
    records = load_dataset_samples(kd_dataset_path, max_samples=50)
    
    issues = []
    for i, record in enumerate(records):
        logits = record.get('teacher_logits')
        if logits is not None:
            # If present, should be a list or dict (not a string or number)
            if not isinstance(logits, (list, dict)):
                issues.append(f"Record {i}: teacher_logits has invalid type {type(logits)}")
    
    assert not issues, f"Teacher logits type issues:\n" + "\n".join(issues)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

