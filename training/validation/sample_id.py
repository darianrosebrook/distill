"""Standardized sample ID generation and deduplication.

This module provides deterministic ID generation for dataset samples,
ensuring consistent identification and enabling safe deduplication.

Author: @darianrosebrook
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional


def compute_sample_hash(
    prompt: str,
    teacher_text: Optional[str] = None,
    working_spec_id: Optional[str] = None,
    additional_fields: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Compute deterministic hash for a sample based on key content fields.
    
    Args:
        prompt: The input prompt text
        teacher_text: Optional teacher response text
        working_spec_id: Optional working spec ID
        additional_fields: Optional additional fields to include in hash
        
    Returns:
        SHA256 hash hex digest (first 16 characters for readability)
    """
    components = {
        "prompt": prompt or "",
        "teacher_text": teacher_text or "",
        "working_spec_id": working_spec_id or "",
    }
    
    if additional_fields:
        components.update(additional_fields)
    
    # Sort keys for deterministic ordering
    content_str = json.dumps(components, sort_keys=True, ensure_ascii=False)
    
    # Compute hash
    hash_obj = hashlib.sha256(content_str.encode("utf-8"))
    return hash_obj.hexdigest()[:16]


def generate_sample_id(
    prompt: str,
    teacher_text: Optional[str] = None,
    working_spec_id: Optional[str] = None,
    role: str = "worker",
    prefix: Optional[str] = None,
    additional_fields: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate a standardized sample ID.
    
    Format: {prefix}-{role}-{hash}
    Example: kd-worker-a1b2c3d4e5f6g7h8
    
    Args:
        prompt: The input prompt text
        teacher_text: Optional teacher response text
        working_spec_id: Optional working spec ID
        role: Dataset role (worker, judge, drafter)
        prefix: Optional prefix (e.g., "kd", "judge", "drafter")
        additional_fields: Optional additional fields to include in hash
        
    Returns:
        Standardized sample ID string
    """
    hash_str = compute_sample_hash(
        prompt=prompt,
        teacher_text=teacher_text,
        working_spec_id=working_spec_id,
        additional_fields=additional_fields,
    )
    
    if prefix is None:
        prefix = role
    
    return f"{prefix}-{role}-{hash_str}"


def ensure_sample_id(sample: Dict[str, Any], role: str = "worker") -> Dict[str, Any]:
    """
    Ensure a sample has a valid ID, generating one if missing.
    
    This function:
    - Uses existing ID if present and valid
    - Generates new ID from content if missing
    - Stores hash in provenance for tracking
    
    Args:
        sample: Sample dictionary
        role: Dataset role
        
    Returns:
        Sample with guaranteed ID field
    """
    # Check if ID exists and is non-empty
    existing_id = sample.get("id")
    if existing_id and isinstance(existing_id, str) and existing_id.strip():
        # ID exists, compute hash for provenance
        hash_str = compute_sample_hash(
            prompt=sample.get("prompt", ""),
            teacher_text=sample.get("teacher_text"),
            working_spec_id=sample.get("working_spec", {}).get("id") if isinstance(sample.get("working_spec"), dict) else None,
        )
        
        # Store hash in provenance if not already present
        if "provenance" not in sample:
            sample["provenance"] = {}
        if not isinstance(sample["provenance"], dict):
            sample["provenance"] = {}
        
        sample["provenance"]["hash"] = hash_str
        return sample
    
    # Generate new ID
    prompt = sample.get("prompt", "")
    teacher_text = sample.get("teacher_text")
    
    # Extract working_spec ID
    working_spec_id = None
    working_spec = sample.get("working_spec") or sample.get("caws_context", {}).get("working_spec")
    if isinstance(working_spec, dict):
        working_spec_id = working_spec.get("id")
    
    # For Judge samples, use both sides for hash
    additional_fields = None
    if role == "judge":
        a_text = sample.get("a", {}).get("text", "") if isinstance(sample.get("a"), dict) else ""
        b_text = sample.get("b", {}).get("text", "") if isinstance(sample.get("b"), dict) else ""
        additional_fields = {"a_text": a_text, "b_text": b_text}
    
    # Generate ID
    prefix = sample.get("metadata", {}).get("prefix") if isinstance(sample.get("metadata"), dict) else None
    sample_id = generate_sample_id(
        prompt=prompt,
        teacher_text=teacher_text,
        working_spec_id=working_spec_id,
        role=role,
        prefix=prefix,
        additional_fields=additional_fields,
    )
    
    sample["id"] = sample_id
    
    # Store hash in provenance
    hash_str = compute_sample_hash(
        prompt=prompt,
        teacher_text=teacher_text,
        working_spec_id=working_spec_id,
        additional_fields=additional_fields,
    )
    
    if "provenance" not in sample:
        sample["provenance"] = {}
    if not isinstance(sample["provenance"], dict):
        sample["provenance"] = {}
    
    sample["provenance"]["hash"] = hash_str
    
    return sample


def deduplicate_samples(
    samples: list[Dict[str, Any]],
    role: str = "worker",
    keep_first: bool = True,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    """
    Deduplicate samples based on content hash.
    
    Args:
        samples: List of sample dictionaries
        role: Dataset role
        keep_first: If True, keep first occurrence; if False, keep last
        
    Returns:
        Tuple of (deduplicated_samples, duplicates_removed)
    """
    seen_hashes: Dict[str, int] = {}
    deduplicated: list[Dict[str, Any]] = []
    duplicates: list[Dict[str, Any]] = []
    
    for sample in samples:
        # Ensure sample has ID
        sample = ensure_sample_id(sample, role=role)
        
        # Get hash from provenance
        hash_str = sample.get("provenance", {}).get("hash")
        if not hash_str:
            # Compute hash if not present
            hash_str = compute_sample_hash(
                prompt=sample.get("prompt", ""),
                teacher_text=sample.get("teacher_text"),
                working_spec_id=sample.get("working_spec", {}).get("id") if isinstance(sample.get("working_spec"), dict) else None,
            )
            if "provenance" not in sample:
                sample["provenance"] = {}
            sample["provenance"]["hash"] = hash_str
        
        # Check for duplicates
        if hash_str in seen_hashes:
            if keep_first:
                duplicates.append(sample)
                continue
            else:
                # Remove previous occurrence
                prev_idx = seen_hashes[hash_str]
                duplicates.append(deduplicated[prev_idx])
                deduplicated[prev_idx] = sample
                seen_hashes[hash_str] = len(deduplicated) - 1
                continue
        
        # New unique sample
        seen_hashes[hash_str] = len(deduplicated)
        deduplicated.append(sample)
    
    return deduplicated, duplicates


