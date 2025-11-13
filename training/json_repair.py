"""
JSON repair utilities for training-time validation.

Provides grammar-based JSON repair for invalid JSON generated during training.
Used to penalize models that generate invalid JSON and track repair metrics.
"""
import json
import re
from typing import Tuple, Optional, Dict, Any

# Try to import jsonrepair, but make it optional
try:
    import jsonrepair
    JSONREPAIR_AVAILABLE = True
except ImportError:
    JSONREPAIR_AVAILABLE = False


def validate_json(text: str) -> bool:
    """
    Check if text contains valid JSON.
    
    Args:
        text: Text to check
        
    Returns:
        True if valid JSON found, False otherwise
    """
    # Try to find JSON in text
    json_patterns = [
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Simple JSON object
        r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]',  # JSON array
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                json.loads(match)
                return True
            except json.JSONDecodeError:
                continue
    
    # Try parsing entire text
    try:
        json.loads(text.strip())
        return True
    except json.JSONDecodeError:
        pass
    
    return False


def extract_json_from_text(text: str) -> Optional[str]:
    """
    Extract JSON string from text.
    
    Args:
        text: Text that may contain JSON
        
    Returns:
        JSON string if found, None otherwise
    """
    # Look for JSON objects
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text)
    
    for match in matches:
        try:
            # Validate it's valid JSON
            json.loads(match)
            return match
        except json.JSONDecodeError:
            continue
    
    # Try parsing entire text
    try:
        json.loads(text.strip())
        return text.strip()
    except json.JSONDecodeError:
        pass
    
    return None


def simple_json_repair(invalid_json: str) -> Optional[str]:
    """
    Simple grammar-based JSON repair (without jsonrepair library).
    
    Handles common issues:
    - Missing quotes around keys
    - Trailing commas
    - Single quotes instead of double quotes
    
    Args:
        invalid_json: Invalid JSON string
        
    Returns:
        Repaired JSON string if repairable, None otherwise
    """
    repaired = invalid_json.strip()
    
    # Fix single quotes to double quotes
    repaired = re.sub(r"'([^']*)':", r'"\1":', repaired)
    repaired = re.sub(r":\s*'([^']*)'", r': "\1"', repaired)
    
    # Remove trailing commas before } or ]
    repaired = re.sub(r',(\s*[}\]])', r'\1', repaired)
    
    # Try to fix missing quotes around keys (simple cases)
    # This is limited - full repair needs jsonrepair
    
    return repaired


def repair_json(invalid_json: str, use_jsonrepair: bool = True) -> Tuple[bool, Optional[Dict], bool]:
    """
    Attempt to repair invalid JSON.
    
    Strategy:
    1. Try parsing first (might already be valid)
    2. Try grammar-based repair (simple fixes)
    3. Try jsonrepair library if available (more sophisticated)
    
    Args:
        invalid_json: Invalid JSON string
        use_jsonrepair: Whether to use jsonrepair library if available
        
    Returns:
        (success, repaired_dict, was_repaired)
        - success: True if JSON is valid (original or repaired)
        - repaired_dict: Parsed JSON dict if successful, None otherwise
        - was_repaired: True if repair was needed, False if already valid
    """
    # Step 1: Try parsing first
    try:
        obj = json.loads(invalid_json)
        return True, obj, False
    except json.JSONDecodeError:
        pass
    
    # Step 2: Try simple grammar-based repair
    simple_repaired = simple_json_repair(invalid_json)
    if simple_repaired:
        try:
            obj = json.loads(simple_repaired)
            return True, obj, True
        except json.JSONDecodeError:
            pass
    
    # Step 3: Try jsonrepair library if available and requested
    if use_jsonrepair and JSONREPAIR_AVAILABLE:
        try:
            repaired_text = jsonrepair.repair_json(invalid_json)
            repaired_obj = json.loads(repaired_text)
            return True, repaired_obj, True
        except Exception:
            pass
    
    # All repair attempts failed
    return False, None, True


def check_json_repair_needed(text: str, use_jsonrepair: bool = True) -> Tuple[bool, bool]:
    """
    Check if JSON repair is needed for text.
    
    Args:
        text: Text that may contain JSON
        use_jsonrepair: Whether to use jsonrepair library if available
        
    Returns:
        (has_json, needs_repair)
        - has_json: True if text contains JSON-like content
        - needs_repair: True if JSON is invalid and needs repair
    """
    # Extract JSON from text
    json_str = extract_json_from_text(text)
    
    if json_str is None:
        return False, False
    
    # Check if valid
    is_valid = validate_json(json_str)
    
    if is_valid:
        return True, False
    
    # Try repair to see if it's repairable
    success, _, was_repaired = repair_json(json_str, use_jsonrepair=use_jsonrepair)
    
    if success:
        return True, was_repaired
    else:
        return True, True  # Has JSON but can't be repaired


def batch_check_json_repair(texts: list[str], use_jsonrepair: bool = True) -> Dict[str, Any]:
    """
    Check JSON repair needs for a batch of texts.
    
    Args:
        texts: List of text strings
        use_jsonrepair: Whether to use jsonrepair library if available
        
    Returns:
        Dictionary with metrics:
        - total: Total number of texts
        - has_json_count: Number of texts with JSON
        - valid_json_count: Number of texts with valid JSON
        - needs_repair_count: Number of texts needing repair
        - repair_rate: Percentage needing repair
    """
    total = len(texts)
    has_json_count = 0
    valid_json_count = 0
    needs_repair_count = 0
    
    for text in texts:
        has_json, needs_repair = check_json_repair_needed(text, use_jsonrepair=use_jsonrepair)
        
        if has_json:
            has_json_count += 1
            if not needs_repair:
                valid_json_count += 1
            else:
                needs_repair_count += 1
    
    repair_rate = needs_repair_count / total if total > 0 else 0.0
    
    return {
        "total": total,
        "has_json_count": has_json_count,
        "valid_json_count": valid_json_count,
        "needs_repair_count": needs_repair_count,
        "repair_rate": repair_rate,
    }

