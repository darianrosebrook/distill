from __future__ import annotations
import re
from jsonschema import Draft202012Validator
from typing import Any, Dict

SENSITIVE_KEYS = {"password", "token", "secret", "authorization", "api_key", "apikey", "auth"}


def redact(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            k: ("[REDACTED]" if k.lower() in SENSITIVE_KEYS else redact(v)) for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [redact(x) for x in obj]
    if isinstance(obj, str):
        obj = re.sub(r"[\w\.-]+@[\w\.-]+", "[REDACTED_EMAIL]", obj)
        obj = re.sub(r"[0-9a-fA-F-]{36}", "[REDACTED_UUID]", obj)
        return obj
    return obj


def validate_trace(trace: Dict[str, Any], schema: Dict[str, Any]) -> None:
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(trace), key=lambda e: e.path)
    if errors:
        messages = []
        for e in errors:
            path = "/".join(map(str, e.path))
            messages.append(f"{path}: {e.message}")
        raise ValueError("Trace validation failed:\n" + "\n".join(messages))
