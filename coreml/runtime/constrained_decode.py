# coreml/runtime/constrained_decode.py
# Lightweight validator to keep function-call JSON well-formed during generation.
# @author: @darianrosebrook

import json
from jsonschema import Draft202012Validator


class JsonConstrainedDecoder:
    def __init__(self, schema: dict):
        self.validator = Draft202012Validator(schema)

    def is_valid(self, text: str) -> bool:
        try:
            obj = json.loads(text)
        except Exception:
            return False
        return self.validator.is_valid(obj)

    def close_json(self, partial: str) -> str:
        # naive bracket closure; acceptable for constrained outputs
        opens = partial.count("{") - partial.count("}")
        closes = "}" * max(0, opens)
        return partial + closes
