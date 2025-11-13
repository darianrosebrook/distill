from typing import Iterator, Dict, Any


def load_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    import json

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)
