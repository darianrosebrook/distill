"""CI smoke test for broker fixture hit rate."""
from __future__ import annotations
import json
import os
import re
from pathlib import Path

import pytest

from eval.tool_broker.broker import ToolBroker


FIXTURES_DIR = Path("eval/tool_broker/fixtures")


def _variant_args(name: str, args: dict):
    """
    Generate argument variants that exercise the broker's normalization:
      - whitespace and case changes for query fields
      - dropping optional keys like top_k
      - inserting None-valued keys (should be ignored)
    """
    args = dict(args or {})
    variants = [dict(args)]
    
    # Query normalization (q/query)
    for qk in ("q", "query"):
        if qk in args and isinstance(args[qk], str):
            v = args[qk]
            variants.append({**args, qk: re.sub(r"\s+", "  ", v.upper())})
            variants.append({**args, qk: f"  {v}   "})
    
    # Omit top_k for web.search* (broker should default to 3)
    if name in ("web.search", "web.search_async") and "top_k" in args:
        v2 = dict(args)
        v2.pop("top_k")
        variants.append(v2)
    
    # None-valued keys should be dropped
    variants.append({**args, "unused": None})
    
    # Deduplicate variants
    dedup = []
    seen = set()
    for v in variants:
        key = tuple(sorted(v.items()))
        if key not in seen:
            seen.add(key)
            dedup.append(v)
    return dedup


@pytest.mark.smoke
def test_broker_fixtures_hit_rate():
    """
    Read every record from every fixture JSONL file and ensure the broker
    returns a result for ≥95% of (record, variant) lookups.
    """
    assert FIXTURES_DIR.exists(), f"Missing fixtures dir: {FIXTURES_DIR}"
    broker = ToolBroker(str(FIXTURES_DIR))
    
    total = 0
    hits = 0
    per_file_stats = []
    
    for fn in os.listdir(FIXTURES_DIR):
        if not fn.endswith(".jsonl"):
            continue
        path = FIXTURES_DIR / fn
        file_total = 0
        file_hits = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                name = rec["name"]
                key = rec.get("key", {})
                # Generate key variants that should still match
                for variant in _variant_args(name, key):
                    total += 1
                    file_total += 1
                    if broker.lookup(name, variant) is not None:
                        hits += 1
                        file_hits += 1
        if file_total:
            per_file_stats.append((fn, file_hits / file_total))
    
    assert total > 0, "No fixture records found to test"
    hit_rate = hits / total
    
    # Helpful diagnostics on failure
    if hit_rate < 0.95:
        details = "\n".join(f"- {fn}: {rate:.3f}" for fn, rate in per_file_stats)
        pytest.fail(f"Broker fixture hit-rate {hit_rate:.3f} < 0.95\nPer-file rates:\n{details}")
    
    # Soft assertion for very high coverage (not a gate)
    assert hit_rate >= 0.95


def test_broker_miss_for_unknown_key():
    """Test that broker returns None for unknown tools/arguments."""
    broker = ToolBroker(str(FIXTURES_DIR))
    # Unknown tool
    assert broker.lookup("nonexistent.tool", {"q": "x"}) is None
    # Known tool, non-matching arguments
    assert broker.lookup("web.search", {"q": "this string should never be in fixtures", "top_k": 3}) is None

