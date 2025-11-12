"""Deterministic tool broker that replays fixtures."""
from __future__ import annotations
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Tuple


class ToolBroker:
    """Deterministic tool broker that returns fixtures instead of making live calls."""
    
    def __init__(self, fixtures_dir: str):
        """
        Initialize tool broker.
        
        Args:
            fixtures_dir: Directory containing fixture JSONL files (one per tool)
        """
        self.fixtures_dir = Path(fixtures_dir)
        self.index: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        self._load_fixtures()
    
    # ------------------------
    # Normalization utilities
    # ------------------------
    @staticmethod
    def _collapse_ws(s: str) -> str:
        """Collapse whitespace to single space."""
        return re.sub(r"\s+", " ", s.strip())
    
    @classmethod
    def _normalize_args(cls, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize argument dicts so fixture matches are robust across runners:
          - Lowercase + collapse whitespace for common query fields (q, query).
          - Provide sensible defaults (e.g., top_k = 3) if missing.
          - Remove None-valued keys.
          - Sort keys via stable JSON (handled when we dump).
        """
        args = dict(args or {})
        
        # Remove None-valued keys to avoid noisy mismatches
        args = {k: v for k, v in args.items() if v is not None}
        
        # Query-like normalization
        for qk in ("q", "query"):
            if qk in args and isinstance(args[qk], str):
                args[qk] = cls._collapse_ws(args[qk].lower())
        
        # Provide a tolerant default for "top_k" if omitted
        if name in ("web.search", "web.search_async"):
            args.setdefault("top_k", 3)
        
        return args
    
    @classmethod
    def _norm_key(cls, name: str, args: Dict[str, Any]) -> Tuple[str, str]:
        """Normalize tool name and arguments to canonical form."""
        norm_args = cls._normalize_args(name, args)
        key_json = json.dumps(norm_args, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return name, key_json
    
    def _load_fixtures(self):
        """Load all fixture files."""
        if not self.fixtures_dir.exists():
            return
        
        # Look for JSONL files named after tools (e.g., web.search.jsonl, read_file.jsonl)
        for fixture_file in self.fixtures_dir.glob("*.jsonl"):
            tool_name = fixture_file.stem.replace(".", "_")  # web.search -> web_search
            self._load_tool_fixtures(tool_name, fixture_file)
    
    def _load_tool_fixtures(self, tool_name: str, fixture_file: Path):
        """Load fixtures for a specific tool."""
        with open(fixture_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    name = entry.get("name", tool_name)
                    key = entry.get("key", {})
                    result = entry.get("result", {})
                    
                    # Normalize key for lookup
                    key_name, key_norm = self._norm_key(name, key)
                    self.index[key_name][key_norm] = result
                except json.JSONDecodeError:
                    continue
    
    def call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get deterministic tool result for given name and arguments.
        
        Args:
            name: Tool name (e.g., "web.search", "read_file")
            arguments: Tool arguments dict
            
        Returns:
            Tool result dict (from fixtures) or error if not found
        """
        # Normalize tool name (handle dots vs underscores)
        normalized_name = name.replace(".", "_")
        
        # Try exact name first
        tool_fixtures = self.index.get(name, {})
        if not tool_fixtures:
            tool_fixtures = self.index.get(normalized_name, {})
        
        # Normalize arguments for lookup
        _, key_norm = self._norm_key(name, arguments or {})
        
        # Lookup result
        hit = tool_fixtures.get(key_norm)
        if hit is not None:
            return hit
        
        # Return error result
        return {
            "ok": False,
            "error": "fixture_miss",
            "name": name,
            "arguments": arguments,
            "key_normalized": key_norm,
        }
    
    def lookup(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any] | None:
        """
        Lookup tool result (alias for call, returns None on miss instead of error dict).
        
        Args:
            name: Tool name
            arguments: Tool arguments dict
            
        Returns:
            Tool result dict or None if not found
        """
        result = self.call(name, arguments)
        if result.get("ok") is False and result.get("error") == "fixture_miss":
            return None
        return result

