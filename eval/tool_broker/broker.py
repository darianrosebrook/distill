"""Deterministic tool broker that replays fixtures."""
from __future__ import annotations
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


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
                    key_str = self._normalize_key(key)
                    self.index[name][key_str] = result
                except json.JSONDecodeError:
                    continue
    
    def _normalize_key(self, key: Dict[str, Any]) -> str:
        """
        Normalize tool arguments to a canonical string for lookup.
        
        Args:
            key: Tool arguments dict
            
        Returns:
            Canonical string representation
        """
        # Sort keys and normalize values
        normalized = {}
        for k, v in sorted(key.items()):
            if isinstance(v, str):
                normalized[k] = v.lower().strip()
            elif isinstance(v, (int, float)):
                normalized[k] = v
            elif isinstance(v, list):
                normalized[k] = tuple(sorted(v) if all(isinstance(x, (str, int, float)) for x in v) else v)
            elif isinstance(v, dict):
                normalized[k] = self._normalize_key(v)
            else:
                normalized[k] = v
        
        # Create canonical JSON (sorted keys, no whitespace)
        return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    
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
        key_str = self._normalize_key(arguments)
        
        # Lookup result
        result = tool_fixtures.get(key_str)
        
        if result is None:
            # Return error result
            return {
                "ok": False,
                "error": "fixture_miss",
                "name": name,
                "arguments": arguments,
                "key_normalized": key_str,
            }
        
        return result

