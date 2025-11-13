"""
Teacher API response cache with integrity checks.

Provides caching for teacher API responses with:
- Version compatibility checks
- Prompt hash validation
- Cache hit/miss tracking
- Budget tracking
@author: @darianrosebrook
"""
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class CacheEntry:
    """Cache entry for teacher API response."""
    prompt_hash: str  # SHA-256 hash of prompt
    teacher_version: str  # Teacher model version
    response: Dict[str, Any]  # Full response payload
    created_at: float  # Timestamp
    prompt_text: str  # Original prompt (for debugging)
    metadata: Dict[str, Any]  # Additional metadata


class TeacherCache:
    """Cache manager for teacher API responses."""
    
    def __init__(self, cache_dir: Path, teacher_version: str):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory for cache files
            teacher_version: Current teacher model version
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.teacher_version = teacher_version
        self.cache_file = self.cache_dir / "teacher_cache.json"
        self._cache: Dict[str, CacheEntry] = {}
        self._stats = {
            "hits": 0,
            "misses": 0,
            "version_mismatches": 0,
            "hash_mismatches": 0,
        }
        self._load_cache()
    
    def _load_cache(self):
        """Load cache from disk."""
        if not self.cache_file.exists():
            return
        
        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
            
            for key, entry_data in data.items():
                # Convert dict back to CacheEntry
                entry = CacheEntry(**entry_data)
                self._cache[key] = entry
        except Exception as e:
            print(f"[TeacherCache] WARNING: Failed to load cache: {e}")
            self._cache = {}
    
    def _save_cache(self):
        """Save cache to disk."""
        try:
            data = {}
            for key, entry in self._cache.items():
                data[key] = asdict(entry)
            
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[TeacherCache] WARNING: Failed to save cache: {e}")
    
    def _compute_prompt_hash(self, prompt: str) -> str:
        """Compute SHA-256 hash of prompt."""
        return hashlib.sha256(prompt.encode('utf-8')).hexdigest()
    
    def _get_cache_key(self, prompt: str) -> str:
        """Generate cache key from prompt."""
        return self._compute_prompt_hash(prompt)
    
    def get(
        self,
        prompt: str,
        validate_version: bool = True,
        validate_hash: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached response for prompt.
        
        Args:
            prompt: Prompt text
            validate_version: If True, check teacher version compatibility
            validate_hash: If True, verify prompt hash matches
            
        Returns:
            Cached response if found and valid, None otherwise
        """
        cache_key = self._get_cache_key(prompt)
        
        if cache_key not in self._cache:
            self._stats["misses"] += 1
            return None
        
        entry = self._cache[cache_key]
        
        # Version check
        if validate_version and entry.teacher_version != self.teacher_version:
            self._stats["version_mismatches"] += 1
            print(f"[TeacherCache] Version mismatch: cached={entry.teacher_version}, current={self.teacher_version}")
            return None
        
        # Hash check
        if validate_hash:
            expected_hash = self._compute_prompt_hash(prompt)
            if entry.prompt_hash != expected_hash:
                self._stats["hash_mismatches"] += 1
                print(f"[TeacherCache] Hash mismatch for prompt")
                return None
        
        self._stats["hits"] += 1
        return entry.response
    
    def put(
        self,
        prompt: str,
        response: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Store response in cache.
        
        Args:
            prompt: Prompt text
            response: Teacher API response
            metadata: Optional metadata
        """
        cache_key = self._get_cache_key(prompt)
        prompt_hash = self._compute_prompt_hash(prompt)
        
        entry = CacheEntry(
            prompt_hash=prompt_hash,
            teacher_version=self.teacher_version,
            response=response,
            created_at=time.time(),
            prompt_text=prompt,
            metadata=metadata or {},
        )
        
        self._cache[cache_key] = entry
        self._save_cache()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / max(1, total_requests)
        
        return {
            **self._stats,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "cache_size": len(self._cache),
        }
    
    def clear(self, older_than_days: Optional[int] = None):
        """
        Clear cache entries.
        
        Args:
            older_than_days: If provided, only clear entries older than N days
        """
        if older_than_days is None:
            self._cache.clear()
        else:
            cutoff_time = time.time() - (older_than_days * 24 * 3600)
            keys_to_remove = [
                key for key, entry in self._cache.items()
                if entry.created_at < cutoff_time
            ]
            for key in keys_to_remove:
                del self._cache[key]
        
        self._save_cache()
    
    def migrate_cache(self, new_version: str):
        """
        Migrate cache to new teacher version.
        
        Args:
            new_version: New teacher version
        """
        print(f"[TeacherCache] Migrating cache from {self.teacher_version} to {new_version}")
        
        # Clear entries with old version
        keys_to_remove = [
            key for key, entry in self._cache.items()
            if entry.teacher_version != new_version
        ]
        
        for key in keys_to_remove:
            del self._cache[key]
        
        self.teacher_version = new_version
        self._save_cache()
        print(f"[TeacherCache] Migrated cache: removed {len(keys_to_remove)} entries")


if __name__ == "__main__":
    # Example usage
    cache_dir = Path("cache/teacher")
    cache = TeacherCache(cache_dir, teacher_version="kimi-k2-thinking-v1")
    
    prompt = "What is 2+2?"
    response = {"text": "4", "logits": [0.1, 0.2, 0.3]}
    
    # Store
    cache.put(prompt, response)
    
    # Retrieve
    cached = cache.get(prompt)
    print(f"Cached response: {cached}")
    
    # Stats
    stats = cache.get_stats()
    print(f"Cache stats: {stats}")

