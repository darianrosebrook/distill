"""Base runner interface for evaluation harness."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class Runner(ABC):
    """Base interface for model runners."""
    
    def __init__(self, model: str, seed: int = 42, temperature: float = 0.0, max_tokens: int = 1024):
        """
        Initialize runner.
        
        Args:
            model: Model identifier (name or path)
            seed: Random seed for deterministic generation
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.model = model
        self.seed = seed
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        *,
        stop: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate model output with tool calls.
        
        Args:
            prompt: Input prompt
            tools: List of tool schemas (name, description, parameters)
            stop: Stop sequences
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            seed: Override default seed
            
        Returns:
            Dict with:
                - "model_output": str (full model response text)
                - "tool_trace": List[Dict] (tool calls with name, arguments)
        
        Raises:
            NotImplementedError: If subclass does not implement this method
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.generate() must be implemented by subclass"
        )
    
    def fingerprint(self) -> Dict[str, Any]:
        """Return runner fingerprint for reproducibility."""
        return {
            "runner_type": self.__class__.__name__,
            "model": self.model,
            "seed": self.seed,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
    
    def model_fingerprint(self) -> Dict[str, Any]:
        """Return model-specific fingerprint."""
        return {"model": self.model}

