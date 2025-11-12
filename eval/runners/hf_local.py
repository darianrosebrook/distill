"""HuggingFace Transformers local runner."""
from __future__ import annotations
import json
import re
from typing import Any, Dict, List, Optional

from eval.runners.base import Runner


class HFLocalRunner(Runner):
    """Runner for local HuggingFace Transformers models."""
    
    def __init__(
        self,
        model: str,
        seed: int = 42,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        tokenizer_path: Optional[str] = None,
    ):
        """
        Initialize HuggingFace local runner.
        
        Args:
            model: Model path or HuggingFace model ID
            tokenizer_path: Optional separate tokenizer path
        """
        super().__init__(model, seed, temperature, max_tokens)
        self.tokenizer_path = tokenizer_path or model
        self._tokenizer = None
        self._model = None
    
    def _load_model(self):
        """Lazy load model and tokenizer."""
        if self._model is None:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch
                
                self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, use_fast=True)
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto",
                )
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
            except ImportError:
                raise ImportError("transformers and torch required for HFLocalRunner")
    
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
        """Generate using local HuggingFace model."""
        self._load_model()
        
        import torch
        
        # Build prompt with tool schemas
        prompt_with_tools = self._build_prompt_with_tools(prompt, tools)
        
        # Tokenize
        inputs = self._tokenizer(prompt_with_tools, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        temp = temperature if temperature is not None else self.temperature
        max_new_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temp if temp > 0 else None,
                do_sample=temp > 0,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        
        # Decode
        generated = self._tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        # Extract tool calls from generated text
        tool_trace = self._extract_tool_calls_from_text(generated, tools)
        
        return {
            "model_output": generated,
            "tool_trace": tool_trace,
        }
    
    def _build_prompt_with_tools(self, prompt: str, tools: List[Dict[str, Any]]) -> str:
        """Build prompt with tool schemas embedded."""
        if not tools:
            return prompt
        
        tools_desc = []
        for tool in tools:
            name = tool.get("name", "")
            desc = tool.get("description", "")
            params = tool.get("parameters", {})
            tools_desc.append(f"- {name}: {desc}")
        
        tools_text = "\n".join(tools_desc)
        return f"{prompt}\n\nAvailable tools:\n{tools_text}\n\nUse JSON format: {{\"name\": \"<tool_name>\", \"arguments\": {{...}}}}"
    
    def _extract_tool_calls_from_text(self, text: str, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract tool calls from model output text."""
        tool_trace = []
        
        # Look for JSON tool call blocks
        patterns = [
            r'TOOL_CALL:\s*(\{.*?\})',
            r'<tool_call>(.*?)</tool_call>',
            r'```json\s*(\{.*?\})\s*```',
            r'\{\s*"name"\s*:\s*"[^"]+",\s*"arguments"\s*:\s*\{[^}]*\}\s*\}',  # Inline JSON
        ]
        
        tool_names = {t.get("name", "") for t in tools}
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            for match in matches:
                json_str = match.group(1) if len(match.groups()) > 0 else match.group(0)
                try:
                    call_obj = json.loads(json_str)
                    name = call_obj.get("name", "")
                    if name in tool_names:
                        tool_trace.append({
                            "name": name,
                            "arguments": call_obj.get("arguments", {}),
                        })
                except json.JSONDecodeError:
                    # Try repair
                    try:
                        repaired = self._repair_json(json_str)
                        name = repaired.get("name", "")
                        if name in tool_names:
                            tool_trace.append({
                                "name": name,
                                "arguments": repaired.get("arguments", {}),
                            })
                    except Exception:
                        pass
        
        return tool_trace
    
    def _repair_json(self, text: str) -> Dict[str, Any]:
        """Attempt to repair malformed JSON."""
        # Remove trailing commas
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}

