"""HuggingFace Transformers local runner."""

from __future__ import annotations
import hashlib
import json
import re
from string import Template
from typing import Any, Dict, List, Optional

from eval.runners.base import Runner

try:
    import jinja2  # type: ignore
except Exception:  # pragma: no cover
    jinja2 = None


class HFLocalRunner(Runner):
    """Runner for local HuggingFace Transformers models."""

    def __init__(
        self,
        model: str,
        seed: int = 42,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        tokenizer_path: Optional[str] = None,
        prompt_wrapper: Optional[str] = None,
    ):
        """
        Initialize HuggingFace local runner.

        Args:
            model: Model path or HuggingFace model ID
            tokenizer_path: Optional separate tokenizer path
            prompt_wrapper: Optional path to prompt wrapper template (Jinja2 or string.Template)
        """
        super().__init__(model, seed, temperature, max_tokens)
        self.tokenizer_path = tokenizer_path or model
        self._tokenizer = None
        self._model = None
        self._wrapper_path = prompt_wrapper
        self._wrapper_tpl = None
        self._wrapper_sha256 = None
        if prompt_wrapper:
            with open(prompt_wrapper, "r", encoding="utf-8") as f:
                content = f.read()
            self._wrapper_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if jinja2:
                self._wrapper_tpl = jinja2.Environment(
                    autoescape=False, undefined=jinja2.StrictUndefined
                ).from_string(content)
            else:
                self._wrapper_tpl = Template(content)

    def _load_model(self):
        """Lazy load model and tokenizer."""
        if self._model is None:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch

                from training.safe_model_loading import (
                    safe_from_pretrained_tokenizer,
                    safe_from_pretrained_causal_lm,
                )
                self._tokenizer = safe_from_pretrained_tokenizer(self.tokenizer_path, use_fast=True)
                self._model = safe_from_pretrained_causal_lm(
                    self.model,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto",
                )
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
            except ImportError:
                raise ImportError("transformers and torch required for HFLocalRunner")

    def fingerprint(self) -> Dict[str, Any]:
        """Return runner fingerprint for reproducibility."""
        fp = super().fingerprint()
        if self._wrapper_sha256:
            fp["prompt_wrapper_sha256"] = self._wrapper_sha256
        return fp

    def _render_text(self, prompt: str, tools: List[Dict[str, Any]]) -> str:
        """
        Local models typically ingest a single concatenated prompt. Wrapper can
        return either:
          - JSON {"system": "...", "user": "..."} which we join as "<s>[SYSTEM]..</s>\n[USER].."
          - A flat string we pass through.
        """
        system_default = (
            "You are a careful assistant. Use tools only when necessary. "
            "For control/decline items, do not call tools."
        )
        if not self._wrapper_tpl:
            return f"<s>[SYSTEM]{system_default}</s>\n[USER]{prompt}"
        if jinja2 and isinstance(self._wrapper_tpl, jinja2.environment.Template):
            rendered = self._wrapper_tpl.render(system=system_default, user=prompt, tools=tools)
            try:
                as_json = json.loads(rendered)
                sys_txt = as_json.get("system") or system_default
                usr_txt = as_json.get("user") or prompt
                return f"<s>[SYSTEM]{sys_txt}</s>\n[USER]{usr_txt}"
            except Exception:
                return rendered
        return self._wrapper_tpl.safe_substitute(system=system_default, user=prompt)

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
        # Apply wrapper if available
        merged = self._render_text(prompt_with_tools, tools)

        # Tokenize the merged prompt
        inputs = self._tokenizer(merged, return_tensors="pt")
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
        generated = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

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
            tool.get("parameters", {})
            tools_desc.append(f"- {name}: {desc}")

        tools_text = "\n".join(tools_desc)
        return f'{prompt}\n\nAvailable tools:\n{tools_text}\n\nUse JSON format: {{"name": "<tool_name>", "arguments": {{...}}}}'

    def _extract_tool_calls_from_text(
        self, text: str, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract tool calls from model output text."""
        tool_trace = []

        # Look for JSON tool call blocks
        patterns = [
            r"TOOL_CALL:\s*(\{.*?\})",
            r"<tool_call>(.*?)</tool_call>",
            r"```json\s*(\{.*?\})\s*```",
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
                        tool_trace.append(
                            {
                                "name": name,
                                "arguments": call_obj.get("arguments", {}),
                            }
                        )
                except json.JSONDecodeError:
                    # Try repair
                    try:
                        repaired = self._repair_json(json_str)
                        name = repaired.get("name", "")
                        if name in tool_names:
                            tool_trace.append(
                                {
                                    "name": name,
                                    "arguments": repaired.get("arguments", {}),
                                }
                            )
                    except Exception:
                        pass

        return tool_trace

    def _repair_json(self, text: str) -> Dict[str, Any]:
        """Attempt to repair malformed JSON."""
        # Remove trailing commas
        text = re.sub(r",\s*}", "}", text)
        text = re.sub(r",\s*]", "]", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}
