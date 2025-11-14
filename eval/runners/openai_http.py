"""OpenAI-compatible HTTP runner."""

from __future__ import annotations
import hashlib
import json
import os
import re
from string import Template
from typing import Any, Dict, List, Optional

import requests

from eval.runners.base import Runner

try:
    # Optional. If missing, we fall back to string.Template
    import jinja2  # type: ignore
except Exception:  # pragma: no cover
    jinja2 = None


class OpenAIHTTPRunner(Runner):
    """Runner for OpenAI-compatible HTTP endpoints."""

    def __init__(
        self,
        model: str,
        seed: int = 42,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        prompt_wrapper: Optional[str] = None,
        determinism_mode: bool = False,
        top_p: Optional[float] = None,
    ):
        """
        Initialize OpenAI HTTP runner.

        Args:
            model: Model name (e.g., "gpt-4", "gpt-3.5-turbo")
            base_url: API base URL (defaults to OPENAI_BASE_URL env or OpenAI default)
            api_key: API key (defaults to OPENAI_API_KEY env)
            prompt_wrapper: Optional path to prompt wrapper template (Jinja2 or string.Template)
        """
        super().__init__(model, seed, temperature, max_tokens)
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.determinism_mode = determinism_mode
        self.top_p = top_p
        self._wrapper_path = prompt_wrapper
        self._wrapper_tpl = None
        self._wrapper_sha256 = None
        if prompt_wrapper:
            with open(prompt_wrapper, "r", encoding="utf-8") as f:
                content = f.read()
            self._wrapper_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if jinja2:
                # Use autoescape=True for security (prevents XSS in templates)
                # Templates are system-controlled, but autoescape is a defense-in-depth measure
                self._wrapper_tpl = jinja2.Environment(
                    autoescape=True, undefined=jinja2.StrictUndefined
                ).from_string(content)
            else:
                self._wrapper_tpl = Template(content)

    def fingerprint(self) -> Dict[str, Any]:
        """Return runner fingerprint for reproducibility."""
        fp = super().fingerprint()
        if self._wrapper_sha256:
            fp["prompt_wrapper_sha256"] = self._wrapper_sha256
        return fp

    def _render_messages(self, prompt: str, tools: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        If a wrapper is provided, render using the template. We pass:
          - system: recommended system text
          - user:   the dataset "prompt"
          - tools:  list of tool schemas/names (informational)
        The template may produce either:
          - a JSON with {"system": "...", "user": "..."} (Jinja path), or
          - a flat string (Template path) that we put as user with a default system.
        """
        system_default = (
            "You are a careful assistant. Use tools only when necessary. "
            "When a sample is a control/decline case, do not call tools."
        )
        if not self._wrapper_tpl:
            return [
                {"role": "system", "content": system_default},
                {"role": "user", "content": prompt},
            ]
        # Jinja: allow structured output
        if jinja2 and isinstance(self._wrapper_tpl, jinja2.environment.Template):
            rendered = self._wrapper_tpl.render(system=system_default, user=prompt, tools=tools)
            try:
                as_json = json.loads(rendered)
                sys_txt = as_json.get("system") or system_default
                usr_txt = as_json.get("user") or prompt
                return [
                    {"role": "system", "content": sys_txt},
                    {"role": "user", "content": usr_txt},
                ]
            except Exception:
                # treat as plain text
                return [
                    {"role": "system", "content": system_default},
                    {"role": "user", "content": rendered},
                ]
        # string.Template fallback: provide $system and $user
        rendered = self._wrapper_tpl.safe_substitute(system=system_default, user=prompt)
        return [
            {"role": "system", "content": system_default},
            {"role": "user", "content": rendered},
        ]

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
        """Generate using OpenAI-compatible HTTP API."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # Build messages using wrapper if available
        messages = self._render_messages(prompt, tools)

        # Build tools/functions format
        functions = []
        for tool in tools:
            tool_name = tool.get("name", "")
            tool_desc = tool.get("description", "")
            tool_params = tool.get("parameters", {})
            functions.append(
                {
                    "name": tool_name,
                    "description": tool_desc,
                    "parameters": tool_params,
                }
            )

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "seed": seed if seed is not None else self.seed,
        }

        # Determinism mode: enforce top_p=1.0
        if self.determinism_mode:
            payload["top_p"] = 1.0
        elif self.top_p is not None:
            payload["top_p"] = self.top_p

        if functions:
            payload["tools"] = [{"type": "function", "function": f} for f in functions]
            payload["tool_choice"] = "auto"

        if stop:
            payload["stop"] = stop

        # Make request
        # Determinism mode: disable retries, fail on any retry
        determinism_mode = getattr(self, "determinism_mode", False)
        if determinism_mode:
            # Use requests without retries (requests doesn't retry by default, but we track for determinism)
            # In determinism mode, any network retry should be considered a failure
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                # In determinism mode, fail immediately on any error (no retries)
                raise RuntimeError(f"Determinism mode: Request failed (no retries allowed): {e}")
        else:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()

        data = response.json()

        # Extract model output and tool calls
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        model_output = message.get("content", "")

        # Extract tool calls
        tool_trace = []
        tool_calls = message.get("tool_calls", [])
        for tc in tool_calls:
            function = tc.get("function", {})
            name = function.get("name", "")
            args_str = function.get("arguments", "{}")
            try:
                # Try to parse JSON arguments
                arguments = json.loads(args_str)
            except json.JSONDecodeError:
                # Try JSON repair for common issues
                arguments = self._repair_json(args_str)

            tool_trace.append(
                {
                    "name": name,
                    "arguments": arguments,
                }
            )

        # If no tool calls but model_output contains tool call patterns, try to extract
        if not tool_trace and model_output:
            tool_trace = self._extract_tool_calls_from_text(model_output, tools)

        return {
            "model_output": model_output,
            "tool_trace": tool_trace,
        }

    def _repair_json(self, text: str) -> Dict[str, Any]:
        """Attempt to repair malformed JSON."""
        # Remove trailing commas
        text = re.sub(r",\s*}", "}", text)
        text = re.sub(r",\s*]", "]", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}

    def _extract_tool_calls_from_text(
        self, text: str, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract tool calls from model output text (fallback for non-function-calling models)."""
        tool_trace = []

        # Look for JSON tool call blocks
        # Pattern: TOOL_CALL: {...} or <tool_call>...</tool_call>
        patterns = [
            r"TOOL_CALL:\s*(\{.*?\})",
            r"<tool_call>(.*?)</tool_call>",
            r"```json\s*(\{.*?\})\s*```",
        ]

        tool_names = {t.get("name", "") for t in tools}

        for pattern in patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            for match in matches:
                json_str = match.group(1)
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
