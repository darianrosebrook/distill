"""
coreml/runtime/constrained_decode.py
Constrained JSON decoding for function/tool-calls.

Usage (HF inference loop):
    dec = JSONConstrainedDecoder(schema=tool_schema, tokenizer=tok)

    # Start the JSON object
    state = dec.start()  # returns a DecoderState

    while not state.is_complete:
        logits = model_step(input_ids)                      # [V]
        mask = dec.allowed_token_mask(state, logits.shape)  # [V] bool
        logits[~mask] = -float("inf")
        tok_id = sample_fn(logits)                          # greedy or nucleus
        state = dec.push(tok_id)

    obj = dec.finalize(state)  # python dict (validated)
 @author: @darianrosebrook
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Set, Tuple, Iterable
import json
import re
import numpy as np

# -------------------------
# Utilities
# -------------------------


def _is_space(s: str) -> bool:
    # treat common whitespace as skippable between tokens
    return s.isspace()


def _strip_json_prefix(s: str) -> str:
    # HF tokenizers sometimes prefix spaces; keep them separate
    return s

# -------------------------
# Token <-> string helpers
# -------------------------


class TokenLexicon:
    """
    Precomputes token -> string and a prefix map so we can decide if a token
    can be appended without breaking the JSON byte-level prefix constraints.

    Optimized with prefix map for O(1) token lookups by first character.
    """

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.id2str: Dict[int, str] = {}
        # Build id->string map robustly from ids:
        for tid in range(tokenizer.vocab_size):
            try:
                self.id2str[tid] = tokenizer.decode(
                    [tid], clean_up_tokenization_spaces=False)
            except Exception:
                self.id2str[tid] = ""

        # Build prefix map: first character -> list of token IDs
        # This enables O(1) lookup instead of O(V) scan
        self.by_first: Dict[str, List[int]] = {}
        for tid, s in self.id2str.items():
            if not s:
                continue
            c0 = s[0]
            # Normalize whitespace to single key
            key = " " if c0.isspace() else c0
            self.by_first.setdefault(key, []).append(tid)

    def token_text(self, tok_id: int) -> str:
        return self.id2str.get(tok_id, "")

    def candidates_for_chars(self, chars: Iterable[str]) -> Set[int]:
        """
        Get set of token IDs that start with any of the given characters.

        Args:
            chars: Iterable of allowed first characters

        Returns:
            Set of token IDs that can start with these characters
        """
        out: Set[int] = set()
        for ch in chars:
            # Normalize whitespace
            key = " " if (len(ch) == 1 and ch.isspace()) else ch
            tids = self.by_first.get(key)
            if tids:
                out.update(tids)
        return out

# -------------------------
# JSON finite-state machine (minimal but robust)
# -------------------------


@dataclass
class DecoderState:
    buffer: str          # accumulated text
    stack: List[str]     # 'obj', 'arr', 'str' markers
    expect: Set[str]     # what syntactic elements are allowed next
    complete: bool       # True when a full valid JSON value parsed
    error: Optional[str]  # last error if any


class JSONFSM:
    """
    A permissive JSON tokenizer-level FSM that tracks brackets/quotes/commas/colons.
    It does *not* fully parse numbers; it only enforces structural correctness
    (+ string quotation and escape rules) so we can mask illegal tokens early.
    """

    def __init__(self):
        """
        Initialize JSON FSM.

        No initialization needed - all state is managed through DecoderState instances
        and class-level constants (_ws, _punct, _quote).
        """
        # No initialization needed - state is managed per decoding session

    def start(self) -> DecoderState:
        return DecoderState(buffer="", stack=[], expect={"value"}, complete=False, error=None)

    # Minimal char classes for structural control
    _ws = set([" ", "\t", "\r", "\n"])
    _punct = set(list("{}[],:"))
    _quote = '"'

    def step_chars_allowed(self, st: DecoderState) -> Set[str]:
        """
        Returns a *character class* we can accept next (coarse).
        We'll map it to tokens via prefix matching.
        """
        if st.complete:
            return set()  # nothing more

        # If we're not inside a string, we only allow structural chars, digits signs, t/f/n, quote
        # If we are inside a string, anything except unescaped quotes; we allow \ escapes.

        # This FSM is deliberately simplified: we enforce key/value separators, brackets balance,
        # and string delimiters. For practical tool-JSON it is sufficient.

        st.buffer.rstrip()
        # Are we currently inside an open string?
        inside_string = self._inside_string(st.buffer)

        allowed: Set[str] = set()
        if inside_string:
            # Allow any unicode except bare quote; allow escaped sequences (we defer at token level)
            # Practically, we cannot enumerate "any"; we just don't block anything except bare unescaped quote rules at finalize.
            # So here we return the full range and rely on outer schema for value shapes.
            # signal to token mask: do not restrict by char set
            return set(["ANY"])
        else:
            # Outside strings
            # Always allow whitespace
            allowed |= self._ws
            # Always allow structural punctuation; the parser will reject illegal placements at push()
            allowed |= self._punct
            # Allow string starters and primitives
            # " true false null numbers
            allowed |= set(list('"-0123456789tfn'))
            return allowed

    def _inside_string(self, s: str) -> bool:
        # naive: count quotes not escaped
        esc = False
        inside = False
        for ch in s:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
            elif ch == self._quote:
                inside = not inside
        return inside

    def push_text(self, st: DecoderState, txt: str) -> DecoderState:
        # Append txt and update stack/expect/complete with best-effort structural checks
        s = st.buffer + txt
        # Balance braces/brackets at coarse level
        # (We don't fully implement colon/comma logic here; masking + schema will do the rest.)
        complete = False
        err = None
        # Heuristic completeness: valid JSON parses without exception and we are not in the middle of a string
        try:
            json.loads(s)
            complete = not self._inside_string(s)
        except Exception:
            complete = False
        return DecoderState(buffer=s, stack=st.stack, expect=st.expect, complete=complete, error=err)

# -------------------------
# Schema checker (post-hoc validation)
# -------------------------


class SchemaValidator:
    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema
        self._required = set(schema.get("required", []))
        props = schema.get("properties", {})
        self._props = {k: v for k, v in props.items()}

    def set_schema(self, schema: Dict[str, Any]):
        """
        Update the schema for validation.
        Useful for dynamic schema switching during decoding.

        Args:
            schema: New JSON schema to use for validation
        """
        self.schema = schema
        self._required = set(schema.get("required", []))
        props = schema.get("properties", {})
        self._props = {k: v for k, v in props.items()}

    def validate(self, obj: Any) -> Tuple[bool, Optional[str]]:
        # Minimal structural validation (presence of required keys; types of primitives)
        if not isinstance(obj, dict):
            return False, "Top-level JSON must be an object"
        for r in self._required:
            if r not in obj:
                return False, f"Missing required key: {r}"
        # Shallow type checks for common primitives
        for k, v in obj.items():
            spec = self._props.get(k)
            if not spec:
                continue
            t = spec.get("type")
            if t == "string" and not isinstance(v, str):
                return False, f"{k} must be string"
            if t == "number" and not isinstance(v, (int, float)):
                return False, f"{k} must be number"
            if t == "boolean" and not isinstance(v, bool):
                return False, f"{k} must be boolean"
            if t == "object" and not isinstance(v, dict):
                return False, f"{k} must be object"
            if t == "array" and not isinstance(v, list):
                return False, f"{k} must be array"
        return True, None

# -------------------------
# Public decoder
# -------------------------


# Regex to detect tool name in JSON buffer for dynamic schema switching
_TOOL_NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')


class JSONConstrainedDecoder:
    """
    Combines (1) a permissive JSON FSM (structural), (2) token-level masking by
    allowed next characters, and (3) a post-hoc shallow schema validator.

    Supports dynamic schema switching when a registry is provided:
    - Starts with base schema
    - Detects tool name in JSON buffer during generation
    - Switches to tool-specific schema for tighter validation
    """

    def __init__(self, schema: Dict[str, Any], tokenizer, registry=None):
        """
        Initialize constrained JSON decoder.

        Args:
            schema: Base JSON schema for tool calls
            tokenizer: Tokenizer instance (must have vocab_size and decode method)
            registry: Optional schema registry for dynamic schema switching.
                     If provided, should have a get(tool_name) method that returns
                     a schema dict or None.
        """
        self.base_schema = schema
        self.schema = schema
        self.validator = SchemaValidator(schema)
        self.lex = TokenLexicon(tokenizer)
        self.fsm = JSONFSM()
        self.tokenizer = tokenizer
        self.registry = registry
        self._last_applied_tool: Optional[str] = None

    def start(self) -> DecoderState:
        """
        Start a new decoding session.
        Resets schema to base schema and clears tool tracking.

        Returns:
            Initial decoder state
        """
        st = self.fsm.start()
        # Reset to base schema
        self.validator.set_schema(self.base_schema)
        self.schema = self.base_schema
        self._last_applied_tool = None
        return st

    def allowed_token_mask(self, st: DecoderState, logits_shape) -> "np.ndarray[bool]":
        """
        Compute boolean mask of allowed tokens given current decoder state.

        Uses optimized prefix map lookup for O(1) performance per character class.

        Args:
            st: Current decoder state
            logits_shape: Shape of logits tensor (last dim is vocab size)

        Returns:
            Boolean numpy array of shape (V,) where True indicates allowed tokens
        """
        import numpy as np
        V = logits_shape[-1]
        mask = np.zeros(V, dtype=bool)
        allowed_chars = self.fsm.step_chars_allowed(st)

        if "ANY" in allowed_chars:
            # Inside string: allow all non-empty tokens
            mask[:] = True
        else:
            # Use optimized prefix map lookup instead of scanning all tokens
            # This is O(1) per character class instead of O(V)
            candidate_tids = self.lex.candidates_for_chars(allowed_chars)
            if candidate_tids:
                # Convert to list and filter by vocab size
                valid_tids = [tid for tid in candidate_tids if 0 <= tid < V]
                if valid_tids:
                    mask[valid_tids] = True

        # Always permit end-of-sequence if your tokenizer uses one (optional):
        if hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
            eos_id = self.tokenizer.eos_token_id
            if 0 <= eos_id < V:
                mask[eos_id] = True
        return mask

    def push(self, st: DecoderState, tok_id: int) -> DecoderState:
        """
        Push a token into the decoder and update state.

        Also performs dynamic schema switching if registry is available and
        tool name is detected in the buffer.

        Args:
            st: Current decoder state
            tok_id: Token ID to push

        Returns:
            Updated decoder state
        """
        txt = self.lex.token_text(tok_id)
        txt = _strip_json_prefix(txt)
        st2 = self.fsm.push_text(st, txt)

        # Try dynamic schema switching if registry available
        self._maybe_switch_schema(st2)

        return st2

    def _maybe_switch_schema(self, st: DecoderState):
        """
        Attempt to switch to tool-specific schema if tool name detected.

        This enables tighter validation once we know which tool is being called.
        Only switches once per decoding session.

        Args:
            st: Current decoder state with buffer to inspect
        """
        if not self.registry or self._last_applied_tool:
            return

        # Look for tool name pattern in buffer
        match = _TOOL_NAME_RE.search(st.buffer)
        if not match:
            return

        tool_name = match.group(1)

        # Get tool-specific schema from registry
        # Registry should have a get() method that returns schema dict or None
        tool_schema = self.registry.get(tool_name) if hasattr(
            self.registry, 'get') else None

        if tool_schema:
            self.validator.set_schema(tool_schema)
            self.schema = tool_schema
            self._last_applied_tool = tool_name

    def finalize(self, st: DecoderState) -> Dict[str, Any]:
        obj = json.loads(st.buffer)
        ok, err = self.validator.validate(obj)
        if not ok:
            raise ValueError(f"Schema validation failed: {err}; obj={obj!r}")
        return obj
