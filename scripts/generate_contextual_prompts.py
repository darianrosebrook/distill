"""
Enhanced contextual prompt generator with stratified coverage, control cases,
adversarial cases, multi-lingual support, and long-context samples.

Author: @darianrosebrook
"""
from __future__ import annotations
import argparse
import json
import random
import os
import re
import hashlib
import unicodedata
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from tools.schema_registry import ToolSchemaRegistry
from scripts.util_sanitize import redact_pii, allowlist_urls, scan_safety

# Dataset version for schema evolution tracking
DATASET_VERSION = "1.1.0"

# Stratification dimensions
SCENARIOS = ["file_ops", "web_search", "code_exec", "multi_step"]
COMPLEXITY = ["single_call", "multi_call", "branching_error_recovery"]
STRUCTURES = [
    "flat_args",
    "nested_args",
    "arrays",
    "enums",
    "numeric_ranges",
    "optional_keys",
]

# Minimum coverage requirements (scenario × complexity)
MIN_COVERAGE = {
    "file_ops": {"single_call": 6, "multi_call": 4, "branching_error_recovery": 2},
    "web_search": {"single_call": 4, "multi_call": 4, "branching_error_recovery": 2},
    "code_exec": {"single_call": 3, "multi_call": 3, "branching_error_recovery": 2},
    "multi_step": {"single_call": 0, "multi_call": 4, "branching_error_recovery": 2},
}

# Fixture files (use repo:// prefix for privacy)
FIXTURE_FILES = [
    "repo:///examples/configs/student_9b_gqa.yaml",
    "repo:///examples/scripts/check_readiness.py",
    "repo:///examples/docs/coreml_notes.md",
    "repo:///examples/training/distill_process.py",
    "repo:///examples/data/kd_mix.jsonl",
]

# Multi-lingual templates
MULTILINGUAL_TASKS = {
    "es": {
        "file_ops": "Lee el archivo especificado y extrae todas las entradas de learning rate con sus números de línea.",
        "web_search": "Busca consejos de optimización de CoreML e integra el hallazgo principal en un resumen conciso.",
        "code_exec": "Ejecuta el fragmento e informa stdout/stderr y cualquier estado de salida distinto de cero.",
        "multi_step": "Lee el archivo, luego verifica sus parámetros contra las mejores prácticas que encuentres mediante búsqueda web.",
    },
    "de": {
        "file_ops": "Lese die angegebene Datei und extrahiere alle Learning-Rate-Einträge mit ihren Zeilennummern.",
        "web_search": "Suche nach CoreML-Optimierungstipps und integriere die wichtigste Erkenntnis in eine prägnante Zusammenfassung.",
        "code_exec": "Führe den Code-Snippet aus und melde stdout/stderr und jeden Exit-Status ungleich Null.",
        "multi_step": "Lese die Datei und überprüfe dann ihre Parameter gegen Best Practices, die du über Websuche findest.",
    },
    "fr": {
        "file_ops": "Lisez le fichier spécifié et extrayez toutes les entrées de learning rate avec leurs numéros de ligne.",
        "web_search": "Recherchez des conseils d'optimisation CoreML et intégrez la découverte principale dans un résumé concis.",
        "code_exec": "Exécutez le fragment et signalez stdout/stderr et tout statut de sortie non nul.",
        "multi_step": "Lisez le fichier, puis vérifiez ses paramètres par rapport aux meilleures pratiques trouvées via la recherche web.",
    },
}


def compact_caws(tier: int = 2) -> Dict[str, Any]:
    """Generate compact CAWS JSON header (≤30 tokens)."""
    return {
        "caws": {
            "tier": tier,
            "max_files": 25,
            "max_loc": 1000,
            "cov": 80,
            "mut": 50,
            "in": ["distill"],
            "out": ["infra"],
        }
    }


def build_stratified_cells(total: int) -> List[Dict[str, str]]:
    """
    Build cells with strict stratification enforcement.
    
    Ensures minimum coverage per (scenario × complexity) cell and
    at least one sample per (scenario × structure) combination.
    
    For small totals (<30), prioritizes diversity over strict minimums.
    """
    cells = []
    coverage = defaultdict(int)
    structure_coverage = defaultdict(set)

    # Calculate slots needed for diversity features upfront
    # Control: max(1, 10% of total), Adversarial: min(3, total), 
    # Multilingual: max(1, 7.5% of total), Long-context: 2-3 samples for N≥20
    want_long_context = (0 if total < 20 else min(3, max(2, total // 10)))  # 2-3 for N≥20
    slots_for_diversity = (
        max(1, int(total * 0.1)) +  # Control
        min(3, total) +              # Adversarial
        max(1, int(total * 0.075)) + # Multilingual
        want_long_context            # Long-context
    )
    max_for_minimums = max(0, total - slots_for_diversity - len(STRUCTURES) * len(SCENARIOS) // 2)  # Reserve some for structure coverage
    
    # Scale minimums for small totals or when minimums exceed available slots
    total_minimums = sum(
        MIN_COVERAGE[scenario].get(complexity, 0)
        for scenario in SCENARIOS
        for complexity in COMPLEXITY
    )
    scale_factor = min(1.0, max_for_minimums / total_minimums) if total_minimums > 0 else 1.0
    
    scaled_min_coverage = {}
    for scenario in SCENARIOS:
        scaled_min_coverage[scenario] = {}
        for complexity in COMPLEXITY:
            min_count = MIN_COVERAGE[scenario].get(complexity, 0)
            scaled_min_coverage[scenario][complexity] = max(0, int(min_count * scale_factor))

    # First pass: fill scaled minimums for (scenario × complexity), but bounded
    for scenario in SCENARIOS:
        for complexity in COMPLEXITY:
            if len(cells) >= max_for_minimums:
                break
            min_count = scaled_min_coverage[scenario].get(complexity, 0)
            for _ in range(min_count):
                if len(cells) >= max_for_minimums:
                    break
                structure = random.choice(STRUCTURES)
                cells.append(
                    {
                        "scenario": scenario,
                        "complexity": complexity,
                        "structure": structure,
                    }
                )
                coverage[(scenario, complexity)] += 1
                structure_coverage[(scenario, structure)].add(complexity)
        if len(cells) >= max_for_minimums:
            break

    # Second pass: ensure at least one per (scenario × structure), but leave room for diversity
    # Reserve slots for control/adversarial/multilingual/long-context based on what's needed
    # Control: max(1, 10% of total), Adversarial: min(3, total), Multilingual: max(1, 7.5% of total), Long-context: want_long_context
    slots_needed = max(1, int(total * 0.1)) + min(3, total) + max(1, int(total * 0.075)) + want_long_context
    available_for_structure = max(0, total - slots_needed)
    
    for scenario in SCENARIOS:
        for structure in STRUCTURES:
            if len(cells) >= available_for_structure:
                break
            if (scenario, structure) not in structure_coverage:
                # Pick a complexity that still needs samples
                complexity = random.choice(COMPLEXITY)
                cells.append(
                    {
                        "scenario": scenario,
                        "complexity": complexity,
                        "structure": structure,
                    }
                )
                coverage[(scenario, complexity)] += 1
                structure_coverage[(scenario, structure)].add(complexity)
        if len(cells) >= available_for_structure:
            break

    # Third pass: add control cases (≥10% of total, but bounded)
    num_controls = max(1, min(int(total * 0.1), total - len(cells)))
    control_behaviors = ["no_tool", "decline", "retry"]
    for i in range(num_controls):
        behavior = control_behaviors[i % len(control_behaviors)]
        scenario = random.choice(SCENARIOS)
        cells.append(
            {
                "scenario": scenario,
                "complexity": "single_call",
                "structure": random.choice(STRUCTURES),
                "expected_behaviour": behavior,
            }
        )

    # Fourth pass: add adversarial cases (at least one per type, but bounded)
    adversarial_types = ["range_violation", "malformed_json", "ambiguity"]
    num_adversarial = min(len(adversarial_types), total - len(cells))
    for i in range(num_adversarial):
        adv_type = adversarial_types[i]
        scenario = random.choice(["web_search", "file_ops", "code_exec"])
        cells.append(
            {
                "scenario": scenario,
                "complexity": "single_call",
                "structure": random.choice(STRUCTURES),
                "adversarial": {"type": adv_type, "expected": "ask_clarify"},
            }
        )

    # Fifth pass: add long-context samples (reserved upfront, 2-3 samples for N≥20)
    # Add them before multilingual to ensure they're included
    num_long_context = min(want_long_context, total - len(cells))
    for _ in range(num_long_context):
        scenario = random.choice(SCENARIOS)
        cells.append(
            {
                "scenario": scenario,
                "complexity": random.choice(COMPLEXITY),
                "structure": random.choice(STRUCTURES),
                "long_context": True,
            }
        )

    # Sixth pass: add multi-lingual samples (5-10% of total)
    num_multilingual = max(1, int(total * 0.075))  # 7.5% of total
    languages = list(MULTILINGUAL_TASKS.keys())
    for i in range(min(num_multilingual, total - len(cells))):  # Don't exceed available slots
        lang = languages[i % len(languages)]
        scenario = random.choice(SCENARIOS)
        cells.append(
            {
                "scenario": scenario,
                "complexity": random.choice(COMPLEXITY),
                "structure": random.choice(STRUCTURES),
                "language": lang,
            }
        )

    # Fill remaining slots randomly
    while len(cells) < total:
        cells.append(
            {
                "scenario": random.choice(SCENARIOS),
                "complexity": random.choice(COMPLEXITY),
                "structure": random.choice(STRUCTURES),
            }
        )

    return cells[:total]


def normalize_text(text: str) -> str:
    """Normalize text to NFC and collapse CRLF→LF."""
    # Normalize to NFC
    text = unicodedata.normalize("NFC", text)
    # Collapse CRLF→LF
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def make_long_context(paragraph: str = "log", chars: int = 9000) -> str:
    """Create neutral filler that won't trip PII/URL gates."""
    base = (paragraph + " ") * 200
    out = []
    while sum(len(s) for s in out) < chars:
        out.append(base)
    return "".join(out)[:chars]


def ensure_long_with_tokenizer(
    text: str, target_tokens: int, tokenizer=None, max_chars: int = 120_000
) -> str:
    """Ensure text meets token threshold using tokenizer if available."""
    if not tokenizer:
        return text  # Fall back to byte path
    
    try:
        current_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        while current_tokens < target_tokens and len(text) < max_chars:
            text += "\n" + make_long_context("audit trail", chars=4000)
            current_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        return text
    except Exception:
        # If tokenization fails, fall back to byte path
        return text


def generate_distraction_text() -> str:
    """Generate plausible distraction text (logs, stack traces)."""
    distractions = [
        """
[2024-01-15 10:23:45] INFO: Starting training loop
[2024-01-15 10:23:46] DEBUG: Loading checkpoint from models/student/checkpoints/latest.pt
[2024-01-15 10:23:47] INFO: Initialized optimizer with lr=2e-4
[2024-01-15 10:23:48] WARN: GPU memory usage at 85%
""",
        """
Traceback (most recent call last):
  File "train.py", line 142, in <module>
    loss = model(input_ids, labels)
  File "/usr/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1102, in __call__
    return self._call_impl(*input, **kwargs)
ValueError: Expected input tensor of shape [batch, seq_len] but got [batch, seq_len, vocab]
""",
        """
Performance metrics:
- Throughput: 12.3 samples/sec
- Memory: 8.2 GB / 16 GB
- GPU utilization: 78%
- ETA: 2h 15m
""",
    ]
    return random.choice(distractions)


def synthesize_prompt(
    cell: Dict[str, str], reg: ToolSchemaRegistry, tokenizer=None
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """Synthesize a prompt with metadata from a cell specification."""
    s = cell["scenario"]
    c = cell["complexity"]
    structure = cell.get("structure", random.choice(STRUCTURES))
    expected_behaviour = cell.get("expected_behaviour", "normal")
    adversarial = cell.get("adversarial")
    language = cell.get("language")
    long_context = cell.get("long_context", False)

    # Generate CAWS header
    caws = compact_caws(tier=random.choice([1, 2, 3]))
    header = json.dumps(caws, ensure_ascii=False)

    # Generate tool calls based on scenario and complexity
    calls = []
    if expected_behaviour in {"no_tool", "decline"}:
        # Control cases: no tool should be called
        calls = []
    elif adversarial and adversarial["type"] == "range_violation":
        # Adversarial: invalid range
        calls.append(
            {
                "name": "web.search",
                "arguments": {"q": "test", "top_k": -1},  # Invalid: negative
            }
        )
    elif adversarial and adversarial["type"] == "malformed_json":
        # Adversarial: malformed JSON (will be handled in teacher text)
        calls.append(
            {"name": "read_file", "arguments": {"path": random.choice(FIXTURE_FILES)}}
        )
    elif adversarial and adversarial["type"] == "ambiguity":
        # Adversarial: ambiguous file name
        calls.append(
            {
                "name": "read_file",
                "arguments": {"path": "repo:///examples/config.yaml"},  # Ambiguous
            }
        )
    elif s == "web_search":
        calls.append(
            {
                "name": "web.search",
                "arguments": {
                    "q": "CoreML ANE optimization",
                    "top_k": 3,
                    "site": "example.org",
                },
            }
        )
        if c != "single_call":
            calls.append(
                {
                    "name": "web.open",
                    "arguments": {"url": "https://example.org/article/coreml-ane"},
                }
            )
    elif s == "file_ops":
        calls.append(
            {
                "name": "read_file",
                "arguments": {
                    "path": random.choice(FIXTURE_FILES),
                    "encoding": "utf-8",
                },
            }
        )
    elif s == "code_exec":
        calls.append(
            {
                "name": "code.execute",
                "arguments": {
                    "language": "python",
                    "code": "print('ok')",
                    "timeout_ms": 500,
                },
            }
        )
    elif s == "multi_step":
        calls.append(
            {"name": "read_file", "arguments": {"path": random.choice(FIXTURE_FILES)}}
        )
        calls.append(
            {
                "name": "web.search",
                "arguments": {"q": "verify config parameters", "top_k": 2},
            }
        )

    # Generate task description
    if language and language in MULTILINGUAL_TASKS:
        task = MULTILINGUAL_TASKS[language][s]
    else:
        task = {
            "web_search": "Search for CoreML optimization tips and integrate the top finding into a concise summary.",
            "file_ops": "Read the specified file and extract all learning rate entries with their line numbers.",
            "code_exec": "Execute the snippet and report stdout/stderr and any nonzero exit status.",
            "multi_step": "Read the file, then cross-check its parameters against best practices you find via web search.",
        }[s]

    # Build user prompt (optionally with long filler)
    # Use token-aware generation if tokenizer provided, else byte-based
    base_user_parts = []
    if expected_behaviour == "no_tool":
        base_user_parts.append(f"{header}\n")
        base_user_parts.append(f"Task: {task}\nNote: You do not have access to any tools. Provide a response without using tools.\n")
    elif expected_behaviour == "decline":
        base_user_parts.append(f"{header}\n")
        base_user_parts.append(f"Task: {task}\nNote: The requested file does not exist. Please ask for clarification or decline politely.\n")
    elif s == "file_ops":
        target_path = calls[0]["arguments"]["path"] if calls else "unknown"
        base_user_parts.append(f"{header}\n")
        base_user_parts.append(f"You have access to file fixtures. Task: {task}\nTarget: {target_path}\n")
    elif s == "web_search":
        base_user_parts.append(f"{header}\n")
        base_user_parts.append(f"Task: {task}\nPrefer results from example.org and include the reference.\n")
    elif s == "code_exec":
        base_user_parts.append(f"{header}\n")
        base_user_parts.append(f"Task: {task}\nLanguage: python. Provide a short explanation of the output.\n")
    else:
        base_user_parts.append(f"{header}\n")
        base_user_parts.append(f"Task: {task}\nUse reading first, then search, then integrate.\n")
    
    user = "".join(base_user_parts)
    
    # Add long-context filler if needed (token-aware if tokenizer provided)
    if long_context:
        if tokenizer:
            # Token-aware: ensure we meet 8000 token threshold
            user = ensure_long_with_tokenizer(user, target_tokens=8000, tokenizer=tokenizer)
        else:
            # Byte-based fallback: use 25000 chars to exceed 24000 byte threshold
            filler = make_long_context("audit trail", chars=25000)
            user = f"{header}\n{filler}\n" + "".join(base_user_parts[1:])

    # Handle error recovery and build attempts array for retry cases
    # Skip for control cases
    attempts = []
    if c == "branching_error_recovery" and calls and expected_behaviour not in {"no_tool", "decline"}:
        if any(x["name"] == "web.open" for x in calls):
            calls[1]["arguments"]["url"] = "https://example.org/404"
            expected_behaviour = "retry"
            # Build attempts array: first call fails, second succeeds
            if len(calls) >= 2:
                attempts.append({
                    "call": calls[0],
                    "ok": True,
                    "result": {"ok": True, "summary": "First call succeeded"}
                })
                attempts.append({
                    "call": calls[1],
                    "ok": False,
                    "error": "404 Not Found"
                })
                # Add a successful retry
                retry_call = dict(calls[1])
                retry_call["arguments"]["url"] = "https://example.org/article/coreml-ane"
                attempts.append({
                    "call": retry_call,
                    "ok": True,
                    "result": {"ok": True, "summary": "Retry succeeded"}
                })

    # Determine if integration should be emitted
    # Ambiguity adversarial cases don't emit integration (they ask for clarification)
    emit_integration = (expected_behaviour not in {"no_tool", "decline"}) and not (adversarial and adversarial.get("type") == "ambiguity")
    
    # Generate teacher response
    if expected_behaviour == "no_tool":
        teacher = (
            "I understand the task, but I don't have access to tools. "
            "Based on the information provided, I can offer general guidance: "
            "For learning rate extraction, you would typically look for patterns "
            "like 'lr:', 'learning_rate:', or 'learning-rate:' in configuration files."
        )
        tool_json = None
        start_tool = None
        end_tool = None
        tool_jsons = []
        tool_results = []
    elif expected_behaviour == "decline":
        teacher = (
            "I'm sorry, but I cannot access the requested file as it doesn't exist. "
            "Could you please clarify which file you'd like me to read? "
            "Or if you meant a different path, please provide the correct location."
        )
        tool_json = None
        start_tool = None
        end_tool = None
        tool_jsons = []
        tool_results = []
    elif adversarial and adversarial["type"] == "range_violation":
        # Use normalized JSON
        corrected_call = {"name": "web.search", "arguments": {"q": "test", "top_k": 3}}
        tool_json = json.dumps(corrected_call, separators=(",", ":"), ensure_ascii=False)
        tool_result = {"ok": True, "summary": "Corrected search with top_k=3 returned valid results", "results": ["result1", "result2"]}
        teacher = (
            "I notice that top_k=-1 is invalid (must be between 1 and 10). "
            "Let me correct this and use top_k=3 instead.\n"
            f"TOOL_CALL: {tool_json}\n"
            f"TOOL_RESULT: {json.dumps(tool_result, separators=(',', ':'), ensure_ascii=False)}\n"
        )
        if emit_integration:
            summary = str(tool_result.get("summary", "")).strip()
            if summary:
                teacher += f"Integration: {summary}."
            else:
                teacher += "Integration: Based on the corrected search, here are the results..."
        start_tool = teacher.index(tool_json)
        end_tool = start_tool + len(tool_json)
        tool_jsons = [tool_json]
        tool_results = [tool_result]
    elif adversarial and adversarial["type"] == "malformed_json":
        # Use normalized JSON for corrected version
        corrected_call = {"name": "read_file", "arguments": {"path": "repo:///examples/configs/student_9b_gqa.yaml"}}
        tool_json = json.dumps(corrected_call, separators=(",", ":"), ensure_ascii=False)
        tool_result = {"ok": True, "summary": "After reading the file, I found the configuration parameters", "content": "..."}
        teacher = (
            "I'll attempt to read the file.\n"
            'TOOL_CALL: {"name": "read_file", "arguments": {"path": "repo:///examples/configs/student_9b_gqa.yaml"\n'
            "Note: The JSON is malformed (missing closing brace). Let me fix it:\n"
            f"TOOL_CALL: {tool_json}\n"
            f"TOOL_RESULT: {json.dumps(tool_result, separators=(',', ':'), ensure_ascii=False)}\n"
        )
        if emit_integration:
            summary = str(tool_result.get("summary", "")).strip()
            if summary:
                teacher += f"Integration: {summary}."
            else:
                teacher += "Integration: After reading the file, I found..."
        # Find the corrected JSON
        start_tool = teacher.rfind(tool_json)
        end_tool = start_tool + len(tool_json) if start_tool >= 0 else None
        tool_jsons = [tool_json] if start_tool >= 0 else []
        tool_results = [tool_result] if start_tool >= 0 else []
    elif adversarial and adversarial["type"] == "ambiguity":
        teacher = (
            "I notice there are multiple files matching 'config.yaml': "
            "repo:///examples/configs/student_9b_gqa.yaml and "
            "repo:///examples/configs/student_8b_gqa.yaml. "
            "Could you please clarify which one you'd like me to read?"
        )
        tool_json = None
        start_tool = None
        end_tool = None
        tool_jsons = []
        tool_results = []
    else:
        # Normal case
        if calls:
            # Use normalized JSON for consistent formatting
            tool_jsons = []
            tool_results = []
            for call in calls:
                tool_json = json.dumps(call, separators=(",", ":"), ensure_ascii=False)
                tool_jsons.append(tool_json)
                # Create tool result with fields for grounding
                tool_result = {
                    "ok": True,
                    "summary": "ANE prefers fp16 kernels when ops are supported",
                    "lines": 128
                }
                tool_results.append(tool_result)
            
            # Build teacher response with all calls
            teacher_parts = ["I will start with a tool call.\n"]
            for i, (tool_json, tool_result) in enumerate(zip(tool_jsons, tool_results)):
                teacher_parts.append(f"TOOL_CALL: {tool_json}\n")
                teacher_parts.append(f"TOOL_RESULT: {json.dumps(tool_result, separators=(',', ':'), ensure_ascii=False)}\n")
            
            if emit_integration:
                # Always include the summary from tool_result to ensure grounding
                summary = str(tool_results[0].get("summary", "")).strip() if tool_results else ""
                if summary:
                    teacher_parts.append(f"Integration: {summary}.")
                else:
                    teacher_parts.append("Integration: Based on the result, the top insight is: ANE prefers fp16 kernels when ops are supported.")
            
            teacher = "".join(teacher_parts)
            # For single call, use first tool_json for backward compatibility
            tool_json = tool_jsons[0] if tool_jsons else None
            start_tool = teacher.index(tool_jsons[0]) if tool_jsons else None
            end_tool = start_tool + len(tool_jsons[0]) if start_tool is not None else None
        else:
            teacher = "I'll proceed with the task."
            tool_json = None
            start_tool = None
            end_tool = None
            tool_jsons = []
            tool_results = []

    # Normalize text BEFORE any span calculations or JSON embedding
    teacher = normalize_text(teacher)
    user = normalize_text(user)

    # Extract integration spans using robust regex (after normalization)
    integration_spans_bytes = []
    if teacher and calls and emit_integration:
        # Use regex to find all Integration: sentences
        for m in re.finditer(r'Integration:\s*([^\n]+?)(?:[\.!?…]\s|$)', teacher, flags=re.UNICODE):
            integration_spans_bytes.append([m.start(1), m.end(1)])

    # Sanitize and scan safety (after normalization)
    teacher = redact_pii(teacher)
    ok_urls = allowlist_urls(teacher)
    safety_scan = scan_safety(teacher + user)

    # Build metadata with all required fields
    meta = {
        "dataset_version": DATASET_VERSION,
        "call_sequence": calls,
        "scenario": s,
        "complexity": c,
        "structure": structure,
        "expected_behaviour": expected_behaviour,
        "url_allowlist_ok": ok_urls,
        "caws_header_ok": True,
        "text_norm": "NFC",
        "line_endings": "LF",
        "safety_scan": safety_scan,
        "spans_index_text": {
            "teacher": "NFC_LF",
            "user": "NFC_LF"
        },
    }

    # Add language tag (default to "en" if not multilingual)
    meta["lang"] = language if language else "en"
    
    # Add attempts array for retry cases
    if attempts:
        meta["attempts"] = attempts

    # Compute tool spans (exact anchoring for single and multi-call)
    json_args_spans_bytes = []
    tool_name_spans_bytes = []
    json_pointers = []
    
    if calls and tool_jsons:
        for i, (call, tool_json) in enumerate(zip(calls, tool_jsons)):
            # Find this tool_json in teacher text
            tool_start = teacher.find(tool_json)
            if tool_start >= 0:
                tool_end = tool_start + len(tool_json)
                json_args_spans_bytes.append([tool_start, tool_end])
                
                # Exact name anchoring: find "name":"..." inside the JSON substring
                inner = teacher[tool_start:tool_end]
                name_pair = f'"name":"{call["name"]}"'
                if name_pair in inner:
                    name_rel = inner.index(name_pair)
                    name_abs_start = tool_start + name_rel + len('"name":"')
                    name_abs_end = name_abs_start + len(call["name"])
                    tool_name_spans_bytes.append([name_abs_start, name_abs_end])
                
                # JSON pointer for semantic anchoring
                json_pointers.append(f"/calls/{i}/arguments" if len(calls) > 1 else "/arguments")
        
        # For backward compatibility, also set single spans if only one call
        if len(calls) == 1 and json_args_spans_bytes:
            meta["json_args_span_bytes"] = json_args_spans_bytes[0]
            if tool_name_spans_bytes:
                meta["tool_name_span_bytes"] = tool_name_spans_bytes[0]
            meta["json_pointer_args"] = json_pointers[0]
            if tool_name_spans_bytes:
                meta["json_pointer_name"] = "/name"
        
        # Multi-call spans (always set for multi-call, also for single for consistency)
        meta["json_args_spans_bytes"] = json_args_spans_bytes
        if tool_name_spans_bytes:
            meta["tool_name_spans_bytes"] = tool_name_spans_bytes
        if len(calls) > 1:
            meta["json_pointers"] = json_pointers

    # Add tool result fields for grounding checks
    if tool_results:
        tool_result_fields = {}
        for result in tool_results:
            for key, value in result.items():
                tool_result_fields[key] = str(value)
        meta["tool_result_fields"] = tool_result_fields

    if integration_spans_bytes:
        meta["integration_spans_bytes"] = integration_spans_bytes

    if adversarial:
        meta["adversarial"] = adversarial

    if long_context:
        meta["long_context"] = True

    return user, [{"role": "assistant", "content": teacher}], meta


def generate_sample_id(seed: Optional[int] = None, index: int = 0) -> str:
    """Generate deterministic sample ID."""
    if seed is not None:
        rng = random.Random(seed + index)
        return f"sample_{rng.randint(100000, 999999)}"
    return f"sample_{random.randint(100000, 999999)}"


def main():
    ap = argparse.ArgumentParser(
        description="Generate contextual prompts with stratified coverage"
    )
    ap.add_argument("--out", required=True, help="Output JSONL file path")
    ap.add_argument("--total", type=int, default=60, help="Total number of samples")
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic generation",
    )
    ap.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer path for token-aware long-context generation (optional)",
    )
    ap.add_argument(
        "--enforce-stratification",
        action="store_true",
        help="Enforce strict stratification (default: True)",
    )
    args = ap.parse_args()

    # Set seed for determinism
    if args.seed is not None:
        random.seed(args.seed)

    # Load tokenizer if provided
    tokenizer = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
        except Exception:
            print(f"[generate_contextual_prompts] WARN: Failed to load tokenizer {args.tokenizer}, using byte-based thresholds")

    reg = ToolSchemaRegistry()
    cells = build_stratified_cells(args.total)
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)

    # Track generation plan with adversarial taxonomy
    adversarial_counts = defaultdict(int)
    for cell in cells:
        if "adversarial" in cell:
            adv_type = cell["adversarial"].get("type", "unknown")
            adversarial_counts[adv_type] += 1
    
    generation_plan = {
        "total": args.total,
        "seed": args.seed,
        "dataset_version": DATASET_VERSION,
        "counts": {
            "control": sum(1 for c in cells if "expected_behaviour" in c and c["expected_behaviour"] != "normal"),
            "adversarial": sum(1 for c in cells if "adversarial" in c),
            "adversarial_by_type": dict(adversarial_counts),
            "multilingual": sum(1 for c in cells if "language" in c),
            "long_context": sum(1 for c in cells if c.get("long_context")),
        }
    }

    # Generate samples with provenance
    items = []
    for i, cell in enumerate(cells):
        prompt, history, meta = synthesize_prompt(cell, reg, tokenizer=tokenizer)
        
        # Add provenance
        sample_id = generate_sample_id(args.seed, i)
        meta["sample_id"] = sample_id
        meta["provenance"] = {
            "seed": args.seed,
            "scenario": cell.get("scenario"),
            "tags": [k for k in ["control", "adversarial", "multilingual", "long_context"] if cell.get(k)],
        }
        
        item = {
            "prompt": prompt,
            "teacher_text": history[0]["content"],
            "metadata": meta,
        }
        items.append(item)

    # Write to file and compute SHA256
    with open(args.out, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    
    # Compute SHA256 for integrity
    with open(args.out, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    
    generation_plan["sha256"] = file_hash

    print(f"[generate_contextual_prompts] Generated {len(cells)} samples to {args.out}")
    print(f"[generate_contextual_prompts] Generation plan: {json.dumps(generation_plan, indent=2)}")
    print(f"[generate_contextual_prompts] SHA256: {file_hash}")


if __name__ == "__main__":
    main()

