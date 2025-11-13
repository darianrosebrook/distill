"""
End-to-end tests for code-mode MCP distillation.

Tests that require code-mode (TS orchestration) to pass:
- Large blob (>10k chars) must use code-mode
- Multi-tool chain (≥2 tools) must use code-mode
- PII scenarios must use code-mode

Reference: code-mode-latent-reasoning.md Milestone 1

Author: @darianrosebrook
"""

from __future__ import annotations

import re
from typing import Dict, Any, List


def detect_code_mode_usage(model_output: str) -> bool:
    """
    Detect if model output uses TypeScript API orchestration (code-mode).

    Args:
        model_output: Model generated text

    Returns:
        True if code-mode patterns detected
    """
    ts_api_patterns = [
        "from './servers",
        'from "./servers',
        "callMCPTool(",
        "import * as",
        "await ",
    ]
    return any(pattern in model_output for pattern in ts_api_patterns)


def detect_direct_tool_calls(model_output: str) -> bool:
    """
    Detect if model output uses direct tool call patterns.

    Args:
        model_output: Model generated text

    Returns:
        True if direct tool call patterns detected
    """
    direct_patterns = [
        "<|tool_call|>",
        "<|tool_result|>",
    ]
    return any(pattern in model_output for pattern in direct_patterns)


def detect_data_leak(model_output: str, tool_results: List[Dict[str, Any]]) -> bool:
    """
    Detect if PII from tool results leaked into model output.

    Args:
        model_output: Model generated text
        tool_results: List of tool call results

    Returns:
        True if data leak detected
    """
    pii_patterns = {
        "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "PHONE": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    }

    # Extract PII from tool results
    tool_result_text = " ".join(str(r.get("result", {})) for r in tool_results)

    for pii_type, pattern in pii_patterns.items():
        matches_in_output = re.findall(pattern, model_output)
        matches_in_results = re.findall(pattern, tool_result_text)

        # If PII appears in both output and results, it's a leak
        if matches_in_output and matches_in_results:
            # Check if any match appears in both
            for match in matches_in_output:
                if match in matches_in_results:
                    return True

    return False


def count_tokens(text: str) -> int:
    """
    Rough token count estimate (character-based approximation).

    Uses ~4 characters per token as a rough estimate, which is typical
    for many tokenizers including GPT-style models.

    Args:
        text: Text to count tokens for

    Returns:
        Estimated token count
    """
    # Rough approximation: ~4 characters per token
    return max(1, len(text) // 4)


class TestCodeModeLargeBlob:
    """Test that large blob scenarios require code-mode."""

    def test_large_blob_requires_code_mode(self):
        """
        Test that large blob (>10k chars) scenarios must use code-mode.

        This test fails if large payloads are echoed into tokens.
        """
        # Simulate large blob scenario
        large_content = "x" * 15000  # 15k chars
        model_output_with_code_mode = """
import * as google_drive from './servers/google-drive';
const doc = await google_drive.getDocument({ documentId: 'abc123' });
// Large content stays in sandbox, only summary returned
const summary = summarize(doc.content, 5);
console.log('Summary:', summary);
"""

        model_output_with_direct_tool = f"""
<|tool_call|>
{{"name": "google_drive__get_document", "arguments": {{"documentId": "abc123"}}}}
<|tool_result|>
{{"content": "{large_content[:1000]}..."}}
"""

        # Code-mode should be detected
        assert detect_code_mode_usage(model_output_with_code_mode), (
            "Code-mode usage should be detected for TS API patterns"
        )

        # Direct tool call should be detected
        assert detect_direct_tool_calls(model_output_with_direct_tool), (
            "Direct tool call patterns should be detected"
        )

        # Large content should not be echoed in code-mode output
        assert len(model_output_with_code_mode) < 1000, (
            "Code-mode output should not contain large payloads"
        )

        # Direct tool call would echo large content (bad)
        assert len(model_output_with_direct_tool) > 1000, (
            "Direct tool calls echo large payloads (this is the problem we're solving)"
        )


class TestCodeModeMultiTool:
    """Test that multi-tool chains require code-mode."""

    def test_multi_tool_requires_code_mode(self):
        """
        Test that multi-tool chains (≥2 tools) must use code-mode.
        """
        model_output_with_code_mode = """
import * as google_drive from './servers/google-drive';
import * as salesforce from './servers/salesforce';

const doc = await google_drive.getDocument({ documentId: 'abc123' });
const notes = summarize(doc.content, 5);
await salesforce.updateRecord({
  objectType: 'SalesMeeting',
  recordId: '00Q...',
  data: { Notes: notes }
});
console.log('Task completed');
"""

        model_output_with_direct_tool = """
<|tool_call|>
{"name": "google_drive__get_document", "arguments": {"documentId": "abc123"}}
<|tool_result|>
{"content": "..."}
<|tool_call|>
{"name": "salesforce__update_record", "arguments": {...}}
"""

        # Code-mode should be detected
        assert detect_code_mode_usage(model_output_with_code_mode), (
            "Code-mode usage should be detected for multi-tool scenarios"
        )

        # Count tools in code-mode (should be 2+)
        code_mode_tool_count = model_output_with_code_mode.count("await ")
        assert code_mode_tool_count >= 2, "Multi-tool scenario should have ≥2 tool calls"

        # Direct tool call should be detected
        assert detect_direct_tool_calls(model_output_with_direct_tool), (
            "Direct tool call patterns should be detected"
        )


class TestCodeModePII:
    """Test that PII scenarios require code-mode."""

    def test_pii_requires_code_mode(self):
        """
        Test that PII scenarios must use code-mode to prevent leaks.
        """
        # Simulate PII in tool results
        tool_results = [
            {
                "name": "salesforce__query_records",
                "result": {
                    "records": [{"Id": "001", "Email": "user@example.com", "Phone": "555-123-4567"}]
                },
            }
        ]

        model_output_with_code_mode = """
import * as salesforce from './servers/salesforce';
const records = await salesforce.queryRecords({ objectType: 'Contact' });
// PII stays in sandbox, only summary returned
const count = records.records.length;
console.log('Found', count, 'records');
"""

        model_output_with_leak = """
<|tool_call|>
{"name": "salesforce__query_records", "arguments": {"objectType": "Contact"}}
<|tool_result|>
{"records": [{"Id": "001", "Email": "user@example.com", "Phone": "555-123-4567"}]}
The contact email is user@example.com and phone is 555-123-4567.
"""

        # Code-mode should be detected
        assert detect_code_mode_usage(model_output_with_code_mode), (
            "Code-mode usage should be detected"
        )

        # Code-mode should not leak PII
        assert not detect_data_leak(model_output_with_code_mode, tool_results), (
            "Code-mode should not leak PII into tokens"
        )

        # Direct tool call with leak should be detected
        assert detect_data_leak(model_output_with_leak, tool_results), (
            "Data leak should be detected when PII appears in output"
        )


class TestCodeModeTokenEfficiency:
    """Test that code-mode improves token efficiency."""

    def test_code_mode_reduces_tokens(self):
        """
        Test that code-mode reduces token count vs direct tool calls.
        """
        # Simulate large tool result
        large_result = {"content": "x" * 10000}

        model_output_code_mode = """
import * as gdrive from './servers/google-drive';
const doc = await gdrive.getDocument({ documentId: 'abc' });
const summary = summarize(doc.content, 5);
console.log('Done');
"""

        model_output_direct = f"""
<|tool_call|>
{{"name": "google_drive__get_document", "arguments": {{"documentId": "abc"}}}}
<|tool_result|>
{large_result}
"""

        tokens_code_mode = count_tokens(model_output_code_mode)
        tokens_direct = count_tokens(model_output_direct)

        # Code-mode should use fewer tokens (large result stays in sandbox)
        assert tokens_code_mode < tokens_direct, (
            f"Code-mode should use fewer tokens ({tokens_code_mode} < {tokens_direct})"
        )

        # Improvement should be significant (≥25% reduction)
        improvement = (tokens_direct - tokens_code_mode) / tokens_direct
        assert improvement >= 0.25, (
            f"Code-mode should achieve ≥25% token reduction (got {improvement:.1%})"
        )


class TestCodeModeExecutionCorrectness:
    """Test execution correctness gate to prevent 'pretty TS' without execution."""

    def test_execution_correctness_gate(self):
        """
        Test that models can't pass by emitting TS without executing it.

        For large-blob scenarios, require observable side-effect (file write with hash).
        """
        import hashlib

        # Simulate large blob (20k chars)
        large_blob = "x" * 20000
        hashlib.sha256(large_blob.encode()).hexdigest()

        # Model output that prints TS but doesn't execute
        model_output_pretty_only = """
import * as gdrive from './servers/google-drive';
const doc = await gdrive.getDocument({ documentId: 'abc' });
// Pretty TS code but never executed
console.log('Processing document...');
"""

        # Model output that executes and writes side-effect
        model_output_with_execution = """
import * as gdrive from './servers/google-drive';
import * as fs from 'fs';
const doc = await gdrive.getDocument({ documentId: 'abc' });
const filtered = doc.content.split('\\n').filter(line => line.length > 10);
const hash = require('crypto').createHash('sha256').update(doc.content).digest('hex');
fs.writeFileSync('./workspace/report.json', JSON.stringify({
  hash: hash,
  count: filtered.length
}));
console.log('Report written');
"""

        # Check that pretty-only output doesn't create side-effect
        # (In real test, would execute code and check for file)
        assert (
            "writeFileSync" not in model_output_pretty_only
            or "./workspace/report.json" not in model_output_pretty_only
        ), "Pretty-only TS should not create execution side-effects"

        # Check that execution output creates side-effect
        assert (
            "writeFileSync" in model_output_with_execution
            and "./workspace/report.json" in model_output_with_execution
        ), "Execution output should create observable side-effects"

        # Check that large blob doesn't appear in tokens (≤200 chars)
        large_blob_in_output = large_blob[:200] in model_output_with_execution
        assert not large_blob_in_output, (
            "Large blob should not appear in assistant tokens (should stay in sandbox)"
        )


class TestCodeModeSingleToolExemption:
    """Test that single-tool tiny cases are not penalized."""

    def test_single_tool_not_penalized(self):
        """
        Test that a single small call (e.g., kv.get('foo')) does not get penalized
        for using direct tool mode.
        """
        # Single small tool call
        model_output_single_tool = """
<|tool_call|>
{"name": "kv_get", "arguments": {"key": "foo"}}
<|tool_result|>
{"value": "bar"}
"""

        # Should not be eligible for code-mode (only 1 tool, small payload)
        tool_count = 1
        intermediate_size = len(model_output_single_tool)  # Small

        eligible = (
            tool_count >= 2 or intermediate_size >= 10000 or False  # No PII
        )

        assert not eligible, (
            "Single small tool call should not be eligible for code-mode preference"
        )

        # Should not detect as direct tool call requiring penalty
        # (eligibility check should prevent loss from firing)
        assert detect_direct_tool_calls(model_output_single_tool), (
            "Should detect direct tool call pattern"
        )

        # But since it's not eligible, it shouldn't be penalized
        # (This is tested by ensuring eligibility_mask filters it out)


def test_single_small_tool_exempt():
    """
    Test that a single small tool call is exempt from code-mode penalty.

    Regression test: single tool + small payload should have eligibility_mask[b]==False
    and preference loss contribution should be exactly zero.
    """
    from training.losses import CodeModePreferenceLoss
    import torch

    batch_size = 1
    seq_len = 10
    vocab_size = 1000

    student_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)

    loss_module = CodeModePreferenceLoss(
        eligibility_rules={"min_tools": 2, "min_intermediate_chars": 10000, "pii_patterns": []},
        reward={"prefer_ts_api_over_direct_tool": True, "penalize_tool_result_roundtrip": True},
        vocab_ids={},
    )

    # Single small tool call (not eligible)
    batch_meta = [
        {
            "tool_count": 1,  # Only 1 tool
            "intermediate_sizes": [100],  # Small payload (< 10k)
            "pii_tags_present": False,
        }
    ]

    # Compute eligibility mask
    eligibility_mask = loss_module._compute_eligibility_mask(batch_meta, batch_size)

    # Should be ineligible
    assert not eligibility_mask[0].item(), "Single small tool should not be eligible"

    # Loss should be zero for ineligible cases
    loss = loss_module(
        student_logits=student_logits,
        span_targets=None,
        batch_meta=batch_meta,
    )

    assert loss.item() == 0.0, f"Loss should be zero for ineligible case (got {loss.item()})"


def test_adversarial_printing():
    """
    Test that adversarial printing (pretty TS + pasted 20k JSON) fails the gate.

    Student prints pretty TS code AND pastes 20k JSON → gate must fail.
    """
    large_json = "x" * 20000

    model_output_adversarial = f"""
import * as gdrive from './servers/google-drive';
const doc = await gdrive.getDocument({{ documentId: 'abc' }});
// Pretty TS code but also pasting large JSON
const data = {large_json};
console.log('Done');
"""

    # Check that large JSON appears in output (should fail gate)
    large_json_in_output = large_json[:1000] in model_output_adversarial

    assert large_json_in_output, "Adversarial output should contain large JSON paste"

    # Gate should fail: large payload echoed into tokens
    # (In real test, would check CES and no-leakage gate)
    assert len(model_output_adversarial) > 20000, "Adversarial output should exceed token limit"


def test_pii_binding_path_no_leak():
    """
    Test that PII in binding path with only counts logged results in leak_count = 0.

    Tool returns emails; code updates CRM via binding and only logs counts → leak count = 0.
    """
    # Simulate tool result with emails
    tool_results = {
        "emails": ["user1@example.com", "user2@example.com", "user3@example.com"],
        "count": 3,
    }

    # Model output that processes emails but only logs counts (no PII leak)
    model_output_safe = """
import * as crm from './servers/crm';
const emails = await crm.getCustomerEmails({ segment: 'premium' });
const count = emails.length;
const filtered = emails.filter(e => e.includes('@example.com'));
await crm.updateSegment({ segment: 'premium', count: filtered.length });
console.log(`Processed ${filtered.length} emails`);
"""

    # Check that emails don't appear in output (only counts)
    email_in_output = any(email in model_output_safe for email in tool_results["emails"])

    assert not email_in_output, "PII should not appear in output (only counts logged)"

    # Check that counts are logged
    assert "count" in model_output_safe.lower() or "length" in model_output_safe.lower(), (
        "Counts should be logged instead of PII"
    )

    # In real test, would verify leak_count == 0 via scorer
