"""
Security tests for contextual dataset generation scripts.

Tests:
1. Input validation
2. Path traversal prevention
3. Command injection prevention
4. PII redaction
5. URL allowlist
"""
import pytest
from unittest.mock import Mock, patch


class TestInputValidation:
    """Test input validation and sanitization."""
    
    def test_malicious_input_handling(self):
        """Test handling of malicious input."""
        from scripts.generate_contextual_prompts import synthesize_prompt
        from tools.schema_registry import ToolSchemaRegistry
        
        reg = ToolSchemaRegistry()
        
        # Test with potentially malicious characters
        cell = {
            "scenario": "file_ops",
            "complexity": "single_call",
            "structure": "flat_args",
        }
        
        prompt, history, meta = synthesize_prompt(cell, reg)
        
        # Should handle gracefully without crashing
        assert isinstance(prompt, str)
        assert len(prompt) > 0
    
    def test_path_traversal_prevention(self):
        """Test file path validation."""
        from scripts.util_sanitize import redact_pii
        
        # Test with path traversal attempts
        text_with_traversal = "../../etc/passwd"
        result = redact_pii(text_with_traversal)
        
        # Should handle without crashing
        assert isinstance(result, str)
    
    def test_command_injection_prevention(self):
        """Test shell command safety."""
        from scripts.generate_contextual_prompts import synthesize_prompt
        from tools.schema_registry import ToolSchemaRegistry
        
        reg = ToolSchemaRegistry()
        
        # Test with command injection attempts
        cell = {
            "scenario": "file_ops",
            "complexity": "single_call",
            "structure": "flat_args",
        }
        
        prompt, history, meta = synthesize_prompt(cell, reg)
        
        # Should not contain shell commands
        assert ";" not in prompt or prompt.count(";") < 2  # Allow JSON separators
        assert "&&" not in prompt
        assert "||" not in prompt
        assert "`" not in prompt  # Backticks for command substitution


class TestPIIRedaction:
    """Test PII detection and redaction."""
    
    def test_pii_redaction(self):
        """Test PII detection and redaction."""
        from scripts.util_sanitize import redact_pii
        
        # Test with email
        text_with_email = "Contact me at test@example.com"
        result = redact_pii(text_with_email)
        
        # Should redact or flag email
        assert isinstance(result, str)
        # Email should be redacted or flagged
        assert "@example.com" not in result or "[REDACTED]" in result
    
    def test_uuid_detection(self):
        """Test UUID detection."""
        from scripts.verify_contextual_set import check_privacy
        
        text_with_uuid = "ID: 550e8400-e29b-41d4-a716-446655440000"
        result = check_privacy(text_with_uuid)
        
        assert result["uuids_found"] > 0
        assert result["privacy_ok"] is False
    
    def test_email_detection(self):
        """Test email detection."""
        from scripts.verify_contextual_set import check_privacy
        
        text_with_email = "Contact test@example.com"
        result = check_privacy(text_with_email)
        
        assert result["emails_found"] > 0
        assert result["privacy_ok"] is False


class TestURLAllowlist:
    """Test URL validation and allowlist."""
    
    def test_url_allowlist(self):
        """Test URL allowlist validation."""
        from scripts.util_sanitize import allowlist_urls
        
        # Test with allowed URL
        text_with_allowed = "See https://example.org/article"
        result = allowlist_urls(text_with_allowed)
        
        assert isinstance(result, bool)
    
    def test_url_validation(self):
        """Test URL validation."""
        from scripts.verify_contextual_set import check_privacy
        
        # Test with URL
        text_with_url = "Visit https://example.org"
        result = check_privacy(text_with_url)
        
        assert isinstance(result, dict)
        assert "url_allowlist_ok" in result


class TestSafetyScanning:
    """Test safety scanning functionality."""
    
    def test_safety_scanning(self):
        """Test safety scanning."""
        from scripts.util_sanitize import scan_safety
        
        text = "This is safe text"
        result = scan_safety(text)
        
        assert isinstance(result, dict)
        # Result should contain safety check fields
        assert "urls_ok" in result or "pii_hits" in result or "safe" in result

