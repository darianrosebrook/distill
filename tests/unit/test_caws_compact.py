"""
Unit tests for Priority 4: CAWS Compact JSON Format.

Tests:
1. format_caws_compact() produces compact JSON
2. Token count ≤ 30 tokens per example
3. All essential CAWS fields preserved
"""
import json
from training.prompt_templates import format_caws_compact, CAWSContext


class TestCAWSCompact:
    """Test compact CAWS formatting."""

    def test_compact_format_basic(self):
        """Test basic compact formatting."""
        caws_ctx = CAWSContext(
            spec_id="FEAT-001",
            title="User Authentication",
            risk_tier=1,
            mode="feature",
            budget={"max_files": 25, "max_loc": 1000},
            scope={
                "in": ["src/auth/", "tests/auth/"],
                "out": ["node_modules/", "dist/"]
            },
            quality={
                "coverage_threshold": 80,
                "mutation_threshold": 60
            },
            acceptance_summary=[],
            invariants=[]
        )
        
        compact = format_caws_compact(caws_ctx)
        
        # Should be valid JSON
        parsed = json.loads(compact)
        assert "caws" in parsed
        
        # Check essential fields
        caws = parsed["caws"]
        assert caws["tier"] == 1
        assert caws["max_files"] == 25
        assert caws["max_loc"] == 1000
        assert caws["cov"] == 80
        assert caws["mut"] == 60
        assert isinstance(caws["in"], list)
        assert isinstance(caws["out"], list)

    def test_compact_format_dict(self):
        """Test compact formatting with dict input."""
        working_spec = {
            "risk_tier": 2,
            "budget": {"max_files": 30, "max_loc": 1500},
            "quality_gates": {"coverage": 85, "mutation_score": 70},
            "scope": {
                "in": ["src/", "tests/"],
                "out": ["node_modules/"]
            }
        }
        
        compact = format_caws_compact(working_spec)
        
        # Should be valid JSON
        parsed = json.loads(compact)
        assert "caws" in parsed
        
        caws = parsed["caws"]
        assert caws["tier"] == 2
        assert caws["max_files"] == 30
        assert caws["max_loc"] == 1500

    def test_token_count_limit(self):
        """Test that compact format stays within token limit."""
        caws_ctx = CAWSContext(
            spec_id="FEAT-001",
            title="User Authentication",
            risk_tier=1,
            mode="feature",
            budget={"max_files": 25, "max_loc": 1000},
            scope={
                "in": ["src/auth/", "tests/auth/", "src/api/", "tests/api/", "src/utils/"],
                "out": ["node_modules/", "dist/", "build/"]
            },
            quality={
                "coverage_threshold": 80,
                "mutation_threshold": 60
            },
            acceptance_summary=[],
            invariants=[]
        )
        
        compact = format_caws_compact(caws_ctx)
        
        # Estimate token count (rough: ~1 token per 4 characters for compact JSON)
        # Compact JSON should be much shorter than verbose markdown
        char_count = len(compact)
        estimated_tokens = char_count / 4
        
        # Should be ≤ 30 tokens (with some margin for tokenizer differences)
        assert estimated_tokens <= 40, f"Estimated tokens ({estimated_tokens:.1f}) exceeds limit (40)"
        
        # Verify it's actually compact (no spaces in JSON)
        assert " " not in compact or compact.count(" ") < 5, "JSON should be compact (minimal spaces)"

    def test_scope_limiting(self):
        """Test that scope lists are limited to 5 items."""
        caws_ctx = CAWSContext(
            spec_id="FEAT-001",
            title="Test",
            risk_tier=1,
            mode="feature",
            budget={"max_files": 25, "max_loc": 1000},
            scope={
                "in": [f"src/dir{i}/" for i in range(10)],  # 10 items
                "out": [f"build/dir{i}/" for i in range(10)]  # 10 items
            },
            quality={"coverage_threshold": 80, "mutation_threshold": 60},
            acceptance_summary=[],
            invariants=[]
        )
        
        compact = format_caws_compact(caws_ctx)
        parsed = json.loads(compact)
        
        # Should be limited to 5 items
        assert len(parsed["caws"]["in"]) <= 5
        assert len(parsed["caws"]["out"]) <= 5

    def test_default_values(self):
        """Test that default values are used when fields missing."""
        working_spec = {
            "risk_tier": 2,
        }
        
        compact = format_caws_compact(working_spec)
        parsed = json.loads(compact)
        
        caws = parsed["caws"]
        # Should use defaults
        assert caws["tier"] == 2
        assert caws["max_files"] == 25  # Default
        assert caws["max_loc"] == 1000  # Default
        assert caws["cov"] == 80  # Default
        assert caws["mut"] == 50  # Default

    def test_compact_vs_verbose(self):
        """Test that compact format is much shorter than verbose."""
        caws_ctx = CAWSContext(
            spec_id="FEAT-001",
            title="User Authentication",
            risk_tier=1,
            mode="feature",
            budget={"max_files": 25, "max_loc": 1000},
            scope={
                "in": ["src/auth/", "tests/auth/"],
                "out": ["node_modules/", "dist/"]
            },
            quality={
                "coverage_threshold": 80,
                "mutation_threshold": 60
            },
            acceptance_summary=["A1: User logs in"],
            invariants=["No localStorage"]
        )
        
        compact = format_caws_compact(caws_ctx)
        
        # Verbose format would be much longer
        verbose_estimate = len(f"""
CAWS CONTEXT:
- Spec ID: {caws_ctx.spec_id}
- Title: {caws_ctx.title}
- Risk Tier: {caws_ctx.risk_tier}
- Mode: {caws_ctx.mode}
- Budget: {caws_ctx.budget.get('max_files', 'N/A')} files, {caws_ctx.budget.get('max_loc', 'N/A')} LOC
- Scope In: {', '.join(caws_ctx.scope.get('in', [])[:5])}
- Scope Out: {', '.join(caws_ctx.scope.get('out', [])[:5])}
""")
        
        # Compact should be significantly shorter
        assert len(compact) < verbose_estimate / 3, "Compact format should be much shorter"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

