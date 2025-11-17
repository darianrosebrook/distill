"""
Tests for training/caws_context.py - CAWS context extraction utilities.

Tests CAWS context extraction, formatting, and budget derivation.
"""
# @author: @darianrosebrook

import yaml
from training.caws_context import (
    CAWSContext,
    extract_caws_context,
    format_caws_context_for_prompt,
    extract_caws_context_dict,
    derive_budget,
)


class TestCAWSContext:
    """Test CAWSContext dataclass."""

    def test_caws_context_creation(self):
        """Test creating a CAWSContext."""
        context = CAWSContext(
            spec_id="FEAT-001",
            title="Test Feature",
            risk_tier=2,
            mode="feature",
            budget={"max_files": 25, "max_loc": 1000},
            scope={"in": ["src/"], "out": ["tests/"]},
            quality={"coverage_threshold": 80, "mutation_threshold": 50},
            acceptance_summary=["A: Given X → When Y → Then Z"],
            invariants=["Invariant 1", "Invariant 2"],
        )
        assert context.spec_id == "FEAT-001"
        assert context.title == "Test Feature"
        assert context.risk_tier == 2
        assert context.budget["max_files"] == 25


class TestExtractCAWSContext:
    """Test extract_caws_context function."""

    def test_extract_caws_context_no_caws_dir(self, tmp_path):
        """Test extracting context when .caws directory doesn't exist."""
        result = extract_caws_context(str(tmp_path))
        assert result is None

    def test_extract_caws_context_legacy_spec(self, tmp_path):
        """Test extracting context from legacy working-spec.yaml."""
        caws_dir = tmp_path / ".caws"
        caws_dir.mkdir()

        spec_file = caws_dir / "working-spec.yaml"
        spec = {
            "id": "LEGACY-001",
            "title": "Legacy Feature",
            "risk_tier": 2,
            "mode": "feature",
            "scope": {"in": ["src/"], "out": ["tests/"]},
            "acceptance": [],
            "invariants": [],
        }
        with open(spec_file, "w") as f:
            yaml.dump(spec, f)

        result = extract_caws_context(str(tmp_path))
        assert result is not None
        assert result.spec_id == "LEGACY-001"
        assert result.title == "Legacy Feature"

    def test_extract_caws_context_feature_spec(self, tmp_path):
        """Test extracting context from feature-specific spec."""
        caws_dir = tmp_path / ".caws"
        caws_dir.mkdir()
        specs_dir = caws_dir / "specs"
        specs_dir.mkdir()

        spec_file = specs_dir / "FEAT-001.yaml"
        spec = {
            "id": "FEAT-001",
            "title": "Test Feature",
            "risk_tier": 2,
            "mode": "feature",
            "scope": {"in": ["src/"], "out": ["tests/"]},
            "acceptance": [],
            "invariants": [],
        }
        with open(spec_file, "w") as f:
            yaml.dump(spec, f)

        result = extract_caws_context(str(tmp_path), spec_id="FEAT-001")
        assert result is not None
        assert result.spec_id == "FEAT-001"

    def test_extract_caws_context_with_policy(self, tmp_path):
        """Test extracting context with policy file."""
        caws_dir = tmp_path / ".caws"
        caws_dir.mkdir()

        spec_file = caws_dir / "working-spec.yaml"
        spec = {
            "id": "FEAT-001",
            "title": "Test Feature",
            "risk_tier": 2,
            "mode": "feature",
            "scope": {"in": ["src/"], "out": ["tests/"]},
            "acceptance": [],
            "invariants": [],
        }
        with open(spec_file, "w") as f:
            yaml.dump(spec, f)

        policy_file = caws_dir / "policy.yaml"
        policy = {
            "risk_tiers": {
                "2": {
                    "max_files": 30,
                    "max_loc": 1200,
                    "coverage_threshold": 85,
                    "mutation_threshold": 60,
                }
            }
        }
        with open(policy_file, "w") as f:
            yaml.dump(policy, f)

        result = extract_caws_context(str(tmp_path))
        assert result is not None
        assert result.budget["max_files"] == 30
        assert result.quality["coverage_threshold"] == 85

    def test_extract_caws_context_default_budget(self, tmp_path):
        """Test extracting context with default budget."""
        caws_dir = tmp_path / ".caws"
        caws_dir.mkdir()

        spec_file = caws_dir / "working-spec.yaml"
        spec = {
            "id": "FEAT-001",
            "title": "Test Feature",
            "risk_tier": 2,
            "mode": "feature",
            "scope": {"in": ["src/"], "out": ["tests/"]},
            "acceptance": [],
            "invariants": [],
        }
        with open(spec_file, "w") as f:
            yaml.dump(spec, f)

        result = extract_caws_context(str(tmp_path))
        assert result is not None
        # Should use default budget
        assert result.budget["max_files"] == 25
        assert result.budget["max_loc"] == 1000

    def test_extract_caws_context_with_acceptance_criteria(self, tmp_path):
        """Test extracting context with acceptance criteria."""
        caws_dir = tmp_path / ".caws"
        caws_dir.mkdir()

        spec_file = caws_dir / "working-spec.yaml"
        spec = {
            "id": "FEAT-001",
            "title": "Test Feature",
            "risk_tier": 2,
            "mode": "feature",
            "scope": {"in": ["src/"], "out": ["tests/"]},
            "acceptance": [
                {"id": "A1", "given": "X", "when": "Y", "then": "Z"},
                "Simple acceptance criterion",
            ],
            "invariants": [],
        }
        with open(spec_file, "w") as f:
            yaml.dump(spec, f)

        result = extract_caws_context(str(tmp_path))
        assert result is not None
        assert len(result.acceptance_summary) == 2

    def test_extract_caws_context_with_invariants(self, tmp_path):
        """Test extracting context with invariants."""
        caws_dir = tmp_path / ".caws"
        caws_dir.mkdir()

        spec_file = caws_dir / "working-spec.yaml"
        spec = {
            "id": "FEAT-001",
            "title": "Test Feature",
            "risk_tier": 2,
            "mode": "feature",
            "scope": {"in": ["src/"], "out": ["tests/"]},
            "acceptance": [],
            "invariants": ["Invariant 1", "Invariant 2"],
        }
        with open(spec_file, "w") as f:
            yaml.dump(spec, f)

        result = extract_caws_context(str(tmp_path))
        assert result is not None
        assert len(result.invariants) == 2

    def test_extract_caws_context_string_invariants(self, tmp_path):
        """Test extracting context with string invariant (not list)."""
        caws_dir = tmp_path / ".caws"
        caws_dir.mkdir()

        spec_file = caws_dir / "working-spec.yaml"
        spec = {
            "id": "FEAT-001",
            "title": "Test Feature",
            "risk_tier": 2,
            "mode": "feature",
            "scope": {"in": ["src/"], "out": ["tests/"]},
            "acceptance": [],
            "invariants": "Single invariant string",
        }
        with open(spec_file, "w") as f:
            yaml.dump(spec, f)

        result = extract_caws_context(str(tmp_path))
        assert result is not None
        assert isinstance(result.invariants, list)
        assert len(result.invariants) == 1


class TestFormatCAWSContextForPrompt:
    """Test format_caws_context_for_prompt function."""

    def test_format_caws_context_for_prompt_none(self):
        """Test formatting None context."""
        result = format_caws_context_for_prompt(None)
        assert result == ""

    def test_format_caws_context_for_prompt_verbose(self, tmp_path):
        """Test formatting context in verbose mode."""
        caws_dir = tmp_path / ".caws"
        caws_dir.mkdir()

        spec_file = caws_dir / "working-spec.yaml"
        spec = {
            "id": "FEAT-001",
            "title": "Test Feature",
            "risk_tier": 2,
            "mode": "feature",
            "scope": {"in": ["src/"], "out": ["tests/"]},
            "acceptance": [],
            "invariants": [],
        }
        with open(spec_file, "w") as f:
            yaml.dump(spec, f)

        context = extract_caws_context(str(tmp_path))
        result = format_caws_context_for_prompt(context, compact=False)

        assert "CAWS Context" in result
        assert "FEAT-001" in result
        assert "Test Feature" in result
        assert "Risk Tier" in result
        assert "Budget Constraints" in result

    def test_format_caws_context_for_prompt_compact(self, tmp_path):
        """Test formatting context in compact mode."""
        caws_dir = tmp_path / ".caws"
        caws_dir.mkdir()

        spec_file = caws_dir / "working-spec.yaml"
        spec = {
            "id": "FEAT-001",
            "title": "Test Feature",
            "risk_tier": 2,
            "mode": "feature",
            "scope": {"in": ["src/"], "out": ["tests/"]},
            "acceptance": [],
            "invariants": [],
        }
        with open(spec_file, "w") as f:
            yaml.dump(spec, f)

        context = extract_caws_context(str(tmp_path))
        result = format_caws_context_for_prompt(context, compact=True)

        # Compact format should be shorter
        assert isinstance(result, str)

    def test_format_caws_context_for_prompt_with_acceptance(self, tmp_path):
        """Test formatting context with acceptance criteria."""
        caws_dir = tmp_path / ".caws"
        caws_dir.mkdir()

        spec_file = caws_dir / "working-spec.yaml"
        spec = {
            "id": "FEAT-001",
            "title": "Test Feature",
            "risk_tier": 2,
            "mode": "feature",
            "scope": {"in": ["src/"], "out": ["tests/"]},
            "acceptance": [
                {"id": "A1", "given": "X", "when": "Y", "then": "Z"},
            ],
            "invariants": [],
        }
        with open(spec_file, "w") as f:
            yaml.dump(spec, f)

        context = extract_caws_context(str(tmp_path))
        result = format_caws_context_for_prompt(context, compact=False)

        assert "Acceptance Criteria" in result

    def test_format_caws_context_for_prompt_with_invariants(self, tmp_path):
        """Test formatting context with invariants."""
        caws_dir = tmp_path / ".caws"
        caws_dir.mkdir()

        spec_file = caws_dir / "working-spec.yaml"
        spec = {
            "id": "FEAT-001",
            "title": "Test Feature",
            "risk_tier": 2,
            "mode": "feature",
            "scope": {"in": ["src/"], "out": ["tests/"]},
            "acceptance": [],
            "invariants": ["Invariant 1", "Invariant 2"],
        }
        with open(spec_file, "w") as f:
            yaml.dump(spec, f)

        context = extract_caws_context(str(tmp_path))
        result = format_caws_context_for_prompt(context, compact=False)

        assert "Invariants" in result

    def test_format_caws_context_for_prompt_many_scope_items(self, tmp_path):
        """Test formatting context with more than 5 scope items."""
        caws_dir = tmp_path / ".caws"
        caws_dir.mkdir()

        spec_file = caws_dir / "working-spec.yaml"
        spec = {
            "id": "FEAT-001",
            "title": "Test Feature",
            "risk_tier": 2,
            "mode": "feature",
            "scope": {
                "in": ["src/auth/", "src/billing/", "src/api/", "src/utils/", "src/models/", "src/config/", "tests/unit/"],
                "out": ["node_modules/", "dist/", "build/", "logs/", "temp/"]
            },
            "acceptance": [],
            "invariants": [],
        }
        with open(spec_file, "w") as f:
            yaml.dump(spec, f)

        context = extract_caws_context(str(tmp_path))
        result = format_caws_context_for_prompt(context, compact=False)

        # scope.in has 7 items, shows 5 + "and 2 more"
        assert "(and 2 more)" in result
        assert "Out of Scope" in result

    def test_format_caws_context_for_prompt_many_acceptance_criteria(self, tmp_path):
        """Test formatting context with more than 5 acceptance criteria."""
        caws_dir = tmp_path / ".caws"
        caws_dir.mkdir()

        spec_file = caws_dir / "working-spec.yaml"
        spec = {
            "id": "FEAT-001",
            "title": "Test Feature",
            "risk_tier": 2,
            "mode": "feature",
            "scope": {"in": ["src/"], "out": []},
            "acceptance": [
                {"id": "A1", "given": "X1", "when": "Y1", "then": "Z1"},
                {"id": "A2", "given": "X2", "when": "Y2", "then": "Z2"},
                {"id": "A3", "given": "X3", "when": "Y3", "then": "Z3"},
                {"id": "A4", "given": "X4", "when": "Y4", "then": "Z4"},
                {"id": "A5", "given": "X5", "when": "Y5", "then": "Z5"},
                {"id": "A6", "given": "X6", "when": "Y6", "then": "Z6"},
                {"id": "A7", "given": "X7", "when": "Y7", "then": "Z7"},
            ],
            "invariants": [],
        }
        with open(spec_file, "w") as f:
            yaml.dump(spec, f)

        context = extract_caws_context(str(tmp_path))
        result = format_caws_context_for_prompt(context, compact=False)

        # acceptance has 7 items, shows 5 + "and 2 more"
        assert "(and 2 more)" in result
        assert "Acceptance Criteria" in result

    def test_format_caws_context_for_prompt_many_invariants(self, tmp_path):
        """Test formatting context with more than 5 invariants."""
        caws_dir = tmp_path / ".caws"
        caws_dir.mkdir()

        spec_file = caws_dir / "working-spec.yaml"
        spec = {
            "id": "FEAT-001",
            "title": "Test Feature",
            "risk_tier": 2,
            "mode": "feature",
            "scope": {"in": ["src/"], "out": []},
            "acceptance": [],
            "invariants": [
                "Invariant 1: System must remain stable",
                "Invariant 2: Performance requirements met",
                "Invariant 3: Security constraints satisfied",
                "Invariant 4: Compatibility maintained",
                "Invariant 5: Documentation updated",
                "Invariant 6: Testing completed",
                "Invariant 7: Code review passed",
            ],
        }
        with open(spec_file, "w") as f:
            yaml.dump(spec, f)

        context = extract_caws_context(str(tmp_path))
        result = format_caws_context_for_prompt(context, compact=False)

        # invariants has 7 items, shows 5 + "and 2 more"
        assert "(and 2 more)" in result
        assert "Invariants" in result


class TestExtractCAWSContextDict:
    """Test extract_caws_context_dict function."""

    def test_extract_caws_context_dict_no_context(self, tmp_path):
        """Test extracting context dict when no context exists."""
        result = extract_caws_context_dict(str(tmp_path))
        assert result is None

    def test_extract_caws_context_dict_valid(self, tmp_path):
        """Test extracting context as dictionary."""
        caws_dir = tmp_path / ".caws"
        caws_dir.mkdir()

        spec_file = caws_dir / "working-spec.yaml"
        spec = {
            "id": "FEAT-001",
            "title": "Test Feature",
            "risk_tier": 2,
            "mode": "feature",
            "scope": {"in": ["src/"], "out": ["tests/"]},
            "acceptance": [],
            "invariants": [],
        }
        with open(spec_file, "w") as f:
            yaml.dump(spec, f)

        result = extract_caws_context_dict(str(tmp_path))
        assert result is not None
        assert isinstance(result, dict)
        assert result["spec_id"] == "FEAT-001"
        assert result["title"] == "Test Feature"
        assert "budget" in result
        assert "scope" in result
        assert "quality" in result


class TestDeriveBudget:
    """Test derive_budget function."""

    def test_derive_budget_baseline(self, tmp_path):
        """Test deriving baseline budget."""
        spec = {
            "id": "FEAT-001",
            "risk_tier": 2,
        }

        result = derive_budget(spec, str(tmp_path))
        assert "baseline" in result
        assert "effective" in result
        assert "waivers_applied" in result
        assert "derived_at" in result
        assert result["baseline"]["max_files"] == 25
        assert result["baseline"]["max_loc"] == 1000

    def test_derive_budget_with_policy(self, tmp_path):
        """Test deriving budget with policy file."""
        caws_dir = tmp_path / ".caws"
        caws_dir.mkdir()

        policy_file = caws_dir / "policy.yaml"
        policy = {
            "risk_tiers": {
                "2": {
                    "max_files": 30,
                    "max_loc": 1200,
                }
            }
        }
        with open(policy_file, "w") as f:
            yaml.dump(policy, f)

        spec = {
            "id": "FEAT-001",
            "risk_tier": 2,
        }

        result = derive_budget(spec, str(tmp_path))
        assert result["baseline"]["max_files"] == 30
        assert result["baseline"]["max_loc"] == 1200

    def test_derive_budget_with_waivers(self, tmp_path):
        """Test deriving budget with waivers."""
        caws_dir = tmp_path / ".caws"
        caws_dir.mkdir()
        waivers_dir = caws_dir / "waivers"
        waivers_dir.mkdir()

        waiver_file = waivers_dir / "WAIVER-001.yaml"
        waiver = {
            "status": "active",
            "delta": {
                "max_files": 10,
                "max_loc": 500,
            },
        }
        with open(waiver_file, "w") as f:
            yaml.dump(waiver, f)

        spec = {
            "id": "FEAT-001",
            "risk_tier": 2,
            "waiver_ids": ["WAIVER-001"],
        }

        result = derive_budget(spec, str(tmp_path))
        assert "WAIVER-001" in result["waivers_applied"]
        assert result["effective"]["max_files"] == result["baseline"]["max_files"] + 10
        assert result["effective"]["max_loc"] == result["baseline"]["max_loc"] + 500

    def test_derive_budget_inactive_waiver(self, tmp_path):
        """Test deriving budget with inactive waiver."""
        caws_dir = tmp_path / ".caws"
        caws_dir.mkdir()
        waivers_dir = caws_dir / "waivers"
        waivers_dir.mkdir()

        waiver_file = waivers_dir / "WAIVER-001.yaml"
        waiver = {
            "status": "inactive",
            "delta": {
                "max_files": 10,
                "max_loc": 500,
            },
        }
        with open(waiver_file, "w") as f:
            yaml.dump(waiver, f)

        spec = {
            "id": "FEAT-001",
            "risk_tier": 2,
            "waiver_ids": ["WAIVER-001"],
        }

        result = derive_budget(spec, str(tmp_path))
        assert "WAIVER-001" not in result["waivers_applied"]
        assert result["effective"] == result["baseline"]

    def test_derive_budget_multiple_waivers(self, tmp_path):
        """Test deriving budget with multiple waivers."""
        caws_dir = tmp_path / ".caws"
        caws_dir.mkdir()
        waivers_dir = caws_dir / "waivers"
        waivers_dir.mkdir()

        # Create two active waivers
        for i, waiver_id in enumerate(["WAIVER-001", "WAIVER-002"]):
            waiver_file = waivers_dir / f"{waiver_id}.yaml"
            waiver = {
                "status": "active",
                "delta": {
                    "max_files": 5 + i,
                    "max_loc": 200 + i * 100,
                },
            }
            with open(waiver_file, "w") as f:
                yaml.dump(waiver, f)

        spec = {
            "id": "FEAT-001",
            "risk_tier": 2,
            "waiver_ids": ["WAIVER-001", "WAIVER-002"],
        }

        result = derive_budget(spec, str(tmp_path))
        assert len(result["waivers_applied"]) == 2
        assert result["effective"]["max_files"] > result["baseline"]["max_files"]
        assert result["effective"]["max_loc"] > result["baseline"]["max_loc"]

    def test_derive_budget_string_waiver_ids(self, tmp_path):
        """Test deriving budget with string waiver_ids (not list)."""
        caws_dir = tmp_path / ".caws"
        caws_dir.mkdir()
        waivers_dir = caws_dir / "waivers"
        waivers_dir.mkdir()

        waiver_file = waivers_dir / "WAIVER-001.yaml"
        waiver = {
            "status": "active",
            "delta": {
                "max_files": 10,
                "max_loc": 500,
            },
        }
        with open(waiver_file, "w") as f:
            yaml.dump(waiver, f)

        spec = {
            "id": "FEAT-001",
            "risk_tier": 2,
            "waiver_ids": "WAIVER-001",  # String instead of list
        }

        result = derive_budget(spec, str(tmp_path))
        assert "WAIVER-001" in result["waivers_applied"]


class TestCAWSContextIntegration:
    """Test integration of CAWS context components."""

    def test_complete_caws_context_workflow(self, tmp_path):
        """Test complete CAWS context extraction and formatting workflow."""
        caws_dir = tmp_path / ".caws"
        caws_dir.mkdir()

        # Create spec
        spec_file = caws_dir / "working-spec.yaml"
        spec = {
            "id": "FEAT-001",
            "title": "Test Feature",
            "risk_tier": 2,
            "mode": "feature",
            "scope": {"in": ["src/"], "out": ["tests/"]},
            "acceptance": [
                {"id": "A1", "given": "X", "when": "Y", "then": "Z"},
            ],
            "invariants": ["Invariant 1"],
        }
        with open(spec_file, "w") as f:
            yaml.dump(spec, f)

        # Create policy
        policy_file = caws_dir / "policy.yaml"
        policy = {
            "risk_tiers": {
                "2": {
                    "max_files": 30,
                    "max_loc": 1200,
                    "coverage_threshold": 85,
                    "mutation_threshold": 60,
                }
            }
        }
        with open(policy_file, "w") as f:
            yaml.dump(policy, f)

        # Extract context
        context = extract_caws_context(str(tmp_path))
        assert context is not None

        # Format for prompt
        formatted = format_caws_context_for_prompt(context, compact=False)
        assert "CAWS Context" in formatted

        # Extract as dict
        context_dict = extract_caws_context_dict(str(tmp_path))
        assert context_dict is not None

        # Derive budget
        budget = derive_budget(spec, str(tmp_path))
        assert budget["baseline"]["max_files"] == 30
