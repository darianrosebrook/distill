"""
Unit tests for training/prompt_templates.py

Tests prompt template generation for worker, judge, drafter, and planning models.
"""

import pytest
import json
from training.prompt_templates import (
    CAWSContext,
    WorkerPromptTemplate,
    JudgePromptTemplate,
    DrafterPromptTemplate,
    PlanningPromptTemplate,
)


class TestCAWSContext:
    """Test CAWS context dataclass."""

    def test_caws_context_creation(self):
        """Test basic CAWS context creation."""
        context = CAWSContext(
            spec_id="TEST-001",
            title="Test Specification",
            risk_tier=2,
            mode="development",
            budget={"tokens": 1000, "compute": 500},
            scope={"in": ["src/"], "out": ["tests/"]},
            quality={"coverage": 80, "complexity": 5},
            acceptance_summary=["Must pass all tests", "Must be documented"],
            invariants=["No external dependencies", "Follow SOLID principles"],
        )

        assert context.spec_id == "TEST-001"
        assert context.title == "Test Specification"
        assert context.risk_tier == 2
        assert context.mode == "development"
        assert context.budget == {"tokens": 1000, "compute": 500}
        assert context.scope == {"in": ["src/"], "out": ["tests/"]}
        assert context.quality == {"coverage": 80, "complexity": 5}
        assert context.acceptance_summary == ["Must pass all tests", "Must be documented"]
        assert context.invariants == ["No external dependencies", "Follow SOLID principles"]

    def test_caws_context_optional_fields(self):
        """Test CAWS context with minimal required fields."""
        context = CAWSContext(
            spec_id="MINIMAL",
            title="Minimal Spec",
            risk_tier=1,
            mode="test",
            budget={},
            scope={},
            quality={},
            acceptance_summary=[],
            invariants=[],
        )

        assert context.spec_id == "MINIMAL"
        assert context.title == "Minimal Spec"
        assert context.budget == {}
        assert context.acceptance_summary == []
        assert context.invariants == []


class TestWorkerPromptTemplate:
    """Test worker prompt template generation."""

    def test_autonomous_coding_agent_basic(self):
        """Test basic autonomous coding agent template."""
        result = WorkerPromptTemplate.autonomous_coding_agent(
            task_id="TASK-001",
            description="Create a function to calculate fibonacci numbers",
        )

        assert isinstance(result, str)
        assert "TASK-001" in result
        assert "Create a function to calculate fibonacci numbers" in result
        assert "You are an autonomous coding agent" in result

    def test_autonomous_coding_agent_with_caws_context(self):
        """Test autonomous coding agent with CAWS context."""
        caws_context = CAWSContext(
            spec_id="SPEC-001",
            title="Fibonacci Calculator",
            risk_tier=2,
            mode="development",
            budget={"max_files": 25, "max_loc": 1000},
            scope={"in": ["src/"], "out": []},
            quality={"coverage": 80},
            acceptance_summary=["Must handle edge cases"],
            invariants=["Pure functions only"],
        )

        result = WorkerPromptTemplate.autonomous_coding_agent(
            task_id="TASK-001",
            description="Create fibonacci function",
            caws_context=caws_context,
        )

        # Should contain task info and be a valid prompt
        assert isinstance(result, str)
        assert "TASK-001" in result
        assert "Create fibonacci function" in result
        assert "You are an autonomous coding agent" in result
        assert len(result) > 100  # Should be a substantial prompt

    def test_autonomous_coding_agent_with_tools(self):
        """Test autonomous coding agent with tool registry."""
        tools = [
            {"name": "calculator", "description": "Basic calculator", "parameters": {"type": "object"}},
            {"name": "search", "description": "Web search", "parameters": {"type": "string"}},
        ]

        result = WorkerPromptTemplate.autonomous_coding_agent(
            task_id="TASK-001",
            description="Solve math problem",
            tool_registry=tools,
        )

        assert "calculator" in result
        assert "Basic calculator" in result
        assert "search" in result
        assert "Web search" in result

    def test_autonomous_coding_agent_compact_caws(self):
        """Test autonomous coding agent with compact CAWS format."""
        caws_context = CAWSContext(
            spec_id="SPEC-001",
            title="Test Spec",
            risk_tier=1,
            mode="test",
            budget={"max_files": 25, "max_loc": 1000},
            scope={"in": ["."], "out": []},
            quality={"coverage": 80},
            acceptance_summary=["Must work"],
            invariants=["No bugs"],
        )

        result = WorkerPromptTemplate.autonomous_coding_agent(
            task_id="TASK-001",
            description="Test task",
            caws_context=caws_context,
            use_compact_caws=True,
        )

        # Should use compact JSON format
        assert "CAWS:" in result
        assert "TASK-001" in result
        assert "Test task" in result

    def test_autonomous_coding_agent_full_caws(self):
        """Test autonomous coding agent with full CAWS format."""
        caws_context = CAWSContext(
            spec_id="SPEC-001",
            title="Test Spec",
            risk_tier=1,
            mode="test",
            budget={"max_files": 25, "max_loc": 1000},
            scope={"in": ["."], "out": []},
            quality={"coverage": 80},
            acceptance_summary=["Must work"],
            invariants=["No bugs"],
        )

        result = WorkerPromptTemplate.autonomous_coding_agent(
            task_id="TASK-001",
            description="Test task",
            caws_context=caws_context,
            use_compact_caws=False,
        )

        # Should use full format
        assert "CAWS CONTEXT:" in result
        assert "Spec ID: SPEC-001" in result
        assert "Title: Test Spec" in result
        assert "ACCEPTANCE CRITERIA:" in result
        assert "Must work" in result
        assert "INVARIANTS:" in result
        assert "No bugs" in result

    def test_autonomous_coding_agent_with_acceptance_criteria(self):
        """Test autonomous coding agent with acceptance criteria."""
        acceptance_criteria = [
            "Function must handle negative inputs",
            "Function must be efficient for large n",
            "Function must return integer values",
        ]

        result = WorkerPromptTemplate.autonomous_coding_agent(
            task_id="TASK-001",
            description="Create fibonacci function",
            acceptance_criteria=acceptance_criteria,
            use_compact_caws=False,  # Need full format
        )

        # Should generate a valid prompt with acceptance criteria
        assert isinstance(result, str)
        assert "TASK-001" in result
        assert "Create fibonacci function" in result
        assert len(result) > 200  # Should be more detailed in full format

    def test_autonomous_coding_agent_with_context_metrics(self):
        """Test autonomous coding agent with context metrics."""
        context_metrics = {
            "task_complexity": 0.8,
            "cpu_available": 60.0,
            "memory_available": 70.0,
        }

        result = WorkerPromptTemplate.autonomous_coding_agent(
            task_id="TASK-001",
            description="Test task",
            context_metrics=context_metrics,
        )

        assert "CONTEXT METRICS:" in result
        assert "Task Complexity: 0.80" in result
        assert "Available CPU: 60.0%" in result
        assert "Available Memory: 70.0%" in result


class TestJudgePromptTemplate:
    """Test judge prompt template generation."""

    def test_caws_debate_scoring_basic(self):
        """Test basic CAWS debate scoring template."""
        worker_outputs = [
            {
                "worker_id": "worker_1",
                "content": "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
                "evidence_completeness": 0.9,
                "budget_adherence": 0.8,
                "gate_integrity": 0.7,
                "provenance_clarity": 0.6,
            },
            {
                "worker_id": "worker_2",
                "content": "def fib(n): a, b = 0, 1; [a, b] = [b, a + b] for _ in range(n); return a",
                "evidence_completeness": 0.7,
                "budget_adherence": 0.9,
                "gate_integrity": 0.8,
                "provenance_clarity": 0.7,
            }
        ]

        working_spec = {
            "title": "Fibonacci Implementation",
            "description": "Implement efficient fibonacci function",
        }

        result = JudgePromptTemplate.caws_debate_scoring(
            worker_outputs=worker_outputs,
            working_spec=working_spec,
        )

        assert isinstance(result, str)
        assert "SOLUTION 1" in result
        assert "SOLUTION 2" in result
        assert "worker_1" in result
        assert "worker_2" in result
        assert "Fibonacci Implementation" in result

    def test_caws_debate_scoring_with_claim_results(self):
        """Test CAWS debate scoring with claim verification results."""
        worker_outputs = [{"worker_id": "worker_1", "content": "test"}]
        working_spec = {"title": "Test Spec"}

        claim_results = {
            "total_claims": 10,
            "verified_claims": 8,
            "verification_confidence": 0.85,
            "evidence_count": 15,
        }

        result = JudgePromptTemplate.caws_debate_scoring(
            worker_outputs=worker_outputs,
            working_spec=working_spec,
            claim_results=claim_results,
        )

        assert "CLAIM VERIFICATION RESULTS" in result
        assert "Total Claims: 10" in result
        assert "Verified Claims: 8" in result
        assert "Verification Confidence: 0.85" in result
        assert "Evidence Count: 15" in result

    def test_caws_debate_scoring_with_quality_gate(self):
        """Test CAWS debate scoring with quality gate results."""
        worker_outputs = [{"worker_id": "worker_1", "content": "test"}]
        working_spec = {"title": "Test Spec"}

        quality_gate_result = {
            "total_violations": 5,
            "waived_violations": 2,
            "blocking_violations": 1,
        }

        result = JudgePromptTemplate.caws_debate_scoring(
            worker_outputs=worker_outputs,
            working_spec=working_spec,
            quality_gate_result=quality_gate_result,
        )

        assert "QUALITY GATE RESULTS" in result
        assert "Total Violations: 5" in result
        assert "Waived Violations: 2" in result
        assert "Blocking Violations: 1" in result

    def test_caws_debate_scoring_complexity_mode(self):
        """Test CAWS debate scoring with complexity mode."""
        worker_outputs = [{"worker_id": "worker_1", "content": "test"}]
        working_spec = {"title": "Test Spec"}

        result = JudgePromptTemplate.caws_debate_scoring(
            worker_outputs=worker_outputs,
            working_spec=working_spec,
            complexity_mode="Enterprise",
        )

        assert "COMPLEXITY MODE: Enterprise" in result
        assert "SCORING WEIGHTS:" in result
        assert "Evidence Completeness: 50.0%" in result

    def test_caws_debate_scoring_empty_outputs(self):
        """Test CAWS debate scoring with empty worker outputs."""
        worker_outputs = []
        working_spec = {"title": "Test Spec"}

        result = JudgePromptTemplate.caws_debate_scoring(
            worker_outputs=worker_outputs,
            working_spec=working_spec,
        )

        # Should handle empty outputs gracefully
        assert isinstance(result, str)
        assert "COMPETING SOLUTIONS:" in result
        assert "SOLUTION 1" not in result  # No solutions listed


class TestDrafterPromptTemplate:
    """Test drafter prompt template generation."""

    def test_speculative_decoding_basic(self):
        """Test basic speculative decoding template."""
        context = "def calculate_fibonacci(n):\n    if n <= 1:\n        return n\n    else:\n        return"

        result = DrafterPromptTemplate.speculative_decoding(context)

        assert isinstance(result, str)
        assert "def calculate_fibonacci(n):" in result
        assert "speculative decoding" in result.lower()
        assert len(result) > 50  # Should be a substantial prompt

    def test_speculative_decoding_empty_context(self):
        """Test speculative decoding with empty context."""
        result = DrafterPromptTemplate.speculative_decoding("")

        assert isinstance(result, str)
        assert len(result) > 0  # Should still generate a prompt


class TestPlanningPromptTemplate:
    """Test planning prompt template generation."""

    def test_milestone_decomposition_basic(self):
        """Test basic milestone decomposition template."""
        result = PlanningPromptTemplate.milestone_decomposition(
            task_id="TASK-001",
            description="Build a web application with user authentication",
            complexity="medium",
        )

        assert isinstance(result, str)
        assert "TASK-001" in result
        assert "Build a web application with user authentication" in result
        assert "milestone decomposition" in result.lower()

    def test_milestone_decomposition_with_dependencies(self):
        """Test milestone decomposition with dependency information."""
        current_plan = {
            "milestones": ["Design schema", "Implement registration", "Implement login"],
            "dependencies": {
                "Implement registration": ["Design schema"],
                "Implement login": ["Design schema", "Implement registration"],
            }
        }

        result = PlanningPromptTemplate.milestone_decomposition(
            task_id="TASK-002",
            description="Build authentication system",
            complexity="medium",
            current_plan=current_plan,
        )

        assert isinstance(result, str)
        assert "TASK-002" in result
        assert "Design schema" in result
        assert "CURRENT PLAN:" in result

    def test_milestone_decomposition_empty_subtasks(self):
        """Test milestone decomposition with minimal parameters."""
        result = PlanningPromptTemplate.milestone_decomposition(
            task_id="TASK-003",
            description="Simple task",
            complexity="low",
        )

        assert isinstance(result, str)
        assert "TASK-003" in result
        assert "Simple task" in result

    def test_milestone_decomposition_with_caws_context(self):
        """Test milestone decomposition (CAWS context not supported in this method)."""
        result = PlanningPromptTemplate.milestone_decomposition(
            task_id="TASK-004",
            description="Implement feature",
            complexity="high",
        )

        assert isinstance(result, str)
        assert "TASK-004" in result
        assert "Implement feature" in result
        assert "high" in result


class TestPromptTemplateIntegration:
    """Test integration between different prompt templates."""

    def test_worker_judge_integration(self):
        """Test that worker and judge templates can work together."""
        # Generate worker output
        worker_prompt = WorkerPromptTemplate.autonomous_coding_agent(
            task_id="TASK-001",
            description="Implement factorial function",
        )

        # Simulate worker output
        worker_output = {
            "worker_id": "worker_1",
            "content": "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
            "evidence_completeness": 0.9,
            "budget_adherence": 0.8,
            "gate_integrity": 0.7,
            "provenance_clarity": 0.6,
        }

        # Generate judge prompt
        judge_prompt = JudgePromptTemplate.caws_debate_scoring(
            worker_outputs=[worker_output],
            working_spec={"title": "Factorial Implementation", "description": "Implement factorial"},
        )

        # Verify integration
        assert isinstance(worker_prompt, str)
        assert isinstance(judge_prompt, str)
        assert "TASK-001" in worker_prompt
        assert "SOLUTION 1" in judge_prompt
        assert "worker_1" in judge_prompt

    def test_template_output_format(self):
        """Test that all templates return properly formatted strings."""
        templates_to_test = [
            lambda: WorkerPromptTemplate.autonomous_coding_agent("task", "desc"),
            lambda: JudgePromptTemplate.caws_debate_scoring([{"worker_id": "test", "content": "test"}], {"title": "test"}),
            lambda: DrafterPromptTemplate.speculative_decoding("context"),
            lambda: PlanningPromptTemplate.milestone_suggestions("task", "desc", {"title": "spec"}, "medium"),
        ]

        for template_func in templates_to_test:
            result = template_func()
            assert isinstance(result, str)
            assert len(result.strip()) > 0
            # Should contain some basic formatting
            assert "\n" in result or " " in result

    def test_template_error_handling(self):
        """Test error handling in template generation."""
        # Test with None values that should be handled gracefully
        caws_context = CAWSContext(
            spec_id="TEST",
            title="Test",
            risk_tier=1,
            mode="test",
            budget={},
            scope={},
            quality={},
            acceptance_summary=None,  # Should handle None
            invariants=None,  # Should handle None
        )

        result = WorkerPromptTemplate.autonomous_coding_agent(
            task_id="TEST",
            description="Test task",
            caws_context=caws_context,
        )

        assert isinstance(result, str)
        assert "TEST" in result
