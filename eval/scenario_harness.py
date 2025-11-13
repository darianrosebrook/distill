"""
Scenario-based evaluation harness.

Provides task-level behavioral evaluation beyond cosine similarity and CAWS gates.
Evaluates student vs teacher vs baseline on curated scenario suites.
@author: @darianrosebrook
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import typer

app = typer.Typer()


class ScenarioRole(str, Enum):
    """Scenario roles."""

    WORKER = "worker"
    JUDGE = "judge"
    DRAFTER = "drafter"


class ScenarioResult(str, Enum):
    """Scenario evaluation result."""

    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


@dataclass
class Scenario:
    """A single evaluation scenario."""

    id: str
    role: ScenarioRole
    prompt: str
    teacher_output: str
    target_student_behavior: str
    scoring_rubric: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class ScenarioEvaluation:
    """Result of evaluating a scenario."""

    scenario_id: str
    result: ScenarioResult
    student_output: str
    teacher_output: str
    baseline_output: Optional[str]
    score: float
    details: Dict[str, Any]
    errors: List[str]


class ScenarioEvaluator:
    """Evaluator for scenario-based testing."""

    def __init__(self, scenarios: List[Scenario]):
        """
        Initialize evaluator with scenarios.

        Args:
            scenarios: List of scenarios to evaluate
        """
        self.scenarios = scenarios
        self.evaluations: List[ScenarioEvaluation] = []

    def evaluate_scenario(
        self,
        scenario: Scenario,
        student_output: str,
        teacher_output: Optional[str] = None,
        baseline_output: Optional[str] = None,
    ) -> ScenarioEvaluation:
        """
        Evaluate a single scenario.

        Args:
            scenario: Scenario to evaluate
            student_output: Student model output
            teacher_output: Teacher model output (if provided)
            baseline_output: Baseline model output (if provided)

        Returns:
            ScenarioEvaluation with result and details
        """
        errors = []
        score = 0.0
        details = {}

        # Use provided teacher output or scenario's default
        if teacher_output is None:
            teacher_output = scenario.teacher_output

        # Run scoring rubric
        rubric = scenario.scoring_rubric

        # Check for catastrophic failures
        if self._has_catastrophic_failure(student_output, scenario):
            errors.append("Catastrophic failure detected")
            return ScenarioEvaluation(
                scenario_id=scenario.id,
                result=ScenarioResult.FAIL,
                student_output=student_output,
                teacher_output=teacher_output,
                baseline_output=baseline_output,
                score=0.0,
                details={"catastrophic_failure": True},
                errors=errors,
            )

        # Evaluate based on rubric type
        rubric_type = rubric.get("type", "schema_check")

        if rubric_type == "schema_check":
            # Check tool call schema validity
            is_valid, schema_errors = self._check_tool_schema(student_output, rubric)
            if not is_valid:
                errors.extend(schema_errors)
            else:
                score += 0.5

        if rubric_type == "functional_test":
            # Run functional test script
            test_result = self._run_functional_test(student_output, rubric)
            if test_result["passed"]:
                score += 0.5
            else:
                errors.extend(test_result.get("errors", []))

        if rubric_type == "judge_model":
            # Use judge model to evaluate
            judge_score = self._judge_evaluation(student_output, teacher_output, rubric)
            score += judge_score

        # Determine result
        if score >= rubric.get("pass_threshold", 0.8):
            result = ScenarioResult.PASS
        elif score >= rubric.get("inconclusive_threshold", 0.5):
            result = ScenarioResult.INCONCLUSIVE
        else:
            result = ScenarioResult.FAIL

        details["score"] = score
        details["rubric_type"] = rubric_type

        return ScenarioEvaluation(
            scenario_id=scenario.id,
            result=result,
            student_output=student_output,
            teacher_output=teacher_output,
            baseline_output=baseline_output,
            score=score,
            details=details,
            errors=errors,
        )

    def _has_catastrophic_failure(self, output: str, scenario: Scenario) -> bool:
        """Check for catastrophic failures (wrong tool spam, invalid JSON, etc.)."""
        # Check for tool spam (too many tool calls)
        tool_call_count = output.count("<bot>")
        max_tool_calls = scenario.metadata.get("max_tool_calls", 10)
        if tool_call_count > max_tool_calls:
            return True

        # Check for invalid JSON in tool arguments
        import re

        json_pattern = r"\{[^}]*\}"
        json_matches = re.findall(json_pattern, output)
        for json_str in json_matches:
            try:
                json.loads(json_str)
            except json.JSONDecodeError:
                return True

        return False

    def _check_tool_schema(self, output: str, rubric: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Check tool call schema validity."""
        errors = []

        # Extract tool calls from output
        import re

        tool_pattern = r"<bot>([^<]+)</bot>"
        tool_matches = re.findall(tool_pattern, output)

        # Check each tool call against schema
        expected_tools = rubric.get("expected_tools", [])
        for tool_match in tool_matches:
            # Parse tool call
            try:
                tool_call = json.loads(tool_match)
                tool_name = tool_call.get("name")

                if tool_name not in expected_tools:
                    errors.append(f"Unexpected tool: {tool_name}")
            except json.JSONDecodeError:
                errors.append(f"Invalid JSON in tool call: {tool_match}")

        return len(errors) == 0, errors

    def _run_functional_test(self, output: str, rubric: Dict[str, Any]) -> Dict[str, Any]:
        """Run functional test script."""
        # PLACEHOLDER: Functional test execution not implemented
        # In practice, this would execute a test script or unit test
        test_script = rubric.get("test_script")
        if test_script:
            # PLACEHOLDER: Execute test script
            return {"passed": False, "errors": ["Functional test not implemented"]}

        return {"passed": True, "errors": []}

    def _judge_evaluation(
        self, student_output: str, teacher_output: str, rubric: Dict[str, Any]
    ) -> float:
        """Use judge model to evaluate student output."""
        # PLACEHOLDER: Judge model evaluation not implemented
        # In practice, this would call a judge model to compare outputs
        return 0.5

    def evaluate_all(
        self,
        student_outputs: Dict[str, str],
        teacher_outputs: Optional[Dict[str, str]] = None,
        baseline_outputs: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate all scenarios.

        Args:
            student_outputs: Dictionary mapping scenario IDs to student outputs
            teacher_outputs: Optional dictionary mapping scenario IDs to teacher outputs
            baseline_outputs: Optional dictionary mapping scenario IDs to baseline outputs

        Returns:
            Dictionary with evaluation summary
        """
        evaluations = []

        for scenario in self.scenarios:
            student_output = student_outputs.get(scenario.id, "")
            teacher_output = teacher_outputs.get(scenario.id) if teacher_outputs else None
            baseline_output = baseline_outputs.get(scenario.id) if baseline_outputs else None

            evaluation = self.evaluate_scenario(
                scenario,
                student_output,
                teacher_output,
                baseline_output,
            )
            evaluations.append(evaluation)

        self.evaluations = evaluations

        # Compute summary statistics
        total = len(evaluations)
        passed = sum(1 for e in evaluations if e.result == ScenarioResult.PASS)
        failed = sum(1 for e in evaluations if e.result == ScenarioResult.FAIL)
        inconclusive = sum(1 for e in evaluations if e.result == ScenarioResult.INCONCLUSIVE)

        avg_score = sum(e.score for e in evaluations) / max(1, total)

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "inconclusive": inconclusive,
            "pass_rate": passed / max(1, total),
            "avg_score": avg_score,
            "evaluations": [
                {
                    "scenario_id": e.scenario_id,
                    "result": e.result.value,
                    "score": e.score,
                    "errors": e.errors,
                }
                for e in evaluations
            ],
        }


def load_scenarios(scenarios_path: Path, role: Optional[ScenarioRole] = None) -> List[Scenario]:
    """
    Load scenarios from JSONL file.

    Args:
        scenarios_path: Path to scenarios JSONL file
        role: Optional role filter

    Returns:
        List of Scenario objects
    """
    scenarios = []

    with open(scenarios_path, "r") as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)

            # Filter by role if specified
            if role and data.get("role") != role.value:
                continue

            scenario = Scenario(
                id=data["id"],
                role=ScenarioRole(data["role"]),
                prompt=data["prompt"],
                teacher_output=data.get("teacher_output", ""),
                target_student_behavior=data.get("target_student_behavior", ""),
                scoring_rubric=data.get("scoring_rubric", {}),
                metadata=data.get("metadata", {}),
            )
            scenarios.append(scenario)

    return scenarios


@app.command()
def main(
    scenarios_path: str = typer.Argument(..., help="Path to scenarios JSONL file"),
    student_outputs_path: str = typer.Argument(..., help="Path to student outputs JSONL file"),
    teacher_outputs_path: Optional[str] = typer.Option(
        None, help="Path to teacher outputs JSONL file"
    ),
    baseline_outputs_path: Optional[str] = typer.Option(
        None, help="Path to baseline outputs JSONL file"
    ),
    output_path: str = typer.Option(
        "eval/scenario_results.json", help="Output path for evaluation results"
    ),
    role: Optional[str] = typer.Option(
        None, help="Filter scenarios by role (worker/judge/drafter)"
    ),
):
    """
    Run scenario-based evaluation.
    """
    scenarios_file = Path(scenarios_path)
    student_outputs_file = Path(student_outputs_path)

    # Load scenarios
    scenario_role = ScenarioRole(role) if role else None
    scenarios = load_scenarios(scenarios_file, scenario_role)
    print(f"[scenario_harness] Loaded {len(scenarios)} scenarios")

    # Load student outputs
    student_outputs = {}
    with open(student_outputs_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            student_outputs[data["scenario_id"]] = data["output"]

    # Load teacher outputs if provided
    teacher_outputs = None
    if teacher_outputs_path:
        teacher_outputs = {}
        with open(teacher_outputs_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                teacher_outputs[data["scenario_id"]] = data["output"]

    # Load baseline outputs if provided
    baseline_outputs = None
    if baseline_outputs_path:
        baseline_outputs = {}
        with open(baseline_outputs_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                baseline_outputs[data["scenario_id"]] = data["output"]

    # Run evaluation
    evaluator = ScenarioEvaluator(scenarios)
    results = evaluator.evaluate_all(student_outputs, teacher_outputs, baseline_outputs)

    # Save results
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n=== Scenario Evaluation Summary ===")
    print(f"Total scenarios: {results['total']}")
    print(f"Passed: {results['passed']} ({results['pass_rate'] * 100:.1f}%)")
    print(f"Failed: {results['failed']}")
    print(f"Inconclusive: {results['inconclusive']}")
    print(f"Average score: {results['avg_score']:.3f}")

    # Check if pass rate meets threshold
    pass_threshold = 0.8  # 80% pass rate required
    if results["pass_rate"] < pass_threshold:
        print(
            f"\n[scenario_harness] ⚠️ Pass rate {results['pass_rate'] * 100:.1f}% < threshold {pass_threshold * 100:.1f}%"
        )
        sys.exit(1)
    else:
        print("\n[scenario_harness] ✅ Pass rate meets threshold")
        sys.exit(0)


if __name__ == "__main__":
    app()
