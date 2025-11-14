"""
Prompt Templates for Arbiter Stack Model Distillation

This module contains prompt templates extracted from agent-agency v3/v4
for worker, judge, and drafter models. These templates are used during
dataset generation to ensure consistency with production usage patterns.

Author: @darianrosebrook
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json


@dataclass
class CAWSContext:
    """CAWS context structure for prompt augmentation"""

    spec_id: str
    title: str
    risk_tier: int
    mode: str
    budget: Dict[str, int]
    scope: Dict[str, List[str]]
    quality: Dict[str, Any]
    acceptance_summary: List[str]
    invariants: List[str]


class WorkerPromptTemplate:
    """Prompt templates for Worker models (~9B GQA)"""

    @staticmethod
    def autonomous_coding_agent(
        task_id: str,
        description: str,
        caws_context: Optional[CAWSContext] = None,
        acceptance_criteria: Optional[List[str]] = None,
        tool_registry: Optional[List[Dict[str, Any]]] = None,
        context_metrics: Optional[Dict[str, float]] = None,
        use_compact_caws: bool = True,
    ) -> str:
        """
        Template for autonomous coding agent worker.

        Used for: Code generation, file manipulation, tool-use JSON generation

        Source: agent-agency/iterations/v3/agent-orchestration/src/autonomous_integration.rs:215-253
        """
        caws_section = ""
        if caws_context:
            if use_compact_caws:
                # PRIORITY 4: Use compact JSON format (≤ 30 tokens)
                caws_compact = format_caws_compact(caws_context)
                caws_section = f"CAWS: {caws_compact}\n"
            else:
                # Legacy verbose format
                acceptance_text = "\n".join(
                    [f"- {c}" for c in (acceptance_criteria or caws_context.acceptance_summary)]
                )
                caws_section = f"""
CAWS CONTEXT:
- Spec ID: {caws_context.spec_id}
- Title: {caws_context.title}
- Risk Tier: {caws_context.risk_tier}
- Mode: {caws_context.mode}
- Budget: {caws_context.budget.get("max_files", "N/A")} files, {caws_context.budget.get("max_loc", "N/A")} LOC
- Scope In: {", ".join(caws_context.scope.get("in", [])[:5])}
- Scope Out: {", ".join(caws_context.scope.get("out", [])[:5])}

ACCEPTANCE CRITERIA:
{acceptance_text}

INVARIANTS:
{chr(10).join([f"- {inv}" for inv in caws_context.invariants])}
"""

        tool_section = ""
        if tool_registry:
            tool_list = "\n".join(
                [
                    f"- {tool.get('name', 'Unknown')}: {tool.get('description', 'No description')}"
                    for tool in tool_registry[:10]  # Limit to first 10 tools
                ]
            )
            tool_section = f"""
AVAILABLE TOOLS:
{tool_list}

Use tool-use JSON format when invoking tools:
{{
    "tool_name": "tool_name",
    "arguments": {{"arg1": "value1"}}
}}
"""

        metrics_section = ""
        if context_metrics:
            metrics_section = f"""
CONTEXT METRICS:
- Task Complexity: {context_metrics.get("task_complexity", 0.5):.2f}
- Available CPU: {context_metrics.get("cpu_available", 50.0):.1f}%
- Available Memory: {context_metrics.get("memory_available", 50.0):.1f}%
"""

        prompt = f"""You are an autonomous coding agent. Plan and execute the following task:

TASK: {task_id}
DESCRIPTION: {description}
{caws_section}{metrics_section}{tool_section}
REQUIREMENTS:
- Maintain code quality and follow CAWS standards
- Use safe file operations with rollback capabilities
- Provide detailed reasoning for all changes
- Ensure changes are testable and maintainable
- Stay within scope boundaries and budget limits

PLAN the specific file changes needed to complete this task. Format your response as:

REASONING:
[Your step-by-step reasoning process. Explain why each change is necessary and how it addresses the acceptance criteria.]

CHANGES:
[File change specifications, one per line]
- CREATE|REPLACE|INSERT|DELETE path/to/file.rs: description of change

VERIFICATION:
[How to verify the changes work correctly, including test cases or validation steps]
"""
        return prompt

    @staticmethod
    def tool_use_generation(
        task_description: str,
        available_tools: List[Dict[str, Any]],
        caws_context: Optional[CAWSContext] = None,
        use_compact_caws: bool = True,
    ) -> str:
        """
        Template for tool-use JSON generation.

        Used for: Generating structured tool invocation JSON

        Source: agent-agency/iterations/v3/agent-orchestration/src/planning/caws_tool_registry.rs
        """
        tool_list = "\n".join(
            [
                f"- {tool.get('name', 'Unknown')} ({tool.get('tool_id', 'unknown')}): {tool.get('description', 'No description')}"
                for tool in available_tools
            ]
        )

        caws_section = ""
        if caws_context:
            if use_compact_caws:
                # PRIORITY 4: Use compact JSON format
                caws_compact = format_caws_compact(caws_context)
                caws_section = f"CAWS: {caws_compact}\n"
            else:
                # Legacy verbose format
                caws_section = f"""
CAWS COMPLIANCE:
- Risk Tier: {caws_context.risk_tier}
- Budget: {caws_context.budget.get("max_files", "N/A")} files, {caws_context.budget.get("max_loc", "N/A")} LOC
- Scope: {", ".join(caws_context.scope.get("in", [])[:3])}

"""

        prompt = f"""Generate tool-use JSON for the following task:

TASK: {task_description}
{caws_section}
AVAILABLE TOOLS:
{tool_list}

Generate a JSON object with the following structure:
{{
    "tool_name": "exact_tool_name",
    "arguments": {{
        "param1": "value1",
        "param2": "value2"
    }},
    "reasoning": "Why this tool is appropriate for this task"
}}

Ensure the tool name matches exactly one of the available tools.
"""
        return prompt


class JudgePromptTemplate:
    """Prompt templates for Judge models (3-4B or 7B)"""

    @staticmethod
    def caws_debate_scoring(
        worker_outputs: List[Dict[str, Any]],
        working_spec: Dict[str, Any],
        claim_results: Optional[Dict[str, Any]] = None,
        quality_gate_result: Optional[Dict[str, Any]] = None,
        complexity_mode: Optional[str] = None,
    ) -> str:
        """
        Template for CAWS debate scoring by judge model.

        Used for: Evaluating competing worker solutions using CAWS criteria

        Source: agent-agency/iterations/v3/agent-orchestration/src/planning/caws_debate_scorer.rs:200-296
        """
        solutions_text = "\n\n".join(
            [
                f"""SOLUTION {i + 1} (Worker {output.get("worker_id", "unknown")}):
{output.get("content", "No content")}

Evidence Completeness: {output.get("evidence_completeness", 0.0):.2f}
Budget Adherence: {output.get("budget_adherence", 0.0):.2f}
Gate Integrity: {output.get("gate_integrity", 0.0):.2f}
Provenance Clarity: {output.get("provenance_clarity", 0.0):.2f}
"""
                for i, output in enumerate(worker_outputs)
            ]
        )

        claim_section = ""
        if claim_results:
            claim_section = f"""
CLAIM VERIFICATION RESULTS:
- Total Claims: {claim_results.get("total_claims", 0)}
- Verified Claims: {claim_results.get("verified_claims", 0)}
- Verification Confidence: {claim_results.get("verification_confidence", 0.0):.2f}
- Evidence Count: {claim_results.get("evidence_count", 0)}
"""

        gate_section = ""
        if quality_gate_result:
            gate_section = f"""
QUALITY GATE RESULTS:
- Total Violations: {quality_gate_result.get("total_violations", 0)}
- Waived Violations: {quality_gate_result.get("waived_violations", 0)}
- Blocking Violations: {quality_gate_result.get("blocking_violations", 0)}
"""

        mode_section = ""
        if complexity_mode:
            weights = {
                "Simple": (0.3, 0.3, 0.2, 0.2),
                "Standard": (0.4, 0.3, 0.2, 0.1),
                "Enterprise": (0.5, 0.25, 0.2, 0.05),
            }
            e_weight, b_weight, g_weight, p_weight = weights.get(
                complexity_mode, (0.4, 0.3, 0.2, 0.1)
            )
            mode_section = f"""
COMPLEXITY MODE: {complexity_mode}
SCORING WEIGHTS:
- Evidence Completeness: {e_weight:.1%}
- Budget Adherence: {b_weight:.1%}
- Gate Integrity: {g_weight:.1%}
- Provenance Clarity: {p_weight:.1%}
"""

        prompt = f"""You are a CAWS constitutional arbiter judge. Evaluate competing worker solutions using CAWS criteria.

WORKING SPECIFICATION:
- ID: {working_spec.get("id", "unknown")}
- Title: {working_spec.get("title", "unknown")}
- Risk Tier: {working_spec.get("risk_tier", 2)}
- Mode: {working_spec.get("mode", "feature")}
{mode_section}{claim_section}{gate_section}
COMPETING SOLUTIONS:
{solutions_text}

SCORING FORMULA:
Total Score = (Evidence × {e_weight if complexity_mode else 0.4}) + (Budget × {b_weight if complexity_mode else 0.3}) + (Gate × {g_weight if complexity_mode else 0.2}) + (Provenance × {p_weight if complexity_mode else 0.1})

Evaluate each solution and provide:
1. Individual component scores (0.0 to 1.0)
2. Total score calculation
3. Winner determination
4. Confidence level (0.0 to 1.0)
5. Judge notes summarizing the debate

Format your response as:
WINNER: Solution <number>
CONFIDENCE: <0.0-1.0>
SCORES:
- Solution 1: Evidence=<score>, Budget=<score>, Gate=<score>, Provenance=<score>, Total=<score>
- Solution 2: Evidence=<score>, Budget=<score>, Gate=<score>, Provenance=<score>, Total=<score>
NOTES: <summary of evaluation>
"""
        return prompt

    @staticmethod
    def claim_extraction_verification(
        worker_output: str,
        working_spec: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Template for claim extraction and verification.

        Used for: Extracting verifiable claims from worker outputs

        Source: agent-agency/iterations/v3/agent-research/src/processor.rs
        """
        prompt = f"""Extract verifiable claims from the following worker output and verify them against the working specification.

WORKING SPECIFICATION:
- ID: {working_spec.get("id", "unknown")}
- Title: {working_spec.get("title", "unknown")}
- Risk Tier: {working_spec.get("risk_tier", 2)}

WORKER OUTPUT:
{worker_output}

Extract atomic, verifiable claims and verify them using:
1. Cross-references (30% weight): Check against code, documentation, tests
2. Code behavior (25% weight): Verify code actually implements the claim
3. Authority (20% weight): Check against authoritative sources
4. Context (15% weight): Verify consistency with conversation context
5. Semantics (10% weight): Verify logical consistency

For each claim, provide:
- Claim ID
- Claim statement
- Verification confidence (0.0 to 1.0)
- Evidence sources
- Verification method used

Format as JSON:
{{
    "claims": [
        {{
            "id": "claim_1",
            "statement": "exact claim text",
            "confidence": 0.85,
            "evidence": ["source1", "source2"],
            "verification_method": "cross_reference"
        }}
    ],
    "total_claims": 1,
    "verified_claims": 1,
    "verification_confidence": 0.85
}}
"""
        return prompt


class DrafterPromptTemplate:
    """Prompt templates for Drafter models (~4B, speculative decoding)"""

    @staticmethod
    def speculative_decoding(
        prompt: str,
        worker_model_output: Optional[str] = None,
    ) -> str:
        """
        Template for speculative decoding drafter.

        Used for: Fast token generation for latency optimization

        Source: Speculative decoding research patterns
        """
        alignment_section = ""
        if worker_model_output:
            alignment_section = f"""
ALIGN WITH WORKER MODEL OUTPUT:
{worker_model_output}

Generate tokens that align with the worker model's reasoning and output style.
"""

        draft_prompt = f"""Generate a draft response for speculative decoding optimization.

ORIGINAL PROMPT:
{prompt}
{alignment_section}
Generate tokens quickly, maintaining quality and alignment with the target model.
Focus on speed while preserving semantic correctness.
"""
        return draft_prompt


class PlanningPromptTemplate:
    """Prompt templates for planning and decomposition"""

    @staticmethod
    def milestone_decomposition(
        task_id: str,
        description: str,
        complexity: str,
        current_plan: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Template for milestone decomposition planning.

        Source: agent-agency/iterations/v3/agent-orchestration/src/planning/plan_generator.rs:762-808
        """
        plan_section = ""
        if current_plan:
            plan_section = f"""
CURRENT PLAN:
- Milestones: {current_plan.get("milestones", [])}
- Dependencies: {current_plan.get("dependencies", [])}
"""

        prompt = f"""You are an AI planning assistant. Analyze this task and suggest optimal milestone decomposition.

TASK: {task_id}
DESCRIPTION: {description}

COMPLEXITY: {complexity}
{plan_section}
REQUIREMENTS:
- Break down into logical, testable milestones
- Identify dependencies between milestones
- Suggest optimal execution order
- Consider parallel execution opportunities

Provide your analysis and milestone suggestions in a structured format.
"""
        return prompt

    @staticmethod
    def milestone_suggestions(
        task_id: str,
        description: str,
        working_spec: Dict[str, Any],
        complexity: str,
    ) -> str:
        """
        Template for milestone suggestions.

        Source: agent-agency/iterations/v3/agent-orchestration/src/planning/plan_generator.rs:810-860
        """
        acceptance_criteria = working_spec.get("acceptance_criteria", [])
        acceptance_text = "\n".join(
            [
                f"- {c.get('id', 'unknown')}: {c.get('given', '')} → {c.get('when', '')} → {c.get('then', '')}"
                for c in acceptance_criteria
            ]
        )

        prompt = f"""You are an AI planning assistant. Suggest optimal milestone breakdown for this task.

TASK: {task_id}
DESCRIPTION: {description}

COMPLEXITY: {complexity}
RISK TIER: {working_spec.get("risk_tier", 2)}

ACCEPTANCE CRITERIA:
{acceptance_text}

REQUIREMENTS:
- Create logical milestones that map to acceptance criteria
- Identify dependencies and execution order
- Suggest optimal parallelization opportunities
- Consider risk tier and complexity in milestone sizing

Provide milestone suggestions in a structured format with:
- Milestone IDs and objectives
- Dependencies between milestones
- Suggested execution order
- Parallel execution opportunities
"""
        return prompt


def format_caws_compact(working_spec: Dict[str, Any]) -> str:
    """
    Format CAWS as compact JSON metadata (≤ 30 tokens target).

    PRIORITY 4: Token-light CAWS format for efficiency.

    Args:
        working_spec: Working specification dict (can be CAWSContext or dict)

    Returns:
        Compact JSON string with minimal CAWS metadata
    """
    # Extract fields from CAWSContext (from caws_context.py or prompt_templates.py) or dict
    # Use duck typing: check for attributes instead of isinstance to handle both CAWSContext classes
    if hasattr(working_spec, "risk_tier") and hasattr(working_spec, "budget") and hasattr(working_spec, "scope"):
        # It's a CAWSContext object (from either module)
        tier = working_spec.risk_tier
        max_files = working_spec.budget.get("max_files", 25)
        max_loc = working_spec.budget.get("max_loc", 1000)
        cov = working_spec.quality.get("coverage_threshold", 80)
        mut = working_spec.quality.get("mutation_threshold", 50)
        scope_in = working_spec.scope.get("in", [])[:5]  # Limit to 5
        scope_out = working_spec.scope.get("out", [])[:5]
    else:
        # It's a dict
        tier = working_spec.get("risk_tier", 2)
        budget = working_spec.get("budget", {})
        max_files = budget.get("max_files", 25)
        max_loc = budget.get("max_loc", 1000)
        quality = working_spec.get("quality_gates", {})
        cov = quality.get("coverage", 80)
        mut = quality.get("mutation_score", 50)
        scope = working_spec.get("scope", {})
        scope_in = scope.get("in", [])[:5]
        scope_out = scope.get("out", [])[:5]

    caws_dict = {
        "caws": {
            "tier": tier,
            "max_files": max_files,
            "max_loc": max_loc,
            "cov": cov,
            "mut": mut,
            "in": scope_in,
            "out": scope_out,
        }
    }
    # Use compact JSON (no spaces)
    return json.dumps(caws_dict, separators=(",", ":"))


def format_caws_context_for_prompt(caws_context: CAWSContext, compact: bool = False) -> str:
    """
    Format CAWS context for inclusion in prompts.

    Source: agent-agency/iterations/v3/agent-orchestration/src/planning/caws_plan_bridge.rs

    Args:
        caws_context: CAWSContext object
        compact: If True, use compact JSON format (≤ 30 tokens). If False, use verbose markdown.
    """
    if compact:
        # PRIORITY 4: Use compact JSON format
        return format_caws_compact(caws_context)
    else:
        # Legacy verbose format
        return f"""
CAWS Working Specification:
- ID: {caws_context.spec_id}
- Title: {caws_context.title}
- Risk Tier: {caws_context.risk_tier}
- Mode: {caws_context.mode}
- Budget: {caws_context.budget.get("max_files", "N/A")} files, {caws_context.budget.get("max_loc", "N/A")} LOC
- Scope In: {", ".join(caws_context.scope.get("in", [])[:5])}
- Scope Out: {", ".join(caws_context.scope.get("out", [])[:5])}
- Quality Thresholds: Coverage={caws_context.quality.get("coverage_threshold", "N/A")}%, Mutation={caws_context.quality.get("mutation_threshold", "N/A")}%
"""


# Example usage and testing
if __name__ == "__main__":
    # Example CAWS context
    caws_ctx = CAWSContext(
        spec_id="FEAT-001",
        title="User Authentication",
        risk_tier=1,
        mode="feature",
        budget={"max_files": 25, "max_loc": 1000},
        scope={"in": ["src/auth/", "tests/auth/"], "out": ["node_modules/", "dist/"]},
        quality={"coverage_threshold": 80, "mutation_threshold": 60},
        acceptance_summary=[
            "A1: User submits valid credentials → Authentication succeeds → User is logged in"
        ],
        invariants=[
            "Authentication state never stored in localStorage",
            "All tokens expire within 24h",
        ],
    )

    # Test worker prompt
    worker_prompt = WorkerPromptTemplate.autonomous_coding_agent(
        task_id="TASK-001",
        description="Implement user authentication flow",
        caws_context=caws_ctx,
        acceptance_criteria=["A1: User submits valid credentials → Authentication succeeds"],
        context_metrics={"task_complexity": 0.7, "cpu_available": 60.0, "memory_available": 70.0},
    )
    print("WORKER PROMPT:")
    print(worker_prompt)
    print("\n" + "=" * 80 + "\n")

    # Test judge prompt
    judge_prompt = JudgePromptTemplate.caws_debate_scoring(
        worker_outputs=[
            {
                "worker_id": "worker-1",
                "content": "Solution 1: Implemented auth with JWT",
                "evidence_completeness": 0.8,
                "budget_adherence": 0.9,
                "gate_integrity": 0.7,
                "provenance_clarity": 0.6,
            }
        ],
        working_spec={"id": "FEAT-001", "title": "User Auth", "risk_tier": 1, "mode": "feature"},
        complexity_mode="Standard",
    )
    print("JUDGE PROMPT:")
    print(judge_prompt)
