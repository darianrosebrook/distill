"""
Generate CAWS-specific tool-use examples for Worker dataset.

Generates examples using CAWS MCP tools (validation, auditing, waiver creation),
including budget-aware generation, scope boundary violations, waiver scenarios,
and negative CAWS examples (refusals, denied waivers, "cannot comply under current budget").

Author: @darianrosebrook
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from tools.schema_registry import get_registry
from training.caws_context import extract_caws_context, extract_caws_context_dict


def generate_caws_tool_prompt(
    scenario: str,
    complexity: str,
    include_negative: bool = False,
    caws_context: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Generate a CAWS tool-use prompt based on scenario and complexity.

    Args:
        scenario: One of "file_ops", "web_search", "code_exec", "multi_step"
        complexity: One of "single_call", "multi_call", "branching_error_recovery"
        include_negative: If True, generate a negative example (refusal, violation, denied waiver)
        caws_context: CAWS context object (optional)

    Returns:
        Dictionary with prompt and metadata
    """
    prompts = {
        "file_ops": {
            "single_call": [
                "Read the file config.yaml and extract the learning_rate value.",
                "Write a new file test_config.json with the provided configuration.",
                "Check if the file src/main.py exists and report its size.",
            ],
            "multi_call": [
                "Read config.yaml, then update the learning_rate field, and finally write it back.",
                "Check multiple files in src/ and report which ones contain 'TODO' comments.",
                "Read package.json, extract dependencies, then create a requirements.txt file.",
            ],
            "branching_error_recovery": [
                "Read config.yaml. If it doesn't exist, create it with default values. Then validate the structure.",
                "Try to read src/main.py. If it fails, check if it's in a different location and read from there.",
            ],
        },
        "web_search": {
            "single_call": [
                "Search for 'CoreML optimization best practices' and summarize the top 3 results.",
                "Find information about 'Python async best practices' and extract key points.",
            ],
            "multi_call": [
                "Search for 'TypeScript type safety', then search for 'TypeScript performance tips', and combine insights.",
                "Find documentation on 'Rust error handling', then search for examples, and create a summary.",
            ],
            "branching_error_recovery": [
                "Search for 'Python decorators'. If results are insufficient, try alternative search terms.",
            ],
        },
        "code_exec": {
            "single_call": [
                "Execute the Python script test.py and report stdout/stderr.",
                "Run the test suite with pytest and report results.",
            ],
            "multi_call": [
                "Run linting, then run tests, then generate coverage report.",
                "Execute build script, then run integration tests, then deploy if successful.",
            ],
            "branching_error_recovery": [
                "Run tests. If they fail, run with verbose output and report specific failures.",
            ],
        },
        "multi_step": {
            "single_call": [],
            "multi_call": [
                "Read config.yaml, search for best practices online, update config based on findings, then validate.",
                "Check file structure, search for documentation, create missing files, then run validation.",
            ],
            "branching_error_recovery": [
                "Read config, validate. If invalid, search for correct format, update, then re-validate.",
            ],
        },
    }

    negative_prompts = {
        "refusal": [
            "Modify files outside the allowed scope (third_party/).",
            "Create more than {max_files} files in a single operation.",
            "Modify code that exceeds the {max_loc} LOC budget.",
        ],
        "denied_waiver": [
            "Request a waiver to bypass test coverage requirements without justification.",
            "Ask for a waiver to modify files outside scope without documented necessity.",
        ],
        "explanation": [
            "Explain why you cannot modify files in third_party/ due to scope restrictions.",
            "Explain why you cannot create {max_files + 5} files when the budget allows only {max_files}.",
        ],
    }

    # Select base prompt
    scenario_prompts = prompts.get(scenario, {}).get(complexity, [])
    if not scenario_prompts:
        # Fallback to multi_call if single_call is empty
        scenario_prompts = prompts.get(scenario, {}).get("multi_call", [])

    if include_negative:
        # Select negative prompt type
        negative_type = random.choice(
            ["refusal", "denied_waiver", "explanation"])
        base_prompts = negative_prompts.get(negative_type, [])
        if base_prompts:
            prompt_template = random.choice(base_prompts)
            # Fill in CAWS values if available
            if caws_context:
                max_files = caws_context.budget.get("max_files", 25)
                max_loc = caws_context.budget.get("max_loc", 1000)
                prompt = prompt_template.format(
                    max_files=max_files, max_loc=max_loc)
            else:
                prompt = prompt_template.format(max_files=25, max_loc=1000)
            negative_label = negative_type
        else:
            prompt = random.choice(
                scenario_prompts) if scenario_prompts else f"Perform {scenario} task"
            negative_label = "refusal"
    else:
        prompt = random.choice(
            scenario_prompts) if scenario_prompts else f"Perform {scenario} task"
        negative_label = None

    return {
        "prompt": prompt,
        "scenario": scenario,
        "complexity": complexity,
        "negative": include_negative,
        "negative_label": negative_label,
    }


def generate_evidence_manifest() -> Dict[str, Any]:
    """Generate a synthetic evidence manifest."""
    return {
        "claims": [
            "File operation completed successfully",
            "Budget constraints respected",
            "Scope boundaries maintained",
        ],
        "verification_status": random.choice(["pending", "verified", "rejected"]),
        "evidence_references": [
            "file://config.yaml",
            "caws://working-spec.yaml",
        ],
    }


def generate_provenance_chain() -> Dict[str, Any]:
    """Generate a synthetic provenance chain."""
    return {
        "steps": [
            {
                "step": 1,
                "action": "read_file",
                "target": "config.yaml",
                "timestamp": datetime.now().isoformat(),
            },
            {
                "step": 2,
                "action": "validate",
                "target": "config.yaml",
                "timestamp": datetime.now().isoformat(),
            },
        ],
        "audit_trail": "CAWS-compliant file operation sequence",
    }


def main():
    ap = argparse.ArgumentParser(
        description="Generate CAWS-specific tool-use examples for Worker dataset",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--out", required=True, help="Output JSONL file path")
    ap.add_argument(
        "--total",
        type=int,
        default=500,
        help="Total number of samples to generate",
    )
    ap.add_argument(
        "--scenarios",
        type=str,
        default="file_ops,web_search,code_exec,multi_step",
        help="Comma-separated list of scenarios to generate",
    )
    ap.add_argument(
        "--include-waiver-scenarios",
        action="store_true",
        help="Include waiver-related scenarios",
    )
    ap.add_argument(
        "--include-budget-violations",
        action="store_true",
        help="Include budget violation examples",
    )
    ap.add_argument(
        "--include-refusal-examples",
        action="store_true",
        help="Include refusal examples (Worker refusing to act)",
    )
    ap.add_argument(
        "--include-denied-waiver-examples",
        action="store_true",
        help="Include denied waiver examples",
    )
    ap.add_argument(
        "--caws-spec-id",
        help="CAWS spec ID to use for context extraction",
    )
    ap.add_argument(
        "--negative-ratio",
        type=float,
        default=0.1,
        help="Fraction of samples that are negative examples (default: 0.1)",
    )
    args = ap.parse_args()

    # Parse scenarios
    scenario_list = [s.strip() for s in args.scenarios.split(",")]
    complexities = ["single_call", "multi_call", "branching_error_recovery"]

    # Extract CAWS context
    caws_context = None
    caws_context_dict = None
    try:
        caws_context = extract_caws_context(".", spec_id=args.caws_spec_id)
        if caws_context:
            caws_context_dict = extract_caws_context_dict(
                ".", spec_id=args.caws_spec_id)
            print(
                f"[generate_caws_tool_examples] CAWS context loaded: {caws_context.spec_id}")
    except Exception as e:
        print(
            f"[generate_caws_tool_examples] WARN: Failed to extract CAWS context: {e}")

    # Generate samples
    samples = []
    negative_count = 0
    total_negative = int(args.total * args.negative_ratio)

    for i in range(args.total):
        scenario = random.choice(scenario_list)
        complexity = random.choice(complexities)

        # Determine if this should be a negative example
        include_negative = False
        if args.include_refusal_examples or args.include_denied_waiver_examples or args.include_budget_violations:
            include_negative = negative_count < total_negative and random.random() < args.negative_ratio

        if include_negative:
            negative_count += 1

        prompt_data = generate_caws_tool_prompt(
            scenario=scenario,
            complexity=complexity,
            include_negative=include_negative,
            caws_context=caws_context,
        )

        # Determine caws_level (always 2 for CAWS tool examples)
        caws_level = 2

        sample = {
            "id": f"caws-tool-{i+1:06d}",
            "role": "worker",
            "task_type": "caws_tool",
            "caws_level": caws_level,
            "source": "synthetic",
            "prompt": prompt_data["prompt"],
            "scenario": prompt_data["scenario"],
            "complexity": prompt_data["complexity"],
            "negative": prompt_data["negative"],
            "negative_label": prompt_data["negative_label"],
        }

        # Add CAWS context
        if caws_context_dict:
            sample["caws_context"] = {
                "working_spec": {
                    "id": caws_context_dict.get("spec_id", "unknown"),
                    "title": caws_context_dict.get("title", "Unknown"),
                    "risk_tier": caws_context_dict.get("risk_tier", 2),
                    "budget": caws_context_dict.get("budget", {}),
                    "scope": caws_context_dict.get("scope", {}),
                }
            }
            sample["evidence_manifest"] = generate_evidence_manifest()
            sample["provenance_chain"] = generate_provenance_chain()

        # Add metadata
        sample["metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "generator": "generate_caws_tool_examples",
        }

        samples.append(sample)

    # Write output
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"[generate_caws_tool_examples] Generated {len(samples)} samples")
    print(f"  Scenarios: {scenario_list}")
    print(f"  Negative examples: {negative_count}/{len(samples)}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()





