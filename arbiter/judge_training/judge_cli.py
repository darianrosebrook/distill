# arbiter/judge_training/judge_cli.py
# CLI tool for judge inference
# @author: @darianrosebrook

import argparse
import json
from .runtime import CoreMLJudge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mlpackage")
    ap.add_argument("hf_name")
    ap.add_argument("prompt")
    ap.add_argument("candidate_a")
    ap.add_argument("candidate_b")
    args = ap.parse_args()
    j = CoreMLJudge(args.mlpackage, args.hf_name, [
        "EVIDENCE_COMPLETENESS", "BUDGET_ADHERENCE", "GATE_INTEGRITY", "PROVENANCE_CLARITY", "WAIVER_JUSTIFICATION"
    ])
    out = j.compare(args.prompt, args.candidate_a, args.candidate_b)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

