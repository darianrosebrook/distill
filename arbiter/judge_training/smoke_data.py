# arbiter/judge_training/smoke_data.py
# Generate smoke test data for judge training
# @author: @darianrosebrook

import json
import os


def main(out_train="data/judge/train.jsonl", out_val="data/judge/val.jsonl"):
    os.makedirs(os.path.dirname(out_train), exist_ok=True)
    train = [
        {
            "id": "t1",
            "prompt": "Add input validation to function foo().",
            "a": {
                "text": "Adds type checks and unit tests.",
                "clauses": ["EVIDENCE_COMPLETENESS", "GATE_INTEGRITY"],
            },
            "b": {"text": "Refactors naming only.", "clauses": ["BUDGET_ADHERENCE"]},
            "winner": "a",
        },
        {
            "id": "t2",
            "prompt": "Summarize RFC-123 in 3 bullets.",
            "a": {
                "text": "Three precise bullets with citations.",
                "clauses": ["PROVENANCE_CLARITY", "EVIDENCE_COMPLETENESS"],
            },
            "b": {"text": "Vague summary without sources.", "clauses": ["WAIVER_JUSTIFICATION"]},
            "winner": "a",
        },
        {
            "id": "t3",
            "prompt": "Produce JSON tool call for create_user(name,email).",
            "a": {
                "text": '{"tool":"create_user","args":{"name":"A","email":"a@x.com"}}',
                "clauses": ["GATE_INTEGRITY"],
            },
            "b": {"text": "{name:'A'}", "clauses": ["WAIVER_JUSTIFICATION"]},
            "winner": "a",
        },
    ]
    val = [
        {
            "id": "v1",
            "prompt": "Fix off-by-one in loop.",
            "a": {
                "text": "Corrects index and adds test.",
                "clauses": ["GATE_INTEGRITY", "EVIDENCE_COMPLETENESS"],
            },
            "b": {"text": "Renames variable only.", "clauses": ["BUDGET_ADHERENCE"]},
            "winner": "a",
        }
    ]
    for path, rows in [(out_train, train), (out_val, val)]:
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", out_train, out_val)


if __name__ == "__main__":
    main()
