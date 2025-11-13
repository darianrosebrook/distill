# arbiter/judge_training/build_pairs_from_caws.py
# Transform CAWS adjudication logs -> pairwise judge training lines
# @author: @darianrosebrook

"""
Transform CAWS adjudication logs -> pairwise judge training lines.

Input JSONL rows (minimal):

{
  "id": "job-123",
  "prompt": "...",
  "candidates": [
      {"text": "...", "clauses": ["EVIDENCE_COMPLETENESS"], "score": 0.78},
      {"text": "...", "clauses": ["BUDGET_ADHERENCE"], "score": 0.55}
  ],
  "winner_index": 0
}
"""

import json
import sys


def main(inp, out):
    with open(inp, "r", encoding="utf-8") as f, open(out, "w", encoding="utf-8") as g:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            c = r["candidates"]
            if len(c) < 2:
                continue
            a, b = c[0], c[1]
            winner = "a" if r.get("winner_index", 0) == 0 else "b"
            row = {
                "id": r.get("id"),
                "prompt": r["prompt"],
                "a": {"text": a["text"], "clauses": a.get("clauses", [])},
                "b": {"text": b["text"], "clauses": b.get("clauses", [])},
                "winner": winner,
            }
            g.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
