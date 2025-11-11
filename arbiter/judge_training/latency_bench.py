# arbiter/judge_training/latency_bench.py
# Latency benchmarking for CoreML judge
# @author: @darianrosebrook

import time
import statistics as stats
from .runtime import CoreMLJudge

DEF_PROMPT = "Summarize the changes and justify under CAWS Section 5.2."

A = "Adds tests and cites issue #123. Budget within limits."

B = "Refactors naming; no tests nor evidence."


def bench(judge: CoreMLJudge, iters=100):
    times = []
    for _ in range(iters):
        t0 = time.time()
        judge.compare(DEF_PROMPT, A, B)
        times.append((time.time() - t0) * 1000)
    return {
        "iters": iters,
        "p50_ms": round(stats.median(times), 2),
        "p95_ms": round(stats.quantiles(times, n=20)[18], 2),
        "mean_ms": round(sum(times) / len(times), 2)
    }


if __name__ == "__main__":
    # Edit paths as needed
    j = CoreMLJudge("arbiter/judge_training/artifacts/coreml/judge.mlpackage", "microsoft/deberta-v3-small", [
        "EVIDENCE_COMPLETENESS", "BUDGET_ADHERENCE", "GATE_INTEGRITY", "PROVENANCE_CLARITY", "WAIVER_JUSTIFICATION"
    ])
    print(bench(j))

