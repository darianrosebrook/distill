
#!/usr/bin/env python3

import argparse, json, math, itertools, sys, pathlib, statistics

def _tokenize(s):
    return [t for t in s.lower().split() if t.strip()]

def _overlap(a, b):
    A, B = set(_tokenize(a)), set(_tokenize(b))
    if not A: 
        return 0.0
    return len(A & B) / len(A)

def raw_overlap_logits(evidence, claim):
    """
    Proxy logits for triad classes:
    - support increases with token overlap
    - contradict increases when negation words appear against claim polarity
    - insufficient gets mid-boost for small overlaps
    """
    ov = _overlap(evidence, claim)
    # crude negation detection
    neg_words = {"not", "no", "never", "without", "none", "n't"}
    has_neg = any(w in _tokenize(evidence) for w in neg_words)
    support = 2.0 * ov
    contradict = 1.0 if has_neg and ov > 0.2 else 0.0
    insufficient = 0.6 if 0.05 <= ov <= 0.35 else 0.2
    return {"support": support, "contradict": contradict, "insufficient": insufficient}

def softmax(logits):
    xs = list(logits.values())
    m = max(xs)
    ex = [math.exp(x - m) for x in xs]
    Z = sum(ex)
    out = {}
    i = 0
    for k in logits.keys():
        out[k] = ex[i]/Z
        i += 1
    return out

def apply_calibration(logits, T, bias, priors):
    # Temperature scaling + additive bias + prior smoothing (mixture)
    scaled = {k: (logits[k] / T) + bias.get(k, 0.0) for k in logits.keys()}
    probs = softmax(scaled)
    # prior smoothing: average with empirical priors
    smoothed = {k: 0.5*probs[k] + 0.5*priors[k] for k in probs.keys()}
    # renormalize
    Z = sum(smoothed.values())
    return {k: smoothed[k]/Z for k in smoothed.keys()}

def nll(probs, gold_label, eps=1e-9):
    p = probs.get(gold_label, 0.0)
    p = max(min(p, 1.0 - eps), eps)
    return -math.log(p)

def load_ndjson(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def empirical_priors(rows):
    counts = {"support":0, "contradict":0, "insufficient":0}
    for r in rows:
        counts[r["label"]] += 1
    total = sum(counts.values())
    return {k: counts[k]/total for k in counts.keys()}

def grid_search(rows, T_grid, bias_vals):
    best = None
    # biases per class from bias_vals
    for T in T_grid:
        for b_s in bias_vals:
            for b_c in bias_vals:
                for b_i in bias_vals:
                    bias = {"support": b_s, "contradict": b_c, "insufficient": b_i}
                    # compute empirical priors once (fixed)
                    priors = empirical_priors(rows)
                    losses = []
                    for r in rows:
                        logits = raw_overlap_logits(r["evidence"], r["claim"])
                        probs = apply_calibration(logits, T, bias, priors)
                        losses.append(nll(probs, r["label"]))
                    mean_nll = sum(losses)/len(losses)
                    if (best is None) or (mean_nll < best["mean_nll"]):
                        best = {
                            "T": T, "bias": bias, "priors": priors,
                            "mean_nll": mean_nll,
                            "p50_nll": statistics.median(losses),
                            "p95_nll": sorted(losses)[int(0.95*len(losses))-1]
                        }
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Path to NDJSON calibration set")
    ap.add_argument("--out", dest="out", required=True, help="Path to write calibration JSON")
    args = ap.parse_args()

    rows = load_ndjson(args.inp)
    if not rows:
        print("No rows in calibration set", file=sys.stderr)
        sys.exit(2)

    # Define search grids (small for determinism and speed)
    T_grid = [round(x,2) for x in [0.6, 0.8, 1.0, 1.2, 1.5, 2.0]]
    bias_vals = [-0.5, -0.25, 0.0, 0.25, 0.5]

    best = grid_search(rows, T_grid, bias_vals)
    with open(args.out, "w") as f:
        json.dump({
            "report": {
                "rows": len(rows),
                "mean_nll": best["mean_nll"],
                "p50_nll": best["p50_nll"],
                "p95_nll": best["p95_nll"]
            },
            "calibration": {
                "temperature": best["T"],
                "bias": best["bias"],
                "priors": best["priors"],
                "labels": ["support", "contradict", "insufficient"]
            }
        }, f, indent=2)
    print(f"Wrote calibration to {args.out} (T={best['T']}, bias={best['bias']})")

if __name__ == "__main__":
    main()
