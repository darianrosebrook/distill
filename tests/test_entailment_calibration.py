
import json
import subprocess
import sys
import pathlib
import math

# Use relative paths from project root
PROJECT_ROOT = pathlib.Path(__file__).parent.parent
DATA = PROJECT_ROOT / "data" / "entailment_calibration.ndjson"
OUT = PROJECT_ROOT / "eval" / "config" / "entailment_calibration.json"
SCRIPT = PROJECT_ROOT / "scripts" / "calibrate_entailment.py"


def _tokenize(s):
    return [t for t in s.lower().split() if t.strip()]


def _overlap(a, b):
    A, B = set(_tokenize(a)), set(_tokenize(b))
    if not A:
        return 0.0
    return len(A & B) / len(A)


def raw_overlap_logits(evidence, claim):
    ov = _overlap(evidence, claim)
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
    scaled = {k: (logits[k] / T) + bias.get(k, 0.0) for k in logits.keys()}
    probs = softmax(scaled)
    smoothed = {k: 0.5*probs[k] + 0.5*priors[k] for k in probs.keys()}
    Z = sum(smoothed.values())
    return {k: smoothed[k]/Z for k in smoothed.keys()}


def nll(probs, gold_label, eps=1e-9):
    p = probs.get(gold_label, 0.0)
    p = max(min(p, 1.0 - eps), eps)
    return -math.log(p)


def load_rows(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def test_calibration_script_reduces_mean_nll():
    # run calibration script
    proc = subprocess.run([sys.executable, str(
        SCRIPT), "--in", str(DATA), "--out", str(OUT)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    payload = json.loads(open(OUT, "r").read())
    calib = payload["calibration"]
    rows = load_rows(DATA)

    # compute baseline and calibrated NLL
    base_losses, cal_losses = [], []
    for r in rows:
        logits = raw_overlap_logits(r["evidence"], r["claim"])
        base_probs = softmax(logits)
        base_losses.append(nll(base_probs, r["label"]))
        cal_probs = apply_calibration(
            logits, calib["temperature"], calib["bias"], calib["priors"])
        cal_losses.append(nll(cal_probs, r["label"]))

    base_mean = sum(base_losses)/len(base_losses)
    cal_mean = sum(cal_losses)/len(cal_losses)

    assert cal_mean <= base_mean, f"calibrated NLL {cal_mean:.4f} should be <= baseline {base_mean:.4f}"
