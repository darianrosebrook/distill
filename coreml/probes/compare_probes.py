# coreml/probes/compare_probes.py
# @author: @darianrosebrook

import numpy as np
import typer

app = typer.Typer()


@app.command()
def main(pt: str, ml: str, rel_err_tol: float = 0.02, mse_tol: float = 1e-3):
    a = np.load(pt)
    b = np.load(ml)
    keys = sorted(set(a.files) & set(b.files))
    worst = (None, 0.0, 0.0)
    for k in keys:
        x, y = a[k].astype(np.float32), b[k].astype(np.float32)
        mse = np.mean((x - y) ** 2)
        denom = np.maximum(np.abs(y), 1e-5)
        rel = np.max(np.abs((x - y) / denom))
        print(f"{k:16s}  mse={mse:.3e}  rel_max={rel:.3e}")
        if rel > worst[1]:
            worst = (k, rel, mse)
        if rel > rel_err_tol or mse > mse_tol:
            raise SystemExit(f"FAIL {k}: rel={rel:.3e} mse={mse:.3e} > tol")
    print(f"OK. worst={worst}")


if __name__ == "__main__":
    app()
