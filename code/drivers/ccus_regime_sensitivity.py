"""Regime-transition-intensity sensitivity on the canonical comparator.

Scales the off-diagonal (switching) mass of Q_regime by a factor and
renormalises the diagonal, leaving every other calibration untouched.  All
certification machinery is imported unchanged from the canonical headline
driver, so the scale=1.0 row is a negative control: it must reproduce the
canonical headline gaps exactly.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ccus_canonical_headline as H   # noqa: E402

SCALES = [float(x) for x in os.environ.get("REG_SCALES", "1.0,0.8,1.2").split(",")]
OUT = Path(os.environ["REG_OUTPUT"])
METHODS = tuple(os.environ.get("REG_METHODS", "plain50,nested100").split(","))


def scaled_Q(Q, s):
    Q = np.asarray(Q, dtype=float).copy()
    out = np.empty_like(Q)
    for i in range(Q.shape[0]):
        off = Q[i].copy(); off[i] = 0.0
        off = off * s
        tot = off.sum()
        if tot >= 1.0:
            raise ValueError(f"row {i}: off-diagonal mass {tot} >= 1")
        out[i] = off
        out[i, i] = 1.0 - tot
    return out


rows = []
base_Q = np.asarray(H.PARAMS.Q_regime, dtype=float).copy()
H.METHODS = METHODS
for s in SCALES:
    H.PARAMS.Q_regime = scaled_Q(base_Q, s)
    print(f"--- scale {s}: off-diagonal row sums "
          f"{[round(float(H.PARAMS.Q_regime[i].sum()-H.PARAMS.Q_regime[i,i]),4) for i in range(3)]}",
          flush=True)
    for seed in range(1000, 1000 + int(os.environ.get("REG_SEEDS", "10"))):
        for r in H.run_seed(seed):
            r["q_scale"] = s
            rows.append(r)
H.PARAMS.Q_regime = base_Q

f = pd.DataFrame(rows)
OUT.parent.mkdir(parents=True, exist_ok=True)
f.to_csv(OUT, index=False)
g = f.groupby(["q_scale", "method"]).agg(
    gap=("gap_pct", "mean"),
    se=("gap_pct", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
    rho=("rho_res", "mean"),
    ok=("sandwich_ok", "all"))
print(g.round(4).to_string())
print(f"CSV: {OUT}")
