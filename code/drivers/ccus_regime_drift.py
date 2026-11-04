"""Does a regime-dependent carbon-price drift change the residual share or the route?

The calibrated instance sets mu_P^(r) = r in every regime, so the regime is
valuation-neutral there.  This driver spreads the drift across regimes while
holding the mean at r, and re-measures the diagnostic and both routes on the
canonical comparator.  The (r,r,r) row is the negative control.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ccus_canonical_headline as H  # noqa: E402

OUT = Path(os.environ["DRIFT_OUTPUT"])
H.METHODS = tuple(os.environ.get("DRIFT_METHODS", "plain50,strat50,nested100").split(","))
SEEDS = int(os.environ.get("DRIFT_SEEDS", "10"))
GRIDS = {
    "(r,r,r) baseline":      (0.05, 0.05, 0.05),
    "spread +/-0.02":        (0.03, 0.05, 0.07),
    "spread +/-0.04":        (0.01, 0.05, 0.09),
}
base = tuple(H.PARAMS.mu_P_by_regime)
rows = []
for label, mu in GRIDS.items():
    H.PARAMS.mu_P_by_regime = mu
    print(f"--- {label}: mu_P = {mu}", flush=True)
    for seed in range(1000, 1000 + SEEDS):
        for r in H.run_seed(seed):
            r["drift_case"] = label; r["mu_P"] = str(mu)
            rows.append(r)
H.PARAMS.mu_P_by_regime = base

f = pd.DataFrame(rows); OUT.parent.mkdir(parents=True, exist_ok=True)
f.to_csv(OUT, index=False)
g = f.groupby(["drift_case", "method"]).agg(
    gap=("gap_pct", "mean"),
    se=("gap_pct", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
    rho=("rho_res", "mean"),
    VLB=("V_LB", "mean"),
    ok=("sandwich_ok", "all"))
print(g.round(4).to_string())
print(f"CSV: {OUT}")
