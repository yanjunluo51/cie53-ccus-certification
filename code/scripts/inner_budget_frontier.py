"""Section 7: what does more inner simulation buy?

Fits the nested gap against N_in^{-1/2} and reports the intercept, i.e. the
gap that survives when the inner budget is extrapolated away.  Subsets are
refit so the reader can see how far the intercept moves.
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

CSV = Path(os.environ.get("FRONTIER_INPUT",
           Path(__file__).resolve().parents[2] / "results" / "nested_inner_sweep_canonical.csv"))
g = pd.read_csv(CSV).groupby("N_in").gap_pct.mean()
print(g.round(3).to_string())
print()
for label, idx in [("all", list(g.index)), ("drop 25", [i for i in g.index if i >= 50]),
                   ("drop 25,50", [i for i in g.index if i >= 100]),
                   ("largest two", sorted(g.index)[-2:])]:
    x = 1 / np.sqrt(np.asarray(idx, float)); y = g.loc[idx].values
    sl, ic, r, _, _ = stats.linregress(x, y)
    print(f"  {label:12s} intercept {ic:6.2f}%   R^2 {r**2:.4f}")
