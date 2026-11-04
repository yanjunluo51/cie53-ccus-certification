"""Section 4.5: is the first-moment residual share monotone in the variance
ratio it stands in for?

For each review date the inner draw is replicated R times from the same
F_{t-1} outer states.  rho_res uses the manuscript's Eq. (3) convention,
|T2| / (|T1| + |T2|); rho_var is Var(T2) / Var(T1 + T2).  Both are averaged
over outer states.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "shared"))
import ccus_canonical_headline as H                       # noqa: E402
from ccus_core.core.ccus_model import simulate_paths      # noqa: E402
from hybrid_ab_conservative import _eval_continuation_grid  # noqa: E402
from nestless_3d_calibrated import lsm_3d_stdlog          # noqa: E402
from sample_splitting import SampleSplittingRNGs, simulate_inner_one_step  # noqa: E402

SEED  = int(os.environ.get("RR_SEED", "1000"))
PATHS = int(os.environ.get("RR_PATHS", "2000"))
REPS  = int(os.environ.get("RR_REPS", "200"))
NRES  = int(os.environ.get("RR_NRES", "50"))
TIMES = [int(x) for x in os.environ.get("RR_TIMES", "2,5,8,11,14,17,20,23").split(",")]
OUT   = Path(os.environ["RR_OUTPUT"])

p = H.PARAMS
st = SampleSplittingRNGs.from_seed(SEED)
_, bp, stats_, _, _, _ = lsm_3d_stdlog(
    p, 30000, H.DEGREE, seed=st.training.bit_generator.seed_seq, eps=H.EPS)
bh = H._adapt_betas(bp)
std = dict(zip(["mu_log_P", "sigma_log_P", "mu_log_C", "sigma_log_C"], stats_))
P, C, Rg = simulate_paths(p, PATHS, rng=st.outer)
common = dict(dt=p.dt, sigma_P=p.sigma_P,
              mu_P_by_regime=np.asarray(p.mu_P_by_regime),
              sigma_C=p.sigma_C, mu_C=p.mu_C, rho=0.0,
              Q_regime=np.asarray(p.Q_regime))

rows = []
for t in TIMES:
    rng = np.random.default_rng([4000, t])
    T1, T2 = [], []
    for _ in range(REPS):
        inn = simulate_inner_one_step(rng, P_prev=P[:, t-1], C_prev=C[:, t-1],
                                      R_prev=Rg[:, t-1], n_res=NRES, **common)
        Z = H._payoff(inn["P"], inn["C"], inn["R"], t)
        Ch = _eval_continuation_grid(inn["P"], inn["C"], inn["R"],
                                     bh[t], H.BASIS_POWERS, std)
        T1.append(Ch.mean(axis=1))
        T2.append(np.maximum(Z - Ch, 0.0).mean(axis=1))
    T1 = np.stack(T1); T2 = np.stack(T2); Q = T1 + T2
    a1 = np.abs(T1.mean(axis=0)).mean(); a2 = np.abs(T2.mean(axis=0)).mean()
    rows.append(dict(t=t, rho_res_eq3=a2 / (a1 + a2),
                     rho_var=T2.var(axis=0, ddof=1).mean() / Q.var(axis=0, ddof=1).mean()))
    print(f"  t={t:2d}  rho_res={rows[-1]['rho_res_eq3']:.4f}  "
          f"rho_var={rows[-1]['rho_var']:.4f}", flush=True)

f = pd.DataFrame(rows); OUT.parent.mkdir(parents=True, exist_ok=True)
f.to_csv(OUT, index=False)
print(f"\nSpearman = {stats.spearmanr(f.rho_res_eq3, f.rho_var)[0]:.3f}  "
      f"(rho_var exceeds one at {int((f.rho_var > 1).sum())} of {len(f)} dates)")
print(f"CSV: {OUT}")
