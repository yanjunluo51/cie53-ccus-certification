"""Direct measurement of the residual estimator's conditional variance,
plain vs stratified, at matched n_res.

Replicates the inner draw R times from the SAME F_{t-1} outer states and takes
the variance of T2_hat per outer path across replications.  This is the object
the manuscript's "factor of six to twenty" claim refers to; the outer standard
error of V_UB is a different, outer-dominated quantity.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "shared"))
import ccus_canonical_headline as H          # noqa: E402
from ccus_core.core.ccus_model import simulate_paths  # noqa: E402
from hybrid_ab_conservative import _eval_continuation_grid  # noqa: E402
from nestless_3d_calibrated import lsm_3d_stdlog  # noqa: E402
from sample_splitting import (SampleSplittingRNGs, simulate_inner_one_step,  # noqa: E402
                              simulate_inner_one_step_stratified)

SEED   = int(os.environ.get("RV_SEED", "1000"))
NPATH  = int(os.environ.get("RV_PATHS", "2000"))
REPS   = int(os.environ.get("RV_REPS", "400"))
TIMES  = [int(x) for x in os.environ.get("RV_TIMES", "5,12,20").split(",")]
NRES   = [int(x) for x in os.environ.get("RV_NRES", "10,50,200").split(",")]
OUT    = Path(os.environ["RV_OUTPUT"])

P0 = H.PARAMS
streams = SampleSplittingRNGs.from_seed(SEED)
_, betas_pr, stats, _, _, _ = lsm_3d_stdlog(
    P0, 30000, H.DEGREE, seed=streams.training.bit_generator.seed_seq, eps=H.EPS)
beta_hat = H._adapt_betas(betas_pr)
std = dict(zip(["mu_log_P", "sigma_log_P", "mu_log_C", "sigma_log_C"], stats))
P, C, R = simulate_paths(P0, NPATH, rng=streams.outer)

common = dict(dt=P0.dt, sigma_P=P0.sigma_P,
              mu_P_by_regime=np.asarray(P0.mu_P_by_regime),
              sigma_C=P0.sigma_C, mu_C=P0.mu_C, rho=0.0,
              Q_regime=np.asarray(P0.Q_regime))

def t2_draw(sim, rng, t, n_res):
    inner = sim(rng, P_prev=P[:, t-1], C_prev=C[:, t-1], R_prev=R[:, t-1],
                n_res=n_res, **common)
    Z = H._payoff(inner["P"], inner["C"], inner["R"], t)
    Chat = _eval_continuation_grid(inner["P"], inner["C"], inner["R"],
                                   beta_hat[t], H.BASIS_POWERS, std)
    return np.maximum(Z - Chat, 0.0).mean(axis=1)

rows = []
for t in TIMES:
    for n_res in NRES:
        var = {}
        for name, sim in (("plain", simulate_inner_one_step),
                          ("stratified", simulate_inner_one_step_stratified)):
            rng = np.random.default_rng([SEED, t, n_res, hash(name) % 9973])
            draws = np.stack([t2_draw(sim, rng, t, n_res) for _ in range(REPS)])
            var[name] = draws.var(axis=0, ddof=1)
        # average conditional variance over outer states, and the per-state ratio
        ratio_of_means = var["plain"].mean() / var["stratified"].mean()
        pos = var["stratified"] > 0
        mean_of_ratios = float(np.median(var["plain"][pos] / var["stratified"][pos]))
        rows.append(dict(t=t, n_res=n_res, reps=REPS, n_paths=NPATH,
                         var_plain=var["plain"].mean(),
                         var_strat=var["stratified"].mean(),
                         reduction_pooled=ratio_of_means,
                         reduction_median_per_state=mean_of_ratios))
        print(f"  t={t:2d} n_res={n_res:3d}: pooled {ratio_of_means:6.2f}x | "
              f"median per-state {mean_of_ratios:6.2f}x", flush=True)

f = pd.DataFrame(rows); OUT.parent.mkdir(parents=True, exist_ok=True)
f.to_csv(OUT, index=False)
print(f"\npooled reduction range: {f.reduction_pooled.min():.2f}x - {f.reduction_pooled.max():.2f}x")
print(f"CSV: {OUT}")
