"""
single_regime_ccus_driver.py
============================
Support routines for the documented single-regime CCUS variance-rate check.

The active consumer is `scripts/varrate_single_regime_ccus.py`. It fixes one
trained single-regime policy and measures conditional outer-sampling variance.
Both residual-nested and nested calculations use the training standardization
statistics; this module does not provide a second routing experiment.
"""
import os, sys, time, functools
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "shared"))
# allow running from repo root too
_REPL = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(_REPL, "shared"))

import copy
from nestless_3d_calibrated import (
    lsm_3d_stdlog, evaluate_lsm_policy, basis_indices, make_params)
from ccus_core.core.ccus_model import simulate_paths, exercise_payoff
from sample_splitting import (
    SampleSplittingRNGs,
    simulate_inner_one_step,
    simulate_inner_one_step_stratified,
)
from hybrid_ab_conservative import hybrid_nestless_AB
from ccus_nested_ab import canonical_nested_ab

# ---- Config (paper CCUS production protocol) ----
DEGREE   = 3
EPS      = 0.05
N_RES    = 50
SEEDS    = list(range(1000, 1010))   # 10 seeds, same as paper
N_REGIMES = 3

# N_out / N_train: paper uses 30000. Allow override via env for smoke tests.
N_OUT   = int(os.environ.get("SR_NOUT",   "30000"))
N_TRAIN = int(os.environ.get("SR_NTRAIN", "30000"))
NESTED_NIN = int(os.environ.get("SR_NESTED_NIN", "100"))
RUN_SEEDS = int(os.environ.get("SR_SEEDS", str(len(SEEDS))))
RUN_WORKERS = max(1, int(os.environ.get("SR_WORKERS", "2")))

basis_powers = np.array(basis_indices(DEGREE))
N_BASIS = len(basis_powers)


def make_single_regime_params(base):
    """Collapse regime structure to K=1: identity transition matrix (no
    transitions). All paths started at regime0=1 stay in regime 1 forever.
    Dynamics (sigma_P, mu_P_by_regime, sigma_C, mu_C, r, horizon) UNCHANGED."""
    p = copy.deepcopy(base)
    p.Q_regime = np.eye(3)   # identity: no regime transitions
    return p


def payoff_fn_for(p):
    def payoff_fn(P, C, R, t):
        orig = P.shape
        npv = exercise_payoff(p, P.reshape(-1), C.reshape(-1), R.reshape(-1))
        return np.maximum(npv, 0.0).reshape(orig)
    return payoff_fn


def adapt_betas(betas_pr):
    beta_hat = {}
    for t, regime_dict in betas_pr.items():
        per_regime = []
        for r in range(N_REGIMES):
            entry = regime_dict.get(r)
            if entry is None:
                per_regime.append(np.zeros(N_BASIS))
            else:
                beta_n, ys = entry
                per_regime.append(beta_n * ys)
        beta_hat[t] = per_regime
    return beta_hat


def train_policy(p, training_seed):
    """Train one policy for a setting and return immutable evaluation inputs.

    The pre-budget router trains once.  Pilot and production certification then
    reuse this fitted policy with independent outer/inner stream bundles.
    """
    started = time.perf_counter()
    payoff_fn = payoff_fn_for(p)
    model_params = {
        "Q_regime":       np.asarray(p.Q_regime),
        "dt":             p.dt, "r": p.r,
        "mu_P_by_regime": np.asarray(p.mu_P_by_regime),
        "mu_C":           p.mu_C, "sigma_P": p.sigma_P, "sigma_C": p.sigma_C,
    }
    train_ss = np.random.SeedSequence(int(training_seed))
    _, betas_pr, std_stats, _, _, _ = lsm_3d_stdlog(
        p, N_TRAIN, DEGREE, seed=train_ss, eps=EPS)
    beta_hat = adapt_betas(betas_pr)
    std_dict = dict(zip(["mu_log_P","sigma_log_P","mu_log_C","sigma_log_C"], std_stats))

    return {
        "betas_pr": betas_pr,
        "beta_hat": beta_hat,
        "std_stats": std_stats,
        "std_dict": std_dict,
        "model_params": model_params,
        "payoff_fn": payoff_fn,
        "training_seconds": time.perf_counter() - started,
        "training_seed": int(training_seed),
    }


def evaluate_trained_policy(
    p,
    trained,
    evaluation_seed,
    label,
    *,
    n_out=None,
    n_res=None,
    nested_nin=None,
    residual_mode="stratified",
):
    """Evaluate paired cheap and nested certificates on one fresh stream bundle."""
    n_out = N_OUT if n_out is None else int(n_out)
    n_res = N_RES if n_res is None else int(n_res)
    nested_nin = NESTED_NIN if nested_nin is None else int(nested_nin)
    if residual_mode not in {"stratified", "plain"}:
        raise ValueError("residual_mode must be 'stratified' or 'plain'")

    stream_root = np.random.SeedSequence(int(evaluation_seed))
    outer_ss, residual_ss, nested_ss = stream_root.spawn(3)
    started_outer = time.perf_counter()
    P_out, C_out, R_out = simulate_paths(
        p, n_out, rng=np.random.default_rng(outer_ss))
    V_LB, _ = evaluate_lsm_policy(
        p, P_out, C_out, R_out, trained["betas_pr"], DEGREE,
        trained["std_stats"])
    outer_lb_seconds = time.perf_counter() - started_outer
    # report regime occupancy as a sanity check on the collapse
    n_unique_regimes = len(np.unique(R_out))

    residual_simulator = (
        simulate_inner_one_step_stratified
        if residual_mode == "stratified"
        else simulate_inner_one_step
    )
    inner_sim = functools.partial(
        residual_simulator,
        dt=p.dt, sigma_P=p.sigma_P,
        mu_P_by_regime=np.asarray(p.mu_P_by_regime),
        sigma_C=p.sigma_C, mu_C=p.mu_C,
        rho=0.0, Q_regime=np.asarray(p.Q_regime))

    started_cheap = time.perf_counter()
    rng_h = np.random.default_rng(residual_ss)
    res_h = hybrid_nestless_AB(
        outer_paths={"P": P_out, "C": C_out, "R": R_out},
        inner_simulator=inner_sim, inner_rng=rng_h,
        beta_hat=trained["beta_hat"], basis_powers=basis_powers,
        standardisation_stats=trained["std_dict"],
        model_params=trained["model_params"],
        n_res=n_res, payoff_fn=trained["payoff_fn"], conservative=True)
    V_UB = res_h["V_UB"]; V_UB_se = res_h["SE"]
    gap_h = (V_UB - V_LB) / V_LB * 100.0
    cheap_seconds = time.perf_counter() - started_cheap

    started_nested = time.perf_counter()
    rng_n = np.random.default_rng(nested_ss)
    V_nest, V_nest_se = canonical_nested_ab(
        p, P_out, C_out, R_out, trained["betas_pr"], trained["std_stats"],
        nested_nin, rng_n, degree=DEGREE)
    gap_n = (V_nest - V_LB) / V_LB * 100.0
    nested_seconds = time.perf_counter() - started_nested

    # Use the statistic computed from the exact Term-1/Term-2 quantities that
    # produced this hybrid certificate.  A former second five-replication pass
    # estimated a different, smoothed rho after the certificate was complete;
    # it was redundant, expensive, and inconsistent with the manuscript's
    # same-run diagnostic definition.
    rho = float(res_h["rho_res"])

    values = np.asarray([V_LB, V_UB, V_UB_se, gap_h, V_nest, V_nest_se, gap_n, rho])
    if not np.isfinite(values).all() or V_LB <= 0.0:
        raise FloatingPointError(
            f"invalid CCUS evaluation for {label}/{evaluation_seed}: {values}")
    return dict(variant=label, seed=int(evaluation_seed), V_LB=V_LB,
                V_UB=V_UB, V_UB_se=V_UB_se,
                gap_hybrid=gap_h, V_nest=V_nest, V_nest_se=V_nest_se,
                gap_nested=gap_n, rho_res=rho, n_unique_regimes=n_unique_regimes,
                sandwich_ok=bool(V_UB >= V_LB), residual_mode=residual_mode,
                n_out=n_out, n_res=n_res, nested_nin=nested_nin,
                outer_lb_seconds=outer_lb_seconds,
                cheap_seconds=cheap_seconds, nested_seconds=nested_seconds)


def run_one(p, seed, label, residual_mode="stratified"):
    """Backward-compatible one-seed entry point using the canonical route."""
    root = np.random.SeedSequence(int(seed))
    training_ss, evaluation_ss = root.spawn(2)
    trained = train_policy(p, int(training_ss.generate_state(1, dtype=np.uint64)[0]))
    result = evaluate_trained_policy(
        p,
        trained,
        int(evaluation_ss.generate_state(1, dtype=np.uint64)[0]),
        label,
        residual_mode=residual_mode,
    )
    result["training_seconds"] = trained["training_seconds"]
    result["seed"] = int(seed)
    return result


def timed_run_one(p, seed, label):
    t0 = time.perf_counter()
    result = run_one(p, seed, label)
    return result, time.perf_counter() - t0


def main():
    base = make_params()
    single = make_single_regime_params(base)
    print("="*78)
    print("SINGLE-REGIME CCUS ROBUSTNESS COMPARISON")
    print(f"  calibration: sigma_P={base.sigma_P}, mu_P_by_regime={base.mu_P_by_regime} "
          f"(risk-neutral: all == r={base.r})")
    print(f"  protocol: eps={EPS}, n_res={N_RES}, degree={DEGREE}, "
          f"N_train={N_TRAIN}, N_out={N_OUT}, nested_Nin={NESTED_NIN}")
    print(f"  seeds: {SEEDS[:RUN_SEEDS]}")
    print(f"  seed workers: {RUN_WORKERS}")
    print(f"  MULTI Q_regime row0 = {np.asarray(base.Q_regime)[0]}")
    print(f"  SINGLE Q_regime = identity (no transitions)")
    print("="*78)

    rows = []
    for label, p in [("multi", base), ("single", single)]:
        print(f"\n--- {label.upper()}-REGIME ---")
        seeds = SEEDS[:RUN_SEEDS]
        if RUN_WORKERS > 1:
            with ThreadPoolExecutor(max_workers=min(RUN_WORKERS, len(seeds))) as pool:
                completed = list(pool.map(lambda seed: timed_run_one(p, seed, label), seeds))
        else:
            completed = [timed_run_one(p, seed, label) for seed in seeds]
        for r, elapsed in completed:
            rows.append(r)
            print(f"  seed={r['seed']}: V_LB={r['V_LB']:.4e} V_UB={r['V_UB']:.4e} "
                  f"gap_h={r['gap_hybrid']:.2f}% gap_n={r['gap_nested']:.2f}% "
                  f"rho={r['rho_res']:.4f} nreg={r['n_unique_regimes']} "
                  f"sand_ok={r['sandwich_ok']} ({elapsed:.1f}s)")
            sys.stdout.flush()

    df = pd.DataFrame(rows)
    outdir = os.path.join(HERE, "results")
    os.makedirs(outdir, exist_ok=True)
    out_csv = os.environ.get("SR_OUTPUT", os.path.join(outdir, "single_regime_ccus.csv"))
    df.to_csv(out_csv, index=False)

    print("\n" + "="*78)
    print("SUMMARY (mean +- SE over seeds)")
    print("="*78)
    se = lambda x: x.std(ddof=1)/np.sqrt(len(x)) if len(x) > 1 else 0.0
    hdr = f"{'variant':>8} {'V_LB(1e9)':>12} {'V_UB(1e9)':>12} {'gap_hyb%':>14} {'gap_nest%':>14} {'rho_res':>14}"
    print(hdr)
    summ = {}
    for label in ["multi", "single"]:
        g = df[df.variant == label]
        summ[label] = dict(
            V_LB=g.V_LB.mean(), V_LB_se=se(g.V_LB),
            V_UB=g.V_UB.mean(), V_UB_se=se(g.V_UB),
            gap_h=g.gap_hybrid.mean(), gap_h_se=se(g.gap_hybrid),
            gap_n=g.gap_nested.mean(), gap_n_se=se(g.gap_nested),
            rho=g.rho_res.mean(), rho_se=se(g.rho_res),
            sand_all=bool(g.sandwich_ok.all()))
        s = summ[label]
        print(f"{label:>8} {s['V_LB']/1e9:>7.4f}      {s['V_UB']/1e9:>7.4f}      "
              f"{s['gap_h']:>6.2f}+-{s['gap_h_se']:<5.2f} "
              f"{s['gap_n']:>6.2f}+-{s['gap_n_se']:<5.2f} "
              f"{s['rho']:>6.4f}+-{s['rho_se']:<5.4f}")

    print("\nRELATIVE CHANGE (single vs multi):")
    for key, name in [("rho","rho_res"),("gap_h","hybrid gap"),
                      ("gap_n","nested gap"),("V_LB","V_LB"),("V_UB","V_UB")]:
        m = summ["multi"][key]; s = summ["single"][key]
        rel = (s - m)/abs(m)*100 if abs(m) > 1e-12 else float('nan')
        print(f"  {name:>12}: multi={m:.4g}  single={s:.4g}  rel_change={rel:+.1f}%")

    print(f"\n  Sandwich valid (V_UB>=V_LB) all seeds: "
          f"multi={summ['multi']['sand_all']} single={summ['single']['sand_all']}")
    print(f"\nCSV: {out_csv}")


if __name__ == "__main__":
    main()
