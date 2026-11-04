"""
exp_mlmc_maxcall.py
===================
Lightweight MLMC-AB comparison on the 5D Bermudan max-call benchmark.

The experiment is deliberately narrow. It uses the same LSM policy,
outer paths, payoff, and polynomial basis as exp_bbs_comparison.py and
adds one AB-family accelerator:

  E[Y_100] = E[Y_25] + E[Y_50 - Y_25] + E[Y_100 - Y_50],

where Y_n is the nested-AB pathwise dual payoff computed with n inner
samples per exercise date. The level differences are coupled by reusing
the first n_coarse inner samples inside the n_fine estimator. Outer
allocations are (50000, 25000, 12500) for the three terms, giving a
small MLMC baseline with the same fine inner budget as nested_100.

Output:
  results/exp_mlmc_maxcall.csv
"""

import os
import time

import numpy as np
import pandas as pd

from exp_bbs_comparison import (
    D,
    S0,
    K,
    R,
    DELTA,
    SIGMA,
    T_YEARS,
    N_STEPS,
    DT,
    DEGREE_LSM,
    N_TRAIN,
    N_OUT,
    N_RES,
    simulate_gbm_paths,
    maxcall_payoff,
    compute_std_stats,
    basis_stdlog_nd,
    lsm_train,
    hybrid_nestless,
    nested_ab,
)


N_SEEDS = 5
SEED0 = 42
SEED_STEP = 1000
MLMC_LEVELS = [(25, 50000), (50, 25000), (100, 12500)]


def out_of_sample_lb(S_out, betas, means, stds):
    """Evaluate the trained LSM stopping policy on independent paths."""
    disc = np.exp(-R * DT)
    value = maxcall_payoff(S_out[:, N_STEPS, :], K)
    for t in range(N_STEPS - 1, 0, -1):
        exercise = maxcall_payoff(S_out[:, t, :], K)
        if betas[t] is not None:
            phi, _ = basis_stdlog_nd(S_out[:, t, :], DEGREE_LSM, means, stds)
            cont = phi @ betas[t]
            stop = exercise > cont
        else:
            stop = np.zeros(len(S_out), dtype=bool)
        value = np.where(stop, exercise, disc * value)
    # Discounting fix: the backward loop stops at t=1, so this value is priced at time 1
    # and needs one more discount step to be a time-0 price. The upper bound is discounted
    # to time 0 with exp(-R*arange(T+1)*DT). Mixing the two would shrink the gap systematically.
    value = disc * value
    return float(value.mean())


def make_z_q(S_out, betas, means, stds):
    n, t1, _ = S_out.shape
    T = t1 - 1
    Z_all = np.zeros((n, T + 1))
    Q = np.zeros((n, T + 1))
    for t in range(T + 1):
        Z_all[:, t] = maxcall_payoff(S_out[:, t, :], K)
        if 0 < t < T and betas[t] is not None:
            phi, _ = basis_stdlog_nd(S_out[:, t, :], DEGREE_LSM, means, stds)
            cont = phi @ betas[t]
        else:
            cont = np.zeros(n)
        Q[:, t] = np.maximum(Z_all[:, t], cont)
    return Z_all, Q


def nested_pathwise_pair(S_out, betas, means, stds, n_fine, rng, n_coarse=None):
    """Return coupled nested-AB pathwise values for fine and optional coarse levels."""
    n, t1, d = S_out.shape
    T = t1 - 1
    disc = np.exp(-R * np.arange(T + 1) * DT)
    Z_all, Q = make_z_q(S_out, betas, means, stds)

    M_fine = np.zeros((n, T + 1))
    M_coarse = np.zeros((n, T + 1)) if n_coarse is not None else None

    drift = (R - DELTA - 0.5 * SIGMA**2) * DT
    vol = SIGMA * np.sqrt(DT)

    for t in range(1, T + 1):
        e_fine = np.zeros(n)
        e_coarse = np.zeros(n) if n_coarse is not None else None

        for j in range(n_fine):
            z = rng.standard_normal((n, d))
            s_in = S_out[:, t - 1, :] * np.exp(drift + vol * z)
            z_pay = maxcall_payoff(s_in, K)
            if t < T and betas[t] is not None:
                phi_in, _ = basis_stdlog_nd(s_in, DEGREE_LSM, means, stds)
                c_in = phi_in @ betas[t]
            else:
                c_in = np.zeros(n)
            q_in = np.maximum(z_pay, c_in)
            e_fine += q_in
            if n_coarse is not None and j < n_coarse:
                e_coarse += q_in

        e_fine /= n_fine
        dM_fine = disc[t] * Q[:, t] - disc[t] * e_fine
        M_fine[:, t] = M_fine[:, t - 1] + dM_fine

        if n_coarse is not None:
            e_coarse /= n_coarse
            dM_coarse = disc[t] * Q[:, t] - disc[t] * e_coarse
            M_coarse[:, t] = M_coarse[:, t - 1] + dM_coarse

    disc_Z = disc[None, :] * Z_all
    y_fine = np.max(disc_Z - M_fine, axis=1)
    if n_coarse is None:
        return y_fine, None
    y_coarse = np.max(disc_Z - M_coarse, axis=1)
    return y_fine, y_coarse


def mlmc_ab(S_out, betas, means, stds, rng):
    """Three-level MLMC estimator of the nested-AB fine target Y_100."""
    terms = []
    detail_rows = []

    n0, N0 = MLMC_LEVELS[0]
    y0, _ = nested_pathwise_pair(S_out[:N0], betas, means, stds, n0, rng)
    terms.append(y0)
    detail_rows.append(("level_25", N0, n0, float(y0.mean()), float(y0.var(ddof=1))))

    prev_n = n0
    for n_fine, n_outer in MLMC_LEVELS[1:]:
        y_fine, y_coarse = nested_pathwise_pair(
            S_out[:n_outer], betas, means, stds, n_fine, rng, n_coarse=prev_n
        )
        diff = y_fine - y_coarse
        terms.append(diff)
        detail_rows.append((
            f"diff_{n_fine}_{prev_n}",
            n_outer,
            n_fine,
            float(diff.mean()),
            float(diff.var(ddof=1)),
        ))
        prev_n = n_fine

    estimate = float(sum(term.mean() for term in terms))
    variance = float(sum(term.var(ddof=1) / len(term) for term in terms))
    se = float(np.sqrt(variance))
    return estimate, se, detail_rows


def run_one_seed(seed):
    rng_train = np.random.default_rng(seed)
    rng_out = np.random.default_rng(seed + 500)

    S_train = simulate_gbm_paths(
        N_TRAIN, D, N_STEPS, S0, R, DELTA, SIGMA, DT, rng_train
    )[0]
    means, stds = compute_std_stats(S_train)
    betas, indices, _ = lsm_train(S_train, DEGREE_LSM, means, stds)

    S_out = simulate_gbm_paths(
        N_OUT, D, N_STEPS, S0, R, DELTA, SIGMA, DT, rng_out
    )[0]
    V_LB = out_of_sample_lb(S_out, betas, means, stds)

    rows = []

    rng_hybrid = np.random.default_rng(seed + 1000)
    tic = time.perf_counter()
    V_hybrid, se_hybrid = hybrid_nestless(
        S_out, betas, indices, DEGREE_LSM, means, stds, N_RES, rng_hybrid
    )
    wall = time.perf_counter() - tic
    rows.append(dict(
        seed=seed,
        method="Hybrid",
        V_LB=V_LB,
        V_UB=V_hybrid,
        SE=se_hybrid,
        gap_pct=(V_hybrid - V_LB) / V_LB * 100.0,
        wall_s=wall,
    ))

    rng_nested = np.random.default_rng(seed + 2000)
    tic = time.perf_counter()
    V_nested, se_nested = nested_ab(
        S_out, betas, indices, DEGREE_LSM, means, stds, 100, rng_nested
    )
    wall = time.perf_counter() - tic
    rows.append(dict(
        seed=seed,
        method="Nested AB",
        V_LB=V_LB,
        V_UB=V_nested,
        SE=se_nested,
        gap_pct=(V_nested - V_LB) / V_LB * 100.0,
        wall_s=wall,
    ))

    rng_mlmc = np.random.default_rng(seed + 4000)
    tic = time.perf_counter()
    V_mlmc, se_mlmc, details = mlmc_ab(S_out, betas, means, stds, rng_mlmc)
    wall = time.perf_counter() - tic
    rows.append(dict(
        seed=seed,
        method="MLMC-AB",
        V_LB=V_LB,
        V_UB=V_mlmc,
        SE=se_mlmc,
        gap_pct=(V_mlmc - V_LB) / V_LB * 100.0,
        wall_s=wall,
    ))

    detail_rows = [
        dict(seed=seed, term=name, n_outer=n_outer, n_inner=n_inner,
             mean=mean, variance=var)
        for name, n_outer, n_inner, mean, var in details
    ]
    return rows, detail_rows


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    all_rows = []
    all_details = []
    print("5D max-call MLMC-AB baseline")
    print(f"N_train={N_TRAIN}, N_out={N_OUT}, degree={DEGREE_LSM}")
    print(f"MLMC levels={MLMC_LEVELS}")
    for i in range(N_SEEDS):
        seed = SEED0 + SEED_STEP * i
        rows, details = run_one_seed(seed)
        all_rows.extend(rows)
        all_details.extend(details)
        compact = ", ".join(
            f"{r['method']}: gap={r['gap_pct']:.2f}%, wall={r['wall_s']:.2f}s"
            for r in rows
        )
        print(f"seed={seed}: {compact}")

    df = pd.DataFrame(all_rows)
    detail_df = pd.DataFrame(all_details)
    out_csv = os.path.join(results_dir, "exp_mlmc_maxcall.csv")
    detail_csv = os.path.join(results_dir, "exp_mlmc_maxcall_terms.csv")
    # Create the output directory before writing (a no-op when it already exists).
    os.makedirs(results_dir, exist_ok=True)
    df.to_csv(out_csv, index=False)
    detail_df.to_csv(detail_csv, index=False)

    summary = df.groupby("method").agg(
        V_UB_mean=("V_UB", "mean"),
        V_UB_se_seed=("V_UB", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        gap_mean=("gap_pct", "mean"),
        gap_se_seed=("gap_pct", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        wall_mean=("wall_s", "mean"),
        wall_sd=("wall_s", "std"),
    )
    print("\nSummary")
    print(summary)
    print(f"\nWrote {out_csv}")
    print(f"Wrote {detail_csv}")


if __name__ == "__main__":
    main()
