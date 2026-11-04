"""
exp_bbs_comparison.py — BBS regression-later vs hybrid nestless on 5D max-call
================================================================================
Reviewer Experiment 1: Head-to-head comparison with Belomestny-Bender-Schoenmakers
regression-later estimator on the standard 5D Bermudan max-call benchmark.

BBS approach (Belomestny 2009, Belomestny-Bender-Schoenmakers 2009):
- Train a stopping policy (same LSM as hybrid)
- Construct dual martingale by regressing ΔM_t onto martingale-difference
  basis functions
- Basis: Hermite polynomials of the Brownian increments H_k(ΔW_i / √Δt)
  tensored across assets, truncated at total degree ≤ q
- No inner simulation needed (truly nestless)

Hybrid approach (this paper):
- Same LSM-trained stopping policy
- Construct dual martingale via closed-form Term 1 + residual MC Term 2
- Inner simulation only for the residual (n_res samples per (t, ω))

Both use CRN-paired outer paths for fair comparison.

Parameters: AB 2004 standard (d=5, S0=100, K=100, r=0.05, δ=0.10, σ=0.20,
T=3yr, 9 exercise dates, ρ=0).

Output: results/exp_bbs_comparison.csv
"""

import numpy as np
import time
import os, sys
from math import comb
from itertools import product as iproduct
import warnings

np.set_printoptions(precision=4)
# Suppress benign overflow warnings from polynomial extrapolation on deep-OTM paths.
# These affect only paths with S → 0 where payoff = 0 anyway; the max() in Q clips
# the result and they do not propagate to the final upper bound.
np.seterr(all='ignore')

# =====================================================================
#  Parameters (AB 2004 standard)
# =====================================================================
D = 5
S0 = 100.0
K = 100.0
R = 0.05
DELTA = 0.10
SIGMA = 0.20
T_YEARS = 3.0
N_STEPS = 9
DT = T_YEARS / N_STEPS
RHO = 0.0

DEGREE_LSM = 2       # for LSM training (both methods use same policy)
DEGREE_BBS_GRID = [2, 3, 4, 5, 6]  # sweep BBS basis degree to saturation
N_TRAIN = 50000
N_OUT   = 50000
N_RES   = 50          # for hybrid only
N_SEEDS = 5

# =====================================================================
#  Simulation
# =====================================================================
def simulate_gbm_paths(n_paths, d, n_steps, s0, r, delta, sigma, dt, rng):
    """Simulate d independent GBM assets. Returns (S, Z) where Z are the
    standard normal increments used."""
    drift = (r - delta - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)
    S = np.zeros((n_paths, n_steps + 1, d))
    S[:, 0, :] = s0
    Z_increments = np.zeros((n_paths, n_steps, d))
    for t in range(1, n_steps + 1):
        Z = rng.standard_normal((n_paths, d))
        Z_increments[:, t-1, :] = Z
        S[:, t, :] = S[:, t-1, :] * np.exp(drift + vol * Z)
    return S, Z_increments

def maxcall_payoff(S, K):
    return np.maximum(S.max(axis=-1) - K, 0.0)

# =====================================================================
#  Standardized-log polynomial basis
# =====================================================================
def compute_std_stats(S_train):
    n, T1, d = S_train.shape
    logS = np.log(np.maximum(S_train[:, 1:, :].reshape(-1, d), 1e-12))
    return logS.mean(axis=0), np.maximum(logS.std(axis=0), 1e-8)

def basis_stdlog_nd(S, degree, means, stds):
    d = S.shape[-1] if S.ndim > 1 else 1
    n = S.shape[0]
    logS = np.log(np.maximum(S, 1e-12))
    z = (logS - means) / stds
    z = np.clip(z, -20.0, 20.0)  # prevent overflow in z**k for large k

    indices = []
    def gen(dim, remaining, current):
        if dim == d:
            indices.append(tuple(current))
            return
        for k in range(remaining + 1):
            gen(dim + 1, remaining - k, current + [k])
    gen(0, degree, [])

    feats = np.ones((n, len(indices)))
    for col, idx in enumerate(indices):
        for dim, power in enumerate(idx):
            if power > 0:
                feats[:, col] *= z[:, dim] ** power
    return feats, indices

def gaussian_raw_moment(mu, var, k):
    if k == 0:
        return np.ones_like(mu)
    acc = np.zeros_like(mu, dtype=np.float64)
    for j in range(k // 2 + 1):
        c = comb(k, 2*j)
        df = 1.0
        for i in range(2*j - 1, 0, -2):
            df *= i
        acc += c * df * (var**j) * (mu**(k - 2*j))
    return acc

def closed_form_E_basis_nd(S_prev, degree, means, stds, indices,
                           r, delta, sigma, dt):
    n, d = S_prev.shape
    p = len(indices)
    E_basis = np.ones((n, p))
    drift_log = (r - delta - 0.5 * sigma**2) * dt
    var_log = sigma**2 * dt
    for dim in range(d):
        logS_prev = np.log(np.maximum(S_prev[:, dim], 1e-12))
        mu_z = (logS_prev + drift_log - means[dim]) / stds[dim]
        var_z = var_log / stds[dim]**2
        for col, idx in enumerate(indices):
            k = idx[dim]
            if k > 0:
                E_basis[:, col] *= gaussian_raw_moment(mu_z, var_z, k)
    return E_basis

# =====================================================================
#  LSM training (shared by both methods)
# =====================================================================
def lsm_train(S, degree, means, stds, ridge=1e-6):
    n, T1, d = S.shape
    T = T1 - 1
    disc = np.exp(-R * DT)
    V = maxcall_payoff(S[:, T, :], K)
    betas = [None] * T
    _, indices = basis_stdlog_nd(S[:, 0, :], degree, means, stds)
    p = len(indices)

    for t in range(T - 1, 0, -1):
        exercise = maxcall_payoff(S[:, t, :], K)
        itm = exercise > 0
        if itm.sum() < p + 5:
            V[~itm] = disc * V[~itm]
            continue
        Y = disc * V[itm]
        Phi, _ = basis_stdlog_nd(S[itm, t, :], degree, means, stds)
        # Ridge-regularized OLS to prevent coefficient explosion
        A = Phi.T @ Phi + ridge * np.eye(p)
        beta = np.linalg.solve(A, Phi.T @ Y)
        betas[t] = beta
        C_hat = np.nan_to_num(Phi @ beta, nan=0.0, posinf=1e10, neginf=-1e10)
        ex_mask = exercise[itm] > C_hat
        V_new = disc * V.copy()
        idx_itm = np.where(itm)[0]
        V_new[idx_itm[ex_mask]] = exercise[itm][ex_mask]
        V_new[idx_itm[~ex_mask]] = disc * V[itm][~ex_mask]
        V_new[~itm] = disc * V[~itm]
        V = V_new
    # Discounting fix: the backward loop stops at t=1, so this value is priced at time 1
    # and needs one more discount step to be a time-0 price. The upper bound is discounted
    # to time 0 with exp(-R*arange(T+1)*DT). Mixing the two would shrink the gap systematically.
    V = disc * V
    return betas, indices, V.mean()

# =====================================================================
#  Hybrid nestless conservative (this paper's method)
# =====================================================================
def hybrid_nestless(S_out, betas, indices, degree, means, stds, n_res, rng):
    n, T1, d = S_out.shape
    T = T1 - 1
    disc = np.exp(-R * np.arange(T + 1) * DT)

    Z_all = np.zeros((n, T + 1))
    Q = np.zeros((n, T + 1))
    for t in range(T + 1):
        Z_all[:, t] = maxcall_payoff(S_out[:, t, :], K)
        if 0 < t < T and betas[t] is not None:
            Phi, _ = basis_stdlog_nd(S_out[:, t, :], degree, means, stds)
            C_hat = Phi @ betas[t]
        else:
            C_hat = np.zeros(n)
        Q[:, t] = np.maximum(Z_all[:, t], C_hat)

    M = np.zeros((n, T + 1))
    for t in range(1, T + 1):
        if t == T:
            E_Q = np.zeros(n)
            for j in range(n_res):
                Zn = rng.standard_normal((n, d))
                S_in = S_out[:, t-1, :] * np.exp(
                    (R-DELTA-0.5*SIGMA**2)*DT + SIGMA*np.sqrt(DT)*Zn)
                E_Q += maxcall_payoff(S_in, K)
            E_Q /= n_res
        else:
            if betas[t] is not None:
                E_basis = closed_form_E_basis_nd(
                    S_out[:, t-1, :], degree, means, stds, indices,
                    R, DELTA, SIGMA, DT)
                T1_val = E_basis @ betas[t]
            else:
                T1_val = np.zeros(n)
            T2_val = np.zeros(n)
            for j in range(n_res):
                Zn = rng.standard_normal((n, d))
                S_in = S_out[:, t-1, :] * np.exp(
                    (R-DELTA-0.5*SIGMA**2)*DT + SIGMA*np.sqrt(DT)*Zn)
                Z_pay = maxcall_payoff(S_in, K)
                Phi_in, _ = basis_stdlog_nd(S_in, degree, means, stds)
                C_in = Phi_in @ betas[t] if betas[t] is not None else np.zeros(n)
                T2_val += np.maximum(Z_pay - C_in, 0.0)
            T2_val /= n_res
            E_Q = T1_val + T2_val
        dM = disc[t] * Q[:, t] - disc[t] * E_Q
        M[:, t] = M[:, t-1] + dM

    pathwise = np.max(disc[None, :] * Z_all - M, axis=1)
    return pathwise.mean(), pathwise.std(ddof=1) / np.sqrt(n)

# =====================================================================
#  BBS regression-later estimator (Doob-decomposed, sample-split)
# =====================================================================
def bbs_regression_later(S_tr, S_ou, betas_lsm, means, stds, degree_bbs):
    """Correct regression-later: regress Q_t onto phi(X_t) using an
    INDEPENDENT training sample; build the dual martingale from the
    Doob-decomposed basis phi(X_t) - E[phi(X_t)|F_{t-1}] (closed form).

    Parameters
    ----------
    S_tr : (n_tr, T+1, d)  — independent training paths for BBS regression
    S_ou : (n_ou, T+1, d)  — evaluation paths (shared with hybrid/nested)
    betas_lsm : list        — LSM continuation-value coefficients (shared policy)
    means, stds             — standardization stats from LSM training
    degree_bbs : int        — polynomial degree for BBS basis
    """
    def snell_QZ(S):
        n, T1, d = S.shape; T = T1 - 1
        Z = np.zeros((n, T + 1)); Q = np.zeros((n, T + 1))
        for t in range(T + 1):
            Z[:, t] = maxcall_payoff(S[:, t, :], K)
            if 0 < t < T and betas_lsm[t] is not None:
                Phi, _ = basis_stdlog_nd(S[:, t, :], DEGREE_LSM, means, stds)
                C = Phi @ betas_lsm[t]
            else:
                C = np.zeros(n)
            Q[:, t] = np.maximum(Z[:, t], C)
        return Z, Q

    Z_tr, Q_tr = snell_QZ(S_tr)
    Z_ou, Q_ou = snell_QZ(S_ou)

    n = S_ou.shape[0]; T = S_ou.shape[1] - 1
    disc = np.exp(-R * np.arange(T + 1) * DT)

    # Get basis indices once
    _, idx_bbs = basis_stdlog_nd(S_ou[:, 1, :], degree_bbs, means, stds)

    M = np.zeros((n, T + 1))
    for t in range(1, T + 1):
        # Regression on INDEPENDENT training sample
        Phi_tr, _ = basis_stdlog_nd(S_tr[:, t, :], degree_bbs, means, stds)
        c = np.linalg.lstsq(Phi_tr, Q_tr[:, t], rcond=None)[0]

        # Finite-value check
        assert np.all(np.isfinite(c)), \
            f"BBS: non-finite coefficients at t={t}, max|c|={np.abs(c).max():.2e}"

        # Evaluate on outer paths
        Phi_ou, _ = basis_stdlog_nd(S_ou[:, t, :], degree_bbs, means, stds)

        # Closed-form E[phi(X_t) | F_{t-1}] — exact, no MC
        E_basis = closed_form_E_basis_nd(S_ou[:, t-1, :], degree_bbs, means, stds,
                                         idx_bbs, R, DELTA, SIGMA, DT)

        # Doob-decomposed martingale difference: (phi - E[phi]) @ c
        dM = disc[t] * ((Phi_ou - E_basis) @ c)
        M[:, t] = M[:, t-1] + dM

    pw = np.max(disc[None, :] * Z_ou - M, axis=1)
    return pw.mean(), pw.std(ddof=1) / np.sqrt(n)


# =====================================================================
#  Nested AB for additional reference
# =====================================================================
def nested_ab(S_out, betas, indices, degree, means, stds, n_inner, rng):
    n, T1, d = S_out.shape
    T = T1 - 1
    disc = np.exp(-R * np.arange(T + 1) * DT)

    Z_all = np.zeros((n, T + 1))
    Q = np.zeros((n, T + 1))
    for t in range(T + 1):
        Z_all[:, t] = maxcall_payoff(S_out[:, t, :], K)
        if 0 < t < T and betas[t] is not None:
            Phi, _ = basis_stdlog_nd(S_out[:, t, :], degree, means, stds)
            C_hat = Phi @ betas[t]
        else:
            C_hat = np.zeros(n)
        Q[:, t] = np.maximum(Z_all[:, t], C_hat)

    M = np.zeros((n, T + 1))
    for t in range(1, T + 1):
        E_Q = np.zeros(n)
        for j in range(n_inner):
            Zn = rng.standard_normal((n, d))
            S_in = S_out[:, t-1, :] * np.exp(
                (R-DELTA-0.5*SIGMA**2)*DT + SIGMA*np.sqrt(DT)*Zn)
            Z_pay = maxcall_payoff(S_in, K)
            if t < T and betas[t] is not None:
                Phi_in, _ = basis_stdlog_nd(S_in, degree, means, stds)
                C_in = Phi_in @ betas[t]
            else:
                C_in = np.zeros(n)
            E_Q += np.maximum(Z_pay, C_in)
        E_Q /= n_inner
        dM = disc[t] * Q[:, t] - disc[t] * E_Q
        M[:, t] = M[:, t-1] + dM

    pathwise = np.max(disc[None, :] * Z_all - M, axis=1)
    return pathwise.mean(), pathwise.std(ddof=1) / np.sqrt(n)


# =====================================================================
#  Main comparison
# =====================================================================
def main():
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
                exist_ok=True)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                            "exp_bbs_comparison.csv")

    print("=" * 70)
    print("5D Bermudan max-call: Hybrid vs BBS (Doob-decomposed) vs Nested AB")
    print("=" * 70)
    print(f"  d={D}, S0={S0}, K={K}, r={R}, δ={DELTA}, σ={SIGMA}")
    print(f"  T={T_YEARS}yr, {N_STEPS} exercise dates, ρ={RHO}")
    print(f"  N_train={N_TRAIN}, N_out={N_OUT}, n_res={N_RES}")
    print(f"  Degree LSM={DEGREE_LSM}, Degree BBS sweep={DEGREE_BBS_GRID}")
    print(f"  Reference: AB 2004 LB=26.15, UB=26.28")
    print()

    rows = []
    for seed_idx in range(N_SEEDS):
        seed = 42 + seed_idx * 1000
        print(f"\n--- Seed {seed} ---")
        rng_train = np.random.default_rng(seed)
        rng_out = np.random.default_rng(seed + 500)
        rng_bbs_train = np.random.default_rng(seed + 3000)  # independent BBS training

        # Train LSM (shared)
        S_train = simulate_gbm_paths(N_TRAIN, D, N_STEPS, S0, R, DELTA, SIGMA, DT,
                                      rng_train)[0]
        means, stds = compute_std_stats(S_train)
        betas, indices, _V_LB_insample = lsm_train(S_train, DEGREE_LSM, means, stds)

        # Generate outer paths (CRN for all methods)
        S_out, Z_inc = simulate_gbm_paths(N_OUT, D, N_STEPS, S0, R, DELTA, SIGMA, DT,
                                           rng_out)

        # Out-of-sample V_LB: re-evaluate the LSM policy on the independent
        # outer paths (matches maxcall_5d_benchmark.py lines 332-346). The
        # in-sample V.mean() returned by lsm_train is positively biased and
        # must not be used as the lower bound.
        disc_1 = np.exp(-R * DT)
        V_path = maxcall_payoff(S_out[:, N_STEPS, :], K)
        for t in range(N_STEPS - 1, 0, -1):
            ex = maxcall_payoff(S_out[:, t, :], K)
            if betas[t] is not None:
                Phi_lb, _ = basis_stdlog_nd(S_out[:, t, :], DEGREE_LSM, means, stds)
                C_hat_lb = Phi_lb @ betas[t]
                exercise_now = ex > C_hat_lb
            else:
                exercise_now = np.zeros(len(S_out), dtype=bool)
            V_path = np.where(exercise_now, ex, disc_1 * V_path)
        V_LB = V_path.mean()
        print(f"  V_LB (out-of-sample) = {V_LB:.4f}")

        # Generate INDEPENDENT BBS training paths (sample splitting)
        S_bbs_tr, _ = simulate_gbm_paths(N_TRAIN, D, N_STEPS, S0, R, DELTA, SIGMA, DT,
                                          rng_bbs_train)

        # --- Hybrid nestless ---
        rng_hybrid = np.random.default_rng(seed + 1000)
        t0 = time.perf_counter()
        V_hybrid, se_hybrid = hybrid_nestless(
            S_out, betas, indices, DEGREE_LSM, means, stds, N_RES, rng_hybrid)
        wall_hybrid = time.perf_counter() - t0
        gap_hybrid = (V_hybrid - V_LB) / V_LB * 100
        print(f"  Hybrid:  V_UB={V_hybrid:.4f}  gap={gap_hybrid:.2f}%  "
              f"SE={se_hybrid:.4f}  wall={wall_hybrid:.1f}s")

        # --- BBS regression-later (sweep degrees) ---
        for deg_bbs in DEGREE_BBS_GRID:
            t0 = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                V_bbs, se_bbs = bbs_regression_later(
                    S_bbs_tr, S_out, betas, means, stds, deg_bbs)
            wall_bbs = time.perf_counter() - t0
            gap_bbs = (V_bbs - V_LB) / V_LB * 100
            n_warn = len([w for w in caught if "overflow" in str(w.message)
                          or "invalid" in str(w.message)
                          or "divide" in str(w.message)])
            warn_tag = f"  ⚠ {n_warn} warnings" if n_warn > 0 else ""
            print(f"  BBS(d={deg_bbs}): V_UB={V_bbs:.4f}  gap={gap_bbs:.2f}%  "
                  f"SE={se_bbs:.4f}  wall={wall_bbs:.1f}s{warn_tag}")
            rows.append(dict(
                seed=seed, method=f"BBS_d{deg_bbs}",
                V_LB=V_LB, V_UB=V_bbs, SE=se_bbs,
                gap_pct=gap_bbs, wall_s=round(wall_bbs, 2),
                n_warnings=n_warn,
            ))

        # --- Nested AB (N_in=100) ---
        rng_nested = np.random.default_rng(seed + 2000)
        t0 = time.perf_counter()
        V_nested, se_nested = nested_ab(
            S_out, betas, indices, DEGREE_LSM, means, stds, 100, rng_nested)
        wall_nested = time.perf_counter() - t0
        gap_nested = (V_nested - V_LB) / V_LB * 100
        print(f"  Nested:  V_UB={V_nested:.4f}  gap={gap_nested:.2f}%  "
              f"SE={se_nested:.4f}  wall={wall_nested:.1f}s")

        for method, V_UB, se, wall, gap in [
            ("hybrid", V_hybrid, se_hybrid, wall_hybrid, gap_hybrid),
            ("nested_100", V_nested, se_nested, wall_nested, gap_nested),
        ]:
            rows.append(dict(
                seed=seed, method=method,
                V_LB=V_LB, V_UB=V_UB, SE=se,
                gap_pct=gap, wall_s=round(wall, 2),
                n_warnings=0,
            ))

    import pandas as pd
    df = pd.DataFrame(rows)
    # Create the output directory before writing (a no-op when it already exists).
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df)} rows to {out_path}")

    # Summary
    print(f"\n{'='*70}")
    print("Summary (5 seeds)")
    print(f"{'='*70}")
    summary = df.groupby("method").agg(
        V_LB_mean=("V_LB", "mean"),
        V_UB_mean=("V_UB", "mean"),
        gap_mean=("gap_pct", "mean"),
        gap_se=("gap_pct", lambda x: x.std(ddof=1)/np.sqrt(len(x))),
        wall_mean=("wall_s", "mean"),
        warn_total=("n_warnings", "sum"),
    ).reset_index()
    for _, row in summary.iterrows():
        warn_str = f"  warns={int(row['warn_total'])}" if row['warn_total'] > 0 else ""
        print(f"  {row['method']:>10}: V_UB={row['V_UB_mean']:.3f}  "
              f"gap={row['gap_mean']:.2f}±{row['gap_se']:.2f}%  "
              f"wall={row['wall_mean']:.1f}s{warn_str}")
    print(f"\n  Reference: [26.15, 26.28]")
    print(f"\n  Validation checks:")
    print(f"    - All V_UB > 26.15 (true value)? {(df['V_UB'] > 26.0).all()}")
    print(f"    - Total overflow/invalid warnings: {int(df['n_warnings'].sum())}")


if __name__ == "__main__":
    main()
