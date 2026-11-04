"""
nestless_3d_calibrated.py — residual-nested support routines for calibrated 3D CCUS
====================================================================================
Self-contained LSM with STANDARDIZED LOG basis (not sqrt-ridge LSM).

Basis: z_P = (logP - mu_logP) / sd_logP, z_C = (logC - mu_logC) / sd_logC
Features: monomials in (z_P, z_C)
Closed-form: Gaussian moments on standardized log (not lognormal).
"""
import argparse, logging, time, os
import numpy as np
import pandas as pd
from math import comb
from itertools import product as iproduct
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ccus_core.core.ccus_model import CCUSParams3D, simulate_paths, exercise_payoff
from ccus_core.core.ridge_regression import ols_fit, dro_fit


def make_params():
    return CCUSParams3D(
        P0=60.65, sigma_P=0.278, mu_P_by_regime=(0.05, 0.05, 0.05),
        C0=350.0, mu_C=-0.03, sigma_C=0.10,
        r=0.05, T=25, dt=1.0,
    )


def double_factorial(m):
    if m <= 0: return 1.0
    out = 1.0
    for i in range(m, 0, -2): out *= i
    return out


def gaussian_raw_moment_vec(mu, var, k):
    """E[X^k] where X ~ N(mu, var), vectorized over mu."""
    acc = np.zeros_like(mu, dtype=np.float64)
    for j in range(k // 2 + 1):
        coef = comb(k, 2*j) * double_factorial(2*j - 1)
        acc += coef * (var**j) * (mu**(k - 2*j))
    return acc


def basis_stdlog(P, C, degree, std_stats):
    """Monomial basis on standardized log(P), log(C). Ordering: for i..for j..
    std_stats: tuple (z_mean_P, z_std_P, z_mean_C, z_std_C) or 4 separate args.
    """
    if isinstance(std_stats, tuple) and len(std_stats) == 4:
        z_mean_P, z_std_P, z_mean_C, z_std_C = std_stats
    else:
        raise ValueError("std_stats must be (z_mean_P, z_std_P, z_mean_C, z_std_C)")
    zP = (np.log(np.maximum(P, 1e-12)) - z_mean_P) / z_std_P
    zC = (np.log(np.maximum(C, 1e-12)) - z_mean_C) / z_std_C
    feats = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            feats.append((zP**i) * (zC**j))
    return np.column_stack(feats)


def basis_indices(degree):
    """Return (i,j) pairs matching basis_stdlog ordering."""
    return [(i, j) for i in range(degree+1) for j in range(degree+1-i)]


def closed_form_E_basis_stdlog(P_t, C_t, mu_P_r, p, degree, std_stats):
    """
    E[basis(P_{t+dt}, C_{t+dt}) | P_t, C_t, regime r] closed form.

    logP_{t+dt} ~ N(logP_t + (mu_P_r - sig^2/2)*dt, sig_P^2*dt)
    z_P_{t+dt} = (logP_{t+dt} - z_mean_P) / z_std_P ~ N(mu_z, var_z)
    Independent of z_C (rho=0).
    E[z_P^i * z_C^j] = E[z_P^i] * E[z_C^j] (Gaussian moments)
    """
    z_mean_P, z_std_P, z_mean_C, z_std_C = std_stats
    n = P_t.shape[0]
    dt = p.dt
    mu_zP = (np.log(np.maximum(P_t, 1e-12)) + (mu_P_r - 0.5*p.sigma_P**2)*dt - z_mean_P) / z_std_P
    mu_zC = (np.log(np.maximum(C_t, 1e-12)) + (p.mu_C - 0.5*p.sigma_C**2)*dt - z_mean_C) / z_std_C
    var_zP = p.sigma_P**2 * dt / z_std_P**2
    var_zC = p.sigma_C**2 * dt / z_std_C**2

    feats = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            E_zPi = gaussian_raw_moment_vec(mu_zP, var_zP, i)
            E_zCj = gaussian_raw_moment_vec(mu_zC, var_zC, j)
            feats.append(E_zPi * E_zCj)
    return np.column_stack(feats)


def lsm_3d_stdlog(p, n_paths, degree, seed, basis_fn=None, eps=0.0):
    """
    Self-contained 3D LSM with per-regime regression on standardized log basis.
    Returns V_LB, betas_per_regime, std_stats, paths.
    basis_fn: callable(P, C, degree, std_stats) -> (n, p) matrix. Default: monomial.
    eps: square-root-ridge radius. eps=0 -> OLS, eps>0 -> BKM ridge via dro_fit.
    """
    if basis_fn is None:
        basis_fn = basis_stdlog
    rng = np.random.default_rng(seed)
    P, C, R = simulate_paths(p, n_paths, rng=rng, regime0=1)
    M = p.n_steps + 1

    # Global standardization stats from all paths
    logP_all = np.log(np.maximum(P[:, 1:], 1e-12))
    logC_all = np.log(np.maximum(C[:, 1:], 1e-12))
    z_mean_P = float(logP_all.mean())
    z_std_P = float(logP_all.std())
    z_mean_C = float(logC_all.mean())
    z_std_C = float(logC_all.std())
    if z_std_P < 1e-12: z_std_P = 1.0
    if z_std_C < 1e-12: z_std_C = 1.0
    std_stats = (z_mean_P, z_std_P, z_mean_C, z_std_C)

    # LSM backward induction with (exercise_time, exercise_payoff) tracking
    ex_time = np.full(n_paths, M - 1, dtype=np.int64)
    ex_payoff = exercise_payoff(p, P[:, -1], C[:, -1], R[:, -1])
    ex_payoff = np.maximum(ex_payoff, 0.0)

    betas_pr = {}  # {t: {regime: beta}}
    discount = np.exp(-p.r * p.dt)

    for t in range(M - 2, 0, -1):
        Pt, Ct, Rt = P[:, t], C[:, t], R[:, t].astype(int)
        intrinsic = np.maximum(exercise_payoff(p, Pt, Ct, Rt), 0.0)

        betas_pr[t] = {}
        for r_val in range(3):
            mask_r = (Rt == r_val)
            mask_itm = mask_r & (intrinsic > 0)
            if mask_itm.sum() < 20:
                betas_pr[t][r_val] = None
                continue

            Phi = basis_fn(Pt[mask_itm], Ct[mask_itm], degree, std_stats)

            # Regression target: discounted future payoff
            dt_to_ex = (ex_time[mask_itm] - t) * p.dt
            Y = np.exp(-p.r * dt_to_ex) * ex_payoff[mask_itm]

            # Normalize Y for numerical stability of beta
            y_scale = max(float(np.abs(Y).max()), 1.0)
            if eps < 1e-9:
                beta_norm = ols_fit(Phi, Y / y_scale)
            else:
                beta_norm, _ = dro_fit(Phi, Y / y_scale, eps)
            # Store (beta_norm, y_scale) so nestless can use beta_norm (small)
            betas_pr[t][r_val] = (beta_norm, y_scale)

            # Exercise decision: cont_est = Phi @ beta_norm * y_scale = Phi @ beta_raw
            Phi_all = basis_fn(Pt[mask_r], Ct[mask_r], degree, std_stats)
     
            cont_est = np.einsum("ij,j->i", Phi_all, beta_norm) * y_scale

            do_ex = (intrinsic[mask_r] > 0) & (intrinsic[mask_r] >= cont_est)
            idx_ex = np.where(mask_r)[0][do_ex]
            ex_time[idx_ex] = t
            ex_payoff[idx_ex] = intrinsic[idx_ex]

    pv = np.exp(-p.r * ex_time * p.dt) * ex_payoff
    V_LB = float(pv.mean())
    std_stats = (z_mean_P, z_std_P, z_mean_C, z_std_C)
    return V_LB, betas_pr, std_stats, P, C, R


def evaluate_lsm_policy(p, P, C, R, betas_pr, degree, std_stats,
                        basis_fn=None, return_pathwise=False):
    """Evaluate a trained LSM stopping rule on independent paths.

    The value returned by :func:`lsm_3d_stdlog` is an in-training diagnostic.
    A reportable primal lower bound must execute the frozen policy on paths
    independent of the regression sample; this function provides that step.
    """
    if basis_fn is None:
        basis_fn = basis_stdlog
    P = np.asarray(P)
    C = np.asarray(C)
    R = np.asarray(R, dtype=int)
    if P.shape != C.shape or P.shape != R.shape:
        raise ValueError("P, C, and R must have identical shapes")
    n_paths, n_dates = P.shape
    if n_dates != p.n_steps + 1:
        raise ValueError("path horizon does not match p.n_steps")

    alive = np.ones(n_paths, dtype=bool)
    exercise_time = np.full(n_paths, p.n_steps, dtype=int)
    terminal = np.maximum(
        exercise_payoff(p, P[:, -1], C[:, -1], R[:, -1]), 0.0
    )
    exercise_value = terminal.copy()

    for t in range(1, p.n_steps):
        active = np.where(alive)[0]
        if active.size == 0:
            break
        intrinsic = np.maximum(
            exercise_payoff(p, P[active, t], C[active, t], R[active, t]),
            0.0,
        )
        continuation = np.zeros(active.size, dtype=float)
        if t in betas_pr:
            Phi = basis_fn(P[active, t], C[active, t], degree, std_stats)
            if not np.isfinite(Phi).all():
                raise FloatingPointError(f"non-finite policy basis at t={t}")
            regimes = R[active, t]
            for regime, entry in betas_pr[t].items():
                if entry is None:
                    continue
                beta_norm, y_scale = entry
                mask = regimes == int(regime)
                if mask.any():
                    continuation[mask] = (
                        np.einsum("ij,j->i", Phi[mask], beta_norm) * y_scale
                    )
        if not np.isfinite(continuation).all():
            raise FloatingPointError(f"non-finite policy continuation at t={t}")
        stop_local = (intrinsic > 0.0) & (intrinsic >= continuation)
        stopped = active[stop_local]
        exercise_time[stopped] = t
        exercise_value[stopped] = intrinsic[stop_local]
        alive[stopped] = False

    pathwise = np.exp(-p.r * p.dt * exercise_time) * exercise_value
    if not np.isfinite(pathwise).all():
        raise FloatingPointError("non-finite independent lower-bound paths")
    value = float(pathwise.mean())
    se = float(pathwise.std(ddof=1) / np.sqrt(n_paths))
    if return_pathwise:
        return value, se, pathwise
    return value, se


def hybrid_nestless_3d(P, C, R, betas_pr, p, degree, std_stats,
                        n_inner_res=50, seed=99999,
                        basis_fn=None, closed_form_fn=None,
                        return_per_path=False):
    """
    Retired prototype for the residual decomposition.

    WARNING (do not use for reported numbers): this function and this module's
    __main__ block evaluate the certificate on the SAME paths used to fit the
    coefficients, which violates the sample-splitting of Assumption 1 (no
    independent outer stream). It is retained only as an exploratory prototype
    and is NOT wired into any manuscript number. The canonical, sample-split
    certificate is `hybrid_ab_conservative.hybrid_nestless_AB`, driven by
    `single_regime_ccus_driver.py` with `sample_splitting.SampleSplittingRNGs`.
    """
    raise RuntimeError(
        "retired non-sample-split prototype; use hybrid_ab_conservative.hybrid_nestless_AB")
    if basis_fn is None:
        basis_fn = basis_stdlog
    if closed_form_fn is None:
        closed_form_fn = closed_form_E_basis_stdlog
    z_mean_P, z_std_P, z_mean_C, z_std_C = std_stats
    n_paths = P.shape[0]; n_steps = p.n_steps
    dt = p.dt; r = p.r; rng = np.random.default_rng(seed)
    mu_by_regime = np.asarray(p.mu_P_by_regime)
    Q_regime = np.asarray(p.Q_regime)
    Qcum = np.cumsum(Q_regime, axis=1)
    disc_arr = np.exp(-r * dt * np.arange(n_steps + 1))

    # Z on outer
    Z_arr = np.zeros((n_paths, n_steps + 1))
    for t in range(n_steps + 1):
        Z_arr[:, t] = np.maximum(exercise_payoff(p, P[:, t], C[:, t], R[:, t]), 0.0)

    # Economic bound: continuation value must be in [0, V_UB_bound]
    V_CLIP = 5.0e9  # ~1.7x paper V_LB=2.92e9; generous upper bound

    # Q on outer
    Q = np.zeros((n_paths, n_steps + 1))
    Q[:, n_steps] = Z_arr[:, n_steps]
    for t in range(1, n_steps):
        Pt, Ct, Rt = P[:, t], C[:, t], R[:, t].astype(int)
        cont = np.zeros(n_paths)
        if t in betas_pr:
            for rv, entry in betas_pr[t].items():
                if entry is None: continue
                beta_n, ys = entry
                mask = (Rt == rv)
                if not mask.any(): continue
                Phi = basis_fn(Pt[mask], Ct[mask], degree, std_stats)
                cont[mask] = np.einsum("ij,j->i", Phi, beta_n) * ys
        # CLIP: continuation must be non-negative and bounded
        cont = np.clip(cont, 0.0, V_CLIP)
        Q[:, t] = np.maximum(Z_arr[:, t], cont)

    # Martingale
    M = np.zeros((n_paths, n_steps + 1))
    for t in range(1, n_steps + 1):
        P_tm1 = P[:, t-1]; C_tm1 = C[:, t-1]; R_tm1 = R[:, t-1].astype(int)

        # Term 1: E[C_hat_t | F_{t-1}] closed form
        E_C_hat = np.zeros(n_paths)
        if t < n_steps and t in betas_pr:
            for r_prime in range(3):
                entry_rp = betas_pr[t].get(r_prime)
                if entry_rp is None: continue
                beta_n_rp, ys_rp = entry_rp
                trans_prob = Q_regime[R_tm1, r_prime]
                mu_P_rp = mu_by_regime[r_prime]
                E_phi = closed_form_fn(
                    P_tm1, C_tm1, mu_P_rp, p, degree, std_stats)
                # KEY: use beta_norm (small) * y_scale, then CLIP
                E_cont_rp = np.einsum("ij,j->i", E_phi, beta_n_rp) * ys_rp
                E_cont_rp = np.clip(E_cont_rp, 0.0, V_CLIP)
                E_C_hat += trans_prob * E_cont_rp

        # Term 2: E[(Z_t - C_hat_t)^+ | F_{t-1}] small MC
        Z1 = rng.standard_normal((n_paths, n_inner_res))
        Z2 = rng.standard_normal((n_paths, n_inner_res))
        U = rng.random((n_paths, n_inner_res))
        R_in = np.zeros((n_paths, n_inner_res), dtype=int)
        for rv in range(3):
            m = (R_tm1 == rv)
            if m.any():
                R_in[m] = (U[m, :, None] >= Qcum[rv][None, None, :-1]).sum(axis=2)
        mu_P_in = mu_by_regime[R_in]
        P_in = P_tm1[:, None] * np.exp(
            (mu_P_in - 0.5*p.sigma_P**2)*dt + p.sigma_P*np.sqrt(dt)*Z1)
        C_in = C_tm1[:, None] * np.exp(
            (p.mu_C - 0.5*p.sigma_C**2)*dt + p.sigma_C*np.sqrt(dt)*Z2)

        Z_inner = np.zeros((n_paths, n_inner_res))
        C_hat_inner = np.zeros((n_paths, n_inner_res))
        for j in range(n_inner_res):
            Z_inner[:, j] = np.maximum(
                exercise_payoff(p, P_in[:, j], C_in[:, j], R_in[:, j]), 0.0)
            if t < n_steps and t in betas_pr:
                for rv, entry in betas_pr[t].items():
                    if entry is None: continue
                    beta_n, ys = entry
                    mask = (R_in[:, j] == rv)
                    if not mask.any(): continue
                    Phi_j = basis_fn(P_in[:, j][mask], C_in[:, j][mask], degree, std_stats)
                    C_hat_inner[mask, j] = np.einsum("ij,j->i", Phi_j, beta_n) * ys

        # CLIP inner continuation
        C_hat_inner = np.clip(C_hat_inner, 0.0, V_CLIP)
        residual = np.maximum(Z_inner - C_hat_inner, 0.0)
        E_residual = residual.mean(axis=1)

        if t == n_steps:
            E_Q = Z_inner.mean(axis=1)
        else:
            E_Q = E_C_hat + E_residual

        Delta = disc_arr[t] * Q[:, t] - disc_arr[t] * E_Q
        M[:, t] = M[:, t-1] + Delta

    disc_Z = Z_arr * disc_arr[None, :]
    pathmax = np.max(disc_Z - M, axis=1)
    # V_LB from outer paths (replay policy via exercise decisions)
    pv_outer = disc_Z.max(axis=1)  # simplified; real V_LB from LSM
    V_UB = float(pathmax.mean())
    V_UB_se = float(pathmax.std(ddof=1) / np.sqrt(n_paths))
    if return_per_path:
        return V_UB, V_UB_se, M, pathmax, pv_outer
    return V_UB, V_UB_se


def main():
    raise SystemExit(
        "retired non-sample-split prototype; this module only supplies LSM support functions")
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_paths", type=int, default=30000)
    parser.add_argument("--degree", type=int, default=5)
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--n_inner_res", type=int, default=50)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.n_paths = 10000; args.n_seeds = 3; args.degree = 3

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger("nestless3d")
    p = make_params()
    logger.info(f"P0={p.P0}, sigma={p.sigma_P}, C0={p.C0}, r={p.r}, degree={args.degree}")

    rows = []
    t_start = time.time()
    for seed in range(1000, 1000 + args.n_seeds):
        t0 = time.time()
        V_LB, betas_pr, std_stats, Pa, Ca, Ra = lsm_3d_stdlog(
            p, args.n_paths, args.degree, seed)

        # Check max beta
        max_beta = 0
        for t_key, bdict in betas_pr.items():
            for rv, entry in bdict.items():
                if entry is not None:
                    beta_n, ys = entry
                    max_beta = max(max_beta, float(np.abs(beta_n).max()))

        V_UB, V_UB_se = hybrid_nestless_3d(
            Pa, Ca, Ra, betas_pr, p, args.degree, std_stats,
            n_inner_res=args.n_inner_res, seed=seed*1000)
        gap = (V_UB - V_LB) / V_LB * 100
        elapsed = time.time() - t0
        logger.info(f"seed={seed} LB={V_LB:.4e} UB={V_UB:.4e} gap={gap:+.2f}% "
                    f"max|beta|={max_beta:.2e} ({elapsed:.0f}s)")
        rows.append(dict(seed=seed, V_LB=V_LB, V_UB=V_UB, gap_pct=gap,
                         max_beta=max_beta, time_s=elapsed))

    df = pd.DataFrame(rows)
    os.makedirs("nestless_3d_results", exist_ok=True)
    df.to_csv("nestless_3d_results/form_A_baseline.csv", index=False)

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print("3D CALIBRATED HYBRID NESTLESS (stdlog basis)")
    print(f"{'='*60}")
    print(f"V_LB: {df.V_LB.mean():.4e} ± {df.V_LB.std():.2e}  (paper: 2.92e9)")
    print(f"V_UB: {df.V_UB.mean():.4e} ± {df.V_UB.std():.2e}")
    print(f"Gap:  {df.gap_pct.mean():.2f}% ± {df.gap_pct.std():.2f}%")
    print(f"max|beta|: {df.max_beta.mean():.2e}")
    print(f"Time: {total:.0f}s total, {total/args.n_seeds:.0f}s/seed")
    all_pos = all(r['gap_pct'] > 0 for r in rows)
    print(f"All positive: {'OK' if all_pos else 'FAIL'}")


if __name__ == "__main__":
    main()
