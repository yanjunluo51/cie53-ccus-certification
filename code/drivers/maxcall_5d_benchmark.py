"""
maxcall_5d_benchmark.py — 5D Bermudan max-call benchmark
=========================================================
Andersen-Broadie (2004) standard parameters:
  d=5 assets, S0=100, K=100, r=0.05, delta=0.10, sigma=0.20,
  T=3 years, 9 equal exercise intervals (10 grid points including time 0),
  rho=0 (independent assets).

This is the K=1 (single regime) special case of the residual-nested hybrid
construction. Lemma 5.6 applies verbatim.

Reference values from literature:
  AB 2004 Table 2 (d=5, S0=100): LB=26.128, UB=26.253,
  reported 95% interval [26.109, 26.292].
  Becker et al. (2019) report point estimate 26.159 under the same parameters.
"""
import csv
import os
from pathlib import Path
import numpy as np
import time
from math import comb

np.set_printoptions(precision=4)

# =====================================================================
#  Parameters (AB 2004 standard)
# =====================================================================
D = 5           # number of assets
S0 = 100.0      # initial price per asset
K = 100.0       # strike
R = 0.05        # risk-free rate
DELTA = 0.10    # continuous dividend yield
SIGMA = 0.20    # volatility (same for all assets)
T_YEARS = 3.0   # maturity
N_STEPS = 9     # equal intervals; 10 grid points including time 0
DT = T_YEARS / N_STEPS
RHO = 0.0       # correlation between assets

DEGREE = 2      # polynomial degree for d=5 → p = C(5+2,2) = 21 at deg 2
N_TRAIN = 50000
N_OUT = 50000
N_RES = 50
N_SEEDS = 5

# =====================================================================
#  Simulation
# =====================================================================
def simulate_gbm_paths(n_paths, d, n_steps, s0, r, delta, sigma, dt, rng):
    """Simulate d independent GBM assets."""
    drift = (r - delta - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)
    S = np.zeros((n_paths, n_steps + 1, d))
    S[:, 0, :] = s0
    for t in range(1, n_steps + 1):
        Z = rng.standard_normal((n_paths, d))
        S[:, t, :] = S[:, t-1, :] * np.exp(drift + vol * Z)
    return S

def maxcall_payoff(S, K):
    """max(max_i S_i - K, 0)"""
    return np.maximum(S.max(axis=-1) - K, 0.0)

# =====================================================================
#  Standardized-log polynomial basis for d dimensions
# =====================================================================
def compute_std_stats(S_train):
    """Compute (mean, std) of log(S) for each asset across all t>=1."""
    n, T1, d = S_train.shape
    logS = np.log(np.maximum(S_train[:, 1:, :].reshape(-1, d), 1e-12))
    means = logS.mean(axis=0)
    stds = logS.std(axis=0)
    stds = np.maximum(stds, 1e-8)
    return means, stds

def basis_stdlog_nd(S, degree, means, stds):
    """Monomial basis on standardized log coordinates for d assets.
    Returns features for monomials of total degree <= degree.
    """
    d = S.shape[-1] if S.ndim > 1 else 1
    n = S.shape[0]
    logS = np.log(np.maximum(S, 1e-12))
    z = (logS - means) / stds  # (n, d)

    # Generate all multi-indices with total degree <= degree
    indices = []
    def gen(dim, remaining, current):
        if dim == d:
            indices.append(tuple(current))
            return
        for k in range(remaining + 1):
            gen(dim + 1, remaining - k, current + [k])
    gen(0, degree, [])

    # Build feature matrix
    feats = np.ones((n, len(indices)))
    for col, idx in enumerate(indices):
        for dim, power in enumerate(idx):
            if power > 0:
                feats[:, col] *= z[:, dim] ** power
    return feats, indices

def gaussian_raw_moment(mu, var, k):
    """E[X^k] for X ~ N(mu, var), vectorized over mu."""
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
    """Closed-form E[basis(S_t) | S_{t-1}] for d independent GBMs.
    Each z_i(t) = (log S_i(t) - mean_i) / std_i is conditionally Gaussian.
    E[prod z_i^{k_i}] = prod E[z_i^{k_i}] (independence).
    """
    n, d = S_prev.shape
    p = len(indices)
    E_basis = np.ones((n, p))

    # For each asset, compute conditional mean and variance of z_i(t)
    drift_log = (r - delta - 0.5 * sigma**2) * dt
    var_log = sigma**2 * dt

    for dim in range(d):
        logS_prev = np.log(np.maximum(S_prev[:, dim], 1e-12))
        # log S_i(t) ~ N(log S_i(t-1) + drift_log, var_log)
        # z_i(t) = (log S_i(t) - mean_i) / std_i
        mu_z = (logS_prev + drift_log - means[dim]) / stds[dim]
        var_z = var_log / stds[dim]**2

        for col, idx in enumerate(indices):
            k = idx[dim]
            if k > 0:
                E_basis[:, col] *= gaussian_raw_moment(mu_z, var_z, k)

    return E_basis

# =====================================================================
#  LSM training (OLS, no ridge for simplicity — matches ε=0 baseline)
# =====================================================================
def lsm_train(S, degree, means, stds):
    """Standard LSM backward induction on max-call."""
    n, T1, d = S.shape
    T = T1 - 1  # number of exercise dates
    disc = np.exp(-R * DT)

    # Terminal value
    V = maxcall_payoff(S[:, T, :], K)
    betas = [None] * T  # betas[t] for t=1..T-1

    _, indices = basis_stdlog_nd(S[:, 0, :], degree, means, stds)

    for t in range(T - 1, 0, -1):
        exercise = maxcall_payoff(S[:, t, :], K)
        itm = exercise > 0
        if itm.sum() < len(indices) + 5:
            V[~itm] = disc * V[~itm]
            continue

        Y = disc * V[itm]
        Phi, _ = basis_stdlog_nd(S[itm, t, :], degree, means, stds)

        # OLS through an explicit SVD pseudoinverse.  This is algebraically the
        # same minimum-norm fit as lstsq while avoiding the macOS BLAS matmul
        # floating-status defect caught by the warning-as-error gate.
        U, singular, Vt = np.linalg.svd(Phi, full_matrices=False)
        projected = np.einsum("ij,i->j", U, Y, optimize=True)
        tolerance = np.finfo(float).eps * max(Phi.shape) * singular[0]
        inverse = np.divide(
            1.0, singular, out=np.zeros_like(singular), where=singular > tolerance)
        beta = np.einsum("ji,j->i", Vt, inverse * projected, optimize=True)
        betas[t] = beta

        C_hat = np.einsum("ij,j->i", Phi, beta, optimize=True)
        ex_mask = exercise[itm] > C_hat
        # Update V
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
#  Residual-nested hybrid AB certification (conservative variant)
# =====================================================================
def residual_nested_conservative(S_out, betas, indices, degree, means, stds,
                                 n_res, rng):
    """Conservative residual-nested AB upper bound (Algorithm 3).
    K=1 regime special case — no regime switching.
    """
    n, T1, d = S_out.shape
    T = T1 - 1
    disc_factors = np.exp(-R * np.arange(T + 1) * DT)

    # Compute Q_t = max(Z_t, C_hat_t) on outer paths
    Q = np.zeros((n, T + 1))
    C_hat_all = np.zeros((n, T + 1))
    Z_all = np.zeros((n, T + 1))

    for t in range(1, T + 1):
        Z_all[:, t] = maxcall_payoff(S_out[:, t, :], K)
        if t < T and betas[t] is not None:
            Phi, _ = basis_stdlog_nd(S_out[:, t, :], degree, means, stds)
            C_hat_all[:, t] = np.einsum(
                "ij,j->i", Phi, betas[t], optimize=True)
        else:
            C_hat_all[:, t] = 0.0  # terminal: no continuation
        Q[:, t] = np.maximum(Z_all[:, t], C_hat_all[:, t])

    Z_all[:, 0] = maxcall_payoff(S_out[:, 0, :], K)
    Q[:, 0] = Z_all[:, 0]

    # Build martingale
    M = np.zeros((n, T + 1))
    for t in range(1, T + 1):
        if t == T:
            # Terminal: E[Q_T | F_{T-1}] = E[Z_T | F_{T-1}] via MC
            E_Q = np.zeros(n)
            for j in range(n_res):
                Z_inner = rng.standard_normal((n, d))
                S_inner = S_out[:, t-1, :] * np.exp(
                    (R - DELTA - 0.5*SIGMA**2)*DT + SIGMA*np.sqrt(DT)*Z_inner)
                E_Q += maxcall_payoff(S_inner, K)
            E_Q /= n_res
        else:
            # Term 1: closed-form E[C_hat_t | F_{t-1}]
            if betas[t] is not None:
                E_basis = closed_form_E_basis_nd(
                    S_out[:, t-1, :], degree, means, stds, indices,
                    R, DELTA, SIGMA, DT)
                T1_val = np.einsum(
                    "ij,j->i", E_basis, betas[t], optimize=True)
            else:
                T1_val = np.zeros(n)

            # Term 2: small MC for E[(Z_t - C_hat_t)+ | F_{t-1}]
            T2_val = np.zeros(n)
            for j in range(n_res):
                Z_inner = rng.standard_normal((n, d))
                S_inner = S_out[:, t-1, :] * np.exp(
                    (R - DELTA - 0.5*SIGMA**2)*DT + SIGMA*np.sqrt(DT)*Z_inner)
                Z_pay = maxcall_payoff(S_inner, K)
                Phi_inner, _ = basis_stdlog_nd(S_inner, degree, means, stds)
                C_inner = (np.einsum(
                    "ij,j->i", Phi_inner, betas[t], optimize=True)
                    if betas[t] is not None else np.zeros(n))
                T2_val += np.maximum(Z_pay - C_inner, 0.0)
            T2_val /= n_res

            E_Q = T1_val + T2_val

        dM = disc_factors[t] * Q[:, t] - disc_factors[t] * E_Q
        M[:, t] = M[:, t-1] + dM

    # V_cons = mean of max_t (disc * Z_t - M_t)
    pathwise = np.zeros(n)
    for t in range(T + 1):
        val = disc_factors[t] * Z_all[:, t] - M[:, t]
        pathwise = np.maximum(pathwise, val)

    V_cons = pathwise.mean()
    se = pathwise.std(ddof=1) / np.sqrt(n)
    return V_cons, se

# =====================================================================
#  Nested AB for comparison
# =====================================================================
def nested_ab(S_out, betas, indices, degree, means, stds, n_inner, rng):
    """Standard nested AB upper bound."""
    n, T1, d = S_out.shape
    T = T1 - 1
    disc_factors = np.exp(-R * np.arange(T + 1) * DT)

    Q = np.zeros((n, T + 1))
    Z_all = np.zeros((n, T + 1))

    for t in range(1, T + 1):
        Z_all[:, t] = maxcall_payoff(S_out[:, t, :], K)
        if t < T and betas[t] is not None:
            Phi, _ = basis_stdlog_nd(S_out[:, t, :], degree, means, stds)
            C_hat = np.einsum("ij,j->i", Phi, betas[t], optimize=True)
        else:
            C_hat = np.zeros(n)
        Q[:, t] = np.maximum(Z_all[:, t], C_hat)

    Z_all[:, 0] = maxcall_payoff(S_out[:, 0, :], K)
    Q[:, 0] = Z_all[:, 0]

    M = np.zeros((n, T + 1))
    for t in range(1, T + 1):
        E_Q = np.zeros(n)
        for j in range(n_inner):
            Z_inner = rng.standard_normal((n, d))
            S_inner = S_out[:, t-1, :] * np.exp(
                (R - DELTA - 0.5*SIGMA**2)*DT + SIGMA*np.sqrt(DT)*Z_inner)
            Z_pay = maxcall_payoff(S_inner, K)
            if t < T and betas[t] is not None:
                Phi_inner, _ = basis_stdlog_nd(S_inner, degree, means, stds)
                C_inner = np.einsum(
                    "ij,j->i", Phi_inner, betas[t], optimize=True)
            else:
                C_inner = np.zeros(n)
            E_Q += np.maximum(Z_pay, C_inner)
        E_Q /= n_inner

        dM = disc_factors[t] * Q[:, t] - disc_factors[t] * E_Q
        M[:, t] = M[:, t-1] + dM

    pathwise = np.zeros(n)
    for t in range(T + 1):
        val = disc_factors[t] * Z_all[:, t] - M[:, t]
        pathwise = np.maximum(pathwise, val)

    V_ub = pathwise.mean()
    se = pathwise.std(ddof=1) / np.sqrt(n)
    return V_ub, se

# =====================================================================
#  Main experiment
# =====================================================================
def run_one_seed(seed):
    rng_train = np.random.default_rng(seed * 1000)
    rng_out = np.random.default_rng(seed * 1000 + 500)
    rng_hybrid = np.random.default_rng(seed * 1000 + 700)
    rng_nested = np.random.default_rng(seed * 1000 + 800)

    # Train
    S_train = simulate_gbm_paths(N_TRAIN, D, N_STEPS, S0, R, DELTA, SIGMA, DT, rng_train)
    means, stds = compute_std_stats(S_train)

    t0 = time.time()
    betas, indices, V_LB = lsm_train(S_train, DEGREE, means, stds)
    t_train = time.time() - t0

    p_basis = len(indices)

    # Outer paths (independent)
    S_out = simulate_gbm_paths(N_OUT, D, N_STEPS, S0, R, DELTA, SIGMA, DT, rng_out)

    # Re-evaluate LB on outer paths
    disc = np.exp(-R * DT)
    V_path = maxcall_payoff(S_out[:, N_STEPS, :], K)
    for t in range(N_STEPS - 1, 0, -1):
        ex = maxcall_payoff(S_out[:, t, :], K)
        if betas[t] is not None:
            Phi, _ = basis_stdlog_nd(S_out[:, t, :], DEGREE, means, stds)
            C_hat = np.einsum("ij,j->i", Phi, betas[t], optimize=True)
            exercise_now = ex > C_hat
        else:
            exercise_now = np.zeros(len(S_out), dtype=bool)
        V_path = np.where(exercise_now, ex, disc * V_path)
    # Discounting fix: the backward loop stops at t=1, so this value is priced at time 1
    # and needs one more discount step to be a time-0 price. The upper bound is discounted
    # to time 0 with exp(-R*arange(T+1)*DT). Mixing the two would shrink the gap systematically.
    V_path = disc * V_path
    V_LB_out = V_path.mean()

    # Hybrid conservative
    t0 = time.time()
    V_cons, se_cons = residual_nested_conservative(
        S_out, betas, indices, DEGREE, means, stds, N_RES, rng_hybrid)
    t_hybrid = time.time() - t0

    # Nested AB (N_in=100)
    t0 = time.time()
    V_nested, se_nested = nested_ab(
        S_out, betas, indices, DEGREE, means, stds, 100, rng_nested)
    t_nested = time.time() - t0

    gap_hybrid = (V_cons - V_LB_out) / V_LB_out * 100
    gap_nested = (V_nested - V_LB_out) / V_LB_out * 100

    return {
        'seed': seed, 'p_basis': p_basis,
        'V_LB': V_LB_out, 'V_cons': V_cons, 'se_cons': se_cons,
        'V_nested': V_nested, 'se_nested': se_nested,
        'gap_hybrid': gap_hybrid, 'gap_nested': gap_nested,
        't_train': t_train, 't_hybrid': t_hybrid, 't_nested': t_nested,
    }

if __name__ == "__main__":
    print(f"5D Max-Call Benchmark: d={D}, S0={S0}, K={K}, σ={SIGMA}, "
          f"r={R}, δ={DELTA}, T={T_YEARS}, {N_STEPS} dates")
    print(f"Degree={DEGREE}, N_train={N_TRAIN}, N_out={N_OUT}, n_res={N_RES}")
    print("Literature ref (AB 2004, d=5, S0=100): "
          "LB=26.128, UB=26.253, 95% CI=[26.109, 26.292]")
    print("=" * 70)

    results = []
    for seed in range(1, N_SEEDS + 1):
        print(f"\n--- Seed {seed}/{N_SEEDS} ---")
        r = run_one_seed(seed)
        results.append(r)
        print(f"  V_LB = {r['V_LB']:.3f}")
        print(f"  V_cons (hybrid) = {r['V_cons']:.3f} ± {r['se_cons']:.3f}  "
              f"gap = {r['gap_hybrid']:.1f}%  time = {r['t_hybrid']:.1f}s")
        print(f"  V_nested (N_in=100) = {r['V_nested']:.3f} ± {r['se_nested']:.3f}  "
              f"gap = {r['gap_nested']:.1f}%  time = {r['t_nested']:.1f}s")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    V_LBs = [r['V_LB'] for r in results]
    V_cons_list = [r['V_cons'] for r in results]
    V_nested_list = [r['V_nested'] for r in results]
    gap_h = [r['gap_hybrid'] for r in results]
    gap_n = [r['gap_nested'] for r in results]
    t_h = [r['t_hybrid'] for r in results]
    t_n = [r['t_nested'] for r in results]

    print(f"V_LB:     {np.mean(V_LBs):.3f} ± {np.std(V_LBs, ddof=1):.3f}")
    print(f"V_cons:   {np.mean(V_cons_list):.3f} ± {np.std(V_cons_list, ddof=1):.3f}")
    print(f"V_nested: {np.mean(V_nested_list):.3f} ± {np.std(V_nested_list, ddof=1):.3f}")
    print(f"Gap hybrid:  {np.mean(gap_h):.1f}% ± {np.std(gap_h, ddof=1):.1f}%")
    print(f"Gap nested:  {np.mean(gap_n):.1f}% ± {np.std(gap_n, ddof=1):.1f}%")
    print(f"Time hybrid: {np.mean(t_h):.1f}s ± {np.std(t_h, ddof=1):.1f}s")
    print(f"Time nested: {np.mean(t_n):.1f}s ± {np.std(t_n, ddof=1):.1f}s")
    print(f"Speedup:     {np.mean(t_n)/np.mean(t_h):.1f}x")
    print("\nLiterature ref (AB 2004): "
          "LB=26.128, UB=26.253, 95% CI=[26.109, 26.292]")
    print(f"Our sandwich:   [{np.mean(V_LBs):.2f}, {np.mean(V_cons_list):.2f}]")

    # Paired gap difference
    diff = [gap_n[i] - gap_h[i] for i in range(len(gap_h))]
    mean_diff = np.mean(diff)
    se_diff = np.std(diff, ddof=1) / np.sqrt(len(diff))
    print(f"\nPaired gap difference (nested - hybrid):")
    print(f"  Δ = {mean_diff:.3f} ± {se_diff:.3f} pp (mean ± standard error)")

    default_output = Path(__file__).resolve().parents[1] / "results" / "maxcall_5d_canonical.csv"
    output = Path(os.environ.get("MAXCALL5D_OUTPUT", str(default_output)))
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(results[0])
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"wrote {output}")
