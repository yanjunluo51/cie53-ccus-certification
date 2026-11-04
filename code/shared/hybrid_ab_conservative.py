"""
hybrid_ab_conservative.py
=========================

Algorithm 2' from §5.3 (conservative variant): residual-nested hybrid AB upper
bound WITHOUT clipping. Validity is provided by Theorem 5.2 (renamed
thm:cons-bound) under the sample-splitting protocol of Assumption 1.

Differences from the original Algorithm 2:

    Line 1 (Modification A): Q_hat_t = max(Z_t, C_hat_t), no clip.
    Line 8 (Modification B): T1_hat from Lemma 5.1 closed form, no clip.

The backward-compatible function name predates the manuscript terminology. The
official route is the conservative un-clipped residual-nested estimator; any
clipped output is an explicit diagnostic perturbation, never a comparator used
by the router.

NOTE on numerical safety. The Doob-decomposed basis psi_k = phi_k(X_t) -
mu_k(X_{t-1}) can produce per-path values of large magnitude when the
regression coefficients beta_hat are not regularised (OLS case). The
estimator is still mathematically valid in expectation; the variance
constant C_path(beta_hat) in Eq. (cons-variance) can be empirically
large. To avoid silent overflow on float64, we log-warn whenever
|beta_hat^T psi| > 1e15 anywhere in the run. Under the canonical
warning-as-error configuration, this fails the run rather than emitting a
reportable numerical value.
"""

from __future__ import annotations
import warnings
import numpy as np


def gaussian_raw_moments(mu: np.ndarray, var: np.ndarray, max_order: int) -> np.ndarray:
    """
    Recurrence m_k = mu * m_{k-1} + (k-1) * var * m_{k-2}, m_0 = 1, m_1 = mu.
    Vectorised over the leading dimensions of mu and var (must broadcast).

    Returns array of shape mu.shape + (max_order + 1,).
    """
    mu = np.asarray(mu)
    var = np.asarray(var)
    out_shape = np.broadcast_shapes(mu.shape, var.shape) + (max_order + 1,)
    m = np.empty(out_shape, dtype=np.float64)
    m[..., 0] = 1.0
    if max_order >= 1:
        m[..., 1] = np.broadcast_to(mu, out_shape[:-1])
    for k in range(2, max_order + 1):
        m[..., k] = mu * m[..., k - 1] + (k - 1) * var * m[..., k - 2]
    return m


def closed_form_T1(
    *,
    log_P_prev: np.ndarray, log_C_prev: np.ndarray, R_prev: np.ndarray,
    beta_per_regime: list,       # list of (p,) arrays, one per regime r' = 0..K-1
    basis_powers: np.ndarray,    # shape (p, 2): (i, j) such that phi_{ij} = z_P^i z_C^j
    Q_regime: np.ndarray,
    mu_log_P: float, sigma_log_P: float,
    mu_log_C: float, sigma_log_C: float,
    mu_P_by_regime: np.ndarray,  # shape (K,)
    mu_C: float, sigma_P: float, sigma_C: float, dt: float,
    V_clip: float | None = None,
) -> np.ndarray:
    """
    Implements Lemma 5.1 in its closed form (no truncation, Modification B):

        T1_hat^(omega) = sum_{r'=0}^{K-1} Q_regime[R_{t-1}^(omega), r']
                         * mu_t^(r')(P_{t-1}, C_{t-1})^T beta_hat_t^(r')

    Vectorised over outer paths. Returns shape (N_out,).
    """
    N_out = log_P_prev.shape[0]
    K = Q_regime.shape[0]
    p = basis_powers.shape[0]
    max_i = int(basis_powers[:, 0].max())
    max_j = int(basis_powers[:, 1].max())

    # Conditional moments of (z_P, z_C) given F_{t-1} and regime r'
    # standardised log coordinates: z = (log X - mu_log)/sigma_log
    # log P_t | F_{t-1}, R_t = r' ~ N(log P_{t-1} + (mu_P_r' - 0.5 sigma_P^2) dt, sigma_P^2 dt)
    # so z_P_t | . ~ N(mu_bar_zP_r', var_bar_zP)
    var_bar_zP = (sigma_P ** 2 * dt) / (sigma_log_P ** 2)
    var_bar_zC = (sigma_C ** 2 * dt) / (sigma_log_C ** 2)

    # mu_bar_zP_r' shape (N_out, K)
    mu_bar_zP = (
        log_P_prev[:, None]
        + (mu_P_by_regime[None, :] - 0.5 * sigma_P ** 2) * dt
        - mu_log_P
    ) / sigma_log_P
    mu_bar_zC = (
        log_C_prev
        + (mu_C - 0.5 * sigma_C ** 2) * dt
        - mu_log_C
    ) / sigma_log_C                                    # shape (N_out,)

    # raw moments
    # m_i_zP shape: (N_out, K, max_i+1)
    m_zP = gaussian_raw_moments(mu_bar_zP, var_bar_zP, max_i)
    # m_j_zC shape: (N_out, max_j+1)
    m_zC = gaussian_raw_moments(mu_bar_zC, var_bar_zC, max_j)

    # basis expectation vector mu_t^(r')(...) shape (N_out, K, p)
    # phi_{ij} expectation = m_i_zP * m_j_zC
    i_powers = basis_powers[:, 0]
    j_powers = basis_powers[:, 1]
    # gather via advanced indexing
    mu_phi = m_zP[:, :, i_powers] * m_zC[:, None, j_powers]   # (N_out, K, p)

    # inner product with beta_hat_per_regime
    beta_stack = np.stack(beta_per_regime, axis=0)    # (K, p)
    proj = np.einsum("nkp,kp->nk", mu_phi, beta_stack)   # (N_out, K)

    # diagnostic variant (Algorithm 2): clip per-regime BEFORE marginalising.
    # conservative variant passes V_clip=None and skips this entirely.
    if V_clip is not None:
        proj = np.clip(proj, 0.0, V_clip)

    # marginalise over R_t = r' using Q_regime[R_{t-1}, :]
    weights = Q_regime[R_prev]                           # (N_out, K)
    T1_hat = np.einsum("nk,nk->n", weights, proj)       # (N_out,)

    return T1_hat


def hybrid_nestless_AB(
    *,
    outer_paths: dict,
    inner_simulator,                  # callable: (rng, P_prev, C_prev, R_prev, n_res) -> dict
    inner_rng: np.random.Generator,
    beta_hat: list,                    # beta_hat[t][r] = (p,) for t = 0..T-1, r = 0..K-1
    basis_powers: np.ndarray,
    standardisation_stats: dict,       # mu_log_P, sigma_log_P, mu_log_C, sigma_log_C
    model_params: dict,                # sigma_P, sigma_C, mu_P_by_regime, mu_C, dt, r, Q_regime
    n_res: int,
    payoff_fn,                         # callable(P, C, R, t) -> Z_t
    conservative: bool = True,
    V_clip: float | None = None,       # only used if conservative=False
) -> dict:
    """
    Compute V_hat_UB on the supplied outer paths.

    conservative=True   -> Algorithm 2' (no clipping, Theorem 5.2 valid)
    conservative=False  -> Algorithm 2 original (with V_clip; original Theorem 5.2)

    A single call can be run twice (toggling the flag) on the same outer_paths
    and inner draws, but to respect the sample-splitting protocol cleanly the
    recommended pattern is: run conservative=True with the inner_rng provided
    here, then for the diagnostic comparison run conservative=False with a
    separate inner_rng spawned from the same SeedSequence (otherwise the two
    estimates share inner-sample noise and their difference is not the right
    quantity to interpret).

    Returns dict with V_UB estimate, standard error, p_clip (NaN if
    conservative=True), per-path martingale realisations for diagnostics.
    """
    P = outer_paths["P"]
    C = outer_paths["C"]
    R = outer_paths["R"]
    N_out, T_plus_1 = P.shape
    T = T_plus_1 - 1
    K = model_params["Q_regime"].shape[0]
    dt = model_params["dt"]
    r_rate = model_params["r"]

    log_P = np.log(P)
    log_C = np.log(C)

    M_hat = np.zeros((N_out, T + 1), dtype=np.float64)
    discounted_Z = np.empty((N_out, T + 1), dtype=np.float64)
    for t in range(T + 1):
        discounted_Z[:, t] = np.exp(-r_rate * t * dt) * payoff_fn(P[:, t], C[:, t], R[:, t], t)

    n_clip_events = 0
    t1_abs_sum = 0.0
    t2_abs_sum = 0.0
    # Keep the path-level numerator and denominator contributions so that a
    # low-budget routing probe can attach sampling uncertainty to rho_res.
    # The ratio below is identical to the legacy scalar calculation; these
    # arrays only expose its delta-method standard error.
    t1_abs_by_path = np.zeros(N_out, dtype=np.float64)
    t2_abs_by_path = np.zeros(N_out, dtype=np.float64)

    for t in range(1, T + 1):
        # --- Q_hat at time t ---
        if t == T:
            # terminal: no continuation
            Q_hat_t = payoff_fn(P[:, t], C[:, t], R[:, t], t)
        else:
            # continuation value from trained regression
            C_hat_t = _eval_continuation(
                P[:, t], C[:, t], R[:, t], beta_hat[t], basis_powers,
                standardisation_stats,
            )
            if not conservative and V_clip is not None:
                # Original Algorithm 2 (clipped variant)
                C_hat_t_used = np.clip(C_hat_t, 0.0, V_clip)
                n_clip_events += int(np.sum(np.abs(C_hat_t) > V_clip))
            else:
                # Modification A: no clip on the pathwise continuation
                C_hat_t_used = C_hat_t

            Z_t = payoff_fn(P[:, t], C[:, t], R[:, t], t)
            Q_hat_t = np.maximum(Z_t, C_hat_t_used)

        # --- E_hat[Q_t | F_{t-1}]: Term 1 closed form + Term 2 residual MC ---
        if t == T:
            # terminal case: estimate E[Z_T | F_{T-1}] by n_res samples
            inner = inner_simulator(
                inner_rng,
                P_prev=P[:, t - 1], C_prev=C[:, t - 1], R_prev=R[:, t - 1],
                n_res=n_res,
            )
            Z_inner = payoff_fn(inner["P"], inner["C"], inner["R"], t)
            E_hat_Q = Z_inner.mean(axis=1)
        else:
            # Term 1 (closed form, Lemma 5.1)
            T1_hat = closed_form_T1(
                log_P_prev=log_P[:, t - 1], log_C_prev=log_C[:, t - 1], R_prev=R[:, t - 1],
                beta_per_regime=beta_hat[t], basis_powers=basis_powers,
                Q_regime=model_params["Q_regime"],
                mu_log_P=standardisation_stats["mu_log_P"],
                sigma_log_P=standardisation_stats["sigma_log_P"],
                mu_log_C=standardisation_stats["mu_log_C"],
                sigma_log_C=standardisation_stats["sigma_log_C"],
                mu_P_by_regime=model_params["mu_P_by_regime"],
                mu_C=model_params["mu_C"],
                sigma_P=model_params["sigma_P"],
                sigma_C=model_params["sigma_C"],
                dt=dt,
                V_clip=(V_clip if (not conservative) else None),
            )
            T1_hat_used = T1_hat

            # Term 2 (residual MC on inner samples)
            inner = inner_simulator(
                inner_rng,
                P_prev=P[:, t - 1], C_prev=C[:, t - 1], R_prev=R[:, t - 1],
                n_res=n_res,
            )
            Z_inner = payoff_fn(inner["P"], inner["C"], inner["R"], t)
            C_inner = _eval_continuation_grid(
                inner["P"], inner["C"], inner["R"],
                beta_hat[t], basis_powers, standardisation_stats,
            )
            if not conservative and V_clip is not None:
                C_inner = np.clip(C_inner, 0.0, V_clip)
            T2_hat = np.maximum(Z_inner - C_inner, 0.0).mean(axis=1)
            t1_abs_sum += float(np.mean(np.abs(T1_hat_used)))
            t2_abs_sum += float(np.mean(np.abs(T2_hat)))
            t1_abs_by_path += np.abs(T1_hat_used)
            t2_abs_by_path += np.abs(T2_hat)

            E_hat_Q = T1_hat_used + T2_hat

        # martingale increment
        disc = np.exp(-r_rate * t * dt)
        dM = disc * Q_hat_t - disc * E_hat_Q
        M_hat[:, t] = M_hat[:, t - 1] + dM

        # numerical safety log
        if np.any(np.abs(dM) > 1e15):
            warnings.warn(
                f"[hybrid_nestless_AB] step t={t}: |dM| > 1e15 on some paths. "
                f"This signals pathological polynomial extrapolation from an "
                f"unregularised regression. Result is mathematically valid in "
                f"expectation but variance is empirically infinite.",
                RuntimeWarning,
            )

    # AB pathwise functional
    pathwise = np.max(discounted_Z - M_hat, axis=1)       # (N_out,)
    V_UB_hat = pathwise.mean()
    se = pathwise.std(ddof=1) / np.sqrt(N_out)

    p_clip = (
        np.nan if conservative
        else n_clip_events / (N_out * (T - 1) * 2)   # crude estimate, refine if needed
    )
    rho_res = (
        t2_abs_sum / (t1_abs_sum + t2_abs_sum)
        if (t1_abs_sum + t2_abs_sum) > 0.0 else np.nan
    )
    rho_denominator = t1_abs_by_path + t2_abs_by_path
    rho_denominator_mean = float(np.mean(rho_denominator))
    if N_out > 1 and rho_denominator_mean > 0.0 and np.isfinite(rho_res):
        rho_influence = (
            t2_abs_by_path - rho_res * rho_denominator
        ) / rho_denominator_mean
        rho_res_se = float(np.std(rho_influence, ddof=1) / np.sqrt(N_out))
        rho_res_ci_low = float(max(0.0, rho_res - 1.959963984540054 * rho_res_se))
        rho_res_ci_high = float(min(1.0, rho_res + 1.959963984540054 * rho_res_se))
    else:
        rho_res_se = np.nan
        rho_res_ci_low = np.nan
        rho_res_ci_high = np.nan
    return {
        "V_UB": float(V_UB_hat),
        "SE": float(se),
        "p_clip": p_clip,
        "pathwise": pathwise,
        "M_final": M_hat[:, -1],
        "conservative": conservative,
        "rho_res": float(rho_res),
        "rho_res_se": rho_res_se,
        "rho_res_ci_low": rho_res_ci_low,
        "rho_res_ci_high": rho_res_ci_high,
    }


def _eval_continuation(P, C, R, beta_t_per_regime, basis_powers, stats):
    """Evaluate phi(X_t)^T beta_hat at outer paths."""
    z_P = (np.log(P) - stats["mu_log_P"]) / stats["sigma_log_P"]
    z_C = (np.log(C) - stats["mu_log_C"]) / stats["sigma_log_C"]
    p = basis_powers.shape[0]
    Phi = z_P[:, None] ** basis_powers[None, :, 0] * z_C[:, None] ** basis_powers[None, :, 1]
    out = np.zeros(P.shape[0])
    for r_idx, beta in enumerate(beta_t_per_regime):
        mask = R == r_idx
        out[mask] = np.einsum("ij,j->i", Phi[mask], beta)
    return out


def _eval_continuation_grid(P_grid, C_grid, R_grid, beta_t_per_regime, basis_powers, stats):
    """Evaluate phi(X)^T beta on an (N_out, n_res) grid of inner samples."""
    N_out, n_res = P_grid.shape
    flat_P = P_grid.reshape(-1)
    flat_C = C_grid.reshape(-1)
    flat_R = R_grid.reshape(-1)
    flat_out = _eval_continuation(flat_P, flat_C, flat_R, beta_t_per_regime, basis_powers, stats)
    return flat_out.reshape(N_out, n_res)
