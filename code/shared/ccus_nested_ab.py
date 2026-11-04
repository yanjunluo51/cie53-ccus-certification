"""Canonical nested Andersen--Broadie implementation for the CCUS model.

The paper's hybrid certificate uses

    Q_t = max(Z_t, C_hat_t)

without clipping ``C_hat_t``.  The canonical nested comparator must use the
same pathwise target; otherwise a hybrid/nested gap comparison mixes the
effect of inner-simulation noise with a change in the dual martingale itself.

``continuation_clip`` exists only for the clipping audit and is deliberately
keyword-only.  Production callers leave it at ``None``.
"""
from __future__ import annotations

import numpy as np

from ccus_core.core.ccus_model import exercise_payoff
from nestless_3d_calibrated import basis_stdlog
from sample_splitting import simulate_inner_one_step


def _positive_payoff(params, P, C, R):
    shape = np.asarray(P).shape
    value = exercise_payoff(
        params,
        np.asarray(P).reshape(-1),
        np.asarray(C).reshape(-1),
        np.asarray(R).reshape(-1),
    )
    return np.maximum(value, 0.0).reshape(shape)


def _continuation(P, C, R, t, betas_pr, degree, std_stats, *, clip=None):
    """Evaluate the trained continuation with one shared transformation."""
    P_arr = np.asarray(P)
    C_arr = np.asarray(C)
    R_arr = np.asarray(R, dtype=int)
    flat_P = P_arr.reshape(-1)
    flat_C = C_arr.reshape(-1)
    flat_R = R_arr.reshape(-1)
    out = np.zeros(flat_P.size, dtype=float)

    if t in betas_pr:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            Phi = basis_stdlog(flat_P, flat_C, degree, std_stats)
        if not np.isfinite(Phi).all():
            raise FloatingPointError(f"non-finite CCUS basis at t={t}")
        for regime, entry in betas_pr[t].items():
            if entry is None:
                continue
            beta_norm, y_scale = entry
            mask = flat_R == int(regime)
            if not mask.any():
                continue
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                out[mask] = (np.dot(Phi[mask], beta_norm)) * y_scale

    if not np.isfinite(out).all():
        raise FloatingPointError(f"non-finite CCUS continuation at t={t}")
    if clip is not None:
        out = np.clip(out, 0.0, float(clip))
    return out.reshape(P_arr.shape)


def canonical_nested_ab(
    params,
    P_out,
    C_out,
    R_out,
    betas_pr,
    std_stats,
    n_in,
    rng,
    *,
    degree=3,
    continuation_clip=None,
    return_pathwise=False,
):
    """Evaluate nested AB using the same ``Q_t`` as the hybrid certificate.

    ``continuation_clip=None`` is the production definition.  A finite value
    is permitted only in targeted diagnostics that quantify the historical
    clipped/unclipped discrepancy.
    """
    P_out = np.asarray(P_out)
    C_out = np.asarray(C_out)
    R_out = np.asarray(R_out, dtype=int)
    if P_out.shape != C_out.shape or P_out.shape != R_out.shape:
        raise ValueError("P_out, C_out, and R_out must have identical shapes")
    if int(n_in) < 1:
        raise ValueError("n_in must be positive")

    n_out, n_dates = P_out.shape
    n_steps = n_dates - 1
    if n_steps != params.n_steps:
        raise ValueError("outer-path horizon does not match params.n_steps")

    discount = np.exp(-params.r * params.dt * np.arange(n_dates))
    Z = np.zeros((n_out, n_dates), dtype=float)
    Q = np.zeros_like(Z)
    for t in range(n_dates):
        Z[:, t] = _positive_payoff(params, P_out[:, t], C_out[:, t], R_out[:, t])
        if 0 < t < n_steps:
            cont = _continuation(
                P_out[:, t], C_out[:, t], R_out[:, t], t,
                betas_pr, degree, std_stats, clip=continuation_clip,
            )
            Q[:, t] = np.maximum(Z[:, t], cont)
        else:
            Q[:, t] = Z[:, t]

    martingale = np.zeros_like(Z)
    for t in range(1, n_dates):
        inner = simulate_inner_one_step(
            rng,
            P_prev=P_out[:, t - 1],
            C_prev=C_out[:, t - 1],
            R_prev=R_out[:, t - 1],
            n_res=int(n_in),
            dt=params.dt,
            sigma_P=params.sigma_P,
            mu_P_by_regime=np.asarray(params.mu_P_by_regime),
            sigma_C=params.sigma_C,
            mu_C=params.mu_C,
            rho=0.0,
            Q_regime=np.asarray(params.Q_regime),
        )
        Z_inner = _positive_payoff(params, inner["P"], inner["C"], inner["R"])
        if t < n_steps:
            C_inner = _continuation(
                inner["P"], inner["C"], inner["R"], t,
                betas_pr, degree, std_stats, clip=continuation_clip,
            )
            Q_inner = np.maximum(Z_inner, C_inner)
        else:
            Q_inner = Z_inner
        conditional_mean = Q_inner.mean(axis=1)
        martingale[:, t] = (
            martingale[:, t - 1]
            + discount[t] * (Q[:, t] - conditional_mean)
        )

    pathwise = np.max(discount[None, :] * Z - martingale, axis=1)
    if not np.isfinite(pathwise).all():
        raise FloatingPointError("non-finite canonical nested-AB path values")
    estimate = float(pathwise.mean())
    se = float(pathwise.std(ddof=1) / np.sqrt(n_out))
    if return_pathwise:
        return estimate, se, pathwise
    return estimate, se
