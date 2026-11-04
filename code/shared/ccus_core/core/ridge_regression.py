"""
ridge_regression.py
===================
Least-squares and square-root-ridge regression utilities used by the LSM
training routines in the replication package.

The penalized fit minimizes the empirical square-root loss plus
``sqrt(eps) * ||beta||_2``.  The routine below solves the corresponding
fixed-point ridge equations.  The historical function name ``dro_fit`` is kept
as a backward-compatible alias inside the replication package; the submitted
paper does not claim a distributionally robust optimization contribution.
"""
from __future__ import annotations

import numpy as np


def ols_fit(Phi: np.ndarray, y: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """Plain or ridge least squares."""
    if ridge > 0:
        d = Phi.shape[1]
        return np.linalg.solve(np.dot(Phi.T, Phi) + ridge * np.eye(d), np.dot(Phi.T, y))
    return np.linalg.lstsq(Phi, y, rcond=None)[0]


def sqrt_ridge_fit(Phi: np.ndarray,
                   y: np.ndarray,
                   eps: float,
                   n_iter: int = 50,
                   tol: float = 1e-8,
                   verbose: bool = False) -> tuple[np.ndarray, dict]:
    """Solve the square-root-ridge regression fixed-point equations."""
    n, d = Phi.shape
    if eps <= 0.0:
        beta = ols_fit(Phi, y)
        rmse = float(np.linalg.norm(y - np.dot(Phi, beta)) / np.sqrt(n))
        return beta, dict(iters=0, final_lambda=0.0, rmse=rmse,
                          beta_norm=float(np.linalg.norm(beta)),
                          converged=True)

    sqrt_eps = float(np.sqrt(eps))
    PtP = np.dot(Phi.T, Phi)
    Pty = np.dot(Phi.T, y)
    lam = n * sqrt_eps / max(np.sqrt(d), 1e-12)
    beta = np.linalg.solve(PtP + lam * np.eye(d), Pty)
    converged = False
    history = []
    for k in range(n_iter):
        rmse = float(np.linalg.norm(y - np.dot(Phi, beta)) / np.sqrt(n))
        bnorm = float(np.linalg.norm(beta))
        if bnorm < 1e-12:
            bnorm = 1e-6
            beta = beta + 1e-6 * np.ones_like(beta) / np.sqrt(d)
        lam_new = n * sqrt_eps * rmse / bnorm
        beta_new = np.linalg.solve(PtP + lam_new * np.eye(d), Pty)
        delta = float(np.linalg.norm(beta_new - beta))
        history.append((k, rmse, bnorm, lam_new, delta))
        if verbose:
            print(f"  iter {k:3d}: rmse={rmse:.6e}  ||beta||={bnorm:.6e}  "
                  f"lambda={lam_new:.4e}  ||delta||={delta:.4e}")
        beta = beta_new
        if delta < tol * max(bnorm, 1e-12):
            converged = True
            break
    rmse = float(np.linalg.norm(y - np.dot(Phi, beta)) / np.sqrt(n))
    bnorm = float(np.linalg.norm(beta))
    return beta, dict(iters=len(history), final_lambda=lam_new, rmse=rmse,
                      beta_norm=bnorm, converged=converged, history=history)


# Backward-compatible alias used by older driver code.
def dro_fit(Phi: np.ndarray, y: np.ndarray, eps: float, **kwargs):
    return sqrt_ridge_fit(Phi, y, eps, **kwargs)
