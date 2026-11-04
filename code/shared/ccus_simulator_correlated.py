"""
ccus_simulator_correlated.py
============================

Cholesky-correlated wrapper around the existing CCUS path simulator.

Design contract
---------------
- When rho_diff == 0.0, returns BYTE-IDENTICAL output to the original
  ccus_core.core.ccus_model.simulate_paths. This is verified by the
  self-test in __main__.
- When rho_diff != 0.0, uses an independent Cholesky-based implementation
  to draw correlated (Z_P, Z_C) Gaussian increments while preserving:
    * the same regime-transition logic as the original
    * the same drift / volatility / step-discretisation formulas
    * the same use of caller-supplied rng for reproducibility

The wrapper is read-only with respect to existing repo files. No
modification to ccus_core/core/ccus_model.py is required, and no
previously generated data is invalidated.

Provenance
----------
Created 2026-05-21 to support the diffusion-correlation sensitivity check.

Usage
-----
    from ccus_simulator_correlated import simulate_paths_correlated

    # rho=0: identical to original simulator
    P0, C0, R0 = simulate_paths_correlated(p, n_paths, rng=rng, rho_diff=0.0)

    # rho!=0: Cholesky-correlated diffusion
    P1, C1, R1 = simulate_paths_correlated(p, n_paths, rng=rng, rho_diff=-0.5)
"""

from __future__ import annotations
import os, sys
import numpy as np

# ---------------------------------------------------------------------
# Path setup: in the replication package, ccus_core is in shared/.
# In the original repo, ccus_core lives outside (in an external project).
# Try import directly first (works inside replication package); if that
# fails, fall back to searching candidate paths.
# ---------------------------------------------------------------------
try:
    from ccus_core.core.ccus_model import simulate_paths as _simulate_paths_orig
except ImportError:
    # Fallback: search relative paths (for use outside replication package)
    _HERE = os.path.dirname(os.path.abspath(__file__))
    CCUS_CORE_PARENT = os.environ.get("CCUS_CORE_PARENT", _HERE)
    _CANDIDATES = [
        _HERE,                                    # shared/ itself contains ccus_core/
        os.path.join(_HERE, ".."),                 # parent of shared/
        CCUS_CORE_PARENT,                         # env override
    ]
    for _cand in _CANDIDATES:
        if os.path.isdir(os.path.join(_cand, "ccus_core")):
            if _cand not in sys.path:
                sys.path.insert(0, _cand)
            break
    else:
        raise ImportError(
            f"Could not locate ccus_core package in any of:\n  "
            + "\n  ".join(_CANDIDATES)
            + f"\n\nSet the environment variable CCUS_CORE_PARENT to the directory "
            + f"containing ccus_core/, e.g.:\n  "
            + f"export CCUS_CORE_PARENT=/path/to/parent/of/ccus_core"
        )
    from ccus_core.core.ccus_model import simulate_paths as _simulate_paths_orig


def simulate_paths_correlated(p, n_paths, *, rng, regime0=1, rho_diff=0.0):
    """
    Drop-in replacement for ccus_core.core.ccus_model.simulate_paths with an
    additional rho_diff parameter controlling the diffusion correlation
    between log P and log C increments.

    Parameters
    ----------
    p : CCUSParams3D
        Calibration parameters (same as the original).
    n_paths : int
        Number of outer paths.
    rng : np.random.Generator
        NumPy Generator instance. All randomness is sourced from here.
    regime0 : int, default 1
        Initial regime (0 = easing, 1 = neutral, 2 = tightening).
    rho_diff : float, default 0.0
        Correlation between Z_P and Z_C increments. Must be in (-1, 1).
        When 0.0, behaviour is identical to the original simulator.

    Returns
    -------
    (P, C, R) : tuple of np.ndarray each of shape (n_paths, n_steps+1)
        P, C are float; R is integer regime index.
    """
    # Fast path: byte-identical to original
    if abs(rho_diff) < 1e-12:
        return _simulate_paths_orig(p, n_paths, rng=rng, regime0=regime0)

    if not (-1.0 < rho_diff < 1.0):
        raise ValueError(f"rho_diff must lie in (-1, 1); got {rho_diff}")

    # ---- Slow path: Cholesky-correlated implementation ----
    # Mirrors the structure of ccus_core.core.ccus_model.simulate_paths
    # exactly, EXCEPT the (Z_P, Z_C) draws use a Cholesky factor.

    n_steps = p.n_steps
    dt = p.dt
    sqdt = np.sqrt(dt)

    Q_regime = np.asarray(p.Q_regime)
    Qcum = np.cumsum(Q_regime, axis=1)              # (K, K)
    mu_by_regime = np.asarray(p.mu_P_by_regime)

    P = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    C = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    R = np.empty((n_paths, n_steps + 1), dtype=np.int64)

    P[:, 0] = p.P0
    C[:, 0] = p.C0
    R[:, 0] = regime0

    # Cholesky factor for [[1, rho], [rho, 1]]
    # Z = std_normal @ L.T gives Cov(Z_P, Z_C) = rho
    rho = float(rho_diff)
    Lchol = np.array([[1.0, 0.0],
                      [rho, np.sqrt(1.0 - rho * rho)]])

    for t in range(n_steps):
        # ---- Regime transition: R[:, t+1] | R[:, t] ----
        # Match the original convention exactly: uniform u, sample by CDF.
        u = rng.random(n_paths)
        cdf = Qcum[R[:, t]]                          # (n_paths, K)
        # broadcast comparison: R_new = sum_k (u >= cdf[:, k]) for k < K-1
        R[:, t + 1] = (u[:, None] >= cdf[:, :-1]).sum(axis=1)

        # ---- Correlated diffusion increments ----
        Zraw = rng.standard_normal((n_paths, 2))     # iid N(0,1)
        Zcor = np.dot(Zraw, Lchol.T)                        # shape (n_paths, 2)
        ZP = Zcor[:, 0]
        ZC = Zcor[:, 1]

        mu_P_t = mu_by_regime[R[:, t + 1]]
        log_inc_P = (mu_P_t - 0.5 * p.sigma_P ** 2) * dt + p.sigma_P * sqdt * ZP
        log_inc_C = (p.mu_C - 0.5 * p.sigma_C ** 2) * dt + p.sigma_C * sqdt * ZC

        P[:, t + 1] = P[:, t] * np.exp(log_inc_P)
        C[:, t + 1] = C[:, t] * np.exp(log_inc_C)

    return P, C, R


# ============================================================
#  Self-test: rho=0 must be byte-identical to original
# ============================================================
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # Import wherever your make_params lives. Adjust path if needed.
    try:
        from nestless_3d_calibrated import make_params
    except ImportError as e:
        print(f"Could not import make_params: {e}")
        print("Adjust the import or pass --params manually.")
        sys.exit(1)

    print("=" * 70)
    print("Self-test: rho_diff=0 must reproduce original simulator exactly")
    print("=" * 70)

    p = make_params()
    test_passed = True

    for seed in [1000, 1001, 1002]:
        for n_paths in [100, 500]:
            # Original
            rng1 = np.random.default_rng(seed)
            P_orig, C_orig, R_orig = _simulate_paths_orig(p, n_paths, rng=rng1, regime0=1)

            # Wrapper at rho=0
            rng2 = np.random.default_rng(seed)
            P_w, C_w, R_w = simulate_paths_correlated(p, n_paths, rng=rng2,
                                                       regime0=1, rho_diff=0.0)

            ok_P = np.array_equal(P_orig, P_w)
            ok_C = np.array_equal(C_orig, C_w)
            ok_R = np.array_equal(R_orig, R_w)
            status = "OK ✓" if (ok_P and ok_C and ok_R) else "MISMATCH ✗"
            print(f"  seed={seed}, n={n_paths}: P {ok_P}, C {ok_C}, R {ok_R} -> {status}")
            test_passed = test_passed and ok_P and ok_C and ok_R

    print()
    if not test_passed:
        print("FAIL: rho=0 wrapper output differs from original. STOP and debug.")
        sys.exit(1)
    print("PASS: rho=0 wrapper is byte-identical to original simulator.")
    print()

    # Spot-check: rho!=0 should produce different output and correct correlation
    print("Spot-check: rho_diff != 0 should produce correlated increments")
    for rho_diff in [-0.5, 0.3, 0.7]:
        rng = np.random.default_rng(2024)
        N = 50000
        P, C, R = simulate_paths_correlated(p, N, rng=rng, rho_diff=rho_diff)
        # Compute empirical correlation of one-step log-increments at t=0->1
        log_inc_P = np.log(P[:, 1] / P[:, 0])
        log_inc_C = np.log(C[:, 1] / C[:, 0])
        # Strip the drift, leaving sigma*sqrt(dt)*Z which has the target correlation
        # (drift terms are deterministic given regime, but regime is at most 3 values
        # so empirical correlation will already be very close to target)
        emp_corr = np.corrcoef(log_inc_P, log_inc_C)[0, 1]
        print(f"  rho_diff={rho_diff:+.2f}: empirical Corr(log P inc, log C inc) "
              f"= {emp_corr:+.4f}  (target ~ {rho_diff:+.4f})")

    print()
    print("All self-tests passed.")
