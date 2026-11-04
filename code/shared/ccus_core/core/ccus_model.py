"""
ccus_model.py
=============
3D CCUS optimal-stopping problem.

State: (P_t, C_t, regime_t)
  P_t : carbon price (CNY/t),  GBM with regime-dependent drift
  C_t : capture/abatement cost (CNY/t),  exogenous learning curve
  regime_t : Markov chain in {0=easing, 1=neutral, 2=tightening}

Payoff at exercise: NPV of installing CCUS retrofit at time t,
                    discounted at risk-free rate r.

This module is parameter-agnostic: synthetic defaults are provided,
but every parameter can be replaced by calibrated values via
CCUSParams3D(**calibrated_kwargs).

Used by the LSM and Andersen--Broadie certification drivers in this replication package.
  - experiments/*           (all phase-3 experiments)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# =============================================================================
#  Parameter container
# =============================================================================
@dataclass
class CCUSParams3D:
    """All economic and physical parameters for the 3D CCUS problem.

    Defaults below are SYNTHETIC and used only for algorithm validation.
    For real-data runs, replace via CCUSParams3D(**calibrated_kwargs).
    """

    # --- carbon price GBM ---
    P0: float = 96.0              # initial carbon price, CNY/t
    sigma_P: float = 0.20         # carbon price volatility (annual)
    mu_P_by_regime: tuple[float, float, float] = (0.01, 0.04, 0.07)
    # easing  / neutral / tightening: drift increases with regime stringency

    # --- capture cost (exogenous learning curve) ---
    C0: float = 350.0             # initial unit abatement cost, CNY/t
    sigma_C: float = 0.10         # cost shock volatility (annual)
    mu_C: float = -0.03           # cost decline rate (negative drift), annual
    # i.e. unit abatement cost falls ~3 %/yr on average

    # --- regime Markov chain (annual transition probabilities) ---
    # Row i = current regime, column j = next regime.
    # Defaults: persistent regimes with moderate switching.
    Q_regime: np.ndarray = field(default_factory=lambda: np.array([
        [0.85, 0.13, 0.02],   # easing
        [0.10, 0.80, 0.10],   # neutral
        [0.02, 0.13, 0.85],   # tightening
    ]))

    # --- physical / engineering ---
    Q_CO2: float = 3.0e6          # tCO2/yr captured
    capture_rate: float = 0.90    # capture efficiency
    om_overhead: float = 30.0     # CNY/tCO2 fixed O&M, on top of unit cost

    # --- financial ---
    r: float = 0.05               # risk-free / discount rate
    I_invest: float = 13.37e8     # one-shot investment, CNY (600 MW retrofit)
    project_life: int = 25        # years post-investment

    # --- numerical ---
    T: int = 25                   # decision horizon (years)
    dt: float = 1.0               # year step (annual exercise dates)

    def __post_init__(self):
        Q = np.asarray(self.Q_regime, dtype=float)
        if Q.shape != (3, 3):
            raise ValueError("Q_regime must be 3x3")
        if not np.allclose(Q.sum(axis=1), 1.0):
            raise ValueError("Q_regime rows must sum to 1")
        self.Q_regime = Q

    # ------------------------------------------------------------------
    @property
    def n_regimes(self) -> int:
        return 3

    @property
    def n_steps(self) -> int:
        return int(self.T / self.dt)


# =============================================================================
#  Path simulator
# =============================================================================
def simulate_paths(p: CCUSParams3D,
                   n_paths: int,
                   rng: Optional[np.random.Generator] = None,
                   regime0: int = 1
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate n_paths trajectories of (P_t, C_t, regime_t).

    Returns
    -------
    P : (n_paths, n_steps+1) carbon price
    C : (n_paths, n_steps+1) capture cost
    R : (n_paths, n_steps+1) regime in {0,1,2}
    """
    if rng is None:
        rng = np.random.default_rng(42)

    N = n_paths
    M = p.n_steps + 1                 # include t=0
    dt = p.dt

    P = np.empty((N, M), dtype=np.float64)
    C = np.empty((N, M), dtype=np.float64)
    R = np.empty((N, M), dtype=np.int8)

    P[:, 0] = p.P0
    C[:, 0] = p.C0
    R[:, 0] = regime0

    mu_by_regime = np.asarray(p.mu_P_by_regime)
    Qcum = np.cumsum(p.Q_regime, axis=1)   # for inverse-CDF sampling

    sP = p.sigma_P
    sC = p.sigma_C
    muC = p.mu_C

    # pre-draw all standard normals
    ZP = rng.standard_normal((N, p.n_steps))
    ZC = rng.standard_normal((N, p.n_steps))
    U  = rng.uniform(size=(N, p.n_steps))

    for k in range(p.n_steps):
        r_now = R[:, k]                                 # shape (N,)
        mu_now = mu_by_regime[r_now]                    # shape (N,)
        # GBM increment for carbon price
        P[:, k+1] = P[:, k] * np.exp(
            (mu_now - 0.5 * sP**2) * dt + sP * np.sqrt(dt) * ZP[:, k]
        )
        # GBM-like cost decline
        C[:, k+1] = C[:, k] * np.exp(
            (muC - 0.5 * sC**2) * dt + sC * np.sqrt(dt) * ZC[:, k]
        )
        # regime transition: inverse-CDF sampling
        u = U[:, k]
        cdf_for_each = Qcum[r_now]                      # (N, 3)
        R[:, k+1] = (u[:, None] >= cdf_for_each).sum(axis=1)

    return P, C, R


# =============================================================================
#  Payoff (NPV of investing at time t)
# =============================================================================
def exercise_payoff(p: CCUSParams3D,
                    P_t: np.ndarray,
                    C_t: np.ndarray,
                    regime_t: np.ndarray
                    ) -> np.ndarray:
    """NPV (in time-t money) of installing CCUS at time t.

    Cash flow per year (t' >= t):
        CF(t') = Q_CO2 * (P_{t'} - C_{t'} - om_overhead)

    Approximation used here (consistent with Phase 2): replace future
    P_{t'} by P_t * exp(mu * (t'-t)) and C_{t'} by C_t * exp(mu_C * (t'-t)),
    then discount by r. This is a standard semi-analytic NPV used inside LSM
    so that the payoff at time t depends only on (P_t, C_t, regime_t) and
    not on future random paths (no look-ahead dependence).
    """
    n = P_t.shape[0]
    L = p.project_life
    r = p.r
    om = p.om_overhead

    mu_by_regime = np.asarray(p.mu_P_by_regime)
    mu_P_now = mu_by_regime[regime_t]                   # (n,)
    mu_C_now = p.mu_C

    # vector of future years 1..L
    tau = np.arange(1, L + 1, dtype=np.float64)         # (L,)

    # expected price/cost at each future year
    # Use risk-neutral expectation of GBM:  E[X_{t+tau}] = X_t * exp(mu * tau)
    P_fut = P_t[:, None] * np.exp(mu_P_now[:, None] * tau[None, :])  # (n, L)
    C_fut = C_t[:, None] * np.exp(mu_C_now * tau[None, :])           # (n, L)

    # cash flows
    cf = p.Q_CO2 * (P_fut - C_fut - om)                              # (n, L)
    disc = np.exp(-r * tau)                                          # (L,)
    NPV_op = (cf * disc).sum(axis=1)                                 # (n,)

    return NPV_op - p.I_invest


# =============================================================================
#  Smoke test
# =============================================================================
if __name__ == "__main__":
    p = CCUSParams3D()
    P, C, R = simulate_paths(p, n_paths=5, rng=np.random.default_rng(0))
    print("Shapes:", P.shape, C.shape, R.shape)
    print("P[0]:", P[0, :6])
    print("C[0]:", C[0, :6])
    print("R[0]:", R[0, :6])

    payoff_t0 = exercise_payoff(p, P[:, 0], C[:, 0], R[:, 0])
    print("Payoff at t=0 (sample):", payoff_t0[:3])
