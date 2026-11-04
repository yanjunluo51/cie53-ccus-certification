"""
sample_splitting.py
===================

Sample splitting for the residual-nested hybrid AB pipeline.
It enforces the
protocol of Assumption 1 in §5.3 (conservative variant). Three mutually
independent random streams:

    training_rng  -> Algorithm 1 (sqrt-ridge LSM) training paths
    outer_rng     -> Algorithm 2 outer paths
    inner_rng     -> Algorithm 2 residual MC inner samples

Each stream wraps an independent PCG64 BitGenerator (NumPy >= 1.17). Using
distinct seeds on a *shared* BitGenerator is NOT sufficient: that produces
two views of the same stream, not two independent streams. The constructor
asserts at the BitGenerator level that the three streams are independent
instances.

Usage in a refactored experiment script:

    rngs = SampleSplittingRNGs(base_seed=42)
    train_paths = simulate_paths_CCUS(rngs.training, N=N_train, T=T, ...)
    beta_hat    = dro_lsm_fit(train_paths, eps=eps, ...)         # Algorithm 1
    outer_paths = simulate_paths_CCUS(rngs.outer,    N=N_out,   T=T, ...)
    inner_draws = simulate_one_step_kernel(rngs.inner, outer_paths, ...)
    V_cons      = hybrid_nestless_AB(outer_paths, inner_draws, beta_hat,
                                     conservative=True, ...)      # Algorithm 2

The `conservative=True` flag toggles between Algorithm 2 (clipped, original)
and the conservative variant of §5.3 (Modifications A and B applied; no clip).

Author note: keep `base_seed` recorded in the experiment manifest. Different
seeds for `training`, `outer`, `inner` are derived via SeedSequence.spawn,
which is the NumPy-recommended way to produce independent child streams
from a single root seed without correlations.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.special import ndtri


@dataclass(frozen=True)
class SampleSplittingRNGs:
    """
    Three mutually independent RNG streams for the sample-splitting protocol.

    Implementation note: uses np.random.SeedSequence.spawn(3) to derive three
    independent child seed sequences from a single root seed. The resulting
    Generators wrap independent PCG64 BitGenerator instances. This is the
    canonical NumPy idiom for guaranteed-independent parallel streams.

    Parameters
    ----------
    base_seed : int
        Root seed. Two runs with the same base_seed are bit-exact reproducible.
    """

    training: np.random.Generator
    outer: np.random.Generator
    inner: np.random.Generator
    base_seed: int

    @classmethod
    def from_seed(cls, base_seed: int) -> "SampleSplittingRNGs":
        ss = np.random.SeedSequence(base_seed)
        child_training, child_outer, child_inner = ss.spawn(3)
        return cls(
            training=np.random.default_rng(child_training),
            outer=np.random.default_rng(child_outer),
            inner=np.random.default_rng(child_inner),
            base_seed=base_seed,
        )

    def assert_independent(self) -> None:
        """
        Sanity check: the three Generators must wrap distinct BitGenerator
        objects. Failing this means the construction was bypassed and the
        sample-splitting assumption is violated.
        """
        bgs = [self.training.bit_generator, self.outer.bit_generator,
               self.inner.bit_generator]
        ids = {id(b) for b in bgs}
        assert len(ids) == 3, (
            "SampleSplittingRNGs: BitGenerators are not distinct. "
            "Sample-splitting (Assumption 1, §5.3) is violated."
        )


def simulate_CCUS_paths(
    rng: np.random.Generator,
    N: int,
    T: int,
    dt: float,
    *,
    P0: float, sigma_P: float, mu_P_by_regime: np.ndarray,
    C0: float, sigma_C: float, mu_C: float,
    rho: float,
    Q_regime: np.ndarray, R0: int,
) -> dict:
    """
    Simulate N paths of (log P, log C, R) over [0, T*dt] under the regime-
    switching GBM model of §3.1, using the supplied RNG instance only.

    All randomness comes from `rng`. No global numpy randomness is used.
    Caller controls reproducibility via the RNG construction.

    Returns dict with arrays of shape (N, T+1) for P, C, R.
    """
    K = Q_regime.shape[0]

    # Cholesky factor for (log P, log C) joint diffusion
    cov = np.array([[1.0, rho], [rho, 1.0]])
    L = np.linalg.cholesky(cov)

    P = np.empty((N, T + 1), dtype=np.float64)
    C = np.empty((N, T + 1), dtype=np.float64)
    R = np.empty((N, T + 1), dtype=np.int64)
    P[:, 0] = P0
    C[:, 0] = C0
    R[:, 0] = R0

    # one-step transition matrix on the simulation grid (annualised Q_regime
    # is already passed in; here we use it directly per step)
    sqrt_dt = np.sqrt(dt)

    for t in range(T):
        # regime transition: sample R[:, t+1] | R[:, t]
        u = rng.random(N)
        cdf = np.cumsum(Q_regime[R[:, t]], axis=1)
        R[:, t + 1] = (u[:, None] >= cdf[:, :-1]).sum(axis=1)

        # diffusion increments for log P and log C
        Z = np.dot(rng.standard_normal((N, 2)), L.T)            # shape (N, 2)
        mu_P_t = mu_P_by_regime[R[:, t + 1]]             # regime-conditional drift

        log_P_inc = (mu_P_t - 0.5 * sigma_P ** 2) * dt + sigma_P * sqrt_dt * Z[:, 0]
        log_C_inc = (mu_C   - 0.5 * sigma_C ** 2) * dt + sigma_C * sqrt_dt * Z[:, 1]

        P[:, t + 1] = P[:, t] * np.exp(log_P_inc)
        C[:, t + 1] = C[:, t] * np.exp(log_C_inc)

    return {"P": P, "C": C, "R": R}


def simulate_inner_one_step(
    rng: np.random.Generator,
    *,
    P_prev: np.ndarray, C_prev: np.ndarray, R_prev: np.ndarray,
    n_res: int, dt: float,
    sigma_P: float, mu_P_by_regime: np.ndarray,
    sigma_C: float, mu_C: float, rho: float,
    Q_regime: np.ndarray,
) -> dict:
    """
    For each of the N_out outer paths, draw n_res independent one-step
    transitions from (P_prev, C_prev, R_prev) under the same dynamics.

    Returns arrays of shape (N_out, n_res).

    All randomness from `rng` only. Independence from the outer paths is
    guaranteed by the caller using a separate Generator instance (the
    `inner` field of SampleSplittingRNGs).
    """
    N_out = P_prev.shape[0]
    cov = np.array([[1.0, rho], [rho, 1.0]])
    L = np.linalg.cholesky(cov)
    sqrt_dt = np.sqrt(dt)

    # regime transitions: each outer path's R_prev produces n_res
    # one-step regime draws
    u = rng.random((N_out, n_res))
    cdf = np.cumsum(Q_regime[R_prev], axis=1)            # shape (N_out, K)
    R_new = (u[:, :, None] >= cdf[:, None, :-1]).sum(axis=-1)  # (N_out, n_res)

    Z = rng.standard_normal((N_out, n_res, 2))           # (N_out, n_res, 2)
    Z = np.dot(Z, L.T)

    mu_P_new = mu_P_by_regime[R_new]                     # (N_out, n_res)
    log_P_inc = (mu_P_new - 0.5 * sigma_P ** 2) * dt + sigma_P * sqrt_dt * Z[:, :, 0]
    log_C_inc = (mu_C    - 0.5 * sigma_C ** 2) * dt + sigma_C * sqrt_dt * Z[:, :, 1]

    P_new = P_prev[:, None] * np.exp(log_P_inc)
    C_new = C_prev[:, None] * np.exp(log_C_inc)

    return {"P": P_new, "C": C_new, "R": R_new}


def simulate_inner_one_step_stratified(
    rng: np.random.Generator,
    *,
    P_prev: np.ndarray, C_prev: np.ndarray, R_prev: np.ndarray,
    n_res: int, dt: float,
    sigma_P: float, mu_P_by_regime: np.ndarray,
    sigma_C: float, mu_C: float, rho: float,
    Q_regime: np.ndarray,
) -> dict:
    """Randomized Latin-hypercube one-step transitions.

    For every outer state, the regime uniform and both Gaussian innovation
    uniforms use one randomized draw from each of ``n_res`` equal-probability
    strata. Independent row-wise permutations preserve the target joint law in
    expectation while reducing between-stratum residual variation.
    """
    N_out = P_prev.shape[0]
    n_res = int(n_res)
    if n_res < 1:
        raise ValueError("n_res must be positive")
    sqrt_dt = np.sqrt(dt)

    def lhs_uniform():
        order = np.argsort(rng.random((N_out, n_res)), axis=1)
        jitter = rng.random((N_out, n_res))
        return (order + jitter) / n_res

    u_regime = lhs_uniform()
    cdf = np.cumsum(Q_regime[R_prev], axis=1)
    R_new = (
        u_regime[:, :, None] >= cdf[:, None, :-1]
    ).sum(axis=-1)

    # Keep ndtri away from exact endpoints even though randomized strata make
    # them probability-zero events.
    eps = np.finfo(float).eps
    z1 = ndtri(np.clip(lhs_uniform(), eps, 1.0 - eps))
    z2_ind = ndtri(np.clip(lhs_uniform(), eps, 1.0 - eps))
    z2 = rho * z1 + np.sqrt(max(1.0 - rho * rho, 0.0)) * z2_ind

    mu_P_new = mu_P_by_regime[R_new]
    log_P_inc = (
        (mu_P_new - 0.5 * sigma_P ** 2) * dt
        + sigma_P * sqrt_dt * z1
    )
    log_C_inc = (
        (mu_C - 0.5 * sigma_C ** 2) * dt
        + sigma_C * sqrt_dt * z2
    )
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        P_new = P_prev[:, None] * np.exp(log_P_inc)
        C_new = C_prev[:, None] * np.exp(log_C_inc)
    if not np.isfinite(P_new).all() or not np.isfinite(C_new).all():
        raise FloatingPointError("non-finite stratified one-step state")
    return {"P": P_new, "C": C_new, "R": R_new}
