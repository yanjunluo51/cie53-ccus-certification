"""Canonical CCUS headline, residual-budget, and stratified comparison.

Every method shares one trained policy and one independent outer evaluation
sample per seed.  The nested comparator uses the same un-clipped
``Q_t=max(Z_t,C_hat_t)`` as the residual-nested hybrid certificate.
"""
from __future__ import annotations

import functools
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "shared"))

from ccus_core.core.ccus_model import simulate_paths, exercise_payoff  # noqa: E402
from ccus_nested_ab import canonical_nested_ab  # noqa: E402
from hybrid_ab_conservative import hybrid_nestless_AB  # noqa: E402
from nestless_3d_calibrated import (  # noqa: E402
    lsm_3d_stdlog, evaluate_lsm_policy, basis_indices, make_params)
from sample_splitting import (  # noqa: E402
    SampleSplittingRNGs, simulate_inner_one_step,
    simulate_inner_one_step_stratified)


DEGREE = 3
EPS = 0.05
N_TRAIN = int(os.environ.get("CCUS_NTRAIN", "30000"))
N_OUT = int(os.environ.get("CCUS_NOUT", "30000"))
N_SEEDS = int(os.environ.get("CCUS_SEEDS", "10"))
WORKERS = max(1, int(os.environ.get("CCUS_WORKERS", "2")))
SEED_BASE = int(os.environ.get("CCUS_SEED_BASE", "1000"))
OUTPUT = Path(os.environ.get(
    "CCUS_OUTPUT", ROOT / "results" / "ccus_canonical_headline.csv"))
METHODS = tuple(
    x.strip() for x in os.environ.get(
        "CCUS_METHODS",
        "plain10,plain50,plain200,strat10,strat50,strat200,nested100",
    ).split(",") if x.strip()
)

PARAMS = make_params()
BASIS_POWERS = np.asarray(basis_indices(DEGREE), dtype=int)


def _adapt_betas(betas_pr):
    out = {}
    for t, by_regime in betas_pr.items():
        values = []
        for regime in range(3):
            entry = by_regime.get(regime)
            values.append(
                np.zeros(len(BASIS_POWERS))
                if entry is None else entry[0] * entry[1]
            )
        out[t] = values
    return out


def _payoff(P, C, R, t):
    shape = np.asarray(P).shape
    raw = exercise_payoff(
        PARAMS,
        np.asarray(P).reshape(-1),
        np.asarray(C).reshape(-1),
        np.asarray(R).reshape(-1),
    )
    return np.maximum(raw, 0.0).reshape(shape)


def _hybrid_row(method, n_res, simulator, P, C, R, betas_pr, stats,
                beta_hat, model_params, inner_ss, V_LB, V_LB_se):
    inner_sim = functools.partial(
        simulator,
        dt=PARAMS.dt,
        sigma_P=PARAMS.sigma_P,
        mu_P_by_regime=np.asarray(PARAMS.mu_P_by_regime),
        sigma_C=PARAMS.sigma_C,
        mu_C=PARAMS.mu_C,
        rho=0.0,
        Q_regime=np.asarray(PARAMS.Q_regime),
    )
    std_dict = dict(zip(
        ["mu_log_P", "sigma_log_P", "mu_log_C", "sigma_log_C"], stats))
    start = time.perf_counter()
    result = hybrid_nestless_AB(
        outer_paths={"P": P, "C": C, "R": R},
        inner_simulator=inner_sim,
        inner_rng=np.random.default_rng(inner_ss),
        beta_hat=beta_hat,
        basis_powers=BASIS_POWERS,
        standardisation_stats=std_dict,
        model_params=model_params,
        n_res=n_res,
        payoff_fn=_payoff,
        conservative=True,
        V_clip=None,
    )
    wall = time.perf_counter() - start
    return {
        "method": method,
        "inner_budget": n_res,
        "V_LB": V_LB,
        "V_LB_se": V_LB_se,
        "V_UB": result["V_UB"],
        "V_UB_se": result["SE"],
        "gap_pct": (result["V_UB"] - V_LB) / V_LB * 100.0,
        "rho_res": result["rho_res"],
        "wall_s": wall,
        "sandwich_ok": bool(result["V_UB"] >= V_LB),
    }


def run_seed(seed):
    setup_start = time.perf_counter()
    streams = SampleSplittingRNGs.from_seed(seed)
    streams.assert_independent()
    _, betas_pr, stats, _, _, _ = lsm_3d_stdlog(
        PARAMS, N_TRAIN, DEGREE,
        seed=streams.training.bit_generator.seed_seq,
        eps=EPS,
    )
    P, C, R = simulate_paths(PARAMS, N_OUT, rng=streams.outer)
    V_LB, V_LB_se = evaluate_lsm_policy(
        PARAMS, P, C, R, betas_pr, DEGREE, stats)
    beta_hat = _adapt_betas(betas_pr)
    model_params = {
        "Q_regime": np.asarray(PARAMS.Q_regime),
        "dt": PARAMS.dt,
        "r": PARAMS.r,
        "mu_P_by_regime": np.asarray(PARAMS.mu_P_by_regime),
        "mu_C": PARAMS.mu_C,
        "sigma_P": PARAMS.sigma_P,
        "sigma_C": PARAMS.sigma_C,
    }
    shared_setup_wall = time.perf_counter() - setup_start
    inner_children = streams.inner.bit_generator.seed_seq.spawn(len(METHODS))
    rows = []
    for method, child in zip(METHODS, inner_children):
        if method.startswith("plain"):
            n_res = int(method.removeprefix("plain"))
            row = _hybrid_row(
                method, n_res, simulate_inner_one_step,
                P, C, R, betas_pr, stats, beta_hat, model_params,
                child, V_LB, V_LB_se)
        elif method.startswith("strat"):
            n_res = int(method.removeprefix("strat"))
            row = _hybrid_row(
                method, n_res, simulate_inner_one_step_stratified,
                P, C, R, betas_pr, stats, beta_hat, model_params,
                child, V_LB, V_LB_se)
        elif method.startswith("nested"):
            n_in = int(method.removeprefix("nested"))
            start = time.perf_counter()
            V_UB, V_UB_se = canonical_nested_ab(
                PARAMS, P, C, R, betas_pr, stats,
                n_in, np.random.default_rng(child), degree=DEGREE)
            row = {
                "method": method,
                "inner_budget": n_in,
                "V_LB": V_LB,
                "V_LB_se": V_LB_se,
                "V_UB": V_UB,
                "V_UB_se": V_UB_se,
                "gap_pct": (V_UB - V_LB) / V_LB * 100.0,
                "rho_res": np.nan,
                "wall_s": time.perf_counter() - start,
                "sandwich_ok": bool(V_UB >= V_LB),
            }
        else:
            raise ValueError(f"unknown CCUS method: {method}")
        row.update(seed=seed, N_train=N_TRAIN, N_out=N_OUT,
                   shared_setup_wall_s=shared_setup_wall,
                   total_if_run_alone_wall_s=shared_setup_wall + row["wall_s"])
        rows.append(row)
    return rows


def main():
    seeds = list(range(SEED_BASE, SEED_BASE + N_SEEDS))
    print("=" * 78)
    print("CANONICAL CCUS HEADLINE / STRATIFIED / NESTED COMPARISON")
    print(f"N_train={N_TRAIN} N_out={N_OUT} seeds={seeds} methods={METHODS}")
    print(f"workers={WORKERS}; lower bound evaluated on independent outer paths")
    print("=" * 78, flush=True)
    if WORKERS == 1:
        blocks = [run_seed(seed) for seed in seeds]
    else:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            blocks = list(pool.map(run_seed, seeds))
    rows = [row for block in blocks for row in block]
    frame = pd.DataFrame(rows).sort_values(["method", "seed"])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, index=False)

    summary = frame.groupby("method", sort=False).agg(
        gap_mean=("gap_pct", "mean"),
        gap_se=("gap_pct", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        rho_mean=("rho_res", "mean"),
        wall_mean=("wall_s", "mean"),
        sandwich_all=("sandwich_ok", "all"),
    )
    print(summary.to_string())
    print(f"CSV: {OUTPUT}")


if __name__ == "__main__":
    main()
