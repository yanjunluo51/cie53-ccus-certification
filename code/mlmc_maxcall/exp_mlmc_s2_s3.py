"""
exp_mlmc_s2_s3.py — MLMC-AB schedules S2 and S3 on 5D max-call
================================================================
Extends exp_mlmc_maxcall.py with two additional schedules:
  S2: coarse-fine 50/100/200, uniform allocation (50000/25000/12500)
  S3: geom 25/50/100, variance-optimal allocation via Giles (2008)

Uses S1 detail data (V_ℓ, C_ℓ) to compute S3 allocations.
"""
import os, sys, time
import numpy as np
import pandas as pd

# Add the replication code directory to path
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # all modules co-located here

# Also need exp_bbs_comparison for the shared infrastructure
# (path already set above)

from exp_bbs_comparison import (
    D, S0, K, R, DELTA, SIGMA, T_YEARS, N_STEPS, DT,
    DEGREE_LSM, N_TRAIN, N_OUT, N_RES,
    simulate_gbm_paths, maxcall_payoff, compute_std_stats,
    basis_stdlog_nd, lsm_train, hybrid_nestless, nested_ab,
)
from exp_mlmc_maxcall import (
    out_of_sample_lb, make_z_q, nested_pathwise_pair,
    N_SEEDS, SEED0, SEED_STEP,
)

# ============================================================
# Schedule definitions
# ============================================================
# S1 (reference, already computed): geom 25/50/100, alloc 50000/25000/12500
S1_LEVELS = [(25, 50000), (50, 25000), (100, 12500)]

# S2: coarse-fine 50/100/200, same uniform allocation
S2_LEVELS = [(50, 50000), (100, 25000), (200, 12500)]

# S3: geom 25/50/100, var-optimal allocation (computed from S1 V_ℓ/C_ℓ)
# Allocations filled after S1 empirical V_ℓ, C_ℓ are measured
S3_INNER = [25, 50, 100]  # same as S1
S3_TOTAL_BUDGET = 50000 + 25000 + 12500  # same total outer paths

def mlmc_ab_general(S_out, betas, means, stds, rng, levels):
    """General MLMC-AB with arbitrary (n_inner, n_outer) levels."""
    terms = []
    detail_rows = []

    n0, N0 = levels[0]
    y0, _ = nested_pathwise_pair(S_out[:N0], betas, means, stds, n0, rng)
    terms.append(y0)
    detail_rows.append(("level_base", N0, n0, float(y0.mean()), float(y0.var(ddof=1))))

    prev_n = n0
    for n_fine, n_outer in levels[1:]:
        y_fine, y_coarse = nested_pathwise_pair(
            S_out[:n_outer], betas, means, stds, n_fine, rng, n_coarse=prev_n
        )
        diff = y_fine - y_coarse
        terms.append(diff)
        detail_rows.append((
            f"diff_{n_fine}_{prev_n}",
            n_outer, n_fine,
            float(diff.mean()), float(diff.var(ddof=1)),
        ))
        prev_n = n_fine

    estimate = float(sum(term.mean() for term in terms))
    variance = float(sum(term.var(ddof=1) / len(term) for term in terms))
    se = float(np.sqrt(variance))
    return estimate, se, detail_rows


def measure_level_costs(S_out, betas, means, stds, seed):
    """Measure per-level V_ℓ and C_ℓ for Giles allocation."""
    rng = np.random.default_rng(seed + 8000)
    n_probe = 5000  # small probe for cost/variance estimation

    costs = []
    variances = []

    # Level 0: Y_25
    t0 = time.perf_counter()
    y0, _ = nested_pathwise_pair(S_out[:n_probe], betas, means, stds, 25, rng)
    c0 = (time.perf_counter() - t0) / n_probe
    v0 = float(y0.var(ddof=1))
    costs.append(c0)
    variances.append(v0)

    # Level 1: Y_50 - Y_25
    t0 = time.perf_counter()
    y_fine, y_coarse = nested_pathwise_pair(
        S_out[:n_probe], betas, means, stds, 50, rng, n_coarse=25
    )
    c1 = (time.perf_counter() - t0) / n_probe
    v1 = float((y_fine - y_coarse).var(ddof=1))
    costs.append(c1)
    variances.append(v1)

    # Level 2: Y_100 - Y_50
    t0 = time.perf_counter()
    y_fine, y_coarse = nested_pathwise_pair(
        S_out[:n_probe], betas, means, stds, 100, rng, n_coarse=50
    )
    c2 = (time.perf_counter() - t0) / n_probe
    v2 = float((y_fine - y_coarse).var(ddof=1))
    costs.append(c2)
    variances.append(v2)

    return variances, costs


def giles_allocation(variances, costs, total_budget):
    """Giles (2008) optimal allocation: N_ℓ ∝ √(V_ℓ / C_ℓ)."""
    weights = [np.sqrt(v / max(c, 1e-12)) for v, c in zip(variances, costs)]
    total_w = sum(weights)
    allocs = [max(int(total_budget * w / total_w), 100) for w in weights]
    return allocs


def run_one_seed(seed):
    rng_train = np.random.default_rng(seed)
    rng_out = np.random.default_rng(seed + 500)

    S_train = simulate_gbm_paths(
        N_TRAIN, D, N_STEPS, S0, R, DELTA, SIGMA, DT, rng_train
    )[0]
    means, stds = compute_std_stats(S_train)
    betas, indices, _ = lsm_train(S_train, DEGREE_LSM, means, stds)

    S_out = simulate_gbm_paths(
        N_OUT, D, N_STEPS, S0, R, DELTA, SIGMA, DT, rng_out
    )[0]
    V_LB = out_of_sample_lb(S_out, betas, means, stds)

    rows = []

    # --- S2: coarse-fine 50/100/200 ---
    rng_s2 = np.random.default_rng(seed + 5000)
    tic = time.perf_counter()
    V_s2, se_s2, details_s2 = mlmc_ab_general(S_out, betas, means, stds, rng_s2, S2_LEVELS)
    wall_s2 = time.perf_counter() - tic
    rows.append(dict(
        seed=seed, method="MLMC-AB (S2)",
        schedule="coarse-fine 50/100/200",
        levels_str="50000/25000/12500",
        V_LB=V_LB, V_UB=V_s2, SE=se_s2,
        gap_pct=(V_s2 - V_LB) / V_LB * 100.0,
        wall_s=wall_s2,
    ))

    # --- S3: var-optimal allocation ---
    # First measure V_ℓ / C_ℓ
    variances, costs = measure_level_costs(S_out, betas, means, stds, seed)
    allocs = giles_allocation(variances, costs, S3_TOTAL_BUDGET)
    s3_levels = list(zip(S3_INNER, allocs))

    rng_s3 = np.random.default_rng(seed + 6000)
    tic = time.perf_counter()
    V_s3, se_s3, details_s3 = mlmc_ab_general(S_out, betas, means, stds, rng_s3, s3_levels)
    wall_s3 = time.perf_counter() - tic

    alloc_str = "/".join(f"{a:,}" for a in allocs)
    rows.append(dict(
        seed=seed, method="MLMC-AB (S3)",
        schedule=f"var-opt 25/50/100",
        levels_str=alloc_str,
        V_LB=V_LB, V_UB=V_s3, SE=se_s3,
        gap_pct=(V_s3 - V_LB) / V_LB * 100.0,
        wall_s=wall_s3,
    ))

    return rows, allocs, variances, costs


def main():
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)

    all_rows = []
    all_allocs = []

    print("=" * 70)
    print("MLMC-AB Schedules S2 and S3 — 5D max-call")
    print(f"  S2: coarse-fine 50/100/200, uniform alloc 50k/25k/12.5k")
    print(f"  S3: geom 25/50/100, Giles (2008) var-optimal alloc")
    print(f"  Seeds: {N_SEEDS} (starting {SEED0})")
    print("=" * 70)

    for i in range(N_SEEDS):
        seed = SEED0 + SEED_STEP * i
        rows, allocs, variances, costs = run_one_seed(seed)
        all_rows.extend(rows)
        all_allocs.append(allocs)

        for r in rows:
            print(f"  seed={seed} {r['method']}: gap={r['gap_pct']:.2f}% "
                  f"wall={r['wall_s']:.2f}s alloc={r['levels_str']}")

    df = pd.DataFrame(all_rows)
    out_csv = os.path.join(out_dir, "exp_mlmc_s2_s3.csv")
    # Create the output directory before writing (a no-op when it already exists).
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(out_csv, index=False)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for method in ["MLMC-AB (S2)", "MLMC-AB (S3)"]:
        sub = df[df['method'] == method]
        gap_mean = sub['gap_pct'].mean()
        gap_se = sub['gap_pct'].std(ddof=1) / np.sqrt(len(sub))
        wall_mean = sub['wall_s'].mean()
        print(f"  {method}: gap={gap_mean:.2f} ± {gap_se:.2f}%  wall={wall_mean:.1f}s/seed")

    # S3 allocation summary
    mean_allocs = np.mean(all_allocs, axis=0).astype(int)
    print(f"\n  S3 mean allocation (Giles): {mean_allocs[0]:,} / {mean_allocs[1]:,} / {mean_allocs[2]:,}")
    print(f"  (Paper TODO placeholder: 60,000 / 22,000 / 8,000)")

    print(f"\nCSV: {out_csv}")


if __name__ == "__main__":
    main()
