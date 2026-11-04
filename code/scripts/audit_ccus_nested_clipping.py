"""Paired audit of the historical clipped and canonical CCUS nested AB.

Both rows share the trained policy, outer paths, and inner random numbers.  The
only changed quantity is whether the continuation is clipped before forming
``Q_t=max(Z_t,C_hat_t)``.  This isolates the 21.95% versus 29.78% discrepancy.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))

from ccus_core.core.ccus_model import simulate_paths  # noqa: E402
from ccus_nested_ab import canonical_nested_ab  # noqa: E402
from nestless_3d_calibrated import (  # noqa: E402
    lsm_3d_stdlog, evaluate_lsm_policy, make_params)
from sample_splitting import SampleSplittingRNGs  # noqa: E402


def main():
    seed = int(os.environ.get("AUDIT_SEED", "1000"))
    n_train = int(os.environ.get("AUDIT_NTRAIN", "30000"))
    n_out = int(os.environ.get("AUDIT_NOUT", "30000"))
    n_in = int(os.environ.get("AUDIT_NIN", "200"))
    out_csv = Path(os.environ.get(
        "AUDIT_OUTPUT", ROOT / "results" / "ccus_nested_clipping_audit.csv"
    ))

    params = make_params()
    streams = SampleSplittingRNGs.from_seed(seed)
    train_ss = streams.training.bit_generator.seed_seq
    _, betas, stats, _, _, _ = lsm_3d_stdlog(
        params, n_train, 3, seed=train_ss, eps=0.05
    )
    P, C, R = simulate_paths(params, n_out, rng=streams.outer)
    V_LB, _ = evaluate_lsm_policy(params, P, C, R, betas, 3, stats)
    inner_ss = streams.inner.bit_generator.seed_seq.spawn(1)[0]

    rows = []
    for label, clip in [("canonical_unclipped", None), ("historical_clipped", 5.0e9)]:
        rng = np.random.default_rng(inner_ss)
        start = time.perf_counter()
        V_UB, se = canonical_nested_ab(
            params, P, C, R, betas, stats, n_in, rng,
            degree=3, continuation_clip=clip,
        )
        wall = time.perf_counter() - start
        rows.append({
            "seed": seed,
            "variant": label,
            "continuation_clip": "none" if clip is None else clip,
            "N_train": n_train,
            "N_out": n_out,
            "N_in": n_in,
            "V_LB": V_LB,
            "V_UB": V_UB,
            "SE": se,
            "gap_pct": (V_UB - V_LB) / V_LB * 100.0,
            "wall_s": wall,
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out_csv, index=False)
    print(frame.to_string(index=False))
    print(f"\npaired gap difference (unclipped - clipped): "
          f"{frame.gap_pct.iloc[0] - frame.gap_pct.iloc[1]:.4f} pp")
    print(f"CSV: {out_csv}")


if __name__ == "__main__":
    main()
