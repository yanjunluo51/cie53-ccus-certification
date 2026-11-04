#!/usr/bin/env python3
"""Section 6.2 sensitivity: regime-transition intensity scaled by +/-20%.

Only H.PARAMS.Q_regime is changed; everything else follows the headline driver.
The scaling acts on the off-diagonal entries and the diagonal absorbs the rest,
so every row still sums to one.

Built-in check: the mult=1.0 run must match the single-thread baseline bit for bit.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "shared"))
import ccus_canonical_headline as H

OUT = HERE.parents[1] / "out"
SEEDS = [1000 + i for i in range(int(os.environ.get("RIS_SEEDS", "10")))]
MULTS = [0.8, 1.0, 1.2]
BASE_Q = np.asarray(H.PARAMS.Q_regime, dtype=float).copy()
# The claim concerns plain50 / strat50 / nested100 only, so the other budgets are not run.
H.METHODS = ("plain50", "strat50", "nested100")


def scale_generator(Q, mult):
    """Scale the off-diagonal switching intensity; the diagonal absorbs the rest so rows sum to one."""
    out = Q.copy()
    for i in range(out.shape[0]):
        off = [j for j in range(out.shape[1]) if j != i]
        scaled = out[i, off] * mult
        if scaled.sum() >= 1.0:
            raise ValueError(f"row {i}: off-diagonal mass {scaled.sum():.3f} ≥ 1")
        out[i, off] = scaled
        out[i, i] = 1.0 - scaled.sum()
    return out


def run(mult):
    H.PARAMS.Q_regime = scale_generator(BASE_Q, mult)
    rows = []
    for s in SEEDS:
        rows.extend(H.run_seed(s))
    for r in rows:
        r["Q_mult"] = mult
    return rows


def main():
    print("=" * 92)
    print(" Regime-transition-intensity sensitivity (Q_regime off-diagonals scaled)")
    print(f" {len(SEEDS)} seeds, multipliers {MULTS}, methods {H.METHODS}")
    print("=" * 92)
    all_rows = []
    for m in MULTS:
        t0 = time.perf_counter()
        rows = run(m)
        all_rows.extend(rows)
        print(f"  mult={m}  done ({time.perf_counter()-t0:.0f}s)")
    H.PARAMS.Q_regime = BASE_Q.copy()

    d = pd.DataFrame(all_rows)
    # Create the output directory before writing.
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "regime_intensity_sensitivity.csv", index=False)

    # --- Check: mult=1.0 must match the baseline run with the SAME METHODS tuple ---
    # Inner RNG streams are assigned by the position of a method in METHODS
    # (zip(METHODS, seed_seq.spawn(len(METHODS))) in run_seed), so only runs with the
    # same methods in the same order are comparable bit for bit. The matching baseline
    # is the single-thread timing run ("plain50","strat50","nested100"), not the
    # seven-method headline run (same V_LB, different V_UB).
    base = pd.read_csv(HERE.parents[1] / "results/timing_1thread_20260825.csv",
                       float_precision="round_trip")
    mine = d[d.Q_mult == 1.0]
    ok = True
    for meth in sorted(set(mine.method)):
        a = mine[mine.method == meth].sort_values("seed")
        b = base[base.method == meth].sort_values("seed")
        if len(b) == 0: continue
        for col in ("V_LB", "V_UB", "gap_pct"):
            same = np.array_equal(a[col].to_numpy(), b[col].to_numpy()[:len(a)])
            print(f"  {'PASS' if same else 'FAIL'}  bit-identical mult=1.0 {meth}.{col}")
            ok &= same
    print(f"  baseline check {'PASSED' if ok else 'FAILED'}")

    print("\n gap / V_LB / rho_res change with intensity (relative to mult=1.0)")
    g = d.groupby(["Q_mult", "method"]).agg(
        gap=("gap_pct", "mean"), V_LB=("V_LB", "mean"), rho=("rho_res", "mean")).reset_index()
    ref = g[g.Q_mult == 1.0].set_index("method")
    for _, r in g.iterrows():
        dg = r["gap"] - ref.loc[r["method"], "gap"]
        dv = (r["V_LB"] / ref.loc[r["method"], "V_LB"] - 1) * 100
        print(f"   mult={r['Q_mult']:.1f} {r['method']:<10} gap={r['gap']:7.3f} "
              f"(Δ{dg:+.3f} pp)  V_LB={r['V_LB']/1e9:.4f}e9 (Δ{dv:+.2f}%)  rho={r['rho']:.4f}")
    mx = max(abs(r["gap"] - ref.loc[r["method"], "gap"]) for _, r in g.iterrows())
    mv = max(abs(r["V_LB"] / ref.loc[r["method"], "V_LB"] - 1) * 100 for _, r in g.iterrows())
    print(f"\n  max gap shift {mx:.3f} pp, max V_LB shift {mv:.2f}%")
    json.dump({"max_gap_shift_pp": mx, "max_VLB_shift_pct": mv,
               "seeds": len(SEEDS), "mults": MULTS, "baseline_bitidentical": bool(ok)},
              open(OUT / "regime_intensity_sensitivity.json", "w"), indent=1)
    print(f"\nCSV: {OUT/'regime_intensity_sensitivity.csv'}")


if __name__ == "__main__":
    main()
