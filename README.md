# Code for "Pre-budget screening for timing certification of carbon capture, utilization, and storage retrofit investment in China"

Yanjun Luo, Yu Zhang, Nan Sun, Lawrence Loh, Qiang Mai.
*CIE53 Proceedings*, Khalifa University, Abu Dhabi, 20–23 October 2026.

This repository holds the code, configuration and result files behind the paper's
computational study. The paper refers to it as the "released code" and the "released configuration".

## Environment

Python 3.9.6, NumPy 2.0.2, SciPy 1.13.1, pandas 2.3.3, macOS arm64 (Apple Accelerate).
`pip install -r requirements.txt` installs the pinned packages.
Wall times in the paper were measured single-threaded on an Apple M4 Pro
(set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and `VECLIB_MAXIMUM_THREADS` to 1).
Gaps and bounds do not depend on the hardware.

Run every command from `code/`.

## Layout

| Path | Content |
|---|---|
| `code/shared/` | Model, simulator, least-squares Monte Carlo training, nested and hybrid dual estimators |
| `code/drivers/` | Drivers for the retrofit instance, the residual-share checks and the transition-intensity sweep |
| `code/mlmc_maxcall/` | Five-asset Bermudan max-call rows of Table 2 and Section 5.1 |
| `code/geoasian/` | Geometric-Asian row of Table 2 and its rerun (Section 5.2) |
| `code/scripts/` | Inner-budget frontier, variance-rate check, comparator audit |
| `data/` | Calibrated parameters (Table 1), regime calibration and quarterly regime labels |
| `results/`, `out/` | Result files as shipped with the paper |

## Numerical guard

`code/run_guarded.py` wraps each driver. It turns every `RuntimeWarning` into an error
except the spurious floating-point flags that Apple Accelerate raises inside `matmul`.
At start-up it checks that `A.T @ A` still agrees exactly with `einsum`, and that genuine
divide-by-zero, overflow and invalid operations still abort. For the same reason, ten `@`
operators in `shared/` were replaced by `np.dot`, which is the same BLAS call on this platform.
`code/patch_manifest.json` records the MD5 of each edited file before and after the edit.

## Commands and the numbers they produce

| Paper | Command |
|---|---|
| Table 2 retrofit row (ρ_res 0.269, gaps 43.5 / 34.2). Section 5.2 stratified gap 33.1. Section 6.1 bounds 2.670, 3.553 and 3.829, gaps 32.3 (200 draws) and 38.6 (10 draws) | `CCUS_OUTPUT=../results/headline.csv python3 run_guarded.py drivers/ccus_canonical_headline.py` |
| Section 6.1 one-sided 95% limit 2.642 (mean of the ten per-seed `strat50` lower estimates less t(9, 0.95) times their standard error) | computed from the output of the command above |
| Section 5.2 retrofit wall times (23.9 s per seed for the 100-draw nested run, stratified hybrid at 70% of it) | the same driver with `CCUS_WORKERS=1 CCUS_METHODS=plain50,strat50,nested100` (shipped output `results/timing_1thread_20260825.csv`) |
| Table 2 max-call row, Section 5.1 bounds 25.40 / 27.56, Section 5.2 paired t = 11.95 and the plain hybrid at half the nested wall time | `cd mlmc_maxcall && PYTHONPATH=. python3 ../run_guarded.py exp_mlmc_maxcall.py` (writes `mlmc_maxcall/results/`) |
| Section 4.5 Spearman 0.98 between the residual share and the variance ratio | `RR_OUTPUT=../results/rho.csv python3 run_guarded.py drivers/rho_res_vs_rho_var.py` |
| Section 4.6 residual variance reduced by a factor of 8 to 80 | `RV_OUTPUT=../results/rv.csv python3 run_guarded.py drivers/residual_variance_check.py` |
| Section 6.2 transition intensity ±20% (gap within one point, lower bound within 0.5%, share inside (0.12, 0.30)) | `python3 run_guarded.py drivers/regime_intensity_sensitivity.py` (writes `../out/regime_intensity_sensitivity.csv`) |
| Section 6.1 asymptotic nested gap 29.5% (R² 0.98) | `python3 scripts/inner_budget_frontier.py` (reads `results/nested_inner_sweep_canonical.csv`) |
| Section 5.1 variance rate (slope −1.05, 95% CI [−1.12, −0.97], R² 0.99) | `python3 scripts/varrate_single_regime_ccus.py` (long run, shipped output `results/varrate_single_regime_ccus.csv`) |
| Table 2 geometric-Asian row and its rerun | `GEO_OUTPUT=geoasian/results/rerun.csv python3 run_guarded.py geoasian/newfamily_geoasian_tableEC15.py` |

The headline, max-call, Spearman, residual-variance, intensity and frontier commands were
rerun from this layout on 31 August 2026 and reproduced the numbers listed above.

The geometric-Asian row of Table 2 is carried over from an archived run
(`code/geoasian/results/geoasian_9point_canonical.csv`). Running the script above at its
default configuration, the released configuration of Section 5.2, gives
`geoasian_9point_rerun_20260831.csv`. At σ = 1.5 the rerun reproduces the archived nested upper
estimate to within 1% and returns a hybrid upper estimate 12% lower. Without `GEO_OUTPUT` the
script overwrites the archived file.

Two result files are shipped without a generating script in this repository.
`results/sigma_p_sweep_canonical.csv` holds the Section 5.3 volatility-level sweep
(Spearman 1, log–log R² 0.76) and `results/prebudget_routing_settings.csv` its gap ratios
(1.05 to 1.86 on the calibrated instance). `results/clipping_audit_20260825.csv` comes from
`scripts/audit_ccus_nested_clipping.py` and is not reported in the paper.

## Scope

This repository holds the code and configuration released with the paper. It does not
regenerate every number in the text. The following are not produced here: the one- and
two-dimensional reference checks of Section 5.1, the one-dimensional and two-regime put rows
of Table 2, the term correlations of Section 4.5, the probe-run checks and the jump family of
Section 5.3, the matched 50-draw comparator and the basis sweep of Section 5.2, the
fixed-rule limits of Section 6.1, and the drift and volatility sweeps of Section 6.2.

Inner random streams are assigned by the position of a method in the method list, so runs
with a different `CCUS_METHODS` list differ in the fifth significant figure.

## Comparator

The certificate and its nested comparator share one un-clipped continuation target
`Q_t = max(Z_t, C_hat_t)`. The lower bound is the trained rule evaluated on outer paths
drawn independently of training. Clipping is available only through
`scripts/audit_ccus_nested_clipping.py`, as a separate comparator.

## Data

The code carries the calibrated values of Table 1 in `make_params()` of `code/shared/nestless_3d_calibrated.py`. `data/calibrated_params.json` and `data/regime_calibration.json` record the same calibration. The daily China
Emissions Allowance settlement prices used to estimate the initial price and σ_P are
third-party data and are not redistributed here. The 682 policy documents behind the regime
index are not redistributed either. `data/pui_regime_labels.csv` gives the quarterly regime
labels derived from them.

Figure 1 was drawn in draw.io and is not produced by this code.
