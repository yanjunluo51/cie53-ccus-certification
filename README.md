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
| `code/drivers/` | Drivers for the retrofit instance, the max-call check and the sensitivity sweeps |
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
| Table 2 retrofit gaps 43.5 / 34.2 / 33.1, ρ_res 0.269. Section 6.1 bounds 2.670, 3.553 and 3.829, gaps 32.3 (200 draws) and 38.6 (10 draws) | `CCUS_OUTPUT=../results/headline.csv python3 run_guarded.py drivers/ccus_canonical_headline.py` |
| Section 6.1 one-sided 95% limit 2.642 (mean of the ten per-seed `strat50` lower estimates less t(9, 0.95) times their standard error) | computed from the output of the command above |
| Section 5.2 wall times (23.9 s per seed for the 100-draw nested run, and the two wall-time ratios) | the same driver with `CCUS_WORKERS=1 CCUS_METHODS=plain50,strat50,nested100` |
| Table 2 max-call row, Section 5.1 bounds 25.40 / 27.56, paired t = 11.95 | `cd mlmc_maxcall && PYTHONPATH=. python3 ../run_guarded.py exp_mlmc_maxcall.py`, then `PYTHONPATH=. python3 ../run_guarded.py exp_mlmc_s2_s3.py` |
| Section 4.5 Spearman 0.98 between the residual share and the variance ratio | `RR_OUTPUT=../results/rho.csv python3 run_guarded.py drivers/rho_res_vs_rho_var.py` |
| Section 4.6 residual variance reduced by a factor of 8 to 80 | `RV_OUTPUT=../results/rv.csv python3 run_guarded.py drivers/residual_variance_check.py` |
| Section 6.2 transition intensity ±20% | `python3 run_guarded.py drivers/regime_intensity_sensitivity.py` (writes `../out/regime_intensity_sensitivity.csv`) |
| Section 6.2 regime-dependent drift (+46% and +23% on the lower bound, share moves under 0.006) | `DRIFT_OUTPUT=../results/drift.csv python3 run_guarded.py drivers/ccus_regime_drift.py` |
| Section 6.1 asymptotic nested gap 29.5% | `python3 scripts/inner_budget_frontier.py` |
| Table 2 geometric-Asian row and its rerun | `GEO_OUTPUT=geoasian/results/rerun.csv python3 run_guarded.py geoasian/newfamily_geoasian_tableEC15.py` |

The geometric-Asian row of Table 2 is carried over from an archived run
(`code/geoasian/results/geoasian_9point_canonical.csv`). Rerunning the script above at its
default configuration gives `geoasian_9point_rerun_20260831.csv`. That rerun reproduces the
archived nested upper estimate to within 1% and returns a hybrid upper estimate 12% lower,
as reported in Section 5.2. Without `GEO_OUTPUT` the script overwrites the archived file.

Several results are shipped as files only:

| Paper | File |
|---|---|
| Section 5.1 variance rate (slope −1.05, 95% CI [−1.12, −0.97], R² 0.99) | `results/varrate_single_regime_ccus.csv` (from `scripts/varrate_single_regime_ccus.py`) |
| Section 5.3 volatility-level sweep and the solid curve of Figure 2 | `results/sigma_p_sweep_canonical.csv`, `results/prebudget_routing_settings.csv` |
| Section 5.3 screened instances on the correct side of the crossing | `results/routing_accuracy_canonical.csv` |
| Section 6.1 nested inner-budget sweep fitted by the frontier script | `results/nested_inner_sweep_canonical.csv` |

The variance-optimal multilevel schedule allocates outer paths from per-level costs measured
during the run, so its gap moves slightly between runs and machines (6.97 shipped, 7.00 and
6.96 on repeat runs). All other rows reproduce exactly.

Inner random streams are assigned by the position of a method in the method list, so runs
with a different `CCUS_METHODS` list differ in the fifth significant figure.
`drivers/ccus_regime_sensitivity.py` and `drivers/ccus_regime_drift.py` each rerun their
unperturbed case as a control, which must reproduce the headline driver.

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
