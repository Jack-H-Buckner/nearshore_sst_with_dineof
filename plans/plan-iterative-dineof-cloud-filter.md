# Plan: iterative DINEOF-baseline cloud filter

## Context
The cloud/outlier detector currently scores each ECOSTRESS/Landsat scene against a smoothed **MODIS** composite. That runs in `build_cube.py`, uses `simple_outlier_detection.classify`, and writes the `*_dineof` channels. MODIS is sparse (123/365 days), coarse (1 km) and a night-time retrieval, so the baseline is weak exactly where cloud detection matters.

`prototypes/DINEOF/src/compare_lowrank.py` already shows the DINEOF low-rank field separates removed from kept pixels, but it names two biases in its docstring:
- the field was fit on data the MODIS filter had already cleaned;
- the field for day j is pulled toward day j's own pixels.

This plan tests whether that result holds without the bootstrap advantage. The loop:
1. Fit DINEOF.
2. Take the rank-k field for each day as the baseline.
3. Classify the raw scenes against it.
4. Mask the outliers.
5. Refit, and repeat until the flags stop changing.

The key question is how to schedule the DINEOF hyperparameters (T_c, k). The plan builds three schedules so cost and quality can be compared directly.

Decisions already made:
- iteration 0 starts from **raw scenes + sensor QC only**;
- detection runs **per sensor scene** in kelvin at native 100 m;
- flagged pixels are **hard-masked** into gaps.

## Approach

### New files
- `prototypes/DINEOF/src/iterative_filter.py`: the driver. Its CLI follows `edineof.main`: `--config --tag --max-iter --strategy --no-figures`.
- `prototypes/DINEOF/configs/config.iterative.admiralty_inlet.yaml`
- `prototypes/DINEOF/tests/test_iterative_filter.py`

### Reuse (no changes to these modules)
**`edineof.py`**
- `load_config`/`validate` pattern: strict `DEFAULTS`, where an unknown key raises.
- `select_matrix` (:927), `unstandardize` (:989) and `upsample` (:857).
- `filter_settings` (:329), `fill` (:257), `top_k_modes` (:165), `reconstruct` (:204).
- `edineof` (:479) for full CV searches.
- The `write_cube`/`build_dataset` provenance pattern.

**`composite.py`**
- `composite()` (:704) re-averages the masked members.
- The fitted per-date offsets are read from the composite cube's `<id>_offset` channels. The fallback is `offset_mean` in `data/datacube/offsets.csv`, the same approach `compare_lowrank.py` uses. Apply the slope if `offset.slope` was on.

**`cloud_mixture_model/src/simple_outlier_detection.py`**, via the same `sys.path` insertion `edineof.py:71-75` uses:
- `qc_masks` (:168), `classify` (:219) and `flag_offset` (:259).
- `classify(y, land, tidal, covariate=<DINEOF field in K>, ...)` is used unchanged.
- The mixture/clear/qc config sections are copied from `simple_outlier_detection.yaml`.

**Data sources**
- Raw scenes: `admiralty_inlet_filtered.zarr` (`eco_sst_v002`, `lst_sst`, QC channels, hours).
- Seasonal coefficients, scale and `validation_msk`: `admiralty_inlet_standardized.zarr`.
- Offsets: the composite cube.

### Algorithm (`run_loop`)
**0. Setup (once)**
- Load the raw member stacks and put them on the composite scale: `raw − offset(t)`.
- Build the QC masks per scene.
- Hold these fixed for the whole run:
  - the seasonal coefficients and scale (standardization is not refit inside the loop);
  - the offsets;
  - `validation_msk`.
- MODIS stays a composite member but is never a detection baseline.

**1. Build the matrix from the current keep-masks**
- `adj_masked[s] = where(keep[s], adj[s], nan)`, then `composite()`, then `z = (comp − seasonal)/scale`.
- Wrap the result in an in-memory copy of the standardized dataset with `sst_z` replaced.
- Set `validation_msk &= finite(z)`, so flagged pixels leave the CV point set.
- Call `select_matrix(ds_iter, cfg)`. This reuses coarsening, thin-date masking and the pixel drop exactly as they are.

**2. Fit DINEOF according to the schedule (`loop.strategy`)**

| Strategy | In the loop | At the end |
|---|---|---|
| `fixed` | `fit_fixed` at the config's `loop.k` and `loop.t_c` | one full `edineof()` CV search |
| `ends` | full `edineof()` CV at iteration 0, then `fit_fixed` at the settings it chose | full CV again |
| `every` | full `edineof()` CV every iteration | none needed |

- `fit_fixed` mirrors `final_fit` inside `edineof` (centre, `fill`, `top_k_modes`, `reconstruct`).
- `fit_fixed` warm-starts the gaps from the previous iteration's analysis. This is the main saving: per the timings, EM iteration count is dominated by temporal diffusion.
- `loop.baseline: day | point` chooses which CV fit's `lowrank` is used as the baseline under `ends` and `every`. The default is `day`, because the smoother day-opt field (k≈2, T_c≈8 on this AoI) absorbs less contamination than point-opt (k≈15, T_c=0).

**3. Baseline**
- `unstandardize(lowrank)`, then `upsample` to native if `coarsen > 1`.
- This gives K on (time, y, x).
- Use the **low-rank** field, not the analysis `X`, since `X` holds the observations themselves.

**4. Classify every raw scene, from scratch each iteration**
- `classify(y=adj_raw[s][j], covariate=baseline[j], ...)` with the QC priors from `qc_masks`.
- `keep = p_valid ≥ loop.p_threshold` (default 0.5) and the pixel passed QC.
- Drop a whole scene when `flag_offset(center) != 0`.
- The flags are not cumulative, so a pixel wrongly flagged in an early iteration can come back.

**5. Convergence**
- Compute the fraction of flags that flipped relative to the previous iteration, per sensor.
- Stop when it falls below `loop.tol` (default 1e-3) for `patience` iterations, or when `max_iter` (default 8) is reached.
- Log a warning if oscillation is detected, i.e. the flip count stops shrinking.

**6. Write outputs**
- `data/datacube/admiralty_inlet_iterative[_tag].zarr`: a copy of the filtered cube's channels plus `<var>_iter` (the raw channel masked by the final keep) and `<id>_p_valid_iter`.
  - `<var>_iter` is a drop-in for `eco_sst_v002_dineof`/`lst_sst_dineof`, so the existing `composite → standardize → edineof` chain can be re-run on it just by pointing `members.*.var` at it.
  - This is the honest "select DINEOF params once at the end" path, because it also refits standardization.
- `iterations.csv`, one row per iteration, with columns:
  - `iter, strategy, k, t_c, fit_seconds, classify_seconds, n_kept_eco, n_kept_lst, flip_frac_eco, flip_frac_lst, cloud_frac_mean, rmse_point, rmse_day`
  - the RMSE columns are filled only when CV ran.
- Figures under `figures/iterative[_tag]/admiralty_inlet/`, reusing `plotting` and `cube_figures` styles:
  - flips and kept-count against iteration;
  - for 6 busy days: raw | baseline | residual | q_cloud | final keep vs the MODIS-filter keep.

### Evaluation (inside the script, `--compare`)
1. **Agreement with the MODIS-baseline filter.** Per scene, kept/removed agreement and Cohen's κ against `*_dineof` finiteness. Report the scenes where the two disagree most, so they can be inspected.
2. **AUC of the final residual.** Compute it the way `compare_lowrank.auc` does, but now without bootstrap bias. Import the function; do not copy it.
3. **Downstream skill.** Compare the final `edineof()` point/day CV RMSE with `dineof_cv*.csv` from the MODIS-filtered run. Score on the **intersection** of the two CV point sets, since the held-out points differ.
4. **Cost.** Wall time per iteration for each strategy. The decision rule for which schedule to adopt should come from `iterations.csv` for the three strategies run side by side (`--tag fixed|ends|every`).

### Tests (`tests/test_iterative_filter.py`, synthetic)
- A rank-2 field with irregular days, plus injected cold patches (−4 K, spatially coherent). The loop should recover the patches with AUC > 0.95, and flips should reach 0 within `max_iter`.
- A clean field: nothing is flagged beyond the clear-tail rate, and iteration 1 has no flips.
- Restoration: seed an over-aggressive iteration-0 mask and check the pixels return.
- `fit_fixed` at k, T_c matches `edineof`'s `final_fit` on the same gap set, to within tol.

## Cost expectations (from the existing timings)
- A fixed fit takes about 8–25 s at `coarsen: 4`. At full resolution it is ~6× slower per EM iteration.
- A full CV search takes about 45–200 s.
- Classifying ~212 scenes with Ising sweeps is expected to take tens of seconds.
- So `fixed`/`ends` should run roughly 0.5–1.5 min per iteration, and `every` 1.5–4 min.
- The default is `coarsen: 4` for the loop. Run full resolution once at the end through the existing chain.

## Verification
1. `pytest prototypes/DINEOF/tests/test_iterative_filter.py prototypes/DINEOF/tests/test_edineof.py` (the existing tests must still pass).
2. Smoke run: `python prototypes/DINEOF/src/iterative_filter.py --config prototypes/DINEOF/configs/config.iterative.admiralty_inlet.yaml --max-iter 2 --tag smoke`. Check that the zarr, CSV and figures are written and that the flip fraction falls.
3. Full runs with `--strategy fixed|ends|every` and distinct tags. Compare the `iterations.csv` files, the κ and AUC against the MODIS filter, and the downstream CV RMSE.
4. Re-run `composite.py` → `standardize.py` → `edineof.py`, pointed at the `_iter` channels. Compare the result with the current `admiralty_inlet_dineof.zarr`.

## Open risks (to watch, not to block)
- **Self-fit leakage.** The day-j baseline still sees day j's contaminated pixels in the first iteration. With low k and T_c > 0 this should be small. If AUC stalls, the fallback is a leave-day-out baseline for heavily clouded days only (`compare_recon.py` has the machinery), at a cost of about 2 min per day.
- **Fixed seasonal and scale coefficients.** These come from the MODIS-filtered composite. That is acceptable inside the loop; the end-of-run chain refits them.

## Implementation notes (found after approval, before any code was written)
- **Cube size.** The cubes now span **912 days** (2023-09-01 → 2026-02-28), not 365. At coarsen 4 the matrix is about 3,694 px × 912.
- **Where to read raw scenes and QC.** Use the SOURCE cube `prototypes/cloud_mixture_model/data/datacube/admiralty_inlet.zarr`, which has `eco_sst_v002`, `eco_valid_v002`, `eco_cloud_v002`, `lst_sst`, `lst_valid`, `lst_cloud`, the hours and `depth_cudem`. The filtered cube has no QC channels. Both cubes share the same time axis.
- **`build_cube.py` is the closest template.** Mirror its config shape (`data`, `detector`, `sensors`, `filter`) and reuse these by import:
  - `validate_detector`, `validate_sensors`
  - `detector_cfg`, which builds the per-sensor `sod` config, including `qc_nodata_prior`
  - `scene_verdict`, the offset band gate on `center`
  - The current band is [-2, +4] K and the p_valid cut is 0.5.
- **What to pass to `classify`.** Pass the raw scene (sensor scale) as `y` and the DINEOF baseline (composite / MODIS-night scale) as the covariate. `center` absorbs the sensor offset, so the offset band keeps its build_cube meaning.
- **Offsets.** All offsets in `data/datacube/offsets.csv` are constant (K=0): eco +1.1219, lst +0.6964, modis 0. So `adj = raw − offset_mean` when compositing.
- **Validation mask.**
  - `validation_msk` comes from `composite.composite_with_validation`, via `composite.scratch_pad`.
  - Recover the validation dates as `validation_msk.any(axis=(1,2))`.
  - Rebuild it each iteration with `composite_with_validation(adj_masked, valid_inds, ccfg, order)`.
  - Standardize reads `sst_composite` (the full composite) plus `validation_msk`. edineof holds out the msk pixels as its point CV set.
- **Baseline construction.** Upsample the low-rank **z** from the coarse grid, then unstandardize with the NATIVE coef and scale. This avoids the damped coarse scale. Where the baseline is NaN, fill with climatology (z = 0).
- **`fit_fixed` details.**
  - For `sd`, use `std(X0[observed])`.
  - Seed the gaps from the previous analysis in z.
  - Zero the all-gap columns when T_c = 0; this is the same guard as `apply_seed` in edineof.
- **Current selections and costs** (`dineof_cv.csv`, coarsen 4):
  - point: k=2, T_c=[0.01, 3] (≈1.09 d)
  - day: k=2, T_c=7
  - A full CV search takes about 200 s; a single fixed fit takes about 2–10 s.
  - Default for the `fixed` strategy: k=2, T_c=7.
- **Environment.** Base miniconda python has `coastal_sst_data` and pytest.
- **Pre-existing test failure.** `tests/test_edineof.py::test_7_synthetic_rank5_recovery` fails with a TypeError at line 151, most likely a stale test signature after the `valid_msk` argument was added to `edineof()`. Check it before relying on that suite for verification.
