# Plan: build a filtered DINEOF input cube from simple_outlier_detection

> On approval, step 0 is to copy this file to
> `prototypes/DINEOF/plans/plan-build-cube.md` (plan mode only permits editing this scratch
> path, so it lands there as the first implementation action).

## Context

The scene-level criteria filter built earlier (`filter_images.py` + `config.filtering.*.yaml`)
is being **retired as the gate**. The evidence that killed it: on 2025-03-07 the criteria
scored the scene 0.970 valid / 0.986 clear — the cleanest in its window — while
`simple_outlier_detection` flagged 26.4% of the accepted pixels, and **100% of those flagged
pixels are called clear by `eco_cloud_v002` on every date tested**. Haze is invisible to the
QC and cloud rasters the criteria read, so no threshold on them can reach it.

`simple_outlier_detection.py` can see it, and it produces exactly two verdicts the pipeline
needs: a per-pixel `p_valid` and a per-scene `center` offset against the MODIS covariate.

This plan builds the formal process: run that detector over every acquisition, apply a pixel
threshold and a scene offset band, and write a **new zarr cube** holding the surviving
ECOSTRESS and Landsat SST plus carried-over channels (MODIS and the statics), ready for DINEOF.

`filter_images.py` is **not deleted** — it stays as a visual QA tool for looking at scenes —
but nothing in this pipeline depends on it.

## What gets built

```
prototypes/DINEOF/
  configs/config.build_cube.admiralty_inlet.yaml   # new: the ONE config
  src/build_cube.py                                # new: the driver
  outputs/outlier_cache/{eco,lst}/*.nc             # stage 1 cache (+ offset_flags.csv)
  data/datacube/admiralty_inlet_filtered.zarr      # the deliverable
```

## Two stages, with a cache

**Stage 1 — detect.** For each sensor, find its acquisitions and run
`simple_outlier_detection.run_date` on each, writing one `.nc` per date into
`outputs/outlier_cache/<sensor>/`. Skipped for dates already cached.

**Stage 2 — threshold and assemble.** Read the cached `.nc` files, apply the thresholds, and
write the cube.

The split is what makes the thresholds tunable: re-running the mixture model over 212
acquisitions costs ~9 minutes, while re-thresholding from the cache costs seconds. Crucially
the builder gates on the **`center` attribute directly**, not on the detector's precomputed
`offset_flag`, so changing the accepted band does *not* invalidate the cache.

**Cache key.** Each `.nc` already embeds its config as a YAML string attr (`write_netcdf`,
`simple_outlier_detection.py:338`). The builder compares the *detector* section of the current
config against that attr and re-runs any date whose detector settings changed. Threshold-only
edits leave the cache valid. `--refresh` forces a full re-run.

## Config

One file, `configs/config.build_cube.admiralty_inlet.yaml`:

```yaml
data:
  aoi: admiralty_inlet
  cube: ../cloud_mixture_model/data/datacube/admiralty_inlet.zarr   # source
  out:  data/datacube/admiralty_inlet_filtered.zarr                 # destination
  watervar: landcover_water

# Passed through to simple_outlier_detection verbatim. Section names and keys match its
# DEFAULTS exactly (reference / covariate / clear / qc / offset / mixture / solver), so this
# block is its config minus the per-sensor `data.*` names and its `output.*`.
detector:
  reference:  {window_days: 4, qc_var: null, qc_max: 1, quantile: 0.5, min_obs: 1,
               min_cover: 0.05}
  covariate:  {init_prior: 285.0, init_range_px: 200.0, init_marg_sd: 0.05,
               init_obs_error: 0.5, alpha: 2}
  clear:      {sd: 0.75, sd_floor: 0.71}
  qc:         {enabled: true, prior_cloud: 0.9, nodata_prior: null}   # per-sensor override
  mixture:    {lambda_cloud: 4.0, prior_cloud: 0.5, lambda_hot: 6.5, prior_hot: 0.005,
               p_h_tidal: 0.5, beta_cloud: 1.75, n_sweeps_cloud: 20, beta_hot: 0.125,
               n_sweeps_hot: 10, estep_damp: 0.5, p_floor: 0.01}
  solver:     {tol: 1.0e-2, pcg_max_iter: 25, precondition: true}
  ref_var: modis_sst_aqua
  depthvar: depth_cudem
  tidal_depth_m: 3.0

sensors:
  eco:
    label: ECOSTRESS v002
    sst: eco_sst_v002
    valid: eco_valid_v002
    cloud: eco_cloud_v002
    min_pixels: 64
    # ECOSTRESS drops QC-rejected pixels rather than keeping them, so its gaps carry a usable
    # cloud/no-data distinction; Landsat's do not. This is the ONE detector key that differs
    # between the two sensors.
    qc_nodata_prior: 0.5
  lst:
    label: Landsat
    sst: lst_sst
    valid: lst_valid
    cloud: lst_cloud
    min_pixels: 64
    qc_nodata_prior: null

filter:
  p_valid_min: 0.8          # per-PIXEL: keep pixels with P(clear) >= this
  offset_lower: -2.0        # per-SCENE: keep scenes whose `center` vs the MODIS covariate
  offset_upper:  4.0        #            falls in [lower, upper] degrees C. null disables.
  require_reference: false  # see "A flagged risk" below -- shipped off, one keystroke to arm
  min_kept_frac: null       # null = no minimum; e.g. 0.20 to drop near-empty survivors

# Channels copied from the source cube UNCHANGED. This is the "what else comes over" list.
carry:
  - modis_sst_aqua          # the coarse daily reference DINEOF is anchored against
  - landcover_water
  - depth_cudem
  - elevation_cudem

output:
  chunks: {time: 64, y: 128, x: 128}
  compression: {codec: zstd, level: 5, shuffle: shuffle}
  cache_dir: outputs/outlier_cache
  write_detector_figures: false     # the 6-panel diagnostic per date; off for a full run
```

Validation follows the house pattern from
[outlier_detection.py:121-140](prototypes/cloud_mixture_model/src/outlier_detection.py#L121-L140):
a strict two-level walk where an unknown section or key is an error. `detector` and `sensors`
are `OPAQUE_SECTIONS` (deeper than two levels), validated by dedicated functions —
`validate_detector` checks every key against `simple_outlier_detection.DEFAULTS` so a typo
cannot reach the model as a silent default, and `validate_sensors` checks each sensor block
and that every named channel exists in the source cube.

## Driver — `src/build_cube.py`

```
# ============ config
DEFAULTS, SENSOR_DEFAULTS, OPAQUE_SECTIONS
load_config(path) -> dict
validate_detector(cfg, path) -> None        # against simple_outlier_detection.DEFAULTS
validate_sensors(cfg, ds, path) -> None
detector_cfg(cfg, sid, ds) -> dict          # our config -> the detector's own cfg dict

# ============ stage 1: detect
cache_path(cfg, sid, date) -> Path
cache_is_current(path, dcfg) -> bool        # compares the embedded config YAML attr
detect_sensor(ds, cfg, sid, refresh) -> list[Path]

# ============ stage 2: threshold
scene_verdict(nc_attrs, cfg) -> (keep: bool, reason: str)
masked_stack(ds, cfg, sid, paths) -> (np.ndarray float32 (t,y,x), pd.DataFrame)

# ============ assemble
carry_vars(ds, cfg) -> dict
build_dataset(ds, cfg, stacks) -> xr.Dataset
write_cube(out, cfg, src_attrs, report) -> None

# ============ driver
main(argv=None)   # --config, --sensors, --refresh, --dry-run, --limit
```

### `detector_cfg` — the bridge

`simple_outlier_detection.run_date(ds, date, cfg)` takes a config dict matching its own
`DEFAULTS` exactly. Build it by deep-copying `simple_outlier_detection.DEFAULTS`, overlaying
`cfg["detector"]`, then filling the per-sensor `data.*` names:

```python
dcfg = copy.deepcopy(sod.DEFAULTS)
for section in ("reference", "covariate", "clear", "qc", "mixture", "solver"):
    dcfg[section].update(cfg["detector"][section])
dcfg["data"].update(cube=cfg["data"]["cube"], var=s["sst"], validvar=s["valid"],
                    cloudvar=s["cloud"], ref_var=cfg["detector"]["ref_var"],
                    landvar=cfg["data"]["watervar"], depthvar=cfg["detector"]["depthvar"],
                    tidal_depth_m=cfg["detector"]["tidal_depth_m"],
                    dates=[], min_pixels=s["min_pixels"])
dcfg["qc"]["nodata_prior"] = s["qc_nodata_prior"]
dcfg["offset"] = {"lower": cfg["filter"]["offset_lower"],      # kept in sync so the .nc's
                  "upper": cfg["filter"]["offset_upper"]}      # own flag agrees with ours
dcfg["output"].update(dir=..., fig_dir=..., write_netcdf=True,
                      write_figures=cfg["output"]["write_detector_figures"])
```

Do **not** call the detector's `load_config` — it reads YAML from disk. Build the dict.

Acquisition dates come from `outlier_detection.acquisition_dates(ds, var, land, min_pixels)`,
which returns `list[str]` of `YYYY-MM-DD` — the same function `filter_images.acquisition_mask`
was cross-checked against (156 eco, 56 lst).

**Cross-prototype import.** `build_cube.py` must put `prototypes/cloud_mixture_model/src` on
`sys.path` before importing, because `simple_outlier_detection` does a flat
`from outlier_detection import ...`. This is a real coupling and the only one; a comment
should say so. DINEOF's own `src/` is already `sys.path[0]` when the script is run by path.

### Stage 2 — masking

```python
sst = ds[s["sst"]].values.astype("float32")          # (365, y, x) from the SOURCE cube
out = np.full_like(sst, np.nan)
for path in paths:
    d = xr.open_dataset(path)
    keep, reason = scene_verdict(d.attrs, cfg)
    if not keep:
        continue                                      # scene stays all-NaN
    i = index_of(d.attrs["date"])
    px = (d["p_valid"].values >= cfg["filter"]["p_valid_min"]) & water
    out[i] = np.where(px, sst[i], np.nan)
```

`scene_verdict` gates on `float(attrs["center"])` against the band, plus `ref_flag` only if
`require_reference` is on, plus `min_kept_frac` if set. It returns the reason so the log and
the report CSV can say *why* each scene was dropped.

Note the source `sst` is used, not the `.nc`'s `obs` — they are the same array, but reading
from the cube keeps the cube the single source of truth for values and reserves the `.nc` for
verdicts.

### Assembly and write

Reuse the package rather than hand-rolling zarr:

- [`datacube.build_encoding(ds, compression, chunks)`](src/coastal_sst_data/processes/datacube.py#L2367)
  — per-variable chunks + blosc codec, with the uint8 bitshuffle branch.
- [`datacube.write_zarr(ds, path, encoding)`](src/coastal_sst_data/processes/datacube.py#L2397)
  — `mode="w-"`, consolidated, with the zarr-v3 consolidated-metadata warning suppressed.
- [`store.atomic(dest)`](src/coastal_sst_data/store.py#L356) and
  [`store.sweep_scratch`](src/coastal_sst_data/store.py#L292) — write to `.part-*`, swap on
  success. Drive `store.atomic` directly rather than using `write_zarr_safe`, following
  [preprocess.run](src/coastal_sst_data/processes/preprocess.py#L1118-L1126): the source store
  must stay open across the write.

**The encoding gotcha** ([preprocess._for_write](src/coastal_sst_data/processes/preprocess.py#L928-L947)):
every channel carried over from an opened store still holds the *source* store's `encoding`
(chunks, codecs), which conflicts with the new `encoding=` and silently wins. Clear it on
**`data_vars` only** — the `time` coord's `units`/`calendar` must survive:

```python
for v in out.data_vars:
    out[v].encoding = {}
```

**Root attrs**, following `finalize_preprocess_attrs`
([preprocess.py:827-866](src/coastal_sst_data/processes/preprocess.py#L827-L866)) — spread the
source attrs first so nothing the assembler stamped is lost, then stamp this stage:

```python
{**src_attrs,
 "aoi_id": cfg["data"]["aoi"],
 "dineof_filter": json.dumps({...filter + detector...}, sort_keys=True),
 "dineof_filter_channels": json.dumps(sorted(kept_channels)),
 "dineof_filtered_at": provenance.now_utc(),      # created_at keeps the ASSEMBLY date
 "package_version": provenance.package_version(),
 "code_version": provenance.code_version(),
 "config_yaml": <this config>}
```

Per-variable attrs record the filter on the channels it touched: `p_valid_min`,
`offset_band`, `n_scenes_kept`, `source_channel`.

A `filter_report.csv` (date, sensor, center, ref_cover, cloud_frac, kept_frac, verdict,
reason) is written next to the cube — the audit trail for which scenes made it in.

### Memory

365 × 303 × 303 float32 = **134 MB per channel**. Output is 2 sensor channels + `modis_sst_aqua`
+ 3 statics ≈ **420 MB**, assembled in memory and written in one pass. No blocking needed;
`resolve_block_days` and `append_zarr` exist if the AoI ever grows.

## A flagged risk I am shipping as you specified

You chose the scene gate as the offset band alone. One case that leaves open: when MODIS
coverage is too sparse, the SPDE returns `covariate.init_prior` and the detector compares the
scene against a **constant**, which makes both `p_valid` and `center` meaningless — this is
exactly what produced `0.0% of water pixels had reference data` on the six March dates. Such a
scene can land *inside* the offset band by accident and be admitted with a garbage pixel mask.

So `require_reference` ships as a config key defaulting to `false`, as chosen, and the builder
**logs a warning naming every admitted scene with `ref_flag == 1`** and writes `ref_cover` into
`filter_report.csv`. The risk is visible rather than silent, and arming it is a one-word edit.
Same for `min_kept_frac` — present, `null`, one edit away.

## Verification

1. **Config guards fire before any I/O**: a typo'd section, `detector.mixture.beta_clod`,
   `sensors.eco.sstt`, and a channel name absent from the source cube each raise a
   `ValueError` naming the exact path, with no cube read.
2. **Cache behaviour**: run twice. The second run must re-run **zero** dates and log
   `cache hit: N/N`. Then change `filter.p_valid_min` and rerun — still zero re-runs
   (thresholds are not part of the cache key). Then change `detector.clear.sd` and rerun —
   **all** dates re-run. Then `--refresh` — all dates re-run.
3. **Detector agreement**: for 2025-03-07, the cached `.nc`'s `center` must equal the
   `+1.019` already in `outputs/simple_outlier_check/offset_flags.csv`, and its `p_valid`
   array must be identical to the existing cached run — the same config must give the same
   numbers.
4. **Masking is exactly the threshold** — the core assertion:
   ```python
   fc = xr.open_zarr(out); src = xr.open_zarr(cube); d = xr.open_dataset(nc_for_date)
   kept = np.isfinite(fc["eco_sst_v002"].sel(time=date).values)
   want = (d["p_valid"].values >= 0.8) & water & np.isfinite(src["eco_sst_v002"].sel(time=date).values)
   assert (kept == want).all()
   assert np.array_equal(fc["eco_sst_v002"].sel(time=date).values[kept],
                         src["eco_sst_v002"].sel(time=date).values[kept])   # values UNCHANGED
   ```
5. **Scene gate**: every date whose `.nc` `center` is outside `[-2, +4]` must be all-NaN in the
   output; every date inside must have at least one finite pixel. Cross-check the kept count
   against `filter_report.csv`. From the runs already done, 2025-05-25 (+4.32) must be
   **excluded** and 2025-08-08 (+2.79) **included** at this band.
6. **Carry-over is byte-identical**: `xr.testing.assert_identical` on `modis_sst_aqua` and each
   static between source and output. A carried channel must not be masked or re-scaled.
7. **Cube integrity**: reopens with `xr.open_zarr`; dims `(365, 303, 303)`; chunks are
   `(64,128,128)` for 3-D and `(128,128)` for statics (read `.encoding["chunks"]`, not
   `.chunks`); `landcover_water` stays `uint8`; root attrs carry both `created_at` (the
   original assembly date) and `dineof_filtered_at`.
8. **End to end**:
   ```bash
   python prototypes/DINEOF/src/build_cube.py \
     --config prototypes/DINEOF/configs/config.build_cube.admiralty_inlet.yaml --limit 6
   # then the full run, ~9 min for 212 acquisitions on a cold cache
   ```
   `--dry-run` must print the per-scene verdict table and write nothing.
