# Plan: config-driven scene filtering + visual inspection for DINEOF

> On approval, step 0 is to copy this file to
> `prototypes/DINEOF/plans/plan-image-filtering.md` (plan mode only permits editing this
> scratch path, so it lands there as the first implementation action).

## Context

DINEOF needs a clean input set: the dates where Landsat or ECOSTRESS actually saw the water,
minus the ones ruined by cloud or sensor error. Reconstructing from a contaminated stack
propagates the contamination into the gap-filled field, so the selection step has to be right
before any of the DINEOF work starts.

`prototypes/DINEOF/src/helpers.py` already holds the whole scoring engine — a `Criterion`
DSL, `evaluate_criteria`, `filter_valid_images`. What is missing is (a) a config that states
the thresholds instead of hard-coding them, and (b) figures that show *why* each scene passed
or failed, because a cutoff cannot be chosen without seeing the scenes just either side of it.

Intended outcome: edit a YAML, rerun, look at a contact sheet, and see the pass/fail boundary
as a visible seam that moves as you retune.

## Hard requirement: every proportion is over WATER pixels, never the full frame

This governs the whole design, so it is stated once here and enforced in three places:

- **Criterion fractions** — every channel is masked to `landcover_water > 0.5` before it
  reaches `evaluate_criteria`, and scored with `nan_policy: "ignore"` so land leaves both the
  numerator and the denominator. A `prop:` in the config therefore always means *"this
  fraction of the AoI's water"*. See "Water masking" below.
- **`frac_obs`** (the "% of water observed" in every figure title) is
  `n_obs / int(water.sum())` — water pixels carrying a finite SST, over water pixels. Never
  over `y * x`.
- **The colour limits** are percentiles over water-and-sensor-valid pixels only.

The denominator is asserted in verification step 2 with a known number: `eco_valid_v002 == 1`
on 2025-03-05 must score **0.761** (water) and never **0.463** (full frame). `n_water` is
logged once per run so the denominator is visible in the output, not merely assumed.

## What gets built

```
prototypes/DINEOF/
  configs/config.filtering.admiralty_inlet.yaml   # new
  src/plotting.py                                 # new — drawing primitives
  src/filter_images.py                            # new — the driver
  figures/filtering/admiralty_inlet/
    scenes/{pass,fail}_<date>_<sensor>.png
    contact_eco.png  contact_lst.png
```

`helpers.py` is **not** modified. Everything below consumes it.

## Three findings from the cube that drive the design

Measured against `prototypes/cloud_mixture_model/data/datacube/admiralty_inlet.zarr`
(the only cube that exists; 365 daily steps × 303 × 303):

1. **Land is 39% of the frame** (55,722 water px of 91,809). `evaluate_criteria` reduces over
   every pixel it is handed, so on 2025-03-05 `eco_valid_v002 == 1` scores **0.463 over the
   frame vs 0.761 over water**. An unmasked threshold measures the AoI's land share, not the
   scene — and transfers to no other AoI. Masking is not a refinement here, it is required.
2. **Most dates are absences, not failures.** ECOSTRESS observes on **156 of 365** days,
   Landsat on **56**. Empty days score 0 and `.fillna(False)` ([helpers.py:320-322](prototypes/DINEOF/src/helpers.py#L320-L322))
   sends them to FAIL. Without acquisition detection a 49/156 pass rate renders as 49/365.
3. **A label collision silently corrupts the report.** `evaluate_criteria` does
   `out["valid"] = combine` *after* the concat ([helpers.py:326-327](prototypes/DINEOF/src/helpers.py#L326-L327)),
   so a criterion labelled `valid` has its fraction column overwritten by the boolean verdict.
   Same for any label starting `pass: `. The config loader must reject both.

## Design

### Water masking — `channel_view()`

Build a lazy Dataset of just the criterion channels, each `.where(water)`, and score with
`nan_policy: "ignore"` and explicit `spatial_dims=["y", "x"]`. Land becomes NaN and
[`Criterion.fraction`](prototypes/DINEOF/src/helpers.py#L178-L183) drops it from numerator
*and* denominator, so every fraction reads "of the AoI water, how much satisfies this".

Two shapes need preparation first:
- **uint8 channels** (`<pre>_valid`) carry no NaN. Cast to `float32` before `.where()` so the
  upcast is not to float64.
- **`(time,)` channels** (`eco_georef_flag`, `<pre>_hour`) have no spatial dim, so
  `fraction`'s `dims = [d for d in dims if d in da.dims]` comes back empty and it raises
  ([helpers.py:171-175](prototypes/DINEOF/src/helpers.py#L171-L175)). `broadcast_like(water)`
  stays lazy and makes the fraction exactly 0 or 1, so `prop: 0.99` reads "this flag must hold".

Plus one local key, `fill`, popped before `helpers` sees it (`from_spec` rejects unknown keys,
[helpers.py:156](prototypes/DINEOF/src/helpers.py#L156)): `fill: 1.0` on a cloud raster means
"undefined water is cloud", putting the 96 dates the ECO cloud raster does not cover back in
the denominator as failures. Without it the denominator is the raster's footprint, not the AoI.

Consequences to document, not fix: with a masked view `nan_policy: "invalid"` is *unusable*
(it divides by `y*x`, [helpers.py:187](prototypes/DINEOF/src/helpers.py#L187), putting land
straight back), and `"valid"` is a trap (land satisfies everything, fraction floors at 0.39).
Ship `ignore` on every criterion; `log.warning` if a config sets otherwise.

### Config schema

Per-sensor, because ECOSTRESS and Landsat need different channels *and* different thresholds.
Criteria are written as the dicts `Criterion.from_spec` already accepts.

```yaml
data:
  aoi: admiralty_inlet
  cube: ../cloud_mixture_model/data/datacube/admiralty_inlet.zarr
  watervar: landcover_water
  sensors: [eco, lst]
output:
  fig_dir: figures/filtering        # -> <fig_dir>/<aoi>/
  write_scenes: true
  write_contact_sheet: true
  write_report: false               # report_<sid>.csv; one line, off by default
plot:
  shared_scale: true
  pct_lo: 1.0
  pct_hi: 99.0
  ncols: 8
  scene_dpi: 130
  sheet_dpi: 150
  sort_by: leading                  # leading | date
  descending: true
sensors:
  eco:
    label: ECOSTRESS v002
    short: ECO
    sst: eco_sst_v002
    valid: eco_valid_v002
    cloud: eco_cloud_v002
    hour: eco_hour_v002
    min_pixels: 64
    require: all
    default_prop: 0.5
    criteria:                       # FIRST one is the leading criterion: it sorts the sheet
      - {channel: eco_valid_v002, value: 1, prop: 0.35, nan_policy: ignore,
         label: "valid>=0.35"}
      - {channel: eco_cloud_v002, value: "<0.5", prop: 0.50, nan_policy: ignore, fill: 1.0,
         label: "clear<0.5"}
      - {channel: eco_cloud_cover_hrrr, value: "<40", prop: 0.50, nan_policy: ignore,
         label: "hrrr<40%"}         # HRRR is PERCENT 0-100; <pre>_cloud is a FRACTION 0-1
      - {channel: eco_georef_flag, value: [0, 1], prop: 0.99, nan_policy: ignore,
         label: "georef ok"}
  lst:
    # lst_valid ALREADY has the cloud raster + 1 km buffer applied, so a surviving Landsat
    # scene should be cleaner than an ECOSTRESS one at the same nominal threshold.
    ... sst: lst_sst, valid: lst_valid, cloud: lst_cloud, hour: lst_hour
    criteria: valid>=0.60, clear<0.5 @0.70 (fill 1.0), hrrr<30% @0.60
```

`label` is **required**, unique, and may not be `valid` or start with `pass: ` — the guard for
finding 3. These are also what keeps the annotations short enough for a 1.85-inch panel
(`helpers`' default label would be `eco_cloud_v002 <0.5`).

### Config loader

Copy `load_config` from
[outlier_detection.py:121-140](prototypes/cloud_mixture_model/src/outlier_detection.py#L121-L140)
verbatim — the strict two-level walk that turns a typo into an error — with **three added
lines**: an `OPAQUE_SECTIONS = {"sensors"}` check that takes that section wholesale, because
it is nested deeper than two levels and its criteria lists are free-form. It is then validated
by a dedicated `validate_sensors(cfg, path)` which merges each block over `SENSOR_DEFAULTS`
one level, rejects unknown `sensors.<id>.<key>`, enforces the label rules, and calls
`Criterion.from_spec` on every criterion **at config time** so a bad value spec fails in
milliseconds rather than after the cube is open.

One deviation from the house loader: `.resolve()` the paths, because the cube lives in a
sibling prototype and `ROOT / "../cloud_mixture_model/..."` otherwise leaves a `..` in every
log line.

### Acquisition detection

```python
n = ds[sst].where(water).count(dim=("y", "x")).compute()   # lazy, ~1 s, returns 365 ints
acq = np.asarray(n) >= min_pixels
```

Scored on the **SST** channel, not the cloud raster (`eco_cloud_v002` is finite on 269 days,
113 of which carry no SST). Score all 365 dates in one pass, then restrict the report and the
figures to `report["acquisition"]` — the saving from pre-subsetting is a fraction of the 1.7 s
scoring pass, and keeping the full table answers "did it fail, or was nothing there?".

Do **not** call `filter_valid_images` / `valid_dates` / `iter_valid_images`: each re-runs
`evaluate_criteria` internally ([helpers.py:342](prototypes/DINEOF/src/helpers.py#L342),
[384](prototypes/DINEOF/src/helpers.py#L384),
[407](prototypes/DINEOF/src/helpers.py#L407)). Call it once, `.sel` yourself.

### Colour scale — shared across scenes, per sensor, over valid pixels only

Measured p1/p99 in degC: ECO **3.03..25.33**, LST **0.82..20.32**; LST *raw* p1 is **-33.12**
(cloud tops). Three decisions:
- **Shared, not per-scene** — the contact sheet exists so the eye can compare scenes;
  per-scene rescaling makes an overcast 8 degC scene and a clear 18 degC scene look identical.
  `plot.shared_scale: false` stays available for reading one marginal scene.
- **Valid-only** — letting -33 degC set the floor squeezes every real Landsat scene into the
  top 40% of the bar. Scoring on the sensor-valid subset pushes contamination off the cold
  end, where a filtering tool wants it conspicuous. Display choice only.
- **Per sensor** — a union scale wastes ~5 degC of bar on each and flattens the real seasonal
  range; the sheets are per-sensor and never seen side by side.

`to_celsius` applied once to the whole loaded block, not per scene.

### Figures

**Per scene — 3 panels, SST | valid | cloud**, every acquisition, pass and fail:

```
ECOSTRESS v002   2025-06-07   20.41Z                          [ PASS ]
61% of AoI water observed (33,912 of 55,722 px)
valid>=0.35 0.76/0.35 ok   clear<0.5 0.79/0.50 ok   hrrr<40% 0.63/0.50 ok
```

`label observed/required ok|FAIL` per criterion in config order, read from `row[label]` and
`row["pass: " + label]`. Stamp is a boxed `fig.text` in `PASS_COLOR #2f9e44` / `FAIL_COLOR
#c92a2a`. Filenames `{pass,fail}_<date>_<sid>.png` — so `ls` sorts the groups apart and each
group stays chronological. **Retuning flips a scene to a different filename**, so clear
`{pass,fail}_*_<sid>.png` at the start of each sensor's run or you review a mix of two tunings.

**Contact sheet — one per sensor**, sorted by the leading criterion descending. Verdict is
carried by title colour *and* all four `ax.spines`, because text is marginal at 1.85 in. The
sheet then reads as a green block, a seam, a red block — and a green-framed panel appearing
below the seam is instantly legible as "passed the leading criterion, vetoed by another",
which is the diagnostic an `require: all` filter needs.

### Performance

| step | cost |
|---|---|
| `evaluate_criteria` over 4 masked channels × 365 dates | **1.7 s** (one fused dask pass) |
| `ds[[sst,valid,cloud]].isel(time=acq).compute()` for ECO | **0.9 s, 129 MB** |
| 156 scene figures + sheet | ~90-110 s — matplotlib dominates |

Per-scene plotting **must not** re-read zarr per date: the time chunk is 64, so one
`isel(time=t)` decompresses ~37 MB to deliver 0.37 MB — ~100× amplification across 156 dates.
Hold the per-sensor block in memory.

## Code to reuse

- **[helpers.py](prototypes/DINEOF/src/helpers.py)** — `Criterion.from_spec`,
  `evaluate_criteria`. The entire scoring engine; do not reimplement any of it.
- **[plot_daily.py](prototypes/cloud_mixture_model/src/plot_daily.py)** — port (not import;
  the prototypes stay independent trees) into `DINEOF/src/plotting.py`: `LAND_COLOR #c8c8c8`,
  `NODATA_COLOR #f4f4f2`, `CLOUD_COLOR #e03131`, `INVALID_COLOR #c2255c`, `sst_cmap()`,
  `to_celsius()`, `_extent_km()`, `water_mask()`, `open_cube()` (SystemExit + rebuild hint),
  `panel()`, and the contact-sheet layout constants
  (`figsize=(1.85*ncols, 2.30*nrows)`, `dpi=150`, `hspace=0.42, wspace=0.06`,
  `subplots_adjust(top=1-0.70/nrows, ...)`, colorbar in `fig.add_axes`).
  Keep the colour vocabulary identical so DINEOF and cloud-mixture figures read side by side.
  One new primitive: `mask_panel()` for the valid/cloud panels.
- **[outlier_detection.py](prototypes/cloud_mixture_model/src/outlier_detection.py)** — the
  house skeleton: `load_config`, `main(argv=None)` with `description=__doc__` +
  `RawDescriptionHelpFormatter`, `logging.basicConfig` inside `main`, `raise SystemExit` for
  user errors, per-item try/except that logs and continues then a summary `log.warning`,
  and the `mkdir -> savefig -> close -> log.info("wrote %s")` saving contract.

## Module structure — `filter_images.py`

```
# ============ config
DEFAULTS, SENSOR_DEFAULTS, OPAQUE_SECTIONS, LOCAL_KEYS = {"fill"}
load_config(path, defaults=DEFAULTS) -> dict
validate_sensors(cfg, path) -> None
_strip_local(spec) -> dict

# ============ scoring
channel_view(ds, water, specs) -> (xr.Dataset, list[dict])
acquisition_mask(ds, water, sst, min_pixels) -> (np.ndarray, np.ndarray)   # mask, n_obs
score_sensor(ds, water, scfg) -> pd.DataFrame
    # <label>*, "pass: <label>"*, "valid", "acquisition", "n_obs", "frac_obs"
    # frac_obs = n_obs / int(water.sum())  -- WATER denominator, asserted in step 2

# ============ scene data
load_block(ds, scfg, idx) -> xr.Dataset          # (n_acq, y, x), degC, in memory
sensor_limits(block, water, scfg, pcfg) -> (vmin, vmax)
scene_arrays(block, scfg, i, water) -> (sst, valid, invalid, cloud)

# ============ figures
criteria_line(row, criteria) -> str
scene_figure(block, scfg, pcfg, i, row, water, extent, vmin, vmax, out_dir) -> None
contact_sheet(block, scfg, pcfg, report, water, extent, vmin, vmax, out_path, aoi) -> None

# ============ driver
run_sensor(ds, water, extent, sid, cfg) -> (n_pass, n_acq)
main(argv=None) -> None
```

CLI: `--config`, `--sensors eco` (subset), `--no-scenes` (sheets only, ~30 s — the threshold
sweep loop), `--limit N` (first N acquisitions, smoke test).

`import helpers` / `import plotting` are bare sibling imports, working because the script is
invoked by path and `src/` lands on `sys.path[0]` — the same way every cloud_mixture_model
script imports its siblings. Worth one comment, since it is the only reason they are not
package-relative.

## Verification

1. **Config errors fire before any I/O.** Feed it `label: valid`, a typo'd section, and
   `sensors.eco.sstt` — each raises its own `ValueError` naming the exact path, zarr untouched.
2. **Fractions are water-only — the denominator check.**
   `score_sensor(...)["valid>=0.35"].loc["2025-03-05"]` must be **0.761**, not 0.463. Seeing
   0.46 means the water mask is not reaching `evaluate_criteria`; the ratio 0.463/0.761 =
   0.607 is exactly the water share of the frame, which is the signature of the bug. Assert
   the same for `frac_obs`: `rep["n_obs"] / rep["frac_obs"]` must equal **55,722** (the water
   pixel count) on every acquisition row, not 91,809. Also confirm `rep["valid"]` is bool
   while `rep["valid>=0.35"]` is float — a float column turned boolean means a label collided.
3. **Acquisition counts** must be `eco 156/365`, `lst 56/365`. Cross-check against
   `python prototypes/cloud_mixture_model/src/plot_daily.py --no-singles --min-pixels 64`,
   whose log line reports the same per-sensor counts independently.
4. **Full run** (~2-3 min):
   ```bash
   PROJ_DATA=$CONDA_PREFIX/share/proj python prototypes/DINEOF/src/filter_images.py \
     --config prototypes/DINEOF/configs/config.filtering.admiralty_inlet.yaml
   ```
   Then `ls figures/filtering/admiralty_inlet/scenes | wc -l` must be exactly **212**
   (156 eco + 56 lst), split pass/fail matching the log. Any other count means
   non-acquisition dates leaked into the figures.
5. **Read the figures** — this is the actual deliverable:
   - `contact_eco.png`: green-framed panels form one contiguous block ending at a single seam;
     panels either side of it should look genuinely similar. Green *below* the seam = passed
     the leading criterion, vetoed by another — open that `fail_` scene and read which.
   - One `pass_` and one `fail_` scene: land flat grey in all three panels, unobserved water
     flat off-white, SST spread across the bar (a single flat colour means the limits came
     from raw percentiles).
   - A `fail_` scene with low `clear<0.5`: the cloud panel must be visibly flagged where the
     SST panel is anomalously cold. An off-white cloud panel with a 0.2 fraction means the
     `fill: 1.0` path is wrong.
   - Landsat's invalid overlay should track its cloud raster closely (`use_cloud=True`);
     ECOSTRESS's should not (QC-gated only). If ECO's hugs its cloud raster, the wrong `valid`
     channel is wired up.
6. **The intended loop:** edit `prop`, rerun `--no-scenes` (~30 s), watch the seam move; once
   it sits right, rerun in full for the per-scene review.

## Not included (per your selection)

No CSV report by default (`output.write_report: false` toggles a one-line `to_csv`), and no
fraction histograms/ECDF.
