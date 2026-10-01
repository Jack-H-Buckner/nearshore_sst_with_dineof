# Nearshore sea surface temperature estimtes with DINEOF

This repo uses DINEOF to make a composite level 4 sea surace temperature data product
combining data from high resolution thermal infared remote sensing instuments. The method
leverages daily MODIS-aqua night time SST retrievals as the back bone of the data product 
and suppliments these observaiton with high resolution information from the Landsat and 
ECOSTRESS missions. The MODIS data provide high quiality night time retrievals at a 1km 
resolution. Thse provide the most accurate information because but do not cover near and inshore
waterways and can miss fine scale structure and patterns with length scales less than 1km. 
ECOSTRSS and Landsat provide less a precise by very high resolution ocean temeprautre data at
70m and 100m respectively. These instuments are able to observe narrow inshore water ways not 
covered by MODIS, but have lower over pass frequencies reducing the temproal coverage. 

Combining MODIS with Landsat and ECOSTRESS data serves two purposes. First, MODIS provides 
accurate retrievals that are used to calibrate the Landsat and ECOSTRESS retrievals. MODIS 
did is also aviable more frequently, but it is often unavaible in nearshore areas where many
aquaculture operations and ecosystems that can be impacted by ocean temperatures are located. 
The ECOSTRESS and Landsat data provide information about the climatology in these areas and 
the correlations between near shore areas and off shore locations wherne MODIS is avaible that 
can be used to predict temperatures in those location on day swhen only the lower resolution 
MODIS data is avaible. 

Here we use the DINEOF algorithm to recover patterns of spatial correlation and use those 
correlations to fill in temperature estimates in areas not covered by MODIS. This produces a 
sptaitly contiguous estiates with temporal gaps on days wher the full scene is obstructed by 
clouds. 

## Overview

This repo applies three primary steps. First, it applies quality control measures to mask 
pixels adn images that have corrupted temperature retrievals. Next The three instruments are 
calibrated against one another using matchups between MODIS which is used as the benchmark
and the other two. After filtering and calibration the data are merged into a single composite
sea surface temeprature field. The DINEOF algothim is applied to the composite field to fill
missing values that are filtered out the qualtiy control process or near shore areas not covred
by MODIS on days without landsat or ECOSTRESS retrievals. The qualtiy of the data interpolations
are evaluated using cross validation and the parameters goverining the DIEOF algorthim are tuned 
to minimize the cross validation error. 

## Quick start

The pipeline runs in three stages, each driven by one YAML file per region:

```bash
# 1. sensor offsets against MODIS (fixed vs fixed + slope), with diagnostics
python src/run_region.py --config configs/region.admiralty_inlet.yaml offsets

# 2. apply the offsets, run the iterative DINEOF cloud filter and the gap-filling fit
python src/run_region.py --config configs/region.admiralty_inlet.yaml dineof

# 3. validate the filled fields against in-situ water temperature
python src/run_region.py --config configs/region.admiralty_inlet.yaml validate

# or all three in order
python src/run_region.py --config configs/region.admiralty_inlet.yaml all
```

Each stage writes its outputs under `<region.out_dir>/<aoi>[_<tag>]/`:

```
data/regions/admiralty_inlet/
├── stage1_offsets/     offsets.json, models.csv, scenes.csv, pairs.csv, figures/
├── stage2_dineof/      admiralty_inlet_dineof.zarr, reports/, figures/
└── stage3_validation/  matchups.csv, metrics.csv, stations.csv, figures
```

Common options:

| Option | Applies to | Effect |
|---|---|---|
| `--tag T` | all stages | writes to `<aoi>_T/` so that variants sit side by side; later stages read the same tag |
| `--masks CUBE` | `offsets` | uses the cloud masks from a stage-2 cube instead of the sensor QC alone (see stage 1) |
| `--max-iter N` | `dineof` | overrides `loop.max_iter`, e.g. for a quick trial run |
| `--no-figures` | all stages | writes data and reports only |

### Setting up a new region

1. Build the source datacube with `coastal_sst_data` (see [Input data](#input-data)).
2. Copy `configs/region.template.yaml` to `configs/region.<aoi>.yaml`. Every section is
   documented there with its default value.
3. Set the required keys:
   - `region.aoi`: names the output directory;
   - `region.source`: path to the cube;
   - `validation.insitu`: a file, or `null` to use in-situ channels carried in the cube.

   Check the `sensors` channel names against your cube. They default to `coastal_sst_data`'s
   names.
4. Run `offsets` and review `stage1_offsets/figures/` (see [Stage 1](#stage-1-offsets)).
5. Run `dineof` and review the CV and cloud-filter figures.
6. Run `validate`.

### Environment

The code runs in the `coastal_sst_data` conda environment, which provides numpy, scipy,
pandas, xarray, zarr, matplotlib, pyproj and pyyaml. Two code dependencies live outside this
repo:

- **`seasonal_smoothing`**, from a checkout of `coastal_sst_data` at
  `../coastal_sst_data/prototypes/cloud_mixture_model/src`. `iterative_filter.py` adds that
  path automatically. If `coastal_sst_data` is checked out somewhere else, put that directory
  on `PYTHONPATH`.
- **`scripts/fetch_insitu.py`**, which needs the `coastal_sst_data` source checkout
  (`../coastal_sst_data/src`, v0.7 or later) and network access.

Run the tests with `python -m pytest tests -q`. They use small synthetic cubes and take about
2 minutes.

## Input data

The input is one daily `coastal_sst_data` datacube (zarr) on a projected grid: `x`/`y` in
metres, a `crs` attribute such as `EPSG:32610`, and one `time` step per day. It must hold:

| Role | Channels (default names) |
|---|---|
| Water mask, depth | `landcover_water`, `depth_cudem` |
| MODIS Aqua (the reference) | `modis_sst_aqua`, `modis_valid_aqua`, `modis_hour_aqua` |
| ECOSTRESS | `eco_sst_v002`, `eco_valid_v002`, `eco_cloud_v002`, `eco_hour_v002` |
| Landsat | `lst_sst`, `lst_valid`, `lst_cloud`, `lst_hour` |
| In situ (optional) | `insitu_sst`, `modis_insitu_sst`, `insitu_station` and the `insitu_stations` attribute, written when the cube is built with `datacube.insitu: true` |

All SST values are in kelvin, and hours are UTC.

MODIS has to be **nearest-neighbour resampled** onto the grid (`coastal_sst_data`'s default).
Stage 1 reconstructs the native ~1 km MODIS footprints from patches of identical values.

The other high-resolution sensors are configured under `sensors:`; their ids (`eco`, `lst`)
name the outputs.

## Quality control 

We filter the Landsat and ECOSTRESS data for cloud contraminaiton. The level 2 data from each 
instrument provide cloud masks, but these often miss contamination at the edge of clouds and
small ammounts of contaminiation from thin clouds and fog. We identify these outliers by 
comparing each retrieval to a base estiamte of sea surface temperature in the region created
by a composit of MODIS retrievals from the neigboring `n` days. These are composited by taking 
taking the median accross the pixels. 

The MODIS composites are interpolated accross the full scene by estimating a gausian process
with an SPDE approximation and  Matern covariance function. We solve for the smooth field
using preconditioned conjugate gradients. The Matern covariance function has a length scale `l`
and marginal variance parameter `\alpha`, which ideally would be tuned using cross validation
to match the length scale SST patterns over the region. However we have chosen to fixed these values at
"sensible" defualts. Tuning can be added as a future enhacement. 

The cloud outliers are identified by a mixture model that includes a central normal distiubton component
for valid pixels, and an upper and lower tail components modeled by exponential distribtions for cold
and warm outliers. The mixture component assigments are spatailly smoothed using a Ising prior. 
The mixture components and 

> **In the three-stage pipeline** the per-scene classifier above (the mixture model with an
> Ising prior, `simple_outlier_detection.classify`) is unchanged. What changes is the baseline
> each scene is compared against. A smoothed MODIS composite is weak exactly where cloud
> detection matters: it is sparse, coarse, night-time, and absent nearshore. So stage 2 uses
> a DINEOF field instead. The spatial modes are fit on all sensors, but each day's amplitudes
> are fit to MODIS pixels only. The classification and the DINEOF fit are iterated until the
> cloud flags stop changing (see [Stage 2](#stage-2-dineof-and-the-cloud-filter)). The
> MODIS-composite detector is still available through `build_cube.py`.

## Interinstrument calibration

ECOSTRESS and Landsat are placed on MODIS Aqua's scale. MODIS is the reference because it is
the most accurate retrieval and covers the most days. The calibration is stage 1, and two
choices shape it.

- **Matchups are built at MODIS footprint scale.** MODIS is a ~1 km average, while ECOSTRESS
  (70 m) and Landsat (100 m) resolve structure inside that footprint. Pairing each 100 m pixel
  with its footprint's single MODIS value inflates the scatter and biases any slope fit.
  Instead, each MODIS footprint is paired with the **median of the sensor's kept pixels inside
  it**.
- **Two models are compared:** a constant offset, and a constant offset with a slope. The slope
  is estimated by reduced major axis (RMA) **across scenes**. Within one scene MODIS varies by
  only ~0.2–0.3 K, which is close to its own noise, so within-scene slopes are not identifiable.
  Across scenes the seasonal range (5–8 K here) provides the leverage. Leave-one-scene-out
  cross-validation decides between the two models.

## DINEOF

DINEOF fills gaps by repeated truncated SVD of the space × time matrix. It fits the leading
`k` EOF modes, uses them to replace the missing values, and repeats until the filled values
stop changing.

This implementation is eDINEOF (`src/edineof.py`). It adds a diffusion filter along time, with
a cutoff period `T_c` in days, applied to the temporal covariance. The filter couples each day
to its neighbours, which is what lets days with no observations be reconstructed at all.

The data are first standardized per pixel: a seasonal harmonic is removed and the result is
divided by a robust scale. That way the EOFs describe anomalies, not the seasonal cycle.

## Cross validation

Two holdouts answer different questions, so both are used:

- **Point holdout (selects `k`).** On a seeded 20% of the dates where ECOSTRESS or Landsat
  covers most of the water (`holdout:`), those sensors are removed from the composite. The
  pixels that only they observed are held out. The reconstruction is scored there, so it is
  tested on recovering the high-resolution field from MODIS and the rest of the record.
- **Day holdout (selects `T_c`).** 10% of the days with data are held out whole. A held-out
  day has no pixels of its own, so it tests what the temporal filter contributes, which is the
  situation of every cloudy day.

## Parameter tuning

The search covers a grid of `k` (`dineof.modes.k_grid`) by `T_c` (`dineof.filter.t_c_grid`),
on a 4×-coarsened grid for speed (`dineof.matrix.coarsen`). It picks two settings:

- the **point-optimal** (`k`, `T_c`), the best point-holdout error, used on dates that have
  data;
- the **day-optimal** (`k`, `T_c`), the best day-holdout error, used on dates without data.

Within the winning `T_c`, the smallest `k` within 1% of the best error is chosen
(`modes.rule: parsimonious`). The final fill then runs once at native resolution with those
settings, warm-started from the coarse fit.

---

# Running the pipeline, stage by stage

## Stage 1: offsets

```bash
python src/run_region.py --config configs/region.<aoi>.yaml offsets [--tag T] [--masks CUBE]
```

Code: `src/stage1_offsets.py`, configured by the `offsets:` section.

### What it does, per sensor

1. **Load** the raw scenes and the sensor's cloud and valid flags. Pixels outside
   `filter.valid_range` (271–310 K) count as flagged; this catches gross errors such as a
   505 K ECOSTRESS scene that the sensor QC passes.
2. **Build footprint pairs.**
   - Each day's MODIS footprints are reconstructed as 4-connected patches of identical value.
   - Each footprint becomes one pair: its MODIS value against the median of the sensor's kept
     pixels inside it.
   - A footprint needs `min_footprint_cells` water cells, and the sensor has to cover
     `min_footprint_cover` of them.
   - A scene needs `min_footprints` pairs to be used.
   - `aggregate: pixel` falls back to per-pixel pairs.
3. **Clip outliers iteratively, within scenes.**
   - Each round fits every scene's own offset (the median of sensor − MODIS) on the pairs kept
     so far.
   - Every pair is judged against its own scene: it is dropped if its residual exceeds
     `outlier_k` (1.5) × a robust SD (1.4826 × MAD) computed over all pairs.
   - Pairs can come back in a later round. Clipping stops when the removed set stops changing,
     or at `outlier_max_iter`.
   - This removes footprints that disagree with the rest of their scene: cloud edges, fronts
     that moved between overpasses, and coastal mixing. It cannot remove a whole scene that is
     uniformly wrong (see the two-pass note below).
4. **Fit two models on the kept pairs.**
   - **fixed:** sensor = MODIS + *a*.
   - **rma:** sensor = x₀ + *a* + *b*·(MODIS − x₀).
     - *b* is the RMA slope over the per-scene medians in absolute temperature.
     - x₀ is the median scene MODIS value, so *a* stays the physical offset at a typical
       temperature.
     - When applied, the sensor is converted as (sensor − off) / *b*, with off = *a* + x₀(1 − *b*).
5. **Score both by leave-one-scene-out (LOSO).** For each scene, refit without it and predict
   its footprints from MODIS. The scores are the footprint RMSE, the scene-median RMSE and the
   bias.
6. **Recommend.**
   - `rma` is recommended only if its LOSO RMSE beats `fixed` by `recommend_min_gain` (2%),
     and its slope lies inside `slope_clip`. Otherwise `fixed`.
   - `offsets.mode: auto` uses the recommendation; `fixed` or `rma` overrides it.

### Outputs

| File | Contents |
|---|---|
| `offsets.json` | **The hand-off to stage 2.** Per sensor: both models (`a`, `b`, `x0`, `off`, `slope`, LOSO metrics), the diagnostics, `recommended` and `chosen` |
| `models.csv` | One row per sensor × model: the parameters, LOSO metrics, slope interval, within-scene correlation and clipping statistics |
| `scenes.csv` | The LOSO error of every held-out scene under each model |
| `pairs.csv` | Every footprint pair, with its residual and kept flag |
| `figures/summary.png` | LOSO RMSE of fixed vs rma per sensor, the slope and its interval, and the recommended and chosen mode |
| `figures/models_<id>.png` | **Left:** scene medians in absolute °C (coloured by overpass hour), with the fixed and RMA lines and a 90% bootstrap band. **Right:** within-scene anomalies with the correlation *r*, against slope 1, the between-scene RMA slope and the within-scene RMA slope |
| `figures/loso_<id>.png` | LOSO error per scene against date and against temperature, for both models. A slope that matters shows up as a trend with temperature under `fixed` |
| `figures/outliers_<id>.png` | Residual histogram with the clip threshold, the clipping's convergence by round, and the share removed per scene |
| `figures/outlier_maps_<id>.png` | The scenes with the most removed footprints: the sensor field, the MODIS footprints, and each footprint's residual with removals marked |

### How to choose the mode

- Start from `summary.png`.
- **Keep `fixed`** if the RMA slope's interval includes 1, or if the LOSO gain is small.
- **Before accepting `rma`, open `models_<id>.png`:**
  - a slope fitted on a handful of scenes, all at one overpass hour, can reflect daytime
    warming that grows with season rather than a sensor gain;
  - on Admiralty Inlet, Landsat (11 scenes, all at ~19 UTC) comes out at slope 1.37, while
    ECOSTRESS stays consistent with 1.
- Set `offsets.mode` and re-run `offsets` to make the choice explicit.

### Two-pass refinement (recommended for a new region)

With only sensor QC as the mask, scenes that are cloudy almost everywhere can survive into the
offset fit. Their medians are several kelvin cold, which inflates the LOSO error and the slope.
Within-scene clipping cannot remove them. Once stage 2 has produced cloud masks, re-run:

```bash
python src/run_region.py --config ... offsets --tag v2 --masks data/regions/<aoi>/stage2_dineof/<aoi>_dineof.zarr
python src/run_region.py --config ... dineof  --tag v2
```

On Admiralty Inlet this lowered the ECOSTRESS LOSO RMSE from 1.05 K to 0.75 K.

## Stage 2: DINEOF and the cloud filter

```bash
python src/run_region.py --config configs/region.<aoi>.yaml dineof [--tag T] [--max-iter N]
```

Code: `src/stage2_dineof.py`, which wraps `src/pipeline.py`. It needs stage 1's `offsets.json`
for the same tag, and errors with the command to run if that is missing.

### What it does

1. **Load and apply the offsets.**
   - Raw scenes, QC flags (plus `valid_range`) and MODIS are loaded.
   - Each sensor is converted as (sensor − off) / slope, using **the model stage 1 chose, with
     no refit**.
2. **Choose holdout dates and standardize.**
   - Seeded validation dates are drawn (`holdout:`).
   - A per-pixel seasonal harmonic and robust scale are fit to the composite of the kept pixels
     (`seasonal:`, `scale:`).
   - Compositing runs in `composite.block_days` blocks to bound memory.
3. **Run the cloud filter: iterative DINEOF** (`loop:`, `filter:`, `detector:`). Starting from
   QC-only masks, each iteration does four things.
   1. **Fit DINEOF** on the 4×-coarsened composite at fixed `loop.k` / `loop.t_c`,
      warm-started from the previous iteration.
   2. **Build the smooth field.** The spatial EOFs come from all sensors, but each day's
      loadings are fit to **MODIS pixels only**. They are pooled across neighbouring days with
      the same temporal filter, so days without MODIS borrow from their neighbours, and days
      none reaches fall back to climatology. A scene therefore never shapes the field it is
      judged against.
   3. **Classify every raw scene** against the smooth field with the mixture + Ising model.
      - A pixel is kept when P(clear) ≥ `filter.p_valid_min`.
      - A scene is dropped whole when its offset against the field falls outside
        [`offset_lower`, `offset_upper`].
      - Flags are recomputed from scratch each iteration, so wrongly flagged pixels can return.
   4. **Check convergence.** The loop stops when the share of flags that changed falls below
      `loop.tol`, or at `loop.max_iter`.
4. **Refit** the holdout dates and the seasonal standardization on the final cloud masks.
5. **Search the CV grid** on the coarse grid, selecting the point-optimal and day-optimal
   (`k`, `T_c`) (see [Parameter tuning](#parameter-tuning)).
6. **Run the final fill** at `final.coarsen` (1 = native 100 m), warm-started from the coarse
   fit.
7. **Build the products.**
   - **Filled field:** the point-tuned fit on dates with data, the day-tuned fit otherwise.
   - **Smooth field:** MODIS loadings on the final EOFs.
8. **Reconstruct the CV holdout.** One more coarse fit at the selected setting, with the held-out
   high-resolution pixels removed, gives the reconstruction that is compared with them.

### Output cube: `stage2_dineof/<aoi>_dineof.zarr`

| Channel(s) | Meaning |
|---|---|
| `sst_filled` | Gap-filled SST (K): point-tuned fit on dates with data, day-tuned otherwise. NaN on water pixels never observed |
| `sst_filled_point`, `sst_filled_day` | Each tuning on every date |
| `sst_filled_observed`, `sst_filled_constrained` | Whether a pixel was observed and held fixed; whether the date had any data of its own |
| `sst_smooth` | The smooth field the cloud filter judged scenes against |
| `smooth_loading_status` | Per date: 0 own MODIS, 1 borrowed from neighbours, 2 climatology |
| `eof_U`, `eof_sigma`, `eof_V`, `loadings_full`, `loadings_modis` | EOFs and loadings of the smooth-field fit; `*_point` holds the point-tuned fit's EOFs |
| `<sst>`, `<sst>_filtered` | Raw sensor SST, and raw SST where the cloud filter kept it |
| `<id>_keep`, `<id>_p_valid`, `<id>_center`, `<id>_keep_history` | Per sensor: the cloud filter's keep mask, P(clear), scene offset against the smooth field, and each iteration's verdict as bits |
| `<id>_offset` (attribute `slope`) | The offset and slope applied |
| `sst_composite`, `sst_composite_src` | The offset-corrected composite that was fitted, and which sensors contributed |
| `sst_seasonal_coef`, `sst_seasonal_sd`, `validation_msk` | The standardization and the CV holdout |

The attributes record the selected settings (`pipeline_fit`), the full config, and stage 1's
offsets (`stage1_offsets`).

### Diagnostics: `stage2_dineof/figures/` and `reports/`

| File | What to look for |
|---|---|
| `cv_curves.png` | Point- and day-holdout RMSE against `k` for each `T_c`, with the selection circled |
| `cv_holdout_scenes.png` | Held-out high-res scenes, the reconstruction made without them, and the difference. Whole-scene colour in the difference column means a day-level bias, not a texture error |
| `cv_holdout_scatter.png` | Reconstruction against held-out observation, and RMSE / bias per held-out date (also in `reports/cv_holdout.csv`) |
| `removed_scenes_<id>.png` | The scenes the cloud filter removed most from: raw, smooth field, residual, kept vs removed |
| `cloud_filter_<id>.png` | Per pixel: share removed, and verdict flips across iterations (persistent flips mark unstable areas such as tidal flats). Per scene: offset against the smooth field, with the accepted band |
| `flag_changes.png` | Pixels newly flagged and restored at each iteration; both should shrink toward zero |
| `eofs.png`, `eofs_point.png` | Spatial modes, with their loadings over time: full-data against MODIS-only, MODIS days ticked, climatology days shaded |
| `fields_<n>.png` | For selected days: the input composite, the filled field, the smooth field and their difference |
| `reports/*.csv` | `iterations.csv` (per iteration: flips, kept pixels, timings), `scenes.csv` / `scenes_history.csv` (per-scene verdicts), `cv_curve.csv`, `cv_holdout.csv` |

### Runtime and memory

On Admiralty Inlet (303 × 303 cells, 365 days) a run takes about 10–20 minutes on a 16 GB
laptop. Each cloud-loop iteration is about 1 minute, the CV search 5–15 minutes, and the native
final fill about 5 minutes. Peak memory is about 3 GB.

It slows down sharply if the machine starts swapping. To reduce memory:
- shorten `region.time_range`;
- lower `composite.block_days`;
- raise `final.coarsen`.

## Stage 3: in-situ validation

```bash
python src/run_region.py --config configs/region.<aoi>.yaml validate [--tag T]
```

Code: `src/stage3_validate.py`, which wraps `src/validate_insitu.py`. It needs the stage-2 cube
for the same tag.

### In-situ data

Set `validation.insitu` to one of:

- a **netCDF** in `coastal_sst_data`'s format: dims `(station, time)`, with `sst` in °C, `qc`
  as QARTOD flags (`reader.qc_pass`, default 1 and 2), and coordinates `station_id`, `lat` and
  `lon`;
- a **long CSV** with columns `station_id, time, latitude, longitude, value[, z]`. Column names,
  units (`degC` or `K`), time zone and the maximum sensor depth are set under
  `validation.reader`;
- **`null`**, to use the in-situ channels carried in the cube.

To download IOOS stations (NDBC, NOAA CO-OPS, regional associations) for a cube's footprint:

```bash
python scripts/fetch_insitu.py --cube data/unfiltered/<aoi>.zarr \
    --start 2025-03-01 --end 2026-02-28 --out data/insitu/<aoi>_insitu.nc [--halo-km 5] [--dry-run]
```

### Matching

- **Placement.** Each station is placed on the nearest grid cell. A station on land, which is
  typical for a gauge on a pier, moves to the nearest water cell within `match.max_snap_m`
  (300 m) and is dropped beyond that. The snap distance is reported.
- **Overpass match.** Each product day is matched to the observation nearest the MODIS
  overpass that day, within `match.max_dt_min` (60 min). This is like for like with the
  product, which is on MODIS's night-time scale.
- **Daily-mean match.** Each product day is also matched to the mean of the UTC day, when
  observations cover `match.min_coverage` (75%) of its hours.
- **Products sampled:** `sst_filled`, `sst_filled_point`, `sst_filled_day`, `sst_smooth`, and
  the seasonal climatology, which is the reference for skill.

### Outputs: `stage3_validation/`

| File | Contents |
|---|---|
| `metrics.csv` | Per match type × product × stratum: n, bias, **MSE**, RMSE, MAE, median \|error\|, robust SD, r, and skill against climatology (1 − MSE / MSE_climatology). Strata: all, per site, pixel observed vs gap-filled, day with no data, season, smooth-field loading status |
| `matchups.csv` | Every matched station-day, with the in-situ value, every product and the gap flags |
| `stations.csv` | Station placement: status, pixel and snap distance |
| `mse_by_site.png` | MSE per site and for all sites, per product, labelled with n |
| `residuals.png` | Residual (product − in situ) against time for each site, residual distributions, residual against in-situ temperature, and residual by gap category |
| `stations_map.png`, `timeseries_<site>.png`, `scatter.png`, `error_by_category.png` | Station positions and matched pixels; per-site series; scatter per product; RMSE by gap category |

## Configuration reference

`configs/region.template.yaml` documents every key. The ones you are most likely to change:

| Key | Default | Effect |
|---|---|---|
| `region.time_range` | `null` | Dates to process; memory and runtime scale with it |
| `offsets.outlier_k` | 1.5 | Within-scene clipping threshold, × robust SD. 1.5 removes about a quarter of the footprint pairs, including real fronts; 2–2.5 is gentler |
| `offsets.mode` | `auto` | `fixed` or `rma` to override the recommendation |
| `filter.p_valid_min` | 0.5 | Pixel cut on P(clear) |
| `filter.offset_lower` / `offset_upper` | −2 / +4 K | Scene gate: a scene whose offset against the smooth field falls outside this band is dropped whole |
| `loop.k`, `loop.t_c`, `loop.max_iter` | 2, 7 d, 8 | Fixed DINEOF settings inside the cloud loop, and its iteration cap |
| `dineof.filter.t_c_grid`, `dineof.modes.k_grid` | see template | The CV search grid |
| `dineof.em.max_iter` | 250 | EM iteration cap for each fit in the search |
| `final.coarsen` | 1 | Grid of the final fill (1 = native) |
| `composite.block_days` | 30 | Memory/speed trade-off when compositing; results do not depend on it |

## Known limitations

- **The CV search may favour `T_c = 0`.** On Admiralty Inlet, fits with the temporal filter on
  often hit `dineof.em.max_iter` before converging, and `T_c = 0` is then chosen by default.
  Days with no data then come back as climatology. Check the `converged` column in
  `reports/cv_curve.csv`; raising `dineof.em.max_iter` is the first thing to try.
- **The RMA slope is confounded with overpass time.** For a sensor whose scenes all come from
  one overpass hour (Landsat), a slope ≠ 1 may be daytime warming that varies with season. See
  [How to choose the mode](#how-to-choose-the-mode).
- **QC-only offsets can be contaminated** by wholly cloudy scenes. Use the
  [two-pass refinement](#two-pass-refinement-recommended-for-a-new-region).
- **MODIS footprints are inferred from identical values,** so MODIS must be nearest-neighbour
  resampled. Adjacent footprints that happen to share a value merge; this is harmless for the
  matchup.

## Other scripts

These are the building blocks the stages import. Each can also be run on its own.

| Script | Purpose |
|---|---|
| `src/pipeline.py` | The end-to-end run that stage 2 wraps; it can also fit offsets itself (`configs/pipeline.admiralty_inlet.yaml`) |
| `src/iterative_filter.py`, `src/iterative_diagnostics.py` | The iterative DINEOF cloud filter, and diagnostics comparing it with the MODIS-composite filter |
| `src/offset_diagnostics.py` | Slope estimates by estimator, pair set and support, with scene-bootstrap intervals |
| `src/pipeline_figures.py` | Re-draws a pipeline cube's figures without re-running it |
| `src/validate_insitu.py` | The matcher stage 3 uses, as a standalone CLI |
| `src/build_cube.py`, `src/composite.py`, `src/standardize.py`, `src/edineof.py` | The original stage scripts (MODIS-composite cloud filter, then compositing, standardization and eDINEOF), kept for comparison |


