"""End-to-end DINEOF gap-filling with iterative cloud filtering, from ONE raw datacube.

Input: a source cube holding MODIS (the composite anchor), ECOSTRESS and Landsat SST with their
QC channels -- the cube coastal_sst_data assembles. Output: ONE cube with the gap-filled field,
the smooth field the cloud filter judged scenes against, the cloud filter itself, the EOFs and
their loadings, plus reports and figures.

Every earlier product the stage scripts used to hand each other is fitted here instead:

  1. load      raw scenes, QC (+ valid_range), MODIS, depth -- iterative_filter.load_raw
  2. offsets   each sensor against MODIS on QC-masked scenes -- composite.scene_matchups /
               fit_offset, unchanged
  3. holdout   validation dates for the point CV -- seeded, without replacement
  4. seasonal  per-pixel harmonic climatology and robust scale of the composite --
               standardize.fit_harmonics_gappy / robust_scale, unchanged
  5. loop      the iterative DINEOF-baseline cloud filter -- iterative_filter.run_loop
  6. refit     offsets, holdout and seasonal again, on the loop's final cloud mask; then a full
               eDINEOF CV search on the coarse grid to choose (k, T_c)
  7. final     one eDINEOF fit at `final.coarsen` (100 m by default) at those settings,
               warm-started from the coarse fit
  8. smooth    MODIS-only loadings on the final EOFs -> the smooth field
  9. write     the output cube, reports, figures

Usage (from the repo root):

    python src/pipeline.py --config configs/pipeline.admiralty_inlet.yaml
    ... --tag e2e          # suffix every output
    ... --max-iter 2       # a quicker cloud loop
    ... --no-figures
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

import iterative_filter as F                               # first: bridges seasonal_smoothing

import composite as C                                       # noqa: E402
import edineof as E                                         # noqa: E402
import standardize as S                                     # noqa: E402
from seasonal_smoothing import FIT_FULL, design_matrix      # noqa: E402

from coastal_sst_data import provenance, store              # noqa: E402
from coastal_sst_data.config import CompressionSpec         # noqa: E402
from coastal_sst_data.processes import datacube             # noqa: E402

log = logging.getLogger("pipeline")

ROOT = F.ROOT
DEFAULT_CONFIG = ROOT / "configs" / "pipeline.admiralty_inlet.yaml"


# ==================================================================== config

# Sections handed to iterative_filter.build_config verbatim, and validated there.
ITER_SECTIONS = ("detector", "sensors", "reference", "filter", "composite", "dineof", "loop")

DEFAULTS = {
    "data": {
        "aoi": "admiralty_inlet",
        "source": "data/unfiltered/admiralty_inlet.zarr",
        "out_dir": "data/pipeline",
        "watervar": "landcover_water",
        "time_range": None,             # [start, end] inclusive, or null for the whole cube
        # Channels copied from the source cube unchanged. `insitu_*` channels and the hour
        # channels are added automatically when present, so validation runs off this cube alone.
        "carry": ["landcover_water", "depth_cudem", "elevation_cudem"],
    },
    "matchup": {
        **copy.deepcopy(C.DEFAULTS["matchup"]),
        # footprint: each MODIS footprint (reconstructed from the anchor's nearest-neighbour
        #   patches) against the MEDIAN of the sensor's kept pixels inside it. pixel: every
        #   overlapping 100 m cell is its own pair (composite.scene_matchups).
        "aggregate": "footprint",
        "min_footprints": 10,           # footprint pairs a scene needs to enter the fit
        "min_footprint_cells": 20,      # water + MODIS-valid cells a footprint needs
        "min_footprint_cover": 0.5,     # share of those cells the sensor observed and kept
        # Iterative outlier removal on the pairs: drop |resid| > k * robust SD, refit, repeat
        # until the removed set stops changing. null disables.
        "outlier_k": 1.5,
        "outlier_max_iter": 20,
    },
    "offset": {**copy.deepcopy(C.DEFAULTS["offset"]),
               "diurnal_harmonics": {},     # sensor id -> K requested (default 1)
               "slope_estimator": "ols",    # ols | rma, applied when `slope` is on
               "slope_clip": [0.5, 2.0]},   # a slope outside this is an error
    "holdout": {
        "frac": 0.2,                    # share of eligible dates held out for the point CV
        "hold": ["lst", "eco"],         # members removed on those dates
        "valid_pixels": 0.5,            # a date is eligible when a hold member keeps this share
        "seed": 0,
    },
    "seasonal": {k: v for k, v in S.DEFAULTS["seasonal"].items()},
    "scale": {k: v for k, v in S.DEFAULTS["scale"].items()},
    "final": {
        "coarsen": 1,                   # grid of the final fill; 1 = native
        "em": {},                       # overrides of dineof.em for the final fit
        "reclassify": False,            # re-run the cloud classifier against the final smooth field
    },
    "output": {
        "chunks": {"time": 64, "y": 128, "x": 128},
        "compression": {"codec": "zstd", "level": 5, "shuffle": "shuffle"},
        "write_figures": True,
        "figure_dpi": 130,
    },
}
for _s in ITER_SECTIONS:
    DEFAULTS[_s] = {}                   # OPAQUE here: validated by iterative_filter
OPAQUE = set(ITER_SECTIONS) | {"final"}


def load_config(path: Path) -> dict:
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    return build_config(user, path)


def build_config(user: dict, path: Path | str = "<dict>", *, resolve: bool = True) -> dict:
    """The pipeline config, with the iterative filter's own config built and validated inside
    it as cfg["_iter"]."""
    cfg = copy.deepcopy(DEFAULTS)
    for section, values in user.items():
        if section not in cfg:
            raise ValueError(f"{path}: unknown config section '{section}'")
        if section in ITER_SECTIONS:
            cfg[section] = copy.deepcopy(values or {})
            continue
        for key, value in (values or {}).items():
            if key not in cfg[section]:
                raise ValueError(f"{path}: unknown key '{section}.{key}'")
            cfg[section][key] = value
    for k in cfg["final"]:
        if k not in DEFAULTS["final"]:
            raise ValueError(f"{path}: unknown key 'final.{k}'")
    if resolve:
        for key in ("source", "out_dir"):
            p = Path(cfg["data"][key])
            cfg["data"][key] = (p if p.is_absolute() else ROOT / p).resolve()

    it_user = {s: cfg[s] for s in ITER_SECTIONS if cfg[s]}
    it_user["data"] = {"aoi": cfg["data"]["aoi"], "source": str(cfg["data"]["source"]),
                       "watervar": cfg["data"]["watervar"]}
    # The pipeline runs its own CV search after the refit; the loop's would be wasted.
    it_user.setdefault("loop", {})
    it_user["loop"] = {**it_user["loop"], "final_cv": False}
    cfg["_iter"] = F.build_config(it_user, path, resolve=resolve)

    f = cfg["final"]
    if int(f["coarsen"]) < 1:
        raise ValueError(f"{path}: final.coarsen must be >= 1")
    if int(f["coarsen"]) > int(cfg["_iter"]["_edineof"]["matrix"]["coarsen"]):
        raise ValueError(f"{path}: final.coarsen must not exceed dineof.matrix.coarsen")
    if int(cfg["_iter"]["_edineof"]["matrix"]["coarsen"]) % int(f["coarsen"]):
        raise ValueError(f"{path}: dineof.matrix.coarsen must be a multiple of final.coarsen")
    for k in f["em"]:
        if k not in E.DEFAULTS["em"]:
            raise ValueError(f"{path}: unknown key 'final.em.{k}'")
    h = cfg["holdout"]
    if not 0 <= float(h["frac"]) < 1:
        raise ValueError(f"{path}: holdout.frac must be in [0, 1)")
    for mid in h["hold"]:
        if mid not in cfg["_iter"]["sensors"]:
            raise ValueError(f"{path}: holdout.hold names {mid!r}, which is not a sensor")
    tr = cfg["data"]["time_range"]
    if tr is not None and (len(tr) != 2 or pd.Timestamp(tr[0]) > pd.Timestamp(tr[1])):
        raise ValueError(f"{path}: data.time_range must be [start, end] or null")
    validate_matchup(cfg, path)
    return cfg


AGGREGATES = ("footprint", "pixel")
SLOPE_ESTIMATORS = ("ols", "rma")


def validate_matchup(cfg: dict, path) -> None:
    m, o = cfg["matchup"], cfg["offset"]
    if m["aggregate"] not in AGGREGATES:
        raise ValueError(f"{path}: matchup.aggregate must be one of {AGGREGATES}")
    for k in ("min_footprints", "min_footprint_cells", "outlier_max_iter"):
        if int(m[k]) < 1:
            raise ValueError(f"{path}: matchup.{k} must be >= 1")
    if not 0.0 <= float(m["min_footprint_cover"]) <= 1.0:
        raise ValueError(f"{path}: matchup.min_footprint_cover must be in [0, 1]")
    if m["outlier_k"] is not None and not float(m["outlier_k"]) > 0:
        raise ValueError(f"{path}: matchup.outlier_k must be > 0, or null to disable")
    if o["slope_estimator"] not in SLOPE_ESTIMATORS:
        raise ValueError(f"{path}: offset.slope_estimator must be one of {SLOPE_ESTIMATORS}")
    lo, hi = (float(v) for v in o["slope_clip"])
    if not 0 < lo < hi:
        raise ValueError(f"{path}: offset.slope_clip must be [low, high] with 0 < low < high")


def offset_config(cfg: dict) -> dict:
    """What composite.scene_matchups / fit_offset / reference_fit / apply_offset read."""
    it = cfg["_iter"]
    dh = cfg["offset"].get("diurnal_harmonics") or {}
    members = {it["reference"]["id"]: {"label": "MODIS", "var": it["reference"]["var"],
                                        "diurnal_harmonics": 0, "weight": 1.0}}
    for sid, s in it["sensors"].items():
        members[sid] = {"label": s["label"], "var": s["sst"],
                        "diurnal_harmonics": int(dh.get(sid, 1)), "weight": 1.0}
    off = {k: v for k, v in cfg["offset"].items() if k != "diurnal_harmonics"}
    return {"members": members, "offset": off, "matchup": dict(cfg["matchup"]),
            "reference": it["reference"]["id"]}


# ==================================================================== bootstrap fits

# ==================================================================== offset matchups

def modis_footprints(ref2d: np.ndarray) -> np.ndarray:
    """(H, W) int32 footprint labels, 1..K; 0 where MODIS has no value.

    MODIS was nearest-neighbour resampled from its ~1 km swath, so every 100 m cell drawn from
    one native pixel carries that pixel's exact value. A footprint is therefore a 4-connected
    patch of identical values. Two adjacent footprints that happen to share a value merge into
    one patch, which is harmless here: the patch still has a single MODIS value to compare
    against.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    H, W = ref2d.shape
    fin = np.isfinite(ref2d)
    idx = np.arange(H * W).reshape(H, W)
    eqr = fin[:, :-1] & fin[:, 1:] & (ref2d[:, :-1] == ref2d[:, 1:])
    eqd = fin[:-1] & fin[1:] & (ref2d[:-1] == ref2d[1:])
    r = np.concatenate([idx[:, :-1][eqr], idx[:-1][eqd]])
    c = np.concatenate([idx[:, 1:][eqr], idx[1:][eqd]])
    g = coo_matrix((np.ones(r.size, bool), (r, c)), shape=(H * W, H * W))
    _, comp = connected_components(g, directed=False)
    labels = np.zeros(H * W, "int32")
    _, compact = np.unique(comp[fin.ravel()], return_inverse=True)
    labels[fin.ravel()] = compact.astype("int32") + 1
    return labels.reshape(H, W)


def footprint_pairs(mem2d: np.ndarray, ref2d: np.ndarray, labels: np.ndarray,
                    min_cells: int, min_cover: float) -> pd.DataFrame:
    """One row per usable footprint: fp, modis, sensor_median, n_cells, n_sensor.

    `n_cells` counts the footprint's water + MODIS-valid cells (`ref2d` is already masked to
    both), so a coastal sliver of a footprint is judged on what is actually water.
    """
    on = labels > 0
    df = pd.DataFrame({"fp": labels[on], "modis": ref2d[on].astype(float),
                       "y": mem2d[on].astype(float)})
    g = df.groupby("fp", sort=True)
    out = pd.DataFrame({"modis": g["modis"].first(), "n_cells": g.size(),
                        "n_sensor": g["y"].count(), "sensor_median": g["y"].median()})
    ok = ((out["n_cells"] >= int(min_cells)) & (out["n_sensor"] > 0)
          & (out["n_sensor"] >= float(min_cover) * out["n_cells"]))
    return out[ok].reset_index()


def matchup_pairs(mem: np.ndarray, ref: np.ndarray, hours: np.ndarray, times, cfg: dict,
                  sid: str) -> pd.DataFrame:
    """Every (MODIS, sensor) pair on every date both observed, at footprint or pixel support.

    Columns: t, date, hour, modis, sensor (the footprint median, or the pixel value), n_cells,
    n_sensor. Scenes with fewer than min_footprints / min_pixels pairs are left out.
    """
    m = cfg["matchup"]
    foot = m["aggregate"] == "footprint"
    min_n = int(m["min_footprints"] if foot else m["min_pixels"])
    k = int(m["smooth_to_reference_px"])
    parts = []
    for t in range(mem.shape[0]):
        if not (np.isfinite(mem[t]).any() and np.isfinite(ref[t]).any()):
            continue
        if foot:
            fp = footprint_pairs(mem[t], ref[t], modis_footprints(ref[t]),
                                 m["min_footprint_cells"], m["min_footprint_cover"])
            df = pd.DataFrame({"fp": fp["fp"], "modis": fp["modis"],
                               "sensor": fp["sensor_median"], "n_cells": fp["n_cells"],
                               "n_sensor": fp["n_sensor"]})
        else:
            m2 = C.box_mean(mem[t], k) if k else mem[t]
            ok = np.isfinite(m2) & np.isfinite(ref[t])
            df = pd.DataFrame({"fp": np.flatnonzero(ok.ravel()), "modis": ref[t][ok].astype(float),
                               "sensor": m2[ok].astype(float), "n_cells": 1, "n_sensor": 1})
        if len(df) < min_n:
            continue
        parts.append(df.assign(t=t, date=str(times[t])[:10], hour=float(hours[t])))
    # `fp` is the footprint label (footprint mode) or the flat cell index (pixel mode) in that
    # day's grid, so a pair can be mapped back.
    cols = ["t", "date", "hour", "fp", "modis", "sensor", "n_cells", "n_sensor"]
    if not parts:
        log.warning("%s: no scene has %d %s matchups with MODIS", sid, min_n,
                    "footprint" if foot else "pixel")
        return pd.DataFrame(columns=cols)
    return pd.concat(parts, ignore_index=True)[cols]


def anomaly_slope(x: np.ndarray, y: np.ndarray, scene: np.ndarray,
                  estimator: str) -> float:
    """Slope of y on x from within-scene anomalies (scene means removed), as
    composite.slope_diagnostics: OLS, or RMA = sd(y) / sd(x) signed by the correlation."""
    df = pd.DataFrame({"x": x, "y": y, "s": scene})
    xa = df["x"] - df.groupby("s")["x"].transform("mean")
    ya = df["y"] - df.groupby("s")["y"].transform("mean")
    if len(df) < 3 or xa.std() == 0:
        return float("nan")
    if estimator == "ols":
        return float(np.polyfit(xa, ya, 1)[0])
    r = np.corrcoef(xa, ya)[0, 1]
    return float(np.sign(r) * ya.std() / xa.std())


def robust_anomaly_slope(x: np.ndarray, y: np.ndarray, scene: np.ndarray,
                         max_pairs: int = 3000, seed: int = 0) -> float:
    """Theil-Sen slope of y on x from within-scene anomalies (scene MEDIANS removed).

    The median of all pairwise slopes: robust to ~29% contamination and, unlike OLS or RMA on
    a clipped set, not shaped by which points were removed. Computed on a fixed random sample
    of at most `max_pairs` points, since it is O(n^2).
    """
    from scipy.stats import theilslopes
    df = pd.DataFrame({"x": x, "y": y, "s": scene})
    xa = (df["x"] - df.groupby("s")["x"].transform("median")).to_numpy()
    ya = (df["y"] - df.groupby("s")["y"].transform("median")).to_numpy()
    if xa.size > max_pairs:
        i = np.random.default_rng(seed).choice(xa.size, max_pairs, replace=False)
        xa, ya = xa[i], ya[i]
    if xa.size < 3 or np.ptp(xa) == 0:
        return float("nan")
    return float(theilslopes(ya, xa)[0])


def clip_pairs(pairs: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Iterative outlier removal on the matchup pairs. Returns (pairs + resid, kept; info).

    Each round refits on the KEPT pairs -- each scene's offset as the median of (y - x) -- then
    judges EVERY pair against its own scene's fit, r = y - (offset_scene + x), and keeps
    |r - median(r)| <= k * s with s = 1.4826 * MAD(r) over all pairs.

    The slope inside the loop is FIXED for the whole clipping: 1 when offset.slope is off, and a
    Theil-Sen slope of ALL pairs when it is on. The chosen estimator (OLS or RMA) is then applied
    once, to the pairs that survive. Two alternatives were tried and rejected on data:
      - re-estimating OLS/RMA every round feeds back on itself -- an attenuated slope makes the
        largest-anomaly pairs look like outliers, removing them attenuates it further (OLS ran
        to 0.46 on the real ECOSTRESS pairs, RMA to 2.02 on Landsat);
      - clipping at slope 1 regardless biases a real gain toward 1, because residuals then grow
        with the anomaly (a true 1.15 came back as 1.07). The scale is robust, so the outliers being removed barely inflate it and
    the cut converges; a plain SD of the kept pairs would shrink every round and, at k = 1.5,
    never settle. Every pair is re-judged each round, so one removed early can come back.
    Stops when the kept set repeats the previous round's (converged), revisits an earlier one
    (oscillation), or reaches matchup.outlier_max_iter.
    """
    m, o = cfg["matchup"], cfg["offset"]
    k = m["outlier_k"]
    x = pairs["modis"].to_numpy(float)
    y = pairs["sensor"].to_numpy(float)
    scene = pairs["t"].to_numpy(int)
    slope_on = bool(o["slope"])
    est = o["slope_estimator"]
    keep = np.ones(len(pairs), bool)
    info = dict(n_pairs=int(len(pairs)), n_removed=0, rounds=0, converged=True,
                scale=float("nan"), b=1.0, history=[])

    b_clip = 1.0
    if slope_on and len(pairs):
        b_clip = robust_anomaly_slope(x, y, scene)
        if not np.isfinite(b_clip):
            b_clip = 1.0
    info["b_clip"] = b_clip

    def fit(kmask):
        b = b_clip
        d = pd.Series(y[kmask] - b * x[kmask]).groupby(scene[kmask]).median()
        # A scene with no kept pair is judged against the median of all its pairs.
        d_all = pd.Series(y - b * x).groupby(scene).median()
        off = d.reindex(d_all.index).fillna(d_all)
        return b, y - (off.reindex(scene).to_numpy() + b * x)

    def slope_of(kmask):
        if not slope_on:
            return 1.0
        b = anomaly_slope(x[kmask], y[kmask], scene[kmask], est)
        return b if np.isfinite(b) else 1.0

    if k is None or len(pairs) == 0:
        _, r = fit(keep)
        info["b"] = slope_of(keep)
        return pairs.assign(resid=r, kept=keep), info

    seen = {keep.tobytes()}
    for rnd in range(1, int(m["outlier_max_iter"]) + 1):
        b, r = fit(keep)
        rc = r - np.median(r)
        s = 1.4826 * float(np.median(np.abs(rc)))
        if s <= 0:
            info.update(rounds=rnd, scale=s)
            break
        new = np.abs(rc) <= float(k) * s
        info.update(rounds=rnd, scale=s)
        info["history"].append(dict(round=rnd, removed=int((~new).sum()), scale=s))
        if np.array_equal(new, keep):
            break
        key = new.tobytes()
        keep = new
        if key in seen:
            info["converged"] = False
            log.warning("outlier clipping is cycling between kept sets after %d rounds; "
                        "stopping there", rnd)
            break
        seen.add(key)
    else:
        info["converged"] = False
        log.warning("outlier clipping did not converge in %d rounds", int(m["outlier_max_iter"]))
    _, r = fit(keep)
    info.update(n_removed=int((~keep).sum()), b=slope_of(keep))
    return pairs.assign(resid=r, kept=keep), info


def slope_pivot(pairs: pd.DataFrame) -> float:
    """The temperature the slope pivots about: the median MODIS value of the kept pairs."""
    k = pairs[pairs["kept"]] if "kept" in pairs else pairs
    return float(np.median(k["modis"])) if len(k) else 0.0


def scene_table(pairs: pd.DataFrame, b: float, sid: str, cfg: dict,
                pivot: float = 0.0) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """composite.scene_matchups' (table, ref_px, mem_px) from the KEPT pairs.

    Each scene's delta is the median of sensor - [pivot + b (MODIS - pivot)]: the offset AT the
    pivot temperature. With b = 1 that is plain sensor - MODIS. With b != 1, pivoting keeps the
    offset physical -- an intercept at 0 K would be tens of kelvin and fail max_abs_delta and
    offset.clip for a gain of 1.2.
    """
    m = cfg["matchup"]
    min_n = int(m["min_footprints"] if m["aggregate"] == "footprint" else m["min_pixels"])
    max_abs = float(m["max_abs_delta"])
    rows, rx, ry = [], [], []
    for t, g in pairs[pairs["kept"]].groupby("t", sort=True):
        if len(g) < min_n:
            continue
        x, y = g["modis"].to_numpy(float), g["sensor"].to_numpy(float)
        d = y - (pivot + b * (x - pivot))
        delta = float(np.median(d))
        if abs(delta) > max_abs:
            log.warning("%s: %s rejected from the fit, |delta| = %.2f K exceeds "
                        "matchup.max_abs_delta = %.2f", sid, g["date"].iloc[0], abs(delta),
                        max_abs)
            continue
        rows.append(dict(member=sid, date=g["date"].iloc[0], t=int(t), n=len(g), delta=delta,
                         sd=float(1.4826 * np.median(np.abs(d - delta))),
                         hour=float(g["hour"].iloc[0]), mem_mean=float(y.mean()),
                         ref_mean=float(x.mean())))
        rx.append(x)
        ry.append(y)
    table = pd.DataFrame(rows, columns=["member", "date", "t", "n", "delta", "sd", "hour",
                                        "mem_mean", "ref_mean"])
    return (table, np.concatenate(rx) if rx else np.empty(0),
            np.concatenate(ry) if ry else np.empty(0))


def fit_offsets(raw: F.Raw, keep: dict, cfg: dict) -> tuple[dict, dict, pd.DataFrame,
                                                             pd.DataFrame, pd.DataFrame]:
    """Each sensor's offset (and optional slope) against MODIS on the kept pixels only.

    Returns (sid -> (T,) offset in K on EVERY date, sid -> slope divisor, the offsets.csv-style
    report, the per-scene matchups table, the pair table). The offset is evaluated on every
    date -- not just the ones that matched -- because the loop may keep scenes that had no
    MODIS matchup.

    Order: pairs (footprint or pixel) -> iterative outlier clipping -> slope b (1 unless
    offset.slope) -> per-scene deltas median(sensor - b * MODIS) -> the diurnal offset model
    of composite.fit_offset. The model is sensor = x0 + a(hour) + b (MODIS - x0), pivoted at
    x0 = the median MODIS value of the kept pairs, so `a` is the physical offset at a typical
    temperature. The correction is MODIS-scale = (sensor - off) / b with
    off = a(hour) + x0 (1 - b): the (offset, slope) form the rest of the pipeline applies.
    """
    ocfg = offset_config(cfg)
    # composite.fit_offset's own slope branch only warns that applying slope_ols is a
    # diagnostic mode; the slope is handled here, so its offset fit runs at slope off.
    fcfg = {**ocfg, "offset": {**ocfg["offset"], "slope": False}}
    o = cfg["offset"]
    times = pd.to_datetime(raw.times)
    ref_id = raw.order[0]
    stats, tables, all_pairs = [C.reference_fit(ocfg, ref_id)["stats"]], [], []
    offsets, slope = {}, {}
    unit = cfg["matchup"]["aggregate"]
    for sid, r in raw.raw.items():
        mem = np.where(keep[sid], r, np.nan).astype("float32")
        pairs = matchup_pairs(mem, raw.ref, raw.hours[sid], times, cfg, sid)
        pairs, info = clip_pairs(pairs, cfg)
        b = float(info["b"]) if o["slope"] else 1.0
        if o["slope"]:
            lo, hi = (float(v) for v in o["slope_clip"])
            if not lo <= b <= hi:
                raise ValueError(
                    f"{sid}: fitted slope {b:.3f} ({o['slope_estimator']}) is outside "
                    f"offset.slope_clip [{lo}, {hi}]; inspect the matchup pairs before "
                    "widening it")
        x0 = slope_pivot(pairs) if o["slope"] else 0.0
        table, ref_px, mem_px = scene_table(pairs, b, sid, cfg, x0)
        fit = C.fit_offset(table, ref_px, mem_px, fcfg, sid)
        offsets[sid] = (C.offset_at(fit["coefs"], fit["K"], raw.hours[sid])
                        + x0 * (1.0 - b)).astype("float64")
        slope[sid] = b
        table = table.assign(unit=unit, fitted=C.offset_at(
            fit["coefs"], fit["K"], table["hour"].to_numpy(float)) if len(table) else [])
        stats.append({**fit["stats"], "aggregate": unit, "slope_applied": b, "slope_pivot": x0,
                      "slope_estimator": o["slope_estimator"] if o["slope"] else "none",
                      "n_pairs": info["n_pairs"], "n_removed": info["n_removed"],
                      "clip_rounds": info["rounds"], "clip_converged": info["converged"],
                      "clip_scale": info["scale"]})
        tables.append(table)
        all_pairs.append(pairs.assign(sensor_id=sid))
        log.info("  offset %s: K=%d, mean %+.4f K over %d scenes (%s pairs: %d of %d removed "
                 "in %d rounds%s), slope %.3f%s", sid, fit["K"], float(np.mean(offsets[sid])),
                 int(fit["stats"]["n_scenes"]), unit, info["n_removed"], info["n_pairs"],
                 info["rounds"], "" if info["converged"] else ", NOT converged", b,
                 "" if o["slope"] else " (off)")
    return (offsets, slope, pd.DataFrame(stats),
            pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(),
            pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame())


def select_validation_dates(raw: F.Raw, keep: dict, cfg: dict) -> np.ndarray:
    """Seeded, duplicate-free draw of dates whose hold members keep enough of the water.

    Replaces composite.identify/select_highres_dates, which count raw (unmasked) pixels and
    draw with replacement from the global RNG.
    """
    h = cfg["holdout"]
    nw = raw.n_water
    eligible = set()
    for sid in h["hold"]:
        frac = keep[sid].sum(axis=(1, 2)) / max(nw, 1)
        eligible |= set(np.flatnonzero(frac > float(h["valid_pixels"])).tolist())
    cand = np.array(sorted(eligible), dtype=int)
    n = int(cand.size * float(h["frac"]))
    if n == 0:
        log.warning("holdout: no validation dates (%d eligible); point-CV will be empty",
                    cand.size)
        return np.array([], dtype=int)
    pick = np.random.default_rng(int(h["seed"])).choice(cand, size=n, replace=False)
    return np.sort(pick)


def fit_seasonal(raw: F.Raw, offsets: dict, slope: dict, keep: dict, valid_inds: np.ndarray,
                 cfg: dict) -> dict:
    """Per-pixel seasonal climatology and robust scale of the composite of the kept pixels.

    The same estimator standardize.py uses -- gappy per-pixel harmonic least squares with the
    FULL / MEAN_ONLY / REFERENCE tiers, then a dof-corrected 1.4826 * MAD scale -- called on the
    in-memory composite.
    """
    it = cfg["_iter"]
    sn, sc = cfg["seasonal"], cfg["scale"]
    # Offsets and the cloud mask are applied per block inside the composite: no full corrected
    # or masked copy of any sensor's cube is built.
    adj = {raw.order[0]: raw.ref, **raw.raw}
    comp = F.composite_blocked(adj, valid_inds, it, raw.order,
                               int(it["composite"]["block_days"]), keep=keep,
                               offsets={sid: (offsets[sid], slope[sid]) for sid in raw.raw})
    water = raw.water
    Y = comp["sst"][:, water].astype("float64")
    O = np.isfinite(Y)
    H = int(sn["n_harmonics"])
    P = 1 + 2 * H
    X = design_matrix(pd.to_datetime(raw.times), H, float(sn["period_days"]))
    coef, info = S.fit_harmonics_gappy(np.where(O, Y, 0.0), O, X,
                                       min_dates=int(sn["min_dates"]),
                                       max_cond=float(sn["max_cond"]))
    R = np.where(O, Y - X @ coef, np.nan)
    scale, _ = S.robust_scale(R, info["n_obs"], P, floor=float(sc["floor"]),
                              dof_correction=bool(sc["dof_correction"]),
                              fallback=sc["fallback"], fit_type=info["fit_type"])

    coef_g = np.full((P,) + water.shape, np.nan)
    coef_g[:, water] = coef
    scale_g = np.full(water.shape, np.nan)
    scale_g[water] = scale
    ft = np.full(water.shape, -1, dtype="int8")
    ft[water] = info["fit_type"]
    log.info("  seasonal: %d px full fit, %d mean-only, %d reference; median scale %.3f K",
             int((info["fit_type"] == FIT_FULL).sum()), int((info["fit_type"] == 1).sum()),
             int((info["fit_type"] == 0).sum()), float(np.median(scale)))
    return dict(coef=coef_g, scale=scale_g, fit_type=ft, n_harmonics=H,
                period_days=float(sn["period_days"]), comp=comp["sst"], src=comp["src"])


def bootstrap(raw: F.Raw, keep: dict, cfg: dict, label: str,
              fixed: tuple[dict, dict] | None = None) -> tuple[F.Inputs, dict]:
    """Offsets, validation dates and seasonal standardization on `keep` -> loop `Inputs`.

    `fixed` = (sid -> (T,) offset, sid -> slope) applies offsets decided elsewhere (stage 1 of
    the region pipeline) instead of fitting them: only the holdout dates and the seasonal
    standardization are fitted here.
    """
    t0 = time.time()
    if fixed is None:
        offsets, slope, report, matchups, pairs = fit_offsets(raw, keep, cfg)
    else:
        offsets, slope = fixed
        report = pd.DataFrame([dict(member=sid, offset_mean=float(np.mean(offsets[sid])),
                                    slope_applied=float(slope[sid]), source="fixed")
                               for sid in offsets])
        matchups, pairs = pd.DataFrame(), pd.DataFrame()
    valid_inds = select_validation_dates(raw, keep, cfg)
    sea = fit_seasonal(raw, offsets, slope, keep, valid_inds, cfg)
    inp = F.make_inputs(raw, offsets, slope, sea["coef"], sea["scale"], valid_inds,
                        sea["n_harmonics"], sea["period_days"])
    log.info("%s fits: %d validation dates, %.1fs", label, valid_inds.size, time.time() - t0)
    return inp, dict(offsets=offsets, slope=slope, report=report, matchups=matchups, pairs=pairs,
                     valid_inds=valid_inds, seasonal=sea)


# ==================================================================== final fit

def warm_from_coarse(res_c: dict, sel_c: dict, sel_f: dict) -> dict:
    """edineof's `warm` dict, in memory: the coarse fits' settings and their z fields
    block-repeated onto the finer matrix. The in-memory twin of edineof.load_warm_start."""
    ratio = int(sel_c["coarsen"]) // int(sel_f["coarsen"])
    out = {"k_point": int(res_c["k_opt"]), "tc_point": float(res_c["tc_opt"]),
           "k_day": int(res_c["k_day"]), "tc_day": float(res_c["tc_day"]),
           "coarsen": int(sel_c["coarsen"]), "from": "in-memory coarse fit"}
    for name, fit in (("point", res_c["point_fit"]), ("day", res_c["day_fit"])):
        G = F.to_grid(fit["X"], sel_c)
        if ratio > 1:
            G = E.upsample(G, ratio, sel_f["water"].shape)
        seed = F.from_grid(G, sel_f)
        seed[~np.isfinite(seed)] = 0.0
        out[name] = seed
    return out


def modes_grid(U: np.ndarray, sel: dict) -> np.ndarray:
    """(m, k) spatial modes -> (k, y, x) on the matrix grid, NaN off the matrix."""
    water, keep = sel["water"], sel["keep"]
    k = U.shape[1]
    full = np.full((k, int(water.sum())), np.nan, dtype="float32")
    full[:, keep] = U.T
    out = np.full((k,) + water.shape, np.nan, dtype="float32")
    out[:, water] = full
    return out


def to_native(G: np.ndarray, sel: dict, shape: tuple) -> np.ndarray:
    """A grid on the matrix's (possibly coarsened) grid, block-repeated to the native shape."""
    f = int(sel["coarsen"])
    return E.upsample(G, f, shape) if f > 1 else G


def final_fit(inp: F.Inputs, keep: dict, cfg: dict) -> dict:
    """Coarse CV search on the final mask, then one fit at `final.coarsen`, warm-started."""
    it = cfg["_iter"]
    t0 = time.time()
    sel_c, _ = F.build_matrix(inp, keep, it)
    res_c = E.edineof(sel_c["X"], sel_c["observed"], sel_c["valid_msk"], sel_c["t"],
                      it["_edineof"])
    log.info("coarse CV (%.0fs): point-opt k=%d T_c=%g, day-opt k=%d T_c=%g", time.time() - t0,
             res_c["k_opt"], res_c["tc_opt"], res_c["k_day"], res_c["tc_day"])

    fc = int(cfg["final"]["coarsen"])
    if fc == int(sel_c["coarsen"]):
        log.info("final.coarsen equals the loop's grid: the coarse fit is the final fit")
        return dict(res=res_c, sel=sel_c, res_c=res_c, sel_c=sel_c, cfg=it)
    it_f = copy.deepcopy(it)
    it_f["_edineof"]["matrix"]["coarsen"] = fc
    it_f["_edineof"]["em"].update(cfg["final"]["em"])
    t1 = time.time()
    sel_f, _ = F.build_matrix(inp, keep, it_f)
    warm = warm_from_coarse(res_c, sel_c, sel_f)
    log.info("final fit: %d px x %d dates at coarsen %d, warm-started from coarsen %d",
             sel_f["m"], sel_f["n"], fc, int(sel_c["coarsen"]))
    res_f = E.edineof(sel_f["X"], sel_f["observed"], sel_f["valid_msk"], sel_f["t"],
                      it_f["_edineof"], warm=warm)
    log.info("final fit done in %.0fs", time.time() - t1)
    return dict(res=res_f, sel=sel_f, res_c=res_c, sel_c=sel_c, cfg=it_f)


def smooth_field(inp: F.Inputs, fin: dict, cfg: dict, loop_fit: dict) -> dict:
    """The smooth field on the final grid: MODIS-only loadings on the final EOFs (or the
    full-data low-rank field when loop.baseline_source is `all`).

    The loadings are pooled over the SAME reach the cloud loop used (`loop.loading_tc`, else the
    loop's own fit T_c) -- not the final fit's. The final CV can choose T_c = 0, and pooling at
    0 would leave every day without its own MODIS (two thirds of them here) as climatology, so
    the shipped smooth field would not be the one that judged the scenes.
    """
    lp = cfg["_iter"]["loop"]
    res, sel = fin["res"], fin["sel"]
    fit = res["day_fit"] if lp["baseline"] == "day" else res["point_fit"]
    mu = float(res["mu"])
    if lp["baseline_source"] == "modis":
        s = (F.setting(lp["loading_tc"], fin["cfg"]["_edineof"]) if lp["loading_tc"] is not None
             else {"t_c": loop_fit["t_c"], "alpha": loop_fit["alpha"], "p": loop_fit["p"]})
        Xm, Om = F.modis_matrix(inp, sel)
        a, info = F.modis_loadings(fit["U"], mu, Xm, Om, sel["t"], s,
                                   float(lp["loading_ridge"]), int(lp["loading_min_px"]))
        z = fit["U"] @ a + mu
        log.info("smooth field: MODIS loadings T_c=%g -- %d days direct, %d pooled, %d "
                 "climatology", info["t_c"], info["direct"], info["pooled"], info["fallback"])
    else:
        a, info, z = None, {}, fit["lowrank"]
    return dict(K=F.baseline_kelvin(z, sel, inp), loadings=a, info=info, fit=fit, mu=mu)


def filled_fields(inp: F.Inputs, fin: dict) -> dict:
    """Point-, day- and merged filled fields in K on the native grid.

    Pixels the matrix dropped (never observed) are NaN, not climatology: a pixel with no data
    must not come back as a plausible series indistinguishable from a reconstructed one.
    """
    res, sel = fin["res"], fin["sel"]
    shape = inp.water.shape
    in_matrix = np.isfinite(to_native(F.to_grid(np.ones((sel["m"], 1)), {**sel, "n": 1}),
                                      sel, shape)[0])
    out = {}
    for name in ("point", "day"):
        K = F.baseline_kelvin(res[f"{name}_fit"]["X"], sel, inp)
        K[:, ~in_matrix] = np.nan
        out[name] = K
    has_data = sel["observed"].sum(axis=0) > 0
    out["merged"] = np.where(has_data[:, None, None], out["point"], out["day"]).astype("float32")
    obs = to_native(F.to_grid(sel["observed"].astype("float32"), sel), sel, shape)
    out["observed"] = (np.nan_to_num(obs) > 0.5).astype("int8")
    out["constrained"] = has_data.astype("int8")
    return out


# ==================================================================== output

def read_carry(cfg: dict, times: np.ndarray) -> tuple[dict, dict]:
    """(channels, attrs) copied from the source cube: the configured list, every `insitu_*`
    channel, and the hour channels."""
    it = cfg["_iter"]
    with xr.open_zarr(cfg["data"]["source"]) as src:
        names = list(dict.fromkeys(
            list(cfg["data"]["carry"])
            + [v for v in src.data_vars if str(v).startswith("insitu")
               or "_insitu_" in str(v)]
            + [it["reference"]["hour"]] + [s["hour"] for s in it["sensors"].values()]))
        out = {}
        for n in names:
            if n not in src:
                log.warning("carry: %s is not in the source cube; skipped", n)
                continue
            da = src[n]
            if "time" in da.dims:
                da = da.sel(time=times)
            arr = xr.DataArray(da.values, dims=da.dims, name=n)
            arr.attrs.update(da.attrs, carried_from=str(cfg["data"]["source"]))
            out[n] = arr
        attrs = dict(src.attrs)
    return out, attrs


def build_output(raw: F.Raw, inp: F.Inputs, loop_out: dict, fits: dict, fin: dict,
                 smooth: dict, filled: dict, cloud: dict, cfg: dict) -> xr.Dataset:
    it = cfg["_iter"]
    dims = ("time", "y", "x")
    data, src_attrs = read_carry(cfg, raw.times)
    shape = raw.water.shape

    def put(name, arr, dims_, **attrs):
        da = xr.DataArray(arr, dims=dims_, name=name)
        da.attrs.update(attrs)
        data[name] = da

    # --- filled ---
    put("sst_filled", filled["merged"], dims, units="K",
        long_name="eDINEOF gap-filled SST, per-date best of the point and day tunings",
        comment=("observations held fixed; the point-tuned fit on dates with data, the "
                 "day-tuned fit on dates without. NaN on water pixels never observed."))
    put("sst_filled_point", filled["point"].astype("float32"), dims, units="K",
        long_name="gap-filled SST at the point-CV setting", k_modes=int(fin["res"]["k_opt"]),
        cutoff_days=float(fin["res"]["tc_opt"]))
    put("sst_filled_day", filled["day"].astype("float32"), dims, units="K",
        long_name="gap-filled SST at the day-CV setting", k_modes=int(fin["res"]["k_day"]),
        cutoff_days=float(fin["res"]["tc_day"]))
    put("sst_filled_observed", filled["observed"], dims, units="1",
        long_name="entry was observed and held fixed in the final fit")
    put("sst_filled_constrained", filled["constrained"], ("time",), units="1",
        long_name="date has at least one observation of its own")

    # --- smooth ---
    put("sst_smooth", smooth["K"].astype("float32"), dims, units="K",
        long_name="smooth SST field the cloud filter compares scenes against",
        baseline_source=it["loop"]["baseline_source"], loop_baseline=it["loop"]["baseline"],
        comment=("U a(t) + mu, unstandardized: the final fit's spatial modes U with per-day "
                 "loadings fit to MODIS pixels only (baseline_source modis), so a scene never "
                 "steers the field that judges it."))
    if smooth["loadings"] is not None:
        put("loadings_modis", smooth["loadings"].astype("float32"), ("mode", "time"),
            long_name="MODIS-only loadings of the smooth field", units="1")
        put("smooth_modis_px", smooth["info"]["modis_px"], ("time",), units="1",
            long_name="MODIS cells on the final matrix grid, per day")
        put("smooth_loading_status", smooth["info"]["status"], ("time",), units="1",
            flag_values="0 1 2", flag_meanings="own_modis neighbours_modis none_climatology")

    # --- EOFs ---
    # Each fit gets its own mode dimension: the point- and day-tuned fits can keep different
    # numbers of modes (k=2 and k=1 on Admiralty Inlet), and one shared `mode` cannot hold both.
    sel = fin["sel"]
    for suffix, fit in (("", smooth["fit"]), ("_point", fin["res"]["point_fit"])):
        md = f"mode{suffix}"
        U = to_native(modes_grid(fit["U"], sel), sel, shape)
        put(f"eof_U{suffix}", U, (md, "y", "x"), units="1",
            long_name="spatial EOF modes (unit-norm columns, standardized units)",
            k=int(fit["k"]), cutoff_days=float(fit["t_c"]))
        put(f"eof_sigma{suffix}", fit["sigma"].astype("float32"), (md,), units="1")
        put(f"eof_V{suffix}", fit["V"].T.astype("float32"), (md, "time"), units="1",
            long_name="temporal modes (orthonormal)")
        put(f"loadings_full{suffix}", (fit["sigma"][:, None] * fit["V"].T).astype("float32"),
            (md, "time"), units="1", long_name="full-data loadings sigma * V")

    # --- cloud filter ---
    tbl = cloud["table"]
    for sid in raw.raw:
        s = it["sensors"][sid]
        put(s["sst"], raw.raw[sid], dims, units="K", long_name=f"{s['label']} SST, raw")
        put(f"{s['sst']}_filtered", np.where(cloud["keep"][sid], raw.raw[sid], np.nan)
            .astype("float32"), dims, units="K",
            long_name=f"{s['label']} SST, cloud-filtered (raw value where kept)")
        put(f"{sid}_keep", cloud["keep"][sid].astype("int8"), dims, units="1",
            long_name=f"{s['label']} pixel kept by the cloud filter")
        put(f"{sid}_p_valid", cloud["p_valid"][sid], dims, units="1",
            long_name=f"{s['label']} P(clear) against the smooth field")
        centre = np.full(len(raw.times), np.nan, dtype="float32")
        sub = tbl[tbl["sensor"] == sid]
        centre[sub["t"].to_numpy()] = sub["center"].to_numpy()
        put(f"{sid}_center", centre, ("time",), units="K",
            long_name=f"{s['label']} scene offset against the smooth field")
        put(f"{sid}_keep_history", loop_out["keep_bits"][sid], dims, units="1",
            n_iterations=int(loop_out["n_iter"]),
            long_name="per-iteration keep verdicts as bits (bit 0 = start, i+1 = after iter i)")

    # --- inputs and standardization (the final refit) ---
    fo = fits["final"]
    for sid in raw.raw:
        put(f"{sid}_offset", fo["offsets"][sid].astype("float32"), ("time",), units="K",
            long_name=f"offset removed from {sid}", slope=float(fo["slope"][sid]),
            comment="MODIS-scale = (sensor - offset) / slope")
    sea = fo["seasonal"]
    put("sst_seasonal_coef", sea["coef"].astype("float32"), ("term", "y", "x"), units="K",
        n_harmonics=int(sea["n_harmonics"]), period_days=float(sea["period_days"]),
        long_name="per-pixel seasonal harmonic coefficients (final fit)")
    put("sst_seasonal_sd", sea["scale"].astype("float32"), ("y", "x"), units="K",
        long_name="per-pixel robust residual scale (final fit)")
    put("sst_seasonal_fit_type", sea["fit_type"], ("y", "x"), units="1",
        flag_values="-1 0 1 2", flag_meanings="land reference mean_only full")
    vmsk = np.zeros((len(raw.times),) + shape, dtype="int8")
    if fo["valid_inds"].size:
        comp_valid = fin["res"]["point_cv"]
        vm = to_native(F.to_grid(comp_valid.astype("float32"), sel), sel, shape)
        vmsk = (np.nan_to_num(vm) > 0.5).astype("int8")
    put("validation_msk", vmsk, dims, units="1",
        long_name="point-CV holdout of the final fit")
    put("sst_composite", sea["comp"].astype("float32"), dims, units="K",
        long_name="offset-corrected composite of the kept pixels (final offsets)",
        comment="the observations the final fill was fit to")
    put("sst_composite_src", sea["src"], dims, units="1",
        flag_meanings=" ".join(raw.order),
        flag_masks=json.dumps([1 << i for i in range(len(raw.order))]))

    ds = xr.Dataset(data, coords={"time": raw.times, **raw.coords})
    for v in ds.data_vars:
        ds[v].encoding = {}
    res = fin["res"]
    rp, rd = F.cv_scores(fin["res_c"])
    fit_info = {"k_point": int(res["k_opt"]), "tc_point": float(res["tc_opt"]),
                "k_day": int(res["k_day"]), "tc_day": float(res["tc_day"]),
                "coarse_rmse_point": rp, "coarse_rmse_day": rd,
                "final_coarsen": int(sel["coarsen"]), "mu": float(res["mu"]),
                "loop_converged": bool(loop_out["converged"]),
                "loop_iterations": int(loop_out["n_iter"])}
    spec = {k: v for k, v in cfg.items() if not k.startswith("_")}
    ds.attrs.update({**{k: v for k, v in src_attrs.items()
                        if k in ("crs", "aoi_id", "met_time", "insitu_stations")},
                     "pipeline_config": yaml.safe_dump(json.loads(json.dumps(spec, default=str)),
                                                       sort_keys=False),
                     "pipeline_fit": json.dumps(fit_info),
                     "pipeline_channels": json.dumps(sorted(map(str, ds.data_vars))),
                     "pipeline_created_at": provenance.now_utc(),
                     "source_cube": str(cfg["data"]["source"]),
                     "package_version": provenance.package_version(),
                     "code_version": provenance.code_version()})
    return ds


def write_cube(ds: xr.Dataset, dest: Path, cfg: dict) -> None:
    compression = CompressionSpec(**cfg["output"]["compression"])
    chunks = dict(cfg["output"]["chunks"])
    encoding = datacube.build_encoding(ds, compression, chunks)
    store.sweep_scratch(dest)
    if dest.exists():
        log.info("replacing existing cube at %s", dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with store.atomic(dest) as tmp:
        datacube.write_zarr(ds, tmp, encoding)
    log.info("wrote %s", dest)


# ==================================================================== driver

def select_times(cfg: dict) -> np.ndarray | None:
    """The source cube's dates inside data.time_range, or None for all of them."""
    tr = cfg["data"]["time_range"]
    if tr is None:
        return None
    with xr.open_zarr(cfg["data"]["source"]) as src:
        allt = pd.to_datetime(src["time"].values)
    m = (allt >= pd.Timestamp(tr[0])) & (allt <= pd.Timestamp(tr[1]))
    if not m.any():
        raise SystemExit(f"data.time_range {tr} selects no dates of the source cube")
    return np.asarray(allt[m].values)


def run_pipeline(cfg: dict, *, tag: str | None = None, figures: bool = True,
                 offsets: tuple[dict, dict] | None = None, paths: dict | None = None,
                 extra_attrs: dict | None = None) -> dict:
    """Run every stage. Returns the paths written and the in-memory results.

    offsets      (sid -> (T,) offset, sid -> slope) to apply as given; neither the bootstrap
                 nor the refit then fits offsets (the region pipeline's stage 2).
    paths        override `cube`, `reports` and `figures` output locations.
    extra_attrs  added to the output cube's attributes.
    """
    it = cfg["_iter"]
    sfx = f"_{tag}" if tag else ""
    out_dir = Path(cfg["data"]["out_dir"])
    paths = paths or {}
    cube_path = Path(paths.get("cube", out_dir / f"{cfg['data']['aoi']}_pipeline{sfx}.zarr"))
    rep_dir = Path(paths.get("reports", out_dir / f"reports{sfx}"))
    fig_dir = Path(paths.get("figures", out_dir / f"figures{sfx}"))
    t_all = time.time()

    # 1. load
    t0 = time.time()
    times = select_times(cfg)
    raw = F.load_raw(it, times)
    if offsets is not None:
        missing = [sid for sid in raw.raw if sid not in offsets[0]]
        if missing:
            raise ValueError(f"fixed offsets are missing sensors {missing}")
        offsets = ({sid: np.broadcast_to(np.asarray(offsets[0][sid], float),
                                         (len(raw.times),)).copy() for sid in raw.raw},
                   {sid: float(offsets[1][sid]) for sid in raw.raw})
    log.info("[1/9] loaded %d dates %s .. %s, %d water px in %.0fs", len(raw.times),
             str(raw.times[0])[:10], str(raw.times[-1])[:10], raw.n_water, time.time() - t0)

    # 2-4. bootstrap fits on QC-only data
    keep0 = F.qc_only_keep(raw, it["filter"]["require_qc"])
    inp0, boot = bootstrap(raw, keep0, cfg, "[2-4/9] bootstrap", fixed=offsets)

    # 5. cloud loop
    t0 = time.time()
    loop_out = F.run_loop(inp0, it)
    log.info("[5/9] cloud loop: %d iterations, converged=%s, %.0fs", loop_out["n_iter"],
             loop_out["converged"], time.time() - t0)
    del inp0

    # 6. refit on the final mask, 7. coarse CV + final fit
    inp, fin_fits = bootstrap(raw, loop_out["keep"], cfg, "[6/9] refit", fixed=offsets)
    t0 = time.time()
    fin = final_fit(inp, loop_out["keep"], cfg)
    log.info("[7/9] final fit in %.0fs", time.time() - t0)

    # 8. smooth field (and optional re-classification against it)
    smooth = smooth_field(inp, fin, cfg, loop_out["last"]["fit"])
    cloud = {"keep": loop_out["keep"], "p_valid": loop_out["p_valid"], "table": loop_out["table"]}
    if cfg["final"]["reclassify"]:
        cls = F.classify_all(inp, smooth["K"], it)
        flips = F.flip_fraction(inp, loop_out["keep"], cls["keep"])
        log.info("reclassified against the final smooth field: flips %s",
                 ", ".join(f"{k} {v:.2e}" for k, v in flips.items()))
        cloud = {"keep": cls["keep"], "p_valid": cls["p_valid"], "table": cls["table"]}
    filled = filled_fields(inp, fin)
    log.info("[8/9] smooth and filled fields built")

    # 9. write
    fits = {"bootstrap": boot, "final": fin_fits}
    try:
        ds = build_output(raw, inp, loop_out, fits, fin, smooth, filled, cloud, cfg)
        if extra_attrs:
            ds.attrs.update(extra_attrs)
        write_cube(ds, cube_path, cfg)
    except Exception:
        # Everything above is the expensive part; never lose it to a bug in assembly.
        rescue = out_dir / f"{cfg['data']['aoi']}_pipeline{sfx}_rescue.npz"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            rescue, times=raw.times.astype("datetime64[ns]").astype("int64"),
            sst_filled=filled["merged"], sst_filled_point=filled["point"],
            sst_filled_day=filled["day"], sst_smooth=smooth["K"],
            **{f"{sid}_keep": cloud["keep"][sid] for sid in raw.raw},
            **{f"{sid}_p_valid": cloud["p_valid"][sid] for sid in raw.raw})
        log.exception("building or writing the cube failed; the fields were saved to %s",
                      rescue)
        raise
    rep_dir.mkdir(parents=True, exist_ok=True)
    boot["report"].to_csv(rep_dir / "offsets_bootstrap.csv", index=False)
    fin_fits["report"].to_csv(rep_dir / "offsets_final.csv", index=False)
    boot["matchups"].to_csv(rep_dir / "matchups_bootstrap.csv", index=False)
    fin_fits["matchups"].to_csv(rep_dir / "matchups_final.csv", index=False)
    if cfg["matchup"]["aggregate"] == "footprint":
        for name, f in (("bootstrap", boot), ("final", fin_fits)):
            if len(f["pairs"]):
                f["pairs"].to_csv(rep_dir / f"footprint_pairs_{name}.csv", index=False)
    loop_out["history"].to_csv(rep_dir / "iterations.csv", index=False)
    cloud["table"].to_csv(rep_dir / "scenes.csv", index=False)
    loop_out["scene_history"].to_csv(rep_dir / "scenes_history.csv", index=False)
    fin["res_c"]["curve"].to_csv(rep_dir / "cv_curve.csv", index=False)
    log.info("[9/9] reports in %s", rep_dir)

    if figures and cfg["output"]["write_figures"]:
        try:
            import pipeline_figures
            pipeline_figures.render(cube_path, fig_dir, rep_dir,
                                    dpi=int(cfg["output"]["figure_dpi"]))
        except Exception:
            log.exception("figures failed; the cube and reports were written and are intact")
    log.info("pipeline done in %.1f min -> %s", (time.time() - t_all) / 60, cube_path)
    return dict(cube=cube_path, reports=rep_dir, figures=fig_dir, ds=ds, loop=loop_out,
                fin=fin, fits=fits, raw=raw, inp=inp, smooth=smooth, filled=filled,
                cloud=cloud)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML config")
    p.add_argument("--tag", default=None, help="suffix the cube, reports and figures")
    p.add_argument("--max-iter", type=int, default=None, help="override loop.max_iter")
    p.add_argument("--no-figures", action="store_true", help="skip the figures")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(args.config)
    if args.max_iter is not None:
        cfg["_iter"]["loop"]["max_iter"] = int(args.max_iter)
    if not Path(cfg["data"]["source"]).exists():
        raise SystemExit(f"no source cube at {cfg['data']['source']}")
    run_pipeline(cfg, tag=args.tag, figures=not args.no_figures)


if __name__ == "__main__":
    main()
