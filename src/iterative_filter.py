"""Iterative DINEOF-baseline cloud filter.

The outlier detector in build_cube.py scores every ECOSTRESS / Landsat scene against a smoothed
MODIS composite. MODIS is sparse, coarse (1 km) and a night-time retrieval, so that baseline is
weakest exactly where cloud detection matters. compare_lowrank.py showed the DINEOF low-rank
field separates removed from kept pixels, but with two biases it names itself: the field was fit
on data the MODIS filter had already cleaned, and the field for day j is pulled toward day j's
own pixels. This process removes the first bias by bootstrapping from scratch:

  0. start from the RAW scenes with only the sensor's own QC applied;
  1. composite and standardize them (seasonal coefficients, scale and offsets held fixed);
  2. fit DINEOF according to `loop.strategy`;
  3. take the rank-k field U diag(sigma) V' for each day as the baseline, in kelvin;
  4. classify every raw scene against it, from scratch -- flags are NOT cumulative, so a pixel
     wrongly flagged early can come back;
  5. hard-mask the flagged pixels and repeat until the flags stop changing.

THE BASELINE IS THE LOW-RANK FIELD, NOT THE ANALYSIS. The analysis X holds the observations
themselves at every observed pixel, so a residual against it is identically zero there.

THREE SCHEDULES for the DINEOF hyperparameters (T_c, k), so cost and quality can be compared
directly from iterations.csv:

  fixed  -- every iteration fits at loop.k / loop.t_c, warm-started from the previous analysis;
            one full CV search at the end.
  ends   -- a full CV search at iteration 0, then fixed fits at the settings it chose; a full
            CV search again at the end.
  every  -- a full CV search every iteration.

Usage (from the repo root, in the `coastal_sst_data` env):

    python src/iterative_filter.py --config configs/config.iterative.admiralty_inlet.yaml
    ... --strategy ends --tag ends   # a variant, written beside the others
    ... --max-iter 2 --tag smoke      # smoke test
    ... --compare                     # score against the MODIS-baseline filter
    ... --no-figures
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "config.iterative.admiralty_inlet.yaml"


def _bridge_seasonal_smoothing() -> None:
    """Make `seasonal_smoothing` importable before edineof / composite are.

    Those modules expect it in a sibling `cloud_mixture_model/src`, which does not exist in this
    repo; it lives in the coastal_sst_data checkout. APPENDED, not inserted, so this repo's own
    copies of outlier_detection / simple_outlier_detection -- which that directory also holds,
    in an older form -- still win.
    """
    try:
        import seasonal_smoothing  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    for cand in (ROOT.parent / "cloud_mixture_model" / "src",
                 ROOT.parent / "coastal_sst_data" / "prototypes" / "cloud_mixture_model" / "src"):
        if (cand / "seasonal_smoothing.py").exists():
            sys.path.append(str(cand))
            return
    raise ModuleNotFoundError(
        "seasonal_smoothing is not importable and was not found in ../cloud_mixture_model/src "
        "or ../coastal_sst_data/prototypes/cloud_mixture_model/src")


_bridge_seasonal_smoothing()

from seasonal_smoothing import design_matrix               # noqa: E402

import build_cube as bc                                     # noqa: E402
import composite as C                                       # noqa: E402
import edineof as E                                         # noqa: E402
import simple_outlier_detection as sod                      # noqa: E402
from compare_lowrank import auc                             # noqa: E402
from outlier_detection import DEFAULTS as OD_DEFAULTS       # noqa: E402

from nearshore_sst import provenance, store, datacube       # noqa: E402
from nearshore_sst.datacube import CompressionSpec          # noqa: E402

log = logging.getLogger("iterative_filter")


# ==================================================================== config

DEFAULTS = {
    "data": {
        "aoi": "admiralty_inlet",
        # Raw scenes and their QC channels. Read-only; subset to the standardized cube's time axis.
        "source": "data/unfiltered/admiralty_inlet.zarr",
        # Seasonal coefficients, scale and validation_msk. Held FIXED for the whole run.
        "standardized": "data/datacube/admiralty_inlet_standardized.zarr",
        # Per-date `<id>_offset` channels; `offsets` is the fallback on dates they are NaN.
        "composite": "data/datacube/admiralty_inlet_composite_validation.zarr",
        "offsets": "data/datacube/offsets.csv",
        # The MODIS-baseline filter's cube: copied into the output, and what --compare scores.
        "filtered": "data/datacube/admiralty_inlet_filtered.zarr",
        "out": "data/datacube/admiralty_inlet_iterative.zarr",
        "watervar": "landcover_water",
        "coef": "sst_seasonal_coef",
        "scale": "sst_seasonal_sd",
        "validation_msk": "validation_msk",
        "modis_filtered_suffix": "_dineof",
    },
    "detector": {},     # OPAQUE: validated like build_cube's, minus `reference`/`covariate` use
    "sensors": {},      # OPAQUE: one level per sensor, build_cube's SENSOR_DEFAULTS
    "reference": {      # the composite anchor. A member, never a detection baseline.
        "id": "modis",
        "var": "modis_sst_aqua",
        "valid": "modis_valid_aqua",
        "hour": "modis_hour_aqua",
    },
    "filter": {
        "p_valid_min": 0.5,
        "offset_lower": -2.0,
        "offset_upper": 4.0,
        "min_kept_frac": None,
        "require_qc": True,
        "valid_range": [271.0, 310.0],
    },
    "composite": {
        "min_members": 1,
        "weights": {},              # member id -> weight; absent members weigh 1.0
        "hold": ["lst", "eco"],     # members removed on validation dates
        # Days composited at a time. The composite is built at native resolution every
        # iteration; whole-record it allocates ~a dozen (time, y, x) float64 arrays (~4 GB for
        # 365 days at 303 x 303). In blocks its peak scales with block_days instead. Lower it
        # on a small machine, raise it on a large one; the result does not depend on it.
        "block_days": 30,
    },
    "dineof": {},       # OPAQUE: edineof's matrix / filter / modes / em / cv sections
    "loop": {
        "strategy": "fixed",
        "k": 2,
        "t_c": 7.0,
        "baseline": "day",
        "max_iter": 8,
        "tol": 1.0e-3,
        "patience": 1,
        "final_cv": True,
        # Where the baseline's per-day loadings come from. `modis`: the spatial modes U are fit
        # on all members, but each day's amplitudes on MODIS pixels only, so a scene never
        # steers the field that judges it. `all`: the full-data low-rank field U diag(s) V'.
        "baseline_source": "modis",
        "loading_tc": None,         # pooling cutoff, days; null = the fit's own T_c
        "loading_ridge": 1.0e-3,    # Tikhonov lambda, relative to the median trace(A_j)/k
        "loading_min_px": 5,        # a day with fewer MODIS matrix cells contributes nothing
        "segment_years": None,      # None = no segmentation; 1.0 = annual, 0.5 = semi-annual
        "segment_overlap_days": 30, # overlap on each side of a window (mitigates edge effects)
    },
    "output": {
        "chunks": {"time": 64, "y": 128, "x": 128},
        "compression": {"codec": "zstd", "level": 5, "shuffle": "shuffle"},
        "filtered_suffix": "_iter",
        "report": "iterations.csv",
        "scenes": "iterative_scenes.csv",
        "compare": "iterative_compare.csv",
        "modis_cv": "data/datacube/dineof_cv_coarse.csv",
        "write_figures": True,
        "fig_dir": "figures/iterative",
        "figure_days": 6,
        "figure_dpi": 130,
    },
}

OPAQUE_SECTIONS = {"detector", "sensors", "dineof"}
DINEOF_SECTIONS = ("matrix", "filter", "modes", "em", "cv")
STRATEGIES = ("fixed", "ends", "every")
BASELINES = ("day", "point")
BASELINE_SOURCES = ("modis", "all")
MAX_ITER = 15
PATH_KEYS = (("data", "source"), ("data", "standardized"), ("data", "composite"),
             ("data", "offsets"), ("data", "filtered"), ("data", "out"),
             ("output", "fig_dir"), ("output", "modis_cv"))

# `classify` reads mixture.min_dev_cold / min_dev_hot, which simple_outlier_detection.DEFAULTS
# does not (yet) define, so build_cube.validate_detector would reject them as unknown keys.
# They are validated here instead and defaulted to 0.0 -- the unary before they existed, and
# therefore the one the MODIS-baseline filter this run is compared against was built with.
EXTRA_MIXTURE = tuple(k for k in OD_DEFAULTS["mixture"]
                      if k.startswith("min_dev") and k not in sod.DEFAULTS["mixture"])


def load_config(path: Path) -> dict:
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    return build_config(user, path)


def build_config(user: dict, path: Path | str = "<dict>", *, resolve: bool = True) -> dict:
    """Overlay `user` on DEFAULTS; unknown sections or keys are an error."""
    cfg = copy.deepcopy(DEFAULTS)
    for section, values in user.items():
        if section not in cfg:
            raise ValueError(f"{path}: unknown config section '{section}'")
        if section in OPAQUE_SECTIONS:
            cfg[section] = copy.deepcopy(values)
            continue
        for key, value in (values or {}).items():
            if key not in cfg[section]:
                raise ValueError(f"{path}: unknown key '{section}.{key}'")
            cfg[section][key] = value
    if resolve:
        for section, key in PATH_KEYS:
            p = Path(cfg[section][key])
            cfg[section][key] = (p if p.is_absolute() else ROOT / p).resolve()

    validate_detector(cfg, path)
    bc.validate_filter(cfg, path)
    bc.validate_sensors(cfg, path)
    validate_loop(cfg, path)
    vr = cfg["filter"]["valid_range"]
    if vr is not None and (len(vr) != 2 or float(vr[0]) >= float(vr[1])):
        raise ValueError(f"{path}: filter.valid_range must be [low, high] in K, or null")
    cfg["_edineof"] = edineof_config(cfg, path)
    return cfg


def validate_detector(cfg: dict, path) -> None:
    """build_cube's check, with the EXTRA_MIXTURE keys taken out first and checked here."""
    det = copy.deepcopy(cfg["detector"])
    mix = det.get("mixture") or {}
    for k in EXTRA_MIXTURE:
        if k in mix and float(mix.pop(k)) < 0:
            raise ValueError(f"{path}: detector.mixture.{k} must be >= 0")
    bc.validate_detector({"detector": det}, path)


def validate_loop(cfg: dict, path) -> None:
    lp = cfg["loop"]
    if lp["strategy"] not in STRATEGIES:
        raise ValueError(f"{path}: loop.strategy must be one of {STRATEGIES}")
    if lp["baseline"] not in BASELINES:
        raise ValueError(f"{path}: loop.baseline must be one of {BASELINES}")
    if not 1 <= int(lp["max_iter"]) <= MAX_ITER:
        raise ValueError(f"{path}: loop.max_iter must be in [1, {MAX_ITER}] -- the per-pixel "
                         "keep history is a uint16 bitmask, one bit per iteration plus the start")
    if int(lp["patience"]) < 1:
        raise ValueError(f"{path}: loop.patience must be >= 1")
    if not float(lp["tol"]) > 0:
        raise ValueError(f"{path}: loop.tol must be > 0")
    if int(lp["k"]) < 1:
        raise ValueError(f"{path}: loop.k must be >= 1")
    if float(lp["t_c"]) < 0:
        raise ValueError(f"{path}: loop.t_c must be >= 0")
    if lp["baseline_source"] not in BASELINE_SOURCES:
        raise ValueError(f"{path}: loop.baseline_source must be one of {BASELINE_SOURCES}")
    if lp["loading_tc"] is not None and float(lp["loading_tc"]) < 0:
        raise ValueError(f"{path}: loop.loading_tc must be >= 0 or null")
    if float(lp["loading_ridge"]) < 0:
        raise ValueError(f"{path}: loop.loading_ridge must be >= 0")
    if lp["segment_years"] is not None and float(lp["segment_years"]) <= 0:
        raise ValueError(f"{path}: loop.segment_years must be > 0 or null")
    if int(lp["segment_overlap_days"]) < 0:
        raise ValueError(f"{path}: loop.segment_overlap_days must be >= 0")
    if int(lp["loading_min_px"]) < 1:
        raise ValueError(f"{path}: loop.loading_min_px must be >= 1")
    ref = cfg["reference"]["id"]
    if ref in cfg["sensors"]:
        raise ValueError(f"{path}: reference.id {ref!r} is also a sensor; it is never classified")
    for mid in cfg["composite"]["hold"]:
        if mid not in cfg["sensors"]:
            raise ValueError(f"{path}: composite.hold names {mid!r}, which is not a sensor")
    if int(cfg["composite"]["block_days"]) < 1:
        raise ValueError(f"{path}: composite.block_days must be >= 1")


def edineof_config(cfg: dict, path) -> dict:
    """The dict edineof's functions expect, built over its DEFAULTS and run through its validate."""
    ecfg = copy.deepcopy(E.DEFAULTS)
    for section, values in (cfg["dineof"] or {}).items():
        if section not in DINEOF_SECTIONS:
            raise ValueError(f"{path}: unknown dineof section '{section}' (allowed: "
                             f"{', '.join(DINEOF_SECTIONS)})")
        for key, value in (values or {}).items():
            if key not in ecfg[section]:
                raise ValueError(f"{path}: unknown key 'dineof.{section}.{key}'")
            ecfg[section][key] = value
    d = cfg["data"]
    ecfg["data"] = {"aoi": d["aoi"], "cube": d["standardized"], "out": d["out"],
                    "watervar": d["watervar"], "channel": "sst_z", "coef": d["coef"],
                    "scale": d["scale"], "validation_msk": d["validation_msk"]}
    E.validate(ecfg, path)
    return ecfg


def detector_config(cfg: dict, sid: str) -> dict:
    """build_cube's per-sensor detector config, through a shim holding only what it reads."""
    shim = {"data": {"cube": cfg["data"]["source"], "watervar": cfg["data"]["watervar"]},
            "detector": cfg["detector"], "sensors": cfg["sensors"],
            "filter": {"offset_lower": cfg["filter"]["offset_lower"],
                       "offset_upper": cfg["filter"]["offset_upper"]},
            "output": {"cache_dir": Path("unused"), "write_detector_figures": False}}
    det_mix = cfg["detector"].get("mixture") or {}
    # build_cube.detector_cfg updates sod.DEFAULTS with the whole detector.mixture section, so
    # the EXTRA_MIXTURE keys are already there when configured; this fills them when not.
    dcfg = bc.detector_cfg(shim, sid)
    for k in EXTRA_MIXTURE:
        dcfg["mixture"][k] = float(det_mix.get(k, 0.0))
    return dcfg


def verdict_config(cfg: dict) -> dict:
    """What build_cube.scene_verdict reads. There is always a baseline, so no reference gate."""
    f = cfg["filter"]
    return {"filter": {"offset_lower": f["offset_lower"], "offset_upper": f["offset_upper"],
                       "require_reference": False}}


def composite_config(cfg: dict, order: list[str]) -> dict:
    """What composite.composite_with_validation reads."""
    w = cfg["composite"]["weights"] or {}
    return {"members": {mid: {"weight": float(w.get(mid, 1.0))} for mid in order},
            "composite": {"min_members": int(cfg["composite"]["min_members"])},
            "validation": {"hold": list(cfg["composite"]["hold"])}}


def composite_blocked(adj: dict, valid_inds: np.ndarray, cfg: dict, order: list[str],
                      block_days: int, *, keep: dict | None = None,
                      offsets: dict | None = None, seasonal: np.ndarray | None = None,
                      scale: np.ndarray | None = None,
                      water: np.ndarray | None = None) -> dict:
    """composite.composite_with_validation's `sst`, `src` and `msk`, built `block_days` at a time
    -- and, optionally, the cloud masking, offset correction and standardization fused in.

    Same arithmetic as the original: a weighted mean over the members that saw each pixel
    (NaN below `composite.min_members`), the bitmask of those members, and the validation mask
    -- pixels covered ONLY by `hold` members on a validation date. Each block is independent
    in time, so blocking changes nothing but the peak memory, which scales with `block_days`
    instead of the record length. The spread and count channels the original also builds are
    not needed here and are skipped.

    Fused per block, so no full-record intermediate is ever built:
      keep      member id -> (T, y, x) bool; pixels outside it are treated as unobserved
                (replaces a full masked copy of every sensor's cube)
      offsets   member id -> ((T,) offset, slope); the member is (adj - offset) / slope
                (replaces a full offset-corrected copy)
      seasonal, scale, water
                also return z = (sst - seasonal) / scale, NaN off water, and restrict `msk`
                to finite z. Computed in float32: `scale` is cast once, so the division does
                not silently promote a whole cube to float64.
    """
    ccfg = composite_config(cfg, order)
    w = {mid: ccfg["members"][mid]["weight"] for mid in order}
    min_members = ccfg["composite"]["min_members"]
    hold = set(ccfg["validation"]["hold"])
    keep = keep or {}
    offsets = offsets or {}
    T = adj[order[0]].shape[0]
    shape = adj[order[0]].shape
    is_valid_day = np.zeros(T, bool)
    is_valid_day[np.asarray(valid_inds, int)] = True
    want_z = seasonal is not None
    if want_z:
        if scale is None or water is None:
            raise ValueError("z needs seasonal, scale and water together")
        scale32 = np.asarray(scale, "float32")[None]
        land = ~np.asarray(water, bool)

    sst = np.full(shape, np.nan, "float32")
    src = np.zeros(shape, "uint8")
    msk = np.zeros(shape, bool)
    z = np.full(shape, np.nan, "float32") if want_z else None
    step = max(1, int(block_days))
    for t0 in range(0, T, step):
        sl = slice(t0, min(t0 + step, T))
        bshape = (sl.stop - sl.start,) + shape[1:]
        vday = is_valid_day[sl][:, None, None]
        num = np.zeros(bshape, "float64")
        den = np.zeros(bshape, "float64")
        cnt = np.zeros(bshape, "uint8")
        bits = np.zeros(bshape, "uint8")
        cnt_msk = np.zeros(bshape, "uint8")
        for bit, mid in enumerate(order):
            a = adj[mid][sl]
            if mid in offsets:
                off, slope = offsets[mid]
                a = ((a - np.asarray(off)[sl][:, None, None]) / slope).astype("float32")
            if mid in keep:
                a = np.where(keep[mid][sl], a, np.nan)
            ok = np.isfinite(a)
            num += np.where(ok, np.nan_to_num(a) * w[mid], 0.0)
            den += np.where(ok, w[mid], 0.0)
            cnt += ok.astype("uint8")
            bits |= ok.astype("uint8") << bit
            cnt_msk += (ok & ~vday).astype("uint8") if mid in hold else ok.astype("uint8")
        enough = cnt >= min_members
        with np.errstate(invalid="ignore", divide="ignore"):
            comp = np.where(enough, num / den, np.nan).astype("float32")
        sst[sl] = comp
        src[sl] = np.where(enough, bits, 0)
        # The original's held-out composite requires only ONE member (hard-coded), so a pixel is
        # in the mask when no non-hold member saw it, on a held-out date, and it is in `sst`.
        m = (cnt_msk < 1) & np.isfinite(comp)
        if want_z:
            with np.errstate(invalid="ignore", divide="ignore"):
                zb = (comp - seasonal[sl]) / scale32
            zb[:, land] = np.nan
            z[sl] = zb
            # Flagged pixels leave the point-CV set along with the fit.
            m &= np.isfinite(zb)
        msk[sl] = m
    out = dict(sst=sst, src=src, msk=msk)
    if want_z:
        out["z"] = z
    return out


# ==================================================================== inputs

@dataclass
class Inputs:
    """Everything held fixed for the run. Built once by `load_inputs`, or directly by tests."""
    times: np.ndarray                       # (T,) datetime64
    water: np.ndarray                       # (y, x) bool
    tidal: np.ndarray                       # (y, x) bool
    coef: np.ndarray                        # (P, y, x) seasonal coefficients, native
    scale: np.ndarray                       # (y, x) float32 native -- float32 so that scaling a
                                            # (T, y, x) float32 cube does not promote it to float64
    seasonal: np.ndarray                    # (T, y, x) float32, K
    raw: dict                               # sid -> (T, y, x) float32 K, sensor scale, water only
    adj: dict                               # member id -> (T, y, x) float32 K, composite scale
    scenes: dict                            # sid -> sorted list of acquisition time indices
    qc: dict                                # sid -> {j: (flagged, gap, nodata)}
    dcfg: dict                              # sid -> detector config
    valid_inds: np.ndarray                  # validation dates (time indices)
    order: list                             # composite member order, reference first
    coords: dict = field(default_factory=dict)
    offsets: dict = field(default_factory=dict)   # sid -> (T,) offset removed

    @property
    def n_water(self) -> int:
        return int(self.water.sum())


def read_offsets(cfg: dict, times: np.ndarray, sids: list[str]) -> tuple[dict, dict]:
    """(sid -> (T,) offset in K, sid -> slope divisor) on the standardized cube's time axis.

    The composite cube's `<id>_offset` channel is NaN on every date the MODIS filter left the
    member empty -- and this run may well keep some of those scenes -- so NaN falls back to
    `offset_mean` from offsets.csv. That is exact for K = 0, which is every member here; with a
    diurnal term it is the member's mean offset, and a warning says so.
    """
    table = pd.read_csv(cfg["data"]["offsets"]).set_index("member")
    with xr.open_zarr(cfg["data"]["composite"]) as cc:
        spec = json.loads(cc.attrs.get("dineof_composite", "{}"))
        ctimes = cc["time"].values
        chan = {sid: cc[f"{sid}_offset"].values for sid in sids if f"{sid}_offset" in cc}
    slope_on = bool(spec.get("offset", {}).get("slope", False))

    idx = pd.Index(ctimes)
    off, slope = {}, {}
    for sid in sids:
        if sid not in table.index:
            raise ValueError(f"{cfg['data']['offsets']} has no row for member {sid!r}")
        row = table.loc[sid]
        mean = float(row["offset_mean"])
        o = np.full(len(times), mean, dtype="float64")
        if sid in chan:
            pos = idx.get_indexer(times)
            have = pos >= 0
            v = np.full(len(times), np.nan)
            v[have] = chan[sid][pos[have]]
            o = np.where(np.isfinite(v), v, mean)
        n_fill = int((~np.isfinite(chan.get(sid, np.full(1, np.nan)))).sum())
        if int(row.get("K", 0)) > 0 and n_fill:
            log.warning("%s: diurnal offset (K=%d) but %d dates have no fitted value; those use "
                        "offset_mean %.4f", sid, int(row["K"]), n_fill, mean)
        off[sid] = o
        slope[sid] = float(row["slope_ols"]) if slope_on else 1.0
    log.info("offsets: %s%s", ", ".join(f"{s} {np.nanmean(o):+.4f}" for s, o in off.items()),
             "  (slope on)" if slope_on else "")
    return off, slope


@dataclass
class Raw:
    """Everything read from the SOURCE cube: the sensors as recorded, their QC, and MODIS.

    Independent of offsets and standardization, so the end-to-end pipeline can fit those from
    it and `make_inputs` can be called again after a refit without re-reading the cube.
    """
    times: np.ndarray                       # (T,) datetime64
    water: np.ndarray                       # (y, x) bool
    tidal: np.ndarray                       # (y, x) bool
    raw: dict                               # sid -> (T, y, x) float32 K, sensor scale, water only
    ref: np.ndarray                         # (T, y, x) float32 K, the reference member
    hours: dict                             # member id -> (T,) overpass hour UTC
    scenes: dict                            # sid -> sorted list of acquisition time indices
    qc: dict                                # sid -> {j: (flagged, gap, nodata)}
    dcfg: dict                              # sid -> detector config
    order: list                             # composite member order, reference first
    coords: dict = field(default_factory=dict)

    @property
    def n_water(self) -> int:
        return int(self.water.sum())


def load_raw(cfg: dict, times: np.ndarray | None = None) -> Raw:
    """Read the source cube on `times` (default: its whole axis) into a `Raw`."""
    d = cfg["data"]
    src_all = xr.open_zarr(d["source"])
    if times is None:
        times = src_all["time"].values
    missing = ~np.isin(times, src_all["time"].values)
    if missing.any():
        raise ValueError(f"{d['source']}: {int(missing.sum())} requested dates are not on its "
                         "time axis")
    src = src_all.sel(time=times)
    water = np.asarray(src[d["watervar"]].compute() > 0.5)
    sids = list(cfg["sensors"])
    ref = cfg["reference"]
    order = [ref["id"]] + sids
    dv = cfg["detector"].get("depthvar")
    if dv and dv in src:
        tidal = np.asarray(src[dv].compute(), float) < float(cfg["detector"]["tidal_depth_m"])
    else:
        tidal = np.zeros(water.shape, bool)   # no intertidal prior

    members = {ref["id"]: {"var": ref["var"], "valid": ref["valid"]}}
    ref_stack = C.member_stack(src, {"members": members}, ref["id"], water)
    hours = {}
    if ref.get("hour") and ref["hour"] in src:
        hours[ref["id"]] = np.asarray(src[ref["hour"]].values, float)
    raw, scenes, qc, dcfgs = {}, {}, {}, {}
    for sid in sids:
        s = cfg["sensors"][sid]
        t0 = time.time()
        r = src[s["sst"]].values.astype("float32")
        r = np.where(water[None], r, np.nan).astype("float32")
        raw[sid] = r
        hours[sid] = np.asarray(src[s["hour"]].values, float)
        n = np.isfinite(r).sum(axis=(1, 2))
        scenes[sid] = [int(j) for j in np.flatnonzero(n >= int(s["min_pixels"]))]

        dcfgs[sid] = dcfg = detector_config(cfg, sid)
        qcds = src[[s["valid"], s["cloud"]]].isel(time=scenes[sid]).load()
        vr = cfg["filter"]["valid_range"]
        qc[sid] = {}
        n_range = 0
        for j in scenes[sid]:
            flagged, gap, nodata = sod.qc_masks(qcds, times[j], water, r[j], dcfg)
            if vr is not None:
                # Gross range check, folded into the QC flag. The sensor QC passes values from
                # 202 K (Landsat) to 505 K (ECOSTRESS, a whole scene on 2025-08-16); left in at
                # iteration 0 they dominate the rank-k fit the loop bootstraps from.
                with np.errstate(invalid="ignore"):
                    out = np.isfinite(r[j]) & ((r[j] < float(vr[0])) | (r[j] > float(vr[1])))
                n_range += int((out & ~flagged).sum())
                flagged = flagged | out
            qc[sid][j] = (flagged, gap, nodata)
        if vr is not None:
            log.info("%s: %d observed px outside valid_range %s K added to the QC flag", sid,
                     n_range, list(vr))
        log.info("%s: %d acquisitions of %s (>= %d water px), QC read in %.1fs", sid,
                 len(scenes[sid]), s["sst"], int(s["min_pixels"]), time.time() - t0)

    coords = {c: src[c].values for c in ("y", "x") if c in src.coords}
    return Raw(times=times, water=water, tidal=tidal, raw=raw, ref=ref_stack, hours=hours,
               scenes=scenes, qc=qc, dcfg=dcfgs, order=order, coords=coords)


def seasonal_field(times: np.ndarray, coef: np.ndarray, n_harmonics: int,
                   period_days: float) -> np.ndarray:
    """(T, y, x) float32 seasonal climatology in K from (P, y, x) coefficients; NaN where coef is."""
    X = design_matrix(pd.to_datetime(times), int(n_harmonics), float(period_days))
    return np.tensordot(X, coef, axes=1).astype("float32")


def make_inputs(raw: Raw, offsets: dict, slope: dict, coef: np.ndarray, scale: np.ndarray,
                valid_inds: np.ndarray, n_harmonics: int, period_days: float) -> Inputs:
    """`Raw` + offsets + standardization -> the `Inputs` the loop runs on."""
    adj = {raw.order[0]: raw.ref}
    for sid, r in raw.raw.items():
        adj[sid] = ((r - np.asarray(offsets[sid])[:, None, None]) / slope[sid]).astype("float32")
    return Inputs(times=raw.times, water=raw.water, tidal=raw.tidal,
                  coef=np.asarray(coef, "float64"), scale=np.asarray(scale, "float32"),
                  seasonal=seasonal_field(raw.times, coef, n_harmonics, period_days),
                  raw=raw.raw, adj=adj, scenes=raw.scenes, qc=raw.qc, dcfg=raw.dcfg,
                  valid_inds=np.asarray(valid_inds, int), order=raw.order, coords=raw.coords,
                  offsets={k: np.asarray(v) for k, v in offsets.items()})


def load_inputs(cfg: dict) -> tuple[Inputs, xr.Dataset]:
    """Read everything held fixed from the earlier stages' products. Returns (inputs, the
    standardized cube, still open)."""
    d = cfg["data"]
    st = xr.open_zarr(d["standardized"])
    times = st["time"].values
    coef = st[d["coef"]].values.astype("float64")
    scale = st[d["scale"]].values.astype("float64")
    a = st[d["coef"]].attrs
    vmsk = np.asarray(st[d["validation_msk"]].values > 0)
    valid_inds = np.flatnonzero(vmsk.any(axis=(1, 2)))
    log.info("standardized: %d dates %s .. %s, %d validation dates", len(times),
             str(times[0])[:10], str(times[-1])[:10], valid_inds.size)
    raw = load_raw(cfg, times)
    offsets, slope = read_offsets(cfg, times, list(cfg["sensors"]))
    inp = make_inputs(raw, offsets, slope, coef, scale, valid_inds, int(a["n_harmonics"]),
                      float(a["period_days"]))
    return inp, st


def qc_only_keep(inp: Inputs, require_qc: bool = True) -> dict:
    """Iteration 0: every observed pixel of every acquisition, minus what the sensor QC flags."""
    keep = {}
    for sid, r in inp.raw.items():
        k = np.zeros(r.shape, bool)
        for j in inp.scenes[sid]:
            k[j] = np.isfinite(r[j]) & inp.water
            if require_qc:
                k[j] &= ~inp.qc[sid][j][0]
        keep[sid] = k
    return keep


# ==================================================================== matrix and fit

def build_matrix(inp: Inputs, keep: dict, cfg: dict) -> tuple[dict, dict]:
    """Current keep-masks -> composite -> z -> edineof's matrix. (sel, composite stats)."""
    # Masking, compositing and standardizing in one blocked pass: no masked sensor copies and no
    # full-record float64 temporary.
    comp = composite_blocked(inp.adj, inp.valid_inds, cfg, inp.order,
                             int(cfg["composite"]["block_days"]),
                             keep={sid: keep[sid] for sid in inp.order[1:]},
                             seasonal=inp.seasonal, scale=inp.scale, water=inp.water)
    z, msk = comp["z"], comp["msk"]

    ec = cfg["_edineof"]
    d = ec["data"]
    dims = ("time", "y", "x")
    ds_iter = xr.Dataset(
        {d["channel"]: (dims, z),
         d["validation_msk"]: (dims, msk),
         d["coef"]: (("term", "y", "x"), inp.coef),
         d["scale"]: (("y", "x"), inp.scale),
         d["watervar"]: (("y", "x"), inp.water.astype("uint8"))},
        coords={"time": inp.times, **inp.coords})
    sel = E.select_matrix(ds_iter, ec)
    stats = {"n_composite": int(np.isfinite(comp["sst"]).sum()), "n_validation": int(msk.sum()),
             "sst": comp["sst"], "src": comp["src"]}
    return sel, stats


def to_grid(Z: np.ndarray, sel: dict) -> np.ndarray:
    """(m, n) matrix -> (T, y, x) on the matrix's (possibly coarsened) grid, NaN elsewhere."""
    water, keep = sel["water"], sel["keep"]
    full = np.full((sel["n"], int(water.sum())), np.nan, dtype="float32")
    full[:, keep] = Z.T
    out = np.full((sel["n"],) + water.shape, np.nan, dtype="float32")
    out[:, water] = full
    return out


def from_grid(G: np.ndarray, sel: dict) -> np.ndarray:
    """The inverse of `to_grid` for the current pixel set; NaN where the grid has no value."""
    return np.ascontiguousarray(G[:, sel["water"]][:, sel["keep"]].T.astype(np.float64))


def setting(t_c: float, ecfg: dict) -> dict:
    """The (alpha, p) edineof would use for a cutoff period of t_c days."""
    return E.filter_settings([float(t_c)], float(ecfg["filter"]["alpha_max"]))[0]


def fit_fixed(X_raw: np.ndarray, observed: np.ndarray, t: np.ndarray, k: int, s: dict,
              ecfg: dict, seed: np.ndarray | None = None, label: str = "fixed",
              smoother=None) -> dict:
    """One eDINEOF fit at fixed (k, T_c): edineof's `final_fit`, without the search.

    Centre, fill, top_k_modes, reconstruct -- exactly as `final_fit` does, so with no seed and
    no CV holdout the two agree. `seed` is a previous analysis in (uncentred) z on this matrix;
    the gaps start from it rather than from the mean, which is the whole saving of a warm start:
    the EM iteration count is dominated by temporal diffusion into the gaps.
    `smoother` (or None) is the optional spatial smoother applied to the data each EM iteration.
    """
    f, mo, em = ecfg["filter"], ecfg["modes"], ecfg["em"]
    E.check_stability(t, s["alpha"], s["p"], float(f["stability_factor"]))
    mu = float(np.mean(X_raw[observed]))
    X0 = np.where(observed, X_raw - mu, 0.0)
    sd = float(np.std(X0[observed]))
    gaps = ~observed
    X = X0.copy()
    if seed is not None:
        sd_seed = np.nan_to_num(seed - mu)
        X[gaps] = sd_seed[gaps]
        if s["p"] == 0 or s["alpha"] == 0:
            # edineof's `apply_seed` guard: without the filter an all-gap column is structurally
            # zero, and a seed would sustain whatever it put there.
            empty = gaps.all(axis=0)
            if empty.any():
                X[:, empty] = 0.0
    X, hist = E.fill(X, gaps, int(k), t, s["alpha"], s["p"], float(em["tol"]),
                     int(em["max_iter"]), sd, label=label, smoother=smoother)
    U, sigma, V = E.top_k_modes(X, int(k), t, s["alpha"], s["p"], use_filter=True,
                                convention=mo["sigma_convention"], fix_sign=bool(mo["fix_sign"]),
                                smoother=smoother)
    return dict(X=X + mu, lowrank=E.reconstruct(U, sigma, V) + mu, U=U, sigma=sigma, V=V, mu=mu,
                hist=hist, k=int(k), t_c=s["t_c"], alpha=s["alpha"], p=s["p"])


def cv_scores(res: dict) -> tuple[float, float]:
    """(point RMSE at the point-opt setting, day RMSE at the day-opt setting) from edineof()."""
    c = res["curve"]
    pt = c[(c["t_c"] == res["tc_opt"]) & (c["k"] == res["k_opt"])]["rmse_point"]
    dy = c[(c["t_c"] == res["tc_day"]) & (c["k"] == res["k_day"])]["rmse_day"]
    return (float(pt.iloc[0]) if len(pt) else np.nan, float(dy.iloc[0]) if len(dy) else np.nan)


def modis_matrix(inp: Inputs, sel: dict) -> tuple[np.ndarray, np.ndarray]:
    """(Xm, Om): the reference member ALONE in z, on the same (m, n) matrix as `sel`.

    Standardized with the same fixed seasonal coefficients and scale as the composite, and
    coarsened with edineof's own block mean, so it sits on exactly the rows the modes U live on.
    Nothing any sensor did enters it.
    """
    ref = inp.order[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        z = ((inp.adj[ref] - inp.seasonal) / inp.scale[None]).astype("float64")
    ok = np.isfinite(z) & inp.water[None]
    f = int(sel["coarsen"])
    if f > 1:
        z, _ = E.block_mean(z, ok, f)
    else:
        z = np.where(ok, z, np.nan)
    Zm = z[:, sel["water"]][:, sel["keep"]].T
    Om = np.isfinite(Zm)
    return np.where(Om, Zm, 0.0), Om


def modis_loadings(U: np.ndarray, mu: float, Xm: np.ndarray, Om: np.ndarray, t: np.ndarray,
                   s: dict, ridge: float, min_px: int) -> tuple[np.ndarray, dict]:
    """Per-day amplitudes of the modes U, fit to MODIS pixels only. Returns (a (k, n), info).

    Each day's least-squares normal equations over its MODIS rows -- A_j = U_M' U_M and
    b_j = U_M' (x_M - mu) -- are POOLED IN TIME with eDINEOF's own diffusion filter before
    solving, because MODIS sees a median 7.5% of the water on a third of the days. Pooling the
    equations rather than the solutions weights each neighbouring day by how much MODIS it
    actually has, and a day with no MODIS of its own is fit entirely from its neighbours --
    the same mechanism, and the same T_c, that populates empty days in the EOF fit itself.

    A day nothing reaches gets a = 0: the seasonal climatology, counted as a fallback.
    """
    k, n = U.shape[1], Xm.shape[1]
    Omf = Om.astype("float64")
    npx = Om.sum(axis=0)
    use = npx >= int(min_px)
    A = np.einsum("ik,il,ij->klj", U, U, Omf)                       # (k, k, n)
    b = np.einsum("ik,ij->kj", U, Omf * (Xm - mu))                  # (k, n)
    A[..., ~use] = 0.0
    b[:, ~use] = 0.0
    if s["alpha"] and s["p"]:
        A = E.filter_time(t, A, s["alpha"], s["p"])
        b = E.filter_time(t, b, s["alpha"], s["p"])
    A = (A + np.swapaxes(A, 0, 1)) / 2.0

    tr = np.einsum("kkj->j", A)
    ref = float(np.median(tr[tr > 0]) / k) if (tr > 0).any() else 0.0
    reached = tr > 1e-9 * max(ref, 1e-300)
    lam = float(ridge) * ref
    a = np.zeros((k, n))
    for j in np.flatnonzero(reached):
        a[:, j] = np.linalg.solve(A[:, :, j] + lam * np.eye(k), b[:, j])
    # Per day: 0 = its own MODIS, 1 = only neighbours' MODIS, 2 = none (climatology).
    status = np.where(use, 0, np.where(reached, 1, 2)).astype("int8")
    info = dict(modis_px=npx.astype("int32"), status=status, direct=int(use.sum()),
                pooled=int((reached & ~use).sum()), fallback=int((~reached).sum()),
                t_c=float(s["t_c"]))
    return a, info


def baseline_kelvin(lowrank: np.ndarray, sel: dict, inp: Inputs) -> np.ndarray:
    """The low-rank field in K on the native grid, (T, y, x).

    Upsampled in z and THEN unstandardized with the native coefficients and scale: the coarse
    scale is a block mean and damps the field (see edineof.coarsen_inputs). Anywhere the matrix
    had no pixel, z = 0 -- the seasonal climatology.
    """
    G = to_grid(lowrank, sel)
    f = int(sel["coarsen"])
    if f > 1:
        G = E.upsample(G, f, inp.water.shape)
    G = np.where(np.isfinite(G), G, 0.0)
    K = (G * inp.scale[None] + inp.seasonal).astype("float32")
    K[:, ~inp.water] = np.nan
    bad = inp.water[None] & ~np.isfinite(K)
    if bad.any():
        # No seasonal coefficient at all: fall back to that day's median baseline.
        med = np.nanmedian(np.where(inp.water[None], K, np.nan), axis=(1, 2))
        K = np.where(bad, med[:, None, None], K).astype("float32")
    return K


# ==================================================================== classification

def classify_all(inp: Inputs, base: np.ndarray, cfg: dict, fig_scenes=()) -> dict:
    """Every acquisition of every sensor against `base`, from scratch.

    `classify` is handed the RAW scene (sensor scale) and the baseline (composite scale), so its
    `center` absorbs the sensor offset and the offset band keeps its build_cube meaning.
    """
    f = cfg["filter"]
    p_min = float(f["p_valid_min"])
    min_kept = f["min_kept_frac"]
    vcfg = verdict_config(cfg)
    keep, p_valid, rows, figs = {}, {}, [], {}
    for sid, r in inp.raw.items():
        k = np.zeros(r.shape, bool)
        pv = np.full(r.shape, np.nan, dtype="float32")
        dcfg = inp.dcfg[sid]
        for j in inp.scenes[sid]:
            y = r[j].astype("float64")
            flagged, gap, nodata = inp.qc[sid][j]
            res = sod.classify(y, inp.water, inp.tidal, base[j].astype("float64"),
                               flagged, gap, nodata, dcfg)
            ok, reason = bc.scene_verdict({"center": res["center"]}, vcfg)
            px = (res["p_valid"] >= p_min) & inp.water & np.isfinite(y)
            if f["require_qc"]:
                px &= ~flagged
            frac = float(px.sum()) / inp.n_water
            if ok and min_kept is not None and frac < float(min_kept):
                ok, reason = False, f"only {frac:.1%} of water survives the pixel cut"
            if ok:
                k[j] = px
            pv[j] = np.where(inp.water, res["p_valid"], np.nan)
            rows.append(dict(sensor=sid, date=str(inp.times[j])[:10], t=j,
                             center=float(res["center"]), sd=float(res["sd"]),
                             cloud_frac=float(res["cloud_frac"]),
                             warm_frac=float(res["warm_frac"]),
                             n_obs=int((np.isfinite(y) & inp.water).sum()),
                             n_qc=int(flagged.sum()), n_pass=int(px.sum()),
                             n_kept=int(px.sum()) if ok else 0,
                             verdict="KEPT" if ok else "DROPPED", reason=reason))
            if (sid, j) in fig_scenes:
                figs[(sid, j)] = dict(residual=res["residual"], q_cloud=res["q_cloud"],
                                      p_valid=res["p_valid"], center=res["center"])
        keep[sid], p_valid[sid] = k, pv
    return dict(keep=keep, p_valid=p_valid, table=pd.DataFrame(rows), figs=figs)


def flip_fraction(inp: Inputs, old: dict, new: dict) -> dict:
    """Per sensor: the share of observed acquisition pixels whose verdict changed."""
    out = {}
    for sid, r in inp.raw.items():
        js = inp.scenes[sid]
        if not js:
            out[sid] = 0.0
            continue
        obs = np.isfinite(r[js]) & inp.water[None]
        out[sid] = float((old[sid][js] != new[sid][js])[obs].sum() / max(int(obs.sum()), 1))
    return out


# ==================================================================== the loop

def run_loop(inp: Inputs, cfg: dict, *, init_keep: dict | None = None,
             fig_scenes=()) -> dict:
    """The iterative filter. Returns the final keep-masks, the history and the last fit."""
    lp, ecfg = cfg["loop"], cfg["_edineof"]
    strategy, which = lp["strategy"], lp["baseline"]
    keep = init_keep if init_keep is not None else qc_only_keep(inp, cfg["filter"]["require_qc"])
    fixed = setting(lp["t_c"], ecfg)
    chosen_k = int(lp["k"])
    prev_grid, rows, totals = None, [], []
    # Bit 0 is the starting mask, bit i+1 the verdict after iteration i. One uint16 per pixel
    # holds the whole trajectory: which pixels were flagged when, and which came back.
    bits = {sid: keep[sid].astype("uint16") for sid in inp.raw}
    scene_hist = []
    stable, converged = 0, False
    last = None
    cls = None
    source = lp["baseline_source"]
    warned_tc0 = False

    for it in range(int(lp["max_iter"])):
        t0 = time.time()
        sel, cstats = build_matrix(inp, keep, cfg)
        geom = (sel["water"], sel["keep"])          # for the optional spatial EOF smoother
        smoother = E.make_spatial_smoother(sel["water"], sel["keep"],
                                           float(ecfg["filter"].get("l_c", 0.0)),
                                           float(ecfg["filter"].get("spatial_alpha_max", 0.25)))
        rmse_point = rmse_day = np.nan
        if strategy == "every" or (strategy == "ends" and it == 0):
            res = E.edineof(sel["X"], sel["observed"], sel["valid_msk"], sel["t"], ecfg, geom=geom)
            fit = dict(res["day_fit"] if which == "day" else res["point_fit"], mu=res["mu"])
            rmse_point, rmse_day = cv_scores(res)
            if strategy == "ends":
                chosen_k = fit["k"]
                fixed = {"t_c": fit["t_c"], "alpha": fit["alpha"], "p": fit["p"]}
                log.info("ends: fixing k=%d, T_c=%g for the loop (%s-opt)", chosen_k,
                         fixed["t_c"], which)
        else:
            seed = from_grid(prev_grid, sel) if prev_grid is not None else None
            fit = fit_fixed(sel["X"], sel["observed"], sel["t"], chosen_k, fixed, ecfg,
                            seed=seed, label=f"iter {it}", smoother=smoother)
        # Seed the next fit from the LOW-RANK field, not the analysis: at a fixed point the two
        # agree on the gaps, but at observed pixels the analysis holds the observation -- and a
        # pixel this iteration flags as cloud would then start the next fit at its cloudy value.
        prev_grid = to_grid(fit["lowrank"], sel)
        fit_s = time.time() - t0

        t1 = time.time()
        linfo = {}
        if source == "modis":
            s_load = (setting(lp["loading_tc"], ecfg) if lp["loading_tc"] is not None
                      else {"t_c": fit["t_c"], "alpha": fit["alpha"], "p": fit["p"]})
            if s_load["t_c"] == 0 and not warned_tc0:
                warned_tc0 = True
                log.warning("loading T_c is 0: MODIS loadings are same-day only, and every day "
                            "without MODIS is filtered against climatology. Set "
                            "loop.loading_tc to pool across days.")
            Xm, Om = modis_matrix(inp, sel)
            a, linfo = modis_loadings(fit["U"], fit["mu"], Xm, Om, sel["t"], s_load,
                                      float(lp["loading_ridge"]), int(lp["loading_min_px"]))
            base = baseline_kelvin(fit["U"] @ a + fit["mu"], sel, inp)
            log.info("  MODIS loadings (T_c=%g): %d days direct, %d pooled from neighbours, "
                     "%d fallback to climatology", linfo["t_c"], linfo["direct"],
                     linfo["pooled"], linfo["fallback"])
        else:
            a = None
            base = baseline_kelvin(fit["lowrank"], sel, inp)
        cls = classify_all(inp, base, cfg, fig_scenes)
        cls_s = time.time() - t1

        flips = flip_fraction(inp, keep, cls["keep"])
        tbl = cls["table"]
        scene_hist.append(tbl.assign(iter=it))
        for sid in inp.raw:
            bits[sid] |= cls["keep"][sid].astype("uint16") << (it + 1)
        row = dict(iter=it, strategy=strategy, k=int(fit["k"]), t_c=float(fit["t_c"]),
                   em_iter=int(fit["hist"]["n_iter"]), fit_seconds=fit_s,
                   classify_seconds=cls_s, baseline_source=source,
                   loading_t_c=linfo.get("t_c", np.nan),
                   modis_days_direct=linfo.get("direct", np.nan),
                   modis_days_pooled=linfo.get("pooled", np.nan),
                   modis_days_fallback=linfo.get("fallback", np.nan))
        for sid in inp.raw:
            row[f"n_kept_{sid}"] = int(cls["keep"][sid].sum())
        for sid in inp.raw:
            row[f"flip_frac_{sid}"] = flips[sid]
        row.update(cloud_frac_mean=float(tbl["cloud_frac"].mean()) if len(tbl) else np.nan,
                   n_scenes_dropped=int(tbl["verdict"].eq("DROPPED").sum()) if len(tbl) else 0,
                   n_validation=cstats["n_validation"],
                   rmse_point=rmse_point, rmse_day=rmse_day)
        rows.append(row)
        log.info("iter %d [%s k=%d T_c=%g]: fit %.1fs (%d EM), classify %.1fs; flips %s; "
                 "kept %s", it, strategy, row["k"], row["t_c"], fit_s, row["em_iter"], cls_s,
                 ", ".join(f"{s} {v:.2e}" for s, v in flips.items()),
                 ", ".join(f"{s} {row[f'n_kept_{s}']:,}" for s in inp.raw))

        n_flip = sum(int((keep[s] != cls["keep"][s]).sum()) for s in inp.raw)
        totals.append(n_flip)
        keep = cls["keep"]
        last = dict(fit=fit, sel=sel, base=base, res=res if strategy == "every" else None,
                    loadings=a, full_loadings=fit["sigma"][:, None] * fit["V"].T,
                    modis_px=linfo.get("modis_px"), status=linfo.get("status"),
                    source=source, composite=cstats["sst"], composite_src=cstats["src"])

        if max(flips.values(), default=0.0) < float(lp["tol"]):
            stable += 1
            if stable >= int(lp["patience"]):
                converged = True
                log.info("converged after %d iterations (flip fraction < %g for %d)",
                         it + 1, float(lp["tol"]), int(lp["patience"]))
                break
        else:
            stable = 0
            if len(totals) >= 3 and totals[-1] >= totals[-2]:
                log.warning("iter %d: %d flips, not fewer than the previous iteration's %d -- "
                            "the flags may be oscillating rather than converging",
                            it, totals[-1], totals[-2])
    if not converged:
        log.warning("did not converge in %d iterations (last flip fractions: %s)",
                    int(lp["max_iter"]), ", ".join(f"{k} {v:.2e}" for k, v in flips.items()))

    final_res = None
    if strategy == "every":
        final_res = last["res"]
    elif lp["final_cv"]:
        # The settings the whole loop assumed, re-chosen on the final mask. This is also what
        # makes the CV RMSE comparable with the MODIS-baseline filter's own run.
        t0 = time.time()
        sel, _ = build_matrix(inp, keep, cfg)
        final_res = E.edineof(sel["X"], sel["observed"], sel["valid_msk"], sel["t"], ecfg,
                              geom=(sel["water"], sel["keep"]))
        rp, rd = cv_scores(final_res)
        rows.append(dict(iter="final_cv", strategy=strategy, k=int(final_res["k_opt"]),
                         t_c=float(final_res["tc_opt"]), fit_seconds=time.time() - t0,
                         rmse_point=rp, rmse_day=rd))
        log.info("final CV: point-opt k=%d T_c=%g (%.4f), day-opt k=%d T_c=%g (%.4f)",
                 final_res["k_opt"], final_res["tc_opt"], rp, final_res["k_day"],
                 final_res["tc_day"], rd)

    return dict(keep=keep, p_valid=cls["p_valid"], table=cls["table"], figs=cls["figs"],
                history=pd.DataFrame(rows), converged=converged, last=last,
                keep_bits=bits, n_iter=len(scene_hist),
                scene_history=pd.concat(scene_hist, ignore_index=True),
                final_res=final_res)


# ==================================================================== segmented runs


def subset_inputs(inp: Inputs, idx: np.ndarray) -> Inputs:
    """Return a new Inputs restricted to the given time indices.

    idx: 1-D integer array of positions into inp.times (not a boolean mask).
    scenes and qc are remapped to the new index space.
    """
    idx = np.asarray(idx)
    old_to_new = np.full(len(inp.times), -1, dtype=int)
    old_to_new[idx] = np.arange(len(idx))

    new_scenes, new_qc = {}, {}
    for sid in inp.raw:
        kept = [j for j in inp.scenes[sid] if old_to_new[j] >= 0]
        new_scenes[sid] = [int(old_to_new[j]) for j in kept]
        new_qc[sid] = {int(old_to_new[j]): inp.qc[sid][j] for j in kept}

    new_valid = old_to_new[inp.valid_inds]
    new_valid = new_valid[new_valid >= 0]

    return dataclasses.replace(
        inp,
        times=inp.times[idx],
        seasonal=inp.seasonal[idx],
        raw={sid: v[idx] for sid, v in inp.raw.items()},
        adj={mid: v[idx] for mid, v in inp.adj.items()},
        offsets={sid: v[idx] for sid, v in inp.offsets.items()},
        valid_inds=new_valid,
        scenes=new_scenes,
        qc=new_qc,
    )


def make_windows(
    times: np.ndarray,
    segment_years: float,
    overlap_days: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split a time axis into overlapping windows of `segment_years` years.

    Returns [(window_idx, central_idx), ...] where indices are into `times`.
    The central_idx arrays partition times exactly — no date is double-counted.
    Overlap on each side mitigates the zero-flux boundary effect in filter_time.
    """
    t = times.astype("datetime64[D]")
    t0, t1 = t[0], t[-1]
    span_days = int(segment_years * 365.25)
    overlap = np.timedelta64(int(overlap_days), "D")

    windows = []
    cursor = t0
    while cursor <= t1:
        c_start = cursor
        c_end = min(cursor + np.timedelta64(span_days, "D") - np.timedelta64(1, "D"), t1)
        w_start = max(t0, c_start - overlap)
        w_end = min(t1, c_end + overlap)

        win_idx = np.flatnonzero((t >= w_start) & (t <= w_end))
        cen_idx = np.flatnonzero((t >= c_start) & (t <= c_end))
        windows.append((win_idx, cen_idx))
        cursor = c_end + np.timedelta64(1, "D")
    return windows


def run_segmented_loop(inp: Inputs, cfg: dict) -> dict:
    """Run run_loop on overlapping annual windows and stitch the results.

    Climatology and offsets (already in inp) were fit on the full time series — only the
    DINEOF gap-filling runs per window, keeping n ≈ segment_years×365 regardless of the
    total record length.
    """
    lp = cfg["loop"]
    windows = make_windows(inp.times, float(lp["segment_years"]),
                           int(lp["segment_overlap_days"]))
    log.info("segmented DINEOF: %d windows of ~%.2g yr with %d-day overlap",
             len(windows), float(lp["segment_years"]), int(lp["segment_overlap_days"]))

    # Pre-allocate full-T output arrays
    all_keep   = {sid: np.zeros(inp.raw[sid].shape, bool)           for sid in inp.raw}
    all_bits   = {sid: np.zeros(inp.raw[sid].shape, dtype="uint16") for sid in inp.raw}
    all_pvalid = {sid: np.full(inp.raw[sid].shape, np.nan, "float32") for sid in inp.raw}
    all_table, all_scenes, all_history = [], [], []
    last_out = None

    for w_num, (w_idx, c_idx) in enumerate(windows):
        t_start = str(inp.times[c_idx[0]])[:10]
        t_end   = str(inp.times[c_idx[-1]])[:10]
        log.info("window %d/%d: %s – %s (%d dates, %d with overlap)",
                 w_num + 1, len(windows), t_start, t_end, len(c_idx), len(w_idx))
        w_inp = subset_inputs(inp, w_idx)
        out = run_loop(w_inp, cfg)

        # positions of the central dates inside the window array
        c_in_win = np.flatnonzero(np.isin(w_idx, c_idx))
        for sid in inp.raw:
            all_keep[sid][c_idx]   = out["keep"][sid][c_in_win]
            all_bits[sid][c_idx]   = out["keep_bits"][sid][c_in_win]
            all_pvalid[sid][c_idx] = out["p_valid"][sid][c_in_win]

        central_set = set(inp.times[c_idx].astype("datetime64[D]").astype(str))
        # table = final-iteration classifications; scene_history = all iterations
        all_table.append(out["table"][out["table"]["date"].isin(central_set)])
        all_scenes.append(out["scene_history"][out["scene_history"]["date"].isin(central_set)])
        all_history.append(out["history"].assign(window=w_num))
        last_out = out

    return dict(
        last_out,
        keep=all_keep,
        keep_bits=all_bits,
        p_valid=all_pvalid,
        table=pd.concat(all_table, ignore_index=True),
        scene_history=pd.concat(all_scenes, ignore_index=True),
        history=pd.concat(all_history, ignore_index=True),
        n_iter=last_out["n_iter"],
        converged=last_out["converged"],
    )


# ==================================================================== evaluation (--compare)

def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Agreement of two boolean verdicts beyond chance."""
    if a.size == 0:
        return float("nan")
    po = float((a == b).mean())
    pa, pb = float(a.mean()), float(b.mean())
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def compare_scenes(inp: Inputs, keep: dict, base: np.ndarray, modis_keep: dict,
                   min_removed: int = 200) -> pd.DataFrame:
    """Per scene: agreement and kappa with the MODIS-baseline filter, and the AUC of the final
    residual against what THAT filter removed.

    The AUC is compare_lowrank's statistic without its bootstrap advantage: the field scoring
    the pixels was never fit on data the MODIS filter had cleaned.
    """
    rows = []
    for sid, r in inp.raw.items():
        for j in inp.scenes[sid]:
            obs = np.isfinite(r[j]) & inp.water
            a, b = keep[sid][j][obs], modis_keep[sid][j][obs]
            score = np.abs(inp.adj[sid][j] - base[j])[obs]
            ok = np.isfinite(score)
            removed = ~b
            a_auc = (auc(score[ok], removed[ok])
                     if removed[ok].sum() >= min_removed and b[ok].any() else np.nan)
            rows.append(dict(sensor=sid, date=str(inp.times[j])[:10], n_obs=int(obs.sum()),
                             kept_iter=int(a.sum()), kept_modis=int(b.sum()),
                             agree=float((a == b).mean()) if obs.any() else np.nan,
                             kappa=cohen_kappa(a, b),
                             iter_only=int((a & ~b).sum()), modis_only=int((~a & b).sum()),
                             auc_vs_modis=a_auc))
    return pd.DataFrame(rows)


def downstream_skill(inp: Inputs, keep: dict, final_res: dict, st: xr.Dataset, cfg: dict,
                     modis_cv: Path) -> dict:
    """Point-CV RMSE of the two filters' DINEOF fits, scored on the SAME held-out points.

    Each run's own CV curve holds out a different point set -- the validation pixels a filter
    removed are not in it -- so the numbers in the two CSVs are not comparable directly. Here
    the common points (in both validation sets, observed in both matrices, with the same
    composite value) are held out of both, each run is refit at its own point-opt setting, and
    both are scored in z on those points. Standardization is shared, so z is comparable.
    """
    ecfg = cfg["_edineof"]
    sel_i, _ = build_matrix(inp, keep, cfg)
    sel_m = E.select_matrix(st, ecfg)

    def grid_bool(M, sel):
        return np.nan_to_num(to_grid(M.astype("float32"), sel)) > 0.5

    Xi = to_grid(np.where(sel_i["observed"], sel_i["X"], np.nan), sel_i)
    Xm = to_grid(np.where(sel_m["observed"], sel_m["X"], np.nan), sel_m)
    common = (grid_bool(sel_i["valid_msk"].T, sel_i) & grid_bool(sel_m["valid_msk"].T, sel_m)
              & np.isfinite(Xi) & np.isfinite(Xm) & (np.abs(Xi - Xm) < 1e-5))
    n = int(common.sum())
    out = {"n_common_points": n}
    if n == 0:
        log.warning("downstream: no common held-out points; skipped")
        return out

    curve = pd.read_csv(modis_cv)
    mo = ecfg["modes"]
    tc_m, k_m = E._pick_setting(curve, "rmse_point", mo["rule"], float(mo["rel_tol"]))
    r = curve[(curve["t_c"] == tc_m) & (curve["k"] == k_m)].iloc[0]
    runs = {"iter": (sel_i, int(final_res["k_opt"]),
                     {"t_c": final_res["tc_opt"], "alpha": final_res["alpha_opt"],
                      "p": final_res["p_opt"]}),
            "modis": (sel_m, k_m, {"t_c": tc_m, "alpha": float(r["alpha"]), "p": int(r["p"])})}
    for name, (sel, k, s) in runs.items():
        hold = from_grid(common.astype("float32"), sel) > 0.5
        truth = sel["X"][hold]
        smoother = E.make_spatial_smoother(sel["water"], sel["keep"],
                                           float(ecfg["filter"].get("l_c", 0.0)),
                                           float(ecfg["filter"].get("spatial_alpha_max", 0.25)))
        fit = fit_fixed(sel["X"], sel["observed"] & ~hold, sel["t"], k, s, ecfg,
                        label=f"downstream/{name}", smoother=smoother)
        err = fit["X"][hold] - truth
        scl = np.broadcast_to(sel["scale"][sel["water"]][sel["keep"]][:, None], hold.shape)[hold]
        out[f"rmse_z_{name}"] = float(np.sqrt(np.mean(err ** 2)))
        out[f"rmse_K_{name}"] = float(np.sqrt(np.nanmean((err * scl) ** 2)))
        out[f"k_{name}"], out[f"t_c_{name}"] = int(k), float(s["t_c"])
    rp, rd = cv_scores(final_res)
    out.update(own_rmse_point_iter=rp, own_rmse_day_iter=rd,
               own_rmse_point_modis=float(r["rmse_point"]),
               own_rmse_day_modis=float(curve["rmse_day"].min()))
    log.info("downstream on %d common points: iter %.4f z (k=%d, T_c=%g) vs MODIS-filter "
             "%.4f z (k=%d, T_c=%g)", n, out["rmse_z_iter"], out["k_iter"], out["t_c_iter"],
             out["rmse_z_modis"], out["k_modis"], out["t_c_modis"])
    return out


def modis_keep_masks(inp: Inputs, cfg: dict) -> dict:
    """The MODIS-baseline filter's verdict per sensor: finite in `<var>_dineof`."""
    sfx = cfg["data"]["modis_filtered_suffix"]
    out = {}
    with xr.open_zarr(cfg["data"]["filtered"]) as ft:
        idx = pd.Index(ft["time"].values).get_indexer(inp.times)
        for sid in inp.raw:
            name = f"{cfg['sensors'][sid]['sst']}{sfx}"
            arr = ft[name].values
            k = np.zeros(inp.raw[sid].shape, bool)
            have = idx >= 0
            k[have] = np.isfinite(arr[idx[have]]) & inp.water[None]
            out[sid] = k
    return out


# ==================================================================== output

def build_dataset(inp: Inputs, cfg: dict, out: dict) -> xr.Dataset:
    """The filtered cube's channels, plus `<var>_iter`, `<id>_p_valid_iter`, `<id>_center_iter`.

    `<var>_iter` is a drop-in for `<var>_dineof`: point composite's `members.<id>.var` at it and
    the existing composite -> standardize -> edineof chain re-runs on this filter, which also
    refits the standardization this loop held fixed.
    """
    sfx = cfg["output"]["filtered_suffix"]
    dims = ("time", "y", "x")
    data = {}
    with xr.open_zarr(cfg["data"]["filtered"]) as ft:
        same = (ft.sizes["time"] == len(inp.times)
                and np.array_equal(ft["time"].values, inp.times))
        if not same:
            raise ValueError(f"{cfg['data']['filtered']} is not on the standardized cube's time "
                             "axis; cannot copy its channels")
        for name in ft.data_vars:
            src = ft[name]
            da = xr.DataArray(src.values, dims=src.dims, name=name)
            da.attrs.update(src.attrs, carried_from=str(cfg["data"]["filtered"]))
            data[name] = da
        time_attrs = dict(ft["time"].attrs)
        src_attrs = dict(ft.attrs)

    tbl = out["table"]
    for sid in inp.raw:
        s = cfg["sensors"][sid]
        name = f"{s['sst']}{sfx}"
        arr = np.where(out["keep"][sid], inp.raw[sid], np.nan).astype("float32")
        da = xr.DataArray(arr, dims=dims, name=name)
        da.attrs.update(
            long_name=f"{s['label']} SST, iterative DINEOF-baseline filter", units="K",
            source_channel=s["sst"], filter_p_valid_min=cfg["filter"]["p_valid_min"],
            filter_offset_band=json.dumps([cfg["filter"]["offset_lower"],
                                           cfg["filter"]["offset_upper"]]),
            loop_strategy=cfg["loop"]["strategy"],
            comment=("raw sensor SST where the final iteration kept the pixel, else NaN. A "
                     f"drop-in for {s['sst']}{cfg['data']['modis_filtered_suffix']}."))
        data[name] = da

        pv = xr.DataArray(out["p_valid"][sid], dims=dims, name=f"{sid}_p_valid_iter")
        pv.attrs.update(long_name=f"{s['label']} P(clear) against the final DINEOF baseline",
                        units="1", comment="NaN on dates that are not acquisitions")
        data[f"{sid}_p_valid_iter"] = pv

        centre = np.full(len(inp.times), np.nan, dtype="float32")
        sub = tbl[tbl["sensor"] == sid]
        centre[sub["t"].to_numpy()] = sub["center"].to_numpy()
        c = xr.DataArray(centre, dims=("time",), name=f"{sid}_center_iter")
        c.attrs.update(long_name=f"{s['label']} scene offset against the DINEOF baseline",
                       units="K")
        data[f"{sid}_center_iter"] = c

    for sid in inp.raw:
        s = cfg["sensors"][sid]
        kb = xr.DataArray(out["keep_bits"][sid], dims=dims, name=f"{sid}_keep_history_iter")
        kb.attrs.update(
            long_name=f"{s['label']} per-iteration keep verdicts, as a bitmask", units="1",
            n_iterations=int(out["n_iter"]),
            comment=("bit 0 = kept in the starting mask (sensor QC and valid_range only); bit "
                     "i+1 = kept after iteration i's classification. 0 everywhere a pixel was "
                     "never observed. The highest set bit that matters is n_iterations."))
        data[f"{sid}_keep_history_iter"] = kb

    bl = xr.DataArray(out["last"]["base"], dims=dims, name="sst_baseline_iter")
    bl.attrs.update(baseline_source=out["last"]["source"],
                    long_name="final DINEOF low-rank baseline the scenes were classified against",
                    units="K", comment=("upsampled in z and unstandardized with the native "
                                        "seasonal coefficients and scale"))
    data["sst_baseline_iter"] = bl

    # The final fit's own input and output, so the smooth baseline can be read against both.
    L = out["last"]
    ff = xr.DataArray(baseline_kelvin(L["fit"]["X"], L["sel"], inp), dims=dims,
                      name="sst_filled_iter")
    ff.attrs.update(
        long_name="DINEOF gap-filled analysis of the final fit", units="K",
        coarsen=int(L["sel"]["coarsen"]),
        comment=("observations held fixed, gaps filled by the full-data rank-k EOFs; on the "
                 "matrix grid, block-repeated to this grid when coarsen > 1. Unlike "
                 "sst_baseline_iter, day j's own ECOSTRESS / Landsat pixels are IN this."))
    data["sst_filled_iter"] = ff
    comp = xr.DataArray(L["composite"].astype("float32"), dims=dims, name="sst_composite_iter")
    comp.attrs.update(long_name="composite the final fit was given (native grid)", units="K",
                      comment="MODIS plus the ECOSTRESS / Landsat pixels the loop kept going "
                              "into the final fit, offset-corrected")
    data["sst_composite_iter"] = comp
    src = xr.DataArray(L["composite_src"], dims=dims, name="sst_composite_src_iter")
    src.attrs.update(long_name="members contributing to sst_composite_iter, as bits",
                     flag_meanings=" ".join(inp.order),
                     flag_masks=json.dumps([1 << i for i in range(len(inp.order))]))
    data["sst_composite_src_iter"] = src

    L = out["last"]
    fl = xr.DataArray(L["full_loadings"].astype("float32"), dims=("mode", "time"),
                      name="full_loadings_iter")
    fl.attrs.update(long_name="full-data temporal loadings sigma * V of the final fit",
                    units="1 (z per unit-norm mode)")
    data["full_loadings_iter"] = fl
    if L["loadings"] is not None:
        ml = xr.DataArray(L["loadings"].astype("float32"), dims=("mode", "time"),
                          name="baseline_loadings_iter")
        ml.attrs.update(long_name="MODIS-only loadings the baseline was built from",
                        units="1 (z per unit-norm mode)",
                        comment="0 on days no MODIS reached: the baseline is climatology there")
        data["baseline_loadings_iter"] = ml
        mp = xr.DataArray(L["modis_px"], dims=("time",), name="baseline_modis_px")
        mp.attrs.update(long_name="MODIS cells on the DINEOF matrix grid, per day", units="1")
        data["baseline_modis_px"] = mp
        st = xr.DataArray(L["status"], dims=("time",), name="baseline_loading_status")
        st.attrs.update(long_name="where each day's MODIS loadings came from", units="1",
                        flag_values="0 1 2",
                        flag_meanings="own_modis neighbours_modis none_climatology")
        data["baseline_loading_status"] = st

    ds = xr.Dataset(data, coords={"time": inp.times, **inp.coords})
    for v in ds.data_vars:
        ds[v].encoding = {}
    ds["time"].attrs.update(time_attrs)
    spec = {k: cfg[k] for k in ("filter", "loop", "detector", "sensors", "reference",
                                "composite", "dineof")}
    hist = out["history"]
    ds.attrs.update({**src_attrs,
                     "aoi_id": cfg["data"]["aoi"],
                     "iterative_filter": json.dumps(spec, sort_keys=True, default=str),
                     "iterative_converged": int(bool(out["converged"])),
                     "iterative_n_iter": int((hist["iter"] != "final_cv").sum()),
                     "iterative_filtered_at": provenance.now_utc(),
                     "package_version": provenance.package_version(),
                     "code_version": provenance.code_version()})
    return ds


def write_cube(ds: xr.Dataset, cfg: dict) -> None:
    dest = cfg["data"]["out"]
    compression = CompressionSpec(**cfg["output"]["compression"])
    encoding = datacube.build_encoding(ds, compression, dict(cfg["output"]["chunks"]))
    store.sweep_scratch(dest)
    if dest.exists():
        log.info("replacing existing cube at %s", dest)
    with store.atomic(dest) as tmp:
        datacube.write_zarr(ds, tmp, encoding)
    log.info("wrote %s", dest)


def pick_fig_scenes(inp: Inputs, n: int) -> list[tuple[str, int]]:
    """The n busiest acquisitions, split across sensors as evenly as they allow."""
    ranked = {sid: sorted(inp.scenes[sid], key=lambda j: -int(np.isfinite(inp.raw[sid][j]).sum()))
              for sid in inp.raw}
    out, i = [], 0
    while len(out) < n and any(i < len(v) for v in ranked.values()):
        for sid, v in ranked.items():
            if i < len(v) and len(out) < n:
                out.append((sid, v[i]))
        i += 1
    return out


def apply_tag(cfg: dict, tag: str) -> None:
    """Suffix the cube, the CSVs and the figure dir, like edineof's --tag."""
    out = cfg["data"]["out"]
    cfg["data"]["out"] = out.with_name(f"{out.stem}_{tag}{out.suffix}")
    fig = cfg["output"]["fig_dir"]
    cfg["output"]["fig_dir"] = fig.with_name(f"{fig.name}_{tag}")
    for key in ("report", "scenes", "compare"):
        v = Path(cfg["output"][key])
        cfg["output"][key] = f"{v.stem}_{tag}{v.suffix}"


# ==================================================================== driver

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML config")
    p.add_argument("--tag", default=None, help="suffix every output, for side-by-side variants")
    p.add_argument("--max-iter", type=int, default=None, help="override loop.max_iter")
    p.add_argument("--strategy", choices=STRATEGIES, default=None, help="override loop.strategy")
    p.add_argument("--compare", action="store_true",
                   help="score against the MODIS-baseline filter (agreement, AUC, downstream)")
    p.add_argument("--no-figures", action="store_true", help="skip the figures")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    cfg = load_config(args.config)
    if args.max_iter is not None:
        cfg["loop"]["max_iter"] = int(args.max_iter)
    if args.strategy is not None:
        cfg["loop"]["strategy"] = args.strategy
    if args.tag:
        apply_tag(cfg, args.tag)
        log.info("tag %r: writing %s", args.tag, cfg["data"]["out"].name)

    for key in ("source", "standardized", "composite", "offsets"):
        if not cfg["data"][key].exists():
            raise SystemExit(f"data.{key}: no such path {cfg['data'][key]}")

    t0 = time.time()
    inp, st = load_inputs(cfg)
    log.info("inputs loaded in %.1fs", time.time() - t0)
    figs_wanted = (cfg["output"]["write_figures"] and not args.no_figures)
    fig_scenes = pick_fig_scenes(inp, int(cfg["output"]["figure_days"])) if figs_wanted else []

    out = run_loop(inp, cfg, fig_scenes=set(fig_scenes))

    out_dir = cfg["data"]["out"].parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out["history"].to_csv(out_dir / cfg["output"]["report"], index=False)
    out["table"].to_csv(out_dir / cfg["output"]["scenes"], index=False)
    sh = Path(cfg["output"]["scenes"])
    out["scene_history"].to_csv(out_dir / f"{sh.stem}_history{sh.suffix}", index=False)
    log.info("wrote %s and %s", out_dir / cfg["output"]["report"],
             out_dir / cfg["output"]["scenes"])

    if cfg["data"]["filtered"].exists():
        write_cube(build_dataset(inp, cfg, out), cfg)
    else:
        log.warning("no filtered cube at %s; the output cube is not written",
                    cfg["data"]["filtered"])

    modis_keep, scenes_cmp = None, None
    if args.compare or figs_wanted:
        modis_keep = modis_keep_masks(inp, cfg)
    if args.compare:
        scenes_cmp = compare_scenes(inp, out["keep"], out["last"]["base"], modis_keep)
        worst = scenes_cmp.sort_values("agree").head(8)
        log.info("agreement with the MODIS-baseline filter: median kappa %s, median AUC %s\n"
                 "largest disagreements:\n%s",
                 ", ".join(f"{s} {g['kappa'].median():.3f}" for s, g in
                           scenes_cmp.groupby("sensor")),
                 ", ".join(f"{s} {g['auc_vs_modis'].median():.3f}" for s, g in
                           scenes_cmp.groupby("sensor")),
                 worst[["sensor", "date", "n_obs", "kept_iter", "kept_modis", "agree", "kappa"]]
                 .to_string(index=False))
        summary = {}
        if out["final_res"] is not None and cfg["output"]["modis_cv"].exists():
            summary = downstream_skill(inp, out["keep"], out["final_res"], st, cfg,
                                       cfg["output"]["modis_cv"])
        else:
            log.warning("downstream skill skipped (no final CV, or no %s)",
                        cfg["output"]["modis_cv"])
        path = out_dir / cfg["output"]["compare"]
        scenes_cmp.to_csv(path, index=False)
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(summary, f, indent=2, default=float)
        log.info("wrote %s and %s", path, path.with_suffix(".json"))

    if figs_wanted:
        try:
            import iterative_figures
            iterative_figures.render(inp, cfg, out, modis_keep, fig_scenes,
                                     cfg["output"]["fig_dir"] / cfg["data"]["aoi"])
        except Exception:
            log.exception("figures failed; the cube and CSVs were written and are intact")
    log.info("done in %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
