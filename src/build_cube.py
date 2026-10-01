"""Build the filtered DINEOF input cube.

Runs `simple_outlier_detection` over every acquisition, keeps the pixels it calls clear and
the scenes whose offset against the MODIS covariate is credible, and writes a new zarr cube.

Each sensor lands as a PAIR of channels -- the raw SST under its original name and the
filtered one under `<name>_dineof` -- beside a configured list of carried channels. Only the
suffixed channel is meant to feed DINEOF; the raw one sits next to it so the filter stays
auditable and a different cut can be re-derived without going back to the source cube.

WHY THE DETECTOR AND NOT SCENE CRITERIA. The earlier approach gated scenes on eco_valid_v002
(QC bits) and eco_cloud_v002 (a near-binary raster). Neither sees haze. On 2025-03-07 those
criteria scored the scene 0.970 valid / 0.986 clear -- the cleanest in its window -- while the
detector flagged 26.4% of the accepted pixels, and 100% of the pixels it flagged are called
CLEAR by eco_cloud_v002 on every date tested. The information simply is not in the rasters the
criteria read, so no threshold on them reaches it. The detector works on the residual against
an independent reference, so it can.

TWO STAGES, WITH A CACHE:

  1. detect   -- run_date() per acquisition -> one .nc per date under output.cache_dir.
  2. threshold -- read those, apply `filter`, assemble, write the cube.

The cache key is the DETECTOR settings only. `filter` and `output` are excluded, and the scene
gate is applied here from the stored `center` rather than from the detector's own precomputed
`offset_flag`, so retuning a threshold -- including the offset band -- re-reads the cache in
seconds instead of re-running the mixture model over 212 acquisitions (~9 min).

Usage (from the repo root, in the `coastal_sst_data` env):

    python prototypes/DINEOF/src/build_cube.py \\
        --config prototypes/DINEOF/configs/config.build_cube.admiralty_inlet.yaml
    ... --dry-run      # print the per-scene verdict table, write nothing
    ... --limit 6      # first N acquisitions per sensor, for a smoke test
    ... --refresh      # re-run the detector even where the cache is current
    ... --sensors eco  # one sensor only
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[1]              # prototypes/DINEOF
DEFAULT_CONFIG = ROOT / "configs" / "config.build_cube.admiralty_inlet.yaml"

# The one cross-prototype coupling. `simple_outlier_detection` imports its siblings flatly
# (`from outlier_detection import ...`), so that directory has to be on sys.path before it can
# be imported at all. DINEOF's own src/ is already sys.path[0] when this is run by path.
_CMM_SRC = ROOT.parent / "cloud_mixture_model" / "src"
if str(_CMM_SRC) not in sys.path:
    sys.path.insert(1, str(_CMM_SRC))

import simple_outlier_detection as sod                  # noqa: E402
import simple_outlier_detection_em as sodem
from outlier_detection import _plain, acquisition_dates  # noqa: E402

import cube_figures                                      # noqa: E402  (DINEOF's own src/)
import plotting                                          # noqa: E402

from nearshore_sst import provenance, store, datacube    # noqa: E402
from nearshore_sst.datacube import CompressionSpec       # noqa: E402

log = logging.getLogger("build_cube")


# ==================================================================== config

DEFAULTS = {
    "data": {
        "aoi": "admiralty_inlet",
        "cube": "../cloud_mixture_model/data/datacube/admiralty_inlet.zarr",
        "out": "data/datacube/admiralty_inlet_filtered.zarr",
        "watervar": "landcover_water",
    },
    "detector": {},     # OPAQUE: validated against simple_outlier_detection.DEFAULTS
    "sensors": {},      # OPAQUE: one level per sensor
    "carry": [],        # OPAQUE: a list, not a mapping
    "filter": {
        "p_valid_min": 0.8,
        "offset_lower": -2.0,
        "offset_upper": 4.0,
        "require_reference": False,
        "min_kept_frac": None,
    },
    "output": {
        "cache_dir": "outputs/outlier_cache",
        "chunks": {"time": 64, "y": 128, "x": 128},
        "compression": {"codec": "zstd", "level": 5, "shuffle": "shuffle"},
        "write_detector_figures": False,
        "report": "filter_report.csv",
        "filtered_suffix": "_dineof",
        "write_figures": True,
        "fig_dir": "figures/build_cube",
        "figure_scenes": True,
        "figure_contact": True,
        "figure_offsets": True,
        "figure_histograms": True,
        "figure_scene_hist": True,
        "figure_dpi": 130,
        "figure_ncols": 8,
    },
}

SENSOR_DEFAULTS = {
    "label": None,
    "sst": None,
    "valid": None,
    "cloud": None,
    # The (time,) overpass hour. Carried into the cube automatically so the composite stage can
    # fit a diurnal offset without reopening the source cube -- ECOSTRESS's overpasses span all
    # 24 hours while MODIS Aqua is a night-time retrieval, so the hour is not a diagnostic here,
    # it is a regressor.
    "hour": None,
    "min_pixels": 64,
    "qc_nodata_prior": None,
}

# Scalar keys `detector` carries in addition to simple_outlier_detection's own sections.
DETECTOR_SCALARS = {"ref_var", "depthvar", "tidal_depth_m"}
# Sections of the detector config this process owns rather than the user.
DETECTOR_OWNED = {"data", "output", "offset"}

OPAQUE_SECTIONS = {"detector", "sensors", "carry"}


def load_config(path: Path, defaults: dict = DEFAULTS) -> dict:
    """Read the YAML config over `defaults`. Unknown sections or keys are an error, so a typo
    cannot silently fall back to a default.

    `detector`, `sensors` and `carry` are nested deeper than the two levels this walk handles
    (or, for `carry`, are a list rather than a mapping), so they are taken wholesale and
    validated by their own functions. Everything else keeps outlier_detection.py's walk.

    Relative paths resolve against ROOT and are then made absolute -- the source cube lives in
    a sibling prototype, so its configured path starts `../` and would otherwise leave a `..`
    in every log line.
    """
    with open(path) as f:
        user = yaml.safe_load(f) or {}

    cfg = copy.deepcopy(defaults)
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

    for section, key in (("data", "cube"), ("data", "out"), ("output", "cache_dir"),
                         ("output", "fig_dir")):
        p = Path(cfg[section][key])
        cfg[section][key] = (p if p.is_absolute() else ROOT / p).resolve()

    validate_detector(cfg, path)
    validate_filter(cfg, path)
    validate_sensors(cfg, path)
    validate_carry(cfg, path)
    return cfg


def validate_detector(cfg: dict, path: Path) -> None:
    """Check `detector` against simple_outlier_detection.DEFAULTS.

    Done here rather than left to the detector because its own `load_config` is never called
    (it reads YAML from disk; this process hands it a dict). Without this check a typo would
    reach the model as a silent default and quietly change every verdict in the cube.
    """
    det = cfg["detector"]
    if not isinstance(det, dict) or not det:
        raise ValueError(f"{path}: `detector` must be a non-empty mapping")

    for key, value in det.items():
        if key in DETECTOR_SCALARS:
            continue
        if key in DETECTOR_OWNED:
            raise ValueError(
                f"{path}: detector.{key} is owned by this process, not the config "
                f"({'the scene gate lives in `filter`' if key == 'offset' else 'set via data/output'})")
        if key not in sod.DEFAULTS:
            raise ValueError(f"{path}: unknown detector section '{key}'")
        if not isinstance(value, dict):
            raise ValueError(f"{path}: detector.{key} must be a mapping")
        for k in value:
            if k not in sod.DEFAULTS[key]:
                raise ValueError(f"{path}: unknown key 'detector.{key}.{k}'")

    for key in DETECTOR_SCALARS:
        if key not in det:
            raise ValueError(f"{path}: detector.{key} is required")


def validate_filter(cfg: dict, path: Path) -> None:
    f = cfg["filter"]
    p = f["p_valid_min"]
    if not 0.0 <= float(p) <= 1.0:
        raise ValueError(f"{path}: filter.p_valid_min must be in [0, 1], got {p!r}")
    lo, hi = f["offset_lower"], f["offset_upper"]
    if lo is not None and hi is not None and float(lo) >= float(hi):
        raise ValueError(f"{path}: filter.offset_lower must be below offset_upper")
    mk = f["min_kept_frac"]
    if mk is not None and not 0.0 <= float(mk) <= 1.0:
        raise ValueError(f"{path}: filter.min_kept_frac must be in [0, 1] or null")


def validate_sensors(cfg: dict, path: Path) -> None:
    sensors = cfg["sensors"]
    if not isinstance(sensors, dict) or not sensors:
        raise ValueError(f"{path}: `sensors` must be a non-empty mapping of id -> block")
    for sid, block in sensors.items():
        merged = copy.deepcopy(SENSOR_DEFAULTS)
        for key, value in (block or {}).items():
            if key not in merged:
                raise ValueError(f"{path}: unknown key 'sensors.{sid}.{key}'")
            merged[key] = value
        for key in ("sst", "valid", "cloud", "hour"):
            if not merged[key]:
                raise ValueError(f"{path}: sensors.{sid}.{key} is required")
        merged["label"] = merged["label"] or sid
        sensors[sid] = merged


def validate_carry(cfg: dict, path: Path) -> None:
    carry = cfg["carry"]
    if not isinstance(carry, list):
        raise ValueError(f"{path}: `carry` must be a list of channel names")
    for name in carry:
        if not isinstance(name, str):
            raise ValueError(f"{path}: carry entries must be strings, got {name!r}")
    if len(set(carry)) != len(carry):
        raise ValueError(f"{path}: duplicate entries in `carry`")
    # Every sensor's raw channel is written automatically beside its filtered counterpart, so
    # listing it here would build the same variable twice. Its `hour` is written automatically
    # too, for the same reason.
    for sid, s in cfg["sensors"].items():
        if s["sst"] in carry:
            raise ValueError(
                f"{path}: carry lists {s['sst']!r}, which is already written as "
                f"sensors.{sid}'s raw channel; drop it from `carry`")
        if s["hour"] in carry:
            raise ValueError(
                f"{path}: carry lists {s['hour']!r}, which is already written as "
                f"sensors.{sid}'s overpass hour; drop it from `carry`")

    # The MODIS anchor is not a sensor -- no detector runs on it -- so it reaches the cube only
    # through `carry`. Without it the composite stage has nothing to fit its offsets against,
    # and that failure would surface one stage later against a cube that looks complete.
    ref = cfg["detector"]["ref_var"]
    if ref not in carry:
        raise ValueError(
            f"{path}: detector.ref_var = {ref!r} must also be listed in `carry`; the composite "
            "stage fits every sensor's offset against it and reads it from this cube")

    suffix = cfg["output"]["filtered_suffix"]
    if not isinstance(suffix, str) or not suffix:
        raise ValueError(f"{path}: output.filtered_suffix must be a non-empty string -- the "
                         "raw and filtered channels would otherwise collide")
    for sid, s in cfg["sensors"].items():
        if f"{s['sst']}{suffix}" in carry:
            raise ValueError(f"{path}: carry lists {s['sst']}{suffix}, which this process writes")


def validate_channels(cfg: dict, ds: xr.Dataset, path: Path) -> None:
    """Every configured channel must exist in the source cube. Checked once, up front, so a
    misspelling fails before the first acquisition is processed rather than 9 minutes in."""
    available = sorted(map(str, ds.data_vars))
    wanted = [(f"data.watervar", cfg["data"]["watervar"]),
              ("detector.ref_var", cfg["detector"]["ref_var"]),
              ("detector.depthvar", cfg["detector"]["depthvar"])]
    for sid, s in cfg["sensors"].items():
        wanted += [(f"sensors.{sid}.{k}", s[k]) for k in ("sst", "valid", "cloud", "hour")]
    wanted += [(f"carry[{i}]", n) for i, n in enumerate(cfg["carry"])]
    for where, name in wanted:
        if name not in ds:
            raise ValueError(
                f"{path}: {where} = {name!r} is not in the cube; available: {available}")


# ==================================================================== the detector bridge

def detector_cfg(cfg: dict, sid: str) -> dict:
    """Our config -> the dict simple_outlier_detection.run_date expects.

    Built by overlaying onto a deep copy of its DEFAULTS rather than by calling its
    `load_config`, which reads YAML from disk. Every key is already validated against those
    same DEFAULTS by `validate_detector`.
    """
    s = cfg["sensors"][sid]
    det = cfg["detector"]
    dcfg = copy.deepcopy(sod.DEFAULTS)

    for section in ("reference", "covariate", "clear", "qc", "mixture", "solver"):
        dcfg[section].update(det.get(section, {}))

    dcfg["data"].update(
        cube=cfg["data"]["cube"], var=s["sst"], validvar=s["valid"], cloudvar=s["cloud"],
        ref_var=det["ref_var"], landvar=cfg["data"]["watervar"], depthvar=det["depthvar"],
        tidal_depth_m=det["tidal_depth_m"], dates=[], min_pixels=s["min_pixels"])

    # Per-sensor QC nuance: ECOSTRESS gaps carry a cloud/no-data distinction, Landsat's do not.
    dcfg["qc"]["nodata_prior"] = s["qc_nodata_prior"]

    # Kept in sync so the .nc's own offset_flag agrees with this build's verdict -- but the
    # gate is applied from `center`, so these values are NOT part of the cache key.
    dcfg["offset"] = {"lower": cfg["filter"]["offset_lower"],
                      "upper": cfg["filter"]["offset_upper"]}

    cache = cfg["output"]["cache_dir"] / sid
    dcfg["output"].update(dir=cache, fig_dir=cache / "figures", write_netcdf=True,
                          write_figures=bool(cfg["output"]["write_detector_figures"]))
    return dcfg


def cache_key(dcfg: dict) -> str:
    """The part of the detector config that changes the NUMBERS.

    `output` (where files land), `offset` (only stamps a flag this process does not read) and
    `data.dates` (loop control) are excluded, which is what lets a threshold change re-use the
    cache. Everything else -- every mixture, solver, covariate and QC parameter, and the
    channel names -- invalidates it.
    """
    key = {k: v for k, v in _plain(dcfg).items() if k not in ("output", "offset")}
    key["data"] = {k: v for k, v in key["data"].items() if k != "dates"}
    return yaml.safe_dump(key, sort_keys=True)


def cache_is_current(path: Path, want: str) -> bool:
    """True when `path` holds a run of the same detector settings."""
    if not path.exists():
        return False
    try:
        with xr.open_dataset(path) as d:
            got = d.attrs.get("config")
    except Exception:
        return False
    if not got:
        return False
    stored = yaml.safe_load(got)
    stored = {k: v for k, v in stored.items() if k not in ("output", "offset")}
    stored["data"] = {k: v for k, v in stored.get("data", {}).items() if k != "dates"}
    return yaml.safe_dump(stored, sort_keys=True) == want


def detect_sensor(ds: xr.Dataset, cfg: dict, sid: str, *, refresh: bool,
                  limit: int | None) -> list[Path]:
    """Stage 1. Ensure a current .nc exists for every acquisition of this sensor."""
    s = cfg["sensors"][sid]
    dcfg = detector_cfg(cfg, sid)
    want = cache_key(dcfg)
    cache = dcfg["output"]["dir"]

    water = np.asarray(ds[cfg["data"]["watervar"]].compute() > 0.5)
    dates = acquisition_dates(ds, s["sst"], water, s["min_pixels"])
    if limit:
        dates = dates[:limit]
    log.info("%s: %d acquisitions of %s", sid, len(dates), s["sst"])

    paths, hits, failed = [], 0, []
    for date in dates:
        path = cache / f"{s['sst']}_{date}.nc"
        if not refresh and cache_is_current(path, want):
            hits += 1
            paths.append(path)
            continue
        try:
            result = sod.run_date(ds, date, dcfg)
            sod.write_netcdf(result, ds, dcfg, path)
            if dcfg["output"]["write_figures"]:
                sod.plot_panels(result, dcfg, dcfg["output"]["fig_dir"] / f"{s['sst']}_{date}.png")
            paths.append(path)
        except Exception:
            log.exception("%s: detector failed on %s", sid, date)
            failed.append(date)

    log.info("%s: cache hit %d/%d, ran %d, failed %d",
             sid, hits, len(dates), len(dates) - hits - len(failed), len(failed))
    if failed:
        log.warning("%s: %d dates have no detector output: %s",
                    sid, len(failed), ", ".join(failed))
    return paths


# ==================================================================== stage 2: threshold

def scene_verdict(attrs: dict, cfg: dict) -> tuple[bool, str]:
    """Keep or drop a whole scene, and why. Applied to the .nc's stored attributes.

    Gated on `center` directly rather than on the stored `offset_flag`, so the band can be
    retuned without re-running the detector.
    """
    f = cfg["filter"]
    center = float(attrs["center"])
    lo, hi = f["offset_lower"], f["offset_upper"]
    if lo is not None and center < float(lo):
        return False, f"offset {center:+.2f} below {float(lo):+.2f}"
    if hi is not None and center > float(hi):
        return False, f"offset {center:+.2f} above {float(hi):+.2f}"
    if f["require_reference"] and int(attrs.get("ref_flag", 0)):
        return False, f"no reference (cover {float(attrs.get('ref_cover', 0)):.0%})"
    return True, "kept"


def masked_stack(ds: xr.Dataset, cfg: dict, sid: str, paths: list[Path],
                 water: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    """Stage 2. The sensor's SST with rejected pixels and rejected scenes set to NaN.

    Values come from the SOURCE cube, not from the .nc's `obs` copy of them: the cube stays the
    single source of truth for measurements, and the .nc is consulted only for verdicts. A kept
    pixel therefore carries exactly the number the assembler wrote.
    """
    s = cfg["sensors"][sid]
    p_min = float(cfg["filter"]["p_valid_min"])
    min_kept = cfg["filter"]["min_kept_frac"]
    n_water = int(water.sum())

    sst = ds[s["sst"]].values.astype("float32")
    out = np.full(sst.shape, np.nan, dtype="float32")
    index = {str(t)[:10]: i for i, t in enumerate(pd.to_datetime(ds["time"].values))}

    rows = []
    for path in paths:
        with xr.open_dataset(path) as d:
            attrs = dict(d.attrs)
            p_valid = d["p_valid"].values
        date = str(attrs["date"])[:10]
        i = index.get(date)
        if i is None:
            log.warning("%s: %s is not on the cube's time axis; skipped", sid, date)
            continue

        keep, reason = scene_verdict(attrs, cfg)
        px = (p_valid >= p_min) & water & np.isfinite(sst[i])
        # Recorded for EVERY scene, kept or not. `pixel_frac` is what the pixel cut leaves
        # standing, which for a dropped scene is what it WOULD have contributed -- the whole
        # question the offset figure asks. What actually reached the cube is `cube_px`.
        pixel_frac = px.sum() / n_water

        if keep and min_kept is not None and pixel_frac < float(min_kept):
            keep, reason = False, f"only {pixel_frac:.1%} of water survives the pixel cut"
        if keep:
            out[i] = np.where(px, sst[i], np.nan)

        rows.append(dict(
            date=date, sensor=sid, var=s["sst"], center=float(attrs["center"]),
            ref_cover=float(attrs.get("ref_cover", np.nan)),
            ref_flag=int(attrs.get("ref_flag", 0)),
            cloud_frac=float(attrs.get("cloud_frac", np.nan)),
            pixel_frac=float(pixel_frac), pixel_px=int(px.sum()),
            cube_px=int(px.sum()) if keep else 0,
            verdict="KEPT" if keep else "DROPPED", reason=reason))

    report = pd.DataFrame(rows).sort_values("date", ignore_index=True)
    kept = report["verdict"].eq("KEPT")
    log.info("%s: %d of %d scenes kept; %.1f%% of AoI water retained on a kept scene (median)",
             sid, int(kept.sum()), len(report),
             100 * report.loc[kept, "pixel_frac"].median() if kept.any() else 0.0)

    # The risk `filter.require_reference: false` leaves open, made visible rather than silent.
    admitted_blind = report[kept & report["ref_flag"].astype(bool)]
    if len(admitted_blind):
        log.warning(
            "%s: %d scenes admitted with NO usable MODIS reference (ref_cover below "
            "detector.reference.min_cover). The detector compared them against a constant, so "
            "their p_valid and offset are not meaningful. Set filter.require_reference: true "
            "to drop them. Dates: %s",
            sid, len(admitted_blind), ", ".join(admitted_blind["date"]))
    return out, report


# ==================================================================== assemble

def build_dataset(ds: xr.Dataset, cfg: dict, stacks: dict[str, np.ndarray],
                  reports: dict[str, pd.DataFrame]) -> xr.Dataset:
    """The output cube, on the source's time axis.

    Each sensor contributes a PAIR: the raw channel under its original name, carried over
    untouched, and the filtered channel under `<name><output.filtered_suffix>`. The raw one is
    what makes the filter auditable and reversible from the cube alone -- you can see what was
    removed, and re-derive a different cut, without going back to the source cube. Only the
    suffixed channel is meant to be fed to DINEOF.
    """
    f = cfg["filter"]
    suffix = cfg["output"]["filtered_suffix"]
    band = [f["offset_lower"], f["offset_upper"]]
    data, coord_names = {}, ("time", "y", "x")

    for sid, arr in stacks.items():
        s = cfg["sensors"][sid]
        src = ds[s["sst"]]
        name = f"{s['sst']}{suffix}"

        raw = xr.DataArray(src.values, dims=src.dims, name=s["sst"])
        raw.attrs.update(
            {k: v for k, v in src.attrs.items() if k != "_FillValue"},
            long_name=f"{s['label']} SST, raw",
            carried_from=str(cfg["data"]["cube"]),
            filtered_counterpart=name,
            comment=("unfiltered sensor SST, identical to the source cube. The DINEOF input "
                     f"is {name}."))
        data[s["sst"]] = raw

        da = xr.DataArray(arr, dims=src.dims, name=name)
        da.attrs.update(
            {k: v for k, v in src.attrs.items() if k != "_FillValue"},
            long_name=f"{s['label']} SST, outlier-filtered",
            source_channel=s["sst"],
            filter_p_valid_min=f["p_valid_min"],
            filter_offset_band=json.dumps(band),
            filter_n_scenes_kept=int(reports[sid]["verdict"].eq("KEPT").sum()),
            filter_n_scenes_scored=int(len(reports[sid])),
            comment=("pixels with P(clear) below filter_p_valid_min are NaN; scenes whose "
                     "offset against the MODIS covariate fell outside filter_offset_band are "
                     f"wholly NaN. Kept pixels carry the {s['sst']} value unchanged."))
        data[name] = da

        # The overpass hour, carried unchanged. It is (time,) -- build_encoding clamps the
        # configured time chunk to the axis, which is what the source store already does.
        hsrc = ds[s["hour"]]
        hour = xr.DataArray(hsrc.values, dims=hsrc.dims, name=s["hour"])
        hour.attrs.update(hsrc.attrs, carried_from=str(cfg["data"]["cube"]),
                          overpass_hour_for=sid)
        data[s["hour"]] = hour

    for name in cfg["carry"]:
        src = ds[name]
        da = xr.DataArray(src.values, dims=src.dims, name=name)
        da.attrs.update(src.attrs, carried_from=str(cfg["data"]["cube"]))
        data[name] = da

    coords = {c: ds[c].values for c in coord_names if c in ds.coords}
    out = xr.Dataset(data, coords=coords)

    # Carried channels still hold the SOURCE store's encoding (chunk shape, codecs), which
    # conflicts with the encoding built below and silently wins. Cleared on data_vars ONLY --
    # the time coord's units/calendar must survive, or the axis is rewritten as plain integers.
    for v in out.data_vars:
        out[v].encoding = {}
    if "time" in out.coords:
        out["time"].attrs.update(ds["time"].attrs)
    return out


def cube_attrs(cfg: dict, src_attrs: dict, out: xr.Dataset) -> dict:
    """Source attrs first, so nothing the assembler stamped is dropped, then stamp this stage.

    `created_at` is deliberately left at the ASSEMBLY date and the derivation is dated
    separately, following the package's own preprocess stage.
    """
    # `cfg["detector"]` comes straight from YAML and holds scalars beside its sections, so it
    # is already JSON-clean -- `_plain` is for the two-level-dict detector config built in
    # `detector_cfg`, and would trip over `ref_var` here.
    spec = {"filter": cfg["filter"],
            "detector": cfg["detector"],
            "sensors": {k: dict(v) for k, v in cfg["sensors"].items()},
            "carry": list(cfg["carry"]),
            "source_cube": str(cfg["data"]["cube"])}
    return {**src_attrs,
            "aoi_id": cfg["data"]["aoi"],
            "dineof_filter": json.dumps(spec, sort_keys=True, default=str),
            "dineof_filter_channels": json.dumps(sorted(map(str, out.data_vars))),
            "dineof_filtered_at": provenance.now_utc(),
            "package_version": provenance.package_version(),
            "code_version": provenance.code_version()}


def write_cube(out: xr.Dataset, cfg: dict, src_attrs: dict) -> None:
    """Write the cube atomically, reusing the package's zarr layer rather than to_zarr direct.

    `store.atomic` is driven here rather than via `datacube.write_zarr_safe` because the source
    store must stay open across the write -- the same reason the package's preprocess stage
    does it this way.
    """
    dest = cfg["data"]["out"]
    out.attrs.update(cube_attrs(cfg, src_attrs, out))
    compression = CompressionSpec(**cfg["output"]["compression"])
    encoding = datacube.build_encoding(out, compression, dict(cfg["output"]["chunks"]))

    store.sweep_scratch(dest)
    if dest.exists():
        log.info("replacing existing cube at %s", dest)
    with store.atomic(dest) as tmp:
        datacube.write_zarr(out, tmp, encoding)
    log.info("wrote %s", dest)


# ==================================================================== driver

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML config")
    p.add_argument("--sensors", nargs="+", default=None, help="subset of the sensor blocks")
    p.add_argument("--refresh", action="store_true",
                   help="re-run the detector even where the cache is current")
    p.add_argument("--dry-run", action="store_true",
                   help="print the per-scene verdict table and write no cube")
    p.add_argument("--limit", type=int, default=None,
                   help="first N acquisitions per sensor (smoke test)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_config(args.config)
    sids = args.sensors if args.sensors is not None else list(cfg["sensors"])
    unknown = [s for s in sids if s not in cfg["sensors"]]
    if unknown:
        raise SystemExit(f"--sensors names undefined blocks: {unknown}")

    if not cfg["data"]["cube"].exists():
        raise SystemExit(
            f"no cube at {cfg['data']['cube']}\n"
            "build it with: coastal-sst-data run --config "
            "prototypes/cloud_mixture_model/configs/config.admiralty.yaml --assemble")

    ds = xr.open_zarr(cfg["data"]["cube"])
    validate_channels(cfg, ds, args.config)
    src_attrs = dict(ds.attrs)          # captured before anything else touches the store
    water = np.asarray(ds[cfg["data"]["watervar"]].compute() > 0.5)
    log.info("%s: %d water pixels of %d; p_valid >= %.2f, offset band [%s, %s]",
             cfg["data"]["aoi"], int(water.sum()), water.size,
             cfg["filter"]["p_valid_min"], cfg["filter"]["offset_lower"],
             cfg["filter"]["offset_upper"])

    stacks, reports, sensor_paths = {}, {}, {}
    for sid in sids:
        log.info("=== %s", sid)
        paths = detect_sensor(ds, cfg, sid, refresh=args.refresh, limit=args.limit)
        if not paths:
            log.warning("%s: no detector output; sensor skipped", sid)
            continue
        stacks[sid], reports[sid] = masked_stack(ds, cfg, sid, paths, water)
        sensor_paths[sid] = paths

    if not stacks:
        raise SystemExit("no sensor produced any scenes; nothing to write")

    report = pd.concat(reports.values(), ignore_index=True).sort_values(
        ["sensor", "date"], ignore_index=True)

    if args.dry_run:
        with pd.option_context("display.width", 200, "display.max_rows", None):
            print(report.to_string(index=False))
        log.info("dry run: nothing written")
        return

    out = build_dataset(ds, cfg, stacks, reports)
    write_cube(out, cfg, src_attrs)

    path = cfg["data"]["out"].parent / cfg["output"]["report"]
    path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(path, index=False)
    log.info("wrote %s (%d scenes scored, %d kept)",
             path, len(report), int(report["verdict"].eq("KEPT").sum()))

    if cfg["output"]["write_figures"]:
        # After the cube and the report, and isolated: a drawing bug must not cost a build
        # that already succeeded. The figures are reproducible from the cache alone.
        try:
            render_figures(ds, cfg, stacks, reports, sensor_paths, water)
        except Exception:
            log.exception("figures failed; the cube and report were written and are intact")


def render_figures(ds, cfg, stacks, reports, sensor_paths, water) -> None:
    """Draw the per-pixel and per-image views of what the filter flagged."""
    extent = plotting.extent_km(ds)
    out_dir = cfg["output"]["fig_dir"] / cfg["data"]["aoi"]
    drawn = {}
    for sid, rep in reports.items():
        drawn[sid] = cube_figures.render(ds, cfg, sid, sensor_paths[sid], rep,
                                         stacks[sid], water, extent)
    if cfg["output"]["figure_offsets"]:
        cube_figures.offset_figure(drawn, cfg, out_dir / "scene_offsets.png",
                                   cfg["data"]["aoi"])


if __name__ == "__main__":
    main()
