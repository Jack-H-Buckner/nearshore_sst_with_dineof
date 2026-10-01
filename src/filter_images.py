"""Score every acquisition against per-sensor validity criteria, and show the verdicts.

DINEOF reconstructs a gap-filled field from the scenes it is given, so contamination in the
input stack propagates into the output. This picks the input stack: the dates where Landsat
or ECOSTRESS genuinely saw the water, minus the ones ruined by cloud or sensor error.

The criteria engine is src/helpers.py -- `Criterion` / `evaluate_criteria`. This module adds
the three things a threshold needs before it means anything:

  1. WATER MASKING. `evaluate_criteria` reduces over every pixel it is handed, and land is
     39% of this AoI (55,722 water pixels of 91,809). Unmasked, `eco_valid_v002 == 1` on
     2025-03-05 scores 0.463 where the water scores 0.761 -- the true fraction scaled by the
     AoI's water share, which is a different number for every AoI and therefore a threshold
     that transfers nowhere. Every channel is masked to `landcover_water` and scored with
     nan_policy="ignore", so land leaves the numerator AND the denominator and a `prop:` in
     the config always means "this fraction of the AoI's water".

  2. ACQUISITION DETECTION. The cube's time axis is daily and complete; ECOSTRESS observes
     on 156 of 365 days, Landsat on 56. The empty days score 0 on every criterion and
     `evaluate_criteria` turns their NaNs into False, so without this they would land in the
     FAIL bucket and render a 49-of-156 pass rate as 49 of 365. They are absences, not
     failures, and are reported separately.

  3. EVERY acquisition is drawn, not only the survivors -- SST | valid | cloud, annotated
     with each criterion's measured fraction against its threshold and stamped PASS or FAIL,
     with the two groups written to separate filename prefixes. A cutoff cannot be chosen
     without seeing the scenes just either side of it, so a filter you can only see the
     output of is a filter you cannot tune.

The contact sheet is the tuning instrument: sorted by the leading (first-listed) criterion,
it reads as a block of green, a seam, then a block of red. Move a `prop:` and the seam moves.
A green-framed panel below the seam passed the leading criterion and was vetoed by another
one -- open that scene and the annotation line names which.

Usage (from the repo root, in the `coastal_sst_data` env):

    python prototypes/DINEOF/src/filter_images.py \\
        --config prototypes/DINEOF/configs/config.filtering.admiralty_inlet.yaml
    ... --sensors eco          # one sensor only
    ... --no-scenes            # contact sheets only (~30 s): the threshold sweep loop
    ... --limit 12             # first N acquisitions per sensor, for a smoke test
"""

from __future__ import annotations

import argparse
import copy
import logging
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Bare sibling imports: the script is invoked by path, so `src/` is sys.path[0]. That is how
# every cloud_mixture_model script imports its siblings, and the only reason these are not
# package-relative -- prototypes/ holds no package.
import helpers
import plotting

log = logging.getLogger("filter_images")

ROOT = Path(__file__).resolve().parents[1]              # prototypes/DINEOF
DEFAULT_CONFIG = ROOT / "configs" / "config.filtering.admiralty_inlet.yaml"

# Keys this module consumes itself and strips before helpers.Criterion.from_spec sees them
# (from_spec rejects anything it does not recognise, which is the behaviour we want to keep).
LOCAL_KEYS = {"fill"}

# Contact-sheet margins, in inches: a two-line suptitle and a shared colour bar need the
# same absolute room whether the sheet is 1 row or 20.
HEADER_IN = 0.85
FOOTER_IN = 0.72       # the bar itself, plus its tick labels and axis label beneath it


# ==================================================================== config

DEFAULTS = {
    "data": {
        "aoi": "admiralty_inlet",
        "cube": "../cloud_mixture_model/data/datacube/admiralty_inlet.zarr",
        "watervar": "landcover_water",
        "sensors": [],
    },
    "output": {
        "fig_dir": "figures/filtering",
        "write_scenes": True,
        "write_contact_sheet": True,
        "write_report": False,
    },
    "plot": {
        "shared_scale": True,
        "pct_lo": 1.0,
        "pct_hi": 99.0,
        "ncols": 8,
        "scene_dpi": 130,
        "sheet_dpi": 150,
        "sort_by": "leading",
        "descending": True,
    },
    "sensors": {},      # OPAQUE: nested deeper than two levels, validated separately
}

SENSOR_DEFAULTS = {
    "label": None,
    "short": None,
    "sst": None,
    "valid": None,
    "cloud": None,
    "hour": None,
    "min_pixels": 64,
    "require": "all",
    "default_prop": 0.5,
    "criteria": [],
}

OPAQUE_SECTIONS = {"sensors"}


def load_config(path: Path, defaults: dict = DEFAULTS) -> dict:
    """Read the YAML config over `defaults`. Unknown sections or keys are an error, so a typo
    cannot silently fall back to a default.

    `sensors` is nested deeper than the two levels this walk handles, and its criteria are
    free-form dicts, so it is taken wholesale and validated by `validate_sensors`. Everything
    else keeps outlier_detection.py's two-level walk unchanged.

    Relative paths resolve against ROOT and are then resolved absolute -- the cube lives in a
    sibling prototype, so the configured path starts `../` and would otherwise put a `..` in
    every log line.
    """
    with open(path) as f:
        user = yaml.safe_load(f) or {}

    cfg = copy.deepcopy(defaults)
    for section, values in user.items():
        if section not in cfg:
            raise ValueError(f"{path}: unknown config section '{section}'")
        if section in OPAQUE_SECTIONS:
            cfg[section] = copy.deepcopy(values or {})
            continue
        for key, value in (values or {}).items():
            if key not in cfg[section]:
                raise ValueError(f"{path}: unknown key '{section}.{key}'")
            cfg[section][key] = value

    for section, key in (("data", "cube"), ("output", "fig_dir")):
        p = Path(cfg[section][key])
        cfg[section][key] = (p if p.is_absolute() else ROOT / p).resolve()

    if cfg["plot"]["sort_by"] not in {"leading", "date"}:
        raise ValueError(f"{path}: plot.sort_by must be 'leading' or 'date'")

    validate_sensors(cfg, path)
    return cfg


def _strip_local(spec: dict) -> dict:
    """A criterion dict as helpers understands it: ours minus the keys we handle ourselves."""
    return {k: v for k, v in spec.items() if k not in LOCAL_KEYS}


def validate_sensors(cfg: dict, path: Path) -> None:
    """Validate the opaque `sensors` section, one level per sensor, and every criterion via
    helpers.Criterion.from_spec -- which already rejects unknown keys, and which is called
    HERE so a bad value spec fails in milliseconds instead of after the cube is open.

    Labels are required, unique, and guarded. `evaluate_criteria` writes the fraction columns
    first and only then assigns `out["valid"]` and the `pass: <label>` columns
    (helpers.py:311-327), so a criterion labelled `valid` has its fraction column silently
    overwritten by the boolean verdict -- the annotation code would then print `False` where
    a number belongs. Same hazard for any label starting `pass: `.
    """
    sensors = cfg["sensors"]
    if not isinstance(sensors, dict) or not sensors:
        raise ValueError(f"{path}: `sensors` must be a non-empty mapping of id -> block")

    for sid, block in sensors.items():
        merged = copy.deepcopy(SENSOR_DEFAULTS)
        for key, value in (block or {}).items():
            if key not in merged:
                raise ValueError(f"{path}: unknown key 'sensors.{sid}.{key}'")
            merged[key] = value

        for key in ("sst", "valid", "cloud"):
            if not merged[key]:
                raise ValueError(f"{path}: sensors.{sid}.{key} is required")
        if merged["require"] not in {"all", "any"}:
            raise ValueError(f"{path}: sensors.{sid}.require must be 'all' or 'any'")
        if not merged["criteria"]:
            raise ValueError(f"{path}: sensors.{sid}.criteria is empty")

        seen: set[str] = set()
        for spec in merged["criteria"]:
            if not isinstance(spec, dict):
                raise ValueError(f"{path}: sensors.{sid} criteria must be dicts, got {spec!r}")
            label = spec.get("label")
            if not label:
                raise ValueError(
                    f"{path}: every criterion in sensors.{sid} needs a `label` -- the report "
                    "columns and the figure annotations are indexed by it")
            if label == "valid" or str(label).startswith("pass: "):
                raise ValueError(
                    f"{path}: sensors.{sid} criterion label {label!r} collides with a column "
                    "evaluate_criteria writes itself, which would overwrite its fraction")
            if label in seen:
                raise ValueError(f"{path}: duplicate criterion label {label!r} in {sid}")
            seen.add(label)

            if spec.get("nan_policy", "invalid") != "ignore":
                # With a water-masked view, "invalid" divides by the whole frame again
                # (helpers.py:187) and "valid" makes every land pixel satisfy the criterion.
                log.warning("sensors.%s criterion %r: nan_policy=%r undoes the water masking; "
                            "use 'ignore' (and `fill:` if undefined pixels should count "
                            "against the scene)", sid, label, spec.get("nan_policy"))

            # Parses and range-checks `prop`; raises on an unknown key or a bad value spec.
            helpers.Criterion.from_spec(_strip_local(spec),
                                        default_prop=merged["default_prop"])
            spec.setdefault("prop", merged["default_prop"])

        merged["label"] = merged["label"] or sid
        merged["short"] = merged["short"] or sid.upper()
        sensors[sid] = merged

    unknown = [s for s in cfg["data"]["sensors"] if s not in sensors]
    if unknown:
        raise ValueError(f"{path}: data.sensors names undefined blocks: {unknown}")
    if not cfg["data"]["sensors"]:
        cfg["data"]["sensors"] = list(sensors)


# ==================================================================== scoring

def channel_view(ds: xr.Dataset, water: xr.DataArray,
                 specs: list[dict]) -> tuple[xr.Dataset, list[dict]]:
    """A lazy Dataset of the criterion channels, masked to water, one variable per criterion.

    Keyed by criterion LABEL rather than channel name, so two criteria may read the same
    channel with different preparation (the usual case: a raw cloud raster and a filled one).

    Land becomes NaN and nan_policy="ignore" drops it from numerator and denominator
    (helpers.py:178-183), which is what makes every fraction read "of the AoI water".

    Two shapes need help first:
      * integer channels (`<pre>_valid`, `eco_georef_flag`) carry no NaN at all, so there is
        nothing for `.where()` to write land into until they are floats. Cast to float32 so
        the upcast is not to float64 -- 134 MB per channel rather than 268.
      * (time,) channels have no spatial dim, so Criterion.fraction's
        `dims = [d for d in dims if d in da.dims]` comes back empty and it raises
        (helpers.py:171-175). `broadcast_like` stays lazy and turns a per-date flag into a
        constant field whose fraction is exactly 0 or 1, so `prop: 0.99` reads as "this flag
        must hold". It costs a 33M-cell reduction to answer a 365-element question, which is
        cheap enough here and is why it is not worth special-casing.

    `fill` substitutes a value for water pixels the channel does not define, putting them
    back into the denominator as failures -- "undefined cloud is cloud".
    """
    arrays: dict[str, xr.DataArray] = {}
    clean: list[dict] = []
    for spec in specs:
        spec = dict(spec)
        fill = spec.pop("fill", None)
        name = spec["channel"]
        label = spec["label"]
        if name not in ds:
            raise SystemExit(
                f"cube has no channel {name!r}; available: {sorted(map(str, ds.data_vars))}")

        da = ds[name]
        if da.dtype.kind in "uib":
            da = da.astype("float32")
        if not {"y", "x"} <= set(da.dims):
            da = da.broadcast_like(water)
        if fill is not None:
            da = da.fillna(fill)

        arrays[label] = da.where(water)
        spec["channel"] = label         # the view's variable name; `label` stays the column
        clean.append(_strip_local(spec))

    return xr.Dataset(arrays), clean


def acquisition_mask(ds: xr.Dataset, water: xr.DataArray, sst: str,
                     min_pixels: int) -> tuple[np.ndarray, np.ndarray]:
    """(mask, n_obs) over the daily time axis: days this sensor retrieved water SST.

    `.count()` is a lazy reduction -- dask streams the 134 MB channel chunk by chunk and only
    365 integers come back. Scored on the SST channel, NOT on the cloud raster:
    `eco_cloud_v002` is finite on 269 days, 113 of which carry no SST at all, so the cloud
    raster would invent acquisitions that never happened.
    """
    n = ds[sst].where(water).count(dim=("y", "x")).compute()
    n = np.asarray(n).astype(int)
    return n >= int(min_pixels), n


def score_sensor(ds: xr.Dataset, water: xr.DataArray, n_water: int,
                 scfg: dict) -> pd.DataFrame:
    """Score every date in the cube for one sensor. One dask pass over the masked channels.

    `evaluate_criteria` fuses the per-criterion reductions into a single traversal
    (helpers.py:311-312), so all 365 dates cost about as much as the acquisitions alone --
    and keeping the full table is what lets the report answer "did this date fail, or was
    there nothing there?".

    The helpers wrappers (`filter_valid_images`, `valid_dates`, `iter_valid_images`) each
    re-run `evaluate_criteria` internally, so they are deliberately not used here.
    """
    view, clean = channel_view(ds, water, scfg["criteria"])
    report = helpers.evaluate_criteria(
        view, clean,
        time_dim="time",
        spatial_dims=["y", "x"],
        default_prop=scfg["default_prop"],
        require=scfg["require"],
    )
    acq, n_obs = acquisition_mask(ds, water, scfg["sst"], scfg["min_pixels"])
    report["acquisition"] = acq
    report["n_obs"] = n_obs
    # WATER denominator, never y*x -- the same rule the criterion fractions obey.
    report["frac_obs"] = n_obs / float(n_water)
    return report


# ==================================================================== scene data

def load_block(ds: xr.Dataset, scfg: dict, idx: np.ndarray) -> xr.Dataset:
    """The acquisition-only (t, y, x) block for one sensor, in memory, SST in degC.

    ~129 MB / 0.9 s for ECOSTRESS (156 dates x 3 channels); Landsat is a third of that.
    Re-reading per date instead would be far worse: the cube's time chunk is 64, so a single
    `isel(time=t)` decompresses ~37 MB to hand back 0.37 MB, and across 156 dates that is
    ~5.8 GB of I/O to deliver 57 MB.
    """
    names = [scfg["sst"], scfg["valid"], scfg["cloud"]]
    block = ds[names].isel(time=idx).compute()
    sst = scfg["sst"]
    block[sst] = (block[sst].dims,
                  plotting.to_celsius(block[sst].values.astype("float32")))
    return block


def scene_arrays(block: xr.Dataset, scfg: dict, i: int,
                 water: np.ndarray) -> tuple[np.ndarray, ...]:
    """(sst_degC, valid, invalid, cloud) for one scene. `invalid` is observed-but-rejected."""
    sst = block[scfg["sst"]].values[i]
    valid = np.asarray(block[scfg["valid"]].values[i]) > 0.5
    cloud = np.asarray(block[scfg["cloud"]].values[i], dtype="float32")
    observed = np.isfinite(sst)
    return sst, valid & water, observed & ~valid & water, cloud


def sensor_limits(block: xr.Dataset, water: np.ndarray, scfg: dict,
                  pcfg: dict) -> tuple[float, float]:
    """One shared SST colour range for this sensor, over water-and-sensor-valid pixels."""
    sst = block[scfg["sst"]].values
    valid = np.asarray(block[scfg["valid"]].values) > 0.5
    keep = valid & water[None, :, :] & np.isfinite(sst)
    return plotting.robust_limits(sst, keep, pcfg["pct_lo"], pcfg["pct_hi"])


# ==================================================================== figures

def criteria_line(row: pd.Series, criteria: list[dict], sep: str = "   ") -> str:
    """`label observed/required ok|FAIL` per criterion, in config order.

    Reads `row[label]` for the fraction and `row["pass: " + label]` for the verdict -- the two
    column families evaluate_criteria writes. Labels are validated unique at config time, so
    `_make_labels_unique` never renames one behind our back.
    """
    parts = []
    for c in criteria:
        label = c["label"]
        frac = row[label]
        shown = "  n/a" if not np.isfinite(frac) else f"{frac:.2f}"
        mark = "ok" if bool(row[f"pass: {label}"]) else "FAIL"
        parts.append(f"{label} {shown}/{c['prop']:.2f} {mark}")
    return sep.join(parts)


def _date_str(value) -> str:
    return str(value)[:10]


def scene_figure(block, scfg, pcfg, i, date, row, water, extent, vmin, vmax,
                 hour, out_dir: Path) -> None:
    """One acquisition: SST | <pre>_valid | <pre>_cloud, annotated and stamped."""
    sst, valid, invalid, cloud = scene_arrays(block, scfg, i, water)
    ok = bool(row["valid"])

    if not pcfg["shared_scale"]:
        w = sst[water & np.isfinite(sst)]
        if w.size:
            vmin, vmax = float(np.percentile(w, 2)), float(np.percentile(w, 98))

    n_water = int(water.sum())
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.7), dpi=pcfg["scene_dpi"])

    im = plotting.panel(axes[0], sst, water, vmin=vmin, vmax=vmax, extent=extent,
                        cmap=plotting.sst_cmap(), title=f"{scfg['sst']} [degC]")
    fig.colorbar(im, ax=axes[0], shrink=0.80, label="SST [degC]")

    plotting.mask_panel(
        axes[1], valid, water, extent=extent, color=plotting.VALID_COLOR, second=invalid,
        title=f"{scfg['valid']}  ({valid.sum() / n_water:.0%} of water accepted)")
    axes[1].legend(
        handles=[Patch(facecolor=plotting.VALID_COLOR, label="valid"),
                 Patch(facecolor=plotting.INVALID_COLOR, label="observed, rejected"),
                 Patch(facecolor=plotting.NODATA_COLOR, edgecolor="#b0b0b0",
                       label="not observed")],
        loc="lower left", fontsize=6, framealpha=0.9)

    imc = plotting.panel(axes[2], cloud, water, vmin=0.0, vmax=1.0, extent=extent,
                         cmap=plotting.cloud_cmap(),
                         title=f"{scfg['cloud']}  (undefined = not observed)")
    fig.colorbar(imc, ax=axes[2], shrink=0.80, label="cloud fraction")

    for ax in axes:
        ax.set_xlabel("km east", fontsize=7)
        ax.tick_params(labelsize=6)
    axes[0].set_ylabel("km north", fontsize=7)

    hour_txt = f"{hour:05.2f}Z" if hour is not None and np.isfinite(hour) else "--:--"
    head = (f"{scfg['label']}   {date}   {hour_txt}\n"
            f"{row['frac_obs']:.0%} of AoI water observed "
            f"({int(row['n_obs']):,} of {n_water:,} px)\n"
            + textwrap.fill(criteria_line(row, scfg["criteria"]), width=118))
    fig.suptitle(head, fontsize=9, ha="left", x=0.01, y=0.985)
    plotting.stamp(fig, ok)
    fig.subplots_adjust(top=0.84, bottom=0.09, left=0.05, right=0.98, wspace=0.16)

    plotting.save(fig, out_dir / f"{'pass' if ok else 'fail'}_{date}_{scfg['sid']}.png")


def contact_sheet(block, scfg, pcfg, report, order_pos, water, extent, vmin, vmax,
                  hours, out_path: Path, aoi: str) -> None:
    """Every acquisition for one sensor in a grid, sorted so the boundary is a visible seam.

    Sorted by the LEADING criterion (the first in the config) descending by default, so the
    sheet reads green block / seam / red block. A green panel below the seam passed the
    leading criterion and was vetoed by another -- which is the diagnostic an `require: all`
    filter needs and which no single number gives you.
    """
    n = len(order_pos)
    if not n:
        log.warning("%s: no acquisitions to draw", scfg["sid"])
        return

    ncols = int(pcfg["ncols"])
    nrows = int(np.ceil(n / ncols))
    cmap = plotting.sst_cmap()

    # Explicit spacing rather than tight_layout: the two-line panel titles need room, and
    # tight_layout does not account for a colour bar added to a whole axes list afterwards.
    # The margins are reserved in INCHES and converted, not as a fraction of nrows -- the
    # header a two-line suptitle needs does not shrink when the sheet does, and a fractional
    # reserve collapses the grid to nothing on a short sheet (a `--limit` run, or a sensor
    # with one row of acquisitions).
    fig_h = 2.30 * nrows + HEADER_IN + FOOTER_IN
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.85 * ncols, fig_h),
                             dpi=pcfg["sheet_dpi"],
                             gridspec_kw=dict(hspace=0.46, wspace=0.06))
    axes = np.atleast_1d(axes).ravel()

    lead = scfg["criteria"][0]["label"]
    im = None
    for ax, i in zip(axes, order_pos):
        row = report.iloc[i]
        date = _date_str(report.index[i])
        sst, _, _, _ = scene_arrays(block, scfg, i, water)
        im = plotting.panel(ax, sst, water, vmin=vmin, vmax=vmax, extent=extent, cmap=cmap)
        ok = bool(row["valid"])
        hour = hours[i] if hours is not None else None
        hour_txt = f" {hour:04.1f}Z" if hour is not None and np.isfinite(hour) else ""
        frac = row[lead]
        shown = "n/a" if not np.isfinite(frac) else f"{frac:.2f}"
        ax.set_title(f"{date[5:]} {scfg['short']}{hour_txt}\n{lead} {shown}  "
                     f"{'PASS' if ok else 'FAIL'}",
                     fontsize=6, pad=3,
                     color=plotting.PASS_COLOR if ok else plotting.FAIL_COLOR)
        ax.set_xticks([])
        ax.set_yticks([])
        plotting.frame_verdict(ax, ok)
    for ax in axes[n:]:
        ax.axis("off")

    n_pass = int(report.iloc[list(order_pos)]["valid"].sum())
    sort_txt = (f"sorted by {lead}, {'descending' if pcfg['descending'] else 'ascending'}"
                if pcfg["sort_by"] == "leading" else "in date order")
    fig.suptitle(
        f"{aoi} -- {scfg['label']}: {n_pass} of {n} acquisitions pass "
        f"(require={scfg['require']})\n"
        f"{sort_txt}; shared scale [{vmin:.1f}, {vmax:.1f}] degC over sensor-valid water; "
        f"green frame = PASS",
        fontsize=10, y=1 - 0.20 / fig_h)
    fig.subplots_adjust(top=1 - HEADER_IN / fig_h, bottom=FOOTER_IN / fig_h,
                        left=0.015, right=0.985)
    cax = fig.add_axes((0.36, 0.46 / fig_h, 0.28, 0.10 / fig_h))
    fig.colorbar(im, cax=cax, orientation="horizontal", label="SST [degC]")

    plotting.save(fig, out_path)


def _clear_stale(scene_dir: Path, sid: str) -> None:
    """Drop this sensor's scene figures from the previous run.

    Retuning a threshold flips a scene between the `pass_` and `fail_` prefixes, so the old
    file would survive under its old name and the directory would hold a mix of two tunings.
    """
    if not scene_dir.exists():
        return
    stale = [p for prefix in ("pass", "fail") for p in scene_dir.glob(f"{prefix}_*_{sid}.png")]
    for p in stale:
        p.unlink()
    if stale:
        log.info("%s: cleared %d scene figures from the previous run", sid, len(stale))


# ==================================================================== driver

def run_sensor(ds: xr.Dataset, water_da: xr.DataArray, water: np.ndarray, extent: list[float],
               sid: str, cfg: dict, limit: int | None, write_scenes: bool) -> tuple[int, int]:
    """Score one sensor and draw it. Returns (n_pass, n_acquisitions)."""
    scfg = dict(cfg["sensors"][sid], sid=sid)
    pcfg, ocfg = cfg["plot"], cfg["output"]
    out_dir = ocfg["fig_dir"] / cfg["data"]["aoi"]
    n_water = int(water.sum())

    report = score_sensor(ds, water_da, n_water, scfg)
    acq = report["acquisition"].to_numpy()
    if not acq.any():
        log.warning("%s: no acquisitions with >= %d water pixels", sid, scfg["min_pixels"])
        return 0, 0

    idx = np.flatnonzero(acq)
    if limit:
        idx = idx[:limit]
    scored = report.iloc[idx]
    n_pass = int(scored["valid"].sum())
    log.info("%s: %d of %d dates are acquisitions; %d pass, %d fail (require=%s)",
             sid, len(idx), len(report), n_pass, len(idx) - n_pass, scfg["require"])
    for c in scfg["criteria"]:
        col = scored[c["label"]]
        log.info("    %-14s median %.2f, threshold %.2f -> %d/%d pass",
                 c["label"], float(col.median(skipna=True)), c["prop"],
                 int(scored[f"pass: {c['label']}"].sum()), len(idx))

    block = load_block(ds, scfg, idx)
    vmin, vmax = sensor_limits(block, water, scfg, pcfg)
    log.info("%s: shared scale %.2f..%.2f degC over %s-valid water (%d water px)",
             sid, vmin, vmax, sid, n_water)

    hours = None
    if scfg["hour"] and scfg["hour"] in ds:
        hours = np.asarray(ds[scfg["hour"]].values, dtype="float64")[idx]

    # `scored` rows and `block` time positions share one order: both come from `idx` applied
    # to the cube's time axis.
    assert len(scored) == block.sizes["time"]

    failed: list[str] = []
    if write_scenes:
        _clear_stale(out_dir / "scenes", sid)
        for i in range(len(scored)):
            date = _date_str(scored.index[i])
            try:
                scene_figure(block, scfg, pcfg, i, date, scored.iloc[i], water, extent,
                             vmin, vmax, hours[i] if hours is not None else None,
                             out_dir / "scenes")
            except Exception:
                log.exception("%s: failed to draw %s", sid, date)
                failed.append(date)

    if ocfg["write_contact_sheet"]:
        if pcfg["sort_by"] == "leading":
            lead = scfg["criteria"][0]["label"]
            order = np.argsort(-scored[lead].to_numpy() if pcfg["descending"]
                               else scored[lead].to_numpy(), kind="stable")
        else:
            order = np.arange(len(scored))
        contact_sheet(block, scfg, pcfg, scored, list(order), water, extent, vmin, vmax,
                      hours, out_dir / f"contact_{sid}.png", cfg["data"]["aoi"])

    if ocfg["write_report"]:
        path = out_dir / f"report_{sid}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(path)
        log.info("wrote %s", path)

    if failed:
        log.warning("%s: %d/%d scenes failed to draw: %s",
                    sid, len(failed), len(scored), ", ".join(failed))
    return n_pass, len(scored)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML config")
    p.add_argument("--sensors", nargs="+", default=None, help="override data.sensors")
    p.add_argument("--no-scenes", action="store_true",
                   help="contact sheets only -- the fast loop for sweeping a threshold")
    p.add_argument("--limit", type=int, default=None,
                   help="draw only the first N acquisitions per sensor (smoke test)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_config(args.config)
    if args.sensors is not None:
        unknown = [s for s in args.sensors if s not in cfg["sensors"]]
        if unknown:
            raise SystemExit(f"--sensors names undefined blocks: {unknown}")
        cfg["data"]["sensors"] = args.sensors

    ds = plotting.open_cube(cfg["data"]["cube"])
    water = plotting.water_mask(ds, cfg["data"]["watervar"])
    water_da = ds[cfg["data"]["watervar"]] > 0.5
    extent = plotting.extent_km(ds)
    log.info("%s: %d water pixels of %d (%.0f%% of the frame) -- every proportion below is "
             "scored on water", cfg["data"]["aoi"], int(water.sum()), water.size,
             100 * water.sum() / water.size)

    write_scenes = cfg["output"]["write_scenes"] and not args.no_scenes
    failed = []
    for sid in cfg["data"]["sensors"]:
        log.info("=== %s", sid)
        try:
            run_sensor(ds, water_da, water, extent, sid, cfg, args.limit, write_scenes)
        except Exception:
            log.exception("failed on sensor %s", sid)
            failed.append(sid)

    if failed:
        log.warning("%d/%d sensors failed: %s",
                    len(failed), len(cfg["data"]["sensors"]), ", ".join(failed))


if __name__ == "__main__":
    main()
