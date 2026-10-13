"""Stage 4: point-level state-space (Kalman) smoothing of the gap-filled fields.

Reads stage 2's cube and a CSV of lat/lon sites (`site_id, lat, lon, group`), extracts each
site's daily series, and smooths it with a per-group state-space model (see `state_space.py`):
the latent temperature is modelled as a random walk / AR(1) / AR(1)+harmonics / harmonic+trend,
and each day's available estimate -- a direct observation, a point- or day-filled DINEOF value, or
the smooth field (further split by context) -- enters as a measurement whose error variance is
estimated by MLE and may be shared across the group's sites.

    python src/run_region.py --config configs/region.<aoi>.yaml pointfilter [--tag T]

Outputs under <out_dir>/<aoi>[_tag]/stage4_pointfilter/:
  reports/point_series.csv   long: site, group, time, y_obs, obs_type, smoothed_mean/sd, ci_lo/hi
  reports/point_params.csv   fitted parameters per group, with llf / aic
  <aoi>_points.zarr          dims (time, site): smoothed_mean, smoothed_sd, y_obs, obs_type
  figures/<site>.png         observations coloured by type, smoothed mean + 95% band
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plotting
import region as R
import state_space as SS
import validate_insitu as V
from nearshore_sst import datacube, provenance, store
from nearshore_sst.datacube import CompressionSpec

log = logging.getLogger("stage4_pointfilter")

# Channels we sample at each site pixel. F / S are SST (K -> degC); the rest are per-day flags.
_FILLED, _SMOOTH = "sst_filled", "sst_smooth"
_OBSERVED, _CONSTRAINED, _LOADING = ("sst_filled_observed", "sst_filled_constrained",
                                     "smooth_loading_status")


def _read_points(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"site_id", "lat", "lon"}
    if not need.issubset(df.columns):
        raise SystemExit(f"{path}: points CSV needs columns {sorted(need)} (+ optional `group`)")
    df = df.rename(columns={"site_id": "station_id"})
    df["station_name"] = df.get("station_name", df["station_id"])
    df["group"] = df["group"] if "group" in df else "default"
    return df


def extract_series(ds: xr.Dataset, placed: pd.DataFrame) -> list[dict]:
    """Per placed site (row>=0), the series state_space.fit_group needs, SST in degC."""
    iy = xr.DataArray(placed["row"].to_numpy(), dims="station")
    ix = xr.DataArray(placed["col"].to_numpy(), dims="station")
    F = ds[_FILLED].isel(y=iy, x=ix).values.astype(float) - 273.15          # (T, n)
    S = ds[_SMOOTH].isel(y=iy, x=ix).values.astype(float) - 273.15
    observed = (ds[_OBSERVED].isel(y=iy, x=ix).values.astype(float) > 0.5     # (T, n)
                if _OBSERVED in ds else np.zeros_like(F, bool))
    constrained = (np.asarray(ds[_CONSTRAINED].values).astype(float) > 0.5    # (T,)
                   if _CONSTRAINED in ds else np.ones(F.shape[0], bool))
    clim = (np.asarray(ds[_LOADING].values).astype(int) == 2                  # (T,) none => clim
            if _LOADING in ds else np.zeros(F.shape[0], bool))
    out = []
    for j in range(F.shape[1]):
        ctx = SS.day_context(observed[:, j], constrained, clim)
        out.append(dict(F=F[:, j], S=S[:, j], ctx=ctx, climatology=clim))
    return out


def insitu_series(ds: xr.Dataset, ok: pd.DataFrame, rc: dict, days) -> np.ndarray:
    """(T, n_sites) in-situ daily-mean series (degC) for each site: the nearest in-situ station
    within point_filter.max_snap_m, from validation.insitu. NaN where no station is in range."""
    insitu_path = rc["validation"]["insitu"]
    if not insitu_path or not Path(insitu_path).exists():
        raise SystemExit("a group uses the 'insitu' source but validation.insitu is not set to an "
                         "existing netCDF / CSV")
    vcfg = R.validate_vcfg(rc)
    df = V.read_insitu(Path(insitu_path), vcfg)
    water = np.asarray(ds[rc["region"]["watervar"]].values > 0.5)
    stations = df.groupby("station_id", as_index=False).agg(
        station_name=("station_name", "first"), lat=("lat", "median"), lon=("lon", "median"))
    placed = V.place_stations(stations, ds, water, float(rc["point_filter"]["max_snap_m"]))
    placed = placed[placed["row"] >= 0].reset_index(drop=True)
    min_cov = float(vcfg["match"]["min_coverage"])
    max_snap = float(rc["point_filter"]["max_snap_m"])
    I = np.full((len(days), len(ok)), np.nan)
    for i in range(len(ok)):
        if placed.empty:
            break
        d = np.hypot(placed["x"].to_numpy() - ok.loc[i, "x"], placed["y"].to_numpy() - ok.loc[i, "y"])
        k = int(np.argmin(d))
        if d[k] > max_snap:
            continue
        sid = placed.loc[k, "station_id"]
        dm, _ = V.match_daily_mean(df[df["station_id"] == sid], days, min_cov)
        I[:, i] = dm
        log.info("  in-situ: site %s <- station %s (%.0f m, %d days)",
                 ok.loc[i, "station_id"], sid, d[k], int(np.isfinite(dm).sum()))
    return I


def _primary_obs(site: dict, sources: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """For reporting: the first available source each day (precedence order), as a global
    ALL_TYPES code (-1 none) and its value. Independent of the fitting mode."""
    T = len(site["F"])
    code = np.full(T, -1, dtype=int)
    value = np.full(T, np.nan)
    filled = np.zeros(T, bool)
    for tp in sources:
        avail, vals = SS._type_available(tp, site)
        take = avail & ~filled
        code[take] = SS.ALL_TYPES.index(tp)
        value[take] = vals[take]
        filled |= take
    return code, value


def _group_cfg(pf: dict, name: str) -> dict:
    g = dict(pf["groups"][name])
    g.setdefault("period_days", pf["period_days"])
    g.setdefault("mode", "hierarchy")
    g.setdefault("n_harmonics", 0)
    g.setdefault("share", [])
    g.setdefault("dineof_bias", "none")
    return g


def run(rc: dict, *, tag: str | None = None, figures: bool = True) -> dict:
    t0 = time.time()
    dirs = R.stage_dirs(rc, tag)
    cube = dirs["dineof"] / f"{rc['region']['aoi']}_dineof.zarr"
    if not cube.exists():
        raise SystemExit(f"no stage-2 cube at {cube}; run:\n  python src/run_region.py "
                         f"--config {rc['path']}{' --tag ' + tag if tag else ''} dineof")
    pf = rc["point_filter"]
    if not pf.get("points"):
        raise SystemExit("point_filter.points is null; set it to a CSV of site_id, lat, lon, group")
    if not Path(pf["points"]).exists():
        raise SystemExit(f"point_filter.points: no such file {pf['points']}")

    out = dirs["pointfilter"]
    (out / "reports").mkdir(parents=True, exist_ok=True)
    points = _read_points(Path(pf["points"]))

    with xr.open_zarr(cube) as ds:
        water = np.asarray(ds[rc["region"]["watervar"]].values > 0.5)
        placed = V.place_stations(points, ds, water, float(pf["max_snap_m"]))
        placed = placed.merge(points[["station_id", "group"]], on="station_id", how="left")
        ok = placed[placed["row"] >= 0].reset_index(drop=True)
        if ok.empty:
            raise SystemExit("no sites placed on water; check lat/lon and point_filter.max_snap_m")
        groups = list(ok["group"].unique())
        for name in groups:
            if name not in pf["groups"]:
                raise SystemExit(f"points CSV has group {name!r} with no point_filter.groups entry")
        times = pd.to_datetime(ds["time"].values)
        series = extract_series(ds, ok)
        crs = ds.attrs.get("crs")
        # In-situ channel: only read when some group actually uses it.
        if any("insitu" in _group_cfg(pf, g)["sources"] for g in groups):
            I = insitu_series(ds, ok, rc, times.normalize())
            for i in range(len(ok)):
                series[i]["I"] = I[:, i]

    series_rows, param_rows, fitted = [], [], {}
    for name in groups:
        pos = [i for i in range(len(ok)) if ok.loc[i, "group"] == name]
        gcfg = _group_cfg(pf, name)
        grp_series = [series[i] for i in pos]
        log.info("group %s: %d site(s), model %s, mode %s, sources %s, bias %s", name, len(pos),
                 gcfg["model"], gcfg["mode"], gcfg["sources"], gcfg["dineof_bias"])
        fit = SS.fit_group(times, grp_series, gcfg)
        fitted[name] = (pos, fit)
        bias = fit["model"].bias_series(fit["res"])
        for pname, pval in fit["params"].items():
            param_rows.append(dict(group=name, name=pname, value=pval))
        param_rows.append(dict(group=name, name="llf", value=fit["llf"]))
        param_rows.append(dict(group=name, name="aic", value=fit["aic"]))
        for k, i in enumerate(pos):
            mean, sd = fit["smoothed"][k]
            code, yobs = _primary_obs(series[i], gcfg["sources"])
            otype = np.where(code >= 0, np.array(["", *SS.ALL_TYPES])[code + 1], "")
            ins = series[i].get("I")
            series_rows.append(pd.DataFrame(dict(
                site=ok.loc[i, "station_id"], group=name, time=times,
                y_obs=yobs, obs_type=otype, smoothed_mean=mean, smoothed_sd=sd,
                ci_lo=mean - 1.96 * sd, ci_hi=mean + 1.96 * sd,
                insitu_obs=(ins if ins is not None else np.full(len(times), np.nan)),
                dineof_bias=bias[k])))

    series_df = pd.concat(series_rows, ignore_index=True)
    series_df.to_csv(out / "reports" / "point_series.csv", index=False)
    pd.DataFrame(param_rows).to_csv(out / "reports" / "point_params.csv", index=False)

    _write_zarr(out / f"{rc['region']['aoi']}_points.zarr", ok, times, fitted, series_df,
                rc, cube, crs)

    if figures:
        try:
            for i in range(len(ok)):
                _site_figure(ok.loc[i, "station_id"], series_df, out / "figures" /
                             f"{ok.loc[i, 'station_id']}.png", int(pf["dpi"]))
        except Exception as e:                       # a plotting bug must not lose the outputs
            log.warning("stage 4 figures failed: %s", e)

    log.info("stage 4 done in %.1f min -> %s (%d sites, %d groups)",
             (time.time() - t0) / 60, out, len(ok), len(fitted))
    return dict(dir=out, n_sites=len(ok), groups=list(fitted), series=series_df)


def _write_zarr(dest: Path, ok: pd.DataFrame, times, fitted: dict, series_df: pd.DataFrame,
                rc: dict, cube: Path, crs) -> None:
    """(time, site) cube of the smoothed series + its inputs, written atomically with provenance."""
    sites = list(ok["station_id"])
    T = len(times)
    mean = np.full((T, len(sites)), np.nan, "float32")
    sd = np.full((T, len(sites)), np.nan, "float32")
    for name, (pos, fit) in fitted.items():
        for k, i in enumerate(pos):
            m, s = fit["smoothed"][k]
            col = sites.index(ok.loc[i, "station_id"])
            mean[:, col], sd[:, col] = m, s
    def piv(col):
        return (series_df.pivot(index="time", columns="site", values=col)
                .reindex(columns=sites).to_numpy().astype("float32"))
    ds = xr.Dataset(
        {"smoothed_mean": (("time", "site"), mean, {"units": "degC",
            "long_name": "state-space smoothed SST (debiased, in-situ scale)"}),
         "smoothed_sd": (("time", "site"), sd, {"units": "degC",
            "long_name": "smoothed-state standard deviation"}),
         "y_obs": (("time", "site"), piv("y_obs"), {"units": "degC",
            "long_name": "observation fed to the filter (primary source that day)"}),
         "insitu_obs": (("time", "site"), piv("insitu_obs"), {"units": "degC",
            "long_name": "in-situ daily-mean observation at the site"}),
         "dineof_bias": (("time", "site"), piv("dineof_bias"), {"units": "degC",
            "long_name": "estimated DINEOF measurement bias (added to DINEOF observations)"})},
        coords={"time": times.values, "site": sites,
                "lat": ("site", ok["lat"].to_numpy()), "lon": ("site", ok["lon"].to_numpy()),
                "row": ("site", ok["row"].to_numpy()), "col": ("site", ok["col"].to_numpy()),
                "group": ("site", ok["group"].astype(str).to_numpy())})
    ds.attrs.update({k: v for k, v in (("crs", crs),) if v is not None})
    ds.attrs.update(point_filter_config=_json(rc["point_filter"]), source_cube=str(cube),
                    created_at=provenance.now_utc(), package_version=provenance.package_version(),
                    code_version=provenance.code_version())
    comp = CompressionSpec()
    enc = datacube.build_encoding(ds, comp, {"time": T, "site": len(sites)})
    store.sweep_scratch(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with store.atomic(dest) as tmp:
        datacube.write_zarr(ds, tmp, enc)
    log.info("wrote %s", dest)


def _json(obj):
    import json
    return json.dumps(obj, default=str, sort_keys=True)


def _site_figure(site: str, series_df: pd.DataFrame, path: Path, dpi: int) -> None:
    d = series_df[series_df["site"] == site].sort_values("time")
    fig, ax = plt.subplots(figsize=(10, 3.4), dpi=dpi, layout="constrained")
    ax.fill_between(d["time"], d["ci_lo"], d["ci_hi"], color="#9ec5fe", alpha=0.5, lw=0,
                    label="95% band")
    ax.plot(d["time"], d["smoothed_mean"], color="#1864ab", lw=1.3, label="smoothed")
    seen = d[d["obs_type"] != ""]
    for tp, sub in seen.groupby("obs_type"):
        ax.plot(sub["time"], sub["y_obs"], ".", ms=3.5, label=tp)
    if "insitu_obs" in d and d["insitu_obs"].notna().any():
        ins = d[d["insitu_obs"].notna()]
        ax.plot(ins["time"], ins["insitu_obs"], "x", ms=4, color="#c92a2a", label="in-situ")
    if "dineof_bias" in d and (d["dineof_bias"].abs() > 1e-9).any():
        ax.plot(d["time"], d["dineof_bias"], color="#e8590c", lw=1.0, ls="--",
                label="DINEOF bias")
    ax.set_title(f"{site}: state-space smoothed SST", fontsize=10)
    ax.set_ylabel("SST [degC]", fontsize=8)
    ax.legend(fontsize=6, ncol=5, loc="upper right")
    plotting.save(fig, path, dpi=dpi)
