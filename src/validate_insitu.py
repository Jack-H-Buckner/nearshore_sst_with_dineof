"""Validate the pipeline's gap-filled and smooth SST against in-situ water temperature.

In-situ data comes from one of three places:

  cube    the pipeline cube's own `insitu_*` channels, carried from a source cube that
          coastal_sst_data built with `datacube.insitu: true`. Station pixels come from
          `insitu_station` and the `insitu_stations` attr. Used when no --insitu file is given.
  netCDF  coastal_sst_data's in-situ product: dims (station, time), `sst` (degC), `qc`
          (QARTOD), coords `station_id`, `station_name`, `lat`, `lon`.
  CSV     long format, one row per observation: station_id, time, latitude, longitude, value
          [, z]. Column names, units and time zone are configurable.

The product is daily and sits on MODIS Aqua's night-time scale (the composite anchor), while
in-situ sensors sample every few minutes, so each station-day is matched TWO ways and both
are reported:

  overpass    the observation nearest the MODIS overpass that day (`modis_hour_aqua`; the
              record's median hour when MODIS did not fly), within `max_dt_min` -- like for
              like with the product's scale. From a cube: `modis_insitu_sst`.
  daily_mean  the mean over the UTC day, when observations cover `min_coverage` of its hours.
              Not available from a cube, which stores instants only. A cube's `insitu_sst`
              (at the met reference time) is reported as `reference` instead.

Every matchup samples `sst_filled`, `sst_filled_point`, `sst_filled_day`, `sst_smooth` and the
seasonal climatology at the station's pixel, so the skill over climatology is always shown.

Usage:

    python src/validate_insitu.py --cube data/pipeline/admiralty_inlet_pipeline.zarr \\
        --insitu data/insitu/admiralty_inlet_insitu.nc
    python src/validate_insitu.py --cube <cube>               # in-situ from the cube itself
    ... --config configs/validate.yaml --tag e2e --no-figures
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

import iterative_filter  # noqa: F401  (bridges seasonal_smoothing)
from seasonal_smoothing import design_matrix   # noqa: E402

log = logging.getLogger("validate_insitu")

DEFAULTS = {
    "products": ["sst_filled", "sst_filled_point", "sst_filled_day", "sst_smooth"],
    "match": {
        "max_dt_min": 60.0,         # overpass: nearest observation within this many minutes
        "min_coverage": 0.75,       # daily_mean: share of the day's 24 hours with an obs
        "max_snap_m": 300.0,        # a land-pixel station moves to water within this, or drops
        "hour_var": "modis_hour_aqua",
        "default_hour": None,       # UTC hour when the cube has no hour channel at all
    },
    "insitu": {
        "format": "auto",           # auto | netcdf | csv
        "columns": {"station_id": "station_id", "time": "time", "latitude": "latitude",
                    "longitude": "longitude", "value": "value", "z": "z",
                    "station_name": "station_name"},
        "units": "degC",            # degC | K
        "time_zone": "UTC",         # of naive CSV times
        "qc_pass": [1, 2],          # netCDF QARTOD flags kept
        "max_depth_m": 5.0,         # CSV `z` deeper than this is not surface truth
    },
    "output": {"dpi": 130},
}

MATCHES = ("overpass", "daily_mean", "reference")
MATCHUP_COLS = ["station_id", "station_name", "row", "col", "snap_m", "date", "t", "match",
                "insitu", "dt_min", "coverage", "pixel_observed", "day_constrained",
                "loading_status", "seasonal_fit", "season"]
SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
           6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
# sst_seasonal_fit_type codes -> label (seasonal_smoothing FIT_*: 0 ref, 1 mean-only, 2 full)
SEASONAL_FITS = {-1: "land", 0: "reference", 1: "mean-only", 2: "full"}


def _seasonal_fit_label(ftype: np.ndarray | None, row: int, col: int) -> str:
    """The seasonal climatology fit method at a station's pixel, as a label."""
    if ftype is None or row < 0 or col < 0:
        return "unknown"
    return SEASONAL_FITS.get(int(ftype[row, col]), "unknown")


def load_config(path: Path | None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path is None:
        return cfg
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    for section, values in user.items():
        if section not in cfg:
            raise ValueError(f"{path}: unknown config section '{section}'")
        if section == "products":
            cfg["products"] = list(values)
            continue
        for key, value in (values or {}).items():
            if key not in cfg[section]:
                raise ValueError(f"{path}: unknown key '{section}.{key}'")
            if isinstance(cfg[section][key], dict):
                cfg[section][key] = {**cfg[section][key], **value}
            else:
                cfg[section][key] = value
    return cfg


# ==================================================================== in-situ readers

def read_insitu(path: Path, cfg: dict) -> pd.DataFrame:
    """Any supported file -> long DataFrame: station_id, station_name, lat, lon, time (UTC,
    naive), value_C."""
    path = Path(path)
    ic = cfg["insitu"]
    fmt = ic["format"]
    if fmt == "auto":
        fmt = "netcdf" if path.suffix in (".nc", ".nc4", ".cdf") else "csv"
    if fmt == "netcdf":
        df = _read_netcdf(path, ic)
    elif fmt == "csv":
        df = _read_csv(path, ic)
    else:
        raise ValueError(f"insitu.format must be auto, netcdf or csv, got {fmt!r}")
    if ic["units"] == "K":
        df["value_C"] = df["value_C"] - 273.15
    elif ic["units"] != "degC":
        raise ValueError(f"insitu.units must be degC or K, got {ic['units']!r}")
    df = df.dropna(subset=["time", "value_C", "lat", "lon"])
    log.info("in-situ: %d observations from %d stations (%s)", len(df),
             df["station_id"].nunique(), path.name)
    return df.sort_values(["station_id", "time"], ignore_index=True)


def _read_netcdf(path: Path, ic: dict) -> pd.DataFrame:
    with xr.open_dataset(path) as ds:
        sst = ds["sst"].values
        qc = ds["qc"].values if "qc" in ds else None
        times = pd.to_datetime(ds["time"].values)
        ids = [str(v) for v in ds["station_id"].values]
        names = ([str(v) for v in ds["station_name"].values] if "station_name" in ds.coords
                 else ids)
        lat, lon = ds["lat"].values, ds["lon"].values
    rows = []
    for i in range(sst.shape[0]):
        ok = np.isfinite(sst[i])
        if qc is not None:
            ok &= np.isin(qc[i], list(ic["qc_pass"]))
        rows.append(pd.DataFrame({"station_id": ids[i], "station_name": names[i],
                                  "lat": float(lat[i]), "lon": float(lon[i]),
                                  "time": times[ok], "value_C": sst[i][ok].astype(float)}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["station_id", "station_name", "lat", "lon", "time", "value_C"])


def _read_csv(path: Path, ic: dict) -> pd.DataFrame:
    c = ic["columns"]
    raw = pd.read_csv(path)
    for key in ("station_id", "time", "latitude", "longitude", "value"):
        if c[key] not in raw:
            raise ValueError(f"{path}: no column {c[key]!r} (insitu.columns.{key})")
    t = pd.to_datetime(raw[c["time"]], errors="coerce")
    if t.dt.tz is None:
        t = t.dt.tz_localize(ic["time_zone"])
    t = t.dt.tz_convert("UTC").dt.tz_localize(None)
    df = pd.DataFrame({"station_id": raw[c["station_id"]].astype(str),
                       "station_name": (raw[c["station_name"]].astype(str)
                                        if c["station_name"] in raw
                                        else raw[c["station_id"]].astype(str)),
                       "lat": raw[c["latitude"]].astype(float),
                       "lon": raw[c["longitude"]].astype(float),
                       "time": t, "value_C": raw[c["value"]].astype(float)})
    if c["z"] in raw and ic["max_depth_m"] is not None:
        deep = raw[c["z"]].abs() > float(ic["max_depth_m"])
        if deep.any():
            log.info("in-situ: %d observations deeper than %g m dropped", int(deep.sum()),
                     float(ic["max_depth_m"]))
        df = df[~deep.to_numpy()]
    return df


# ==================================================================== placement

def place_stations(stations: pd.DataFrame, ds: xr.Dataset, water: np.ndarray,
                   max_snap_m: float) -> pd.DataFrame:
    """station_id, lat, lon -> + row, col, snap_m, status (ok | snapped | outside | no_water).

    Nearest pixel centre in the cube CRS; a station whose pixel is land moves to the nearest
    water pixel within `max_snap_m`. Coastal gauges usually sit on a pier, i.e. on land in a
    100 m mask, so snapping is the norm rather than the exception -- and the distance is
    reported so a 290 m snap is not mistaken for a co-located match.
    """
    from pyproj import Transformer
    crs = ds.attrs.get("crs")
    if not crs:
        raise ValueError("the cube has no `crs` attr; cannot place lat/lon stations")
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = ds["x"].values, ds["y"].values
    dx, dy = float(np.median(np.abs(np.diff(xs)))), float(np.median(np.abs(np.diff(ys))))
    wr, wc = np.nonzero(water)
    out = []
    for _, s in stations.iterrows():
        X, Y = tr.transform(float(s["lon"]), float(s["lat"]))
        col, row = int(np.argmin(np.abs(xs - X))), int(np.argmin(np.abs(ys - Y)))
        rec = dict(station_id=s["station_id"], station_name=s["station_name"],
                   lat=float(s["lat"]), lon=float(s["lon"]), x=X, y=Y, row=row, col=col,
                   snap_m=0.0, status="ok")
        if abs(xs[col] - X) > dx / 2 + 1e-6 or abs(ys[row] - Y) > dy / 2 + 1e-6:
            rec.update(status="outside", row=-1, col=-1)
        elif not water[row, col]:
            d = np.hypot(xs[wc] - X, ys[wr] - Y)
            if d.size == 0 or d.min() > max_snap_m:
                rec.update(status="no_water", row=-1, col=-1,
                           snap_m=float(d.min()) if d.size else np.nan)
            else:
                i = int(np.argmin(d))
                rec.update(row=int(wr[i]), col=int(wc[i]), snap_m=float(d[i]),
                           status="snapped")
        out.append(rec)
    placed = pd.DataFrame(out)
    for _, r in placed[placed["status"].isin(["outside", "no_water"])].iterrows():
        log.warning("station %s dropped: %s%s", r["station_id"], r["status"],
                    f" (nearest water {r['snap_m']:.0f} m)" if r["status"] == "no_water" else "")
    return placed


def cube_stations(ds: xr.Dataset) -> pd.DataFrame:
    """Station table from a cube's `insitu_stations` attr (row/col already on the grid)."""
    meta = json.loads(ds.attrs["insitu_stations"])
    return pd.DataFrame([dict(station_id=m["id"], station_name=m.get("name", m["id"]),
                              lat=m.get("lat", np.nan), lon=m.get("lon", np.nan),
                              row=int(m["row"]), col=int(m["col"]), snap_m=0.0, status="ok",
                              index=int(m["index"])) for m in meta])


# ==================================================================== time matching

def overpass_targets(ds: xr.Dataset, cfg: dict) -> pd.DatetimeIndex:
    """The MODIS overpass instant on each product day (UTC)."""
    days = pd.to_datetime(ds["time"].values).normalize()
    hv = cfg["match"]["hour_var"]
    if hv in ds:
        h = np.asarray(ds[hv].values, float)
        fill = np.nanmedian(h) if np.isfinite(h).any() else np.nan
    else:
        h = np.full(len(days), np.nan)
        fill = np.nan
    if not np.isfinite(fill):
        if cfg["match"]["default_hour"] is None:
            raise ValueError(f"the cube has no usable {hv!r}; set match.default_hour")
        fill = float(cfg["match"]["default_hour"])
    h = np.where(np.isfinite(h), h, fill)
    return days + pd.to_timedelta(h, unit="h")


def match_overpass(obs: pd.DataFrame, targets: pd.DatetimeIndex,
                   max_dt_min: float) -> tuple[np.ndarray, np.ndarray]:
    """(value, signed dt in minutes) of the observation nearest each target, NaN beyond the
    tolerance. `obs` is one station's rows, sorted by time."""
    t = obs["time"].to_numpy("datetime64[ns]")
    v = obs["value_C"].to_numpy(float)
    tg = targets.to_numpy("datetime64[ns]")
    val = np.full(tg.size, np.nan)
    dt = np.full(tg.size, np.nan)
    if t.size == 0:
        return val, dt
    i = np.clip(np.searchsorted(t, tg), 1, max(t.size - 1, 1))
    lo = np.clip(i - 1, 0, t.size - 1)
    hi = np.clip(i, 0, t.size - 1)
    d_lo = (tg - t[lo]).astype("timedelta64[s]").astype(float) / 60.0
    d_hi = (t[hi] - tg).astype("timedelta64[s]").astype(float) / 60.0
    pick = np.where(np.abs(d_hi) < np.abs(d_lo), hi, lo)
    d = (t[pick] - tg).astype("timedelta64[s]").astype(float) / 60.0
    ok = np.abs(d) <= float(max_dt_min)
    val[ok], dt[ok] = v[pick[ok]], d[ok]
    return val, dt


def match_daily_mean(obs: pd.DataFrame, days: pd.DatetimeIndex,
                     min_coverage: float) -> tuple[np.ndarray, np.ndarray]:
    """(daily mean, hour coverage) per product day; NaN mean below `min_coverage`."""
    if obs.empty:
        return np.full(len(days), np.nan), np.zeros(len(days))
    d = obs.assign(day=obs["time"].dt.normalize(), hour=obs["time"].dt.hour)
    g = d.groupby("day")
    mean = g["value_C"].mean()
    cov = g["hour"].nunique() / 24.0
    m = mean.reindex(days).to_numpy(float)
    c = cov.reindex(days).fillna(0.0).to_numpy(float)
    return np.where(c >= float(min_coverage), m, np.nan), c


# ==================================================================== sampling

def sample_products(ds: xr.Dataset, placed: pd.DataFrame, products: list[str]) -> dict:
    """name -> (T, n_station) values at the station pixels (degC for SST; flags as-is)."""
    ok = placed[placed["row"] >= 0]
    iy = xr.DataArray(ok["row"].to_numpy(), dims="station")
    ix = xr.DataArray(ok["col"].to_numpy(), dims="station")
    out = {}
    for p in products:
        if p not in ds:
            log.warning("product %s is not in the cube; skipped", p)
            continue
        out[p] = ds[p].isel(y=iy, x=ix).values.astype(float) - 273.15
    a = ds["sst_seasonal_coef"].attrs
    X = design_matrix(pd.to_datetime(ds["time"].values), int(a["n_harmonics"]),
                      float(a["period_days"]))
    coef = ds["sst_seasonal_coef"].isel(y=iy, x=ix).values                # (P, station)
    out["climatology"] = X @ coef - 273.15
    if "sst_filled_observed" in ds:
        out["_observed"] = ds["sst_filled_observed"].isel(y=iy, x=ix).values.astype(bool)
    return out


def build_matchups(ds: xr.Dataset, cfg: dict, insitu: pd.DataFrame | None) -> tuple[
        pd.DataFrame, pd.DataFrame]:
    """(matchups, station table). One row per station x day x match type with an in-situ value."""
    water = np.asarray(ds["landcover_water"].values > 0.5)
    days = pd.to_datetime(ds["time"].values).normalize()
    products = [p for p in cfg["products"] if p in ds]

    if insitu is None:
        if "insitu_stations" not in ds.attrs:
            raise SystemExit("no --insitu file and the cube carries no in-situ channels")
        placed = cube_stations(ds)
        values = {}
        for match, var in (("overpass", "modis_insitu_sst"), ("reference", "insitu_sst")):
            if var in ds:
                iy = xr.DataArray(placed["row"].to_numpy(), dims="station")
                ix = xr.DataArray(placed["col"].to_numpy(), dims="station")
                v = ds[var].isel(y=iy, x=ix).values.astype(float)
                # The cube stores degC for in-situ, as the package writes it.
                values[match] = (v, np.full(v.shape, np.nan))
        log.info("in-situ from cube channels: %d stations; daily_mean unavailable (instants "
                 "only)", len(placed))
    else:
        stations = insitu.groupby("station_id", as_index=False).agg(
            station_name=("station_name", "first"), lat=("lat", "median"),
            lon=("lon", "median"))
        placed = place_stations(stations, ds, water, float(cfg["match"]["max_snap_m"]))
        targets = overpass_targets(ds, cfg)
        ok = placed[placed["row"] >= 0]
        ov = np.full((len(days), len(ok)), np.nan)
        ovdt = np.full_like(ov, np.nan)
        dm = np.full_like(ov, np.nan)
        cov = np.zeros_like(ov)
        for k, (_, s) in enumerate(ok.iterrows()):
            o = insitu[insitu["station_id"] == s["station_id"]]
            ov[:, k], ovdt[:, k] = match_overpass(o, targets, cfg["match"]["max_dt_min"])
            dm[:, k], cov[:, k] = match_daily_mean(o, days, cfg["match"]["min_coverage"])
        values = {"overpass": (ov, ovdt), "daily_mean": (dm, cov)}

    empty = pd.DataFrame(columns=MATCHUP_COLS + products + ["climatology"])
    ok = placed[placed["row"] >= 0].reset_index(drop=True)
    if ok.empty:
        return empty, placed
    samp = sample_products(ds, ok, products)
    constrained = (ds["sst_filled_constrained"].values.astype(bool)
                   if "sst_filled_constrained" in ds else np.ones(len(days), bool))
    status = (ds["smooth_loading_status"].values if "smooth_loading_status" in ds
              else np.full(len(days), -1))
    ftype = (ds["sst_seasonal_fit_type"].values if "sst_seasonal_fit_type" in ds else None)
    rows = []
    for match, (val, aux) in values.items():
        for k, s in ok.iterrows():
            have = np.flatnonzero(np.isfinite(val[:, k]))
            for j in have:
                r = dict(station_id=s["station_id"], station_name=s["station_name"],
                         row=int(s["row"]), col=int(s["col"]), snap_m=float(s["snap_m"]),
                         date=days[j].strftime("%Y-%m-%d"), t=int(j), match=match,
                         insitu=float(val[j, k]),
                         dt_min=float(aux[j, k]) if match != "daily_mean" else np.nan,
                         coverage=float(aux[j, k]) if match == "daily_mean" else np.nan,
                         pixel_observed=bool(samp["_observed"][j, k]) if "_observed" in samp
                         else False,
                         day_constrained=bool(constrained[j]),
                         loading_status=int(status[j]),
                         seasonal_fit=_seasonal_fit_label(ftype, int(s["row"]), int(s["col"])),
                         season=SEASONS[days[j].month])
                for p in products + ["climatology"]:
                    if p in samp:
                        r[p] = float(samp[p][j, k])
                rows.append(r)
    return (pd.DataFrame(rows) if rows else empty), placed


# ==================================================================== metrics

def scores(err: np.ndarray, pred: np.ndarray, obs: np.ndarray,
           err_clim: np.ndarray | None) -> dict:
    """Error statistics of `pred - obs`, with skill over climatology on the same rows."""
    ok = np.isfinite(err)
    e, p, o = err[ok], pred[ok], obs[ok]
    n = int(ok.sum())
    if n == 0:
        return dict(n=0)
    out = dict(n=n, bias=float(e.mean()), mse=float(np.mean(e ** 2)),
               rmse=float(np.sqrt(np.mean(e ** 2))),
               mae=float(np.mean(np.abs(e))), med_abs=float(np.median(np.abs(e))),
               robust_sd=float(1.4826 * np.median(np.abs(e - np.median(e)))),
               r=float(np.corrcoef(p, o)[0, 1]) if n > 2 and p.std() > 0 and o.std() > 0
               else np.nan)
    if err_clim is not None:
        ec = err_clim[ok]
        mse_c = float(np.mean(ec ** 2)) if np.isfinite(ec).all() else np.nan
        out["skill_vs_clim"] = (1.0 - float(np.mean(e ** 2)) / mse_c) if mse_c > 0 else np.nan
    return out


def metrics(mu: pd.DataFrame, products: list[str]) -> pd.DataFrame:
    """Scores per match x product x stratum."""
    if mu.empty:
        return pd.DataFrame()
    strata = [("all", None), ("station", "station_id"), ("pixel_observed", "pixel_observed"),
              ("day_constrained", "day_constrained"), ("season", "season"),
              ("loading_status", "loading_status")]
    rows = []
    for match, g0 in mu.groupby("match"):
        for sname, col in strata:
            groups = [("all", g0)] if col is None else list(g0.groupby(col))
            for key, g in groups:
                obs = g["insitu"].to_numpy(float)
                clim = g["climatology"].to_numpy(float) if "climatology" in g else None
                ec = None if clim is None else clim - obs
                for p in [q for q in products + ["climatology"] if q in g]:
                    pred = g[p].to_numpy(float)
                    s = scores(pred - obs, pred, obs, ec)
                    rows.append(dict(match=match, product=p, stratum=sname, group=str(key),
                                     **s))
    return pd.DataFrame(rows)


# ==================================================================== figures

PRODUCT_COLORS = {"sst_filled": "#2a78d6", "sst_filled_point": "#1baf7a",
                  "sst_filled_day": "#eda100", "sst_smooth": "#eb6834",
                  "climatology": "#898781"}

WATER_FILL = "#4a90d9"      # water: a readable mid blue
STATION_FILL = "#e03131"    # the matched station pixel: red


def station_pixel_map(ds: xr.Dataset, placed: pd.DataFrame, out: Path, dpi: int = 130) -> None:
    """Map of the monitoring stations as highlighted pixels: land grey, water blue, each matched
    station pixel red, labelled by station name. Shared by stage 3 and the standalone script.

    `placed` is the station table from place_stations / cube_stations (station_name, row, col,
    status). Stations that fell outside the grid or off water (row < 0) are listed in the title
    but have no pixel to highlight.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.patches import Patch
    import plotting
    from cube_figures import INK, INK_SECONDARY

    water = plotting.water_mask(ds)
    extent = plotting.extent_km(ds)
    xs, ys = ds["x"].values, ds["y"].values
    km = lambda X, Y: ((X - xs.min()) / 1000.0, (Y - ys.min()) / 1000.0)  # noqa: E731

    on = placed[placed["row"] >= 0]
    station_mask = np.zeros(water.shape, bool)
    for _, s in on.iterrows():
        station_mask[int(s["row"]), int(s["col"])] = True

    fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=dpi, layout="constrained")
    ax.set_facecolor(plotting.NODATA_COLOR)
    plotting.flat(ax, ~water, plotting.LAND_COLOR, extent)              # land grey
    plotting.flat(ax, water & ~station_mask, WATER_FILL, extent)        # water blue
    plotting.flat(ax, station_mask, STATION_FILL, extent)              # station pixels red

    stroke = [pe.withStroke(linewidth=2.0, foreground="white")]
    for _, s in on.iterrows():
        cx, cy = km(xs[int(s["col"])], ys[int(s["row"])])
        ax.plot(cx, cy, "o", color=STATION_FILL, markersize=4, markeredgecolor="white",
                markeredgewidth=0.6, zorder=5)                          # keep the pixel visible
        ax.annotate(str(s["station_name"]), (cx, cy), xytext=(6, 4), textcoords="offset points",
                    fontsize=7, color=INK, zorder=6, path_effects=stroke)

    n_off = int((placed["row"] < 0).sum())
    ax.legend(handles=[Patch(color=plotting.LAND_COLOR, label="land"),
                       Patch(color=WATER_FILL, label="water"),
                       Patch(color=STATION_FILL, label="station pixel")],
              fontsize=7, frameon=False, loc="upper right")
    title = f"monitoring stations as matched pixels ({len(on)} on water"
    title += f", {n_off} off-grid/land)" if n_off else ")"
    ax.set_title(title, fontsize=9, color=INK)
    ax.set_xlabel("km east", fontsize=7, color=INK_SECONDARY)
    ax.set_ylabel("km north", fontsize=7, color=INK_SECONDARY)
    plotting.save(fig, out)


def figures(ds: xr.Dataset, mu: pd.DataFrame, placed: pd.DataFrame, met: pd.DataFrame,
            products: list[str], out_dir: Path, dpi: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotting
    from cube_figures import GRID, INK, INK_SECONDARY

    def style(ax):
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.tick_params(colors=INK_SECONDARY, labelsize=7)

    water = plotting.water_mask(ds)
    extent = plotting.extent_km(ds)
    xs, ys = ds["x"].values, ds["y"].values
    # plotting.extent_km spans [0, (xmax - xmin) km] x [0, (ymax - ymin) km]; map metres onto it.
    km = lambda X, Y: ((X - xs.min()) / 1000.0, (Y - ys.min()) / 1000.0)  # noqa: E731

    # stations map
    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=dpi, layout="constrained")
    ax.set_facecolor(plotting.NODATA_COLOR)
    plotting.flat(ax, ~water, plotting.LAND_COLOR, extent)
    plotting.flat(ax, water, "#d8e6f5", extent)
    for _, s in placed.iterrows():
        if np.isfinite(s.get("x", np.nan)):
            ax.plot(*km(s["x"], s["y"]), "o", color="#eb6834", markersize=6, zorder=4)
        if s["row"] >= 0:
            cx, cy = km(xs[s["col"]], ys[s["row"]])
            ax.plot(cx, cy, "s", markerfacecolor="none", markeredgecolor="#2a78d6",
                    markersize=8, markeredgewidth=1.5, zorder=5)
            ax.annotate(f"{s['station_id']}" + (f"\nsnap {s['snap_m']:.0f} m"
                                                if s["snap_m"] > 0 else ""),
                        (cx, cy), xytext=(6, 6), textcoords="offset points", fontsize=7,
                        color=INK)
    ax.set_title("in-situ stations: circle = reported position, square = matched pixel",
                 fontsize=8, color=INK)
    ax.set_xlabel("km east", fontsize=7)
    ax.set_ylabel("km north", fontsize=7)
    plotting.save(fig, out_dir / "stations_map.png")

    # station pixels highlighted over the land/water mask, labelled by name
    station_pixel_map(ds, placed, out_dir / "station_pixels.png", dpi)

    if mu.empty:
        return
    # time series per station
    t_all = pd.to_datetime(ds["time"].values)
    for sid, g in mu.groupby("station_id"):
        fig, ax = plt.subplots(figsize=(13, 4), dpi=dpi, layout="constrained")
        row, col = int(g["row"].iloc[0]), int(g["col"].iloc[0])
        for p in products:
            if p in ds and p in ("sst_filled", "sst_smooth"):
                ax.plot(t_all, ds[p].values[:, row, col] - 273.15, color=PRODUCT_COLORS[p],
                        linewidth=1.6, label=p)
        if "climatology" in g:
            cl = g.drop_duplicates("t").sort_values("t")
            ax.plot(pd.to_datetime(cl["date"]), cl["climatology"], color=PRODUCT_COLORS[
                "climatology"], linewidth=1.2, linestyle="--", label="climatology")
        for match, mk in (("overpass", "o"), ("daily_mean", "s"), ("reference", "^")):
            h = g[g["match"] == match]
            if len(h):
                ax.plot(pd.to_datetime(h["date"]), h["insitu"], mk, color="#0b0b0b",
                        markersize=3.5 if match != "daily_mean" else 3, alpha=0.8,
                        markerfacecolor="none" if match == "daily_mean" else "#0b0b0b",
                        label=f"in situ ({match})")
        style(ax)
        ax.set_ylabel("temperature [degC]", fontsize=8)
        ax.set_title(f"{sid}  {g['station_name'].iloc[0]}  (pixel row {row}, col {col}, "
                     f"snap {g['snap_m'].iloc[0]:.0f} m)", fontsize=9, color=INK, loc="left")
        ax.legend(fontsize=7, frameon=False, ncol=6, loc="upper left")
        plotting.save(fig, out_dir / f"timeseries_{str(sid).replace('/', '_')}.png")

    # scatter, overpass match preferred
    match = "overpass" if (mu["match"] == "overpass").any() else mu["match"].iloc[0]
    g = mu[mu["match"] == match]
    ps = [p for p in products + ["climatology"] if p in g]
    fig, axes = plt.subplots(1, len(ps), figsize=(3.6 * len(ps), 3.8), dpi=dpi,
                             layout="constrained", squeeze=False)
    lo = np.nanmin(g[["insitu"] + ps].to_numpy(float))
    hi = np.nanmax(g[["insitu"] + ps].to_numpy(float))
    for ax, p in zip(axes[0], ps):
        ax.plot([lo, hi], [lo, hi], color="#898781", linewidth=1, linestyle="--")
        ax.scatter(g["insitu"], g[p], s=10, color=PRODUCT_COLORS.get(p, "#2a78d6"), alpha=0.7,
                   edgecolor="none")
        m = met[(met["match"] == match) & (met["product"] == p) & (met["stratum"] == "all")]
        txt = (f"n={int(m['n'].iloc[0])}  bias {m['bias'].iloc[0]:+.2f}\n"
               f"RMSE {m['rmse'].iloc[0]:.2f}  r {m['r'].iloc[0]:.2f}") if len(m) else ""
        ax.set_title(f"{p}\n{txt}", fontsize=7.5, color=INK)
        style(ax)
        ax.set_xlabel("in situ [degC]", fontsize=7)
    axes[0][0].set_ylabel("product [degC]", fontsize=7)
    fig.suptitle(f"matchups ({match})", fontsize=9, color=INK, ha="left", x=0.01)
    plotting.save(fig, out_dir / "scatter.png")

    # RMSE by gap category, overpass match
    cats = [("pixel_observed", "True", "pixel observed"),
            ("pixel_observed", "False", "pixel gap-filled"),
            ("day_constrained", "False", "day with no data")]
    sub = met[(met["match"] == match)]
    fig, ax = plt.subplots(figsize=(9, 4), dpi=dpi, layout="constrained")
    w = 0.8 / max(len(ps), 1)
    for i, p in enumerate(ps):
        vals, ns = [], []
        for st, grp, _ in cats:
            m = sub[(sub["product"] == p) & (sub["stratum"] == st) & (sub["group"] == grp)]
            vals.append(float(m["rmse"].iloc[0]) if len(m) and m["n"].iloc[0] else np.nan)
            ns.append(int(m["n"].iloc[0]) if len(m) else 0)
        xpos = np.arange(len(cats)) + (i - (len(ps) - 1) / 2) * w
        ax.bar(xpos, vals, w * 0.92, color=PRODUCT_COLORS.get(p, "#2a78d6"), label=p)
        for xp, v, n in zip(xpos, vals, ns):
            if np.isfinite(v):
                ax.annotate(f"{v:.2f}\nn={n}", (xp, v), ha="center", va="bottom", fontsize=6,
                            color=INK_SECONDARY)
    ax.set_xticks(np.arange(len(cats)), [c[2] for c in cats])
    style(ax)
    ax.set_ylabel("RMSE vs in situ [K]", fontsize=8)
    ax.legend(fontsize=7, frameon=False, ncol=len(ps))
    ax.set_title(f"error by gap category ({match} match)", fontsize=9, color=INK)
    plotting.save(fig, out_dir / "error_by_category.png")


# ==================================================================== driver

def validate(cube: Path, insitu_path: Path | None, cfg: dict, out_dir: Path,
             figs: bool = True) -> dict:
    ds = xr.open_zarr(cube)
    insitu = read_insitu(insitu_path, cfg) if insitu_path is not None else None
    mu, placed = build_matchups(ds, cfg, insitu)
    products = [p for p in cfg["products"] if p in ds]
    met = metrics(mu, products)
    out_dir.mkdir(parents=True, exist_ok=True)
    mu.to_csv(out_dir / "matchups.csv", index=False)
    met.to_csv(out_dir / "metrics.csv", index=False)
    placed.to_csv(out_dir / "stations.csv", index=False)
    if mu.empty:
        log.warning("no matchups: no station fell on the cube's water within the time range")
    else:
        head = met[met["stratum"] == "all"][["match", "product", "n", "bias", "rmse", "r",
                                             "skill_vs_clim"]]
        log.info("\n%s", head.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    log.info("wrote %s, %s, %s", out_dir / "matchups.csv", out_dir / "metrics.csv",
             out_dir / "stations.csv")
    if figs:
        try:
            figures(ds, mu, placed, met, products, out_dir, int(cfg["output"]["dpi"]))
        except Exception:
            log.exception("figures failed; the CSVs were written and are intact")
    return dict(matchups=mu, metrics=met, stations=placed)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cube", type=Path, required=True, help="a pipeline output cube")
    p.add_argument("--insitu", type=Path, default=None,
                   help="in-situ netCDF or CSV; default: the cube's own insitu_* channels")
    p.add_argument("--config", type=Path, default=None, help="optional YAML over DEFAULTS")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="default: validation[_tag]/ next to the cube")
    p.add_argument("--tag", default=None)
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(args.config)
    out_dir = args.out_dir or args.cube.parent / (f"validation_{args.tag}" if args.tag
                                                  else "validation")
    validate(args.cube, args.insitu, cfg, out_dir, figs=not args.no_figures)


if __name__ == "__main__":
    main()
