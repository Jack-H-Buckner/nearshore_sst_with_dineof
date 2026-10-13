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
from cube_figures import GRID, INK, INK_MUTED, INK_SECONDARY, SURFACE
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


def _load_insitu(ds: xr.Dataset, rc: dict, times) -> tuple[pd.DataFrame, dict]:
    """(placed stations with row/col/x/y, {station_id: (T,) daily in-situ series, degC}).

    Uses `validation.insitu` when it points at a file (daily means from the raw observations);
    otherwise falls back to the cube's embedded in-situ channels (`insitu_stations` attr +
    `insitu_sst` / `modis_insitu_sst`), which already hold one value per station pixel per day.
    """
    xs, ys = ds["x"].values, ds["y"].values
    path = rc["validation"]["insitu"]
    if path and Path(path).exists():
        vcfg = R.validate_vcfg(rc)
        df = V.read_insitu(Path(path), vcfg)
        water = np.asarray(ds[rc["region"]["watervar"]].values > 0.5)
        stations = df.groupby("station_id", as_index=False).agg(
            station_name=("station_name", "first"), lat=("lat", "median"), lon=("lon", "median"))
        placed = V.place_stations(stations, ds, water, float(rc["point_filter"]["max_snap_m"]))
        placed = placed[placed["row"] >= 0].reset_index(drop=True)
        days, min_cov = times.normalize(), float(vcfg["match"]["min_coverage"])
        daily = {s["station_id"]: V.match_daily_mean(df[df["station_id"] == s["station_id"]],
                                                     days, min_cov)[0]
                 for _, s in placed.iterrows()}
        return placed, daily
    # Fallback: the in-situ channels carried in the cube itself.
    if "insitu_stations" not in ds.attrs:
        raise SystemExit("validation.insitu is null and the cube carries no in-situ data "
                         "(no `insitu_stations` attr); set validation.insitu to a netCDF / CSV")
    var = next((v for v in ("insitu_sst", "modis_insitu_sst") if v in ds), None)
    if var is None:
        raise SystemExit("validation.insitu is null and the cube has no insitu_sst / "
                         "modis_insitu_sst channel to read")
    placed = V.cube_stations(ds).reset_index(drop=True)
    placed["x"] = xs[placed["col"].to_numpy()]
    placed["y"] = ys[placed["row"].to_numpy()]
    iy = xr.DataArray(placed["row"].to_numpy(), dims="station")
    ix = xr.DataArray(placed["col"].to_numpy(), dims="station")
    vals = ds[var].isel(y=iy, x=ix).values.astype(float)         # (T, n); the cube stores degC
    daily = {sid: vals[:, k] for k, sid in enumerate(placed["station_id"])}
    log.info("in-situ from cube channel %s: %d stations (validation.insitu is null)",
             var, len(placed))
    return placed, daily


def insitu_series(ds: xr.Dataset, ok: pd.DataFrame, rc: dict, times) -> np.ndarray:
    """(T, n_sites) in-situ daily series (degC) for each points-CSV site: the nearest in-situ
    station within point_filter.max_snap_m. NaN where no station is in range."""
    placed, daily = _load_insitu(ds, rc, times)
    max_snap = float(rc["point_filter"]["max_snap_m"])
    I = np.full((len(times), len(ok)), np.nan)
    if placed.empty:
        return I
    px, py = placed["x"].to_numpy(), placed["y"].to_numpy()
    for i in range(len(ok)):
        d = np.hypot(px - ok.loc[i, "x"], py - ok.loc[i, "y"])
        k = int(np.argmin(d))
        if d[k] > max_snap:
            continue
        sid = placed.loc[k, "station_id"]
        I[:, i] = daily[sid]
        log.info("  in-situ: site %s <- station %s (%.0f m, %d days)",
                 ok.loc[i, "station_id"], sid, d[k], int(np.isfinite(daily[sid]).sum()))
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
    ev = pf.get("insitu_eval") or {}
    do_points = bool(pf.get("points"))
    do_eval = bool(ev.get("enabled"))
    if not do_points and not do_eval:
        raise SystemExit("point_filter: set `points` (a CSV of site_id, lat, lon, group) and/or "
                         "insitu_eval.enabled: true")
    if do_points and not Path(pf["points"]).exists():
        raise SystemExit(f"point_filter.points: no such file {pf['points']}")

    out = dirs["pointfilter"]
    (out / "reports").mkdir(parents=True, exist_ok=True)
    result = {"dir": out}
    with xr.open_zarr(cube) as ds:
        times = pd.to_datetime(ds["time"].values)
        crs = ds.attrs.get("crs")
        if do_points:
            result.update(_run_points(ds, rc, pf, out, times, crs, cube, figures))
        if do_eval:
            result["insitu_eval"] = _run_insitu_eval(ds, rc, pf, ev, out, times, figures)
    log.info("stage 4 done in %.1f min -> %s", (time.time() - t0) / 60, out)
    return result


def _run_points(ds, rc: dict, pf: dict, out: Path, times, crs, cube: Path,
                figures: bool) -> dict:
    """The points-CSV workflow: smooth each site's series per group, write series/params/zarr."""
    points = _read_points(Path(pf["points"]))
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
    series = extract_series(ds, ok)
    # In-situ channel: only read when some group actually uses it.
    if any("insitu" in _group_cfg(pf, g)["sources"] for g in groups):
        I = insitu_series(ds, ok, rc, times)
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

    log.info("stage 4 points: %d sites, %d groups -> %s", len(ok), len(fitted), out)
    return dict(n_sites=len(ok), groups=list(fitted), series=series_df)


def _run_insitu_eval(ds, rc: dict, pf: dict, ev: dict, out: Path, times, figures: bool) -> dict:
    """Diagnostic: smooth each in-situ site's DINEOF series (NO in-situ in the fit), fit a POST-HOC
    bias on (insitu - smoothed), and evaluate it out-of-sample by leave-one-site-out (LOSO).

    Two steps, deliberately post-hoc: (1) the state-space smoother runs on DINEOF sources only and
    gives a point estimate at each in-situ pixel; (2) a bias (constant or seasonal day-of-year) is
    fit by OLS on (insitu - smoothed) pooled across sites. LOSO refits the bias with each site left
    out and scores the correction at that held-out site -- the filter's accuracy after a bias
    adjustment that did not see the site. Reuses stage 5's offset / metric / LOSO helpers.
    """
    import stage5_calibrate as C5                   # OLS offset + metrics + LOSO on a matchup frame
    # Sites and daily in-situ come from validation.insitu, or the cube's own channels when it is null.
    ok, daily = _load_insitu(ds, rc, times)
    ok = ok.reset_index(drop=True)
    if ok.empty:
        raise SystemExit("insitu_eval: no in-situ stations placed on water")

    # (1) smoother on DINEOF only -- one group, no in-situ channel.
    series = extract_series(ds, ok)
    gcfg = dict(model=ev["model"], mode=ev["mode"], sources=list(ev["sources"]),
                n_harmonics=int(ev.get("n_harmonics", 0)), share=list(ev.get("share") or []),
                period_days=float(pf["period_days"]))
    log.info("insitu_eval: smoothing %d in-situ site(s), model %s, sources %s",
             len(ok), gcfg["model"], gcfg["sources"])
    fit = SS.fit_group(times, series, gcfg)

    # matchup frame: the smoothed point estimate vs the daily in-situ value, per site x day.
    days = times.normalize()
    rows = []
    for k in range(len(ok)):
        sm = fit["smoothed"][k][0]
        dm = daily[ok.loc[k, "station_id"]]
        for t in np.flatnonzero(np.isfinite(dm)):
            rows.append(dict(station_id=ok.loc[k, "station_id"], date=days[t].strftime("%Y-%m-%d"),
                             t=int(t), insitu=float(dm[t]), smoothed=float(sm[t])))
    mu = pd.DataFrame(rows)
    if mu.empty:
        raise SystemExit("insitu_eval: no in-situ matchups against the smoothed estimate")
    n_sites = int(mu["station_id"].nunique())

    # (2) post-hoc bias on (insitu - smoothed); form chosen by stage 5's selection.
    H, P = int(ev["bias_n_harmonics"]), float(pf["period_days"])
    cal = dict(n_harmonics=H, period_days=P, min_gain=float(ev.get("min_gain", 0.0)),
               selection=ev.get("selection", "aic"),
               force_form=(None if ev["bias_form"] == "auto" else ev["bias_form"]))
    sel = C5._select(mu, "smoothed", cal, n_sites)
    coef, form = sel["fit"]["coef"], sel["form"]
    bias_info = dict(form=form, how=sel["how"], n=int(sel["fit"]["n"]),
                     coef={nm: float(c) for nm, c in zip(C5._coef_names(form, H), coef)},
                     aic_constant=C5._finite(sel["aic_constant"]),
                     aic_seasonal=C5._finite(sel["aic_seasonal"]))
    if "loso_cv_constant" in sel:
        bias_info["loso_cv_constant"] = C5._finite(sel["loso_cv_constant"])
        bias_info["loso_cv_seasonal"] = C5._finite(sel["loso_cv_seasonal"])

    met = pd.DataFrame(C5._metrics(mu, "smoothed", coef, form, H, P))
    loso = C5._loso_table(mu, "smoothed", coef, form, H, P) if n_sites >= 2 else pd.DataFrame()
    loso_mse = C5._loso_cv_error(mu, "smoothed", form, H, P) if n_sites >= 2 else np.nan
    loso_rmse = float(loso_mse ** 0.5) if np.isfinite(loso_mse) else None

    (out / "reports" / "insitu_eval_bias.json").write_text(
        _json(dict(bias_info, selection=cal["selection"], loso_rmse=loso_rmse)))
    met.to_csv(out / "reports" / "insitu_eval_metrics.csv", index=False)
    if not loso.empty:
        loso.to_csv(out / "reports" / "insitu_eval_loso.csv", index=False)
    elif n_sites < 2:
        log.info("insitu_eval: only %d site(s); leave-one-site-out diagnostics skipped", n_sites)

    if (met["site"] == "all").any():
        a = met[met["site"] == "all"].iloc[0]
        log.info("insitu_eval (%s bias, %s): RMSE %.3f -> %.3f K, bias %+.3f -> %+.3f K; "
                 "LOSO RMSE %s", form, sel["how"], a["rmse_before"], a["rmse_after"],
                 a["bias_before"], a["bias_after"],
                 f"{loso_rmse:.3f} K" if loso_rmse is not None else "n/a (need >= 2 sites)")

    if figures:
        try:
            dpi = int(pf["dpi"])
            # The bias-corrected estimate and residuals before/after, for the validation plots.
            mu = mu.assign(corrected=mu["smoothed"] + C5._predict(coef, mu["date"], form, H, P))
            mu = mu.assign(resid_before=mu["smoothed"] - mu["insitu"],
                           resid_after=mu["corrected"] - mu["insitu"])
            curve = C5._predict(coef, times, form, H, P)
            C5._offset_figure(mu, "smoothed", bias_info, curve, times, H, P,
                              out / "figures" / "insitu_eval_offset.png", dpi)
            if not loso.empty:
                C5._loso_figure(loso, bias_info, "smoothed",
                                out / "figures" / "insitu_eval_loso.png", dpi)
            # Validation-style plots of the smoothed point estimate, before vs after the bias.
            _eval_mse_by_site(met, out / "figures" / "insitu_eval_mse_by_site.png", dpi)
            _eval_scatter(mu, out / "figures" / "insitu_eval_scatter.png", dpi)
            _eval_residuals(mu, out / "figures" / "insitu_eval_residuals.png", dpi)
        except Exception as e:                        # a plotting bug must not lose the diagnostics
            log.warning("insitu_eval figures failed: %s", e)

    return dict(n_sites=n_sites, bias=bias_info, metrics=met, loso=loso, loso_rmse=loso_rmse)


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


# --------------------------------------------------------------------------- insitu_eval figures

_BEFORE, _AFTER = "#adb5bd", "#1864ab"      # before / after the post-hoc bias


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=7)


def _eval_mse_by_site(met: pd.DataFrame, path: Path, dpi: int) -> None:
    """RMSE of the smoothed estimate vs in-situ, per site and for all sites, before vs after the
    post-hoc bias (mirrors stage 3's mse_by_site)."""
    sites = [s for s in met["site"] if s != "all"]
    order = (["all"] + sorted(sites)) if (met["site"] == "all").any() else sorted(sites)
    m = met.set_index("site").reindex(order)
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(order) + 2), 4.0), dpi=dpi,
                           layout="constrained")
    ax.bar(x - 0.2, m["rmse_before"].to_numpy(float), 0.4, color=_BEFORE, label="before bias")
    ax.bar(x + 0.2, m["rmse_after"].to_numpy(float), 0.4, color=_AFTER, label="after bias")
    for xi, n in zip(x, m["n"].to_numpy()):
        ax.annotate(f"n={int(n)}", (xi, 0), xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.5, color=INK_SECONDARY)
    ax.set_xticks(x, order, fontsize=8, rotation=0)
    _style(ax)
    ax.set_ylabel("RMSE vs in situ [K]", fontsize=8)
    ax.set_title("smoothed point estimate: RMSE before vs after the post-hoc bias", fontsize=10,
                 color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False)
    plotting.save(fig, path, dpi=dpi)


def _eval_scatter(mu: pd.DataFrame, path: Path, dpi: int) -> None:
    """Smoothed (before) and bias-corrected (after) estimate vs in-situ, 1:1 line."""
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2), dpi=dpi, layout="constrained", sharex=True,
                             sharey=True)
    lo = float(np.nanmin([mu["insitu"].min(), mu["smoothed"].min(), mu["corrected"].min()]))
    hi = float(np.nanmax([mu["insitu"].max(), mu["smoothed"].max(), mu["corrected"].max()]))
    for ax, col, title, resid in ((axes[0], "smoothed", "before bias", "resid_before"),
                                  (axes[1], "corrected", "after bias", "resid_after")):
        for s, g in mu.groupby("station_id"):
            ax.scatter(g["insitu"], g[col], s=12, alpha=0.6, edgecolor="none", label=str(s))
        ax.plot([lo, hi], [lo, hi], color=INK_MUTED, lw=1)
        r = mu[resid].to_numpy(float)
        _style(ax)
        ax.set_xlabel("in situ [degC]", fontsize=8)
        ax.set_title(f"{title}: bias {np.mean(r):+.2f}, RMSE {np.sqrt(np.mean(r**2)):.2f} K",
                     fontsize=9, color=INK, loc="left")
    axes[0].set_ylabel("smoothed estimate [degC]", fontsize=8)
    axes[1].legend(fontsize=6.5, frameon=False, ncol=2)
    plotting.save(fig, path, dpi=dpi)


def _eval_residuals(mu: pd.DataFrame, path: Path, dpi: int) -> None:
    """Residual (estimate - in situ) against time per site, and against in-situ temperature,
    before vs after the bias (mirrors stage 3's residuals)."""
    t = pd.to_datetime(mu["date"])
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10.5, 3.8), dpi=dpi, layout="constrained")
    for s, g in mu.groupby("station_id"):
        gi = mu.index.get_indexer(g.index)
        a0.plot(t.iloc[gi], g["resid_after"], ".", ms=3.5, alpha=0.7, label=str(s))
    a0.axhline(0, color=INK_MUTED, lw=1)
    _style(a0)
    a0.set_ylabel("corrected - in situ [K]", fontsize=8)
    a0.set_title("residual over time (after bias)", fontsize=9, color=INK, loc="left")
    a0.legend(fontsize=6.5, frameon=False, ncol=2)
    a1.scatter(mu["insitu"], mu["resid_before"], s=12, alpha=0.5, edgecolor="none",
               color=_BEFORE, label="before")
    a1.scatter(mu["insitu"], mu["resid_after"], s=12, alpha=0.5, edgecolor="none",
               color=_AFTER, label="after")
    a1.axhline(0, color=INK_MUTED, lw=1)
    _style(a1)
    a1.set_xlabel("in situ [degC]", fontsize=8)
    a1.set_ylabel("estimate - in situ [K]", fontsize=8)
    a1.set_title("residual vs in-situ temperature", fontsize=9, color=INK, loc="left")
    a1.legend(fontsize=7, frameon=False)
    plotting.save(fig, path, dpi=dpi)
