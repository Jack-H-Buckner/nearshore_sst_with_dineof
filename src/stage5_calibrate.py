"""Stage 5: calibrate stage 2's DINEOF fields against in-situ water temperature.

Stage 3 measures the bias of each DINEOF product (sst_filled, sst_filled_point, sst_filled_day,
sst_smooth, ...) against in-situ observations; this stage estimates and removes it. For each
product it fits an additive offset `insitu - product` on the in-situ matchups -- either a single
constant or a seasonal (day-of-year harmonic) curve -- selects the form per product by an
information criterion or by leave-one-site-out (LOSO) cross-validation, and applies the chosen
offset uniformly across space to produce a calibrated field.

The offset form is picked per product by `calibration.selection`:
  aic       pick the form with the lower AIC; keep the seasonal form only when it beats the
            constant form by at least `calibration.min_gain` AIC units.
  loso_cv   pick the form with the lower leave-one-site-out prediction error, weighted by the
            number of observations at each held-out site. Needs >= 2 sites; with fewer it falls
            back to aic.
`calibration.force_form` skips the selection.

When >= 2 in-situ sites are present, the chosen form is refit with each site left out, so the
sensitivity of the offset to the sites included is reported.

    python src/run_region.py --config configs/region.<aoi>.yaml calibrate [--tag T]

Outputs under <out_dir>/<aoi>[_tag]/stage5_calibration/:
  reports/offsets.json           per product: chosen form, coefficients, n, AIC of each form,
                                 selection criterion, loso-fallback flag
  reports/calibration_metrics.csv  bias / rmse vs in-situ before and after calibration, per product
  reports/calibration_loso.csv   per product x left-out site: refit coefficients and the offset
                                 and residual at the held-out site (only when >= 2 sites)
  <aoi>_calibrated.zarr          dims (time, y, x): <product>_cal (K) and <product>_offset (degC)
  figures/offset_<product>.png   fitted offset curve over the matchup residuals by day of year
  figures/loso_<product>.png     offset sensitivity across the LOSO refits (only when >= 2 sites)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import iterative_filter  # noqa: F401  (bridges seasonal_smoothing)
import plotting
import region as R
import validate_insitu as V
from cube_figures import GRID, INK, INK_MUTED, INK_SECONDARY, SURFACE
from seasonal_smoothing import design_matrix, term_names
from nearshore_sst import datacube, provenance, store
from nearshore_sst.datacube import CompressionSpec

log = logging.getLogger("stage5_calibrate")

CHUNKS = {"time": 64, "y": 128, "x": 128}


# ==================================================================== offset model

def _design(times, form: str, n_harmonics: int, period_days: float) -> np.ndarray:
    """Day-of-year design matrix for an offset form. constant -> (T, 1); seasonal -> (T, 1+2H)."""
    if form == "constant":
        return np.ones((len(pd.DatetimeIndex(times)), 1))
    return design_matrix(times, n_harmonics, period_days)


def _residual(mu: pd.DataFrame, product: str) -> np.ndarray:
    """The per-matchup offset to estimate: insitu - product (degC)."""
    return mu["insitu"].to_numpy(float) - mu[product].to_numpy(float)


def _fit_offset(mu: pd.DataFrame, product: str, form: str, n_harmonics: int,
                period_days: float) -> dict:
    """OLS fit of `insitu - product` on the day-of-year design. -> coef, rss, n, k."""
    y = _residual(mu, product)
    X = _design(mu["date"], form, n_harmonics, period_days)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    Xo, yo = X[ok], y[ok]
    n, k = int(ok.sum()), X.shape[1]
    if n < k:                                         # too few points to fit this form
        return dict(coef=np.full(k, np.nan), rss=np.nan, n=n, k=k)
    coef, *_ = np.linalg.lstsq(Xo, yo, rcond=None)
    resid = yo - Xo @ coef
    return dict(coef=coef, rss=float(resid @ resid), n=n, k=k)


def _predict(coef: np.ndarray, times, form: str, n_harmonics: int,
             period_days: float) -> np.ndarray:
    """The fitted offset (degC) at each time."""
    return _design(times, form, n_harmonics, period_days) @ coef


def _aic(rss: float, n: int, k: int) -> float:
    """Gaussian-OLS AIC: n*ln(RSS/n) + 2k. inf when it cannot be formed."""
    if not np.isfinite(rss) or n <= 0 or rss <= 0:
        return np.inf
    return n * np.log(rss / n) + 2 * k


def _loso_cv_error(mu: pd.DataFrame, product: str, form: str, n_harmonics: int,
                   period_days: float) -> float:
    """Leave-one-site-out prediction MSE, weighted by the held-out site's obs count (== pooled
    squared error / total held-out obs). inf when it cannot be formed."""
    sites = list(mu["station_id"].unique())
    sq, nobs = 0.0, 0
    for s in sites:
        train, test = mu[mu["station_id"] != s], mu[mu["station_id"] == s]
        fit = _fit_offset(train, product, form, n_harmonics, period_days)
        if not np.isfinite(fit["coef"]).all():
            return np.inf
        resid = _residual(test, product) - _predict(fit["coef"], test["date"], form,
                                                     n_harmonics, period_days)
        resid = resid[np.isfinite(resid)]
        sq += float(resid @ resid)
        nobs += resid.size
    return sq / nobs if nobs else np.inf


def _select(mu: pd.DataFrame, product: str, cal: dict, n_sites: int) -> dict:
    """Fit both forms and choose one. -> chosen form, its fit, per-form AIC, how it was chosen."""
    H, P = int(cal["n_harmonics"]), float(cal["period_days"])
    fits = {f: _fit_offset(mu, product, f, H, P) for f in R.CAL_FORMS}
    aic = {f: _aic(fits[f]["rss"], fits[f]["n"], fits[f]["k"]) for f in R.CAL_FORMS}
    info = dict(aic_constant=aic["constant"], aic_seasonal=aic["seasonal"])

    if cal["force_form"] is not None:
        chosen, how = cal["force_form"], "forced"
    elif cal["selection"] == "loso_cv" and n_sites >= 2:
        cv = {f: _loso_cv_error(mu, product, f, H, P) for f in R.CAL_FORMS}
        info["loso_cv_constant"], info["loso_cv_seasonal"] = cv["constant"], cv["seasonal"]
        chosen = min(R.CAL_FORMS, key=lambda f: cv[f])
        how = "loso_cv"
    else:
        if cal["selection"] == "loso_cv":
            log.warning("calibration.selection=loso_cv needs >= 2 sites (have %d); "
                        "falling back to aic for %s", n_sites, product)
        # seasonal only when it beats the constant form by at least min_gain AIC units
        chosen = ("seasonal" if aic["seasonal"] <= aic["constant"] - float(cal["min_gain"])
                  else "constant")
        how = "aic_fallback" if cal["selection"] == "loso_cv" else "aic"
    return dict(form=chosen, fit=fits[chosen], how=how, **info)


# ==================================================================== run

def run(rc: dict, *, tag: str | None = None, figures: bool = True) -> dict:
    t0 = time.time()
    cal = rc["calibration"]
    dirs = R.stage_dirs(rc, tag)
    cube = dirs["dineof"] / f"{rc['region']['aoi']}_dineof.zarr"
    if not cube.exists():
        raise SystemExit(f"no stage-2 cube at {cube}; run:\n  python src/run_region.py "
                         f"--config {rc['path']}{' --tag ' + tag if tag else ''} dineof")
    insitu = rc["validation"]["insitu"]
    if insitu is None:
        raise SystemExit("calibration needs validation.insitu (a netCDF / CSV); the cube's own "
                         "insitu_* channels carry overpass instants only, not daily means")
    if not Path(insitu).exists():
        raise SystemExit(f"validation.insitu: no such file {insitu}")

    out = dirs["calibration"]
    (out / "reports").mkdir(parents=True, exist_ok=True)
    vcfg = R.validate_vcfg(rc)
    insitu_df = V.read_insitu(Path(insitu), vcfg)

    H, P = int(cal["n_harmonics"]), float(cal["period_days"])
    with xr.open_zarr(cube) as ds:
        mu_all, _ = V.build_matchups(ds, vcfg, insitu_df)
        times = pd.to_datetime(ds["time"].values)
        crs = ds.attrs.get("crs")
        mu = mu_all[mu_all["match"] == cal["match"]].reset_index(drop=True)
        if mu.empty:
            raise SystemExit(f"no '{cal['match']}' matchups between the cube and {insitu}; "
                             f"check validation.insitu and calibration.match")
        products = [p for p in cal["products"] if p in mu.columns and p in ds]
        if not products:
            raise SystemExit(f"none of calibration.products {cal['products']} are both in the "
                             f"cube and matched in situ")
        n_sites = int(mu["station_id"].nunique())

        offsets, metric_rows, loso_frames, cal_fields, offset_curves = {}, [], [], {}, {}
        for p in products:
            sel = _select(mu, p, cal, n_sites)
            coef = sel["fit"]["coef"]
            offsets[p] = dict(form=sel["form"], how=sel["how"], n=sel["fit"]["n"],
                              coef={n: float(c) for n, c in zip(_coef_names(sel["form"], H),
                                                                coef)},
                              aic_constant=_finite(sel["aic_constant"]),
                              aic_seasonal=_finite(sel["aic_seasonal"]),
                              selection=cal["selection"])
            if "loso_cv_constant" in sel:
                offsets[p]["loso_cv_constant"] = _finite(sel["loso_cv_constant"])
                offsets[p]["loso_cv_seasonal"] = _finite(sel["loso_cv_seasonal"])
            log.info("%-18s offset: %s (%s), n=%d, mean %+0.3f K", p, sel["form"], sel["how"],
                     sel["fit"]["n"], float(coef[0]))

            metric_rows += _metrics(mu, p, coef, sel["form"], H, P)
            curve = _predict(coef, times, sel["form"], H, P)              # (T,) degC
            offset_curves[p] = curve
            cal_fields[p] = ds[p].values.astype("float32") + curve[:, None, None].astype("float32")

            if n_sites >= 2:
                loso_frames.append(_loso_table(mu, p, coef, sel["form"], H, P))

        yx = {"y": ds["y"].values, "x": ds["x"].values}

    # ---- reports
    (out / "reports" / "offsets.json").write_text(json.dumps(offsets, indent=2, sort_keys=True))
    met = pd.DataFrame(metric_rows)
    met.to_csv(out / "reports" / "calibration_metrics.csv", index=False)
    loso = pd.concat(loso_frames, ignore_index=True) if loso_frames else pd.DataFrame()
    if not loso.empty:
        loso.to_csv(out / "reports" / "calibration_loso.csv", index=False)
    elif n_sites < 2:
        log.info("only %d in-situ site(s); leave-one-site-out diagnostics skipped", n_sites)

    # ---- calibrated cube
    dest = out / f"{rc['region']['aoi']}_calibrated.zarr"
    _write_calibrated(dest, times, yx, cal_fields, offset_curves, offsets, rc, cube, crs)

    # ---- figures
    if figures:
        try:
            dpi = int(cal["dpi"])
            for p in products:
                _offset_figure(mu, p, offsets[p], offset_curves[p], times, H, P,
                               out / "figures" / f"offset_{p}.png", dpi)
                if not loso.empty:
                    _loso_figure(loso[loso["product"] == p], offsets[p], p,
                                 out / "figures" / f"loso_{p}.png", dpi)
        except Exception as e:                        # a plotting bug must not lose the outputs
            log.warning("stage 5 figures failed: %s", e)

    log.info("stage 5 MSE/bias (all sites):\n%s",
             met[["product", "bias_before", "bias_after", "rmse_before", "rmse_after"]]
             .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    log.info("stage 5 done in %.1f min -> %s (%d products, %d sites)",
             (time.time() - t0) / 60, out, len(products), n_sites)
    return dict(dir=out, offsets=offsets, metrics=met, loso=loso, n_sites=n_sites)


# ==================================================================== helpers

def _coef_names(form: str, n_harmonics: int) -> list[str]:
    return ["mean"] if form == "constant" else term_names(n_harmonics)


def _finite(v: float):
    return float(v) if np.isfinite(v) else None


def _metrics(mu: pd.DataFrame, product: str, coef: np.ndarray, form: str, H: int,
             P: float) -> list[dict]:
    """bias / rmse of (product - insitu) before and after the offset, overall and per site."""
    off = _predict(coef, mu["date"], form, H, P)
    rows = []
    for key, g in [("all", mu)] + list(mu.groupby("station_id")):
        m = g.index
        obs = g["insitu"].to_numpy(float)
        before = V.scores(g[product].to_numpy(float) - obs, g[product].to_numpy(float), obs, None)
        after_pred = g[product].to_numpy(float) + off[mu.index.get_indexer(m)]
        after = V.scores(after_pred - obs, after_pred, obs, None)
        rows.append(dict(product=product, site=str(key), n=before.get("n", 0),
                         bias_before=before.get("bias", np.nan), bias_after=after.get("bias",
                         np.nan), rmse_before=before.get("rmse", np.nan),
                         rmse_after=after.get("rmse", np.nan)))
    return rows


def _loso_table(mu: pd.DataFrame, product: str, full_coef: np.ndarray, form: str, H: int,
                P: float) -> pd.DataFrame:
    """Refit the chosen form with each site left out; report the offset and the residual at the
    held-out site, so the offset's sensitivity to the sites included is visible."""
    rows = []
    for s in mu["station_id"].unique():
        train, test = mu[mu["station_id"] != s], mu[mu["station_id"] == s]
        fit = _fit_offset(train, product, form, H, P)
        pred = _predict(fit["coef"], test["date"], form, H, P)
        actual = _residual(test, product)
        resid = actual - pred
        resid = resid[np.isfinite(resid)]
        rows.append(dict(
            product=product, left_out=str(s), n_train=fit["n"], n_held=int(np.isfinite(actual).sum()),
            mean_offset=float(fit["coef"][0]), full_mean_offset=float(full_coef[0]),
            held_pred_mean=float(np.nanmean(pred)), held_actual_mean=float(np.nanmean(actual)),
            held_rmse=float(np.sqrt(np.mean(resid ** 2))) if resid.size else np.nan,
            coef=";".join(f"{n}={c:.4f}" for n, c in zip(_coef_names(form, H), fit["coef"]))))
    return pd.DataFrame(rows)


def _write_calibrated(dest: Path, times, yx: dict, cal_fields: dict, offset_curves: dict,
                      offsets: dict, rc: dict, cube: Path, crs) -> None:
    """(time, y, x) calibrated fields + their (time,) offset curves, atomic with provenance."""
    data = {}
    for p, field in cal_fields.items():
        data[f"{p}_cal"] = (("time", "y", "x"), field,
                            {"units": "K", "long_name": f"{p} calibrated against in situ",
                             "offset_form": offsets[p]["form"]})
        data[f"{p}_offset"] = (("time",), offset_curves[p].astype("float32"),
                               {"units": "degC", "long_name": f"additive offset applied to {p}"})
    ds = xr.Dataset(data, coords={"time": times.values, "y": yx["y"], "x": yx["x"]})
    if crs is not None:
        ds.attrs["crs"] = crs
    ds.attrs.update(calibration_config=json.dumps(rc["calibration"], default=str, sort_keys=True),
                    calibration_offsets=json.dumps(offsets, sort_keys=True), source_cube=str(cube),
                    created_at=provenance.now_utc(), package_version=provenance.package_version(),
                    code_version=provenance.code_version())
    enc = datacube.build_encoding(ds, CompressionSpec(), CHUNKS)
    store.sweep_scratch(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with store.atomic(dest) as tmp:
        datacube.write_zarr(ds, tmp, enc)
    log.info("wrote %s", dest)


# ==================================================================== figures

def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=7)


def _offset_figure(mu: pd.DataFrame, product: str, info: dict, curve: np.ndarray, times, H: int,
                   P: float, path: Path, dpi: int) -> None:
    """Matchup residuals (insitu - product) by day of year, with the fitted offset curve."""
    doy = pd.DatetimeIndex(mu["date"]).dayofyear
    resid = _residual(mu, product)
    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=dpi, layout="constrained")
    for s, g in mu.groupby("station_id"):
        gi = mu.index.get_indexer(g.index)
        ax.scatter(doy[gi], resid[gi], s=12, alpha=0.6, edgecolor="none", label=str(s))
    order = np.argsort(pd.DatetimeIndex(times).dayofyear.to_numpy())
    cd = pd.DatetimeIndex(times).dayofyear.to_numpy()[order]
    ax.plot(cd, curve[order], color=INK, lw=1.6, label=f"{info['form']} offset")
    ax.axhline(0, color=INK_MUTED, lw=1)
    _style(ax)
    ax.set_xlabel("day of year", fontsize=8)
    ax.set_ylabel("in situ - product [K]", fontsize=8)
    ax.set_title(f"{product}: fitted offset ({info['form']}, {info['how']}, n={info['n']})",
                 fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=6.5, frameon=False, ncol=4)
    plotting.save(fig, path, dpi=dpi)


def _loso_figure(loso: pd.DataFrame, info: dict, product: str, path: Path, dpi: int) -> None:
    """Mean offset from the full fit vs each leave-one-site-out refit."""
    color = V.PRODUCT_COLORS.get(product, "#2a78d6")
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(loso) + 2), 3.6), dpi=dpi,
                           layout="constrained")
    xs = np.arange(len(loso))
    ax.bar(xs, loso["mean_offset"].to_numpy(float), 0.6, color=color,
           label="refit (site left out)")
    ax.axhline(float(info["coef"].get("mean", loso["full_mean_offset"].iloc[0])), color=INK,
               lw=1.4, label="all sites")
    for x, v, n in zip(xs, loso["mean_offset"], loso["n_held"]):
        ax.annotate(f"{v:+.3f}\nn={int(n)}", (x, v), xytext=(0, 2), textcoords="offset points",
                    ha="center", fontsize=6.5, color=INK_SECONDARY)
    ax.set_xticks(xs, loso["left_out"], fontsize=8)
    _style(ax)
    ax.set_ylabel("mean offset [K]", fontsize=8)
    ax.set_title(f"{product}: offset sensitivity to sites (leave one out)", fontsize=10,
                 color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False)
    plotting.save(fig, path, dpi=dpi)
