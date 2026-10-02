"""Stage 3: validate stage 2's DINEOF fields against in-situ water temperature.

Matching, metrics and the base figures are validate_insitu's: stations placed on the grid
(land pixels snapped to water), each product day matched at the MODIS overpass and as a daily
mean, scored per product (filled, smooth, climatology) and stratum. Added here:

  mse_by_site.png   MSE per site and for all sites together, per product, labelled with n
  residuals.png     residual (product - in situ) against time per site, residual
                    distributions per product, residual against in-situ temperature, and
                    residual by gap category (pixel observed / gap-filled / day with no data)

    python src/run_region.py --config configs/region.<aoi>.yaml validate [--tag T]
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
import validate_insitu as V
from cube_figures import GRID, INK, INK_MUTED, INK_SECONDARY, SURFACE

log = logging.getLogger("stage3_validate")

PRODUCT_COLORS = V.PRODUCT_COLORS
SHOWN = ("sst_filled", "sst_smooth", "climatology")


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=7)


def _match(mu: pd.DataFrame) -> str:
    return "overpass" if (mu["match"] == "overpass").any() else str(mu["match"].iloc[0])


def mse_by_site_figure(met: pd.DataFrame, match: str, out: Path, dpi: int) -> None:
    m = met[(met["match"] == match)]
    sites = sorted(m.loc[m["stratum"] == "station", "group"].unique())
    groups = [("all sites", m[m["stratum"] == "all"])] + \
             [(s, m[(m["stratum"] == "station") & (m["group"] == s)]) for s in sites]
    prods = [p for p in SHOWN if p in set(m["product"])]
    fig, ax = plt.subplots(figsize=(max(7, 1.9 * len(groups) + 3), 4.6), dpi=dpi,
                           layout="constrained")
    w = 0.8 / max(len(prods), 1)
    for i, p in enumerate(prods):
        xs = np.arange(len(groups)) + (i - (len(prods) - 1) / 2) * w
        vals, ns = [], []
        for _, g in groups:
            r = g[g["product"] == p]
            vals.append(float(r["mse"].iloc[0]) if len(r) and r["n"].iloc[0] else np.nan)
            ns.append(int(r["n"].iloc[0]) if len(r) else 0)
        ax.bar(xs, vals, w * 0.92, color=PRODUCT_COLORS.get(p, "#2a78d6"), label=p)
        for x, v, n in zip(xs, vals, ns):
            if np.isfinite(v):
                ax.annotate(f"{v:.2f}\nn={n}", (x, v), xytext=(0, 2), textcoords="offset points",
                            ha="center", fontsize=6.5, color=INK_SECONDARY)
    ax.set_xticks(np.arange(len(groups)), [g[0] for g in groups], fontsize=8)
    _style(ax)
    ax.set_ylabel("MSE vs in situ [K$^2$]", fontsize=8)
    ax.legend(fontsize=7, frameon=False, ncol=len(prods))
    ax.set_title(f"mean squared error by site ({match} match)", fontsize=10, color=INK)
    plotting.save(fig, out)


def residuals_figure(mu: pd.DataFrame, match: str, out: Path, dpi: int) -> None:
    g = mu[mu["match"] == match].copy()
    g["date"] = pd.to_datetime(g["date"])
    prods = [p for p in SHOWN if p in g]
    sites = sorted(g["station_id"].unique())
    nrow_ts = len(sites)
    fig = plt.figure(figsize=(15, 2.6 * nrow_ts + 4.6), dpi=dpi, layout="constrained")
    gs = fig.add_gridspec(nrow_ts + 1, 3, height_ratios=[1] * nrow_ts + [1.6])
    for i, s in enumerate(sites):
        ax = fig.add_subplot(gs[i, :])
        h = g[g["station_id"] == s].sort_values("date")
        for p in prods:
            ax.plot(h["date"], h[p] - h["insitu"], "o-", markersize=2.5, linewidth=1,
                    color=PRODUCT_COLORS.get(p), label=p, alpha=0.85)
        ax.axhline(0, color=INK_MUTED, linewidth=1)
        _style(ax)
        ax.set_ylabel("residual [K]", fontsize=8)
        ax.set_title(f"{s}: product - in situ", fontsize=8.5, color=INK, loc="left")
        if i == 0:
            ax.legend(fontsize=7, frameon=False, ncol=len(prods), loc="upper right")
    res = {p: (g[p] - g["insitu"]).to_numpy(float) for p in prods}
    lim = float(np.nanpercentile(np.abs(np.concatenate(list(res.values()))), 99)) if len(g) \
        else 1.0
    ax = fig.add_subplot(gs[-1, 0])
    bins = np.linspace(-lim, lim, 41)
    for p in prods:
        v = res[p][np.isfinite(res[p])]
        ax.hist(v, bins=bins, histtype="step", linewidth=2, color=PRODUCT_COLORS.get(p),
                label=f"{p}: mean {v.mean():+.2f}, sd {v.std():.2f}" if v.size else p)
    _style(ax)
    ax.set_xlabel("residual [K]", fontsize=8)
    ax.set_title("residual distribution", fontsize=9, color=INK)
    ax.legend(fontsize=6.5, frameon=False)
    ax = fig.add_subplot(gs[-1, 1])
    for p in prods:
        ax.scatter(g["insitu"], res[p], s=9, color=PRODUCT_COLORS.get(p), alpha=0.6,
                   edgecolor="none", label=p)
    ax.axhline(0, color=INK_MUTED, linewidth=1)
    _style(ax)
    ax.set_xlabel("in situ [degC]", fontsize=8)
    ax.set_ylabel("residual [K]", fontsize=8)
    ax.set_title("residual vs in-situ temperature", fontsize=9, color=INK)
    ax = fig.add_subplot(gs[-1, 2])
    cats = [("pixel observed", g["pixel_observed"].astype(bool)),
            ("gap-filled", ~g["pixel_observed"].astype(bool) & g["day_constrained"].astype(bool)),
            ("no data that day", ~g["day_constrained"].astype(bool))]
    w = 0.8 / max(len(prods), 1)
    for i, p in enumerate(prods):
        data = [res[p][c.to_numpy()] for _, c in cats]
        pos = np.arange(len(cats)) + (i - (len(prods) - 1) / 2) * w
        for x, d in zip(pos, data):
            d = d[np.isfinite(d)]
            if d.size:
                ax.scatter(np.full(d.size, x) + np.random.default_rng(i).uniform(
                    -w * 0.3, w * 0.3, d.size), d, s=6, color=PRODUCT_COLORS.get(p),
                    alpha=0.45, edgecolor="none")
                ax.plot([x - w * 0.4, x + w * 0.4], [np.median(d)] * 2, color=INK, linewidth=2)
    ax.set_xticks(np.arange(len(cats)), [f"{c}\nn={int(m.sum())}" for c, m in cats],
                  fontsize=7)
    ax.axhline(0, color=INK_MUTED, linewidth=1)
    _style(ax)
    ax.set_ylabel("residual [K]", fontsize=8)
    ax.set_title("residual by gap category (bar = median)", fontsize=9, color=INK)
    fig.suptitle(f"residuals between the products and in situ ({match} match)", fontsize=10,
                 color=INK, ha="left", x=0.01)
    plotting.save(fig, out)


def run(rc: dict, *, tag: str | None = None, figures: bool = True) -> dict:
    t0 = time.time()
    dirs = R.stage_dirs(rc, tag)
    cube = dirs["dineof"] / f"{rc['region']['aoi']}_dineof.zarr"
    if not cube.exists():
        raise SystemExit(f"no stage-2 cube at {cube}; run:\n  python src/run_region.py "
                         f"--config {rc['path']}{' --tag ' + tag if tag else ''} dineof")
    insitu = rc["validation"]["insitu"]
    if insitu is None:
        with xr.open_zarr(cube) as ds:
            if "insitu_stations" not in ds.attrs:
                raise SystemExit(
                    "validation.insitu is null and the cube carries no insitu_* channels. "
                    "Point validation.insitu at a netCDF / CSV, e.g. one written by "
                    "scripts/fetch_insitu.py")
    elif not Path(insitu).exists():
        raise SystemExit(f"validation.insitu: no such file {insitu}")
    vcfg = R.validate_vcfg(rc)
    out = dirs["validation"]
    res = V.validate(cube, insitu, vcfg, out, figs=figures)
    mu, met = res["matchups"], res["metrics"]
    if figures and not mu.empty:
        match = _match(mu)
        dpi = int(rc["validation"]["dpi"])
        mse_by_site_figure(met, match, out / "mse_by_site.png", dpi)
        residuals_figure(mu, match, out / "residuals.png", dpi)
    if not mu.empty and "seasonal_fit" in mu.columns:
        per_station = (mu.drop_duplicates("station_id")
                       .sort_values("station_id")[["station_id", "row", "col", "seasonal_fit"]])
        log.info("stage 3 seasonal fit at each in-situ pixel:\n%s",
                 per_station.to_string(index=False))
    if not met.empty:
        allm = met[met["stratum"] == "all"][["match", "product", "n", "mse", "rmse", "bias"]]
        log.info("stage 3 MSE (all sites):\n%s", allm.to_string(
            index=False, float_format=lambda v: f"{v:.3f}"))
    log.info("stage 3 done in %.1f min -> %s", (time.time() - t0) / 60, out)
    return dict(dir=out, **res)
