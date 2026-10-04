"""Figures for the end-to-end pipeline, drawn from its output cube and reports alone.

  eofs.png, eofs_point.png
                     one row per mode: the spatial EOF on the map, and its loadings against
                     time -- full-data sigma*V and, for the smooth field, MODIS-only. Ticks mark
                     MODIS days, shading the days no MODIS reached (smooth = climatology).
  seasonal.png       the per-pixel seasonal climatology: mean and robust-scale maps, the
                     amplitude and phase of each harmonic, the fit method (full / mean-only /
                     reference) as a categorical map, and the reconstructed annual cycle.
  seasonal_gmrf.png  (seasonal.method=gmrf only) raw vs GMRF-smoothed vs difference, for the
                     mean and each harmonic amplitude -- the shrinkage / gap-fill map.
  cv_curves.png      the coarse CV search: point- and day-holdout RMSE against k, per T_c.
  cloud_filter_<id>.png
                     per pixel: share of observations removed and verdict flips across the
                     loop; per scene: the offset against the smooth field, with the band.
  offsets_<id>.png   the offset fit on MODIS-footprint pairs: within-scene anomalies of the
                     sensor's footprint median against MODIS (outliers marked), with the OLS,
                     RMA and 1:1 lines; and the per-scene offset against overpass hour.
  flag_changes.png   pixels newly flagged / restored at each loop iteration.
  fields_<n>.png     per day: composite input | filled | smooth | filled - smooth.

Re-run without the pipeline:

    python src/pipeline_figures.py --cube data/pipeline/admiralty_inlet_pipeline.zarr
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import iterative_filter  # noqa: F401  (bridges seasonal_smoothing for the imports below)
import iterative_diagnostics as ID
import plotting
from cube_figures import GRID, INK, INK_MUTED, INK_SECONDARY, SURFACE
from seasonal_smoothing import (amplitude_phase, design_matrix,  # noqa: E402
                                FIT_FULL, FIT_MEAN_ONLY, FIT_REFERENCE)

log = logging.getLogger("pipeline_figures")

MODIS_LINE = "#2a78d6"
FULL_LINE = INK_MUTED

# fit-method colours: full fit, mean-only (reference shape), reference wholesale
FIT_COLORS = {FIT_FULL: "#1baf7a", FIT_MEAN_ONLY: "#f0a202", FIT_REFERENCE: "#c92a2a"}
FIT_LABELS = {FIT_FULL: "full", FIT_MEAN_ONLY: "mean-only", FIT_REFERENCE: "reference"}


def _style(ax) -> None:
    ID._style(ax)


def eof_figure(ds: xr.Dataset, suffix: str, out: Path, dpi: int) -> None:
    """Spatial modes beside their loadings, one row per mode."""
    U = ds[f"eof_U{suffix}"].values
    sigma = ds[f"eof_sigma{suffix}"].values
    full = ds[f"loadings_full{suffix}"].values
    modis = ds["loadings_modis"].values if (suffix == "" and "loadings_modis" in ds) else None
    status = ds["smooth_loading_status"].values if "smooth_loading_status" in ds else None
    mpx = ds["smooth_modis_px"].values if "smooth_modis_px" in ds else None
    water = plotting.water_mask(ds)
    extent = plotting.extent_km(ds)
    t = pd.to_datetime(ds["time"].values)
    k = U.shape[0]
    var = sigma ** 2 / max(float((sigma ** 2).sum()), 1e-300)

    fig = plt.figure(figsize=(15, 3.1 * k + 0.8), dpi=dpi, layout="constrained")
    gs = fig.add_gridspec(k, 2, width_ratios=[1, 3.2])
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(alpha=0.0)
    for i in range(k):
        ax = fig.add_subplot(gs[i, 0])
        lim = float(np.nanpercentile(np.abs(U[i][water]), 99)) if np.isfinite(U[i][water]).any() \
            else 1.0
        im = plotting.panel(ax, U[i], water, vmin=-lim, vmax=lim, extent=extent, cmap=cmap)
        ax.set_title(f"EOF {i + 1}: {var[i]:.0%} of retained variance", fontsize=8, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8).ax.tick_params(labelsize=6)

        ax = fig.add_subplot(gs[i, 1])
        if status is not None:
            for j in np.flatnonzero(status == 2):
                ax.axvspan(t[j] - pd.Timedelta(hours=12), t[j] + pd.Timedelta(hours=12),
                           color=GRID, alpha=0.9, linewidth=0, zorder=0)
        ax.plot(t, full[i], color=FULL_LINE, linewidth=2, label="full data (sigma * V)")
        if modis is not None:
            ax.plot(t, modis[i], color=MODIS_LINE, linewidth=2,
                    label="MODIS only (smooth field)")
        _style(ax)
        if mpx is not None:
            md = t[mpx > 0]
            ax.plot(md, np.full(md.size, ax.get_ylim()[0]), "|", color=INK_SECONDARY,
                    markersize=6, label="MODIS day")
        ax.set_ylabel(f"loading, mode {i + 1}", fontsize=8, color=INK_SECONDARY)
        if i == 0:
            ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper right")
    a = ds[f"eof_U{suffix}"].attrs
    title = (f"EOFs of the {'point-CV' if suffix else 'smooth-field'} fit: k={a.get('k')}, "
             f"T_c={a.get('cutoff_days')} d (standardized units)")
    if float(a.get("spatial_cutoff_px", 0.0)) > 0:
        title += f", spatial L_c={a.get('spatial_cutoff_px')} px"
    if status is not None and suffix == "":
        title += f".  Shaded: {int((status == 2).sum())} days no MODIS reached"
    fig.suptitle(title, fontsize=9, color=INK, ha="left", x=0.01)
    plotting.save(fig, out)


def seasonal_figure(ds: xr.Dataset, out: Path, dpi: int) -> None:
    """The per-pixel seasonal climatology: mean, scale and the amplitude/phase of each harmonic
    as maps, the fit method as a categorical map, and the reconstructed annual cycle.

    The map column echoes the EOF figure: one spatial field per row. The right panel is the
    seasonal analogue of the EOF loadings -- the annual cycle across the water, with a
    representative pixel and the reference cycle the thin-data pixels borrow.
    """
    import matplotlib.patches as mpatches
    from matplotlib.colors import BoundaryNorm, ListedColormap

    coef = ds["sst_seasonal_coef"].values                 # (P, y, x)
    scale = ds["sst_seasonal_sd"].values                  # (y, x)
    ftype = ds["sst_seasonal_fit_type"].values            # (y, x), -1 land / 0 / 1 / 2
    H = int(ds["sst_seasonal_coef"].attrs.get("n_harmonics", (coef.shape[0] - 1) // 2))
    period = float(ds["sst_seasonal_coef"].attrs.get("period_days", 365.25))
    water = plotting.water_mask(ds)
    extent = plotting.extent_km(ds)

    nrows = 1 + H                                          # row 0: mean|scale|fit; then amp|phase
    fig = plt.figure(figsize=(15, 3.0 * nrows + 0.8), dpi=dpi, layout="constrained")
    gs = fig.add_gridspec(nrows, 4, width_ratios=[1, 1, 1, 2.6])

    def _map(r, c, data, cmap, title, *, cyclic=False, tight=False, vmax=None):
        ax = fig.add_subplot(gs[r, c])
        w = np.isfinite(data[water])
        if cyclic:
            vmin, vhi = 0.0, period
        elif tight:                                       # absolute field: percentile window
            vmin, vhi = (float(np.nanpercentile(data[water], 1)),
                         float(np.nanpercentile(data[water], 99))) if w.any() else (0.0, 1.0)
            if vmin == vhi:
                vmin, vhi = vmin - 0.5, vhi + 0.5         # flat field: keep a usable range
        else:                                             # magnitude field anchored at 0
            vmin = 0.0
            vhi = vmax or (float(np.nanpercentile(data[water], 99)) if w.any() else 1.0)
        im = plotting.panel(ax, data, water, vmin=vmin, vmax=vhi, extent=extent, cmap=cmap)
        ax.set_title(title, fontsize=8, color=INK)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8).ax.tick_params(labelsize=6)
        return ax

    _map(0, 0, coef[0], plt.get_cmap("RdYlBu_r"), "mean (c0) [K]", tight=True)
    _map(0, 1, scale, plt.get_cmap("viridis"), "robust scale (SD) [K]")

    # fit-method map (categorical): full / mean-only / reference over water, land flat grey
    axf = fig.add_subplot(gs[0, 2])
    axf.set_facecolor("none")
    plotting.flat(axf, ~water, "#d9d9d9", extent)
    counts = {}
    for val, color in FIT_COLORS.items():
        m = (ftype == val) & water
        counts[val] = int(m.sum())
        if m.any():
            plotting.flat(axf, m, color, extent)
    axf.set_title("fit method", fontsize=8, color=INK)
    axf.set_xticks([]); axf.set_yticks([])
    axf.set_xlim(extent[0], extent[1]); axf.set_ylim(extent[2], extent[3])
    axf.legend(handles=[mpatches.Patch(color=FIT_COLORS[v],
                                       label=f"{FIT_LABELS[v]} ({counts[v]:,})")
                        for v in (FIT_FULL, FIT_MEAN_ONLY, FIT_REFERENCE)],
               fontsize=6, frameon=False, loc="lower left")

    twilight = plt.get_cmap("twilight")
    for k in range(1, H + 1):
        amp, peak = amplitude_phase(coef, k, period)      # each (y, x)
        _map(k, 0, amp, plt.get_cmap("magma"), f"harmonic {k}: amplitude [K]")
        _map(k, 1, peak * k, twilight, f"harmonic {k}: phase (peak DOY)", cyclic=True)

    # right: the reconstructed annual cycle across the water, one full period at daily step
    axc = fig.add_subplot(gs[:, 3])
    coef_w = coef[:, water]                                # (P, N_water)
    t0 = pd.to_datetime(ds["time"].values[0]).normalize()
    year = t0 + pd.to_timedelta(np.arange(int(round(period))), unit="D")
    Xy = design_matrix(year, H, period)                   # (365, P)
    cyc = Xy @ coef_w                                      # (365, N_water)
    doy = np.arange(cyc.shape[0])
    lo, med, hi = np.nanpercentile(cyc, [10, 50, 90], axis=1)
    axc.fill_between(doy, lo, hi, color=MODIS_LINE, alpha=0.15,
                     label="10-90th pct across water")
    axc.plot(doy, med, color=MODIS_LINE, linewidth=2, label="median pixel cycle")
    full_mask = (ftype == FIT_FULL) & water
    if full_mask.any():
        ref = coef[:, full_mask].mean(axis=1)             # the reference cycle (mean of FULL)
        axc.plot(doy, Xy @ ref, color=INK, linewidth=1.6, linestyle="--",
                 label="reference cycle (mean of full fits)")
        amp1 = np.hypot(coef[1][full_mask], coef[2][full_mask])
        pick = np.flatnonzero(full_mask.ravel())[int(np.argmin(
            np.abs(amp1 - np.median(amp1))))]
        rep_cycle = Xy @ coef.reshape(coef.shape[0], -1)[:, pick]
        axc.plot(doy, rep_cycle, color=FULL_LINE, linewidth=1.4, alpha=0.9,
                 label="representative full-fit pixel")
    _style(axc)
    axc.set_xlabel("day of year", fontsize=8)
    axc.set_ylabel("seasonal SST [K]", fontsize=8, color=INK_SECONDARY)
    axc.set_xlim(0, cyc.shape[0] - 1)
    axc.legend(fontsize=7, frameon=False, loc="best")

    n_full, n_mean, n_ref = counts[FIT_FULL], counts[FIT_MEAN_ONLY], counts[FIT_REFERENCE]
    fig.suptitle(f"Seasonal climatology: {H} harmonic(s), period {period:g} d.  "
                 f"{n_full:,} full / {n_mean:,} mean-only / {n_ref:,} reference pixels",
                 fontsize=9, color=INK, ha="left", x=0.01)
    plotting.save(fig, out)


def seasonal_gmrf_figure(ds: xr.Dataset, out: Path, dpi: int) -> None:
    """GMRF smoothing diagnostic: raw (per-pixel least-squares) vs smoothed vs their difference,
    for the mean and each harmonic's amplitude. Only drawn when the raw coefficients were kept
    (i.e. seasonal.method == 'gmrf'). The difference column is the shrinkage / gap-fill map: where
    it is large, the GMRF moved thin-data pixels off their noisy own-fit toward their neighbours."""
    raw = ds["sst_seasonal_coef_raw"].values              # (P, y, x), per-pixel LS (with fallback)
    sm = ds["sst_seasonal_coef"].values                   # (P, y, x), GMRF-smoothed
    H = int(ds["sst_seasonal_coef"].attrs.get("n_harmonics", (sm.shape[0] - 1) // 2))
    period = float(ds["sst_seasonal_coef"].attrs.get("period_days", 365.25))
    water = plotting.water_mask(ds)
    extent = plotting.extent_km(ds)
    fit = ds["sst_seasonal_fit_type"].values if "sst_seasonal_fit_type" in ds else None

    # rows: mean, then one amplitude per harmonic. ("mean" [K], "harmonic k amplitude" [K])
    def amp(c, k):
        return c[0] if k == 0 else np.hypot(c[2 * k - 1], c[2 * k])
    rows = [(0, "mean (c0) [K]")] + [(k, f"harmonic {k} amplitude [K]") for k in range(1, H + 1)]

    nrows = len(rows)
    fig = plt.figure(figsize=(13, 3.0 * nrows + 0.8), dpi=dpi, layout="constrained")
    gs = fig.add_gridspec(nrows, 3)

    def _draw(ax, data, cmap, vmin, vmax, title):
        im = plotting.panel(ax, data, water, vmin=vmin, vmax=vmax, extent=extent, cmap=cmap)
        ax.set_title(title, fontsize=8, color=INK)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.82).ax.tick_params(labelsize=6)

    seq = plt.get_cmap("magma")
    div = plt.get_cmap("RdBu_r").copy(); div.set_bad(alpha=0.0)
    moved = []
    for r, (k, label) in enumerate(rows):
        a_raw, a_sm = amp(raw, k), amp(sm, k)
        both = np.concatenate([a_raw[water], a_sm[water]])
        both = both[np.isfinite(both)]
        lo, hi = (float(np.percentile(both, 1)), float(np.percentile(both, 99))) if both.size \
            else (0.0, 1.0)
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        diff = a_sm - a_raw
        dlim = float(np.nanpercentile(np.abs(diff[water]), 99)) if np.isfinite(diff[water]).any() \
            else 1.0
        dlim = max(dlim, 1e-6)
        _draw(fig.add_subplot(gs[r, 0]), a_raw, seq, lo, hi, f"{label} — raw (per-pixel LS)")
        _draw(fig.add_subplot(gs[r, 1]), a_sm, seq, lo, hi, f"{label} — GMRF smoothed")
        _draw(fig.add_subplot(gs[r, 2]), diff, div, -dlim, dlim, f"{label} — smoothed − raw")
        ad = np.abs(diff[water])
        moved.append(float(np.nanpercentile(ad, 95)) if np.isfinite(ad).any() else float("nan"))

    note = ""
    if fit is not None:
        n_borrow = int(((fit == 1) | (fit == 0)).sum())
        note = f".  {n_borrow:,} pixels borrowed from neighbours"
    fig.suptitle("GMRF seasonal smoothing: raw vs smoothed vs difference.  "
                 f"95th-pct |Δ| per row: {', '.join(f'{m:.3g}' for m in moved)} K{note}",
                 fontsize=9, color=INK, ha="left", x=0.01)
    plotting.save(fig, out)


def cv_figure(curve: pd.DataFrame, fit: dict, out: Path, dpi: int) -> None:
    """Point- and day-holdout RMSE against k, one line per T_c (sequential by T_c)."""
    tcs = sorted(curve["t_c"].unique())
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), dpi=dpi, layout="constrained")
    for ax, col, which in ((axes[0], "rmse_point", "point"), (axes[1], "rmse_day", "day")):
        for i, tc in enumerate(tcs):
            g = curve[curve["t_c"] == tc].sort_values("k")
            ax.plot(g["k"], g[col], "o-", color=cmap(i / max(len(tcs) - 1, 1)), linewidth=2,
                    markersize=4, label=f"T_c={tc:g} d")
        kk, tt = fit.get(f"k_{which}"), fit.get(f"tc_{which}")
        sel = curve[(curve["k"] == kk) & (np.isclose(curve["t_c"], tt))]
        if len(sel):
            ax.plot(sel["k"], sel[col], "o", markersize=12, markerfacecolor="none",
                    markeredgecolor=INK, markeredgewidth=1.6, label="selected")
        _style(ax)
        ax.set_xscale("log")
        ax.set_xlabel("k (modes)", fontsize=8)
        ax.set_ylabel("RMSE (standardized units)", fontsize=8)
        ax.set_title(f"{which}-holdout CV" + (f": k={kk}, T_c={tt:g} d" if kk else ""),
                     fontsize=9, color=INK)
    axes[1].legend(fontsize=7, frameon=False)
    plotting.save(fig, out)


def cloud_filter_figure(ds: xr.Dataset, sid: str, sst: str, label: str, band, out: Path,
                        dpi: int) -> None:
    water = plotting.water_mask(ds)
    extent = plotting.extent_km(ds)
    raw = ds[sst].values
    keep = ds[f"{sid}_keep"].values.astype(bool)
    bits = ds[f"{sid}_keep_history"].values
    n_iter = int(ds[f"{sid}_keep_history"].attrs["n_iterations"])
    obs = np.isfinite(raw) & water[None]
    n = obs.sum(axis=0)
    rem = (obs & ~keep).sum(axis=0)
    flips = np.zeros(water.shape)
    for i in range(1, n_iter + 1):
        flips += (obs & (ID.kept_at(bits, i) != ID.kept_at(bits, i - 1))).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        fr = np.where(n > 0, rem / n, np.nan)
        fl = np.where(n > 0, flips / n, np.nan)

    fig = plt.figure(figsize=(15, 8.2), dpi=dpi, layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1])
    for c, (arr, cm, hi, title) in enumerate((
            (fr, "Blues", 1.0, "share of observations removed"),
            (fl, "Purples", max(float(np.nanpercentile(fl, 99)), 0.05)
             if np.isfinite(fl).any() else 1.0, "verdict flips per observation across the loop"))):
        ax = fig.add_subplot(gs[0, c])
        cmap = plt.get_cmap(cm).copy()
        cmap.set_bad(alpha=0.0)
        im = plotting.panel(ax, arr, water, vmin=0, vmax=hi, extent=extent, cmap=cmap)
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.85).ax.tick_params(labelsize=6)

    ax = fig.add_subplot(gs[1, :])
    t = pd.to_datetime(ds["time"].values)
    c = ds[f"{sid}_center"].values
    have = np.isfinite(c)
    kept_scene = keep.reshape(keep.shape[0], -1).any(axis=1)
    lo, hi = band
    if lo is not None and hi is not None:
        ax.axhspan(lo, hi, color=GRID, alpha=0.6, zorder=0, label="accepted band")
    ax.scatter(t[have & kept_scene], c[have & kept_scene], s=22, color=MODIS_LINE,
               edgecolor=SURFACE, linewidth=0.6, label="scene kept", zorder=3)
    ax.scatter(t[have & ~kept_scene], c[have & ~kept_scene], s=30, marker="X",
               color=plotting.FAIL_COLOR, edgecolor=SURFACE, linewidth=0.6,
               label="scene dropped", zorder=3)
    _style(ax)
    ax.set_ylabel("scene offset vs smooth field [K]", fontsize=8)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper left")
    tot = max(int(obs.sum()), 1)
    fig.suptitle(f"{label}: cloud filter. {int(have.sum())} scenes, {int(kept_scene[have].sum())}"
                 f" kept; {int((obs & ~keep).sum()) / tot:.1%} of {tot:,} observed px removed",
                 fontsize=9, color=INK, ha="left", x=0.01)
    plotting.save(fig, out)


def offsets_figure(pairs: pd.DataFrame, scenes: pd.DataFrame, row: pd.Series, label: str,
                   out: Path, dpi: int) -> None:
    """Footprint-pair anomalies (left) and per-scene offset vs overpass hour (right)."""
    kept = pairs["kept"].astype(bool)
    mx = pairs[kept].groupby("t")["modis"].mean()
    my = pairs[kept].groupby("t")["sensor"].mean()
    xa = pairs["modis"] - pairs["t"].map(mx)
    ya = pairs["sensor"] - pairs["t"].map(my)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=dpi, layout="constrained")
    ax = axes[0]
    ax.scatter(xa[kept], ya[kept], s=9, color=MODIS_LINE, alpha=0.55, edgecolor="none",
               label=f"kept ({int(kept.sum()):,})")
    if (~kept).any():
        ax.scatter(xa[~kept], ya[~kept], s=16, marker="x", color="#eb6834", linewidth=0.9,
                   label=f"removed as outliers ({int((~kept).sum()):,})")
    lim = float(np.nanpercentile(np.abs(np.r_[xa, ya]), 99.5)) if len(xa) else 1.0
    xs = np.array([-lim, lim])
    ax.plot(xs, xs, color=INK_MUTED, linewidth=1, linestyle="--", label="1:1")
    for key, ls, name in (("slope_ols", "-", "OLS"), ("slope_rma", ":", "RMA")):
        b = float(row.get(key, np.nan))
        if np.isfinite(b):
            ax.plot(xs, b * xs, color=INK, linewidth=1.6, linestyle=ls, label=f"{name} {b:.3f}")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    _style(ax)
    ax.set_xlabel("MODIS footprint, within-scene anomaly [K]", fontsize=8)
    ax.set_ylabel(f"{label} footprint median, within-scene anomaly [K]", fontsize=8)
    applied = float(row.get("slope_applied", 1.0))
    ax.set_title(f"{row.get('aggregate', 'footprint')} pairs; slope applied {applied:.3f}"
                 f" ({row.get('slope_estimator', 'none')}); clipping "
                 f"{int(row.get('clip_rounds', 0))} rounds"
                 + ("" if bool(row.get("clip_converged", True)) else ", NOT converged"),
                 fontsize=8, color=INK)
    ax.legend(fontsize=7, frameon=False, loc="upper left")

    ax = axes[1]
    if len(scenes):
        ax.scatter(scenes["hour"], scenes["delta"], s=22, color=MODIS_LINE, edgecolor=SURFACE,
                   linewidth=0.6, label="scene offset (median over its footprints)")
        o = scenes.sort_values("hour")
        if "fitted" in o:
            ax.plot(o["hour"], o["fitted"], color=INK, linewidth=1.6, label="fitted a(hour)")
    _style(ax)
    ax.set_xlabel("overpass hour [UTC]", fontsize=8)
    ax.set_ylabel("offset at the pivot temperature [K]", fontsize=8)
    ax.set_title(f"K={int(row.get('K', 0))}, mean offset {float(row.get('offset_mean', np.nan)):+.3f}"
                 f" K over {int(row.get('n_scenes', 0))} scenes", fontsize=8, color=INK)
    ax.legend(fontsize=7, frameon=False)
    fig.suptitle(f"{label}: offset against MODIS", fontsize=9, color=INK, ha="left", x=0.01)
    plotting.save(fig, out)


def diagnostics_adapter(ds: xr.Dataset, cfg: dict, scenes: pd.DataFrame) -> dict:
    """The `D` dict iterative_diagnostics' figure functions read, from a pipeline cube."""
    water = plotting.water_mask(ds)
    sensors = {}
    for sid, s in cfg["sensors"].items():
        raw = ds[s["sst"]].values
        js = np.sort(scenes.loc[scenes["sensor"] == sid, "t"].to_numpy())
        sensors[sid] = dict(label=s.get("label") or sid, raw=raw, scenes=js,
                            bits=ds[f"{sid}_keep_history"].values,
                            n_iter=int(ds[f"{sid}_keep_history"].attrs["n_iterations"]))
    fields = dict(filled=ds["sst_filled"].values, base=ds["sst_smooth"].values,
                  comp=ds["sst_composite"].values, src=ds["sst_composite_src"].values,
                  order=ds["sst_composite_src"].attrs["flag_meanings"].split(),
                  coarsen=1,
                  status=(ds["smooth_loading_status"].values
                          if "smooth_loading_status" in ds else None))
    return dict(water=water, times=ds["time"].values, extent=plotting.extent_km(ds),
                sensors=sensors, fields=fields,
                source=ds["sst_smooth"].attrs.get("baseline_source", "modis"))


def render(cube: Path, fig_dir: Path, rep_dir: Path | None = None, *, dpi: int = 130,
           n_field_days: int = 3, field_dates=()) -> None:
    cube = Path(cube)
    rep_dir = Path(rep_dir) if rep_dir is not None else None
    ds = xr.open_zarr(cube)
    cfg = yaml.safe_load(ds.attrs["pipeline_config"])
    fit = json.loads(ds.attrs["pipeline_fit"])
    fig_dir = Path(fig_dir)

    eof_figure(ds, "", fig_dir / "eofs.png", dpi)
    eof_figure(ds, "_point", fig_dir / "eofs_point.png", dpi)

    if "sst_seasonal_coef" in ds:
        seasonal_figure(ds, fig_dir / "seasonal.png", dpi)
    if "sst_seasonal_coef_raw" in ds:
        seasonal_gmrf_figure(ds, fig_dir / "seasonal_gmrf.png", dpi)

    if rep_dir is not None and (rep_dir / "cv_curve.csv").exists():
        cv_figure(pd.read_csv(rep_dir / "cv_curve.csv"),
                  {"k_point": fit["k_point"], "tc_point": fit["tc_point"],
                   "k_day": fit["k_day"], "tc_day": fit["tc_day"]},
                  fig_dir / "cv_curves.png", dpi)

    band = (cfg.get("filter", {}).get("offset_lower", -2.0),
            cfg.get("filter", {}).get("offset_upper", 4.0))
    for sid, s in cfg["sensors"].items():
        cloud_filter_figure(ds, sid, s["sst"], s.get("label") or sid, band,
                            fig_dir / f"cloud_filter_{sid}.png", dpi)

    if rep_dir is not None and (rep_dir / "footprint_pairs_final.csv").exists():
        pairs = pd.read_csv(rep_dir / "footprint_pairs_final.csv")
        scenes = pd.read_csv(rep_dir / "matchups_final.csv")
        rep = pd.read_csv(rep_dir / "offsets_final.csv").set_index("member")
        for sid, s in cfg["sensors"].items():
            if sid in rep.index:
                offsets_figure(pairs[pairs["sensor_id"] == sid],
                               scenes[scenes["member"] == sid], rep.loc[sid],
                               s.get("label") or sid, fig_dir / f"offsets_{sid}.png", dpi)

    if rep_dir is not None and (rep_dir / "scenes.csv").exists():
        D = diagnostics_adapter(ds, cfg, pd.read_csv(rep_dir / "scenes.csv"))
        ID.flag_changes_figure(D, fig_dir / "flag_changes.png", dpi)
        days = ID.pick_field_days(D, n_field_days, list(field_dates))
        for page, i in enumerate(range(0, len(days), 6)):
            ID.fields_figure(D, days[i:i + 6], fig_dir / f"fields_{page + 1}.png", dpi)
    else:
        log.info("no reports directory; flag_changes and fields figures skipped")
    log.info("figures in %s", fig_dir)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cube", type=Path, required=True)
    p.add_argument("--fig-dir", type=Path, default=None,
                   help="default: figures<suffix>/ next to the cube")
    p.add_argument("--reports", type=Path, default=None,
                   help="default: reports<suffix>/ next to the cube")
    p.add_argument("--field-dates", nargs="*", default=[])
    p.add_argument("--dpi", type=int, default=130)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    stem = args.cube.stem
    sfx = stem.split("_pipeline", 1)[1] if "_pipeline" in stem else ""
    fig_dir = args.fig_dir or args.cube.parent / f"figures{sfx}"
    rep_dir = args.reports or args.cube.parent / f"reports{sfx}"
    render(args.cube, fig_dir, rep_dir, dpi=args.dpi, field_dates=args.field_dates)


if __name__ == "__main__":
    main()
