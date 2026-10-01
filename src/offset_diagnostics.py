"""Slope estimates and removed outliers in the MODIS offset fit, as figures.

Runs the pipeline's own matchup and clipping code on a real cube and draws, per sensor:

  slope_scatter_<id>.png   footprint pairs as within-scene anomalies (sensor vs MODIS). Left:
                           every pair, with the OLS, RMA and Theil-Sen slopes. Right: after
                           outlier clipping, kept vs removed, slopes refit on the kept pairs.
  slope_per_scene_<id>.png each scene's own OLS and Theil-Sen slope against date (marker area
                           ~ footprints), with the pooled estimates as reference lines.
  outliers_<id>.png        the residuals with the clip threshold, the share removed per scene,
                           and the clipping's convergence round by round.
  outlier_maps_<id>.png    the scenes with the most removals: sensor field, and each
                           footprint's residual with removed footprints marked.
  slope_estimates.png      every estimator x pair set x support, with a 90% bootstrap interval
                           from resampling scenes.
  slope_absolute_<id>.png  the slope WITHOUT within-scene centring, so the seasonal range is the
                           leverage: every kept footprint pair pooled in absolute temperature
                           (left), and one point per scene -- the scene's median pair --
                           coloured by overpass hour (right).
  slope_absolute_estimates.png
                           absolute-pooled, scene-median and within-scene slopes side by side,
                           with 90% scene-bootstrap intervals and the offset at the pivot.

Pixel keep-masks come from a pipeline cube's final cloud filter (--cube), or from the sensor
QC alone when no cube is given.

    python src/offset_diagnostics.py --cube data/pipeline/admiralty_inlet_pipeline_e2e.zarr
    python src/offset_diagnostics.py                       # QC-only masks
    ... --out-dir figures/offset_diagnostics --n-boot 200
"""

from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import iterative_filter as F
import pipeline as P
import plotting
from cube_figures import GRID, INK, INK_MUTED, INK_SECONDARY, SURFACE

log = logging.getLogger("offset_diagnostics")

KEPT = "#8fb8e8"
REMOVED = "#eb6834"
EST_STYLE = {"OLS": ("#0b0b0b", "-"), "RMA": ("#1baf7a", "--"),
             "Theil-Sen": ("#4a3aa7", "-.")}
AGG_COLOR = {"pixel": "#898781", "footprint": "#2a78d6"}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=7)


# ==================================================================== estimators

def anomalies(pairs: pd.DataFrame, centre: str) -> tuple[np.ndarray, np.ndarray]:
    """Within-scene anomalies of MODIS and sensor, about each scene's mean or median."""
    g = pairs.groupby("t")
    xa = pairs["modis"] - g["modis"].transform(centre)
    ya = pairs["sensor"] - g["sensor"].transform(centre)
    return xa.to_numpy(float), ya.to_numpy(float)


def estimates(pairs: pd.DataFrame, ts_max: int = 3000) -> dict:
    """OLS and RMA on mean-anomalies, Theil-Sen on median-anomalies, as the pipeline uses them."""
    out = {}
    for name in ("OLS", "RMA"):
        out[name] = P.anomaly_slope(pairs["modis"].to_numpy(float), pairs["sensor"].to_numpy(float),
                                    pairs["t"].to_numpy(int), name.lower())
    out["Theil-Sen"] = P.robust_anomaly_slope(pairs["modis"].to_numpy(float),
                                              pairs["sensor"].to_numpy(float),
                                              pairs["t"].to_numpy(int), max_pairs=ts_max)
    return out


def bootstrap(pairs: pd.DataFrame, n_boot: int, seed: int = 0) -> dict:
    """90% intervals by resampling whole scenes (a scene's pairs share its errors)."""
    rng = np.random.default_rng(seed)
    scenes = pairs["t"].unique()
    groups = {t: g for t, g in pairs.groupby("t")}
    draws = {k: [] for k in EST_STYLE}
    for b in range(n_boot):
        pick = rng.choice(scenes, scenes.size, replace=True)
        # a scene drawn twice must stay two scenes for the anomaly centring
        sample = pd.concat([groups[t].assign(t=i) for i, t in enumerate(pick)],
                           ignore_index=True)
        e = estimates(sample, ts_max=1000)
        for k, v in e.items():
            draws[k].append(v)
    return {k: (np.nanpercentile(v, 5), np.nanpercentile(v, 95)) for k, v in draws.items()}


def absolute_estimates(x: np.ndarray, y: np.ndarray, ts_max: int = 3000,
                       seed: int = 0) -> dict:
    """Slopes of y on x in ABSOLUTE temperature (no centring), and each line's offset at the
    pivot x0 = median(x): name -> (slope, offset at x0)."""
    from scipy.stats import theilslopes
    x, y = np.asarray(x, float), np.asarray(y, float)
    out = {}
    if x.size < 3 or np.ptp(x) == 0:
        return {k: (np.nan, np.nan) for k in EST_STYLE}
    x0 = float(np.median(x))
    b = float(np.polyfit(x, y, 1)[0])
    out["OLS"] = (b, float(np.mean(y) - np.mean(x) + (b - 1) * (np.mean(x) - x0)))
    r = np.corrcoef(x, y)[0, 1]
    b = float(np.sign(r) * y.std() / x.std())
    out["RMA"] = (b, float(np.mean(y) - np.mean(x) + (b - 1) * (np.mean(x) - x0)))
    xs, ys = x, y
    if x.size > ts_max:
        i = np.random.default_rng(seed).choice(x.size, ts_max, replace=False)
        xs, ys = x[i], y[i]
    b, a = theilslopes(ys, xs)[:2]
    out["Theil-Sen"] = (float(b), float(a + b * x0 - x0))
    return out


def scene_medians(pairs: pd.DataFrame) -> pd.DataFrame:
    """One row per scene: median MODIS and median sensor of its kept pairs, hour, n."""
    k = pairs[pairs["kept"]]
    return k.groupby("t").agg(modis=("modis", "median"), sensor=("sensor", "median"),
                              hour=("hour", "first"), n=("modis", "size"),
                              date=("date", "first")).reset_index()


def absolute_bootstrap(pairs: pd.DataFrame, n_boot: int, seed: int = 0) -> dict:
    """90% scene-bootstrap intervals of the absolute-pooled and scene-median slopes."""
    rng = np.random.default_rng(seed)
    k = pairs[pairs["kept"]]
    groups = {t: g for t, g in k.groupby("t")}
    scenes = np.array(list(groups))
    sm = scene_medians(pairs).set_index("t")
    draws = {(kind, e): [] for kind in ("pooled", "scene") for e in EST_STYLE}
    for _ in range(n_boot):
        pick = rng.choice(scenes, scenes.size, replace=True)
        g = pd.concat([groups[t] for t in pick], ignore_index=True)
        for e, (b, _) in absolute_estimates(g["modis"], g["sensor"], ts_max=1000).items():
            draws[("pooled", e)].append(b)
        m = sm.loc[pick]
        for e, (b, _) in absolute_estimates(m["modis"], m["sensor"]).items():
            draws[("scene", e)].append(b)
    return {key: (np.nanpercentile(v, 5), np.nanpercentile(v, 95)) for key, v in draws.items()}


def absolute_figure(pairs: pd.DataFrame, label: str, out: Path, dpi: int) -> None:
    kept = pairs["kept"].to_numpy(bool)
    xc = pairs["modis"].to_numpy(float) - 273.15
    yc = pairs["sensor"].to_numpy(float) - 273.15
    sm = scene_medians(pairs)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.3), dpi=dpi, layout="constrained",
                             sharex=True, sharey=True)
    lo = float(np.nanpercentile(np.r_[xc, yc], 0.5)) - 0.5
    hi = float(np.nanpercentile(np.r_[xc, yc], 99.5)) + 0.5
    xs = np.array([lo, hi])

    def lines(ax, est, x0c):
        ax.plot(xs, xs, color=INK_MUTED, linestyle=":", linewidth=1, label="1:1")
        for name, (b, a) in est.items():
            c, ls = EST_STYLE[name]
            if np.isfinite(b):
                ax.plot(xs, x0c + a + b * (xs - x0c), color=c, linestyle=ls, linewidth=1.8,
                        label=f"{name}: slope {b:.3f}, offset {a:+.2f} K")

    ax = axes[0]
    ax.scatter(xc[~kept], yc[~kept], s=10, marker="x", color=REMOVED, linewidth=0.7,
               alpha=0.6, label=f"removed by clipping ({int((~kept).sum()):,})")
    ax.scatter(xc[kept], yc[kept], s=7, color=KEPT, alpha=0.6, edgecolor="none",
               label=f"kept footprint pairs ({int(kept.sum()):,})")
    est = absolute_estimates(xc[kept], yc[kept])
    lines(ax, est, float(np.median(xc[kept])))
    ax.set_title(f"all scenes pooled, absolute temperature (pivot {np.median(xc[kept]):.1f} "
                 "degC)", fontsize=9, color=INK)

    ax = axes[1]
    cmap = plt.get_cmap("viridis")
    sc = ax.scatter(sm["modis"] - 273.15, sm["sensor"] - 273.15, c=sm["hour"], cmap=cmap,
                    vmin=0, vmax=24, s=np.clip(sm["n"] * 1.5, 15, 200), edgecolor=SURFACE,
                    linewidth=0.7, zorder=3)
    est_s = absolute_estimates(sm["modis"] - 273.15, sm["sensor"] - 273.15)
    lines(ax, est_s, float(np.median(sm["modis"] - 273.15)))
    fig.colorbar(sc, ax=ax, shrink=0.8, label="overpass hour [UTC]").ax.tick_params(labelsize=6)
    ax.set_title(f"one point per scene: median of its kept pairs ({len(sm)} scenes; area ~ "
                 "footprints)", fontsize=9, color=INK)
    for ax in axes:
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        _style(ax)
        ax.set_xlabel("MODIS footprint [degC]", fontsize=8)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    axes[0].set_ylabel(f"{label} footprint median [degC]", fontsize=8)
    within = float(np.median(pairs[pairs["kept"]].groupby("t")["modis"].std()))
    fig.suptitle(f"{label}: slope across scenes. MODIS spans {np.ptp(sm['modis']):.1f} K across "
                 f"scenes vs a median {within:.2f} K within one", fontsize=10, color=INK,
                 ha="left", x=0.01)
    plotting.save(fig, out)


def absolute_estimates_figure(rows: pd.DataFrame, out: Path, dpi: int) -> None:
    sensors = list(dict.fromkeys(rows["sensor"]))
    kinds = ["absolute, all pairs pooled", "absolute, scene medians",
             "within-scene anomalies (reference)"]
    kind_color = {kinds[0]: "#2a78d6", kinds[1]: "#1baf7a", kinds[2]: "#898781"}
    fig, axes = plt.subplots(1, len(sensors), figsize=(6.4 * len(sensors), 4.6), dpi=dpi,
                             layout="constrained", sharex=True, squeeze=False)
    for ax, sid in zip(axes[0], sensors):
        sub = rows[rows["sensor"] == sid]
        labels, y = [], 0
        for est in EST_STYLE:
            for kind in kinds:
                r = sub[(sub["estimator"] == est) & (sub["kind"] == kind)]
                if len(r):
                    r = r.iloc[0]
                    ax.plot([r["lo"], r["hi"]], [y, y], color=kind_color[kind], linewidth=2.2,
                            solid_capstyle="round")
                    ax.plot(r["slope"], y, "o", color=kind_color[kind], markersize=7,
                            markeredgecolor=SURFACE, markeredgewidth=1)
                    txt = f"{r['slope']:.2f}"
                    if np.isfinite(r.get("offset", np.nan)):
                        txt += f"  ({r['offset']:+.2f} K)"
                    ax.annotate(txt, (r["hi"], y), xytext=(4, 0), textcoords="offset points",
                                va="center", fontsize=6.5, color=INK_SECONDARY)
                labels.append(f"{est} | {kind}")
                y += 1
            y += 0.6
        ys = [i + 0.6 * (i // 3) for i in range(len(labels))]
        ax.set_yticks(ys, labels, fontsize=7)
        ax.invert_yaxis()
        ax.axvline(1.0, color=INK_MUTED, linestyle=":", linewidth=1)
        _style(ax)
        ax.set_xlabel("slope (dot), 90% scene-bootstrap interval; label: slope (offset at "
                      "pivot)", fontsize=8)
        ax.set_title(sid, fontsize=10, color=INK)
    fig.suptitle("slope from the seasonal range (absolute) vs within scenes, footprint pairs "
                 "after clipping at slope 1", fontsize=10, color=INK, ha="left", x=0.01)
    plotting.save(fig, out)


# ==================================================================== figures

def scatter_figure(pairs: pd.DataFrame, label: str, k: float, out: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=dpi, layout="constrained",
                             sharex=True, sharey=True)
    kept = pairs["kept"].to_numpy(bool)
    xa, ya = anomalies(pairs, "mean")
    xk, yk = anomalies(pairs[kept], "mean")
    lim = float(np.nanpercentile(np.abs(np.r_[xa, ya]), 99.5))
    xs = np.array([-lim, lim])
    panels = ((axes[0], pairs, xa, ya, None, f"all {len(pairs):,} footprint pairs"),
              (axes[1], pairs[kept], xk, yk, kept,
               f"after clipping at {k:g} x robust SD: {int(kept.sum()):,} kept, "
               f"{int((~kept).sum()):,} removed"))
    for ax, sub, x, y, kmask, title in panels:
        if kmask is None:
            ax.scatter(x, y, s=8, color=KEPT, alpha=0.6, edgecolor="none", label="pair")
        else:
            # removed pairs plotted against the kept-set centring of their own scene
            rem = pairs[~kept]
            mx = pairs[kept].groupby("t")["modis"].mean()
            my = pairs[kept].groupby("t")["sensor"].mean()
            ax.scatter(x, y, s=8, color=KEPT, alpha=0.6, edgecolor="none", label="kept")
            ax.scatter(rem["modis"] - rem["t"].map(mx), rem["sensor"] - rem["t"].map(my),
                       s=14, marker="x", color=REMOVED, linewidth=0.8, label="removed")
        ax.plot(xs, xs, color=INK_MUTED, linewidth=1, linestyle=":", label="1:1")
        for name, b in estimates(sub).items():
            c, ls = EST_STYLE[name]
            if np.isfinite(b):
                ax.plot(xs, b * xs, color=c, linestyle=ls, linewidth=1.8,
                        label=f"{name} {b:.3f}")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        _style(ax)
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel("MODIS footprint, within-scene anomaly [K]", fontsize=8)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    axes[0].set_ylabel(f"{label} footprint median, within-scene anomaly [K]", fontsize=8)
    sd_x = float(np.median(pairs.groupby("t")["modis"].std()))
    fig.suptitle(f"{label}: slope of sensor on MODIS from pooled within-scene anomalies "
                 f"(median within-scene MODIS spread {sd_x:.2f} K)", fontsize=10, color=INK,
                 ha="left", x=0.01)
    plotting.save(fig, out)


def per_scene_figure(pairs: pd.DataFrame, pooled: dict, label: str, out: Path,
                     dpi: int, min_n: int = 15) -> None:
    rows = []
    for t, g in pairs[pairs["kept"]].groupby("t"):
        if len(g) < min_n:
            continue
        x, y = g["modis"].to_numpy(float), g["sensor"].to_numpy(float)
        if np.ptp(x) == 0:
            continue
        ols = float(np.polyfit(x - x.mean(), y - y.mean(), 1)[0])
        from scipy.stats import theilslopes
        ts = float(theilslopes(y - np.median(y), x - np.median(x))[0])
        rows.append(dict(date=pd.Timestamp(g["date"].iloc[0]), n=len(g), ols=ols, ts=ts,
                         sd=float(x.std())))
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.6), dpi=dpi, layout="constrained",
                             sharex=True, height_ratios=[2.2, 1])
    ax = axes[0]
    if len(df):
        ax.scatter(df["date"], df["ols"], s=np.clip(df["n"] * 1.2, 12, 160), color=EST_STYLE[
            "OLS"][0], alpha=0.75, edgecolor=SURFACE, linewidth=0.6, label="scene OLS")
        ax.scatter(df["date"], df["ts"], s=np.clip(df["n"] * 1.2, 12, 160), marker="D",
                   color=EST_STYLE["Theil-Sen"][0], alpha=0.6, edgecolor=SURFACE,
                   linewidth=0.6, label="scene Theil-Sen")
    for name, b in pooled.items():
        c, ls = EST_STYLE[name]
        if np.isfinite(b):
            ax.axhline(b, color=c, linestyle=ls, linewidth=1.4, label=f"pooled {name} {b:.3f}")
    ax.axhline(1.0, color=INK_MUTED, linestyle=":", linewidth=1)
    _style(ax)
    ax.set_ylabel("slope (sensor on MODIS)", fontsize=8)
    if len(df):
        lo, hi = np.nanpercentile(np.r_[df["ols"], df["ts"]], [2, 98])
        ax.set_ylim(min(lo, 0) - 0.2, max(hi, 2) + 0.2)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper left")
    ax.set_title(f"{label}: each scene's own slope on its kept footprints (marker area ~ "
                 f"footprints; scenes with >= {min_n})", fontsize=9, color=INK)
    ax = axes[1]
    if len(df):
        ax.bar(df["date"], df["sd"], width=2.0, color="#2a78d6")
    _style(ax)
    ax.set_ylabel("within-scene MODIS\nspread [K]", fontsize=8)
    plotting.save(fig, out)


def outliers_figure(pairs: pd.DataFrame, info: dict, k: float, label: str, out: Path,
                    dpi: int) -> None:
    kept = pairs["kept"].to_numpy(bool)
    r = pairs["resid"].to_numpy(float)
    rc = r - np.median(r)
    s = float(info["scale"])
    fig = plt.figure(figsize=(14, 7.5), dpi=dpi, layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

    ax = fig.add_subplot(gs[0, 0])
    lim = float(np.nanpercentile(np.abs(rc), 99.5))
    bins = np.linspace(-lim, lim, 81)
    ax.hist([rc[kept], rc[~kept]], bins=bins, stacked=True, color=[KEPT, REMOVED],
            label=[f"kept {int(kept.sum()):,}", f"removed {int((~kept).sum()):,}"])
    for sgn in (-1, 1):
        ax.axvline(sgn * k * s, color=INK, linewidth=1.2, linestyle="--")
    ax.text(k * s, ax.get_ylim()[1] * 0.92, f"  +/-{k:g} x {s:.3f} K", fontsize=7,
            color=INK)
    _style(ax)
    ax.set_xlabel("residual against own scene's fit [K]", fontsize=8)
    ax.set_ylabel("footprint pairs", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("residuals and the clip threshold (robust SD of all pairs)", fontsize=9,
                 color=INK)

    ax = fig.add_subplot(gs[0, 1])
    h = pd.DataFrame(info.get("history", []))
    if len(h):
        ax.plot(h["round"], h["removed"], "o-", color=REMOVED, linewidth=2,
                label="pairs removed")
        for _, row in h.iterrows():
            ax.annotate(f"{row['scale']:.3f}", (row["round"], row["removed"]),
                        xytext=(0, 6), textcoords="offset points", ha="center", fontsize=6,
                        color=INK_SECONDARY)
    _style(ax)
    ax.set_xlabel("clipping round", fontsize=8)
    ax.set_ylabel("pairs removed", fontsize=8)
    ax.set_title(f"convergence: {info['rounds']} rounds, "
                 f"{'converged' if info['converged'] else 'NOT converged'} "
                 "(labels: robust SD that round, K)", fontsize=9, color=INK)

    ax = fig.add_subplot(gs[1, :])
    per = pairs.groupby("date").agg(n=("kept", "size"), removed=("kept", lambda v: (~v).sum()))
    per.index = pd.to_datetime(per.index)
    frac = per["removed"] / per["n"]
    ax.bar(per.index, frac, width=2.5, color=REMOVED)
    for d, f_, n in zip(per.index, frac, per["n"]):
        if f_ > 0.5:
            ax.annotate(f"{n}", (d, f_), xytext=(0, 2), textcoords="offset points",
                        ha="center", fontsize=6, color=INK_SECONDARY)
    _style(ax)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("share of the scene's\nfootprints removed", fontsize=8)
    ax.set_title("removals per scene (labels: footprints in scenes losing > 50%)", fontsize=9,
                 color=INK)
    fig.suptitle(f"{label}: iterative outlier removal", fontsize=10, color=INK, ha="left",
                 x=0.01)
    plotting.save(fig, out)


def outlier_maps_figure(pairs: pd.DataFrame, raw: F.Raw, mem: np.ndarray, label: str,
                        extent, out: Path, dpi: int, n_scenes: int = 4) -> None:
    rem = pairs.groupby("t")["kept"].apply(lambda v: int((~v).sum()))
    pick = rem.sort_values(ascending=False).head(n_scenes).index.tolist()
    if not pick:
        return
    water = raw.water
    fig, axes = plt.subplots(len(pick), 3, figsize=(13, 4.0 * len(pick)), dpi=dpi,
                             layout="constrained", squeeze=False)
    div = plt.get_cmap("RdBu_r").copy()
    div.set_bad(alpha=0.0)
    for row, t in zip(axes, pick):
        g = pairs[pairs["t"] == t]
        labels = P.modis_footprints(raw.ref[t])
        lut_r = np.full(labels.max() + 1, np.nan)
        lut_k = np.zeros(labels.max() + 1, bool)
        lut_r[g["fp"].to_numpy(int)] = g["resid"].to_numpy(float) - np.median(pairs["resid"])
        lut_k[g["fp"].to_numpy(int)] = ~g["kept"].to_numpy(bool)
        res_map = lut_r[labels]
        res_map[labels == 0] = np.nan
        rem_map = lut_k[labels] & (labels > 0)
        sen = mem[t] - 273.15
        mod = raw.ref[t] - 273.15
        vals = np.r_[sen[np.isfinite(sen)], mod[np.isfinite(mod)]]
        lo, hi = np.percentile(vals, [2, 98]) if vals.size else (0, 1)
        im = plotting.panel(row[0], sen, water, vmin=lo, vmax=hi, extent=extent)
        row[0].set_title(f"{g['date'].iloc[0]}  {label} (kept pixels)", fontsize=8, color=INK)
        plotting.panel(row[1], mod, water, vmin=lo, vmax=hi, extent=extent)
        row[1].set_title("MODIS footprints", fontsize=8, color=INK)
        fig.colorbar(im, ax=row[:2], shrink=0.8, label="SST [degC]").ax.tick_params(labelsize=6)
        rl = float(np.nanpercentile(np.abs(res_map), 98)) if np.isfinite(res_map).any() else 1
        imr = plotting.panel(row[2], res_map, water, vmin=-rl, vmax=rl, extent=extent, cmap=div)
        plotting.flat(row[2], rem_map, REMOVED, extent)
        row[2].set_title(f"footprint residual; orange = removed "
                         f"({int((~g['kept']).sum())} of {len(g)})", fontsize=8, color=INK)
        fig.colorbar(imr, ax=row[2], shrink=0.8, label="K").ax.tick_params(labelsize=6)
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"{label}: scenes with the most removed footprints", fontsize=10, color=INK,
                 ha="left", x=0.01)
    plotting.save(fig, out)


def estimates_figure(rows: pd.DataFrame, out: Path, dpi: int) -> None:
    sensors = list(dict.fromkeys(rows["sensor"]))
    fig, axes = plt.subplots(1, len(sensors), figsize=(6.4 * len(sensors), 6.2), dpi=dpi,
                             layout="constrained", sharex=True, squeeze=False)
    sets = ["all pairs", "clipped at slope 1", "clipped at Theil-Sen"]
    for ax, sid in zip(axes[0], sensors):
        sub = rows[rows["sensor"] == sid]
        ylabels, y = [], 0
        for est in EST_STYLE:
            for st in sets:
                for agg in ("pixel", "footprint"):
                    r = sub[(sub["estimator"] == est) & (sub["set"] == st)
                            & (sub["aggregate"] == agg)]
                    if len(r):
                        r = r.iloc[0]
                        ax.plot([r["lo"], r["hi"]], [y, y], color=AGG_COLOR[agg], linewidth=2.2,
                                solid_capstyle="round")
                        ax.plot(r["slope"], y, "o", color=AGG_COLOR[agg], markersize=7,
                                markeredgecolor=SURFACE, markeredgewidth=1)
                        ax.annotate(f"{r['slope']:.2f}", (r["hi"], y), xytext=(4, 0),
                                    textcoords="offset points", va="center", fontsize=6.5,
                                    color=INK_SECONDARY)
                    ylabels.append(f"{est} | {st} | {agg}")
                    y += 1
            y += 0.6
        ax.axvline(1.0, color=INK_MUTED, linestyle=":", linewidth=1)
        ys = [i + 0.6 * (i // 6) for i in range(len(ylabels))]
        ax.set_yticks(ys, ylabels, fontsize=7)
        ax.invert_yaxis()
        _style(ax)
        ax.set_xlabel("slope of sensor on MODIS (dot) with 90% scene-bootstrap interval",
                      fontsize=8)
        ax.set_title(sid, fontsize=10, color=INK)
    fig.legend(handles=[Line2D([], [], color=AGG_COLOR[a], marker="o", linewidth=2.2,
                               label=f"{a} pairs") for a in ("pixel", "footprint")],
               loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle("slope estimates by estimator, pair set and support", fontsize=10, color=INK,
                 ha="left", x=0.01)
    fig.get_layout_engine().set(rect=(0, 0.05, 1, 0.95))
    plotting.save(fig, out)


# ==================================================================== driver

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=P.DEFAULT_CONFIG)
    p.add_argument("--cube", type=Path, default=None,
                   help="pipeline cube whose <id>_keep masks to use (default: QC-only)")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--dpi", type=int, default=130)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("pipeline", "iterative_filter", "composite", "simple_outlier_detection"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cfg = P.load_config(args.config)
    it = cfg["_iter"]
    times = None
    if args.cube is not None:
        with xr.open_zarr(args.cube) as c:
            times = c["time"].values
    elif cfg["data"]["time_range"] is not None:
        with xr.open_zarr(cfg["data"]["source"]) as s:
            t = pd.to_datetime(s["time"].values)
        tr = cfg["data"]["time_range"]
        times = np.asarray(t[(t >= pd.Timestamp(tr[0])) & (t <= pd.Timestamp(tr[1]))].values)
    raw = F.load_raw(it, times)
    if args.cube is not None:
        with xr.open_zarr(args.cube) as c:
            keep = {sid: c[f"{sid}_keep"].values.astype(bool) for sid in raw.raw}
        source = f"cloud filter of {args.cube.name}"
    else:
        keep = F.qc_only_keep(raw, it["filter"]["require_qc"])
        source = "sensor QC only"
    out_dir = args.out_dir or (P.ROOT / "figures" / "offset_diagnostics" / cfg["data"]["aoi"])
    k = float(cfg["matchup"]["outlier_k"] or 1.5)
    times_pd = pd.to_datetime(raw.times)
    extent = ([0.0, float(np.ptp(raw.coords["x"])) / 1000.0, 0.0,
               float(np.ptp(raw.coords["y"])) / 1000.0] if "x" in raw.coords else None)
    log.info("pixel masks: %s", source)

    rows, abs_rows = [], []
    for sid, r in raw.raw.items():
        label = it["sensors"][sid]["label"]
        mem = np.where(keep[sid], r, np.nan).astype("float32")
        for agg in ("footprint", "pixel"):
            c = copy.deepcopy(cfg)
            c["matchup"].update(aggregate=agg, outlier_k=k)
            pairs = P.matchup_pairs(mem, raw.ref, raw.hours[sid], times_pd, c, sid)
            pairs["kept"] = True
            sets = {"all pairs": pairs}
            clipped1, info1 = P.clip_pairs(pairs.drop(columns="kept"), c)
            sets["clipped at slope 1"] = clipped1[clipped1["kept"]]
            c_ts = copy.deepcopy(c)
            c_ts["offset"]["slope"] = True
            clipped_ts, info_ts = P.clip_pairs(pairs.drop(columns="kept"), c_ts)
            sets["clipped at Theil-Sen"] = clipped_ts[clipped_ts["kept"]]
            log.info("%s %s: %d pairs; slope-1 clip removed %d in %d rounds; Theil-Sen clip "
                     "(b=%.3f) removed %d", sid, agg, len(pairs), info1["n_removed"],
                     info1["rounds"], info_ts["b_clip"], info_ts["n_removed"])
            for st, sp in sets.items():
                est = estimates(sp)
                ci = bootstrap(sp, args.n_boot if agg == "footprint" else max(50, args.n_boot // 4))
                for name, b in est.items():
                    rows.append(dict(sensor=sid, aggregate=agg, set=st, estimator=name,
                                     slope=b, lo=ci[name][0], hi=ci[name][1], n=len(sp),
                                     scenes=sp["t"].nunique()))
            if agg == "footprint":
                scatter_figure(clipped1, label, k, out_dir / f"slope_scatter_{sid}.png",
                               args.dpi)
                per_scene_figure(clipped1, estimates(sets["clipped at slope 1"]), label,
                                 out_dir / f"slope_per_scene_{sid}.png", args.dpi)
                outliers_figure(clipped1, info1, k, label, out_dir / f"outliers_{sid}.png",
                                args.dpi)
                outlier_maps_figure(clipped1, raw, mem, label, extent,
                                    out_dir / f"outlier_maps_{sid}.png", args.dpi)
                absolute_figure(clipped1, label, out_dir / f"slope_absolute_{sid}.png",
                                args.dpi)
                kp = clipped1[clipped1["kept"]]
                sm = scene_medians(clipped1)
                ci = absolute_bootstrap(clipped1, args.n_boot)
                ci_w = bootstrap(kp, args.n_boot)
                est_p = absolute_estimates(kp["modis"], kp["sensor"])
                est_s = absolute_estimates(sm["modis"], sm["sensor"])
                for e in EST_STYLE:
                    abs_rows += [
                        dict(sensor=sid, estimator=e, kind="absolute, all pairs pooled",
                             slope=est_p[e][0], offset=est_p[e][1], lo=ci[("pooled", e)][0],
                             hi=ci[("pooled", e)][1], n=len(kp)),
                        dict(sensor=sid, estimator=e, kind="absolute, scene medians",
                             slope=est_s[e][0], offset=est_s[e][1], lo=ci[("scene", e)][0],
                             hi=ci[("scene", e)][1], n=len(sm)),
                        dict(sensor=sid, estimator=e,
                             kind="within-scene anomalies (reference)",
                             slope=estimates(kp)[e], offset=np.nan, lo=ci_w[e][0],
                             hi=ci_w[e][1], n=len(kp))]
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "slope_estimates.csv", index=False)
    estimates_figure(df, out_dir / "slope_estimates.png", args.dpi)
    adf = pd.DataFrame(abs_rows)
    adf.to_csv(out_dir / "slope_absolute_estimates.csv", index=False)
    absolute_estimates_figure(adf, out_dir / "slope_absolute_estimates.png", args.dpi)
    with pd.option_context("display.width", 200):
        log.info("\n%s", adf.round(3).to_string(index=False))
    with pd.option_context("display.width", 200):
        log.info("\n%s", df.round(3).to_string(index=False))
    log.info("figures in %s (pixel masks: %s)", out_dir, source)


if __name__ == "__main__":
    main()
