"""Figures for the iterative DINEOF-baseline filter.

Two views:

  convergence -- flip fraction and kept-pixel count per sensor against iteration. The loop's
                 own health check: flips should fall toward zero, kept counts should settle.
  days        -- for the busiest acquisitions, read left to right: the raw scene, the DINEOF
                 baseline, the residual, q_cloud, and the final verdict against the
                 MODIS-baseline filter's, as four flat categories.

Not a script; imported by iterative_filter.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import plotting
from cube_figures import DROP_PIXEL, GRID, INK, INK_SECONDARY, SURFACE, VALID_COLOR

log = logging.getLogger("iterative_figures")

ITER_ONLY = plotting.PASS_COLOR     # kept here, removed by the MODIS-baseline filter
MODIS_ONLY = "#f08c00"              # removed here, kept by the MODIS-baseline filter
SENSOR_COLORS = ["#1971c2", "#c2255c", "#2f9e44", "#f08c00"]


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.6)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=7)


def convergence_figure(history, sids, out: Path, dpi: int) -> None:
    h = history[history["iter"] != "final_cv"]
    it = h["iter"].astype(int).to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=dpi)
    for i, sid in enumerate(sids):
        c = SENSOR_COLORS[i % len(SENSOR_COLORS)]
        f = h[f"flip_frac_{sid}"].to_numpy(float)
        axes[0].semilogy(it, np.maximum(f, 1e-7), "o-", color=c, label=sid)
        axes[1].plot(it, h[f"n_kept_{sid}"].to_numpy(float), "o-", color=c, label=sid)
    axes[0].set_title("pixels whose verdict flipped (fraction of observed)", fontsize=9,
                      color=INK)
    axes[1].set_title("pixels kept after the iteration", fontsize=9, color=INK)
    for ax in axes:
        _style(ax)
        ax.set_xlabel("iteration (the fit whose baseline classified)", fontsize=8)
        ax.legend(fontsize=7)
    fig.suptitle(f"strategy: {h['strategy'].iloc[0]}", fontsize=9, color=INK)
    fig.tight_layout()
    plotting.save(fig, out)


def day_figure(inp, sid, j, info, base, keep, modis_keep, extent, out: Path, dpi: int) -> None:
    water = inp.water
    raw = inp.raw[sid][j] - 273.15
    b = base[j] - 273.15
    obs = np.isfinite(raw) & water
    vals = np.concatenate([raw[obs], b[obs]])
    vmin, vmax = (np.percentile(vals, [2, 98]) if vals.size else (0.0, 1.0))
    res = info["residual"]
    rl = float(np.nanpercentile(np.abs(res[obs]), 98)) if obs.any() else 1.0

    fig, axes = plt.subplots(1, 5, figsize=(21, 4.6), dpi=dpi)
    im = plotting.panel(axes[0], raw, water, vmin=vmin, vmax=vmax, extent=extent,
                        title="raw scene")
    fig.colorbar(im, ax=axes[0], shrink=0.8, label="SST [degC]")
    im = plotting.panel(axes[1], b, water, vmin=vmin, vmax=vmax, extent=extent,
                        title="DINEOF low-rank baseline")
    fig.colorbar(im, ax=axes[1], shrink=0.8, label="SST [degC]")
    cm = plt.get_cmap("RdBu_r").copy()
    cm.set_bad(alpha=0.0)
    im = plotting.panel(axes[2], np.where(obs, res, np.nan), water, vmin=-rl, vmax=rl,
                        extent=extent, cmap=cm,
                        title=f"residual (centre {info['center']:+.2f} K removed)")
    fig.colorbar(im, ax=axes[2], shrink=0.8, label="K")
    cq = plt.get_cmap("magma").copy()
    cq.set_bad(alpha=0.0)
    im = plotting.panel(axes[3], info["q_cloud"], water, vmin=0, vmax=1, extent=extent,
                        cmap=cq, title="q_cloud")
    fig.colorbar(im, ax=axes[3], shrink=0.8)

    a = keep[j] & obs
    m = (modis_keep[j] & obs) if modis_keep is not None else np.zeros_like(a)
    ax = axes[4]
    ax.set_facecolor(plotting.NODATA_COLOR)
    plotting.flat(ax, ~water, plotting.LAND_COLOR, extent)
    plotting.flat(ax, a & m, VALID_COLOR, extent)
    plotting.flat(ax, obs & ~a & ~m, DROP_PIXEL, extent)
    plotting.flat(ax, a & ~m, ITER_ONLY, extent)
    plotting.flat(ax, ~a & m, MODIS_ONLY, extent)
    n = max(int(obs.sum()), 1)
    ax.set_title("final keep vs MODIS-baseline keep", fontsize=8)
    ax.legend(handles=[
        Patch(facecolor=VALID_COLOR, label=f"both keep {(a & m).sum() / n:.0%}"),
        Patch(facecolor=DROP_PIXEL, label=f"both remove {(obs & ~a & ~m).sum() / n:.0%}"),
        Patch(facecolor=ITER_ONLY, label=f"iterative only {(a & ~m).sum() / n:.0%}"),
        Patch(facecolor=MODIS_ONLY, label=f"MODIS only {(~a & m).sum() / n:.0%}")],
        loc="lower left", fontsize=6, framealpha=0.9)
    for ax in axes:
        ax.tick_params(labelsize=6)
    fig.suptitle(f"{sid}  {str(inp.times[j])[:10]}  --  {int(obs.sum()):,} observed water px",
                 fontsize=9, ha="left", x=0.01, color=INK)
    fig.tight_layout()
    plotting.save(fig, out)


def render(inp, cfg, out, modis_keep, fig_scenes, out_dir: Path) -> None:
    dpi = int(cfg["output"]["figure_dpi"])
    sids = list(inp.raw)
    convergence_figure(out["history"], sids, out_dir / "convergence.png", dpi)
    xs, ys = inp.coords.get("x"), inp.coords.get("y")
    extent = ([0.0, float(xs.max() - xs.min()) / 1000.0, 0.0, float(ys.max() - ys.min()) / 1000.0]
              if xs is not None and ys is not None else None)
    base = out["last"]["base"]
    for sid, j in fig_scenes:
        info = out["figs"].get((sid, j))
        if info is None:
            continue
        day_figure(inp, sid, j, info, base, out["keep"][sid],
                   modis_keep[sid] if modis_keep is not None else None, extent,
                   out_dir / f"day_{str(inp.times[j])[:10]}_{sid}.png", dpi)
