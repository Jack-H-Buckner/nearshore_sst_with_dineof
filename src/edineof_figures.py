"""Figures for the eDINEOF reconstruction: how k and p were chosen, and what came out.

Three views, all optional via `output.figure_*`:

  cv       -- the two cross-validation curves. The left panel selects k on the point holdout,
              the right selects p on the day holdout, and the right panel carries the number
              that decides whether this stage was worth running: the day-CV at p=0, which IS
              the per-pixel seasonal climatology stage 5 removed. Every other p has to beat it.
  modes    -- the leading spatial modes beside their temporal amplitudes. Diverging colour with
              a NEUTRAL midpoint, because an EOF is signed and zero is a real reference point,
              not an arbitrary end of a ramp.
  contact  -- every date of the reconstruction on one sheet, in time order, with the dates that
              have no observation of their own marked. Those are the ones resting entirely on
              the temporal filter, and they are not validated by either CV set.

Not a script; imported by edineof.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import plotting

log = logging.getLogger("edineof_figures")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"

# Categorical slots 1-3 of the validated reference palette, as elsewhere in this prototype.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#e8e8e6", "#eb6834"])

HEADER_IN = 0.85
FOOTER_IN = 0.72


def _axis(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(labelsize=8, colors=INK_SECONDARY)


def cv_figure(cfg: dict, res: dict, scale_k: float, path: Path) -> None:
    """The two CV curves: k from the point holdout, p from the day holdout.

    Two panels rather than one axis with two y-scales -- a dual-axis chart invites reading a
    crossing as meaningful when the two series share no units. They also answer different
    questions, so separating them is honest about that.
    """
    curve = res["curve"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.9), dpi=cfg["output"]["figure_dpi"],
                             facecolor=SURFACE)

    ax = axes[0]
    _axis(ax)
    for i, tc in enumerate(sorted(curve["t_c"].unique())):
        s = curve[curve["t_c"] == tc].sort_values("k")
        sel = tc == res["tc_opt"]
        ax.plot(s["k"], s["rmse_point"] * scale_k, linewidth=2 if sel else 1.2,
                color=SERIES[i % len(SERIES)], alpha=1.0 if sel else 0.55,
                label=f"T_c={tc:g}d" + ("  (selected)" if sel else ""))
    ax.axvline(res["k_opt"], color=INK_MUTED, linewidth=1.5, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate(f"k = {res['k_opt']}", (res["k_opt"], ax.get_ylim()[1]),
                textcoords="offset points", xytext=(5, -12), fontsize=8, color=INK_SECONDARY)
    ax.set_xlabel("modes k", fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel("point-holdout CV RMSE [K]", fontsize=9, color=INK_SECONDARY)
    ax.set_title("k is selected here: scattered points inside days that still have data",
                 fontsize=10.5, color=INK, loc="left")
    ax.legend(fontsize=7.5, frameon=False, ncol=2)

    ax = axes[1]
    _axis(ax)
    per_tc = res["per_tc"]
    ps = sorted(res["day_at"])
    vals = [res["day_at"][x] * scale_k for x in ps]
    tc = list(ps)

    base = res["day_at"].get(0.0)
    if base is not None:
        ax.axhline(base * scale_k, color=INK_MUTED, linewidth=2, linestyle=(0, (5, 3)),
                   zorder=2, label="p=0: the seasonal climatology")
    ax.plot(tc, vals, color=SERIES[0], linewidth=2, zorder=3)
    ax.scatter(tc, vals, s=58, facecolor=SERIES[0], edgecolor=SURFACE, linewidth=1.5, zorder=4)
    sel = ps.index(res["tc_opt"])
    ax.scatter([tc[sel]], [vals[sel]], s=150, marker="o", facecolor="none",
               edgecolor=INK, linewidth=2, zorder=5,
               label=f"selected T_c={res['tc_opt']:g}d")
    for x, y, kk in zip(tc, vals, [per_tc[x] for x in ps]):
        ax.annotate(f"k={kk}", (x, y), textcoords="offset points", xytext=(0, 9),
                    fontsize=7, color=INK_SECONDARY, ha="center")
    ax.set_xlabel("filter cutoff period  T_c = 2*pi*sqrt(alpha*p)  [days]", fontsize=9,
                  color=INK_SECONDARY)
    ax.set_ylabel("day-holdout CV RMSE [K]", fontsize=9, color=INK_SECONDARY)
    ax.set_title("p is selected here: whole days held out, reconstructed by the filter alone",
                 fontsize=10.5, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="best")

    gain = ""
    if base is not None and base > 0:
        g = 100 * (1 - res["day_at"][res["tc_opt"]] / base)
        gain = (f"   |   the filter beats climatology by {g:.1f}%"
                if g > 0 else "   |   THE FILTER BEATS NOTHING: no better than climatology")
    fig.suptitle(
        f"Cross-validation  --  {cfg['data']['aoi']}  --  k={res['k_opt']}, "
        f"T_c={res['tc_opt']:g} d (alpha={res['alpha_opt']:.4g}, p={res['p_opt']}){gain}\n"
        "two holdouts because the point set cannot see what the filter does: its pixels sit in "
        "days whose modes are already pinned by their own data",
        fontsize=11, color=INK, x=0.006, ha="left", y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    plotting.save(fig, path, dpi=cfg["output"]["figure_dpi"])


def convergence_figure(cfg: dict, res: dict, sel: dict, scale_k: float, path: Path) -> None:
    """How the final fit converged, split by whether a date had any observation of its own.

    THE SPLIT IS THE POINT. A date with pixels has its temporal mode pinned by its own data and
    settles quickly. A date with none is filled entirely by the filter diffusing in from its
    neighbours, one step per EM iteration, so it converges on the diffusion timescale instead.
    Pooling the two reports the slow half as if it were the whole field, which is what the
    single `delta` number does -- and it is why `em.tol` looks unreachable when most of the
    field has long since settled.

    Panel 3 is the one to read for "will more iterations help": a contraction rate at 1.0 is a
    stalled iterate, and no budget rescues it. Below 1.0 it is descending and the annotation
    projects how far there is to go.
    """
    h = res["history"]
    d = np.asarray(h["deltas"], float)
    if d.size < 2:
        return
    it = np.arange(1, d.size + 1)
    tol = float(cfg["em"]["tol"])
    alpha, p = res["alpha_opt"], res["p_opt"]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9), dpi=cfg["output"]["figure_dpi"],
                             facecolor=SURFACE)

    ax = axes[0]
    _axis(ax)
    ax.axhline(tol, color=INK_MUTED, linewidth=2, linestyle=(0, (5, 3)), zorder=2,
               label=f"em.tol = {tol:g}")
    for key, color, lab in (("deltas_seen", SERIES[2], "dates WITH data"),
                            ("deltas_empty", SERIES[1], "dates with NO data (filter only)"),
                            ("deltas", SERIES[0], "all gaps (the stopping metric)")):
        v = np.asarray(h.get(key, []), float)
        if v.size == d.size and np.isfinite(v).any():
            ax.plot(it, v, color=color, linewidth=2.2 if key == "deltas" else 1.6,
                    zorder=4 if key == "deltas" else 3, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("EM iteration", fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel("RMS change over gaps / sd", fontsize=9, color=INK_SECONDARY)
    ax.set_title("convergence, split by whether the date has data", fontsize=10.5,
                 color=INK, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")

    ax = axes[1]
    _axis(ax)
    # Split the same way as panel 1: a date with no pixels of its own converges on the
    # filter's diffusion timescale, so pooling it with the rest hides which half is still
    # moving -- and it is the kelvin number that says whether "still moving" matters at all.
    for key, color, lab, lw in (
            ("deltas_abs_seen", SERIES[2], "dates WITH data", 1.6),
            ("deltas_abs_empty", SERIES[1], "dates with NO data (filter only)", 1.6),
            ("deltas_abs", SERIES[0], "all gaps", 2.2)):
        v = np.asarray(h.get(key, []), float)
        if v.size == d.size and np.isfinite(v).any():
            ax.plot(it, v * scale_k, color=color, linewidth=lw,
                    zorder=4 if key == "deltas_abs" else 3, label=lab)
            ax.annotate(f"{v[-1] * scale_k:.4f} K", (it[-1], v[-1] * scale_k),
                        textcoords="offset points", xytext=(6, 0), fontsize=7.5,
                        color=color, va="center")
    ax.set_yscale("log")
    ax.set_xlim(right=it[-1] * 1.18)          # room for the direct labels
    ax.set_xlabel("EM iteration", fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel("mean |change| per gap pixel  [K]", fontsize=9, color=INK_SECONDARY)
    ax.set_title("the same thing in kelvin  (is it physically settled?)", fontsize=10.5,
                 color=INK, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")

    ax = axes[2]
    _axis(ax)
    r = d[1:] / np.maximum(d[:-1], 1e-300)
    ax.axhline(1.0, color=INK_MUTED, linewidth=2, zorder=2, label="1.0 = stalled")
    ax.plot(it[1:], r, color=SERIES[2], linewidth=1.1, alpha=0.7, zorder=3)
    win = 20
    if r.size > win:
        ax.plot(it[win:], np.convolve(r, np.ones(win) / win, mode="valid"),
                color="#08382a", linewidth=2, zorder=4, label=f"{win}-iteration mean")
    ax.set_ylim(0.9, 1.06)
    ax.set_xlabel("EM iteration", fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel("delta[i] / delta[i-1]", fontsize=9, color=INK_SECONDARY)
    ax.set_title("contraction rate  (rho)", fontsize=10.5, color=INK, loc="left")
    rho = float(np.median(r[-100:])) if r.size > 100 else float(np.median(r))
    note = f"tail rho = {rho:.4f}"
    if 0 < rho < 1 and d[-1] > tol:
        note += f"\n-> ~{int(np.log(tol / d[-1]) / np.log(rho)):,} more iterations to tol"
    ax.annotate(note, (0.97, 0.06), xycoords="axes fraction", fontsize=8,
                color="#08382a", ha="right")
    ax.legend(fontsize=8, frameon=False, loc="upper right")

    n_e = int(sel["observed"].sum(axis=0).size - (sel["observed"].sum(axis=0) > 0).sum())
    fig.suptitle(
        f"Convergence of the final fit  --  {cfg['data']['aoi']}  --  k={res['k_opt']}, "
        f"alpha={alpha:g}, p={p} (T_c = {2 * np.pi * np.sqrt(alpha * p):.2f} d), "
        f"{d.size} iterations, {n_e} of {sel['n']} dates with no data of their own",
        fontsize=11.5, color=INK, x=0.006, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    plotting.save(fig, path, dpi=cfg["output"]["figure_dpi"])


def modes_figure(cfg: dict, res: dict, sel: dict, ds, extent, path: Path, n_show: int = 4
                 ) -> None:
    """Leading spatial modes over their temporal amplitudes."""
    water, keep = sel["water"], sel["keep"]
    k = min(n_show, res["k_opt"])
    if k == 0:
        return
    times = pd.to_datetime(ds["time"].values)
    var = res["sigma"] ** 2
    frac = var / max(var.sum(), 1e-300)

    fig, axes = plt.subplots(2, k, figsize=(3.6 * k, 7.4), dpi=cfg["output"]["figure_dpi"],
                             facecolor=SURFACE, squeeze=False,
                             gridspec_kw=dict(height_ratios=[1.45, 1.0], hspace=0.28))
    for j in range(k):
        g = np.full(water.shape, np.nan)
        tmp = np.full(int(water.sum()), np.nan)
        tmp[keep] = res["U"][:, j]
        g[water] = tmp

        ax = axes[0, j]
        ax.set_facecolor(plotting.NODATA_COLOR)
        plotting.flat(ax, ~water, plotting.LAND_COLOR, extent)
        v = np.nanpercentile(np.abs(g), 99)
        im = ax.imshow(g, cmap=DIV, vmin=-v, vmax=v, extent=extent, origin="upper",
                       interpolation="nearest")
        ax.set_title(f"mode {j + 1}  --  {100 * frac[j]:.1f}% of variance", fontsize=9.5,
                     color=INK, loc="left", pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=7, colors=INK_SECONDARY)
        cb.outline.set_visible(False)

        ax = axes[1, j]
        _axis(ax)
        ax.axhline(0, color=INK_MUTED, linewidth=1.0)
        ax.plot(times, res["V"][:, j], color=SERIES[j % len(SERIES)], linewidth=1.4)
        ax.set_title("temporal amplitude", fontsize=8.5, color=INK_SECONDARY, loc="left")
        ax.tick_params(axis="x", labelsize=7, rotation=30)

    fig.suptitle(
        f"Leading modes  --  {cfg['data']['aoi']}  --  k={res['k_opt']}, "
        f"p={res['p_opt']}; these are modes of the STANDARDIZED anomaly, so a map is in units "
        "of local sd -- multiply by sst_seasonal_sd for kelvin",
        fontsize=11, color=INK, x=0.006, ha="left", y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plotting.save(fig, path, dpi=cfg["output"]["figure_dpi"])


def contact_figure(cfg: dict, res: dict, sel: dict, recon: np.ndarray, ds, extent,
                   path: Path) -> None:
    """Every reconstructed date, chronological, on one shared colour scale."""
    water = sel["water"]
    times = pd.to_datetime(ds["time"].values)
    n = recon.shape[0]
    constrained = sel["observed"].sum(axis=0) > 0

    field = recon - 273.15 if np.nanmedian(recon) > 100 else recon
    pool = field[np.isfinite(field) & water[None, :, :]]
    vmin, vmax = float(np.percentile(pool, 1)), float(np.percentile(pool, 99))

    ncols = int(cfg["output"]["figure_ncols"])
    nrows = int(np.ceil(n / ncols))
    fig_h = 1.75 * nrows + HEADER_IN + FOOTER_IN
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.45 * ncols, fig_h), dpi=130,
                             facecolor=SURFACE, gridspec_kw=dict(hspace=0.42, wspace=0.05))
    axes = np.atleast_1d(axes).ravel()
    cmap = plotting.sst_cmap()

    im = None
    for i in range(n):
        ax = axes[i]
        im = plotting.panel(ax, field[i], water, vmin=vmin, vmax=vmax, extent=extent, cmap=cmap)
        ax.set_title(str(times[i])[5:10], fontsize=5.5, pad=2,
                     color=INK if constrained[i] else INK_MUTED)
        ax.set_xticks([])
        ax.set_yticks([])
        if not constrained[i]:
            # No observation of its own: the whole day came from the filter. Dashed, not
            # coloured -- a caveat, not a verdict.
            for s in ax.spines.values():
                s.set_color(INK_MUTED)
                s.set_linewidth(1.0)
                s.set_linestyle((0, (2, 1.5)))
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        f"{cfg['data']['aoi']} -- eDINEOF reconstruction, {n} days, k={res['k_opt']} "
        f"p={res['p_opt']}\ndashed frame = no observation that day ({int((~constrained).sum())} "
        "of them): reconstructed entirely by the temporal filter, and scored by neither CV set",
        fontsize=10, color=INK, y=1 - 0.20 / fig_h)
    fig.subplots_adjust(top=1 - HEADER_IN / fig_h, bottom=FOOTER_IN / fig_h,
                        left=0.015, right=0.985)
    cax = fig.add_axes((0.36, 0.46 / fig_h, 0.28, 0.10 / fig_h))
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", label="SST [degC]")
    cb.ax.tick_params(labelsize=8, colors=INK_SECONDARY)
    plotting.save(fig, path)


def render(cfg: dict, res: dict, sel: dict, recon: np.ndarray, ds, extent,
           out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    o = cfg["output"]
    # CV errors are in standardized units; the median per-pixel scale converts them to the
    # kelvin figure anyone would actually quote. Taken from `sel`, not the cube: under
    # matrix.coarsen the cube is still at source resolution while `sel["water"]` is the
    # coarsened mask, and indexing one with the other is a shape error.
    scale_k = float(np.nanmedian(sel["scale"][sel["water"]]))
    if o["figure_cv"]:
        cv_figure(cfg, res, scale_k, out_dir / "cv_curves.png")
    if o["figure_convergence"]:
        convergence_figure(cfg, res, sel, scale_k, out_dir / "convergence.png")
    if o["figure_modes"]:
        modes_figure(cfg, res, sel, ds, extent, out_dir / "modes.png")
    if o["figure_contact"]:
        contact_figure(cfg, res, sel, recon, ds, extent, out_dir / "contact_reconstruction.png")
