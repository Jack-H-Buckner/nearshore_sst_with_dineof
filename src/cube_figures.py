"""Figures for the filtered DINEOF cube: which pixels and which images got flagged.

Three views, each answering a different question, all optional via `output.figure_*`:

  scenes   -- per acquisition, four panels read left to right: the raw field on its own full
              range, the same field on the clean range with removals painted over it, the
              p_valid evidence, and what survives. This is the per-PIXEL view.
  contact  -- one sheet per sensor, every scene sorted by its offset, framed by verdict. The
              band boundary reads as a seam you can move. This is the per-IMAGE view.
  offsets  -- offset against kept fraction, one point per scene, with the accepted band
              shaded. The scene gate's tuning instrument.

VERDICT COLOURS ARE NEVER ALONE. The status pair (#2f9e44 kept / #c92a2a dropped) separates
well for normal vision (OKLab dE 31.8) and protanopia (21.2) but only 6.1 under deuteranopia,
below the 8 that would make colour sufficient on its own. So every verdict is carried three
ways -- colour, marker or frame, AND a literal KEPT/DROPPED word -- and the figures stay
readable in greyscale. The same holds for the dropped-pixel overlay, which is a hatch as well
as a hue. `filter_report.csv` is the table view of everything drawn here.

Not a script; imported by build_cube.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import plotting

log = logging.getLogger("cube_figures")

# The repo's chart palette (prototypes/cloud_mixture_model/src/compare_modis_eco.py).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"

KEPT_COLOR = plotting.PASS_COLOR      # #2f9e44
DROP_COLOR = plotting.FAIL_COLOR      # #c92a2a
DROP_PIXEL = plotting.INVALID_COLOR   # #c2255c -- a pixel the filter removed
VALID_COLOR = plotting.VALID_COLOR    # #1971c2 -- a pixel the filter kept

HEADER_IN = 0.85
FOOTER_IN = 0.72


def _kelvin_offset(raw: np.ndarray) -> float:
    """273.15 if the stack is Kelvin, else 0.

    Decided once per sensor on the RAW stack and applied to both fields, so the filtered
    stack -- which may be mostly NaN -- cannot land on the other side of the test and put the
    two panels on different scales.
    """
    finite = raw[np.isfinite(raw)]
    return 273.15 if finite.size and np.median(finite) > 150.0 else 0.0


def _limits(filt: np.ndarray, raw: np.ndarray, lo: float, hi: float) -> tuple[float, float]:
    """Colour range from the KEPT pixels: by construction the ones the filter calls clean.

    Falls back to the raw field only when a sensor kept nothing at all, so an empty run still
    draws something rather than raising.
    """
    vals = filt[np.isfinite(filt)]
    if vals.size < 64:
        vals = raw[np.isfinite(raw)]
    if vals.size == 0:
        return 0.0, 1.0
    vmin, vmax = float(np.percentile(vals, lo)), float(np.percentile(vals, hi))
    return (vmin, vmin + 1.0) if vmax <= vmin else (vmin, vmax)


def _verdict_bits(kept: bool) -> tuple[str, str, str]:
    """(word, colour, marker). The word and the marker are what make the colour redundant."""
    return ("KEPT", KEPT_COLOR, "o") if kept else ("DROPPED", DROP_COLOR, "X")


# --------------------------------------------------------------------------- per scene

def _scene_limits(raw2d, p_valid, water, p_min, lo=2.0, hi=98.0) -> tuple[float, float]:
    """Colour range for ONE scene, from the pixels that scene's pixel cut would keep.

    Per scene rather than shared, unlike the contact sheet. The comparison this figure exists
    to support is raw-against-filtered WITHIN the frame, and a range pooled over every scene
    in the run compresses each one into a narrow band of the ramp -- on the March sample the
    pooled range was 2.6 K wide and every panel rendered near-uniformly dark. The contact
    sheet keeps the shared scale, which is where cross-scene comparison belongs.
    """
    sel = (p_valid >= p_min) & water & np.isfinite(raw2d)
    vals = raw2d[sel]
    if vals.size < 64:
        vals = raw2d[water & np.isfinite(raw2d)]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    vmin, vmax = float(np.percentile(vals, lo)), float(np.percentile(vals, hi))
    return (vmin, vmin + 1.0) if vmax <= vmin else (vmin, vmax)


def _raw_limits(raw2d, water, lo=2.0, hi=98.0) -> tuple[float, float]:
    """Colour range over EVERY observed water pixel, contamination included.

    The clean-scale range deliberately pushes contamination off the cold end, which is right
    for judging the surviving field but useless for looking at the raw one -- a heavily hazed
    scene would render as a saturated blob. This range is what the sensor actually recorded.
    """
    vals = raw2d[water & np.isfinite(raw2d)]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    vmin, vmax = float(np.percentile(vals, lo)), float(np.percentile(vals, hi))
    return (vmin, vmin + 1.0) if vmax <= vmin else (vmin, vmax)


def residual_hist(ax, residual, removed, water, sd, *, compact: bool = False) -> None:
    """Stacked histogram of the residual against the covariate, kept vs removed.

    This is the classifier's own decision space, and the point of drawing it is that the split
    is NOT a clean threshold in it. The mixture reads a per-pixel QC prior and then runs 20
    Ising sweeps, so a pixel's neighbours help decide it -- the two colours overlap, and how
    much they overlap is the honest picture of how spatially-driven the call was. A hard cut
    would mean the spatial term was doing nothing.

    The shaded band is +/- the clear-sky sigma the detector assumed. Residuals well outside it
    that are still kept, or inside it and removed, are the pixels worth arguing about.

    kept = #1971c2, removed = #c2255c: OKLab dE 28.9 normal, 17.4 protan, 19.1 deutan, so this
    pair is safe on colour alone, unlike the green/red verdict pair elsewhere in this module.
    """
    obs = water & np.isfinite(residual)
    if obs.sum() < 50:
        ax.set_axis_off()
        return
    r = residual[obs]
    lo, hi = np.percentile(r, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(r)), float(np.nanmin(r)) + 1.0
    pad = 0.05 * (hi - lo)
    edges = np.linspace(lo - pad, hi + pad, (40 if compact else 70) + 1)

    kept_r = residual[obs & ~removed]
    rem_r = residual[obs & removed]
    # Values beyond the axis are CLIPPED INTO the end bins rather than dropped, so the bar
    # heights still sum to every observed pixel. Those two bars are overflow, not a real mode
    # -- cloud has a long cold tail and it would otherwise set the range for the whole plot.
    n_over = int((r < edges[0]).sum() + (r > edges[-1]).sum())
    ax.hist([np.clip(kept_r, edges[0], edges[-1]), np.clip(rem_r, edges[0], edges[-1])],
            bins=edges, stacked=True, color=[VALID_COLOR, DROP_PIXEL],
            label=[f"kept ({kept_r.size:,})", f"removed ({rem_r.size:,})"],
            edgecolor="none")
    if n_over:                       # mark the overflow bins so they cannot be misread
        for x in (edges[0], edges[-1]):
            ax.axvline(x, color=INK_MUTED, linewidth=0.8, linestyle=(0, (2, 2)), zorder=4)

    if sd is not None and np.isfinite(sd):
        ax.axvspan(-sd, sd, color=INK_MUTED, alpha=0.13, zorder=0)
    ax.axvline(0.0, color=INK, linewidth=1.0, zorder=3)

    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    if compact:
        ax.set_yticks([])
        ax.tick_params(labelsize=5, colors=INK_SECONDARY)
    else:
        note = f"\n{n_over:,} px beyond the axis sit in the dashed end bars" if n_over else ""
        ax.set_xlabel(f"residual vs covariate [K]{note}", fontsize=7, color=INK_SECONDARY)
        # No y-axis label: the count is already in the legend, and the label collides with the
        # colour bar of the map panel to its left.
        ax.tick_params(labelsize=7, colors=INK_SECONDARY, left=False, labelleft=False)
        ax.legend(fontsize=6.5, frameon=False, loc="upper left")
        ax.set_title("residual distribution"
                     + (f"\nshaded = +/-{sd:.2f} K assumed clear-sky sd" if sd else ""),
                     fontsize=8)


def scene_figure(raw2d, filt2d, p_valid, water, row, s, cfg, extent, out_dir: Path,
                 residual=None, sd=None) -> None:
    """One acquisition, read left to right: what the sensor recorded, what the filter targets,
    the evidence it used, and what DINEOF gets.

    The raw panel carries its OWN colour range, spanning every observed pixel, while the other
    three share the clean range taken from the pixels that pass the cut. Two scales, each
    labelled with its own span, because they answer different questions: the raw panel shows
    what was actually recorded (contamination and all), and a clean-scaled copy of it would
    saturate to a single colour on exactly the hazy scenes worth inspecting. Panel 2 is the
    same array on the clean scale with the removals painted on, so the two sit side by side.
    """
    kept = row["verdict"] == "KEPT"
    word, color, _ = _verdict_bits(kept)
    p_min = float(cfg["filter"]["p_valid_min"])
    n_water = int(water.sum())
    vmin, vmax = _scene_limits(raw2d, p_valid, water, p_min)
    rmin, rmax = _raw_limits(raw2d, water)

    observed = water & np.isfinite(raw2d)
    # On a dropped scene nothing reaches the cube, so `filt2d` is empty and the removal mask
    # would be "everything". Show what the PIXEL cut alone would have taken, so the figure
    # still says something about the scene rather than going uniformly magenta.
    removed = observed & ~(p_valid >= p_min)

    show_hist = residual is not None and cfg["output"].get("figure_scene_hist", True)
    n_panels = 5 if show_hist else 4
    # The histogram is not a map, so it gets a slightly narrower column than the square panels.
    fig, axes = plt.subplots(1, n_panels, dpi=cfg["output"]["figure_dpi"],
                             figsize=(16.6 + (3.6 if show_hist else 0), 4.7),
                             gridspec_kw=dict(width_ratios=[1, 1, 1, 1] + ([0.92] if show_hist else [])))
    cmap = plotting.sst_cmap()

    im0 = plotting.panel(axes[0], raw2d, water, vmin=rmin, vmax=rmax, extent=extent, cmap=cmap,
                         title=f"{s['sst']} (raw, unfiltered)\nfull range {rmin:.1f}-{rmax:.1f} degC")
    fig.colorbar(im0, ax=axes[0], shrink=0.80, label="SST [degC]")

    im = plotting.panel(axes[1], raw2d, water, vmin=vmin, vmax=vmax, extent=extent, cmap=cmap,
                        title=f"raw + removals  --  "
                              f"{removed.sum() / max(observed.sum(), 1):.0%} of observed "
                              f"water removed\nclean range {vmin:.1f}-{vmax:.1f} degC")
    plotting.flat(axes[1], removed, DROP_PIXEL, extent)
    fig.colorbar(im, ax=axes[1], shrink=0.80, label="SST [degC]")
    axes[1].legend(handles=[Patch(facecolor=DROP_PIXEL, label=f"p_valid < {p_min:.2f}")],
                   loc="lower left", fontsize=6, framealpha=0.9)

    cmp_p = plt.get_cmap("viridis").copy()
    cmp_p.set_bad(alpha=0.0)
    imp = plotting.panel(axes[2], p_valid, water, vmin=0.0, vmax=1.0, extent=extent, cmap=cmp_p,
                         title=f"p_valid  (threshold {p_min:.2f})")
    cb = fig.colorbar(imp, ax=axes[2], shrink=0.80, label="P(clear)")
    cb.ax.axhline(p_min, color=INK, linewidth=1.4)      # the cut, drawn on the bar itself

    fourth = filt2d if kept else np.full_like(raw2d, np.nan)
    im3 = plotting.panel(axes[3], fourth, water, vmin=vmin, vmax=vmax, extent=extent, cmap=cmap,
                         title=f"{s['sst']}{cfg['output']['filtered_suffix']}"
                               + ("  (DINEOF input)" if kept else "  -- EMPTY, scene dropped")
                               + f"\nclean range {vmin:.1f}-{vmax:.1f} degC")
    fig.colorbar(im3, ax=axes[3], shrink=0.80, label="SST [degC]")

    for ax in axes[:4]:
        ax.set_xlabel("km east", fontsize=7)
        ax.tick_params(labelsize=6)
    axes[0].set_ylabel("km north", fontsize=7)

    if show_hist:
        residual_hist(axes[4], residual, removed, water, sd)

    tail = "kept in full" if kept and row["reason"] == "kept" else row["reason"]
    head = (f"{s['label']}   {row['date']}   offset {row['center']:+.2f} K\n"
            f"{row['pixel_px']:,} of {n_water:,} water px pass the pixel cut "
            f"({row['pixel_frac']:.1%})   --   {word}: {tail}")
    fig.suptitle(head, fontsize=9, ha="left", x=0.01, y=0.985)
    plotting.stamp(fig, kept, words=("KEPT", "DROPPED"))
    fig.subplots_adjust(top=0.82, bottom=0.14, left=0.04, right=0.985,
                        wspace=0.24 if show_hist else 0.16)

    stem = f"{'kept' if kept else 'dropped'}_{row['date']}_{row['sensor']}"
    plotting.save(fig, out_dir / f"{stem}.png")


# --------------------------------------------------------------------------- contact sheet

def contact_sheet(raw, removed, water, report, s, cfg, extent, vmin, vmax, out_path: Path,
                  aoi: str) -> None:
    """Every scene for one sensor, sorted by offset, framed by verdict.

    Sorted by the quantity the SCENE gate reads, so the accepted band shows up as a contiguous
    green run with a red block at each end -- move `filter.offset_lower/upper` and the seams
    move. A red panel inside the run is a scene the band accepted but the pixel cut emptied.
    """
    n = len(report)
    if not n:
        log.warning("%s: no scenes to draw", s["sst"])
        return

    order = report["center"].to_numpy().argsort()
    ncols = int(cfg["output"]["figure_ncols"])
    nrows = int(np.ceil(n / ncols))
    fig_h = 2.30 * nrows + HEADER_IN + FOOTER_IN
    cmap = plotting.sst_cmap()

    fig, axes = plt.subplots(nrows, ncols, figsize=(1.85 * ncols, fig_h), dpi=150,
                             gridspec_kw=dict(hspace=0.46, wspace=0.06))
    axes = np.atleast_1d(axes).ravel()

    im = None
    for ax, i in zip(axes, order):
        r = report.iloc[i]
        kept = r["verdict"] == "KEPT"
        word, color, _ = _verdict_bits(kept)
        # Draw the RAW field with removals painted on: an all-NaN dropped scene would
        # otherwise be a blank square that says nothing about why it went.
        im = plotting.panel(ax, raw[r["t"]], water, vmin=vmin, vmax=vmax, extent=extent,
                            cmap=cmap)
        plotting.flat(ax, removed[r["t"]], DROP_PIXEL, extent)
        ax.set_title(f"{r['date'][5:]}  {r['center']:+.1f}K\n"
                     f"{r['pixel_frac']:.0%} clear  {word}",
                     fontsize=6, pad=3, color=color)
        ax.set_xticks([])
        ax.set_yticks([])
        plotting.frame_verdict(ax, kept)
    for ax in axes[n:]:
        ax.axis("off")

    lo, hi = cfg["filter"]["offset_lower"], cfg["filter"]["offset_upper"]
    n_kept = int(report["verdict"].eq("KEPT").sum())
    fig.suptitle(
        f"{aoi} -- {s['label']}: {n_kept} of {n} scenes kept "
        f"(offset band [{lo}, {hi}] K, p_valid >= {cfg['filter']['p_valid_min']})\n"
        f"sorted by offset; green frame = KEPT, red = DROPPED; "
        f"magenta = pixels the filter removed",
        fontsize=10, y=1 - 0.20 / fig_h)
    fig.subplots_adjust(top=1 - HEADER_IN / fig_h, bottom=FOOTER_IN / fig_h,
                        left=0.015, right=0.985)
    cax = fig.add_axes((0.36, 0.46 / fig_h, 0.28, 0.10 / fig_h))
    fig.colorbar(im, cax=cax, orientation="horizontal", label="SST [degC]")
    plotting.save(fig, out_path)


def hist_sheet(residuals, removed, water, sds, report, s, cfg, out_path: Path, aoi: str) -> None:
    """Every scene's residual distribution in one grid, sorted by how much was removed.

    The scanning view: a scene whose removed mass sits in a clean cold tail is the detector
    working as intended, while one where the two colours are interleaved across the whole
    distribution was decided by the spatial prior rather than by the residual, and is worth
    opening. Sorted by removed fraction so those tend to collect at one end.
    """
    n = len(report)
    if not n:
        return
    ncols = int(cfg["output"]["figure_ncols"])
    nrows = int(np.ceil(n / ncols))
    fig_h = 1.65 * nrows + HEADER_IN + 0.25
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.05 * ncols, fig_h), dpi=150,
                             facecolor=SURFACE,
                             gridspec_kw=dict(hspace=0.55, wspace=0.14))
    axes = np.atleast_1d(axes).ravel()

    order = (-report["pixel_frac"].to_numpy()).argsort(kind="stable")
    for ax, i in zip(axes, order):
        r = report.iloc[i]
        t = r["t"]
        residual_hist(ax, residuals[t], removed[t], water, sds.get(r["date"]), compact=True)
        kept = r["verdict"] == "KEPT"
        word, color, _ = _verdict_bits(kept)
        ax.set_title(f"{r['date'][5:]}  {r['pixel_frac']:.0%} clear\n{word}",
                     fontsize=5.5, pad=2, color=color)
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        f"{aoi} -- {s['label']}: residual against the covariate, per scene\n"
        f"blue = kept (p_valid >= {cfg['filter']['p_valid_min']}), magenta = removed; "
        f"shaded = +/- the assumed clear-sky sd; sorted by fraction clear",
        fontsize=10, color=INK, y=1 - 0.22 / fig_h)
    fig.subplots_adjust(top=1 - HEADER_IN / fig_h, bottom=0.25 / fig_h, left=0.02, right=0.99)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    log.info("wrote %s", out_path)


# --------------------------------------------------------------------------- offset chart

def offset_figure(reports: dict, cfg: dict, out_path: Path, aoi: str) -> None:
    """Scene offset against the fraction of water that survives, one panel per sensor.

    The scene gate's tuning instrument: the shaded span is the accepted band, so every point
    outside it is an image this run threw away, and the height of a point says how much that
    image would have contributed had it been kept. A point high and just outside the band is
    the one to argue about.

    One panel per sensor rather than one shared axis because the two offset ranges differ by
    an order of magnitude -- Landsat reaches -21 K where ECOSTRESS spans about -2 to +4 -- and
    a shared scale would compress every ECOSTRESS point into a single stripe. The band is
    drawn in both panels, so it anchors the comparison.
    """
    sids = [s for s in reports if len(reports[s])]
    if not sids:
        return
    lo, hi = cfg["filter"]["offset_lower"], cfg["filter"]["offset_upper"]
    min_kept = cfg["filter"]["min_kept_frac"]

    fig, axes = plt.subplots(1, len(sids), figsize=(6.4 * len(sids), 4.8), dpi=150,
                             facecolor=SURFACE, squeeze=False)
    axes = axes.ravel()

    for ax, sid in zip(axes, sids):
        rep = reports[sid]
        ax.set_facecolor(SURFACE)
        span_lo = lo if lo is not None else rep["center"].min() - 1
        span_hi = hi if hi is not None else rep["center"].max() + 1
        ax.axvspan(span_lo, span_hi, color=KEPT_COLOR, alpha=0.07, zorder=0)
        for b in (lo, hi):
            if b is not None:
                ax.axvline(b, color=KEPT_COLOR, linewidth=2, zorder=1)
        if min_kept is not None:
            ax.axhline(float(min_kept), color=INK_MUTED, linewidth=1.5, linestyle=(0, (4, 3)),
                       zorder=1)

        for kept in (True, False):
            sub = rep[rep["verdict"].eq("KEPT") == kept]
            if sub.empty:
                continue
            word, color, marker = _verdict_bits(kept)
            ax.scatter(sub["center"], sub["pixel_frac"], s=46, marker=marker,
                       facecolor=color if kept else "none", edgecolor=color,
                       linewidth=1.6, alpha=0.9, zorder=3, label=f"{word} ({len(sub)})")

        # Direct-label only the scenes that carry an argument: dropped, but would have
        # contributed more water than the median kept scene.
        kept_med = rep.loc[rep["verdict"].eq("KEPT"), "pixel_frac"].median()
        if np.isfinite(kept_med):
            for _, r in rep[rep["verdict"].eq("DROPPED")].iterrows():
                if r["pixel_frac"] > kept_med:
                    ax.annotate(r["date"][5:], (r["center"], r["pixel_frac"]),
                                textcoords="offset points", xytext=(7, 3),
                                fontsize=7, color=INK_SECONDARY)

        ax.set_xlabel("scene offset vs MODIS covariate [K]", fontsize=9, color=INK_SECONDARY)
        ax.set_ylabel("fraction of AoI water surviving the pixel cut", fontsize=9,
                      color=INK_SECONDARY)
        ax.set_title(f"{cfg['sensors'][sid]['label']}  --  "
                     f"{int(rep['verdict'].eq('KEPT').sum())} of {len(rep)} scenes kept",
                     fontsize=11, color=INK, loc="left")
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(labelsize=8, colors=INK_SECONDARY)
        ax.set_ylim(-0.03, 1.03)
        ax.legend(fontsize=8, frameon=False, loc="upper left")

    band = f"[{lo}, {hi}] K"
    fig.suptitle(f"{aoi} -- scene gate: shaded span is the accepted offset band {band}",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    log.info("wrote %s", out_path)


# --------------------------------------------------------------------------- driver

def render(ds: xr.Dataset, cfg: dict, sid: str, paths: list[Path], report: pd.DataFrame,
           filt: np.ndarray, water: np.ndarray, extent: list[float]) -> pd.DataFrame:
    """Draw everything enabled for one sensor. Returns the report with a `t` index column."""
    o, s = cfg["output"], cfg["sensors"][sid]
    out_dir = o["fig_dir"] / cfg["data"]["aoi"]

    index = {str(t)[:10]: i for i, t in enumerate(pd.to_datetime(ds["time"].values))}
    report = report.copy()
    report["t"] = [index[d] for d in report["date"]]

    raw = ds[s["sst"]].values.astype("float32")
    k = _kelvin_offset(raw)
    raw = raw - k
    filt = filt - k
    vmin, vmax = _limits(filt, raw, 2.0, 98.0)
    log.info("%s: contact-sheet scale %.2f..%.2f degC (scene figures scale per scene)",
             sid, vmin, vmax)

    p_min = float(cfg["filter"]["p_valid_min"])
    by_date = {p.name.rsplit("_", 1)[-1][:-3]: p for p in paths}
    scene_dir = out_dir / "scenes"
    if o["figure_scenes"] and scene_dir.exists():
        # Retuning flips a scene between the kept_ and dropped_ prefixes, so a stale file
        # would survive under its old name and the directory would mix two runs.
        for pre in ("kept", "dropped"):
            for old in scene_dir.glob(f"{pre}_*_{sid}.png"):
                old.unlink()

    # One pass over the cache: p_valid and the residual feed the scene panels, the removal mask
    # the contact sheet paints, and the histograms -- so the .nc files are read once, not four
    # times. `residuals` costs another (t, y, x) float32, ~134 MB, and is only held when a
    # figure that needs it is switched on.
    want_hist = o["figure_histograms"] or (o["figure_scenes"] and o["figure_scene_hist"])
    removed = np.zeros(raw.shape, dtype=bool)
    residuals = np.full(raw.shape, np.nan, dtype="float32") if want_hist else None
    sds: dict = {}
    n_drawn = 0
    for _, r in report.iterrows():
        path = by_date.get(r["date"])
        if path is None:
            continue
        with xr.open_dataset(path) as d:
            p_valid = d["p_valid"].values
            res = d["residual"].values if want_hist else None
            sds[r["date"]] = float(d.attrs.get("sd", np.nan))
        t = r["t"]
        removed[t] = water & np.isfinite(raw[t]) & ~(p_valid >= p_min)
        if want_hist:
            residuals[t] = res
        if o["figure_scenes"]:
            scene_figure(raw[t], filt[t], p_valid, water, r, s, cfg, extent, scene_dir,
                         residual=residuals[t] if want_hist else None,
                         sd=sds.get(r["date"]))
            n_drawn += 1
    if o["figure_scenes"]:
        log.info("%s: wrote %d scene figures to %s", sid, n_drawn, scene_dir)

    if o["figure_contact"]:
        contact_sheet(raw, removed, water, report, s, cfg, extent, vmin, vmax,
                      out_dir / f"contact_{sid}.png", cfg["data"]["aoi"])
    if o["figure_histograms"]:
        hist_sheet(residuals, removed, water, sds, report, s, cfg,
                   out_dir / f"residual_hist_{sid}.png", cfg["data"]["aoi"])
    return report
