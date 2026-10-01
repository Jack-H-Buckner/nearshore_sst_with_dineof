"""Figures for the composite stage: what offset was removed, and what the average is made of.

Five views, each answering one question, all optional via `output.figure_*`:

  offsets  -- per-scene difference against overpass hour, one panel per member, with the
              fitted curve AND the constant-model line drawn together. The question is not
              "what is the offset" (that is one number in offsets.csv) but "does the diurnal
              term earn its parameters", and that can only be judged against the scatter it
              sits inside. Landsat's panel is the control: fourteen points in an eight-minute
              stripe, which is what the condition-number gate sees.
  hexbin   -- member against reference, before and after. The cloud should TRANSLATE along the
              1:1 and not rotate. The residual tilt that remains is the errors-in-variables
              artefact of a 1 km retrieval on a 100 m grid, and leaving it visible is the
              point: it is the evidence for offset.slope defaulting off.
  coverage -- fraction of AoI water each member sees per date, as small multiples over the
              composite's own row. This is the figure that shows the composite is mostly a
              UNION: the member rows barely overlap.
  scenes   -- one figure PER DATE into `scenes/<date>.png`: every member's corrected field,
              then the composite they average into, then the member count. All SST panels on
              one range, so a member still reading warm or cool after correction is obvious.
              The per-PIXEL view, and the one that shows the offsets actually landed.
  contact  -- all dates on one sheet, chronological, shared colour scale. The per-DATE view:
              the whole DINEOF input sequence as one image, with its gaps and its seasonal
              swing in the order the reconstruction will meet them.

COLOUR. Members take slots 1-3 of the repo's categorical palette (blue / orange / aqua),
unchanged from the validated reference instance -- that three-slot subset clears the all-pairs
CVD and normal-vision floors in both modes, which matters here because scatter and small
multiples put every pair on screen at once. Colour is never the only channel: every figure is
small multiples with one member per panel and the member NAMED in the panel title, so identity
survives greyscale, CVD and print. That also means a fourth member (MODIS Terra) is safe
despite slot 4 sitting badly beside slot 2 -- the panel title, not the hue, is what identifies
a series here. Status colours (#2f9e44 / #c92a2a) stay reserved for verdicts and are not used.

`offsets.csv` and `matchups.csv` are the table view of everything drawn here.

Not a script; imported by composite.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

import plotting

log = logging.getLogger("composite_figures")

# The repo's chart palette, shared with cube_figures.py.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"

# Categorical slots 1-3 of the validated reference palette. Assigned in config order and never
# cycled: a member keeps its hue whatever else is on screen.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def _member_color(order: list[str], mid: str) -> str:
    return SERIES[order.index(mid) % len(SERIES)]


def _axis(ax) -> None:
    """Recessive grid and axes, so the marks carry the figure."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(labelsize=8, colors=INK_SECONDARY)


def _density_cmap(color: str):
    """A single-hue light-to-dark ramp for density. Never a rainbow."""
    return LinearSegmentedColormap.from_list("density", ["#eef1f4", color, "#16181a"])


def _scene_offsets(fit: dict) -> np.ndarray:
    """The modelled offset at each matchup scene's hour."""
    from seasonal_smoothing import diurnal_design
    h = np.nan_to_num(fit["table"]["hour"].to_numpy(float))
    return diurnal_design(h, fit["K"]) @ fit["coefs"]


# --------------------------------------------------------------------------- offsets

def offset_curve_figure(cfg: dict, offsets: dict, order: list[str], path: Path) -> None:
    """Per-scene difference against overpass hour, one panel per fitted member.

    Both models are drawn in every panel: the fitted curve and the constant it would collapse
    to. A diurnal term that explains the data would separate them visibly against the points;
    here it does not, which is the honest reading and the reason the LOO gate exists. The
    y-range is the DATA's, never the curve's -- a curve auto-scaled to its own amplitude is the
    classic way to make a 0.4 K wiggle look like a finding.
    """
    mids = [m for m in order if m != cfg["reference"] and len(offsets[m].get("table", []))]
    if not mids:
        return
    ref_label = cfg["members"][cfg["reference"]]["label"]

    fig, axes = plt.subplots(1, len(mids), figsize=(6.2 * len(mids), 4.6),
                             dpi=cfg["output"]["figure_dpi"], facecolor=SURFACE, squeeze=False)
    for ax, mid in zip(axes.ravel(), mids):
        fit = offsets[mid]
        t = fit["table"]
        color = _member_color(order, mid)
        _axis(ax)

        grid_h = np.linspace(0, 24, 241)
        from seasonal_smoothing import diurnal_design
        curve = diurnal_design(grid_h, fit["K"]) @ fit["coefs"]
        const = float(fit["coefs"][0]) if fit["K"] == 0 else float(
            np.average(t["delta"], weights=fit["weights"]))

        ax.axhline(0.0, color=INK_MUTED, linewidth=1.0, zorder=1)
        ax.plot(grid_h, np.full_like(grid_h, const), color=INK_MUTED, linewidth=2,
                linestyle=(0, (5, 3)), zorder=3, label=f"constant  ({const:+.2f} K)")
        if fit["K"] > 0:
            ax.plot(grid_h, curve, color=color, linewidth=2, zorder=4,
                    label=f"K={fit['K']} diurnal")
        ax.scatter(t["hour"], t["delta"], s=52, facecolor=color, edgecolor=SURFACE,
                   linewidth=1.5, alpha=0.9, zorder=5)

        st = fit["stats"]
        ax.set_xlim(-0.5, 24.5)
        ax.set_xticks(range(0, 25, 6))
        span = max(t["delta"].max() - t["delta"].min(), 0.5)
        ax.set_ylim(t["delta"].min() - 0.15 * span, t["delta"].max() + 0.22 * span)
        ax.set_xlabel("overpass hour [UTC]", fontsize=9, color=INK_SECONDARY)
        ax.set_ylabel(f"scene median minus {ref_label} [K]", fontsize=9, color=INK_SECONDARY)
        ax.set_title(f"{st['label']}  --  {len(t)} scenes, K={fit['K']} chosen",
                     fontsize=11, color=INK, loc="left", pad=24)
        ax.text(0.0, 1.015,
                f"LOO RMSE {st['loo_rmse']:.4f} vs {st['loo_rmse_constant']:.4f} constant   "
                f"|   cond {st['cond']:.3g}   |   scene scatter {st['delta_sd']:.2f} K",
                transform=ax.transAxes, fontsize=8, color=INK_MUTED, va="bottom")
        ax.legend(fontsize=8, frameon=False, loc="upper right")

    fig.suptitle(f"What the offset model removes  --  {cfg['data']['aoi']}",
                 fontsize=12.5, color=INK, x=0.008, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plotting.save(fig, path, dpi=cfg["output"]["figure_dpi"])


# --------------------------------------------------------------------------- hexbin

def hexbin_figure(cfg: dict, offsets: dict, order: list[str], path: Path) -> None:
    """Member against reference, before and after the offset, two panels per member.

    Drawn on a shared square axis with the 1:1 line, because the whole claim is about position
    relative to that line. Density is a single hue light to dark -- the cloud spans four orders
    of magnitude in count, and a rainbow would invent structure in it.
    """
    mids = [m for m in order if m != cfg["reference"] and offsets[m]["ref_px"].size]
    if not mids:
        return
    ref_label = cfg["members"][cfg["reference"]]["label"]

    fig, axes = plt.subplots(len(mids), 2, figsize=(10.2, 4.9 * len(mids)),
                             dpi=cfg["output"]["figure_dpi"], facecolor=SURFACE, squeeze=False)
    for row, mid in zip(axes, mids):
        fit = offsets[mid]
        color = _member_color(order, mid)
        x = fit["ref_px"]
        off = np.repeat(_scene_offsets(fit), fit["table"]["n"].to_numpy(int))
        panels = [("before", fit["mem_px"]), ("after", fit["mem_px"] - off)]

        lo = float(min(x.min(), min(y.min() for _, y in panels)))
        hi = float(max(x.max(), max(y.max() for _, y in panels)))
        pad = 0.03 * (hi - lo)
        lo, hi = lo - pad, hi + pad

        for ax, (when, y) in zip(row, panels):
            _axis(ax)
            ax.hexbin(x, y, gridsize=58, extent=(lo, hi, lo, hi), mincnt=1, bins="log",
                      cmap=_density_cmap(color), linewidths=0.0, zorder=2)
            ax.plot([lo, hi], [lo, hi], color=INK, linewidth=2, linestyle=(0, (5, 3)),
                    zorder=3, label="1:1")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal")
            bias = float(np.mean(y - x))
            ax.set_xlabel(f"{ref_label} [K]", fontsize=9, color=INK_SECONDARY)
            ax.set_ylabel(f"{fit['stats']['label']} [K]", fontsize=9, color=INK_SECONDARY)
            # Pooled over PIXELS, so a big scene counts more -- unlike the fit, which targets
            # the equally-weighted scene median and drives that to exactly 0. The two differ by
            # a few tenths and the gap is scene-size imbalance, not residual bias.
            ax.set_title(f"{fit['stats']['label']}, {when} offset  --  pixel mean {bias:+.2f} K",
                         fontsize=10.5, color=INK, loc="left")
            ax.legend(fontsize=8, frameon=False, loc="upper left")

        st = fit["stats"]
        row[1].text(0.98, 0.04,
                    f"anomaly slope: OLS {st['slope_ols']:.2f}, RMA {st['slope_rma']:.2f}\n"
                    "the true slope lies between; the gap is resolution\n"
                    "error, not gain -- see offset.slope",
                    transform=row[1].transAxes, fontsize=7.5, color=INK_MUTED,
                    ha="right", va="bottom")

    fig.suptitle(f"The offset translates the cloud; it does not rotate it  --  "
                 f"{cfg['data']['aoi']}", fontsize=12.5, color=INK, x=0.008, ha="left", y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plotting.save(fig, path, dpi=cfg["output"]["figure_dpi"])


# --------------------------------------------------------------------------- coverage

def coverage_figure(cfg: dict, comp: dict, adj: dict, order: list[str],
                    times: pd.DatetimeIndex, water: np.ndarray, path: Path) -> None:
    """Fraction of AoI water seen per date: one small multiple per member, then the composite.

    Small multiples rather than a stacked area, because stacking would assert that the members
    sum -- and they very nearly do, which is the finding, not an assumption to build into the
    geometry. The rows barely overlap: the composite is mostly a union.

    The composite row flags the dates carried by the reference ALONE. Those are the columns a
    DINEOF run inherits at the reference's coarse support with no high-resolution constraint,
    and `sst_composite_src` is how a later stage finds them.
    """
    n_water = float(water.sum())
    ref = cfg["reference"]
    rows = [(mid, np.isfinite(adj[mid]).sum(axis=(1, 2)) / n_water) for mid in order]
    comp_cov = (comp["n"] > 0).sum(axis=(1, 2)) / n_water

    ref_bit = 1 << order.index(ref)
    seen = comp["src"].reshape(comp["src"].shape[0], -1)
    only_ref = np.array([bool((s != 0).any() and set(np.unique(s[s != 0])) == {ref_bit})
                         for s in seen])

    fig, axes = plt.subplots(len(rows) + 1, 1, figsize=(12.5, 1.55 * (len(rows) + 1)),
                             dpi=cfg["output"]["figure_dpi"], facecolor=SURFACE,
                             sharex=True, squeeze=False)
    axes = axes.ravel()
    top = max(float(comp_cov.max()), max(float(c.max()) for _, c in rows)) * 1.15

    for ax, (mid, cov) in zip(axes, rows):
        _axis(ax)
        color = _member_color(order, mid)
        hit = cov > 0
        ax.vlines(times[hit], 0, cov[hit], color=color, linewidth=1.6, alpha=0.92, zorder=3)
        ax.set_ylim(0, top)
        ax.set_ylabel("water seen", fontsize=8, color=INK_SECONDARY)
        ax.set_title(f"{cfg['members'][mid]['label']}  --  {int(hit.sum())} dates, "
                     f"median {100 * np.median(cov[hit]) if hit.any() else 0:.1f}% of water",
                     fontsize=9.5, color=INK, loc="left")
        ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")

    ax = axes[-1]
    _axis(ax)
    hit = comp_cov > 0
    ax.vlines(times[hit & ~only_ref], 0, comp_cov[hit & ~only_ref], color=INK,
              linewidth=1.6, zorder=3, label="has a high-resolution member")
    ax.vlines(times[only_ref], 0, comp_cov[only_ref], color=INK_MUTED, linewidth=1.6,
              linestyle=(0, (2, 1.5)), zorder=4,
              label=f"{cfg['members'][ref]['label']} only ({int(only_ref.sum())} dates)")
    ax.set_ylim(0, top)
    ax.set_ylabel("water seen", fontsize=8, color=INK_SECONDARY)
    ax.set_title(f"sst_composite  --  {int(hit.sum())} dates, median "
                 f"{100 * np.median(comp_cov[hit]) if hit.any() else 0:.1f}% of water",
                 fontsize=9.5, color=INK, loc="left")
    ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
    ax.legend(fontsize=8, frameon=False, loc="upper right", ncol=2)
    ax.tick_params(axis="x", labelsize=8, colors=INK_SECONDARY)

    fig.suptitle(f"What the composite is made of  --  {cfg['data']['aoi']}",
                 fontsize=12.5, color=INK, x=0.006, ha="left", y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    plotting.save(fig, path, dpi=cfg["output"]["figure_dpi"])


# --------------------------------------------------------------------------- the field itself

HEADER_IN = 0.85
FOOTER_IN = 0.72


def _kelvin_offset(a: np.ndarray) -> float:
    """273.15 if the field is Kelvin, else 0. Decided once over the whole stack and applied
    everywhere, so every panel in every figure shares one temperature convention."""
    v = a[np.isfinite(a)]
    return 273.15 if v.size and float(np.nanmedian(v)) > 100.0 else 0.0


def _members_glyph(src_row: np.ndarray, order: list[str]) -> str:
    """'ME.' -- one character per member, in config order, a dot where it did not contribute.

    A fixed-width glyph rather than a colour or a count, so the contact sheet says WHICH
    members made each panel without a legend and without relying on hue. Position is the
    encoding, which survives greyscale and a thumbnail.
    """
    return "".join(mid[0].upper() if (src_row & (1 << i)).any() else "."
                   for i, mid in enumerate(order))


def _cov(frac: float) -> str:
    """Coverage as a percentage, without rounding a real sliver down to a flat '0%'.

    13 of the 185 composite dates cover under 0.5% of AoI water -- the thinnest is 18 pixels --
    and printing those as '0% of water' reads as a broken panel rather than as the very thin
    date it is.
    """
    if frac <= 0:
        return "0%"
    return f"<1%" if frac < 0.01 else f"{frac:.0%}"


def _count_cmap(n_members: int):
    """A discrete light-to-dark ramp for the member count: sequential, one hue, n steps."""
    steps = ["#cfe0f2", "#6ba3dd", "#1d4f8c", "#0b2444"][:max(n_members, 1)]
    return ListedColormap(steps)


def composite_scene_figure(cfg: dict, comp: dict, adj: dict, order: list[str],
                           t: int, date: str, water: np.ndarray, extent: list[float],
                           koff: float, scene_dir: Path) -> None:
    """One date: every member's corrected field, then the composite they average into.

    All the SST panels share ONE colour range, computed from this date's own data. That is the
    point of the figure -- after the offset correction the members should be reading the same
    temperatures, so a member panel that looks systematically warmer or cooler than its
    neighbours is either a bad offset or a bad scene, and a shared bar is what makes that
    visible at a glance. A per-panel range would hide exactly the failure this is drawn to
    catch.

    Members that saw nothing on this date still get a panel, empty. The gap is the information:
    it says which instrument was absent, which is what the composite's coverage depends on.
    """
    fields = [(cfg["members"][mid]["label"], adj[mid][t] - koff) for mid in order]
    sst = comp["sst"][t] - koff
    n = comp["n"][t]

    pool = np.concatenate([f[np.isfinite(f) & water] for _, f in fields] + [[]])
    if pool.size == 0:
        return
    vmin, vmax = float(np.percentile(pool, 1.0)), float(np.percentile(pool, 99.0))
    if not np.isfinite(vmin) or vmax <= vmin:
        vmin, vmax = float(pool.min()), float(pool.min()) + 1.0

    ncol = len(order) + 2
    fig, axes = plt.subplots(1, ncol, figsize=(2.55 * ncol, 3.35),
                             dpi=cfg["output"]["figure_dpi"], facecolor=SURFACE)
    cmap = plotting.sst_cmap()
    n_water = float(water.sum())

    im = None
    for ax, (label, field) in zip(axes, fields):
        im = plotting.panel(ax, field, water, vmin=vmin, vmax=vmax, extent=extent, cmap=cmap)
        cov = float((np.isfinite(field) & water).sum()) / n_water
        ax.set_title(f"{label}\n{_cov(cov)} of water" if cov else f"{label}\nnot seen",
                     fontsize=7.5, color=INK if cov else INK_MUTED, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])

    ax = axes[len(order)]
    plotting.panel(ax, sst, water, vmin=vmin, vmax=vmax, extent=extent, cmap=cmap)
    ax.set_title(f"sst_composite\n{_cov(float((n > 0).sum()) / n_water)} of water",
                 fontsize=7.5, color=INK, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():          # the one panel that is an output, not an input
        side.set_color(INK)
        side.set_linewidth(1.4)

    ax = axes[-1]
    ax.set_facecolor(plotting.NODATA_COLOR)
    plotting.flat(ax, ~water, plotting.LAND_COLOR, extent)
    nm = len(order)
    imn = ax.imshow(np.where(water & (n > 0), n, np.nan), cmap=_count_cmap(nm),
                    vmin=0.5, vmax=nm + 0.5, extent=extent, origin="upper",
                    interpolation="nearest")
    ax.set_title("members per pixel", fontsize=7.5, color=INK, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.subplots_adjust(top=0.80, bottom=0.14, left=0.012, right=0.988, wspace=0.06)
    cax = fig.add_axes((0.14, 0.055, 0.42, 0.030))
    fig.colorbar(im, cax=cax, orientation="horizontal",
                 label="SST [degC]" if koff else "SST")
    cax.tick_params(labelsize=7, colors=INK_SECONDARY)
    cax.xaxis.label.set(fontsize=7.5, color=INK_SECONDARY)
    caxn = fig.add_axes((0.70, 0.055, 0.22, 0.030))
    cb = fig.colorbar(imn, cax=caxn, orientation="horizontal", label="members",
                      ticks=range(1, nm + 1))
    cb.ax.tick_params(labelsize=7, colors=INK_SECONDARY)
    caxn.xaxis.label.set(fontsize=7.5, color=INK_SECONDARY)

    glyph = _members_glyph(comp["src"][t], order)
    fig.suptitle(f"{date}   --   {cfg['data']['aoi']}   --   members {glyph}",
                 fontsize=11, color=INK, x=0.012, ha="left", y=0.975)
    plotting.save(fig, scene_dir / f"{date}.png", dpi=cfg["output"]["figure_dpi"])


def composite_contact_sheet(cfg: dict, comp: dict, order: list[str], dates: list[str],
                            idx: list[int], water: np.ndarray, extent: list[float],
                            koff: float, path: Path) -> None:
    """Every date the composite has data on, one panel each, in time order.

    Chronological rather than sorted by any quality score, because this IS the DINEOF input
    sequence: read left to right and the gaps, the seasonal swing and the coverage collapses
    are all in the order the reconstruction will meet them.

    ONE colour range across all panels, taken over the whole composite. That makes the seasonal
    cycle the dominant visual signal -- which is honest, since it is the dominant signal in the
    data -- at the cost of within-scene contrast on any single date. The per-date figures in
    `scenes/` carry their own range for that.
    """
    if not idx:
        return
    sst = comp["sst"] - koff
    pool = sst[np.isfinite(sst) & water[None, :, :]]
    vmin, vmax = float(np.percentile(pool, 1.0)), float(np.percentile(pool, 99.0))

    n = len(idx)
    ncols = int(cfg["output"]["figure_ncols"])
    nrows = int(np.ceil(n / ncols))
    fig_h = 2.30 * nrows + HEADER_IN + FOOTER_IN
    n_water = float(water.sum())

    fig, axes = plt.subplots(nrows, ncols, figsize=(1.85 * ncols, fig_h), dpi=150,
                             facecolor=SURFACE, gridspec_kw=dict(hspace=0.46, wspace=0.06))
    axes = np.atleast_1d(axes).ravel()
    cmap = plotting.sst_cmap()

    im = None
    ref_bit = 1 << order.index(cfg["reference"])
    for ax, t, date in zip(axes, idx, dates):
        im = plotting.panel(ax, sst[t], water, vmin=vmin, vmax=vmax, extent=extent, cmap=cmap)
        cov = float((comp["n"][t] > 0).sum()) / n_water
        glyph = _members_glyph(comp["src"][t], order)
        s = comp["src"][t]
        only_ref = bool((s != 0).any() and set(np.unique(s[s != 0])) == {ref_bit})
        ax.set_title(f"{date[5:]}  {glyph}\n{_cov(cov)} of water",
                     fontsize=6, pad=3, color=INK_MUTED if only_ref else INK)
        ax.set_xticks([])
        ax.set_yticks([])
        if only_ref:
            # The dates with no high-resolution constraint, marked so they are countable by
            # eye. Dashed, not coloured: this is a caveat, not a verdict.
            for side in ax.spines.values():
                side.set_color(INK_MUTED)
                side.set_linewidth(1.2)
                side.set_linestyle((0, (2, 1.5)))
    for ax in axes[n:]:
        ax.axis("off")

    ref_label = cfg["members"][cfg["reference"]]["label"]
    glyph_key = " ".join(f"{cfg['members'][m]['label'][0].upper()}={cfg['members'][m]['label']}"
                         for m in order)
    fig.suptitle(
        f"{cfg['data']['aoi']} -- sst_composite on {n} dates, in time order\n"
        f"members glyph: {glyph_key}, a dot where absent; "
        f"dashed frame = {ref_label} alone (no high-resolution member); shared colour scale",
        fontsize=10, color=INK, y=1 - 0.20 / fig_h)
    fig.subplots_adjust(top=1 - HEADER_IN / fig_h, bottom=FOOTER_IN / fig_h,
                        left=0.015, right=0.985)
    cax = fig.add_axes((0.36, 0.46 / fig_h, 0.28, 0.10 / fig_h))
    cb = fig.colorbar(im, cax=cax, orientation="horizontal",
                      label="SST [degC]" if koff else "SST")
    cb.ax.tick_params(labelsize=8, colors=INK_SECONDARY)
    plotting.save(fig, path)


# --------------------------------------------------------------------------- driver

def render(cfg: dict, offsets: dict, comp: dict, adj: dict, order: list[str],
           times: pd.DatetimeIndex, water: np.ndarray, extent: list[float],
           out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    o = cfg["output"]
    if o["figure_offsets"]:
        offset_curve_figure(cfg, offsets, order, out_dir / "offset_curves.png")
    if o["figure_hexbin"]:
        hexbin_figure(cfg, offsets, order, out_dir / "member_vs_reference.png")
    if o["figure_coverage"]:
        coverage_figure(cfg, comp, adj, order, times, water, out_dir / "coverage.png")

    if not (o["figure_scenes"] or o["figure_contact"]):
        return
    idx = [t for t in range(comp["sst"].shape[0]) if np.isfinite(comp["sst"][t]).any()]
    dates = [str(times[t])[:10] for t in idx]
    koff = _kelvin_offset(comp["sst"])

    if o["figure_scenes"]:
        scene_dir = out_dir / "scenes"
        # A rerun that changes min_members or a member list leaves fewer dates with data, and a
        # stale PNG from the previous run would sit in the directory looking current.
        if scene_dir.exists():
            for old in scene_dir.glob("*.png"):
                old.unlink()
        for t, date in zip(idx, dates):
            composite_scene_figure(cfg, comp, adj, order, t, date, water, extent, koff,
                                   scene_dir)
        log.info("wrote %d per-date figures to %s", len(idx), scene_dir)

    if o["figure_contact"]:
        composite_contact_sheet(cfg, comp, order, dates, idx, water, extent, koff,
                                out_dir / "contact_composite.png")
