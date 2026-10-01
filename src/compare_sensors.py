"""ECOSTRESS against Landsat on the days that have both.

A validation of the DINEOF INPUT, not of its output. The composite stage merges the two
sensors into one field per day, and eDINEOF then reconstructs from that merged series -- so
wherever the two disagree, the merge is averaging a disagreement and the reconstruction
inherits it. There are only 14 such days in this cube (ECOSTRESS has 114, Landsat 32, and
they rarely coincide), which is few enough to look at individually.

Four panels per day, read left to right:

    ECOSTRESS | Landsat | difference | histogram of the difference

The two sensor panels SHARE one colour scale -- without that the eye cannot compare them, and
comparing them is the entire point. Each panel shows its sensor's full field, so coverage
differences are visible; the difference and the histogram are restricted to pixels where both
sensors saw water, which is the only place a difference is defined.

Read from the COMPOSITE cube's `<id>_adj` channels, i.e. AFTER the inter-sensor offset that
stage fitted. That is deliberate: the raw difference is dominated by a known calibration
offset (+1.12 K for ECOSTRESS against the MODIS anchor), and subtracting a number we already
estimated tells us nothing. What is left is the disagreement the offset does not explain --
scene-dependent, spatially structured, and the part that actually damages a reconstruction.
Pass --raw to see it the other way.

Usage (from the repo root, in the `coastal_sst_data` env):

    python prototypes/DINEOF/src/compare_sensors.py
    ... --min-overlap 500      # skip days with fewer overlapping water pixels
    ... --raw                  # compare the unadjusted <sensor>_dineof channels instead
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plotting
from cube_figures import GRID, INK, INK_SECONDARY, SURFACE

log = logging.getLogger("compare_sensors")

ROOT = Path(__file__).resolve().parents[1]              # prototypes/DINEOF
DEFAULT_COMPOSITE = ROOT / "data" / "datacube" / "admiralty_inlet_composite.zarr"
DEFAULT_FILTERED = ROOT / "data" / "datacube" / "admiralty_inlet_filtered.zarr"
DEFAULT_FIGDIR = ROOT / "figures" / "sensor_comparison"

# ECOSTRESS minus Landsat. A diverging pair needs a neutral midpoint, so that zero -- perfect
# agreement -- is the colour the eye reads as "nothing here" rather than a hue of its own.
DIFF_CMAP = "RdBu_r"


def load(raw: bool, aoi: str) -> tuple[xr.Dataset, str, str, str]:
    """(dataset, eco channel, lst channel, what the numbers mean)."""
    if raw:
        ds = xr.open_zarr(DEFAULT_FILTERED)
        return ds, "eco_sst_v002_dineof", "lst_sst_dineof", "unadjusted"
    ds = xr.open_zarr(DEFAULT_COMPOSITE)
    return ds, "eco_adj", "lst_adj", "after the composite inter-sensor offset"


def day_figure(eco, lst, water, date, stats, extent, vmin, vmax, dmax, note, out: Path) -> None:
    both = np.isfinite(eco) & np.isfinite(lst) & water
    diff = np.where(both, eco - lst, np.nan)

    fig, axes = plt.subplots(1, 4, figsize=(16.8, 4.7), dpi=130,
                             gridspec_kw=dict(width_ratios=[1, 1, 1, 0.95]))
    cmap = plotting.sst_cmap()

    for ax, field, name in ((axes[0], eco, "ECOSTRESS v002"), (axes[1], lst, "Landsat")):
        im = plotting.panel(ax, field, water, vmin=vmin, vmax=vmax, extent=extent, cmap=cmap,
                            title=f"{name}\n{np.isfinite(field[water]).sum():,} water px")
        fig.colorbar(im, ax=ax, shrink=0.80, label="SST [degC]")

    dcm = plt.get_cmap(DIFF_CMAP).copy()
    dcm.set_bad(alpha=0.0)
    imd = plotting.panel(axes[2], diff, water, vmin=-dmax, vmax=dmax, extent=extent, cmap=dcm,
                         title=f"ECOSTRESS - Landsat\n{int(both.sum()):,} overlapping px")
    fig.colorbar(imd, ax=axes[2], shrink=0.80, label="difference [K]")

    ax = axes[3]
    d = diff[both]
    lo, hi = np.percentile(d, [0.5, 99.5])
    pad = 0.05 * max(hi - lo, 1e-6)
    edges = np.linspace(lo - pad, hi + pad, 61)
    n_over = int((d < edges[0]).sum() + (d > edges[-1]).sum())
    ax.hist(np.clip(d, edges[0], edges[-1]), bins=edges, color="#6da7ec", edgecolor="none")
    ax.axvline(0.0, color=INK, linewidth=1.4, label="agreement")
    ax.axvline(stats["bias"], color="#eb6834", linewidth=1.8, linestyle="--",
               label=f"mean {stats['bias']:+.2f} K")
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(labelsize=7, colors=INK_SECONDARY, left=False, labelleft=False)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    note_over = f"\n{n_over:,} px beyond the axis are in the end bars" if n_over else ""
    ax.set_xlabel(f"ECOSTRESS - Landsat [K]{note_over}", fontsize=7, color=INK_SECONDARY)
    ax.set_title(f"bias {stats['bias']:+.2f}   median {stats['med']:+.2f}\n"
                 f"RMSD {stats['rmsd']:.2f}   sd {stats['sd']:.2f} K", fontsize=8)

    for a in axes[:3]:
        a.set_xlabel("km east", fontsize=7)
        a.tick_params(labelsize=6)
    axes[0].set_ylabel("km north", fontsize=7)

    fig.suptitle(f"{date}   ECOSTRESS vs Landsat, {note}   "
                 f"(shared SST scale {vmin:.1f}-{vmax:.1f} degC)",
                 fontsize=10, ha="left", x=0.01, y=0.985)
    fig.subplots_adjust(top=0.82, bottom=0.14, left=0.04, right=0.985, wspace=0.24)
    plotting.save(fig, out / f"{date}.png")


def summary_figure(df: pd.DataFrame, note: str, out: Path) -> None:
    """Per-day bias with a +/-1 sd bar, so a systematic offset is separable from scatter."""
    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=150, facecolor=SURFACE)
    x = np.arange(len(df))
    ax.axhline(0.0, color=INK, linewidth=1.2, zorder=3)
    ax.errorbar(x, df["bias"], yerr=df["sd"], fmt="o", markersize=7, capsize=4,
                color="#256abf", ecolor="#9ec5f4", elinewidth=2.2, zorder=4,
                label="daily mean difference  +/- 1 sd")
    pooled = float((df["bias"] * df["n"]).sum() / df["n"].sum())
    ax.axhline(pooled, color="#eb6834", linewidth=1.8, linestyle="--", zorder=3,
               label=f"pixel-weighted mean {pooled:+.2f} K")
    ax.set_xticks(x)
    ax.set_xticklabels(df["date"], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("ECOSTRESS - Landsat [K]", fontsize=9, color=INK_SECONDARY)
    ax.set_title(f"ECOSTRESS vs Landsat on the {len(df)} days with both, {note}",
                 fontsize=11, color=INK, loc="left")
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(labelsize=8, colors=INK_SECONDARY)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "summary.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    log.info("wrote %s", out / "summary.png")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--aoi", default="admiralty_inlet")
    p.add_argument("--min-overlap", type=int, default=200,
                   help="skip days with fewer overlapping water pixels (default 200)")
    p.add_argument("--raw", action="store_true",
                   help="compare the unadjusted <sensor>_dineof channels instead")
    p.add_argument("--fig-dir", type=Path, default=DEFAULT_FIGDIR)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ds, eco_v, lst_v, note = load(args.raw, args.aoi)
    water = plotting.water_mask(ds)
    extent = plotting.extent_km(ds)
    t = pd.to_datetime(ds["time"].values)

    # to_celsius decides K-vs-degC on each array's own median, which is safe here because both
    # channels are full-year stacks; the DIFFERENCE is unaffected either way, only the axis
    # labels on the two sensor panels depend on it.
    eco_all = plotting.to_celsius(ds[eco_v].values.astype("float32"))
    lst_all = plotting.to_celsius(ds[lst_v].values.astype("float32"))

    both = np.isfinite(eco_all) & np.isfinite(lst_all) & water[None]
    n = both.sum(axis=(1, 2))
    days = np.where(n >= args.min_overlap)[0]
    log.info("%s: %d days have both sensors with >= %d overlapping water px "
             "(%s / %s, %s)", args.aoi, len(days), args.min_overlap, eco_v, lst_v, note)
    if not len(days):
        raise SystemExit("no days with both sensors")

    out = args.fig_dir / args.aoi / ("raw" if args.raw else "adjusted")
    rows = []
    for i in days:
        m = both[i]
        d = eco_all[i][m] - lst_all[i][m]
        stats = dict(date=str(t[i])[:10], n=int(m.sum()), bias=float(d.mean()),
                     med=float(np.median(d)), rmsd=float(np.sqrt((d ** 2).mean())),
                     sd=float(d.std()))
        rows.append(stats)
        # Shared SST scale across the two sensor panels, taken over the OVERLAP only: scoring
        # it on each sensor's full field would let a swath edge that only one of them saw set
        # the range and push the comparable part into a narrow band.
        vals = np.concatenate([eco_all[i][m], lst_all[i][m]])
        vmin, vmax = np.percentile(vals, [2, 98])
        dmax = max(float(np.percentile(np.abs(d), 98)), 0.1)
        day_figure(eco_all[i], lst_all[i], water, stats["date"], stats, extent,
                   float(vmin), float(vmax), dmax, note, out)

    df = pd.DataFrame(rows)
    summary_figure(df, note, out)
    csv = out / "sensor_comparison.csv"
    df.to_csv(csv, index=False)
    log.info("wrote %s", csv)

    pooled_bias = float((df["bias"] * df["n"]).sum() / df["n"].sum())
    pooled_rmsd = float(np.sqrt((df["rmsd"] ** 2 * df["n"]).sum() / df["n"].sum()))
    log.info("\n%s", df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    log.info("\npooled over %d days / %d px: bias %+.2f K, RMSD %.2f K, "
             "median |daily bias| %.2f K", len(df), int(df["n"].sum()),
             pooled_bias, pooled_rmsd, df["bias"].abs().median())


if __name__ == "__main__":
    main()
