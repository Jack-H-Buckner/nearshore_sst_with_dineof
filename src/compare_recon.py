"""Measured against interpolated, on days eDINEOF was not allowed to see.

The obvious version of this figure is empty, and that is worth stating plainly: DINEOF holds
observed entries FIXED and only ever writes into gaps, so on the 4,445,299 observed pixels the
reconstruction equals the observation to the last bit -- measured minus interpolated is
identically zero. Differencing the delivered cube measures nothing.

So this refits with data held back. It reproduces edineof's own day holdout (same seed, same
`choose_cv_points`), removes those whole days from the matrix, refits at the same k and p, and
compares the prediction against the measurement that was withheld. A whole held-out day has no
pixels of its own, so its entire field comes from the temporal filter reaching in from
neighbouring days -- which is exactly the situation of the 188 days in this cube that have no
data at all. These days are the only honest preview of what those 188 look like.

Four panels per day, matching compare_sensors.py:

    measured | interpolated | difference | histogram of the residual

The two field panels share one colour scale. The difference and the histogram are restricted
to pixels the held-out day actually measured, since that is where a residual is defined.

Cost: one refit, about two minutes at k=7.

Usage (from the repo root, in the `coastal_sst_data` env):

    python prototypes/DINEOF/src/compare_recon.py --k 7 --p 10
    ... --min-obs 2000     # skip held-out days with fewer measured water pixels
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

import edineof as ed
import plotting
from cube_figures import GRID, INK, INK_SECONDARY, SURFACE

log = logging.getLogger("compare_recon")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIGDIR = ROOT / "figures" / "recon_comparison"
DIFF_CMAP = "RdBu_r"


def day_figure(meas, pred, water, date, stats, extent, vmin, vmax, dmax, out: Path) -> None:
    m = np.isfinite(meas) & water
    diff = np.where(m, pred - meas, np.nan)

    fig, axes = plt.subplots(1, 4, figsize=(16.8, 4.7), dpi=130,
                             gridspec_kw=dict(width_ratios=[1, 1, 1, 0.95]))
    cmap = plotting.sst_cmap()

    im = plotting.panel(axes[0], meas, water, vmin=vmin, vmax=vmax, extent=extent, cmap=cmap,
                        title=f"measured (withheld)\n{int(m.sum()):,} water px")
    fig.colorbar(im, ax=axes[0], shrink=0.80, label="SST [degC]")
    # The prediction covers every water pixel, including the ones the day never measured --
    # that is the product. Only the measured subset can be scored.
    im = plotting.panel(axes[1], pred, water, vmin=vmin, vmax=vmax, extent=extent, cmap=cmap,
                        title=f"interpolated (eDINEOF)\nfilter-only, no data from this day")
    fig.colorbar(im, ax=axes[1], shrink=0.80, label="SST [degC]")

    dcm = plt.get_cmap(DIFF_CMAP).copy()
    dcm.set_bad(alpha=0.0)
    imd = plotting.panel(axes[2], diff, water, vmin=-dmax, vmax=dmax, extent=extent, cmap=dcm,
                         title=f"interpolated - measured\nbias {stats['bias']:+.2f} K")
    fig.colorbar(imd, ax=axes[2], shrink=0.80, label="residual [K]")

    ax = axes[3]
    d = diff[m]
    lo, hi = np.percentile(d, [0.5, 99.5])
    pad = 0.05 * max(hi - lo, 1e-6)
    edges = np.linspace(lo - pad, hi + pad, 61)
    n_over = int((d < edges[0]).sum() + (d > edges[-1]).sum())
    ax.hist(np.clip(d, edges[0], edges[-1]), bins=edges, color="#6da7ec", edgecolor="none")
    ax.axvline(0.0, color=INK, linewidth=1.4, label="perfect")
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
    note = f"\n{n_over:,} px beyond the axis are in the end bars" if n_over else ""
    ax.set_xlabel(f"interpolated - measured [K]{note}", fontsize=7, color=INK_SECONDARY)
    ax.set_title(f"bias {stats['bias']:+.2f}   median {stats['med']:+.2f}\n"
                 f"RMSE {stats['rmse']:.2f}   sd {stats['sd']:.2f} K", fontsize=8)

    for a in axes[:3]:
        a.set_xlabel("km east", fontsize=7)
        a.tick_params(labelsize=6)
    axes[0].set_ylabel("km north", fontsize=7)

    fig.suptitle(f"{date}   eDINEOF day-holdout: this day was removed entirely and predicted "
                 f"from its neighbours   (shared scale {vmin:.1f}-{vmax:.1f} degC)",
                 fontsize=10, ha="left", x=0.01, y=0.985)
    fig.subplots_adjust(top=0.82, bottom=0.14, left=0.04, right=0.985, wspace=0.24)
    plotting.save(fig, out / f"{date}.png")


def summary_figure(df: pd.DataFrame, k: int, p: int, clim_rmse: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=150, facecolor=SURFACE)
    x = np.arange(len(df))
    ax.axhline(0.0, color=INK, linewidth=1.2, zorder=3)
    ax.errorbar(x, df["bias"], yerr=df["sd"], fmt="o", markersize=7, capsize=4,
                color="#256abf", ecolor="#9ec5f4", elinewidth=2.2, zorder=4,
                label="daily mean residual  +/- 1 sd")
    pooled = float((df["bias"] * df["n"]).sum() / df["n"].sum())
    ax.axhline(pooled, color="#eb6834", linewidth=1.8, linestyle="--", zorder=3,
               label=f"pixel-weighted mean {pooled:+.2f} K")
    ax.set_xticks(x)
    ax.set_xticklabels(df["date"], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("interpolated - measured [K]", fontsize=9, color=INK_SECONDARY)
    ax.set_title(f"eDINEOF day holdout, k={k} p={p}: {len(df)} days removed whole and predicted"
                 f"\npooled RMSE {np.sqrt((df['rmse']**2 * df['n']).sum()/df['n'].sum()):.2f} K"
                 f"   vs {clim_rmse:.2f} K for the seasonal climatology these days would "
                 f"otherwise get", fontsize=10, color=INK, loc="left")
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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ed.DEFAULT_CONFIG)
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--p", type=int, default=10)
    ap.add_argument("--min-obs", type=int, default=500,
                    help="skip held-out days with fewer measured water pixels")
    ap.add_argument("--fig-dir", type=Path, default=DEFAULT_FIGDIR)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = ed.load_config(args.config)
    ds = xr.open_zarr(cfg["data"]["cube"])
    sel = ed.select_matrix(ds, cfg)
    observed, t = sel["observed"], sel["t"]
    alpha = float(cfg["filter"]["alpha"])
    ed.check_stability(t, alpha, args.p, float(cfg["filter"]["stability_factor"]))

    # Reproduce edineof()'s split exactly: same seed, same helper, same derived gaps.
    mu = float(np.mean(sel["X"][observed]))
    X0 = np.where(observed, sel["X"] - mu, 0.0)
    rng = np.random.default_rng(int(cfg["cv"]["seed"]))
    day_cv, point_cv = ed.choose_cv_points(observed, cfg, rng)
    if not day_cv.any():
        raise SystemExit("cv.day_frac is 0, so there is no day holdout to score against")
    cv_any = day_cv | point_cv
    gaps = (~observed) | cv_any
    # sd BEFORE zeroing, over the entries that remain data -- the same order edineof uses, and
    # the order matters: after the zeroing the held-out entries would drag it toward zero.
    sd = float(np.std(X0[observed & ~cv_any]))
    X0[gaps] = 0.0

    log.info("refitting with %d whole days held out (%d px), k=%d p=%d ...",
             int((day_cv.any(axis=0)).sum()), int(day_cv.sum()), args.k, args.p)
    X, hist = ed.fill(X0, gaps, args.k, t, alpha, args.p, float(cfg["em"]["tol"]),
                      int(cfg["em"]["max_iter"]), sd, label="holdout")
    log.info("  %d EM iterations, converged=%s", hist["n_iter"], hist["converged"])

    # unstandardize returns KELVIN. Everything below is displayed and differenced in degC, so
    # convert once here -- a difference is identical either way, but the axis labels are not.
    pred = plotting.to_celsius(ed.unstandardize(X + mu, ds, cfg, sel))
    # p = 0 in z is exactly 0, which un-standardizes to the seasonal climatology: the baseline
    # these days would get with no filter at all. Scored on the same pixels, it is what the
    # reconstruction has to beat to be worth anything.
    clim = plotting.to_celsius(ed.unstandardize(np.zeros_like(X), ds, cfg, sel))

    # The truth, put through the SAME inversion as the prediction so the two are on one scale
    # and any error in the seasonal/scale fields cancels out of the difference.
    meas = plotting.to_celsius(
        ed.unstandardize(np.where(observed, sel["X"], np.nan), ds, cfg, sel))
    water = sel["water"]
    extent = plotting.extent_km(ds)
    times = pd.to_datetime(ds["time"].values)

    held = np.flatnonzero(day_cv.any(axis=0))          # columns = dates
    out = args.fig_dir / cfg["data"]["aoi"]
    rows, clim_sq, clim_n = [], 0.0, 0
    for j in held:
        m = np.isfinite(meas[j]) & water
        if m.sum() < args.min_obs:
            continue
        d = (pred[j] - meas[j])[m]
        c = (clim[j] - meas[j])[m]
        clim_sq += float((c ** 2).sum()); clim_n += int(m.sum())
        stats = dict(date=str(times[j])[:10], n=int(m.sum()), bias=float(d.mean()),
                     med=float(np.median(d)), rmse=float(np.sqrt((d ** 2).mean())),
                     sd=float(d.std()), clim_rmse=float(np.sqrt((c ** 2).mean())))
        rows.append(stats)
        vals = np.concatenate([meas[j][m], pred[j][m]])
        vmin, vmax = np.percentile(vals, [2, 98])
        dmax = max(float(np.percentile(np.abs(d), 98)), 0.1)
        day_figure(meas[j], pred[j], water, stats["date"], stats, extent,
                   float(vmin), float(vmax), dmax, out)

    if not rows:
        raise SystemExit("no held-out day had enough measured pixels")
    df = pd.DataFrame(rows)
    clim_rmse = float(np.sqrt(clim_sq / clim_n))
    summary_figure(df, args.k, args.p, clim_rmse, out)
    df.to_csv(out / "recon_comparison.csv", index=False)
    log.info("wrote %s", out / "recon_comparison.csv")

    pooled = float(np.sqrt((df["rmse"] ** 2 * df["n"]).sum() / df["n"].sum()))
    log.info("\n%s", df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    log.info("\npooled over %d held-out days / %d px: bias %+.2f K, RMSE %.2f K",
             len(df), int(df["n"].sum()),
             float((df["bias"] * df["n"]).sum() / df["n"].sum()), pooled)
    log.info("seasonal climatology on the same pixels: RMSE %.2f K  -->  eDINEOF is %.0f%% %s",
             clim_rmse, 100 * abs(pooled - clim_rmse) / clim_rmse,
             "BETTER" if pooled < clim_rmse else "WORSE -- the filter is not earning its place")


if __name__ == "__main__":
    main()
