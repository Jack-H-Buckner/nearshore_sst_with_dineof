"""Would the low-rank field make a better outlier filter than the mixture model?

Takes each sensor's RAW, UNFILTERED scene -- every pixel the sensor retrieved, cloud and haze
included -- puts it on the composite scale, and differences it against `sst_lowrank`, the
rank-k model field for that same day. If that residual separates the pixels the current filter
removed from the ones it kept, it is a viable outlier statistic on its own.

    raw (unfiltered) | sst_lowrank | raw - lowrank | histogram split by the filter's verdict

The histogram is the test. Cleanly separated modes mean the low-rank residual carries the same
information the mixture model does, from a completely different direction -- a smooth field fit
across 365 days rather than a per-scene spatial mixture. Heavy overlap means it does not.

The separation is scored as AUC: the probability that a randomly chosen REMOVED pixel has a
larger |residual| than a randomly chosen KEPT one. 0.5 is a coin flip, 1.0 is perfect.

TWO THINGS THAT MAKE THIS FAVOURABLE TO THE LOW-RANK FIELD, and both must be said:

  1. `sst_lowrank` was FIT on the filtered data. The pixels the mixture model removed were
     gaps during that fit, so the model was never contaminated by them and is free to
     disagree with them. An outlier filter built this way in production would have to
     bootstrap from something, and the first pass would not have this advantage.
  2. `sst_lowrank` is not a leave-one-day-out prediction. V[j] for date j is informed by date
     j's own surviving pixels through B = X'X, so the field is already pulled toward that
     day's data -- see the channel's own comment in edineof.build_dataset.

So read a high AUC as "worth building properly and testing honestly", not as "this already
works". The clean test would refit with the day held out, which is what compare_recon.py does.

Usage (from the repo root, in the `coastal_sst_data` env):

    python prototypes/DINEOF/src/compare_lowrank.py
    ... --sensors eco --n-days 8
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
from cube_figures import DROP_PIXEL, GRID, INK, INK_SECONDARY, SURFACE, VALID_COLOR

log = logging.getLogger("compare_lowrank")

ROOT = Path(__file__).resolve().parents[1]
DC = ROOT / "data" / "datacube"
DEFAULT_FIGDIR = ROOT / "figures" / "lowrank_outlier"
DIFF_CMAP = "RdBu_r"

SENSORS = {"eco": ("eco_sst_v002", "eco_sst_v002_dineof", "ECOSTRESS v002"),
           "lst": ("lst_sst", "lst_sst_dineof", "Landsat")}


def auc(score: np.ndarray, positive: np.ndarray) -> float:
    """P(score of a random positive > score of a random negative), via rank sums.

    Mann-Whitney U over ranks rather than a threshold sweep: no bin choice, ties handled by
    average rank, and it is the same number an ROC integration would give.
    """
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if not n_pos or not n_neg:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype="float64")
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks within ties, so a degenerate score cannot manufacture separation
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def day_figure(raw, low, kept, removed, water, date, label, stats, extent, out: Path) -> None:
    obs = kept | removed
    resid = np.where(obs, raw - low, np.nan)

    fig, axes = plt.subplots(1, 4, figsize=(16.8, 4.7), dpi=130,
                             gridspec_kw=dict(width_ratios=[1, 1, 1, 0.95]))
    cmap = plotting.sst_cmap()
    vals = raw[obs]
    vmin, vmax = np.percentile(vals, [2, 98])

    im = plotting.panel(axes[0], raw, water, vmin=vmin, vmax=vmax, extent=extent, cmap=cmap,
                        title=f"{label} RAW, unfiltered\n{int(obs.sum()):,} retrieved px "
                              f"({removed.sum() / max(obs.sum(), 1):.0%} removed by the filter)")
    fig.colorbar(im, ax=axes[0], shrink=0.80, label="SST [degC]")

    im = plotting.panel(axes[1], low, water, vmin=vmin, vmax=vmax, extent=extent, cmap=cmap,
                        title=f"sst_lowrank (k={stats['k']})\nthe rank-k model field")
    fig.colorbar(im, ax=axes[1], shrink=0.80, label="SST [degC]")

    # The scale has to show the OUTLIERS, which is the whole question. A percentile over all
    # observed pixels does not: on a day where the filter removed 7% of the frame, the kept
    # population (sd ~0.4 K) sets the range and the removed pixels at 3-4 K all saturate to
    # the end colour, so the panel cannot show how far out they actually are. Take the wider
    # of a high overall percentile and a mid percentile of the REMOVED population, so the
    # group being judged is always on scale.
    dmax = float(np.percentile(np.abs(resid[obs]), 99.5))
    if removed.any():
        dmax = max(dmax, float(np.percentile(np.abs(resid[removed]), 90)))
    dmax = max(dmax, 0.1)
    dcm = plt.get_cmap(DIFF_CMAP).copy()
    dcm.set_bad(alpha=0.0)
    imd = plotting.panel(axes[2], resid, water, vmin=-dmax, vmax=dmax, extent=extent, cmap=dcm,
                         title=f"raw - lowrank\nthe candidate outlier statistic")
    fig.colorbar(imd, ax=axes[2], shrink=0.80, label="residual [K]")

    ax = axes[3]
    rk, rr = resid[kept], resid[removed]
    allr = resid[obs]
    lo, hi = np.percentile(allr, [0.5, 99.5])
    pad = 0.05 * max(hi - lo, 1e-6)
    edges = np.linspace(lo - pad, hi + pad, 61)
    n_over = int((allr < edges[0]).sum() + (allr > edges[-1]).sum())
    ax.hist([np.clip(rk, edges[0], edges[-1]), np.clip(rr, edges[0], edges[-1])],
            bins=edges, stacked=True, color=[VALID_COLOR, DROP_PIXEL], edgecolor="none",
            label=[f"filter KEPT ({rk.size:,})", f"filter REMOVED ({rr.size:,})"])
    ax.axvline(0.0, color=INK, linewidth=1.2)
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
    ax.set_xlabel(f"raw - lowrank [K]{note}", fontsize=7, color=INK_SECONDARY)
    ax.set_title(f"AUC {stats['auc']:.3f}   (|residual| as an outlier score)\n"
                 f"kept {np.median(rk):+.2f} +/- {rk.std():.2f}   "
                 f"removed {np.median(rr):+.2f} +/- {rr.std():.2f} K", fontsize=8)

    for a in axes[:3]:
        a.set_xlabel("km east", fontsize=7)
        a.tick_params(labelsize=6)
    axes[0].set_ylabel("km north", fontsize=7)

    fig.suptitle(f"{date}   {label}: would the low-rank residual work as an outlier filter?   "
                 f"(raw is on the composite scale, offset {stats['offset']:+.3f} K removed)",
                 fontsize=10, ha="left", x=0.01, y=0.985)
    fig.subplots_adjust(top=0.82, bottom=0.14, left=0.04, right=0.985, wspace=0.24)
    plotting.save(fig, out / f"{date}_{stats['sid']}.png")


def summary_figure(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.4), dpi=150, facecolor=SURFACE)
    for sid, g in df.groupby("sensor"):
        ax.scatter(g["removed_frac"], g["auc"], s=60, alpha=0.85,
                   label=f"{sid} ({len(g)} days)")
        for _, r in g.iterrows():
            ax.annotate(r["date"][5:], (r["removed_frac"], r["auc"]),
                        textcoords="offset points", xytext=(6, 3), fontsize=6,
                        color=INK_SECONDARY)
    ax.axhline(0.5, color=INK, linewidth=1.4, linestyle=(0, (4, 3)), label="coin flip")
    ax.set_xlabel("fraction of retrieved pixels the mixture filter removed", fontsize=9,
                  color=INK_SECONDARY)
    ax.set_ylabel("AUC of |raw - lowrank|", fontsize=9, color=INK_SECONDARY)
    ax.set_title("Does the low-rank residual recover the mixture filter's decisions?",
                 fontsize=11, color=INK, loc="left")
    ax.set_ylim(0.35, 1.02)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
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
    ap.add_argument("--aoi", default="admiralty_inlet")
    ap.add_argument("--sensors", nargs="+", default=["eco", "lst"])
    ap.add_argument("--n-days", type=int, default=6,
                    help="days per sensor, chosen for the most retrieved pixels")
    ap.add_argument("--min-removed", type=int, default=2000,
                    help="a day needs this many removed px for the AUC to mean anything")
    ap.add_argument("--fig-dir", type=Path, default=DEFAULT_FIGDIR)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    dn = xr.open_zarr(DC / f"{args.aoi}_dineof.zarr")
    if "sst_lowrank" not in dn:
        raise SystemExit("this cube has no sst_lowrank; re-run edineof.py")
    ft = xr.open_zarr(DC / f"{args.aoi}_filtered.zarr")
    offs = pd.read_csv(DC / "offsets.csv").set_index("member")["offset_mean"].to_dict()

    water = plotting.water_mask(dn)
    extent = plotting.extent_km(dn)
    times = pd.to_datetime(dn["time"].values)
    k = int(dn["sst_lowrank"].attrs.get("k_modes", -1))

    low_all = dn["sst_lowrank"].values.astype("float32")
    kelvin = 273.15 if np.nanmedian(low_all) > 150 else 0.0   # one offset, both fields
    low_all = low_all - kelvin

    out = args.fig_dir / args.aoi
    rows = []
    for sid in args.sensors:
        raw_v, filt_v, label = SENSORS[sid]
        off = float(offs[sid])
        raw_all = ft[raw_v].values.astype("float32") - off - kelvin
        filt_all = ft[filt_v].values.astype("float32")

        obs = np.isfinite(raw_all) & water[None]
        rem = obs & ~np.isfinite(filt_all)
        cand = np.argsort(-obs.sum(axis=(1, 2)))
        picked = 0
        for j in cand:
            if picked >= args.n_days:
                break
            if rem[j].sum() < args.min_removed or not np.isfinite(low_all[j][water]).any():
                continue
            resid = raw_all[j] - low_all[j]
            # 161 water px were dropped from the matrix for having no observations at all, so
            # sst_lowrank is NaN there. Left in, they sort to the end of the AUC ranking and
            # silently inflate it, and they make the medians NaN.
            ok = np.isfinite(resid)
            kept_j = obs[j] & np.isfinite(filt_all[j]) & ok
            rem_j = rem[j] & ok
            if not kept_j.any() or rem_j.sum() < args.min_removed:
                # A scene the OFFSET gate dropped whole has no kept class, so there is nothing
                # for an outlier score to separate. Excluded rather than reported as AUC = nan.
                log.info("  %s %s: skipped (%d kept, %d removed)", sid, str(times[j])[:10],
                         int(kept_j.sum()), int(rem_j.sum()))
                continue
            sel = kept_j | rem_j
            a = auc(np.abs(resid[sel]), rem_j[sel])
            stats = dict(sid=sid, date=str(times[j])[:10], k=k, offset=off, auc=a,
                         n_obs=int(sel.sum()), n_removed=int(rem_j.sum()),
                         removed_frac=float(rem_j.sum() / sel.sum()),
                         kept_med=float(np.median(resid[kept_j])),
                         kept_sd=float(resid[kept_j].std()),
                         rem_med=float(np.median(resid[rem_j])),
                         rem_sd=float(resid[rem_j].std()))
            rows.append(stats)
            day_figure(raw_all[j], low_all[j], kept_j, rem_j, water, stats["date"], label,
                       stats, extent, out)
            picked += 1

    if not rows:
        raise SystemExit("no day had enough removed pixels")
    df = pd.DataFrame(rows).rename(columns={"sid": "sensor"})
    summary_figure(df, out)
    df.to_csv(out / "lowrank_outlier.csv", index=False)
    log.info("wrote %s", out / "lowrank_outlier.csv")
    cols = ["date", "sensor", "n_obs", "n_removed", "removed_frac", "auc",
            "kept_med", "kept_sd", "rem_med", "rem_sd"]
    log.info("\n%s", df[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    log.info("\nmedian AUC: %s",
             ", ".join(f"{s} {g['auc'].median():.3f}" for s, g in df.groupby("sensor")))


if __name__ == "__main__":
    main()
