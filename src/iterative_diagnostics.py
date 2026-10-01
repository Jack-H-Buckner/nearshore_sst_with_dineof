"""Diagnostics for the iterative DINEOF-baseline filter: what it flags at each iteration, and
how that compares with the MODIS-baseline filter from build_cube.py.

Reads only what iterative_filter.py wrote (the cube's `<id>_keep_history_iter` bitmask and
`sst_baseline_iter`, and its per-scene CSVs) plus build_cube's `filter_report.csv`, so it
re-draws in a minute without re-running the loop.

Figures, under <fig_dir>_diagnostics[_tag]/<aoi>/:

  flag_changes.png        per iteration: pixels newly flagged and pixels restored, per sensor.
                          The loop's convergence, in pixels rather than fractions.
  iterations_<date>_<sid>.png
                          one scene through the loop: raw | start | after iter 0 .. N-1 |
                          final vs MODIS. Each mask panel marks what changed at that step.
  agreement_maps_<sid>.png
                          per pixel, over all acquisitions: share removed by each filter,
                          their difference, and how often the pixel's verdict flipped.
  scene_comparison.png    per scene: kept fraction and scene offset, DINEOF vs MODIS, with the
                          offset band drawn. Whole-scene disagreements are the scene gate.
  residual_by_category.png
                          residual against the final DINEOF baseline, split by the two
                          filters' joint verdict.
  disagreements_<sid>.png contact sheet of the scenes the two filters disagree on most.
  fields_<n>.png          per day, side by side: the composite the final fit was given, the
                          DINEOF gap-filled analysis, the smoothed baseline the scenes were
                          classified against, and filled minus smoothed. Days are picked to
                          cover busy, typical, empty and no-MODIS days.
  loadings.png            per mode: the MODIS-only loadings the baseline was built from
                          against the full-data loadings sigma*V. Where they part is where
                          ECOSTRESS / Landsat were steering the old baseline. (MODIS-baseline
                          runs only.)

Plus diagnostics_scenes[_tag].csv: one row per scene, both filters side by side.

Usage (from the repo root):

    python src/iterative_diagnostics.py --config configs/config.iterative.admiralty_inlet.yaml
    ... --tag smoke                 # the outputs of a tagged run
    ... --scenes 2025-06-03:eco     # add specific scenes to the per-scene strips
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import iterative_filter as F                      # bridges seasonal_smoothing; config loader
import plotting
from cube_figures import GRID, INK, INK_MUTED, INK_SECONDARY, SURFACE

log = logging.getLogger("iterative_diagnostics")

# Categorical roles, from the dataviz reference palette (slots 1-3 are a validated adjacent
# run). Agreement is drawn recessive, disagreement saturated, so the eye lands on the argument.
KEEP = "#8fb8e8"            # kept (by both, on a comparison panel)
REMOVED = "#52514e"         # removed (by both); on an iteration panel, removed before this step
DINEOF_ONLY = "#eb6834"     # removed by the DINEOF filter only / newly flagged this iteration
MODIS_ONLY = "#1baf7a"      # removed by the MODIS filter only / restored this iteration
SENSOR = {"eco": ("#2a78d6", "o"), "lst": ("#eb6834", "s")}
VERDICT_MARK = {("KEPT", "KEPT"): "o", ("KEPT", "DROPPED"): "^",
                ("DROPPED", "KEPT"): "v", ("DROPPED", "DROPPED"): "X"}


# ==================================================================== loading

def load(cfg: dict, modis_report: Path) -> dict:
    """Everything the figures need, as numpy, per sensor."""
    out_path = cfg["data"]["out"]
    if not out_path.exists():
        raise SystemExit(f"no iterative cube at {out_path}; run iterative_filter.py first")
    ds = xr.open_zarr(out_path)
    need = [f"{sid}_keep_history_iter" for sid in cfg["sensors"]] + ["sst_baseline_iter"]
    missing = [v for v in need if v not in ds]
    if missing:
        raise SystemExit(f"{out_path.name} has no {missing}; it predates the history channels "
                         "-- re-run iterative_filter.py")

    water = np.asarray(ds[cfg["data"]["watervar"]].compute() > 0.5)
    times = ds["time"].values
    base = ds["sst_baseline_iter"].values
    offs = pd.read_csv(cfg["data"]["offsets"]).set_index("member")["offset_mean"].to_dict()
    sfx = cfg["data"]["modis_filtered_suffix"]
    out_dir = out_path.parent
    tag_sfx = out_path.stem.replace(Path(DEFAULT_OUT).stem, "")      # "" or "_<tag>"

    scenes = pd.read_csv(out_dir / cfg["output"]["scenes"])
    sh = Path(cfg["output"]["scenes"])
    history = pd.read_csv(out_dir / f"{sh.stem}_history{sh.suffix}")
    modis = pd.read_csv(modis_report) if modis_report.exists() else None
    if modis is None:
        log.warning("no %s; scene-gate comparisons use the cube's verdicts only", modis_report)

    sensors = {}
    for sid, s in cfg["sensors"].items():
        raw = ds[s["sst"]].values.astype("float32")
        raw = np.where(water[None], raw, np.nan)
        bits = ds[f"{sid}_keep_history_iter"].values
        n_iter = int(ds[f"{sid}_keep_history_iter"].attrs["n_iterations"])
        js = scenes.loc[scenes["sensor"] == sid, "t"].to_numpy()
        sensors[sid] = dict(
            label=s["label"], raw=raw, bits=bits, n_iter=n_iter, scenes=np.sort(js),
            modis_keep=np.isfinite(ds[f"{s['sst']}{sfx}"].values) & water[None],
            resid=raw - float(offs.get(sid, 0.0)) - base)
    loads = None
    if "full_loadings_iter" in ds:
        loads = dict(full=ds["full_loadings_iter"].values,
                     modis=(ds["baseline_loadings_iter"].values
                            if "baseline_loadings_iter" in ds else None),
                     modis_px=(ds["baseline_modis_px"].values
                               if "baseline_modis_px" in ds else None))
    source = ds["sst_baseline_iter"].attrs.get("baseline_source", "all")
    fields = None
    if "sst_filled_iter" in ds:
        fields = dict(filled=ds["sst_filled_iter"].values, base=base,
                      comp=ds["sst_composite_iter"].values,
                      src=ds["sst_composite_src_iter"].values,
                      order=ds["sst_composite_src_iter"].attrs["flag_meanings"].split(),
                      coarsen=int(ds["sst_filled_iter"].attrs.get("coarsen", 1)),
                      status=(ds["baseline_loading_status"].values
                              if "baseline_loading_status" in ds else None))
    return dict(ds=ds, water=water, times=times, sensors=sensors, scenes=scenes,
                history=history, modis=modis, extent=plotting.extent_km(ds), tag=tag_sfx,
                loads=loads, source=source, fields=fields)


DEFAULT_OUT = F.DEFAULTS["data"]["out"]


def kept_at(bits: np.ndarray, i: int) -> np.ndarray:
    """Keep verdict at step i: 0 = the starting mask, i = after iteration i-1."""
    return ((bits >> i) & 1).astype(bool)


# ==================================================================== the scene table

def scene_table(D: dict) -> pd.DataFrame:
    """One row per (sensor, scene): both filters' verdicts, offsets and pixel counts."""
    rows = []
    for sid, S in D["sensors"].items():
        final = kept_at(S["bits"], S["n_iter"])
        start = kept_at(S["bits"], 0)
        for j in S["scenes"]:
            obs = np.isfinite(S["raw"][j]) & D["water"]
            a, b = final[j][obs], S["modis_keep"][j][obs]
            flips = sum(int((kept_at(S["bits"], i)[j] != kept_at(S["bits"], i - 1)[j])[obs].sum())
                        for i in range(1, S["n_iter"] + 1))
            rows.append(dict(sensor=sid, date=str(D["times"][j])[:10], t=int(j),
                             n_obs=int(obs.sum()), n_start=int(start[j][obs].sum()),
                             kept_dineof=int(a.sum()), kept_modis=int(b.sum()),
                             both_keep=int((a & b).sum()), both_remove=int((~a & ~b).sum()),
                             dineof_only_remove=int((~a & b).sum()),
                             modis_only_remove=int((a & ~b).sum()),
                             agree=float((a == b).mean()) if obs.any() else np.nan,
                             kappa=F.cohen_kappa(a, b), total_flips=flips))
    df = pd.DataFrame(rows)
    sc = D["scenes"][["sensor", "date", "center", "verdict", "reason"]].rename(
        columns={"center": "center_dineof", "verdict": "verdict_dineof",
                 "reason": "reason_dineof"})
    df = df.merge(sc, on=["sensor", "date"], how="left")
    if D["modis"] is not None:
        m = D["modis"][["sensor", "date", "center", "verdict", "reason", "ref_cover"]].rename(
            columns={"center": "center_modis", "verdict": "verdict_modis",
                     "reason": "reason_modis", "ref_cover": "modis_ref_cover"})
        df = df.merge(m, on=["sensor", "date"], how="left")
    else:
        df["verdict_modis"] = np.where(df["kept_modis"] > 0, "KEPT", "DROPPED")
        df["center_modis"] = np.nan
    df["verdict_modis"] = df["verdict_modis"].fillna("DROPPED")
    df["disagree_px"] = df["dineof_only_remove"] + df["modis_only_remove"]
    return df


# ==================================================================== drawing helpers

def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=7)


def _map_axes(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)


def _categories(ax, water, extent, layers, title) -> None:
    """Flat categorical map: land grey, unobserved off-white, then each (mask, colour)."""
    ax.set_facecolor(plotting.NODATA_COLOR)
    plotting.flat(ax, ~water, plotting.LAND_COLOR, extent)
    for mask, color in layers:
        plotting.flat(ax, mask & water, color, extent)
    ax.set_title(title, fontsize=7, color=INK)
    _map_axes(ax)


def compare_layers(a, b, obs):
    """(DINEOF keep, MODIS keep, observed) -> the four joint-verdict layers."""
    return [(obs & a & b, KEEP), (obs & ~a & ~b, REMOVED),
            (obs & ~a & b, DINEOF_ONLY), (obs & a & ~b, MODIS_ONLY)]


def compare_legend(ax, a, b, obs, loc="lower left") -> None:
    n = max(int(obs.sum()), 1)
    frac = lambda m: f"{int(m.sum()) / n:.0%}"
    ax.legend(handles=[
        Patch(facecolor=KEEP, label=f"both keep {frac(obs & a & b)}"),
        Patch(facecolor=REMOVED, label=f"both remove {frac(obs & ~a & ~b)}"),
        Patch(facecolor=DINEOF_ONLY, label=f"DINEOF only removes {frac(obs & ~a & b)}"),
        Patch(facecolor=MODIS_ONLY, label=f"MODIS only removes {frac(obs & a & ~b)}")],
        loc=loc, fontsize=5.5, framealpha=0.92)


# ==================================================================== figures

def flag_changes_figure(D, out: Path, dpi: int) -> None:
    """Newly flagged and restored pixels at each iteration, one panel per sensor."""
    sids = list(D["sensors"])
    fig, axes = plt.subplots(1, len(sids), figsize=(5.2 * len(sids), 3.6), dpi=dpi,
                             squeeze=False)
    for ax, sid in zip(axes[0], sids):
        S = D["sensors"][sid]
        js = S["scenes"]
        obs = np.isfinite(S["raw"][js]) & D["water"][None]
        new, back = [], []
        for i in range(1, S["n_iter"] + 1):
            prev, cur = kept_at(S["bits"][js], i - 1), kept_at(S["bits"][js], i)
            new.append(int((prev & ~cur & obs).sum()))
            back.append(int((~prev & cur & obs).sum()))
        x = np.arange(S["n_iter"])
        w = 0.38
        # 2px surface gap between the paired bars.
        ax.bar(x - w / 2 - 0.01, new, w, color=DINEOF_ONLY, label="newly flagged")
        ax.bar(x + w / 2 + 0.01, back, w, color=MODIS_ONLY, label="restored")
        _style(ax)
        ax.set_yscale("symlog", linthresh=100)
        ax.set_xticks(x, [f"iter {i}" for i in x])
        ax.set_ylabel("pixels (all acquisitions)", fontsize=8, color=INK_SECONDARY)
        ax.set_title(f"{S['label']}: verdict changes per iteration\n"
                     f"({int(obs.sum()):,} observed px on {len(js)} scenes; iter 0 is "
                     "measured against the QC-only start)", fontsize=8, color=INK)
        ax.legend(fontsize=7, frameon=False)
        for xi, (n1, n2) in enumerate(zip(new, back)):
            ax.annotate(f"{n1:,}", (xi - w / 2, n1), ha="center", va="bottom", fontsize=6,
                        color=INK_SECONDARY)
            ax.annotate(f"{n2:,}", (xi + w / 2, max(n2, 1)), ha="center", va="bottom",
                        fontsize=6, color=INK_SECONDARY)
    fig.tight_layout()
    plotting.save(fig, out)


def iteration_strip(D, sid, j, row, out: Path, dpi: int) -> None:
    """One scene through the loop, then the final verdict against the MODIS filter's."""
    S = D["sensors"][sid]
    water, extent = D["water"], D["extent"]
    raw = S["raw"][j] - 273.15
    obs = np.isfinite(raw) & water
    n = S["n_iter"]
    ncol = n + 3
    fig, axes = plt.subplots(1, ncol, figsize=(2.6 * ncol, 3.3), dpi=dpi)

    vmin, vmax = np.percentile(raw[obs], [2, 98]) if obs.any() else (0, 1)
    im = plotting.panel(axes[0], raw, water, vmin=vmin, vmax=vmax, extent=extent)
    axes[0].set_title("raw scene [degC]", fontsize=7, color=INK)
    _map_axes(axes[0])
    fig.colorbar(im, ax=axes[0], shrink=0.7).ax.tick_params(labelsize=6)

    start = kept_at(S["bits"][j], 0)
    _categories(axes[1], water, extent, [(obs & start, KEEP), (obs & ~start, REMOVED)],
                f"start: QC + range\n{int((obs & ~start).sum()):,} removed")
    for i in range(1, n + 1):
        prev, cur = kept_at(S["bits"][j], i - 1), kept_at(S["bits"][j], i)
        new, back = obs & prev & ~cur, obs & ~prev & cur
        _categories(axes[i + 1], water, extent,
                    [(obs & cur, KEEP), (obs & ~cur & ~prev, REMOVED),
                     (new, DINEOF_ONLY), (back, MODIS_ONLY)],
                    f"after iter {i - 1}\n+{int(new.sum()):,} flagged  "
                    f"-{int(back.sum()):,} restored")

    a, b = kept_at(S["bits"][j], n), S["modis_keep"][j]
    ax = axes[-1]
    _categories(ax, water, extent, compare_layers(a, b, obs),
                f"final vs MODIS filter\nkappa {F.cohen_kappa(a[obs], b[obs]):.2f}")
    compare_legend(ax, a, b, obs)

    fig.legend(handles=[Patch(facecolor=KEEP, label="kept"),
                        Patch(facecolor=REMOVED, label="removed (already)"),
                        Patch(facecolor=DINEOF_ONLY, label="newly flagged this iteration"),
                        Patch(facecolor=MODIS_ONLY, label="restored this iteration")],
               loc="lower center", ncol=4, fontsize=7, frameon=False)
    cm = row.get("center_modis", np.nan)
    fig.suptitle(
        f"{S['label']}  {row['date']}  --  {int(obs.sum()):,} observed px, "
        f"{D['source'].upper()}-loadings baseline.   "
        f"DINEOF filter: {row['verdict_dineof']} (offset {row['center_dineof']:+.2f} K)   "
        f"MODIS filter: {row['verdict_modis']}"
        + (f" (offset {cm:+.2f} K)" if np.isfinite(cm) else ""),
        fontsize=8, ha="left", x=0.01, color=INK)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.78, bottom=0.13, wspace=0.08)
    plotting.save(fig, out)


def agreement_maps(D, sid, out: Path, dpi: int) -> None:
    """Per pixel over all acquisitions: removal share by each filter, difference, flip rate."""
    S = D["sensors"][sid]
    js = S["scenes"]
    water, extent = D["water"], D["extent"]
    obs = np.isfinite(S["raw"][js]) & water[None]
    n_obs = obs.sum(axis=0)
    final = kept_at(S["bits"][js], S["n_iter"])
    rem_d = (obs & ~final).sum(axis=0)
    rem_m = (obs & ~S["modis_keep"][js]).sum(axis=0)
    flips = np.zeros(water.shape)
    for i in range(1, S["n_iter"] + 1):
        flips += (obs & (kept_at(S["bits"][js], i) != kept_at(S["bits"][js], i - 1))).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        fd = np.where(n_obs > 0, rem_d / n_obs, np.nan)
        fm = np.where(n_obs > 0, rem_m / n_obs, np.nan)
        fl = np.where(n_obs > 0, flips / n_obs, np.nan)

    seq = plt.get_cmap("Blues").copy()
    seq.set_bad(alpha=0.0)
    # The same two hues as every categorical panel, through a neutral grey midpoint.
    div = LinearSegmentedColormap.from_list("modis_dineof", [MODIS_ONLY, "#e1e0d9", DINEOF_ONLY])
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), dpi=dpi)
    panels = [(fd, seq, 0, 1, "share of observations removed\nDINEOF filter", "share"),
              (fm, seq, 0, 1, "share of observations removed\nMODIS filter", "share"),
              (fd - fm, div, -0.5, 0.5, "DINEOF minus MODIS\n(orange: DINEOF removes more, aqua: MODIS)",
               "difference in share"),
              (fl, plt.get_cmap("Purples").copy(), 0,
               max(float(np.nanpercentile(fl, 99)), 0.05) if np.isfinite(fl).any() else 1,
               "verdict flips per observation\nacross the loop", "flips / obs")]
    for ax, (arr, cmap, lo, hi, title, lab) in zip(axes, panels):
        cmap.set_bad(alpha=0.0)
        im = plotting.panel(ax, arr, water, vmin=lo, vmax=hi, extent=extent, cmap=cmap)
        ax.set_title(title, fontsize=8, color=INK)
        _map_axes(ax)
        fig.colorbar(im, ax=ax, shrink=0.75, label=lab).ax.tick_params(labelsize=6)
    tot = max(int(obs.sum()), 1)
    fig.suptitle(f"{S['label']}: {len(js)} acquisitions, {tot:,} observed px.  Removed: DINEOF "
                 f"{int(rem_d.sum()) / tot:.1%}, MODIS {int(rem_m.sum()) / tot:.1%}",
                 fontsize=9, color=INK, ha="left", x=0.01)
    fig.tight_layout()
    plotting.save(fig, out)


def scene_comparison(df: pd.DataFrame, cfg: dict, out: Path, dpi: int,
                     source: str = "all") -> None:
    """Kept fraction and scene offset per scene, DINEOF against MODIS."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), dpi=dpi)
    lo, hi = cfg["filter"]["offset_lower"], cfg["filter"]["offset_upper"]

    ax = axes[0]
    for sid, g in df.groupby("sensor"):
        c, _ = SENSOR.get(sid, ("#4a3aa7", "o"))
        for (vd, vm), m in VERDICT_MARK.items():
            h = g[(g["verdict_dineof"] == vd) & (g["verdict_modis"] == vm)]
            if len(h):
                ax.scatter(h["kept_modis"] / h["n_obs"], h["kept_dineof"] / h["n_obs"],
                           s=np.clip(h["n_obs"] / 600, 10, 120), marker=m, color=c,
                           edgecolor=SURFACE, linewidth=0.8, alpha=0.85)
    ax.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=1, linestyle="--")
    _style(ax)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("MODIS filter: share of observed px kept", fontsize=8)
    ax.set_ylabel("DINEOF filter: share of observed px kept", fontsize=8)
    ax.set_title("pixels kept per scene (marker area ~ observed px)\nx = 0 or y = 0: that "
                 "filter dropped the whole scene", fontsize=8, color=INK)

    ax = axes[1]
    ok = np.isfinite(df["center_modis"]) & np.isfinite(df["center_dineof"])
    for sid, g in df[ok].groupby("sensor"):
        c, _ = SENSOR.get(sid, ("#4a3aa7", "o"))
        for (vd, vm), m in VERDICT_MARK.items():
            h = g[(g["verdict_dineof"] == vd) & (g["verdict_modis"] == vm)]
            if len(h):
                ax.scatter(h["center_modis"], h["center_dineof"], s=26, marker=m, color=c,
                           edgecolor=SURFACE, linewidth=0.8, alpha=0.85)
    if ok.any():
        lim = np.nanpercentile(np.abs(df.loc[ok, ["center_modis", "center_dineof"]].values), 99)
        lim = max(float(lim) + 0.5, float(hi or 0) + 1, abs(float(lo or 0)) + 1)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.plot([-lim, lim], [-lim, lim], color=INK_MUTED, linewidth=1, linestyle="--")
    if lo is not None and hi is not None:
        ax.axvspan(lo, hi, color=GRID, alpha=0.5, zorder=0)
        ax.axhspan(lo, hi, color=GRID, alpha=0.5, zorder=0)
    _style(ax)
    ax.set_xlabel("scene offset vs MODIS covariate [K]", fontsize=8)
    ax.set_ylabel(f"scene offset vs DINEOF baseline ({source} loadings) [K]", fontsize=8)
    ax.set_title(f"the scene gate: accepted band [{lo}, {hi}] K shaded on both axes",
                 fontsize=8, color=INK)

    handles = [Line2D([], [], marker="o", linestyle="", color=SENSOR[s][0], label=s)
               for s in df["sensor"].unique() if s in SENSOR]
    names = {("KEPT", "KEPT"): "kept by both", ("KEPT", "DROPPED"): "dropped by MODIS only",
             ("DROPPED", "KEPT"): "dropped by DINEOF only", ("DROPPED", "DROPPED"): "dropped by both"}
    handles += [Line2D([], [], marker=m, linestyle="", color=INK_SECONDARY, label=names[k])
                for k, m in VERDICT_MARK.items()]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=7,
               frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    plotting.save(fig, out)


def residual_by_category(D, out: Path, dpi: int) -> None:
    """Residual vs the final DINEOF baseline, split by the joint verdict of the two filters."""
    sids = list(D["sensors"])
    fig, axes = plt.subplots(1, len(sids), figsize=(6 * len(sids), 3.8), dpi=dpi,
                             squeeze=False)
    bins = np.linspace(-12, 8, 161)
    for ax, sid in zip(axes[0], sids):
        S = D["sensors"][sid]
        js = S["scenes"]
        obs = np.isfinite(S["raw"][js]) & D["water"][None] & np.isfinite(S["resid"][js])
        a, b, r = kept_at(S["bits"][js], S["n_iter"]), S["modis_keep"][js], S["resid"][js]
        # "both keep" drawn in the palette's full-strength blue: the tint used on maps is too
        # light to read as a line.
        for mask, color, lab in ((obs & a & b, "#2a78d6", "both keep"),
                                 (obs & ~a & ~b, REMOVED, "both remove"),
                                 (obs & ~a & b, DINEOF_ONLY, "DINEOF only removes"),
                                 (obs & a & ~b, MODIS_ONLY, "MODIS only removes")):
            v = r[mask]
            if v.size:
                ax.hist(np.clip(v, bins[0], bins[-1]), bins=bins, histtype="step",
                        linewidth=2, color=color, label=f"{lab} (n={v.size:,}, "
                                                        f"median {np.median(v):+.2f} K)")
        _style(ax)
        ax.set_yscale("log")
        ax.set_xlabel("raw - offset - final DINEOF baseline [K]  (clipped to [-12, 8])",
                      fontsize=8)
        ax.set_ylabel("pixels", fontsize=8)
        ax.set_title(f"{S['label']}: where each filter's removals sit against the baseline",
                     fontsize=8, color=INK)
        ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.tight_layout()
    plotting.save(fig, out)


def disagreement_sheet(D, df, sid, out: Path, dpi: int, n: int = 24, ncols: int = 6) -> None:
    """The n scenes with the most disagreeing pixels, as joint-verdict maps."""
    S = D["sensors"][sid]
    top = df[df["sensor"] == sid].sort_values("disagree_px", ascending=False).head(n)
    if top.empty:
        return
    nrows = int(np.ceil(len(top) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.9 * nrows + 0.8), dpi=dpi,
                             squeeze=False, gridspec_kw=dict(hspace=0.35, wspace=0.06))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, r) in zip(axes.flat, top.iterrows()):
        ax.axis("on")
        j = int(r["t"])
        obs = np.isfinite(S["raw"][j]) & D["water"]
        a, b = kept_at(S["bits"][j], S["n_iter"]), S["modis_keep"][j]
        _categories(ax, D["water"], D["extent"], compare_layers(a, b, obs),
                    f"{r['date']}  {int(r['disagree_px']):,} px differ\n"
                    f"DINEOF {r['verdict_dineof'].lower()} / MODIS {r['verdict_modis'].lower()}")
    fig.legend(handles=[Patch(facecolor=KEEP, label="both keep"),
                        Patch(facecolor=REMOVED, label="both remove"),
                        Patch(facecolor=DINEOF_ONLY, label="DINEOF only removes"),
                        Patch(facecolor=MODIS_ONLY, label="MODIS only removes")],
               loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle(f"{S['label']}: the {len(top)} scenes where the filters disagree most",
                 fontsize=10, color=INK, ha="left", x=0.01)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.05)
    plotting.save(fig, out)


STATUS_WORDS = {0: "own MODIS", 1: "MODIS from neighbours", 2: "no MODIS: climatology"}


def pick_field_days(D, n_each: int, extra: list[str]) -> list[int]:
    """Busy, typical and empty days, and days no MODIS reached -- sorted by date."""
    Fd = D["fields"]
    n_obs = (np.isfinite(Fd["comp"]) & D["water"][None]).sum(axis=(1, 2))
    have = np.flatnonzero(n_obs > 0)
    picks = list(have[np.argsort(-n_obs[have])][:n_each])                    # busiest
    mid = have[np.argsort(np.abs(n_obs[have] - np.median(n_obs[have])))]
    picks += [j for j in mid if j not in picks][:n_each]                     # typical
    empty = np.flatnonzero(n_obs == 0)
    if empty.size:
        picks += list(empty[np.linspace(0, empty.size - 1, min(n_each, empty.size)).astype(int)])
    if Fd["status"] is not None:
        fb = np.flatnonzero(Fd["status"] == 2)
        if fb.size:
            picks += list(fb[np.linspace(0, fb.size - 1, min(n_each, fb.size)).astype(int)])
    dates = [str(t)[:10] for t in D["times"]]
    for d in extra:
        if d in dates:
            picks.append(dates.index(d))
        else:
            log.warning("--field-dates %s is not on the cube's time axis", d)
    return sorted(set(int(j) for j in picks))


def fields_figure(D, days: list[int], out: Path, dpi: int) -> None:
    """One row per day: composite input | DINEOF filled | smoothed baseline | filled - smoothed."""
    Fd, water, extent = D["fields"], D["water"], D["extent"]
    fig, axes = plt.subplots(len(days), 4, figsize=(13.5, 3.0 * len(days) + 0.5), dpi=dpi,
                             squeeze=False, layout="constrained")
    cmap = plotting.sst_cmap()
    div = plt.get_cmap("RdBu_r").copy()
    div.set_bad(alpha=0.0)
    for r, j in enumerate(days):
        comp = Fd["comp"][j] - 273.15
        fill = Fd["filled"][j] - 273.15
        base = Fd["base"][j] - 273.15
        obs = np.isfinite(comp) & water
        # One colour range per row, shared by all three temperature panels so they compare.
        vals = np.concatenate([fill[water & np.isfinite(fill)], base[water & np.isfinite(base)]])
        lo, hi = np.percentile(vals, [2, 98]) if vals.size else (0.0, 1.0)
        diff = fill - base
        dl = float(np.nanpercentile(np.abs(diff[water]), 98)) if np.isfinite(diff[water]).any() \
            else 1.0
        dl = max(dl, 0.25)

        bits = np.bitwise_or.reduce(Fd["src"][j][obs]) if obs.any() else 0
        members = [m for i, m in enumerate(Fd["order"]) if bits & (1 << i)] or ["none"]
        status = (STATUS_WORDS[int(Fd["status"][j])] if Fd["status"] is not None else "")
        row = axes[r]
        im = plotting.panel(row[0], comp, water, vmin=lo, vmax=hi, extent=extent, cmap=cmap)
        row[0].set_title(f"{str(D['times'][j])[:10]}  input composite\n"
                         f"{int(obs.sum()):,} px from {'+'.join(members)}", fontsize=7,
                         color=INK)
        plotting.panel(row[1], fill, water, vmin=lo, vmax=hi, extent=extent, cmap=cmap)
        row[1].set_title("DINEOF filled (full-data EOFs;\nobservations held fixed)",
                         fontsize=7, color=INK)
        plotting.panel(row[2], base, water, vmin=lo, vmax=hi, extent=extent, cmap=cmap)
        row[2].set_title(f"smoothed baseline ({D['source']} loadings)\n{status}", fontsize=7,
                         color=INK)
        imd = plotting.panel(row[3], diff, water, vmin=-dl, vmax=dl, extent=extent, cmap=div)
        rms = float(np.sqrt(np.nanmean(diff[water] ** 2)))
        row[3].set_title(f"filled - smoothed\nRMS {rms:.2f} K", fontsize=7, color=INK)
        for ax in row:
            _map_axes(ax)
        fig.colorbar(im, ax=list(row[:3]), shrink=0.85, pad=0.01, label="SST [degC]") \
            .ax.tick_params(labelsize=6)
        fig.colorbar(imd, ax=row[3], shrink=0.85, pad=0.02, label="K").ax.tick_params(labelsize=6)
    f = Fd["coarsen"]
    fig.suptitle("the fit's input, its gap-filled analysis, and the smoothed field the scenes are "
                 "filtered against" + (f"  (DINEOF fields on the {f}x coarsened grid)"
                                       if f > 1 else ""),
                 fontsize=9, color=INK, ha="left", x=0.01)
    plotting.save(fig, out)


def loadings_figure(D, out: Path, dpi: int) -> None:
    """MODIS-only loadings against full-data loadings, one panel per mode, shared time axis."""
    L = D["loads"]
    full, mod, px = L["full"], L["modis"], L["modis_px"]
    k = full.shape[0]
    t = pd.to_datetime(D["times"])
    fallback = np.all(mod == 0, axis=0)
    fig, axes = plt.subplots(k, 1, figsize=(12, 2.3 * k + 0.9), dpi=dpi, sharex=True,
                             squeeze=False)
    for i, ax in enumerate(axes[:, 0]):
        # Days nothing reached: the baseline there is climatology.
        for j in np.flatnonzero(fallback):
            ax.axvspan(t[j] - pd.Timedelta(hours=12), t[j] + pd.Timedelta(hours=12),
                       color=GRID, alpha=0.9, linewidth=0, zorder=0)
        ax.plot(t, full[i], color=INK_MUTED, linewidth=2, label="full data (sigma * V)")
        ax.plot(t, mod[i], color="#2a78d6", linewidth=2, label="MODIS only (baseline)")
        if px is not None:
            md = t[px > 0]
            ax.plot(md, np.full(md.size, ax.get_ylim()[0]), "|", color=INK_SECONDARY,
                    markersize=7, label="MODIS day")
        _style(ax)
        r = np.corrcoef(full[i], mod[i])[0, 1] if np.std(mod[i]) > 0 else np.nan
        rms = float(np.sqrt(np.mean((full[i] - mod[i]) ** 2)))
        ax.set_ylabel(f"mode {i + 1}", fontsize=8, color=INK_SECONDARY)
        ax.set_title(f"mode {i + 1}: r = {r:.2f}, RMS difference {rms:.2f} "
                     f"(RMS of full loading {float(np.sqrt(np.mean(full[i] ** 2))):.2f})",
                     fontsize=8, color=INK, loc="left")
    axes[0, 0].legend(fontsize=7, frameon=False, ncol=3, loc="upper right")
    fig.suptitle(f"temporal loadings of the final fit.  Shaded: {int(fallback.sum())} days no "
                 "MODIS reached (baseline = climatology)", fontsize=9, color=INK, ha="left",
                 x=0.01)
    fig.tight_layout()
    plotting.save(fig, out)


# ==================================================================== driver

def pick_strip_scenes(df: pd.DataFrame, n: int, extra: list[str]) -> list[tuple[str, int]]:
    """The most-disagreeing scenes per sensor, the busiest per sensor, and any asked for."""
    picked = []
    for sid, g in df.groupby("sensor"):
        for j in g.sort_values("disagree_px", ascending=False)["t"].head(n):
            picked.append((sid, int(j)))
        for j in g.sort_values("n_obs", ascending=False)["t"].head(max(1, n // 2)):
            picked.append((sid, int(j)))
    for spec in extra:
        date, _, sid = spec.partition(":")
        hit = df[(df["date"] == date) & ((df["sensor"] == sid) if sid else True)]
        if hit.empty:
            log.warning("--scenes %s: no such acquisition", spec)
        picked += [(r["sensor"], int(r["t"])) for _, r in hit.iterrows()]
    return list(dict.fromkeys(picked))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=F.DEFAULT_CONFIG, help="YAML config")
    p.add_argument("--tag", default=None, help="the --tag the loop was run with")
    p.add_argument("--modis-report", type=Path, default=None,
                   help="build_cube's filter_report.csv (default: next to data.filtered)")
    p.add_argument("--n-strips", type=int, default=4,
                   help="per-scene iteration strips per sensor, most-disagreeing first")
    p.add_argument("--scenes", nargs="*", default=[],
                   help="extra strips, as DATE or DATE:SENSOR")
    p.add_argument("--n-field-days", type=int, default=3,
                   help="days per category (busy, typical, empty, no-MODIS) in fields_*.png")
    p.add_argument("--field-dates", nargs="*", default=[],
                   help="extra days for fields_*.png, as YYYY-MM-DD")
    p.add_argument("--dpi", type=int, default=130)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = F.load_config(args.config)
    if args.tag:
        F.apply_tag(cfg, args.tag)
    report = args.modis_report or cfg["data"]["filtered"].parent / "filter_report.csv"
    D = load(cfg, report)
    fig_dir = cfg["output"]["fig_dir"]
    base_name = fig_dir.name.replace(f"_{args.tag}", "") if args.tag else fig_dir.name
    out_dir = fig_dir.with_name(f"{base_name}_diagnostics{D['tag']}") / cfg["data"]["aoi"]

    df = scene_table(D)
    csv = cfg["data"]["out"].parent / f"diagnostics_scenes{D['tag']}.csv"
    df.to_csv(csv, index=False)
    log.info("wrote %s (%d scenes)", csv, len(df))
    both = df.groupby(["sensor", "verdict_dineof", "verdict_modis"]).size().unstack(fill_value=0)
    log.info("scene verdicts (rows DINEOF, columns MODIS):\n%s", both.to_string())

    flag_changes_figure(D, out_dir / "flag_changes.png", args.dpi)
    scene_comparison(df, cfg, out_dir / "scene_comparison.png", args.dpi, D["source"])
    if D["fields"] is not None:
        days = pick_field_days(D, args.n_field_days, args.field_dates)
        for page, i in enumerate(range(0, len(days), 6)):
            fields_figure(D, days[i:i + 6], out_dir / f"fields_{page + 1}.png", args.dpi)
    else:
        log.info("no sst_filled_iter in this cube; fields_*.png skipped -- re-run the loop")
    if D["loads"] is not None and D["loads"]["modis"] is not None:
        loadings_figure(D, out_dir / "loadings.png", args.dpi)
    else:
        log.info("no MODIS loadings in this cube (baseline_source %s); loadings.png skipped",
                 D["source"])
    residual_by_category(D, out_dir / "residual_by_category.png", args.dpi)
    for sid in D["sensors"]:
        agreement_maps(D, sid, out_dir / f"agreement_maps_{sid}.png", args.dpi)
        disagreement_sheet(D, df, sid, out_dir / f"disagreements_{sid}.png", args.dpi)
    rows = {(r["sensor"], int(r["t"])): r for _, r in df.iterrows()}
    for sid, j in pick_strip_scenes(df, args.n_strips, args.scenes):
        iteration_strip(D, sid, j, rows[(sid, j)],
                        out_dir / f"iterations_{rows[(sid, j)]['date']}_{sid}.png", args.dpi)
    log.info("figures in %s", out_dir)


if __name__ == "__main__":
    main()
