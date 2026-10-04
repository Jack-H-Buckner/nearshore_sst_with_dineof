"""Stage 2: apply stage 1's offsets, run the iterative DINEOF cloud filter and the final fill.

The offsets are taken from `stage1_offsets/offsets.json` (the model chosen there) and applied
UNCHANGED -- neither the bootstrap nor the refit fits offsets -- so what was reviewed is what
is used. Everything else is the end-to-end pipeline (pipeline.run_pipeline): seasonal
standardization, the cloud loop, the coarse CV search, the warm-started full-resolution fill
and the MODIS-loading smooth field.

Diagnostics added here, beside the pipeline's own figures:

  cv_holdout_scenes.png   the held-out high-res scenes against the reconstruction made
                          without them (MODIS and the other dates only), and the difference
  cv_holdout_scatter.png  reconstruction vs held-out observation, and RMSE / bias per date
  removed_scenes_<id>.png the scenes the cloud filter removed most from: raw, smooth field,
                          residual, kept vs removed

    python src/run_region.py --config configs/region.<aoi>.yaml dineof [--tag T]
"""

from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import edineof as E
import iterative_filter as F
import pipeline as P
import plotting
import region as R
import stage1_offsets as S1
from cube_figures import GRID, INK, INK_MUTED, INK_SECONDARY, SURFACE

log = logging.getLogger("stage2_dineof")

KEPT = "#8fb8e8"
REMOVED = "#eb6834"


def chosen_offsets(payload: dict) -> tuple[dict, dict, dict]:
    """(sid -> offset, sid -> slope, sid -> model name) from stage 1's offsets.json."""
    off, slope, which = {}, {}, {}
    for sid, s in payload["sensors"].items():
        m = s["models"][s["chosen"]]
        off[sid], slope[sid], which[sid] = float(m["off"]), float(m["slope"]), s["chosen"]
    return off, slope, which


# ==================================================================== CV holdout

def holdout_predictions(out: dict, pcfg: dict) -> dict | None:
    """Reconstruct the held-out high-res pixels from everything else, on the CV grid.

    One fit at the selected point-CV setting with the validation-date high-res pixels removed
    -- the same holdout the CV search scored -- keeping the predictions, which the search
    itself does not. Returns grids in K on the coarse grid plus a per-date table, or None
    when no date was held out.
    """
    fin = out["fin"]
    sel, res = fin["sel_c"], fin["res_c"]
    held = sel["valid_msk"].T & sel["observed"]
    if not held.any():
        log.warning("no validation pixels were held out; CV holdout figures skipped")
        return None
    ecfg = pcfg["_iter"]["_edineof"]
    s = {"t_c": float(res["tc_opt"]), "alpha": float(res["alpha_opt"]), "p": int(res["p_opt"])}
    smoother = E.make_spatial_smoother(sel["water"], sel["keep"],
                                       float(ecfg["filter"].get("l_c", 0.0)),
                                       float(ecfg["filter"].get("spatial_alpha_max", 0.25)))
    fit = F.fit_fixed(sel["X"], sel["observed"] & ~held, sel["t"], int(res["k_opt"]), s, ecfg,
                      label="cv-holdout", smoother=smoother)
    sea = pcfg["seasonal"]
    seasonal = F.seasonal_field(out["raw"].times, sel["coef"], int(sea["n_harmonics"]),
                                float(sea["period_days"]))
    scale = np.asarray(sel["scale"], float)[None]

    def kelvin(Z):
        return F.to_grid(Z, sel) * scale + seasonal

    obs = kelvin(np.where(held, sel["X"], np.nan))
    pred_held = kelvin(np.where(held, fit["X"], np.nan))
    recon = kelvin(fit["X"])
    times = pd.to_datetime(out["raw"].times)
    rows = []
    for j in np.flatnonzero(held.any(axis=0)):
        d = (pred_held[j] - obs[j])[np.isfinite(obs[j])]
        rows.append(dict(date=times[j].strftime("%Y-%m-%d"), t=int(j), n=int(d.size),
                         rmse=float(np.sqrt(np.mean(d ** 2))), bias=float(d.mean())))
    table = pd.DataFrame(rows)
    allerr = (pred_held - obs)[np.isfinite(obs)]
    log.info("CV holdout: %d dates, %d coarse pixels, RMSE %.3f K, bias %+.3f K "
             "(k=%d, T_c=%g)", len(table), int(held.sum()), float(np.sqrt(np.mean(allerr ** 2))),
             float(allerr.mean()), int(res["k_opt"]), s["t_c"])
    return dict(obs=obs, pred=pred_held, recon=recon, table=table, sel=sel,
                k=int(res["k_opt"]), t_c=s["t_c"])


def _extent(sel: dict) -> list:
    c = sel["coords"]
    return [0.0, float(np.ptp(c["x"])) / 1000.0, 0.0, float(np.ptp(c["y"])) / 1000.0]


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=7)


def holdout_scenes_figure(h: dict, out: Path, dpi: int, n: int = 4) -> None:
    tab = h["table"].sort_values("n", ascending=False).head(n)
    water = h["sel"]["water"]
    extent = _extent(h["sel"])
    div = plt.get_cmap("RdBu_r").copy()
    div.set_bad(alpha=0.0)
    fig, axes = plt.subplots(len(tab), 3, figsize=(13, 3.9 * len(tab)), dpi=dpi,
                             layout="constrained", squeeze=False)
    for row, (_, r) in zip(axes, tab.iterrows()):
        j = int(r["t"])
        o, rc = h["obs"][j] - 273.15, h["recon"][j] - 273.15
        vals = np.r_[o[np.isfinite(o)], rc[np.isfinite(o)]]
        lo, hi = np.percentile(vals, [2, 98]) if vals.size else (0, 1)
        im = plotting.panel(row[0], o, water, vmin=lo, vmax=hi, extent=extent)
        row[0].set_title(f"{r['date']}  held-out high-res ({int(r['n'])} px)", fontsize=8,
                         color=INK)
        plotting.panel(row[1], rc, water, vmin=lo, vmax=hi, extent=extent)
        row[1].set_title("reconstruction without it", fontsize=8, color=INK)
        fig.colorbar(im, ax=row[:2], shrink=0.8, label="SST [degC]").ax.tick_params(labelsize=6)
        diff = h["pred"][j] - h["obs"][j]
        dl = max(float(np.nanpercentile(np.abs(diff), 98)), 0.25) if np.isfinite(diff).any() else 1
        imd = plotting.panel(row[2], diff, water, vmin=-dl, vmax=dl, extent=extent, cmap=div)
        row[2].set_title(f"reconstruction - held out: RMSE {r['rmse']:.2f} K, bias "
                         f"{r['bias']:+.2f} K", fontsize=8, color=INK)
        fig.colorbar(imd, ax=row[2], shrink=0.8, label="K").ax.tick_params(labelsize=6)
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"CV holdout: high-res pixels removed on validation dates and reconstructed "
                 f"(k={h['k']}, T_c={h['t_c']:g} d, CV grid)", fontsize=10, color=INK,
                 ha="left", x=0.01)
    plotting.save(fig, out)


def holdout_scatter_figure(h: dict, out: Path, dpi: int) -> None:
    ok = np.isfinite(h["obs"])
    o, p = h["obs"][ok] - 273.15, h["pred"][ok] - 273.15
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=dpi, layout="constrained",
                             gridspec_kw=dict(width_ratios=[1, 1.6]))
    ax = axes[0]
    ax.scatter(o, p, s=5, color="#2a78d6", alpha=0.35, edgecolor="none")
    lo, hi = float(min(o.min(), p.min())), float(max(o.max(), p.max()))
    ax.plot([lo, hi], [lo, hi], color=INK_MUTED, linestyle=":", linewidth=1)
    e = p - o
    _style(ax)
    ax.set_xlabel("held-out high-res [degC]", fontsize=8)
    ax.set_ylabel("reconstruction [degC]", fontsize=8)
    ax.set_title(f"all held-out pixels: n={e.size:,}, RMSE {np.sqrt(np.mean(e ** 2)):.3f} K, "
                 f"bias {e.mean():+.3f} K", fontsize=9, color=INK)
    ax = axes[1]
    t = h["table"]
    x = np.arange(len(t))
    ax.bar(x - 0.2, t["rmse"], 0.38, color="#2a78d6", label="RMSE")
    ax.bar(x + 0.2, t["bias"], 0.38, color="#1baf7a", label="bias")
    ax.axhline(0, color=INK_MUTED, linewidth=1)
    ax.set_xticks(x, t["date"], rotation=60, fontsize=6.5)
    _style(ax)
    ax.set_ylabel("K", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("per held-out date", fontsize=9, color=INK)
    fig.suptitle("CV holdout: reconstruction against the held-out high-res scenes", fontsize=10,
                 color=INK, ha="left", x=0.01)
    plotting.save(fig, out)


def removed_scenes_figure(ds: xr.Dataset, sid: str, sst: str, label: str, out: Path,
                          dpi: int, n: int = 4) -> None:
    """The scenes the cloud filter removed most from: raw | smooth | residual | kept/removed."""
    water = plotting.water_mask(ds)
    extent = plotting.extent_km(ds)
    raw = ds[sst].values
    keep = ds[f"{sid}_keep"].values.astype(bool)
    obs = np.isfinite(raw) & water[None]
    removed = (obs & ~keep).sum(axis=(1, 2))
    pick = [int(j) for j in np.argsort(-removed)[:n] if removed[j] > 0]
    if not pick:
        return
    off = ds[f"{sid}_offset"].values
    slope = float(ds[f"{sid}_offset"].attrs.get("slope", 1.0))
    times = pd.to_datetime(ds["time"].values)
    div = plt.get_cmap("RdBu_r").copy()
    div.set_bad(alpha=0.0)
    fig, axes = plt.subplots(len(pick), 4, figsize=(17, 3.8 * len(pick)), dpi=dpi,
                             layout="constrained", squeeze=False)
    for row, j in zip(axes, pick):
        r = raw[j] - 273.15
        sm = ds["sst_smooth"].values[j] - 273.15
        vals = np.r_[r[obs[j] & keep[j]], sm[water]]
        lo, hi = np.percentile(vals, [2, 98]) if vals.size else (0, 1)
        im = plotting.panel(row[0], r, water, vmin=lo, vmax=hi, extent=extent)
        row[0].set_title(f"{times[j]:%Y-%m-%d}  {label} raw", fontsize=8, color=INK)
        plotting.panel(row[1], sm, water, vmin=lo, vmax=hi, extent=extent)
        row[1].set_title("smooth field (MODIS loadings)", fontsize=8, color=INK)
        fig.colorbar(im, ax=row[:2], shrink=0.8, label="SST [degC]").ax.tick_params(labelsize=6)
        o = off[j] if np.isfinite(off[j]) else np.nanmean(off)
        res = np.where(obs[j], (raw[j] - o) / slope - ds["sst_smooth"].values[j], np.nan)
        rl = max(float(np.nanpercentile(np.abs(res), 98)), 0.5) if np.isfinite(res).any() else 1
        imr = plotting.panel(row[2], res, water, vmin=-rl, vmax=rl, extent=extent, cmap=div)
        row[2].set_title("residual: offset-corrected raw - smooth", fontsize=8, color=INK)
        fig.colorbar(imr, ax=row[2], shrink=0.8, label="K").ax.tick_params(labelsize=6)
        ax = row[3]
        ax.set_facecolor(plotting.NODATA_COLOR)
        plotting.flat(ax, ~water, plotting.LAND_COLOR, extent)
        plotting.flat(ax, obs[j] & keep[j], KEPT, extent)
        plotting.flat(ax, obs[j] & ~keep[j], REMOVED, extent)
        nobs = max(int(obs[j].sum()), 1)
        ax.set_title(f"cloud filter: {int(removed[j]):,} of {nobs:,} px removed "
                     f"({removed[j] / nobs:.0%})", fontsize=8, color=INK)
        ax.legend(handles=[Patch(facecolor=KEPT, label="kept"),
                           Patch(facecolor=REMOVED, label="removed")], loc="lower left",
                  fontsize=6.5, framealpha=0.9)
        for a in row:
            a.set_xticks([])
            a.set_yticks([])
    fig.suptitle(f"{label}: scenes with the most pixels removed by the iterative filter",
                 fontsize=10, color=INK, ha="left", x=0.01)
    plotting.save(fig, out)


# ==================================================================== driver

def run(rc: dict, *, tag: str | None = None, figures: bool = True) -> dict:
    t0 = time.time()
    payload = S1.read_offsets(rc, tag)
    off, slope, which = chosen_offsets(payload)
    log.info("stage 2: offsets from stage 1 -- %s", ", ".join(
        f"{sid} {which[sid]} (off {off[sid]:+.3f} K, slope {slope[sid]:.3f})" for sid in off))
    pcfg = rc["pipe"]
    d = R.stage_dirs(rc, tag)["dineof"]
    paths = {"cube": d / f"{rc['region']['aoi']}_dineof.zarr", "reports": d / "reports",
             "figures": d / "figures"}
    out = P.run_pipeline(pcfg, figures=figures, offsets=(off, slope), paths=paths,
                         extra_attrs={"stage1_offsets": json.dumps(payload, default=float)})
    # Keep only what the holdout needs (the coarse CV matrix and its selection) and release
    # the full-resolution arrays: holding them pushed a 16 GB machine into swap, and the
    # holdout's few-second coarse fit took 40 minutes.
    slim = {"fin": {"sel_c": out["fin"]["sel_c"], "res_c": out["fin"]["res_c"]},
            "raw": SimpleNamespace(times=out["raw"].times)}
    keep_keys = {"cube", "reports", "figures"}
    for k in [k for k in out if k not in keep_keys]:
        out[k] = None
    gc.collect()

    t1 = time.time()
    h = holdout_predictions(slim, pcfg)
    log.info("CV holdout reconstruction in %.1fs", time.time() - t1)
    if h is not None:
        h["table"].to_csv(paths["reports"] / "cv_holdout.csv", index=False)
    if figures:
        t1 = time.time()
        dpi = int(pcfg["output"]["figure_dpi"])
        try:
            if h is not None:
                holdout_scenes_figure(h, paths["figures"] / "cv_holdout_scenes.png", dpi)
                holdout_scatter_figure(h, paths["figures"] / "cv_holdout_scatter.png", dpi)
            with xr.open_zarr(paths["cube"]) as ds:
                for sid, s in pcfg["_iter"]["sensors"].items():
                    removed_scenes_figure(ds, sid, s["sst"], s["label"],
                                          paths["figures"] / f"removed_scenes_{sid}.png", dpi)
        except Exception:
            log.exception("stage-2 figures failed; the cube and reports are intact")
        log.info("stage-2 figures in %.1fs", time.time() - t1)
    log.info("stage 2 done in %.1f min -> %s", (time.time() - t0) / 60, paths["cube"])
    return dict(dir=d, cube=paths["cube"], holdout=h, out=out)
