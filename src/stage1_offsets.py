"""Stage 1: sensor offsets against MODIS -- fixed offset vs fixed offset + RMA slope.

For each high-resolution sensor, on MODIS-footprint pairs (each MODIS footprint against the
median of the sensor's kept pixels inside it, outliers clipped within scenes):

  fixed   sensor = MODIS + a
  rma     sensor = x0 + a + b (MODIS - x0), b the RMA slope across SCENE MEDIANS in absolute
          temperature (the seasonal range is the leverage; within a scene MODIS varies by about
          its own noise), x0 the median scene MODIS value

Both are scored by leave-one-scene-out (LOSO): refit without a scene, predict its footprints
from MODIS. `rma` is recommended only if it beats `fixed` by `offsets.recommend_min_gain`;
`offsets.mode` (auto | fixed | rma) overrides. Everything stage 2 needs is written to
`offsets.json`.

    python src/run_region.py --config configs/region.<aoi>.yaml offsets [--masks CUBE]
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import composite as C
import iterative_filter as F
import offset_diagnostics as D
import pipeline as P
import plotting
import region as R
from cube_figures import GRID, INK, INK_MUTED, INK_SECONDARY, SURFACE

log = logging.getLogger("stage1_offsets")

MODEL_COLOR = {"fixed": "#2a78d6", "rma": "#1baf7a"}
MODELS = ("fixed", "rma")


# ==================================================================== models

def fit_model(pairs: pd.DataFrame, kind: str, sid: str, pcfg: dict) -> dict:
    """Fit one model on the KEPT pairs. -> {a, b, x0, off, slope, n_scenes}.

    `off` and `slope` are what stage 2 applies: MODIS-scale = (sensor - off) / slope.
    """
    fcfg = P.offset_config(pcfg)
    fcfg = {**fcfg, "offset": {**fcfg["offset"], "model": "constant", "slope": False}}
    if kind == "fixed":
        b, x0 = 1.0, 0.0
    else:
        sm = D.scene_medians(pairs)
        b = float(D.absolute_estimates(sm["modis"], sm["sensor"])["RMA"][0])
        x0 = float(np.median(sm["modis"])) if len(sm) else 0.0
    table, rx, ry = P.scene_table(pairs, b, sid, pcfg, x0)
    fit = C.fit_offset(table, rx, ry, fcfg, sid)
    a = float(fit["coefs"][0])
    return dict(a=a, b=b, x0=x0, off=a + x0 * (1.0 - b), slope=b, n_scenes=len(table))


def predict(m: dict, modis) -> np.ndarray:
    """The sensor value the model expects for a MODIS value."""
    x = np.asarray(modis, float)
    return m["x0"] + m["a"] + m["b"] * (x - m["x0"])


def loso(pairs: pd.DataFrame, kind: str, sid: str, pcfg: dict) -> pd.DataFrame:
    """Leave-one-scene-out errors: refit without each scene, predict its kept footprints."""
    kept = pairs[pairs["kept"]]
    rows = []
    for t, test in kept.groupby("t"):
        try:
            m = fit_model(pairs[pairs["t"] != t], kind, sid, pcfg)
        except ValueError as e:                     # e.g. too few scenes left
            log.debug("LOSO %s %s without scene %s: %s", sid, kind, t, e)
            continue
        err = test["sensor"].to_numpy(float) - predict(m, test["modis"])
        rows.append(dict(sensor=sid, model=kind, t=int(t), date=test["date"].iloc[0],
                         hour=float(test["hour"].iloc[0]), n=len(test),
                         modis_median=float(test["modis"].median()),
                         err_median=float(test["sensor"].median()
                                          - predict(m, [test["modis"].median()])[0]),
                         sse=float(np.sum(err ** 2)), rmse_pairs=float(np.sqrt(np.mean(err ** 2)))))
    return pd.DataFrame(rows)


def slope_interval(sm: pd.DataFrame, n_boot: int, seed: int = 0) -> dict:
    """90% scene-bootstrap intervals of the across-scene RMA and OLS slopes."""
    rng = np.random.default_rng(seed)
    draws = {"RMA": [], "OLS": []}
    for _ in range(int(n_boot)):
        s = sm.iloc[rng.integers(0, len(sm), len(sm))]
        e = D.absolute_estimates(s["modis"], s["sensor"])
        draws["RMA"].append(e["RMA"][0])
        draws["OLS"].append(e["OLS"][0])
    return {k: (float(np.nanpercentile(v, 5)), float(np.nanpercentile(v, 95)))
            for k, v in draws.items()}


def within_scene(pairs: pd.DataFrame) -> dict:
    """Within-scene anomaly diagnostics on the kept pairs."""
    k = pairs[pairs["kept"]]
    xa, ya = D.anomalies(k, "mean")
    r = float(np.corrcoef(xa, ya)[0, 1]) if len(k) > 2 and np.std(xa) > 0 else np.nan
    x, y, s = k["modis"].to_numpy(float), k["sensor"].to_numpy(float), k["t"].to_numpy(int)
    return dict(within_r=r, within_rma=P.anomaly_slope(x, y, s, "rma"),
                within_ols=P.anomaly_slope(x, y, s, "ols"),
                within_modis_sd=float(np.median(k.groupby("t")["modis"].std())))


def analyse_sensor(raw: F.Raw, keep: dict, sid: str, rc: dict) -> dict:
    """Pairs, clipping, both models, LOSO, intervals and the recommendation for one sensor."""
    pcfg, oc = rc["pipe"], rc["offsets"]
    mem = np.where(keep[sid], raw.raw[sid], np.nan).astype("float32")
    pairs = P.matchup_pairs(mem, raw.ref, raw.hours[sid], pd.to_datetime(raw.times), pcfg, sid)
    if pairs.empty:
        raise ValueError(f"{sid}: no matchup scenes with MODIS; check the masks and "
                         "offsets.min_footprints")
    pairs, clip = P.clip_pairs(pairs, pcfg)      # pcfg.offset.slope is off: clipping at slope 1
    sm = D.scene_medians(pairs)
    models, errs = {}, []
    for kind in MODELS:
        m = fit_model(pairs, kind, sid, pcfg)
        lo, hi = (float(v) for v in oc["slope_clip"])
        m["valid"] = bool(np.isfinite(m["b"]) and lo <= m["b"] <= hi)
        e = loso(pairs, kind, sid, pcfg)
        n = int(e["n"].sum()) if len(e) else 0
        m.update(loso_rmse=float(np.sqrt(e["sse"].sum() / n)) if n else np.nan,
                 loso_rmse_scene=float(np.sqrt(np.mean(e["err_median"] ** 2))) if len(e)
                 else np.nan,
                 loso_bias=float(e["err_median"].mean()) if len(e) else np.nan,
                 loso_scenes=int(len(e)))
        models[kind] = m
        errs.append(e)
    ci = slope_interval(sm, oc["n_boot"])
    ols = D.absolute_estimates(sm["modis"], sm["sensor"])["OLS"][0]
    gain = float(oc["recommend_min_gain"])
    f_, r_ = models["fixed"]["loso_rmse"], models["rma"]["loso_rmse"]
    better = models["rma"]["valid"] and np.isfinite(r_) and r_ <= (1.0 - gain) * f_
    recommended = "rma" if better else "fixed"
    chosen = recommended if oc["mode"] == "auto" else oc["mode"]
    if chosen == "rma" and not models["rma"]["valid"]:
        raise ValueError(f"{sid}: offsets.mode is rma but its slope {models['rma']['b']:.3f} is "
                         f"outside offsets.slope_clip {oc['slope_clip']}")
    diag = dict(rma_ci=ci["RMA"], ols_across=float(ols), ols_ci=ci["OLS"],
                modis_range=float(np.ptp(sm["modis"])) if len(sm) else np.nan,
                n_scene_medians=int(len(sm)), clip=dict(
                    n_pairs=clip["n_pairs"], n_removed=clip["n_removed"],
                    rounds=clip["rounds"], converged=bool(clip["converged"]),
                    scale=float(clip["scale"])), **within_scene(pairs))
    log.info("%s: fixed a=%+.3f K (LOSO %.3f K) | rma b=%.3f [%.2f, %.2f] off=%+.3f K "
             "(LOSO %.3f K) -> recommend %s, use %s", sid, models["fixed"]["a"], f_,
             models["rma"]["b"], *ci["RMA"], models["rma"]["off"], r_, recommended, chosen)
    return dict(pairs=pairs, clip=clip, scene_medians=sm, models=models,
                loso=pd.concat(errs, ignore_index=True), diag=diag,
                recommended=recommended, chosen=chosen, mem=mem)


# ==================================================================== figures

def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=7)


def models_figure(res: dict, label: str, out: Path, dpi: int) -> None:
    sm, pairs, m, d = res["scene_medians"], res["pairs"], res["models"], res["diag"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), dpi=dpi, layout="constrained")

    ax = axes[0]
    xc, yc = sm["modis"] - 273.15, sm["sensor"] - 273.15
    sc = ax.scatter(xc, yc, c=sm["hour"], cmap="viridis", vmin=0, vmax=24,
                    s=np.clip(sm["n"] * 1.5, 15, 200), edgecolor=SURFACE, linewidth=0.7,
                    zorder=3)
    lo, hi = float(min(xc.min(), yc.min())) - 0.5, float(max(xc.max(), yc.max())) + 0.5
    xs = np.linspace(lo, hi, 50)
    ax.plot(xs, xs, color=INK_MUTED, linestyle=":", linewidth=1, label="1:1")
    for kind in MODELS:
        mk = m[kind]
        ax.plot(xs, predict(mk, xs + 273.15) - 273.15, color=MODEL_COLOR[kind], linewidth=2,
                label=(f"fixed: offset {mk['a']:+.2f} K" if kind == "fixed" else
                       f"rma: slope {mk['b']:.3f} [{d['rma_ci'][0]:.2f}, {d['rma_ci'][1]:.2f}],"
                       f" offset {mk['a']:+.2f} K at {mk['x0'] - 273.15:.1f} degC"))
    r = m["rma"]
    band = [r["x0"] + r["a"] + b * (xs + 273.15 - r["x0"]) - 273.15 for b in d["rma_ci"]]
    ax.fill_between(xs, band[0], band[1], color=MODEL_COLOR["rma"], alpha=0.15, linewidth=0,
                    label="rma slope, 90% scene bootstrap")
    fig.colorbar(sc, ax=ax, shrink=0.8, label="overpass hour [UTC]").ax.tick_params(labelsize=6)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    _style(ax)
    ax.set_xlabel("MODIS, scene median of footprints [degC]", fontsize=8)
    ax.set_ylabel(f"{label}, scene median of footprint medians [degC]", fontsize=8)
    ax.set_title(f"between scenes: {len(sm)} scenes spanning {d['modis_range']:.1f} K "
                 "(area ~ footprints)", fontsize=9, color=INK)
    ax.legend(fontsize=7, frameon=False, loc="upper left")

    ax = axes[1]
    k = pairs[pairs["kept"]]
    xa, ya = D.anomalies(k, "mean")
    ax.scatter(xa, ya, s=7, color="#8fb8e8", alpha=0.6, edgecolor="none",
               label=f"kept footprint pairs ({len(k):,})")
    lim = float(np.nanpercentile(np.abs(np.r_[xa, ya]), 99.5)) if len(k) else 1.0
    xs2 = np.array([-lim, lim])
    ax.plot(xs2, xs2, color=MODEL_COLOR["fixed"], linewidth=2, label="fixed (slope 1)")
    ax.plot(xs2, r["b"] * xs2, color=MODEL_COLOR["rma"], linewidth=2,
            label=f"between-scene rma slope {r['b']:.3f}")
    if np.isfinite(d["within_rma"]):
        ax.plot(xs2, d["within_rma"] * xs2, color=INK, linestyle="--", linewidth=1.4,
                label=f"within-scene rma slope {d['within_rma']:.3f} (diagnostic)")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    _style(ax)
    ax.set_xlabel("MODIS footprint, within-scene anomaly [K]", fontsize=8)
    ax.set_ylabel(f"{label} footprint median, within-scene anomaly [K]", fontsize=8)
    ax.set_title(f"within scenes: r = {d['within_r']:.2f}; median MODIS spread "
                 f"{d['within_modis_sd']:.2f} K per scene", fontsize=9, color=INK)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle(f"{label}: offset models. Recommended: {res['recommended']}; used: "
                 f"{res['chosen']}", fontsize=10, color=INK, ha="left", x=0.01)
    plotting.save(fig, out)


def loso_figure(res: dict, label: str, out: Path, dpi: int) -> None:
    e, m = res["loso"], res["models"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=dpi, layout="constrained",
                             gridspec_kw=dict(width_ratios=[2.2, 1.6, 1]))
    for kind, mk in (("fixed", "o"), ("rma", "D")):
        g = e[e["model"] == kind]
        axes[0].scatter(pd.to_datetime(g["date"]), g["err_median"], s=22, marker=mk,
                        color=MODEL_COLOR[kind], edgecolor=SURFACE, linewidth=0.6,
                        label=kind, alpha=0.85)
        axes[1].scatter(g["modis_median"] - 273.15, g["err_median"], s=22, marker=mk,
                        color=MODEL_COLOR[kind], edgecolor=SURFACE, linewidth=0.6,
                        label=kind, alpha=0.85)
    for ax in axes[:2]:
        ax.axhline(0, color=INK_MUTED, linewidth=1)
        _style(ax)
        ax.set_ylabel("held-out scene error [K]\n(observed - predicted scene median)",
                      fontsize=8)
        ax.legend(fontsize=7, frameon=False)
    axes[0].set_title("leave-one-scene-out error by date", fontsize=9, color=INK)
    axes[1].set_xlabel("scene MODIS median [degC]", fontsize=8)
    axes[1].set_title("by scene temperature (a slope shows as a trend)", fontsize=9,
                      color=INK)
    ax = axes[2]
    x = np.arange(2)
    for i, (col, name) in enumerate((("loso_rmse", "footprints"),
                                     ("loso_rmse_scene", "scene medians"))):
        vals = [m[k][col] for k in MODELS]
        ax.bar(x + (i - 0.5) * 0.38, vals, 0.36,
               color=[MODEL_COLOR[k] for k in MODELS], alpha=1.0 if i == 0 else 0.55,
               label=name)
        for xi, v in zip(x + (i - 0.5) * 0.38, vals):
            ax.annotate(f"{v:.3f}", (xi, v), xytext=(0, 2), textcoords="offset points",
                        ha="center", fontsize=7, color=INK_SECONDARY)
    ax.set_xticks(x, list(MODELS))
    _style(ax)
    ax.set_ylabel("LOSO RMSE [K]", fontsize=8)
    ax.set_title("LOSO RMSE (solid: footprints,\nlight: scene medians)", fontsize=9, color=INK)
    fig.suptitle(f"{label}: leave-one-scene-out comparison", fontsize=10, color=INK,
                 ha="left", x=0.01)
    plotting.save(fig, out)


def summary_figure(results: dict, labels: dict, gain: float, out: Path, dpi: int) -> None:
    sids = list(results)
    fig, axes = plt.subplots(1, len(sids), figsize=(4.6 * len(sids), 4.2), dpi=dpi,
                             layout="constrained", squeeze=False)
    for ax, sid in zip(axes[0], sids):
        r = results[sid]
        vals = [r["models"][k]["loso_rmse"] for k in MODELS]
        ax.bar(list(MODELS), vals, color=[MODEL_COLOR[k] for k in MODELS], width=0.6)
        for i, v in enumerate(vals):
            ax.annotate(f"{v:.3f} K", (i, v), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8, color=INK)
        _style(ax)
        lo, hi = r["diag"]["rma_ci"]
        ax.set_title(f"{labels[sid]}\nrma slope {r['models']['rma']['b']:.3f} "
                     f"[{lo:.2f}, {hi:.2f}]; recommended {r['recommended']}, used "
                     f"{r['chosen']}", fontsize=8, color=INK)
        ax.set_ylabel("LOSO RMSE [K]", fontsize=8)
    fig.suptitle(f"offset model choice (rma must beat fixed by {gain:.0%})", fontsize=10,
                 color=INK, ha="left", x=0.01)
    plotting.save(fig, out)


# ==================================================================== driver

def load_masks(raw: F.Raw, cube: Path | None, it: dict) -> tuple[dict, str]:
    if cube is None:
        return F.qc_only_keep(raw, it["filter"]["require_qc"]), "sensor QC only"
    with xr.open_zarr(cube) as c:
        idx = pd.Index(c["time"].values).get_indexer(raw.times)
        if (idx < 0).any():
            raise ValueError(f"{cube} does not cover the configured time range")
        keep = {sid: c[f"{sid}_keep"].values[idx].astype(bool) for sid in raw.raw}
    return keep, f"cloud filter of {Path(cube).name}"


def run(rc: dict, *, tag: str | None = None, masks: Path | None = None,
        figures: bool = True) -> dict:
    t0 = time.time()
    for noisy in ("composite", "pipeline", "iterative_filter"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    pcfg, it = rc["pipe"], rc["pipe"]["_iter"]
    out = R.stage_dirs(rc, tag)["offsets"]
    fig_dir = out / "figures"
    raw = F.load_raw(it, P.select_times(pcfg))
    keep, mask_source = load_masks(raw, masks, it)
    log.info("stage 1: %d dates, %d water px; pixel masks: %s", len(raw.times), raw.n_water,
             mask_source)

    results = {sid: analyse_sensor(raw, keep, sid, rc) for sid in raw.raw}
    labels = {sid: it["sensors"][sid]["label"] for sid in raw.raw}

    out.mkdir(parents=True, exist_ok=True)
    rows, sens = [], {}
    for sid, r in results.items():
        for kind, m in r["models"].items():
            rows.append(dict(sensor=sid, model=kind, **{k: v for k, v in m.items()},
                             recommended=r["recommended"] == kind, chosen=r["chosen"] == kind,
                             **{k: v for k, v in r["diag"].items() if k != "clip"},
                             **{f"clip_{k}": v for k, v in r["diag"]["clip"].items()}))
        sens[sid] = dict(label=labels[sid], recommended=r["recommended"], chosen=r["chosen"],
                         models=r["models"], diagnostics=r["diag"])
    pd.DataFrame(rows).to_csv(out / "models.csv", index=False)
    pd.concat([r["loso"] for r in results.values()], ignore_index=True) \
        .to_csv(out / "scenes.csv", index=False)
    pd.concat([r["pairs"].assign(sensor_id=sid) for sid, r in results.items()],
              ignore_index=True).to_csv(out / "pairs.csv", index=False)
    payload = dict(aoi=rc["region"]["aoi"], config=rc["path"], config_sha=rc["sha"],
                   masks=mask_source, time_range=[str(raw.times[0])[:10],
                                                  str(raw.times[-1])[:10]],
                   mode=rc["offsets"]["mode"],
                   recommend_rule=f"rma if its LOSO RMSE <= (1 - "
                                  f"{rc['offsets']['recommend_min_gain']}) x fixed's",
                   sensors=sens)
    with open(out / "offsets.json", "w") as f:
        json.dump(payload, f, indent=2, default=float)
    log.info("wrote %s", out / "offsets.json")

    if figures:
        dpi = int(pcfg["output"]["figure_dpi"])
        k = float(pcfg["matchup"]["outlier_k"] or 1.5)
        extent = ([0.0, float(np.ptp(raw.coords["x"])) / 1000.0, 0.0,
                   float(np.ptp(raw.coords["y"])) / 1000.0] if "x" in raw.coords else None)
        for sid, r in results.items():
            models_figure(r, labels[sid], fig_dir / f"models_{sid}.png", dpi)
            loso_figure(r, labels[sid], fig_dir / f"loso_{sid}.png", dpi)
            D.outliers_figure(r["pairs"], r["clip"], k, labels[sid],
                              fig_dir / f"outliers_{sid}.png", dpi)
            D.outlier_maps_figure(r["pairs"], raw, r["mem"], labels[sid], extent,
                                  fig_dir / f"outlier_maps_{sid}.png", dpi)
        summary_figure(results, labels, float(rc["offsets"]["recommend_min_gain"]),
                       fig_dir / "summary.png", dpi)
    log.info("stage 1 done in %.1f min -> %s", (time.time() - t0) / 60, out)
    return dict(dir=out, offsets=payload, results=results)


def read_offsets(rc: dict, tag: str | None = None) -> dict:
    """Stage 1's hand-off, with an actionable error when it has not been run."""
    path = R.stage_dirs(rc, tag)["offsets"] / "offsets.json"
    if not path.exists():
        raise SystemExit(f"no stage-1 offsets at {path}; run:\n  python src/run_region.py "
                         f"--config {rc['path']}{' --tag ' + tag if tag else ''} offsets")
    with open(path) as f:
        return json.load(f)
