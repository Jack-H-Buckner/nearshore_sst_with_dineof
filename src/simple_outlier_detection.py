"""Single-pass cloud / outlier detection for SST imagery.

The first pass of `outlier_detection.py`, without the outer loop. The MODIS covariate is the
spatial mean throughout: there is no SPDE refit of the target field, so no damping across
iterations and no convergence test.

Per acquisition:

  1. Covariate: robust MODIS composite around the date, smoothed once by the SPDE.
  2. Global sensor offset `center` = robust centre of (y - covariate); residual = y - covariate
     - center. `center` also drives the whole-image offset flag.
  3. Sensor QC -> per-pixel cloud prior. Pixels the sensor flags as cloudy or invalid, and
     pixels with no data at all, get `qc.prior_cloud` instead of `mixture.prior_cloud`. A gap
     has no likelihood term, so its prior IS its unary; the Ising sweeps then carry that
     evidence to its neighbours.
  4. Two-sided mixture (cold cloud, warm anomaly, clear) with a mean-field Ising prior on the
     labels -> p_valid = 1 - q_cloud - q_hot.

Outputs one NetCDF and one PNG panel per date, plus a run-level offset_flags.csv.

Usage (from the repo root, in the `coastal_sst_data` env):

    python prototypes/cloud_mixture_model/src/simple_outlier_detection.py \\
        --config prototypes/cloud_mixture_model/configs/simple_outlier_detection.yaml
    ... --dates 2025-03-05 2025-03-06     # override data.dates
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from outlier_detection import (
    LAND_COLOR,
    ROOT,
    _plain,
    acquisition_dates,
    estep,
    load_config,
    load_inputs,
    robust_centre_scale,
    robust_modis_composite,
    solve_spde,
    unary_logodds,
)

log = logging.getLogger("simple_outlier_detection")

DEFAULT_CONFIG = ROOT / "configs" / "simple_outlier_detection.yaml"


# ==================================================================== config

DEFAULTS = {
    # Which detector to run: the two-sided mixture + Ising prior (`mixture`, the default), or the
    # simpler per-scene Tukey IQR-fence rule on the residual (`tukey`). See `classify_tukey`.
    "method": "mixture",
    "data": {
        "cube": "data/datacube/admiralty_inlet.zarr",
        "var": "lst_sst",
        "validvar": "lst_valid",
        "cloudvar": "lst_cloud",
        "ref_var": "modis_sst_aqua",
        "landvar": "landcover_water",
        "depthvar": None,          # null = no intertidal prior; set to a depth channel to enable
        "tidal_depth_m": 3.0,
        "dates": [],
        "min_pixels": 64,
    },
    "output": {
        "dir": "outputs/simple_outlier_detection",
        "fig_dir": "figures/simple_outlier_detection",
        "write_netcdf": True,
        "write_figures": True,
    },
    "reference": {
        "window_days": 4,
        "qc_var": None,
        "qc_max": 1,
        "quantile": 0.5,
        "min_obs": 1,
        "min_cover": 0.05,
    },
    "covariate": {
        "init_prior": 285.0,
        "init_range_px": 200.0,
        "init_marg_sd": 0.05,
        "init_obs_error": 0.5,
        "alpha": 2,
    },
    "clear": {
        "sd": 0.75,
        "sd_floor": 0.71,
    },
    "qc": {
        "enabled": True,
        "prior_cloud": 0.9,
        "nodata_prior": None,
    },
    "offset": {
        "lower": -2.0,
        "upper": 2.0,
    },
    "mixture": {
        "lambda_cloud": 4.0,
        "prior_cloud": 0.5,
        "lambda_hot": 6.5,
        "prior_hot": 0.005,
        "p_h_tidal": 0.5,
        "beta_cloud": 1.75,
        "n_sweeps_cloud": 20,
        "beta_hot": 0.125,
        "n_sweeps_hot": 10,
        "estep_damp": 0.5,
        "p_floor": 0.01,
    },
    "solver": {
        "tol": 1e-2,
        "pcg_max_iter": 25,
        "precondition": True,
    },
    "tukey": {
        # Fences on the per-scene residual distribution: lo = Q1 - k_low * IQR, hi = Q3 + k_high
        # * IQR. Classic Tukey uses 1.5 (outlier) / 3.0 (far out); 3.0 is the default here so the
        # filter removes the clearly-cloudy tail without nibbling the clear-sky spread.
        "k_low": 3.0,
        "k_high": 3.0,
        # Scenes with fewer observed water pixels than this keep everything -- the quartiles of a
        # handful of pixels are too noisy to fence on. The whole-scene offset gate still applies.
        "min_pixels": 64,
    },
}


# ================================================================== covariate

def build_covariate(ds, date, land, cfg):
    """Robust reference composite, smoothed once by the SPDE. The spatial mean for the run."""
    d, ref, cov, sol = cfg["data"], cfg["reference"], cfg["covariate"], cfg["solver"]
    modis_raw, n_ref = robust_modis_composite(
        ds, date, d["ref_var"], d["landvar"], window_days=ref["window_days"],
        qc_var=ref["qc_var"], qc_max=ref["qc_max"], quantile=ref["quantile"],
        min_obs=ref["min_obs"])

    covariate, k = solve_spde(modis_raw, land, cov["init_prior"],
                              cov["init_range_px"], cov["init_marg_sd"],
                              cov["init_obs_error"], alpha=cov["alpha"],
                              precondition=sol["precondition"], tol=sol["tol"],
                              max_iter=sol["pcg_max_iter"])
    cover = float(np.isfinite(modis_raw[land]).mean())
    log.info("  covariate: pcg %d iters, %.1f%% of water pixels had reference data",
             k, 100 * cover)
    return covariate, modis_raw, n_ref, cover


# ================================================================== sensor QC

def _channel(ds, name, date, shape):
    """A QC channel at `date` as float, or None when it is not configured or not present."""
    if not name or name not in ds:
        return None
    da = ds[name]
    if "time" in da.dims:
        da = da.sel(time=date, method="nearest")
    a = np.asarray(da.compute(), float)
    if a.shape != shape:
        raise ValueError(f"QC channel {name} has shape {a.shape}, expected {shape}")
    return a


def qc_masks(ds, date, land, y, cfg):
    """Water pixels the sensor's own QC distrusts, split by whether they still carry data.

    Returns (flagged, gap, nodata):
      flagged  has data, but cloudvar == 1 or validvar == 0
      gap      no data at all (QC-removed cloud, or never retrieved)
      nodata   the subset of `gap` the cloud channel positively calls NOT cloud

    A NaN QC value is no verdict, not a flag. With qc.enabled false, `flagged` is empty and
    every gap is treated alike.
    """
    d = cfg["data"]
    obs = land & np.isfinite(y)
    gap = land & ~np.isfinite(y)
    flagged = np.zeros(land.shape, bool)
    nodata = np.zeros(land.shape, bool)
    if not cfg["qc"]["enabled"]:
        return flagged, gap, nodata

    cloud = _channel(ds, d["cloudvar"], date, land.shape)
    valid = _channel(ds, d["validvar"], date, land.shape)
    if cloud is not None:
        flagged |= cloud == 1
        # On a gap, cloud == 1 means QC removed cloud; NaN means nothing was retrieved.
        nodata = gap & ~(cloud == 1)
    if valid is not None:
        flagged |= valid == 0
    # `valid` is 0 wherever the sensor has no data, so restrict to pixels that kept a value.
    flagged &= obs
    return flagged, gap, nodata


def cloud_prior(flagged, gap, nodata, cfg):
    """Per-pixel cloud prior: qc.prior_cloud on flagged and missing pixels, baseline elsewhere.

    `unary_logodds` takes the prior as `pi_t` and forms log(pi/(1-pi)), which broadcasts, so an
    array needs no change there.
    """
    qc, mix = cfg["qc"], cfg["mixture"]
    pi = np.full(flagged.shape, float(mix["prior_cloud"]))
    pi[flagged | gap] = float(qc["prior_cloud"])
    if qc["nodata_prior"] is not None:
        # Gaps the cloud channel calls no-data (a swath edge, say) should not seed cloud.
        pi[nodata] = float(qc["nodata_prior"])
    # log(pi / (1 - pi)) is the prior's whole contribution, so a prior of exactly 0 or 1 would
    # be an infinite log-odds no evidence could move.
    return np.clip(pi, 1e-6, 1.0 - 1e-6)


# ================================================================ classification

def classify(y, land, tidal, covariate, flagged, gap, nodata, cfg):
    """Residual against the covariate -> cloud / warm responsibilities -> p_valid."""
    mix, clr = cfg["mixture"], cfg["clear"]
    obs = land & np.isfinite(y)
    if not obs.any():
        raise ValueError("no observed water pixels")

    # The shorth midpoint of (y - covariate) is both the mixture centre and the sensor offset;
    # in the parent's iteration 0 these are the same number, so estimate it once.
    r = (np.asarray(y, float) - np.asarray(covariate, float))[obs]
    center, sd_est = robust_centre_scale(r)
    sd = float(clr["sd"]) if clr["sd"] is not None else max(sd_est, float(clr["sd_floor"]))
    residual = y - covariate - center

    pi = cloud_prior(flagged, gap, nodata, cfg)
    ull, ull_hot = unary_logodds(residual, tidal, 0.0, sd,
                                 mix["lambda_cloud"], mix["min_dev_cold"], pi,
                                 mix["lambda_hot"], mix["min_dev_hot"], mix["prior_hot"], mix["p_h_tidal"],
                                 noise=True)
    # A gap has no likelihood term and `estep` would nan_to_num its unary to 0, losing the
    # prior with it. Write the prior-only log-odds in explicitly.
    ull = np.where(gap, np.log(pi / (1.0 - pi)), ull)

    q = estep(ull, valid=land, beta=mix["beta_cloud"],
              n_sweeps=mix["n_sweeps_cloud"], damp=mix["estep_damp"])
    # Gaps say nothing about warm anomalies; left in `valid` they sit at 0.5 and pull their
    # neighbours warm-ward.
    q_hot = estep(ull_hot, valid=obs, beta=mix["beta_hot"],
                  n_sweeps=mix["n_sweeps_hot"], damp=mix["estep_damp"])

    # q and q_hot come from two independent binary fields and can sum past 1.
    s = np.maximum(q + q_hot, 1.0)
    q, q_hot = q / s, q_hot / s
    p_valid = np.clip(1.0 - q - q_hot, mix["p_floor"], 1.0)

    return dict(residual=residual, p_valid=p_valid, q_cloud=q, q_hot=q_hot,
                center=center, sd=sd,
                cloud_frac=float(q[obs].mean()), warm_frac=float(q_hot[obs].mean()))


def classify_tukey(y, land, tidal, covariate, flagged, gap, nodata, cfg):
    """Per-scene Tukey IQR fence on the residual (y - covariate - center). A hard keep mask.

    The simpler alternative to `classify`: no mixture, no Ising prior, no EM. The DINEOF low-rank
    field is still the central tendency; a pixel is flagged iff its residual falls outside the
    scene's Tukey fences [Q1 - k_low*IQR, Q3 + k_high*IQR]. Returns the SAME dict shape as
    `classify` so it drops into the loop, figures and scene table unchanged. `tidal` is accepted
    for signature parity with `classify` but unused.
    """
    tk = cfg["tukey"]
    obs = land & np.isfinite(y)
    if not obs.any():
        raise ValueError("no observed water pixels")

    # Same shorth centre as the mixture path, so `center` keeps its "sensor offset" meaning and
    # bc.scene_verdict's offset gate fires identically (a near-total-cloud scene still pulls the
    # centre cold and gets dropped).
    r0 = (np.asarray(y, float) - np.asarray(covariate, float))[obs]
    center, sd_est = robust_centre_scale(r0)
    residual = y - covariate - center

    cold = np.zeros(land.shape, bool)
    warm = np.zeros(land.shape, bool)
    iqr = 0.0
    if int(obs.sum()) >= int(tk["min_pixels"]):
        q1, q3 = np.nanpercentile(residual[obs], [25.0, 75.0])
        iqr = float(q3 - q1)
        lo = q1 - float(tk["k_low"]) * iqr
        hi = q3 + float(tk["k_high"]) * iqr
        with np.errstate(invalid="ignore"):
            cold = obs & (residual < lo)
            warm = obs & (residual > hi)

    q_cloud = cold.astype("float64")
    q_hot = warm.astype("float64")
    # Hard keep mask: 1.0 inside the fences, 0.0 outside. Drops straight into the loop's
    # `p_valid >= p_valid_min` cut (p_valid_min default 0.5).
    p_valid = np.where(obs & ~cold & ~warm, 1.0, 0.0)
    # Robust sigma from the IQR, for the scene table only (IQR = 1.349 sigma for a Gaussian).
    sd = iqr / 1.349 if iqr > 0 else float(sd_est)

    return dict(residual=residual, p_valid=p_valid, q_cloud=q_cloud, q_hot=q_hot,
                center=center, sd=sd,
                cloud_frac=float(q_cloud[obs].mean()), warm_frac=float(q_hot[obs].mean()))


def flag_offset(center, cfg):
    """-1 below `offset.lower`, +1 above `offset.upper`, 0 in band. A null bound disables it."""
    lo, hi = cfg["offset"]["lower"], cfg["offset"]["upper"]
    if lo is not None and center < float(lo):
        return -1
    if hi is not None and center > float(hi):
        return 1
    return 0


def flag_reference(ref_cover, cfg):
    """1 when too little of the scene had reference data for the comparison to mean anything.

    Where the reference is absent the SPDE returns covariate.init_prior, so `center` measures
    the target against a constant rather than against MODIS and the offset flag would fire on
    the prior, not on the sensor. Kept separate from offset_flag so the two cannot be confused:
    a large offset with good cover is a real verdict on the image (near-total cloud pulls the
    shorth onto the cloud mode and `center` goes to -20 K), a large offset without cover is not.
    """
    return int(ref_cover < float(cfg["reference"]["min_cover"]))


def run_date(ds, date, cfg):
    """Full pipeline for one acquisition."""
    y, land, tidal, t = load_inputs(ds, date, cfg)
    covariate, modis_raw, n_ref, ref_cover = build_covariate(ds, t, land, cfg)
    flagged, gap, nodata = qc_masks(ds, t, land, y, cfg)
    log.info("  qc: %d flagged-with-data, %d gap (%d no-data) of %d water pixels",
             flagged.sum(), gap.sum(), nodata.sum(), land.sum())

    result = classify(y, land, tidal, covariate, flagged, gap, nodata, cfg)
    result["offset_flag"] = flag_offset(result["center"], cfg)
    result["ref_flag"] = flag_reference(ref_cover, cfg)
    # `obs`, not `y`: `y` is the cube's row coordinate and would clash in the NetCDF.
    result.update(obs=y, land=land, tidal=tidal, time=t, covariate=covariate,
                  modis_raw=modis_raw, n_ref=n_ref, ref_cover=ref_cover,
                  qc_flag=flagged, gap=gap, nodata=nodata)

    marks = ([" <-- OFFSET FLAGGED"] if result["offset_flag"] else []) + \
            ([" <-- NO REFERENCE"] if result["ref_flag"] else [])
    log.info("  centre %+6.3f  sd %5.3f  cloud %.3f  warm %.3f%s",
             result["center"], result["sd"], result["cloud_frac"], result["warm_frac"],
             "".join(marks))
    if result["ref_flag"]:
        log.warning("  only %.1f%% reference cover (< %.1f%%): the offset measures the target "
                    "against covariate.init_prior, not against the reference",
                    100 * ref_cover, 100 * cfg["reference"]["min_cover"])
    elif result["offset_flag"]:
        side = "below" if result["offset_flag"] < 0 else "above"
        log.warning("  offset %+.3f K is %s the [%s, %s] band", result["center"], side,
                    cfg["offset"]["lower"], cfg["offset"]["upper"])
    return result


# =================================================================== outputs

FIELDS = ("obs", "covariate", "modis_raw", "residual", "p_valid", "q_cloud", "q_hot", "n_ref")
MASKS = ("land", "tidal", "qc_flag", "gap", "nodata")


def write_netcdf(result, ds, cfg, path):
    """2-D fields on the cube's grid, run diagnostics and the config as attrs."""
    var = cfg["data"]["var"]
    dims = ds[var].dims[-2:]
    coords = {d: ds[d].values for d in dims if d in ds.coords}
    data = {name: (dims, np.asarray(result[name])) for name in FIELDS}
    for name in MASKS:
        data[name] = (dims, np.asarray(result[name]).astype("int8"))

    out = xr.Dataset(data, coords=coords)
    out.attrs.update(
        date=str(result["time"]), var=var,
        center=result["center"], sd=result["sd"],
        cloud_frac=result["cloud_frac"], warm_frac=result["warm_frac"],
        qc_flag_px=int(result["qc_flag"].sum()), gap_px=int(result["gap"].sum()),
        ref_cover=result["ref_cover"], ref_flag=result["ref_flag"],
        offset_flag=result["offset_flag"],
        offset_lower="null" if cfg["offset"]["lower"] is None else cfg["offset"]["lower"],
        offset_upper="null" if cfg["offset"]["upper"] is None else cfg["offset"]["upper"],
        config=yaml.safe_dump(_plain(cfg), sort_keys=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(path)
    log.info("  wrote %s", path)


def plot_panels(result, cfg, path):
    """obs | covariate | residual | q_cloud | q_hot | p_valid, land grey."""
    land = result["land"]
    sst = np.concatenate([result[k][land & np.isfinite(result[k])]
                          for k in ("obs", "covariate")])
    vmin, vmax = np.percentile(sst, [2, 98]) if sst.size else (None, None)
    res = result["residual"][land & np.isfinite(result["residual"])]
    rlim = float(np.percentile(np.abs(res), 98)) if res.size else 1.0

    panels = [("obs", "observed", "inferno", vmin, vmax),
              ("covariate", "covariate", "inferno", vmin, vmax),
              ("residual", "residual", "RdBu_r", -rlim, rlim),
              ("q_cloud", "q_cloud", "magma", 0.0, 1.0),
              ("q_hot", "q_hot", "magma", 0.0, 1.0),
              ("p_valid", "p_valid", "viridis", 0.0, 1.0)]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4),
                             constrained_layout=True)
    for ax, (key, title, cmap_name, lo, hi) in zip(axes, panels):
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad(LAND_COLOR)
        im = ax.imshow(np.where(land, result[key], np.nan), cmap=cmap, vmin=lo, vmax=hi,
                       interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8)

    flag = {-1: "  -- OFFSET LOW", 0: "", 1: "  -- OFFSET HIGH"}[result["offset_flag"]]
    if result["ref_flag"]:
        flag += "  -- NO REFERENCE"
    fig.suptitle(f"{cfg['data']['var']} {result['time'].date()} -- "
                 f"centre {result['center']:+.2f} K, sd {result['sd']:.2f}, "
                 f"cloud {result['cloud_frac']:.1%}, ref {result['ref_cover']:.0%}, "
                 f"qc {int(result['qc_flag'].sum())} px, gap {int(result['gap'].sum())} px"
                 f"{flag}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    log.info("  wrote %s", path)


SUMMARY_COLUMNS = ["date", "var", "center", "sd", "cloud_frac", "warm_frac",
                   "n_obs_px", "n_qc_px", "n_gap_px", "ref_cover", "ref_flag",
                   "offset_flag"]


def summary_row(result, cfg):
    land, y = result["land"], result["obs"]
    return {
        "date": str(result["time"].date()),
        "var": cfg["data"]["var"],
        "center": result["center"],
        "sd": result["sd"],
        "cloud_frac": result["cloud_frac"],
        "warm_frac": result["warm_frac"],
        "n_obs_px": int((land & np.isfinite(y)).sum()),
        "n_qc_px": int(result["qc_flag"].sum()),
        "n_gap_px": int(result["gap"].sum()),
        "ref_cover": result["ref_cover"],
        "ref_flag": result["ref_flag"],
        "offset_flag": result["offset_flag"],
    }


def append_summary(rows, path):
    """Merge `rows` into the CSV, replacing any existing row for the same (var, date).

    A --dates subset run should update those dates without clobbering the rest.
    """
    new = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    if path.exists():
        old = pd.read_csv(path)
        keys = set(zip(new["var"], new["date"]))
        keep = [k not in keys for k in zip(old.get("var", []), old.get("date", []))]
        new = pd.concat([old[keep], new], ignore_index=True)
    new = new.sort_values(["var", "date"], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)
    log.info("wrote %s (%d rows)", path, len(new))


# =============================================================== entry point

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML config")
    p.add_argument("--dates", nargs="+", default=None, help="override data.dates")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_config(args.config, DEFAULTS)
    if args.dates is not None:
        cfg["data"]["dates"] = args.dates

    d, o = cfg["data"], cfg["output"]
    if not d["cube"].exists():
        raise SystemExit(f"no cube at {d['cube']}")
    ds = xr.open_zarr(d["cube"])

    dates = d["dates"]
    if not dates:
        land = np.asarray(ds[d["landvar"]].compute() > 0.5)
        dates = acquisition_dates(ds, d["var"], land, d["min_pixels"])
        log.info("%d acquisitions of %s found", len(dates), d["var"])

    rows, flagged_dates, no_ref, failed = [], [], [], []
    for date in dates:
        log.info("=== %s %s", d["var"], date)
        try:
            result = run_date(ds, date, cfg)
            stem = f"{d['var']}_{result['time'].date()}"
            if o["write_netcdf"]:
                write_netcdf(result, ds, cfg, o["dir"] / f"{stem}.nc")
            if o["write_figures"]:
                plot_panels(result, cfg, o["fig_dir"] / f"{stem}.png")
            rows.append(summary_row(result, cfg))
            if result["ref_flag"]:
                no_ref.append(f"{result['time'].date()}({result['ref_cover']:.0%})")
            elif result["offset_flag"] != 0:
                flagged_dates.append(f"{result['time'].date()}({result['center']:+.2f})")
        except Exception:
            log.exception("failed on %s", date)
            failed.append(date)

    if rows:
        append_summary(rows, o["dir"] / "offset_flags.csv")
    if no_ref:
        log.warning("%d/%d dates had too little reference cover to screen: %s",
                    len(no_ref), len(rows), ", ".join(no_ref))
    if flagged_dates:
        log.warning("%d/%d dates offset-flagged: %s",
                    len(flagged_dates), len(rows), ", ".join(flagged_dates))
    if failed:
        log.warning("%d/%d dates failed: %s", len(failed), len(dates), ", ".join(failed))


if __name__ == "__main__":
    main()
