"""Standardize the composite: remove a per-pixel seasonal cycle, divide by a robust scale.

Stage 5 of the DINEOF input pipeline. `composite.py` produced one SST field per date; this
turns it into the dimensionless anomaly eDINEOF actually decomposes, and stores everything
needed to invert the transform exactly.

WHY STANDARDIZE AT ALL. Two reasons, and they are separate.

  The SEASONAL MEAN, because without it the leading EOF is just the annual cycle. On this AoI
  the per-pixel harmonic amplitude has a median of 2.50 K against a residual scatter of 1.02 K,
  so mode 1 would spend itself describing the mean state instead of the variability that the
  reconstruction is supposed to interpolate. The eDINEOF spec calls for exactly this (§2:
  "consider removing a per-pixel climatology yourself beforehand... Fit a smooth harmonic
  rather than a raw average of available days").

  The ROBUST SCALE, because the upstream filter does not catch everything -- 2.48% of the
  standardized values exceed |z| = 3 and the largest is 10.4 -- and because the per-pixel
  residual scale genuinely varies, from 0.60 K in the deep tidally-mixed strait to 1.68 K in
  the shallow inner bays. A non-robust scale would be set by the outliers it is meant to
  contain; a single global scale would let the noisy shallow pixels dominate every mode.
  Dividing per pixel makes the downstream analysis a CORRELATION-matrix EOF rather than a
  covariance one. That is a deliberate choice, not an accident of implementation.

THE FIT IS GAP-TOLERANT, which is why it is written here rather than imported. Its sibling
`seasonal_smoothing.fit_harmonics` hard-raises on NaN ("smoothed fields must be finite on
water"): it consumes SPDE-gap-filled MODIS, which is complete by construction. This input is
21.9% populated. The three-tier fallback ladder is taken from it unchanged, though --
FIT_FULL / FIT_MEAN_ONLY / FIT_REFERENCE -- because the problem it solves is the same one.

WHAT THE SEASONAL CYCLE ACTUALLY IS. Observations are not missing-at-random in time: 185 of
365 days, on a composite anchored to a clear-sky night-time retrieval. So `sst_seasonal` is a
CLEAR-SKY-NIGHT climatology, not the climatology. That is self-consistent, since the
reconstruction targets the same quantity -- but the 180 empty days inherit it in full, and
anyone reading those days is reading this assumption.


UPDATE 9/22: we need to apply the scaling to the sst_msk_composite figures. 

Usage (from the repo root, in the `coastal_sst_data` env):

    python prototypes/DINEOF/src/standardize.py \\
        --config prototypes/DINEOF/configs/config.standardize.admiralty_inlet.yaml
    ... --dry-run       # fit and print the summary, write nothing
    ... --no-figures
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[1]              # prototypes/DINEOF
DEFAULT_CONFIG = ROOT / "configs" / "config.standardize.admiralty_inlet.yaml"

# Same cross-prototype bridge build_cube.py and composite.py use.
_CMM_SRC = ROOT.parent / "cloud_mixture_model" / "src"
if str(_CMM_SRC) not in sys.path:
    sys.path.insert(1, str(_CMM_SRC))

from seasonal_smoothing import (                        # noqa: E402
    FIT_FULL, FIT_MEAN_ONLY, FIT_REFERENCE, amplitude_phase, design_matrix, term_names)

import outlier_detection as OD                          # noqa: E402  (solve_spde, GMRF smoother)
import standardize_figures                              # noqa: E402  (DINEOF's own src/)
import plotting                                         # noqa: E402

from nearshore_sst import provenance, store, datacube   # noqa: E402
from nearshore_sst.datacube import CompressionSpec      # noqa: E402

log = logging.getLogger("standardize")


# ==================================================================== config

DEFAULTS = {
    "data": {
        "aoi": "admiralty_inlet",
        "cube": "data/datacube/admiralty_inlet_composite.zarr",
        "out": "data/datacube/admiralty_inlet_standardized_validation.zarr",
        "watervar": "landcover_water",
        "channel": "sst_composite",
        "channel_msk": "sst_msk_composite",
        "validation_msk": "validation_msk"
    },
    "seasonal": {
        "n_harmonics": 1,
        "period_days": 365.25,
        "min_dates": 10,
        "max_cond": 100.0,
        "method": "harmonic",       # "harmonic" (per-pixel LS) | "gmrf" (spatially smoothed)
        "gmrf_range_px": 50.0,      # correlation range of the coefficient-field prior, pixels
        "gmrf_alpha": 2,            # 1 = membrane, 2 = thin-plate (passed to solve_spde)
        "gmrf_min_sd": 1.0e-3,      # floor on the EB-estimated prior SD per coefficient channel
    },
    "scale": {
        "estimator": "mad",
        "floor": 0.5,
        "dof_correction": True,
        "fallback": "median",
    },
    "carry": [],        # OPAQUE: a list, not a mapping
    "output": {
        "chunks": {"time": 64, "y": 128, "x": 128},
        "compression": {"codec": "zstd", "level": 5, "shuffle": "shuffle"},
        "report": "standardize_report.csv",
        "write_figures": True,
        "fig_dir": "figures/standardize",
        "figure_coefficients": True,
        "figure_fit_quality": True,
        "roughness_warn": 0.5,
        "figure_dpi": 130,
    },
}

OPAQUE_SECTIONS = {"carry"}
ESTIMATORS = ("mad", "none")
FALLBACKS = ("median", "floor")
SEASONAL_METHODS = ("harmonic", "gmrf")

# Pixel-block size for the two nanmedian passes. The residual is (T, N) float64 = 162 MB at
# full width; `np.nanmedian` copies and sorts, so it is taken in blocks.
BLOCK = 8192


def load_config(path: Path, defaults: dict = DEFAULTS) -> dict:
    """Read the YAML config over `defaults`. Unknown sections or keys are an error, so a typo
    cannot silently fall back to a default -- the contract every script in these two prototypes
    keeps. Relative paths resolve against ROOT and are then made absolute.
    """
    with open(path) as f:
        user = yaml.safe_load(f) or {}

    cfg = copy.deepcopy(defaults)
    for section, values in user.items():
        if section not in cfg:
            raise ValueError(f"{path}: unknown config section '{section}'")
        if section in OPAQUE_SECTIONS:
            cfg[section] = copy.deepcopy(values)
            continue
        for key, value in (values or {}).items():
            if key not in cfg[section]:
                raise ValueError(f"{path}: unknown key '{section}.{key}'")
            cfg[section][key] = value

    for section, key in (("data", "cube"), ("data", "out"), ("output", "fig_dir")):
        p = Path(cfg[section][key])
        cfg[section][key] = (p if p.is_absolute() else ROOT / p).resolve()

    validate_seasonal(cfg, path)
    validate_scale(cfg, path)
    validate_carry(cfg, path)
    return cfg


def validate_seasonal(cfg: dict, path: Path) -> None:
    s = cfg["seasonal"]
    if int(s["n_harmonics"]) < 0:
        raise ValueError(f"{path}: seasonal.n_harmonics must be >= 0")
    if float(s["period_days"]) <= 0:
        raise ValueError(f"{path}: seasonal.period_days must be positive")
    n_params = 1 + 2 * int(s["n_harmonics"])
    if int(s["min_dates"]) < n_params:
        raise ValueError(
            f"{path}: seasonal.min_dates ({s['min_dates']}) is below the {n_params} parameters "
            f"a {s['n_harmonics']}-harmonic fit needs; the system would be underdetermined")
    if float(s["max_cond"]) <= 1:
        raise ValueError(f"{path}: seasonal.max_cond must be > 1")
    if s.get("method", "harmonic") not in SEASONAL_METHODS:
        raise ValueError(f"{path}: seasonal.method must be one of {SEASONAL_METHODS}")
    if float(s.get("gmrf_range_px", 50.0)) <= 0:
        raise ValueError(f"{path}: seasonal.gmrf_range_px must be positive")
    if int(s.get("gmrf_alpha", 2)) not in (1, 2):
        raise ValueError(f"{path}: seasonal.gmrf_alpha must be 1 (membrane) or 2 (thin-plate)")
    if float(s.get("gmrf_min_sd", 1.0e-3)) <= 0:
        raise ValueError(f"{path}: seasonal.gmrf_min_sd must be positive")


def validate_scale(cfg: dict, path: Path) -> None:
    s = cfg["scale"]
    if s["estimator"] not in ESTIMATORS:
        raise ValueError(f"{path}: scale.estimator must be one of {ESTIMATORS}")
    if s["fallback"] not in FALLBACKS:
        raise ValueError(f"{path}: scale.fallback must be one of {FALLBACKS}")
    if float(s["floor"]) <= 0:
        raise ValueError(
            f"{path}: scale.floor must be positive -- it is the guard against a pixel whose "
            "residuals happen to be near-identical getting a scale of ~0 and a z of ~1e6")


def validate_carry(cfg: dict, path: Path) -> None:
    carry = cfg["carry"]
    if not isinstance(carry, list):
        raise ValueError(f"{path}: `carry` must be a list of channel names")
    for name in carry:
        if not isinstance(name, str):
            raise ValueError(f"{path}: carry entries must be strings, got {name!r}")
    if len(set(carry)) != len(carry):
        raise ValueError(f"{path}: duplicate entries in `carry`")


def validate_channels(cfg: dict, ds: xr.Dataset, path: Path) -> None:
    available = sorted(map(str, ds.data_vars))
    wanted = [("data.watervar", cfg["data"]["watervar"]),
              ("data.channel", cfg["data"]["channel"])]
    wanted += [(f"carry[{i}]", n) for i, n in enumerate(cfg["carry"])]
    for where, name in wanted:
        if name not in ds:
            raise ValueError(
                f"{path}: {where} = {name!r} is not in the cube; available: {available}")


# ==================================================================== the fit

def fit_harmonics_gappy(Y: np.ndarray, O: np.ndarray, X: np.ndarray, *,
                        min_dates: int, max_cond: float) -> tuple[np.ndarray, dict]:
    """Per-pixel least-squares harmonic coefficients over the OBSERVED entries only.

    Y  (T, N) float, the value at observed entries and anything at the others (masked by O)
    O  (T, N) bool, True where Y is a real observation
    X  (T, P) design matrix from `seasonal_smoothing.design_matrix`

    Returns (coef (P, N), info) with info holding per-pixel `n_obs` and `fit_type`.

    Three tiers, taken from `seasonal_smoothing.fit_harmonics` because the problem is the same
    even though the input is not:

      FIT_FULL       >= min_dates observations AND cond(X'WX) <= max_cond -- its own full fit
      FIT_MEAN_ONLY  some observations, but too few or too clustered to identify a cycle --
                     the REFERENCE seasonal shape with only the mean level fitted to its own
                     data. A pixel observed in one season only would otherwise get a wild
                     amplitude extrapolated into seasons it never saw; `max_cond` is the only
                     thing that catches that case, since it has the date COUNT to look fine.
      FIT_REFERENCE  no observations at all -- the reference cycle wholesale. Flagged, and
                     dropped downstream: a pixel with no data must not come back as a smooth
                     plausible series indistinguishable from a reconstructed one.

    The normal equations are accumulated as one gemm rather than an (n,i,j) einsum: the einsum
    is O(T*N*P^2) with a bad access pattern, the gemm is a single (N,T) @ (T,P*P) BLAS call.
    """
    T, P = X.shape
    N = Y.shape[1]
    if T < P:
        raise ValueError(f"{T} dates cannot fit {P} seasonal terms")

    Of = O.astype(np.float64)
    n_obs = Of.sum(axis=0).astype(int)

    A = (Of.T @ (X[:, :, None] * X[:, None, :]).reshape(T, P * P)).reshape(N, P, P)
    b = (Of * np.nan_to_num(Y)).T @ X

    coef = np.zeros((N, P))
    fit_type = np.full(N, FIT_REFERENCE)

    cand = np.flatnonzero(n_obs >= max(min_dates, P))
    if cand.size:
        ok = np.linalg.cond(A[cand]) <= max_cond
        full = cand[ok]
        if full.size:
            coef[full] = np.linalg.solve(A[full], b[full][..., None])[..., 0]
            fit_type[full] = FIT_FULL
    if not (fit_type == FIT_FULL).any():
        raise ValueError(
            f"no pixel has >= {min_dates} well-conditioned observations; lower "
            "seasonal.min_dates or seasonal.n_harmonics")

    # The reference cycle: the mean coefficient vector over the pixels that could be fitted.
    ref = coef[fit_type == FIT_FULL].mean(axis=0)               # (P,)
    rest = np.flatnonzero(fit_type != FIT_FULL)
    coef[rest] = ref
    some = rest[n_obs[rest] > 0]
    if some.size:
        shape_part = X[:, 1:] @ ref[1:]                         # (T,) the reference cycle
        resid = (np.nan_to_num(Y[:, some]) - shape_part[:, None]) * Of[:, some]
        coef[some, 0] = resid.sum(axis=0) / n_obs[some]
        fit_type[some] = FIT_MEAN_ONLY

    return coef.T, {"n_obs": n_obs, "fit_type": fit_type, "A": A}


def robust_scale(R: np.ndarray, n_obs: np.ndarray, n_params: int, *, floor: float,
                 dof_correction: bool, fallback: str,
                 fit_type: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel 1.4826 * MAD of the residuals. Returns (scale, raw_mad).

    Robust rather than an ordinary sd because the point is to be unmoved by the outliers that
    survived the upstream filter: 2.48% of the standardized values here exceed |z| = 3, and an
    ordinary sd would inflate to accommodate them and then fail to flag them.

    `dof_correction` multiplies by sqrt(n / (n - P)). Fitting P parameters to n points removes
    P degrees of freedom and biases the residual scale LOW by sqrt(1 - P/n) -- 4% at a pixel
    with 36 observations, 16% near min_dates=10. Those are the coastal-margin pixels, and
    without the correction they get a scale that is too small and therefore z-scores that are
    systematically too large, exactly where the data is thinnest.
    """
    N = R.shape[1]
    mad = np.full(N, np.nan)
    # A pixel with no observations has an all-NaN residual column, which nanmedian warns
    # about. That case is real, expected, and handled by the fallback below -- so the warning
    # is suppressed here rather than printed once per block for a condition already logged.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
        for i in range(0, N, BLOCK):                            # bounded peak memory
            blk = R[:, i:i + BLOCK]
            med = np.nanmedian(blk, axis=0)
            mad[i:i + BLOCK] = 1.4826 * np.nanmedian(np.abs(blk - med), axis=0)

    scale = mad.copy()
    if dof_correction:
        n = n_obs.astype(float)
        scale = scale * np.sqrt(np.where(n > n_params, n / np.maximum(n - n_params, 1.0), np.nan))

    # A pixel whose own scale is not usable borrows one rather than producing inf or nan z.
    bad = ~np.isfinite(scale) | (scale <= 0) | (fit_type != FIT_FULL)
    if bad.any():
        good = np.isfinite(scale) & (scale > 0) & (fit_type == FIT_FULL)
        sub = float(np.median(scale[good])) if good.any() else float(floor)
        scale[bad] = sub if fallback == "median" else float(floor)
        log.info("scale: %d pixels took the %s fallback (%.3f K)", int(bad.sum()), fallback, sub)

    n_floored = int((scale < floor).sum())
    scale = np.maximum(scale, float(floor))
    if n_floored:
        log.info("scale: %d pixels floored at %.2f K", n_floored, float(floor))
    return scale, mad


def smooth_coefficients_gmrf(coef: np.ndarray, info: dict, scale0: np.ndarray,
                             water: np.ndarray, X: np.ndarray, Y: np.ndarray, O: np.ndarray,
                             s_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Spatially smooth per-pixel harmonic coefficients with a GMRF/CAR prior (empirical Bayes).

    This replaces the global-reference fallback of `fit_harmonics_gappy`. Instead of a thin or
    empty pixel borrowing one basin-wide cycle, every coefficient is treated as a smooth spatial
    field and estimated with a Matern GMRF (`outlier_detection.solve_spde`), so a pixel borrows
    from its NEIGHBOURS, weighted by its own data precision. A FULL pixel with many observations
    stays near its own least-squares value; a thin or empty one shrinks toward the locally
    interpolated field. The prior marginal SD of each channel is estimated from the data by
    method of moments (empirical Bayes), mirroring `composite.vc_weights`' tau2.

    coef    (P, N)  least-squares coefficients from fit_harmonics_gappy; only FIT_FULL columns
                    are trustworthy (A is singular/ill-conditioned elsewhere).
    info    dict with "A" (N, P, P), "n_obs" (N,), "fit_type" (N,).
    scale0  (N,)    an initial per-pixel residual scale, for the per-coefficient obs variance.
    water   (y, x)  bool mask; N == water.sum().
    X       (T, P)  harmonic design matrix. Y (T, N) values (0 off-obs), O (T, N) bool.
    Returns (coef_s (P, N), fit_type_new (N,) int8 with the -1/0/1/2 codes).
    """
    P, N = coef.shape
    A, n_obs = info["A"], info["n_obs"]
    full = info["fit_type"] == FIT_FULL
    range_px = float(s_cfg.get("gmrf_range_px", 50.0))
    alpha = int(s_cfg.get("gmrf_alpha", 2))
    min_sd = float(s_cfg.get("gmrf_min_sd", 1.0e-3))

    # Per-coefficient estimation variance at the FULL pixels: diag(scale^2 * A^-1).
    v = np.full((N, P), np.nan)
    if full.any():
        Ainv = np.linalg.inv(A[full])                       # (n_full, P, P)
        v[full] = (scale0[full] ** 2)[:, None] * np.einsum("ipp->ip", Ainv)

    wflat = np.flatnonzero(water.ravel())                   # column index -> grid cell

    def to_grid(col: np.ndarray) -> np.ndarray:             # (N,) -> (y, x), NaN off-water
        g = np.full(water.size, np.nan)
        g[wflat] = col
        return g.reshape(water.shape)

    def from_grid(field: np.ndarray) -> np.ndarray:         # (y, x) -> (N,)
        return field.reshape(-1)[wflat]

    def eb_marg_sd(obs: np.ndarray, obs_var: np.ndarray, sel: np.ndarray) -> float:
        """Method-of-moments prior SD: spatial variance minus mean estimation variance."""
        if sel.sum() < 2:
            return min_sd
        var_spatial = float(np.var(obs[sel]))
        mean_ev = float(np.nanmean(obs_var[sel]))
        return float(np.sqrt(max(var_spatial - mean_ev, min_sd ** 2)))

    coef_s = coef.copy()

    # 1) Harmonic channels: smooth the FULL-pixel estimates; thin/empty pixels borrow neighbours.
    for p in range(1, P):
        obs = np.where(full, coef[p], np.nan)
        obs_var = np.where(full, v[:, p], np.nan)
        prior_mean = float(np.mean(coef[p, full])) if full.any() else 0.0
        marg_sd = eb_marg_sd(coef[p], obs_var, full)
        field, _ = OD.solve_spde(to_grid(obs), water, prior_mean, range_px, marg_sd,
                                 to_grid(obs_var), alpha=alpha)
        coef_s[p] = from_grid(field)

    # 2) Mean channel, AFTER the cycle shape is known, so each pixel recenters to its own data
    #    where it has any and borrows the mean only where it has none. This is the spatial
    #    counterpart of the old MEAN_ONLY step, with a locally-smoothed shape in place of the
    #    global cycle. The own-data de-cycled mean follows from the normal equations:
    #    c0 = (b0 - A[0, 1:] . coef_s[1:]) / A[0, 0], with A[0, 0] = n_obs, b0 = sum of obs y.
    has = n_obs >= 1
    b0 = (np.asarray(O, float) * Y).sum(axis=0)             # (N,) sum of observed values
    cross = np.einsum("np,pn->n", A[:, 0, 1:], coef_s[1:])  # (N,) A[0,1:] . coef_s[1:]
    m_pix = np.full(N, np.nan)
    m_pix[has] = (b0[has] - cross[has]) / np.maximum(n_obs[has], 1)
    obs_var_mean = np.where(has, scale0 ** 2 / np.maximum(n_obs, 1), np.nan)
    prior_mean0 = float(np.mean(m_pix[full])) if full.any() else 0.0
    marg_sd0 = eb_marg_sd(m_pix, obs_var_mean, full)
    field0, _ = OD.solve_spde(to_grid(m_pix), water, prior_mean0, range_px, marg_sd0,
                              to_grid(obs_var_mean), alpha=alpha)
    coef_s[0] = from_grid(field0)

    fit_type = np.where(full, FIT_FULL, np.where(has, FIT_MEAN_ONLY, FIT_REFERENCE))
    return coef_s, fit_type.astype(np.int8)


def fit_seasonal_coeffs(Y: np.ndarray, O: np.ndarray, X: np.ndarray, water: np.ndarray,
                        s_cfg: dict, sc_cfg: dict
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Per-pixel seasonal coefficients, robust scale and fit_type, honouring `seasonal.method`.

    Shared by both the standalone standardize path and pipeline.fit_seasonal so the two cannot
    drift. `harmonic` is the original gappy least-squares fit with the global-reference fallback;
    `gmrf` additionally smooths the coefficients spatially (`smooth_coefficients_gmrf`).
    `none` subtracts a single grand mean (one scalar across all pixels and times) and lets
    DINEOF estimate the mean spatial pattern and seasonal structure via its low-rank decomposition.

    Y (T, N) values (0 off-obs), O (T, N) bool, X (T, P) design, water (y, x) bool.
    Returns (coef (P, N), scale (N,), mad (N,), fit_type (N,), info) where `info` is the raw
    fit_harmonics_gappy info (A / n_obs / original fit_type) for downstream diagnostics.
    """
    method = s_cfg.get("method", "harmonic")

    if method == "none":
        N = Y.shape[1]
        grand_mean = float(np.nanmean(Y[O])) if O.any() else 0.0
        coef = np.full((1, N), grand_mean)
        n_obs = O.sum(axis=0).astype(float)
        fit_type = np.full(N, FIT_FULL, dtype="int8")
        info = {"n_obs": n_obs, "fit_type": fit_type, "coef_raw": coef.copy()}
        if sc_cfg["estimator"] == "none":
            scale = np.full(N, 1.0)
            mad = np.full(N, np.nan)
        else:
            R = np.where(O, Y - grand_mean, np.nan)
            scale, mad = robust_scale(R, n_obs, 1, floor=float(sc_cfg["floor"]),
                                      dof_correction=bool(sc_cfg["dof_correction"]),
                                      fallback=sc_cfg["fallback"], fit_type=fit_type)
        return coef, scale, mad, fit_type, info

    P = X.shape[1]
    coef, info = fit_harmonics_gappy(Y, O, X, min_dates=int(s_cfg["min_dates"]),
                                     max_cond=float(s_cfg["max_cond"]))
    info["coef_raw"] = coef.copy()          # per-pixel LS coefficients, before any spatial smooth

    def scale_and_mad(c: np.ndarray, ft: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if sc_cfg["estimator"] == "none":
            return np.full(c.shape[1], 1.0), np.full(c.shape[1], np.nan)
        R = np.where(O, Y - X @ c, np.nan)
        return robust_scale(R, info["n_obs"], P, floor=float(sc_cfg["floor"]),
                            dof_correction=bool(sc_cfg["dof_correction"]),
                            fallback=sc_cfg["fallback"], fit_type=ft)

    if method == "gmrf":
        scale0, _ = scale_and_mad(coef, info["fit_type"])   # initial scale for the obs variance
        coef, fit_type = smooth_coefficients_gmrf(coef, info, scale0, water, X, Y, O, s_cfg)
    else:
        fit_type = info["fit_type"]
    scale, mad = scale_and_mad(coef, fit_type)
    return coef, scale, mad, fit_type, info


def roughness_over_se(coef: np.ndarray, A: np.ndarray, sd: np.ndarray, water: np.ndarray,
                      full: np.ndarray) -> dict:
    """Spatial roughness of each coefficient divided by its own standard error.

    THE NOISINESS TEST, and the reason it is a number rather than an impression. Roughness is
    the median |pixel - mean of its 4 neighbours|; the standard error is sqrt(inv(X'WX)_jj) * sd,
    the sampling uncertainty of that pixel's own coefficient. A ratio well below 1 means
    neighbouring pixels -- fitted completely independently, sharing no data and no prior --
    agree far better than their individual uncertainties would require. That can only happen if
    the maps are tracking real spatial structure.

    Measured on admiralty_inlet: 0.16-0.20 for every term. A ratio approaching 1 would say the
    maps are mostly fitting noise and spatial regularization is worth its complexity.
    """
    inv = np.full(A.shape, np.nan)
    if full.any():
        inv[full] = np.linalg.inv(A[full])
    se = np.sqrt(np.einsum("njj->nj", inv)) * sd[:, None]       # (N, P)

    out = {}
    for j in range(coef.shape[0]):
        g = np.full(water.shape, np.nan)
        g[water] = np.where(full, coef[j], np.nan)
        s = np.zeros_like(g)
        cnt = np.zeros_like(g)
        for sh, ax in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
            r = np.roll(g, sh, axis=ax)
            m = np.isfinite(r)
            s = np.where(m, s + np.nan_to_num(r), s)
            cnt = cnt + m
        nb = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
        rough = float(np.nanmedian(np.abs(g - nb)))
        med_se = float(np.nanmedian(se[full, j])) if full.any() else np.nan
        out[j] = {"roughness": rough, "se": med_se,
                  "ratio": rough / med_se if med_se and np.isfinite(med_se) else np.nan}
    return {"per_term": out, "se": se}


# ==================================================================== driver

def standardize(ds: xr.Dataset, cfg: dict) -> dict:
    """Fit, standardize, and gather everything the cube and the figures need."""
    d, s, sc = cfg["data"], cfg["seasonal"], cfg["scale"]
    water = np.asarray(ds[d["watervar"]].compute() > 0.5)
    times = pd.to_datetime(ds["time"].values)
    H = int(s["n_harmonics"])
    P = 1 + 2 * H

    field = ds[d["channel"]].values
    O3 = np.isfinite(field)
    Y = np.where(O3, field, 0.0)[:, water].astype(np.float64)   # (T, N)
    O = O3[:, water]
    N = Y.shape[1]
    log.info("%s: %d water px, %d dates, %d observations (%.1f%% of the matrix)",
             d["aoi"], N, len(times), int(O.sum()), 100 * O.sum() / O.size)

    X = design_matrix(times, H, float(s["period_days"]))        # (T, P)
    coef, scale, mad, fit_type, info = fit_seasonal_coeffs(Y, O, X, water, s, sc)
    counts = np.bincount(fit_type, minlength=3)
    log.info("fit types (%s): %d full, %d %s, %d reference (no data)", s.get("method", "harmonic"),
             counts[FIT_FULL], counts[FIT_MEAN_ONLY],
             "borrowed" if s.get("method") == "gmrf" else "mean-only", counts[FIT_REFERENCE])

    seasonal = X @ coef                                         # (T, N)
    R = np.where(O, Y - seasonal, np.nan)
    full = info["fit_type"] == FIT_FULL                         # from the raw fit, for diagnostics

    Z = R / scale[None, :]
    z_obs = Z[O]
    mu = float(np.nanmean(z_obs))

    validation_msk = ds[d["validation_msk"]]
    field_msk = ds[d["channel_msk"]].values
    O3_msk = np.isfinite(field_msk)
    Y_msk = np.where(O3_msk, field_msk, 0.0)[:, water].astype(np.float64)   # (T, N)
    R_msk = np.where(O, Y_msk - seasonal, np.nan)
    Z_msk = R_msk / scale[None, :]

    # For the per-pixel least-squares fit, each FIT_FULL pixel's residuals sum to exactly zero
    # over its own observed entries, so the grand mean must be ~0. This is the cheapest place to
    # catch an inconsistent design matrix / mask / solve. GMRF smoothing shrinks the coefficients
    # and so does NOT preserve the exact zero-sum, so there the check is a loose warning only.
    if s.get("method", "harmonic") == "harmonic":
        if abs(mu) > 0.01:
            raise ValueError(
                f"mean standardized value over observed entries is {mu:+.4f}, expected ~0. The "
                "per-pixel least-squares residuals should sum to zero by construction; this means "
                "the design matrix, the mask, or the solve is inconsistent")
    elif abs(mu) > 0.5:
        log.warning("gmrf: mean standardized value over observed entries is %+.4f; spatial "
                    "shrinkage does not preserve the exact per-pixel zero-sum, but a value this "
                    "large still suggests a problem worth inspecting", mu)

    rough = roughness_over_se(coef, info["A"], scale, water, full)
    for j, name in enumerate(term_names(H)):
        r = rough["per_term"][j]
        log.info("  %-6s roughness %.4f / SE %.4f = %.2f", name, r["roughness"], r["se"],
                 r["ratio"])
    worst = max((r["ratio"] for r in rough["per_term"].values() if np.isfinite(r["ratio"])),
                default=np.nan)
    warn_at = float(cfg["output"]["roughness_warn"])
    if np.isfinite(worst) and worst > warn_at:
        log.warning(
            "roughness/SE reaches %.2f (> output.roughness_warn = %.2f): neighbouring pixels "
            "no longer corroborate each other, so the coefficient maps are substantially "
            "fitting noise. Inspect figures/standardize/*/coefficients.png -- the amplitude/SE "
            "panel is the one to read -- and consider more observations, fewer harmonics, or "
            "spatial regularization.", worst, warn_at)

    log.info("standardized: sd %.3f, |z|>3 %.2f%%, max|z| %.1f, mean %+.5f",
             float(np.std(z_obs)), 100 * float(np.mean(np.abs(z_obs) > 3)),
             float(np.max(np.abs(z_obs))), mu)

    amp, peak = {}, {}
    for k in range(1, H + 1):
        a, p = amplitude_phase(coef, k, float(s["period_days"]))
        amp[k], peak[k] = a, p

    return dict(water=water, times=times, X=X, coef=coef, scale=scale, mad=mad,
                Z=Z, Z_msk=Z_msk, msk=validation_msk, O=O, O3=O3, n_obs=info["n_obs"],
                fit_type=fit_type,
                amp=amp, peak=peak, rough=rough, mu=mu, P=P, H=H,
                counts=dict(full=int(counts[FIT_FULL]), mean_only=int(counts[FIT_MEAN_ONLY]),
                            reference=int(counts[FIT_REFERENCE])))


def _grid(v: np.ndarray, water: np.ndarray, fill=np.nan, dtype="float32") -> np.ndarray:
    g = np.full(water.shape, fill, dtype=dtype)
    g[water] = v
    return g


def build_dataset(ds: xr.Dataset, cfg: dict, fit: dict) -> xr.Dataset:
    """The standardized cube. Stores the COEFFICIENTS, not a full seasonal cube: the transform
    is exactly as invertible either way, at a hundredth of the bytes."""
    d, s = cfg["data"], cfg["seasonal"]
    water = fit["water"]
    dims = ("time", "y", "x")
    yx = dims[1:]
    data = {}

    z = np.full(fit["O3"].shape, np.nan, dtype="float32")
    z[:, water] = fit["Z"].astype("float32")
    za = xr.DataArray(z, dims=dims, name="sst_z")
    za.attrs.update(
        long_name="standardized SST anomaly", units="1",
        source_channel=d["channel"],
        comment=("(value - seasonal) / sst_seasonal_sd, NaN where unobserved. THIS IS THE "
                 "eDINEOF Validation set. Invert with sst_z * sst_seasonal_sd + the seasonal cycle "
                 "rebuilt from sst_seasonal_coef."))
    data["sst_z"] = za

    # masked sst_channel
    z_msk= np.full(fit["O3"].shape, np.nan, dtype="float32")
    z_msk[:, water] = fit["Z_msk"].astype("float32")
    za_msk = xr.DataArray(z_msk, dims=dims, name="sst_msk_z")
    za_msk.attrs.update(
        long_name="standardized SST anomaly with validation mask", units="1",
        source_channel=d["channel"],
        comment=("(value - seasonal) / sst_seasonal_sd, NaN where unobserved. THIS IS THE "
                 "eDINEOF INPUT. Invert with sst_z * sst_seasonal_sd + the seasonal cycle "
                 "rebuilt from sst_seasonal_coef."))
    data["sst_msk_z"] = za_msk

    # masked sst_channel
    msk= np.full(fit["O3"].shape, np.nan, dtype="float32")
    msk[:,:] = fit["msk"].astype("float32")
    msk = xr.DataArray(msk, dims=dims, name="validation_msk")
    msk.attrs.update(long_name="masked pixels for validation")
    data["validation_msk"] = msk

    coef = np.full((fit["P"],) + water.shape, np.nan, dtype="float32")
    coef[:, water] = fit["coef"].astype("float32")
    ca = xr.DataArray(coef, dims=("term",) + yx, name="sst_seasonal_coef")
    ca.attrs.update(
        long_name="per-pixel seasonal harmonic coefficients", units="K",
        n_harmonics=fit["H"], period_days=float(s["period_days"]),
        # The term names live in an ATTR, not a string coordinate: fixed-width unicode has no
        # stable Zarr V3 specification, so a string coord may be unreadable by other zarr
        # implementations and can change representation between versions.
        terms=json.dumps(term_names(fit["H"])),
        design=("[1, cos(2*pi*k*doy/period), sin(2*pi*k*doy/period) for k in 1..n_harmonics]; "
                "doy is the fractional day of year, so the cycle is periodic and carries no "
                "multi-year trend"),
        comment=("fitted on OBSERVED entries only. Because coverage is 185 of 365 days and the "
                 "composite is anchored on a clear-sky night-time retrieval, this is a "
                 "clear-sky-night climatology rather than the climatology."))
    data["sst_seasonal_coef"] = ca

    sd = xr.DataArray(_grid(fit["scale"], water), dims=yx, name="sst_seasonal_sd")
    sd.attrs.update(long_name="per-pixel robust residual scale", units="K",
                    estimator=cfg["scale"]["estimator"], floor=float(cfg["scale"]["floor"]),
                    dof_correction=int(bool(cfg["scale"]["dof_correction"])),
                    comment="1.4826 * MAD of the seasonal residual, dof-corrected then floored")
    data["sst_seasonal_sd"] = sd

    for k in range(1, fit["H"] + 1):
        a = xr.DataArray(_grid(fit["amp"][k], water), dims=yx, name=f"sst_seasonal_amplitude{k}")
        a.attrs.update(long_name=f"amplitude of harmonic {k}", units="K")
        data[f"sst_seasonal_amplitude{k}"] = a
        p = xr.DataArray(_grid(fit["peak"][k], water), dims=yx, name=f"sst_seasonal_peak_doy{k}")
        p.attrs.update(long_name=f"day of year of the peak of harmonic {k}", units="day",
                       comment="CIRCULAR: 364 and 1 are adjacent. Plot on a cyclic colour scale.")
        data[f"sst_seasonal_peak_doy{k}"] = p

    n = xr.DataArray(_grid(fit["n_obs"], water, fill=0, dtype="int16"), dims=yx, name="sst_n_obs")
    n.attrs.update(long_name="observations used in the seasonal fit", units="1")
    data["sst_n_obs"] = n

    ft = xr.DataArray(_grid(fit["fit_type"], water, fill=-1, dtype="int8"), dims=yx,
                      name="sst_fit_type")
    ft.attrs.update(
        long_name="which seasonal fit each pixel received", units="1",
        flag_values=json.dumps([-1, FIT_REFERENCE, FIT_MEAN_ONLY, FIT_FULL]),
        flag_meanings=(f"-1=land {FIT_REFERENCE}=reference_cycle_no_data "
                       f"{FIT_MEAN_ONLY}=reference_shape_local_mean {FIT_FULL}=full_per_pixel_fit"),
        comment=("pixels below FIT_FULL have no usable cycle of their own. FIT_REFERENCE "
                 "pixels have NO observations at all and must be dropped downstream, not "
                 "reconstructed -- they would come back as plausible fabricated series."))
    data["sst_fit_type"] = ft

    for name in cfg["carry"]:
        src = ds[name]
        da = xr.DataArray(src.values, dims=src.dims, name=name)
        da.attrs.update(src.attrs, carried_from=str(d["cube"]))
        data[name] = da

    coords = {c: ds[c].values for c in dims if c in ds.coords}
    out = xr.Dataset(data, coords=coords)

    # Carried channels still hold the SOURCE store's encoding, which conflicts with the
    # encoding built at write time and silently wins. Cleared on data_vars ONLY -- the time
    # coord's units/calendar must survive or the axis is rewritten as plain integers.
    for v in out.data_vars:
        out[v].encoding = {}
    if "time" in out.coords:
        out["time"].attrs.update(ds["time"].attrs)
    return out


def cube_attrs(cfg: dict, src_attrs: dict, out: xr.Dataset, fit: dict) -> dict:
    spec = {"seasonal": cfg["seasonal"], "scale": cfg["scale"],
            "carry": list(cfg["carry"]), "channel": cfg["data"]["channel"],
            "source_cube": str(cfg["data"]["cube"])}
    quality = {"fit_types": fit["counts"], "mean_z": fit["mu"],
               "roughness_over_se": {term_names(fit["H"])[j]: r["ratio"]
                                     for j, r in fit["rough"]["per_term"].items()}}
    return {**src_attrs,
            "aoi_id": cfg["data"]["aoi"],
            "dineof_standardize": json.dumps(spec, sort_keys=True, default=str),
            "dineof_standardize_quality": json.dumps(quality, sort_keys=True, default=str),
            "dineof_standardize_channels": json.dumps(sorted(map(str, out.data_vars))),
            "dineof_standardized_at": provenance.now_utc(),
            "package_version": provenance.package_version(),
            "code_version": provenance.code_version()}


def write_cube(out: xr.Dataset, cfg: dict, src_attrs: dict, fit: dict) -> None:
    dest = cfg["data"]["out"]
    out.attrs.update(cube_attrs(cfg, src_attrs, out, fit))
    compression = CompressionSpec(**cfg["output"]["compression"])
    encoding = datacube.build_encoding(out, compression, dict(cfg["output"]["chunks"]))

    store.sweep_scratch(dest)
    if dest.exists():
        log.info("replacing existing cube at %s", dest)
    with store.atomic(dest) as tmp:
        datacube.write_zarr(out, tmp, encoding)
    log.info("wrote %s", dest)


def report(fit: dict, cfg: dict) -> pd.DataFrame:
    """One row per seasonal term plus a summary row -- the table view of the figures."""
    rows = []
    for j, name in enumerate(term_names(fit["H"])):
        r = fit["rough"]["per_term"][j]
        full = fit["fit_type"] == FIT_FULL
        v = fit["coef"][j][full]
        rows.append(dict(term=name, median=float(np.median(v)),
                         p5=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95)),
                         median_se=r["se"], roughness=r["roughness"],
                         roughness_over_se=r["ratio"]))
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML config")
    p.add_argument("--dry-run", action="store_true",
                   help="fit and print the summary, write nothing")
    p.add_argument("--no-figures", action="store_true", help="skip the figures")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(args.config)

    if not cfg["data"]["cube"].exists():
        raise SystemExit(
            f"no cube at {cfg['data']['cube']}\n"
            "build it with: python prototypes/DINEOF/src/composite.py --config "
            "prototypes/DINEOF/configs/config.composite.admiralty_inlet.yaml")

    ds = xr.open_zarr(cfg["data"]["cube"])
    validate_channels(cfg, ds, args.config)
    src_attrs = dict(ds.attrs)

    for k in ds.keys():
        print(k)

    fit = standardize(ds, cfg)
    rep = report(fit, cfg)

    if args.dry_run:
        with pd.option_context("display.width", 200, "display.max_columns", None):
            print(rep.to_string(index=False))
        log.info("dry run: nothing written")
        return

    out = build_dataset(ds, cfg, fit)
    write_cube(out, cfg, src_attrs, fit)

    path = cfg["data"]["out"].parent / cfg["output"]["report"]
    path.parent.mkdir(parents=True, exist_ok=True)
    rep.to_csv(path, index=False)
    log.info("wrote %s", path)

    if cfg["output"]["write_figures"] and not args.no_figures:
        # After the cube and the report, and isolated: a drawing bug must not cost a fit that
        # already succeeded. Everything drawn is reproducible from the cube alone.
        try:
            standardize_figures.render(cfg, fit, plotting.extent_km(ds),
                                       cfg["output"]["fig_dir"] / cfg["data"]["aoi"])
        except Exception:
            log.exception("figures failed; the cube and report were written and are intact")


if __name__ == "__main__":
    main()
