"""Cloud / outlier detection for SST imagery.

Two-sided mixture (cold cloud, warm anomaly, clear) with a mean-field Ising prior on the
labels, alternating with a Matern GMRF (SPDE) analysis of the SST field. Ported from the
"new outlier detection code" section of cloud_filter.ipynb; every parameter lives in the
YAML config (configs/outlier_detection.yaml).

Per acquisition:

  1. Frozen covariate: robust MODIS composite around the date, smoothed once by the SPDE.
  2. Global sensor offset `center` = robust centre of (y - covariate).
  3. Outer loop, until the cloud fraction and p_valid stabilise or max_iter:
       residual y - x -> unary log-odds (tidal-aware warm prior) -> mean-field E-steps
       -> p_valid -> SPDE analysis with obs_var = obs_error / p_valid (soft mask).
  4. combined = x + p_valid * (y - center - x)

Outputs one NetCDF (fields + per-iteration history) and one PNG panel per date.

Usage (from the repo root, in the `coastal_sst_data` env):

    python prototypes/cloud_mixture_model/src/outlier_detection.py \\
        --config prototypes/cloud_mixture_model/configs/outlier_detection.yaml
    ... --dates 2025-05-28 2025-06-07     # override data.dates
"""

from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from scipy.fft import dctn, idctn
from scipy.special import log_ndtr

import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = logging.getLogger("outlier_detection")

ROOT = Path(__file__).resolve().parents[1]          # prototypes/cloud_mixture_model
DEFAULT_CONFIG = ROOT / "configs" / "outlier_detection.yaml"

LOG_2PI = np.log(2.0 * np.pi)
LAND_COLOR = "#c8c8c8"


# ==================================================================== config

DEFAULTS = {
    "data": {
        "cube": "data/datacube/admiralty_inlet.zarr",
        "var": "eco_sst_v002",
        "ref_var": "modis_sst_aqua",
        "landvar": "landcover_water",
        "depthvar": "depth_cudem",
        "tidal_depth_m": 3.0,
        "dates": [],
        "min_pixels": 64,
    },
    "output": {
        "dir": "outputs/outlier_detection",
        "fig_dir": "figures/outlier_detection",
        "write_netcdf": True,
        "write_figures": True,
    },
    "reference": {
        "window_days": 4,
        "qc_var": None,
        "qc_max": 1,
        "quantile": 0.5,
        "min_obs": 1,
    },
    "covariate": {
        "init_prior": 285.0,
        "init_range_px": 200.0,
        "init_marg_sd": 0.05,
        "init_obs_error": 0.5,
    },
    "analysis": {
        "range_px": 100.0,
        "marg_sd": 0.05,
        "obs_error": 0.5,
        "clear_sd": 0.75,
        "alpha": 2,
    },
    "mixture": {
        "lambda_cloud": 4.0,
        "min_dev_cold": 1.0,
        "prior_cloud": 0.5,
        "lambda_hot": 6.5,
        "min_dev_hot": 1.0,
        "prior_hot": 0.005,
        "p_h_tidal": 0.5,
        "beta_cloud": 1.75,
        "n_sweeps_cloud": 20,
        "beta_hot": 0.125,
        "n_sweeps_hot": 10,
        "estep_damp": 0.5,
        "outer_damp": 0.5,
        "p_floor": 0.01,
    },
    "convergence": {
        "max_iter": 50,
        "min_iter": 3,
        "cloud_frac_tol": 1e-3,
        "p_valid_tol": 1e-3,
        "patience": 2,
    },
    "solver": {
        "tol": 1e-6,
        "pcg_max_iter": 2000,
        "precondition": True,
    },
}


def load_config(path: Path, defaults: dict = DEFAULTS) -> dict:
    """Read the YAML config over `defaults`. Unknown sections or keys are an error, so a typo
    cannot silently fall back to a default. Relative paths resolve against ROOT."""
    with open(path) as f:
        user = yaml.safe_load(f) or {}

    cfg = copy.deepcopy(defaults)
    for section, values in user.items():
        if section not in cfg:
            raise ValueError(f"{path}: unknown config section '{section}'")
        for key, value in (values or {}).items():
            if key not in cfg[section]:
                raise ValueError(f"{path}: unknown key '{section}.{key}'")
            cfg[section][key] = value

    for section, key in (("data", "cube"), ("output", "dir"), ("output", "fig_dir")):
        p = Path(cfg[section][key])
        cfg[section][key] = p if p.is_absolute() else ROOT / p
    cfg["data"]["dates"] = [str(d) for d in (cfg["data"]["dates"] or [])]
    return cfg


# ============================================================ stencil / solver

def neighbour_sum(x, mask):
    """5-point stencil sum, with edges across blocked cells removed."""
    xm = np.where(mask, x, 0.0)
    s = np.zeros_like(x)
    s[1:, :] += xm[:-1, :]
    s[:-1, :] += xm[1:, :]
    s[:, 1:] += xm[:, :-1]
    s[:, :-1] += xm[:, 1:]
    return s


def pcg(Q, b, mask, Minv=None, tol=1e-8, max_iter=2000, x0=None):
    """Preconditioned conjugate gradient.

    The convergence test is ||r|| / ||b||, NOT ||r|| / ||r0||. With the latter a good warm
    start tightens the absolute target and can cost MORE iterations than a cold start;
    measured against ||b|| the same warm start converges in a handful.
    """
    b = np.where(mask, b, 0.0)
    x = np.zeros_like(b) if x0 is None else np.where(mask, x0, 0.0)
    bnorm = np.linalg.norm(b)
    if bnorm == 0.0:
        return x, 0

    r = b - Q(x)
    z = Minv(r) if Minv else r
    p = z.copy()
    rz = np.sum(r * z)

    k = 0
    for k in range(1, max_iter + 1):
        Qp = Q(p)
        pQp = np.sum(p * Qp)
        if pQp <= 0:
            break
        alpha = rz / pQp
        x += alpha * p
        r -= alpha * Qp
        if np.linalg.norm(r) / bnorm < tol:
            break
        z = Minv(r) if Minv else r
        rz_new = np.sum(r * z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, k


# ============================================================== SPDE / GMRF

def _dct_eigs(shape):
    H, W = shape
    return ((2.0 - 2.0 * np.cos(np.pi * np.arange(H) / H))[:, None]
            + (2.0 - 2.0 * np.cos(np.pi * np.arange(W) / W))[None, :])


def spde_params(shape, range_px, marg_sd, alpha=2):
    """Map (range, marginal sd) -> (kappa2, lam).

    range_px : correlation range in PIXELS. alpha=2 (nu=1): distance at which correlation
               ~0.1, r = sqrt(8*nu)/kappa. alpha=1 (nu=0): that relation degenerates,
               range_px is the e-folding scale 1/kappa.
    marg_sd  : prior sd of (field - prior_mean) at one pixel, in data units. Only
               marg_sd / sqrt(obs_var) affects the posterior MEAN.

    lam is calibrated against the exact DCT spectrum, which avoids grid-spacing and
    mass-matrix factors.
    """
    nu = alpha - 1.0                                     # d = 2
    kappa2 = (8.0 * nu / range_px ** 2) if nu > 0 else (1.0 / range_px ** 2)
    s = np.mean((kappa2 + _dct_eigs(shape)) ** (-float(alpha)))
    return float(kappa2), float((s / marg_sd ** 2) ** (1.0 / alpha))


def marginal_sd(shape, kappa2, lam, alpha=2):
    """Realised marginal sd on a land-free rectangle. For checking."""
    return float(np.sqrt(np.mean((lam * (kappa2 + _dct_eigs(shape))) ** (-float(alpha)))))


def make_operator_spde(mask, kappa2, lam, obs_prec, alpha=2):
    """Q = diag(obs_prec) + P^alpha,  P = lam * (kappa2*I + L).

    Only the PRIOR is raised to the power alpha; the data term is not.
    """
    n_nb = neighbour_sum(np.ones(mask.shape), mask)

    def P(x):
        xm = np.where(mask, x, 0.0)
        return np.where(mask, lam * (kappa2 * xm + n_nb * xm
                                     - neighbour_sum(xm, mask)), 0.0)

    def Qp(x):
        for _ in range(alpha):
            x = P(x)
        return x

    def Q(x):
        xm = np.where(mask, x, 0.0)
        return np.where(mask, obs_prec * xm + Qp(xm), 0.0)

    return Q, Qp


def make_dct_preconditioner_spde(shape, kappa2, lam, obs_prec_bar, mask, alpha=2):
    """Exact inverse of the land-free operator, in O(N log N)."""
    denom = obs_prec_bar + (lam * (kappa2 + _dct_eigs(shape))) ** float(alpha)

    def Minv(r):
        r = np.where(mask, r, 0.0)
        z = idctn(dctn(r, type=2, norm="ortho") / denom, type=2, norm="ortho")
        return np.where(mask, z, 0.0)

    return Minv


def solve_spde(obs, mask, prior_mean, range_px, marg_sd, obs_var,
               alpha=2, precondition=True, tol=1e-6, max_iter=2000, x0=None):
    """Posterior mean of a Matern GMRF given noisy, partially observed data.

    obs        (H, W)  observations, NaN where missing
    mask       (H, W)  bool, True on water
    prior_mean scalar or (H, W) -- the anchor. Where there is no data the solution
               reverts to this.
    range_px   correlation range, pixels
    marg_sd    prior marginal sd of (truth - prior_mean), data units. Small marg_sd keeps
               x near the covariate, so cloud stands out in the residual; large marg_sd
               lets x follow the cloud and detection recall drops.
    obs_var    scalar or (H, W). Pass obs_error / p_valid to soft-mask cloud.
    alpha      1 = membrane (nu=0), 2 = thin plate (nu=1).
    x0         PCG seed. Does not change the answer, only the speed.
    """
    mask = np.asarray(mask, bool)
    obs = np.asarray(obs, float)
    m = np.broadcast_to(np.asarray(prior_mean, float), mask.shape)
    t2 = np.broadcast_to(np.asarray(obs_var, float), mask.shape)

    have = mask & np.isfinite(obs) & np.isfinite(t2) & (t2 > 0)
    obs_prec = np.where(have, 1.0 / np.where(have, t2, 1.0), 0.0)

    kappa2, lam = spde_params(mask.shape, range_px, marg_sd, alpha)
    Q, _ = make_operator_spde(mask, kappa2, lam, obs_prec, alpha)

    # Solve for the anomaly z = x - m. Qp(0) = 0 so the prior term drops out, and ||b|| no
    # longer carries the ~285 K offset.
    b = obs_prec * np.where(have, np.nan_to_num(obs) - m, 0.0)

    Minv = None
    if precondition:
        pos = obs_prec[obs_prec > 0]
        bar = (float(np.exp(np.mean(np.log(pos)))) * pos.size
               / max(int(mask.sum()), 1)) if pos.size else 0.0
        Minv = make_dct_preconditioner_spde(mask.shape, kappa2, lam, bar, mask, alpha)

    z0 = None if x0 is None else np.asarray(x0, float) - m
    z, k = pcg(Q, b, mask, Minv, tol=tol, max_iter=max_iter, x0=z0)
    return np.where(mask, m + z, np.nan), k


# ========================================================== robust estimators

def _shorth(x):
    """Shortest-half interval. Returns (midpoint, length).

    The midpoint is a mode estimate that tolerates up to ~50% contamination on either side;
    with 40% cold cloud the median sits down the cold tail while the shorth stays on the
    clear mode. For a Gaussian the shortest half has length 1.349*sigma.
    """
    s = np.sort(x)
    n = s.size
    h = n // 2
    if h < 1:
        return float(np.median(s)), float(np.ptp(s))
    widths = s[h:] - s[:n - h]
    i = int(np.argmin(widths))
    return float(0.5 * (s[i] + s[i + h])), float(widths[i])


def robust_centre_scale(res, w=None, min_n=64, floor=1e-3):
    """Centre and scale of the CLEAR component of a residual field.

    res  (H, W)  residual, NaN where masked
    w    (H, W)  optional clear-sky weights (p_valid). When given, a weighted moment
                 estimate is used; otherwise a distribution-free one.

    Without weights: shorth midpoint for the centre, then a WARM-SIDE scale,
    sigma = median(res - centre | res > centre) / 0.6745, because cloud contamination is
    one-sided and cold.
    """
    r_all = np.asarray(res, float).ravel()
    finite = np.isfinite(r_all)
    r = r_all[finite]
    if r.size < min_n:
        return 0.0, 1.0

    if w is not None:
        ww = np.clip(np.nan_to_num(np.asarray(w, float).ravel()[finite]), 0.0, 1.0)
        sw = ww.sum()
        if sw > min_n:
            centre = float((ww * r).sum() / sw)
            var = float((ww * (r - centre) ** 2).sum() / sw)
            return centre, float(max(np.sqrt(max(var, 0.0)), floor))

    centre, width = _shorth(r)
    d = r[r > centre] - centre
    scale = float(np.median(d)) / 0.6745 if d.size >= min_n else width / 1.349
    return float(centre), float(max(scale, floor))


def robust_modis_composite(ds, date, ref_var, landvar, window_days=2,
                           qc_var=None, qc_max=1, quantile=0.5, min_obs=1):
    """Cloud-resistant reference composite from the coarse sensor.

    A plain .mean() over the window averages in cloud-contaminated retrievals and is
    cold-biased, and the analysis would anchor to that bias in exactly the gaps where
    nothing else constrains it.

    quantile : 0.5 (median) is the safe default. Raise toward 0.6-0.75 if the reference is
               not already cloud-screened.
    min_obs  : pixels with fewer valid times become NaN and are filled by the GMRF.
    """
    centre = pd.Timestamp(date)
    w = pd.Timedelta(days=window_days)
    window = slice(centre - w, centre + w)
    sub = ds[ref_var].sel(time=window)

    if qc_var is not None and qc_var in ds:
        sub = sub.where(ds[qc_var].sel(time=window) <= qc_max)

    n = sub.count(dim="time")
    comp = (sub.median(dim="time", skipna=True) if quantile == 0.5
            else sub.quantile(quantile, dim="time", skipna=True))
    comp = comp.where(n >= min_obs)
    comp = comp.where(ds[landvar] > 0.5)

    return np.asarray(comp, float), np.asarray(n, int)


def estimate_offset(y, covariate, mask, w=None, min_n=64):
    """Global sensor offset (target minus reference).

    Estimated against the FROZEN covariate, never against the analysis: y - x shrinks as x
    absorbs the data, whereas y - covariate has no such feedback.
    """
    r = np.asarray(y, float) - np.asarray(covariate, float)
    ok = np.asarray(mask, bool) & np.isfinite(r)
    if ok.sum() < min_n:
        return 0.0
    ww = None if w is None else np.asarray(w, float)[ok]
    centre, _ = robust_centre_scale(r[ok], w=ww, min_n=min_n)
    return float(centre)


# ============================================================ mixture E-step

def sigmoid(x):
    """Numerically stable logistic."""
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def unary_logodds(y, tidal, mu, sigma, s_c, min_dev_cold, pi_t, s_ch, min_dev_hot, phi_t, p_h_tidal, noise=True):
    """Log-odds for cold (cloud) and warm anomalies, before spatial coupling.

    Clear:  y ~ N(mu, sigma^2)
    Cloud:  y = mu - delta,  delta ~ Exp(mean s_c)
    Warm :  y = mu + delta,  delta ~ Exp(mean s_ch)

    The warm prior is p_h_tidal on tidal pixels and phi_t elsewhere.

    NOTE the -0.5*log(2*pi) term is required: the components have different functional
    forms, so it does not cancel.
    """
    w = (mu - min_dev_cold) - y                            # cold deviation
    wh = y - (mu + min_dev_hot)                           # warm deviation
    lam, lamh = 1.0 / s_c, 1.0 / s_ch

    ll_clear = -0.5 * (w / sigma) ** 2 - np.log(sigma) - 0.5 * LOG_2PI

    if noise:
        # Exp convolved with N(0, sigma^2); stable over the whole real line.
        ll_cloud = (np.log(lam) + 0.5 * (lam * sigma) ** 2 - lam * w
                    + log_ndtr((w - lam * sigma ** 2) / sigma))
        ll_hot = (np.log(lamh) + 0.5 * (lamh * sigma) ** 2 - lamh * wh
                  + log_ndtr((wh - lamh * sigma ** 2) / sigma))
    else:
        with np.errstate(invalid="ignore"):
            ll_cloud = np.where(w > 0.0, np.log(lam) - lam * w, -np.inf)
            ll_hot = np.where(wh > 0.0, np.log(lamh) - lamh * wh, -np.inf)

    tidal = np.asarray(tidal, float)
    h = np.log(pi_t / (1.0 - pi_t))
    hh = (tidal * np.log(p_h_tidal / (1.0 - p_h_tidal))
          + (1.0 - tidal) * np.log(phi_t / (1.0 - phi_t)))
    return (ll_cloud - ll_clear) + h, (ll_hot - ll_clear) + hh


def _shift0(a, dy, dx):
    """out[y, x] = a[y + dy, x + dx], zero outside the grid."""
    out = np.zeros_like(a)
    H, W = a.shape
    yd = slice(max(0, -dy), H - max(0, dy))
    ys = slice(max(0, dy), H - max(0, -dy))
    xd = slice(max(0, -dx), W - max(0, dx))
    xs = slice(max(0, dx), W - max(0, -dx))
    out[yd, xd] = a[ys, xs]
    return out


_NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def neighbour_mean(q, valid=None, fill=0.5):
    """Mean of the 4 adjacent pixels over valid neighbours only.

    Renormalising by the neighbours actually present handles scene edges and nodata with
    one mechanism. fill=0.5 makes the coupling term vanish for a pixel with no valid
    neighbour, leaving it on its unary evidence.
    """
    q = np.asarray(q, dtype=float)
    if q.ndim != 2:
        raise ValueError(f"q must be (H, W), got {q.shape}")

    finite = np.isfinite(q)
    if valid is None:
        m = finite
    else:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != q.shape:
            raise ValueError(f"valid {valid.shape} does not match q {q.shape}")
        m = finite & valid
    m = m.astype(q.dtype)

    qm = np.where(m > 0.0, q, 0.0)         # np.where, not q*m: keeps NaN out
    num = np.zeros_like(q)
    den = np.zeros_like(q)
    for dy, dx in _NEIGHBOURS:
        num += _shift0(qm, dy, dx)
        den += _shift0(m, dy, dx)

    return np.where(den > 1e-12, num / np.maximum(den, 1e-12), fill)


def estep(u, valid=None, beta=1.5, n_sweeps=8, damp=0.5):
    """Damped mean-field sweeps for one acquisition.

    beta      beta > 2 is supercritical: the field can lock to a label with no evidence.
    n_sweeps  fixed, NOT run to convergence. Each sweep propagates information one pixel,
              so this is also a range parameter, and stopping early limits saturation.
    """
    u = np.nan_to_num(np.asarray(u, float), nan=0.0, posinf=50.0, neginf=-50.0)
    v = np.ones(u.shape) if valid is None else np.asarray(valid, bool).astype(float)
    q = sigmoid(u) * v

    for _ in range(n_sweeps):
        nb = neighbour_mean(q, v > 0)
        target = sigmoid(u + beta * (2.0 * nb - 1.0))
        q = ((1.0 - damp) * q + damp * target) * v

    return q


# ================================================================= pipeline

def load_inputs(ds, date, cfg):
    """Target acquisition on water, plus the water and tidal masks."""
    d = cfg["data"]
    land = np.asarray(ds[d["landvar"]].compute() > 0.5)
    dv = d.get("depthvar")
    tidal = (np.asarray(ds[dv].compute() < d["tidal_depth_m"])
             if dv and dv in ds else np.zeros(land.shape, bool))
    da = ds[d["var"]].sel(time=date, method="nearest").compute()
    y = np.where(land, np.asarray(da, float), np.nan)
    return y, land, tidal, pd.Timestamp(da.time.values)


def build_covariate(ds, date, land, cfg):
    """Frozen covariate: robust reference composite, smoothed once by the SPDE."""
    d, ref, cov = cfg["data"], cfg["reference"], cfg["covariate"]
    sol = cfg["solver"]
    modis_raw, n_ref = robust_modis_composite(
        ds, date, d["ref_var"], d["landvar"], window_days=ref["window_days"],
        qc_var=ref["qc_var"], qc_max=ref["qc_max"], quantile=ref["quantile"],
        min_obs=ref["min_obs"])

    covariate, k = solve_spde(modis_raw, land, cov["init_prior"],
                              cov["init_range_px"], cov["init_marg_sd"],
                              cov["init_obs_error"], alpha=cfg["analysis"]["alpha"],
                              precondition=sol["precondition"], tol=sol["tol"],
                              max_iter=sol["pcg_max_iter"])
    log.info("covariate: pcg %d iters, %.1f%% of water pixels had reference data",
             k, 100 * np.isfinite(modis_raw[land]).mean())
    return covariate, modis_raw, n_ref


def _mixture_update(res, tidal, land, p_valid, first, cfg):
    """Residual -> cloud / warm responsibilities -> damped p_valid."""
    mix, ana = cfg["mixture"], cfg["analysis"]

    res_cent, sd_est = robust_centre_scale(res, w=None if first else p_valid)
    # y - x shrinks as the analysis fits the data, so sd_est drifts below the instrument
    # error and over-flags. Use clear_sd if given, else floor at the measurement noise.
    sd = (float(ana["clear_sd"]) if ana["clear_sd"] is not None
          else max(sd_est, float(np.sqrt(ana["obs_error"]))))

    ull, ull_hot = unary_logodds(res, tidal, res_cent, sd,
                                 mix["lambda_cloud"], mix["min_dev_cold"], mix["prior_cloud"],
                                 mix["lambda_hot"], mix["min_dev_hot"], mix["prior_hot"], mix["p_h_tidal"],
                                 noise=True)
    q = estep(ull, valid=land, beta=mix["beta_cloud"],
              n_sweeps=mix["n_sweeps_cloud"], damp=mix["estep_damp"])
    q_hot = estep(ull_hot, valid=land, beta=mix["beta_hot"],
                  n_sweeps=mix["n_sweeps_hot"], damp=mix["estep_damp"])

    # q and q_hot come from two independent binary fields and can sum past 1. Renormalise
    # only when they do, so p_valid never goes negative (non-SPD solve).
    s = np.maximum(q + q_hot, 1.0)
    q, q_hot = q / s, q_hot / s
    p_new = np.clip(1.0 - q - q_hot, mix["p_floor"], 1.0)

    # Damp across outer iterations; undamped, boundary pixels flip back and forth.
    od = mix["outer_damp"]
    p_valid = p_new if first else (1.0 - od) * p_valid + od * p_new
    return p_valid, q, q_hot, res_cent, sd


def _analysis_update(y, center, land, covariate, p_valid, x, cfg):
    """SPDE analysis of the offset-corrected target, soft-masked by p_valid."""
    ana, sol = cfg["analysis"], cfg["solver"]
    return solve_spde(y - center, land,
                      prior_mean=covariate,                   # frozen anchor
                      range_px=ana["range_px"], marg_sd=ana["marg_sd"],
                      obs_var=ana["obs_error"] / p_valid,     # soft mask
                      alpha=ana["alpha"], precondition=sol["precondition"],
                      tol=sol["tol"], max_iter=sol["pcg_max_iter"], x0=x)


def detect_outliers(y, land, tidal, covariate, cfg):
    """Alternate mixture labelling and SPDE analysis until the cloud fraction and p_valid
    stabilise, or max_iter.

    Converged when |d cloud_frac| < cloud_frac_tol AND mean|d p_valid| < p_valid_tol, both
    over observed water pixels, for `patience` consecutive iterations after min_iter.
    """
    conv = cfg["convergence"]
    obs = land & np.isfinite(y)
    if not obs.any():
        raise ValueError("no observed water pixels")

    center = estimate_offset(y, covariate, land)
    x = covariate.copy()                  # PCG seed only; covariate stays the anchor
    p_valid = np.ones_like(x)
    cloud_prev = None
    quiet = 0
    converged = False
    history = []

    for i in range(conv["max_iter"]):
        p_prev = p_valid
        p_valid, q, q_hot, res_cent, sd = _mixture_update(
            y - x, tidal, land, p_valid, i == 0, cfg)
        x_new, k = _analysis_update(y, center, land, covariate, p_valid, x, cfg)
        step = float(np.nanmax(np.abs(x_new - x)))
        x = x_new

        cloud_frac = float(q[obs].mean())
        warm_frac = float(q_hot[obs].mean())
        dp = np.abs(p_valid[obs] - p_prev[obs])
        d_cloud = np.nan if cloud_prev is None else abs(cloud_frac - cloud_prev)
        d_pvalid_mean = np.nan if i == 0 else float(dp.mean())
        d_pvalid_max = np.nan if i == 0 else float(dp.max())
        cloud_prev = cloud_frac

        history.append(dict(iter=i, pcg=k, step=step, res_cent=res_cent, sd=sd,
                            cloud_frac=cloud_frac, warm_frac=warm_frac, d_cloud=d_cloud,
                            d_pvalid_mean=d_pvalid_mean, d_pvalid_max=d_pvalid_max))
        log.info("  iter %2d  pcg %4d  step %7.4f  centre %+6.3f  sd %5.3f  "
                 "cloud %.3f (d %.2e)  warm %.3f  dp mean %.2e max %.2e",
                 i, k, step, res_cent, sd, cloud_frac, d_cloud, warm_frac,
                 d_pvalid_mean, d_pvalid_max)

        if i > 0 and d_cloud < conv["cloud_frac_tol"] and d_pvalid_mean < conv["p_valid_tol"]:
            quiet += 1
        else:
            quiet = 0
        if i + 1 >= conv["min_iter"] and quiet >= conv["patience"]:
            converged = True
            log.info("  converged after %d iterations", i + 1)
            break
    else:
        log.warning("  hit max_iter=%d without converging", conv["max_iter"])

    residual = y - center - x
    return dict(analysis=x, combined=x + p_valid * residual, p_valid=p_valid,
                q_cloud=q, q_hot=q_hot, residual=residual, center=center, sd=sd,
                converged=converged, n_iter=len(history), history=history)


def run_date(ds, date, cfg):
    """Full pipeline for one acquisition."""
    y, land, tidal, t = load_inputs(ds, date, cfg)
    covariate, modis_raw, n_ref = build_covariate(ds, t, land, cfg)
    result = detect_outliers(y, land, tidal, covariate, cfg)
    # `obs`, not `y`: `y` is the cube's row coordinate and would clash in the NetCDF.
    result.update(obs=y, land=land, tidal=tidal, time=t, covariate=covariate,
                  modis_raw=modis_raw, n_ref=n_ref)
    return result


def acquisition_dates(ds, var, land, min_pixels):
    """Dates of every time step of `var` with at least `min_pixels` finite water pixels."""
    n = ds[var].where(xr.DataArray(land, dims=ds[var].dims[-2:])).count(
        dim=list(ds[var].dims[-2:])).compute()
    times = n.time.values[np.asarray(n) >= min_pixels]
    return [str(pd.Timestamp(t).date()) for t in times]


# =================================================================== outputs

FIELDS = ("obs", "covariate", "modis_raw", "analysis", "combined", "p_valid",
          "q_cloud", "q_hot", "residual", "n_ref")


def write_netcdf(result, ds, cfg, path):
    """2-D fields on the cube's grid, per-iteration history over `iter`, config as attrs."""
    var = cfg["data"]["var"]
    dims = ds[var].dims[-2:]
    coords = {d: ds[d].values for d in dims if d in ds.coords}
    data = {name: (dims, np.asarray(result[name])) for name in FIELDS}
    data["land"] = (dims, result["land"].astype("int8"))
    data["tidal"] = (dims, result["tidal"].astype("int8"))
    hist = pd.DataFrame(result["history"])
    for col in hist.columns.drop("iter"):
        data[f"hist_{col}"] = ("iter", hist[col].to_numpy())
    coords["iter"] = hist["iter"].to_numpy()

    out = xr.Dataset(data, coords=coords)
    out.attrs.update(
        date=str(result["time"]), var=var, center=result["center"], sd=result["sd"],
        converged=int(result["converged"]), n_iter=result["n_iter"],
        config=yaml.safe_dump(_plain(cfg), sort_keys=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(path)
    log.info("wrote %s", path)


def _plain(cfg):
    return {s: {k: (str(v) if isinstance(v, Path) else v) for k, v in vals.items()}
            for s, vals in cfg.items()}


def plot_panels(result, cfg, path):
    """obs | covariate | p_valid | q_hot | combined, land grey, SST panels on one scale."""
    land = result["land"]
    sst = np.concatenate([result[k][land & np.isfinite(result[k])]
                          for k in ("obs", "covariate", "combined")])
    vmin, vmax = np.percentile(sst, [2, 98]) if sst.size else (None, None)

    panels = [("obs", "observed", "inferno", vmin, vmax),
              ("covariate", "covariate", "inferno", vmin, vmax),
              ("p_valid", "p_valid", "viridis", 0.0, 1.0),
              ("q_hot", "q_hot", "magma", 0.0, 1.0),
              ("combined", "combined", "inferno", vmin, vmax)]
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
    status = "converged" if result["converged"] else "max_iter"
    fig.suptitle(f"{cfg['data']['var']} {result['time'].date()} -- {result['n_iter']} iters "
                 f"({status}), cloud {result['history'][-1]['cloud_frac']:.1%}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    log.info("wrote %s", path)


# =============================================================== entry point

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML config")
    p.add_argument("--dates", nargs="+", default=None, help="override data.dates")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_config(args.config)
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

    failed = []
    for date in dates:
        log.info("=== %s %s", d["var"], date)
        try:
            result = run_date(ds, date, cfg)
            stem = f"{d['var']}_{result['time'].date()}"
            if o["write_netcdf"]:
                write_netcdf(result, ds, cfg, o["dir"] / f"{stem}.nc")
            if o["write_figures"]:
                plot_panels(result, cfg, o["fig_dir"] / f"{stem}.png")
        except Exception:
            log.exception("failed on %s", date)
            failed.append(date)

    if failed:
        log.warning("%d/%d dates failed: %s", len(failed), len(dates), ", ".join(failed))


if __name__ == "__main__":
    main()
