"""Point-level state-space (Kalman) smoothers for SST time series.

Stage 4 extracts a handful of named locations from the gap-filled cube and smooths each one's
daily series with a linear-Gaussian state-space model, fit by maximum likelihood. The engine is
statsmodels (`sm.tsa.statespace.MLEModel`); this module only assembles the model matrices and the
parameter bookkeeping on top of it.

The gap-filled cube tells us, per pixel-day, WHICH KIND of estimate each value is. We turn that
into an observation whose error variance depends on its type, and fuse the types through the
filter. The two readable value channels are F = `sst_filled` and S = `sst_smooth`; each day has a
context (observed / point_fill / day_fill / climatology). An observation TYPE is (channel,
context), and each type carries its own, MLE-estimated observation variance:

    observed  point_fill  day_fill                            (channel F, sst_filled)
    smooth_observed  smooth_point_fill  smooth_day_fill  smooth_climatology   (channel S, sst_smooth)

A group of sites selects types with `sources` and a `mode`:
  * hierarchy -- one observation per day, the first available type in `sources` order; its type
                 sets that day's variance (so a hierarchy still uses different errors per type).
  * multiple  -- every available type that day is fed as a separate measurement, each with its own
                 variance. (F and S are correlated, so this is an approximation.)

Four models: random_walk (local level), ar1 (AR(1) about a mean), ar1_harmonic (+ seasonal
harmonics), harmonic_trend (local linear trend + seasonal harmonics). Parameters -- process
variance(s), AR coefficient, mean, harmonic coefficients, and the per-type observation variances
-- are estimated by MLE and can be SHARED across a group's sites (one value fit jointly) or fit
per site, chosen per parameter.

An extra channel I = `insitu` carries independent in-situ measurements (the unbiased anchor). With
`dineof_bias` the DINEOF channels (F/S) get a measurement bias added to their observation
intercept -- a constant, optionally plus a seasonal day-of-year harmonic -- so the model is
y_insitu = SST + eps_i and y_dineof = SST + (c0 + seasonal) + eps_d. In-situ pins the latent level,
making the DINEOF bias identifiable; the smoothed latent is the debiased, in-situ-scale estimate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.mlemodel import MLEModel

log = logging.getLogger("state_space")

MODELS = ("random_walk", "ar1", "ar1_harmonic", "harmonic_trend")
HARMONIC_MODELS = ("ar1_harmonic", "harmonic_trend")

# Observation type -> (channel, context). Channel F = sst_filled, S = sst_smooth, I = in-situ.
# F and S are DINEOF-derived (and can carry a measurement bias); I is the unbiased in-situ anchor.
TYPE_CHANNEL = {
    "observed": "F", "point_fill": "F", "day_fill": "F",
    "smooth_observed": "S", "smooth_point_fill": "S", "smooth_day_fill": "S",
    "smooth_climatology": "S",
    "insitu": "I",
}
TYPE_CONTEXT = {
    "observed": "observed", "point_fill": "point_fill", "day_fill": "day_fill",
    "smooth_observed": "observed", "smooth_point_fill": "point_fill",
    "smooth_day_fill": "day_fill", "smooth_climatology": "climatology",
    "insitu": "insitu",
}
ALL_TYPES = tuple(TYPE_CHANNEL)
DINEOF_CHANNELS = {"F", "S"}      # channels a DINEOF measurement bias applies to (not in-situ)


# ============================================================ observation assembly

def day_context(observed: np.ndarray, constrained: np.ndarray,
                climatology: np.ndarray) -> np.ndarray:
    """(T,) context label per day: observed | point_fill | day_fill. `climatology` is separate.

    observed    -- the pixel carried a real observation that day (sst_filled_observed).
    point_fill  -- not observed, but the day had its own data (sst_filled_constrained).
    day_fill    -- the day had no data of its own.
    """
    ctx = np.where(observed, "observed",
                   np.where(constrained, "point_fill", "day_fill"))
    return ctx.astype(object)


def _type_available(tp: str, site: dict) -> tuple[np.ndarray, np.ndarray]:
    """(available (T,) bool, value (T,)) for one observation type at one site."""
    ch = TYPE_CHANNEL[tp]
    if ch == "I":                                     # in-situ: available wherever finite, no ctx
        vals = site.get("I")
        if vals is None:
            vals = np.full(len(site["F"]), np.nan)
        vals = np.asarray(vals, float)
        return np.isfinite(vals), vals
    vals = site["S"] if ch == "S" else site["F"]
    need = TYPE_CONTEXT[tp]
    if need == "climatology":
        active = site["climatology"]
    else:
        active = (site["ctx"] == need)
        if ch == "S":
            # A smooth value on a climatology day is `smooth_climatology`, not smooth_<ctx>.
            active = active & ~site["climatology"]
    avail = active & np.isfinite(vals)
    return avail, np.asarray(vals, float)


@dataclass
class Rows:
    """The observation rows for a group: endog plus, per row, its site and per-day type."""
    endog: np.ndarray            # (T, k_endog)
    row_site_pos: np.ndarray     # (k_endog,) site position within the group
    row_type_idx: np.ndarray     # (k_endog, T) index into `types`, -1 where no observation
    types: tuple                 # the obs-variance types, in a fixed order


def assemble_rows(series: list[dict], sources: list[str], mode: str) -> Rows:
    """Build the (T, k_endog) observation matrix and per-row type bookkeeping.

    `series`: one dict per site (in group order) with keys F, S (T,) float, ctx (T,) object,
    climatology (T,) bool. `sources`: type keys; `mode`: 'hierarchy' | 'multiple'.
    """
    types = tuple(sources)
    tix = {t: i for i, t in enumerate(types)}
    T = len(series[0]["F"])
    cols, site_pos, type_idx = [], [], []
    for pos, site in enumerate(series):
        if mode == "hierarchy":
            value = np.full(T, np.nan)
            label = np.full(T, -1, dtype=int)
            filled = np.zeros(T, bool)
            for tp in sources:
                avail, vals = _type_available(tp, site)
                take = avail & ~filled
                value[take] = vals[take]
                label[take] = tix[tp]
                filled |= take
            cols.append(value); site_pos.append(pos); type_idx.append(label)
        else:  # multiple: one row per source
            for tp in sources:
                avail, vals = _type_available(tp, site)
                value = np.where(avail, vals, np.nan)
                label = np.where(avail, tix[tp], -1)
                cols.append(value); site_pos.append(pos); type_idx.append(label)
    return Rows(endog=np.column_stack(cols), row_site_pos=np.asarray(site_pos, int),
                row_type_idx=np.vstack(type_idx), types=types)


def seasonal_design(times, n_harmonics: int, period_days: float) -> np.ndarray:
    """(T, 2H) seasonal regressors [cos1, sin1, ...], NO constant (same convention as
    seasonal_smoothing.design_matrix minus its mean column; the level/mu carries the mean)."""
    t = pd.DatetimeIndex(times)
    d = (np.asarray(t.dayofyear - 1, float)
         + np.asarray((t - t.normalize()).total_seconds(), float) / 86400.0)
    cols = []
    for k in range(1, int(n_harmonics) + 1):
        w = 2.0 * np.pi * k * d / float(period_days)
        cols += [np.cos(w), np.sin(w)]
    return np.stack(cols, axis=1) if cols else np.zeros((len(t), 0))


# ============================================================ parameter bookkeeping

_SHARE_PROC = {"process_var", "process_var_level", "process_var_slope"}
_PHI_MAX = 0.999                 # |phi| ceiling; keeps the AR(1) stationary covariance regular


def _is_shared(local: str, share: set) -> bool:
    if local in _SHARE_PROC:
        return "process_var" in share
    if local == "phi":
        return "phi" in share
    if local == "mu":
        return "mu" in share
    if local.startswith("beta"):
        return "beta" in share
    if local == "bias_const":
        return "bias_const" in share
    if local.startswith("biash"):
        return "bias_seasonal" in share
    if local.startswith("obs_var."):
        return local in share
    return False


def _base_locals(model: str, n_harmonics: int) -> list[tuple[str, str]]:
    """(localname, kind) parameters for one site, excluding obs variances. kind in var|phi|real."""
    if model == "random_walk":
        L = [("process_var", "var")]
    elif model == "ar1":
        L = [("process_var", "var"), ("phi", "phi"), ("mu", "real")]
    elif model == "ar1_harmonic":
        L = [("process_var", "var"), ("phi", "phi"), ("mu", "real")]
        L += [(f"beta{i}", "real") for i in range(2 * n_harmonics)]
    elif model == "harmonic_trend":
        L = [("process_var_level", "var"), ("process_var_slope", "var")]
        L += [(f"beta{i}", "real") for i in range(2 * n_harmonics)]
    else:
        raise ValueError(f"unknown model {model!r}")
    return L


class ParamSpec:
    """Packs/unpacks the flat MLE parameter vector with per-parameter sharing across sites."""

    def __init__(self, model: str, n_harmonics: int, types: tuple, n_sites: int, share: set,
                 dineof_bias: str = "none", bias_n_harmonics: int = 0):
        self.model = model
        self.n_harmonics = int(n_harmonics)
        self.types = tuple(types)
        self.n_sites = int(n_sites)
        self.share = set(share or ())
        self.dineof_bias = dineof_bias
        self.bias_n_harmonics = int(bias_n_harmonics)
        bias_locals = []
        if dineof_bias in ("constant", "seasonal"):
            bias_locals.append(("bias_const", "real"))
            if dineof_bias == "seasonal":
                bias_locals += [(f"biash{i}", "real") for i in range(2 * self.bias_n_harmonics)]
        locals_ = _base_locals(model, self.n_harmonics) + \
            [(f"obs_var.{t}", "var") for t in types] + bias_locals
        self.index: dict = {}
        self.kinds: list[str] = []
        self.names: list[str] = []
        for local, kind in locals_:
            if _is_shared(local, self.share):
                self._add(local, None, kind)
            else:
                for pos in range(self.n_sites):
                    self._add(local, pos, kind)

    def _add(self, local, pos, kind):
        self.index[(local, pos)] = len(self.kinds)
        self.kinds.append(kind)
        self.names.append(local if pos is None else f"{local}[{pos}]")

    @property
    def n_params(self) -> int:
        return len(self.kinds)

    def _slot(self, local: str, pos: int) -> int:
        key = (local, None) if _is_shared(local, self.share) else (local, pos)
        return self.index[key]

    def start(self, mean0: float) -> np.ndarray:
        """Natural-space starting values: variances 0.5, phi 0.3, mu the data mean, beta 0."""
        p = np.zeros(self.n_params)
        for (local, pos), i in self.index.items():
            if self.kinds[i] == "var":
                p[i] = 0.5
            elif self.kinds[i] == "phi":
                p[i] = 0.3
            elif local == "mu":
                p[i] = float(mean0)
        return p

    def transform(self, u: np.ndarray) -> np.ndarray:
        """Unconstrained optimizer space -> natural (positive variances, |phi| < 1)."""
        c = np.array(u, float)
        for i, k in enumerate(self.kinds):
            if k == "var":
                c[i] = np.exp(u[i])
            elif k == "phi":
                c[i] = _PHI_MAX * np.tanh(u[i])     # keep |phi| < 1 so stationary init stays regular
        return c

    def untransform(self, c: np.ndarray) -> np.ndarray:
        u = np.array(c, float)
        for i, k in enumerate(self.kinds):
            if k == "var":
                u[i] = np.log(max(float(c[i]), 1e-12))
            elif k == "phi":
                u[i] = np.arctanh(np.clip(c[i] / _PHI_MAX, -0.999, 0.999))
        return u

    def resolve(self, params: np.ndarray) -> dict:
        """Natural params -> per-site arrays. obsvar is (n_sites, n_types)."""
        n, H = self.n_sites, self.n_harmonics
        out = {"obsvar": np.empty((n, len(self.types)), dtype=np.asarray(params).dtype)}
        for pos in range(n):
            for ti, t in enumerate(self.types):
                out["obsvar"][pos, ti] = params[self._slot(f"obs_var.{t}", pos)]
        if self.model == "harmonic_trend":
            out["q_level"] = np.array([params[self._slot("process_var_level", p)] for p in range(n)])
            out["q_slope"] = np.array([params[self._slot("process_var_slope", p)] for p in range(n)])
        else:
            out["q"] = np.array([params[self._slot("process_var", p)] for p in range(n)])
        if self.model in ("ar1", "ar1_harmonic"):
            out["phi"] = np.array([params[self._slot("phi", p)] for p in range(n)])
            out["mu"] = np.array([params[self._slot("mu", p)] for p in range(n)])
        if self.model in HARMONIC_MODELS:
            out["beta"] = np.array([[params[self._slot(f"beta{i}", p)] for i in range(2 * H)]
                                    for p in range(n)])
        if self.dineof_bias in ("constant", "seasonal"):
            out["bias_const"] = np.array([params[self._slot("bias_const", p)] for p in range(n)])
            if self.dineof_bias == "seasonal":
                Hb = self.bias_n_harmonics
                out["bias_harm"] = np.array(
                    [[params[self._slot(f"biash{i}", p)] for i in range(2 * Hb)]
                     for p in range(n)])
        return out


# ============================================================ the state-space model

class PointStateSpace(MLEModel):
    """A block-diagonal multi-site linear-Gaussian model with per-type, time-varying obs error."""

    def __init__(self, rows: Rows, model: str, n_harmonics: int, design: np.ndarray,
                 spec: ParamSpec, mean0: float, dineof_bias: str = "none",
                 bias_design: np.ndarray | None = None):
        self.model_name = model
        self.spec = spec
        self.n_sites = spec.n_sites
        self.state_dim = 2 if model == "harmonic_trend" else 1
        self.design_seasonal = design                    # (T, 2H)
        self.rows = rows
        self._mean0 = float(mean0)
        self.dineof_bias = dineof_bias
        self.bias_design = (bias_design if bias_design is not None
                            else np.zeros((rows.endog.shape[0], 0)))      # (T, 2Hb)
        # Which (row, day) carry a DINEOF channel (F/S) -> the bias applies there, not to in-situ.
        type_is_dineof = np.array([TYPE_CHANNEL[t] in DINEOF_CHANNELS for t in rows.types])
        idx = rows.row_type_idx
        self.dineof_mask = np.where(idx >= 0, type_is_dineof[idx.clip(min=0)], False)  # (k_endog,T)
        k_states = self.n_sites * self.state_dim
        # The AR(1) state is zero-mean stationary, so stationary init identifies phi (diffuse init
        # cannot tell phi ~ 1 from a random walk). The random-walk / trend states are
        # nonstationary and take approximate-diffuse init.
        init = "stationary" if model in ("ar1", "ar1_harmonic") else "diffuse"
        super().__init__(rows.endog, k_states=k_states, k_posdef=k_states, initialization=init)
        # Constant matrices: design (which state each obs row reads) and an identity selection.
        self["design"] = np.zeros((self.k_endog, k_states))
        for i, pos in enumerate(rows.row_site_pos):
            self["design"][i, pos * self.state_dim] = 1.0
        self["selection"] = np.eye(k_states)

    @property
    def start_params(self) -> np.ndarray:
        return self.spec.start(self._mean0)

    @property
    def param_names(self) -> list:
        return list(self.spec.names)

    def transform_params(self, u):
        return self.spec.transform(np.asarray(u, float))

    def untransform_params(self, c):
        return self.spec.untransform(np.asarray(c, float))

    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)
        R = self.spec.resolve(params)
        n, sd, T = self.n_sites, self.state_dim, self.nobs
        # statsmodels differentiates the likelihood by complex step, so the matrices must take
        # the parameter dtype (float64 normally, complex128 while scoring).
        dt = np.asarray(params).dtype

        transition = np.zeros((self.k_states, self.k_states), dtype=dt)
        state_cov = np.zeros((self.k_states, self.k_states), dtype=dt)
        intercept_site = np.zeros((n, T), dtype=dt)             # per-site obs intercept series
        seas = self.design_seasonal
        for pos in range(n):
            b = pos * sd
            if self.model_name == "random_walk":
                transition[b, b] = 1.0
                state_cov[b, b] = R["q"][pos]
            elif self.model_name in ("ar1", "ar1_harmonic"):
                transition[b, b] = R["phi"][pos]
                state_cov[b, b] = R["q"][pos]
                intercept_site[pos] += R["mu"][pos]
            elif self.model_name == "harmonic_trend":
                transition[b, b] = 1.0; transition[b, b + 1] = 1.0
                transition[b + 1, b + 1] = 1.0
                state_cov[b, b] = R["q_level"][pos]
                state_cov[b + 1, b + 1] = R["q_slope"][pos]
            if self.model_name in HARMONIC_MODELS and seas.shape[1]:
                intercept_site[pos] += seas @ R["beta"][pos]

        self["transition"] = transition
        self["state_cov"] = state_cov

        obs_int = intercept_site[self.rows.row_site_pos]                     # (k_endog, T)
        if self.dineof_bias in ("constant", "seasonal"):
            bias_site = np.zeros((n, T), dtype=dt) + R["bias_const"][:, None]
            if self.dineof_bias == "seasonal" and self.bias_design.shape[1]:
                bias_site = bias_site + R["bias_harm"] @ self.bias_design.T  # (n, T)
            # Add the DINEOF bias to DINEOF rows only (in-situ rows stay unbiased).
            obs_int = obs_int + self.dineof_mask * bias_site[self.rows.row_site_pos]
        self["obs_intercept"] = obs_int

        # Time-varying diagonal observation covariance: each row's variance is its day's type.
        obs_cov = np.zeros((self.k_endog, self.k_endog, T), dtype=dt)
        idx = self.rows.row_type_idx
        for i in range(self.k_endog):
            v = np.where(idx[i] >= 0, R["obsvar"][self.rows.row_site_pos[i], idx[i].clip(min=0)],
                         1.0)
            obs_cov[i, i, :] = v
        self["obs_cov"] = obs_cov

    def smoothed_series(self, res) -> dict:
        """pos -> (mean (T,), sd (T,)): the smoothed latent SST (level + intercept) per site."""
        R = self.spec.resolve(res.params)
        seas = self.design_seasonal
        state = res.smoothed_state                     # (k_states, T)
        cov = res.smoothed_state_cov                   # (k_states, k_states, T)
        out = {}
        for pos in range(self.n_sites):
            li = pos * self.state_dim
            ic = np.zeros(self.nobs)
            if self.model_name in ("ar1", "ar1_harmonic"):
                ic += R["mu"][pos]
            if self.model_name in HARMONIC_MODELS and seas.shape[1]:
                ic += seas @ R["beta"][pos]
            mean = ic + state[li]
            sd = np.sqrt(np.clip(cov[li, li], 0.0, np.inf))
            out[pos] = (mean, sd)
        return out

    def bias_series(self, res) -> dict:
        """pos -> (T,) estimated DINEOF measurement bias (zeros when the group has no bias)."""
        out = {pos: np.zeros(self.nobs) for pos in range(self.n_sites)}
        if self.dineof_bias in ("constant", "seasonal"):
            R = self.spec.resolve(res.params)
            for pos in range(self.n_sites):
                b = np.full(self.nobs, float(R["bias_const"][pos]))
                if self.dineof_bias == "seasonal" and self.bias_design.shape[1]:
                    b = b + self.bias_design @ np.asarray(R["bias_harm"][pos], float)
                out[pos] = b
        return out


# ============================================================ fit one group

def fit_group(times, series: list[dict], gcfg: dict) -> dict:
    """Fit one group's sites jointly. Returns smoothed series, obs rows and the fitted model.

    series: one dict per site (group order) with F, S (T,), ctx (T,) object, climatology (T,) bool.
    gcfg: {model, n_harmonics, mode, sources, share, period_days, fit (optional kwargs)}.
    """
    model = gcfg["model"]
    mode = gcfg.get("mode", "hierarchy")
    sources = list(gcfg["sources"])
    n_harm = int(gcfg.get("n_harmonics", 0))
    period = float(gcfg.get("period_days", 365.25))
    dineof_bias = gcfg.get("dineof_bias", "none")
    bias_H = int(gcfg.get("bias_n_harmonics", 0)) if dineof_bias == "seasonal" else 0
    rows = assemble_rows(series, sources, mode)
    spec = ParamSpec(model, n_harm, rows.types, len(series), set(gcfg.get("share") or ()),
                     dineof_bias=dineof_bias, bias_n_harmonics=bias_H)
    design = (seasonal_design(times, n_harm, period)
              if model in HARMONIC_MODELS else np.zeros((len(times), 0)))
    bias_design = (seasonal_design(times, bias_H, period)
                   if dineof_bias == "seasonal" else np.zeros((len(times), 0)))
    if dineof_bias == "seasonal" and model in HARMONIC_MODELS:
        log.warning("group fits a seasonal DINEOF bias with a seasonal latent model (%s); the two "
                    "seasonal terms can trade off where in-situ is sparse", model)
    finite = rows.endog[np.isfinite(rows.endog)]
    mean0 = float(np.mean(finite)) if finite.size else 0.0

    mod = PointStateSpace(rows, model, n_harm, design, spec, mean0, dineof_bias=dineof_bias,
                          bias_design=bias_design)
    fit_kw = dict(method="lbfgs", maxiter=500, disp=False)
    fit_kw.update(gcfg.get("fit") or {})
    res = mod.fit(**fit_kw)
    smoothed = mod.smoothed_series(res)
    return dict(model=mod, res=res, rows=rows, smoothed=smoothed, spec=spec,
                params=dict(zip(spec.names, np.asarray(res.params, float))),
                llf=float(res.llf), aic=float(res.aic))
