"""Tests for the point-level state-space smoother (src/state_space.py) and its config.

Parameter-recovery tests follow tests/test_standardize.py: simulate a model with known
parameters on a seeded RNG, fit, and assert recovery within a tolerance tied to the noise floor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import iterative_filter as F          # noqa: E402,F401  (bridges seasonal_smoothing)
import region as R                    # noqa: E402
import state_space as SS              # noqa: E402
import synthetic_cube as SC           # noqa: E402

T = 400
TIMES = pd.date_range("2023-01-01", periods=T, freq="D")


def _site(F_vals, S_vals=None, observed=None, climatology=None):
    obs = np.zeros(T, bool) if observed is None else observed
    ctx = SS.day_context(obs, np.ones(T, bool), np.zeros(T, bool))
    return dict(F=F_vals, S=(np.full(T, np.nan) if S_vals is None else S_vals),
                ctx=ctx, climatology=(np.zeros(T, bool) if climatology is None else climatology))


# --------------------------------------------------------------------------- recovery

def test_random_walk_recovers():
    rng = np.random.default_rng(0)
    q, ov = 0.02, 0.25
    level = np.cumsum(rng.normal(0, np.sqrt(q), T)) + 10.0
    obs = rng.random(T) < 0.6
    Fv = np.where(obs, level + rng.normal(0, np.sqrt(ov), T), np.nan)
    g = dict(model="random_walk", mode="hierarchy", sources=["observed"], share=[])
    r = SS.fit_group(TIMES, [_site(Fv, observed=obs)], g)
    assert 0.5 < r["params"]["obs_var.observed[0]"] / ov < 2.0
    mean, sd = r["smoothed"][0]
    rmse = np.sqrt(np.mean((mean - level) ** 2))
    assert rmse < 0.6 * np.sqrt(ov), f"smoothed rmse {rmse:.3f}"
    assert (sd > 0).all()


def test_ar1_harmonic_recovers():
    # AR(1) phi under measurement noise is high-variance; assert the well-identified quantities
    # (mean, observation variance, smoothed signal) and that positive persistence is detected
    # without pinning to the stationarity boundary.
    rng = np.random.default_rng(1)
    phi, q, mu, ov = 0.7, 0.12, 12.0, 0.05
    d = np.zeros(T)
    for t in range(1, T):
        d[t] = phi * d[t - 1] + rng.normal(0, np.sqrt(q))
    D = SS.seasonal_design(TIMES, 2, 365.25)
    beta = np.array([2.0, 0.5, -1.0, 0.3])
    signal = mu + d + D @ beta
    obs = rng.random(T) < 0.85
    Fv = np.where(obs, signal + rng.normal(0, np.sqrt(ov), T), np.nan)
    g = dict(model="ar1_harmonic", n_harmonics=2, mode="hierarchy", sources=["observed"],
             share=[], period_days=365.25)
    r = SS.fit_group(TIMES, [_site(Fv, observed=obs)], g)
    assert 0.2 < r["params"]["phi[0]"] < 0.999          # persistence detected, not at boundary
    assert abs(r["params"]["mu[0]"] - mu) < 0.5
    assert 0.3 < r["params"]["obs_var.observed[0]"] / ov < 3.0
    mean, _ = r["smoothed"][0]
    assert np.sqrt(np.mean((mean - signal) ** 2)) < 1.5 * np.sqrt(ov)


def test_harmonic_trend_runs_and_smooths():
    rng = np.random.default_rng(2)
    D = SS.seasonal_design(TIMES, 1, 365.25)
    signal = 10 + 0.003 * np.arange(T) + D @ np.array([1.5, -0.5])
    obs = rng.random(T) < 0.6
    Fv = np.where(obs, signal + rng.normal(0, 0.3, T), np.nan)
    g = dict(model="harmonic_trend", n_harmonics=1, mode="hierarchy",
             sources=["observed", "day_fill"], share=[], period_days=365.25)
    r = SS.fit_group(TIMES, [_site(Fv, observed=obs)], g)
    mean, sd = r["smoothed"][0]
    assert np.isfinite(r["llf"])
    assert np.sqrt(np.mean((mean - signal) ** 2)) < 0.5


def test_shared_parameters_pool_across_sites():
    q, ov = 0.03, 0.3

    def mk(seed):
        rr = np.random.default_rng(seed)
        level = np.cumsum(rr.normal(0, np.sqrt(q), T)) + 10.0
        obs = rr.random(T) < 0.45
        return _site(np.where(obs, level + rr.normal(0, np.sqrt(ov), T), np.nan), observed=obs)

    sites = [mk(1), mk(2)]
    shared = SS.fit_group(TIMES, sites, dict(model="random_walk", mode="hierarchy",
                          sources=["observed"], share=["process_var", "obs_var.observed"]))
    indep = SS.fit_group(TIMES, sites, dict(model="random_walk", mode="hierarchy",
                         sources=["observed"], share=[]))
    assert shared["spec"].n_params == 2            # one q + one obs_var for the whole group
    assert indep["spec"].n_params == 4             # per-site q and obs_var
    assert 0.5 < shared["params"]["obs_var.observed"] / ov < 2.0


# --------------------------------------------------------------------------- insitu + bias

def test_insitu_fusion_recovers_dineof_bias():
    """Fuse sparse unbiased in-situ with biased DINEOF: the bias (constant + seasonal) is
    recovered and the smoothed latent is debiased toward the in-situ truth."""
    rng = np.random.default_rng(0)
    t2 = pd.date_range("2022-01-01", periods=730, freq="D")
    doy = t2.dayofyear.to_numpy()
    truth = 10 + 3 * np.sin(2 * np.pi * doy / 365.25) + np.cumsum(rng.normal(0, 0.05, len(t2)))
    c0, a, b = -0.8, 0.6, -0.4
    bias = c0 + a * np.cos(2 * np.pi * doy / 365.25) + b * np.sin(2 * np.pi * doy / 365.25)
    dine = rng.random(len(t2)) < 0.8
    F = np.where(dine, truth + bias + rng.normal(0, 0.3, len(t2)), np.nan)
    ins = rng.random(len(t2)) < 0.25
    I = np.where(ins, truth + rng.normal(0, 0.15, len(t2)), np.nan)
    site = dict(F=F, S=np.full(len(t2), np.nan), ctx=np.array(["observed"] * len(t2), object),
                climatology=np.zeros(len(t2), bool), I=I)
    g = dict(model="random_walk", mode="multiple", sources=["insitu", "observed"],
             dineof_bias="seasonal", bias_n_harmonics=1, period_days=365.25,
             share=["process_var", "obs_var.insitu", "obs_var.observed", "bias_const",
                    "bias_seasonal"])
    r = SS.fit_group(t2, [site], g)
    assert abs(r["params"]["bias_const"] - c0) < 0.2
    amp = np.hypot(r["params"]["biash0"], r["params"]["biash1"])
    assert abs(amp - np.hypot(a, b)) < 0.25
    mean, _ = r["smoothed"][0]
    assert abs(np.mean(mean - truth)) < 0.15                 # debiased
    # A DINEOF-only fit with no bias is biased by ~ the mean offset.
    g0 = dict(model="random_walk", mode="hierarchy", sources=["observed"])
    m0, _ = SS.fit_group(t2, [site], g0)["smoothed"][0]
    assert abs(np.mean(m0 - truth)) > 0.4


def test_assemble_insitu_channel():
    T3 = 4
    site = dict(F=np.array([5.0, 6.0, np.nan, 8.0]), S=np.full(T3, np.nan),
                ctx=np.array(["observed"] * T3, object), climatology=np.zeros(T3, bool),
                I=np.array([np.nan, 2.0, 2.5, np.nan]))
    rows = SS.assemble_rows([site], ["insitu", "observed"], "multiple")
    assert rows.endog.shape == (4, 2)
    insitu_col = list(rows.types).index("insitu")
    assert np.isfinite(rows.endog[:, insitu_col]).tolist() == [False, True, True, False]


# --------------------------------------------------------------------------- assembly

def test_day_context():
    observed = np.array([True, False, False])
    constrained = np.array([True, True, False])
    clim = np.zeros(3, bool)
    ctx = SS.day_context(observed, constrained, clim)
    assert list(ctx) == ["observed", "point_fill", "day_fill"]


def test_assemble_hierarchy_precedence():
    # day 0 observed, day 1 point_fill, day 2 day_fill
    observed = np.array([True, False, False])
    ctx = SS.day_context(observed, np.array([True, True, False]), np.zeros(3, bool))
    site = dict(F=np.array([5.0, 6.0, 7.0]), S=np.array([1.0, 1.0, 1.0]), ctx=ctx,
                climatology=np.zeros(3, bool))
    rows = SS.assemble_rows([site], ["observed", "point_fill", "day_fill"], "hierarchy")
    assert rows.endog.shape == (3, 1)
    assert list(rows.endog[:, 0]) == [5.0, 6.0, 7.0]           # F value each day
    picked = [rows.types[i] if i >= 0 else None for i in rows.row_type_idx[0]]
    assert picked == ["observed", "point_fill", "day_fill"]


def test_assemble_multiple_rows_and_smooth_context():
    observed = np.array([True, False, False])
    ctx = SS.day_context(observed, np.array([True, True, False]), np.zeros(3, bool))
    site = dict(F=np.array([5.0, 6.0, 7.0]), S=np.array([1.0, 2.0, 3.0]), ctx=ctx,
                climatology=np.array([False, False, False]))
    rows = SS.assemble_rows([site], ["observed", "smooth_observed", "smooth_day_fill"], "multiple")
    assert rows.endog.shape == (3, 3)                          # one row per source
    # observed only on day 0; smooth_observed only on day 0; smooth_day_fill only on day 2
    assert np.isfinite(rows.endog[0, 0]) and not np.isfinite(rows.endog[1, 0])
    assert rows.endog[0, 1] == 1.0 and not np.isfinite(rows.endog[1, 1])
    assert rows.endog[2, 2] == 3.0 and not np.isfinite(rows.endog[0, 2])


def test_smooth_climatology_split():
    clim = np.array([False, True])
    ctx = SS.day_context(np.array([False, False]), np.array([False, False]), clim)
    site = dict(F=np.array([np.nan, np.nan]), S=np.array([1.0, 2.0]), ctx=ctx, climatology=clim)
    # smooth_day_fill is available only on the non-climatology day; smooth_climatology on the other
    av_df, _ = SS._type_available("smooth_day_fill", site)
    av_cl, _ = SS._type_available("smooth_climatology", site)
    assert list(av_df) == [True, False]
    assert list(av_cl) == [False, True]


# --------------------------------------------------------------------------- config validation

def _rc(point_filter):
    user = SC.region_user_config("x.zarr", "out", point_filter=point_filter)
    return R.build_region_config(user, resolve=False)


def test_point_filter_config_accepts_both_modes():
    rc = _rc({"points": "p.csv", "groups": {
        "a": {"model": "ar1_harmonic", "n_harmonics": 2, "mode": "hierarchy",
              "sources": ["observed", "point_fill"], "share": ["process_var", "obs_var.observed"]},
        "b": {"model": "ar1", "mode": "multiple",
              "sources": ["observed", "smooth_observed"]}}})
    assert set(rc["point_filter"]["groups"]) == {"a", "b"}


def test_insitu_eval_config_accepts():
    # validates even with no `groups` (the early return must not skip insitu_eval)
    rc = _rc({"insitu_eval": {"enabled": True, "model": "random_walk",
                              "sources": ["observed", "day_fill"], "bias_form": "seasonal",
                              "bias_n_harmonics": 2, "selection": "loso_cv"}})
    assert rc["point_filter"]["insitu_eval"]["enabled"] is True


@pytest.mark.parametrize("ev, match", [
    ({"enabled": True, "model": "random_walk", "sources": ["insitu", "observed"]}, "DINEOF-only"),
    ({"enabled": True, "model": "random_walk", "sources": ["observed"],
      "bias_form": "nope"}, "bias_form"),
    ({"enabled": True, "model": "random_walk", "sources": ["observed"],
      "selection": "nope"}, "selection"),
    ({"enabled": True, "model": "random_walk", "sources": ["observed"], "bogus": 1}, "bogus"),
])
def test_insitu_eval_config_rejects(ev, match):
    with pytest.raises(ValueError, match=match):
        _rc({"insitu_eval": ev})


def test_point_filter_config_accepts_insitu_bias():
    rc = _rc({"points": "p.csv", "groups": {
        "calib": {"model": "random_walk", "mode": "multiple",
                  "sources": ["insitu", "observed", "day_fill"], "dineof_bias": "seasonal",
                  "bias_n_harmonics": 1,
                  "share": ["process_var", "obs_var.insitu", "bias_const", "bias_seasonal"]}}})
    assert rc["point_filter"]["groups"]["calib"]["dineof_bias"] == "seasonal"


@pytest.mark.parametrize("bad, match", [
    ({"groups": {"a": {"model": "ar1", "sources": ["observed"], "bogus": 1}}}, "bogus"),
    ({"groups": {"a": {"model": "nope", "sources": ["observed"]}}}, "model"),
    ({"groups": {"a": {"model": "ar1", "mode": "weird", "sources": ["observed"]}}}, "mode"),
    ({"groups": {"a": {"model": "ar1", "sources": ["not_a_type"]}}}, "unknown type"),
    ({"groups": {"a": {"model": "random_walk", "sources": ["observed"],
                       "share": ["phi"]}}}, "share"),
    ({"groups": {"a": {"model": "ar1_harmonic", "sources": ["observed"]}}}, "n_harmonics"),
    # bias needs insitu + a DINEOF source
    ({"groups": {"a": {"model": "random_walk", "mode": "multiple", "sources": ["observed"],
                       "dineof_bias": "constant"}}}, "insitu"),
    # bias requires mode: multiple
    ({"groups": {"a": {"model": "random_walk", "mode": "hierarchy",
                       "sources": ["insitu", "observed"], "dineof_bias": "constant"}}}, "multiple"),
    # seasonal bias needs bias_n_harmonics
    ({"groups": {"a": {"model": "random_walk", "mode": "multiple",
                       "sources": ["insitu", "observed"], "dineof_bias": "seasonal"}}},
     "bias_n_harmonics"),
    # bias_seasonal share token invalid when bias is constant
    ({"groups": {"a": {"model": "random_walk", "mode": "multiple",
                       "sources": ["insitu", "observed"], "dineof_bias": "constant",
                       "share": ["bias_seasonal"]}}}, "share"),
])
def test_point_filter_config_rejects(bad, match):
    with pytest.raises(ValueError, match=match):
        _rc(bad)
