"""Tests for the seasonal climatology fit, focused on the GMRF spatial-smoothing method.

The central claim: a pixel with no data borrows its seasonal cycle from its spatial NEIGHBOURS
(GMRF), not from one global average cycle (the old `harmonic` fallback).

    pytest tests/test_standardize.py -q
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

import standardize as S            # noqa: E402
from seasonal_smoothing import FIT_FULL, FIT_REFERENCE, design_matrix   # noqa: E402

H = W = 24
T = 60
LEFT, RIGHT = slice(0, 12), slice(12, W)          # two regions with opposite cycles
BLOCK = (slice(8, 16), slice(16, 22))             # an empty hole, well inside the RIGHT region


def _seasonal(**method):
    return {"n_harmonics": 1, "period_days": 365.25, "min_dates": 10, "max_cond": 100.0,
            "method": "harmonic", "gmrf_range_px": 8.0, "gmrf_alpha": 2, "gmrf_min_sd": 1e-3,
            **method}


SCALE_CFG = {"estimator": "mad", "floor": 0.5, "dof_correction": True, "fallback": "median"}


def _make_data(seed=0):
    """Left region: mean 10, cos1 +2. Right region: mean 14, cos1 -2. A block in the right has
    no observations at all. Returns (Y (T,N), O (T,N), X (T,P), water, coef_true (P,H,W))."""
    rng = np.random.default_rng(seed)
    times = pd.to_datetime("2023-01-01") + pd.to_timedelta(
        np.linspace(0, 364, T).astype(int), unit="D")
    X = design_matrix(times, 1, 365.25)                       # (T, 3): [1, cos1, sin1]

    coef_true = np.zeros((3, H, W))
    coef_true[0, :, LEFT], coef_true[0, :, RIGHT] = 10.0, 14.0
    coef_true[1, :, LEFT], coef_true[1, :, RIGHT] = 2.0, -2.0
    truth = np.tensordot(X, coef_true.reshape(3, -1), axes=(1, 0)).reshape(T, H, W)
    truth = truth + rng.normal(scale=0.05, size=truth.shape)

    O3 = np.ones((T, H, W), bool)
    O3[:, BLOCK[0], BLOCK[1]] = False                         # the hole: never observed
    water = np.ones((H, W), bool)
    Y = np.where(O3, truth, 0.0).reshape(T, -1)
    O = O3.reshape(T, -1)
    return Y, O, X, water, coef_true


def _block_cols():
    m = np.zeros((H, W), bool)
    m[BLOCK[0], BLOCK[1]] = True
    return np.flatnonzero(m.ravel())


def test_gmrf_borrows_cycle_from_neighbours():
    """The empty block's recovered cycle matches the RIGHT region (its neighbours), not the
    global mean. The old harmonic fallback gives it the global (near-zero cos1, mean ~12)."""
    Y, O, X, water, _ = _make_data()
    block = _block_cols()

    coef_h, _, _, ft_h, _ = S.fit_seasonal_coeffs(Y, O, X, water, _seasonal(), SCALE_CFG)
    coef_g, _, _, ft_g, _ = S.fit_seasonal_coeffs(
        Y, O, X, water, _seasonal(method="gmrf"), SCALE_CFG)

    cos_h, cos_g = coef_h[1, block].mean(), coef_g[1, block].mean()
    mean_h, mean_g = coef_h[0, block].mean(), coef_g[0, block].mean()

    # harmonic fallback: the global reference cycle -- cos1 ~ 0, mean ~ 12 (basin average)
    assert abs(cos_h) < 0.5, cos_h
    assert abs(mean_h - 12.0) < 1.0, mean_h
    # gmrf: borrows the RIGHT region's cycle (cos1 ~ -2, mean ~ 14)
    assert cos_g < -1.0, cos_g
    assert mean_g > 13.0, mean_g
    # and it is decisively more local than the global fallback
    assert cos_g < cos_h - 1.0 and mean_g > mean_h + 1.0


def test_gmrf_keeps_contract_and_fit_type_codes():
    """Shapes, the -1/0/1/2 fit_type codes, and the FULL pixels staying near their own fit."""
    Y, O, X, water, coef_true = _make_data()
    block = _block_cols()
    P, N = 3, water.sum()

    coef_g, scale_g, mad_g, ft_g, info = S.fit_seasonal_coeffs(
        Y, O, X, water, _seasonal(method="gmrf"), SCALE_CFG)

    assert coef_g.shape == (P, N)
    assert scale_g.shape == (N,) and ft_g.shape == (N,)
    # the hole has no data -> REFERENCE; everywhere else is a FULL own-fit
    assert np.all(ft_g[block] == FIT_REFERENCE)
    observed = np.setdiff1d(np.arange(N), block)
    assert np.all(ft_g[observed] == FIT_FULL)
    # a FULL right-region pixel keeps its own cos1 near the truth (-2), lightly smoothed
    right = np.flatnonzero(np.tile(np.arange(W) >= 12, H))
    right_obs = np.intersect1d(right, observed)
    assert abs(coef_g[1, right_obs].mean() - (-2.0)) < 0.3


def test_harmonic_method_matches_plain_fit():
    """method='harmonic' must reproduce the bare fit_harmonics_gappy coefficients exactly."""
    Y, O, X, water, _ = _make_data()
    coef_ref, info = S.fit_harmonics_gappy(Y, O, X, min_dates=10, max_cond=100.0)
    coef_h, _, _, ft_h, _ = S.fit_seasonal_coeffs(Y, O, X, water, _seasonal(), SCALE_CFG)
    assert np.allclose(coef_h, coef_ref, equal_nan=True)
    assert np.array_equal(ft_h, info["fit_type"])
