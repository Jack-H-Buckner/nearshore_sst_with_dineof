"""Minimal seasonal_smoothing stub — only the pure functions this repo uses.

The full seasonal_smoothing implementation lives in coastal_sst_data's prototypes;
this file vendors just the functions that the nearshore_sst_with_dineof pipeline
needs so it is self-contained.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


FIT_REFERENCE, FIT_MEAN_ONLY, FIT_FULL = 0, 1, 2


def term_names(n_harmonics):
    return ["mean"] + [f"{f}{k}" for k in range(1, n_harmonics + 1) for f in ("cos", "sin")]


def design_matrix(times, n_harmonics, period_days):
    """(T, 1 + 2H): [1, cos(2 pi k d/P), sin(2 pi k d/P)], d = fractional day of year."""
    t = pd.DatetimeIndex(times)
    d = np.asarray(t.dayofyear - 1, float) + np.asarray(
        (t - t.normalize()).total_seconds(), float) / 86400.0
    cols = [np.ones_like(d)]
    for k in range(1, n_harmonics + 1):
        w = 2.0 * np.pi * k * d / period_days
        cols += [np.cos(w), np.sin(w)]
    return np.stack(cols, axis=1)


def amplitude_phase(coefs, k, period_days):
    """Amplitude and day of year of the peak of harmonic k."""
    a, b = coefs[2 * k - 1], coefs[2 * k]
    amp = np.hypot(a, b)
    peak = (np.arctan2(b, a) / (2.0 * np.pi * k) * period_days) % (period_days / k)
    return amp, peak


def diurnal_design(hours, n_harmonics):
    """(N, 1 + 2K): [1, cos(2 pi k h/24), sin(2 pi k h/24)], h = solar hour."""
    h = np.asarray(hours, float)
    cols = [np.ones_like(h)]
    for k in range(1, n_harmonics + 1):
        w = 2.0 * np.pi * k * h / 24.0
        cols += [np.cos(w), np.sin(w)]
    return np.stack(cols, axis=1)


def offset_terms(coefs, K):
    """'mean=..;cos1=..;sin1=..' — one string column keeps the CSV schema fixed."""
    names = ["mean"] + [f"{f}{k}" for k in range(1, K + 1) for f in ("cos", "sin")]
    return ";".join(f"{n}={c:.4f}" for n, c in zip(names, coefs))
