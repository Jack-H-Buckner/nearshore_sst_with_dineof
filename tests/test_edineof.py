"""Correctness tests for the eDINEOF algorithm.

The eight checks from `plans/eDINEOF_description.md` §10, plus four more that catch bugs those
eight pass over. Every test runs on small synthetic matrices; nothing here touches the cube.

    pytest prototypes/DINEOF/tests -q

(The repo's pyproject sets `testpaths = ["tests"]`, so this directory must be named explicitly.)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import edineof as E  # noqa: E402


# --------------------------------------------------------------------------- helpers

def uniform_t(n=40):
    return np.arange(n, dtype=float)


def irregular_t(n=40, seed=3):
    """Strictly increasing, uneven spacing -- the case the plain-Euclidean adjoint gets wrong."""
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.uniform(0.5, 4.0, size=n))


def random_X(m=80, n=20, seed=0):
    return np.random.default_rng(seed).normal(size=(m, n))


def dtmid(t):
    """Cell widths, the weights the non-uniform Laplacian is self-adjoint under."""
    n = t.size
    dt = np.diff(t)
    tm = np.empty(n + 1)
    tm[1:n] = (t[:-1] + t[1:]) / 2
    tm[0] = t[0] - dt[0] / 2
    tm[n] = t[-1] + dt[-1] / 2
    return np.diff(tm)


# --------------------------------------------------------------------------- doc §10

def test_1_exact_svd_no_filter():
    """Filter off, full rank: the factorization must reproduce X and match numpy's SVD."""
    X = random_X(80, 20)
    t = uniform_t(20)
    U, s, V = E.top_k_modes(X, 20, t, 0.0, 0, use_filter=False)
    assert np.allclose(E.reconstruct(U, s, V), X, atol=1e-10)
    assert np.allclose(np.sort(s), np.sort(np.linalg.svd(X, compute_uv=False)), rtol=1e-10)


def test_2_orthogonality():
    """Filter off: both bases orthonormal. Filter ON: V stays, U does NOT.

    The second half is asserted as an INEQUALITY on purpose. With the filter, V diagonalizes
    F'X'XF rather than X'X, so the spatial modes are unit-norm but not orthogonal. That is a
    documented property of the method, not a defect -- and writing it as a test means a future
    reader who "fixes" it by orthogonalizing U will get a failure instead of silent drift.
    """
    X = random_X(80, 20)
    t = uniform_t(20)
    I = np.eye(20)

    U, s, V = E.top_k_modes(X, 20, t, 0.0, 0, use_filter=False)
    assert np.allclose(U.T @ U, I, atol=1e-12)
    assert np.allclose(V.T @ V, I, atol=1e-12)

    U, s, V = E.top_k_modes(X, 20, t, 0.25, 10, use_filter=True)
    assert np.allclose(V.T @ V, I, atol=1e-12)
    assert np.abs(U.T @ U - I).max() > 1e-6


def test_3_eigenvalues_both_conventions():
    X = random_X(60, 15)
    t = uniform_t(15)
    lam = np.sort(np.linalg.eigvalsh(X.T @ X))[::-1][:8]
    for conv in ("projection", "reference"):
        _, s, _ = E.top_k_modes(X, 8, t, 0.0, 0, use_filter=False, convention=conv)
        assert np.allclose(s ** 2, lam, rtol=1e-10)


def test_4_filter_symmetry():
    t = irregular_t(30)
    X = random_X(50, 30)
    B = E.filter_covariance(t, X.T @ X, 0.1, 7)
    assert np.array_equal(B, B.T), "symmetrization must be exact, not approximate"


def test_5_filter_preserves_a_constant():
    """Zero-flux boundaries: no mass leaks off either end, so a constant is a fixed point."""
    t = irregular_t(50)
    out = E.filter_time(t, np.ones(50), 0.1, 40)
    assert np.allclose(out, 1.0, atol=1e-12)


def test_6_stability_guard():
    t = uniform_t(20)                       # min(dt) = 1, so the ceiling is `factor`
    E.check_stability(t, 0.24, 10, 0.25)    # under: fine
    with pytest.raises(ValueError, match="exceeds"):
        E.check_stability(t, 0.26, 10, 0.25)
    # The filter being off is never unstable, whatever alpha says.
    E.check_stability(t, 10.0, 0, 0.25)
    E.check_stability(t, 0.0, 10, 0.25)


def test_7_synthetic_rank5_recovery():
    """Build a known rank-5 matrix, punch out 40%, confirm both the fill and the rank.

    `day_frac: 0` because this is the doc's §10 check, which has ONE cross-validation set. The
    day holdout is this implementation's addition and is meaningless here: with the filter off
    a whole held-out column is structurally unrecoverable, so it would only add error that has
    nothing to do with the rank being measured.

    A little NOISE is added, and it is what makes "CV selects k = 5" a well-posed claim. On an
    exactly rank-5 matrix every k >= 5 reconstructs it exactly -- measured CV error 0.0015 at
    k=5 and 0.0000 at k=6,7,8 -- so nothing in the data distinguishes rank 5 from rank 11 and
    any answer in that range is defensible. With a noise floor the curve has a real minimum:
    k<5 underfits, k>5 fits noise, and 5 wins by a margin. That is the situation the rule is
    for, and the situation real data is always in.
    """
    rng = np.random.default_rng(1)
    m, n, r = 400, 60, 5
    U = np.linalg.qr(rng.normal(size=(m, r)))[0]
    V = np.linalg.qr(rng.normal(size=(n, r)))[0]
    s = np.array([50.0, 30.0, 18.0, 11.0, 6.0])
    X_clean = U @ np.diag(s) @ V.T
    # 0.01 is ~2.5% of the signal sd and ~26% of mode 5's per-entry amplitude, so the fifth
    # mode stays comfortably identifiable.
    X_true = X_clean + np.random.default_rng(7).normal(scale=0.01, size=X_clean.shape)
    observed = rng.random((m, n)) > 0.40
    t = uniform_t(n)

    cfg = {"filter": {"t_c_grid": [0], "alpha_max": 0.25, "stability_factor": 0.25},
           "modes": {"k_grid": list(range(1, 13)), "rule": "parsimonious", "rel_tol": 0.01,
                     "patience": 3, "sigma_convention": "projection", "fix_sign": True},
           "em": {"tol": 1e-8, "max_iter": 400},
           "cv": {"frac": 0.05, "day_frac": 0.0, "max_date_frac": 0.4,
                  "min_date_obs_after": 20, "donor_lo": 0.5, "donor_hi": 0.95, "seed": 0}}
    # edineof now takes its point holdout as an explicit (n, m) mask rather than drawing one
    # from cv.frac; 5% of the observed entries, scattered, stands in for the old draw.
    valid_msk = ((np.random.default_rng(3).random((m, n)) < 0.05) & observed).T
    res = E.edineof(X_true.copy(), observed, valid_msk, t, cfg)

    # Scored against the CLEAN signal: recovering the noise would be overfitting, not success.
    gaps = ~observed
    err = np.sqrt(np.mean((res["X"][gaps] - X_clean[gaps]) ** 2))
    assert err < 0.05 * X_clean.std(), f"gap RMSE {err:.4f} vs sd {X_clean.std():.4f}"
    assert res["k_opt"] == r, f"selected k={res['k_opt']}, expected {r}"


def test_8_delta_decreases():
    """The EM step size should fall monotonically; oscillation means alpha is over the limit."""
    rng = np.random.default_rng(2)
    m, n = 200, 40
    X_true = (np.linalg.qr(rng.normal(size=(m, 4)))[0] * [20, 12, 7, 4]) @ \
        np.linalg.qr(rng.normal(size=(n, 4)))[0].T
    observed = rng.random((m, n)) > 0.35
    X = np.where(observed, X_true, 0.0)
    gaps = ~observed
    _, hist = E.fill(X, gaps, 4, uniform_t(n), 0.25, 5, 1e-8, 60, float(X_true.std()))
    d = hist["deltas"]
    assert all(d[i + 1] <= d[i] * 1.05 for i in range(1, len(d) - 1)), d[:12]


# --------------------------------------------------------------------------- four more

def test_9_filter_conserves_mass_on_irregular_time():
    """Sum(M * dtmid) is invariant.

    Check 5 (a constant stays constant) passes even if `dt` and `dtmid` are swapped in the
    flux, because on that input both divisions cancel. This one does not: it is what actually
    pins the non-uniform discretization.
    """
    t = irregular_t(45, seed=11)
    w = dtmid(t)
    M = np.random.default_rng(5).normal(size=(7, 45))
    out = E.filter_time(t, M, 0.1, 12)
    assert np.allclose(out @ w, M @ w, rtol=1e-12, atol=1e-12)


def test_10_filter_self_adjoint_under_cell_widths():
    """<Fa, b>_w == <a, Fb>_w with w = dtmid.

    The non-uniform Laplacian is self-adjoint under the cell-width-weighted inner product, not
    the plain Euclidean one -- exactly the trap the doc's §9 warns about for anyone going
    matrix-free. Worth pinning even though this implementation does not.
    """
    t = irregular_t(40, seed=7)
    w = dtmid(t)
    rng = np.random.default_rng(9)
    a, b = rng.normal(size=40), rng.normal(size=40)
    lhs = E.filter_time(t, a, 0.1, 9) @ (b * w)
    rhs = a @ (E.filter_time(t, b, 0.1, 9) * w)
    assert np.isclose(lhs, rhs, rtol=1e-11, atol=1e-11)


def test_11_empty_column_needs_the_filter():
    """An all-gap column is exactly 0 without the filter, and populated with it.

    This is the property the whole design rests on: 180 of the real cube's 365 days are empty,
    and plain DINEOF returns the mean for every one of them.
    """
    rng = np.random.default_rng(4)
    m, n, j = 150, 31, 15
    X_true = (np.linalg.qr(rng.normal(size=(m, 3)))[0] * [15, 9, 5]) @ \
        np.linalg.qr(rng.normal(size=(n, 3)))[0].T
    observed = rng.random((m, n)) > 0.3
    observed[:, j] = False                              # day j sees nothing at all
    t = uniform_t(n)
    gaps = ~observed

    X = np.where(observed, X_true, 0.0)
    E.fill(X, gaps, 3, t, 0.0, 0, 1e-8, 50, float(X_true.std()))
    assert np.abs(X[:, j]).max() < 1e-14, "without the filter an empty day must stay at 0"

    X = np.where(observed, X_true, 0.0)
    E.fill(X, gaps, 3, t, 0.25, 10, 1e-8, 50, float(X_true.std()))
    assert np.linalg.norm(X[:, j]) > 0, "with the filter it must be populated"
    # And it must interpolate rather than overshoot: bounded by its neighbours' range.
    lo = min(X[:, j - 1].min(), X[:, j + 1].min())
    hi = max(X[:, j - 1].max(), X[:, j + 1].max())
    span = hi - lo
    assert X[:, j].min() >= lo - 0.25 * span and X[:, j].max() <= hi + 0.25 * span


def test_12_kernel_width_and_the_parity_ceiling():
    """The kernel sd is sqrt(2*alpha*p), and alpha = min(dt)^2/2 is a parity comb.

    Pins the measured finding behind `check_stability`'s factor of 1/4, so nobody raises the
    default back to the doc's stated ceiling without this failing.
    """
    n = 201
    t = uniform_t(n)
    imp = np.zeros(n)
    imp[100] = 1.0

    for alpha, p in ((0.05, 20), (0.1, 32), (0.25, 8)):
        k = E.filter_time(t, imp, alpha, p)
        sd = np.sqrt(((t - 100) ** 2 * k).sum() / k.sum())
        assert np.isclose(sd, np.sqrt(2 * alpha * p), rtol=0.01)

    comb = E.filter_time(t, imp, 0.5, 10)               # exactly min(dt)^2 / 2
    assert np.abs(comb[101]) < 1e-15, "at alpha=1/2 every odd offset must vanish"
    assert comb[102] > 0.1, "...while even offsets carry all the weight"
