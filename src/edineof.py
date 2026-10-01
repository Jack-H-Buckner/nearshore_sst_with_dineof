"""eDINEOF: fill the gaps by repeated truncated SVD, with a filter on the temporal covariance.

Stage 6, and the last one. Implements `prototypes/DINEOF/plans/eDINEOF_description.md` against
the standardized cube from `standardize.py`, then inverts the standardization to produce a
gap-free daily SST field.

DINEOF is EM for a low-rank model: fit a rank-k SVD, use it to refill the missing entries,
repeat. eDINEOF adds one thing -- a diffusion filter applied to B = X'X before the
eigendecomposition -- and that one thing is doing nearly all the work here. 180 of these 365
days have NO data at all. For an all-gap column j, B[:,j] = 0, so V[j,:] = 0, so the
reconstruction is exactly 0 forever: plain DINEOF cannot populate an empty day, it can only
return the mean. The filter couples column j to its temporal neighbours and is the only reason
those 180 days come back as anything.

THE TWO CROSS-VALIDATION SETS ANSWER DIFFERENT QUESTIONS, and that is why there are two.

  POINT holdout (the doc's, §6) -- scattered points inside days that still have data, using a
    real gap mask pasted from another day. Selects k. It is the standard DINEOF criterion and
    it measures spatial rank: how well the mode basis fills a cloud.
  DAY holdout -- 10% of the days with data, held out whole. Selects p. It has to be separate,
    because the point holdout is STRUCTURALLY BLIND to what the filter does: a point-holdout
    pixel sits in a day whose V is already pinned by its own surviving pixels, and filtering
    can only pull that V toward its neighbours, i.e. distort it. Scored on points alone, the
    optimum is always p = 0 and eDINEOF silently degrades to DINEOF. A whole held-out day has
    no pixels of its own, so its reconstruction comes entirely from the filter -- the same
    situation as the 180 empty days, which is the thing actually being validated.

    p = 0 is therefore a free and very informative baseline: with no filter, a held-out day
    reconstructs to exactly 0 in z, which un-standardizes to exactly its seasonal climatology.
    So the day-CV curve measures precisely what the filter buys over the climatology that
    stage 5 removed. If it buys nothing, the daily product is climatology wearing a hat.

THREE DEPARTURES FROM THE SPEC DOC, each because it was measured rather than assumed:

  1. `eigsh` -> dense `scipy.linalg.eigh`. At n = 365 the full symmetric decomposition takes
     0.021 s, FASTER than eigsh(k=30), and it is deterministic, needs no v0 warm start, cannot
     fail to converge, and is immune to the near-degenerate-eigenvalue trouble §9 warns about.
     The doc's advice targets n in the tens of thousands.
  2. `reconstruct_at`'s fancy-index form allocates 3.8 GB here: gaps are 78% of the matrix, so
     U[rows] materializes (1.6e7, k) float64. One dense gemm is 0.115 s and 162 MB.
  3. The stability ceiling is alpha <= min(dt)^2 / 4, not / 2. See `check_stability`.

Usage (from the repo root, in the `coastal_sst_data` env):

    python prototypes/DINEOF/src/edineof.py \\
        --config prototypes/DINEOF/configs/config.edineof.admiralty_inlet.yaml
    ... --dry-run        # build the matrix, report its shape and the CV split, stop
    ... --k 12 --p 8     # skip the search, run one combination
    ... --no-figures
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[1]              # prototypes/DINEOF
DEFAULT_CONFIG = ROOT / "configs" / "config.edineof.admiralty_inlet.yaml"

_CMM_SRC = ROOT.parent / "cloud_mixture_model" / "src"
if str(_CMM_SRC) not in sys.path:
    sys.path.insert(1, str(_CMM_SRC))

from seasonal_smoothing import design_matrix             # noqa: E402

import edineof_figures                                   # noqa: E402  (DINEOF's own src/)
import plotting                                          # noqa: E402

from coastal_sst_data import provenance, store           # noqa: E402
from coastal_sst_data.config import CompressionSpec      # noqa: E402
from coastal_sst_data.processes import datacube          # noqa: E402

log = logging.getLogger("edineof")


# ==================================================================== the algorithm
# Pure numpy from here to `select_matrix`: no I/O, no config objects, so the tests can drive
# these directly on synthetic matrices.

def check_stability(t: np.ndarray, alpha: float, p: int, factor: float = 0.25) -> None:
    """Raise unless alpha <= factor * min(dt)^2. No-op when the filter is off.

    THE DOC'S CEILING IS WRONG BY A FACTOR OF TWO and it matters. It gives
    alpha <= min(dt)^2 / 2, the limit of numerical stability. But the p-step operator on a
    uniform grid is (1 - 2a + a(S + S^-1))^p with Fourier symbol 1 - 4a sin^2(k/2), and at
    a = 1/2 exactly:

      - the self term (1 - 2a) vanishes, so only EVEN offsets survive. Measured at p=10:
        w(+-1) = 0.00000, w(+-2) = 0.833, support 11 instead of 21. The kernel is a parity
        comb and half the days receive no weight at all.
      - the symbol equals -1 at Nyquist, so the highest-frequency mode does not decay, it
        flips sign every iteration forever.

    At a <= 1/4 the symbol is cos^2(k/2) >= 0 and monotone, which is what a smoother should be.
    """
    if not alpha or not p:
        return
    dt = np.diff(np.asarray(t, float))
    if dt.size == 0:
        return
    ceiling = float(factor) * float(dt.min()) ** 2
    if alpha > ceiling:
        raise ValueError(
            f"alpha = {alpha:g} exceeds {factor:g} * min(dt)^2 = {ceiling:g}. Above this the "
            "diffusion stops being a smoother: at min(dt)^2/2 the kernel drops every odd "
            "offset and the Nyquist mode oscillates forever instead of decaying. Lower alpha "
            "and raise p -- the cutoff period 2*pi*sqrt(alpha*p) depends only on the product.")


def filter_time(t: np.ndarray, M: np.ndarray, alpha: float, p: int) -> np.ndarray:
    """Forward-Euler diffusion along the LAST axis of M, on possibly irregular samples `t`.

    Doc §3. Differences are divided by the real time increment, which is the whole point: two
    columns adjacent in the matrix but three weeks apart in the calendar couple weakly. Ghost
    points at both ends carry zero flux, so nothing leaks out and a constant is preserved.
    """
    if not alpha or not p:
        return np.array(M, dtype=float, copy=True)
    t = np.asarray(t, float)
    n = t.size
    dt = np.diff(t)

    tmid = np.empty(n + 1)
    tmid[1:n] = (t[:-1] + t[1:]) / 2.0
    tmid[0] = t[0] - dt[0] / 2.0
    tmid[n] = t[-1] + dt[-1] / 2.0
    dtmid = np.diff(tmid)

    M = np.array(M, dtype=float, copy=True)
    G = np.zeros(M.shape[:-1] + (n + 1,))
    for _ in range(int(p)):
        G[...] = 0.0
        G[..., 1:n] = alpha * np.diff(M, axis=-1) / dt
        M = M + np.diff(G, axis=-1) / dtmid
    return M


def filter_covariance(t: np.ndarray, B: np.ndarray, alpha: float, p: int) -> np.ndarray:
    """B~ = F' B F, applied as two passes rather than by ever forming the (n, n) matrix F.

    Row i of B is "how much image i resembles image 1, 2, ... n", indexed by time. Smoothing it
    asserts that that resemblance varies smoothly as j walks the calendar -- so an image with
    almost no data, whose row is nearly empty, inherits structure from its temporal neighbours.
    That is the entire mechanism.

    The final symmetrization is not cosmetic: the eigensolver requires an exactly symmetric
    matrix and two floating-point passes will not produce one.
    """
    B = filter_time(t, B, alpha, p)
    B = filter_time(t, B.T, alpha, p).T
    return (B + B.T) / 2.0


def top_k_modes(X: np.ndarray, k: int, t: np.ndarray, alpha: float, p: int,
                use_filter: bool, *, convention: str = "projection", fix_sign: bool = True):
    """(U, sigma, V) for the leading k modes. Doc §4.

    The temporal modes come from the FILTERED covariance; U comes from projecting the
    UNFILTERED X onto them. The filter constrains when things happen, never what the spatial
    patterns look like.

    Dense `eigh` rather than `eigsh`: see the module docstring. It returns all n eigenpairs
    ascending, so take the last k and flip.

    Conventions (identical when the filter is off):
      projection  sigma = ||X V||, so U diag(sigma) V' is exactly the orthogonal projection of
                  X onto span(V). Least-squares optimal given the basis, and the default.
      reference   sigma = sqrt(lambda), what the Fortran does. Diffusion reduces variance, so
                  sqrt(lambda) <= ||X V|| and mode amplitudes come out systematically damped.
    """
    B = X.T @ X
    B = filter_covariance(t, B, alpha, p) if use_filter else (B + B.T) / 2.0

    lam, V = scipy.linalg.eigh(B)
    lam = np.maximum(lam[::-1][:k], 0.0)
    V = np.ascontiguousarray(V[:, ::-1][:, :k])

    W = X @ V                                            # (m, k)
    norms = np.linalg.norm(W, axis=0)
    safe = np.where(norms > 0, norms, 1.0)
    U = W / safe
    sigma = norms if convention == "projection" else np.sqrt(lam)

    if fix_sign:
        # Eigenvector signs are arbitrary and flip between runs; U flips with V so the
        # reconstruction is invariant, but modes are only comparable across runs if pinned.
        flip = V[np.argmax(np.abs(V), axis=0), np.arange(V.shape[1])] < 0
        V = np.where(flip, -V, V)
        U = np.where(flip, -U, U)
    return U, sigma, V


def reconstruct(U: np.ndarray, sigma: np.ndarray, V: np.ndarray) -> np.ndarray:
    """The full (m, n) rank-k product.

    One gemm rather than the doc's `einsum('ik,k,ik->i', U[rows], sigma, V[cols])`: gaps are
    78% of this matrix, so the fancy-index form materializes a (1.6e7, k) float64 temporary --
    3.8 GB at k=30 -- to avoid forming a 162 MB dense result. The gemm is also ~3x faster.
    """
    return U @ (sigma[:, None] * V.T)


class Workspace:
    """Preallocated buffers and the INDEX form of the gap mask, built once per gap layout.

    Boolean masking is what made the original inner loop slow: measured on this matrix,
    `R[gaps]` costs 0.169 s and `X[gaps] = ...` 0.137 s, against 0.023 s for a memcpy and
    0.024 s for an integer gather. The doc's §5 pseudocode uses four boolean ops per iteration.

    Indexing the OBSERVED set rather than the gaps is the other half: gaps are 80% of the
    matrix (16.3M entries) but the observed set is only 4.0M, so every correction term here is
    computed over the smaller side.
    """

    __slots__ = ("R", "D", "prev", "obs_idx", "obs_val", "n_gaps",
                 "empty_col", "n_gap_empty", "n_gap_seen")

    def __init__(self, X: np.ndarray, gaps: np.ndarray):
        self.R = np.empty_like(X)
        self.D = np.empty_like(X)
        self.prev = np.empty_like(X)          # previous k's solution, for the warm-start metric
        self.obs_idx = np.flatnonzero(~gaps.reshape(-1))
        # Observed entries are held fixed for the life of the fill, so one copy is enough.
        self.obs_val = X.reshape(-1)[self.obs_idx].copy()
        self.n_gaps = int(X.size - self.obs_idx.size)

        # Columns with NO observation at all -- either genuinely empty dates or days the CV
        # held out whole. They are filled entirely by the temporal filter diffusing in from
        # neighbours, which is a completely different convergence problem from a date that has
        # pixels of its own, so the two are tracked separately.
        self.empty_col = np.asarray(gaps.all(axis=0))
        self.n_gap_empty = int(X.shape[0] * self.empty_col.sum())
        self.n_gap_seen = int(self.n_gaps - self.n_gap_empty)

    def gap_rms(self, A: np.ndarray, B: np.ndarray) -> float:
        """RMS of (A - B) over the gap entries only, with no boolean selection.

        Taken over everything and corrected by the observed part, which is the 4x smaller side.
        """
        np.subtract(A, B, out=self.D)
        Df = self.D.reshape(-1)
        ss = float(np.dot(Df, Df)) - float(np.dot(Df[self.obs_idx], Df[self.obs_idx]))
        return float(np.sqrt(max(ss, 0.0) / self.n_gaps)) if self.n_gaps else 0.0


def fill(X: np.ndarray, gaps: np.ndarray, k: int, t: np.ndarray, alpha: float, p: int,
         tol: float, max_iter: int, sd: float, *, label: str = "",
         work: Workspace | None = None) -> tuple[np.ndarray, dict]:
    """EM at fixed k: fit modes, refill gaps, repeat until the filled values stop moving.

    Observed entries are held fixed and never updated -- there is no observation-error
    weighting in DINEOF, a pixel is present with weight 1 or absent with weight 0.

    `X` is modified in place and also returned. `delta` is the RMS change in the filled values
    relative to `sd`, and should decrease monotonically; oscillation means alpha is above the
    stability limit.
    """
    w = work if work is not None else Workspace(X, gaps)
    R, D, obs_idx, obs_val, n_gaps = w.R, w.D, w.obs_idx, w.obs_val, w.n_gaps
    Xf, Df = X.reshape(-1), D.reshape(-1)
    ecol = w.empty_col
    deltas, d_empty, d_seen = [], [], []
    d_abs, d_abs_empty, d_abs_seen = [], [], []
    norm = sd if sd > 0 else 1.0
    for it in range(int(max_iter)):
        U, sigma, V = top_k_modes(X, k, t, alpha, p, use_filter=True)
        np.matmul(U, sigma[:, None] * V.T, out=R)

        # ||R - X||^2 restricted to the gaps, without ever forming a boolean selection:
        # take it over the whole matrix, then subtract back the observed part, which is the
        # 4x smaller side. `X` equals `obs_val` at observed entries by construction, so the
        # correction is exact.
        np.subtract(R, X, out=D)
        obs_ss = float(np.dot(Df[obs_idx], Df[obs_idx]))
        col_ss = np.einsum("ij,ij->j", D, D)        # per-date sum of squares, one pass
        ss = float(col_ss.sum()) - obs_ss
        # Split it: dates the EM sees no pixel of (filter-only) against dates that have some.
        ss_empty = float(col_ss[ecol].sum())
        ss_seen = max(ss - ss_empty, 0.0)

        # Mean |change| per gap pixel -- the physical reading of the same quantity -- split the
        # same way. `D` is taken absolute IN PLACE, which is safe because the squares above are
        # already extracted and the next iteration overwrites it; that avoids a second
        # full-size temporary purely for diagnostics.
        np.abs(D, out=D)
        col_abs = D.sum(axis=0)
        abs_gap = float(col_abs.sum()) - float(Df[obs_idx].sum())
        abs_empty = float(col_abs[ecol].sum())
        abs_seen = max(abs_gap - abs_empty, 0.0)

        # Write the new iterate: memcpy everywhere, then restore the observed entries.
        np.copyto(X, R)
        Xf[obs_idx] = obs_val

        delta = float(np.sqrt(max(ss, 0.0) / n_gaps) / norm) if n_gaps else 0.0
        deltas.append(delta)
        d_empty.append(float(np.sqrt(ss_empty / w.n_gap_empty) / norm)
                       if w.n_gap_empty else float("nan"))
        d_seen.append(float(np.sqrt(ss_seen / w.n_gap_seen) / norm)
                      if w.n_gap_seen else float("nan"))
        d_abs.append(abs_gap / n_gaps if n_gaps else 0.0)
        d_abs_empty.append(abs_empty / w.n_gap_empty if w.n_gap_empty else float("nan"))
        d_abs_seen.append(abs_seen / w.n_gap_seen if w.n_gap_seen else float("nan"))
        if delta < tol:
            return X, {"n_iter": it + 1, "deltas": deltas, "deltas_empty": d_empty,
                       "deltas_seen": d_seen, "deltas_abs": d_abs,
                       "deltas_abs_empty": d_abs_empty, "deltas_abs_seen": d_abs_seen,
                       "converged": True}
    log.warning("%sk=%d did not converge in %d EM iterations (last delta %.2e > tol %.1e); "
                "the reconstruction is still usable but is not a fixed point",
                f"{label}: " if label else "", k, max_iter, deltas[-1], tol)
    return X, {"n_iter": int(max_iter), "deltas": deltas, "deltas_empty": d_empty,
               "deltas_seen": d_seen, "deltas_abs": d_abs,
               "deltas_abs_empty": d_abs_empty, "deltas_abs_seen": d_abs_seen,
               "converged": False}


def filter_settings(t_c_grid, alpha_max: float) -> list[dict]:
    """Target cutoff periods (days) -> the (alpha, p) that reaches each one most cheaply.

    THE GRID IS IN T_c, NOT p, because T_c = 2*pi*sqrt(alpha*p) is the quantity that actually
    matters and (alpha, p) is a redundant parameterisation of it. Measured at fixed alpha*p,
    varying the split from alpha=0.1/p=2 to alpha=0.002/p=100: the convergence delta after 400
    iterations was identical to three digits (5.03e-3 vs 5.12e-3) while wall-clock rose 5.7x,
    because the filter runs p times per EM iteration. So for any target reach there is one
    sensible choice -- the LARGEST alpha the stability limit allows, giving the smallest p.

    p is the smallest integer with alpha = (T_c/2pi)^2 / p <= alpha_max, so alpha lands just
    under the ceiling and p stays small. T_c = 0 means no filter.

    An entry may instead be an explicit `[alpha, p]` PAIR, which is used verbatim and its T_c
    derived. That exists to reproduce a published setting exactly -- the method paper's
    reference values are alpha = 0.01, p = 3 -- rather than substituting the cheaper pair with
    the same reach. Given the equal-T_c measurement the two are numerically interchangeable, so
    a pair is for provenance, not for accuracy; the log marks which entries are which.
    """
    out = []
    for raw in t_c_grid:
        if isinstance(raw, (list, tuple)):
            if len(raw) != 2:
                raise ValueError(
                    f"filter.t_c_grid entry {raw!r} must be a cutoff period in days, or an "
                    "explicit [alpha, p] pair")
            alpha, p = float(raw[0]), int(raw[1])
            out.append({"t_c": float(2.0 * np.pi * np.sqrt(max(alpha * p, 0.0))),
                        "alpha": alpha, "p": p, "explicit": True})
            continue
        tc = float(raw)
        if tc <= 0:
            out.append({"t_c": 0.0, "alpha": 0.0, "p": 0, "explicit": False})
            continue
        ap = (tc / (2.0 * np.pi)) ** 2
        p = max(1, int(np.ceil(ap / float(alpha_max))))
        out.append({"t_c": tc, "alpha": ap / p, "p": p, "explicit": False})

    seen = {}
    for s in out:
        key = round(s["t_c"], 6)
        if key in seen:
            log.warning("filter.t_c_grid: %g d appears twice (as %s and %s). Only alpha*p "
                        "affects the result, so these are duplicate work.", s["t_c"],
                        seen[key], f"alpha={s['alpha']:.4g}/p={s['p']}")
        seen[key] = f"alpha={s['alpha']:.4g}/p={s['p']}"
    return out


def choose_cv_points(observed: np.ndarray, msk: np.ndarray, cfg: dict, rng: np.random.Generator
                     ) -> tuple[np.ndarray, np.ndarray]:
    """(day_mask, point_mask): two disjoint held-out subsets of `observed`.

    DAY MASK -- `cv.day_frac` of the days that have data, held out whole. Selects p.

    POINT MASK -- `cv.frac` of the points in the REMAINING days, using donor geometry: the gap
    mask of another real day is pasted onto the target, so the held-out shape is a contiguous
    cloud like the gaps actually being filled rather than scattered pixels each surrounded by
    data. The doc (§6) is explicit that uniform-random points are optimistic for this reason.

    Donor geometry needs two guards the doc does not mention, because pasting is unbounded:
    a sparse donor on a dense target holds out almost the entire target day, which both blows
    the budget and can empty the day, recreating the zero-column degeneracy on a day that is
    supposed to be scoring. So donors are drawn from a coverage band around the target, and
    the result is capped at `cv.max_date_frac` and floored at `cv.min_date_obs_after`.
    """
    c = cfg["cv"]
    m, n = observed.shape
    date_n = observed.sum(axis=0)
    data_days = np.flatnonzero(date_n > 0)

    day_mask = np.zeros_like(observed)
    n_hold = int(round(float(c["day_frac"]) * data_days.size))
    hold = rng.choice(data_days, size=n_hold, replace=False) if n_hold else np.array([], int)
    if n_hold:
        day_mask[:, hold] = observed[:, hold]

    avail = np.setdiff1d(data_days, hold)
    point_mask = np.zeros_like(observed)
    point_mask = np.transpose(msk)
    # budget = float(c["frac"]) * float(observed[:, avail].sum())
    # lo, hi = float(c["donor_lo"]), float(c["donor_hi"])
    #got = 0.0
    # EVERY available day contributes its own share rather than the budget being spent on
    # whichever dense days come first: a holdout concentrated on a handful of days estimates
    # the error on those days, not on the series. Each day's share is drawn from within a
    # donor cloud, so the points stay spatially clustered like a real gap even when the cloud
    # has to be thinned to fit.
    # for tgt in rng.permutation(avail):
    #     cov = date_n[tgt]
    #     want = float(c["frac"]) * cov
    #     cap = min(float(c["max_date_frac"]) * cov, cov - float(c["min_date_obs_after"]), want)
    #     if cap < 1:
    #         continue
    #     band = avail[(date_n[avail] >= lo * cov) & (date_n[avail] <= hi * cov) & (avail != tgt)]
    #     if band.size == 0:
    #         continue
    #     cand = np.flatnonzero(observed[:, tgt] & ~observed[:, rng.choice(band)])
    #     if cand.size == 0:
    #         continue
    #     if cand.size > cap:
    #         cand = rng.choice(cand, size=int(cap), replace=False)
    #     point_mask[cand, tgt] = True
    #     got += cand.size

    log.info("cv: %d of %d data-days held out whole (%.1f%% of observations); "
             "%d scattered points in %d other days (%.2f%%)",
             hold.size, data_days.size, 100 * day_mask.sum() / observed.sum(),
             int(point_mask.sum()), int((point_mask.any(axis=0)).sum()),
             100 * point_mask.sum() / observed.sum())
    # if got < 0.5 * budget:
    #     log.warning("cv: only reached %.2f%% of the %.2f%% point budget -- donor band "
    #                 "[%.2f, %.2f] may be too narrow for this coverage distribution",
    #                 100 * got / observed.sum(), 100 * float(c["frac"]), lo, hi)
    return day_mask, point_mask




def _rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2))) if a.size else float("nan")


def _pick_k(curve: pd.DataFrame, rule: str, rel_tol: float) -> int:
    """Smallest k within rel_tol of the best CV error, or the outright best.

    `parsimonious` is the default because DINEOF CV curves are notoriously flat near the
    optimum, so the argmin is often a coin toss between several k that differ by <1%, and the
    smaller one is the better-conditioned model.
    """
    best = curve["rmse_point"].min()
    if rule == "best":
        return int(curve.loc[curve["rmse_point"].idxmin(), "k"])
    ok = curve[curve["rmse_point"] <= best * (1.0 + rel_tol)]
    return int(ok["k"].min())


def _pick_setting(curve: pd.DataFrame, col: str, rule: str, rel_tol: float):
    """(t_c, k) minimising `col` over the whole grid, parsimony applied within the winning t_c.

    Used once per holdout, which is what produces the two different optima.
    """
    best_tc = float(curve.loc[curve[col].idxmin(), "t_c"])
    sub = curve[curve["t_c"] == best_tc]
    if rule == "best":
        return best_tc, int(sub.loc[sub[col].idxmin(), "k"])
    ok = sub[sub[col] <= sub[col].min() * (1.0 + rel_tol)]
    return best_tc, int(ok["k"].min())


def edineof(X_raw: np.ndarray, observed: np.ndarray, valid_msk: np.ndarray, t: np.ndarray, cfg: dict,
            warm: dict | None = None) -> dict:
    """The full algorithm, doc §7, with the k and p searches split across the two CV sets."""
    f, mo, em, c = cfg["filter"], cfg["modes"], cfg["em"], cfg["cv"]
    if warm:
        # No search: the settings are already chosen, at a resolution where searching was cheap.
        f = dict(f, t_c_grid=sorted({warm["tc_point"], warm["tc_day"]}))
        mo = dict(mo, k_grid=sorted({warm["k_point"], warm["k_day"]}))
    settings = filter_settings(f["t_c_grid"], float(f["alpha_max"]))
    for s in settings:
        check_stability(t, s["alpha"], s["p"], float(f["stability_factor"]))
    log.info("filter grid: %s", ", ".join(
        f"T_c={s['t_c']:.3g}d (a={s['alpha']:.4g}, p={s['p']}"
        f"{', explicit' if s.get('explicit') else ''})" for s in settings))

    mu = float(np.mean(X_raw[observed]))
    X0 = np.where(observed, X_raw - mu, 0.0)
    log.info("centering: mu = %+.6f (near zero is expected -- stage 5 already removed a "
             "per-pixel seasonal mean)", mu)

    rng = np.random.default_rng(int(c["seed"]))
    day_cv, point_cv = choose_cv_points(observed, valid_msk, cfg, rng)
    cv_any = day_cv | point_cv

    gaps = (~observed) | cv_any

    # Index form of the two CV masks, so scoring is a 4M-element integer gather rather than
    # two boolean selections (0.34 s per combination) over the full matrix. The truths must be
    # captured BEFORE the held-out entries are zeroed below, or they are all zero.
    point_idx = np.flatnonzero(point_cv.reshape(-1))
    day_idx = np.flatnonzero(day_cv.reshape(-1))
    truth_point = X0.reshape(-1)[point_idx].copy()
    truth_day = X0.reshape(-1)[day_idx].copy()

    sd = float(np.std(X0[observed & ~cv_any]))
    X0[gaps] = 0.0
    log.info("matrix %d x %d, %.1f%% observed, %.1f%% gaps during the search; sd %.4f",
             *X0.shape, 100 * observed.mean(), 100 * gaps.mean(), sd)

    # An explicit, non-uniform k grid: dense where the CV curve turns, sparse above. Doubling
    # k near 40 changes the reconstruction far less than doubling it near 4, so testing every
    # integer up there costs the most per fit and tells you the least.
    def apply_seed(X: np.ndarray, gap_mask: np.ndarray, tc: float) -> None:
        """Seed the gaps from the coarse field -- except where that would be a lie.

        At T_c = 0 an all-gap column is STRUCTURALLY pinned at zero: B[:,j] = 0 so V[j,:] = 0
        so the reconstruction is 0 forever. A warm start breaks that degeneracy artificially --
        X[:,j] starts non-zero, so B[:,j] is non-zero, and the column then sustains whatever
        the coarse run put there. Measured before this guard: max |z| of 1.6 on empty dates in
        a fit labelled "no filter", where the cold-start value is a structural constant.
        Those values were inherited from a FILTERED coarse run, so the channel did not mean
        what its settings said. Zeroing them keeps T_c = 0 honest, and costs nothing: a column
        the filter cannot reach has no information to start from anyway.
        """
        if not warm:
            return
        seed = warm["point"] if tc == warm["tc_point"] else warm["day"]
        X[gap_mask] = seed[gap_mask]
        if tc == 0:
            empty = gap_mask.all(axis=0)
            if empty.any():
                X[:, empty] = 0.0

    ks = [int(v) for v in mo["k_grid"]]
    work = Workspace(X0, gaps)
    rows = []
    for s in settings:
        alpha, p, tc = s["alpha"], s["p"], s["t_c"]
        X = X0.copy()
        apply_seed(X, gaps, tc)              # coarse field at the gaps, not the mean
        errs = []
        t0 = time.time()
        for k in ks:
            tk = time.time()
            np.copyto(work.prev, X)              # the warm start this fit begins from
            X, hist = fill(X, gaps, k, t, alpha, p, float(em["tol"]), int(em["max_iter"]),
                           sd, label=f"T_c={tc:g}", work=work)
            Xf = X.reshape(-1)
            e_point = _rms(Xf[point_idx], truth_point)
            e_day = _rms(Xf[day_idx], truth_day)
            # How far this k moved the solution from the previous k's. The warm start pays off
            # only insofar as this falls with k -- flat means every k is effectively cold.
            step = work.gap_rms(X, work.prev)
            secs = time.time() - tk
            # Per-k progress. A full sweep is thousands of EM iterations, so a run with only
            # per-setting output is indistinguishable from a hang for minutes at a time.
            log.info("  T_c=%-5g k=%-3d point %.5f  day %.5f  %3d iter  %5.1fs  "
                     "warm-step %.4f%s", tc, k, e_point, e_day, hist["n_iter"], secs, step,
                     "" if hist["converged"] else "  NOT CONVERGED")
            rows.append(dict(t_c=tc, alpha=alpha, p=p, k=k,
                             rmse_point=e_point, rmse_day=e_day, seconds=secs,
                             warm_step=step, n_iter=hist["n_iter"],
                             converged=int(hist["converged"])))
            errs.append(e_point)
            # The doc's rule: stop once CV has risen on `patience` consecutive k.
            n = int(mo["patience"])
            if len(errs) > n and all(errs[-i] > errs[-i - 1] for i in range(1, n + 1)):
                break
        log.info("T_c=%g d (a=%.4g, p=%d): best point-CV %.4f at k=%d, day-CV %.4f "
                 "[%.0fs, %d k]", tc, alpha, p, min(errs), ks[int(np.argmin(errs))],
                 min(r["rmse_day"] for r in rows if r["t_c"] == tc), time.time() - t0,
                 len(errs))

    curve = pd.DataFrame(rows)
    tcs = [s["t_c"] for s in settings]
    if (curve.groupby("t_c")["k"].max() == ks[-1]).all():
        log.warning("the point-CV curve never turned up within the k grid (max %d) at any "
                    "T_c. k_opt is the ceiling, not an optimum -- extend modes.k_grid", ks[-1])

    # k from the point holdout (spatial rank); T_c from the day holdout (temporal reach). See
    # the module docstring for why these cannot share a criterion.
    per_tc = {s["t_c"]: _pick_k(curve[curve["t_c"] == s["t_c"]], mo["rule"],
                                float(mo["rel_tol"])) for s in settings}
    col = "rmse_day" if day_cv.any() else "rmse_point"
    day_at = {tc: float(curve[(curve["t_c"] == tc) & (curve["k"] == per_tc[tc])][col].iloc[0])
              for tc in tcs}
    if not day_cv.any() and len(settings) > 1:
        log.warning(
            "cv.day_frac = 0, so there is no day holdout and T_c is being selected on the "
            "point holdout instead. Expect T_c = 0: a point-holdout pixel sits in a day whose "
            "modes are already pinned by its own surviving pixels, so filtering can only "
            "distort them. This reduces eDINEOF to plain DINEOF.")
    # TWO OPTIMA, because the two holdouts want genuinely different models and measurement says
    # the gap between them is large -- on admiralty_inlet, point-CV picks T_c=0/k=15 while
    # day-CV picks T_c=8/k=2. That is not a tuning wobble, it is the physics: a date with its
    # own pixels wants a rich basis and no smoothing distorting the modes it already pins,
    # while a date with nothing wants a smooth low-rank field and long temporal reach to borrow
    # from its neighbours. Forcing one setting to serve both compromises whichever matters more
    # to the reader, so both are fitted and both are written.
    if warm:
        tc_pt, k_pt = warm["tc_point"], warm["k_point"]
        tc_dy, k_dy = warm["tc_day"], warm["k_day"]
    else:
        tc_pt, k_pt = _pick_setting(curve, "rmse_point", mo["rule"], float(mo["rel_tol"]))
        if day_cv.any():
            tc_dy, k_dy = _pick_setting(curve, "rmse_day", mo["rule"], float(mo["rel_tol"]))
        else:
            tc_dy, k_dy = tc_pt, k_pt

    tc_opt, k_opt = tc_pt, k_pt         # the primary: the product is a gap-filled field
    sel_s = next(s for s in settings if s["t_c"] == tc_opt)
    alpha, p_opt = sel_s["alpha"], sel_s["p"]
    base = day_at.get(0.0)
    log.info("dates WITH data   -> T_c=%g d, k=%d  (point-CV %.4f)",
             tc_pt, k_pt, float(curve[(curve["t_c"] == tc_pt) & (curve["k"] == k_pt)]
                                ["rmse_point"].iloc[0]))
    log.info("dates WITHOUT data-> T_c=%g d, k=%d  (%s %.4f%s)", tc_dy, k_dy, col,
             float(curve[(curve["t_c"] == tc_dy) & (curve["k"] == k_dy)][col].iloc[0]),
             f", vs {base:.4f} at T_c=0 -- the climatology baseline" if base is not None else "")
    # Only meaningful when the grid actually offered an alternative: with a single-entry grid
    # the selection IS the baseline, and comparing it against itself always "fails".
    if base is not None and day_cv.any() and len(settings) > 1 and day_at[tc_opt] >= base:
        log.warning("THE FILTER BUYS NOTHING: day-CV at the selected T_c is not better than "
                    "at T_c=0, which is the per-pixel seasonal climatology stage 5 removed. "
                    "The empty days are being reconstructed no better than by climatology, and "
                    "the daily product should not be presented as more than that.")

    # Final pass: hand the held-out points back and refit at the selected settings, doc §7.
    #
    # Started from X0 rather than from the winning search iterate. The search kept nothing:
    # retaining one filled array per combination cost 0.13 GB apiece and 14 GB across the grid,
    # to save a single fit. It was also never more than an approximate warm start, because the
    # final pass has a DIFFERENT gap set -- the CV points are observed again here -- so the
    # fixed point it converges to is not the one the search found.
    final_gaps = ~observed

    def final_fit(tc: float, k: int, label: str) -> dict:
        s = next(q for q in settings if q["t_c"] == tc)
        X = X0.copy()
        apply_seed(X, final_gaps, tc)
        Xf = X.reshape(-1)
        Xf[point_idx] = truth_point          # the held-out points are observations again
        Xf[day_idx] = truth_day
        X, hist = fill(X, final_gaps, k, t, s["alpha"], s["p"], float(em["tol"]),
                       int(em["max_iter"]), sd, label=label)
        U, sigma, V = top_k_modes(X, k, t, s["alpha"], s["p"], use_filter=True,
                                  convention=mo["sigma_convention"],
                                  fix_sign=bool(mo["fix_sign"]))
        # The rank-k model field itself, U diag(sigma) V', evaluated EVERYWHERE -- including at
        # pixels that were observed, where `X` holds the observation rather than the model's
        # own estimate of it. `X` is the analysis; this is what the basis alone says.
        return dict(X=X + mu, lowrank=reconstruct(U, sigma, V) + mu, U=U, sigma=sigma, V=V,
                    hist=hist, k=k, t_c=tc, alpha=s["alpha"], p=s["p"])

    day_fit = final_fit(tc_dy, k_dy, "final/day-opt")
    same = (tc_pt == tc_dy) and (k_pt == k_dy)
    if same:
        log.info("both holdouts chose the same setting; one fit serves both")
        point_fit = day_fit
    else:
        point_fit = final_fit(tc_pt, k_pt, "final/point-opt")

    X, lowrank = point_fit["X"], point_fit["lowrank"]
    U, sigma, V, hist = point_fit["U"], point_fit["sigma"], point_fit["V"], point_fit["hist"]
    var = sigma ** 2
    return dict(X=X, lowrank=lowrank, U=U, sigma=sigma, V=V,
                day_fit=day_fit, point_fit=point_fit, same_setting=same,
                k_opt=k_opt, p_opt=p_opt, alpha_opt=alpha, tc_opt=tc_opt,
                k_day=k_dy, tc_day=tc_dy, mu=mu, sd=sd,
                curve=curve, day_cv=day_cv, point_cv=point_cv, gaps=final_gaps,
                history=hist, day_at=day_at, per_tc=per_tc, settings=settings,
                var_explained=float(var[:k_opt].sum() / max(var.sum(), 1e-300)))


# ==================================================================== plumbing

DEFAULTS = {
    "data": {
        "aoi": "admiralty_inlet",
        "cube": "data/datacube/admiralty_inlet_standardized.zarr",
        "out": "data/datacube/admiralty_inlet_dineof.zarr",
        "watervar": "landcover_water",
        "channel": "sst_z",
        "coef": "sst_seasonal_coef",
        "scale": "sst_seasonal_sd",
        "validation_msk": "validation_msk"
    },
    "matrix": {"min_pixel_obs": 1, "min_date_obs": 200, "coarsen": 1,
               "warm_start_from": None},
    "filter": {"t_c_grid": [0, 2, 4, 6, 8, 11], "alpha_max": 0.25,
               "stability_factor": 0.25},
    "modes": {"k_grid": [1, 2, 3, 4, 5, 7, 9, 11, 15, 20, 25, 30, 40],
              "rule": "parsimonious", "rel_tol": 0.01,
              "patience": 3, "sigma_convention": "projection", "fix_sign": True},
    "em": {"tol": 1.0e-3, "max_iter": 100},
    "cv": {"frac": 0.02, "day_frac": 0.10, "max_date_frac": 0.4, "min_date_obs_after": 50,
           "donor_lo": 0.5, "donor_hi": 0.95, "seed": 0},
    "carry": [],
    "output": {
        "chunks": {"time": 64, "y": 128, "x": 128},
        "compression": {"codec": "zstd", "level": 5, "shuffle": "shuffle"},
        "modes_file": "dineof_modes.nc",
        "report": "dineof_cv.csv",
        "write_figures": True,
        "fig_dir": "figures/edineof",
        "figure_cv": True,
        "figure_convergence": True,
        "figure_modes": True,
        "figure_contact": True,
        "figure_dpi": 130,
        "figure_ncols": 12,
    },
}

OPAQUE_SECTIONS = {"carry"}
RULES = ("parsimonious", "best")
CONVENTIONS = ("projection", "reference")


def load_config(path: Path, defaults: dict = DEFAULTS) -> dict:
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
    validate(cfg, path)
    return cfg


def validate(cfg: dict, path: Path) -> None:
    f, mo, c, m = cfg["filter"], cfg["modes"], cfg["cv"], cfg["matrix"]
    if not isinstance(f["t_c_grid"], list) or not f["t_c_grid"]:
        raise ValueError(f"{path}: filter.t_c_grid must be a non-empty list of cutoff "
                         "periods in days (0 = no filter)")
    for v in f["t_c_grid"]:
        if isinstance(v, (list, tuple)):
            if len(v) != 2 or float(v[0]) < 0 or int(v[1]) < 0:
                raise ValueError(
                    f"{path}: filter.t_c_grid entry {v!r} must be [alpha, p] with both >= 0")
        elif float(v) < 0:
            raise ValueError(f"{path}: filter.t_c_grid entries must be >= 0")
    if not 0 < float(f["alpha_max"]) <= 0.5:
        raise ValueError(f"{path}: filter.alpha_max must be in (0, 0.5]")
    if not 0 < float(f["stability_factor"]) <= 0.5:
        raise ValueError(f"{path}: filter.stability_factor must be in (0, 0.5]")
    if mo["rule"] not in RULES:
        raise ValueError(f"{path}: modes.rule must be one of {RULES}")
    if mo["sigma_convention"] not in CONVENTIONS:
        raise ValueError(f"{path}: modes.sigma_convention must be one of {CONVENTIONS}")
    if not isinstance(mo["k_grid"], list) or not mo["k_grid"]:
        raise ValueError(f"{path}: modes.k_grid must be a non-empty list of mode counts")
    kg = [int(v) for v in mo["k_grid"]]
    if any(v < 1 for v in kg) or kg != sorted(kg) or len(set(kg)) != len(kg):
        raise ValueError(f"{path}: modes.k_grid must be strictly increasing and >= 1 -- the "
                         "patience rule reads it as an ordered sequence")
    if not 0 < float(c["frac"]) < 1:
        raise ValueError(f"{path}: cv.frac must be in (0, 1)")
    # day_frac = 0 disables the day holdout, which is the doc's own single-CV setup. Allowed,
    # but then nothing can select p -- see `edineof`.
    if not 0 <= float(c["day_frac"]) < 1:
        raise ValueError(f"{path}: cv.day_frac must be in [0, 1)")
    if not 0 < float(c["donor_lo"]) < float(c["donor_hi"]) <= 1:
        raise ValueError(f"{path}: need 0 < cv.donor_lo < cv.donor_hi <= 1")
    if int(m["coarsen"]) < 1:
        raise ValueError(f"{path}: matrix.coarsen must be >= 1 (1 = full resolution)")
    if int(m["min_pixel_obs"]) < 1:
        raise ValueError(
            f"{path}: matrix.min_pixel_obs must be >= 1. A pixel with no observations has an "
            "all-gap row, so U[i,:] = 0 and it reconstructs to exactly the mean forever -- "
            "which un-standardizes to a smooth, plausible, entirely fabricated series")


def block_mean(a: np.ndarray, valid: np.ndarray, f: int) -> tuple[np.ndarray, np.ndarray]:
    """Average the trailing two axes in f x f blocks, over `valid` cells only.

    Returns (mean, count). Blocks with no valid cell come back NaN with count 0. A ragged edge
    is trimmed rather than partially averaged, so every block has the same footprint.
    """
    *lead, H, W = a.shape
    Hc, Wc = (H // f) * f, (W // f) * f
    a = a[..., :Hc, :Wc]
    valid = valid[..., :Hc, :Wc]
    shape = (*lead, Hc // f, f, Wc // f, f)
    n = valid.reshape(shape).sum(axis=(-3, -1))
    s = np.where(valid, np.nan_to_num(a), 0.0).reshape(shape).sum(axis=(-3, -1))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return out, n


def coarsen_inputs(ds: xr.Dataset, cfg: dict, f: int) -> dict:
    """Block-average the standardized field and everything needed to invert it, by f.

    A TUNING mode. `m` falls by f^2 (55,561 -> ~3,500 at f=4) so every term that scales with
    the pixel count gets that much cheaper, while `eigh` and the temporal filter -- which act
    on the (n, n) covariance -- are untouched. So the speedup is well short of f^2, and shrinks
    as p grows.

    It also does NOT reduce the iteration count: convergence is set by diffusion across the
    empty columns, which is purely temporal and does not know about pixel size. Coarsening
    makes each iteration cheaper, not fewer of them.

    THE INVERSE IS APPROXIMATE HERE. The seasonal coefficients average exactly -- the seasonal
    field is linear in them, so the mean of the block's cycles is the cycle of the mean. The
    per-pixel SCALE does not: averaging f^2 cells reduces the variance of the block mean below
    the mean of the cell variances, so `z * sd_coarse + seasonal` recovers a slightly damped
    field. Fine for choosing k and p, which is all this mode is for; rerun at coarsen: 1 for
    the field you actually ship.
    """
    d = cfg["data"]
    water = np.asarray(ds[d["watervar"]].compute() > 0.5)
    z = ds[d["channel"]].values
    msk = ds[d["validation_msk"]].values
    coef = ds[d["coef"]].values
    scale = ds[d["scale"]].values

    zc, _ = block_mean(z, np.isfinite(z) & water[None, :, :], f)
    mskc, _ = block_mean(msk, np.isfinite(z) & water[None, :, :], f)
    mskc = mskc > 0  # get blocks with all valid entries
    wc, wn = block_mean(water.astype(float), np.ones_like(water, bool), f)
    water_c = wn * 0 + (wc > 0)                       # a block is water if ANY cell is
    coef_c, _ = block_mean(coef, np.broadcast_to(water, coef.shape), f)
    scale_c, _ = block_mean(scale, water, f)

    coords = {}
    for name in ("y", "x"):
        if name in ds.coords:
            v = np.asarray(ds[name].values)
            v = v[: (v.size // f) * f]
            coords[name] = v.reshape(-1, f).mean(axis=1)

    log.info("coarsen %dx: %d x %d -> %d x %d cells, %d water -> %d",
             f, *water.shape, *water_c.shape, int(water.sum()), int(water_c.sum()))
    return dict(z=zc, msk=mskc, water=np.asarray(water_c, bool), coef=coef_c, scale=scale_c,
                coords=coords)


def upsample(a: np.ndarray, factor: int, shape: tuple[int, int]) -> np.ndarray:
    """Block-repeat the trailing two axes by `factor`, then pad to `shape` by edge extension.

    The inverse of `block_mean`'s geometry. `block_mean` trims a ragged edge (303 -> 300 at
    factor 4), so repeating gives 300 back and the last few rows/columns have no coarse cell
    above them; they take their nearest neighbour's value. Those are a warm START, not an
    answer -- the EM overwrites them from the data like any other gap.
    """
    up = np.repeat(np.repeat(a, factor, axis=-2), factor, axis=-1)
    H, W = shape
    if up.shape[-2] < H:
        up = np.concatenate([up, np.repeat(up[..., -1:, :], H - up.shape[-2], axis=-2)], -2)
    if up.shape[-1] < W:
        up = np.concatenate([up, np.repeat(up[..., :, -1:], W - up.shape[-1], axis=-1)], -1)
    return up[..., :H, :W]


def load_warm_start(path: Path, cfg: dict, sel: dict) -> dict:
    """Settings and an initial field from a coarser run of this same AoI.

    THE POINT IS TO SKIP THE SEARCH, NOT TO SKIP THE FIT. k and T_c are read from the coarse
    run's attrs and used directly -- both describe structure (spatial rank, temporal reach)
    rather than pixel size, so they carry across resolutions. The coarse field is then
    block-repeated onto the fine grid and used as the EM's starting point instead of zeros.

    CROSS-VALIDATION NUMBERS FROM A WARM-STARTED RUN ARE NOT INDEPENDENT. The coarse field was
    fitted using data that this run holds out, so anything it reports as CV error is
    contaminated. That is acceptable here only because nothing is being selected on it -- the
    settings are already fixed -- and the run says so in the log.
    """
    with xr.open_zarr(path) as cs:
        fit = json.loads(cs.attrs["dineof_edineof_fit"])
        coarse_f = int(fit.get("coarsen", 1))
        out = {"k_day": int(fit["k_opt"]), "tc_day": float(fit["tc_opt"]),
               "k_point": int(fit["k_point"]), "tc_point": float(fit["tc_point"]),
               "from": str(path), "coarsen": coarse_f}
        ratio = coarse_f // max(int(sel["coarsen"]), 1)
        if ratio < 1 or coarse_f % max(int(sel["coarsen"]), 1):
            raise ValueError(
                f"{path} was run at coarsen={coarse_f}, which is not a whole multiple of this "
                f"run's coarsen={sel['coarsen']}; the grids do not nest")
        water, keep = sel["water"], sel["keep"]
        for name in ("point", "day"):
            var = f"sst_recon_z_{name}"
            if var not in cs:
                raise ValueError(
                    f"{path} has no {var}; it predates the two-tuning output and cannot seed "
                    "a warm start")
            up = upsample(cs[var].values, ratio, water.shape)      # (T, y, x)
            seed = np.ascontiguousarray(up[:, water][:, keep].T.astype(np.float64))
            # A fine pixel can sit under a coarse cell the coarse run dropped (no observations
            # there) or past its trimmed ragged edge, and those come back NaN. Fall back to 0,
            # which in the centred field is the mean -- exactly the doc's default initializer.
            # A NaN here would otherwise reach `eigh` and abort the run.
            bad = ~np.isfinite(seed)
            if bad.any():
                log.info("  %s: %d of %d seed entries have no coarse value (dropped cells or "
                         "trimmed edge); starting those from the mean",
                         name, int(bad.sum()), seed.size)
                seed[bad] = 0.0
            out[name] = seed
    log.info("warm start from %s (coarsen %d -> %d, x%d): point k=%d/T_c=%g, day k=%d/T_c=%g",
             path.name, coarse_f, sel["coarsen"], ratio, out["k_point"], out["tc_point"],
             out["k_day"], out["tc_day"])
    log.warning("warm-started run: the CV numbers it reports are NOT independent -- the coarse "
                "field was fitted on data this run holds out. Nothing is selected on them "
                "here, since k and T_c come from the coarse run.")
    return out


def select_matrix(ds: xr.Dataset, cfg: dict) -> dict:
    """Build (m, n) X and its observed mask from the cube.

    Thin dates are MASKED, not dropped: their observations are discarded but the column stays,
    so it becomes an empty day the filter populates like the other 180 and the output keeps a
    complete daily axis. Pixels with no observations are genuinely dropped -- see `validate`.
    """
    d, m = cfg["data"], cfg["matrix"]
    f = int(m["coarsen"])
    if f > 1:
        c = coarsen_inputs(ds, cfg, f)
        water, z, msk, coef, scale, coords = (c["water"], c["z"], c["msk"], c["coef"], c["scale"],
                                         c["coords"])
    else:
        water = np.asarray(ds[d["watervar"]].compute() > 0.5)
        z = ds[d["channel"]].values
        msk = ds[d["validation_msk"]].values > 0
        coef = ds[d["coef"]].values
        scale = ds[d["scale"]].values
        coords = {n: np.asarray(ds[n].values) for n in ("y", "x") if n in ds.coords}
    O3 = np.isfinite(z) & water[None, :, :]
    

    # `min_date_obs` is a pixel COUNT, so it has to be rescaled when the grid is coarsened or
    # it silently becomes a far harsher cut: 200 is 0.36% of the water at 100 m but 5.4% of it
    # at 400 m, which masked 29 dates instead of 8 and changed the problem being tuned.
    min_date = max(1, int(round(int(m["min_date_obs"]) / f ** 2)))
    date_n = O3.sum(axis=(1, 2))
    thin = (date_n > 0) & (date_n < min_date)
    if thin.any():
        log.info("masking %d dates with < %d observed px%s (%d obs, %.3f%% of the data); "
                 "their columns stay and are filled by the filter",
                 int(thin.sum()), min_date,
                 f" (min_date_obs {m['min_date_obs']} / coarsen^2)" if f > 1 else "",
                 int(date_n[thin].sum()), 100 * date_n[thin].sum() / date_n.sum())
        O3[thin] = False

    Zw = z[:, water]
    msk_w = msk[:,water]
    Ow = O3[:, water]
    pix_n = Ow.sum(axis=0)
    keep = pix_n >= int(m["min_pixel_obs"])
    if (~keep).any():
        log.info("dropping %d water px with < %d observations", int((~keep).sum()),
                 int(m["min_pixel_obs"]))

    X = np.where(Ow, np.nan_to_num(Zw), 0.0)[:, keep].T.astype(np.float64)
    observed = Ow[:, keep].T
    t = (pd.to_datetime(ds["time"].values) - pd.to_datetime(ds["time"].values[0])).days
    t = np.asarray(t, float)

    valid_msk = msk_w[:, keep]

    if X.shape[0] < X.shape[1]:
        raise ValueError(
            f"matrix is {X.shape[0]} pixels x {X.shape[1]} dates; the doc requires m >= n "
            "(transpose the problem and swap U/V if you really need fewer pixels than dates)")
    return dict(X=X, observed=observed, valid_msk = valid_msk, t=t, water=water, keep=keep, m=X.shape[0],
                n=X.shape[1], date_n=O3.sum(axis=(1, 2)), coef=coef, scale=scale,
                coords=coords, coarsen=f)


def unstandardize(Z: np.ndarray, ds: xr.Dataset, cfg: dict, sel: dict) -> np.ndarray:
    """(m, n) z-scores -> (time, y, x) kelvin, re-adding the seasonal cycle and scale.

    Takes the coefficients and scale from `sel`, not from the cube, because under
    `matrix.coarsen` they have been block-averaged to match the matrix -- see `coarsen_inputs`
    for why the scale half of that is approximate.
    """
    a = ds[cfg["data"]["coef"]].attrs
    times = pd.to_datetime(ds["time"].values)
    X = design_matrix(times, int(a["n_harmonics"]), float(a["period_days"]))
    seasonal = np.tensordot(X, np.nan_to_num(sel["coef"]), axes=1)      # (T, y, x)

    water, keep = sel["water"], sel["keep"]
    full = np.full((sel["n"], int(water.sum())), np.nan)
    full[:, keep] = Z.T
    out = np.full(seasonal.shape, np.nan, dtype="float32")
    out[:, water] = (full * sel["scale"][water][None, :]
                     + seasonal[:, water]).astype("float32")
    return out


def _grid(v, water, fill=np.nan, dtype="float32"):
    g = np.full(water.shape, fill, dtype=dtype)
    g[water] = v
    return g


def build_dataset(ds: xr.Dataset, cfg: dict, res: dict, sel: dict,
                  fields: dict) -> xr.Dataset:
    dims = ("time", "y", "x")
    water, keep = sel["water"], sel["keep"]
    nw = int(water.sum())
    data = {}

    def scatter(flat_mn, dtype="float32", fill=np.nan):
        full = np.full((sel["n"], nw), fill, dtype=dtype)
        full[:, keep] = flat_mn.T
        out = np.full((sel["n"],) + water.shape, fill, dtype=dtype)
        out[:, water] = full
        return out

    # Which dates have an observation of their own. This is what selects between the two fits
    # below, and it is also what each fit was cross-validated against.
    has_data = sel["observed"].sum(axis=0) > 0                      # (n,) bool

    # --- the two tuned fits, each written whole so either can be used on its own ---
    for name, why, kk, tc in (
            ("point", "dates WITH their own observations", res["k_opt"], res["tc_opt"]),
            ("day", "dates with NO observations (filter-interpolated)", res["k_day"],
             res["tc_day"])):
        f = fields[name]
        v = xr.DataArray(f["recon"], dims=dims, name=f"sst_recon_{name}")
        v.attrs.update(
            long_name=f"eDINEOF reconstruction, tuned for {why}", units="K",
            k_modes=int(kk), cutoff_days=float(tc),
            selected_on="point-holdout CV" if name == "point" else "day-holdout CV",
            comment=(
                "A COMPLETE field at these settings, every date. The two tunings differ "
                "because the holdouts want different models: a date with its own pixels wants "
                "a rich basis and no smoothing distorting the modes it already pins, while a "
                "date with none wants a smooth low-rank field and long temporal reach to "
                "borrow from neighbours. Use sst_recon for the per-date pick of the two."))
        data[f"sst_recon_{name}"] = v

        lr = xr.DataArray(f["lowrank"], dims=dims, name=f"sst_lowrank_{name}")
        lr.attrs.update(
            long_name=f"rank-k model field U diag(sigma) V' at the {name}-tuned settings",
            units="K", k_modes=int(kk), cutoff_days=float(tc),
            comment=("evaluated at EVERY pixel including observed ones, unlike sst_recon_* "
                     "which holds observations fixed. sst_composite minus this is the "
                     "residual against the model. NOT a leave-one-day-out prediction: V[j] is "
                     "still informed by date j's own pixels through B = X'X."))
        data[f"sst_lowrank_{name}"] = lr

    # --- the recommended field: each date taken from the fit validated for it ---
    merged = np.where(has_data[:, None, None], fields["point"]["recon"],
                      fields["day"]["recon"]).astype("float32")
    da = xr.DataArray(merged, dims=dims, name="sst_recon")
    da.attrs.update(
        long_name="eDINEOF-reconstructed SST, per-date best of the two tunings", units="K",
        k_point=int(res["k_opt"]), cutoff_days_point=float(res["tc_opt"]),
        k_interpolated=int(res["k_day"]), cutoff_days_interpolated=float(res["tc_day"]),
        same_setting=int(bool(res["same_setting"])),
        var_explained=res["var_explained"],
        comment=(
            "sst_recon_point on dates with an observation, sst_recon_day on dates without -- "
            "each date taken from the fit that was cross-validated for its own situation. "
            "Read sst_recon_constrained to see which is which. NOTE the two halves come from "
            "different mode bases, so this field can step discontinuously at the boundary "
            "between a date with data and one without; use a single sst_recon_* channel "
            "instead wherever temporal continuity matters more than per-date accuracy."))
    data["sst_recon"] = da

    # Both tunings in standardized units. Kept per-tuning rather than as one ambiguous
    # `sst_recon_z` because a coarse run's z field is what a full-resolution run warm-starts
    # from, and it has to start from the field fitted at the SAME settings.
    for name, kk, tc in (("point", res["k_opt"], res["tc_opt"]),
                         ("day", res["k_day"], res["tc_day"])):
        zf = res["point_fit"] if name == "point" else res["day_fit"]
        zz = xr.DataArray(scatter(zf["X"]), dims=dims, name=f"sst_recon_z_{name}")
        zz.attrs.update(long_name=f"reconstruction in standardized units, {name}-tuned",
                        units="1", k_modes=int(kk), cutoff_days=float(tc))
        data[f"sst_recon_z_{name}"] = zz


    obs = xr.DataArray(scatter(sel["observed"], dtype="int8", fill=0), dims=dims,
                       name="sst_recon_observed")
    obs.attrs.update(long_name="entry was observed and held fixed", units="1",
                     comment="0 here means the value in sst_recon was reconstructed")
    data["sst_recon_observed"] = obs

    for nm, mask, why in (("sst_recon_cv_day", res["day_cv"], "whole-day holdout, selected p"),
                          ("sst_recon_cv_point", res["point_cv"],
                           "scattered-point holdout, selected k")):
        cv = xr.DataArray(scatter(mask, dtype="int8", fill=0), dims=dims, name=nm)
        cv.attrs.update(long_name=f"cross-validation holdout: {why}", units="1",
                        comment="stored so the CV numbers are reproducible without re-deriving "
                                "them from the seed")
        data[nm] = cv

    dn = sel["observed"].sum(axis=0).astype("int32")
    d1 = xr.DataArray(dn, dims=("time",), name="sst_recon_n_obs")
    d1.attrs.update(long_name="observed pixels contributing on each date", units="1")
    data["sst_recon_n_obs"] = d1

    d2 = xr.DataArray((dn > 0).astype("int8"), dims=("time",), name="sst_recon_constrained")
    d2.attrs.update(
        long_name="date has at least one observation of its own", units="1",
        comment=("0 = the entire day is interpolated by the temporal filter from neighbouring "
                 "dates. Those days are NOT validated by cross-validation, which can only "
                 "score days that have data to hold out."))
    data["sst_recon_constrained"] = d2

    pn = np.zeros(nw, dtype="int32")
    pn[keep] = sel["observed"].sum(axis=1)
    p1 = xr.DataArray(_grid(pn, water, fill=0, dtype="int32"), dims=dims[1:],
                      name="sst_recon_pixel_n_obs")
    p1.attrs.update(long_name="observations per pixel", units="1")
    data["sst_recon_pixel_n_obs"] = p1

    if sel["coarsen"] > 1:
        # Carried channels live on the source grid and would not align with the coarsened one.
        # Rather than block-average each by a rule that depends on what it means, this mode
        # carries only the water mask it actually used. It is a tuning artifact, not a product.
        log.info("coarsen %dx: skipping `carry` (%d channels) -- they are on the source grid",
                 sel["coarsen"], len(cfg["carry"]))
        wm = xr.DataArray(sel["water"].astype("uint8"), dims=dims[1:], name="landcover_water")
        wm.attrs.update(long_name="water mask, block-averaged to the coarsened grid",
                        coarsen_factor=sel["coarsen"])
        data["landcover_water"] = wm
    else:
        for name in cfg["carry"]:
            src = ds[name]
            d_ = xr.DataArray(src.values, dims=src.dims, name=name)
            d_.attrs.update(src.attrs, carried_from=str(cfg["data"]["cube"]))
            data[name] = d_

    coords = {"time": ds["time"].values, **sel["coords"]}
    out = xr.Dataset(data, coords={c: v for c, v in coords.items() if c in dims})
    for v in out.data_vars:
        out[v].encoding = {}
    if "time" in out.coords:
        out["time"].attrs.update(ds["time"].attrs)
    return out


def cube_attrs(cfg: dict, src_attrs: dict, out: xr.Dataset, res: dict, sel: dict) -> dict:
    spec = {k: cfg[k] for k in ("matrix", "filter", "modes", "em", "cv")}
    spec["source_cube"] = str(cfg["data"]["cube"])
    fit = {"k_opt": res["k_opt"], "p_opt": res["p_opt"], "tc_opt": res["tc_opt"],
           "alpha_opt": res["alpha_opt"], "k_day": res["k_day"],
           "tc_day": res["tc_day"], "same_setting": bool(res["same_setting"]),
           "mu": res["mu"],
           "coarsen": sel["coarsen"],
           "m": sel["m"], "n": sel["n"], "var_explained": res["var_explained"],
           "day_cv_by_tc": res["day_at"], "k_by_tc": res["per_tc"],
           "converged": bool(res["history"]["converged"])}
    return {**src_attrs,
            "aoi_id": cfg["data"]["aoi"],
            "dineof_edineof": json.dumps(spec, sort_keys=True, default=str),
            "dineof_edineof_fit": json.dumps(fit, sort_keys=True, default=str),
            "dineof_edineof_channels": json.dumps(sorted(map(str, out.data_vars))),
            "dineof_reconstructed_at": provenance.now_utc(),
            "package_version": provenance.package_version(),
            "code_version": provenance.code_version()}


def write_cube(out: xr.Dataset, cfg: dict, src_attrs: dict, res: dict, sel: dict) -> None:
    dest = cfg["data"]["out"]
    out.attrs.update(cube_attrs(cfg, src_attrs, out, res, sel))
    compression = CompressionSpec(**cfg["output"]["compression"])
    encoding = datacube.build_encoding(out, compression, dict(cfg["output"]["chunks"]))
    store.sweep_scratch(dest)
    if dest.exists():
        log.info("replacing existing cube at %s", dest)
    with store.atomic(dest) as tmp:
        datacube.write_zarr(out, tmp, encoding)
    log.info("wrote %s", dest)


def write_modes(ds: xr.Dataset, cfg: dict, res: dict, sel: dict, path: Path) -> None:
    """(U, sigma, V) scattered back to the grid. The truncated basis is what error maps and
    residual-based outlier tests need, so it is worth keeping beside the field."""
    water, keep = sel["water"], sel["keep"]
    k = res["k_opt"]
    U = np.full((k,) + water.shape, np.nan, dtype="float32")
    tmp = np.full((int(water.sum()), k), np.nan)
    tmp[keep] = res["U"]
    U[:, water] = tmp.T
    out = xr.Dataset(
        {"U": (("mode", "y", "x"), U),
         "sigma": (("mode",), res["sigma"].astype("float32")),
         "V": (("mode", "time"), res["V"].T.astype("float32"))},
        coords={"mode": np.arange(1, k + 1), "time": ds["time"].values, **sel["coords"]})
    out["U"].attrs.update(long_name="spatial modes", comment=(
        "unit-norm columns. With the filter ON these are NOT orthogonal: V diagonalizes "
        "F'X'XF rather than X'X, so U'U is only close to the identity. That is expected, does "
        "not affect the reconstruction, and does make 'EOF' a slight misnomer on this side."))
    out["sigma"].attrs.update(long_name="singular values",
                              convention=cfg["modes"]["sigma_convention"])
    out["V"].attrs.update(long_name="temporal modes",
                          comment="orthonormal; signs pinned by the largest-magnitude entry")
    out.attrs.update(k_opt=k, p_opt=res["p_opt"], alpha=res["alpha_opt"],
                     t_c=res["tc_opt"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(path)
    log.info("wrote %s", path)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML config")
    p.add_argument("--dry-run", action="store_true",
                   help="build the matrix, report its shape and the CV split, write nothing")
    p.add_argument("--k", type=int, default=None, help="skip the k search, use this k")
    p.add_argument("--warm-from", type=Path, default=None,
                   help="a coarser run's cube: take k and T_c from it and start the EM from "
                        "its field upsampled onto this grid")
    p.add_argument("--tc", type=float, default=None,
                   help="skip the filter search, use this cutoff period in days (0 = off)")
    p.add_argument("--tag", default=None,
                   help="suffix the cube, figure dir and CSVs, so variant runs from one "
                        "config land side by side instead of overwriting each other")
    p.add_argument("--no-figures", action="store_true", help="skip the figures")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(args.config)
    if args.k is not None:
        cfg["modes"]["k_grid"] = [args.k]
    if args.tc is not None:
        cfg["filter"]["t_c_grid"] = [args.tc]
    if args.warm_from is not None:
        cfg["matrix"]["warm_start_from"] = str(args.warm_from)
    if args.tag:
        # Applied after load_config so the paths are already absolute. Everything a run
        # produces gets the same suffix, so a comparison never half-overwrites a previous one.
        out = cfg["data"]["out"]
        cfg["data"]["out"] = out.with_name(f"{out.stem}_{args.tag}{out.suffix}")
        fig = cfg["output"]["fig_dir"]
        cfg["output"]["fig_dir"] = fig.with_name(f"{fig.name}_{args.tag}")
        for key in ("modes_file", "report"):
            v = Path(cfg["output"][key])
            cfg["output"][key] = f"{v.stem}_{args.tag}{v.suffix}"
        log.info("tag %r: writing %s", args.tag, cfg["data"]["out"].name)

    if not cfg["data"]["cube"].exists():
        raise SystemExit(
            f"no cube at {cfg['data']['cube']}\n"
            "build it with: python prototypes/DINEOF/src/standardize.py --config "
            "prototypes/DINEOF/configs/config.standardize.admiralty_inlet.yaml")

    ds = xr.open_zarr(cfg["data"]["cube"])
    src_attrs = dict(ds.attrs)
    sel = select_matrix(ds, cfg)
    log.info("%s: matrix %d px x %d dates, %d observations (%.1f%%), %d dates with no data",
             cfg["data"]["aoi"], sel["m"], sel["n"], int(sel["observed"].sum()),
             100 * sel["observed"].mean(), int((sel["observed"].sum(axis=0) == 0).sum()))
    
    rng = np.random.default_rng(int(cfg["cv"]["seed"]))

    if args.dry_run:
        rng = np.random.default_rng(int(cfg["cv"]["seed"]))
        choose_cv_points(sel["observed"],sel["valid_msk"], cfg, rng)
        log.info("dry run: nothing written")
        return

    warm = None
    if cfg["matrix"]["warm_start_from"]:
        warm = load_warm_start(Path(cfg["matrix"]["warm_start_from"]), cfg, sel)
    res = edineof(sel["X"], sel["observed"], sel["valid_msk"] , sel["t"], cfg, warm=warm)
    fields = {}
    for name, fit in (("day", res["day_fit"]), ("point", res["point_fit"])):
        if name == "point" and res["same_setting"]:
            fields["point"] = fields["day"]
            continue
        fields[name] = {"recon": unstandardize(fit["X"], ds, cfg, sel),
                        "lowrank": unstandardize(fit["lowrank"], ds, cfg, sel)}

    out = build_dataset(ds, cfg, res, sel, fields)
    write_cube(out, cfg, src_attrs, res, sel)
    write_modes(ds, cfg, res, sel, cfg["data"]["out"].parent / cfg["output"]["modes_file"])

    path = cfg["data"]["out"].parent / cfg["output"]["report"]
    res["curve"].to_csv(path, index=False)
    log.info("wrote %s", path)

    if cfg["output"]["write_figures"] and not args.no_figures:
        try:
            edineof_figures.render(cfg, res, sel, fields["day"]["recon"], ds,
                                   plotting.extent_km(ds),
                                   cfg["output"]["fig_dir"] / cfg["data"]["aoi"])
        except Exception:
            log.exception("figures failed; the cube and CSVs were written and are intact")


if __name__ == "__main__":
    main()
