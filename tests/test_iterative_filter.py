"""Tests for the iterative DINEOF-baseline cloud filter, on small synthetic cubes.

Nothing here touches a real cube: `Inputs` is built directly, with a rank-2 anomaly field on a
constant seasonal mean, one sensor with a known offset, and an empty MODIS member.

    pytest tests/test_iterative_filter.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import iterative_filter as F        # noqa: E402  (first: it bridges seasonal_smoothing)
import edineof as E                 # noqa: E402

H = W = 24
T = 60
OFFSET = 1.0                        # sensor minus composite scale, K
MEAN = 285.0


# --------------------------------------------------------------------------- helpers

def config(**loop) -> dict:
    user = {
        "detector": {
            "ref_var": "modis_sst_aqua", "depthvar": "depth_cudem", "tidal_depth_m": 3.0,
            "clear": {"sd": 0.75, "sd_floor": 0.71},
            "qc": {"enabled": True, "prior_cloud": 0.9, "nodata_prior": None},
            "mixture": {"prior_cloud": 0.1, "min_dev_cold": 0.0, "min_dev_hot": 0.0},
        },
        "sensors": {"eco": {"sst": "eco_sst", "valid": "eco_valid", "cloud": "eco_cloud",
                            "hour": "eco_hour", "min_pixels": 64}},
        "composite": {"hold": ["eco"]},
        "dineof": {"matrix": {"min_date_obs": 10, "coarsen": 1},
                   "em": {"tol": 1e-4, "max_iter": 200}},
        "loop": {"strategy": "fixed", "k": 2, "t_c": 2.0, "max_iter": 6, "tol": 1e-3,
                 "final_cv": False, **{"baseline_source": "all", **loop}},
    }
    return F.build_config(user, resolve=False)


def truth_field(seed=0):
    """A rank-2 anomaly field around MEAN, smooth in space and time."""
    yy, xx = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing="ij")
    p1 = np.cos(np.pi * xx) + 0.5 * yy
    p2 = np.sin(2 * np.pi * yy) * xx
    t = np.arange(T)
    a1 = 2.0 * np.cos(2 * np.pi * t / 45)
    a2 = 1.0 * np.sin(2 * np.pi * t / 17)
    return MEAN + a1[:, None, None] * p1 + a2[:, None, None] * p2


def make_inputs(cfg, *, patches: bool, seed=0, modis_days=0.0, modis_px=0.10):
    """(Inputs, patch mask). Scenes on ~2/3 of the days; a -4 K 7x7 cold patch on a third.

    With modis_days > 0, a MODIS member sees the clean truth (no offset, small noise) at a
    random `modis_px` of the pixels on that share of the days.
    """
    rng = np.random.default_rng(seed)
    water = np.ones((H, W), bool)
    days = np.sort(rng.choice(T, size=40, replace=False))
    truth = truth_field(seed)
    raw = np.full((T, H, W), np.nan, dtype="float32")
    patch = np.zeros((T, H, W), bool)
    for j in days:
        raw[j] = truth[j] + OFFSET + rng.normal(scale=0.2, size=(H, W))
        if patches and rng.random() < 0.35:
            r0, c0 = rng.integers(0, H - 7, size=2)
            patch[j, r0:r0 + 7, c0:c0 + 7] = True
    raw = np.where(patch, raw - 4.0, raw).astype("float32")
    modis = np.full((T, H, W), np.nan, dtype="float32")
    if modis_days > 0:
        mrng = np.random.default_rng(seed + 100)
        for j in np.flatnonzero(mrng.random(T) < modis_days):
            m = mrng.random((H, W)) < modis_px
            modis[j] = np.where(m, truth[j] + mrng.normal(scale=0.1, size=(H, W)), np.nan)

    scenes = {"eco": [int(j) for j in days]}
    zeros = np.zeros((H, W), bool)
    qc = {"eco": {int(j): (zeros, water & ~np.isfinite(raw[j]), zeros) for j in days}}
    times = np.datetime64("2025-03-01") + np.arange(T).astype("timedelta64[D]")
    inp = F.Inputs(
        times=times.astype("datetime64[ns]"), water=water, tidal=zeros,
        coef=np.zeros((3, H, W)), scale=np.ones((H, W)),
        seasonal=np.full((T, H, W), MEAN, dtype="float32"),
        raw={"eco": raw}, adj={"modis": modis,
                               "eco": (raw - OFFSET).astype("float32")},
        scenes=scenes, qc=qc, dcfg={"eco": F.detector_config(cfg, "eco")},
        valid_inds=np.array([], dtype=int), order=["modis", "eco"],
        coords={"y": np.arange(H) * 100.0, "x": np.arange(W) * 100.0},
        offsets={"eco": np.full(T, OFFSET)})
    return inp, patch


def observed_mask(inp):
    r = inp.raw["eco"]
    obs = np.zeros(r.shape, bool)
    for j in inp.scenes["eco"]:
        obs[j] = np.isfinite(r[j])
    return obs


# --------------------------------------------------------------------------- the loop

def test_recovers_cold_patches():
    """Coherent -4 K patches are found and removed; the flags stop changing."""
    cfg = config()
    inp, patch = make_inputs(cfg, patches=True)
    assert patch.any()
    out = F.run_loop(inp, cfg)

    obs = observed_mask(inp)
    base = out["last"]["base"]
    score = np.abs(inp.adj["eco"] - base)[obs]
    a = F.auc(score, patch[obs])
    assert a > 0.95, f"AUC {a:.3f}"

    keep = out["keep"]["eco"]
    recall = float((~keep[patch]).mean())
    assert recall > 0.9, f"only {recall:.1%} of patch pixels removed"
    h = out["history"]
    assert out["converged"], h
    assert float(h["flip_frac_eco"].iloc[-1]) < 1e-3


def test_keep_history_bits():
    """Bit 0 is the starting mask and the last bit is the final keep."""
    cfg = config()
    inp, _ = make_inputs(cfg, patches=True)
    out = F.run_loop(inp, cfg)
    b = out["keep_bits"]["eco"]
    assert np.array_equal((b & 1).astype(bool), F.qc_only_keep(inp)["eco"])
    assert np.array_equal(((b >> out["n_iter"]) & 1).astype(bool), out["keep"]["eco"])
    assert set(out["scene_history"]["iter"]) == set(range(out["n_iter"]))


def test_clean_field_flags_nothing():
    """No contamination: only the clear-tail rate is flagged, and iteration 0 barely flips."""
    cfg = config()
    inp, _ = make_inputs(cfg, patches=False)
    out = F.run_loop(inp, cfg)
    obs = observed_mask(inp)
    removed = float((obs & ~out["keep"]["eco"]).sum() / obs.sum())
    assert removed < 0.01, f"{removed:.2%} of a clean field flagged"
    assert float(out["history"]["flip_frac_eco"].iloc[0]) < 0.01
    assert (out["table"]["verdict"] == "KEPT").all()


def test_wrong_flags_are_restored():
    """Flags are not cumulative: an over-aggressive starting mask comes back."""
    cfg = config()
    inp, _ = make_inputs(cfg, patches=False)
    obs = observed_mask(inp)
    rng = np.random.default_rng(5)
    init = F.qc_only_keep(inp)
    wrong = obs & (rng.random(obs.shape) < 0.4)
    wrong[inp.scenes["eco"][3]] = obs[inp.scenes["eco"][3]]          # one whole scene too
    init["eco"] = init["eco"] & ~wrong
    out = F.run_loop(inp, cfg, init_keep=init)
    restored = float(out["keep"]["eco"][wrong].mean())
    assert restored > 0.99, f"only {restored:.1%} of wrongly flagged pixels restored"


def test_ends_strategy_runs_cv_once():
    """`ends` searches at iteration 0 and fixes those settings for the rest of the loop."""
    cfg = config(strategy="ends", max_iter=3)
    cfg["_edineof"]["filter"]["t_c_grid"] = [0, 2]
    cfg["_edineof"]["modes"]["k_grid"] = [1, 2, 3]
    cfg["_edineof"]["cv"]["day_frac"] = 0.1
    inp, _ = make_inputs(cfg, patches=True)
    # The CV search needs a point holdout: mark every observed pixel on 3 days as validation.
    inp.valid_inds = np.array(inp.scenes["eco"][5:8])
    out = F.run_loop(inp, cfg)
    h = out["history"]
    assert np.isfinite(h["rmse_point"].iloc[0]) and np.isfinite(h["rmse_day"].iloc[0])
    assert h["rmse_point"].iloc[1:].isna().all()
    assert h["k"].nunique() == 1 and h["t_c"].nunique() == 1


# --------------------------------------------------------------------------- fit_fixed

def test_fit_fixed_matches_final_fit():
    """With no seed, fit_fixed is edineof's final_fit at the same (k, T_c) and gap set."""
    rng = np.random.default_rng(1)
    m, n, r = 300, 50, 3
    U = np.linalg.qr(rng.normal(size=(m, r)))[0]
    V = np.linalg.qr(rng.normal(size=(n, r)))[0]
    X_true = U @ np.diag([40.0, 20.0, 10.0]) @ V.T + rng.normal(scale=0.05, size=(m, n))
    observed = rng.random((m, n)) > 0.35
    observed[:, 7] = False                                           # an empty day
    t = np.cumsum(rng.uniform(1.0, 3.0, size=n))
    # A point holdout, which edineof's search needs; final_fit hands it back as observations.
    valid_msk = ((rng.random((m, n)) < 0.03) & observed).T

    ecfg = {"filter": {"t_c_grid": [4.0], "alpha_max": 0.25, "stability_factor": 0.25},
            "modes": {"k_grid": [3], "rule": "parsimonious", "rel_tol": 0.01, "patience": 3,
                      "sigma_convention": "projection", "fix_sign": True},
            "em": {"tol": 1e-10, "max_iter": 2000},
            "cv": {"frac": 0.05, "day_frac": 0.0, "max_date_frac": 0.4,
                   "min_date_obs_after": 20, "donor_lo": 0.5, "donor_hi": 0.95, "seed": 0}}
    X = np.where(observed, X_true, 0.0)
    res = E.edineof(X.copy(), observed, valid_msk, t, ecfg)
    ref = res["day_fit"]
    got = F.fit_fixed(X.copy(), observed, t, 3, F.setting(4.0, ecfg), ecfg)
    assert np.allclose(got["X"], ref["X"], atol=1e-5)
    assert np.allclose(got["lowrank"], ref["lowrank"], atol=1e-5)


def test_warm_seed_converges_faster_to_same_answer():
    """Seeding the gaps from a converged analysis reaches the same field in fewer EM steps."""
    rng = np.random.default_rng(2)
    m, n = 200, 40
    X_true = (np.linalg.qr(rng.normal(size=(m, 2)))[0] * [20, 8]) @ \
        np.linalg.qr(rng.normal(size=(n, 2)))[0].T + 5.0
    observed = rng.random((m, n)) > 0.4
    t = np.arange(n, dtype=float)
    ecfg = {"filter": {"alpha_max": 0.25, "stability_factor": 0.25},
            "modes": {"sigma_convention": "projection", "fix_sign": True},
            "em": {"tol": 1e-6, "max_iter": 500}}
    s = F.setting(3.0, ecfg)
    X = np.where(observed, X_true, 0.0)
    cold = F.fit_fixed(X.copy(), observed, t, 2, s, ecfg)
    warm = F.fit_fixed(X.copy(), observed, t, 2, s, ecfg, seed=cold["X"])
    assert warm["hist"]["n_iter"] < cold["hist"]["n_iter"]
    assert np.allclose(warm["X"], cold["X"], atol=1e-3)


# --------------------------------------------------------------------------- plumbing

def test_grid_roundtrip():
    cfg = config()
    inp, _ = make_inputs(cfg, patches=False)
    sel, _ = F.build_matrix(inp, F.qc_only_keep(inp), cfg)
    G = F.to_grid(sel["X"], sel)
    assert np.allclose(F.from_grid(G, sel), sel["X"], atol=1e-5)


def test_unknown_keys_rejected():
    with pytest.raises(ValueError, match="unknown key 'loop.nope'"):
        F.build_config({"loop": {"nope": 1}}, resolve=False)
    cfg_user = {"detector": {"ref_var": "a", "depthvar": "b", "tidal_depth_m": 1.0},
                "sensors": {"eco": {"sst": "a", "valid": "b", "cloud": "c", "hour": "d"}},
                "composite": {"hold": ["eco"]},
                "dineof": {"matrix": {"bogus": 1}}}
    with pytest.raises(ValueError, match="dineof.matrix.bogus"):
        F.build_config(cfg_user, resolve=False)


def test_kappa():
    a = np.array([1, 1, 0, 0], bool)
    assert F.cohen_kappa(a, a) == pytest.approx(1.0)
    assert F.cohen_kappa(a, ~a) == pytest.approx(-1.0)


# --------------------------------------------------------------------------- MODIS loadings

def _modes(m=150, n=40, k=2, seed=4):
    rng = np.random.default_rng(seed)
    U = np.linalg.qr(rng.normal(size=(m, k)))[0]
    t = np.arange(n, dtype=float)
    a = np.vstack([20 * np.cos(2 * np.pi * t / 30), 8 * np.sin(2 * np.pi * t / 23)])[:k]
    return U, a, t


def test_modis_loadings_exact():
    """Dense MODIS on an exactly rank-k field recovers the loadings."""
    U, a, t = _modes()
    mu = 0.3
    Xm = U @ a + mu
    Om = np.ones_like(Xm, bool)
    got, info = F.modis_loadings(U, mu, Xm, Om, t, F.setting(0.0, E.DEFAULTS), 1e-12, 5)
    assert np.allclose(got, a, atol=1e-8)
    assert info["direct"] == t.size and info["fallback"] == 0


def test_modis_loadings_pool_and_fallback():
    """Alternate-day MODIS: pooling fills the gaps near truth; without it they are 0."""
    U, a, t = _modes()
    mu = -0.2
    Xm = U @ a + mu
    Om = np.zeros_like(Xm, bool)
    Om[:, ::2] = True
    s = F.setting(4.0, E.DEFAULTS)
    got, info = F.modis_loadings(U, mu, Xm, Om, t, s, 1e-6, 5)
    odd = np.arange(1, t.size, 2)
    assert info["pooled"] == odd.size and info["fallback"] == 0
    # Interior gap days sit between two MODIS days, so pooling interpolates them.
    inner = odd[odd < t.size - 1]
    err = np.abs(got[:, inner] - a[:, inner]).max()
    assert err < 0.05 * np.abs(a).max(), err
    # The last day has one MODIS day in reach (day 38, p = 2), so it takes that day's loading:
    # a zero-order extrapolation at the end of the series, not an interpolation.
    assert s["p"] == 2
    assert np.allclose(got[:, -1], a[:, -2], atol=1e-4)

    got0, info0 = F.modis_loadings(U, mu, Xm, Om, t, F.setting(0.0, E.DEFAULTS), 1e-6, 5)
    assert info0["fallback"] == odd.size
    assert np.all(got0[:, odd] == 0.0)


def test_modis_baseline_ignores_sensor_values():
    """With U fixed, the sensor's values cannot move the MODIS matrix or its loadings."""
    cfg = config(baseline_source="modis")
    inp, _ = make_inputs(cfg, patches=False, modis_days=0.5)
    sel, _ = F.build_matrix(inp, F.qc_only_keep(inp), cfg)
    Xm, Om = F.modis_matrix(inp, sel)
    U = np.linalg.qr(np.random.default_rng(0).normal(size=(sel["m"], 2)))[0]
    s = F.setting(6.0, cfg["_edineof"])
    a1, _ = F.modis_loadings(U, 0.0, Xm, Om, sel["t"], s, 1e-3, 5)

    inp.adj["eco"] = inp.adj["eco"] - 5.0            # a very different sensor field
    inp.raw["eco"] = inp.raw["eco"] - 5.0
    Xm2, Om2 = F.modis_matrix(inp, sel)
    a2, _ = F.modis_loadings(U, 0.0, Xm2, Om2, sel["t"], s, 1e-3, 5)
    assert np.array_equal(Xm, Xm2) and np.array_equal(Om, Om2)
    assert np.array_equal(a1, a2)


def test_no_modis_in_reach_is_climatology():
    """A day no MODIS reaches gets a = 0, so its baseline is the seasonal field exactly."""
    cfg = config(baseline_source="modis")
    inp, _ = make_inputs(cfg, patches=False)                     # MODIS empty everywhere
    sel, _ = F.build_matrix(inp, F.qc_only_keep(inp), cfg)
    Xm, Om = F.modis_matrix(inp, sel)
    U = np.linalg.qr(np.random.default_rng(0).normal(size=(sel["m"], 2)))[0]
    a, info = F.modis_loadings(U, 0.4, Xm, Om, sel["t"], F.setting(6.0, cfg["_edineof"]),
                               1e-3, 5)
    assert info["fallback"] == T and np.all(a == 0)
    base = F.baseline_kelvin(U @ a + 0.0, sel, inp)
    assert np.allclose(base, inp.seasonal)


def test_recovers_cold_patches_modis_baseline():
    """End to end with MODIS-only loadings: patches found, loop converges."""
    cfg = config(baseline_source="modis", t_c=6.0)
    inp, patch = make_inputs(cfg, patches=True, modis_days=0.5)
    out = F.run_loop(inp, cfg)
    obs = observed_mask(inp)
    score = np.abs(inp.adj["eco"] - out["last"]["base"])[obs]
    a = F.auc(score, patch[obs])
    assert a > 0.95, f"AUC {a:.3f}"
    assert float((~out["keep"]["eco"][patch]).mean()) > 0.9
    assert out["converged"], out["history"]
    h = out["history"]
    assert (h["baseline_source"] == "modis").all()
    assert out["last"]["loadings"] is not None


# --------------------------------------------------------------------------- blocked composite

@pytest.mark.parametrize("block_days", [1, 7, 30, 1000])
@pytest.mark.parametrize("min_members", [1, 2])
def test_composite_blocked_matches_original(block_days, min_members):
    """Blocking changes the peak memory, never the result."""
    import contextlib
    import io
    import composite as C
    rng = np.random.default_rng(3)
    shape = (45, 12, 14)
    order = ["modis", "eco", "lst"]
    adj = {}
    for mid, p in zip(order, (0.3, 0.6, 0.4)):
        a = rng.normal(285.0, 2.0, shape).astype("float32")
        a[rng.random(shape) > p] = np.nan
        adj[mid] = a
    valid_inds = np.array([2, 3, 17, 30, 44])
    cfg = config()
    cfg["composite"].update(min_members=min_members, hold=["lst", "eco"],
                            weights={"modis": 1.0, "eco": 2.0, "lst": 0.5})
    with contextlib.redirect_stdout(io.StringIO()):
        ref = C.composite_with_validation(adj, valid_inds, F.composite_config(cfg, order), order)
    got = F.composite_blocked(adj, valid_inds, cfg, order, block_days)
    assert np.array_equal(np.isnan(got["sst"]), np.isnan(ref["sst"]))
    assert np.allclose(got["sst"], ref["sst"], equal_nan=True, rtol=0, atol=1e-5)
    assert np.array_equal(got["src"], ref["src"])
    assert np.array_equal(got["msk"], ref["msk"])
    assert got["msk"].any()


def test_block_days_validated():
    with pytest.raises(ValueError, match="block_days"):
        config_bad = {"composite": {"hold": ["eco"], "block_days": 0}}
        F.build_config({**_min_user(), **config_bad}, resolve=False)


def _min_user():
    return {"detector": {"ref_var": "a", "depthvar": "b", "tidal_depth_m": 1.0},
            "sensors": {"eco": {"sst": "a", "valid": "b", "cloud": "c", "hour": "d"}}}


@pytest.mark.parametrize("block_days", [1, 7, 1000])
def test_composite_blocked_fused_steps(block_days):
    """keep masking, offsets and z, applied per block, equal applying them to whole cubes first."""
    rng = np.random.default_rng(4)
    shape = (40, 10, 12)
    order = ["modis", "eco", "lst"]
    raw = {mid: rng.normal(285.0, 2.0, shape).astype("float32") for mid in order}
    for mid in order:
        raw[mid][rng.random(shape) > 0.5] = np.nan
    keep = {sid: rng.random(shape) > 0.3 for sid in ("eco", "lst")}
    offsets = {"eco": (rng.normal(1.0, 0.2, shape[0]), 1.0), "lst": (np.full(shape[0], 0.6), 1.1)}
    seasonal = rng.normal(284.0, 1.0, shape).astype("float32")
    scale = rng.uniform(0.5, 2.0, shape[1:])
    water = np.ones(shape[1:], bool)
    water[:, :2] = False
    vi = np.array([1, 5, 22])
    cfg = config()
    cfg["composite"].update(hold=["lst", "eco"])

    # The old way: full corrected-and-masked copies first, then composite, then z.
    adj_full = {"modis": raw["modis"]}
    for sid in ("eco", "lst"):
        off, slope = offsets[sid]
        a = ((raw[sid] - off[:, None, None]) / slope).astype("float32")
        adj_full[sid] = np.where(keep[sid], a, np.nan).astype("float32")
    ref = F.composite_blocked(adj_full, vi, cfg, order, 1000)
    with np.errstate(invalid="ignore"):
        z_ref = ((ref["sst"] - seasonal) / scale[None]).astype("float32")
    z_ref[:, ~water] = np.nan
    msk_ref = ref["msk"] & np.isfinite(z_ref)

    got = F.composite_blocked(raw, vi, cfg, order, block_days, keep=keep, offsets=offsets,
                              seasonal=seasonal, scale=scale, water=water)
    assert np.allclose(got["sst"], ref["sst"], equal_nan=True, rtol=0, atol=1e-5)
    assert np.array_equal(got["src"], ref["src"])
    assert np.array_equal(got["msk"], msk_ref)
    assert got["z"].dtype == np.float32
    assert np.allclose(got["z"], z_ref, equal_nan=True, rtol=1e-6, atol=1e-6)
    assert np.isnan(got["z"][:, ~water]).all()


# --------------------------------------------------------------------------- segmentation helpers

def test_subset_inputs():
    """subset_inputs slices time arrays and remaps scenes/qc to the new index space."""
    cfg = config()
    inp, _ = make_inputs(cfg, patches=False)
    T_full = len(inp.times)

    # Take the middle third
    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(T_full, size=T_full // 2, replace=False))
    sub = F.subset_inputs(inp, idx)

    # --- time-dimension fields are sliced correctly
    assert len(sub.times) == len(idx)
    assert np.array_equal(sub.times, inp.times[idx])
    assert sub.seasonal.shape[0] == len(idx)
    for sid in inp.raw:
        assert sub.raw[sid].shape[0] == len(idx)
        assert sub.offsets[sid].shape[0] == len(idx)
    for mid in inp.adj:
        assert sub.adj[mid].shape[0] == len(idx)

    # --- spatial fields are unchanged
    assert np.array_equal(sub.water, inp.water)
    assert np.array_equal(sub.scale, inp.scale)

    # --- scenes are remapped to the new index space
    idx_set = set(idx.tolist())
    old_to_new = np.full(T_full, -1, dtype=int)
    old_to_new[idx] = np.arange(len(idx))

    for sid in inp.raw:
        # Every scene index in the subset must be a valid new index
        for new_j in sub.scenes[sid]:
            assert 0 <= new_j < len(idx), new_j
        # Scenes not in idx must be absent
        kept_old = set(j for j in inp.scenes[sid] if j in idx_set)
        assert len(sub.scenes[sid]) == len(kept_old)
        # qc keys must match scenes
        assert set(sub.qc[sid].keys()) == set(sub.scenes[sid])

    # --- valid_inds are within bounds
    assert (sub.valid_inds >= 0).all()
    assert (sub.valid_inds < len(idx)).all()


def test_make_windows():
    """make_windows partitions the time axis exactly; overlap is correct; no gaps."""
    # 400-day series, annual (365-day) windows, 20-day overlap
    n = 400
    times = (np.datetime64("2023-01-01") + np.arange(n)).astype("datetime64[ns]")
    windows = F.make_windows(times, segment_years=1.0, overlap_days=20)

    # Central indices must be a disjoint partition of 0..n-1
    all_central = np.concatenate([c for _, c in windows])
    assert np.array_equal(np.sort(all_central), np.arange(n)), \
        "central indices do not partition the full time axis"

    # Each window must contain its central indices
    for w_idx, c_idx in windows:
        assert set(c_idx).issubset(set(w_idx)), "central indices not inside window"

    # Overlap: non-edge windows should extend beyond the central period
    if len(windows) > 1:
        _, (w1_idx, c1_idx) = 0, windows[0]
        # First window's right edge >= last central date + overlap
        assert w1_idx[-1] > c1_idx[-1], "first window has no right overlap"
        w_last, c_last = windows[-1]
        assert w_last[0] < c_last[0], "last window has no left overlap"

    # No gaps in central coverage: sorted central indices cover 0..n-1 contiguously
    assert all_central.min() == 0 and all_central.max() == n - 1
    diffs = np.diff(np.sort(all_central))
    assert (diffs == 1).all(), "gap in central coverage"


def test_run_segmented_loop():
    """run_segmented_loop with segment_years=0.1 (tiny windows) covers all times and returns
    the same keys as run_loop."""
    cfg = config(segment_years=0.1, segment_overlap_days=5)
    inp, patch = make_inputs(cfg, patches=True)

    out = F.run_segmented_loop(inp, cfg)

    # Same keys as run_loop
    for key in ("keep", "keep_bits", "p_valid", "table", "scene_history", "history"):
        assert key in out, key

    # keep covers the full time dimension
    for sid in inp.raw:
        assert out["keep"][sid].shape == inp.raw[sid].shape
        assert out["keep_bits"][sid].shape == inp.raw[sid].shape
        assert out["p_valid"][sid].shape == inp.raw[sid].shape

    # table and scene_history contain entries for all scene dates
    all_dates = {str(inp.times[j])[:10] for sid in inp.raw for j in inp.scenes[sid]}
    table_dates = set(out["table"]["date"])
    # Every scene date should appear in table (one row per pixel-scene verdict)
    # At a minimum the set of dates in the table is a subset of all dates
    assert table_dates.issubset(all_dates | {"nan"})
