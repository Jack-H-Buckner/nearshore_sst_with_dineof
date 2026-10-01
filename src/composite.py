"""Put every sensor on one scale and collapse them into a single SST field per date.

Stage 3+4 of the DINEOF input pipeline. `build_cube.py` produced a cube whose `_dineof`
channels are outlier-filtered but still sit on three different instrument scales; this reads
that cube, fits and removes a per-member offset against the MODIS anchor, and writes
`sst_composite` -- the field DINEOF reconstructs.

WHY AN OFFSET AT ALL. ECOSTRESS overpasses span all 24 hours, Landsat is pinned at ~19:00 UTC,
and MODIS Aqua is a night-time retrieval. ECOSTRESS reads +1.23 K against MODIS on average over
the 50 dates where both see the same water. Averaging the channels as they stand would fold
that instrument-and-time-of-day bias straight into the reconstructed field, and it would enter
as a step wherever the mix of contributing sensors changes from one date to the next.

TWO GATES ON THE DIURNAL TERM, AND BOTH ARE NEEDED:

  cond  -- is the harmonic NUMERICALLY estimable? Landsat's overpasses span 8 minutes, so its
           K=1 design has condition number 1.1e5 and the ladder drops it to a constant. Without
           this it would happily fit a curve to corr(delta, hour) = -0.72, which over an
           8-minute span is seasonal aliasing and nothing else.
  LOO   -- is it JUSTIFIED? ECOSTRESS's hours are so well spread that cond stays at 1.6-3.0 for
           K=1,2,3 and the ladder never fires. Leave-one-out is what notices that K=1 helps
           (RMSE 0.9400 -> 0.9220) while K=2 hurts (0.9327).

THE SLOPE TERM SHIPS OFF. Pooled per-scene-anomaly OLS gives b = 0.639 while the ratio of
anomaly standard deviations is 1.124. MODIS is a ~1 km retrieval nearest-neighbour-resampled
onto this 100 m grid, so the regressor carries resolution error and OLS is attenuated; dividing
by 0.639 would inflate every real SST gradient by 1.56x before DINEOF ever saw it. The true
slope is somewhere in [0.64, 1.12] and is not identifiable without a declared error-variance
ratio. Both estimates are reported in offsets.csv as diagnostics either way.

READ `sst_composite_n` BEFORE YOU TRUST THE AVERAGE. 84.9% of covered pixels have exactly one
member, so the composite is overwhelmingly a union rather than a mean, and 53 of the 185
covered dates are MODIS-only at ~7% water coverage. `sst_composite_src` is a bitmask naming
which members contributed, so the DINEOF step can drop or downweight those columns without
re-running this stage.

This process can also optionally create a training esting split mask based on the avaiabiltiy 
of data on differnt dates. Most days in the data set are only covered by MODIS, which as a 
smaller spatial footprint than landsat or ecostress. On these dates interpolation algoritmhs
wil ned to fill in the rest of the footprint. This optional step will select some days with
a highresoltion overpass and hold the highres data out. This wil produce three channels
a training channel that leave the high res data out of the compsite, a validation channel
that uses every avaibly instrument to form the composite and a make channel that indicates
the areas that were interpolated. This is configured with cfg["validation.frac"]

Usage (from the repo root, in the `coastal_sst_data` env):

    python prototypes/DINEOF/src/composite.py \\
        --config prototypes/DINEOF/configs/config.composite.admiralty_inlet.yaml
    ... --dry-run         # fit and print the offset table, write nothing
    ... --members eco     # fit a subset (the reference is always included)
    ... --no-figures
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[1]              # prototypes/DINEOF
DEFAULT_CONFIG = ROOT / "configs" / "config.composite.admiralty_inlet.yaml"

# Same cross-prototype bridge build_cube.py uses, and for the same reason: the cloud_mixture
# modules import their siblings flatly, so that directory has to be on sys.path first.
_CMM_SRC = ROOT.parent / "cloud_mixture_model" / "src"
if str(_CMM_SRC) not in sys.path:
    sys.path.insert(1, str(_CMM_SRC))

from seasonal_smoothing import diurnal_design, offset_terms  # noqa: E402

import composite_figures                                 # noqa: E402  (DINEOF's own src/)
import plotting                                          # noqa: E402

from coastal_sst_data import provenance, store           # noqa: E402
from coastal_sst_data.config import CompressionSpec      # noqa: E402
from coastal_sst_data.processes import datacube          # noqa: E402

log = logging.getLogger("composite")


# ==================================================================== config

DEFAULTS = {
    "data": {
        "aoi": "admiralty_inlet",
        "cube": "data/datacube/admiralty_inlet_filtered.zarr",
        "out": "data/datacube/admiralty_inlet_composite.zarr",
        "watervar": "landcover_water",
        "date": "time"
    },
    "reference": None,  # SCALAR: the member id every other member is fitted against
    "members": {},      # OPAQUE: one block per member
    "carry": [],        # OPAQUE: a list, not a mapping
    "matchup": {
        "min_pixels": 200,
        "max_abs_delta": 8.0,
        "smooth_to_reference_px": 0,
    },
    "offset": {
        "model": "diurnal",
        "max_harmonics": 1,
        "max_cond": 100.0,
        "loo_gate": True,
        "weights": "variance_components",
        "slope": False,
        "min_scenes": 8,
        "clip": [-5.0, 5.0],
    },
    "composite": {
        "min_members": 1,
        "write_source_bits": True,
    },
    "output": {
        "chunks": {"time": 64, "y": 128, "x": 128},
        "compression": {"codec": "zstd", "level": 5, "shuffle": "shuffle"},
        "offsets": "offsets.csv",
        "matchups": "matchups.csv",
        "write_figures": True,
        "fig_dir": "figures/composite",
        "figure_offsets": True,
        "figure_hexbin": True,
        "figure_coverage": True,
        "figure_scenes": True,
        "figure_contact": True,
        "figure_dpi": 130,
        "figure_ncols": 8,
    },
    "validation":{
        "frac": 0.2,
        "hold": ["lst","eco"],
        "valid_pixels": 0.5
    }
}

MEMBER_DEFAULTS = {
    "label": None,
    "var": None,
    "valid": None,          # null -> the variable's own finite mask is the only gate
    "hour": None,
    "weight": 1.0,
    "diurnal_harmonics": 1,
}

SCALAR_KEYS = {"reference"}
OPAQUE_SECTIONS = {"members", "carry"}

WEIGHT_MODES = ("variance_components", "equal", "n_pixels")
OFFSET_MODELS = ("diurnal", "constant")

# One bit per member in sst_composite_src, so the whole set has to fit in a uint8.
MAX_MEMBERS = 8


def load_config(path: Path, defaults: dict = DEFAULTS) -> dict:
    """Read the YAML config over `defaults`. Unknown sections or keys are an error, so a typo
    cannot silently fall back to a default -- the same contract every script in these two
    prototypes keeps.

    `members` and `carry` are nested deeper than this two-level walk handles (or, for `carry`,
    are a list), so they are taken wholesale and validated separately. `reference` is a bare
    scalar at the top level rather than a section, because it names one of the member ids and
    would read strangely as a one-key mapping.

    Relative paths resolve against ROOT and are then made absolute.
    """
    with open(path) as f:
        user = yaml.safe_load(f) or {}

    cfg = copy.deepcopy(defaults)
    for section, values in user.items():
        if section not in cfg:
            raise ValueError(f"{path}: unknown config section '{section}'")
        if section in SCALAR_KEYS:
            cfg[section] = values
            continue
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

    validate_members(cfg, path)
    validate_carry(cfg, path)
    validate_matchup(cfg, path)
    validate_offset(cfg, path)
    validate_composite(cfg, path)
    validate_validation(cfg)
    return cfg


def validate_members(cfg: dict, path: Path) -> None:
    members = cfg["members"]
    if not isinstance(members, dict) or not members:
        raise ValueError(f"{path}: `members` must be a non-empty mapping of id -> block")
    if len(members) > MAX_MEMBERS:
        raise ValueError(
            f"{path}: {len(members)} members, but sst_composite_src is a uint8 bitmask and "
            f"holds at most {MAX_MEMBERS}")

    for mid, block in members.items():
        merged = copy.deepcopy(MEMBER_DEFAULTS)
        for key, value in (block or {}).items():
            if key not in merged:
                raise ValueError(f"{path}: unknown key 'members.{mid}.{key}'")
            merged[key] = value
        if not merged["var"]:
            raise ValueError(f"{path}: members.{mid}.var is required")
        if not merged["hour"]:
            raise ValueError(
                f"{path}: members.{mid}.hour is required -- the offset model is a function of "
                "overpass hour, and a missing hour would be silently fitted as hour 0")
        if float(merged["weight"]) <= 0:
            raise ValueError(f"{path}: members.{mid}.weight must be positive")
        if int(merged["diurnal_harmonics"]) < 0:
            raise ValueError(f"{path}: members.{mid}.diurnal_harmonics must be >= 0")
        merged["label"] = merged["label"] or mid
        members[mid] = merged

    ref = cfg["reference"]
    if ref not in members:
        raise ValueError(
            f"{path}: reference = {ref!r} is not a member; choose one of {sorted(members)}")


def validate_carry(cfg: dict, path: Path) -> None:
    carry = cfg["carry"]
    if not isinstance(carry, list):
        raise ValueError(f"{path}: `carry` must be a list of channel names")
    for name in carry:
        if not isinstance(name, str):
            raise ValueError(f"{path}: carry entries must be strings, got {name!r}")
    if len(set(carry)) != len(carry):
        raise ValueError(f"{path}: duplicate entries in `carry`")
    for mid, m in cfg["members"].items():
        for key in ("var", "hour"):
            if m[key] in carry:
                raise ValueError(
                    f"{path}: carry lists {m[key]!r}, which is already written as "
                    f"members.{mid}'s {key}; drop it from `carry`")


def validate_matchup(cfg: dict, path: Path) -> None:
    mt = cfg["matchup"]
    if int(mt["min_pixels"]) < 1:
        raise ValueError(f"{path}: matchup.min_pixels must be >= 1")
    if float(mt["max_abs_delta"]) <= 0:
        raise ValueError(f"{path}: matchup.max_abs_delta must be positive")
    k = int(mt["smooth_to_reference_px"])
    if k < 0 or k == 1:
        raise ValueError(f"{path}: matchup.smooth_to_reference_px must be 0 (off) or >= 2")


def validate_offset(cfg: dict, path: Path) -> None:
    o = cfg["offset"]
    if o["model"] not in OFFSET_MODELS:
        raise ValueError(f"{path}: offset.model must be one of {OFFSET_MODELS}")
    if o["weights"] not in WEIGHT_MODES:
        raise ValueError(f"{path}: offset.weights must be one of {WEIGHT_MODES}")
    if int(o["max_harmonics"]) < 0:
        raise ValueError(f"{path}: offset.max_harmonics must be >= 0")
    if float(o["max_cond"]) <= 1:
        raise ValueError(f"{path}: offset.max_cond must be > 1")
    if int(o["min_scenes"]) < 1:
        raise ValueError(f"{path}: offset.min_scenes must be >= 1")
    clip = o["clip"]
    if (not isinstance(clip, (list, tuple)) or len(clip) != 2
            or float(clip[0]) >= float(clip[1])):
        raise ValueError(f"{path}: offset.clip must be [lower, upper] with lower < upper")


def validate_composite(cfg: dict, path: Path) -> None:
    c = cfg["composite"]
    n = int(c["min_members"])
    if not 1 <= n <= len(cfg["members"]):
        raise ValueError(
            f"{path}: composite.min_members must be in [1, {len(cfg['members'])}], got {n}")



def validate_validation(cfg: dict) -> None:
    c = cfg["validation"]
    n = c["frac"] 
    if not 0 <= n <= 1.0:
        raise ValueError(
            f"{path}: v.frac must be btween zero and one, got {n}")


def validate_channels(cfg: dict, ds: xr.Dataset, path: Path) -> None:
    """Every configured channel must exist in the cube, checked up front."""
    available = sorted(map(str, ds.data_vars))
    wanted = [("data.watervar", cfg["data"]["watervar"])]
    for mid, m in cfg["members"].items():
        wanted += [(f"members.{mid}.{k}", m[k]) for k in ("var", "valid", "hour")
                   if m[k] is not None]
    wanted += [(f"carry[{i}]", n) for i, n in enumerate(cfg["carry"])]
    for where, name in wanted:
        if name not in ds:
            raise ValueError(
                f"{path}: {where} = {name!r} is not in the cube; available: {available}")


def validate_nonempty(cfg: dict, stacks: dict[str, np.ndarray], path: Path) -> None:
    """A member with no finite water pixels anywhere is a hard error.

    Without this the member fits an offset against nothing, contributes an all-NaN channel, and
    the failure only shows up as a composite that is quietly thinner than expected. This is not
    hypothetical: `mur_sst` sits in the source cube with 0 finite water pixels across all 365
    days, and reads as a perfectly plausible member name.
    """
    for mid, arr in stacks.items():
        n = int(np.isfinite(arr).sum())
        if n == 0:
            raise ValueError(
                f"{path}: members.{mid} ({cfg['members'][mid]['var']}) has no finite water "
                "pixels anywhere in the cube; it would fit an offset against nothing and "
                "contribute an all-NaN channel")
        log.info("%s: %d finite water px on %d dates", mid, n,
                 int((np.isfinite(arr).sum(axis=(1, 2)) > 0).sum()))


# ==================================================================== member stacks

def member_stack(ds: xr.Dataset, cfg: dict, mid: str, water: np.ndarray) -> np.ndarray:
    """The member's SST as (time, y, x) float32, NaN off-water and where its QC rejects it."""
    m = cfg["members"][mid]
    arr = ds[m["var"]].values.astype("float32")
    keep = np.isfinite(arr) & water[None, :, :]
    if m["valid"] is not None:
        keep &= np.asarray(ds[m["valid"]].values) > 0.5
    return np.where(keep, arr, np.nan).astype("float32")



def member_hours(ds: xr.Dataset, cfg: dict, mid: str) -> np.ndarray:
    return np.asarray(ds[cfg["members"][mid]["hour"]].values, dtype=float)


def box_mean(a: np.ndarray, k: int) -> np.ndarray:
    """NaN-aware k x k box mean of a single 2-D field.

    Used by `matchup.smooth_to_reference_px` to degrade a fine member to the reference's own
    support before differencing. Degrading the FINE instrument is the right direction: MODIS is
    the coarse one, so comparing at its support removes resolution mismatch from the residual
    instead of pretending the coarse field resolves what it does not. Measured effect on this
    AoI: within-scene sd falls 0.339 -> 0.241 K while the offset itself is unchanged.
    """
    from scipy.ndimage import uniform_filter

    ok = np.isfinite(a)
    filled = np.where(ok, a, 0.0)
    num = uniform_filter(filled, size=k, mode="constant", cval=0.0)
    den = uniform_filter(ok.astype(float), size=k, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    return np.where(den > 0, out, np.nan)


# ==================================================================== matchups

def scene_matchups(mem: np.ndarray, ref: np.ndarray, hours: np.ndarray,
                   times: np.ndarray, cfg: dict, mid: str
                   ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """One robust difference per date where the member and the reference see the same water.

    Returns (per-scene table, pooled reference values, pooled member values). The pooled arrays
    are kept for the slope diagnostics and the before/after figure.

    The centre is a plain MEDIAN and the spread a plain MAD -- deliberately NOT
    `outlier_detection.robust_centre_scale`, which is otherwise the house estimator here. Two
    reasons. Its unweighted scale is warm-side by construction
    (`median(res - centre | res > centre) / 0.6745`), which is right for a residual field whose
    contamination is one-sided and cold but wrong for a two-sided member-minus-reference
    difference on pixels the detector has already cleaned. And below `min_n=64` it silently
    returns `(0.0, 1.0)` -- a fabricated zero offset indistinguishable from a measured one.
    Empirically the choice barely matters (corr(median, shorth) = 0.991 over these 50 scenes),
    so take the estimator without the hidden fallback.
    """
    mt = cfg["matchup"]
    min_px = int(mt["min_pixels"])
    max_abs = float(mt["max_abs_delta"])
    k = int(mt["smooth_to_reference_px"])

    rows, pooled_ref, pooled_mem = [], [], []
    n_rejected = 0
    for t in range(mem.shape[0]):
        m2 = box_mean(mem[t], k) if k else mem[t]
        ok = np.isfinite(m2) & np.isfinite(ref[t])
        n = int(ok.sum())
        if n < min_px:
            continue
        x, y = ref[t][ok].astype(float), m2[ok].astype(float)
        d = y - x
        delta = float(np.median(d))
        sd = float(1.4826 * np.median(np.abs(d - delta)))
        if abs(delta) > max_abs:
            n_rejected += 1
            log.warning("%s: %s rejected from the fit, |delta| = %.2f K exceeds "
                        "matchup.max_abs_delta = %.2f",
                        mid, str(times[t])[:10], abs(delta), max_abs)
            continue
        rows.append(dict(member=mid, date=str(times[t])[:10], t=t, n=n,
                         delta=delta, sd=sd, hour=float(hours[t]),
                         mem_mean=float(y.mean()), ref_mean=float(x.mean())))
        pooled_ref.append(x)
        pooled_mem.append(y)

    table = pd.DataFrame(rows, columns=["member", "date", "t", "n", "delta", "sd", "hour",
                                        "mem_mean", "ref_mean"])
    log.info("%s: %d matchup scenes, %d px%s", mid, len(table), int(table["n"].sum()),
             f" ({n_rejected} rejected on max_abs_delta)" if n_rejected else "")
    ref_px = np.concatenate(pooled_ref) if pooled_ref else np.empty(0)
    mem_px = np.concatenate(pooled_mem) if pooled_mem else np.empty(0)
    return table, ref_px, mem_px


# ==================================================================== the offset fit

def vc_weights(delta: np.ndarray, sd: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-scene weights 1/(tau2 + sd^2/n), and tau2.

    tau2 is the BETWEEN-scene variance -- real day-to-day differences in the offset -- estimated
    by method of moments from the constant-model residual: tau2 = Var(delta) - mean(sd^2/n).

    This is the whole weighting argument in one number. On this AoI tau2 ~ 0.86 K^2 against a
    median within-scene variance of ~0.004, a factor of 200, so the weights collapse to
    near-equal and each scene counts once. That is the right answer, and it is why
    `weights: n_pixels` is wrong rather than merely different: weighting a scene by its pixel
    count asserts a precision that the between-scene scatter says does not exist, and it is
    algebraically the pooled per-pixel regression written the long way round (verified: two-
    level sqrt(n) gives a constant of 1.091, pooled per-pixel 1.094, against 1.232 equal).
    """
    v = np.asarray(sd, float) ** 2 / np.maximum(np.asarray(n, float), 1.0)
    tau2 = float(max(np.var(np.asarray(delta, float), ddof=1) - v.mean(), 0.0)) \
        if delta.size > 1 else 0.0
    return 1.0 / (tau2 + v), tau2


def scene_weights(table: pd.DataFrame, mode: str) -> tuple[np.ndarray, float]:
    if mode == "equal":
        return np.ones(len(table)), float("nan")
    if mode == "n_pixels":
        return table["n"].to_numpy(float), float("nan")
    return vc_weights(table["delta"].to_numpy(), table["sd"].to_numpy(),
                      table["n"].to_numpy())


def wls(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    sw = np.sqrt(w)
    return np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]


def loo_rmse(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Leave-one-out RMSE of a weighted least-squares fit, in closed form.

    For WLS the deleted residual is r_i / (1 - h_ii) with h_ii = w_i x_i' (X'WX)^-1 x_i, so no
    refitting loop is needed. Scored UNWEIGHTED, because the question is how well the model
    predicts an arbitrary held-out scene, not a precision-weighted one.
    """
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    coefs = np.linalg.lstsq(Xw, y * sw, rcond=None)[0]
    r = y - X @ coefs
    try:
        h = np.einsum("ij,jk,ik->i", Xw, np.linalg.pinv(Xw.T @ Xw), Xw)
    except np.linalg.LinAlgError:
        return float("inf")
    denom = np.clip(1.0 - h, 1e-8, None)
    return float(np.sqrt(np.mean((r / denom) ** 2)))


def choose_harmonics(hours: np.ndarray, y: np.ndarray, w: np.ndarray, K0: int,
                     cfg: dict, name: str) -> tuple[int, float, dict]:
    """Pick K under both gates, and return the LOO curve that justified the choice.

    The two gates answer different questions and neither substitutes for the other -- see the
    module docstring. cond runs first because a design that cannot be solved cannot be scored.
    """
    o = cfg["offset"]
    max_cond = float(o["max_cond"])
    h = np.nan_to_num(hours)
    K = K0

    # Gate 1: numerical identifiability.
    while K > 0:
        X = diurnal_design(h, K)
        if X.shape[0] >= X.shape[1] + 1 and np.linalg.cond(X * np.sqrt(w)[:, None]) <= max_cond:
            break
        log.info("%s: overpass hours cannot identify %d diurnal harmonic(s) "
                 "(cond %.3g > %.3g); trying %d",
                 name, K, np.linalg.cond(diurnal_design(h, K) * np.sqrt(w)[:, None]),
                 max_cond, K - 1)
        K -= 1

    curve = {k: loo_rmse(diurnal_design(h, k), y, w) for k in range(0, K + 1)}

    # Gate 2: statistical justification. Step down while a simpler model predicts as well.
    if o["loo_gate"]:
        while K > 0 and curve[K] >= curve[K - 1]:
            log.info("%s: K=%d does not beat K=%d out of sample (LOO RMSE %.4f vs %.4f); "
                     "stepping down", name, K, K - 1, curve[K], curve[K - 1])
            K -= 1

    cond = float(np.linalg.cond(diurnal_design(h, K) * np.sqrt(w)[:, None]))
    return K, cond, curve


def slope_diagnostics(ref_px: np.ndarray, mem_px: np.ndarray,
                      table: pd.DataFrame) -> tuple[float, float, float]:
    """(OLS slope, RMA slope, correlation) on per-scene-anomaly pairs.

    Both slopes are reported and NEITHER is applied by default. Pooling anomalies removes the
    per-scene offset, which is what makes the two comparable across dates, but it does nothing
    about errors-in-variables: the regressor is a ~1 km retrieval on a 100 m grid, so OLS is
    attenuated toward zero and RMA (the ratio of standard deviations) is its upper counterpart.
    The honest statement is that the true slope lies between them, and on this AoI that is
    [0.64, 1.12] -- an interval wide enough that picking either end distorts the anomaly field
    more than leaving the slope at 1.
    """
    if ref_px.size == 0:
        return float("nan"), float("nan"), float("nan")
    xa, ya, i = [], [], 0
    for n in table["n"].to_numpy(int):
        x, y = ref_px[i:i + n], mem_px[i:i + n]
        xa.append(x - x.mean())
        ya.append(y - y.mean())
        i += n
    x, y = np.concatenate(xa), np.concatenate(ya)
    if x.size < 2 or x.std() == 0:
        return float("nan"), float("nan"), float("nan")
    return (float(np.polyfit(x, y, 1)[0]), float(y.std() / x.std()),
            float(np.corrcoef(x, y)[0, 1]))


def fit_offset(table: pd.DataFrame, ref_px: np.ndarray, mem_px: np.ndarray,
               cfg: dict, mid: str) -> dict:
    """Fit one member's offset model. Returns coefficients plus everything offsets.csv reports."""
    m = cfg["members"][mid]
    o = cfg["offset"]
    name = m["label"]

    b_ols, b_rma, corr = slope_diagnostics(ref_px, mem_px, table)
    base = dict(member=mid, label=name, var=m["var"], reference=False,
                n_scenes=len(table), n_pixels=int(table["n"].sum()),
                slope_ols=b_ols, slope_rma=b_rma, anomaly_corr=corr)

    if len(table) == 0:
        raise ValueError(
            f"members.{mid} has no matchup scene against the reference (min_pixels = "
            f"{cfg['matchup']['min_pixels']}); it cannot be placed on the reference's scale")

    y = table["delta"].to_numpy(float)
    w, tau2 = scene_weights(table, o["weights"])
    hours = table["hour"].to_numpy(float)

    K0 = 0 if o["model"] == "constant" else min(int(m["diurnal_harmonics"]),
                                                int(o["max_harmonics"]))
    if len(table) < int(o["min_scenes"]) and K0 > 0:
        log.warning("%s: only %d matchup scenes (< offset.min_scenes = %d); forcing a "
                    "constant offset", name, len(table), int(o["min_scenes"]))
        K0 = 0

    K, cond, curve = choose_harmonics(hours, y, w, K0, cfg, name)
    coefs = wls(diurnal_design(np.nan_to_num(hours), K), y, w)

    fitted = offset_at(coefs, K, hours)
    resid = y - fitted
    lo, hi = float(o["clip"][0]), float(o["clip"][1])
    if not (lo <= fitted.min() and fitted.max() <= hi):
        raise ValueError(
            f"members.{mid}: fitted offset spans [{fitted.min():+.2f}, {fitted.max():+.2f}] K, "
            f"outside offset.clip [{lo:+.2f}, {hi:+.2f}]. Either the matchups are contaminated "
            "or the model is extrapolating; inspect matchups.csv before widening the clip")

    if o["slope"]:
        log.warning(
            "%s: offset.slope is ON. The pooled anomaly slope is %.3f and the RMA slope %.3f; "
            "dividing by the former inflates every spatial anomaly by %.2fx. MODIS is ~1 km "
            "resampled onto a 100 m grid, so OLS here is attenuated by resolution error, not "
            "measuring a real gain. This is a diagnostic mode.", name, b_ols, b_rma,
            1.0 / b_ols if b_ols else float("nan"))

    stats = dict(base, K=K, cond=cond, tau2=tau2, weights=o["weights"],
                 coefs=offset_terms(coefs, K),
                 delta_mean=float(y.mean()), delta_sd=float(y.std(ddof=1)) if len(y) > 1 else 0.0,
                 delta_min=float(y.min()), delta_max=float(y.max()),
                 within_scene_sd=float(table["sd"].median()),
                 offset_mean=float(fitted.mean()),
                 offset_min=float(fitted.min()), offset_max=float(fitted.max()),
                 resid_sd=float(resid.std(ddof=1)) if len(resid) > 1 else 0.0,
                 loo_rmse=curve.get(K, float("nan")),
                 loo_rmse_constant=curve.get(0, float("nan")))
    log.info("%s: K=%d (cond %.3g), offset %+.3f K [%+.3f, %+.3f], "
             "delta sd %.3f -> resid sd %.3f, LOO %.4f (K=0: %.4f)",
             name, K, cond, fitted.mean(), fitted.min(), fitted.max(),
             stats["delta_sd"], stats["resid_sd"], stats["loo_rmse"],
             stats["loo_rmse_constant"])
    return dict(coefs=coefs, K=K, stats=stats, curve=curve, weights=w,
                ref_px=ref_px, mem_px=mem_px)


def reference_fit(cfg: dict, mid: str) -> dict:
    """The anchor's own fit: identically zero, by definition, recorded so it appears in the
    CSV beside the others rather than as a gap a reader has to explain to themselves."""
    m = cfg["members"][mid]
    return dict(coefs=np.zeros(1), K=0, curve={},
                weights=np.empty(0), ref_px=np.empty(0), mem_px=np.empty(0),
                stats=dict(member=mid, label=m["label"], var=m["var"], reference=True,
                           n_scenes=0, n_pixels=0, K=0, coefs="mean=0.0000",
                           offset_mean=0.0, offset_min=0.0, offset_max=0.0))


# ==================================================================== apply

def offset_at(coefs: np.ndarray, K: int, hours: np.ndarray) -> np.ndarray:
    """The modelled offset at each hour. Dates with no recorded hour fall back to the constant
    term rather than being evaluated at hour 0, which would be a silent 3 a.m. prediction."""
    h = np.asarray(hours, float)
    off = diurnal_design(np.nan_to_num(h), K) @ coefs
    return np.where(np.isfinite(h) | (K == 0), off, coefs[0])


def apply_offset(stack: np.ndarray, hours: np.ndarray, fit: dict,
                 cfg: dict, mid: str) -> tuple[np.ndarray, np.ndarray]:
    """(corrected stack, per-date offset). The offset channel is NaN on dates the member did
    not see, so it reads as what was actually removed rather than as a model evaluated into
    the void."""
    off = offset_at(fit["coefs"], fit["K"], hours)
    if cfg["offset"]["slope"] and not fit["stats"].get("reference"):
        b = fit["stats"]["slope_ols"]
        adj = (stack - off[:, None, None]) / b
    else:
        adj = stack - off[:, None, None]

    seen = np.isfinite(stack).any(axis=(1, 2))
    n_nohour = int((~np.isfinite(hours) & seen).sum())
    if n_nohour and fit["K"] > 0:
        log.warning("%s: %d dates with data have no overpass hour; the constant term was used",
                    mid, n_nohour)
    return adj.astype("float32"), np.where(seen, off, np.nan).astype("float32")

# ==================================================================== validation
# step 1:  Identify dates that can be used in the validation. 
# step 1a: Add list of high res instruments to hold for validation 
# step 2:  Select dates to use for validation.
# step 3:  build composite with highres data removed on validaiton dates
# step 4:  build mask indicating where data was removed 
# step 5:  build composite without 

def identify_highres_dates(ds: xr.Dataset, cfg: dict) -> dict:
    """Find dates with highres retrievals with sufficent valid observations.

    The ECOSTESS and Landsat have differnt footprints than MODIS, often covering areas MODIS
    misses. To properly validate the reconstructions on days without ECOSTESS and Landsat 
    date the algoritmh needs to be tuned to capture these gaps. This function finds dates
    with ECOSTESS and/ or Landsat data that are candidates for the validaiton set. 
    """
    dates = ds[cfg["data"]["date"]]
    dat_inds = np.asarray(range(len(dates)))
    valid_dates = []
    valid_inds = []
    for var in cfg["validation"]["hold"]:
        channel = cfg["members"][var]["var"]
        water = cfg["data"]["watervar"]
        n_valid = ds[channel].notnull().sum(dim=["y", "x"])
        n_water = ds[water].sum(dim=["y", "x"])
        p_valid = n_valid/n_water
        valid = np.asarray(p_valid > cfg["validation"]["valid_pixels"])
        valid_dates.append(dates[valid])
        valid_inds.append(dat_inds[valid])

    var_dates = dict(zip(cfg["validation"]["hold"],valid_dates))
    var_inds = dict(zip(cfg["validation"]["hold"],valid_inds))
    return var_dates, var_inds


def select_highres_dates(var_inds: dict,cfg: dict) -> np.array:
    """Select dates to leave highres data out of compisite for validation.

    Select a random subsample of dates of size cfg["validation]["frac"]
    """
    unique_inds = np.unique(np.concatenate(list(var_inds.values())))
    k = int(len(unique_inds) * cfg["validation"]["frac"])
    sampled_inds = np.random.choice(unique_inds.ravel(), k)
    return sampled_inds



# ==================================================================== composite
def composite(adj: dict[str, np.ndarray], cfg: dict, order: list[str]) -> dict:
    """The per-date field: a weighted mean over whichever members saw each pixel.

    `order` fixes the bit assignment in sst_composite_src, so a bit means the same member for
    the life of the cube. Members are iterated in config order for that reason.

    READ THE COUNT. On this AoI 84.9% of covered pixels have exactly one member, so for most of
    the cube this is a union and the weights never come into play. The 53 MODIS-only dates are
    the case that matters downstream: they are the only constraint DINEOF gets on those dates,
    at ~7% water coverage and 1 km support, and the bitmask is what lets a later stage find
    them without re-running this one.
    """
    w = {mid: float(cfg["members"][mid]["weight"]) for mid in order}
    min_members = int(cfg["composite"]["min_members"])
    shape = adj[order[0]].shape

    num = np.zeros(shape, "float64")
    den = np.zeros(shape, "float64")
    cnt = np.zeros(shape, "uint8")
    src = np.zeros(shape, "uint8")

    for bit, mid in enumerate(order):
        ok = np.isfinite(adj[mid])
        num += np.where(ok, np.nan_to_num(adj[mid]) * w[mid], 0.0)
        den += np.where(ok, w[mid], 0.0)
        cnt += ok.astype("uint8")
        src |= (ok.astype("uint8") << bit)

    enough = cnt >= min_members
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = num / den
    comp = np.where(enough, mean, np.nan).astype("float32")

    # Weighted sample spread across members, with the reliability correction that reduces to
    # the familiar n-1 denominator when the weights are equal. Only defined where two members
    # actually met -- 15.1% of covered pixels here.
    ss = np.zeros(shape, "float64")
    sw2 = np.zeros(shape, "float64")
    for mid in order:
        ok = np.isfinite(adj[mid])
        ss += np.where(ok, w[mid] * (np.nan_to_num(adj[mid]) - np.nan_to_num(mean)) ** 2, 0.0)
        sw2 += np.where(ok, w[mid] ** 2, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        denom = den - sw2 / den
        sd = np.sqrt(ss / denom)
    sd = np.where(enough & (cnt >= 2) & (denom > 0), sd, np.nan).astype("float32")

    covered = int((cnt > 0).sum())
    hist = {k: int((cnt[cnt > 0] == k).sum()) for k in range(1, len(order) + 1)}
    log.info("composite: %d covered px on %d dates; members per px %s",
             int(np.isfinite(comp).sum()), int((np.isfinite(comp).any(axis=(1, 2))).sum()),
             {k: f"{100 * v / covered:.2f}%" for k, v in hist.items() if v})
    return dict(sst=comp, n=np.where(enough, cnt, 0).astype("uint8"), sd=sd,
                src=np.where(enough, src, 0).astype("uint8"), order=order)


# ==================================================================== composite
def composite_with_validation(adj: dict[str, np.ndarray], valid_inds: np.array, cfg: dict, order: list[str]) -> dict:
    """The per-date field with mask: a weighted mean over whichever members saw each pixel.

    This is largely the same as the composite function but it also builds a masked composite
    that leaves the high resolution data out on selected dates to use as a validation set. 

    The full composite for the validation, the masked composite for training and the masked values 
    are all returned in the output dictionary. 
    """

    # build list of arrays with masking on validation dates
    print(valid_inds)
    adj_msk={}
    hold = cfg["validation"]["hold"]
    for mid in order:
        adj_msk[mid] = adj[mid].copy()
        if mid in hold:
            adj_msk[mid][valid_inds,:,:] = np.nan


    w = {mid: float(cfg["members"][mid]["weight"]) for mid in order}
    min_members = int(cfg["composite"]["min_members"])
    shape = adj[order[0]].shape

    num = np.zeros(shape, "float64")
    den = np.zeros(shape, "float64")
    cnt = np.zeros(shape, "uint8")
    src = np.zeros(shape, "uint8")

    for bit, mid in enumerate(order):
        ok = np.isfinite(adj[mid])
        num += np.where(ok, np.nan_to_num(adj[mid]) * w[mid], 0.0)
        den += np.where(ok, w[mid], 0.0)
        cnt += ok.astype("uint8")
        src |= (ok.astype("uint8") << bit)

    enough = cnt >= min_members
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = num / den
    
    comp = np.where(enough, mean, np.nan).astype("float32")

    num_msk = np.zeros(shape, "float64")
    den_msk = np.zeros(shape, "float64")
    cnt_msk = np.zeros(shape, "uint8")
    src_msk = np.zeros(shape, "uint8")

    for bit, mid in enumerate(order):
        ok_msk = np.isfinite(adj_msk[mid]) 
        num_msk += np.where(ok_msk, np.nan_to_num(adj_msk[mid]) * w[mid], 0.0)
        den_msk += np.where(ok_msk, w[mid], 0.0)
        cnt_msk += ok_msk.astype("uint8")
        src_msk |= (ok_msk.astype("uint8") << bit)

    enough_msk = cnt_msk >= 1 #min_members
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_msk = num_msk / den_msk


    comp_msk = np.where(enough_msk, mean_msk, np.nan).astype("float32")

    msk = np.isnan(comp_msk) & np.isfinite(comp)


    # Weighted sample spread across members, with the reliability correction that reduces to
    # the familiar n-1 denominator when the weights are equal. Only defined where two members
    # actually met -- 15.1% of covered pixels here.
    ss = np.zeros(shape, "float64")
    sw2 = np.zeros(shape, "float64")
    for mid in order:
        ok = np.isfinite(adj[mid])
        ss += np.where(ok, w[mid] * (np.nan_to_num(adj[mid]) - np.nan_to_num(mean)) ** 2, 0.0)
        sw2 += np.where(ok, w[mid] ** 2, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        denom = den - sw2 / den
        sd = np.sqrt(ss / denom)
    sd = np.where(enough & (cnt >= 2) & (denom > 0), sd, np.nan).astype("float32")

    covered = int((cnt > 0).sum())
    hist = {k: int((cnt[cnt > 0] == k).sum()) for k in range(1, len(order) + 1)}
    log.info("composite: %d covered px on %d dates; members per px %s",
             int(np.isfinite(comp).sum()), int((np.isfinite(comp).any(axis=(1, 2))).sum()),
             {k: f"{100 * v / covered:.2f}%" for k, v in hist.items() if v})
    return dict(sst=comp, sst_msk = comp_msk, msk=msk, n=np.where(enough, cnt, 0).astype("uint8"), sd=sd,
                src=np.where(enough, src, 0).astype("uint8"), order=order)


# ==================================================================== assemble

def build_dataset(ds: xr.Dataset, cfg: dict, adj: dict, offsets: dict,
                  comp: dict) -> xr.Dataset:
    """The composite cube.

    Members are keyed by their CONFIG ID, not by their source channel name: threading source
    names through three stages produces `eco_sst_v002_dineof_adj`, and this is a new cube, so
    it is free to name things for what they are. `source_channel` in the attrs is the link back.
    """
    dims = ("time", "y", "x")
    data = {}

    for mid in comp["order"]:
        m = cfg["members"][mid]
        st = offsets[mid]["stats"]
        da = xr.DataArray(adj[mid], dims=dims, name=f"{mid}_adj")
        da.attrs.update(
            long_name=f"{m['label']} SST, offset-corrected",
            units=ds[m["var"]].attrs.get("units", "K"),
            source_channel=m["var"],
            reference_member=cfg["reference"],
            offset_model="reference (identically zero)" if st.get("reference")
            else f"diurnal K={st['K']}",
            offset_coefficients=st["coefs"],
            offset_n_scenes=int(st["n_scenes"]),
            comment=("the source channel minus the fitted offset against "
                     f"{cfg['members'][cfg['reference']]['label']}; see "
                     f"{mid}_offset for the value removed on each date"))
        data[f"{mid}_adj"] = da

        off = xr.DataArray(offsets[mid]["offset"], dims=("time",), name=f"{mid}_offset")
        off.attrs.update(long_name=f"{m['label']} offset removed", units="K",
                         applies_to=f"{mid}_adj",
                         comment="NaN on dates this member did not observe")
        data[f"{mid}_offset"] = off

    # SST data
    sst = xr.DataArray(comp["sst"], dims=dims, name="sst_composite")
    sst.attrs.update(
        long_name="composite SST validation set",
        units=ds[cfg["members"][cfg["reference"]]["var"]].attrs.get("units", "K"),
        members=json.dumps(comp["order"]),
        member_weights=json.dumps({m: float(cfg["members"][m]["weight"])
                                   for m in comp["order"]}),
        min_members=int(cfg["composite"]["min_members"]),
        reference_member=cfg["reference"],
        comment=("weighted per-pixel mean of the <id>_adj channels. THIS IS THE DINEOF VALIDATION SET. "))
    data["sst_composite"] = sst

    # Masked SST data
    sst_msk = xr.DataArray(comp["sst_msk"], dims=dims, name="sst_composite")
    sst_msk.attrs.update(
        long_name="composite SST training data.",
        units=ds[cfg["members"][cfg["reference"]]["var"]].attrs.get("units", "K"),
        members=json.dumps(comp["order"]),
        member_weights=json.dumps({m: float(cfg["members"][m]["weight"])
                                    for m in comp["order"]}),
        min_members=int(cfg["composite"]["min_members"]),
        reference_member=cfg["reference"],
        comment=("weighted per-pixel mean of the <id>_adj channels with eco and lst data removed on validation dates. THIS IS THE DINEOF INPUT. "))
    data["sst_msk_composite"] = sst_msk 

    # Masked points data
    msk = xr.DataArray(comp["msk"], dims=dims, name="masked observations")
    msk.attrs.update(
        long_name="Pixels masked for cross validation.",
        comment=("Pixels with only lst or eco data points on masked dates."))
    data["validation_msk"] = msk


    n = xr.DataArray(comp["n"], dims=dims, name="sst_composite_n")
    n.attrs.update(long_name="number of members contributing to sst_composite", units="1")
    data["sst_composite_n"] = n

    sd = xr.DataArray(comp["sd"], dims=dims, name="sst_composite_sd")
    sd.attrs.update(long_name="spread across contributing members", units="K",
                    comment="defined only where sst_composite_n >= 2; a cross-sensor "
                            "disagreement estimate, not a retrieval uncertainty")
    data["sst_composite_sd"] = sd

    if cfg["composite"]["write_source_bits"]:
        bits = xr.DataArray(comp["src"], dims=dims, name="sst_composite_src")
        bits.attrs.update(
            long_name="bitmask of members contributing to sst_composite", units="1",
            flag_masks=json.dumps([1 << i for i in range(len(comp["order"]))]),
            flag_meanings=" ".join(comp["order"]),
            comment=("bit i is set where member i of `flag_meanings` contributed. Lets a "
                     "downstream stage drop or downweight, say, MODIS-only dates without "
                     "re-running this one."))
        data["sst_composite_src"] = bits

    for name in cfg["carry"]:
        src = ds[name]
        da = xr.DataArray(src.values, dims=src.dims, name=name)
        da.attrs.update(src.attrs, carried_from=str(cfg["data"]["cube"]))
        data[name] = da

    out = xr.Dataset(data, coords={c: ds[c].values for c in dims if c in ds.coords})

    # Carried channels still hold the SOURCE store's encoding, which conflicts with the
    # encoding built at write time and silently wins. Cleared on data_vars ONLY -- the time
    # coord's units/calendar must survive or the axis is rewritten as plain integers.
    for v in out.data_vars:
        out[v].encoding = {}
    if "time" in out.coords:
        out["time"].attrs.update(ds["time"].attrs)
    return out


def cube_attrs(cfg: dict, src_attrs: dict, out: xr.Dataset, offsets: dict) -> dict:
    spec = {"reference": cfg["reference"],
            "members": {k: dict(v) for k, v in cfg["members"].items()},
            "matchup": cfg["matchup"],
            "offset": cfg["offset"],
            "composite": cfg["composite"],
            "carry": list(cfg["carry"]),
            "source_cube": str(cfg["data"]["cube"])}
    fitted = {mid: {k: f["stats"][k] for k in ("K", "coefs", "n_scenes", "offset_mean")
                    if k in f["stats"]} for mid, f in offsets.items()}
    return {**src_attrs,
            "aoi_id": cfg["data"]["aoi"],
            "dineof_composite": json.dumps(spec, sort_keys=True, default=str),
            "dineof_composite_offsets": json.dumps(fitted, sort_keys=True, default=str),
            "dineof_composite_channels": json.dumps(sorted(map(str, out.data_vars))),
            "dineof_composited_at": provenance.now_utc(),
            "package_version": provenance.package_version(),
            "code_version": provenance.code_version()}


def write_cube(out: xr.Dataset, cfg: dict, src_attrs: dict, offsets: dict) -> None:
    """Atomic write through the package's zarr layer. `store.atomic` is driven here rather than
    via `datacube.write_zarr_safe` because the source store stays open across the write."""
    dest = cfg["data"]["out"]
    out.attrs.update(cube_attrs(cfg, src_attrs, out, offsets))
    compression = CompressionSpec(**cfg["output"]["compression"])
    encoding = datacube.build_encoding(out, compression, dict(cfg["output"]["chunks"]))

    store.sweep_scratch(dest)
    if dest.exists():
        log.info("replacing existing cube at %s", dest)
    with store.atomic(dest) as tmp:
        datacube.write_zarr(out, tmp, encoding)
    log.info("wrote %s", dest)


# ==================================================================== driver
def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML config")
    p.add_argument("--members", nargs="+", default=None,
                   help="subset of the member blocks; the reference is always included")
    p.add_argument("--dry-run", action="store_true",
                   help="fit and print the offset table, write nothing")
    p.add_argument("--no-figures", action="store_true", help="skip the figures")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(args.config)
    ref = cfg["reference"]

    order = list(cfg["members"])
    if args.members is not None:
        unknown = [m for m in args.members if m not in cfg["members"]]
        if unknown:
            raise SystemExit(f"--members names undefined blocks: {unknown}")
        order = [m for m in order if m in set(args.members) | {ref}]

    if not cfg["data"]["cube"].exists():
        raise SystemExit(
            f"no cube at {cfg['data']['cube']}\n"
            "build it with: python prototypes/DINEOF/src/build_cube.py --config "
            "prototypes/DINEOF/configs/config.build_cube.admiralty_inlet.yaml")

    ds = xr.open_zarr(cfg["data"]["cube"])
    validate_channels(cfg, ds, args.config)
    src_attrs = dict(ds.attrs)
    water = np.asarray(ds[cfg["data"]["watervar"]].compute() > 0.5)
    times = pd.to_datetime(ds["time"].values)
    log.info("%s: %d water px, members %s, reference %s",
             cfg["data"]["aoi"], int(water.sum()), order, ref)

    stacks = {mid: member_stack(ds, cfg, mid, water) for mid in order}
    hours = {mid: member_hours(ds, cfg, mid) for mid in order}
    validate_nonempty(cfg, stacks, args.config)

    # -------------------------------------------------- stage 3: offsets
    offsets, tables = {}, []
    for mid in order:
        if mid == ref:
            offsets[mid] = reference_fit(cfg, mid)
            continue
        table, ref_px, mem_px = scene_matchups(
            stacks[mid], stacks[ref], hours[mid], times, cfg, mid)
        offsets[mid] = fit_offset(table, ref_px, mem_px, cfg, mid)
        offsets[mid]["table"] = table
        tables.append(table)

    report = pd.DataFrame([offsets[m]["stats"] for m in order])
    matchups = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()

    if args.dry_run:
        with pd.option_context("display.width", 240, "display.max_columns", None):
            print(report.to_string(index=False))
        log.info("dry run: nothing written")
        return

    # -------------------------------------------------- stage 4: composite
    adj = {}
    for mid in order:
        adj[mid], offsets[mid]["offset"] = apply_offset(
            stacks[mid], hours[mid], offsets[mid], cfg, mid)
    comp = composite(adj, cfg, order)

    out = build_dataset(ds, cfg, adj, offsets, comp)
    write_cube(out, cfg, src_attrs, offsets)

    out_dir = cfg["data"]["out"].parent
    out_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_dir / cfg["output"]["offsets"], index=False)
    matchups.to_csv(out_dir / cfg["output"]["matchups"], index=False)
    log.info("wrote %s and %s", out_dir / cfg["output"]["offsets"],
             out_dir / cfg["output"]["matchups"])

    if cfg["output"]["write_figures"] and not args.no_figures:
        # After the cube and the CSVs, and isolated: a drawing bug must not cost a build that
        # already succeeded. Everything drawn is reproducible from the two CSVs and the cube.
        try:
            composite_figures.render(cfg, offsets, comp, adj, order, times, water,
                                     plotting.extent_km(ds),
                                     cfg["output"]["fig_dir"] / cfg["data"]["aoi"])
        except Exception:
            log.exception("figures failed; the cube and CSVs were written and are intact")

import matplotlib.pyplot as plt
import random
def scratch_pad(argv=None)->None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML config")
    p.add_argument("--members", nargs="+", default=None,
                   help="subset of the member blocks; the reference is always included")
    p.add_argument("--dry-run", action="store_true",
                   help="fit and print the offset table, write nothing")
    p.add_argument("--no-figures", action="store_true", help="skip the figures")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(args.config)
    ref = cfg["reference"]

    order = list(cfg["members"])
    if args.members is not None:
        unknown = [m for m in args.members if m not in cfg["members"]]
        if unknown:
            raise SystemExit(f"--members names undefined blocks: {unknown}")
        order = [m for m in order if m in set(args.members) | {ref}]

    if not cfg["data"]["cube"].exists():
        raise SystemExit(
            f"no cube at {cfg['data']['cube']}\n"
            "build it with: python prototypes/DINEOF/src/build_cube.py --config "
            "prototypes/DINEOF/configs/config.build_cube.admiralty_inlet.yaml")

    ds = xr.open_zarr(cfg["data"]["cube"])
    validate_channels(cfg, ds, args.config)
    src_attrs = dict(ds.attrs)
    water = np.asarray(ds[cfg["data"]["watervar"]].compute() > 0.5)
    times = pd.to_datetime(ds["time"].values)
    log.info("%s: %d water px, members %s, reference %s",
             cfg["data"]["aoi"], int(water.sum()), order, ref)

    stacks = {mid: member_stack(ds, cfg, mid, water) for mid in order}
    hours = {mid: member_hours(ds, cfg, mid) for mid in order}
    validate_nonempty(cfg, stacks, args.config)

    # -------------------------------------------------- stage 3: offsets
    offsets, tables = {}, []
    for mid in order:
        if mid == ref:
            offsets[mid] = reference_fit(cfg, mid)
            continue
        table, ref_px, mem_px = scene_matchups(
            stacks[mid], stacks[ref], hours[mid], times, cfg, mid)
        offsets[mid] = fit_offset(table, ref_px, mem_px, cfg, mid)
        offsets[mid]["table"] = table
        tables.append(table)

    report = pd.DataFrame([offsets[m]["stats"] for m in order])
    matchups = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()

    if args.dry_run:
        with pd.option_context("display.width", 240, "display.max_columns", None):
            print(report.to_string(index=False))
        log.info("dry run: nothing written")
        return



    dates,inds = identify_highres_dates(ds, cfg)
    valid_inds = select_highres_dates(inds, cfg)
    all_dates = ds[cfg["data"]["date"]]

    adj = {}
    for mid in order:
        adj[mid], offsets[mid]["offset"] = apply_offset(
            stacks[mid], hours[mid], offsets[mid], cfg, mid)

    comp = composite_with_validation(adj, valid_inds, cfg, order)
    out = build_dataset(ds, cfg, adj, offsets, comp)
    write_cube(out, cfg, src_attrs, offsets)

    # out_dir = cfg["data"]["out"].parent
    # out_dir.mkdir(parents=True, exist_ok=True)
    # report.to_csv(out_dir / cfg["output"]["offsets"], index=False)
    # matchups.to_csv(out_dir / cfg["output"]["matchups"], index=False)
    # log.info("wrote %s and %s", out_dir / cfg["output"]["offsets"],
    #          out_dir / cfg["output"]["matchups"])

    # if cfg["output"]["write_figures"] and not args.no_figures:
    #     # After the cube and the CSVs, and isolated: a drawing bug must not cost a build that
    #     # already succeeded. Everything drawn is reproducible from the two CSVs and the cube.
    #     try:
    #         composite_figures.render(cfg, offsets, comp, adj, order, times, water,
    #                                  plotting.extent_km(ds),
    #                                  cfg["output"]["fig_dir"] / cfg["data"]["aoi"])
    #     except Exception:
    #         log.exception("figures failed; the cube and CSVs were written and are intact")


if __name__ == "__main__":
    scratch_pad()
