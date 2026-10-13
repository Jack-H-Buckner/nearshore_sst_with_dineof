"""Region configuration and output layout for the three-stage pipeline.

One YAML per region (`configs/region.<aoi>.yaml`; see `configs/region.template.yaml`):

  region       aoi, source cube, out_dir, time_range, watervar, carry
  offsets      STAGE 1: footprint matchups, outlier clipping, model selection
  detector, sensors, reference, filter, composite, holdout, seasonal, scale, dineof, loop,
  final, output
               STAGE 2: exactly the end-to-end pipeline's sections (pipeline.py)
  validation   STAGE 3: in-situ source and matching (validate_insitu.py)

Outputs land under <out_dir>/<aoi>[_tag]/{stage1_offsets, stage2_dineof, stage3_validation}.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import yaml

import pipeline as P
import validate_insitu as V

ROOT = P.ROOT

REGION_DEFAULTS = {
    "aoi": None,                    # required
    "source": None,                 # required: the coastal_sst_data cube
    "out_dir": "data/regions",
    "watervar": "landcover_water",
    "time_range": None,
    "carry": ["landcover_water", "depth_cudem", "elevation_cudem"],
}

# Stage 1. The matchup keys are pipeline.py's `matchup` section; the rest select the model.
MATCHUP_KEYS = tuple(P.DEFAULTS["matchup"])
OFFSETS_DEFAULTS = {
    **{k: v for k, v in P.DEFAULTS["matchup"].items()},
    "mode": "auto",                 # auto (stage 1's recommendation) | fixed | rma
    "recommend_min_gain": 0.02,     # rma must beat fixed's LOSO RMSE by this share
    "slope_clip": [0.5, 2.0],       # an RMA slope outside this cannot be chosen
    "n_boot": 200,                  # scene-bootstrap draws for the slope interval
}
OFFSET_MODES = ("auto", "fixed", "rma")

VALIDATION_DEFAULTS = {
    "insitu": None,                 # netCDF / CSV path; null = the cube's own insitu_* channels
    "products": list(V.DEFAULTS["products"]),
    "match": dict(V.DEFAULTS["match"]),
    "reader": copy.deepcopy(V.DEFAULTS["insitu"]),
    "dpi": 130,
}

# Stage 4 (optional): point-level state-space smoother. `groups` maps a site group name to its
# model config; see state_space.py and validate_point_filter.
POINT_FILTER_DEFAULTS = {
    "points": None,                 # CSV of site_id, lat, lon, group; null disables the stage
    "period_days": 365.25,          # seasonal period for the harmonic models
    "max_snap_m": 300.0,            # a site on land snaps to the nearest water pixel within this
    "dpi": 130,
    "groups": {},                   # name -> {model, mode, sources, n_harmonics, share, ...}
    # Optional diagnostic: smooth each in-situ site (from validation.insitu, no CSV) on DINEOF
    # sources only, then fit a POST-HOC bias (insitu - smoothed) and evaluate it out-of-sample by
    # leave-one-site-out. Enabled on its own; `points` may be null.
    "insitu_eval": {
        "enabled": False,
        "model": "random_walk",     # smoother model (DINEOF-only)
        "mode": "hierarchy",
        "sources": ["observed", "point_fill", "day_fill"],   # DINEOF types only, no `insitu`
        "n_harmonics": 0,           # smoother harmonics (for ar1_harmonic / harmonic_trend)
        "share": [],
        "bias_form": "auto",        # constant | seasonal | auto (auto -> chosen by `selection`)
        "bias_n_harmonics": 2,      # harmonics of the post-hoc seasonal bias
        "selection": "aic",         # when bias_form is auto: aic | loso_cv
        "min_gain": 0.0,            # AIC units the seasonal form must beat the constant form by
    },
}

# Stage 5 (optional): calibrate the DINEOF fields against in-situ. A per-product additive offset
# (constant or seasonal) is fit on the in-situ matchups and applied to produce calibrated fields.
CALIBRATION_DEFAULTS = {
    "enabled": False,               # include in `all` only when true (always runnable on its own)
    "products": list(V.DEFAULTS["products"]),
    "match": "daily_mean",          # matchup basis: daily_mean | overpass
    "selection": "aic",             # constant-vs-seasonal criterion: aic | loso_cv
    "n_harmonics": 2,               # harmonics of the seasonal offset form
    "period_days": 365.25,          # seasonal period
    "min_gain": 0.0,                # AIC units the seasonal form must beat the constant form by
    "force_form": None,             # None | constant | seasonal (skip the selection)
    "dpi": 130,
}
CAL_MATCHES = ("daily_mean", "overpass")
CAL_SELECTIONS = ("aic", "loso_cv")
CAL_FORMS = ("constant", "seasonal")

STAGE2_SECTIONS = ("detector", "sensors", "reference", "filter", "composite", "holdout",
                   "seasonal", "scale", "dineof", "loop", "final", "output")


def load_region_config(path: Path, *, resolve: bool = True) -> dict:
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    return build_region_config(user, path, resolve=resolve)


def build_region_config(user: dict, path: Path | str = "<dict>", *,
                        resolve: bool = True) -> dict:
    """-> {region, offsets, validation, pipe (pipeline.py's validated config), sha}."""
    allowed = {"region", "offsets", "validation", "point_filter", "calibration",
               *STAGE2_SECTIONS}
    for section in user:
        if section not in allowed:
            raise ValueError(f"{path}: unknown config section '{section}'")

    def merged(defaults, values, name):
        out = copy.deepcopy(defaults)
        for k, v in (values or {}).items():
            if k not in out:
                raise ValueError(f"{path}: unknown key '{name}.{k}'")
            if isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = {**out[k], **v}
            else:
                out[k] = v
        return out

    region = merged(REGION_DEFAULTS, user.get("region"), "region")
    offsets = merged(OFFSETS_DEFAULTS, user.get("offsets"), "offsets")
    validation = merged(VALIDATION_DEFAULTS, user.get("validation"), "validation")
    point_filter = merged(POINT_FILTER_DEFAULTS, user.get("point_filter"), "point_filter")
    calibration = merged(CALIBRATION_DEFAULTS, user.get("calibration"), "calibration")
    for key in ("aoi", "source"):
        if not region[key]:
            raise ValueError(f"{path}: region.{key} is required")
    if offsets["mode"] not in OFFSET_MODES:
        raise ValueError(f"{path}: offsets.mode must be one of {OFFSET_MODES}")
    if not 0 <= float(offsets["recommend_min_gain"]) < 1:
        raise ValueError(f"{path}: offsets.recommend_min_gain must be in [0, 1)")
    validate_point_filter(point_filter, path)
    validate_calibration(calibration, path)
    if resolve:
        for key in ("source", "out_dir"):
            p = Path(region[key])
            region[key] = (p if p.is_absolute() else ROOT / p).resolve()
        if validation["insitu"]:
            p = Path(validation["insitu"])
            validation["insitu"] = (p if p.is_absolute() else ROOT / p).resolve()
        if point_filter["points"]:
            p = Path(point_filter["points"])
            point_filter["points"] = (p if p.is_absolute() else ROOT / p).resolve()

    # Stage 2 runs on pipeline.py's own config. Offsets are fixed by stage 1, so the pipeline's
    # offset section only matters for its validation; the matchup section is stage 1's.
    pipe_user = {s: user[s] for s in STAGE2_SECTIONS if s in user}
    pipe_user["data"] = {"aoi": region["aoi"], "source": str(region["source"]),
                         "out_dir": str(region["out_dir"]), "watervar": region["watervar"],
                         "time_range": region["time_range"], "carry": region["carry"]}
    pipe_user["matchup"] = {k: offsets[k] for k in MATCHUP_KEYS}
    pipe_user["offset"] = {"model": "constant", "slope": False,
                           "slope_clip": offsets["slope_clip"]}
    pipe = P.build_config(pipe_user, path, resolve=resolve)

    sha = hashlib.sha256(yaml.safe_dump(user, sort_keys=True, default_flow_style=None)
                         .encode()).hexdigest()[:12]
    return dict(region=region, offsets=offsets, validation=validation,
                point_filter=point_filter, calibration=calibration, pipe=pipe, sha=sha,
                path=str(path))


def validate_point_filter(pf: dict, path: Path | str) -> None:
    """Strictly check the optional stage-4 `point_filter` section (merged() does not reject
    unknown nested keys). No-op when no groups are configured."""
    if not float(pf["period_days"]) > 0:
        raise ValueError(f"{path}: point_filter.period_days must be > 0")
    if float(pf["max_snap_m"]) < 0:
        raise ValueError(f"{path}: point_filter.max_snap_m must be >= 0")
    if int(pf["dpi"]) <= 0:
        raise ValueError(f"{path}: point_filter.dpi must be > 0")
    groups = pf["groups"] or {}
    if not isinstance(groups, dict):
        raise ValueError(f"{path}: point_filter.groups must be a mapping of name -> block")
    ev = pf.get("insitu_eval") or {}
    if not groups and not ev.get("enabled"):
        return
    import state_space as SS                       # imported only when stage 4 is configured
    if ev.get("enabled"):
        _validate_insitu_eval(ev, "point_filter.insitu_eval", path, SS)
    gkeys = {"model", "n_harmonics", "mode", "sources", "share", "period_days", "fit",
             "dineof_bias", "bias_n_harmonics"}
    for name, g in groups.items():
        where = f"point_filter.groups.{name}"
        if not isinstance(g, dict):
            raise ValueError(f"{path}: {where} must be a mapping")
        for k in g:
            if k not in gkeys:
                raise ValueError(f"{path}: unknown key '{where}.{k}'")
        model = g.get("model")
        if model not in SS.MODELS:
            raise ValueError(f"{path}: {where}.model must be one of {SS.MODELS}, got {model!r}")
        mode = g.get("mode", "hierarchy")
        if mode not in ("hierarchy", "multiple"):
            raise ValueError(f"{path}: {where}.mode must be 'hierarchy' or 'multiple'")
        sources = g.get("sources")
        if not sources or not isinstance(sources, list):
            raise ValueError(f"{path}: {where}.sources must be a non-empty list of types")
        for s in sources:
            if s not in SS.ALL_TYPES:
                raise ValueError(f"{path}: {where}.sources: unknown type {s!r}; "
                                 f"allowed {SS.ALL_TYPES}")
        H = int(g.get("n_harmonics", 0))
        if H < 0:
            raise ValueError(f"{path}: {where}.n_harmonics must be >= 0")
        if model in SS.HARMONIC_MODELS and H < 1:
            raise ValueError(f"{path}: {where}.model {model!r} needs n_harmonics >= 1")
        if model not in SS.HARMONIC_MODELS and H > 0:
            raise ValueError(f"{path}: {where}.model {model!r} takes no harmonics; drop n_harmonics")
        bias = g.get("dineof_bias", "none")
        if bias not in ("none", "constant", "seasonal"):
            raise ValueError(f"{path}: {where}.dineof_bias must be none | constant | seasonal")
        if bias != "none":
            dineof = [s for s in sources if SS.TYPE_CHANNEL[s] in SS.DINEOF_CHANNELS]
            if "insitu" not in sources or not dineof:
                raise ValueError(f"{path}: {where}.dineof_bias needs both 'insitu' and a DINEOF "
                                 "source (observed/point_fill/day_fill/smooth_*) in sources")
            if mode != "multiple":
                raise ValueError(f"{path}: {where}.dineof_bias requires mode: multiple so in-situ "
                                 "and DINEOF are observed on the same days")
        bias_H = int(g.get("bias_n_harmonics", 0))
        if bias == "seasonal" and bias_H < 1:
            raise ValueError(f"{path}: {where}.bias_n_harmonics must be >= 1 for a seasonal bias")
        valid_share = {"process_var"} | {f"obs_var.{t}" for t in sources}
        if model in ("ar1", "ar1_harmonic"):
            valid_share |= {"phi", "mu"}
        if model in SS.HARMONIC_MODELS:
            valid_share |= {"beta"}
        if bias in ("constant", "seasonal"):
            valid_share |= {"bias_const"}
        if bias == "seasonal":
            valid_share |= {"bias_seasonal"}
        for sh in (g.get("share") or []):
            if sh not in valid_share:
                raise ValueError(f"{path}: {where}.share: {sh!r} is not a shareable parameter "
                                 f"for model {model!r}; allowed {sorted(valid_share)}")


def _validate_insitu_eval(ev: dict, where: str, path: Path | str, SS) -> None:
    """Check the point_filter.insitu_eval diagnostic block (smoother + post-hoc bias + LOSO)."""
    allowed = {"enabled", "model", "mode", "sources", "n_harmonics", "share", "bias_form",
               "bias_n_harmonics", "selection", "min_gain"}
    for k in ev:
        if k not in allowed:
            raise ValueError(f"{path}: unknown key '{where}.{k}'")
    model = ev.get("model")
    if model not in SS.MODELS:
        raise ValueError(f"{path}: {where}.model must be one of {SS.MODELS}, got {model!r}")
    if ev.get("mode", "hierarchy") not in ("hierarchy", "multiple"):
        raise ValueError(f"{path}: {where}.mode must be 'hierarchy' or 'multiple'")
    sources = ev.get("sources")
    if not sources or not isinstance(sources, list):
        raise ValueError(f"{path}: {where}.sources must be a non-empty list of DINEOF types")
    for s in sources:
        if s not in SS.ALL_TYPES:
            raise ValueError(f"{path}: {where}.sources: unknown type {s!r}; allowed {SS.ALL_TYPES}")
        if SS.TYPE_CHANNEL[s] not in SS.DINEOF_CHANNELS:
            raise ValueError(f"{path}: {where}.sources must be DINEOF-only (the smoother is fit "
                             f"without in-situ); drop {s!r}")
    H = int(ev.get("n_harmonics", 0))
    if model in SS.HARMONIC_MODELS and H < 1:
        raise ValueError(f"{path}: {where}.model {model!r} needs n_harmonics >= 1")
    if model not in SS.HARMONIC_MODELS and H > 0:
        raise ValueError(f"{path}: {where}.model {model!r} takes no harmonics; drop n_harmonics")
    if ev.get("bias_form", "auto") not in ("constant", "seasonal", "auto"):
        raise ValueError(f"{path}: {where}.bias_form must be constant | seasonal | auto")
    if int(ev.get("bias_n_harmonics", 2)) < 1:
        raise ValueError(f"{path}: {where}.bias_n_harmonics must be >= 1")
    if ev.get("selection", "aic") not in CAL_SELECTIONS:
        raise ValueError(f"{path}: {where}.selection must be one of {CAL_SELECTIONS}")
    if float(ev.get("min_gain", 0.0)) < 0:
        raise ValueError(f"{path}: {where}.min_gain must be >= 0")
    valid_share = {"process_var"} | {f"obs_var.{t}" for t in sources}
    if model in ("ar1", "ar1_harmonic"):
        valid_share |= {"phi", "mu"}
    if model in SS.HARMONIC_MODELS:
        valid_share |= {"beta"}
    for sh in (ev.get("share") or []):
        if sh not in valid_share:
            raise ValueError(f"{path}: {where}.share: {sh!r} is not a shareable smoother "
                             f"parameter for model {model!r}; allowed {sorted(valid_share)}")


def validate_calibration(cal: dict, path: Path | str) -> None:
    """Strictly check the optional stage-5 `calibration` section (merged() does not reject
    unknown nested keys)."""
    if cal["match"] not in CAL_MATCHES:
        raise ValueError(f"{path}: calibration.match must be one of {CAL_MATCHES}")
    if cal["selection"] not in CAL_SELECTIONS:
        raise ValueError(f"{path}: calibration.selection must be one of {CAL_SELECTIONS}")
    if cal["force_form"] is not None and cal["force_form"] not in CAL_FORMS:
        raise ValueError(f"{path}: calibration.force_form must be null or one of {CAL_FORMS}")
    if int(cal["n_harmonics"]) < 1:
        raise ValueError(f"{path}: calibration.n_harmonics must be >= 1")
    if not float(cal["period_days"]) > 0:
        raise ValueError(f"{path}: calibration.period_days must be > 0")
    if float(cal["min_gain"]) < 0:
        raise ValueError(f"{path}: calibration.min_gain must be >= 0")
    if int(cal["dpi"]) <= 0:
        raise ValueError(f"{path}: calibration.dpi must be > 0")
    if not cal["products"] or not isinstance(cal["products"], list):
        raise ValueError(f"{path}: calibration.products must be a non-empty list")


def stage_dirs(rc: dict, tag: str | None = None) -> dict:
    """The stage directories for this region (and tag)."""
    base = Path(rc["region"]["out_dir"]) / (rc["region"]["aoi"] + (f"_{tag}" if tag else ""))
    return {"base": base, "offsets": base / "stage1_offsets", "dineof": base / "stage2_dineof",
            "validation": base / "stage3_validation",
            "pointfilter": base / "stage4_pointfilter",
            "calibration": base / "stage5_calibration"}


def validate_vcfg(rc: dict) -> dict:
    """The dict validate_insitu functions expect, from the `validation` section."""
    v = rc["validation"]
    cfg = copy.deepcopy(V.DEFAULTS)
    cfg["products"] = list(v["products"])
    cfg["match"].update(v["match"])
    cfg["insitu"].update(v["reader"])
    cfg["output"]["dpi"] = int(v["dpi"])
    return cfg
