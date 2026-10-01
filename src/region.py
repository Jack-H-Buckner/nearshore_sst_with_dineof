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

STAGE2_SECTIONS = ("detector", "sensors", "reference", "filter", "composite", "holdout",
                   "seasonal", "scale", "dineof", "loop", "final", "output")


def load_region_config(path: Path, *, resolve: bool = True) -> dict:
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    return build_region_config(user, path, resolve=resolve)


def build_region_config(user: dict, path: Path | str = "<dict>", *,
                        resolve: bool = True) -> dict:
    """-> {region, offsets, validation, pipe (pipeline.py's validated config), sha}."""
    allowed = {"region", "offsets", "validation", *STAGE2_SECTIONS}
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
    for key in ("aoi", "source"):
        if not region[key]:
            raise ValueError(f"{path}: region.{key} is required")
    if offsets["mode"] not in OFFSET_MODES:
        raise ValueError(f"{path}: offsets.mode must be one of {OFFSET_MODES}")
    if not 0 <= float(offsets["recommend_min_gain"]) < 1:
        raise ValueError(f"{path}: offsets.recommend_min_gain must be in [0, 1)")
    if resolve:
        for key in ("source", "out_dir"):
            p = Path(region[key])
            region[key] = (p if p.is_absolute() else ROOT / p).resolve()
        if validation["insitu"]:
            p = Path(validation["insitu"])
            validation["insitu"] = (p if p.is_absolute() else ROOT / p).resolve()

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
    return dict(region=region, offsets=offsets, validation=validation, pipe=pipe, sha=sha,
                path=str(path))


def stage_dirs(rc: dict, tag: str | None = None) -> dict:
    """The three stage directories for this region (and tag)."""
    base = Path(rc["region"]["out_dir"]) / (rc["region"]["aoi"] + (f"_{tag}" if tag else ""))
    return {"base": base, "offsets": base / "stage1_offsets", "dineof": base / "stage2_dineof",
            "validation": base / "stage3_validation"}


def validate_vcfg(rc: dict) -> dict:
    """The dict validate_insitu functions expect, from the `validation` section."""
    v = rc["validation"]
    cfg = copy.deepcopy(V.DEFAULTS)
    cfg["products"] = list(v["products"])
    cfg["match"].update(v["match"])
    cfg["insitu"].update(v["reader"])
    cfg["output"]["dpi"] = int(v["dpi"])
    return cfg
