"""A small synthetic source cube with the channels the pipeline reads, written as zarr.

48 x 48 cells at 100 m in EPSG:32610, 365 daily steps. Truth is a seasonal cycle plus a rank-2
anomaly field. MODIS sees it unbiased and sparse; ECOSTRESS +1.0 K and Landsat +0.6 K see most
of the water on their days, ECOSTRESS with -4 K coherent cold patches (the "cloud") on about a
third of its scenes. The first `LAND_COLS` columns are land.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

H = W = 48
T = 365
LAND_COLS = 4
X0, Y0, DX = 515000.0, 5330000.0, 100.0
OFFSETS = {"eco": 1.0, "lst": 0.6}
FOOT = 6            # MODIS footprint, in cells: a block mean replicated to every cell it covers
MEAN, AMP, PEAK_DOY = 283.0, 4.0, 220.0


def grid():
    x = X0 + DX * (np.arange(W) + 0.5)
    y = Y0 - DX * (np.arange(H) + 0.5)
    return x, y


def truth_field(times: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """(truth (T, H, W) K, seasonal (T,) K)."""
    doy = np.asarray(times.dayofyear, float)
    seasonal = MEAN + AMP * np.cos(2 * np.pi * (doy - PEAK_DOY) / 365.25)
    yy, xx = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing="ij")
    p1 = np.cos(np.pi * xx) + 0.5 * yy
    p2 = np.sin(2 * np.pi * yy) * xx
    t = np.arange(len(times))
    a1 = 1.5 * np.cos(2 * np.pi * t / 45)
    a2 = 0.8 * np.sin(2 * np.pi * t / 17)
    truth = (seasonal[:, None, None] + a1[:, None, None] * (p1 - p1.mean())
             + a2[:, None, None] * (p2 - p2.mean()))
    return truth, seasonal


def build(path: Path, *, seed: int = 0, patches: bool = True,
          gains: dict | None = None) -> dict:
    """`gains`: sensor id -> b, so the sensor reads offset + MEAN + b (truth - MEAN)."""
    gains = gains or {}
    """Write the cube to `path`; return the ground truth needed by tests."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2025-03-01", periods=T, freq="D")
    x, y = grid()
    water = np.ones((H, W), bool)
    water[:, :LAND_COLS] = False
    truth, seasonal = truth_field(times)

    def sensor(p_day, offset, noise, cover_fn, gain=1.0):
        sst = np.full((T, H, W), np.nan, "float32")
        days = np.flatnonzero(rng.random(T) < p_day)
        for j in days:
            m = cover_fn() & water
            val = MEAN + gain * (truth[j] - MEAN)
            sst[j] = np.where(m, val + offset + rng.normal(scale=noise, size=(H, W)),
                              np.nan)
        return sst, days

    # MODIS is footprint-like, as on the real grid: each FOOT x FOOT block carries ONE value
    # (the block's water mean plus retrieval noise), nearest-neighbour style, on a random half
    # of the blocks. The sensors resolve the variability inside each block.
    modis = np.full((T, H, W), np.nan, "float32")
    modis_days = np.flatnonzero(rng.random(T) < 0.5)
    for j in modis_days:
        for bi in range(0, H, FOOT):
            for bj in range(0, W, FOOT):
                cells = water[bi:bi + FOOT, bj:bj + FOOT]
                if not cells.any() or rng.random() > 0.5:
                    continue
                v = truth[j, bi:bi + FOOT, bj:bj + FOOT][cells].mean() + rng.normal(scale=0.1)
                modis[j, bi:bi + FOOT, bj:bj + FOOT] = np.where(cells, v, np.nan)

    def eco_cover():
        m = np.ones((H, W), bool)
        r0, c0 = rng.integers(0, H - 12, size=2)
        m[r0:r0 + 12, c0:c0 + 12] = False                     # a QC-removed cloud gap
        return m

    eco, eco_days = sensor(0.3, OFFSETS["eco"], 0.2, eco_cover, gains.get("eco", 1.0))
    lst, lst_days = sensor(0.15, OFFSETS["lst"], 0.2, lambda: np.ones((H, W), bool),
                           gains.get("lst", 1.0))
    patch = np.zeros((T, H, W), bool)
    if patches:
        for j in eco_days:
            if rng.random() < 0.35:
                r0, c0 = rng.integers(0, H - 8, size=2)
                c0 = max(c0, LAND_COLS)
                patch[j, r0:r0 + 8, c0:c0 + 8] = True
        patch &= np.isfinite(eco)
        eco = np.where(patch, eco - 4.0, eco).astype("float32")

    def valid(a):
        return np.isfinite(a).astype("float32")

    eco_cloud = np.where(water[None] & ~np.isfinite(eco), 1.0, 0.0).astype("float32")
    hours = {
        "modis": np.where(np.isin(np.arange(T), modis_days), 10.4, np.nan),
        "eco": np.where(np.isin(np.arange(T), eco_days), rng.uniform(0, 24, T), np.nan),
        "lst": np.where(np.isin(np.arange(T), lst_days), 19.0, np.nan),
    }
    dims = ("time", "y", "x")
    ds = xr.Dataset(
        {"landcover_water": (("y", "x"), water.astype("uint8")),
         "depth_cudem": (("y", "x"), np.where(water, 10.0, 0.0)),
         "elevation_cudem": (("y", "x"), np.where(water, -10.0, 5.0)),
         "modis_sst_aqua": (dims, modis), "modis_valid_aqua": (dims, valid(modis)),
         "modis_hour_aqua": (("time",), hours["modis"]),
         "eco_sst_v002": (dims, eco), "eco_valid_v002": (dims, valid(eco)),
         "eco_cloud_v002": (dims, eco_cloud), "eco_hour_v002": (("time",), hours["eco"]),
         "lst_sst": (dims, lst), "lst_valid": (dims, valid(lst)),
         "lst_cloud": (dims, np.zeros((T, H, W), "float32")),
         "lst_hour": (("time",), hours["lst"])},
        coords={"time": times, "y": y, "x": x})
    ds.attrs.update(crs="EPSG:32610", aoi_id="synthetic")
    ds.to_zarr(path, mode="w")
    return dict(truth=truth, seasonal=seasonal, patch=patch, water=water, times=times,
                eco_days=eco_days, lst_days=lst_days, modis_days=modis_days)


def pipeline_user_config(source: Path, out_dir: Path, **over) -> dict:
    """A small, fast pipeline config for the synthetic cube."""
    user = {
        "data": {"aoi": "synthetic", "source": str(source), "out_dir": str(out_dir),
                 "time_range": None, "carry": ["landcover_water", "depth_cudem"]},
        "detector": {
            "ref_var": "modis_sst_aqua", "depthvar": "depth_cudem", "tidal_depth_m": 3.0,
            "clear": {"sd": 0.75, "sd_floor": 0.71},
            "qc": {"enabled": True, "prior_cloud": 0.9, "nodata_prior": None},
            "mixture": {"prior_cloud": 0.1, "min_dev_cold": 0.0, "min_dev_hot": 0.0}},
        "sensors": {
            "eco": {"label": "ECOSTRESS", "sst": "eco_sst_v002", "valid": "eco_valid_v002",
                    "cloud": "eco_cloud_v002", "hour": "eco_hour_v002", "min_pixels": 64,
                    "qc_nodata_prior": 0.5},
            "lst": {"label": "Landsat", "sst": "lst_sst", "valid": "lst_valid",
                    "cloud": "lst_cloud", "hour": "lst_hour", "min_pixels": 64}},
        "reference": {"id": "modis", "var": "modis_sst_aqua", "valid": "modis_valid_aqua",
                      "hour": "modis_hour_aqua"},
        "composite": {"hold": ["lst", "eco"]},
        "dineof": {"matrix": {"min_date_obs": 20, "coarsen": 2},
                   "filter": {"t_c_grid": [0, 4]},
                   "modes": {"k_grid": [1, 2, 3]},
                   "em": {"tol": 1.0e-3, "max_iter": 100},
                   "cv": {"day_frac": 0.1}},
        "loop": {"k": 2, "t_c": 4.0, "max_iter": 3},
        "matchup": {"min_pixels": 50},
        "offset": {"min_scenes": 4, "diurnal_harmonics": {"eco": 0, "lst": 0}},
        "final": {"coarsen": 1, "em": {"max_iter": 100}},
        "output": {"write_figures": False, "figure_dpi": 60},
    }
    for k, v in over.items():
        user[k] = {**user.get(k, {}), **v}
    return user


def region_user_config(source: Path, out_dir: Path, **over) -> dict:
    """The synthetic pipeline config re-expressed as a three-stage REGION config."""
    u = pipeline_user_config(source, out_dir)
    d = u.pop("data")
    region = {"aoi": d["aoi"], "source": d["source"], "out_dir": d["out_dir"],
              "time_range": d["time_range"], "carry": d["carry"]}
    offsets = {**u.pop("matchup"), "n_boot": 50}
    off = u.pop("offset")
    offsets["slope_clip"] = off.get("slope_clip", [0.5, 2.0])
    user = {"region": region, "offsets": offsets, **u,
            "validation": {"insitu": None}}
    for k, v in over.items():
        user[k] = {**user.get(k, {}), **v}
    return user
