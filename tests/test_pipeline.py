"""End-to-end pipeline tests on a synthetic source cube (see synthetic_cube.py)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import iterative_filter as F        # noqa: F401  (bridges seasonal_smoothing)
import pipeline as P
import synthetic_cube as SC


@pytest.fixture(scope="module")
def cube(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipe")
    src = root / "source.zarr"
    truth = SC.build(src)
    return root, src, truth


@pytest.fixture(scope="module")
def run(cube):
    root, src, truth = cube
    cfg = P.build_config(SC.pipeline_user_config(src, root / "out"), resolve=False)
    out = P.run_pipeline(cfg, tag="t", figures=False)
    return cfg, out, truth


def _raw_and_keep(cfg):
    raw = F.load_raw(cfg["_iter"])
    return raw, F.qc_only_keep(raw, True)


def test_fit_offsets_recovers_known_offsets(cube):
    root, src, truth = cube
    cfg = P.build_config(SC.pipeline_user_config(src, root / "o"), resolve=False)
    raw, keep = _raw_and_keep(cfg)
    offsets, slope, report, matchups, pairs = P.fit_offsets(raw, keep, cfg)
    for sid, want in SC.OFFSETS.items():
        assert abs(float(np.mean(offsets[sid])) - want) < 0.1, (sid, np.mean(offsets[sid]))
        assert slope[sid] == 1.0
    assert set(report["member"]) == {"modis", "eco", "lst"}


def test_validation_dates_seeded_and_unique(cube):
    root, src, _ = cube
    cfg = P.build_config(SC.pipeline_user_config(src, root / "o"), resolve=False)
    raw, keep = _raw_and_keep(cfg)
    a = P.select_validation_dates(raw, keep, cfg)
    b = P.select_validation_dates(raw, keep, cfg)
    assert a.size > 0 and np.array_equal(a, b)
    assert np.unique(a).size == a.size
    cfg["holdout"]["seed"] = 1
    assert not np.array_equal(a, P.select_validation_dates(raw, keep, cfg))


def test_fit_seasonal_recovers_cycle(tmp_path):
    src = tmp_path / "clean.zarr"
    SC.build(src, patches=False)
    cfg = P.build_config(SC.pipeline_user_config(src, tmp_path / "o"), resolve=False)
    raw, keep = _raw_and_keep(cfg)
    offsets, slope, _, _, _ = P.fit_offsets(raw, keep, cfg)
    sea = P.fit_seasonal(raw, offsets, slope, keep, np.array([], int), cfg)
    w = raw.water
    mean = sea["coef"][0][w]
    amp = np.hypot(sea["coef"][1][w], sea["coef"][2][w])
    assert abs(float(np.median(mean)) - SC.MEAN) < 0.3
    assert abs(float(np.median(amp)) - SC.AMP) < 0.4
    assert np.all(sea["scale"][w] >= cfg["scale"]["floor"])


def test_pipeline_output_channels(run):
    cfg, out, truth = run
    ds = xr.open_zarr(out["cube"])
    for name in ("sst_filled", "sst_filled_point", "sst_filled_day", "sst_smooth",
                 "sst_filled_observed", "sst_filled_constrained", "loadings_modis",
                 "smooth_loading_status", "eof_U", "eof_sigma", "eof_V", "loadings_full",
                 "eof_U_point", "eco_keep", "eco_p_valid", "eco_center", "eco_keep_history",
                 "eco_sst_v002_filtered", "lst_keep", "eco_offset", "lst_offset",
                 "sst_seasonal_coef", "sst_seasonal_sd", "validation_msk", "sst_composite",
                 "landcover_water", "modis_hour_aqua"):
        assert name in ds, name
    w = truth["water"]
    assert np.isfinite(ds["sst_filled"].values[:, w]).all()
    assert np.isfinite(ds["sst_smooth"].values[:, w]).all()
    k = ds.sizes["mode"]
    assert ds["eof_U_point"].dims[0] == "mode_point"
    assert ds["eof_U"].shape == (k, SC.H, SC.W)
    assert ds["eof_V"].shape == (k, SC.T)
    assert ds.attrs["crs"] == "EPSG:32610"
    for f in ("offsets_bootstrap.csv", "offsets_final.csv", "iterations.csv", "scenes.csv",
              "cv_curve.csv"):
        assert (out["reports"] / f).exists(), f


def test_pipeline_removes_cloud_patches(run):
    _, out, truth = run
    ds = xr.open_zarr(out["cube"])
    keep = ds["eco_keep"].values.astype(bool)
    patch = truth["patch"]
    recall = float((~keep[patch]).mean())
    assert recall > 0.9, recall


def test_pipeline_filled_is_close_to_truth(run):
    _, out, truth = run
    ds = xr.open_zarr(out["cube"])
    w = truth["water"]
    err = (ds["sst_filled"].values - truth["truth"])[:, w]
    rmse = float(np.sqrt(np.nanmean(err ** 2)))
    clim = (truth["seasonal"][:, None] - truth["truth"][:, w])
    rmse_clim = float(np.sqrt(np.mean(clim ** 2)))
    assert rmse < 0.6 * rmse_clim, (rmse, rmse_clim)


def test_figures_render(run, tmp_path):
    import pipeline_figures
    _, out, _ = run
    pipeline_figures.render(out["cube"], tmp_path, out["reports"], dpi=50)
    for f in ("eofs.png", "eofs_point.png", "cv_curves.png", "cloud_filter_eco.png",
              "cloud_filter_lst.png", "flag_changes.png", "fields_1.png", "offsets_eco.png",
              "offsets_lst.png"):
        assert (tmp_path / f).exists(), f


def test_build_output_with_unequal_k(run):
    """The point- and day-tuned fits may keep different numbers of modes (k=2 vs k=1 on the
    real data); each gets its own mode dimension."""
    import copy
    cfg, out, _ = run
    fin = copy.copy(out["fin"])
    res = dict(fin["res"])
    pf = dict(res["point_fit"])
    pf.update(U=pf["U"][:, :1], sigma=pf["sigma"][:1], V=pf["V"][:, :1], k=1)
    res["point_fit"] = pf
    fin["res"] = res
    ds = P.build_output(out["raw"], out["inp"], out["loop"], out["fits"], fin, out["smooth"],
                        out["filled"], out["cloud"], cfg)
    assert ds.sizes["mode_point"] == 1
    assert ds.sizes["mode"] == out["smooth"]["fit"]["k"]



# --------------------------------------------------------------------------- footprint matchups

def test_modis_footprints_labels_patches():
    a = np.full((12, 12), np.nan, "float32")
    a[0:4, 0:4], a[0:4, 4:8], a[4:8, 0:4] = 280.0, 281.0, 282.0
    a[8:12, 8:12] = 280.0                          # same value as block 1, not adjacent
    lab = P.modis_footprints(a)
    assert (lab[np.isnan(a)] == 0).all()
    assert len(np.unique(lab[lab > 0])) == 4
    a[4:8, 0:4] = 281.0                            # touches block 2 diagonally only
    a[0:4, 4:8] = 280.0                            # now equal and adjacent to block 1 -> merge
    lab = P.modis_footprints(a)
    assert lab[0, 0] == lab[0, 5]
    assert len(np.unique(lab[lab > 0])) == 3


def test_footprint_pairs_median_and_filters():
    ref = np.full((8, 8), np.nan)
    ref[:4, :4], ref[:4, 4:] = 280.0, 281.0
    ref[4:, :4] = 282.0
    mem = np.full((8, 8), np.nan)
    mem[:4, :4] = np.arange(16).reshape(4, 4)      # full cover: median 7.5
    mem[0, 4:] = 1.0                               # 4 of 16 cells: below a 0.5 cover
    lab = P.modis_footprints(ref)
    out = P.footprint_pairs(mem, ref, lab, min_cells=10, min_cover=0.5)
    assert len(out) == 1
    assert out["sensor_median"].iloc[0] == pytest.approx(7.5)
    assert out["modis"].iloc[0] == pytest.approx(280.0)
    assert len(P.footprint_pairs(mem, ref, lab, min_cells=20, min_cover=0.5)) == 0


def _scenes(n_scene=25, H=36, F=6, gain=1.0, off=1.0, outlier_frac=0.0, seed=0):
    """MODIS footprint means and a fine sensor field. Returns raw-like arrays (T, H, W)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.arange(H), np.arange(H), indexing="ij")
    truth = np.empty((n_scene, H, H))
    for j in range(n_scene):
        large = rng.normal(0, 1.5) * np.cos(2 * np.pi * xx / H + rng.uniform(0, 6)) \
            + rng.normal(0, 1.5) * np.sin(2 * np.pi * yy / H + rng.uniform(0, 6))
        truth[j] = 285.0 + rng.normal(0, 1) + large + rng.normal(0, 0.8, (H, H))
    modis = np.empty_like(truth)
    for bi in range(0, H, F):
        for bj in range(0, H, F):
            blk = truth[:, bi:bi + F, bj:bj + F].mean(axis=(1, 2))
            modis[:, bi:bi + F, bj:bj + F] = (blk + rng.normal(0, 0.05, n_scene))[:, None, None]
    sensor = off + gain * truth + rng.normal(0, 0.1, truth.shape)
    if outlier_frac:
        for j in range(n_scene):
            for bi in range(0, H, F):
                for bj in range(0, H, F):
                    if rng.random() < outlier_frac:
                        sensor[j, bi:bi + F, bj:bj + F] += 5.0
    return modis.astype("float32"), sensor.astype("float32")


def _offset_cfg(tmp_path, **matchup):
    cfg = P.build_config(SC.pipeline_user_config(tmp_path / "s.zarr", tmp_path / "o"),
                         resolve=False)
    cfg["matchup"].update(min_pixels=50, **matchup)
    return cfg


def _fit_one(modis, sensor, cfg):
    T = modis.shape[0]
    times = pd.date_range("2025-06-01", periods=T, freq="D")
    pairs = P.matchup_pairs(sensor, modis, np.full(T, 12.0), times, cfg, "eco")
    pairs, info = P.clip_pairs(pairs, cfg)
    b = info["b"] if cfg["offset"]["slope"] else 1.0
    x0 = P.slope_pivot(pairs) if cfg["offset"]["slope"] else 0.0
    table, rx, ry = P.scene_table(pairs, b, "eco", cfg, x0)
    return pairs, info, table, rx, ry, b


import pandas as pd  # noqa: E402
import composite as C  # noqa: E402


def test_footprints_remove_support_mismatch(tmp_path):
    """Pixel pairs inflate the RMA slope (the sensor's sub-footprint variance); footprint
    medians bring OLS and RMA together near the true gain of 1, and both recover the offset."""
    modis, sensor = _scenes()
    res = {}
    for agg in ("pixel", "footprint"):
        cfg = _offset_cfg(tmp_path, aggregate=agg, outlier_k=None)
        _, _, table, rx, ry, _ = _fit_one(modis, sensor, cfg)
        b_ols, b_rma, _ = C.slope_diagnostics(rx, ry, table)
        res[agg] = (b_ols, b_rma, float(np.median(table["delta"])))
    assert res["pixel"][1] > 1.1, res
    assert abs(res["footprint"][0] - 1.0) < 0.05 and abs(res["footprint"][1] - 1.0) < 0.05, res
    for agg in res:
        assert abs(res[agg][2] - 1.0) < 0.05, res


def test_slope_on_recovers_gain_and_offset(tmp_path):
    """sensor = 285 + 0.5 + 1.2 (truth - 285): gain 1.2, offset 0.5 K at 285 K. The offset is
    reported at the pivot x0 (median MODIS), where it is 0.5 + 0.2 (x0 - 285)."""
    modis, sensor = _scenes(gain=1.2, off=0.5)
    sensor = sensor - 1.2 * 285.0 + 285.0
    cfg = _offset_cfg(tmp_path, aggregate="footprint", outlier_k=None)
    cfg["offset"].update(slope=True, slope_estimator="ols")
    pairs, info, table, _, _, b = _fit_one(modis, sensor, cfg)
    x0 = P.slope_pivot(pairs)
    assert abs(b - 1.2) < 0.05, b
    assert abs(float(np.median(table["delta"])) - (0.5 + 0.2 * (x0 - 285.0))) < 0.1


def test_slope_correction_maps_sensor_onto_modis(cube):
    """End to end through fit_offsets: with the slope on, (sensor - off) / b lands on MODIS
    at the footprint level, and a slope outside slope_clip is an error."""
    root, src, _ = cube
    cfg = P.build_config(SC.pipeline_user_config(src, root / "o"), resolve=False)
    raw, keep = _raw_and_keep(cfg)
    raw.raw["eco"] = (285.0 + 1.15 * (raw.raw["eco"] - 285.0)).astype("float32")
    cfg["offset"].update(slope=True)
    offsets, slope, report, _, pairs = P.fit_offsets(raw, keep, cfg)
    assert abs(slope["eco"] - 1.15) < 0.08, slope
    k = pairs[(pairs["sensor_id"] == "eco") & pairs["kept"]]
    corr = (k["sensor"] - offsets["eco"][k["t"].to_numpy()]) / slope["eco"]
    assert abs(float(np.median(corr - k["modis"]))) < 0.1
    cfg["offset"]["slope_clip"] = [0.5, 1.05]
    with pytest.raises(ValueError, match="slope_clip"):
        P.fit_offsets(raw, keep, cfg)


def test_outlier_clipping(tmp_path):
    modis, sensor = _scenes(outlier_frac=0.08, seed=2)
    cfg_off = _offset_cfg(tmp_path, aggregate="footprint", outlier_k=None)
    _, _, t_off, _, _, _ = _fit_one(modis, sensor, cfg_off)
    cfg = _offset_cfg(tmp_path, aggregate="footprint", outlier_k=1.5)
    pairs, info, t_on, _, _, _ = _fit_one(modis, sensor, cfg)
    bad = pairs["sensor"] - pairs["modis"] > 3.5
    assert bad.any() and not pairs.loc[bad, "kept"].any()
    assert info["converged"] and 1 <= info["rounds"] <= cfg["matchup"]["outlier_max_iter"]
    assert abs(float(t_on["delta"].mean()) - 1.0) < 0.05
    assert float(t_off["delta"].mean()) > float(t_on["delta"].mean())
    # fixed point: one more round over ALL pairs, fit on the final kept set, reproduces it
    k = pairs["kept"].to_numpy()
    d = (pairs["sensor"] - pairs["modis"])
    off = d[k].groupby(pairs["t"][k]).median().reindex(pairs["t"]).to_numpy()
    r = (d.to_numpy() - off)
    rc = r - np.median(r)
    s = 1.4826 * np.median(np.abs(rc))
    assert np.array_equal(np.abs(rc) <= 1.5 * s, k)


def test_outlier_off_keeps_all_and_thin_scene_dropped(tmp_path):
    modis, sensor = _scenes(n_scene=3)
    cfg = _offset_cfg(tmp_path, aggregate="footprint", outlier_k=None)
    pairs, info, table, _, _, _ = _fit_one(modis, sensor, cfg)
    assert pairs["kept"].all() and info["n_removed"] == 0
    pairs.loc[pairs["t"] == 0, "kept"] = False
    pairs.loc[(pairs["t"] == 0).to_numpy().nonzero()[0][:5], "kept"] = True
    table, _, _ = P.scene_table(pairs, 1.0, "eco", cfg)
    assert 0 not in set(table["t"])


def test_matchup_config_validated(tmp_path):
    for over, msg in ((dict(matchup={"aggregate": "nope"}), "aggregate"),
                      (dict(matchup={"outlier_k": 0}), "outlier_k"),
                      (dict(offset={"slope_estimator": "x"}), "slope_estimator")):
        with pytest.raises(ValueError, match=msg):
            P.build_config(SC.pipeline_user_config(tmp_path / "s", tmp_path / "o", **over),
                           resolve=False)
