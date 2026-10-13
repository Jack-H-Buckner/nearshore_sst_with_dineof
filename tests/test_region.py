"""The three-stage region pipeline on the synthetic cube (see synthetic_cube.py)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml
from pyproj import Transformer

import iterative_filter as F       # noqa: F401  (bridges seasonal_smoothing)
import region as R
import run_region
import stage1_offsets as S1
import stage2_dineof as S2
import stage4_pointfilter as S4
import stage3_validate as S3
import synthetic_cube as SC


def _rc(src, out, **over):
    return R.build_region_config(SC.region_user_config(src, out, **over), resolve=False)


@pytest.fixture(scope="module")
def plain(tmp_path_factory):
    root = tmp_path_factory.mktemp("region")
    src = root / "source.zarr"
    truth = SC.build(src)
    rc = _rc(src, root / "out", output={"write_figures": True, "figure_dpi": 60})
    s1 = S1.run(rc, figures=True)
    s2 = S2.run(rc, figures=True)
    return dict(root=root, src=src, truth=truth, rc=rc, s1=s1, s2=s2)


# --------------------------------------------------------------------------- stage 1

def test_stage1_no_gain_recommends_fixed(plain):
    sens = plain["s1"]["offsets"]["sensors"]
    for sid, want in SC.OFFSETS.items():
        assert sens[sid]["recommended"] == "fixed", (sid, sens[sid]["models"])
        assert abs(sens[sid]["models"]["fixed"]["off"] - want) < 0.1
        assert sens[sid]["chosen"] == "fixed"


def test_stage1_offsets_json_schema(plain):
    path = R.stage_dirs(plain["rc"])["offsets"] / "offsets.json"
    payload = json.loads(path.read_text())
    assert payload == json.loads(json.dumps(plain["s1"]["offsets"], default=float))
    for sid, s in payload["sensors"].items():
        assert set(s) >= {"recommended", "chosen", "models", "diagnostics"}
        for kind in ("fixed", "rma"):
            m = s["models"][kind]
            assert set(m) >= {"a", "b", "x0", "off", "slope", "loso_rmse", "loso_rmse_scene",
                              "valid"}
        assert s["models"]["fixed"]["slope"] == 1.0
        assert len(s["diagnostics"]["rma_ci"]) == 2


def test_stage1_loso_matches_hand_refit(plain):
    res = plain["s1"]["results"]["eco"]
    pairs = res["pairs"]
    row = res["loso"][res["loso"]["model"] == "fixed"].iloc[0]
    m = S1.fit_model(pairs[pairs["t"] != row["t"]], "fixed", "eco", plain["rc"]["pipe"])
    test = pairs[(pairs["t"] == row["t"]) & pairs["kept"]]
    err = test["sensor"] - S1.predict(m, test["modis"])
    assert row["rmse_pairs"] == pytest.approx(float(np.sqrt(np.mean(err ** 2))))


def test_stage1_figures(plain):
    fig = R.stage_dirs(plain["rc"])["offsets"] / "figures"
    for f in ("models_eco.png", "loso_eco.png", "outliers_eco.png", "outlier_maps_eco.png",
              "models_lst.png", "summary.png"):
        assert (fig / f).exists(), f


def test_stage1_gain_recommends_rma_and_mode_overrides(tmp_path):
    src = tmp_path / "gain.zarr"
    SC.build(src, gains={"eco": 1.3})
    rc = _rc(src, tmp_path / "out")
    s = S1.run(rc, figures=False)["offsets"]["sensors"]["eco"]
    assert s["recommended"] == "rma", s["models"]
    assert abs(s["models"]["rma"]["b"] - 1.3) < 0.1
    rc2 = _rc(src, tmp_path / "out2", offsets={"mode": "fixed"})
    s2 = S1.run(rc2, figures=False)["offsets"]["sensors"]["eco"]
    assert s2["recommended"] == "rma" and s2["chosen"] == "fixed"


# --------------------------------------------------------------------------- stage 2

def test_stage2_applies_stage1_offsets_unchanged(plain):
    ds = xr.open_zarr(plain["s2"]["cube"])
    sens = plain["s1"]["offsets"]["sensors"]
    for sid in ("eco", "lst"):
        m = sens[sid]["models"][sens[sid]["chosen"]]
        assert np.allclose(ds[f"{sid}_offset"].values, m["off"], atol=1e-5)
        assert ds[f"{sid}_offset"].attrs["slope"] == pytest.approx(m["slope"])
    assert "stage1_offsets" in ds.attrs


def test_stage2_holdout(plain):
    h = plain["s2"]["holdout"]
    assert h is not None and len(h["table"])
    ok = np.isfinite(h["obs"])
    assert np.isfinite(h["pred"][ok]).all()
    for _, r in h["table"].iterrows():
        j = int(r["t"])
        d = (h["pred"][j] - h["obs"][j])[np.isfinite(h["obs"][j])]
        assert r["rmse"] == pytest.approx(float(np.sqrt(np.mean(d ** 2))))
    assert h["table"]["rmse"].median() < 1.0


def test_stage2_figures(plain):
    fig = R.stage_dirs(plain["rc"])["dineof"] / "figures"
    for f in ("cv_curves.png", "cv_holdout_scenes.png", "cv_holdout_scatter.png",
              "removed_scenes_eco.png", "cloud_filter_eco.png", "flag_changes.png",
              "eofs.png", "fields_1.png"):
        assert (fig / f).exists(), f


def test_stage2_requires_stage1(tmp_path, plain):
    rc = _rc(plain["src"], tmp_path / "empty")
    with pytest.raises(SystemExit, match="offsets"):
        S2.run(rc, figures=False)


# --------------------------------------------------------------------------- stage 3

def _insitu_csv(plain, path):
    """A station at a water pixel reading the synthetic truth, every 10 minutes."""
    truth = plain["truth"]
    row, col = 20, 30
    x = SC.X0 + SC.DX * (col + 0.5)
    y = SC.Y0 - SC.DX * (row + 0.5)
    lon, lat = Transformer.from_crs("EPSG:32610", "EPSG:4326", always_xy=True).transform(x, y)
    rows = []
    for j, t in enumerate(truth["times"][:120]):
        for h in np.arange(0, 24, 1 / 6):
            rows.append(dict(station_id="S1", time=t + pd.Timedelta(hours=h), latitude=lat,
                             longitude=lon, value=truth["truth"][j, row, col] - 273.15))
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_stage3(plain, tmp_path):
    csv = _insitu_csv(plain, tmp_path / "insitu.csv")
    rc = plain["rc"]
    rc["validation"]["insitu"] = csv
    out = S3.run(rc, figures=True)
    met = out["metrics"]
    ok = met["n"] > 0
    assert np.allclose(met.loc[ok, "mse"], met.loc[ok, "rmse"] ** 2)
    mu = out["matchups"]
    g = mu[mu["match"] == "overpass"]
    site = met[(met["match"] == "overpass") & (met["product"] == "sst_filled")
               & (met["stratum"] == "station") & (met["group"] == "S1")].iloc[0]
    assert site["mse"] == pytest.approx(float(np.mean((g["sst_filled"] - g["insitu"]) ** 2)))
    d = R.stage_dirs(rc)["validation"]
    for f in ("mse_by_site.png", "residuals.png", "matchups.csv", "metrics.csv"):
        assert (d / f).exists(), f


def test_stage3_requires_insitu(tmp_path, plain):
    rc = plain["rc"]
    rc["validation"]["insitu"] = None
    with pytest.raises(SystemExit, match="insitu"):
        S3.run(rc, figures=False)


# --------------------------------------------------------------------------- stage 4

def _points_csv(path):
    tr = Transformer.from_crs("EPSG:32610", "EPSG:4326", always_xy=True)
    rows = []
    for sid, (row, col), grp in [("A", (20, 30), "g1"), ("B", (22, 28), "g1")]:
        x = SC.X0 + SC.DX * (col + 0.5)
        y = SC.Y0 - SC.DX * (row + 0.5)
        lon, lat = tr.transform(x, y)
        rows.append(dict(site_id=sid, lat=lat, lon=lon, group=grp))
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_stage4_pointfilter(plain, tmp_path):
    csv = _points_csv(tmp_path / "points.csv")
    rc = dict(plain["rc"])
    rc["point_filter"] = {"points": str(csv), "period_days": 365.25, "max_snap_m": 500.0,
                          "dpi": 60, "groups": {"g1": {
                              "model": "random_walk", "mode": "hierarchy",
                              "sources": ["observed", "day_fill"],
                              "share": ["process_var", "obs_var.observed"]}}}
    out = S4.run(rc, figures=True)
    d = R.stage_dirs(rc)["pointfilter"]
    assert (d / "reports" / "point_series.csv").exists()
    assert (d / "reports" / "point_params.csv").exists()
    cube = d / "synthetic_points.zarr"
    assert cube.exists()
    with xr.open_zarr(cube) as ps:
        assert ps.sizes["site"] == 2 and ps.sizes["time"] == len(plain["truth"]["times"])
        assert np.isfinite(ps["smoothed_mean"].values).mean() > 0.9
        assert (ps["smoothed_sd"].values >= 0).all()
    series = out["series"]
    assert set(series["site"]) == {"A", "B"}
    # a shared group fits one process_var and one obs_var.observed for both sites
    params = pd.read_csv(d / "reports" / "point_params.csv")
    assert (params["name"] == "process_var").any()
    for f in ("A.png", "B.png"):
        assert (d / "figures" / f).exists(), f


def test_stage4_insitu_bias(plain, tmp_path):
    """A group that fuses in-situ with DINEOF and fits a constant DINEOF bias runs end to end."""
    insitu = _insitu_csv(plain, tmp_path / "insitu.csv")           # station S1 at pixel (20, 30)
    tr = Transformer.from_crs("EPSG:32610", "EPSG:4326", always_xy=True)
    x = SC.X0 + SC.DX * (30 + 0.5); y = SC.Y0 - SC.DX * (20 + 0.5)
    lon, lat = tr.transform(x, y)
    csv = tmp_path / "points_bias.csv"
    pd.DataFrame([dict(site_id="A", lat=lat, lon=lon, group="calib")]).to_csv(csv, index=False)
    rc = dict(plain["rc"])
    rc["validation"] = {**rc["validation"], "insitu": str(insitu)}
    rc["point_filter"] = {"points": str(csv), "period_days": 365.25, "max_snap_m": 500.0,
                          "dpi": 60, "groups": {"calib": {
                              "model": "random_walk", "mode": "multiple",
                              "sources": ["insitu", "observed", "day_fill"],
                              "dineof_bias": "constant", "share": ["bias_const"]}}}
    out = S4.run(rc, figures=False)
    params = pd.read_csv(R.stage_dirs(rc)["pointfilter"] / "reports" / "point_params.csv")
    assert (params["name"] == "bias_const").any()
    with xr.open_zarr(R.stage_dirs(rc)["pointfilter"] / "synthetic_points.zarr") as ps:
        assert "insitu_obs" in ps and "dineof_bias" in ps
        assert np.isfinite(ps["insitu_obs"].values).any()      # in-situ was matched and fed


def _insitu_csv_multi(plain, path):
    """Two in-situ stations at water pixels, 10-min sampling (passes daily-mean coverage)."""
    truth = plain["truth"]
    tr = Transformer.from_crs("EPSG:32610", "EPSG:4326", always_xy=True)
    rows = []
    for sid, (row, col) in [("S1", (20, 30)), ("S2", (22, 28))]:
        x = SC.X0 + SC.DX * (col + 0.5); y = SC.Y0 - SC.DX * (row + 0.5)
        lon, lat = tr.transform(x, y)
        for j, t in enumerate(truth["times"][:120]):
            for h in np.arange(0, 24, 1 / 6):
                rows.append(dict(station_id=sid, time=t + pd.Timedelta(hours=h), latitude=lat,
                                 longitude=lon, value=truth["truth"][j, row, col] - 273.15))
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_stage4_insitu_eval(plain, tmp_path):
    """Post-hoc bias + LOSO diagnostic at the in-situ sites (no points CSV)."""
    csv = _insitu_csv_multi(plain, tmp_path / "insitu_multi.csv")
    rc = dict(plain["rc"])
    rc["validation"] = {**rc["validation"], "insitu": str(csv)}
    rc["point_filter"] = {"points": None, "period_days": 365.25, "max_snap_m": 500.0, "dpi": 60,
                          "groups": {}, "insitu_eval": {
                              "enabled": True, "model": "random_walk", "mode": "hierarchy",
                              "sources": ["observed", "point_fill", "day_fill"], "n_harmonics": 0,
                              "share": [], "bias_form": "constant", "bias_n_harmonics": 2,
                              "selection": "aic", "min_gain": 0.0}}
    S4.run(rc, figures=True)
    base = R.stage_dirs(rc)["pointfilter"]
    d = base / "reports"
    assert (d / "insitu_eval_bias.json").exists()
    met = pd.read_csv(d / "insitu_eval_metrics.csv")
    allrow = met[met["site"] == "all"].iloc[0]
    assert np.isfinite(allrow["rmse_before"]) and np.isfinite(allrow["rmse_after"])
    loso = pd.read_csv(d / "insitu_eval_loso.csv")
    assert set(loso["left_out"].astype(str)) == {"S1", "S2"}
    for f in ("insitu_eval_offset.png", "insitu_eval_loso.png", "insitu_eval_mse_by_site.png",
              "insitu_eval_scatter.png", "insitu_eval_residuals.png",
              "timeseries_S1.png", "timeseries_S2.png"):
        assert (base / "figures" / f).exists(), f


def test_load_insitu_cube_fallback():
    """With validation.insitu null, in-situ sites/series come from the cube's embedded channels."""
    import stage4_pointfilter as S4
    T = 10
    xs = np.arange(5) * 100.0; ys = np.arange(4) * 100.0
    times = pd.date_range("2023-01-01", periods=T, freq="D")
    insitu = np.full((T, 4, 5), np.nan)
    insitu[:, 2, 3] = np.arange(T) + 10.0        # station A pixel, every day
    insitu[0:5, 1, 1] = 20.0                      # station B pixel, first 5 days
    ds = xr.Dataset({"insitu_sst": (("time", "y", "x"), insitu),
                     "landcover_water": (("y", "x"), np.ones((4, 5), "uint8"))},
                    coords={"time": times, "x": xs, "y": ys})
    ds.attrs["insitu_stations"] = json.dumps([
        {"id": "A", "row": 2, "col": 3, "index": 0, "lat": 0.0, "lon": 0.0},
        {"id": "B", "row": 1, "col": 1, "index": 1, "lat": 0.0, "lon": 0.0}])
    rc = {"validation": {"insitu": None}, "region": {"watervar": "landcover_water"},
          "point_filter": {"max_snap_m": 300.0}}
    placed, daily = S4._load_insitu(ds, rc, times)
    assert set(placed["station_id"]) == {"A", "B"}
    assert np.allclose(daily["A"], np.arange(T) + 10.0)
    assert int(np.isfinite(daily["B"]).sum()) == 5


def test_load_insitu_no_source_errors():
    """validation.insitu null and no in-situ channels in the cube -> a clear error."""
    import stage4_pointfilter as S4
    ds = xr.Dataset({"landcover_water": (("y", "x"), np.ones((3, 3), "uint8"))},
                    coords={"time": pd.date_range("2023-01-01", periods=3), "x": np.arange(3.0),
                            "y": np.arange(3.0)})
    rc = {"validation": {"insitu": None}, "region": {"watervar": "landcover_water"},
          "point_filter": {"max_snap_m": 300.0}}
    with pytest.raises(SystemExit, match="no in-situ data"):
        S4._load_insitu(ds, rc, pd.to_datetime(ds["time"].values))


# --------------------------------------------------------------------------- CLI

def test_run_region_all(tmp_path, plain):
    csv = _insitu_csv(plain, tmp_path / "insitu.csv")
    user = SC.region_user_config(plain["src"], tmp_path / "cli",
                                 validation={"insitu": str(csv)},
                                 loop={"max_iter": 2})
    cfg = tmp_path / "region.yaml"
    cfg.write_text(yaml.safe_dump(json.loads(json.dumps(user, default=str))))
    run_region.main(["all", "--config", str(cfg), "--no-figures"])
    base = tmp_path / "cli" / "synthetic"
    assert (base / "stage1_offsets" / "offsets.json").exists()
    assert (base / "stage2_dineof" / "synthetic_dineof.zarr").exists()
    assert (base / "stage3_validation" / "metrics.csv").exists()


def test_region_config_validation(tmp_path):
    with pytest.raises(ValueError, match="region.source"):
        R.build_region_config({"region": {"aoi": "x"}}, resolve=False)
    with pytest.raises(ValueError, match="offsets.mode"):
        R.build_region_config(SC.region_user_config(tmp_path, tmp_path,
                                                    offsets={"mode": "x"}), resolve=False)
    with pytest.raises(ValueError, match="unknown config section"):
        R.build_region_config({"bogus": {}}, resolve=False)
