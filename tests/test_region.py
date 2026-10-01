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
