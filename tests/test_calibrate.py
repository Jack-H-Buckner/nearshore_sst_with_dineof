"""Stage 5 in-situ calibration (src/stage5_calibrate.py) and its config.

The offset-fit tests follow tests/test_state_space.py: build matchups with a known injected
offset on a seeded RNG, fit, and assert recovery / that the right form is selected. One
end-to-end test runs the stage on the synthetic cube from synthetic_cube.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyproj import Transformer

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import iterative_filter as F          # noqa: E402,F401  (bridges seasonal_smoothing)
import region as R                    # noqa: E402
import stage1_offsets as S1           # noqa: E402
import stage2_dineof as S2            # noqa: E402
import stage5_calibrate as C5         # noqa: E402
import synthetic_cube as SC           # noqa: E402

PRODUCT = "sst_filled"


def _cal(**over):
    c = dict(R.CALIBRATION_DEFAULTS)
    c.update(over)
    return c


def _matchups(sites, offset_fn, *, n_days=200, noise=0.0, seed=0):
    """Matchups shaped like validate_insitu.build_matchups: one row per site x day, with a
    product column and an in-situ value = product + offset(day_of_year) + noise."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2024-01-01", periods=n_days, freq="D")
    doy = times.dayofyear.to_numpy()
    rows = []
    for s in sites:
        prod = 10.0 + 3.0 * np.sin(2 * np.pi * np.arange(n_days) / n_days)
        ins = prod + offset_fn(doy) + rng.normal(0.0, noise, n_days)
        for t, p, i in zip(times, prod, ins):
            rows.append({"station_id": s, "date": t.strftime("%Y-%m-%d"), "match": "daily_mean",
                         "insitu": float(i), PRODUCT: float(p)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- fit recovery

def test_constant_offset_recovers():
    mu = _matchups(["S1"], lambda d: np.full_like(d, 0.8, float), noise=0.02, seed=0)
    fit = C5._fit_offset(mu, PRODUCT, "constant", 2, 365.25)
    assert fit["k"] == 1 and fit["n"] == len(mu)
    assert fit["coef"][0] == pytest.approx(0.8, abs=0.02)


def test_seasonal_offset_recovers():
    off = lambda d: 0.5 + 1.5 * np.cos(2 * np.pi * d / 365.25)
    mu = _matchups(["S1"], off, noise=0.02, seed=1)
    fit = C5._fit_offset(mu, PRODUCT, "seasonal", 2, 365.25)
    coef = dict(zip(C5._coef_names("seasonal", 2), fit["coef"]))
    assert coef["mean"] == pytest.approx(0.5, abs=0.05)
    assert coef["cos1"] == pytest.approx(1.5, abs=0.05)
    assert abs(coef["sin1"]) < 0.05


# --------------------------------------------------------------------------- selection

def test_aic_picks_constant_when_flat():
    mu = _matchups(["S1"], lambda d: np.full_like(d, 0.6, float), noise=0.05, seed=2)
    sel = C5._select(mu, PRODUCT, _cal(selection="aic"), n_sites=1)
    assert sel["form"] == "constant" and sel["how"] == "aic"


def test_aic_picks_seasonal_when_seasonal():
    off = lambda d: 0.3 + 2.0 * np.cos(2 * np.pi * d / 365.25)
    mu = _matchups(["S1"], off, noise=0.05, seed=3)
    sel = C5._select(mu, PRODUCT, _cal(selection="aic"), n_sites=1)
    assert sel["form"] == "seasonal"
    assert sel["aic_seasonal"] < sel["aic_constant"]


def test_loso_cv_selection_runs_with_two_sites():
    off = lambda d: 0.3 + 2.0 * np.cos(2 * np.pi * d / 365.25)
    mu = _matchups(["S1", "S2"], off, noise=0.1, seed=4)
    sel = C5._select(mu, PRODUCT, _cal(selection="loso_cv"), n_sites=2)
    assert sel["how"] == "loso_cv"
    assert "loso_cv_constant" in sel and "loso_cv_seasonal" in sel
    assert sel["form"] == "seasonal"


def test_loso_cv_falls_back_to_aic_with_one_site():
    mu = _matchups(["S1"], lambda d: np.full_like(d, 0.6, float), noise=0.05, seed=5)
    sel = C5._select(mu, PRODUCT, _cal(selection="loso_cv"), n_sites=1)
    assert sel["how"] == "aic_fallback" and sel["form"] == "constant"


def test_force_form_overrides_selection():
    off = lambda d: 0.3 + 2.0 * np.cos(2 * np.pi * d / 365.25)
    mu = _matchups(["S1"], off, noise=0.05, seed=6)
    sel = C5._select(mu, PRODUCT, _cal(force_form="constant"), n_sites=1)
    assert sel["form"] == "constant" and sel["how"] == "forced"


def test_min_gain_blocks_marginal_seasonal():
    off = lambda d: 0.6 + 0.02 * np.cos(2 * np.pi * d / 365.25)   # tiny seasonal term
    mu = _matchups(["S1"], off, noise=0.3, seed=7)
    cheap = C5._select(mu, PRODUCT, _cal(min_gain=0.0), n_sites=1)
    strict = C5._select(mu, PRODUCT, _cal(min_gain=1e6), n_sites=1)
    assert strict["form"] == "constant"              # a huge gain threshold forces constant
    assert cheap["aic_constant"] >= strict["aic_constant"]  # same numbers, selection differs


# --------------------------------------------------------------------------- apply / metrics

def test_metrics_reduce_bias():
    mu = _matchups(["S1"], lambda d: np.full_like(d, 1.2, float), noise=0.05, seed=8)
    fit = C5._fit_offset(mu, PRODUCT, "constant", 2, 365.25)
    rows = C5._metrics(mu, PRODUCT, fit["coef"], "constant", 2, 365.25)
    allrow = [r for r in rows if r["site"] == "all"][0]
    assert abs(allrow["bias_after"]) < abs(allrow["bias_before"])
    assert allrow["bias_before"] == pytest.approx(-1.2, abs=0.05)   # product is cold by 1.2 K
    assert abs(allrow["bias_after"]) < 0.02


def test_loso_table_shape():
    off = lambda d: 0.3 + 1.0 * np.cos(2 * np.pi * d / 365.25)
    mu = _matchups(["S1", "S2", "S3"], off, noise=0.1, seed=9)
    fit = C5._fit_offset(mu, PRODUCT, "seasonal", 2, 365.25)
    tbl = C5._loso_table(mu, PRODUCT, fit["coef"], "seasonal", 2, 365.25)
    assert set(tbl["left_out"]) == {"S1", "S2", "S3"}
    assert (tbl["n_train"] > 0).all() and (tbl["n_held"] > 0).all()


# --------------------------------------------------------------------------- config validation

def _rc(src, out, **over):
    return R.build_region_config(SC.region_user_config(src, out, **over), resolve=False)


@pytest.mark.parametrize("bad, match", [
    ({"match": "weekly"}, "calibration.match"),
    ({"selection": "bic"}, "calibration.selection"),
    ({"force_form": "linear"}, "calibration.force_form"),
    ({"n_harmonics": 0}, "calibration.n_harmonics"),
    ({"period_days": 0}, "calibration.period_days"),
    ({"min_gain": -1}, "calibration.min_gain"),
    ({"products": []}, "calibration.products"),
])
def test_calibration_config_rejects(tmp_path, bad, match):
    with pytest.raises(ValueError, match=match):
        _rc(tmp_path / "s.zarr", tmp_path / "o", calibration=bad)


def test_calibration_defaults_disabled(tmp_path):
    rc = _rc(tmp_path / "s.zarr", tmp_path / "o")
    assert rc["calibration"]["enabled"] is False
    assert rc["calibration"]["selection"] == "aic"


# --------------------------------------------------------------------------- end to end

@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("calibrate")
    src = root / "source.zarr"
    truth = SC.build(src)
    rc = R.build_region_config(SC.region_user_config(src, root / "out"), resolve=False)
    S1.run(rc, figures=False)
    S2.run(rc, figures=False)
    return dict(root=root, truth=truth, rc=rc)


def _insitu_two_sites(truth, path):
    """Two stations at water pixels reading the synthetic truth hourly (daily-mean coverage)."""
    tr = Transformer.from_crs("EPSG:32610", "EPSG:4326", always_xy=True)
    rows = []
    for sid, (row, col) in [("A", (20, 30)), ("B", (22, 28))]:
        x = SC.X0 + SC.DX * (col + 0.5)
        y = SC.Y0 - SC.DX * (row + 0.5)
        lon, lat = tr.transform(x, y)
        for j, t in enumerate(truth["times"][:120]):
            for h in range(24):
                rows.append(dict(station_id=sid, time=t + pd.Timedelta(hours=h), latitude=lat,
                                 longitude=lon, value=truth["truth"][j, row, col] - 273.15))
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_stage5_end_to_end(pipeline, tmp_path):
    csv = _insitu_two_sites(pipeline["truth"], tmp_path / "insitu.csv")
    rc = dict(pipeline["rc"])
    rc["validation"] = {**rc["validation"], "insitu": csv}
    rc["calibration"] = _cal(enabled=True, selection="loso_cv")
    out = C5.run(rc, figures=True)

    assert out["n_sites"] == 2
    d = R.stage_dirs(rc)["calibration"]
    for f in ("reports/offsets.json", "reports/calibration_metrics.csv",
              "reports/calibration_loso.csv"):
        assert (d / f).exists(), f
    offs = json.loads((d / "reports" / "offsets.json").read_text())
    assert PRODUCT in offs and offs[PRODUCT]["form"] in R.CAL_FORMS

    # LOSO ran for two sites, one row per product x left-out site
    loso = out["loso"]
    assert set(loso[loso["product"] == PRODUCT]["left_out"]) == {"A", "B"}

    # the calibrated field is the original plus the (spatially uniform) offset curve
    cube = R.stage_dirs(rc)["dineof"] / "synthetic_dineof.zarr"
    calz = d / "synthetic_calibrated.zarr"
    assert calz.exists()
    with xr.open_zarr(cube) as ds, xr.open_zarr(calz) as cz:
        applied = cz[f"{PRODUCT}_cal"].values - cz[f"{PRODUCT}_offset"].values[:, None, None]
        orig = ds[PRODUCT].values
        both = np.isfinite(applied) & np.isfinite(orig)
        assert np.allclose(applied[both], orig[both], atol=1e-3)


def test_stage5_requires_insitu(pipeline):
    rc = dict(pipeline["rc"])
    rc["validation"] = {**rc["validation"], "insitu": None}
    rc["calibration"] = _cal(enabled=True)
    with pytest.raises(SystemExit, match="insitu"):
        C5.run(rc, figures=False)
