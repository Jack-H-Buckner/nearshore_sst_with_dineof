"""In-situ validation tests on a synthetic pipeline-style cube with known fields."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyproj import Transformer

import validate_insitu as V

H = W = 30
T = 20
X0, Y0, DX = 515000.0, 5330000.0, 100.0
CRS = "EPSG:32610"
HOUR = 10.0
CLIM_K = 282.0


def field(t, row, col):
    """The filled field in K, known in closed form."""
    return 284.0 + 0.1 * t + 0.01 * row + 0.02 * col


def make_cube(path, *, insitu_channels=None):
    x = X0 + DX * (np.arange(W) + 0.5)
    y = Y0 - DX * (np.arange(H) + 0.5)
    times = pd.date_range("2025-06-01", periods=T, freq="D")
    tt, rr, cc = np.meshgrid(np.arange(T), np.arange(H), np.arange(W), indexing="ij")
    filled = field(tt, rr, cc).astype("float32")
    water = np.ones((H, W), bool)
    water[:, :3] = False                                         # land strip, west edge
    coef = np.zeros((3, H, W), "float32")
    coef[0] = CLIM_K
    hour = np.full(T, HOUR)
    hour[5] = np.nan                                             # a day MODIS did not fly
    dims = ("time", "y", "x")
    data = {"sst_filled": (dims, filled), "sst_filled_point": (dims, filled),
            "sst_filled_day": (dims, filled), "sst_smooth": (dims, filled - 0.5),
            "sst_filled_observed": (dims, (tt % 2).astype("int8")),
            "sst_filled_constrained": (("time",), (np.arange(T) % 3 != 0).astype("int8")),
            "smooth_loading_status": (("time",), np.zeros(T, "int8")),
            "landcover_water": (("y", "x"), water.astype("uint8")),
            "modis_hour_aqua": (("time",), hour),
            "sst_seasonal_coef": (("term", "y", "x"), coef),
            "sst_seasonal_fit_type": (("y", "x"),
                                      np.where(water, 2, -1).astype("int8"))}
    ds = xr.Dataset(data, coords={"time": times, "y": y, "x": x})
    ds["sst_seasonal_coef"].attrs.update(n_harmonics=1, period_days=365.25)
    ds.attrs["crs"] = CRS
    if insitu_channels is not None:
        for k, v in insitu_channels["vars"].items():
            ds[k] = (dims, v)
        ds["insitu_station"] = (("y", "x"), insitu_channels["station"])
        ds.attrs["insitu_stations"] = json.dumps(insitu_channels["meta"])
    ds.to_zarr(path, mode="w")
    return ds


def lonlat(row, col):
    tr = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    return tr.transform(X0 + DX * (col + 0.5), Y0 - DX * (row + 0.5))


def station_obs(sid, row, col, *, hours=np.arange(0, 24, 0.1), value_fn=None, days=range(T),
                lat_lon=None):
    """Observations every 6 min; by default exactly the filled field, in degC."""
    lon, lat = lat_lon if lat_lon is not None else lonlat(row, col)
    rows = []
    for d in days:
        for h in hours:
            v = (value_fn(d, h) if value_fn else field(d, row, col)) - 273.15
            rows.append(dict(station_id=sid, station_name=sid, lat=lat, lon=lon,
                             time=pd.Timestamp("2025-06-01") + pd.Timedelta(days=d, hours=h),
                             value_C=v))
    return pd.DataFrame(rows)


@pytest.fixture()
def cube(tmp_path):
    p = tmp_path / "c.zarr"
    make_cube(p)
    return p


def run(cube, insitu, **match):
    cfg = V.load_config(None)
    cfg["match"].update(match)
    ds = xr.open_zarr(cube)
    return V.build_matchups(ds, cfg, insitu)


def test_exact_match_overpass_and_daily_mean(cube):
    obs = station_obs("A", 10, 20)
    mu, placed = run(cube, obs)
    assert placed.loc[0, "status"] == "ok"
    assert set(mu["match"]) == {"overpass", "daily_mean"}
    assert np.allclose(mu["sst_filled"], mu["insitu"], atol=1e-4)
    assert np.allclose(mu["sst_smooth"], mu["insitu"] - 0.5, atol=1e-4)
    assert np.allclose(mu["climatology"], CLIM_K - 273.15, atol=1e-4)
    assert (mu["match"] == "overpass").sum() == T and (mu["match"] == "daily_mean").sum() == T
    # day 5 has no MODIS hour: it falls back to the record's median hour, still matched
    assert 5 in set(mu.loc[mu["match"] == "overpass", "t"])
    # the seasonal fit method at the station's water pixel (fit_type 2 -> "full")
    assert (mu["seasonal_fit"] == "full").all()


def test_overpass_tolerance(cube):
    # observations only 90 min after the overpass: outside a 60 min tolerance, inside 120
    obs = station_obs("A", 10, 20, hours=[HOUR + 1.5])
    mu, _ = run(cube, obs, max_dt_min=60.0)
    assert (mu["match"] == "overpass").sum() == 0
    mu, _ = run(cube, obs, max_dt_min=120.0)
    ov = mu[mu["match"] == "overpass"]
    assert len(ov) == T and np.allclose(ov["dt_min"], 90.0, atol=0.5)


def test_daily_mean_coverage(cube):
    obs = station_obs("A", 10, 20, hours=np.arange(8, 14, 0.1))     # 6 of 24 hours
    mu, _ = run(cube, obs, min_coverage=0.75)
    assert (mu["match"] == "daily_mean").sum() == 0
    mu, _ = run(cube, obs, min_coverage=0.2)
    assert (mu["match"] == "daily_mean").sum() == T


def test_daily_mean_is_the_mean(cube):
    # value = field + (h - 12) / 12: the daily mean over symmetric hours equals the field
    hrs = np.arange(0, 24, 0.5)
    obs = station_obs("A", 10, 20, hours=hrs,
                      value_fn=lambda d, h: field(d, 10, 20) + (h - 11.75) / 12)
    mu, _ = run(cube, obs)
    dm = mu[mu["match"] == "daily_mean"]
    assert np.allclose(dm["insitu"], dm["sst_filled"], atol=1e-6)


def test_land_station_snaps_and_far_station_drops(cube):
    lon, lat = lonlat(10, 1)                                    # land, 200 m from water (col 3)
    near = station_obs("L", 10, 3, lat_lon=(lon, lat), days=[0])
    far_lon, far_lat = lonlat(10, 0)                            # land, 300 m from water
    far = station_obs("F", 10, 3, lat_lon=(far_lon, far_lat), days=[0])
    out_lon, out_lat = lonlat(10, W + 20)                       # 2 km off the grid
    outside = station_obs("O", 10, 3, lat_lon=(out_lon, out_lat), days=[0])
    mu, placed = run(cube, pd.concat([near, far, outside]), max_snap_m=250.0)
    st = placed.set_index("station_id")
    assert st.loc["L", "status"] == "snapped" and st.loc["L", "col"] == 3
    assert abs(st.loc["L", "snap_m"] - 200.0) < 1.0
    assert st.loc["F", "status"] == "no_water"
    assert st.loc["O", "status"] == "outside"
    assert set(mu["station_id"]) == {"L"}


def test_cube_channels_match_file_path(tmp_path):
    row, col = 12, 18
    tt = np.arange(T)
    modis_insitu = np.full((T, H, W), np.nan, "float32")
    modis_insitu[:, row, col] = field(tt, row, col) - 273.15
    station = np.zeros((H, W), "uint16")
    station[row, col] = 1
    lon, lat = lonlat(row, col)
    meta = [{"index": 1, "id": "S1", "name": "S1", "source": "ioos", "lat": lat, "lon": lon,
             "row": row, "col": col}]
    p = tmp_path / "ci.zarr"
    make_cube(p, insitu_channels={"vars": {"modis_insitu_sst": modis_insitu,
                                           "insitu_sst": modis_insitu},
                                  "station": station, "meta": meta})
    mu_cube, _ = run(p, None)
    mu_file, _ = run(p, station_obs("S1", row, col, hours=[HOUR]))
    a = mu_cube[mu_cube["match"] == "overpass"].sort_values("t")
    b = mu_file[mu_file["match"] == "overpass"].sort_values("t")
    assert len(a) == T == len(b)
    assert np.allclose(a["insitu"].to_numpy(), b["insitu"].to_numpy(), atol=1e-4)
    assert np.allclose(a["sst_filled"].to_numpy(), b["sst_filled"].to_numpy())
    assert (mu_cube["match"] == "reference").sum() == T


def test_scores_by_hand():
    obs = np.array([10.0, 11.0, 12.0, 13.0])
    pred = obs + np.array([0.5, -0.5, 1.0, 0.0])
    clim = np.full(4, 11.5)
    s = V.scores(pred - obs, pred, obs, clim - obs)
    assert s["n"] == 4
    assert s["bias"] == pytest.approx(0.25)
    assert s["rmse"] == pytest.approx(np.sqrt((0.25 + 0.25 + 1 + 0) / 4))
    assert s["mae"] == pytest.approx(0.5)
    mse_c = np.mean((clim - obs) ** 2)
    assert s["skill_vs_clim"] == pytest.approx(1 - 0.375 / mse_c)
    sc = V.scores(clim - obs, clim, obs, clim - obs)
    assert sc["skill_vs_clim"] == pytest.approx(0.0)


def test_netcdf_and_csv_readers(tmp_path):
    obs = station_obs("A", 10, 20, days=[0, 1], hours=[HOUR])
    # package netCDF: (station, time) with QARTOD flags; a flag-4 value must be dropped
    times = pd.DatetimeIndex(obs["time"])
    sst = obs["value_C"].to_numpy()[None, :].astype("float32")
    qc = np.array([[1, 4]], "uint8")
    ds = xr.Dataset({"sst": (("station", "time"), sst), "qc": (("station", "time"), qc)},
                    coords={"station": [0], "time": times, "station_id": ("station", ["A"]),
                            "station_name": ("station", ["A"]),
                            "lat": ("station", [obs["lat"].iloc[0]]),
                            "lon": ("station", [obs["lon"].iloc[0]])})
    nc = tmp_path / "s.nc"
    ds.to_netcdf(nc)
    cfg = V.load_config(None)
    got = V.read_insitu(nc, cfg)
    assert len(got) == 1 and got["value_C"].iloc[0] == pytest.approx(sst[0, 0])

    # long CSV in kelvin with a depth column: the deep row is dropped
    csv = tmp_path / "s.csv"
    pd.DataFrame({"station_id": ["A", "A"], "time": times.strftime("%Y-%m-%dT%H:%M:%S"),
                  "latitude": obs["lat"], "longitude": obs["lon"],
                  "value": obs["value_C"] + 273.15, "z": [1.0, 12.0]}).to_csv(csv, index=False)
    cfg["insitu"]["units"] = "K"
    got = V.read_insitu(csv, cfg)
    assert len(got) == 1 and got["value_C"].iloc[0] == pytest.approx(obs["value_C"].iloc[0])


def test_validate_writes_outputs(cube, tmp_path):
    obs = station_obs("A", 10, 20, hours=np.arange(0, 24, 1.0))
    nc_rows = obs
    csv = tmp_path / "obs.csv"
    nc_rows.rename(columns={"lat": "latitude", "lon": "longitude", "value_C": "value"}) \
        .to_csv(csv, index=False)
    out = V.validate(cube, csv, V.load_config(None), tmp_path / "val", figs=True)
    for f in ("matchups.csv", "metrics.csv", "stations.csv", "stations_map.png",
              "scatter.png", "error_by_category.png", "timeseries_A.png"):
        assert (tmp_path / "val" / f).exists(), f
    m = out["metrics"]
    allrow = m[(m["match"] == "overpass") & (m["product"] == "sst_filled") &
               (m["stratum"] == "all")].iloc[0]
    assert allrow["rmse"] == pytest.approx(0.0, abs=1e-4)
