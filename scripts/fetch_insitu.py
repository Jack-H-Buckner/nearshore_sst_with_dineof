"""Download in-situ water temperature for a cube's footprint from the IOOS Sensors ERDDAP.

Writes coastal_sst_data's (station, time) in-situ netCDF -- the format
`src/validate_insitu.py --insitu` reads -- using that package's own IOOS fetch, QARTOD
filtering and sensor-depth cut. Needs network access and a coastal_sst_data that has
`processes.insitu_ioos.fetch_aoi` and `processes.insitu_acquire.build_dataset`; the checkout
at ../coastal_sst_data/src is preferred over an older installed copy.

Known station inside the Admiralty Inlet cube: NOAA CO-OPS 9444900, Port Townsend.

    python scripts/fetch_insitu.py --cube data/unfiltered/admiralty_inlet.zarr \\
        --start 2025-03-01 --end 2026-02-28 --out data/insitu/admiralty_inlet_insitu.nc
    ... --halo-km 5          # also search this far outside the cube
    ... --dry-run            # list candidate stations only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
_PKG = ROOT.parent / "coastal_sst_data" / "src"
if (_PKG / "coastal_sst_data").exists() and str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

log = logging.getLogger("fetch_insitu")


def cube_bbox(cube: Path, halo_km: float) -> tuple[float, float, float, float]:
    """(W, S, E, N) in EPSG:4326 covering the cube's grid plus `halo_km`."""
    from pyproj import Transformer
    with xr.open_zarr(cube) as ds:
        crs = ds.attrs["crs"]
        xs, ys = ds["x"].values, ds["y"].values
    h = float(halo_km) * 1000.0
    x0, x1 = xs.min() - h, xs.max() + h
    y0, y1 = ys.min() - h, ys.max() + h
    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    ex = np.array([x0, x1, x1, x0, (x0 + x1) / 2, (x0 + x1) / 2, x0, x1])
    ey = np.array([y0, y0, y1, y1, y0, y1, (y0 + y1) / 2, (y0 + y1) / 2])
    lon, lat = tr.transform(ex, ey)
    return float(np.min(lon)), float(np.min(lat)), float(np.max(lon)), float(np.max(lat))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cube", type=Path, required=True, help="any cube on the target grid")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--out", type=Path, required=True, help="output netCDF")
    p.add_argument("--halo-km", type=float, default=0.0)
    p.add_argument("--max-depth-m", type=float, default=5.0)
    p.add_argument("--stations", nargs="*", default=[], help="only these IOOS dataset ids")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        from coastal_sst_data.processes import insitu_acquire, insitu_ioos
        fetch_aoi, build_dataset = insitu_ioos.fetch_aoi, insitu_acquire.build_dataset
    except (ImportError, AttributeError) as e:
        raise SystemExit(f"coastal_sst_data has no IOOS in-situ fetch ({e}); install the "
                         f"checkout at {_PKG.parent} (pip install -e)")

    bbox = cube_bbox(args.cube, args.halo_km)
    log.info("search bbox (W, S, E, N): %.4f, %.4f, %.4f, %.4f", *bbox)
    cfg = {"variables": insitu_ioos.DEFAULT_VARIABLES, "pad_deg": 0.0,
           "stations": list(args.stations), "exclude_stations": [],
           "qc_flags": insitu_ioos.DEFAULT_QC_FLAGS, "max_sensor_depth_m": args.max_depth_m}
    records = fetch_aoi(SimpleNamespace(search_bbox=bbox), args.start, args.end, cfg,
                        dry_run=args.dry_run)
    if args.dry_run:
        return
    if not records:
        raise SystemExit("no station returned usable data in this window")
    ds = build_dataset(records)
    ds.attrs.update(source="ioos", qc_flags=str(cfg["qc_flags"]),
                    requested_start=args.start, requested_end=args.end,
                    bbox=str(bbox), cube=str(args.cube))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(args.out, encoding={"sst": {"zlib": True, "complevel": 4},
                                     "qc": {"zlib": True, "complevel": 4}})
    log.info("wrote %s: %d stations, %d time steps", args.out, ds.sizes["station"],
             ds.sizes["time"])


if __name__ == "__main__":
    main()
