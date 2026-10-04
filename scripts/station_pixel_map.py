"""Draw the monitoring stations as highlighted pixels over the land/water mask, standalone.

Land is grey, water blue, and each station's matched pixel red, labelled by station name. This is
the same figure stage 3 writes as `station_pixels.png`, runnable on any pipeline cube without
re-running validation.

    python scripts/station_pixel_map.py --cube CUBE.zarr [--insitu FILE] [--out map.png]

With --insitu (a netCDF/CSV of stations) the stations are snapped onto the grid exactly as stage 3
does; without it, the cube's own `insitu_stations` attribute is used (if present).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import validate_insitu as V          # noqa: E402

log = logging.getLogger("station_pixel_map")


def placed_stations(ds: xr.Dataset, cfg: dict, insitu_path: Path | None):
    """Station table with grid (row, col), matching stage 3's placement."""
    if insitu_path is None:
        if "insitu_stations" not in ds.attrs:
            raise SystemExit("no --insitu file and the cube carries no `insitu_stations` attribute")
        return V.cube_stations(ds)
    insitu = V.read_insitu(insitu_path, cfg)
    stations = insitu.groupby("station_id", as_index=False).agg(
        station_name=("station_name", "first"), lat=("lat", "median"), lon=("lon", "median"))
    water = np.asarray(ds["landcover_water"].values > 0.5)
    return V.place_stations(stations, ds, water, float(cfg["match"]["max_snap_m"]))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cube", type=Path, required=True, help="a pipeline output cube (zarr)")
    p.add_argument("--insitu", type=Path, default=None, help="station netCDF/CSV (else cube attr)")
    p.add_argument("--out", type=Path, default=None, help="output PNG (default beside the cube)")
    p.add_argument("--dpi", type=int, default=130)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ds = xr.open_zarr(args.cube)
    cfg = V.load_config(None)
    placed = placed_stations(ds, cfg, args.insitu)
    on = int((placed["row"] >= 0).sum())
    log.info("%d stations, %d on water", len(placed), on)

    out = args.out or Path(args.cube).with_name(f"{Path(args.cube).stem}_station_pixels.png")
    V.station_pixel_map(ds, placed, out, args.dpi)
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
