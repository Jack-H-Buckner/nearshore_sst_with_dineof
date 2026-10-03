"""Fit ONLY the seasonal climatology for a region and draw its figures, skipping the (slow)
DINEOF cloud loop. Use it to iterate on the GMRF settings (seasonal.method / gmrf_range_px / ...)
before committing to a full `nearshore-sst dineof` run.

    python scripts/debug_seasonal.py --config configs/region.<aoi>.yaml [--tag T] [--out DIR]

It loads the source cube, applies stage-1 offsets if they exist for the tag (otherwise fits
offsets inline), runs the seasonal fit (the same `pipeline.bootstrap` the real run uses), and
writes seasonal.png -- plus seasonal_gmrf.png when seasonal.method is 'gmrf' -- to the output
directory. Nothing else from the pipeline runs, so it finishes in seconds to tens of seconds.
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

import region as R                       # noqa: E402
import pipeline as P                     # noqa: E402
import pipeline_figures as PF            # noqa: E402
import iterative_filter as F             # noqa: E402
import stage1_offsets as S1              # noqa: E402
import stage2_dineof as S2               # noqa: E402

log = logging.getLogger("debug_seasonal")


def _fixed_offsets(rc: dict, tag: str | None, sids, n_times: int):
    """Stage-1 offsets as bootstrap wants them, or None to fit offsets inline."""
    path = R.stage_dirs(rc, tag)["offsets"] / "offsets.json"
    if not path.exists():
        log.info("no stage-1 offsets at %s; fitting offsets inline instead", path)
        return None
    off, slope, which = S2.chosen_offsets(S1.read_offsets(rc, tag))
    log.info("stage-1 offsets: %s", ", ".join(
        f"{sid} {which[sid]} (off {off[sid]:+.3f} K, slope {slope[sid]:.3f})" for sid in off))
    return ({sid: np.full(n_times, float(off[sid])) for sid in sids},
            {sid: float(slope[sid]) for sid in sids})


def run(config: Path, tag: str | None, out: Path | None, dpi: int) -> None:
    rc = R.load_region_config(config)
    pcfg = rc["pipe"]
    it = pcfg["_iter"]
    method = pcfg["seasonal"].get("method", "harmonic")

    times = P.select_times(pcfg)
    raw = F.load_raw(it, times)
    keep = F.qc_only_keep(raw, it["filter"]["require_qc"])
    fixed = _fixed_offsets(rc, tag, list(raw.raw), len(raw.times))

    _, boot = P.bootstrap(raw, keep, pcfg, "debug-seasonal", fixed=fixed)
    sea = boot["seasonal"]
    ft = sea["fit_type"]
    log.info("seasonal method %s: %d full, %d borrowed/mean-only, %d reference (no data)",
             method, int((ft == 2).sum()), int((ft == 1).sum()), int((ft == 0).sum()))

    out = Path(out) if out is not None else R.stage_dirs(rc, tag)["dineof"] / "figures"
    out.mkdir(parents=True, exist_ok=True)
    ds = _dataset(raw, sea)
    PF.seasonal_figure(ds, out / "seasonal.png", dpi)
    log.info("wrote %s", out / "seasonal.png")
    if "sst_seasonal_coef_raw" in ds:
        PF.seasonal_gmrf_figure(ds, out / "seasonal_gmrf.png", dpi)
        log.info("wrote %s", out / "seasonal_gmrf.png")


def _dataset(raw: F.Raw, sea: dict) -> xr.Dataset:
    """The minimal cube the seasonal figures read: the coefficient channels, scale, fit_type,
    the water mask and the y/x coords."""
    data = {
        "sst_seasonal_coef": (("term", "y", "x"), sea["coef"].astype("float32")),
        "sst_seasonal_sd": (("y", "x"), sea["scale"].astype("float32")),
        "sst_seasonal_fit_type": (("y", "x"), sea["fit_type"]),
        "landcover_water": (("y", "x"), raw.water.astype("uint8")),
    }
    if "coef_raw" in sea:
        data["sst_seasonal_coef_raw"] = (("term", "y", "x"), sea["coef_raw"].astype("float32"))
    coords = {k: raw.coords[k] for k in ("y", "x") if k in raw.coords}
    coords["time"] = raw.times
    ds = xr.Dataset(data, coords=coords)
    ds["sst_seasonal_coef"].attrs.update(n_harmonics=int(sea["n_harmonics"]),
                                         period_days=float(sea["period_days"]))
    if "sst_seasonal_coef_raw" in ds:
        ds["sst_seasonal_coef_raw"].attrs.update(n_harmonics=int(sea["n_harmonics"]),
                                                 period_days=float(sea["period_days"]))
    return ds


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", "-c", type=Path, required=True, help="region YAML")
    p.add_argument("--tag", default=None, help="stage-1 offsets tag; also the output variant")
    p.add_argument("--out", type=Path, default=None, help="figure directory (default the tag's)")
    p.add_argument("--dpi", type=int, default=130)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    run(args.config, args.tag, args.out, args.dpi)
    return 0


if __name__ == "__main__":
    sys.exit(main())
