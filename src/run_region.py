"""Run the three-stage pipeline for one region.

    python src/run_region.py --config configs/region.<aoi>.yaml offsets
    python src/run_region.py --config configs/region.<aoi>.yaml dineof
    python src/run_region.py --config configs/region.<aoi>.yaml validate
    python src/run_region.py --config configs/region.<aoi>.yaml all

  offsets   stage 1: fixed offset vs fixed offset + RMA slope per sensor; writes
            stage1_offsets/offsets.json (the model used downstream) and diagnostics. Review
            figures/summary.png and models_<id>.png; set offsets.mode to override.
            --masks CUBE uses a stage-2 cube's cloud masks instead of sensor QC alone.
  dineof    stage 2: applies the stage-1 offsets unchanged, runs the iterative cloud filter,
            the CV search and the final fill; writes stage2_dineof/<aoi>_dineof.zarr.
  validate  stage 3: matches the filled and smooth fields to in situ; MSE overall and by
            site, residual plots.
  all       the three in order.

Every output lands under <region.out_dir>/<aoi>[_tag]/. --tag keeps variants side by side.
New region: copy configs/region.template.yaml and fill in region.aoi and region.source.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import region as R


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("offsets", "dineof", "validate", "all"))
    p.add_argument("--config", type=Path, required=True, help="region YAML")
    p.add_argument("--tag", default=None, help="suffix the region's output directory")
    p.add_argument("--masks", type=Path, default=None,
                   help="offsets: take pixel masks from this stage-2 cube")
    p.add_argument("--max-iter", type=int, default=None, help="dineof: override loop.max_iter")
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    rc = R.load_region_config(args.config)
    if args.max_iter is not None:
        rc["pipe"]["_iter"]["loop"]["max_iter"] = int(args.max_iter)
    if not Path(rc["region"]["source"]).exists():
        raise SystemExit(f"region.source: no cube at {rc['region']['source']}")
    figs = not args.no_figures
    stages = ("offsets", "dineof", "validate") if args.stage == "all" else (args.stage,)
    for st in stages:
        if st == "offsets":
            import stage1_offsets
            stage1_offsets.run(rc, tag=args.tag, masks=args.masks, figures=figs)
        elif st == "dineof":
            import stage2_dineof
            stage2_dineof.run(rc, tag=args.tag, figures=figs)
        else:
            import stage3_validate
            stage3_validate.run(rc, tag=args.tag, figures=figs)


if __name__ == "__main__":
    main()
