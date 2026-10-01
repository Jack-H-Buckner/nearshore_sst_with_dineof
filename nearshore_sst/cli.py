"""`nearshore-sst`: the command-line interface to the three-stage region pipeline.

    nearshore-sst init my_region --source data/unfiltered/my_region.zarr
    nearshore-sst check    --config configs/region.my_region.yaml
    nearshore-sst offsets  --config configs/region.my_region.yaml [--masks CUBE]
    nearshore-sst dineof   --config configs/region.my_region.yaml [--max-iter N]
    nearshore-sst validate --config configs/region.my_region.yaml
    nearshore-sst all      --config configs/region.my_region.yaml
    nearshore-sst status   --config configs/region.my_region.yaml
    nearshore-sst figures  --config configs/region.my_region.yaml
    nearshore-sst fetch-insitu --cube CUBE --start YYYY-MM-DD --end YYYY-MM-DD --out FILE.nc

Every stage command takes --tag T to keep variants side by side (outputs go to
<region.out_dir>/<aoi>_T/); later stages read the same tag.

The pipeline's modules live in the repo's src/ as flat modules with generic names (plotting,
composite, ...), so they are not installed into site-packages: this package adds src/ to the
import path when a command runs. Install with `pip install -e .` from the repo root.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
TEMPLATE = REPO / "configs" / "region.template.yaml"


def _bootstrap() -> None:
    """Make the pipeline's src/ modules importable (and, through iterative_filter, the
    external seasonal_smoothing)."""
    if not SRC.is_dir():
        raise SystemExit(f"cannot find the pipeline sources at {SRC}; nearshore-sst must be "
                         "installed from the repository (pip install -e .)")
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))


def _region(args):
    _bootstrap()
    import region as R
    rc = R.load_region_config(args.config)
    if getattr(args, "max_iter", None) is not None:
        rc["pipe"]["_iter"]["loop"]["max_iter"] = int(args.max_iter)
    return R, rc


# ==================================================================== commands

def cmd_init(args) -> int:
    """Write a new region config from the template with aoi and source filled in."""
    out = args.out or REPO / "configs" / f"region.{args.aoi}.yaml"
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; use --force to overwrite")
    import re
    text = TEMPLATE.read_text()
    # The first `  aoi:` / `  source:` lines are region's; `  insitu:` is validation's.
    for key, value in (("aoi", args.aoi), ("source", args.source), ("insitu", args.insitu)):
        if value is not None:
            text, n = re.subn(rf"^(  {key}:).*$", rf"\g<1> {value}", text, count=1,
                              flags=re.M)
            if n != 1:
                raise SystemExit(f"{TEMPLATE} has no `{key}:` line to fill in")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out}\nnext: nearshore-sst check --config {out}")
    return 0


def cmd_check(args) -> int:
    """Validate the config and the source cube without running anything."""
    R, rc = _region(args)
    import numpy as np
    import pandas as pd
    import xarray as xr
    errors, warnings = [], []
    reg, it = rc["region"], rc["pipe"]["_iter"]
    src = Path(reg["source"])
    print(f"config   {args.config}  (aoi {reg['aoi']}, sha {rc['sha']})")
    if not src.exists():
        print(f"ERROR    source cube not found: {src}")
        return 1
    ds = xr.open_zarr(src)
    print(f"source   {src}")
    crs = ds.attrs.get("crs")
    if crs:
        print(f"crs      {crs}")
    else:
        errors.append("the cube has no `crs` attribute")
    for c in ("x", "y", "time"):
        if c not in ds.coords:
            errors.append(f"coordinate `{c}` is missing")
    times = pd.to_datetime(ds["time"].values)
    tr = reg["time_range"]
    if tr is not None:
        sel = times[(times >= pd.Timestamp(tr[0])) & (times <= pd.Timestamp(tr[1]))]
    else:
        sel = times
    print(f"time     {len(times)} days in the cube {times[0]:%Y-%m-%d} .. {times[-1]:%Y-%m-%d}; "
          f"{len(sel)} selected"
          + (f" by time_range {pd.Timestamp(tr[0]):%Y-%m-%d} .. {pd.Timestamp(tr[1]):%Y-%m-%d}"
             if tr else ""))
    if len(sel) == 0:
        errors.append("time_range selects no dates")
    wanted = {"region.watervar": reg["watervar"], "detector.depthvar": it["detector"]["depthvar"]}
    ref = it["reference"]
    for k in ("var", "valid", "hour"):
        if ref.get(k):
            wanted[f"reference.{k}"] = ref[k]
    for sid, s in it["sensors"].items():
        for k in ("sst", "valid", "cloud", "hour"):
            wanted[f"sensors.{sid}.{k}"] = s[k]
    missing = [f"{k} = {v}" for k, v in wanted.items() if v not in ds]
    for m in missing:
        errors.append(f"channel missing: {m}")
    if not missing and len(sel):
        water = np.asarray(ds[reg["watervar"]].values > 0.5)
        print(f"water    {int(water.sum()):,} of {water.size:,} cells")
        sub = ds.sel(time=sel.values)
        nm = int((np.isfinite(sub[ref["var"]]).where(water).sum(("y", "x")) > 0)
                 .sum().compute())
        print(f"MODIS    {int(nm)} days with data")
        if nm == 0:
            errors.append("MODIS has no data in the selected dates")
        for sid, s in it["sensors"].items():
            n = int((np.isfinite(sub[s["sst"]]).where(water).sum(("y", "x"))
                     >= int(s["min_pixels"])).sum().compute())
            print(f"{sid:8s} {int(n)} acquisitions (>= {s['min_pixels']} water px)")
            if n == 0:
                errors.append(f"sensor {sid} has no acquisitions in the selected dates")
            elif n < 8:
                warnings.append(f"sensor {sid} has only {n} acquisitions; offsets will be weak")
    ins = rc["validation"]["insitu"]
    if ins is not None and Path(ins).exists():
        print(f"in situ  {ins}")
    elif ins is not None:
        warnings.append(f"validation.insitu not found: {ins} (needed for stage 3)")
    elif "insitu_stations" in ds.attrs:
        print("in situ  the cube's own insitu_* channels")
    else:
        warnings.append("no in-situ data: set validation.insitu for stage 3")
    for w in warnings:
        print(f"WARNING  {w}")
    for e in errors:
        print(f"ERROR    {e}")
    print("OK" if not errors else f"{len(errors)} error(s)")
    return 1 if errors else 0


def cmd_stage(args) -> int:
    R, rc = _region(args)
    if not Path(rc["region"]["source"]).exists():
        raise SystemExit(f"region.source: no cube at {rc['region']['source']}")
    figs = not args.no_figures
    stages = ("offsets", "dineof", "validate") if args.command == "all" else (args.command,)
    for st in stages:
        if st == "offsets":
            import stage1_offsets
            stage1_offsets.run(rc, tag=args.tag, masks=getattr(args, "masks", None),
                               figures=figs)
        elif st == "dineof":
            import stage2_dineof
            stage2_dineof.run(rc, tag=args.tag, figures=figs)
        else:
            import stage3_validate
            stage3_validate.run(rc, tag=args.tag, figures=figs)
    return 0


def cmd_status(args) -> int:
    """What each stage has produced for this region and tag."""
    R, rc = _region(args)
    d = R.stage_dirs(rc, args.tag)
    print(f"{rc['region']['aoi']}{' [' + args.tag + ']' if args.tag else ''}: {d['base']}")
    off = d["offsets"] / "offsets.json"
    if off.exists():
        p = json.loads(off.read_text())
        print(f"  stage 1  done  (masks: {p['masks']})")
        for sid, s in p["sensors"].items():
            m = s["models"][s["chosen"]]
            print(f"           {sid}: {s['chosen']} (recommended {s['recommended']}), "
                  f"offset {m['a']:+.3f} K, slope {m['b']:.3f}, LOSO {m['loso_rmse']:.3f} K")
    else:
        print("  stage 1  not run")
    cube = d["dineof"] / f"{rc['region']['aoi']}_dineof.zarr"
    if cube.exists():
        import xarray as xr
        with xr.open_zarr(cube) as ds:
            fit = json.loads(ds.attrs.get("pipeline_fit", "{}"))
        print(f"  stage 2  done  point k={fit.get('k_point')} T_c={fit.get('tc_point')}, "
              f"day k={fit.get('k_day')} T_c={fit.get('tc_day')}, loop converged="
              f"{fit.get('loop_converged')} after {fit.get('loop_iterations')}")
        hold = d["dineof"] / "reports" / "cv_holdout.csv"
        if hold.exists():
            import pandas as pd
            h = pd.read_csv(hold)
            print(f"           CV holdout: {len(h)} dates, median RMSE {h['rmse'].median():.3f} K")
    else:
        print("  stage 2  not run")
    met = d["validation"] / "metrics.csv"
    if met.exists():
        import pandas as pd
        m = pd.read_csv(met)
        a = m[(m["stratum"] == "all")]
        print("  stage 3  done")
        for _, r in a.iterrows():
            print(f"           {r['match']:10s} {r['product']:17s} n={int(r['n']):5d}  "
                  f"MSE {r.get('mse', float('nan')):.3f}  bias {r['bias']:+.3f}")
    else:
        print("  stage 3  not run")
    return 0


def cmd_figures(args) -> int:
    """Re-draw stage 2's figures from its cube and reports, without re-running it."""
    R, rc = _region(args)
    import pipeline_figures
    import stage2_dineof
    import xarray as xr
    d = R.stage_dirs(rc, args.tag)["dineof"]
    cube = d / f"{rc['region']['aoi']}_dineof.zarr"
    if not cube.exists():
        raise SystemExit(f"no stage-2 cube at {cube}")
    dpi = int(rc["pipe"]["output"]["figure_dpi"])
    pipeline_figures.render(cube, d / "figures", d / "reports", dpi=dpi)
    with xr.open_zarr(cube) as ds:
        for sid, s in rc["pipe"]["_iter"]["sensors"].items():
            stage2_dineof.removed_scenes_figure(ds, sid, s["sst"], s["label"],
                                                d / "figures" / f"removed_scenes_{sid}.png", dpi)
    print(f"figures in {d / 'figures'} (the CV holdout figures need a full `dineof` run)")
    return 0


def cmd_fetch_insitu(args, rest) -> int:
    path = REPO / "scripts" / "fetch_insitu.py"
    spec = importlib.util.spec_from_file_location("fetch_insitu", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(rest)
    return 0


# ==================================================================== parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nearshore-sst", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")

    s = sub.add_parser("init", help="create a region config from the template")
    s.add_argument("aoi", help="region name; names the config and the output directory")
    s.add_argument("--source", required=True, help="the source datacube (zarr)")
    s.add_argument("--insitu", default=None, help="in-situ netCDF / CSV for stage 3")
    s.add_argument("--out", type=Path, default=None,
                   help="config path (default configs/region.<aoi>.yaml)")
    s.add_argument("--force", action="store_true", help="overwrite an existing config")
    s.set_defaults(func=cmd_init)

    def with_config(name, help_, func, **kw):
        s = sub.add_parser(name, help=help_, **kw)
        s.add_argument("--config", "-c", type=Path, required=True, help="region YAML")
        s.add_argument("--tag", default=None,
                       help="output variant: <out_dir>/<aoi>_<tag>/ (later stages read it)")
        s.set_defaults(func=func)
        return s

    with_config("check", "validate a config and its source cube; runs nothing", cmd_check)
    s = with_config("offsets", "stage 1: fixed vs fixed + RMA-slope offsets", cmd_stage)
    s.add_argument("--masks", type=Path, default=None,
                   help="pixel masks from a stage-2 cube instead of sensor QC alone")
    s.add_argument("--no-figures", action="store_true")
    s = with_config("dineof", "stage 2: offsets applied, iterative DINEOF cloud filter, fill",
                    cmd_stage)
    s.add_argument("--max-iter", type=int, default=None, help="override loop.max_iter")
    s.add_argument("--no-figures", action="store_true")
    s = with_config("validate", "stage 3: validation against in situ", cmd_stage)
    s.add_argument("--no-figures", action="store_true")
    s = with_config("all", "stages 1-3 in order", cmd_stage)
    s.add_argument("--max-iter", type=int, default=None, help="override loop.max_iter")
    s.add_argument("--no-figures", action="store_true")
    with_config("status", "show what each stage has produced", cmd_status)
    with_config("figures", "re-draw stage 2's figures from its cube", cmd_figures)

    s = sub.add_parser("fetch-insitu", add_help=False,
                       help="download IOOS in-situ temperature for a cube's footprint "
                            "(arguments as scripts/fetch_insitu.py; --help for them)")
    s.set_defaults(func=None)
    return p


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["fetch-insitu"]:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        return cmd_fetch_insitu(None, argv[1:])
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(message)s")
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
