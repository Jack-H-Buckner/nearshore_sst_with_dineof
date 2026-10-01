"""Drawing primitives for the DINEOF scene-filtering figures.

Ported from prototypes/cloud_mixture_model/src/plot_daily.py rather than imported: the two
prototypes are deliberately independent trees, and this one needs mask panels and verdict
framing that the cloud-mixture figures have no use for. The colour VOCABULARY is kept
identical so a DINEOF figure and a cloud-mixture figure can be read side by side -- grey is
land, off-white is water this pass did not see, red is the sensor's own cloud raster,
magenta is observed-but-dropped.

Three visual states must stay distinguishable in every panel, because the whole point of the
filtering figures is to tell them apart: a MEASUREMENT, water the pass did not observe, and
land. Land is painted flat under the data and every colormap here sets `set_bad` transparent,
so land reads grey no matter which quantity a panel is showing.

Not a script; imported by filter_images.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

log = logging.getLogger("plotting")

# Shared with prototypes/cloud_mixture_model/src/plot_daily.py -- do not diverge.
LAND_COLOR = "#c8c8c8"      # land: not water, and never a temperature
NODATA_COLOR = "#f4f4f2"    # water this pass did not see
CLOUD_COLOR = "#e03131"     # the sensor's own cloud raster says cloud here
INVALID_COLOR = "#c2255c"   # observed, but <pre>_valid == 0: what the filter drops
VALID_COLOR = "#1971c2"     # observed and accepted by the sensor's own gate

PASS_COLOR = "#2f9e44"
FAIL_COLOR = "#c92a2a"


# --------------------------------------------------------------------------- cube access

def open_cube(path: Path) -> xr.Dataset:
    """Open an assembled cube. The package has no `load_cube` helper; `xr.open_zarr` on
    `<output_dir>/datacube/<aoi>.zarr` is the idiom used across both prototypes."""
    if not path.exists():
        raise SystemExit(
            f"no cube at {path}\n"
            "build it with: coastal-sst-data run --config "
            "prototypes/cloud_mixture_model/configs/config.admiralty.yaml --assemble")
    return xr.open_zarr(path)


def water_mask(ds: xr.Dataset, var: str = "landcover_water") -> np.ndarray:
    """`landcover_water` as boolean (y, x); True = water.

    EVERY proportion in this prototype is scored against this mask, never against the frame.
    On Admiralty Inlet it is 55,722 of 91,809 pixels -- land is 39% of the box, so a fraction
    taken over the frame is the true one scaled by 0.607 and transfers to no other AoI.
    """
    if var not in ds:
        raise SystemExit(f"cube has no `{var}`; acquire the `landcover` product")
    return ds[var].values > 0.5


def to_celsius(a: np.ndarray) -> np.ndarray:
    """The cube stores Kelvin unless `grid.to_celsius` was set; display in degC either way.

    Decided on the data, not on an attribute -- the assembler leaves `units` unset on these
    channels. Applied once to a whole per-sensor block rather than per scene, so a single
    all-cloud scene cannot flip the verdict for that scene alone.
    """
    finite = a[np.isfinite(a)]
    if finite.size and np.median(finite) > 150.0:
        return a - 273.15
    return a


def extent_km(ds: xr.Dataset) -> list[float]:
    """imshow extent in km. `y` descends (top-left origin), which `origin="upper"` expects."""
    x, y = ds["x"].values, ds["y"].values
    return [0.0, float(x.max() - x.min()) / 1000.0, 0.0, float(y.max() - y.min()) / 1000.0]


# --------------------------------------------------------------------------- colormaps

def sst_cmap():
    """Perceptually uniform, and cmocean's thermal map when it happens to be installed."""
    try:
        import cmocean
        cm = cmocean.cm.thermal
    except ImportError:
        cm = plt.get_cmap("viridis")
    cm = cm.copy()
    cm.set_bad(alpha=0.0)       # unobserved falls through to the axes facecolor
    return cm


def cloud_cmap():
    """Cloud fraction 0..1.

    `plasma`, not a grey or a white-to-red ramp, for one specific reason: the clear end of
    those ramps is near-white, which is indistinguishable from NODATA_COLOR (#f4f4f2), and
    "this raster says clear" would then look identical to "this raster says nothing here".
    That distinction is exactly what the `fill:` config key exists to handle, so the figure
    must not hide it. plasma runs dark indigo (clear) to bright yellow (cloud); neither end
    collides with the land grey or the nodata off-white, and bright-means-cloud matches how
    clouds read in the visible bands anyway.
    """
    cm = plt.get_cmap("plasma").copy()
    cm.set_bad(alpha=0.0)
    return cm


# --------------------------------------------------------------------------- drawing

def flat(ax, mask: np.ndarray, color: str, extent: list[float]) -> None:
    """Paint one boolean mask in a single flat colour; everything else transparent."""
    if mask is None or not mask.any():
        return
    cm = ListedColormap([color])
    cm.set_bad(alpha=0.0)
    ax.imshow(np.where(mask, 1.0, np.nan), cmap=cm, vmin=0, vmax=1,
              extent=extent, origin="upper", interpolation="nearest")


def panel(ax, data: np.ndarray, water: np.ndarray, *, vmin: float, vmax: float,
          extent: list[float], cmap=None, title: str | None = None,
          flag: np.ndarray | None = None, flag_color: str = CLOUD_COLOR):
    """Draw one (y, x) continuous field over the water mask; returns the image, for a colour
    bar.

    Three visual states, deliberately distinguishable: a value, water this pass did not
    observe (the axes facecolor), and land (flat grey). An optional `flag` mask is painted
    flat on top.
    """
    cmap = cmap if cmap is not None else sst_cmap()
    ax.set_facecolor(NODATA_COLOR)

    flat(ax, ~water, LAND_COLOR, extent)
    im = ax.imshow(np.where(water, data, np.nan), cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=extent, origin="upper", interpolation="nearest")
    if flag is not None and flag.any():
        flat(ax, flag, flag_color, extent)
    if title:
        ax.set_title(title, fontsize=8)
    return im


def mask_panel(ax, mask: np.ndarray, water: np.ndarray, *, extent: list[float], color: str,
               title: str | None = None, second: np.ndarray | None = None,
               second_color: str = INVALID_COLOR) -> None:
    """Draw one or two boolean masks flat over the water mask.

    The categorical counterpart to `panel`. Used for the `<pre>_valid` gate, where the three
    states that matter are accepted (blue), observed-but-rejected (magenta) and never
    observed (the facecolor) -- a continuous colormap would blur the first two together.
    """
    ax.set_facecolor(NODATA_COLOR)
    flat(ax, ~water, LAND_COLOR, extent)
    flat(ax, mask & water, color, extent)
    if second is not None:
        flat(ax, second & water, second_color, extent)
    if title:
        ax.set_title(title, fontsize=8)


def robust_limits(sst: np.ndarray, keep: np.ndarray,
                  lo: float = 1.0, hi: float = 99.0) -> tuple[float, float]:
    """One colour range per sensor, over the pixels that sensor's own gate calls valid.

    `sst` (t, y, x) in degC and `keep` the same shape: water & <pre>_valid & finite.

    Raw percentiles do not work on this cube. It runs from 227 K (Landsat cloud tops) to
    506 K (bad ECOSTRESS pixels), and the 1st percentile of ALL observed Landsat water is
    -33 degC. Letting that set the floor squeezes every real scene into the top of the bar.
    That contamination is precisely what this tool is being tuned to remove, so it must not
    be allowed to set the scale it is judged against. Scoring the limits on the sensor-clear
    subset instead spreads genuine SST over the full ramp and pushes contaminated pixels off
    the cold end, where a filtering figure wants them conspicuous.

    A DISPLAY choice only -- `<pre>_valid` is a reference the criteria are tuned against,
    never an input to the colour decision for the other panels.
    """
    vals = sst[keep]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    vmin, vmax = float(np.percentile(vals, lo)), float(np.percentile(vals, hi))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return vmin, vmin + 1.0
    return vmin, vmax


# --------------------------------------------------------------------------- verdict marks

def frame_verdict(ax, ok: bool, lw: float = 1.2) -> None:
    """Colour all four spines by the verdict.

    The contact sheet's panels are 1.85 inches wide, where a text stamp is marginal; the
    frame carries the pass/fail at a glance and the title text carries the detail.
    """
    color = PASS_COLOR if ok else FAIL_COLOR
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(lw)
        spine.set_visible(True)


def stamp(fig, ok: bool, x: float = 0.995, y: float = 0.985,
          words: tuple[str, str] = ("PASS", "FAIL")) -> None:
    """The boxed verdict mark on a single-scene figure.

    `words` is (affirmative, negative). The status green and red separate by an OKLab dE of
    only 6.1 under deuteranopia -- below the 8 that would let colour stand alone -- so the
    word is not decoration, it is the accessible channel. Callers pass the vocabulary their
    figure uses ("KEPT"/"DROPPED" for the cube build) rather than restating PASS/FAIL.
    """
    color = PASS_COLOR if ok else FAIL_COLOR
    fig.text(x, y, f" {words[0] if ok else words[1]} ", ha="right", va="top",
             fontsize=11, fontweight="bold", color=color,
             bbox=dict(facecolor="white", edgecolor=color, linewidth=1.4,
                       boxstyle="round,pad=0.35"))


# --------------------------------------------------------------------------- saving

def save(fig, path: Path, dpi: int | None = None) -> None:
    """mkdir -> savefig -> close -> log. The saving contract used across both prototypes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if dpi:
        fig.savefig(path, dpi=dpi)
    else:
        fig.savefig(path)
    plt.close(fig)
    log.info("wrote %s", path)
