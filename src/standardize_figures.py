"""Figures for the standardization: are the per-pixel seasonal fits any good?

Two views, both optional via `output.figure_*`:

  coefficients -- eight maps of the fitted cycle. THE figure for the question "do these look
                  reasonable", which is worth asking because 55k pixels are fitted completely
                  independently, sharing no data and no spatial prior, so nothing in the method
                  forces the maps to be smooth. Read the amplitude/SE panel first: it is the
                  per-pixel signal-to-noise of the thing being removed.
  fit_quality  -- where the three-tier ladder fell back, how many observations each pixel had,
                  and the roughness/SE ratio printed in the header so the judgement is a number.

THE NOISINESS TEST IS A NUMBER, NOT AN IMPRESSION. Roughness over standard error compares how
much neighbouring pixels disagree against how much each pixel's own sampling uncertainty says
they should. Well below 1 means independent fits are corroborating each other, which can only
happen if they are tracking real structure. On admiralty_inlet it is 0.16-0.20 for every term,
and a 3x3 spatial smooth would move the amplitude by 17% of one standard error -- which is why
this pipeline does no spatial regularization. If a new AoI comes back near or above 1, that
conclusion does not transfer, and `standardize.py` logs a warning saying so.

PHASE IS CIRCULAR. Day 364 and day 1 are adjacent, so peak-day-of-year gets a cyclic colour
scale. A linear ramp would paint a hard seam across new year that no physical process put
there, and would make a perfectly uniform phase field look like it has a front in it.

Not a script; imported by standardize.py.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap

import plotting

log = logging.getLogger("standardize_figures")

# The repo's chart palette, shared with cube_figures.py and composite_figures.py.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"

# Sequential ramps: one hue, light to dark. Never a rainbow for magnitude.
SEQ_COOL = LinearSegmentedColormap.from_list("seq_cool", ["#eef1f4", "#2a78d6", "#0b2444"])
SEQ_WARM = LinearSegmentedColormap.from_list("seq_warm", ["#fdf0e8", "#eb6834", "#5c1f06"])
SEQ_GREEN = LinearSegmentedColormap.from_list("seq_green", ["#eaf6f1", "#1baf7a", "#08382a"])
# Diverging: two hues, NEUTRAL GREY midpoint -- never a hue at zero.
DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#e8e8e6", "#eb6834"])


def _map(ax, g: np.ndarray, water: np.ndarray, extent, *, cmap, vmin, vmax, title,
         unit="", norm=None, ticks=None):
    """One (y, x) field over the water mask, with land flat grey and no-data distinct."""
    ax.set_facecolor(plotting.NODATA_COLOR)
    plotting.flat(ax, ~water, plotting.LAND_COLOR, extent)
    kw = dict(cmap=cmap, extent=extent, origin="upper", interpolation="nearest")
    im = ax.imshow(g, norm=norm, **kw) if norm is not None else \
        ax.imshow(g, vmin=vmin, vmax=vmax, **kw)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, ticks=ticks)
    cb.ax.tick_params(labelsize=7, colors=INK_SECONDARY)
    cb.outline.set_visible(False)
    if unit:
        cb.set_label(unit, fontsize=7.5, color=INK_SECONDARY)
    return im


def _grid(v: np.ndarray, water: np.ndarray) -> np.ndarray:
    g = np.full(water.shape, np.nan)
    g[water] = v
    return g


def _lim(g, lo=1.0, hi=99.0):
    v = g[np.isfinite(g)]
    if v.size == 0:
        return 0.0, 1.0
    a, b = float(np.percentile(v, lo)), float(np.percentile(v, hi))
    return (a, b) if b > a else (a, a + 1.0)


def coefficients_figure(cfg: dict, fit: dict, extent, path: Path) -> None:
    """The fitted cycle, its uncertainty, and what it leaves behind."""
    from seasonal_smoothing import FIT_FULL

    water, coef = fit["water"], fit["coef"]
    full = fit["fit_type"] == FIT_FULL
    keep = lambda v: np.where(full, v, np.nan)                  # noqa: E731

    mean_k = _grid(keep(coef[0]) - 273.15, water)
    amp = _grid(keep(fit["amp"][1]), water) if fit["H"] >= 1 else None
    se = fit["rough"]["se"]
    amp_se = None
    if fit["H"] >= 1:
        a, b = coef[1], coef[2]
        denom = np.maximum(np.hypot(a, b) ** 2, 1e-12)
        amp_se = np.sqrt((a ** 2 * se[:, 1] ** 2 + b ** 2 * se[:, 2] ** 2) / denom)

    fig, axes = plt.subplots(2, 4, figsize=(17.5, 8.8), dpi=cfg["output"]["figure_dpi"],
                             facecolor=SURFACE)
    a = axes.ravel()

    lo, hi = _lim(mean_k)
    _map(a[0], mean_k, water, extent, cmap=SEQ_COOL, vmin=lo, vmax=hi,
         title="mean term  (annual mean SST)", unit="degC")

    if amp is not None:
        _map(a[1], amp, water, extent, cmap=SEQ_WARM, vmin=0, vmax=_lim(amp)[1],
             title="harmonic amplitude", unit="K")

        # Phase is meaningless where there is no cycle to have a phase, so it is only drawn
        # where the amplitude clears its own uncertainty by 3x. Otherwise the panel fills with
        # the arctan2 of two numbers that are both noise.
        snr = np.where(full, fit["amp"][1] / np.maximum(amp_se, 1e-9), np.nan)
        pk = _grid(np.where(full & (snr > 3), fit["peak"][1], np.nan), water)
        _map(a[2], pk, water, extent, cmap="twilight", vmin=0, vmax=365.25,
             title="peak day of year  (cyclic scale; SNR > 3 only)", unit="day",
             ticks=[0, 91, 182, 274, 365])

    for idx, j, name in ((3, 1, "cos1"), (4, 2, "sin1")):
        if fit["H"] >= 1:
            g = _grid(keep(coef[j]), water)
            v = max(abs(_lim(g)[0]), abs(_lim(g)[1]))
            _map(a[idx], g, water, extent, cmap=DIV, vmin=-v, vmax=v,
                 title=f"{name} coefficient", unit="K")

    no = _grid(fit["n_obs"].astype(float), water)
    _map(a[5], no, water, extent, cmap=SEQ_GREEN, vmin=0, vmax=_lim(no)[1],
         title="observations per pixel", unit="n")

    sd = _grid(fit["scale"], water)
    lo, hi = _lim(sd)
    _map(a[6], sd, water, extent, cmap=SEQ_WARM, vmin=lo, vmax=hi,
         title="residual robust SD  (the standardizing scale)", unit="K")

    if amp_se is not None:
        snr_g = _grid(np.where(full, fit["amp"][1] / np.maximum(amp_se, 1e-9), np.nan), water)
        _map(a[7], snr_g, water, extent, cmap=SEQ_COOL, vmin=0, vmax=_lim(snr_g)[1],
             title="amplitude / its standard error  (SNR)  <- read this one first", unit="")

    ratios = "   ".join(f"{n}: {fit['rough']['per_term'][j]['ratio']:.2f}"
                        for j, n in enumerate(_terms(fit["H"])))
    fig.suptitle(
        f"Per-pixel seasonal fit  --  {cfg['data']['aoi']}  --  "
        f"{fit['counts']['full']} full fits of {fit['counts']['full'] + fit['counts']['mean_only'] + fit['counts']['reference']} water px\n"
        f"roughness / standard error   {ratios}    "
        "(well below 1 = independent neighbouring fits corroborate each other, so the maps "
        "are structure and not noise)",
        fontsize=11.5, color=INK, x=0.006, ha="left", y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    plotting.save(fig, path, dpi=cfg["output"]["figure_dpi"])


def _terms(H: int) -> list[str]:
    return ["mean"] + [f"{f}{k}" for k in range(1, H + 1) for f in ("cos", "sin")]


def fit_quality_figure(cfg: dict, fit: dict, extent, path: Path) -> None:
    """Where the three-tier ladder fell back, and how thin the data got there."""
    from seasonal_smoothing import FIT_FULL, FIT_MEAN_ONLY, FIT_REFERENCE

    water = fit["water"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), dpi=cfg["output"]["figure_dpi"],
                             facecolor=SURFACE)

    # Three ordered classes -> three ordered steps of ONE hue, not three categorical colours:
    # the tiers are a quality ladder, so darker should mean better rather than just different.
    tiers = ListedColormap(["#f2c0a8", "#e8834f", "#1d4f8c"])
    g = _grid(fit["fit_type"].astype(float), water)
    _map(axes[0], g, water, extent, cmap=tiers, vmin=None, vmax=None,
         norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], tiers.N),
         title="seasonal fit type", ticks=[0, 1, 2])
    axes[0].set_xlabel(
        f"0 reference ({fit['counts']['reference']})   "
        f"1 shape+mean ({fit['counts']['mean_only']})   "
        f"2 full ({fit['counts']['full']})",
        fontsize=7.5, color=INK_SECONDARY)

    no = _grid(fit["n_obs"].astype(float), water)
    _map(axes[1], no, water, extent, cmap=SEQ_GREEN, vmin=0, vmax=_lim(no)[1],
         title="observations per pixel", unit="n")

    zc = np.full(water.shape, np.nan)
    with warnings.catch_warnings():        # pixels with no observations: expected, handled
        warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
        zc[water] = np.nanmax(np.abs(fit["Z"]), axis=0)
    _map(axes[2], zc, water, extent, cmap=SEQ_WARM, vmin=0, vmax=_lim(zc, hi=99.5)[1],
         title="max |z| over time  (surviving outliers)", unit="")

    obs = fit["Z"][fit["O"]]
    fig.suptitle(
        f"Fit quality  --  {cfg['data']['aoi']}  --  "
        f"standardized sd {np.std(obs):.3f}, |z|>3 {100 * np.mean(np.abs(obs) > 3):.2f}%, "
        f"max |z| {np.max(np.abs(obs)):.1f}   "
        "(sd above 1 and a heavy tail are expected: a robust scale is deliberately not set "
        "by the outliers it exists to expose)",
        fontsize=11, color=INK, x=0.006, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    plotting.save(fig, path, dpi=cfg["output"]["figure_dpi"])


def render(cfg: dict, fit: dict, extent, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    o = cfg["output"]
    if o["figure_coefficients"]:
        coefficients_figure(cfg, fit, extent, out_dir / "coefficients.png")
    if o["figure_fit_quality"]:
        fit_quality_figure(cfg, fit, extent, out_dir / "fit_quality.png")
