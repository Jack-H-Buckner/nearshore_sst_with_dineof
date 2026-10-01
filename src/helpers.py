"""Filter an xarray time series down to the "valid" images.

Validity is expressed as a list of criteria, one per QA/quality channel::

    criteria = [
        {"channel": "eco_valid", "value": 1,       "prop": 0.75},
        {"channel": "eco_cloud", "value": "<0.5",  "prop": 0.75},
    ]

Each criterion means: *at least `prop` of the pixels in this scene must satisfy
`value` on channel `channel`*. A date is kept when every criterion passes
(or any, with ``require="any"``).

Accepted forms for ``value``
----------------------------
1                       exact match            -> da == 1
[1, 2, 5]               membership             -> da.isin([1, 2, 5])
"<0.5"                  comparison             -> da < 0.5
["<0.5", ">=0.1"]       all comparisons (AND)  -> (da < 0.5) & (da >= 0.1)
lambda da: da % 2 == 0  arbitrary predicate, returns a boolean DataArray

Supported operators: <, <=, >, >=, ==, !=
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
import xarray as xr

__all__ = [
    "Criterion",
    "build_predicate",
    "evaluate_criteria",
    "filter_valid_images",
    "iter_valid_images",
    "valid_dates",
]

# --------------------------------------------------------------------------- #
# value spec -> predicate
# --------------------------------------------------------------------------- #

_OPS: dict[str, Callable[[Any, float], Any]] = {
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
}

_COMPARISON = re.compile(
    r"^\s*(<=|>=|==|!=|<|>)\s*(-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)


def build_predicate(value: Any) -> Callable[[xr.DataArray], xr.DataArray]:
    """Turn a ``value`` spec into a callable DataArray -> boolean DataArray."""
    if callable(value):
        return value

    if isinstance(value, str):
        match = _COMPARISON.match(value)
        if match is None:
            # not a comparison expression: treat as a plain categorical value
            return lambda da, v=value: da == v
        op, number = match.groups()
        return lambda da, op=op, num=float(number): _OPS[op](da, num)

    if isinstance(value, (list, tuple, set, frozenset, np.ndarray)):
        items = list(value)
        if not items:
            raise ValueError("`value` list is empty")
        # plain scalars -> membership test; anything else -> AND of predicates
        if all(not callable(v) and not _looks_like_expression(v) for v in items):
            return lambda da, vals=items: da.isin(vals)
        preds = [build_predicate(v) for v in items]

        def _all(da: xr.DataArray, preds=preds) -> xr.DataArray:
            out = preds[0](da)
            for pred in preds[1:]:
                out = out & pred(da)
            return out

        return _all

    return lambda da, v=value: da == v


def _looks_like_expression(value: Any) -> bool:
    return isinstance(value, str) and _COMPARISON.match(value) is not None


# --------------------------------------------------------------------------- #
# criterion
# --------------------------------------------------------------------------- #

_CHANNEL_KEYS = ("channel", "chanel", "name", "var", "variable", "band")
_VALUE_KEYS = ("value", "values", "condition")
_PROP_KEYS = ("prop", "proportion", "min_prop", "threshold", "frac")
_NAN_KEYS = ("nan_policy", "nan")


@dataclass
class Criterion:
    """A single per-channel validity rule."""

    channel: Hashable
    value: Any
    prop: float = 1.0
    #   "invalid" - NaN pixels count against you (denominator = all pixels)
    #   "ignore"  - NaN pixels are dropped from numerator *and* denominator
    #   "valid"   - NaN pixels count as satisfying the condition
    nan_policy: str = "invalid"
    label: str | None = None
    _predicate: Callable[[xr.DataArray], xr.DataArray] = field(
        init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.prop) <= 1.0:
            raise ValueError(
                f"prop must be between 0 and 1, got {self.prop!r} "
                f"for channel {self.channel!r}"
            )
        if self.nan_policy not in {"invalid", "ignore", "valid"}:
            raise ValueError(f"unknown nan_policy {self.nan_policy!r}")
        self._predicate = build_predicate(self.value)
        if self.label is None:
            self.label = f"{self.channel} {_describe(self.value)}"

    @classmethod
    def from_spec(cls, spec: Any, default_prop: float = 1.0) -> "Criterion":
        """Build from a dict (or pass a Criterion straight through)."""
        if isinstance(spec, Criterion):
            return spec
        if not isinstance(spec, dict):
            raise TypeError(f"criterion must be a dict or Criterion, got {type(spec)}")

        spec = dict(spec)
        channel = _pop_first(spec, _CHANNEL_KEYS)
        if channel is None:
            raise KeyError(f"criterion is missing a channel key: {spec!r}")
        if not any(k in spec for k in _VALUE_KEYS):
            raise KeyError(f"criterion for {channel!r} is missing a value key")
        value = _pop_first(spec, _VALUE_KEYS)
        prop = _pop_first(spec, _PROP_KEYS)
        nan_policy = _pop_first(spec, _NAN_KEYS) or "invalid"
        label = spec.pop("label", None)
        if spec:
            raise KeyError(f"unrecognised keys in criterion: {sorted(spec)}")

        return cls(
            channel=channel,
            value=value,
            prop=default_prop if prop is None else float(prop),
            nan_policy=nan_policy,
            label=label,
        )

    def fraction(
        self, data: xr.Dataset | xr.DataArray, dims: Sequence[Hashable]
    ) -> xr.DataArray:
        """Fraction of pixels satisfying this criterion, per remaining dim."""
        da = _get_channel(data, self.channel)
        dims = [d for d in dims if d in da.dims]
        if not dims:
            raise ValueError(
                f"channel {self.channel!r} has no dimensions to reduce over"
            )
        mask = self._predicate(da)

        if self.nan_policy == "ignore":
            finite = da.notnull()
            count = finite.sum(dim=dims)
            hits = (mask & finite).sum(dim=dims)
            # all-NaN scenes -> NaN fraction -> fails the >= test below
            frac = hits / count.where(count > 0)
        else:
            if self.nan_policy == "valid":
                mask = mask | da.isnull()
            frac = mask.sum(dim=dims) / _size(da, dims)

        return frac.rename(str(self.label))


def _make_labels_unique(criteria: Sequence[Criterion]) -> None:
    """Two identical rules would otherwise collide into one report column."""
    seen: dict[str, int] = {}
    for crit in criteria:
        label = str(crit.label)
        if label in seen:
            seen[label] += 1
            crit.label = f"{label} #{seen[label]}"
        else:
            seen[label] = 1


def _pop_first(mapping: dict, keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping.pop(key)
    return None


def _describe(value: Any) -> str:
    if callable(value):
        return getattr(value, "__name__", "custom")
    if isinstance(value, str) and _COMPARISON.match(value):
        return value.replace(" ", "")
    if isinstance(value, (list, tuple, set, frozenset, np.ndarray)):
        return "in " + str(list(value))
    return f"== {value}"


def _size(da: xr.DataArray, dims: Sequence[Hashable]) -> int:
    n = 1
    for dim in dims:
        n *= da.sizes[dim]
    return n


# --------------------------------------------------------------------------- #
# channel lookup
# --------------------------------------------------------------------------- #


def _get_channel(data: xr.Dataset | xr.DataArray, channel: Hashable) -> xr.DataArray:
    """Pull a channel out of a Dataset (variable) or DataArray (coord label)."""
    if isinstance(data, xr.Dataset):
        if channel not in data:
            raise KeyError(
                f"channel {channel!r} not found; available: {sorted(map(str, data.data_vars))}"
            )
        return data[channel]

    for dim in data.dims:
        if dim in data.coords:
            labels = data[dim].values
            if labels.dtype.kind in "USO" and channel in set(labels.tolist()):
                # drop the scalar label coord, else channels won't merge later
                return data.sel({dim: channel}).drop_vars(dim, errors="ignore")

    if data.name == channel:
        return data

    raise KeyError(
        f"channel {channel!r} not found on DataArray {data.name!r} "
        f"(dims {tuple(data.dims)})"
    )


def _spatial_dims(
    data: xr.Dataset | xr.DataArray,
    criteria: Sequence[Criterion],
    time_dim: Hashable,
    spatial_dims: Sequence[Hashable] | None,
) -> list[Hashable]:
    if spatial_dims is not None:
        return list(spatial_dims)
    dims: list[Hashable] = []
    for crit in criteria:
        for dim in _get_channel(data, crit.channel).dims:
            if dim != time_dim and dim not in dims:
                dims.append(dim)
    if not dims:
        raise ValueError(
            f"no dimensions left to reduce over besides {time_dim!r}; "
            "pass spatial_dims explicitly"
        )
    return dims


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #


def evaluate_criteria(
    data: xr.Dataset | xr.DataArray,
    criteria: Sequence[Any],
    *,
    time_dim: Hashable = "time",
    spatial_dims: Sequence[Hashable] | None = None,
    default_prop: float = 1.0,
    require: str = "all",
) -> pd.DataFrame:
    """Score every date against every criterion.

    Returns a DataFrame indexed by date, with one column of observed fractions
    per criterion, one boolean ``pass: ...`` column per criterion, and a final
    ``valid`` column. Useful on its own for tuning thresholds.
    """
    if require not in {"all", "any"}:
        raise ValueError("require must be 'all' or 'any'")
    if time_dim not in data.dims:
        raise KeyError(f"{time_dim!r} is not a dimension of the data: {tuple(data.dims)}")

    parsed = [Criterion.from_spec(spec, default_prop=default_prop) for spec in criteria]
    if not parsed:
        raise ValueError("no criteria supplied")
    _make_labels_unique(parsed)

    dims = _spatial_dims(data, parsed, time_dim, spatial_dims)

    fractions = xr.Dataset({c.label: c.fraction(data, dims) for c in parsed})
    fractions = fractions.compute()  # single pass, dask-friendly

    frame = fractions.to_dataframe()
    frame = frame[[str(c.label) for c in parsed]]  # keep criterion order

    checks = {}
    for crit in parsed:
        # NaN fractions (e.g. an all-NaN scene under nan_policy="ignore") fail.
        checks[f"pass: {crit.label}"] = (frame[str(crit.label)] >= crit.prop).fillna(
            False
        )
    checks_frame = pd.DataFrame(checks, index=frame.index)

    combine = checks_frame.all(axis=1) if require == "all" else checks_frame.any(axis=1)
    out = pd.concat([frame, checks_frame], axis=1)
    out["valid"] = combine
    out.index.name = str(time_dim)
    return out


def valid_dates(
    data: xr.Dataset | xr.DataArray,
    criteria: Sequence[Any],
    *,
    time_dim: Hashable = "time",
    spatial_dims: Sequence[Hashable] | None = None,
    default_prop: float = 1.0,
    require: str = "all",
) -> np.ndarray:
    """The dates that pass, as an array of coordinate values."""
    report = evaluate_criteria(
        data,
        criteria,
        time_dim=time_dim,
        spatial_dims=spatial_dims,
        default_prop=default_prop,
        require=require,
    )
    return report.index[report["valid"]].to_numpy()


def filter_valid_images(
    data: xr.Dataset | xr.DataArray,
    criteria: Sequence[Any],
    *,
    time_dim: Hashable = "time",
    spatial_dims: Sequence[Hashable] | None = None,
    default_prop: float = 1.0,
    require: str = "all",
    return_report: bool = False,
):
    """Subset ``data`` to the dates whose images meet every criterion.

    Parameters
    ----------
    data
        Dataset (channels as variables) or DataArray (channels as labels along
        a dim such as ``band``), with a time dimension.
    criteria
        Sequence of dicts like ``{"channel": "eco_valid", "value": 1, "prop": 0.75}``.
    time_dim
        Name of the time dimension. Default ``"time"``.
    spatial_dims
        Dims to compute the pixel proportion over. Default: every dim except
        ``time_dim``.
    default_prop
        ``prop`` to use for criteria that omit it.
    require
        ``"all"`` (default) or ``"any"``.
    return_report
        Also return the per-date DataFrame of observed fractions.
    """
    report = evaluate_criteria(
        data,
        criteria,
        time_dim=time_dim,
        spatial_dims=spatial_dims,
        default_prop=default_prop,
        require=require,
    )
    keep = report.index[report["valid"]].to_numpy()
    subset = data.sel({time_dim: keep})
    return (subset, report) if return_report else subset


def iter_valid_images(
    data: xr.Dataset | xr.DataArray,
    criteria: Sequence[Any],
    *,
    time_dim: Hashable = "time",
    spatial_dims: Sequence[Hashable] | None = None,
    default_prop: float = 1.0,
    require: str = "all",
) -> Iterator[tuple[Any, xr.Dataset | xr.DataArray]]:
    """Yield ``(date, image)`` for each date that passes, one at a time."""
    report = evaluate_criteria(
        data,
        criteria,
        time_dim=time_dim,
        spatial_dims=spatial_dims,
        default_prop=default_prop,
        require=require,
    )
    for date in report.index[report["valid"]]:
        yield date, data.sel({time_dim: date})