"""Zarr encoding and write helpers for output cubes.

Extracted from coastal_sst_data — only the three things this pipeline uses:
  * CompressionSpec          -- Pydantic model for Blosc compression settings
  * build_encoding(ds, c, chunks)  -- per-variable zarr encoding dict
  * write_zarr(ds, path, encoding) -- write xarray Dataset to zarr
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr
from pydantic import BaseModel, Field


class CompressionSpec(BaseModel):
    """Per-variable Zarr compression for the assembled datacube.

    Lossless: values are kept as-is (float32 / uint8); only a Blosc entropy
    codec is applied.
    """
    model_config = {"extra": "forbid"}
    codec: str = "zstd"
    level: int = Field(5, ge=0, le=9)
    shuffle: Literal["shuffle", "bitshuffle", "noshuffle"] = "shuffle"


_SHUFFLE = {"noshuffle": 0, "shuffle": 1, "bitshuffle": 2}


def _blosc_codec(cname: str, clevel: int, shuffle: str):
    """A Blosc codec for the installed Zarr (v3 BloscCodec, else numcodecs Blosc)."""
    import zarr
    if int(zarr.__version__.split(".")[0]) >= 3:
        from zarr.codecs import BloscCodec
        return BloscCodec(cname=cname, clevel=clevel, shuffle=shuffle), "compressors"
    from numcodecs import Blosc
    return Blosc(cname=cname, clevel=clevel, shuffle=_SHUFFLE[shuffle]), "compressor"


def build_encoding(ds: xr.Dataset, compression: CompressionSpec, chunks: dict,
                   *, sizes=None) -> dict:
    """Per-variable chunk + Blosc encoding dict for ds.to_zarr(encoding=...)."""
    dim_sizes = {**ds.sizes, **(sizes or {})}
    enc = {}
    for v in ds.data_vars:
        dims = ds[v].dims
        ch = tuple(min(chunks.get(d, dim_sizes[d]), dim_sizes[d]) for d in dims)
        shuffle = "bitshuffle" if ds[v].dtype == np.uint8 else compression.shuffle
        codec, key = _blosc_codec(compression.codec, compression.level, shuffle)
        e = {key: (codec,) if key == "compressors" else codec}
        if ch:
            e["chunks"] = ch
        enc[v] = e
    return enc


def write_zarr(ds: xr.Dataset, zpath: Path, encoding: dict, *, consolidated: bool = True):
    """Write a zarr cube to zpath. NOT atomic — callers drive store.atomic themselves."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*[Cc]onsolidated metadata.*")
        ds.to_zarr(zpath, mode="w-", consolidated=consolidated, encoding=encoding)
