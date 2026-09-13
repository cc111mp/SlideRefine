"""Exact uint8/uint16 tissue histogram reduction over owned, nonoverlapping regions."""
from __future__ import annotations
import numpy as np
from tsclahe.preprocess import Normalization
from ..contracts import regions

def fit_uint_window(source, percentiles=(0.5, 99.5), *, chunk_shape=(1024,1024)):
    if source.dtype.kind != "u" or source.dtype.itemsize not in (1,2):
        raise ValueError("Exact histogram fitting supports uint8/uint16 only; supply fixed limits for floats")
    if len(percentiles) != 2 or not 0 <= percentiles[0] < percentiles[1] <= 100:
        raise ValueError("Require 0 <= lower < upper <= 100")
    hist = np.zeros(int(np.iinfo(source.dtype).max)+1, dtype=np.uint64)
    for region in regions(source.shape, chunk_shape):
        part = source.read_region(region)
        values = part.pixels[part.valid & part.tissue]
        hist += np.bincount(values, minlength=len(hist)).astype(np.uint64)
    total = int(hist.sum())
    if total == 0:
        raise ValueError("No valid tissue pixels for slide-level normalization")
    cdf = np.cumsum(hist)
    def quantile(percent):
        rank = (total-1)*(float(percent)/100)
        lo, hi = int(np.floor(rank)), int(np.ceil(rank))
        a, b = np.searchsorted(cdf, [lo,hi], side="right")
        return float(a + (b-a)*(rank-lo))
    lo, hi = map(quantile, percentiles)
    if hi <= lo:
        return Normalization(0.0, float(len(hist)-1), "slide_uint_histogram_constant_fallback", True)
    return Normalization(lo, hi, "slide_uint_histogram_linear_percentiles")
