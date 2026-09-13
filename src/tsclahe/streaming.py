"""Reader-based WSI fitting/application without a full-slide pixel array.

The caller owns slide I/O, a GLOBAL mask, and fixed normalization limits.
A reader has signature reader(y0, y1, x0, x1) -> 2-D array. Any dark/flat
correction must be consistently applied inside the image reader in both passes.
No independent crop normalization or crop-specific CLAHE grids are used.
"""
from __future__ import annotations

from typing import Callable, Iterator
import numpy as np
from .config import Config
from .core import TileGrid, FittedTransform, transform_from_statistics
from .preprocess import Normalization, validate_image, validate_mask
from .statistics import TileStatistics, summarize_tile

Reader = Callable[[int, int, int, int], np.ndarray]


def _read(reader: Reader, bounds, name):
    y0, y1, x0, x1 = bounds
    a = np.asarray(reader(y0, y1, x0, x1))
    if a.shape != (y1 - y0, x1 - x0):
        raise ValueError(f"{name} reader returned {a.shape} for region {bounds}")
    return a


def fit_streaming(shape: tuple[int, int], image_reader: Reader, mask_reader: Reader,
                  normalization: Normalization, config: Config | None = None,
                  *, predictor=None, saturation_reader: Reader | None = None) -> FittedTransform:
    """Fit global LUTs in one tile pass; memory scales with the LUT grid.

    Limits must be fixed from a prior slide-level/reference measurement.
    A previously calculated percentile Normalization is also acceptable;
    it is held constant throughout this pass.
    """
    if len(shape) != 2 or any(type(v) is not int or v < 1 for v in shape):
        raise ValueError("shape must contain two positive Python integers")
    cfg = config or Config()
    if predictor is not None and hasattr(predictor, "validate_normalization"):
        predictor.validate_normalization(normalization)
    if cfg.sensor_max is not None and saturation_reader is None:
        raise ValueError("Streaming with sensor_max requires a pre-calibration saturation_reader")
    grid = TileGrid(shape, cfg.tile_size)
    stats = TileStatistics.allocate(grid.grid_shape, cfg.bins)
    for i, j, y0, y1, x0, x1 in grid.tiles():
        bounds = y0, y1, x0, x1
        tile = normalization.apply(validate_image(_read(image_reader, bounds, "Image")))
        mask = validate_mask(_read(mask_reader, bounds, "Mask"), tile.shape)
        saturated = None if saturation_reader is None else validate_mask(
            _read(saturation_reader, bounds, "Saturation"), tile.shape)
        summarize_tile(tile, mask, stats, (i, j), cfg, sensor_saturated=saturated)
    result = transform_from_statistics(grid, stats, cfg, predictor)
    result.normalization = normalization
    if normalization.degenerate:
        result.strengths.fill(0)
    return result


def apply_streaming(transform: FittedTransform, image_reader: Reader, mask_reader: Reader,
                    normalization: Normalization, *, chunk_size=(1024, 1024)) -> Iterator:
    """Yield (y0, x0, output_chunk); write chunks to a caller-owned slide store.

    Chunk size does NOT change the transform. Images, masks, calibration,
    normalization limits and global coordinates must match the fitting pass.
    """
    if len(chunk_size) != 2 or any(type(v) is not int or v < 1 for v in chunk_size):
        raise ValueError("chunk_size must contain two positive integers")
    if transform.normalization is not None and transform.normalization != normalization:
        raise ValueError("Application normalization differs from the fitted transform")
    h, w = transform.grid.shape
    ch, cw = chunk_size
    for y0 in range(0, h, ch):
        for x0 in range(0, w, cw):
            bounds = y0, min(y0 + ch, h), x0, min(x0 + cw, w)
            tile = normalization.apply(validate_image(_read(image_reader, bounds, "Image")))
            mask = validate_mask(_read(mask_reader, bounds, "Mask"), tile.shape)
            yield y0, x0, transform.apply_region(tile, mask, origin=(y0, x0))
