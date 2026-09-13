# Whole-slide integration

## Why use one global grid?

Calling CLAHE independently for each neural-network patch changes the histogram context, normalization, LUT centers and interpolation boundary conditions. Adjacent model patches can then have different mappings for the same tissue. Overlap alone does not correct independently fitted statistics.

This package separates fitting from application. Both passes use one globally registered tissue mask, one set of intensity limits, and one global tile grid. “Chunk” means an I/O unit. “Tile” means a histogram/statistics unit. They do not need the same dimensions.

## Reader contract

```python
reader(y0: int, y1: int, x0: int, x1: int) -> numpy.ndarray
```

Coordinates are absolute, zero-based, half-open, and expressed at one chosen image resolution. The returned array must have shape `(y1-y0, x1-x0)`. Image readers return a grayscale plane, and mask readers return the corresponding binary mask.

A low-resolution mask must be mapped into exactly the selected resolution and origin by your adapter. No coordinate registration is silently inferred. Apply dark/flat correction identically inside the image reader in both passes when required. Do not compute the median of a flat-field reference independently per chunk: use a fixed calibrated scale for the slide/reference.

## Two-pass use

```python
import numpy as np
from tsclahe import Config, Normalization
from tsclahe.streaming import fit_streaming, apply_streaming

raw = np.load("raw.npy", mmap_mode="r", allow_pickle=False)
mask = np.load("mask.npy", mmap_mode="r", allow_pickle=False)
read_raw = lambda y0, y1, x0, x1: raw[y0:y1, x0:x1]
read_mask = lambda y0, y1, x0, x1: mask[y0:y1, x0:x1]

normalization = Normalization(low=100, high=24000, source="fixed")
config = Config(tile_size=(128, 128))
transform = fit_streaming(raw.shape, read_raw, read_mask, normalization, config)

output = np.lib.format.open_memmap(
    "enhanced.npy", mode="w+", dtype="float32", shape=raw.shape
)
for y0, x0, block in apply_streaming(
    transform, read_raw, read_mask, normalization, chunk_size=(1024, 1536)
):
    output[y0:y0 + block.shape[0], x0:x0 + block.shape[1]] = block
output.flush()
```

The intensity limits are illustrative. Estimate suitable slide-level or fixed-reference limits before this call. `fit_streaming` does not estimate streaming percentiles or an automatic global tissue mask. It expects these decisions to have been made consistently upstream.

The v0.2 transform NPZ contains LUTs, control fields and the fitted normalization, but not raw images, calibration references or masks. Retain the matching acquisition/mask metadata. A low-level fit_normalized transform has no raw-unit window unless the caller supplies one. The regular CLI already writes `metadata.json`. `examples/streaming_npy.py` is a runnable memory-map example.

## Memory and compute

Fitting retains tile histograms, statistics and LUTs; it does not retain a whole-slide image. Core state therefore scales approximately with `(grid_y * grid_x * histogram_bins)` rather than WSI pixel count. Small histogram tiles on a very large slide can still produce a large LUT grid. Application requires an output chunk, row-block indexing buffers and the global LUTs. It does not allocate a pixel-wise histogram-bin dimension.

A learned controller processes the whole tile-feature grid and may need substantial accelerator memory for a huge grid. Chunking the controller itself with the appropriate tile-grid halo is a future optimization; it must preserve its 3x3 neighborhood exactly. The default heuristic avoids this neural activation cost.

No native OpenSlide/NDPI/SVS adapter or pyramidal TIFF writer is supplied. Plug in a reader/writer tested on your own scanner metadata. A valid 2-D output is not equivalent to a scanner-compatible WSI. A real large-WSI performance test has not been run for this release.

## Numerical consistency versus image quality

Tests compare reader-based fitting and non-divisible output chunks to in-memory processing on synthetic arrays, including uint16 scaling and irregular edge tiles. They verify equal numerical results. This avoids **chunk-induced** inconsistency, but does not prove the absence of every visible artifact caused by histogram scale, tissue-mask boundaries, optical shading or spatially changing biological content.

The physical size of a tile is `tile_pixels * micrometers_per_pixel`. Record pixel size and keep physical context comparable across magnifications. Do not assume 128 pixels represents the same tissue neighborhood on different scanners.

## v0.2 reliability and saturation

Rejected tile footprints are rasterized in global coordinates. Application computes an inward
reliability taper with a pixel halo at least its radius, so crop boundaries do not introduce
confidence leakage. The code does not allocate an entire WSI reliability image.

When `Config.sensor_max` is supplied, `fit_streaming(..., saturation_reader=...)` requires a
reader returning the corresponding binary saturation mask measured on RAW data, before
any calibration applied by the image reader. It is not inferred from corrected brightness.

Learned checkpoint fitting validates the normalization policy. Application also rejects
a normalization that differs from the fitted transform. Masks and calibration are caller-owned
and must match both passes; identical normalization does not certify matching acquisition.
