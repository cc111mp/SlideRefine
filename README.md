# SlideRefine

**Tile-aware correction and enhancement for gigapixel microscopy.**

SlideRefine is a research workspace for coordinate-aware, out-of-core microscopy processing. It combines a preserved Tissue/SNR-aware CLAHE v0.2 backend with a working tile-manifest reader, shared slide statistics, region rendering, a command-line interface, and tests.

**Bootstrap release 0.1.0.** This is not a completed five-paper implementation, a clinically validated product, or a giant-WSI performance benchmark. Four additional method families are deliberately marked **planned** and fail explicitly when selected.

## What runs now

| Component | Status |
|---|---|
| Tissue/SNR-aware CLAHE, including optional PyTorch controller/trainer | Preserved tsclahe v0.2 source and original 110-test suite |
| Nonoverlapping tile files plus global coordinates -> virtual slide | Implemented; explicit single-channel, level-0 YX manifest |
| Neighboring-region reads, outside-slide halos, missing-data masks | Implemented and tested |
| Slide-wide uint8/uint16 histogram percentiles | Implemented; pooled exact counts, not average tile percentiles |
| Fixed-window normalization and CLAHE region rendering | Implemented reference runner |
| Separate baseline/enhanced NPY tiles, coordinates, validity, state, provenance | Implemented |
| HiFiEM, visual-prior HE, multiscale redistribution, Simple Tone Curves | Planned; see method status |
| Native SVS/NDPI/OME-Zarr reader/writer, output pyramid | Not yet integrated |
| Disk-backed CLAHE state, distributed scheduling, resume | Planned; current state is RAM-backed with an explicit budget guard |
| Actual gigapixel throughput and real AF/downstream validation | Not yet measured |

## Install

Python 3.10 or newer. From this repository:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
python -m pytest -q
sliderefine methods
```

Core inference does not require Torch. Optional controller/training support:

```bash
python -m pip install -e ".[learn,test]"
```

Select an appropriate PyTorch build for the intended machine. Do not replace a working accelerator build unnecessarily. This bootstrap was exercised on Linux CPU only.

## Synthetic end-to-end example

Use output directories that do not already exist:

```bash
sliderefine demo demo_input
sliderefine inspect demo_input/manifest.json
sliderefine run demo_input/manifest.json demo_output \
  --method tissue_snr_clahe \
  --config configs/af_conservative.json \
  --limits 0 16000 \
  --chunk-shape 128 160
```

The limits above are only for the synthetic example, not recommended AF settings. To fit a slide-wide window from supported integer data instead:

```bash
sliderefine run demo_input/manifest.json demo_percentile \
  --method normalization_only --percentiles 0.5 99.5
```

The histogram reducer pools valid tissue counts from disjoint regions. Floating/calibrated inputs require explicit bounds unless they remain supported uint8/uint16. No automatic per-tile normalization or masking is performed.

## Tile manifest

```json
{
  "schema": "sliderefine.tiles/v1",
  "slide_id": "example",
  "shape": [100000, 100000],
  "dtype": "uint16",
  "channel_id": "AF_0",
  "axes": "YX",
  "level": 0,
  "intensity_domain": "raw",
  "mask_policy": "supplied",
  "pixel_size_um": [0.5, 0.5],
  "tiles": [
    {"path": "tiles/tile_0.tif", "mask_path": "masks/tile_0.tif",
     "x": 0, "y": 0, "width": 1024, "height": 1024}
  ]
}
```

Paths resolve relative to the manifest (absolute local paths also work). Every source tile must be a selected 2-D single-channel TIFF/PNG/NPY. Real biological scale/pixel spacing must come from acquisition metadata. Unknown spacing can be null. `valid_path` optionally supplies a binary coverage mask. `mask_policy: all_valid` is an explicit assumption that all covered pixels are foreground.

Source rectangles must not overlap. Raw stage coordinates need registration/compositing first. Missing regions return zeros with valid=False and are excluded from statistics. Do not interpret the zero fill as measured darkness.

```python
from sliderefine import Region
from sliderefine.wsi import TileManifestSource

slide = TileManifestSource("manifest.json")
part = slide.read_region(Region(x=1000, y=2000, width=1152, height=1152))
# part.pixels, part.valid, part.tissue
```

## Output

```text
demo_output/
  run.json
  transform.npz                 # CLAHE only
  baseline/manifest.json
  baseline/y..._x....npy
  enhanced/manifest.json
  enhanced/y..._x....npy
  masks/y..._x....npy
  valid/y..._x....npy
```

Each branch manifest is readable by the same virtual-slide reader. Float32 output represents normalized intensity, not detector counts. Output writing is incremental, but completion metadata and output manifests are finalized only after a successful run. Interrupted outputs are labeled incomplete; automatic resume is not implemented.

## Existing backend commands remain available

```bash
tsclahe sample.tif --mask sample_mask.tif --config configs/af_conservative.json --output-dir results/sample
python examples/demo.py --out backend_demo
```

`tsclahe` remains version 0.2.0; `sliderefine` is the new workspace version 0.1.0. The old backend's plane CLI is not a giant-slide reader. Its optional training entry point is `tsclahe-train`; see [training](docs/TRAINING.md).

## WSI boundary

Pixel working memory is bounded by requested regions and the tile cache. Each source tile is decoded as a whole under a size guard; large native pyramids need a future region adapter. The CLAHE fit still retains global histograms/LUTs and working temporaries in RAM. The runner estimates that allocation and rejects runs above `--max-state-mib` (default 512). The estimate is a planning guard, not a measured peak-memory guarantee. Do not enlarge the biological analysis scale merely to bypass the budget.

The tile reader has tests on a sparse 100000x100000 coordinate canvas; that is **not** processing 10 billion real pixels. Full gigapixel performance, multichannel joint transforms, learned WSI execution, native pyramids, and production restart behavior remain milestones.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [WSI contract](docs/WSI_CONTRACT.md)
- [Method implementation status](docs/METHOD_STATUS.md)
- [Next Codex milestones](docs/IMPLEMENTATION_PLAN.md)
- [Bootstrap validation](docs/BOOTSTRAP_VALIDATION.md)
- [Import identity](docs/review/BOOTSTRAP_IMPORT.json)
- [Backend equations](docs/METHOD.md), [paper relationship](docs/PAPER_RELATIONSHIP.md), [recorded v0.2 review](docs/review/REVIEW.md)

Historical v0.2 reports in docs/validation and docs/VALIDATION.md describe that release, not new SlideRefine results. Two large generated historical assets (the comparison PNG and detailed coverage JSON) are omitted; regenerate the demo/coverage locally. Source, tests, configs, and examples are retained.

No real images, private weights, or credentials are included. No public-release license has been selected; see LICENSE_NOTICE.md and THIRD_PARTY_NOTICES.md. Keep raw/calibrated data for quantification and QC/OOD: enhancement can alter intensities, spectral ratios, and failure cues.
