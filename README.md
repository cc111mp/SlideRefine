# SlideRefine

Reusable [Nyul-style and empirical-CDF normalization](docs/HISTOGRAM_NORMALIZATION.md)
is available through `sliderefine.normalization` and `fit-histogram-lut`. Fit once
to an explicit frozen reference, save validated state, and apply it unchanged to
raw uint16 chunks. Reference population selection remains a study responsibility.

**Tile-aware correction and enhancement for gigapixel microscopy.**

SlideRefine is a research workspace for coordinate-aware, out-of-core microscopy processing. It combines a preserved Tissue/SNR-aware CLAHE v0.2 backend with a working tile-manifest reader, shared slide statistics, region rendering, a command-line interface, and tests.

**Release 0.2.0.** HiFiEM local/global contrast and a Simple Tone Curves discrete optimizer now have reference code and tile adapters. This is not a completed five-paper implementation or a giant-WSI benchmark. HiFiEM destriping/full restoration is NOT included. Visual-prior HE and multiscale redistribution remain unavailable; selecting them fails explicitly.

## What runs now

| Component | Status |
|---|---|
| Tissue/SNR-aware CLAHE, including optional PyTorch controller/trainer | Preserved tsclahe v0.2 source and original 110-test suite |
| Nonoverlapping tile files plus global coordinates -> virtual slide | Implemented; explicit single-channel, level-0 YX manifest |
| Neighboring-region reads, outside-slide halos, missing-data masks | Implemented and tested |
| Slide-wide uint8/uint16 histogram percentiles | Implemented; pooled exact counts, not average tile percentiles |
| Fixed-window normalization and CLAHE region rendering | Implemented reference runner |
| Separate baseline/enhanced NPY tiles, coordinates, validity, state, provenance | Implemented |
| HiFiEM local/global contrast (`contrast_af`) | Source-pinned contrast reference + explicit AF/halo adapter; not full HiFiEM |
| Simple Tone Curves | Independent discrete QP fitter + fixed-curve streaming; supplied target required |
| Visual-prior HE and multiscale redistribution | Unavailable; see method status and reference gaps |
| Native SVS/NDPI/OME-Zarr reader/writer, output pyramid | Not yet integrated |
| Disk-backed CLAHE state | Optional heuristic block store; reference equations retained; synthetic parity/RSS measured |
| Distributed scheduling and resume | Not implemented |
| Actual gigapixel throughput and real AF/downstream validation | Not yet measured |

## CLAHE provenance and training

`tissue_snr_clahe` is a custom microscopy CLAHE variant inspired by IA-CLAHE's
adaptive-control approach. Its default controls are deterministic tissue/SNR-aware
rules and require no enhancer training. The optional learned controller is our own
histogram/statistics-grid CNN and requires explicit paired reference targets.

The paper's no-ground-truth-clip-limits claim concerns parameter labels, not the
absence of reference images or training. See [paper supervision and provenance](docs/PAPER_RELATIONSHIP.md#training-supervision-in-the-paper)
and [our training contract](docs/TRAINING.md). There is no established AF target
standard or bundled pretrained AF controller. Training a downstream MIL classifier
on heuristic-enhanced images does not evaluate the paper's learned method.

## Tone-curve targets and fitting

`simple_tone_curves` fits a supplied brightness mapping with a constrained numerical
optimizer; it does not train a neural network or infer an ideal AF target. The paper's
expert-derived targets and Athena's separate patient-reference policy are described
in [the tone-curve reference](docs/references/simple_tone_curves.md). In the AF policy,
the cohort reference and each slide's curve are distinct fitted objects. Statistical
brightness matching is not diagnostic ground truth, and subsequent MIL training is
separate from curve fitting.

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

Select an appropriate PyTorch build for the intended machine. Do not replace a working accelerator build unnecessarily. Local validation of the new integrations uses Linux CPU. See the version-specific validation report and actual CI results; historical checks are not new runs.

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

## New reference methods: runnable examples

Create `demo_input` with the command above, then use fresh output paths:

```bash
sliderefine fit-tone-curve configs/tone_target.example.json fitted_curve.json
sliderefine run demo_input/manifest.json tone_output \
  --method simple_tone_curves --tone-curve fitted_curve.json \
  --limits 0 16000 --chunk-shape 128 160

sliderefine run demo_input/manifest.json hifiem_output \
  --method hifiem --method-config configs/hifiem_af.json \
  --limits 0 16000 --chunk-shape 128 160
```

**The target curve and HiFiEM parameters are synthetic/engineering examples, not
validated AF settings.** Simple Tone Curves fits a supplied mapping; it does not
infer the desired brightness from a slide. Its sampled constraints and PCHIP
interpolation limitations are explained in [the reference note](docs/references/simple_tone_curves.md).
HiFiEM here means **local/global contrast only**, not stripe removal, denoising or
background fitting. The [HiFiEM note](docs/references/hifiem.md) records the source
commit, license, exact included scope and AF adaptations.

The current source `src/tsclahe` remains unchanged. The new adapters share its
declared normalization object but do not replace its equations or controller.
New methods use fixed small state and owned-region rendering. Missing-neighbor
coverage in the HiFiEM halo triggers explicit baseline bypass, not invented pixels.
Pixel streaming alone does not establish measured gigapixel speed or a native
pyramid output implementation. Metadata lists still scale with tile count.

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
  fitted_state.json             # New reference methods only
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

`tsclahe` remains version 0.2.0; `sliderefine` is the new workspace version 0.2.0. The old backend's plane CLI is not a giant-slide reader. Its optional training entry point is `tsclahe-train`; see [training](docs/TRAINING.md).

## WSI boundary

Pixel working memory is bounded by requested regions and the tile cache. Each source tile is decoded as a whole under a size guard; large native pyramids need a future region adapter. The default CLAHE fit retains global histograms/LUTs and working temporaries in RAM. The optional `--state-backend disk` fits blocks of analysis cells and renders through a bounded state cache. `--state-block-shape` is measured in analysis cells; it does not change `Config.tile_size` in image pixels. `--max-state-mib` checks a conservative state working estimate, not whole-process RSS or the separate pixel cache. Do not enlarge the biological analysis scale merely to bypass the budget. See [disk-state validation](docs/DISK_STATE_VALIDATION.md) for measured synthetic memory, timing, and limitations.

The tile reader has tests on a sparse 100000x100000 coordinate canvas; that is **not** processing 10 billion real pixels. Full gigapixel performance, multichannel joint transforms, learned WSI execution, native pyramids, and production restart behavior remain milestones.

## Optional disk-backed CLAHE state

```bash
sliderefine run demo_input/manifest.json demo_disk \
  --config configs/af_conservative.json --limits 0 16000 \
  --state-backend disk --state-block-shape 8 8 --state-cache-mib 16 \
  --chunk-shape 128 160
```

Disk mode stores versioned `state/state.json`, independently hashed NPZ blocks
(histograms, diagnostics, features, controls, LUTs), block ledgers and a final
`COMPLETE.json`. It preserves the reference heuristic math and renderer. Loading
rejects incomplete/mismatched state; rendering verifies blocks on cache misses.
A supplied learned predictor is explicitly unsupported by this disk prototype.
Source pixels, masks and validity files are content-hashed and must stay immutable.
Memory mode and its `transform.npz` format remain available as the reference.
No automatic resume, native WSI adapter or output pyramid is added.

## Documentation

- [Disk-state validation](docs/DISK_STATE_VALIDATION.md)
- [AF parameter study design](docs/studies/AF_PARAMETER_STUDY.md) — classification not yet executed
- [New reference-method validation](docs/REFERENCE_METHODS_VALIDATION.md)
- [HiFiEM reference](docs/references/hifiem.md) and [Simple Tone Curves reference](docs/references/simple_tone_curves.md)
- [Multiscale source gap](docs/references/multiscale_redistribution.md)
- [Architecture](docs/ARCHITECTURE.md)
- [WSI contract](docs/WSI_CONTRACT.md)
- [Method implementation status](docs/METHOD_STATUS.md)
- [Next Codex milestones](docs/IMPLEMENTATION_PLAN.md)
- [Bootstrap validation](docs/BOOTSTRAP_VALIDATION.md)
- [Import identity](docs/review/BOOTSTRAP_IMPORT.json)
- [Backend equations](docs/METHOD.md), [paper relationship](docs/PAPER_RELATIONSHIP.md), [recorded v0.2 review](docs/review/REVIEW.md)

Historical v0.2 reports in docs/validation and docs/VALIDATION.md describe that release, not new SlideRefine results. Two large generated historical assets (the comparison PNG and detailed coverage JSON) are omitted; regenerate the demo/coverage locally. Source, tests, configs, and examples are retained.

No real images, private weights, or credentials are included. No public-release license has been selected; see LICENSE_NOTICE.md and THIRD_PARTY_NOTICES.md. Keep raw/calibrated data for quantification and QC/OOD: enhancement can alter intensities, spectral ratios, and failure cues.
