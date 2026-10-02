# Disk-backed heuristic CLAHE state — executed validation, 2026-09-14

Base: `00d5454` (GitHub main when cloned). Local branch:
`feat/wsi-disk-backed-state`. This milestone changes execution/storage, not the
reference equations. No native reader/writer or learned controller is added.

## Implementation

`wsi/state_store.py` fits bounded blocks of analysis cells with the existing
`summarize_tile`, `transform_from_statistics`, and `TileStatistics.features`.
The heuristic is pointwise in the analysis grid, so its fit requires no grid halo.
Each cell still reads its exact original pixel bounds, including partial edges.
Histograms, scalar statistics, feature channels, controls and LUTs are saved per
block. Global grid origin and centers remain the reference `TileGrid` values.

Integer-gather array proxies let the unchanged `FittedTransform.apply_region`
fetch neighbouring LUTs/strengths. A lazy eligibility proxy avoids its otherwise
full-grid `strengths > 0` allocation. The original interpolation and reliability
pixel halo are retained. No local refitting or substitute CLAHE is performed.

Versioned metadata binds normalization, config, source/mask/validity hashes,
reference backend hashes and explicitly null model identity for the heuristic.
Ordered block ledgers and final completion marker reject incomplete/mismatched
state. Blocks are checksum-verified and validated on cache misses. Cache arrays
are read-only. The manifest runner checks content identity before normalization
and again after fitting/rendering; callers of direct reader APIs must provide
truthful immutable identities. Inputs and state must remain immutable throughout
use, including while cached. This is integrity checking, not adversarial storage
security or atomic transactional snapshotting of concurrently modified inputs.

## Executed tests

- Untouched checkout: **212 passed, 2 expected warnings in 5.65 s**.
- Full available suite after implementation: **240 passed, 2 expected warnings in 10.47 s**.
- Added 28 cases. The existing tests and `src/tsclahe` are unchanged.
- Original two warnings concern intentionally unbound research Torch checkpoints.
- Executed CLI examples: disk CLAHE (9 output chunks), HiFiEM contrast (9), and
  Simple Tone Curves fit/application (9), all complete on synthetic data.

New tests compare every saved statistic, feature channel, control and LUT against
the RAM path exactly; rendered outputs use atol=2e-7, rtol=0. Test fixtures include
active enhancement, multiple state block sizes, shifted render origins, reversed
processing order, mask bypass, missing source coverage, partial/single-cell edges,
saturation, degenerate normalization, cache budgets, corrupted arrays/metadata,
changed input identity, incomplete fits and explicit rejection of learned fitting.

Commands (existing environment, no dependency replacement):

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /mnt/4TB_ssd/conda/envs/lazyslide-prototype/bin/python -m pytest -q
```

Logs are in `disk_state_validation/`. Python 3.11.16, NumPy 2.4.6; optional Torch
2.5.1+cu121 was already installed and its tests passed. The declared `learn` extra
requests Torch >=2.6, so this is not verification of the minimum declared optional
installation or a fresh wheel installation. No accelerator performance claim.

## Synthetic memory benchmark

Fresh Linux process for each case, `resource.getrusage(RUSAGE_SELF).ru_maxrss`.
The deterministic procedural reader generates only requested regions: no full
image array, source tile I/O or output tile files. Both modes fully fit and render
the stated number of real synthetic pixels, using identical cell size 16x16,
256 bins and 128x128 render chunks. Disk blocks: 8x8 cells; cache budget: 1 MiB.

| Pixels / analysis cells | Backend | Peak process RSS MiB | Fit seconds | Fit + render seconds |
|---|---|---:|---:|---:|
| 1024x1024 / 64x64 | RAM | 138.49 | 1.376 | 1.535 |
| 1024x1024 / 64x64 | Disk | 81.24 | 1.470 | 2.323 |
| 2048x2048 / 128x128 | RAM | 275.96 | 5.411 | 6.047 |
| 2048x2048 / 128x128 | Disk | 81.46 | 5.949 | 9.495 |

Initial process peaks were 76.7–76.9 MiB. Disk-cache peak was 994,560 bytes in both
cases; on-disk state was approximately 13.4 MB and 53.6 MB. Output SHA-256 was
identical between modes at each size. Enhancement was active on 898,047 and
3,594,237 pixels, respectively, so this was not an identity-only comparison.

Reproduce in fresh output paths:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/disk_state.py \
  --mode memory --size 2048 --output /tmp/clahe-memory-new
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/disk_state.py \
  --mode disk --size 2048 --output /tmp/clahe-disk-new
```

This is one run per case on a warm local machine, not a throughput distribution.
Disk mode is slower here; the objective was bounded grid-state memory and unchanged
results. The benchmark measures state fitting/rendering, not native WSI I/O,
source hashing overhead, output export, training, or biological quality.

## Remaining limitations

- No actual gigapixel or real-AF execution was run in this milestone.
- No DINO extraction, MIL retraining, or classification improvement is claimed.
- Learned controller grid halos are explicitly unsupported; no heuristic substitute
  is supplied under a learned model identity.
- No native TIFF/Zarr region adapter, pyramid output, distributed scheduler, resume,
  power-loss/fsync durability, or transactional multi-process writing.
- Cache and fit blocks are bounded; total RSS also includes source tile decoding,
  pixel chunks/halos, index/manifest metadata, and one-dimensional grid-center
  vectors. Extremely large pixel halos remain a separate resource concern.
- The manifest runner's source identity list and output tile manifests scale with
  tile count. State metadata scanning is serial. The block cache can be slower
  when configured too small, though numerical results remain unchanged.
- Original RAM mode and optional methods retain their existing source-immutability
  assumptions; content-hash strengthening here is specific to the new disk path.

The AF normalization/enhancement study is specified separately in
`studies/AF_PARAMETER_STUDY.md`. It deliberately separates execution parameters
from mathematical parameters and feature similarity from classification accuracy.
