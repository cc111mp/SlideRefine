# Executed SlideRefine bootstrap validation

Date: 2026-09-13. Linux CPU; synthetic data only. Historical v0.2 reports are separate.

## Actual checks

| Check | Result |
|---|---|
| Untouched reviewed v0.2 baseline | 110 passed, 2 expected warnings |
| SlideRefine source suite | 148 passed, 2 expected warnings (38 new workspace cases) |
| Editable installation with existing dependencies | Succeeded with --no-build-isolation --no-deps |
| Built wheel installed into a separate target directory | 148 passed, 2 expected warnings |
| CLI method registry and synthetic tile generation | Succeeded |
| CLI fixed-window CLAHE tile run | Completed; baseline/enhanced manifests and validity exported |
| Source/config Git tree identities before final upload | Matched the tested local source/config trees |

Warnings come from two preserved tests deliberately making unbound research checkpoints; they are not unexpected numerical failures.

The ordinary isolated pip installation could not fetch build dependencies because this runtime has no working network/DNS access to its package index. The successful editable/wheel builds used the installed dependencies. This does not verify clean installation, minimum versions, or other operating systems. GitHub Actions is configured separately; consult the actual run status, not this local report, for remote results.

## Commands exercised

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider
python -m pip install -e . --no-build-isolation --no-deps
python -m pip wheel . --no-build-isolation --no-deps -w /tmp/sliderefine-dist
python -m pip install --no-deps --target /tmp/sliderefine-wheel /tmp/sliderefine-dist/*.whl
PYTHONPATH=/tmp/sliderefine-wheel python -m pytest -q -p no:cacheprovider
sliderefine methods
sliderefine demo /tmp/sliderefine-input
sliderefine run /tmp/sliderefine-input/manifest.json /tmp/sliderefine-output --limits 0 16000 --chunk-shape 128 160 --config configs/af_conservative.json
```

Paths above are portable equivalents of the run directories. Use fresh output directories.

## Environment
Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, Pillow 12.3.0, tifffile 2026.5.15, PyTorch 2.10.0+cpu, pytest 9.0.2, setuptools 82.0.1.

## What the new tests establish
Neighbor-crossing reads, halo boundaries, dtype and mask contracts, missing-data propagation, overlap rejection, exact pooled integer percentiles, unavailable-method failures before writes, state allocation budget rejection, full/reference equivalence for three execution chunk shapes and shifted output origins, output rereading, and CLI behavior.

A sparse 100000x100000 coordinate canvas is queried in one small-region test. No test in this bootstrap processes 10 billion real pixels. Native pyramids, real AF, accelerator execution, throughput, restart, and disk-backed state remain unverified or unimplemented. There are no pretrained microscopy weights.
