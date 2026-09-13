# SlideRefine development instructions

Read README.md, docs/WSI_CONTRACT.md, docs/METHOD_STATUS.md, and docs/IMPLEMENTATION_PLAN.md before changes.

## Non-negotiable contracts
- Storage tiles and execution chunks are not independent images. Use global level-0 XY coordinates; NumPy arrays are YX.
- With the same fitted slide state, changing processing chunk size/origin/order must preserve the intended output within a declared numerical tolerance.
- Separate neighboring raw pixels, neighboring analysis-grid state, and truly global statistics. Derive halos from the complete operation; do not assume 3x3 stored tiles always suffice.
- Missing coverage is valid=False, not measured zero background. Count overlap only once. Reject unregistered/overlapping source tiles until an explicit compositing policy exists.
- Do not fit normalization per inference patch. Do not calibrate or normalize twice. Do not treat AF channels as photographic RGB.
- Preserve raw/calibrated data and the normalization-only branch. Enhancement does not establish quantitative fluorescence, harmonization, OOD safety, or clinical validity.
- Keep src/tsclahe and its original tests byte-identical to the imported v0.2 baseline during infrastructure work. Any later mathematical change requires a separate PR, updated method specification, and new regression tests.
- Planned paper methods must fail explicitly; never silently substitute ordinary CLAHE, gamma, identity, or an invented objective under a paper's name.
- Do not publish real specimens, private manifests, calibration acquisitions, credentials, or pretrained private weights. Synthetic fixtures only.
- No automatic upstream downloads or heavyweight imports at package import time. Pin and document external code/license decisions before vendoring.

## Commands
```bash
python -m pip install -e ".[test]"
python -m pytest -q
python -m pip install -e ".[learn,test]"  # optional CPU/GPU Torch already chosen for this machine
python -m sliderefine methods
python -m sliderefine demo /tmp/sliderefine-demo
```
Use fresh output paths. The demo and runner intentionally refuse existing outputs.

## Definition of done
Run affected tests and the full suite where dependencies permit. Record exact commands/results and remaining limitations. Never cite historical tests as a new run. Distinguish reference fidelity, synthetic correctness, real-AF quality, and gigapixel performance. Do not invent missing metadata, paper equations, benchmarks, or licenses.

## Next work
Start with bounded disk-backed state and native TIFF/Zarr region adapters, preserving the existing runner as a reference. See docs/IMPLEMENTATION_PLAN.md. Work one milestone per PR; do not implement all pending algorithms at once or expand into learned operator mixtures yet.
