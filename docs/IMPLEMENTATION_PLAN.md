# Implementation plan for Codex

Work on one milestone per branch/PR. Read AGENTS.md first. Preserve the original backend and tests during infrastructure work.

## Implemented bootstrap
- Source-identity-checked tsclahe v0.2 import.
- Method registry and explicit unavailable-method errors.
- Selected-channel, coordinate-aware nonoverlapping tile reader with validity masks.
- Native integer histogram reduction, fixed normalization, reference rendering and NPY-tile output.
- Synthetic neighborhood, chunk-equivalence, mask, dtype, state-budget and CLI tests.

## Next PR: disk-backed state and native region adapters
1. Profile the retained global LUT/state allocation on representative inputs without changing the algorithm.
2. Define a versioned state store for histogram/features/controls/LUTs; include source/config/normalization/mask/model digests.
3. Fit and render chunks of the analysis grid using appropriate pixel and controller-grid halos. Prove equality with the retained in-memory reference at multiple chunk origins/sizes.
4. Add a dtype-preserving native TIFF/Zarr region adapter with explicit series/level/channel axes. Pin a compatible dependency set in an optional group. Never route detector data through RGBA rendering.
5. Add a minimal chunked output adapter and inspect axes/scale/translation roundtrips. Do not claim pyramid support before testing it.

## Later PR: scheduler and resumable outputs
Use immutable fitted state and bounded worker/cache budgets. Write a run fingerprint, completed-chunk ledger and atomic completion marker. A resumed run must reject changed inputs/config/state. Reject duplicate writes; avoid silently updating mappings while workers render. Measure real input I/O, peak RSS and output throughput.

## Paper-method PRs
- Simple Tone Curves: verify fitting equations, monotonicity and inflection constraints, and declare target-curve source. Add an independently checked curve fixture.
- Visual-prior HE: retrieve full equations/code, document optimizer and convergence, identify global versus spatial sufficient statistics.
- Multiscale redistribution: preserve scale-specific redistribution/fusion; define global lattices at all scales and native-versus-downsampled semantics.
- HiFiEM: pin verified upstream revision/licenses; reference adapter first, then separate AF stage configurations. Do not assume all EM stages are local or fluorescence-compatible.

## Final empirical milestone
Actual gigapixel dataset on target hardware, with native I/O, restart checks and full/chunk tests on reference ROIs. Evaluate weak tissue, oriented structures, stripes/grids, low-SNR, saturation, and mask failures. Compare normalization-only and independent methods before combinations. Distinguish frozen-model preprocessing swaps from matched retraining; split by slide/patient/session.

## Suggested first task prompt
Read AGENTS.md and docs/WSI_CONTRACT.md. Implement only a disk-backed state-store prototype behind the current CLAHE adapter. Keep the original backend untouched as the reference. Add chunk-size/origin equivalence tests, versioned state/provenance checks, and a memory measurement. Run the full suite and report unimplemented native readers/writers rather than silently substituting them.
