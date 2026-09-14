# Executed validation — reference methods integration, 2026-09-14

This report concerns SlideRefine 0.2.0, not the historical tsclahe release.
No real AF, patient images, or gigapixel throughput benchmark was used.

## Baseline and scope

GitHub base commit: `bffdbc1b481664d57a917f14f821e7b21dba2a5d`.
Its root tree matched the locally imported snapshot:
`c5c1b90aa700a5d296b82e395d90ad28becbcd64`.
The untouched workspace was rerun: **148 passed, 2 expected warnings**.
`src/tsclahe` remains byte-identical to that snapshot. Original backend test files
are unchanged. `tests/test_workspace.py` only migrates unavailable-method assertions
for the two newly implemented IDs; multiscale and visual-prior IDs still fail.

## Executed checks

| Check | Actual result |
|---|---|
| Source suite including optional Torch | **212 passed**, 2 expected warnings |
| Separately installed wheel in an isolated directory | **212 passed**, 2 expected warnings |
| New reference-method test file | 66 parametrized cases |
| Synthetic example, both new methods | Completed, nine owned output chunks per method |
| License/notice packaging | Both files present in installed wheel |
| Real AF, native pyramids, actual gigapixel slide | Not executed |

The test-count calculation is 148 existing cases minus two migrated unavailable-method
parameters plus 66 new cases = 212. No backend regression assertion was weakened.
The warnings are preserved tests deliberately constructing an unbound research
checkpoint without a saved normalization policy.

Captured logs: `reference_validation/source_tests.txt`, `wheel_tests.txt`, `demo.txt`.
Runtime versions: `reference_validation/environment.json`.

Source command:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q
python -m pip wheel . --no-deps --no-build-isolation -w /external/wheels
```

For the wheel check, the wheel was installed with `--no-deps --target` into a new
directory; tests/configs were copied outside the source repository. PYTHONPATH
selected only that installed directory, and both module paths were asserted to be
inside it. Existing runtime dependencies were reused. This is not a clean minimum-
dependency installation or a cross-platform result. GitHub Actions results, when
available, are separate evidence associated with the final PR commit.

## Numerical evidence, not efficacy claims

HiFiEM's normalized AF contrast output is compared on tissue against the pinned
contrast excerpt for masked/all-tissue cases, three redistribution rules and two
local-strength settings (absolute tolerance 5e-7). A separate analytic expression
checks the global-only CDF stage. Input immutability and background bypass are tested.
Full/cropped execution and different owned chunk sizes agree within 2e-7 in the
synthetic fixtures, including missing-context bypass and true slide edges.

Simple Tone Curves checks all four shape families and full nontrivial candidate
coverage through n=100, plus a closed five-sample example. Five independent tiny-QP
checks use exhaustive active-face linear algebra rather than the production solver.
Serialization, endpoints, monotonic samples, rejection of invalid targets and solver
failures, curve application and tile rendering are tested. A stress check found tiny
out-of-range solver values at flat endpoints; redundant endpoint-range constraints
fixed this without changing the feasible set, and four regression cases were added.

These checks support the implemented subset under those examples. They do not prove
absence of all numerical problems, continuous PCHIP curvature simplicity, a favorable
AF target policy, preservation of diagnostic detail, or full HiFiEM fidelity.

## Remaining boundaries

- HiFiEM means local/global contrast only; no stripe correction or full restoration.
- Simple Tone Curves requires an explicitly supplied monotone target mapping.
- Multiscale redistribution and visual-prior HE remain unimplemented and fail clearly.
- NPY tile output, input tile indexing and per-tile metadata remain the existing model.
- These two adapters use small fixed state, but the older CLAHE state is still RAM-backed.
- No native pyramidal reader/writer, resumable scheduler or optimized GPU path is added.
- No new project-wide publication license is assigned; the selected vendor excerpt
  retains Apache-2.0 with its LICENSE and modification/provenance NOTICE.
