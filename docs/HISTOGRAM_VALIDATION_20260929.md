# Histogram normalization validation — 2026-09-29

This records a new run on the normalization publication branch, based on
`00d5454`, using Python 3.11 on Linux. It is an engineering correctness review,
not an efficacy evaluation or an unchanged upstream-paper reproduction.

## Results

| Check | Result |
|---|---|
| Full source suite: `PYTHONPATH=src python -m pytest -q -p no:cacheprovider` | 248 passed |
| Full suite against an installed wheel, from outside the checkout | 248 passed |
| Existing `src/tsclahe` versus base commit | Byte-identical |
| Pinned scikit-image helper, 20 seeded synthetic fixtures | Maximum absolute error 0 |
| Pinned TorchIO mapping, 65,536 uint16 codes and distinct source landmarks, explicit output clipping | Maximum uint8 difference 1 |
| Existing fitted AF Nyul lookup states (local, read-only comparison) | All 166 tables exactly equal |
| LumaPath companion valid-input regression | All 65,536 uint16 codes exactly equal to historical arithmetic at five scales, both byte orders |

Both full SlideRefine suites emit the same two existing warnings for deliberately
unbound research-checkpoint fixtures in `test_torch.py`. No failure is suppressed.
Wheel validation uses the shared scientific runtime without replacing packages.
Windows and other Python versions are delegated to repository CI.

## Reference checks

Source revisions and declared AF deviations are linked in
[histogram normalization](HISTOGRAM_NORMALIZATION.md). The optional local reference
comparison executes the pinned numerical helper functions on synthetic arrays;
no upstream source is vendored or imported at package startup.

The CDF fixtures use uint16 random arrays of shapes 23x31 and 17x43 for seeds
0..19, comparing every observed source pixel to `_match_cumulative_cdf`.
The TorchIO fixture uses every uint16 code, its default 13 percentiles, and a
uniform target from 0 to 100, then clips and quantizes to the AF uint8 domain.
Float32 upstream arithmetic can differ by one quantized level. Duplicate source
landmarks and out-of-range behavior follow the explicitly different AF policy.

The historical LUT comparison accesses existing local research state read-only;
no raw specimens, reference population manifests or fitted cohort states are
distributed. Portable repository tests instead provide hand-calculated values,
sample-rank oracles and synthetic failure cases without private dependencies.

## Boundaries

The review found invalid-input handling gaps in research adapters: NaN landmark
targets could cast to plausible uint8 output, and NaN/Inf CDF masses could propagate.
The new API rejects those inputs, malformed domains and numerically unsafe masses.
It does not alter or reseal any prior study source or state.

The existing HiFiEM contrast and Simple Tone Curves numerical/reference tests also
pass in the full suite. HiFiEM remains contrast-only; tone targets remain explicit;
CLAHE remains the preserved custom implementation. Unavailable paper methods are
unchanged. No new claim about AF diagnostic quality, faithful learned IA-CLAHE,
clinical validity, native WSI export or gigapixel throughput follows from this review.
