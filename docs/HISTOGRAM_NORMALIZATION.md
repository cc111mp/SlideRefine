# Frozen histogram normalization

`sliderefine.normalization` provides reusable uint16-to-uint8 Nyul-style
landmark and dense empirical-CDF operators. Both fit one slide against a supplied
reference, save immutable state and apply the same lookup to every raw chunk.
These are independent AF adaptations. They do not reproduce the entire original
Nyul training protocol or infer the ideal brightness of a specimen.

The operators were extracted from research adapters after finding that NaN
landmarks could silently quantize to an apparently valid image, and non-finite
CDF masses could propagate NaNs. The reusable API rejects non-finite, negative,
empty, degenerate, incorrectly shaped and numerically unsafe inputs explicitly.
Existing valid-input formulas and quantization are retained.

## Fit once, apply to raw pixels

```python
from sliderefine.normalization import HistogramLUT, fit_uint16_histogram, fit_nyul
from sliderefine.wsi import TileManifestSource
from sliderefine.contracts import regions

source = TileManifestSource("manifest.json")
counts = fit_uint16_histogram(source)  # Pool valid tissue over the entire slide.
# Supply a reference fitted using the training patients for this evaluation fold.
# This vector is a synthetic engineering example only.
target = [0, .08, .18, .23, .28, .38, .5, .62, .72, .77, .82, .92, 1]
state = fit_nyul(counts, target)
state.save("slide_lut.npz")  # Refuses overwrite.
loaded = HistogramLUT.load("slide_lut.npz")
for region in regions(source.shape, (512, 768)):
    part = source.read_region(region)
    normalized = loaded.apply(part.pixels, valid=part.valid)
    # Feed this uint8 result to the encoder or a separately configured writer.
```

The selected tissue mask determines fitting only. Application covers all valid
pixels, including background. Missing coverage is zero-filled with its independent
validity mask retained by the caller. Repeated-grayscale RGB conversion and the
encoder's declared mean/std transform are separate. Do not fit per patch or apply
another percentile window after the LUT. Raw/calibrated data remain the quantitative
source; the uint8 output is a mapped encoder/display input.

There is also a command to fit from the existing tile-manifest source:

```bash
python -m sliderefine demo /tmp/histogram-demo
python -m sliderefine fit-histogram-lut /tmp/histogram-demo/manifest.json \
  configs/nyul_target.synthetic.json /tmp/histogram-demo-lut.npz
```

Use fresh paths. This command does not write normalized WSIs; application is the
Python API above. The existing `run --method` registry is unchanged. Native WSI
pyramids, scheduling and resume are separate capabilities. Source tiles must stay
immutable throughout fitting. State records the manifest and fitted histogram
digests, not a complete inventory of raw tile hashes.

## Numerical policies

- `histogram_quantiles` reproduces NumPy's linear sample-percentile convention
  using integer counts without expanding pixels. Counts must have an exactly
  representable total below 2^53; probability masses are not pixel counts.
- Nyul defaults to 13 percentiles: P1/P10/P20/P25/P30/P40/P50/P60/P70/P75/P80/P90/P99.
  The supplied target must be nondecreasing and span exactly [0,1]. Repeated source
  landmarks merge using the median target; endpoints stay fixed at 0 and 1.
  Intensities outside the endpoint landmarks clip to 0 or 1. Output is floored
  to uint8 only after applying the complete mapping. Constant source ranges fail.
- `cdf_values` interpolates each source cumulative probability on occupied target
  bins. `fit_cdf` requires 65,536 source and target bins. Target coordinates span
  normalized [0,1]; each bin can hold a finite count or probability mass. Output
  is `floor(mapped_bin / 65535 * 255)`. A target concentrated in one bin is valid
  and produces a constant mapping; it is not evidence of preserved tissue signal.
- `HistogramLUT` stores read-only lookup bytes and versioned method parameters,
  reference/histogram digests, range and quantization. Loading checks size, dtype,
  monotonicity, metadata and the LUT content digest. A checksum detects accidental
  LUT changes; it does not authenticate an untrusted reference or state file.

For CDF CLI fitting, supply JSON with exactly `method: "empirical_cdf_af"`,
`domain: [0,1]`, and a `target_histogram` array of 65,536 finite nonnegative masses.
For Nyul, use the schema of the supplied synthetic configuration. Never use the
synthetic reference as a validated AF default.

## Reference construction is a separate study policy

Use only the designated training population to fit a shared reference, and freeze
it before applying to held-out slides. A held-out slide may supply its own source
histogram; it must not update that reference. Patient balancing, mask selection,
outlier trimming and split membership belong in the study contract, not a hidden
default in these numerical operators. Refit references for each evaluation fold.

Historical AF adaptations used patient-balanced median Nyul landmarks and
patient-balanced mean CDF profiles. These aggregation rules differ from each
other and from upstream defaults. Their scientific utility needs independent
validation; numerical correctness does not establish diagnostic improvement.

## Reference relationship and tests

- [TorchIO histogram standardization, pinned source](https://github.com/TorchIO-project/torchio/blob/ced7e41bf796f709bfd9ba58abf2d1828c99d7e3/src/torchio/transforms/intensity/histogram_standardization.py):
  numerical reference for piecewise-linear mappings. The AF policy differs in
  masked slide fitting, explicit endpoint clipping, repeated landmarks and
  reference aggregation. No TorchIO source is vendored or required at import.
- [scikit-image histogram matching, pinned source](https://github.com/scikit-image/scikit-image/blob/533b7694d2004ae84e49e2cfd0bcfc5f8e562f22/src/_skimage2/exposure/histogram_matching.py):
  numerical CDF reference. Tissue masking, fixed lookup state and target population
  selection are AF adaptations. No scikit-image dependency is introduced.

Portable tests cover a hand-calculated CDF example, independent sample-rank
comparisons, exact histogram percentiles, duplicate landmarks, invalid inputs,
serialization, source preservation, byte order, validity, and identical output
for changed chunk sizes, origins and order. Real-AF quality and whole-slide
throughput are separate from these engineering checks.
