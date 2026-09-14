# HiFiEM: source-pinned contrast reference and AF adapter

## Source and included scope

Kreinin et al., *High-fidelity Image Restoration of Large 3D Electron Microscopy Volume*.
Journal paper: https://academic.oup.com/mam/article/30/5/889/7826595
Author repository: https://github.com/image-infomatics/hifiem
Pinned commit: `477efb57dc2b2d24e1ad56d1044cf99d22d39957`.
Upstream `python/fltemd.py` Git blob: `885114c73e2c04224750340508856c5f2b07dfef`.
Function: `adjust_histogram_v_3_02` (starting near upstream line 1650).

**Only local/global contrast is integrated.** The paper's full restoration pipeline,
stripe prediction, denoising, cross-section correction and fitted background models
are NOT implemented by the SlideRefine `hifiem` method. Those can be added separately
against their own pinned references. Do not call this a full HiFiEM reproduction.

`_vendor/contrast_reference.py` contains the function's computational body. The long
upstream docstring is shortened and a minimal `kernel.convn` compatibility shim is
provided for its explicit one-dimensional kernels. Imports are limited to the needed
NumPy/SciPy components. The retained function preserves upstream defaults, side effects
and inclusive output-bin convention; it is not the public AF input-validation API.
Apache-2.0 LICENSE and change/provenance NOTICE are included adjacent to it and in wheels.
No specimen example, full notebook, paper PDF, or external helper library is vendored.

## Public AF variant

`src/sliderefine/methods/hifiem/contrast.py` implements `HiFiEMContrastConfig` and
`enhance_contrast_af`. The runner requires the explicit `contrast_af` variant.
Inputs are one normalized [0,1] channel and a supplied Boolean tissue mask. Calibration
and normalization are performed before this operator, once at a declared scope.

The supported local path is upstream `method="image"`, with no range-compression path.
Let A be a separable triangular smoothing operator formed by correlating a box kernel
with itself; m is the tissue mask. The low-pass and local midpoint are computed from
masked brightness and morphological extrema, then a brightness-dependent multiplier
scales the difference between the image and the midpoint. The following global stage
uses the upstream clipped normal-CDF mapping and `leftmost`, `left`, or `full`
redistribution. See the preserved function for exact arithmetic.

| Component | Code | Test |
|---|---|---|
| Local/global reference arithmetic | `_vendor/contrast_reference.py` | `test_hifiem_contrast_against_pinned_excerpt` |
| Normalized intensity interface | `enhance_contrast_af` | Masked and all-tissue upstream comparison |
| Global CDF stage with local enhancement disabled | `enhance_contrast_af`, `ratio=0` | Independent analytic expression test |
| Neighborhood extent and ownership | `HiFiEMContrastConfig.halo`; `wsi/reference_methods.py` | Shifted-origin and multiple-chunk tests |
| Missing coverage | Post-operator full-neighborhood validity check | Gap bypass and crop consistency test |

## Explicit adaptations

- No EM intensity inversion and no uint8 quantization. Float data are internally
  scaled to 0..255 computational units and scaled back afterward.
- Upstream uses `(high-low+1)*CDF`; naively using `drange=(0,1)` would produce an
  incorrect factor of two. The adapter retains the 256/255 convention deliberately.
- Supplied normalized-unit sigma, offset, mean and hp_sigma are converted to that
  computational scale. Values in `configs/hifiem_af.json` are **engineering examples**,
  not tuned or validated AF defaults.
- Input is copied and non-tissue output is restored exactly to the normalized baseline.
  Upstream can modify floating input outside its mask.
- Mask-normalized smoothing is always used, even on all-tissue regions. Avoiding the
  upstream all-mask shortcut prevents chunk-dependent switching. Small floating-point
  differences versus that shortcut are allowed; tests use absolute tolerance 5e-7.
- This variant supports only odd finite window sizes and symmetric CDF clipping.
  Other options are rejected rather than described as equivalent to upstream.

## WSI contract and limitations

With `smoothing=s`, the triangular kernel has width `2*s-1`, hence radius `s-1`.
The min/max path runs in parallel with radius `minmax_size//2`. Their maximum is the
required halo for the supported `method="image"`, no-compression variant. At ratio=0,
the operator is pointwise and halo=0. Do not reuse this halo for other HiFiEM stages.

The halo is read in slide coordinates and clipped only at true slide edges, where
SciPy mirror conditions reproduce the reference. Internal missing coverage is not
measured dark background: output pixels whose dependency neighborhood includes a gap
are conservatively bypassed. This is an AF/WSI policy, not an upstream paper claim.

The retained normal-CDF contrast transform can greatly alter brightness and amplify
noise. It is not a denoiser, artifact detector, quantitative normalizer, or established
improvement for virtual staining. Evaluate against the normalization-only branch.
