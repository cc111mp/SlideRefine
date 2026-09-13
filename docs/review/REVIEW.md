# Recorded review and dispositions — v0.2.0

**Date:** 2026-09-13. **Deliverable:** modified source, tests, examples, documentation and an installable wheel.
**Status:** executable research prototype; real microscopy efficacy and product suitability remain unverified.

## 1. Source identity: a necessary correction

The mounted `tissue-snr-ia-clahe-v0.1.0.zip` contains `src/tsclahe`, JSON configs, the `tsclahe`
CLI, a histogram/statistics tile-grid CNN and a reader-based streaming API. Its original 61 tests
passed when rerun before modification. The previous standalone audit instead identifies
`src/tissue_snr_clahe/clahe.py`, an MLP and a different preprocessing/operator implementation.
Those are not interchangeable source trees.

Original source ZIP SHA-256:

```text
fe55fe831305e13e2d2b178fdc90563ab62ca6cc8993ab9e91d76097606f59cc
```

The complete original source-file identity is in [SOURCE_BASELINE.json](SOURCE_BASELINE.json).
This review is specific to that available archive. The earlier audit supplied useful questions,
but its 60-test count, line numbers and numerical measurements are not claimed as evidence for
this `tsclahe` tree. The missing alternate source was not recreated from chat prose.

The archive and original audit were not overwritten. v0.2 is a separate release. Any downstream
consumer should use this release's equations and commands, not descriptions of the alternate variant.

## 2. What was actually executed

| Check | Result |
|---|---|
| Original available source, unchanged | 61 tests passed |
| Reviewed source, core and optional Torch | 110 tests passed |
| Additional cases in new regression files | 49 parametrized test cases |
| Separately installed wheel, same suite | 110 tests passed |
| Statement coverage, actual source | 91.31% |
| Random histogram stress | 224 cases, 32–4096 bins |
| Paired training smoke | 3 epochs, 4 training and 2 validation synthetic groups |
| Best-checkpoint load and CLI inference | Completed |
| Real AF, clinical endpoints, CUDA/MPS/GB10/Windows | Not executed |

Commands/logs are collected in [VALIDATION.md](../VALIDATION.md). Tests do not prove absence of
all defects. The wheel was installed into a separate target directory, using the same runtime
dependencies; this was not a cross-platform or minimum-dependency-version test.

## 3. Reproduced before/after cases against the actual archive

All images here are synthetic, use fixed [0,1] windows where appropriate, and use reviewed/all-tissue
masks. The new conservative artifact policy is enabled. These are controlled regression examples,
not a benchmark of microscopy enhancement quality.

| Case | Actual v0.1 source | v0.2 source |
|---|---:|---:|
| Constant rejected central tile: maximum change | 0.01068181 | 0.00000000 |
| Constant rejected tile: changed pixels | 100.0% | 0.0% |
| Simulated axis-aligned stripe: output/input standard-deviation ratio | 2.79247 | 1.00000 |
| 32 high-offset uint32 counts: distinct normalized values | 1 | 32 |
| Boolean mask loading | Rejected by intensity loader | Dedicated mask loader succeeds |

The stripe result is **bypass**, not stripe removal or improved SNR. Its HH-derived SNR proxy
remains high; the new separate directional uncertainty gate prevents enhancement in that test.
Real fibers can also be coherent and complex/non-axis-aligned artifacts may evade the gate.

Exact numerical evidence: [before.json](../validation/before.json) and
[after.json](../validation/after.json). Run `python examples/reproduce_review.py` to exercise the
current source. To reproduce historical results, use the original available source in a separate
interpreter with `PYTHONPATH` selecting its `src` directory.

## 4. Disposition of each prior audit concern

| Prior ID | Applicability to actual archive | v0.2 disposition |
|---|---|---|
| F01 rejected regions altered by neighboring mappings | Reproduced, although renderer differs | Fixed: global rejected-tile raster, post-interpolation bypass, inward taper and crop parity tests |
| F02 HH noise proxy blind to directional components | Reproduced | Partially mitigated: directional-uncertainty feature and configurable abstention; general artifact detection remains unresolved |
| F03 synthetic-degradation/default-normalization mismatch | That synthetic training path is absent; current paired path already calls `prepare` | Public checkpoint policy gaps fixed: fixed/percentile limits, explicit masks and calibration policy enforced in Python/CLI; equivalent training input tested |
| F04 regularly sampled percentiles alias periodic content | Not present; available source uses exact tissue percentiles | Retained exact behavior and added the alternating-column regression |
| F05 normalized clipping labeled sensor saturation | The available source did not contain that measurement | Added distinct lower/upper clipping and raw sensor state; unknown remains explicit |
| F06 Boolean masks rejected by intensity loader | Reproduced | Fixed with dedicated binary-mask reader in CLI/trainer; uint16 binary masks supported |
| F07 silent TIFF series selection | Already rejected by available source | Preserve rejection; add explicit checked selection and output provenance |
| F08 palette indices treated as image intensity | Already rejected by available source | Preserve intensity rejection; document/test binary palette-label mask semantics |
| F09 premature float32 conversion | Reproduced | Fixed across calibration, masking input and window subtraction; unsupported precision rejected |
| F10 missing loaded-plan endpoints/gain checks | Endpoints were already checked; available operator had no slope cap | Preserve endpoint checks; add actual gain cap, final histogram cap and schema checks |
| F11 impossible restoration targets | Exact gain-0.4 augmentation not present; arbitrary real targets can still be unattainable | Synthetic fixture replaced with reachable same-operator targets; real-reference attainability remains an evaluation task |
| F12 finite losses without useful learning | Training did not enforce active/gradient coverage | Fixed diagnostics and all-inactive/zero-gradient stopping; identity comparison and group-balanced validation added |

Rows marked “not present” or “already rejected” are not retroactively claimed as fixed bugs.
General scientific limitations are not reclassified as solved because a regression case passes.

## 5. Additional source-specific changes

### Pair controls with their own mappings

The archived renderer separately interpolated a strength field and an equalized image, then
multiplied them. That introduces cross terms between one tile's strength and another tile's
mapping. v0.2 instead computes each tile's bounded residual with its own strength and interpolates
the residuals. The NumPy and PyTorch renderers use the same expression.

### Cap the final histogram, not only the preliminary clip

The archived one-pass uniform redistribution can raise bins above its pre-redistribution cap.
v0.2 uses explicitly defined mass-preserving capped water filling. c=1 now yields an identity CDF.
A fixed-context gain bound is added to blending. This is a method change, not a claim of a faithful
paper implementation. Therefore old serialized plans and weights are rejected.

### Enforce a reliability decision at the final output

An identity LUT or zero strength at a tile center alone is insufficient. The final raster has
exact zeros throughout every rejected tile. Its transition is computed inward into accepted
pixels with a sufficient global-coordinate halo; no neighboring confidence is smoothed into
rejected pixels. Soft nonzero strengths still influence accepted neighboring regions; the raster
is not a validated pixel-wise probability or a complete solution to seams.

### Reference units and checkpoint policy

The paired trainer retains a shared input preparation path. Unit-encoded and raw-unit targets are
now explicit modes, and raw targets require shared bounds. Fixed limits are reused by loaded
checkpoints at public entry points and conflicting overrides fail. New in-pipeline calibration
is rejected for precorrected-input training policies. Low-level research APIs still require the
caller to supply consistent already-normalized arrays and masks.

### Reproducibility

Outputs record version/semantics, configuration, normalization, raw dtype, selected channel/series,
file digests and calibration/reference identity. Fitted plans include the normalization when
available. Training stores group/file/manifest provenance. Do not publish private manifests or
use metadata alone to infer that matching acquisition settings have been verified.

## 6. Important limitations retained explicitly

**Directional uncertainty is not biological validity.** The conservative gate can suppress real
oriented tissue and may miss oblique stripes, grid artifacts, texture-dependent noise or shading.
`artifact_policy="warn"` is an explicit ablation, not a validated default replacement.

**HH and robust-span statistics remain model-dependent proxies.** Correlation, Poisson behavior,
resampling and fine tissue texture can invalidate their interpretation. A calibrated noise model
and repeated acquisitions may offer stronger evidence but are not implemented here.

**Enhancement is not quantification or harmonization.** Location-dependent mappings can change
cross-region intensity relationships and multichannel ratios. Preserve raw/calibrated values and
the normalization-only branch. Do not erase QC/OOD cues upstream and assume the enhancer is a QC
system. No task improvement is established by a better-looking image.

**Reference attainability and learning validity remain research questions.** Synthetic reachable
targets only check optimization. Real reference targets may demand denoising, geometry repair or
out-of-family corrections. Group/path checks cannot establish biological independence or detect
all duplicated tissue stored under new filenames.

**Streaming is an adapter-level implementation.** It keeps global histogram/LUT state and requires
consistent image/mask/calibration readers. No native scanner-format reader/pyramid writer, giant-WSI
benchmark, automatic streaming percentiles or GPU-optimized reliability kernel is bundled. The
learned CNN uses a full tile-grid context, which can differ from standalone training ROIs.

**Numerical guardrails have stated scope.** Monotonicity and gain bounds apply with fixed fitted
context, not to derivatives through normalization/statistics/gates. They are not an SNR or diagnostic
safety theorem. File checks do not certify safe loading of arbitrary hostile large archives.

**Release permissions remain unresolved.** No publication license has been selected. The code is
not official author code or an employer-approved product release. No real-AF weights are shipped.

## 7. Test maintenance note

The original external-predictor test assumed clamped blend must equal the heuristic blend.
A higher proposed clip in v0.2 can require a LOWER blend under the new gain cap. The updated test
checks the original deterministic upper bound, the stronger gain restriction, and plan validation;
it does not remove the protective assertion to make a test pass.

The full executed suite retains the 61 original cases and adds 49 reviewed cases. Two warnings
are intentional: old unit tests manually construct unbound research models without a saved
preprocessing policy, and the public API now warns about that omission.

## 8. Next empirical decisions

Measure real-AF false rejection versus artifact amplification, calibrate noise/sensor settings,
establish same-modality reference scales, and run downstream patient/slide/session-held-out
ablations. Maintain a separate paper-aligned baseline before claiming superiority over IA-CLAHE.
Run native scanner I/O and speed/memory validation on the actual target machines before promotion.
