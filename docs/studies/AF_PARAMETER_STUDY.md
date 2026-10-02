# AF normalization and enhancement study specification

Status: staged study design, not an executed classification result. The disk-state
engineering milestone is evaluated separately in `../DISK_STATE_VALIDATION.md`.
The existing AF cohort has already been inspected in many experiments; further
results on it are exploratory, even with careful held-patient evaluation. An
untouched patient/session cohort is needed to confirm a selected method.

## Questions and units

1. Does input normalization improve the frozen-DINO plus retrained-MIL classifier?
2. Does each enhancement add useful signal beyond its matched normalization-only input?
3. Are outputs independent of storage layout and execution chunking?

The patient is the split and resampling unit. Slides are classification units;
patches are nested measurements, never independent accuracy samples. A change in
feature cosine is not an accuracy change. A frozen-head preprocessing swap is a
robustness experiment, not a replacement for matched head retraining.

## Stage 0: execution and implementation gates

Keep tsclahe equations and reference tests unchanged. Compare RAM/disk statistics,
controls, LUTs, features and rendered outputs, including shifted chunk origins,
reversed order, rejected regions, gaps, partial cells and saturation handling.
Tolerance: float32 output absolute error <=2e-7, relative tolerance zero. Measure
fresh-process peak RSS and elapsed fit/render time at fixed biological analysis
scale. Block size, cache budget and processing chunks are execution parameters;
they must not be selected by classification accuracy. Synthetic parity and RSS are
not evidence for AF efficacy, native I/O, restart safety or actual gigapixel use.

## Stage 1: normalization, no enhancement

Retain a byte-reproducible original P99.5 input. The current Athena baseline is
zero-floor, whole-slide strided P99.5 -> uint8, then RGB replication, resize and
DINO ImageNet normalization. SlideRefine's pooled tissue histogram percentile is
NOT that same baseline. Do not swap percentile scope, quantization or subtraction
while calling an arm a baseline reproduction.

Fixed candidate grid (record all results, do not quietly discard poor arms):

| Family | Statistics scope | Parameters | Image mapping |
|---|---|---|---|
| Original baseline | Current whole-slide strided sample | P99.5 | clip(raw/P,0,1) then original uint8 arithmetic |
| Percentile sensitivity | Same sample | P99.0, P99.9 | Same zero-floor mapping |
| Z-score | Same whole-slide sample | k=3,4,6 | clip(0.5+(raw-mu)/(2*k*sigma),0,1) |
| Z-score | Supplied fold-correct tissue mask on the same sample lattice | k=3,4,6 | Same fixed mapping |

This is nine arms including the baseline. Z-score is the input intensity transform,
not standardization of already extracted feature dimensions. Both scopes must use
one frozen per-slide state for all patches and scales. Background scope is an
explicit factor; no per-patch auto contrast or display min/max normalization.
Use population sigma and float64 accumulation. Preserve the original baseline's
float32 percentile arithmetic. Every arm uses the same declared final uint8 and
DINO transform so a precision change is not hidden in a normalization comparison.
Do not infer sensor saturation from uint16 dtype or confuse it with window clipping.
Reject empty tissue, invalid moments and zero variance explicitly; do not add an
undocumented epsilon or substitute an unrelated normalization.

Before extraction, report low/high clipping separately over covered pixels,
supplied tissue pixels, and actual DINO crop pixels. Record source/mask/config
hashes, sample lattice, mean, sigma, percentile and raw extrema. Map masks using the
existing documented coordinate transform; do not infer physical spacing from a
filename or use a held-out patient's trained mask model. Preserve acquisition
metadata as unknown when unknown. Save fixed-scale previews with identical display
limits. Label-free diagnostics can expose unsuitable mappings, but all exclusion
rules and any resulting protocol revision must be recorded before MIL evaluation.

The previous +/-3 SD whole-slide experiment already showed substantial bright
clipping. It motivates the wider and tissue-only arms; it does not prove that they
improve accuracy. This is not a preregistration made before all cohort inspection.

## Stage 2: individual SlideRefine methods

The CLAHE arm below evaluates **our deterministic tissue/SNR-aware heuristic**.
It requires no enhancement targets or controller training. Retraining MIL on its
outputs does not evaluate learned IA-CLAHE or our optional `TileController`.
See [paper supervision and implementation provenance](../PAPER_RELATIONSHIP.md).
A learned-enhancer study would need a separate protocol with explicit same-modality
reference targets and controller-training roles; no established AF target standard
is supplied. Synthetic reconstruction of original AF appearance alone would not
establish improved diagnostic features. See [custom training](../TRAINING.md).

Freeze one normalization using ONLY inner training/validation patients before
adding an enhancement. Include its paired normalization-only control in every
fold. Evaluate methods separately before combinations; compare against the original
baseline as well as the selected-normalization control.

- Tissue/SNR-aware CLAHE: start with the complete `af_conservative.json` configuration.
  First vary analysis-cell size 64/128/256 pixels, then strength 0/0.225/0.45,
  clip_max 2/3/4, and max_gain 1.5/2. Keep the other fields explicit. Zero strength
  is the identity control. Do not run an uncontrolled Cartesian search; fix the
  ordering and candidate budget in a separate method-stage lock before labels are
  evaluated. Record active-cell fraction and actual pixel changes: a heavily
  gated identity output has not exercised enhancement. Log tissue/SNR/directional
  and sensor gates separately; directional coherence is not an artifact diagnosis.
- HiFiEM: only the pinned `contrast_af` stage. A ratio=0 arm still includes the
  global CDF transform and is NOT an identity control. Include normalization-only
  and global-only controls. Lock local ratio/window and global CDF parameters
  separately, with sufficient pixel halos. Existing example parameters have no
  AF efficacy validation; do not call the stage denoising or full restoration.
- Simple Tone Curves: defer efficacy evaluation until an explicit deployment-
  available target policy is frozen. Fit any learned/reference target using inner
  training patients only, then reuse the curve. No held-out H&E or preferred target
  image is available to inference by assumption. Do not substitute gamma curves
  and label them the paper's optimizer.

These later method grids are proposed design axes, not an executed or tuned final
protocol. Version the method-stage lock before extraction/training. The learned
CLAHE controller, multiscale redistribution, visual-prior HE, native readers and
resumable scheduling are outside this state-store milestone.

## Matched extraction and training

Use the clean 21-patient, 157-slide AF release and its outer-patient exclusions.
Retain identical coordinates, tissue selection, DINO weights/revision, x1/x4 fields
of view, resize, precision, patch order and valid instances. Use the same original
PSA architecture, locked training settings, 15 epochs, and five seeds
20261001..20261005. Retrain each head from scratch on that arm's training features;
do not train on P99.5 features and evaluate on z-score features as the main test.
One normalization comparison with nine arms is 9*21*5=945 outer fits, before any
inner selection fits. Reuse a baseline only after full source/recipe identity
verification. The previous x1-only held-out cache is insufficient for x1+x4 training.
Feature caching may reuse identical source coordinates and preprocessing identities;
fold-dependent tissue masks/statistics cannot be shared across incompatible folds.

For parameter selection, implement nested patient splits. The outer test patient
must be excluded from mask fitting, MIL fitting, epoch selection, normalization
selection, enhancement selection and calibration. When a learned mask is used,
inner validation patients must also be excluded from mask training/validation used
to construct inner-fit inputs. Existing outer-only mask models do not establish
that stronger inner independence. If properly nested preprocessing is unavailable,
report prespecified fixed-arm comparisons without selecting a winner and calling
it an unbiased tuned estimate. Store explicit fit-role ledgers and hashes.

## Endpoints and decision rules

Primary endpoint: difference in outer-held-out slide accuracy at threshold 0.5,
using the mean of the five seed probabilities per slide. Secondary endpoints:
balanced accuracy, malignant sensitivity, specificity, AUROC, AP, log loss and
Brier score. Save every slide probability and paired correction/regression case.
Report patient-macro accuracy and site strata descriptively. Do not optimize the
threshold on outer predictions; any calibration/threshold choice belongs inside
training folds. Site stratification alone is not a site-held-out/OOD evaluation.

Use paired patient-cluster bootstrap intervals (10,000 resamples, fixed seed
20260914), keeping all slides per patient together and comparing the same resamples.
Seeds measure training variability, not five independent cohorts. State undefined
metrics in one-class bootstrap samples and their effective counts. Report all arm
comparisons with simultaneous bootstrap intervals for the family of baseline
contrasts, in addition to unadjusted descriptive intervals. Do not use patch-level
significance or an ordinary independent-slide test.

A highest observed accuracy does not establish superiority. Require a positive
multiplicity-adjusted lower confidence bound for the primary accuracy difference
to claim improvement within this exploratory cohort. Report sensitivity harms even
when total accuracy rises. No clinical noninferiority margin is assumed. With only
21 patients and nine current baseline errors, one extra correct slide is 0.64
percentage points and should not be called a robust gain without uncertainty and
new-patient confirmation. If intervals are wide, conclude inconclusive; do not
force a parameter winner.

## Reproducibility and outputs

Freeze a machine-readable stage lock with ordered candidates, source digests,
folds, seeds, endpoints, selection policy, extraction/training code and stopping
rules before each stage. Save source snapshots, per-slide normalization state,
features, heads, fit roles, predictions, runtime/memory and explicit completion
inventories. Keep private specimen manifests/images/results outside this public
repository. Publish only synthetic tests and non-identifying methodology unless
separately authorized. No AF classification result is produced by this document.
