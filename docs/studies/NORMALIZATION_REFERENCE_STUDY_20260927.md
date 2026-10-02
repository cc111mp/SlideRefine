# AF normalization reference study

Status: references acquired and initial source review completed on 2026-09-27.
This is an Athena workspace study entry, not a new implemented SlideRefine method
or an executed AF comparison. Existing algorithms and the existing study plan
are unchanged.

SlideRefine is the appropriate home for reusable normalization operators,
slide-wide fitted state, chunk-invariance checks and quality measurements.
LumaPath owns matched feature extraction; afmil owns classifier fitting.
Keep experiment-specific target construction and selection policies separate from
the reusable image operator.

The author checkouts live in the central
[reference library](../../../../refs/MIL-References/README.md#af-normalization-references)
as required by Athena's organization rules. This document is the entry from `code/`.
Each checkout is detached and pinned by commit and file hashes. Method adaptations
belong under `experiments/sliderefine/<experiment>/`, never in those checkouts.

## What we acquired

| Reference | Local source | Paper | Role in this study |
|---|---|---|---|
| [Nyúl standardization](../../../../refs/MIL-References/notes/nyul_standardization.md) | Original code not located; later implementation below | DOI/bibliography; PDF unavailable | Sparse percentile landmarks mapped to a training reference |
| [TorchIO](../../../../refs/MIL-References/notes/torchio.md) | Official project cloned | Author manuscript PDF | Numerical implementation reference for Nyúl; not original Nyúl author code |
| [scikit-image](../../../../refs/MIL-References/notes/skimage_histogram.md) | Official project cloned | Software paper PDF | Dense empirical CDF matching comparator |
| [Simple Tone Curves](../../../../refs/MIL-References/notes/simple_tone_curves.md) | No verified author repo; independent solver already in SlideRefine | Publisher PDF | Constrained curve approximation with an explicit target |
| [UniFORM](../../../../refs/MIL-References/notes/uniform.md) | Author project cloned | Published article HTML; PDF unavailable | IF distribution alignment; assumptions need separate AF validation |
| [BaSiC](../../../../refs/MIL-References/notes/basic.md) | Original MATLAB project cloned | Publisher PDF | Within-field background and shading correction |
| [BaSiCPy](../../../../refs/MIL-References/notes/basicpy.md) | Author-linked Python project cloned | 2026 preprint metadata; PDF unavailable | Practical illumination-correction source |
| [EVEN](../../../../refs/MIL-References/notes/even.md) | Official GitLab verified, clone blocked by TLS failure | Publisher PDF | Assessment and optimization of illumination correction |

Exact versions, licenses, source entry points and download gaps are in the linked
reading cards and [source lock](../../../../refs/MIL-References/sources.lock.json).
There are **five new checkouts, five verified PDFs and one full article archived
as HTML**, across eight bibliographic entries. No upstream package was installed.
The [acquisition report](../../../../work/af_normalization_references_20260927/README.md)
records checks and the [CSV inventory](../../../../work/af_normalization_references_20260927/reference_inventory.csv).

## Separate the questions

| Family | What changes | Uses other training slides? | Main AF risk |
|---|---|---|---|
| Own-slide P99.5 | One intensity scale and upper clipping | No | Bright tissue can clip; other distribution differences remain |
| Nyúl-style landmarks | Piecewise-linear intensity mapping | Yes, for the reference landmarks | Tissue composition can be mistaken for acquisition variation |
| Dense histogram matching | Almost the entire intensity distribution | Yes, for the reference CDF | Strong matching may remove useful biological variation |
| Bounded reference tone | A constrained, limited intensity mapping | Yes, for our AF target policy | Benefit depends on the reference and how strongly we move toward it |
| UniFORM | Marker distribution alignment, primarily rigid log shifts | Reference selection must be frozen | AF lacks the same marker-negative population assumption |
| BaSiC/BaSiCPy | Spatial illumination/background field | Calibration or suitable field collections | Tissue structure can be mistaken for illumination |
| EVEN | Scores/selects correction outputs | Its quality model needs validated provenance | An illumination score may not transfer to AF or predict classifier benefit |

Nyúl and dense histogram matching directly address the requested use of several
percentiles to reduce between-slide variation. Illumination correction is a
different upstream question. Better histogram agreement is a process measure,
not evidence that diagnostic information was preserved.

## Initial code findings

1. **TorchIO needs an AF adapter.** The pinned development snapshot is 2.0.0a2.
   Training accepts a mask, while application flattens the whole source tensor.
   Its default quantile list has 13 points; a nearby comment incorrectly says 12.
   It can extrapolate beyond its nominal 0–100 range. Do not call it on each DINO
   patch or assume that training and inference use the same tissue domain.
2. **scikit-image is a numerical reference.** `match_histograms` does not provide
   masked slide-state fitting. The current public module forwards to `_skimage2`.
   A tissue-aware, patient-weighted uint16 LUT implementation would be our own
   adaptation and needs explicit comparison to the pinned source on small arrays.
3. **Simple Tone Curves already has local work.** See the
   [solver/target distinction](../references/simple_tone_curves.md) and
   [method status](../METHOD_STATUS.md). The core solver requires a supplied
   target; it does not decide what a well-normalized AF image should look like.
4. **UniFORM is not ready to drop into this AF pipeline.** Its README still marks
   the pixel-level tutorial/data as being assembled. Inspected source supports
   feature-histogram registration and intensity rescaling. No explicit source
   license was found. Also, its feature inputs are per-marker fluorescence
   intensities, not DINO embedding dimensions.
5. **BaSiCPy needs a geometry and runtime audit.** Its pinned `VERSION` is 1.2.0;
   the checkout predates the 2026 preprint, so paper/release equivalence is not
   established. Dependencies include `scipy<1.13`, `hyperactive<5` and Torch.
   The original MATLAB BaSiC README declares CC BY-NC-ND 4.0; BaSiCPy has MIT.
6. **EVEN is currently a paper reference.** Its official code host fails TLS from
   this machine. No source-level conclusions or execution claims are made for it.

## Evidence we already have

Read the [sealed preprocessing conclusions](../../../../data/runs/athena-study/af_preprocessing_conclusions_20260927_v1/REPORT.md)
before choosing another large grid. Original P99.5 remains the useful control.
Tested percentile changes, tissue scaling, CLAHE and reference tone have not
established a replacement under their matched historical comparisons.
This does not establish that all histogram-standardization methods fail.

The existing
[AF parameter study](AF_PARAMETER_STUDY.md) describes a nine-arm percentile/z-score
plan on the historical 157-slide view. The new
[Round 2 protocol](../../../../experiments/afmil/prostate_round2_20260927/PROTOCOL.md)
describes a corrected 165-slide view and a 16/5 patient rehearsal split, with target
and inner validation still pending. They are different protocols. This reference
study extends the candidate rationale; it does not silently replace either grid,
reuse historical predictions as new results, or authorize a combined parameter sweep.

## Proposed order of work

**1. Establish an exact control and characterize variation.** Preserve the original
whole-slide strided P99.5 -> uint8 recipe, masks, coordinates, encoder revision,
resize and normalization. SlideRefine's pooled tissue histogram is not an exact
reproduction of that baseline. Use a separate tissue-scope control when testing
methods fitted only on tissue. Report covered-pixel, tissue-pixel and actual encoder
crop statistics separately: percentiles, dynamic range, low/high clipping,
flat/invalid values and spatial brightness patterns. Do not infer sensor saturation
from dtype or window clipping.

**2. Implement and verify a small histogram family.** Start with a Nyúl-style
piecewise-linear operator, then dense CDF matching, with the existing bounded
tone adaptation as a comparator. First reproduce the upstream numerical behavior
on synthetic arrays; then name and document each AF deviation. Candidate Nyúl
landmarks are P1/P10/P20/P25/P30/P40/P50/P60/P70/P75/P80/P90/P99, matching the inspected
TorchIO default list. This is a starting design, not an AF optimum. Specify duplicate
landmarks, empty tissue, extrapolation, clipping and quantization before real-data use.

For the AF reference, prevent patients with more slides or more pixels dominating.
Freeze a patient-balanced aggregation policy using development patients only.
Equal-patient aggregation differs from an unweighted upstream image average and
must be labelled as our adaptation. The landmark and dense-CDF comparisons should
share the declared source sampling, reference population and output-range policy;
otherwise changes in sampling or clipping confound the comparison.

**3. Screen outputs before classifier fitting.** Use fixed-scale paired previews,
clipping/gain measurements, preservation of local intensity order, and acquired
state diagnostics. Check weak tissue and bright outliers explicitly. Measure
between-slide histogram distance, but do not select solely for minimum distance:
perfect agreement can remove biologically meaningful differences. Report failure
cases rather than silently dropping slides. Keep any inspection-driven revision
in the protocol history.

**4. Run matched feature/MIL comparisons.** Bind the target, cohort and inner-patient
split first. Fit references, choose parameters/epochs and calibrate only within
development folds. A held-out slide may supply its own source histogram to a frozen
mapping policy; it must not update the cohort target or any shared calibration.
Refit every arm's classifier on that arm's features using matched seeds/settings.
Report paired held-out predictions, sensitivity/specificity, accuracy/BA, AUROC and
calibration with patient-cluster uncertainty. Lock the primary metric and selection
rule before this readout; the new Round 2 protocol has not yet done so.

**5. Study illumination separately if supported by the raw acquisition.** Locate
camera-field geometry, overlaps, dark/flat acquisitions, exposure settings and
scanner/session metadata first. Storage chunks are not camera fields. If those
contracts cannot be established, report BaSiC applicability as unresolved. Validate
any EVEN score on AF artifacts before using it for automatic parameter selection.
Do not combine shading correction and histogram mapping until each has its own
matched normalization-only control.

**6. Freeze for the real new cases.** After internal selection, freeze the entire
recipe and all reference states. Cases awaiting pathology can be inventoried and
checked under a prespecified policy, but cannot contribute to the reference,
tuning or claimed labelled evaluation. Previous preprocessing-training or
checkpoint-selection exposure must remain in each patient's provenance.

## State and result contract

Every fitted normalization state should identify source/mask hashes, sample lattice,
intensity units, fit patients/fold, reference hash, quantiles or CDF, output window,
clipping, quantization, code version and configuration. Fit once per slide/fold;
apply unchanged to all extraction chunks. Reference-dependent features are indexed
by fold/reference identity and cannot be reused across different references.

Required engineering checks include monotonicity, constant/repeated landmarks,
invalid coverage, fit/apply serialization, a small upstream parity fixture, and
identical output under changed chunk sizes/origins/orders. Do not renormalize after
the selected operator or independently inside encoder patches.

New empirical outputs belong in a fresh
`data/runs/sliderefine/<experiment-id>/` directory with a configuration/fit-role
manifest, source/reference hashes, per-slide state, QC table, failure inventory,
runtime/memory and (when evaluated) predictions. Feature releases remain external
and are linked by their immutable identities. Use the workspace run contracts and
seal only completed artifacts. Reference acquisition itself is recorded in `work/`;
it is not an AF experiment result.

No Nyúl/CDF/BaSiC/UniFORM/EVEN adapter or new AF classifier was implemented or run
by this acquisition. The existing SlideRefine test result is recorded in the
[acquisition report](../../../../work/af_normalization_references_20260927/README.md).
