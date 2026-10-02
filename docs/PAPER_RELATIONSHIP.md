# Relationship to the original IA-CLAHE paper

## Verified source

Otsuka, R.; Shoji, Y.; Ogino, Y.; Toizumi, T.; Ito, A. **IA-CLAHE: Image-Adaptive Clip Limit
Estimation for CLAHE**. CVPR Workshops / NTIRE 2026; arXiv:2604.16010.

Primary sources checked on 2026-09-13:
- https://arxiv.org/abs/2604.16010
- https://arxiv.org/html/2604.16010v1 (Sections 3–5.1)
- https://www.nec.com/en/global/rd/publications/2026.html

The paper predicts tile-wise clip limits from a resized luminance image. Its small CNN yields
a local map and a global positive scale. It trains through differentiable CLAHE using image-wise
L1 reference reconstruction, without searched clip-limit labels. Image features, pretrained
initialization, and the paper's training data/augmentation differ from this repository.

## Training supervision in the paper

The paper trains with reference images, without pre-searched clip-limit labels.
Section 5.1 uses MSEC (derived from MIT–Adobe FiveK), taking expert photographer C's
retouched images as clean references. Histogram compression and intensity shift
create degraded inputs. The CNN predicts tile-wise clip limits; differentiable
CLAHE produces an output that is compared with the clean reference using image-wise
L1 reconstruction loss (Sections 4–4.1).

```text
Clean reference -> synthetic histogram degradation -> CNN -> CLAHE -> output
       |                                                              |
       +----------------------- L1 comparison ------------------------+
```

Thus, no ground-truth clip limits does not mean no reference-image supervision.
Zero-shot downstream evaluation means using the trained enhancer on other tasks
without task-specific training; it does not mean that the enhancer was never trained.
These photographic targets are not AF diagnostic ground truth.
Source: [IA-CLAHE v1, Sections 4–5.1](https://arxiv.org/html/2604.16010v1#S4).

## What is shared, changed, and unproven

| Component | Relationship |
|---|---|
| Predict controls, then apply a classical operator | Shared high-level idea |
| Different clip limits across tiles | Shared high-level idea |
| End-to-end gradients into the controller | Shared idea; independently implemented/tested operator |
| Controller input/architecture | Replaced by full histogram/statistics tile-grid CNN |
| Foreground-only histogram and exact foreground bypass | Independently specified microscopy extension |
| Independent blend control and residual/gain bounds | Independently specified restrictions |
| Structural-to-noise and directional-uncertainty gates | Independent engineering hypotheses, not validated biology |
| Final rejected-pixel envelope | Independent v0.2 repair |
| Capped water filling | Independent operator, not a faithful reproduction |
| Paired same-modality trainer | Independent training recipe and preprocessing contract |
| Training supervision | Paper uses photographic reconstruction pairs; our optional trainer requires explicit aligned same-modality targets supplied by the caller |
| Zero-shot microscopy improvement or scanner invariance | Not established |

The delivered controller is neither the paper's image-CNN nor the 2,434-parameter MLP mentioned
in an earlier source description. It is explicitly documented in METHOD.md and the code.
No results or safety benefits are inherited by citation. No claim about present availability
of the authors' official code is made here.

## Which implementation is being evaluated

- **Default `tissue_snr_clahe`:** deterministic heuristic controls, with no enhancer
  training or reference targets required. Adaptive here means image-dependent rules.
- **Optional `TileController`:** our histogram/statistics-grid CNN, trained through
  our bounded operator. Its architecture and checkpoint format differ from the
  paper; author weights are not directly interchangeable. See [TRAINING.md](TRAINING.md).
- **Downstream MIL training:** training a classifier on enhanced-image features
  does not train the enhancer. A heuristic-enhancement classification experiment
  does not evaluate learned IA-CLAHE or our optional learned controller.

The repository has no established AF reference-target standard. Reconstructing an
original AF image from synthetic brightness/contrast perturbations would teach
recovery of that appearance, including its existing defects; it would not establish
an ideal AF appearance or improved diagnostic features. Such a target-generation
policy would be a separate experiment, not the current synthetic software fixture.

## Appropriate project description

> An independent microscopy-oriented contrast-enhancement prototype inspired by IA-CLAHE's
> adaptive parameter-learning framework. It adds tissue-aware histograms, reliability-related
> diagnostics, bounded corrections, and explicit rejection. Its implementation and training
> differ from the paper, and microscopy efficacy remains to be evaluated.

A future experiment should maintain a separate paper-aligned baseline and add each independent
change in an ablation. Changing the controller, operator, data and safeguards together does not
identify which change improves a downstream task. A faithful paper baseline is not bundled.
