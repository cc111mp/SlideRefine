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
| Zero-shot microscopy improvement or scanner invariance | Not established |

The delivered controller is neither the paper's image-CNN nor the 2,434-parameter MLP mentioned
in an earlier source description. It is explicitly documented in METHOD.md and the code.
No results or safety benefits are inherited by citation. No claim about present availability
of the authors' official code is made here.

## Appropriate project description

> An independent microscopy-oriented contrast-enhancement prototype inspired by IA-CLAHE's
> adaptive parameter-learning framework. It adds tissue-aware histograms, reliability-related
> diagnostics, bounded corrections, and explicit rejection. Its implementation and training
> differ from the paper, and microscopy efficacy remains to be evaluated.

A future experiment should maintain a separate paper-aligned baseline and add each independent
change in an ablation. Changing the controller, operator, data and safeguards together does not
identify which change improves a downstream task. A faithful paper baseline is not bundled.
