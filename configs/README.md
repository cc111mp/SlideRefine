# Configurations

The imported JSON files configure the unchanged tsclahe backend. `tile_size` is the algorithm's analysis scale in pixels, not processing chunk size. It is not a validated physical AF default.

Select the execution chunk shape using `sliderefine run --chunk-shape HEIGHT WIDTH` independently. Global normalization is explicit via `--limits` or pooled integer `--percentiles`. Do not normalize again inside an algorithm adapter.

`af_conservative.json` supplies heuristic bounds and gates; it is not a trained
controller or a validated AF target policy. Without a predictor/checkpoint, controls
are deterministic. The optional custom controller requires separately supplied
paired targets and training; see [TRAINING.md](../docs/TRAINING.md) and
[paper provenance](../docs/PAPER_RELATIONSHIP.md). `training_demo.json` supports a
synthetic software fixture, not the paper's training recipe or AF ground truth.

`tone_target.example.json` supplies desired output brightness samples for numerical
curve fitting. It is a synthetic mapping example, not a target photograph or a
validated AF reference. No neural network is trained by `fit-tone-curve`. A cohort
reference policy must be defined outside the core solver; see
[tone targets and AF fitting](../docs/references/simple_tone_curves.md).
