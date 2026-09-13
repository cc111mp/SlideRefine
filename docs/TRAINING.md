# Optional controller training — v0.2

Start with the training-free baseline. A learned clip controller does not by itself fix missing
scanner metadata, weak tissue masking, artifact ambiguity, or inconsistent reference targets.

## Input and target contract

Provide group-disjoint CSV files with columns:

```csv
input,target,mask,group_id
images/slide001_roi01.tif,targets/slide001_roi01.tif,masks/slide001_roi01.tif,patient001
```

Paths are relative to the CSV. Every input/target/mask must be an aligned 2-D plane. Masks are
explicit binary arrays; Boolean formats and 0/1, 0/255 and 0/65535 are supported. Target must
be the same modality; AF/H&E or adjacent-section pairs are not pixel-aligned AF tone references.

Input calibration must already be consistent upstream (or absent consistently). Inputs are
normalized through exactly the public `prepare` implementation. Supply fixed `--limits LOW HIGH`,
or use exact per-image tissue percentiles. Export manageable ROIs; no random crop loader is hidden
in the trainer. Use ROI size and tile-grid context comparable to deployment.

Two target encodings are explicit:

- `--target-space unit` (default): float [0,1] or unsigned integer whose full dtype range encodes
  the prepared tone reference. A uint16 AF raw-count image is NOT automatically a unit target.
- `--target-space raw`: apply the SAME explicit `--limits` to the raw-unit target. Without fixed
  bounds this mode fails rather than separately normalizing source and target.

References need not lie in the operator's achievable family. Geometry changes, denoising,
saturation inversion and stronger-than-permitted amplification are not recoverable by this
restricted tone map. This remains a limitation on real data; training loss alone cannot diagnose it.

With `sensor_max`, inputs must still be in actual detector units so the threshold is meaningful.
For externally corrected data that has lost raw saturation state, omit `sensor_max` and maintain
raw-saturation QC separately; do not apply a detector threshold to corrected intensity values.

## Splits and provenance

The loader checks nonempty data, duplicate input paths within each manifest, and intersecting
group IDs or resolved input/target paths across train/validation. It does not establish biological
independence of differently named files or discover duplicate tissue. Choose patient/slide/session
groups according to the dependence in your data. Keep a third test set outside tuning.

`run.json` records file digests, manifest digests, groups, versions and preprocessing semantics.
It is local; do not publish sensitive group labels or filenames. No data are uploaded.

## Training commands

```bash
python -m pip install -e ".[learn,test]"
tsclahe-train --train data/train.csv --val data/val.csv --config configs/af_conservative.json --epochs 30 --output-dir runs/af
```

For raw-unit paired references:

```bash
tsclahe-train --train data/train.csv --val data/val.csv --target-space raw --limits 100 24000 --config configs/af_conservative.json --output-dir runs/af_raw
```

The numeric bounds above are placeholders for an experimentally established input policy.
One image is processed per step. There is no distributed trainer, minibatch loader, silent
resizing, optimizer-state resume, or pretrained model.

## Objective and diagnostics

```text
loss = tissue_L1(output, target)
     + 0.2 * tissue_neighbor_gradient_L1(output, target)
     + 0.001 * TV(normalized_clip_control, normalized_strength)
```

Only adjacent tissue pixels enter gradient comparisons. Control TV is normalized by the
configured control ranges, not a hard-coded clip scale. Coefficients are engineering defaults.

Every epoch records active/total tiles, skipped all-inactive samples, nonzero-gradient steps,
gradient magnitude, input-baseline error, and output error. All-inactive epochs and completely
zero-gradient epochs stop instead of silently writing a new checkpoint. Skipped images do not
optimize control-TV alone. Finite training and validation metrics are required.

Validation is first averaged within `group_id`, then equally across groups; `best.pt` minimizes
that group-balanced tissue MAE. Image-level means and per-image latest metrics are also saved.
`beats_identity` compares to the normalized input. A checkpoint that does not beat the baseline
emits a warning and must not be promoted as useful. Improvements on artificial data do not imply
segmentation, virtual staining or OOD improvement.

Artifacts are `config.json`, `run.json`, `history.json`, `validation_latest.json`, `best.pt`, `last.pt`.
Checkpoints contain model state, not optimizer/RNG continuation state.

## Same preprocessing at inference

```bash
tsclahe sample.tif --mask mask.tif --checkpoint runs/af/best.pt --output-dir results/learned
```

Fixed training limits are automatically reused. A conflicting fixed range or replacing a saved
percentile policy with fixed limits fails. A supplied config must match exactly. Explicit mask
or `--all-tissue` is required; a saved explicit mask policy cannot silently become automatic.
New dark/flat arguments are rejected for the paired trainer's precorrected-input policy.

These checks also run in `TissueSNRCLAHE` Python calls. Streaming fitting checks the saved
normalization policy. Low-level `fit_normalized` deliberately accepts already-prepared values;
its caller remains responsible for matching normalization/masks/calibration and provenance.

`save_checkpoint` can save a manually constructed research model without a preprocessing policy;
using that model in the public pipeline warns. Do not mistake an unbound research checkpoint for
a deployable training artifact. The loader uses restricted state-dict loading, finite-state checks,
and semantic versions; load only trusted artifacts.

## Synthetic optimization fixture

```bash
python examples/synthetic_training.py --out synthetic_pairs --config configs/training_demo.json
tsclahe-train --train synthetic_pairs/train.csv --val synthetic_pairs/val.csv --config synthetic_pairs/generator_config.json --limits 0 1 --epochs 3 --threads 1 --output-dir runs/smoke
```

Targets are produced by the same operator with known constant control proposals of 0.75,
including the same gates and gain cap. They are therefore attainable within this operator's
family. Initial sigmoid proposals are 0.5. This checks learning, saving, loading and applying a
checkpoint without demanding denoising or an impossible inverse gain.

This is intentionally a software fixture, not an independent benchmark, a reproduction of the
paper's data augmentation, or a simulation validated against a microscope. For real reference
pairs, an oracle/constrained-reference analysis remains future evaluation work.
