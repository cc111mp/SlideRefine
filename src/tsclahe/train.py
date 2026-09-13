"""Paired same-modality training for the experimental controller.

Targets must be aligned, acceptable-contrast images in the SAME modality.
H&E is not an AF intensity target. No entropy-maximization objective is used.
Training and validation group IDs must be disjoint (patient/slide/session as
appropriate). A group split is necessary but is not proof against all leakage.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import warnings
from pathlib import Path
import random
import sys
import numpy as np
from .config import Config
from .core import fit_normalized
from .io import read_image, read_mask, file_provenance
from .preprocess import prepare, validate_mask
from .version import __version__, PREPROCESS_VERSION, OPERATOR_VERSION


@dataclass(frozen=True)
class Record:
    image: Path
    target: Path
    mask: Path
    group: str


def read_manifest(path: Path) -> list[Record]:
    path = Path(path).resolve()
    records = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not {"input", "target", "mask", "group_id"}.issubset(reader.fieldnames or []):
            raise ValueError("Manifest requires input,target,mask,group_id columns")
        for row in reader:
            if any(not row.get(k, "").strip() for k in ("input", "target", "mask", "group_id")):
                raise ValueError("Manifest fields cannot be blank; an explicit mask is required")
            paths = [(path.parent / row[k]).resolve() for k in ("input", "target", "mask")]
            if not all(p.is_file() for p in paths):
                raise ValueError(f"Missing file in manifest row {len(records) + 2}")
            records.append(Record(*paths, row["group_id"].strip()))
    if not records:
        raise ValueError("Manifest is empty")
    if len({r.image for r in records}) != len(records):
        raise ValueError("Duplicate input images in manifest")
    return records


def validate_split(train: list[Record], val: list[Record]):
    if {r.group for r in train} & {r.group for r in val}:
        raise ValueError("Training and validation group_id values overlap")
    train_paths = {p for r in train for p in (r.image, r.target)}
    val_paths = {p for r in val for p in (r.image, r.target)}
    if train_paths & val_paths:
        raise ValueError("Training and validation image/target paths overlap")


def target_unit_range(image):
    if image.dtype.kind == "u":
        return image.astype(np.float32) / np.iinfo(image.dtype).max
    if image.dtype.kind == "f" and np.isfinite(image).all() and image.min() >= 0 and image.max() <= 1:
        return image.astype(np.float32)
    raise ValueError("Targets must be float [0,1] or unsigned integer images encoding [0,1]")


def load_sample(record: Record, config: Config, device="cpu", limits=None, target_space="unit"):
    import torch
    from .torch_backend import prepare_tensors
    image, raw_target = read_image(record.image), read_image(record.target)
    mask = validate_mask(read_mask(record.mask), image.shape)
    if target_space == "unit":
        target = target_unit_range(raw_target)
    elif target_space == "raw":
        if limits is None:
            raise ValueError("Raw targets require explicit shared --limits LOW HIGH")
        from .preprocess import Normalization
        target = Normalization(float(limits[0]), float(limits[1]), "fixed").apply(raw_target)
    else:
        raise ValueError("target_space must be 'unit' or 'raw'")
    if target.shape != image.shape or not mask.any():
        raise ValueError("Target must align with input; mask must contain tissue")
    prepared = prepare(image, config, mask=mask, limits=limits)
    transform = fit_normalized(prepared.image, prepared.mask, config,
                               sensor_saturated=prepared.sensor_saturated)
    tensors = prepare_tensors(prepared.image, prepared.mask, transform.statistics, config, device)
    if prepared.normalization.degenerate:
        tensors["cap"].zero_()
    tensors["target"] = torch.as_tensor(target[None, None], device=device)
    return tensors, transform.grid


def enhancement_loss(output, target, mask, clips, strengths, config):
    import torch
    def masked_mean(x, m):
        return (x * m).sum() / m.sum().clamp_min(1)
    mae = masked_mean((output - target).abs(), mask)
    gradient = output.sum() * 0
    for axis in (-2, -1):
        if output.shape[axis] < 2:
            continue
        left = [slice(None)] * 4
        right = [slice(None)] * 4
        left[axis], right[axis] = slice(1, None), slice(None, -1)
        l, r = tuple(left), tuple(right)
        edge_mask = mask[l] * mask[r]
        error = ((output[l] - output[r]) - (target[l] - target[r])).abs()
        gradient = gradient + masked_mean(error, edge_mask)
    tv = output.sum() * 0
    for control in ((clips - config.clip_min) / max(config.clip_max - config.clip_min, 1e-6), strengths / max(config.max_strength, 1e-6)):
        for axis in (-2, -1):
            if control.shape[axis] > 1:
                tv = tv + torch.diff(control, dim=axis).abs().mean()
    total = mae + 0.2 * gradient + 0.001 * tv
    return total, mae


def train_model(args):
    import torch
    from .torch_backend import TileController, forward_prepared, save_checkpoint
    if args.epochs < 1 or not math.isfinite(args.lr) or args.lr <= 0 or args.threads < 1:
        raise ValueError("epochs, lr and threads must be positive")
    if not 0 <= args.seed < 2**32:
        raise ValueError("seed must be in [0,2**32)")
    if args.target_space == "raw" and args.limits is None:
        raise ValueError("Raw target training requires shared --limits")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    config = Config.from_json(args.config) if args.config else Config()
    training, validation = read_manifest(args.train), read_manifest(args.val)
    validate_split(training, validation)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Training output directory must be new or empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = TileController(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best, history = float("inf"), []
    preprocessing = {"version": PREPROCESS_VERSION,
                     "limits": list(args.limits) if args.limits else None,
                     "input_policy": "fixed" if args.limits else "tissue_percentiles",
                     "mask_policy": "explicit_binary", "calibration_policy": "precorrected_or_none",
                     "target_encoding": args.target_space + "_same_modality"}
    run = {"package_version": __version__, "operator_version": OPERATOR_VERSION,
           "preprocessing": preprocessing, "seed": args.seed,
           "manifests": {"train": file_provenance(args.train), "val": file_provenance(args.val)},
           "files": [{"split": split, "group_id": record.group,
                      "input": file_provenance(record.image), "target": file_provenance(record.target),
                      "mask": file_provenance(record.mask)}
                     for split, rows in (("train", training), ("val", validation)) for record in rows]}
    (args.output_dir / "run.json").write_text(json.dumps(run, indent=2, allow_nan=False), encoding="utf-8")
    (args.output_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        active_tiles = total_tiles = active_steps = nonzero_gradient_steps = skipped_steps = 0
        gradient_l1_total = 0.0
        order = list(training)
        random.shuffle(order)
        for record in order:
            tensors, grid = load_sample(record, config, args.device, args.limits, args.target_space)
            optimizer.zero_grad(set_to_none=True)
            output, clips, strengths = forward_prepared(model, tensors, grid)
            total_tiles += strengths.numel()
            active_tiles += int((strengths > 0).sum())
            if not bool((strengths > 0).any()):
                skipped_steps += 1
                continue  # Do not optimize only control-TV on wholly rejected images.
            active_steps += 1
            loss, _ = enhancement_loss(output, tensors["target"], tensors["mask"], clips, strengths, config)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            gradient_l1 = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
            gradient_l1_total += gradient_l1
            nonzero_gradient_steps += int(gradient_l1 > 0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        if active_steps == 0:
            raise ValueError("All training regions rejected: no active tiles; no checkpoint written for this epoch")
        if nonzero_gradient_steps == 0:
            raise ValueError("No nonzero controller gradients: inspect targets, caps and eligibility")
        model.eval()
        val_maes, baseline_maes = [], []
        group_values = {}
        validation_rows = []
        with torch.inference_mode():
            for record in validation:
                tensors, grid = load_sample(record, config, args.device, args.limits, args.target_space)
                output, clips, strengths = forward_prepared(model, tensors, grid)
                _, mae = enhancement_loss(output, tensors["target"], tensors["mask"], clips, strengths, config)
                baseline = ((tensors["image"] - tensors["target"]).abs() * tensors["mask"]).sum() / tensors["mask"].sum()
                if not torch.isfinite(mae) or not torch.isfinite(baseline):
                    raise ValueError("Non-finite validation metrics")
                val_maes.append(float(mae))
                baseline_maes.append(float(baseline))
                group_values.setdefault(record.group, []).append((float(mae), float(baseline)))
                validation_rows.append({"input": record.image.name, "group_id": record.group,
                    "tissue_mae": float(mae), "input_mae": float(baseline),
                    "active_tiles": int((strengths > 0).sum()), "total_tiles": strengths.numel()})
        group_means = np.asarray([np.mean(v, axis=0) for v in group_values.values()])
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "val_tissue_mae": float(group_means[:, 0].mean()),
               "val_normalized_input_mae": float(group_means[:, 1].mean()),
               "val_image_tissue_mae": float(np.mean(val_maes)),
               "active_training_tiles": active_tiles, "total_training_tiles": total_tiles,
               "active_steps": active_steps, "skipped_steps": skipped_steps,
               "nonzero_gradient_steps": nonzero_gradient_steps,
               "gradient_l1_total": gradient_l1_total,
               "validation_groups": len(group_values),
               "beats_identity": bool(group_means[:, 0].mean() < group_means[:, 1].mean())}
        (args.output_dir / "validation_latest.json").write_text(
            json.dumps(validation_rows, indent=2, allow_nan=False), encoding="utf-8")
        history.append(row)
        metadata = {"epoch": epoch, "metrics": row, "seed": args.seed,
                    "training_images": len(training), "validation_images": len(validation),
                    "preprocessing": preprocessing,
                    "status": "experimental; requires independent held-out microscopy validation"}
        save_checkpoint(args.output_dir / "last.pt", model, metadata=metadata)
        if row["val_tissue_mae"] < best:
            best = row["val_tissue_mae"]
            save_checkpoint(args.output_dir / "best.pt", model, metadata=metadata)
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(row), flush=True)
    if min(row["val_tissue_mae"] for row in history) >= history[-1]["val_normalized_input_mae"]:
        warnings.warn("Best checkpoint did not beat the normalized-input baseline. Do not promote it.", RuntimeWarning)
    return history


def parser():
    p = argparse.ArgumentParser(description="Train the experimental Tissue/SNR-aware CLAHE controller")
    p.add_argument("--train", type=Path, required=True, help="Training CSV")
    p.add_argument("--val", type=Path, required=True, help="Group-disjoint validation CSV")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--config", type=Path)
    p.add_argument("--limits", type=float, nargs=2, help="Fixed INPUT normalization; otherwise tissue percentiles")
    p.add_argument("--target-space", choices=["unit", "raw"], default="unit",
                   help="unit: prepared [0,1] reference; raw: shared input/target --limits")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    return p


def main(argv=None):
    try:
        train_model(parser().parse_args(argv))
        return 0
    except (ValueError, OSError, ImportError, RuntimeError) as e:
        print(f"tsclahe-train: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
