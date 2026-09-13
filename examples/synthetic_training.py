"""Generate artificial paired data to check training plumbing, NOT efficacy."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from tsclahe.io import write_image
from tsclahe import Config, TissueSNRCLAHE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", type=Path, help="Same configuration used for training")
    p.add_argument("--train-count", type=int, default=8)
    p.add_argument("--val-count", type=int, default=2)
    args = p.parse_args()
    if args.train_count < 1 or args.val_count < 1:
        raise SystemExit("Counts must be positive")
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("Choose a new or empty output directory")
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = Config.from_json(args.config) if args.config else Config(tile_size=(32,32), bins=64,
        min_tissue_pixels=16, min_noise_coefficients=4)
    (args.out / "generator_config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    def known_controller(stats, config):
        snr, deficit, cap = stats.gates(config)
        return config.clip_min + .75 * (config.clip_max-config.clip_min) * snr * deficit, .75 * cap
    reference_operator = TissueSNRCLAHE(cfg, predictor=known_controller)
    yy, xx = np.mgrid[:128, :128]
    for split, count, offset in (("train", args.train_count, 0), ("val", args.val_count, 1000)):
        with (args.out / f"{split}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["input", "target", "mask", "group_id"])
            for index in range(count):
                rng = np.random.default_rng(index + offset)
                mask = ((xx - 64) / 55) ** 2 + ((yy - 64) / 55) ** 2 < 1
                phase = rng.uniform(0, 2 * np.pi)
                signal = .24 + .065 * np.sin(xx / 5 + phase) * np.cos(yy / 7 + phase)
                raw = np.where(mask, signal, .015) + rng.normal(0, .002, xx.shape)
                raw = np.clip(raw, 0, 1).astype(np.float32)
                # Target is reachable by THIS operator/config: a known constant
                # controller proposal (0.75). This is an optimization fixture,
                # not a microscope reference, independent benchmark or denoiser.
                target = reference_operator(raw, mask=mask, limits=(0,1)).enhanced
                stem = f"{split}_{index:03d}"
                write_image(args.out / f"{stem}_input.tif", raw)
                write_image(args.out / f"{stem}_target.tif", target)
                write_image(args.out / f"{stem}_mask.png", mask.astype(np.float32), dtype="uint8")
                writer.writerow([f"{stem}_input.tif", f"{stem}_target.tif", f"{stem}_mask.png", f"synthetic_{offset + index}"])


if __name__ == "__main__":
    main()
