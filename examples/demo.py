"""Synthetic-only diagnostic demo. Contains no real microscopy or patient data."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from tsclahe import Config, TissueSNRCLAHE
from tsclahe.core import clipped_luts, FittedTransform
from tsclahe.io import write_image, write_result


def phantom(size=512, seed=17):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size] / size
    mask = (xx > .06) & (xx < .94) & (yy > .06) & (yy < .94)
    weak = mask & (xx < .5) & (yy < .5)
    noisy = mask & (xx >= .5) & (yy < .5)
    strong = mask & (xx < .5) & (yy >= .5)
    dim = mask & (xx >= .5) & (yy >= .5)
    texture = (np.sin(xx * 143) * np.sin(yy * 119) + .5 * np.cos((xx + yy) * 211)) / 1.5
    clean = np.full((size, size), .012)
    clean[weak] = .25 + .055 * texture[weak]
    clean[noisy] = .20
    clean[strong] = .50 + .40 * texture[strong]
    clean[dim] = .13 + .075 * texture[dim]
    sigma = np.full(clean.shape, .002)
    sigma[noisy] = .035
    sigma[strong] = .006
    image = np.clip(clean + rng.normal(size=clean.shape) * sigma, 0, 1).astype(np.float32)
    return image, mask, {"weak_structured": weak, "noisy_flat": noisy,
                         "high_contrast": strong, "dim_structured": dim}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("demo_output"))
    args = p.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("Choose a new or empty demo output directory")
    args.out.mkdir(parents=True, exist_ok=True)
    image, mask, regions = phantom()
    cfg = Config(tile_size=(64, 64))
    raw = np.rint(image * 65535).astype(np.uint16)
    result = TissueSNRCLAHE(cfg)(raw, mask=mask, limits=(0, 65535))
    write_result(args.out / "adaptive", result)
    write_image(args.out / "input_16bit.tif", image, dtype="uint16")
    write_image(args.out / "mask.png", mask.astype(np.float32), dtype="uint8")
    # Fixed-CLAHE comparison uses the SAME continuous operator, full-image
    # histograms, no tissue/SNR gates, no bounded residual. Not OpenCV CLAHE.
    fixed_cfg = replace(cfg, max_strength=1.0, max_delta=1.0)
    base = TissueSNRCLAHE(fixed_cfg)(image, mask_mode="all", limits=(0, 1))
    clips = np.full(base.transform.grid.grid_shape, 2, dtype=np.float32)
    fixed = FittedTransform(base.transform.grid, fixed_cfg,
                            clipped_luts(base.transform.statistics.histograms, clips), clips,
                            np.ones_like(clips)).apply_region(image, np.ones_like(mask))
    write_image(args.out / "fixed_clahe.tif", fixed)
    width, gap, title_h = 512, 14, 48
    panel = Image.new("RGB", (width * 3 + gap * 4, width + title_h + gap * 2), "white")
    draw = ImageDraw.Draw(panel)
    for i, (name, array) in enumerate([( "Normalized input", result.normalized),
                                       ("Fixed CLAHE, c=2", fixed),
                                       ("Tissue/SNR-aware", result.enhanced)]):
        x0 = gap + i * (width + gap)
        draw.text((x0, 12), name, fill="black")
        panel.paste(Image.fromarray(np.rint(array * 255).astype(np.uint8)).convert("RGB"), (x0, title_h))
    panel.save(args.out / "comparison.png")
    summary = result.summary()
    summary["synthetic_only"] = True
    summary["region_mean_absolute_changes"] = {
        name: float(np.abs(result.enhanced - result.normalized)[m].mean()) for name, m in regions.items()}
    # Rejected tiles now bypass exactly. A mixed tile may still be eligible,
    # so not every pixel in a user-defined noisy REGION is guaranteed unchanged.
    (args.out / "synthetic_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
