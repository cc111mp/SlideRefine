"""Memory-mapped example; swap readers/writer for an actual WSI backend."""
import argparse
import json
from pathlib import Path
import numpy as np
from tsclahe import Config, Normalization
from tsclahe.streaming import fit_streaming, apply_streaming


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image", type=Path, help="2-D NPY image")
    p.add_argument("mask", type=Path, help="2-D binary NPY mask, globally registered")
    p.add_argument("output", type=Path, help="New float32 NPY output")
    p.add_argument("--limits", type=float, nargs=2, required=True)
    p.add_argument("--config", type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite the output")
    raw = np.load(args.image, mmap_mode="r", allow_pickle=False)
    mask = np.load(args.mask, mmap_mode="r", allow_pickle=False)
    if raw.ndim != 2 or raw.shape != mask.shape:
        raise SystemExit("Image and mask must be aligned 2-D arrays")
    read_image = lambda y0, y1, x0, x1: raw[y0:y1, x0:x1]
    read_mask = lambda y0, y1, x0, x1: mask[y0:y1, x0:x1]
    norm = Normalization(*args.limits, source="fixed")
    cfg = Config.from_json(args.config) if args.config else Config()
    fitted = fit_streaming(raw.shape, read_image, read_mask, norm, cfg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(args.output, mode="w+", dtype=np.float32, shape=raw.shape)
    for y0, x0, block in apply_streaming(fitted, read_image, read_mask, norm):
        out[y0:y0 + block.shape[0], x0:x0 + block.shape[1]] = block
    out.flush()
    fitted.save(args.output.with_suffix(".transform.npz"))
    args.output.with_suffix(".normalization.json").write_text(
        json.dumps(norm.to_dict(), indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
