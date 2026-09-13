from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from .config import Config
from .core import TissueSNRCLAHE
from .io import read_image, read_mask, write_result, file_provenance


def parser():
    p = argparse.ArgumentParser(description="Tissue/SNR-aware CLAHE: independent microscopy enhancement")
    p.add_argument("input", type=Path, help="2-D TIFF/PNG/NPY, not a full SVS slide")
    p.add_argument("--output-dir", required=True, type=Path, help="New or empty output directory")
    masks = p.add_mutually_exclusive_group()
    masks.add_argument("--mask", type=Path, help="Binary tissue mask")
    masks.add_argument("--all-tissue", action="store_true", help="Input is a tissue-only crop")
    p.add_argument("--config", type=Path, help="JSON configuration; checkpoint config must match")
    p.add_argument("--limits", type=float, nargs=2, metavar=("LOW", "HIGH"), help="Fixed slide/reference scaling limits")
    p.add_argument("--dark", type=Path, help="Dark reference in raw detector units")
    p.add_argument("--flat", type=Path, help="Bright flat reference in raw detector units")
    p.add_argument("--series", type=int, help="Explicit TIFF series index, zero-based")
    p.add_argument("--channel", type=int)
    p.add_argument("--channel-axis", type=int)
    p.add_argument("--replicated-rgb", action="store_true", help="Accept only EXACTLY identical HxWx3 channels")
    p.add_argument("--output-dtype", choices=["float32", "uint16", "uint8"], default="float32")
    p.add_argument("--max-pixels", type=int, default=64_000_000, help="Maximum total loaded scalar samples")
    p.add_argument("--checkpoint", type=Path, help="Optional validated learned controller; no weights bundled")
    p.add_argument("--device", default="cpu", help="Device for optional learned controller")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        predictor = None
        if args.checkpoint:
            from .torch_backend import load_predictor
            predictor = load_predictor(args.checkpoint, device=args.device)
        cfg = Config.from_json(args.config) if args.config else (predictor.config if predictor else Config())
        if predictor and cfg != predictor.config:
            raise ValueError("Configuration differs from checkpoint")
        limits = args.limits
        if predictor is not None and limits is None:
            limits = predictor.metadata.get("preprocessing", {}).get("limits")
        image = read_image(args.input, channel=args.channel, channel_axis=args.channel_axis,
                           replicated_rgb=args.replicated_rgb, max_pixels=args.max_pixels, series_index=args.series)
        # References must already be exported to the same single channel/plane.
        read = lambda path: read_image(path, max_pixels=args.max_pixels) if path else None
        result = TissueSNRCLAHE(cfg, predictor=predictor)(
            image, mask=read_mask(args.mask, max_pixels=args.max_pixels) if args.mask else None, mask_mode="all" if args.all_tissue else "auto",
            limits=limits, dark=read(args.dark), flat=read(args.flat))
        result.provenance.update({
            "files": {name: file_provenance(path) for name, path in
                      (("input", args.input), ("mask", args.mask), ("dark", args.dark),
                       ("flat", args.flat), ("checkpoint", args.checkpoint)) if path is not None},
            "selection": {"series": args.series, "channel": args.channel,
                          "channel_axis": args.channel_axis, "replicated_rgb": args.replicated_rgb}})
        write_result(args.output_dir, result, output_dtype=args.output_dtype)
        print(json.dumps(result.summary(), indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, ImportError, RuntimeError) as e:
        print(f"tsclahe: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
