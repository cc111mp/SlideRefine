"""Small reproducible CLI. Planned methods never silently fall back to CLAHE."""
import argparse
import json
from pathlib import Path
from tsclahe import Config
from .registry import available_methods
from .demo import make_demo
from .wsi.tiles import TileManifestSource
from .wsi.executor import run_manifest

def main(argv=None):
    parser = argparse.ArgumentParser(prog="sliderefine")
    sub = parser.add_subparsers(dest="command",required=True)
    sub.add_parser("methods",help="List actual method implementation status")
    inspect = sub.add_parser("inspect",help="Validate a tile manifest's metadata and geometry")
    inspect.add_argument("manifest",type=Path)
    demo = sub.add_parser("demo",help="Create synthetic source tiles plus coordinates")
    demo.add_argument("output",type=Path)
    tone = sub.add_parser("fit-tone-curve",help="Fit the discrete Simple Tone Curves QP to an explicit target")
    tone.add_argument("target",type=Path,help='JSON containing {"samples": [...]} on a uniform [0,1] input grid')
    tone.add_argument("output",type=Path)
    tone.add_argument("--interpolation",choices=["pchip","linear"],default="pchip")
    histogram = sub.add_parser("fit-histogram-lut", help="Fit a uint16 slide LUT to an explicit frozen reference")
    histogram.add_argument("manifest", type=Path)
    histogram.add_argument("reference", type=Path, help="Nyul landmarks or CDF target JSON; see docs/HISTOGRAM_NORMALIZATION.md")
    histogram.add_argument("output", type=Path)
    histogram.add_argument("--chunk-shape", nargs=2, type=int, default=[1024, 1024])
    run = sub.add_parser("run",help="Fit one slide state and stream output NPY tiles")
    run.add_argument("manifest",type=Path)
    run.add_argument("output",type=Path)
    run.add_argument("--method",default="tissue_snr_clahe")
    run.add_argument("--config",type=Path,help="CLAHE config only")
    run.add_argument("--method-config",type=Path,help="Explicit HiFiEM contrast_af configuration")
    run.add_argument("--tone-curve",type=Path,help="Previously fitted Simple Tone Curves JSON")
    window = run.add_mutually_exclusive_group(required=True)
    window.add_argument("--limits",nargs=2,type=float,metavar=("LOW","HIGH"))
    window.add_argument("--percentiles",nargs=2,type=float,metavar=("LOW","HIGH"))
    run.add_argument("--chunk-shape",nargs=2,type=int,default=[1024,1024],metavar=("HEIGHT","WIDTH"))
    run.add_argument("--max-state-mib",type=int,default=512)
    args = parser.parse_args(argv)
    try:
        if args.command == "methods":
            result = available_methods()
        elif args.command == "demo":
            result = {"manifest":str(make_demo(args.output))}
        elif args.command == "fit-tone-curve":
            from .methods.simple_tone_curves import fit_simple_tone_curve
            if args.target.stat().st_size > 100_000:
                raise ValueError("Target curve file exceeds the size limit")
            target = json.loads(args.target.read_text(encoding="utf-8"))
            if not isinstance(target,dict) or set(target) != {"samples"}:
                raise ValueError('Target JSON must contain exactly {"samples": [...]}')
            if args.output.exists():
                raise FileExistsError(args.output)
            curve = fit_simple_tone_curve(target["samples"],interpolation=args.interpolation)
            curve.save(args.output)
            result = curve.to_dict()
        elif args.command == "fit-histogram-lut":
            from .normalization import HistogramLUT, fit_uint16_histogram, fit_nyul, fit_cdf
            if args.output.exists():
                raise FileExistsError(args.output)
            if args.reference.stat().st_size > 2_000_000:
                raise ValueError("Histogram reference file exceeds the size limit")
            reference = json.loads(args.reference.read_text(encoding="utf-8"))
            if not isinstance(reference, dict):
                raise ValueError("Reference must be a JSON object")
            method = reference.get("method")
            required = ({"method", "percentiles", "target_landmarks"} if method == "nyul_landmarks_af"
                        else {"method", "target_histogram", "domain"})
            if method not in ("nyul_landmarks_af", "empirical_cdf_af") or set(reference) != required:
                raise ValueError("Unsupported or incomplete histogram reference")
            if method == "empirical_cdf_af" and reference["domain"] != [0, 1]:
                raise ValueError("CDF target bins must span normalized [0,1]")
            source = TileManifestSource(args.manifest)
            if source.manifest["intensity_domain"] == "normalized":
                raise ValueError("Refusing to normalize an already normalized slide")
            counts = fit_uint16_histogram(source, chunk_shape=tuple(args.chunk_shape))
            state = (fit_nyul(counts, reference["target_landmarks"], percentiles=reference["percentiles"])
                     if method == "nyul_landmarks_af" else fit_cdf(counts, reference["target_histogram"]))
            parameters = json.loads(state.parameters_json)
            parameters.update(manifest_sha256=source.digest, intensity_domain=source.manifest["intensity_domain"],
                              fit_scope="valid_tissue", application_scope="all_valid_pixels")
            state = HistogramLUT(state.method, state.lookup, json.dumps(parameters, allow_nan=False))
            state.save(args.output)
            result = state.to_dict()
        elif args.command == "inspect":
            s = TileManifestSource(args.manifest)
            result = {"slide_id":s.manifest["slide_id"],"shape":s.shape,"dtype":str(s.dtype),
                      "tiles":len(s.tiles),"channel_id":s.manifest["channel_id"],"sha256":s.digest,
                      "note":"Geometry verified; pixels are decoded/validated lazily during reads."}
        else:
            cfg = Config.from_json(args.config) if args.config else None
            result = run_manifest(args.manifest,args.output,method=args.method,config=cfg,
                                  limits=args.limits,percentiles=args.percentiles,
                                  chunk_shape=tuple(args.chunk_shape),max_state_bytes=args.max_state_mib*1024**2,
                                  method_config=json.loads(args.method_config.read_text(encoding="utf-8")) if args.method_config else None,
                                  tone_curve=args.tone_curve)
        print(json.dumps(result,indent=2,allow_nan=False))
        return 0
    except (ValueError,OSError,NotImplementedError,MemoryError,KeyError,TypeError,RuntimeError) as exc:
        parser.exit(2,f"sliderefine: {exc}\n")
