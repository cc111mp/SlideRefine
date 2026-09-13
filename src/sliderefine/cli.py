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
    run = sub.add_parser("run",help="Fit one slide state and stream output NPY tiles")
    run.add_argument("manifest",type=Path)
    run.add_argument("output",type=Path)
    run.add_argument("--method",default="tissue_snr_clahe")
    run.add_argument("--config",type=Path)
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
        elif args.command == "inspect":
            s = TileManifestSource(args.manifest)
            result = {"slide_id":s.manifest["slide_id"],"shape":s.shape,"dtype":str(s.dtype),
                      "tiles":len(s.tiles),"channel_id":s.manifest["channel_id"],"sha256":s.digest,
                      "note":"Geometry verified; pixels are decoded/validated lazily during reads."}
        else:
            cfg = Config.from_json(args.config) if args.config else Config()
            result = run_manifest(args.manifest,args.output,method=args.method,config=cfg,
                                  limits=args.limits,percentiles=args.percentiles,
                                  chunk_shape=tuple(args.chunk_shape),max_state_bytes=args.max_state_mib*1024**2)
        print(json.dumps(result,indent=2,allow_nan=False))
        return 0
    except (ValueError,OSError,NotImplementedError,MemoryError,KeyError,TypeError) as exc:
        parser.exit(2,f"sliderefine: {exc}\n")
