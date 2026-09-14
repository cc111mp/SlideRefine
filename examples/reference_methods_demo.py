"""Run the two reference-method subsets on locally generated synthetic tiles."""
from pathlib import Path
import argparse
from sliderefine.demo import make_demo
from sliderefine.methods.simple_tone_curves import fit_simple_tone_curve
from sliderefine.wsi.executor import run_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, help="Fresh directory; contains synthetic inputs and results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = make_demo(args.output / "input")
    curve = fit_simple_tone_curve([0, 0.1, 0.5, 0.6, 1])
    curve.save(args.output / "curve.json")
    for method, options in (
        ("simple_tone_curves", {"tone_curve": curve}),
        ("hifiem", {"method_config": {"variant": "contrast_af", "smoothing": 31}}),
    ):
        report = run_manifest(manifest, args.output / method, method=method,
                              limits=(0, 16000), chunk_shape=(128, 160), **options)
        print(method, report["status"], report["output_chunks"])
    print("Synthetic execution only; not evidence of microscopy improvement.")


if __name__ == "__main__":
    main()
