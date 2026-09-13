# Executed release validation — v0.2.0

This report records local execution, not real-microscopy efficacy or device validation.
Review date: 2026-09-13. Full source identity and issue dispositions are in
[the review](review/REVIEW.md).

## Environment

Python 3.13.5, Linux x86_64, CPU only.
NumPy 2.3.5; SciPy 1.17.0;
PyTorch 2.10.0+cpu; tifffile 2026.5.15;
Pillow 12.3.0; pytest 9.0.2.
See [environment.json](validation/environment.json) for exact runtime metadata.

Minimum versions in pyproject.toml are not a universal environment lock. Minimum-version
compatibility and other platforms/accelerators have not been executed.

## Tests

```text
Original available source: 61 passed
Reviewed source:           110 passed, 2 expected warnings
Installed wheel:           110 passed, 2 expected warnings
Statement coverage:        91.31%
```

The warnings concern manually created unbound research checkpoints in two existing tests;
they demonstrate the new missing-preprocessing-policy warning. The trained checkpoint path
stores and enforces the policy.

Logs: [source tests + coverage](validation/pytest.txt), [wheel tests](validation/wheel_tests.txt),
[original baseline](validation/original_archive_tests.txt), and [coverage JSON](validation/coverage.json).

Executed source command:

```bash
pytest --cov=tsclahe --cov-report=term-missing --cov-report=json:docs/validation/coverage.json -q
```

Tests cover exact background/rejected-tile bypass, halo/crop consistency, odd edge tiles,
masked histograms, final caps and mass, gain/delta bounds, 8/16-bit handling, high-offset uint32
and float64 normalization, Boolean and palette mask semantics, TIFF series selection, CLI output,
NumPy/PyTorch parity, finite-difference gradients, checkpoint constraints, real optimizer steps,
all-inactive training failure, grouped splits and sensor-saturation streaming consistency.

## Additional numerical stress

224 random histograms across 32,64,128,256,512,1024,4096 bins passed mass/cap/nonnegativity and
fixed-context gain checks. The largest observed effective fitted gain was
2.000000053 with configured max_gain=2
(floating-point tolerance applies). This is not a robustness benchmark for all possible images.

```bash
python examples/stress_histograms.py --output docs/validation/histogram_stress.json
```

[Raw stress data](validation/histogram_stress.json).

## Reproduced review cases

[before.json](validation/before.json) and [after.json](validation/after.json) were produced in
separate interpreters using the actual original/reviewed source paths. They show rejection
leakage fixed, the tested stripe pattern bypassed, high-offset detail preserved, and mask loading
repaired. See the review for exact setup and scope.

```bash
python examples/reproduce_review.py
```

## Three-epoch synthetic training and inference

Executed with four artificial training groups, two artificial validation groups, the stored
training-demo config, fixed limits [0,1], one CPU thread and three epochs. Targets came from a
known controller in this operator family, not from patient/microscope acquisitions.

At epoch 3, group-balanced tissue MAE to that artificial target was
0.01255872; the unchanged input MAE was
0.02407188. Each epoch had four nonzero-gradient optimization
steps. These numbers only exercise optimization; they must not be reported as AF performance.

Best/last checkpoints were saved. The best checkpoint was loaded in a separate CLI invocation
and applied successfully using its saved normalization bounds. No smoke checkpoint is bundled.

[Training stdout](validation/training_stdout.txt), [history](validation/synthetic_history.json),
[configuration](validation/synthetic_config.json), [run provenance](validation/synthetic_run.json),
[validation samples](validation/synthetic_validation_latest.json), and
[checkpoint inference](validation/checkpoint_inference_stdout.txt).

## Packaging

An editable installation and a wheel build were executed. The wheel was installed into a separate
target directory with current runtime dependencies, then imported from that directory outside
the source working directory. All 110 tests passed against it.

[Wheel build](validation/wheel_build.txt), [installation](validation/wheel_install.txt), and
[import-path verification](validation/wheel_import.txt). The source distribution includes its
examples and documents; the wheel contains the importable package and command entry points.

## Not executed

Real AF or patient/clinical validation; GPU/CUDA/MPS; GB10/ARM64; Windows/macOS execution;
minimum-dependency-version testing; giant real-WSI stress; native scanner pyramid I/O; remote
GitHub Actions; independent original-paper reproduction; downstream segmentation, classification,
virtual-staining or OOD efficacy. The CI YAML is supplied configuration, not a remote test result.
