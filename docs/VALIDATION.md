# Historical backend validation — tsclahe v0.2.0

This is the imported backend's historical review, not the current SlideRefine test report.
For the workspace's newly executed results, read [BOOTSTRAP_VALIDATION.md](BOOTSTRAP_VALIDATION.md).
The full issue dispositions are in [review/REVIEW.md](review/REVIEW.md).

## Historical checks

| Check | Recorded result |
|---|---|
| Original v0.1 source | 61 tests passed |
| Reviewed v0.2 source | 110 tests passed, two expected warnings |
| Separately installed v0.2 wheel | 110 tests passed, two expected warnings |
| Statement coverage | 91.31% |
| Additional random-histogram checks | 224 cases, 32–4096 bins |
| Synthetic paired training | Three epochs followed by checkpoint reload/inference |

The two warnings are from deliberately unbound research checkpoints in preserved tests.
All historical execution was Linux CPU with synthetic images, not real microscopy efficacy.
Versions and runtime metadata are in [environment.json](validation/environment.json).

## Included evidence

The [source log](validation/pytest.txt), [wheel log](validation/wheel_tests.txt), and
[original baseline log](validation/original_archive_tests.txt) record those historical tests.
The full coverage JSON is omitted from this workspace; regenerate coverage with pytest-cov.
The raw [histogram stress data](validation/histogram_stress.json) and
[before](validation/before.json)/[after](validation/after.json) cases support the recorded review.

Synthetic training [history](validation/synthetic_history.json),
[stdout](validation/training_stdout.txt), [configuration](validation/synthetic_config.json), and
[validation sample results](validation/synthetic_validation_latest.json) are retained.
Epoch-3 group-balanced tissue MAE was 0.01255872 versus normalized-input MAE 0.02407188;
these only test optimization on reachable artificial targets and are not AF benchmarks.
No smoke checkpoint is shipped as a pretrained microscopy model.

Historical wheel [build](validation/wheel_build.txt), [installation](validation/wheel_install.txt),
and [import path](validation/wheel_import.txt) records are retained. Other duplicate or generated
historical artifacts remain in the original v0.2 source archive rather than this bootstrap.

## Scope

No real AF, clinical, CUDA/MPS, GB10/ARM64, giant-slide throughput, native scanner-pyramid,
or downstream-task validation is established by these historical checks. Minimum dependency
versions and other platforms were not tested in that release. Current remote CI results,
when available, must be checked on the corresponding SlideRefine commit.
