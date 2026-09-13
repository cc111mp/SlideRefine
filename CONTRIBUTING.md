# Contributing

Keep numerical changes independently testable. Run the full test suite before
submitting a change and update docs/METHOD.md when equations or conventions change.
Distinguish new functionality, synthetic sanity checks and real-data validation.

Report input shape/dtype, pixel size, configuration, normalized intensity range,
mask source, NumPy/SciPy/PyTorch versions and a non-sensitive reproducer for bugs.
Avoid uploading patient or company images to a public issue.

A test-passing algorithm change can still alter clinical/biological interpretation.
Do not label a change as safer, more accurate or SOTA without appropriate evidence.
