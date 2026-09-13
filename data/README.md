# Data stays outside Git

Use `sliderefine demo demo_input` to create a synthetic tile manifest. Real slide tiles, patient identifiers, calibration acquisitions, trained private weights, and credentials must remain outside this repository. Paths in private manifests can point to workstation/NAS storage. The virtual reader is local-file based; no uploads occur during enhancement.
