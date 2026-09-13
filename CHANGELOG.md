# Changelog

## SlideRefine 0.1.0 bootstrap
- Import the reviewed tsclahe 0.2.0 source, tests, configs, examples, and textual review records without algorithm changes.
- Add the SlideRefine namespace/CLI and explicit implemented/planned method registry.
- Add a spatially indexed tile-manifest reader with bounded cache, missing-data masks, geometry checks, and native dtype validation.
- Add slide-wide exact uint8/uint16 tissue histogram percentiles.
- Add a serial fixed-state reference runner and incremental NPY-tile output, preserving baseline and coverage separately.
- Add WSI contracts, import hashes, synthetic tests, and Codex milestones.
- Native pyramid I/O, disk-backed state, distributed scheduling, resume, four paper implementations, and real gigapixel/AF validation remain pending.

The imported backend retains its own 0.2.0 version and operator/checkpoint semantics. Historical release changes are documented in docs/review/REVIEW.md.
