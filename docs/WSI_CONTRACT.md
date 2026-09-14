# Whole-slide execution contract

## Coordinates and domains
Manifest schema is `sliderefine.tiles/v1`. Coordinates are integer level-0 XY origins; arrays/shapes/pixel spacing are YX. The bootstrap accepts one explicitly identified 2-D channel. Do not guess RGB, axes, stage registration, magnification, or physical spacing. Lower pyramid levels and affine mosaics need future explicit transforms.

Store raw/calibrated/normalized domain separately from dtype. Bounds refer to supplied input units. The reference runner performs no physical correction. Correct upstream under one calibrated contract, not independently with per-chunk flat-field medians. Configured sensor saturation is measured only on raw counts.

## Geometry, masks, overlap
Source rectangles must be registered and nonoverlapping. Source tiles can omit regions; output has valid=False there. Zeros are fill values, not measurements. A supplied binary tissue mask and validity are intersected. `all_valid` must be chosen explicitly. Do not infer tissue independently per processing tile. Do not count halos more than once during histogram reduction.

`read_region` supports negative/outside-slide halo coordinates, returning invalid fill outside coverage. The requested region itself has a size guard. A source storage tile is decoded as a whole under its own size guard, so a giant native TIFF is not supported by this reader.

## Neighborhood types
1. Pixel halos for finite-neighborhood operators.
2. Adjacent analysis-cell features/LUTs for interpolation or the controller.
3. Global reductions/models that cannot be replaced by a fixed neighborhood.
Derive composed halos and boundary conventions per method. Raw tile coordinates only establish adjacency, not biological correlation. Do not smooth neighboring means automatically.

## Shared state and equivalence
Hold normalization, analysis grid origin/size, masks, model, thresholds, and method state fixed while rendering. Change only execution chunk size/order. Test manageable whole-image references against chunked rendering with shifted origins. For the current NumPy backend, compare float32 outputs at atol=2e-7, rtol=0 unless a more specific test establishes exact equality.

The reader returns validity independently from foreground. The final CLAHE rejection envelope must still apply after neighboring mappings are interpolated. Keep source files immutable for the whole run. The initial runner records a manifest hash but does not compute every source pixel digest; stronger immutable-source identity is a later task.

## Memory and output
Never allocate a full slide pixel array, mask, or reliability map for actual WSI execution. The current grid state is in RAM; a conservative estimate prevents exceeding the chosen budget, but is not a peak-RSS theorem. Implement chunked state rather than silently enlarging analysis cells.

The current writer stores NPY output tiles and branch manifests, not a native WSI pyramid. If a run fails, leave incomplete status and do not claim valid export. Resume is pending. For a canonical future pyramid, enhance the designated full-resolution image then downsample it: nonlinear enhancement and downsampling do not generally commute.

## Acceptance before production use
Native format/channel/metadata roundtrips; actual gigapixel peak RAM and throughput; chunk-size/origin/order parity; rejected regions and noise/artifact fixtures; missing/edge coverage; restart integrity; no duplicated normalization; real AF and downstream slide/patient/session-held-out evaluation.

## Reference-method adapters added in 0.2.0

Simple Tone Curves uses one explicitly fitted curve for the run, with no local fitting
or pixel halo at application time. Its first/last output knots preserve the supplied
target endpoints. HiFiEM requires `variant="contrast_af"`; only local/global contrast
is included. Its fixed, normalized-unit configuration is serialized for the whole run.

For the supported HiFiEM path, halo=max(smoothing-1,minmax_size//2), or zero at ratio=0.
Clip the read rectangle at actual slide edges and apply mirror conditions there. Do not
reflect or treat gaps between source tiles as measured pixels. If the full dependency
window includes missing coverage, preserve the owned pixel's normalized baseline.

Both adapters keep supplied non-tissue pixels unchanged and publish validity masks.
Neither performs dark/flat calibration or physical saturation detection. Fitted state
identity and normalization policy are separate recorded fields. Source tiles must remain
immutable across all passes. These are selected-channel reference runners with incremental
NPY outputs, not native pyramidal formats or a production scheduler.
