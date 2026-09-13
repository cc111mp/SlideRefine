# Configurations

The imported JSON files configure the unchanged tsclahe backend. `tile_size` is the algorithm's analysis scale in pixels, not processing chunk size. It is not a validated physical AF default.

Select the execution chunk shape using `sliderefine run --chunk-shape HEIGHT WIDTH` independently. Global normalization is explicit via `--limits` or pooled integer `--percentiles`. Do not normalize again inside an algorithm adapter.
