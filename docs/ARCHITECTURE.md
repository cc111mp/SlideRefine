# Architecture

## Layers
- `src/tsclahe`: unchanged numerical backend v0.2.0, including its training CLI.
- `src/sliderefine/contracts.py`: global region coordinates and source protocol.
- `src/sliderefine/wsi/tiles.py`: selected-channel tile-manifest source, validity/mask handling, spatial index, bounded cache.
- `src/sliderefine/wsi/reducers.py`: pooled exact uint8/uint16 histogram/window estimation.
- `src/sliderefine/wsi/executor.py`: serial fit-then-render reference runner, state budget, and incremental output manifests.
- `src/sliderefine/registry.py`: truthful method metadata and unavailable-method errors.
- `src/sliderefine/methods`: method-specific integration boundaries; four paper methods are placeholders explicitly marked planned.
- `docs/review`: source identity and earlier recorded reviews.

## Execution
Registered tiles -> one virtual slide -> one declared global normalization -> method-specific slide state -> output regions + needed neighboring context -> tiled baseline/enhanced outputs.

Pixels, analysis cells, and worker chunks are different abstractions. A worker chunk may span many storage tiles and analysis cells. The same slide coordinate must use the same transform regardless of the output chunk containing it.

The current CLAHE fit uses the original in-memory global grid. Large-state disk-backed fitting is deliberately not disguised as solved. The conservative state budget rejects excessive allocations before fitting. Native Zarr/TIFF region sources and disk-backed state should be added behind these interfaces, not by changing the CLAHE math.

## Reuse boundaries
Use existing NumPy/SciPy/tifffile/Pillow infrastructure now. Evaluate Dask overlap, OME-Zarr storage, and native TIFF/Zarr region access as later adapters with a tested compatible version set. Do not import all optional libraries from the registry. No automatic downloads or unpinned copies of research repositories.

Every paper reference and AF adaptation must have separate provenance, equations, and tests. Do not unify differing histogram-redistribution rules merely to reduce code duplication. The four planned papers are alternative methods, not a mandatory sequential stack.
