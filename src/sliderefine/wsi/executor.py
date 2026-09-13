"""Serial reference runner with fixed slide state and incrementally written NPY tiles.

Pixels are streamed. The retained tsclahe state/temporary buffers are in-memory;
an explicit conservative budget estimate prevents unbounded grid allocation.
No native pyramid writer, resume implementation, learned WSI executor, or GPU
scheduler is claimed. Use the unchanged tsclahe APIs for training/plane inference.
"""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys
import numpy as np
from tsclahe import Config
from tsclahe.preprocess import Normalization
from tsclahe.streaming import fit_streaming
from .. import __version__
from ..contracts import Region, regions
from ..registry import require_method
from .tiles import TileManifestSource, SCHEMA
from .reducers import fit_uint_window

class _Readers:
    def __init__(self, source):
        self.source, self._bounds, self._part = source, None, None
    def part(self, bounds):
        if bounds != self._bounds:
            y0,y1,x0,x1 = bounds
            self._part = self.source.read_region(Region(x0,y0,x1-x0,y1-y0))
            self._bounds = bounds
        return self._part
    def image(self,*bounds): return self.part(bounds).pixels
    def mask(self,*bounds): return self.part(bounds).tissue

def estimate_state_bytes(shape, cfg):
    cells = ((shape[0]+cfg.tile_size[0]-1)//cfg.tile_size[0])*((shape[1]+cfg.tile_size[1]-1)//cfg.tile_size[1])
    # Includes generous allowance for float64 sorting/redistribution temporaries;
    # this is a planning bound, not a measured peak-RSS guarantee.
    return cells*(128*cfg.bins+1024)

def run_manifest(manifest, output, *, method="tissue_snr_clahe", config=None,
                 limits=None, percentiles=None, chunk_shape=(1024,1024),
                 max_state_bytes=512*1024**2):
    require_method(method)  # Fail before creating files for planned methods.
    cfg = config or Config()
    if (limits is None) == (percentiles is None):
        raise ValueError("Supply exactly one of fixed limits or slide histogram percentiles")
    if type(max_state_bytes) is not int or max_state_bytes < 1:
        raise ValueError("max_state_bytes must be positive")
    source = TileManifestSource(manifest)
    # Validate even when a previous phase would happen to skip iteration.
    next(regions(source.shape, chunk_shape))
    if max(chunk_shape[0]*chunk_shape[1], cfg.tile_size[0]*cfg.tile_size[1]) > source.max_region_pixels:
        raise ValueError("Processing/analysis chunk exceeds the region-size guard")
    estimate = estimate_state_bytes(source.shape, cfg) if method != "normalization_only" else 0
    if estimate > max_state_bytes:
        raise MemoryError(f"Estimated state/working allocation {estimate} bytes exceeds budget {max_state_bytes}. "
                          "Do not silently change analysis scale; disk-backed state is a future milestone.")
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError(f"Output must not exist: {out}")
    if limits is not None:
        if len(limits) != 2:
            raise ValueError("limits must contain low and high")
        norm = Normalization(float(limits[0]),float(limits[1]),"fixed")
    else:
        norm = fit_uint_window(source,percentiles,chunk_shape=chunk_shape)
    reader = _Readers(source)
    saturation_reader = None
    if cfg.sensor_max is not None:
        if source.manifest["intensity_domain"] != "raw":
            raise ValueError("sensor_max requires raw input; corrected pixels cannot recover saturation")
        saturation_reader = lambda *b: reader.image(*b) >= cfg.sensor_max
    transform = None
    if method == "tissue_snr_clahe":
        transform = fit_streaming(source.shape,reader.image,reader.mask,norm,cfg,
                                  saturation_reader=saturation_reader)
    out.mkdir(parents=True)
    for folder in ("baseline","enhanced","masks","valid"):
        (out/folder).mkdir()
    metadata = {"status":"incomplete","version":__version__,"backend":"tsclahe-0.2.0",
                "method":method,"manifest_sha256":source.digest,"normalization":norm.to_dict(),
                "config":cfg.to_dict(),"chunk_shape":list(chunk_shape),"state_budget_estimate":estimate,
                "channel_id":source.manifest["channel_id"],"intensity_domain_in":source.manifest["intensity_domain"],
                "python":sys.version.split()[0],"platform":platform.platform(),"numpy":np.__version__,
                "warnings":["Research output: no real-AF or gigapixel validation.",
                            "No calibration is performed by this runner; windows use input units.",
                            "Sensor state is unknown unless explicitly configured.",
                            "Input files must remain immutable throughout fitting and rendering."]}
    report = out/"run.json"
    report.write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    if transform is not None:
        transform.save(out/"transform.npz")
    result_manifests = {}
    for branch in ("baseline","enhanced"):
        result_manifests[branch] = {k:source.manifest[k] for k in
                                   ("slide_id","shape","channel_id","axes","level")}
        result_manifests[branch].update(schema=SCHEMA,dtype="float32",intensity_domain="normalized",
                                       mask_policy="supplied",pixel_size_um=source.manifest.get("pixel_size_um"),tiles=[])
    try:
        count = 0
        for region in regions(source.shape,chunk_shape):
            part = source.read_region(region)
            baseline = norm.apply(part.pixels)
            enhanced = (baseline.copy() if transform is None else
                        transform.apply_region(baseline,part.tissue,origin=(region.y,region.x)))
            # Missing coverage is not measured dark tissue; export validity separately.
            baseline[~part.valid] = 0
            enhanced[~part.valid] = 0
            name = f"y{region.y:09d}_x{region.x:09d}.npy"
            np.save(out/"masks"/name,part.tissue,allow_pickle=False)
            np.save(out/"valid"/name,part.valid,allow_pickle=False)
            for branch,array in (("baseline",baseline),("enhanced",enhanced)):
                np.save(out/branch/name,array,allow_pickle=False)
                result_manifests[branch]["tiles"].append(dict(asdict(region),path=name,
                                                            mask_path="../masks/"+name,valid_path="../valid/"+name))
            count += 1
        for branch,m in result_manifests.items():
            (out/branch/"manifest.json").write_text(json.dumps(m,indent=2),encoding="utf-8")
        metadata.update(status="complete",output_chunks=count)
    except Exception as exc:
        metadata.update(status="incomplete",error_type=type(exc).__name__)
        raise
    finally:
        report.write_text(json.dumps(metadata,indent=2,allow_nan=False),encoding="utf-8")
    return metadata
