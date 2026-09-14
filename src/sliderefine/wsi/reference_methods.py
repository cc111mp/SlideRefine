"""Streaming adapters for verified reference-method subsets.

The fitted curve or fixed HiFiEM contrast configuration is immutable across
chunks. No whole-slide image array, local percentile refit, or periodic padding
at execution boundaries. This does NOT implement full HiFiEM or native WSI I/O.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import minimum_filter
from tsclahe.preprocess import Normalization
from .. import __version__
from ..contracts import Region, regions
from .tiles import TileManifestSource, SCHEMA
from .reducers import fit_uint_window


@dataclass(frozen=True)
class PreparedMethod:
    method: str
    operator: object
    halo: int

    def to_dict(self):
        from ..methods.hifiem.contrast import UPSTREAM_COMMIT
        if self.method == "simple_tone_curves":
            return {"method":self.method,"curve":self.operator.to_dict()}
        return {"method":self.method,"variant":"contrast_af", "config":self.operator.to_dict(),
                "upstream_commit":UPSTREAM_COMMIT,"semantics":"hifiem-contrast-af/v1"}


def prepare_method(method, *, method_config=None, tone_curve=None):
    """Validate scope before source reads or output creation; no hidden defaults."""
    if method == "simple_tone_curves":
        from ..methods.simple_tone_curves import SimpleToneCurve
        if method_config is not None or tone_curve is None:
            raise ValueError("Simple Tone Curves requires tone_curve and no method_config")
        curve = tone_curve if isinstance(tone_curve,SimpleToneCurve) else SimpleToneCurve.load(tone_curve)
        return PreparedMethod(method,curve,0)
    if method == "hifiem":
        from ..methods.hifiem import HiFiEMContrastConfig
        if tone_curve is not None:
            raise ValueError("tone_curve only applies to Simple Tone Curves")
        cfg = HiFiEMContrastConfig.from_dict(method_config)
        return PreparedMethod(method,cfg,cfg.halo)
    raise ValueError(f"Unsupported reference-method adapter {method!r}")


def render_region(source, normalization, prepared: PreparedMethod, region: Region):
    """Return (baseline, enhanced, owned pixels/masks, missing-context bypass count).

    This API accepts regions INSIDE the slide. Pixel halo is clipped at genuine
    slide edges, where the operator uses its reference mirror boundary condition.
    Internal missing coverage is not a slide edge and is conservatively bypassed.
    """
    if region.x < 0 or region.y < 0 or region.x+region.width > source.shape[1] or region.y+region.height > source.shape[0]:
        raise ValueError("Output region must be inside the slide")
    part = source.read_region(region)
    baseline = normalization.apply(part.pixels)
    if normalization.degenerate or not part.tissue.any():
        enhanced = baseline.copy()
        bypassed = 0
    elif prepared.method == "simple_tone_curves":
        enhanced = prepared.operator.apply(baseline)
        enhanced[~part.tissue] = baseline[~part.tissue]
        bypassed = 0
    else:
        from ..methods.hifiem import enhance_contrast_af
        h = prepared.halo
        x0, y0 = max(0,region.x-h), max(0,region.y-h)
        x1 = min(source.shape[1],region.x+region.width+h)
        y1 = min(source.shape[0],region.y+region.height+h)
        context = source.read_region(Region(x0,y0,x1-x0,y1-y0))
        normalized = normalization.apply(context.pixels)
        full = enhance_contrast_af(normalized,context.tissue,prepared.operator)
        crop = np.s_[region.y-y0:region.y-y0+region.height,
                     region.x-x0:region.x-x0+region.width]
        # Conservative full-square support contains both separable triangular
        # filter and local min/max supports. A gap is not reflected or invented.
        supported = minimum_filter(context.valid,size=2*h+1,mode="mirror")[crop]
        enhanced = full[crop].copy()
        reject = ~supported | ~part.tissue
        enhanced[reject] = baseline[reject]
        bypassed = int(np.count_nonzero(~supported & part.tissue))
    baseline[~part.valid] = 0
    enhanced[~part.valid] = 0
    return baseline, enhanced, part, bypassed


def run_reference_manifest(manifest, output, *, method, method_config=None, tone_curve=None,
                           limits=None, percentiles=None, chunk_shape=(1024,1024),
                           max_state_bytes=512*1024**2):
    prepared = prepare_method(method,method_config=method_config,tone_curve=tone_curve)
    if (limits is None) == (percentiles is None):
        raise ValueError("Supply exactly one of fixed limits or slide histogram percentiles")
    if type(max_state_bytes) is not int or max_state_bytes < 1:
        raise ValueError("max_state_bytes must be positive")
    source = TileManifestSource(manifest)
    next(regions(source.shape,chunk_shape))
    h = prepared.halo
    max_context = min(source.shape[0],chunk_shape[0]+2*h)*min(source.shape[1],chunk_shape[1]+2*h)
    if chunk_shape[0]*chunk_shape[1] > source.max_region_pixels or max_context > source.max_region_pixels:
        raise MemoryError("Chunk plus required pixel halo exceeds region guard; reduce execution chunk size")
    state = prepared.to_dict()
    state_text = json.dumps(state,sort_keys=True,allow_nan=False)
    if len(state_text.encode("utf-8")) > max_state_bytes:
        raise MemoryError("Serialized fitted state exceeds the supplied state budget")
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError(f"Output must not exist: {out}")
    if limits is not None:
        if len(limits) != 2:
            raise ValueError("limits must contain low and high")
        norm = Normalization(float(limits[0]),float(limits[1]),"fixed")
    else:
        norm = fit_uint_window(source,percentiles,chunk_shape=chunk_shape)
    metadata = dict(status="incomplete",version=__version__,method=method,
                    variant="contrast_af" if method == "hifiem" else "discrete-qp",
                    manifest_sha256=source.digest,normalization=norm.to_dict(),
                    state_sha256=hashlib.sha256(state_text.encode()).hexdigest(),
                    chunk_shape=list(chunk_shape),pixel_halo=h,
                    channel_id=source.manifest["channel_id"],
                    intensity_domain_in=source.manifest["intensity_domain"],
                    warnings=["Research software; no real-AF or gigapixel validation.",
                              "Input files must stay immutable during all passes.",
                              "No calibration or physical saturation assessment is performed.",
                              "The tissue mask is supplied; mapping outside tissue is bypassed."])
    if method == "hifiem":
        metadata["warnings"].extend([
            "Only HiFiEM local/global contrast is integrated, NOT destriping/denoising/background fitting.",
            "AF parameters and missing-context bypass are independent adaptations, not EM defaults."])
    out.mkdir(parents=True)
    report = out/"run.json"
    results = {}
    try:
        report.write_text(json.dumps(metadata,indent=2),encoding="utf-8")
        (out/"fitted_state.json").write_text(state_text,encoding="utf-8")
        for folder in ("baseline","enhanced","masks","valid"):
            (out/folder).mkdir()
        for branch in ("baseline","enhanced"):
            results[branch] = {k:source.manifest[k] for k in ("slide_id","shape","channel_id","axes","level")}
            results[branch].update(schema=SCHEMA,dtype="float32",intensity_domain="normalized",
                                   mask_policy="supplied",pixel_size_um=source.manifest.get("pixel_size_um"),tiles=[])
        count, bypassed = 0, 0
        for region in regions(source.shape,chunk_shape):
            baseline, enhanced, part, n = render_region(source,norm,prepared,region)
            bypassed += n
            name = f"y{region.y:09d}_x{region.x:09d}.npy"
            np.save(out/"masks"/name,part.tissue,allow_pickle=False)
            np.save(out/"valid"/name,part.valid,allow_pickle=False)
            for branch,array in (("baseline",baseline),("enhanced",enhanced)):
                np.save(out/branch/name,array,allow_pickle=False)
                results[branch]["tiles"].append(dict(asdict(region),path=name,
                                                    mask_path="../masks/"+name,valid_path="../valid/"+name))
            count += 1
        for branch,m in results.items():
            (out/branch/"manifest.json").write_text(json.dumps(m,indent=2),encoding="utf-8")
        metadata.update(status="complete",output_chunks=count,missing_context_bypass_pixels=bypassed)
    except Exception as exc:
        metadata.update(status="incomplete",error_type=type(exc).__name__)
        raise
    finally:
        report.write_text(json.dumps(metadata,indent=2,allow_nan=False),encoding="utf-8")
    return metadata
