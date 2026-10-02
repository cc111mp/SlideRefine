"""Fit once, apply unchanged: uint16 AF landmark and empirical-CDF mappings.

These are independent AF adaptations, not unmodified Nyul/TorchIO pipelines.
Reference construction and patient/fold membership belong to the caller.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

QUANTILES = (1, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 99)
SCHEMA = "sliderefine.histogram-lut/v1"


def _vector(values, name, *, size=None):
    a = np.asarray(values)
    if a.ndim != 1 or a.dtype.kind not in "uif" or not a.size:
        raise ValueError(f"{name} must be a nonempty real vector")
    a = a.astype(np.float64)
    if not np.isfinite(a).all() or (size is not None and a.size != size):
        raise ValueError(f"{name} must be finite with the required length")
    return a


def _mass(values, name, *, size=None):
    a = _vector(values, name, size=size)
    if np.any(a < 0) or not np.any(a > 0):
        raise ValueError(f"{name} must have nonnegative mass and positive total")
    # Bound every accumulation in float64; overflow is an error, not a NaN LUT.
    if float(a.max()) > np.finfo(np.float64).max / a.size:
        raise ValueError(f"{name} mass is too large to accumulate safely")
    return a


def histogram_quantiles(counts, percentiles):
    """NumPy linear sample quantiles without expanding integer pixel counts."""
    counts = _mass(counts, "counts")
    if np.any(counts != np.floor(counts)) or counts.sum() > 2**53 - 1:
        raise ValueError("counts must be integers with an exactly representable total")
    q = _vector(percentiles, "percentiles")
    if np.any((q < 0) | (q > 100)):
        raise ValueError("percentiles must be in [0,100]")
    ranks = q / 100 * (int(counts.sum()) - 1)
    lo, hi = np.floor(ranks), np.ceil(ranks)
    cdf = np.cumsum(counts.astype(np.int64), dtype=np.int64)
    a = np.searchsorted(cdf, lo.astype(np.int64), side="right")
    b = np.searchsorted(cdf, hi.astype(np.int64), side="right")
    return a + (ranks - lo) * (b - a)


def fit_uint16_histogram(source, *, chunk_shape=(1024, 1024)):
    """Pool each valid tissue pixel once on the source's global slide grid.

    Accepts the existing RegionSource/TileManifestSource interface. Storage tiles
    must be registered and nonoverlapping; no image/mask is inferred per chunk.
    """
    from .contracts import regions
    dtype = np.dtype(source.dtype)
    if dtype.kind != "u" or dtype.itemsize != 2:
        raise ValueError("Histogram fitting requires uint16 source pixels")
    if int(source.shape[0]) * int(source.shape[1]) > 2**53 - 1:
        raise ValueError("Slide pixel count exceeds exact histogram quantile support")
    counts = np.zeros(65536, dtype=np.int64)
    for region in regions(source.shape, chunk_shape):
        part = source.read_region(region)
        if (part.pixels.dtype != dtype or part.pixels.shape != region.shape
                or part.valid.dtype != np.bool_ or part.tissue.dtype != np.bool_
                or part.valid.shape != region.shape or part.tissue.shape != region.shape):
            raise ValueError("Source region violates its pixel/mask contract")
        counts += np.bincount(part.pixels[part.valid & part.tissue], minlength=65536)
    if not counts.any():
        raise ValueError("No valid tissue pixels for histogram fitting")
    return counts


def nyul_lut(source, target):
    """Monotone piecewise-linear AF map, with clipped 0/1 target endpoints.

    Repeated source landmarks receive their median target, with endpoints
    fixed at 0 and 1. This declared tie policy differs from TorchIO's epsilon
    denominator; target vectors must already use the normalized [0,1] domain.
    """
    x = _vector(source, "source landmarks")
    y = _vector(target, "target landmarks", size=x.size)
    if x.size < 2 or np.any(np.diff(x) < 0) or np.any(np.diff(y) < 0):
        raise ValueError("Landmarks must be matching nondecreasing vectors of length >= 2")
    if x[0] < 0 or x[-1] > 65535 or x[-1] <= x[0]:
        raise ValueError("Source landmarks require a nondegenerate uint16 range")
    if y[0] != 0 or y[-1] != 1:
        raise ValueError("Target landmarks must span exactly [0,1]")
    unique = np.unique(x)
    mapped = np.array([np.median(y[x == v]) for v in unique])
    mapped[0], mapped[-1] = 0., 1.
    values = np.interp(np.arange(65536, dtype=float), unique, mapped, left=0., right=1.)
    lut = np.clip(values * 255., 0, 255).astype(np.uint8)
    return lut, dict(input_landmarks=x.tolist(), target_landmarks=y.tolist(),
                     merged_input_landmarks=unique.tolist(), merged_targets=mapped.tolist(),
                     repeated_landmark_count=int(x.size - unique.size),
                     lower_endpoint=float(x[0]), upper_endpoint=float(x[-1]))


def cdf_values(source_hist, target_hist):
    """Empirical inverse-CDF interpolation on integer bin coordinates.

    Matches scikit-image's numerical rule at occupied source intensities.
    Empty source bins inherit their cumulative probability. Counts and finite
    probability masses are accepted; source and target supports may differ.
    """
    source = _mass(source_hist, "source histogram")
    target = _mass(target_hist, "target histogram")
    values = np.flatnonzero(target)
    src = np.clip(np.cumsum(source) / source.sum(), 0, 1)
    dst = np.cumsum(target[values]) / target.sum()
    src[np.flatnonzero(source)[-1]:] = 1
    dst[-1] = 1
    return np.interp(src, dst, values)


def cdf_lut(source_hist, target_hist):
    """uint16 source -> uint8 using a target on 65,536 uniform [0,1] bins."""
    source = _mass(source_hist, "source histogram", size=65536)
    target = _mass(target_hist, "target histogram", size=65536)
    return np.clip(cdf_values(source, target) / 65535 * 255, 0, 255).astype(np.uint8)


def _digest(values):
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


@dataclass(frozen=True)
class HistogramLUT:
    """Immutable lookup state; application never refits or normalizes twice."""
    method: str
    lookup: bytes
    parameters_json: str

    def __post_init__(self):
        if self.method not in ("nyul_landmarks_af", "empirical_cdf_af"):
            raise ValueError("Unsupported histogram method")
        if type(self.lookup) is not bytes or len(self.lookup) != 65536:
            raise ValueError("Lookup must contain exactly 65,536 uint8 values")
        if np.any(np.diff(self.values.astype(np.int16)) < 0):
            raise ValueError("Lookup must be nondecreasing")
        parameters = json.loads(self.parameters_json)
        if not isinstance(parameters, dict):
            raise ValueError("Parameters must be a JSON object")
        json.dumps(parameters, allow_nan=False)

    @property
    def values(self):
        return np.frombuffer(self.lookup, dtype=np.uint8)

    def apply(self, image, *, valid=None):
        image = np.asarray(image)
        if image.ndim != 2 or image.dtype.kind != "u" or image.dtype.itemsize != 2:
            raise ValueError("Lookup application requires a 2-D uint16 image")
        result = self.values[image]
        if valid is not None:
            valid = np.asarray(valid)
            if valid.dtype != np.bool_ or valid.shape != image.shape:
                raise ValueError("valid must be a matching Boolean coverage mask")
            result[~valid] = 0
        return result

    def to_dict(self):
        return dict(schema=SCHEMA, method=self.method, input_dtype="uint16",
                    output_dtype="uint8", quantization="floor_after_clip_0_255",
                    lut_sha256=hashlib.sha256(self.lookup).hexdigest(),
                    parameters=json.loads(self.parameters_json))

    def save(self, path):
        # Exclusive creation: fitted state is never silently overwritten.
        metadata = json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)
        with Path(path).open("xb") as stream:
            np.savez_compressed(stream, lut=self.values, metadata=np.array(metadata))

    @classmethod
    def load(cls, path):
        # Bound expanded archive size before np.load can allocate its arrays.
        from zipfile import ZipFile
        with ZipFile(path) as archive:
            entries = archive.infolist()
            if (len(entries) != 2 or {x.filename for x in entries} != {"lut.npy", "metadata.npy"}
                    or sum(x.file_size for x in entries) > 1_000_000):
                raise ValueError("Invalid or oversized histogram state archive")
        with np.load(path, allow_pickle=False) as data:
            lut, metadata = data["lut"], data["metadata"]
            if lut.dtype != np.uint8 or lut.shape != (65536,):
                raise ValueError("Invalid stored lookup shape/dtype")
            if metadata.shape != () or metadata.dtype.kind != "U":
                raise ValueError("Invalid stored metadata")
            info = json.loads(str(metadata))
        if not isinstance(info, dict) or set(info) != {
            "schema", "method", "input_dtype", "output_dtype", "quantization", "lut_sha256", "parameters"
        }:
            raise ValueError("Unsupported histogram state schema")
        obj = cls(info["method"], lut.tobytes(), json.dumps(info["parameters"], allow_nan=False))
        if obj.to_dict() != info:
            raise ValueError("Histogram state metadata or digest mismatch")
        return obj


def fit_nyul(source_hist, target_landmarks, *, percentiles=QUANTILES):
    """Fit one slide against an explicitly supplied, frozen training reference."""
    source = _mass(source_hist, "source histogram", size=65536)
    q = _vector(percentiles, "percentiles")
    if q.size < 2 or np.any(np.diff(q) <= 0):
        raise ValueError("Percentiles must be strictly increasing")
    lut, info = nyul_lut(histogram_quantiles(source, q), target_landmarks)
    info.update(percentiles=q.tolist(), source_histogram_sha256=_digest(source),
                reference_sha256=_digest(target_landmarks),
                tie_policy="median_target_at_ties_with_fixed_endpoints",
                outside_endpoints="clip")
    return HistogramLUT("nyul_landmarks_af", lut.tobytes(), json.dumps(info, allow_nan=False))


def fit_cdf(source_hist, target_hist):
    """Fit one slide to an explicit frozen target on uniform normalized bins."""
    lut = cdf_lut(source_hist, target_hist)
    info = dict(source_histogram_sha256=_digest(source_hist), reference_sha256=_digest(target_hist),
                reference_bins=65536, reference_domain=[0., 1.],
                interpolation="empirical_inverse_cdf")
    return HistogramLUT("empirical_cdf_af", lut.tobytes(), json.dumps(info, allow_nan=False))
