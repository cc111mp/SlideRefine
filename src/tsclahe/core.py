"""Continuous-count, per-tile CLAHE with globally anchored interpolation.

No calls to OpenCV are used. Histogram clipping is explicitly per tile.
This is an independent CLAHE variant, not a bit-exact OpenCV reproduction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Callable
import numpy as np
from scipy import ndimage as ndi
from .version import __version__, OPERATOR_VERSION, PREPROCESS_VERSION
from .config import Config
from .preprocess import PreparedImage, Normalization, prepare, validate_image, validate_mask
from .statistics import TileStatistics, summarize_tile, heuristic_controls


@dataclass(frozen=True)
class TileGrid:
    shape: tuple[int, int]
    tile_size: tuple[int, int]

    def __post_init__(self):
        for name in ("shape", "tile_size"):
            value = getattr(self, name)
            if len(value) != 2 or any(type(v) is not int or v < 1 for v in value):
                raise ValueError(f"{name} must contain two positive Python integers")

    @property
    def grid_shape(self):
        return tuple((n + t - 1) // t for n, t in zip(self.shape, self.tile_size))

    @property
    def centers(self):
        result = []
        for n, t in zip(self.shape, self.tile_size):
            start = np.arange(0, n, t)
            stop = np.minimum(start + t, n)
            result.append((start + stop - 1) / 2.0)
        return tuple(result)

    def tiles(self):
        ty, tx = self.tile_size
        for i, y0 in enumerate(range(0, self.shape[0], ty)):
            for j, x0 in enumerate(range(0, self.shape[1], tx)):
                yield i, j, y0, min(y0 + ty, self.shape[0]), x0, min(x0 + tx, self.shape[1])


def axis_weights(coordinates: np.ndarray, centers: np.ndarray):
    """Interpolate at actual centers, including non-full edge tiles."""
    if centers.size == 1:
        z = np.zeros(coordinates.shape, dtype=np.int64)
        return z, z, np.zeros(coordinates.shape, dtype=np.float32)
    right = np.clip(np.searchsorted(centers, coordinates, side="right"), 1, centers.size - 1)
    left = right - 1
    weight = np.clip((coordinates - centers[left]) / (centers[right] - centers[left]), 0, 1)
    return left, right, weight.astype(np.float32)


def clipped_luts(histograms: np.ndarray, clips: np.ndarray) -> np.ndarray:
    """Mass-preserving capped water filling; c is a multiple of 1/K.

    q_b = min(p_b + t, c/K), with sum(q)=1. c>=1 is required.
    Unlike v0.1's one-pass redistribution, the FINAL histogram obeys the cap.
    The sorted active-set solution is piecewise differentiable in c.
    """
    p = np.asarray(histograms, dtype=np.float64)
    c = np.asarray(clips, dtype=np.float64)
    if p.ndim != 3 or c.shape != p.shape[:2] or p.shape[-1] < 2:
        raise ValueError("Expected histograms (GY,GX,K) and clips (GY,GX)")
    if not np.isfinite(p).all() or np.any(p < 0) or not np.isfinite(c).all() or np.any(c < 1):
        raise ValueError("Histograms must be nonnegative; finite clip limits must be >= 1")
    mass = p.sum(axis=-1, keepdims=True)
    if np.any(mass <= 0) or not np.isfinite(mass).all():
        raise ValueError("Every histogram must have finite positive mass")
    p = p / mass
    bins = p.shape[-1]
    counts = np.arange(1, bins + 1, dtype=np.float64)
    cumulative = np.cumsum(np.sort(p, axis=-1), axis=-1)
    cap = c[..., None] / bins
    shift = np.maximum(0, ((1 - (bins - counts) * cap - cumulative) / counts).max(-1))
    redistributed = np.minimum(p + shift[..., None], cap)
    cdf = np.concatenate([np.zeros((*p.shape[:2], 1)), np.cumsum(redistributed, axis=-1)], axis=-1)
    cdf[..., -1] = 1.0
    return np.clip(cdf, 0, 1).astype(np.float32)


def limit_strengths(luts: np.ndarray, strengths: np.ndarray, config: Config) -> np.ndarray:
    """Bound slopes of u+a*clip(F(u)-u,+/-max_delta) for FIXED context."""
    gain = config.bins * np.diff(luts.astype(np.float64), axis=-1).max(axis=-1)
    denominator = np.maximum(gain - 1, np.finfo(np.float64).eps)
    limit = np.where(gain > 1, (config.max_gain - 1) / denominator, 1.0)
    return np.minimum(strengths, limit).clip(0, config.max_strength).astype(np.float32)


def reliability_region(grid: TileGrid, eligible: np.ndarray, shape, origin=(0, 0), feather=8):
    """Rasterize rejected GLOBAL tiles, then feather ONLY into accepted tiles.

    A halo at least as wide as the taper radius makes this independent of
    application chunk size. No full-slide pixel reliability array is required.
    Rejected tiles remain exactly zero; foreground masking is separate.
    """
    if eligible.shape != grid.grid_shape:
        raise ValueError("Eligibility grid shape mismatch")
    y0, x0 = origin
    h, w = shape
    halo = int(feather) + 2
    ya, yb = max(0, y0 - halo), min(grid.shape[0], y0 + h + halo)
    xa, xb = max(0, x0 - halo), min(grid.shape[1], x0 + w + halo)
    iy = np.arange(ya, yb) // grid.tile_size[0]
    ix = np.arange(xa, xb) // grid.tile_size[1]
    allowed = eligible[iy[:, None], ix[None, :]]
    selection = np.s_[y0-ya:y0-ya+h, x0-xa:x0-xa+w]
    if not allowed.any():
        return np.zeros(shape, dtype=np.float32)
    if allowed.all():
        return np.ones(shape, dtype=np.float32)
    if feather == 0:
        return allowed[selection].astype(np.float32)
    distance = ndi.distance_transform_edt(allowed)
    t = np.clip(distance / feather, 0, 1)
    return (t * t * (3 - 2 * t))[selection].astype(np.float32)


@dataclass
class FittedTransform:
    grid: TileGrid
    config: Config
    luts: np.ndarray
    clip_limits: np.ndarray
    strengths: np.ndarray
    statistics: TileStatistics | None = None
    normalization: Normalization | None = None

    def apply_region(self, image: np.ndarray, mask: np.ndarray, *, origin=(0, 0)) -> np.ndarray:
        """Apply to a normalized region using GLOBAL coordinates.

        Reuse one fitted transform for every chunk of a WSI. Do not refit on
        each inference crop. Foreground-excluded pixels equal normalized input.
        """
        a = validate_image(image).astype(np.float32, copy=False)
        if np.min(a) < 0 or np.max(a) > 1:
            raise ValueError("apply_region expects normalized values in [0,1]")
        m = validate_mask(mask, a.shape)
        if len(origin) != 2 or any(type(v) is not int or v < 0 for v in origin):
            raise ValueError("origin must contain two nonnegative Python integers")
        y0, x0 = origin
        if y0 + a.shape[0] > self.grid.shape[0] or x0 + a.shape[1] > self.grid.shape[1]:
            raise ValueError("Region lies outside the fitted global image")
        cy, cx = self.grid.centers
        xl, xr, wx = axis_weights(np.arange(x0, x0 + a.shape[1]), cx)
        out = np.empty_like(a)
        bins = self.config.bins
        for start in range(0, a.shape[0], self.config.apply_block_rows):
            stop = min(start + self.config.apply_block_rows, a.shape[0])
            yl, yr, wy = axis_weights(np.arange(y0 + start, y0 + stop), cy)
            block = a[start:stop]
            pos = block * bins
            k = np.minimum(pos.astype(np.int64), bins - 1)
            fraction = pos - k
            correction = np.zeros_like(block)
            for yy, yw in ((yl, 1 - wy), (yr, wy)):
                for xx, xw in ((xl, 1 - wx), (xr, wx)):
                    weight = yw[:, None] * xw[None, :]
                    low = self.luts[yy[:, None], xx[None, :], k]
                    high = self.luts[yy[:, None], xx[None, :], k + 1]
                    local = low + fraction * (high - low)
                    residual = np.clip(local - block, -self.config.max_delta, self.config.max_delta)
                    correction += weight * self.strengths[yy[:, None], xx[None, :]] * residual
            reliability = reliability_region(self.grid, self.strengths > 0, block.shape,
                                             (y0 + start, x0), self.config.reliability_feather)
            adjusted = np.clip(block + reliability * correction, 0, 1)
            out[start:stop] = np.where(m[start:stop], adjusted, block)
        return out

    def save(self, path: str | Path) -> None:
        """Save the spatial transform; raw images and tissue masks are not stored."""
        self.validate()
        meta = {"schema_version": 2, "operator_version": OPERATOR_VERSION,
                "preprocess_version": PREPROCESS_VERSION, "shape": list(self.grid.shape),
                "config": self.config.to_dict(),
                "normalization": self.normalization.to_dict() if self.normalization else None}
        with Path(path).open("wb") as stream:
            np.savez_compressed(stream, metadata=json.dumps(meta), luts=self.luts,
                                clip_limits=self.clip_limits, strengths=self.strengths)

    @classmethod
    def load(cls, path: str | Path) -> "FittedTransform":
        with np.load(path, allow_pickle=False) as data:
            if not {"metadata", "luts", "clip_limits", "strengths"}.issubset(data.files):
                raise ValueError("Fitted-transform file is missing required arrays")
            meta = json.loads(str(data["metadata"]))
            if not isinstance(meta, dict) or not {"shape", "config"}.issubset(meta):
                raise ValueError("Fitted-transform metadata is malformed")
            if (meta.get("schema_version") != 2 or meta.get("operator_version") != OPERATOR_VERSION
                    or meta.get("preprocess_version") != PREPROCESS_VERSION):
                raise ValueError("Unsupported fitted-transform schema/operator; refit old v0.1 plans")
            cfg = Config.from_dict(meta["config"])
            shape = tuple(meta["shape"])
            if len(shape) != 2 or any(type(v) is not int or v < 1 for v in shape):
                raise ValueError("Invalid fitted image shape")
            grid = TileGrid(shape, cfg.tile_size)
            luts = data["luts"].astype(np.float32)
            clips = data["clip_limits"].astype(np.float32)
            strength = data["strengths"].astype(np.float32)
        norm = Normalization(**meta["normalization"]) if meta.get("normalization") else None
        result = cls(grid, cfg, luts, clips, strength, normalization=norm)
        result.validate()
        return result

    def validate(self):
        grid, cfg = self.grid, self.config
        luts, clips, strength = self.luts, self.clip_limits, self.strengths
        if grid.tile_size != cfg.tile_size:
            raise ValueError("Transform grid and config tile sizes differ")
        if luts.shape != (*grid.grid_shape, cfg.bins + 1) or clips.shape != grid.grid_shape or strength.shape != grid.grid_shape:
            raise ValueError("Malformed fitted-transform dimensions")
        if not all(np.isfinite(v).all() for v in (luts, clips, strength)):
            raise ValueError("Non-finite fitted-transform values")
        if np.any(np.diff(luts, axis=-1) < -2e-7) or luts.min() < 0 or luts.max() > 1:
            raise ValueError("LUTs must be monotone and bounded")
        if not np.allclose(luts[..., 0], 0, rtol=0, atol=2e-7) or not np.allclose(luts[..., -1], 1, rtol=0, atol=2e-7):
            raise ValueError("LUT endpoints must be 0 and 1")
        if np.any(strength < 0) or np.any(strength > cfg.max_strength + 1e-6):
            raise ValueError("Invalid enhancement strengths")
        if np.any(clips < cfg.clip_min) or np.any(clips > cfg.clip_max + 1e-6):
            raise ValueError("Invalid clip limits")
        gain = cfg.bins * np.diff(luts.astype(np.float64), axis=-1).max(-1)
        tolerance = 8 * np.finfo(np.float32).eps * cfg.bins
        if np.any(gain > clips + tolerance):
            raise ValueError("LUT histogram violates its clip cap")
        if np.any(1 + strength * (gain - 1) > cfg.max_gain + tolerance):
            raise ValueError("LUT blend violates configured max_gain")


def transform_from_statistics(grid: TileGrid, stats: TileStatistics, config: Config,
                              predictor: Callable | None = None) -> FittedTransform:
    clips, strengths = heuristic_controls(stats, config)
    if predictor is not None:
        proposed_clips, proposed_strengths = predictor(stats, config)
        for field in (proposed_clips, proposed_strengths):
            if np.shape(field) != grid.grid_shape or not np.isfinite(field).all():
                raise ValueError("Predictor returned invalid shapes or non-finite values")
        clips = np.clip(proposed_clips, config.clip_min, config.clip_max).astype(np.float32)
        # A learned predictor cannot override the heuristic evidence cap.
        strengths = np.clip(proposed_strengths, 0, strengths).astype(np.float32)
    luts = clipped_luts(stats.histograms, clips)
    # Low-support and constant tiles contribute an identity LUT, not a noisy CDF.
    luts[~stats.valid] = np.linspace(0, 1, config.bins + 1, dtype=np.float32)
    strengths = limit_strengths(luts, strengths, config)
    return FittedTransform(grid, config, luts, clips, strengths, stats)


def fit_normalized(image: np.ndarray, mask: np.ndarray, config: Config | None = None,
                   *, predictor=None, sensor_saturated=None) -> FittedTransform:
    cfg = config or Config()
    a = validate_image(image).astype(np.float32, copy=False)
    if a.min() < 0 or a.max() > 1:
        raise ValueError("fit_normalized requires values in [0,1]")
    m = validate_mask(mask, a.shape)
    if sensor_saturated is not None:
        sensor_saturated = validate_mask(sensor_saturated, a.shape)
    if cfg.sensor_max is not None and sensor_saturated is None:
        raise ValueError("Normalized fitting requires the pre-calibration sensor saturation mask")
    grid = TileGrid(a.shape, cfg.tile_size)
    stats = TileStatistics.allocate(grid.grid_shape, cfg.bins)
    for i, j, y0, y1, x0, x1 in grid.tiles():
        saturated = None if sensor_saturated is None else sensor_saturated[y0:y1, x0:x1]
        summarize_tile(a[y0:y1, x0:x1], m[y0:y1, x0:x1], stats, (i, j), cfg,
                       sensor_saturated=saturated)
    return transform_from_statistics(grid, stats, cfg, predictor)


@dataclass
class EnhancementResult:
    enhanced: np.ndarray
    normalized: np.ndarray
    mask: np.ndarray
    normalization: Normalization
    transform: FittedTransform
    warnings: list[str]
    provenance: dict = field(default_factory=dict)

    def summary(self) -> dict:
        st = self.transform.statistics
        masked = self.mask
        changed = np.abs(self.enhanced - self.normalized)
        return {
            "package_version": __version__, "operator_version": OPERATOR_VERSION, "preprocess_version": PREPROCESS_VERSION, "method": "independent_tissue_snr_aware_clahe",
            "shape": list(self.enhanced.shape), "config": self.transform.config.to_dict(),
            "normalization": self.normalization.to_dict(),
            "tissue_fraction": float(masked.mean()),
            "tile_grid": list(self.transform.grid.grid_shape),
            "valid_tiles": int(st.valid.sum()) if st is not None else None,
            "mean_strength": float(self.transform.strengths.mean()),
            "max_strength": float(self.transform.strengths.max()),
            "max_absolute_change_from_normalized": float(changed.max()),
            "background_max_absolute_change_from_normalized": float(changed[~masked].max()) if (~masked).any() else 0.0,
            "warnings": self.warnings,
            "provenance": self.provenance,
            "active_tiles": int((self.transform.strengths > 0).sum()),
            "directional_uncertainty_tiles": int((st.directional_coherence >= self.transform.config.artifact_low).sum()) if st else None,
            "sensor_saturation_measured": bool(st.sensor_saturation_known.any()) if st else False,
            "interpretation": "Enhanced intensities are not quantitative fluorescence measurements.",
        }


class TissueSNRCLAHE:
    def __init__(self, config: Config | None = None, *, predictor=None):
        self.config = config or Config()
        self.predictor = predictor

    def __call__(self, image: np.ndarray, *, mask=None, mask_mode="auto", limits=None,
                 dark=None, flat=None) -> EnhancementResult:
        if self.predictor is not None and hasattr(self.predictor, "resolve_preprocessing"):
            limits = self.predictor.resolve_preprocessing(limits, mask=mask, mask_mode=mask_mode,
                                                         dark=dark, flat=flat)
        p: PreparedImage = prepare(image, self.config, mask=mask, mask_mode=mask_mode,
                                   limits=limits, dark=dark, flat=flat)
        fitted = fit_normalized(p.image, p.mask, self.config, predictor=self.predictor,
                                sensor_saturated=p.sensor_saturated)
        fitted.normalization = p.normalization
        if (fitted.statistics.directional_coherence >= self.config.artifact_low).any():
            p.warnings.append("Directional coherence detected: could be scan artifact OR real oriented tissue; inspect diagnostics.")
        if self.config.sensor_max is None:
            p.warnings.append("Sensor saturation is unknown: no acquisition-specific sensor_max supplied.")
        if p.normalization.degenerate:
            fitted.strengths.fill(0)
        enhanced = fitted.apply_region(p.image, p.mask)
        result = EnhancementResult(enhanced, p.image, p.mask, p.normalization, fitted, p.warnings)
        result.provenance = {"input_dtype": str(np.asarray(image).dtype),
                             "calibration": {"dark": dark is not None, "flat": flat is not None},
                             "mask_mode": "supplied" if mask is not None else mask_mode,
                             "sensor_max": self.config.sensor_max,
                             "checkpoint_sha256": getattr(self.predictor, "checkpoint_sha256", None)}
        return result
