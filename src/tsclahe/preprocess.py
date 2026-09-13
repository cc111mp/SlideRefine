"""Calibration, explicit foreground handling, and separately recorded scaling."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
from scipy import ndimage as ndi
from .config import Config

MAD_NORMAL = 0.6744897501960817


def validate_image(image: np.ndarray) -> np.ndarray:
    a = np.asarray(image)
    if a.ndim != 2 or 0 in a.shape:
        raise ValueError(f"Expected a non-empty 2-D grayscale image, got {a.shape}")
    if a.dtype.kind not in "uif" or not np.isfinite(a).all():
        raise ValueError("Image must contain finite real numeric values (not bool/complex)")
    if a.dtype.itemsize > 8:
        raise ValueError("Input precision above float64 is not supported")
    # Every integer up to 2**53 is exactly representable by our float64 arithmetic.
    if a.dtype.kind in "ui" and (int(a.min()) < -(2**53) or int(a.max()) > 2**53):
        raise ValueError("Integer counts beyond exact float64 precision (2**53) are unsupported")
    return a


def validate_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    a = np.asarray(mask)
    if a.shape != shape or a.dtype.kind not in "buif" or not np.isfinite(a).all():
        raise ValueError("Mask must be finite, numeric/bool, and the same shape as the image")
    # Accept binary 0/1 and conventional 0/255 masks; reject probability masks.
    values = np.unique(a)
    if not (np.isin(values, [0, 1]).all() or np.isin(values, [0, 255]).all() or np.isin(values, [0, 65535]).all()):
        raise ValueError("Mask must be binary (0/1, 0/255, or 0/65535), not a probability map")
    return a != 0


def haar_noise(image: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, int]:
    """MAD of orthonormal 2x2 HH coefficients; iid Gaussian noise proxy.

    The four weights are (+1,-1,-1,+1)/2; their squared sum is 1.
    This avoids using the iid MAD factor on an uncalibrated blur residual.
    Real fine texture, correlated scanner noise, and Poisson noise remain caveats.
    """
    h, w = image.shape
    h -= h % 2
    w -= w % 2
    if h == 0 or w == 0:
        return 0.0, 0
    a = image[:h, :w].astype(np.float64, copy=False)
    hh = (a[0::2, 0::2] - a[0::2, 1::2] - a[1::2, 0::2] + a[1::2, 1::2]) / 2.0
    if mask is not None:
        m = mask[:h, :w]
        valid = m[0::2, 0::2] & m[0::2, 1::2] & m[1::2, 0::2] & m[1::2, 1::2]
        hh = hh[valid]
    else:
        hh = hh.ravel()
    if hh.size == 0:
        return 0.0, 0
    median = np.median(hh)
    sigma = np.median(np.abs(hh - median)) / MAD_NORMAL
    return float(sigma), int(hh.size)


def flat_field_correct(image: np.ndarray, dark=None, flat=None) -> np.ndarray:
    """(raw - dark) * median(flat - dark) / (flat - dark).

    `flat` is a bright reference in raw detector units, NOT a pre-normalized gain
    map. Scalars or shape-matched references are accepted. Invalid gains fail.
    Negative dark-subtracted values are retained until intensity scaling.
    """
    a = validate_image(image).astype(np.float64, copy=True)

    def reference(v, name):
        b = np.asarray(v, dtype=np.float64)
        if b.ndim != 0 and b.shape != a.shape:
            raise ValueError(f"{name} must be a scalar or match image shape")
        if not np.isfinite(b).all():
            raise ValueError(f"{name} must be finite")
        return b

    d = np.float64(0) if dark is None else reference(dark, "dark")
    a = a - d
    if flat is not None:
        gain = reference(flat, "flat") - d
        if np.any(gain <= 0):
            raise ValueError("flat - dark must be strictly positive everywhere")
        a *= np.median(gain) / gain
    if not np.isfinite(a).all():
        raise ValueError("Calibration produced non-finite values")
    return a


def auto_tissue_mask(image: np.ndarray, config: Config) -> np.ndarray:
    """Conservative bright-foreground heuristic, NOT a trained tissue segmenter.

    Supplied masks or mask_mode='all' are preferred for tissue-only AF crops.
    Dark tissue, blood, and very dim signal may be missed by this heuristic.
    """
    a = validate_image(image).astype(np.float64, copy=False)
    smoothed = ndi.gaussian_filter(a, config.mask_sigma, mode="reflect")
    low, high = np.percentile(smoothed, [20, 99.5])
    if high - low <= max(np.finfo(np.float64).tiny, 8 * abs(np.spacing(float(high)))):
        return np.zeros(a.shape, dtype=bool)
    background = float(np.median(smoothed[smoothed <= low]))
    sigma, _ = haar_noise(a)
    threshold = max(background + config.mask_noise_multiplier * sigma,
                    background + 0.08 * (float(high) - background))
    mask = smoothed > threshold
    if min(mask.shape) >= 5:
        # Padding prevents closing from unnecessarily deleting border tissue.
        mask = ndi.binary_closing(np.pad(mask, 2, mode="edge"), iterations=1)[2:-2, 2:-2]
    labels, n = ndi.label(mask)
    if n:
        counts = np.bincount(labels.ravel())
        keep = counts >= config.mask_min_component
        keep[0] = False
        mask = keep[labels]
    return mask.astype(bool)


@dataclass(frozen=True)
class Normalization:
    low: float
    high: float
    source: str
    degenerate: bool = False

    def __post_init__(self):
        if not np.isfinite([self.low, self.high, self.high - self.low]).all() or self.high <= self.low:
            raise ValueError("Normalization requires finite low < high")

    def apply(self, image: np.ndarray) -> np.ndarray:
        a = validate_image(image).astype(np.float64, copy=False)
        return np.clip((a - self.low) / (self.high - self.low), 0, 1).astype(np.float32)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PreparedImage:
    image: np.ndarray
    mask: np.ndarray
    normalization: Normalization
    warnings: list[str]
    sensor_saturated: np.ndarray | None = None


def _fallback_limits(image: np.ndarray, raw_dtype) -> tuple[float, float]:
    if np.issubdtype(raw_dtype, np.unsignedinteger):
        return 0.0, float(np.iinfo(raw_dtype).max)
    # Do not min/max-stretch a nearly constant floating image.
    return min(0.0, float(image.min())), max(1.0, float(image.max()))


def prepare(image: np.ndarray, config: Config, *, mask=None, mask_mode="auto",
            limits: tuple[float, float] | None = None, dark=None, flat=None) -> PreparedImage:
    raw = validate_image(image)
    corrected = flat_field_correct(raw, dark, flat)
    messages = []
    if mask is not None:
        if mask_mode == "all":
            raise ValueError("Supply a mask OR mask_mode='all', not both")
        m = validate_mask(mask, raw.shape)
    elif mask_mode == "all":
        m = np.ones(raw.shape, dtype=bool)
    elif mask_mode == "auto":
        m = auto_tissue_mask(corrected, config)
        messages.append("Automatic bright-foreground mask is heuristic; inspect it for missed dim tissue.")
    else:
        raise ValueError("mask_mode must be 'auto' or 'all'")
    if not m.any():
        messages.append("Empty tissue mask: local enhancement is bypassed.")
    if limits is not None:
        if len(limits) != 2:
            raise ValueError("limits must contain low and high")
        norm = Normalization(float(limits[0]), float(limits[1]), "fixed")
    else:
        values = corrected[m]
        if values.size:
            lo, hi = np.percentile(values, [config.lower_percentile, config.upper_percentile])
        else:
            lo, hi = 0.0, 0.0
        degenerate = hi - lo <= max(np.finfo(np.float64).tiny, 8 * abs(np.spacing(float(hi))))
        if degenerate:
            lo, hi = _fallback_limits(corrected, raw.dtype)
            norm = Normalization(float(lo), float(hi), "fallback_no_tissue_or_constant", True)
            messages.append("No usable tissue intensity range: sensor/unit scaling used, enhancement bypassed.")
        else:
            norm = Normalization(float(lo), float(hi), "tissue_percentiles")
    saturated = None if config.sensor_max is None else raw >= config.sensor_max
    return PreparedImage(norm.apply(corrected), m, norm, messages, saturated)
