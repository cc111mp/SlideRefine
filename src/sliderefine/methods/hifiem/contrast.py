"""Explicit AF adaptation of HiFiEM's local/global contrast stage only.

No upstream EM inversion, uint8 quantization, stripe correction, or automatic
per-chunk mean fitting. Parameters use normalized intensity units. The same fixed
configuration and native-pixel neighborhood are used for every output chunk.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from scipy.ndimage import convolve1d, maximum_filter, minimum_filter
from scipy.stats import norm

UPSTREAM_COMMIT = "477efb57dc2b2d24e1ad56d1044cf99d22d39957"


@dataclass(frozen=True)
class HiFiEMContrastConfig:
    variant: str = "contrast_af"
    smoothing: int = 31
    minmax_size: int = 9
    ratio: float = 1.0
    sigma: float = 0.03
    offset: float = -0.06
    hp_sigma: float = 0.20
    mean: float = 0.50
    clip: float = 1.0
    redistribute: str = "leftmost"

    def __post_init__(self):
        if self.variant != "contrast_af":
            raise ValueError("Only HiFiEM variant='contrast_af' is integrated; full pipeline is not")
        for name in ("smoothing", "minmax_size"):
            x = getattr(self,name)
            if type(x) is not int or x < 1 or x % 2 != 1 or x > 1023:
                raise ValueError(f"{name} must be an odd integer in [1,1023]")
        for name in ("ratio","sigma","offset","hp_sigma","mean","clip"):
            x = getattr(self,name)
            if isinstance(x,(bool,np.bool_)) or not np.isscalar(x) or not np.isfinite(x):
                raise ValueError(f"{name} must be finite")
        if not 0 <= self.ratio <= 10 or self.sigma <= 0 or self.hp_sigma <= 0:
            raise ValueError("Require ratio in [0,10] and positive sigma/hp_sigma")
        if not 0 <= self.mean <= 1 or not 0 < self.clip <= 4:
            raise ValueError("Require mean in [0,1] and symmetric clip half-width in (0,4]")
        if self.redistribute not in ("leftmost","left","full"):
            raise ValueError("Unknown HiFiEM redistribution")

    @property
    def halo(self):
        # The triangular kernel has length 2*smoothing-1. The min/max path is
        # parallel, not composed with that convolution, for method='image'.
        return max(self.smoothing-1,self.minmax_size//2) if self.ratio > 0 else 0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls,data):
        if not isinstance(data,dict) or data.get("variant") != "contrast_af":
            raise ValueError("Declare HiFiEM variant='contrast_af' explicitly")
        return cls(**data)


def enhance_contrast_af(image, mask, config: HiFiEMContrastConfig):
    """Normalized 2-D input -> float32; supplied-mask background is unchanged.

    Always use masked low-pass normalization, even in all-tissue chunks, so
    numerical behavior does not switch with a processing boundary. This differs
    slightly from upstream's all-mask shortcut. Missing neighbors require a
    separately enforced execution policy (the WSI runner conservatively bypasses
    pixels whose full dependency neighborhood contains missing coverage).
    """
    x = np.asarray(image)
    m = np.asarray(mask)
    if x.ndim != 2 or not x.size or x.dtype.kind not in "uif" or not np.isfinite(x).all():
        raise ValueError("HiFiEM expects a nonempty finite real 2-D normalized plane")
    if x.min() < 0 or x.max() > 1 or m.dtype != np.bool_ or m.shape != x.shape:
        raise ValueError("Require normalized [0,1] intensities and a matching Boolean mask")
    baseline = x.astype(np.float32, copy=True)
    if not m.any():
        return baseline
    # Preserve upstream's 0..255 computational units WITHOUT quantizing samples.
    work = baseline * np.float32(255)
    center = config.mean * 255
    work[~m] = center
    if config.ratio > 0:
        box = np.ones(config.smoothing,np.float32) / np.float32(config.smoothing)
        weights = np.correlate(box,box,mode="full")
        def conv(a):
            a = convolve1d(a,weights,axis=1,mode="mirror")
            return convolve1d(a,weights,axis=0,mode="mirror")
        mf = m.astype(np.float32)
        hp = mf * (work-conv(work*mf)/(conv(mf)+np.finfo(np.float32).eps))
        lp = work-hp
        multi = config.ratio*mf*(1-norm.cdf(work,loc=lp+config.offset*255,scale=config.sigma*255))
        midpoint = 0.5*(maximum_filter(work,size=config.minmax_size,mode="mirror")+
                        minimum_filter(work,size=config.minmax_size,mode="mirror"))
        work = work + multi*(work-midpoint)
    z = (work-center)/(config.hp_sigma*255)
    ncdf = norm.cdf(z)
    bounds = np.array([-config.clip,config.clip])
    cdf, pdf = norm.cdf(bounds), norm.pdf(bounds)
    width = bounds[1]-bounds[0]
    delta = cdf[1]-cdf[0]-0.5*(pdf[0]+pdf[1])*width
    u = np.clip(z-bounds[0],0,width)
    middle = u*0.5*(2*pdf[0]+(pdf[1]-pdf[0])*u/width)
    left, right = np.minimum(ncdf,cdf[0]), np.maximum(ncdf-cdf[1],0)
    if config.redistribute == "leftmost":
        ncdf = (1+delta/cdf[0])*left + middle + right
    elif config.redistribute == "left":
        ncdf = (1+delta/(cdf[0]+0.5*(pdf[0]+pdf[1])*width))*(left+middle)+right
    else:
        ncdf = (left+middle+right)/(1-delta)
    # Match the upstream inclusive 256 levels over a 0..255 interval, then
    # rescale. Passing drange=(0,1) into upstream would instead double the CDF.
    out = (np.clip(256*ncdf,0,255)/255).astype(np.float32)
    out[~m] = baseline[~m]
    return out
