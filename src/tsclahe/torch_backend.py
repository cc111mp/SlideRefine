"""Optional learned controller and differentiable continuous CLAHE.

Gradients are supported through clip limits, LUTs, spatial interpolation,
and blending. Histograms and tissue/noise statistics are PRECOMPUTED constants:
this is NOT fully differentiable with respect to upstream image acquisition.
The NumPy and Torch paths implement the same continuous-count operator.
"""
from __future__ import annotations

import numpy as np
import hashlib
import warnings
from pathlib import Path
import torch
from torch import nn
from .config import Config
from .core import TileGrid, axis_weights, reliability_region
from .statistics import TileStatistics, FEATURE_NAMES
from .version import OPERATOR_VERSION, PREPROCESS_VERSION


def clipped_luts_torch(histograms: torch.Tensor, clips: torch.Tensor) -> torch.Tensor:
    """histograms (B,GY,GX,K); clips (B,GY,GX); output (B,GY,GX,K+1)."""
    if histograms.ndim != 4 or clips.shape != histograms.shape[:-1]:
        raise ValueError("Histogram/clip shape mismatch")
    if not torch.isfinite(histograms).all() or not torch.isfinite(clips).all():
        raise ValueError("Histograms and clips must be finite")
    if torch.any(histograms < 0) or torch.any(clips < 1):
        raise ValueError("Histograms must be nonnegative and clips >= 1")
    mass = histograms.sum(-1, keepdim=True)
    if torch.any(mass <= 0):
        raise ValueError("Histogram mass must be positive")
    p = histograms / mass
    bins = p.shape[-1]
    counts = torch.arange(1, bins + 1, device=p.device, dtype=p.dtype)
    cumulative = p.sort(dim=-1).values.cumsum(-1)
    cap = clips[..., None] / bins
    shift = ((1 - (bins - counts) * cap - cumulative) / counts).amax(-1).clamp_min(0)
    redistributed = torch.minimum(p + shift[..., None], cap)
    cdf = redistributed.cumsum(-1)
    # Exact endpoints, without in-place modification of autograd tensors.
    return torch.cat([torch.zeros_like(cdf[..., :1]), cdf[..., :-1], torch.ones_like(cdf[..., :1])], dim=-1)


def limit_strengths_torch(luts: torch.Tensor, strengths: torch.Tensor, config: Config):
    # Float64 cap calculation agrees with the NumPy fitted operator. Cast the
    # bounded strengths back; gradients remain connected through the cast.
    gain = config.bins * torch.diff(luts.to(torch.float64), dim=-1).amax(-1)
    limit = torch.where(gain > 1, (config.max_gain - 1) / (gain - 1).clamp_min(1e-15),
                        torch.ones_like(gain))
    return torch.minimum(strengths, limit.to(strengths.dtype)).clamp(0, config.max_strength)


def apply_luts_torch(image: torch.Tensor, mask: torch.Tensor, luts: torch.Tensor,
                     strengths: torch.Tensor, grid: TileGrid, config: Config) -> torch.Tensor:
    """Apply the global operator to full Bx1xHxW training crops/images.

    No HxWxK LUT volume is allocated. Row blocking keeps indexing arrays small.
    At training time autograd still retains the output computation graph.
    """
    if image.ndim != 4 or image.shape[1] != 1 or tuple(image.shape[-2:]) != grid.shape:
        raise ValueError("Expected Bx1xHxW image matching the fitted grid")
    if mask.shape != image.shape:
        raise ValueError("Mask/image shape mismatch")
    if luts.shape != (image.shape[0], *grid.grid_shape, config.bins + 1):
        raise ValueError("LUT shape mismatch")
    if strengths.shape != (image.shape[0], *grid.grid_shape):
        raise ValueError("Strength shape mismatch")
    if not torch.isfinite(image).all() or torch.any(image < 0) or torch.any(image > 1):
        raise ValueError("Image must be finite and in [0,1]")
    if not torch.isfinite(mask).all() or torch.any((mask != 0) & (mask != 1)):
        raise ValueError("Torch mask must be binary")
    x = image[:, 0]
    device, dtype = image.device, image.dtype
    cy, cx = grid.centers

    def weights(coords, centers):
        lo, hi, w = axis_weights(coords, centers)
        return (torch.as_tensor(lo, device=device), torch.as_tensor(hi, device=device),
                torch.as_tensor(w, device=device, dtype=dtype))

    xl, xr, wx = weights(np.arange(grid.shape[1]), cx)
    batch = torch.arange(x.shape[0], device=device)[:, None, None]
    # Nonlearned rejection raster, same global-coordinate code as CPU. This
    # transfers the small eligibility grid to CPU; it is not a production GPU kernel.
    reliability = torch.as_tensor(np.stack([
        reliability_region(grid, a, grid.shape, feather=config.reliability_feather)
        for a in (strengths.detach() > 0).cpu().numpy()
    ]), device=device, dtype=dtype)
    chunks = []
    for start in range(0, grid.shape[0], config.apply_block_rows):
        stop = min(start + config.apply_block_rows, grid.shape[0])
        yl, yr, wy = weights(np.arange(start, stop), cy)
        block = x[:, start:stop]
        pos = block * config.bins
        k = pos.long().clamp(0, config.bins - 1)
        frac = pos - k.to(dtype)
        correction = torch.zeros_like(block)
        for yy, yw in ((yl, 1 - wy), (yr, wy)):
            for xx, xw in ((xl, 1 - wx), (xr, wx)):
                weight = yw[None, :, None] * xw[None, None, :]
                low = luts[batch, yy[None, :, None], xx[None, None, :], k]
                high = luts[batch, yy[None, :, None], xx[None, None, :], k + 1]
                local = low + frac * (high - low)
                residual = (local - block).clamp(-config.max_delta, config.max_delta)
                correction = correction + weight * strengths[batch, yy[None, :, None], xx[None, None, :]] * residual
        out = (block + reliability[:, start:stop] * correction).clamp(0, 1)
        chunks.append(torch.where(mask[:, 0, start:stop].bool(), out, block))
    return torch.cat(chunks, dim=-2)[:, None]


class TileController(nn.Module):
    """Histogram/statistics-conditioned CNN, NOT the published IA-CLAHE CNN.

    A 1x1 -> 3x3 -> 1x1 CNN predicts two bounded controls on the tile grid.
    Clip prediction and blending retain non-learned tissue/SNR/contrast caps.
    There are no pretrained weights. Use a checkpoint only after validation.
    """
    def __init__(self, config: Config, width: int = 32):
        super().__init__()
        self.config = config
        if type(width) is not int or not 1 <= width <= 1024:
            raise ValueError("Controller width must be an integer in [1,1024]")
        self.width = width
        self.network = nn.Sequential(
            nn.Conv2d(config.bins + len(FEATURE_NAMES), width, 1), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width, 2, 1),
        )
        # Conservative half-cap initialization; not a trained model.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor, strength_cap: torch.Tensor):
        if features.ndim != 4 or features.shape[1] != self.config.bins + len(FEATURE_NAMES):
            raise ValueError("Feature tensor has incorrect channel count")
        controls = torch.sigmoid(self.network(features))
        snr_gate = features[:, self.config.bins + FEATURE_NAMES.index("snr_gate")]
        deficit = features[:, self.config.bins + FEATURE_NAMES.index("contrast_deficit")]
        clips = self.config.clip_min + (self.config.clip_max - self.config.clip_min) * controls[:, 0] * snr_gate * deficit
        strengths = strength_cap * controls[:, 1]
        return clips, strengths


def prepare_tensors(image: np.ndarray, mask: np.ndarray, stats: TileStatistics,
                    config: Config, device="cpu") -> dict:
    _, _, cap = stats.gates(config)
    def tensor(a):
        return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device).unsqueeze(0)
    return {
        "image": tensor(image[None]), "mask": tensor(mask[None]),
        "histograms": tensor(stats.histograms), "features": tensor(stats.features(config)),
        "cap": tensor(cap), "valid": tensor(stats.valid),
    }


def forward_prepared(model: TileController, tensors: dict, grid: TileGrid):
    clips, strengths = model(tensors["features"], tensors["cap"])
    luts = clipped_luts_torch(tensors["histograms"], clips)
    identity = torch.linspace(0, 1, model.config.bins + 1, device=luts.device, dtype=luts.dtype)
    luts = torch.where(tensors["valid"][..., None].bool(), luts, identity)
    strengths = limit_strengths_torch(luts, strengths, model.config)
    output = apply_luts_torch(tensors["image"], tensors["mask"], luts, strengths, grid, model.config)
    return output, clips, strengths


class TorchPredictor:
    """Bridge a trained controller into the regular NumPy/file inference path."""
    def __init__(self, model: TileController, device="cpu", metadata=None):
        self.metadata = metadata or {}
        self.model = model.to(device).eval()
        self.device = device

    def resolve_preprocessing(self, limits, *, mask=None, mask_mode="auto", dark=None, flat=None):
        policy = self.metadata.get("preprocessing")
        if not policy:
            warnings.warn("Unbound research checkpoint: normalization policy was not saved.", RuntimeWarning)
            return limits
        if policy.get("version") != PREPROCESS_VERSION:
            raise ValueError("Checkpoint preprocessing semantics differ from this release")
        if mask is None and mask_mode != "all":
            raise ValueError("Checkpoint requires an explicit tissue mask or all-tissue input")
        if dark is not None or flat is not None:
            raise ValueError("Paired trainer expects pre-corrected inputs; apply identical calibration before both paths")
        expected = policy.get("limits")
        if expected is None:
            if limits is not None:
                raise ValueError("Checkpoint expects percentile normalization, not fixed limits")
        elif limits is not None and list(limits) != list(expected):
            raise ValueError("Normalization limits differ from checkpoint preprocessing contract")
        return expected

    def validate_normalization(self, normalization):
        policy = self.metadata.get("preprocessing")
        if not policy:
            return
        if policy.get("version") != PREPROCESS_VERSION:
            raise ValueError("Checkpoint preprocessing semantics differ")
        expected = policy.get("limits")
        if expected is not None:
            if [normalization.low, normalization.high] != list(expected):
                raise ValueError("Streaming normalization differs from checkpoint")
        elif normalization.source not in ("tissue_percentiles", "fallback_no_tissue_or_constant"):
            raise ValueError("Checkpoint requires a shared tissue-percentile normalization")

    @property
    def config(self):
        return self.model.config

    @torch.inference_mode()
    def __call__(self, stats: TileStatistics, config: Config):
        if config != self.config:
            raise ValueError("Inference configuration differs from training checkpoint")
        features = torch.as_tensor(stats.features(config)[None], device=self.device)
        cap = torch.as_tensor(stats.gates(config)[2][None], device=self.device)
        clip, strength = self.model(features, cap)
        return clip[0].cpu().numpy(), strength[0].cpu().numpy()


def save_checkpoint(path, model: TileController, *, metadata=None):
    torch.save({"schema_version": 2, "operator_version": OPERATOR_VERSION,
                "preprocess_version": PREPROCESS_VERSION, "config": model.config.to_dict(),
                "width": model.width, "feature_names": list(FEATURE_NAMES),
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "metadata": metadata or {}}, path)


def load_predictor(path, *, device="cpu") -> TorchPredictor:
    # Restricted state-dict loading, never torch.load(..., weights_only=False).
    # As with all serialized models, only open checkpoints from trusted sources.
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or not {"config", "width", "state_dict"}.issubset(data):
        raise ValueError("Checkpoint is missing required metadata")
    if not isinstance(data.get("metadata", {}), dict) or not isinstance(data["state_dict"], dict):
        raise ValueError("Malformed checkpoint metadata/state_dict")
    if (data.get("schema_version") != 2 or data.get("feature_names") != list(FEATURE_NAMES)
            or data.get("operator_version") != OPERATOR_VERSION
            or data.get("preprocess_version") != PREPROCESS_VERSION):
        raise ValueError("Unsupported checkpoint schema or feature definition")
    cfg = Config.from_dict(data["config"])
    model = TileController(cfg, width=data["width"])
    if not all(isinstance(v, torch.Tensor) and torch.isfinite(v).all() for v in data["state_dict"].values()):
        raise ValueError("Checkpoint parameters must be finite")
    model.load_state_dict(data["state_dict"], strict=True)
    predictor = TorchPredictor(model, device, metadata=data.get("metadata", {}))
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    predictor.checkpoint_sha256 = digest.hexdigest()
    return predictor
