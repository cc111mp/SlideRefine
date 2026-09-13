"""Configuration: clip limits are multiples of the mean tissue-bin occupancy."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
from numbers import Real
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # Spatial scale is in pixels, NOT a number of CLAHE tiles.
    tile_size: tuple[int, int] = (128, 128)
    bins: int = 256
    clip_min: float = 1.0
    clip_max: float = 4.0
    max_strength: float = 0.65
    max_delta: float = 0.25
    contrast_target: float = 0.35
    snr_low: float = 1.0
    snr_high: float = 4.0
    noise_floor: float = 0.0001
    min_tissue_fraction: float = 0.02
    full_tissue_fraction: float = 0.30
    min_tissue_pixels: int = 64
    min_noise_coefficients: int = 16
    lower_percentile: float = 0.5
    upper_percentile: float = 99.5
    mask_sigma: float = 2.0
    mask_noise_multiplier: float = 4.0
    mask_min_component: int = 64
    apply_block_rows: int = 256
    # v0.2: separate artifact uncertainty, sensor saturation, and output safeguards.
    max_gain: float = 2.0
    reliability_feather: int = 8
    artifact_policy: str = "conservative"  # "warn" records without suppressing
    artifact_low: float = 0.80
    artifact_high: float = 0.95
    sensor_max: float | None = None  # actual acquisition threshold, never inferred from dtype
    max_sensor_saturated_fraction: float = 0.05

    def __post_init__(self) -> None:
        if len(self.tile_size) != 2 or any(type(v) is not int or v < 4 for v in self.tile_size):
            raise ValueError("tile_size must contain two integer pixel sizes >= 4")
        object.__setattr__(self, "tile_size", tuple(self.tile_size))
        for name in ("bins", "min_tissue_pixels", "min_noise_coefficients", "mask_min_component", "apply_block_rows"):
            v = getattr(self, name)
            if type(v) is not int or v < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 16 <= self.bins <= 4096:
            raise ValueError("bins must be between 16 and 4096")
        for f in fields(self):
            if f.name in ("tile_size", "artifact_policy"):
                continue
            v = getattr(self, f.name)
            if f.name == "sensor_max" and v is None:
                continue
            if isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v):
                raise ValueError(f"{f.name} must be a finite real number")
        if type(self.reliability_feather) is not int or self.reliability_feather < 0:
            raise ValueError("reliability_feather must be a nonnegative integer")
        if self.max_gain < 1:
            raise ValueError("max_gain must be >= 1")
        if self.artifact_policy not in ("conservative", "warn"):
            raise ValueError("artifact_policy must be 'conservative' or 'warn'")
        if not 0 <= self.artifact_low < self.artifact_high <= 1:
            raise ValueError("invalid artifact thresholds")
        if not 0 < self.max_sensor_saturated_fraction <= 1:
            raise ValueError("max_sensor_saturated_fraction must be in (0,1]")
        if not 1 <= self.clip_min <= self.clip_max:
            raise ValueError("require 1 <= clip_min <= clip_max")
        if not 0 <= self.max_strength <= 1 or not 0 < self.max_delta <= 1:
            raise ValueError("max_strength must be in [0,1]; max_delta in (0,1]")
        if not 0 < self.contrast_target <= 1:
            raise ValueError("contrast_target must be in (0,1]")
        if not 0 <= self.snr_low < self.snr_high or self.noise_floor <= 0:
            raise ValueError("invalid SNR thresholds or noise_floor")
        if not 0 <= self.min_tissue_fraction < self.full_tissue_fraction <= 1:
            raise ValueError("invalid tissue fraction thresholds")
        if not 0 <= self.lower_percentile < self.upper_percentile <= 100:
            raise ValueError("invalid normalization percentiles")
        if self.mask_sigma < 0 or self.mask_noise_multiplier < 0:
            raise ValueError("mask parameters cannot be negative")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["tile_size"] = list(self.tile_size)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
