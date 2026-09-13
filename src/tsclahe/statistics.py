"""Masked distributions, uncalibrated noise proxies, and explicit uncertainty cues.

Directional coherence is NOT an artifact diagnosis: real fibers may trigger it.
The conservative policy abstains; 'warn' keeps diagnostics without suppression.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .config import Config
from .preprocess import haar_noise

FEATURE_NAMES = ("tissue_fraction", "contrast_span", "median", "noise_sigma",
                 "log_snr", "snr_gate", "contrast_deficit", "valid",
                 "directional_coherence", "artifact_gate", "upper_window_clip",
                 "lower_window_clip", "sensor_saturation", "sensor_saturation_known")


def smoothstep(x, lo, hi):
    t = np.clip((np.asarray(x) - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def directional_coherence(image: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of centered energy explained by row OR column means.

    A row/column stripe has a score near 1; independent noise is much lower
    for adequately supported tiles. Tissue orientation and mask geometry can
    also raise this score, so it represents uncertainty, not ground truth.
    """
    n = int(mask.sum())
    if n < 2:
        return 0.0
    residual = np.where(mask, image.astype(np.float64) - image[mask].mean(dtype=np.float64), 0.0)
    energy = float(np.square(residual).sum())
    if energy <= np.finfo(np.float64).tiny:
        return 0.0
    scores = []
    for axis in (0, 1):
        count = mask.sum(axis=axis)
        summed = residual.sum(axis=axis)
        explained = np.divide(summed * summed, count, out=np.zeros_like(summed), where=count > 0).sum()
        scores.append(explained / energy)
    return float(np.clip(max(scores), 0, 1))


@dataclass
class TileStatistics:
    histograms: np.ndarray
    tissue_fraction: np.ndarray
    contrast_span: np.ndarray
    median: np.ndarray
    noise_sigma: np.ndarray
    snr_proxy: np.ndarray
    tissue_count: np.ndarray
    noise_count: np.ndarray
    valid: np.ndarray
    directional_coherence: np.ndarray
    upper_window_clip: np.ndarray
    lower_window_clip: np.ndarray
    sensor_saturation: np.ndarray
    sensor_saturation_known: np.ndarray

    @classmethod
    def allocate(cls, grid_shape, bins):
        z = lambda: np.zeros(grid_shape, dtype=np.float32)
        return cls(np.zeros((*grid_shape, bins), dtype=np.float32),
                   z(), z(), z(), z(), z(),
                   np.zeros(grid_shape, dtype=np.int64),
                   np.zeros(grid_shape, dtype=np.int64),
                   np.zeros(grid_shape, dtype=bool), z(), z(), z(), z(),
                   np.zeros(grid_shape, dtype=bool))

    def artifact_gate(self, config: Config):
        if config.artifact_policy == "warn":
            return np.ones_like(self.directional_coherence)
        return 1 - smoothstep(self.directional_coherence, config.artifact_low, config.artifact_high)

    def gates(self, config: Config):
        snr = smoothstep(self.snr_proxy, config.snr_low, config.snr_high)
        tissue = smoothstep(self.tissue_fraction, config.min_tissue_fraction, config.full_tissue_fraction)
        deficit = np.clip(1 - self.contrast_span / config.contrast_target, 0, 1)
        sensor_gate = np.where(self.sensor_saturation_known,
                               np.clip(1 - self.sensor_saturation / config.max_sensor_saturated_fraction, 0, 1), 1)
        cap = config.max_strength * snr * tissue * deficit * self.valid * self.artifact_gate(config) * sensor_gate
        return snr.astype(np.float32), deficit.astype(np.float32), cap.astype(np.float32)

    def features(self, config: Config) -> np.ndarray:
        snr, deficit, _ = self.gates(config)
        scalars = np.stack([self.tissue_fraction, self.contrast_span, self.median,
                            self.noise_sigma, np.log1p(np.minimum(self.snr_proxy, 100)) / np.log(101),
                            snr, deficit, self.valid.astype(np.float32), self.directional_coherence,
                            self.artifact_gate(config), self.upper_window_clip, self.lower_window_clip,
                            self.sensor_saturation, self.sensor_saturation_known.astype(np.float32)], axis=0)
        hist = np.moveaxis(np.sqrt(self.histograms), -1, 0)
        return np.concatenate([hist, scalars]).astype(np.float32)


def summarize_tile(image, mask, stats: TileStatistics, index, config: Config, *, sensor_saturated=None):
    y, x = index
    values = image[mask]
    n = values.size
    stats.tissue_count[y, x] = n
    stats.tissue_fraction[y, x] = n / image.size
    stats.sensor_saturation_known[y, x] = sensor_saturated is not None
    if n == 0:
        stats.histograms[y, x] = 1 / config.bins
        return
    counts = np.histogram(values, bins=config.bins, range=(0.0, 1.0))[0]
    stats.histograms[y, x] = counts / n
    p10, median, p90 = np.percentile(values, [10, 50, 90])
    span = float(p90 - p10)
    noise, noise_count = haar_noise(image, mask)
    robust_sigma = span / 2.5631031310892007
    structure_sigma = np.sqrt(max(robust_sigma ** 2 - noise ** 2, 0.0))
    snr = min(float(structure_sigma / (noise + config.noise_floor)), 1e6)
    stats.contrast_span[y, x] = span
    stats.median[y, x] = median
    stats.noise_sigma[y, x] = noise
    stats.noise_count[y, x] = noise_count
    stats.snr_proxy[y, x] = snr
    stats.directional_coherence[y, x] = directional_coherence(image, mask)
    # These are normalization-window endpoints, NOT detector saturation.
    stats.upper_window_clip[y, x] = np.mean(values >= 1.0)
    stats.lower_window_clip[y, x] = np.mean(values <= 0.0)
    if sensor_saturated is not None:
        stats.sensor_saturation[y, x] = np.mean(sensor_saturated[mask])
    stats.valid[y, x] = (n >= config.min_tissue_pixels and
                        n / image.size >= config.min_tissue_fraction and
                        noise_count >= config.min_noise_coefficients and span > 1e-7)


def heuristic_controls(stats: TileStatistics, config: Config):
    snr, deficit, cap = stats.gates(config)
    clip = config.clip_min + (config.clip_max - config.clip_min) * snr * deficit
    return clip.astype(np.float32), cap
