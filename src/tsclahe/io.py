"""Conservative 2-D I/O. Does not silently select a channel or destroy input files."""
from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
import numpy as np
from PIL import Image
import tifffile
from .preprocess import validate_image, validate_mask


def _read_array(path, *, channel=None, channel_axis=None, replicated_rgb=False,
               max_pixels=64_000_000, series_index=None, mask_input=False) -> np.ndarray:
    """Read TIFF/PNG/NPY; stacks and colored channels require explicit selection.

    max_pixels guards total loaded scalar samples, before TIFF decompression.
    For larger slides use the reader-based streaming API, not this convenience CLI.
    """
    path = Path(path)
    if (channel is None) != (channel_axis is None):
        raise ValueError("Provide both channel and channel_axis")
    if channel is not None and replicated_rgb:
        raise ValueError("Choose explicit channel selection OR replicated_rgb")
    if type(max_pixels) is not int or max_pixels < 1:
        raise ValueError("max_pixels must be positive")
    ext = path.suffix.lower()
    if ext in (".tif", ".tiff"):
        with tifffile.TiffFile(path) as tf:
            if len(tf.series) != 1 and series_index is None:
                raise ValueError("Multiple TIFF series: supply series_index / --series explicitly")
            index = 0 if series_index is None else series_index
            if type(index) is not int or not 0 <= index < len(tf.series):
                raise ValueError("TIFF series index is out of range")
            series = tf.series[index]
            if math.prod(series.shape) > max_pixels:
                raise ValueError("Image exceeds max_pixels; use streaming readers or an explicit larger limit")
            a = series.asarray()
    elif ext == ".npy":
        if series_index is not None:
            raise ValueError("series_index is only supported for TIFF")
        a = np.load(path, mmap_mode="r", allow_pickle=False)
        if a.size > max_pixels:
            raise ValueError("Array exceeds max_pixels; use streaming readers")
        a = np.asarray(a)
    elif ext == ".png":
        if series_index is not None:
            raise ValueError("series_index is only supported for TIFF")
        with Image.open(path) as im:
            if im.width * im.height * len(im.getbands()) > max_pixels:
                raise ValueError("PNG exceeds max_pixels")
            if im.mode == "P" and not mask_input:
                raise ValueError("Palette PNG is ambiguous; export explicit grayscale or RGB")
            a = np.asarray(im)
    else:
        raise ValueError("Supported inputs: .tif, .tiff, .png, .npy")
    if channel is not None:
        if a.ndim != 3:
            raise ValueError("Explicit channel selection requires a 3-D array")
        if channel_axis not in (0, 1, 2, -1, -2, -3):
            raise ValueError("channel_axis is out of range")
        if channel < 0 or channel >= a.shape[channel_axis]:
            raise ValueError("Channel index is out of range")
        a = np.take(a, channel, axis=channel_axis)
    elif replicated_rgb:
        if a.ndim != 3 or a.shape[-1] != 3:
            raise ValueError("replicated_rgb expects HxWx3")
        if not np.array_equal(a[..., 0], a[..., 1]) or not np.array_equal(a[..., 0], a[..., 2]):
            raise ValueError("RGB channels differ; refusing to discard biological/spectral information")
        a = a[..., 0]
    if a.size > max_pixels:
        raise ValueError("Loaded array exceeds max_pixels")
    return a


def read_image(path, *, channel=None, channel_axis=None, replicated_rgb=False,
               max_pixels=64_000_000, series_index=None) -> np.ndarray:
    return validate_image(_read_array(path, channel=channel, channel_axis=channel_axis,
                                      replicated_rgb=replicated_rgb, max_pixels=max_pixels,
                                      series_index=series_index))


def read_mask(path, *, max_pixels=64_000_000) -> np.ndarray:
    """Read bool TIFF/NPY, 1-bit PNG, or explicit binary masks independently of image dtype.

    Palette masks use label INDICES, not displayed palette brightness. Only
    binary encodings 0/1, 0/255 and 0/65535 are accepted; multilabel masks fail.
    """
    a = _read_array(path, max_pixels=max_pixels, mask_input=True)
    if a.ndim != 2 or 0 in a.shape:
        raise ValueError("Mask must be a nonempty 2-D array")
    return validate_mask(a, a.shape)


def file_provenance(path) -> dict:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"name": path.name, "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def write_image(path, image: np.ndarray, *, dtype="float32", overwrite=False):
    """Write unit-range images. Integer outputs encode [0,1], NOT raw counts."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}")
    a = validate_image(image)
    if a.min() < 0 or a.max() > 1:
        raise ValueError("write_image expects [0,1]")
    if dtype == "float32":
        out = a.astype(np.float32)
    elif dtype in ("uint8", "uint16"):
        out = np.rint(a * np.iinfo(dtype).max).astype(dtype)
    else:
        raise ValueError("Output dtype must be float32, uint8, or uint16")
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext in (".tif", ".tiff"):
        tifffile.imwrite(path, out, photometric="minisblack", metadata={"axes": "YX"})
    elif ext == ".npy":
        np.save(path, out, allow_pickle=False)
    elif ext == ".png" and dtype != "float32":
        Image.fromarray(out).save(path)
    else:
        raise ValueError("Use TIFF/NPY for float32; PNG is supported only for integer outputs")


def write_result(directory, result, *, output_dtype="float32"):
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    write_image(directory / "enhanced.tif", result.enhanced, dtype=output_dtype)
    write_image(directory / "normalized.tif", result.normalized)
    write_image(directory / "tissue_mask.png", result.mask.astype(np.float32), dtype="uint8")
    result.transform.save(directory / "transform.npz")
    st = result.transform.statistics
    if st is not None:
        np.savez_compressed(directory / "diagnostics.npz",
                            clip_limits=result.transform.clip_limits,
                            strengths=result.transform.strengths,
                            tissue_fraction=st.tissue_fraction,
                            contrast_span=st.contrast_span,
                            noise_sigma=st.noise_sigma, snr_proxy=st.snr_proxy,
                            valid=st.valid, tissue_count=st.tissue_count,
                            noise_count=st.noise_count,
                            directional_coherence=st.directional_coherence,
                            artifact_gate=st.artifact_gate(result.transform.config),
                            upper_window_clip=st.upper_window_clip,
                            lower_window_clip=st.lower_window_clip,
                            sensor_saturation=st.sensor_saturation,
                            sensor_saturation_known=st.sensor_saturation_known,
                            eligible=result.transform.strengths > 0)
    (directory / "metadata.json").write_text(json.dumps(result.summary(), indent=2, allow_nan=False), encoding="utf-8")
