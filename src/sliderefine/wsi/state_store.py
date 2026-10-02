"""Versioned block store for the unchanged heuristic tsclahe operator.

No full-grid histogram, LUT, feature, or eligibility array is allocated. Block
fitting uses the reference functions; array proxies let its unchanged renderer
fetch only the indexed cells. This is serial immutable state, not a scheduler.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import fields
import hashlib
import json
from pathlib import Path

import numpy as np
from tsclahe.config import Config
from tsclahe.core import TileGrid, FittedTransform, transform_from_statistics
from tsclahe.preprocess import Normalization, validate_image, validate_mask
from tsclahe.statistics import TileStatistics, summarize_tile, FEATURE_NAMES
from tsclahe.streaming import _read
from tsclahe.version import OPERATOR_VERSION, PREPROCESS_VERSION

SCHEMA = "sliderefine.clahe-blocks/v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def backend_identity():
    import tsclahe
    root = Path(tsclahe.__file__).parent
    return {name: file_digest(root / name) for name in
            ("core.py", "statistics.py", "preprocess.py", "config.py", "version.py")}


def _json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def _pair(value):
    if len(value) != 2 or any(type(n) is not int or n < 1 for n in value):
        raise ValueError("block_shape must contain two positive integers (analysis cells)")
    return tuple(value)


def block_origins(grid_shape, block_shape):
    for y in range(0, grid_shape[0], block_shape[0]):
        for x in range(0, grid_shape[1], block_shape[1]):
            yield y, x


def block_name(y, x):
    return f"y{y:09d}_x{x:09d}"


def estimate_block_bytes(config, block_shape):
    by, bx = _pair(block_shape)
    # Conservative temporary allowance as in the retained RAM runner, plus
    # explicitly saved feature channels. Pixel readers/cache are separate.
    return by * bx * (132 * config.bins + 2048)


def manifest_identity(source):
    """Bind all declared pixel/mask/valid files, not just manifest text."""
    if file_digest(source.path) != source.digest:
        raise ValueError("Input manifest changed during execution")
    root = source.path.parent
    ledgers = {"pixels": [], "masks": [], "validity": []}
    for tile in source.manifest["tiles"]:
        for key, group in (("path", "pixels"), ("mask_path", "masks"), ("valid_path", "validity")):
            if tile.get(key) is not None:
                p = Path(tile[key])
                ledgers[group].append({"path": str(p), "sha256": file_digest(root / p)})
    return {"manifest_sha256": source.digest,
            **{k + "_sha256": digest(v) for k, v in ledgers.items()},
            "mask_policy": source.manifest["mask_policy"], "model_sha256": None,
            "controller": "unchanged_heuristic"}


def fit_disk(shape, image_reader, mask_reader, normalization, config, path, *,
             identity, block_shape=(8, 8), cache_bytes=16 * 1024**2,
             max_state_bytes=512 * 1024**2, saturation_reader=None, predictor=None):
    """Fit cell blocks with exactly the reference per-cell math.

    Caller-supplied identity must bind immutable pixels and masks. The manifest
    runner supplies content hashes. Direct readers must provide their own hashes.
    A failed fit has no COMPLETE marker and cannot be loaded or resumed.
    """
    if predictor is not None:
        raise NotImplementedError("Learned controller grid halos are not implemented in the disk store")
    grid = TileGrid(shape, config.tile_size)
    block_shape = _pair(block_shape)
    if type(cache_bytes) is not int or cache_bytes < 0:
        raise ValueError("cache_bytes must be a nonnegative integer")
    if type(max_state_bytes) is not int or max_state_bytes < 1:
        raise ValueError("max_state_bytes must be positive")
    if estimate_block_bytes(config, block_shape) + cache_bytes > max_state_bytes:
        raise MemoryError("Disk fit block plus cache exceeds state working budget")
    if not isinstance(identity, dict) or not identity:
        raise ValueError("An explicit immutable source/mask identity is required")
    if config.sensor_max is not None and saturation_reader is None:
        raise ValueError("sensor_max requires a pre-calibration saturation_reader")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    metadata = dict(schema=SCHEMA, shape=list(shape), config=config.to_dict(),
                    normalization=normalization.to_dict(), block_shape=list(block_shape),
                    identity=identity, identity_sha256=digest(identity),
                    config_sha256=digest(config.to_dict()),
                    normalization_sha256=digest(normalization.to_dict()),
                    operator_version=OPERATOR_VERSION, preprocess_version=PREPROCESS_VERSION,
                    backend_sha256=backend_identity(), feature_names=list(FEATURE_NAMES),
                    feature_layout="sqrt_histogram_channels_then_FEATURE_NAMES; CYX",
                    model_sha256=None, controller="unchanged_heuristic")
    _json(path / "state.json", metadata)
    chain = hashlib.sha256()
    ty, tx = config.tile_size
    for y, x in block_origins(grid.grid_shape, block_shape):
        ny, nx = min(block_shape[0], grid.grid_shape[0]-y), min(block_shape[1], grid.grid_shape[1]-x)
        stats = TileStatistics.allocate((ny, nx), config.bins)
        for i in range(ny):
            for j in range(nx):
                bounds = ((y+i)*ty, min((y+i+1)*ty, shape[0]),
                          (x+j)*tx, min((x+j+1)*tx, shape[1]))
                pixels = normalization.apply(validate_image(_read(image_reader, bounds, "Image")))
                mask = validate_mask(_read(mask_reader, bounds, "Mask"), pixels.shape)
                saturated = None if saturation_reader is None else validate_mask(
                    _read(saturation_reader, bounds, "Saturation"), pixels.shape)
                summarize_tile(pixels, mask, stats, (i, j), config, sensor_saturated=saturated)
        local_grid = TileGrid((min(ny*ty, shape[0]-y*ty), min(nx*tx, shape[1]-x*tx)), config.tile_size)
        fitted = transform_from_statistics(local_grid, stats, config)
        if normalization.degenerate:
            fitted.strengths.fill(0)
        fitted.validate()
        arrays = {"stat_"+f.name: getattr(stats, f.name) for f in fields(stats)}
        arrays.update(luts=fitted.luts, clip_limits=fitted.clip_limits,
                      strengths=fitted.strengths, features=stats.features(config))
        name = block_name(y, x)
        np.savez(path / (name+".npz"), **arrays)
        record = dict(origin=[y, x], shape=[ny, nx], sha256=file_digest(path / (name+".npz")))
        _json(path / (name+".json"), record)
        chain.update(bytes.fromhex(digest(record)))
    _json(path / "COMPLETE.json", dict(schema=SCHEMA, state_sha256=file_digest(path / "state.json"),
                                       blocks_sha256=chain.hexdigest()))
    return DiskTransform(path, expected_identity=identity, cache_bytes=cache_bytes)


class _GridArray:
    """Only the reference renderer's integer gather and eligibility operation."""
    def __init__(self, store, name, eligible=False):
        self.store, self.name, self.eligible = store, name, eligible
        self.shape = store.grid.grid_shape + ((store.config.bins+1,) if name == "luts" else ())

    def __gt__(self, value):
        if self.name != "strengths" or value != 0:
            raise TypeError("Only strengths > 0 eligibility is supported")
        return _GridArray(self.store, self.name, eligible=True)

    def __getitem__(self, indices):
        indices = np.broadcast_arrays(*indices)
        if len(indices) != len(self.shape) or any(a.dtype.kind not in "iu" for a in indices):
            raise TypeError("Disk grid arrays require integer gathers")
        if any(np.any((a < 0) | (a >= n)) for a, n in zip(indices, self.shape)):
            raise IndexError("Grid gather outside fitted state")
        y, x = indices[:2]
        by, bx = self.store.block_shape
        gy, gx = y // by * by, x // bx * bx
        code = gy * self.store.grid.grid_shape[1] + gx
        out = np.empty(y.shape, dtype=bool if self.eligible else np.float32)
        for value in np.unique(code):
            selected = code == value
            oy, ox = divmod(int(value), self.store.grid.grid_shape[1])
            block = self.store.block(oy, ox)[self.name]
            local = (y[selected]-oy, x[selected]-ox)
            if len(indices) == 3:
                local += (indices[2][selected],)
            v = block[local]
            out[selected] = v > 0 if self.eligible else v
        return out


class DiskTransform:
    def __init__(self, path, *, expected_identity=None, cache_bytes=16 * 1024**2):
        if type(cache_bytes) is not int or cache_bytes < 0:
            raise ValueError("cache_bytes must be a nonnegative integer")
        self.path = Path(path)
        if not (self.path / "COMPLETE.json").exists():
            raise ValueError("Incomplete disk state")
        complete = json.loads((self.path / "COMPLETE.json").read_text())
        meta = json.loads((self.path / "state.json").read_text())
        if complete.get("schema") != SCHEMA or meta.get("schema") != SCHEMA:
            raise ValueError("Unsupported disk state schema")
        if file_digest(self.path / "state.json") != complete["state_sha256"]:
            raise ValueError("Disk state metadata hash mismatch")
        if (meta["operator_version"] != OPERATOR_VERSION or meta["preprocess_version"] != PREPROCESS_VERSION
                or meta["backend_sha256"] != backend_identity()):
            raise ValueError("Disk state backend identity mismatch")
        for key in ("identity", "config", "normalization"):
            if digest(meta[key]) != meta[key+"_sha256"]:
                raise ValueError(f"Disk state {key} digest mismatch")
        if expected_identity is not None and digest(expected_identity) != meta["identity_sha256"]:
            raise ValueError("Disk state source/mask/model identity mismatch")
        self.config = Config.from_dict(meta["config"])
        self.grid = TileGrid(tuple(meta["shape"]), self.config.tile_size)
        self.block_shape = _pair(meta["block_shape"])
        self.normalization = Normalization(**meta["normalization"])
        self.metadata = meta
        self.cache_bytes, self.cache_size, self.peak_cache_bytes = cache_bytes, 0, 0
        self._cache = OrderedDict()
        chain = hashlib.sha256()
        for y, x in block_origins(self.grid.grid_shape, self.block_shape):
            record = self._record(y, x)
            chain.update(bytes.fromhex(digest(record)))
        if chain.hexdigest() != complete["blocks_sha256"]:
            raise ValueError("Disk state block ledger mismatch")
        self.luts = _GridArray(self, "luts")
        self.strengths = _GridArray(self, "strengths")
        self.clip_limits = _GridArray(self, "clip_limits")
        self._reference = FittedTransform(self.grid, self.config, self.luts,
                                         self.clip_limits, self.strengths,
                                         normalization=self.normalization)

    def _record(self, y, x):
        record = json.loads((self.path / (block_name(y, x)+".json")).read_text())
        expected = [min(b, n-o) for b, n, o in zip(self.block_shape, self.grid.grid_shape, (y, x))]
        if record["origin"] != [y, x] or record["shape"] != expected:
            raise ValueError("Disk state block geometry mismatch")
        return record

    def block(self, y, x):
        key = y, x
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        path = self.path / (block_name(y, x)+".npz")
        record = self._record(y, x)
        if file_digest(path) != record["sha256"]:
            raise ValueError("Disk state block checksum mismatch")
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data[k] for k in ("luts", "clip_limits", "strengths")}
        ny, nx = record["shape"]
        local_grid = TileGrid((ny*self.config.tile_size[0], nx*self.config.tile_size[1]), self.config.tile_size)
        FittedTransform(local_grid, self.config, **arrays).validate()
        for a in arrays.values():
            a.flags.writeable = False
        size = sum(a.nbytes for a in arrays.values())
        while self._cache and self.cache_size + size > self.cache_bytes:
            _, old = self._cache.popitem(last=False)
            self.cache_size -= sum(a.nbytes for a in old.values())
        if size <= self.cache_bytes:
            self._cache[key] = arrays
            self.cache_size += size
            self.peak_cache_bytes = max(self.peak_cache_bytes, self.cache_size)
        return arrays

    def apply_region(self, image, mask, *, origin=(0, 0)):
        return self._reference.apply_region(image, mask, origin=origin)
