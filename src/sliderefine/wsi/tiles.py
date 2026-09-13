"""Virtual single-channel slide assembled lazily from registered, nonoverlapping tiles.

Storage tiles may be TIFF/PNG/NPY; each must already be one 2-D channel. Each
stored tile is decoded as a whole with a size guard. This is NOT a native
pyramidal WSI reader. Missing pixels carry valid=False; zero is only a fill.
"""
from __future__ import annotations
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import numpy as np
from tsclahe.io import read_image, read_mask
from ..contracts import Region, RegionPixels

SCHEMA = "sliderefine.tiles/v1"

def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value

def _intersects(a: Region, b: Region) -> bool:
    return (a.x < b.x+b.width and b.x < a.x+a.width and
            a.y < b.y+b.height and b.y < a.y+a.height)

class TileManifestSource:
    def __init__(self, path, *, cache_bytes=128*1024**2,
                 max_tile_pixels=16_777_216, max_region_pixels=16_777_216):
        self.path = Path(path).resolve()
        raw = self.path.read_bytes()
        self.digest = hashlib.sha256(raw).hexdigest()
        self.manifest = json.loads(raw)
        m = self.manifest
        if m.get("schema") != SCHEMA:
            raise ValueError(f"Expected schema {SCHEMA}")
        if not isinstance(m.get("slide_id"), str) or not m["slide_id"]:
            raise ValueError("slide_id must be a nonempty string")
        if not isinstance(m.get("channel_id"), str) or not m["channel_id"]:
            raise ValueError("channel_id must explicitly identify the selected channel")
        if m.get("axes") != "YX" or m.get("level") != 0:
            raise ValueError("Bootstrap supports explicit single-channel YX, level=0 only")
        if m.get("intensity_domain") not in ("raw", "calibrated", "normalized"):
            raise ValueError("Declare intensity_domain: raw, calibrated, or normalized")
        if m.get("mask_policy") not in ("supplied", "all_valid"):
            raise ValueError("Declare mask_policy: supplied or all_valid (never automatic per tile)")
        if not isinstance(m.get("shape"), list) or len(m["shape"]) != 2:
            raise ValueError("shape must be [height,width]")
        self.shape = tuple(_integer(v, "shape", 1) for v in m["shape"])
        self.dtype = np.dtype(m.get("dtype", "INVALID"))
        if self.dtype.kind not in "uif" or self.dtype.itemsize > 8:
            raise ValueError("dtype must be a supported real numeric type, not bool")
        spacing = m.get("pixel_size_um")
        if spacing is not None:
            if len(spacing) != 2 or not np.isfinite(spacing).all() or min(spacing) <= 0:
                raise ValueError("pixel_size_um must be null or positive finite [Y,X]")
        self.cache_bytes = _integer(cache_bytes, "cache_bytes")
        self.max_tile_pixels = _integer(max_tile_pixels, "max_tile_pixels", 1)
        self.max_region_pixels = _integer(max_region_pixels, "max_region_pixels", 1)
        self._cache = OrderedDict()
        self._cache_size = 0
        self._bucket_size = 2048
        self._index = {}
        self.tiles = []
        entries = m.get("tiles")
        if not isinstance(entries, list) or not entries:
            raise ValueError("tiles must be a nonempty list")
        for entry in entries:
            rect = Region(_integer(entry["x"], "x"), _integer(entry["y"], "y"),
                          _integer(entry["width"], "width", 1), _integer(entry["height"], "height", 1))
            if rect.x+rect.width > self.shape[1] or rect.y+rect.height > self.shape[0]:
                raise ValueError("A source tile extends beyond the slide")
            if rect.width*rect.height > self.max_tile_pixels:
                raise ValueError("Source tile is too large for this full-tile reader")
            keys = list(self._keys(rect))
            previous = {i for k in keys for i in self._index.get(k, [])}
            if any(_intersects(rect, self.tiles[i][0]) for i in previous):
                raise ValueError("Overlapping source rectangles require registration/compositing; not supported")
            paths = {}
            for key in ("path", "mask_path", "valid_path"):
                value = entry.get(key)
                if key == "path" or (key == "mask_path" and m["mask_policy"] == "supplied"):
                    if not isinstance(value, str) or not value:
                        raise ValueError(f"Every tile requires {key}")
                if value is not None:
                    if not isinstance(value, str):
                        raise ValueError(f"{key} must be a path string")
                    p = (self.path.parent/value).resolve()
                    if not p.is_file():
                        raise FileNotFoundError(p)
                    paths[key] = p
            idx = len(self.tiles)
            self.tiles.append((rect, paths))
            for k in keys:
                self._index.setdefault(k, []).append(idx)

    def _keys(self, rect):
        for iy in range(rect.y//self._bucket_size, (rect.y+rect.height-1)//self._bucket_size+1):
            for ix in range(rect.x//self._bucket_size, (rect.x+rect.width-1)//self._bucket_size+1):
                yield iy, ix

    def _load(self, idx):
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]
        rect, paths = self.tiles[idx]
        image = read_image(paths["path"], max_pixels=self.max_tile_pixels)
        if image.shape != rect.shape or image.dtype != self.dtype:
            raise ValueError(f"Tile {idx} shape/dtype differs from the manifest")
        valid = (read_mask(paths["valid_path"], max_pixels=self.max_tile_pixels)
                 if "valid_path" in paths else np.ones(rect.shape, bool))
        tissue = (read_mask(paths["mask_path"], max_pixels=self.max_tile_pixels)
                  if self.manifest["mask_policy"] == "supplied" else np.ones(rect.shape, bool))
        if valid.shape != rect.shape or tissue.shape != rect.shape:
            raise ValueError(f"Tile {idx} mask shape differs from the manifest")
        result = RegionPixels(image, valid, tissue & valid)
        size = image.nbytes + valid.nbytes + result.tissue.nbytes
        while self._cache and self._cache_size + size > self.cache_bytes:
            _, old = self._cache.popitem(last=False)
            self._cache_size -= old.pixels.nbytes + old.valid.nbytes + old.tissue.nbytes
        if size <= self.cache_bytes:
            self._cache[idx] = result
            self._cache_size += size
        return result

    def read_region(self, region: Region) -> RegionPixels:
        if region.width*region.height > self.max_region_pixels:
            raise ValueError("Requested region exceeds the explicit region-memory guard")
        image = np.zeros(region.shape, dtype=self.dtype)
        valid = np.zeros(region.shape, bool)
        tissue = np.zeros(region.shape, bool)
        x0, y0 = max(0, region.x), max(0, region.y)
        x1 = min(self.shape[1], region.x+region.width)
        y1 = min(self.shape[0], region.y+region.height)
        if x0 >= x1 or y0 >= y1:
            return RegionPixels(image, valid, tissue)
        clipped = Region(x0, y0, x1-x0, y1-y0)
        indices = {i for k in self._keys(clipped) for i in self._index.get(k, [])}
        for idx in sorted(indices):
            tile, _ = self.tiles[idx]
            if not _intersects(clipped, tile):
                continue
            a = self._load(idx)
            xa, xb = max(x0, tile.x), min(x1, tile.x+tile.width)
            ya, yb = max(y0, tile.y), min(y1, tile.y+tile.height)
            src = np.s_[ya-tile.y:yb-tile.y, xa-tile.x:xb-tile.x]
            dst = np.s_[ya-region.y:yb-region.y, xa-region.x:xb-region.x]
            image[dst], valid[dst], tissue[dst] = a.pixels[src], a.valid[src], a.tissue[src]
        return RegionPixels(image, valid, tissue)
