"""Global level-0 coordinates. Coordinates are XY; array shapes are YX."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Protocol
import numpy as np

@dataclass(frozen=True)
class Region:
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self):
        if any(type(v) is not int for v in (self.x, self.y, self.width, self.height)):
            raise ValueError("Region coordinates and dimensions must be Python integers")
        if self.width < 1 or self.height < 1:
            raise ValueError("Region dimensions must be positive")

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        return self.y, self.y + self.height, self.x, self.x + self.width

    def expand(self, halo: int) -> Region:
        if type(halo) is not int or halo < 0:
            raise ValueError("halo must be a nonnegative integer")
        return Region(self.x-halo, self.y-halo, self.width+2*halo, self.height+2*halo)

@dataclass
class RegionPixels:
    pixels: np.ndarray
    valid: np.ndarray
    tissue: np.ndarray

class RegionSource(Protocol):
    shape: tuple[int, int]
    dtype: np.dtype
    def read_region(self, region: Region) -> RegionPixels: ...

def regions(shape: tuple[int, int], chunk_shape=(1024, 1024)) -> Iterator[Region]:
    if len(shape) != 2 or len(chunk_shape) != 2:
        raise ValueError("shape and chunk_shape must be YX pairs")
    if any(type(v) is not int or v < 1 for v in (*shape, *chunk_shape)):
        raise ValueError("shape/chunk dimensions must be positive integers")
    for y in range(0, shape[0], chunk_shape[0]):
        for x in range(0, shape[1], chunk_shape[1]):
            yield Region(x, y, min(chunk_shape[1], shape[1]-x), min(chunk_shape[0], shape[0]-y))
