import numpy as np
import pytest
from tsclahe import Config


@pytest.fixture
def config():
    return Config(tile_size=(32, 32), bins=64, min_tissue_pixels=16,
                  min_noise_coefficients=4, apply_block_rows=19)


@pytest.fixture
def scene():
    rng = np.random.default_rng(17)
    y, x = np.mgrid[:137, :173]
    mask = ((x - 86) / 78) ** 2 + ((y - 68) / 60) ** 2 < 1
    image = np.where(mask, .25 + .055 * np.sin(x / 5) * np.cos(y / 6), .01)
    image += rng.normal(0, .002, image.shape)
    return image.astype(np.float32), mask


def pytest_sessionstart(session):
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
