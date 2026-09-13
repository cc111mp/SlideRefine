import numpy as np
import pytest
from tsclahe import Normalization, TissueSNRCLAHE
from tsclahe.streaming import fit_streaming, apply_streaming


def test_streaming_matches_in_memory(scene, config):
    image, mask = scene
    # Exercise native uint16 + fixed global scaling in both passes.
    raw = np.rint(image * 65535).astype(np.uint16)
    norm = Normalization(0, 65535, "fixed")
    reader = lambda y0, y1, x0, x1: raw[y0:y1, x0:x1]
    mask_reader = lambda y0, y1, x0, x1: mask[y0:y1, x0:x1]
    fitted = fit_streaming(raw.shape, reader, mask_reader, norm, config)
    result = np.empty(raw.shape, np.float32)
    for y0, x0, block in apply_streaming(fitted, reader, mask_reader, norm, chunk_size=(37, 51)):
        result[y0:y0 + block.shape[0], x0:x0 + block.shape[1]] = block
    expected = TissueSNRCLAHE(config)(raw, mask=mask, limits=(0, 65535))
    np.testing.assert_array_equal(fitted.luts, expected.transform.luts)
    np.testing.assert_array_equal(result, expected.enhanced)


def test_streaming_bad_reader(config):
    bad = lambda *args: np.zeros((5, 5))
    with pytest.raises(ValueError, match="reader"):
        fit_streaming((32, 32), bad, bad, Normalization(0, 1, "fixed"), config)
