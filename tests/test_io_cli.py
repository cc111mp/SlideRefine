import json
import numpy as np
import pytest
from tsclahe.cli import main
from tsclahe.io import write_image, read_image


@pytest.mark.parametrize("dtype,suffix,tolerance", [("float32", ".tif", 1e-7), ("uint16", ".tif", 2/65535),
                                                   ("uint8", ".png", 2/255), ("float32", ".npy", 1e-7)])
def test_io_roundtrip(tmp_path, dtype, suffix, tolerance):
    x = np.random.default_rng(0).random((13, 17)).astype(np.float32)
    path = tmp_path / ("image" + suffix)
    write_image(path, x, dtype=dtype)
    loaded = read_image(path)
    if loaded.dtype.kind == "u":
        loaded = loaded.astype(np.float32) / np.iinfo(loaded.dtype).max
    np.testing.assert_allclose(x, loaded, atol=tolerance)
    with pytest.raises(FileExistsError):
        write_image(path, x, dtype=dtype)


def test_cli_end_to_end(tmp_path, scene, config):
    image, mask = scene
    write_image(tmp_path / "input.tif", image, dtype="uint16")
    write_image(tmp_path / "mask.png", mask.astype(np.float32), dtype="uint8")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.to_dict()))
    out = tmp_path / "result"
    args = [str(tmp_path / "input.tif"), "--mask", str(tmp_path / "mask.png"),
            "--output-dir", str(out), "--config", str(config_path), "--limits", "0", "65535"]
    assert main(args) == 0
    assert {"enhanced.tif", "normalized.tif", "tissue_mask.png", "transform.npz", "diagnostics.npz", "metadata.json"} <= {p.name for p in out.iterdir()}
    metadata = json.loads((out / "metadata.json").read_text())
    assert metadata["background_max_absolute_change_from_normalized"] == 0
    assert main(args) == 2  # Never silently overwrite a previous result.


def test_explicit_replicated_rgb(tmp_path):
    from PIL import Image
    x = np.arange(100, dtype=np.uint8).reshape(10, 10)
    rgb = np.repeat(x[..., None], 3, axis=-1)
    path = tmp_path / "rgb.png"
    Image.fromarray(rgb).save(path)
    with pytest.raises(ValueError):
        read_image(path)
    np.testing.assert_array_equal(read_image(path, replicated_rgb=True), x)
    rgb[0, 0, 1] = 99
    Image.fromarray(rgb).save(path)
    with pytest.raises(ValueError, match="differ"):
        read_image(path, replicated_rgb=True)
    np.testing.assert_array_equal(read_image(path, channel=0, channel_axis=-1), x)


def test_max_pixels_prevents_load(tmp_path):
    write_image(tmp_path / "big.tif", np.zeros((33, 37), np.float32))
    with pytest.raises(ValueError, match="max_pixels"):
        read_image(tmp_path / "big.tif", max_pixels=100)
