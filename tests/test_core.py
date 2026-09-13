from dataclasses import replace
import numpy as np
import pytest
from tsclahe import Config, TissueSNRCLAHE, FittedTransform, fit_normalized
from tsclahe.core import clipped_luts, TileGrid, axis_weights
from tsclahe.preprocess import flat_field_correct, haar_noise, prepare


@pytest.mark.parametrize("kwargs", [
    {"tile_size": (0, 32)}, {"bins": 8}, {"clip_min": 0}, {"clip_min": 5, "clip_max": 2},
    {"max_strength": 1.1}, {"max_delta": 0}, {"snr_high": 1, "snr_low": 2},
    {"noise_floor": 0}, {"lower_percentile": 99, "upper_percentile": 1},
    {"contrast_target": float("nan")}, {"apply_block_rows": -1}, {"tile_size": (32.0, 32)},
])
def test_invalid_config(kwargs):
    with pytest.raises((ValueError, TypeError)):
        Config(**kwargs)


def test_unknown_config_key():
    with pytest.raises(ValueError, match="Unknown"):
        Config.from_dict({"clip_limt": 2})


def test_config_roundtrip(config):
    assert Config.from_dict(config.to_dict()) == config


@pytest.mark.parametrize("image", [np.zeros((2, 3, 3)), np.zeros((0, 3)),
                                   np.full((4, 4), np.nan), np.full((4, 4), np.inf),
                                   np.ones((4, 4), complex), np.ones((4, 4), bool)])
def test_invalid_image(image):
    with pytest.raises(ValueError):
        TissueSNRCLAHE()(image)


def test_background_exactly_preserved_and_input_immutable(scene, config):
    image, mask = scene
    before, mask_before = image.copy(), mask.copy()
    result = TissueSNRCLAHE(config)(image, mask=mask, limits=(0, 1))
    np.testing.assert_array_equal(image, before)
    np.testing.assert_array_equal(mask, mask_before)
    np.testing.assert_array_equal(result.enhanced[~mask], result.normalized[~mask])
    assert np.abs(result.enhanced[mask] - result.normalized[mask]).mean() > 0.001
    assert result.enhanced.dtype == np.float32
    assert 0 <= result.enhanced.min() <= result.enhanced.max() <= 1


def test_strength_and_delta_bounds(scene, config):
    image, mask = scene
    result = TissueSNRCLAHE(config)(image, mask=mask, limits=(0, 1))
    assert result.transform.strengths.min() >= 0
    assert result.transform.strengths.max() <= config.max_strength
    assert np.max(np.abs(result.enhanced - result.normalized)) <= config.max_delta * config.max_strength + 1e-6


def test_zero_strength_bypasses_enhancement(scene, config):
    image, mask = scene
    result = TissueSNRCLAHE(replace(config, max_strength=0))(image, mask=mask)
    np.testing.assert_array_equal(result.enhanced, result.normalized)


@pytest.mark.parametrize("value", [0, 12000, 65535])
def test_constant_uint16_is_not_equalized(value):
    image = np.full((55, 73), value, np.uint16)
    r = TissueSNRCLAHE()(image, mask_mode="all")
    np.testing.assert_allclose(r.enhanced, value / 65535, atol=1e-7)
    assert not r.transform.strengths.any()
    assert r.normalization.degenerate


def test_empty_mask(scene, config):
    image, _ = scene
    r = TissueSNRCLAHE(config)(image, mask=np.zeros(image.shape, bool), limits=(0, 1))
    np.testing.assert_array_equal(r.enhanced, image)
    assert any("Empty" in w for w in r.warnings)


def test_tiny_image_conservative_bypass(config):
    image = np.arange(9, dtype=np.float32).reshape(3, 3) / 10
    r = TissueSNRCLAHE(config)(image, mask_mode="all", limits=(0, 1))
    np.testing.assert_array_equal(r.enhanced, image)


def test_mask_validation(scene, config):
    image, _ = scene
    with pytest.raises(ValueError):
        TissueSNRCLAHE(config)(image, mask=np.full(image.shape, .5))
    with pytest.raises(ValueError):
        TissueSNRCLAHE(config)(image, mask=np.zeros((3, 4)))
    with pytest.raises(ValueError):
        TissueSNRCLAHE(config)(image, mask=np.ones(image.shape), mask_mode="all")


def test_uint8_uint16_equivalence(config):
    image = np.arange(256, dtype=np.uint8).reshape(16, 16)
    r8 = TissueSNRCLAHE(config)(image, mask_mode="all")
    r16 = TissueSNRCLAHE(config)(image.astype(np.uint16) * 257, mask_mode="all")
    np.testing.assert_allclose(r8.enhanced, r16.enhanced, atol=2e-6)


def test_sixteen_bit_not_forced_to_eight_bit(config):
    image = np.tile(np.arange(4096, dtype=np.uint16), (8, 1))
    r = TissueSNRCLAHE(config)(image, mask_mode="all", limits=(0, 65535))
    assert np.unique(r.normalized).size == 4096
    assert np.unique(r.enhanced).size > 256


def test_flat_field_algebra():
    gain = np.tile(np.linspace(.5, 1.5, 41), (31, 1)).astype(np.float32)
    truth = np.full_like(gain, 80)
    dark = np.full_like(gain, 10)
    flat = 100 * gain + dark
    raw = truth * gain + dark
    np.testing.assert_allclose(flat_field_correct(raw, dark, flat), truth, atol=2e-5)
    with pytest.raises(ValueError):
        flat_field_correct(raw, dark=10, flat=5)
    with pytest.raises(ValueError):
        flat_field_correct(raw, dark=np.zeros((2, 2)))


def test_haar_gaussian_noise_calibration():
    rng = np.random.default_rng(10)
    sigma, count = haar_noise(rng.normal(0, .02, (512, 512)))
    assert count == 65536
    assert sigma == pytest.approx(.02, rel=.025)


def test_noisy_flat_receives_less_strength_than_structured(config):
    rng = np.random.default_rng(91)
    y, x = np.mgrid[:64, :64]
    clean = .2 + .06 * np.sin(x / 4) * np.cos(y / 5) + rng.normal(0, .002, (64, 64))
    noise = .2 + rng.normal(0, .025, (64, 64))
    good = TissueSNRCLAHE(config)(clean, mask_mode="all", limits=(0, 1))
    bad = TissueSNRCLAHE(config)(noise, mask_mode="all", limits=(0, 1))
    assert good.transform.strengths.mean() > .2
    assert bad.transform.strengths.mean() < .02


def test_background_is_excluded_from_histogram(config):
    image = np.zeros((32, 32), np.float32)
    mask = np.zeros_like(image, bool)
    mask[8:24, 8:24] = True
    image[mask] = .4
    fitted = fit_normalized(image, mask, config)
    hist = fitted.statistics.histograms[0, 0]
    assert hist.sum() == pytest.approx(1)
    assert hist[0] == 0
    assert hist[int(.4 * config.bins)] == 1
    assert fitted.statistics.tissue_count[0, 0] == 256


def test_uniform_histogram_maps_to_identity():
    p = np.full((2, 3, 64), 1 / 64)
    lut = clipped_luts(p, np.full((2, 3), 2))
    np.testing.assert_allclose(lut, np.broadcast_to(np.linspace(0, 1, 65), lut.shape), atol=1e-7)


def test_luts_monotonic_mass_and_endpoints():
    rng = np.random.default_rng(23)
    hist = rng.dirichlet(np.full(64, .2), size=(3, 5))
    lut = clipped_luts(hist, rng.uniform(1, 4, (3, 5)))
    assert np.all(np.diff(lut, axis=-1) >= 0)
    np.testing.assert_array_equal(lut[..., 0], 0)
    np.testing.assert_array_equal(lut[..., -1], 1)
    np.testing.assert_allclose(np.diff(lut).sum(-1), 1, atol=1e-7)


def test_irregular_edge_tile_centers():
    grid = TileGrid((70, 99), (32, 32))
    np.testing.assert_array_equal(grid.centers[0], [15.5, 47.5, 66.5])
    np.testing.assert_array_equal(grid.centers[1], [15.5, 47.5, 79.5, 97])
    lo, hi, w = axis_weights(np.array([-10., 97., 110.]), grid.centers[1])
    assert w[0] == 0 and w[-1] == 1


def test_chunk_application_matches_full_image(scene, config):
    image, mask = scene
    fitted = fit_normalized(image, mask, config)
    expected = fitted.apply_region(image, mask)
    actual = np.empty_like(image)
    for y in range(0, image.shape[0], 37):
        for x in range(0, image.shape[1], 41):
            view = np.s_[y:y+37, x:x+41]
            actual[view] = fitted.apply_region(image[view], mask[view], origin=(y, x))
    np.testing.assert_array_equal(actual, expected)


def test_transform_roundtrip(tmp_path, scene, config):
    image, mask = scene
    fitted = fit_normalized(image, mask, config)
    path = tmp_path / "fitted.npz"
    fitted.save(path)
    loaded = FittedTransform.load(path)
    np.testing.assert_array_equal(loaded.apply_region(image, mask), fitted.apply_region(image, mask))


def test_bad_region_is_rejected(scene, config):
    image, mask = scene
    fitted = fit_normalized(image, mask, config)
    with pytest.raises(ValueError):
        fitted.apply_region(image, mask, origin=(1, 0))
    with pytest.raises(ValueError):
        fitted.apply_region(image * 10, mask)


def test_learned_predictor_cannot_override_caps(scene, config):
    image, mask = scene
    bad_predictor = lambda stats, cfg: (np.full(stats.valid.shape, 1000), np.full(stats.valid.shape, 1000))
    normal = fit_normalized(image, mask, config)
    fitted = fit_normalized(image, mask, config, predictor=bad_predictor)
    # v0.2: larger proposed clips may require an even LOWER blend under max_gain.
    assert np.all(fitted.strengths <= normal.statistics.gates(config)[2] + 1e-7)
    assert np.all(fitted.strengths <= normal.strengths + 1e-7)
    fitted.validate()
    assert fitted.clip_limits.max() == config.clip_max


def test_auto_mask_simple_bright_foreground():
    from tsclahe.preprocess import auto_tissue_mask
    y, x = np.mgrid[:128, :128]
    true_mask = (x - 64) ** 2 + (y - 64) ** 2 < 40 ** 2
    rng = np.random.default_rng(48)
    image = np.where(true_mask, .3, .01) + rng.normal(0, .002, (128, 128))
    inferred = auto_tissue_mask(image, Config())
    assert inferred[64, 64]
    assert not inferred[:10, :10].any()
    assert (inferred & true_mask).sum() / true_mask.sum() > .95


def test_auto_mask_noise_only_is_empty():
    from tsclahe.preprocess import auto_tissue_mask
    image = np.random.default_rng(54).normal(.02, .002, (128, 128))
    assert not auto_tissue_mask(image, Config()).any()
