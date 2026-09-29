"""Independent numerical checks and failure regressions; synthetic inputs only."""
import json

import numpy as np
import pytest

from sliderefine.contracts import RegionPixels, regions
from sliderefine.cli import main
from sliderefine.demo import make_demo
from sliderefine.normalization import (
    HistogramLUT, cdf_lut, cdf_values, fit_cdf, fit_nyul,
    fit_uint16_histogram, histogram_quantiles, nyul_lut,
)


def histogram(raw):
    return np.bincount(np.asarray(raw).ravel(), minlength=65536)


def test_histogram_quantiles_equal_numpy_with_ties_gaps_and_endpoints():
    raw = np.array([0, 0, 10, 10, 10, 300, 65535], dtype=np.uint16)
    q = [0, 1, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 99, 99.5, 100]
    np.testing.assert_allclose(histogram_quantiles(histogram(raw), q),
                               np.percentile(raw, q), rtol=0, atol=1e-10)


def test_nyul_piecewise_values_and_explicit_tie_policy():
    lut, info = nyul_lut([10, 20, 20, 30], [0, .25, .5, 1])
    assert info['repeated_landmark_count'] == 1
    assert info['merged_targets'] == [0, .375, 1]
    assert lut[[0, 10, 15, 20, 25, 30, 65535]].tolist() == [0, 0, 47, 95, 175, 255, 255]
    assert np.all(np.diff(lut.astype(int)) >= 0)


def test_cdf_hand_calculated_fixture():
    source = np.array([0, 0, 3, 3, 3, 100, 400, 65535], dtype=np.uint16)
    target = np.array([5, 5, 5, 30, 90, 90, 90, 90, 300, 65535], dtype=np.uint16)
    actual = cdf_values(histogram(source), histogram(target))[source]
    np.testing.assert_allclose(actual, [5, 5, 63.75, 63.75, 63.75, 82.5, 247.5, 65535],
                               rtol=0, atol=1e-10)


@pytest.mark.parametrize('seed', range(5))
def test_cdf_against_explicit_sorted_sample_ranks(seed):
    rng = np.random.default_rng(seed)
    source = rng.integers(0, 300, 400, dtype=np.uint16)
    target = rng.integers(0, 65536, 500, dtype=np.uint16)
    # Oracle operates on unique observed samples, not a fixed-bin histogram.
    sx, inverse, sc = np.unique(source, return_inverse=True, return_counts=True)
    tx, tc = np.unique(target, return_counts=True)
    expected = np.interp(np.cumsum(sc) / len(source), np.cumsum(tc) / len(target), tx)
    actual = cdf_values(histogram(source), histogram(target))[source]
    np.testing.assert_allclose(actual, expected[inverse], rtol=0, atol=1e-10)


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_nonfinite_landmarks_and_cdf_masses_fail_before_cast(bad):
    with np.errstate(all='raise'):
        for source, target in [([10, bad, 30], [0, .5, 1]), ([10, 20, 30], [0, bad, 1])]:
            with pytest.raises(ValueError):
                nyul_lut(source, target)
        for source, target in [([1, bad, 1], [1, 1, 1]), ([1, 1, 1], [1, bad, 1])]:
            with pytest.raises(ValueError):
                cdf_values(source, target)


@pytest.mark.parametrize('source,target', [
    ([], []), ([1], [0]), ([1, 1], [0, 1]), ([10, 5], [0, 1]),
    ([-1, 10], [0, 1]), ([0, 65536], [0, 1]), ([0, 10], [.1, .9]),
    ([0, 5, 10], [0, 2, 1]), ([0, 5, 10], [0, 1]),
    ([[0, 10]], [[0, 1]]), ([False, True], [0, 1]),
])
def test_invalid_landmark_contracts(source, target):
    with pytest.raises(ValueError):
        nyul_lut(source, target)


@pytest.mark.parametrize('bad', [[], [0, 0], [-1, 2], [[1, 1]], [True, True],
                                [1e308, 1e308], [1+2j, 3]])
def test_invalid_histogram_mass(bad):
    with np.errstate(all='raise'), pytest.raises(ValueError):
        cdf_values(bad, [1, 1])


def test_counts_and_percentiles_are_validated():
    for counts, q in [([1.5, 2], [50]), ([2**53], [50]), ([1, 1], [-1]),
                      ([1, 1], [101]), ([1, 1], [np.nan])]:
        with pytest.raises(ValueError):
            histogram_quantiles(counts, q)
    with pytest.raises(ValueError):
        cdf_lut([1, 1], [1, 1])
    h = histogram([10, 20, 30])
    for q in [[0, 100, 50], [0, 0, 100], [50]]:
        with pytest.raises(ValueError):
            fit_nyul(h, [0, .5, 1], percentiles=q)


@pytest.mark.parametrize('method', ['nyul', 'cdf'])
def test_frozen_state_roundtrip_shifted_chunks_and_invalid_coverage(tmp_path, method):
    raw = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    original = raw.copy()
    source = histogram(raw)
    target = histogram((raw.astype(np.uint32) // 2 + 100).astype(np.uint16))
    state = (fit_nyul(source, [0, .2, 1], percentiles=[0, 50, 100])
             if method == 'nyul' else fit_cdf(source, target))
    path = tmp_path / 'state.npz'
    state.save(path)
    loaded = HistogramLUT.load(path)
    assert state.to_dict() == loaded.to_dict()
    expected = state.apply(raw)
    actual = np.empty_like(expected)
    for r in reversed(list(regions(raw.shape, (31, 47)))):
        ys, xs = slice(r.y, r.y+r.height), slice(r.x, r.x+r.width)
        actual[ys, xs] = loaded.apply(raw[ys, xs].astype('>u2'))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(loaded.apply(raw[17:133, 31:197]), expected[17:133, 31:197])
    valid = np.ones(raw.shape, bool)
    valid[50:99, 80:120] = False
    result = loaded.apply(raw, valid=valid)
    assert not result[~valid].any()
    np.testing.assert_array_equal(result[valid], expected[valid])
    np.testing.assert_array_equal(raw, original)
    with pytest.raises(FileExistsError):
        loaded.save(path)
    with pytest.raises(ValueError):
        loaded.values[0] = 7
    with pytest.raises(ValueError):
        loaded.apply(raw.astype(np.uint8))
    with pytest.raises(ValueError):
        loaded.apply(raw, valid=np.ones(raw.shape, np.uint8))


def test_serialized_lut_mutation_detected(tmp_path):
    state = fit_nyul(histogram([0, 100, 1000]), [0, .5, 1], percentiles=[0, 50, 100])
    values = state.values.copy()
    values[0] = 1  # Still monotone, so the content digest must reject it.
    path = tmp_path/'changed.npz'
    np.savez_compressed(path, lut=values, metadata=np.array(json.dumps(state.to_dict())))
    with pytest.raises(ValueError, match='digest'):
        HistogramLUT.load(path)


class Source:
    dtype = np.dtype('uint16')
    shape = (29, 37)

    def __init__(self):
        self.raw = np.arange(np.prod(self.shape), dtype=np.uint16).reshape(self.shape)
        self.valid = np.ones(self.shape, bool)
        self.valid[8:13, 10:15] = False
        self.tissue = self.raw % 3 != 0

    def read_region(self, r):
        ys, xs = slice(r.y, r.y+r.height), slice(r.x, r.x+r.width)
        return RegionPixels(self.raw[ys, xs], self.valid[ys, xs], self.tissue[ys, xs])


def test_histogram_fit_counts_only_valid_tissue_once_across_chunks():
    source = Source()
    expected = histogram(source.raw[source.valid & source.tissue])
    for chunks in [(29, 37), (7, 11), (1, 1)]:
        np.testing.assert_array_equal(fit_uint16_histogram(source, chunk_shape=chunks), expected)
    source.tissue[:] = False
    with pytest.raises(ValueError, match='No valid tissue'):
        fit_uint16_histogram(source)


def test_cli_fits_synthetic_manifest_and_rejects_bad_reference(tmp_path, capsys):
    manifest = make_demo(tmp_path/'input')
    reference = tmp_path/'reference.json'
    reference.write_text(json.dumps(dict(method='nyul_landmarks_af',
                                         percentiles=[1, 50, 99], target_landmarks=[0, .4, 1])))
    output = tmp_path/'state.npz'
    assert main(['fit-histogram-lut', str(manifest), str(reference), str(output),
                 '--chunk-shape', '31', '47']) == 0
    info = json.loads(capsys.readouterr().out)
    state = HistogramLUT.load(output)
    assert state.to_dict() == info
    assert info['parameters']['fit_scope'] == 'valid_tissue'
    assert len(info['parameters']['manifest_sha256']) == 64
    with pytest.raises(SystemExit):
        main(['fit-histogram-lut', str(manifest), str(reference), str(output)])
    reference.write_text(json.dumps(dict(method='nyul_landmarks_af',
                                         percentiles=[1, 50, 99], target_landmarks=[0, float('nan'), 1])))
    bad_output = tmp_path/'bad.npz'
    with pytest.raises(SystemExit):
        main(['fit-histogram-lut', str(manifest), str(reference), str(bad_output)])
    assert not bad_output.exists()


def test_cli_cdf_and_already_normalized_input(tmp_path, capsys):
    manifest = make_demo(tmp_path/'input')
    reference = tmp_path/'reference.json'
    target = [0] * 65536
    target[32768] = 1
    reference.write_text(json.dumps(dict(method='empirical_cdf_af', domain=[0, 1],
                                         target_histogram=target)))
    output = tmp_path/'state.npz'
    assert main(['fit-histogram-lut', str(manifest), str(reference), str(output)]) == 0
    assert (HistogramLUT.load(output).values == 127).all()
    record = json.loads(manifest.read_text())
    record['intensity_domain'] = 'normalized'
    manifest.write_text(json.dumps(record))
    with pytest.raises(SystemExit):
        main(['fit-histogram-lut', str(manifest), str(reference), str(tmp_path/'bad.npz')])
    assert not (tmp_path/'bad.npz').exists()
