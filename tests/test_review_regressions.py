"""New regression checks mapped to docs/review/REVIEW.md.

The prior audit named a different package. These tests target the actual
available tsclahe source, not transplanted assertions against missing modules.
"""
from dataclasses import replace
import json
import numpy as np
import pytest
import tifffile
from PIL import Image

from tsclahe import Config, TissueSNRCLAHE, FittedTransform, Normalization, fit_normalized
from tsclahe.cli import main
from tsclahe.core import TileGrid, clipped_luts, reliability_region
from tsclahe.io import read_image, read_mask
from tsclahe.preprocess import prepare, flat_field_correct


def constant_hole():
    y, x = np.mgrid[:192, :192]
    a = (.3 + .02 * np.sin(x / 4) * np.cos(y / 6)).astype(np.float32)
    a[64:128, 64:128] = .3
    return a, np.ones_like(a, bool)


def test_rejected_tile_exact_bypass_after_interpolation():
    a, mask = constant_hole()
    cfg = Config(tile_size=(64, 64), bins=64)
    r = TissueSNRCLAHE(cfg)(a, mask=mask, limits=(0, 1))
    assert r.transform.strengths[1, 1] == 0
    assert (r.transform.strengths > 0).any()
    np.testing.assert_array_equal(r.enhanced[64:128, 64:128], a[64:128, 64:128])
    r.transform.validate()


@pytest.mark.parametrize('feather', [0, 1, 8, 30])
def test_reliability_global_crop_invariant(feather):
    grid = TileGrid((193, 197), (64, 64))
    allowed = np.ones(grid.grid_shape, bool)
    allowed[1, 1] = False
    full = reliability_region(grid, allowed, grid.shape, feather=feather)
    assembled = np.empty(grid.shape, np.float32)
    for y in range(0, 193, 29):
        for x in range(0, 197, 37):
            shape = min(29, 193-y), min(37, 197-x)
            assembled[y:y+shape[0], x:x+shape[1]] = reliability_region(grid, allowed, shape, (y, x), feather)
    np.testing.assert_array_equal(assembled, full)
    assert not full[64:128, 64:128].any()
    assert full[20, 20] == 1


def test_stripe_uncertainty_and_explicit_warn_policy():
    y, x = np.mgrid[:256, :256]
    a = (.3 + .03 * np.sin(x / 6) + np.random.default_rng(23).normal(0, .001, x.shape)).astype(np.float32)
    cfg = Config(tile_size=(64, 64))
    conservative = TissueSNRCLAHE(cfg)(a, mask_mode='all', limits=(0, 1))
    warning_only = TissueSNRCLAHE(replace(cfg, artifact_policy='warn'))(a, mask_mode='all', limits=(0, 1))
    assert conservative.transform.statistics.directional_coherence.min() > cfg.artifact_high
    np.testing.assert_array_equal(conservative.enhanced, a)
    assert np.abs(warning_only.enhanced-a).max() > .005
    assert any('Directional' in w for w in conservative.warnings)


@pytest.mark.parametrize('suffix', ['.tif', '.npy', '.png'])
def test_boolean_mask_file_and_cli(tmp_path, suffix):
    y, x = np.mgrid[:64, :64]
    a = (.2 + .04 * np.sin(x/4) * np.cos(y/6)).astype(np.float32)
    mask = np.ones(a.shape, bool)
    input_path, mask_path = tmp_path/'input.tif', tmp_path/('mask'+suffix)
    tifffile.imwrite(input_path, a)
    if suffix == '.tif':
        tifffile.imwrite(mask_path, mask)
    elif suffix == '.npy':
        np.save(mask_path, mask)
    else:
        Image.fromarray(mask).save(mask_path)
    np.testing.assert_array_equal(read_mask(mask_path), mask)
    assert main([str(input_path), '--mask', str(mask_path), '--limits', '0', '1',
                 '--output-dir', str(tmp_path/'output')]) == 0
    metadata = json.loads((tmp_path/'output/metadata.json').read_text())
    assert len(metadata['provenance']['files']['input']['sha256']) == 64
    assert metadata['provenance']['files']['mask']['name'] == mask_path.name
    assert metadata['package_version'] == '0.2.0'


def test_uint16_mask_foreground_encoding(tmp_path):
    m = np.zeros((16, 16), np.uint16)
    m[3:13, 3:13] = 65535
    tifffile.imwrite(tmp_path/'mask.tif', m)
    np.testing.assert_array_equal(read_mask(tmp_path/'mask.tif'), m != 0)


def test_palette_intensity_rejected_but_binary_label_mask_supported(tmp_path):
    a = np.tile([0, 1], (16, 8)).astype(np.uint8)
    im = Image.fromarray(a).convert('P')
    im.putpalette([200,200,200,10,10,10] + [0]*762)
    path = tmp_path/'palette.png'
    im.save(path)
    with pytest.raises(ValueError, match='Palette'):
        read_image(path)
    np.testing.assert_array_equal(read_mask(path), a != 0)


def test_multiseries_default_rejected_explicit_selection_works(tmp_path):
    path = tmp_path/'two_series.tif'
    with tifffile.TiffWriter(path) as writer:
        writer.write(np.ones((32, 32), np.uint16))
        writer.write(np.full((16, 16), 2, np.uint16))
    with pytest.raises(ValueError, match='Multiple'):
        read_image(path)
    selected = read_image(path, series_index=1)
    assert selected.shape == (16,16) and np.all(selected == 2)
    with pytest.raises(ValueError, match='index'):
        read_image(path, series_index=2)


@pytest.mark.parametrize('dtype', [np.uint32, np.float64])
def test_high_offset_values_survive_calibration_and_normalization(dtype):
    a = (2**30 + np.arange(32)).astype(dtype).reshape(4,8)
    norm = Normalization(2**30, 2**30+31, 'fixed')
    expected = np.arange(32, dtype=np.float32).reshape(4,8)/31
    np.testing.assert_allclose(norm.apply(flat_field_correct(a)), expected, atol=1e-7)
    assert np.unique(norm.apply(a)).size == 32


def test_narrow_uint16_window_is_not_false_constant():
    a = np.tile(np.array([65000,65001], np.uint16), (16,16))
    p = prepare(a, Config(), mask_mode='all')
    assert not p.normalization.degenerate
    np.testing.assert_array_equal(np.unique(p.image), [0,1])


def test_reject_integer_counts_beyond_float64_exact_range():
    a = np.full((4,4), 2**60+1, np.uint64)
    with pytest.raises(ValueError, match='precision'):
        prepare(a, Config(), mask_mode='all')


def test_periodic_percentile_case_uses_all_tissue_not_stride():
    # This defect belonged to the other source variant; exact fitting is
    # retained in tsclahe and must not regress to lattice subsampling.
    a = np.broadcast_to(np.tile(np.array([100,200], np.uint16), 1024), (2048,2048))
    p = prepare(a, Config(), mask_mode='all')
    assert (p.normalization.low, p.normalization.high) == (100,200)
    np.testing.assert_array_equal(np.unique(p.image), [0,1])


def test_raw_sensor_saturation_is_not_normalization_clipping():
    raw = np.tile(np.linspace(100,300,64), (64,1))
    cfg = Config(tile_size=(64,64), sensor_max=4095)
    a = TissueSNRCLAHE(cfg)(raw, mask_mode='all', limits=(0,400))
    b = TissueSNRCLAHE(cfg)(raw, mask_mode='all', limits=(0,250))
    assert a.transform.statistics.sensor_saturation[0,0] == b.transform.statistics.sensor_saturation[0,0] == 0
    assert a.transform.statistics.upper_window_clip[0,0] == 0
    assert b.transform.statistics.upper_window_clip[0,0] == pytest.approx(.25)
    assert b.transform.statistics.sensor_saturation_known.all()
    unknown = TissueSNRCLAHE(replace(cfg,sensor_max=None))(raw, mask_mode='all', limits=(0,400))
    assert not unknown.transform.statistics.sensor_saturation_known.any()


def test_sensor_saturation_computed_before_flat_correction():
    raw = np.full((32,32), 4095, np.uint16)
    cfg = Config(tile_size=(32,32), sensor_max=4095)
    p = prepare(raw, cfg, mask_mode='all', limits=(0,10000), dark=100, flat=5000)
    assert p.sensor_saturated.all()
    r = TissueSNRCLAHE(cfg)(raw, mask_mode='all', limits=(0,10000), dark=100,flat=5000)
    assert r.transform.statistics.sensor_saturation[0,0] == 1
    assert r.transform.statistics.upper_window_clip[0,0] == 0
    np.testing.assert_array_equal(r.enhanced, r.normalized)


@pytest.mark.parametrize('bins', [16,32,64,128,256,1024,4096])
def test_final_histogram_cap_and_identity_at_one(bins):
    rng = np.random.default_rng(bins)
    hist = rng.dirichlet(np.full(bins,.1),size=(2,3))
    clips = rng.uniform(1,4,(2,3))
    luts = clipped_luts(hist,clips)
    mass = np.diff(luts.astype(np.float64),axis=-1)
    assert mass.min() >= 0
    assert np.all(mass <= clips[...,None]/bins+2e-7)
    np.testing.assert_allclose(mass.sum(-1),1,atol=2e-7)
    identity = clipped_luts(hist,np.ones((2,3)))
    np.testing.assert_allclose(identity,np.broadcast_to(np.linspace(0,1,bins+1),identity.shape),atol=2e-7)


def _alter_plan(path, **replace_arrays):
    with np.load(path,allow_pickle=False) as saved:
        values = {k:saved[k] for k in saved.files}
    values.update(replace_arrays)
    np.savez_compressed(path,**values)


@pytest.mark.parametrize('case', ['endpoints','gain','schema','nan','shape'])
def test_serialized_transform_rejects_corrupted_state(tmp_path,case):
    a,mask = constant_hole()
    cfg = Config(tile_size=(64,64),bins=64)
    t = fit_normalized(a,mask,cfg)
    path = tmp_path/'plan.npz';t.save(path)
    if case == 'endpoints':
        _alter_plan(path,luts=.2+.6*t.luts)
    elif case == 'gain':
        # Valid c=4 final histograms, but invalid blend slope > max_gain=2.
        peaked = np.zeros((*t.grid.grid_shape,cfg.bins));peaked[...,12] = 1
        clips = np.full(t.grid.grid_shape,4.)
        _alter_plan(path,luts=clipped_luts(peaked,clips),clip_limits=clips,
                    strengths=np.full(t.grid.grid_shape,cfg.max_strength))
    elif case == 'schema':
        with np.load(path) as saved: meta = json.loads(str(saved['metadata']))
        meta['schema_version'] = 1
        _alter_plan(path,metadata=json.dumps(meta))
    elif case == 'nan':
        invalid=t.luts.copy();invalid[0,0,2]=np.nan;_alter_plan(path,luts=invalid)
    else:
        _alter_plan(path,strengths=np.zeros((1,1)))
    with pytest.raises(ValueError):
        FittedTransform.load(path)


def test_fitted_gain_zero_enhancement_and_fixed_context_bound():
    a,mask = constant_hole()
    cfg=Config(tile_size=(64,64),bins=64,max_gain=1)
    r=TissueSNRCLAHE(cfg)(a,mask=mask,limits=(0,1))
    np.testing.assert_array_equal(r.enhanced,a)
    cfg=replace(cfg,max_gain=2)
    t=fit_normalized(a,mask,cfg)
    t.validate()
    # Perturb only the applied image, not the fitted plan, mask or coordinates.
    x=np.full(a.shape,.3331,np.float32);y=x+np.float32(.0001)
    change=t.apply_region(y,mask)-t.apply_region(x,mask)
    assert change.min() >= -1e-7
    assert change.max() <= 2*float((y-x).max())+2e-7


@pytest.mark.parametrize('kwargs',[{'max_gain':.9},{'artifact_policy':'none'},
                                  {'artifact_low':1.0},{'reliability_feather':-1},
                                  {'reliability_feather':1.2},{'sensor_max':float('nan')},
                                  {'max_strength':True}])
def test_new_configuration_validation(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)
