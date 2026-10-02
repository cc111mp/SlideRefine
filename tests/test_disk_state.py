"""Execution equivalence against the retained, independent in-memory path."""
from dataclasses import fields, replace
import json
import numpy as np
import pytest
from tsclahe import Config
from tsclahe.preprocess import Normalization
from tsclahe.streaming import fit_streaming
from sliderefine import Region
from sliderefine.demo import make_demo
from sliderefine.wsi.executor import run_manifest
from sliderefine.wsi.tiles import TileManifestSource
from sliderefine.wsi.state_store import fit_disk, DiskTransform, block_name

IDENTITY = {'pixels_sha256': 'synthetic-fixture-v1', 'masks_sha256': 'synthetic-fixture-mask-v1', 'model_sha256': None}


def fixture(shape=(79,113), degenerate=False):
    y,x=np.indices(shape)
    raw=(1000+400*np.sin(x*.6)*np.cos(y*.4)+2*x+3*y).astype(np.uint16)
    mask=np.ones(shape,bool); mask[:8]=False; mask[21:38,30:65]=False
    cfg=Config(tile_size=(16,20),bins=32,min_tissue_pixels=8,min_noise_coefficients=2,
               snr_low=0,snr_high=1,contrast_target=.9,artifact_policy='warn',apply_block_rows=13)
    norm=Normalization(0,1,'fixed',True) if degenerate else Normalization(0,2400,'fixed')
    image=lambda y0,y1,x0,x1:raw[y0:y1,x0:x1]
    tissue=lambda y0,y1,x0,x1:mask[y0:y1,x0:x1]
    return raw,mask,cfg,norm,image,tissue


@pytest.mark.parametrize('block',[(1,1),(2,3),(20,20)])
@pytest.mark.parametrize('chunk',[(7,11),(29,43),(79,113)])
def test_disk_reference_parity_shifted_and_reversed(tmp_path,block,chunk):
    raw,mask,cfg,norm,image,tissue=fixture()
    ref=fit_streaming(raw.shape,image,tissue,norm,cfg)
    disk=fit_disk(raw.shape,image,tissue,norm,cfg,tmp_path/'state',identity=IDENTITY,
                  block_shape=block,cache_bytes=1300)
    expected=ref.apply_region(norm.apply(raw),mask)
    assert np.any(expected!=norm.apply(raw))  # Exercise enhancement, not identity-only parity.
    out=np.empty_like(expected)
    ys=sorted(set([0,3]+list(range(3,raw.shape[0],chunk[0]))+[raw.shape[0]]))
    xs=sorted(set([0,5]+list(range(5,raw.shape[1],chunk[1]))+[raw.shape[1]]))
    for ya,yb in reversed(list(zip(ys[:-1],ys[1:]))):
        for xa,xb in reversed(list(zip(xs[:-1],xs[1:]))):
            out[ya:yb,xa:xb]=disk.apply_region(norm.apply(raw[ya:yb,xa:xb]),mask[ya:yb,xa:xb],origin=(ya,xa))
    np.testing.assert_allclose(out,expected,atol=2e-7,rtol=0)
    np.testing.assert_array_equal(out[~mask],norm.apply(raw)[~mask])
    assert disk.cache_size<=1300 and disk.peak_cache_bytes<=1300
    # Every diagnostic/statistic and control is also compared, not just output.
    for p in (tmp_path/'state').glob('y*.json'):
        m=json.loads(p.read_text()); y,x=m['origin']; ny,nx=m['shape']; sl=np.s_[y:y+ny,x:x+nx]
        with np.load(p.with_suffix('.npz')) as a:
            for f in fields(ref.statistics):
                np.testing.assert_array_equal(a['stat_'+f.name],getattr(ref.statistics,f.name)[sl])
            for key in ['luts','strengths','clip_limits']:
                np.testing.assert_array_equal(a[key],getattr(ref,key)[sl])
            np.testing.assert_array_equal(a['features'],ref.statistics.features(cfg)[:,y:y+ny,x:x+nx])


@pytest.mark.parametrize('shape',[(1,1),(3,51),(43,2)])
def test_small_and_partial_cells(tmp_path,shape):
    raw,mask,cfg,norm,image,tissue=fixture(shape)
    ref=fit_streaming(shape,image,tissue,norm,cfg)
    d=fit_disk(shape,image,tissue,norm,cfg,tmp_path/'s',identity=IDENTITY)
    np.testing.assert_array_equal(d.apply_region(norm.apply(raw),mask),ref.apply_region(norm.apply(raw),mask))


def test_degenerate_saturation_and_reopen(tmp_path):
    raw,mask,cfg,norm,image,tissue=fixture(degenerate=True)
    cfg=replace(cfg,sensor_max=1200)
    saturation=lambda *b:image(*b)>=1200
    d=fit_disk(raw.shape,image,tissue,norm,cfg,tmp_path/'s',identity=IDENTITY,saturation_reader=saturation)
    d=DiskTransform(tmp_path/'s',expected_identity=IDENTITY,cache_bytes=0)
    np.testing.assert_array_equal(d.apply_region(norm.apply(raw),mask),norm.apply(raw))
    assert d.cache_size==0
    with pytest.raises(ValueError,match='identity mismatch'):
        DiskTransform(tmp_path/'s',expected_identity={'pixels_sha256':'changed'})


def test_corruption_and_incomplete_rejected(tmp_path):
    raw,mask,cfg,norm,image,tissue=fixture()
    fit_disk(raw.shape,image,tissue,norm,cfg,tmp_path/'s',identity=IDENTITY)
    p=tmp_path/'s'/ (block_name(0,0)+'.npz')
    with p.open('ab') as f:f.write(b'corruption')
    d=DiskTransform(tmp_path/'s',expected_identity=IDENTITY)
    with pytest.raises(ValueError,match='checksum'):d.apply_region(norm.apply(raw),mask)
    (tmp_path/'s'/'COMPLETE.json').unlink()
    with pytest.raises(ValueError,match='Incomplete'):DiskTransform(tmp_path/'s')


@pytest.mark.parametrize('change',['metadata','ledger'])
def test_metadata_and_ledger_integrity(tmp_path,change):
    raw,mask,cfg,norm,image,tissue=fixture()
    fit_disk(raw.shape,image,tissue,norm,cfg,tmp_path/'s',identity=IDENTITY)
    p=tmp_path/'s'/('state.json' if change=='metadata' else block_name(0,0)+'.json')
    obj=json.loads(p.read_text());obj['unexpected']='changed';p.write_text(json.dumps(obj))
    with pytest.raises(ValueError,match='hash|ledger'):DiskTransform(tmp_path/'s')


@pytest.mark.parametrize('kwargs,error',[
    ({'predictor':object()},NotImplementedError),({'max_state_bytes':1},MemoryError),
    ({'cache_bytes':-1},ValueError),({'block_shape':(0,2)},ValueError),
    ({'block_shape':(True,2)},ValueError)])
def test_fail_before_output(tmp_path,kwargs,error):
    raw,mask,cfg,norm,image,tissue=fixture()
    with pytest.raises(error):fit_disk(raw.shape,image,tissue,norm,cfg,tmp_path/'s',identity=IDENTITY,**kwargs)
    assert not (tmp_path/'s').exists()


@pytest.mark.parametrize('chunk',[(35,51),(128,160)])
def test_manifest_runner_and_gaps(tmp_path,chunk):
    manifest=make_demo(tmp_path/'input'); m=json.loads(manifest.read_text());m['tiles'].pop(0);manifest.write_text(json.dumps(m))
    cfg=Config(tile_size=(32,32),bins=32,min_tissue_pixels=8,min_noise_coefficients=2)
    ram=run_manifest(manifest,tmp_path/'ram',config=cfg,limits=(0,16000),chunk_shape=(71,93))
    disk=run_manifest(manifest,tmp_path/'disk',config=cfg,limits=(0,16000),chunk_shape=chunk,
                      state_backend='disk',state_block_shape=(2,3),state_cache_bytes=4000)
    assert disk['status']==ram['status']=='complete'
    assert (tmp_path/'disk/state/COMPLETE.json').exists()
    assert not (tmp_path/'disk/transform.npz').exists()
    for branch in ['baseline','enhanced']:
        a=TileManifestSource(tmp_path/'ram'/branch/'manifest.json').read_region(Region(0,0,389,257))
        b=TileManifestSource(tmp_path/'disk'/branch/'manifest.json').read_region(Region(0,0,389,257))
        np.testing.assert_allclose(a.pixels,b.pixels,atol=2e-7,rtol=0)
        np.testing.assert_array_equal(a.valid,b.valid)
        assert np.all(b.pixels[~b.valid]==0)


def test_non_degenerate_saturation_parity(tmp_path):
    raw,mask,cfg,norm,image,tissue=fixture()
    cfg=replace(cfg,sensor_max=1100)
    saturation=lambda *b:image(*b)>=1100
    ref=fit_streaming(raw.shape,image,tissue,norm,cfg,saturation_reader=saturation)
    disk=fit_disk(raw.shape,image,tissue,norm,cfg,tmp_path/'s',identity=IDENTITY,
                  saturation_reader=saturation,block_shape=(2,2))
    np.testing.assert_array_equal(disk.apply_region(norm.apply(raw),mask),ref.apply_region(norm.apply(raw),mask))


def test_changed_manifest_detected(tmp_path):
    from sliderefine.wsi.state_store import manifest_identity
    manifest=make_demo(tmp_path/'input'); source=TileManifestSource(manifest)
    identity=manifest_identity(source)
    assert identity['model_sha256'] is None
    m=json.loads(manifest.read_text());m['slide_id']='changed';manifest.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='manifest changed'):manifest_identity(source)


def test_source_pixels_masks_validity_bind_identity(tmp_path):
    from sliderefine.wsi.state_store import manifest_identity
    manifest=make_demo(tmp_path/'input');source=TileManifestSource(manifest)
    old=manifest_identity(source)
    pixels=source.tiles[0][1]['path']
    a=np.load(pixels);a[0,0]+=1;np.save(pixels,a)
    new=manifest_identity(source)
    assert new['pixels_sha256']!=old['pixels_sha256']
    assert new['masks_sha256']==old['masks_sha256']


def test_disk_cli_and_non_clahe_rejection(tmp_path,capsys):
    from sliderefine.cli import main
    manifest=make_demo(tmp_path/'input')
    assert main(['run',str(manifest),str(tmp_path/'out'),'--limits','0','16000',
                 '--state-backend','disk','--state-block-shape','2','3','--state-cache-mib','1'])==0
    report=json.loads(capsys.readouterr().out)
    assert report['state_backend']=='disk' and report['status']=='complete'
    with pytest.raises(ValueError,match='only'):
        run_manifest(manifest,tmp_path/'bad',method='normalization_only',limits=(0,16000),state_backend='disk')
    assert not (tmp_path/'bad').exists()


def test_fit_failure_leaves_unloadable_state(tmp_path):
    raw,mask,cfg,norm,image,tissue=fixture()
    def broken(*bounds):raise RuntimeError('simulated reader failure')
    with pytest.raises(RuntimeError,match='reader failure'):
        fit_disk(raw.shape,broken,tissue,norm,cfg,tmp_path/'s',identity=IDENTITY)
    assert (tmp_path/'s/state.json').exists()
    assert not (tmp_path/'s/COMPLETE.json').exists()
    with pytest.raises(ValueError,match='Incomplete'):DiskTransform(tmp_path/'s')
