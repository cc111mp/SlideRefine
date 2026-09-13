import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from tsclahe import Config, TissueSNRCLAHE
from sliderefine import Region, available_methods, require_method, MethodUnavailableError
from sliderefine.contracts import regions
from sliderefine.demo import make_demo
from sliderefine.wsi.tiles import TileManifestSource
from sliderefine.wsi.reducers import fit_uint_window
from sliderefine.wsi.executor import run_manifest, estimate_state_bytes
from sliderefine.cli import main

@pytest.fixture
def source_path(tmp_path):
    return make_demo(tmp_path/'input')

@pytest.fixture
def cfg():
    return Config(tile_size=(32,32),bins=64,min_tissue_pixels=16,min_noise_coefficients=4)

@pytest.mark.parametrize('values',[(0,0,0,1),(False,0,1,1),(0,0,1.5,3),(0,0,1,-2)])
def test_bad_region(values):
    with pytest.raises(ValueError): Region(*values)

@pytest.mark.parametrize('shape',[(0,5),(2,False),(2,1.2)])
def test_bad_chunks(shape):
    with pytest.raises(ValueError): list(regions((10,11),shape))

def test_region_halo():
    r=Region(10,20,30,40).expand(3)
    assert r.bounds==(17,63,7,43)
    assert r.shape==(46,36)

@pytest.mark.parametrize('box',[(110,80,80,90),(0,0,389,257),(300,199,89,58)])
def test_reader_crosses_storage_tiles(source_path,box):
    s=TileManifestSource(source_path)
    full=s.read_region(Region(0,0,389,257))
    r=Region(*box); p=s.read_region(r)
    np.testing.assert_array_equal(p.pixels,full.pixels[r.y:r.y+r.height,r.x:r.x+r.width])
    np.testing.assert_array_equal(p.tissue,full.tissue[r.y:r.y+r.height,r.x:r.x+r.width])
    assert p.valid.all()

def test_halo_outside_slide(source_path):
    s=TileManifestSource(source_path)
    p=s.read_region(Region(-5,-7,30,40))
    assert not p.valid[:7].any() and not p.valid[:,:5].any()
    assert p.valid[7:,5:].all()
    assert not p.tissue[~p.valid].any()

def test_cache_bound(source_path):
    s=TileManifestSource(source_path,cache_bytes=80_000)
    for r in regions(s.shape,(64,64)): s.read_region(r)
    assert s._cache_size <= s.cache_bytes

def test_missing_tile_not_dark(source_path):
    m=json.loads(source_path.read_text()); m['tiles'].pop(0)
    source_path.write_text(json.dumps(m))
    s=TileManifestSource(source_path); p=s.read_region(Region(0,0,129,97))
    assert not p.valid[:96,:128].any()
    assert p.valid[-1,-1]
    assert not p.tissue[~p.valid].any()

def test_overlap_fails(source_path):
    m=json.loads(source_path.read_text()); m['tiles'].append(m['tiles'][0])
    source_path.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='Overlapping'): TileManifestSource(source_path)

@pytest.mark.parametrize('field,value',[('axes','CYX'),('level',1),('mask_policy','auto'),('channel_id',''),('intensity_domain','guess')])
def test_ambiguous_metadata_fails(source_path,field,value):
    m=json.loads(source_path.read_text()); m[field]=value
    source_path.write_text(json.dumps(m))
    with pytest.raises(ValueError): TileManifestSource(source_path)

def test_wrong_dtype_not_silently_cast(source_path):
    m=json.loads(source_path.read_text()); m['dtype']='uint8'; source_path.write_text(json.dumps(m))
    s=TileManifestSource(source_path)
    with pytest.raises(ValueError,match='dtype'): s.read_region(Region(0,0,10,10))

def test_region_budget(source_path):
    s=TileManifestSource(source_path,max_region_pixels=100)
    with pytest.raises(ValueError,match='guard'): s.read_region(Region(0,0,11,11))

def test_huge_sparse_geometry_without_full_allocation(source_path):
    m=json.loads(source_path.read_text()); m['shape']=[100_000,100_000]
    source_path.write_text(json.dumps(m))
    s=TileManifestSource(source_path); p=s.read_region(Region(99_900,99_900,25,29))
    assert p.pixels.shape==(29,25) and not p.valid.any()

@pytest.mark.parametrize('chunk',[(31,47),(100,120),(400,500)])
def test_histogram_percentiles_exact(source_path,chunk):
    s=TileManifestSource(source_path)
    a=s.read_region(Region(0,0,389,257))
    expected=np.percentile(a.pixels[a.tissue],[0.5,99.5])
    norm=fit_uint_window(s,chunk_shape=chunk)
    np.testing.assert_allclose([norm.low,norm.high],expected,rtol=0,atol=1e-10)

@pytest.mark.parametrize('name',['hifiem','visual_prior_he','multiscale_redistribution','simple_tone_curves'])
def test_pending_methods_fail_before_writes(tmp_path,name):
    assert len(available_methods())==6
    with pytest.raises(MethodUnavailableError):
        run_manifest('does-not-exist.json',tmp_path/'out',method=name,limits=(0,1))
    assert not (tmp_path/'out').exists()

def test_state_budget(source_path,tmp_path,cfg):
    assert estimate_state_bytes((100000,100000),cfg)>1024**3
    with pytest.raises(MemoryError):
        run_manifest(source_path,tmp_path/'out',limits=(0,16000),config=cfg,max_state_bytes=1)
    assert not (tmp_path/'out').exists()

@pytest.mark.parametrize('chunks',[(35,51),(96,128),(300,400)])
def test_streaming_matches_unchanged_backend(source_path,tmp_path,cfg,chunks):
    source=TileManifestSource(source_path)
    part=source.read_region(Region(0,0,389,257))
    reference=TissueSNRCLAHE(cfg)(part.pixels,mask=part.tissue,limits=(0,16000))
    report=run_manifest(source_path,tmp_path/'out',config=cfg,limits=(0,16000),chunk_shape=chunks)
    result=TileManifestSource(tmp_path/'out/enhanced/manifest.json')
    got=result.read_region(Region(0,0,389,257))
    assert report['status']=='complete'
    np.testing.assert_allclose(got.pixels,reference.enhanced,rtol=0,atol=2e-7)
    np.testing.assert_array_equal(got.tissue,part.tissue)
    # Shifted origin through the output manifest matches the same full reference.
    r=Region(29,37,137,111)
    np.testing.assert_allclose(result.read_region(r).pixels,reference.enhanced[37:148,29:166],rtol=0,atol=2e-7)

def test_normalization_only_and_no_overwrite(source_path,tmp_path):
    out=tmp_path/'out'
    report=run_manifest(source_path,out,method='normalization_only',percentiles=(0.5,99.5),chunk_shape=(100,120))
    assert report['normalization']['source']=='slide_uint_histogram_linear_percentiles'
    a=TileManifestSource(out/'baseline/manifest.json').read_region(Region(0,0,389,257))
    b=TileManifestSource(out/'enhanced/manifest.json').read_region(Region(0,0,389,257))
    np.testing.assert_array_equal(a.pixels,b.pixels)
    with pytest.raises(FileExistsError): run_manifest(source_path,out,limits=(0,16000))

def test_missing_coverage_roundtrip(source_path,tmp_path):
    m=json.loads(source_path.read_text());m['tiles'].pop(0);source_path.write_text(json.dumps(m))
    run_manifest(source_path,tmp_path/'out',method='normalization_only',limits=(500,16000))
    p=TileManifestSource(tmp_path/'out/enhanced/manifest.json').read_region(Region(0,0,129,97))
    assert not p.valid[:96,:128].any()
    assert (p.pixels[~p.valid]==0).all()

def test_sensor_threshold_on_calibrated_fails(source_path,tmp_path,cfg):
    m=json.loads(source_path.read_text());m['intensity_domain']='calibrated';source_path.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='sensor_max'):
        run_manifest(source_path,tmp_path/'out',config=replace(cfg,sensor_max=65535),limits=(0,16000))

def test_cli(tmp_path,capsys):
    assert main(['methods'])==0
    assert 'planned' in capsys.readouterr().out
    assert main(['demo',str(tmp_path/'demo')])==0
    assert main(['inspect',str(tmp_path/'demo/manifest.json')])==0
    assert main(['run',str(tmp_path/'demo/manifest.json'),str(tmp_path/'out'),
                 '--method','normalization_only','--limits','0','16000'])==0
    with pytest.raises(SystemExit) as e:
        main(['run',str(tmp_path/'demo/manifest.json'),str(tmp_path/'bad'),
              '--method','hifiem','--limits','0','16000'])
    assert e.value.code==2
