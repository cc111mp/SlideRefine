"""Independent numerical fixtures and WSI-context tests for new method subsets."""
from dataclasses import replace
from itertools import combinations
import json
import numpy as np
import pytest
from scipy.stats import norm as gaussian
from sliderefine.methods.simple_tone_curves import SimpleToneCurve, fit_simple_tone_curve
from sliderefine.methods.simple_tone_curves.reference import constraint_matrix
from sliderefine.methods.hifiem import HiFiEMContrastConfig, enhance_contrast_af
from sliderefine.methods.hifiem._vendor.contrast_reference import adjust_histogram_v_3_02
from sliderefine.wsi.reference_methods import prepare_method, render_region
from sliderefine.wsi.executor import run_manifest
from sliderefine.wsi.tiles import TileManifestSource
from sliderefine.contracts import Region, RegionPixels
from sliderefine.demo import make_demo
from sliderefine.cli import main
from tsclahe.preprocess import Normalization


@pytest.mark.parametrize("case",[1,2,3,4])
def test_already_simple_targets_have_zero_error(case):
    n=21
    if case == 1:
        slopes=np.linspace(.1,2,n-1)
    elif case == 2:
        slopes=np.linspace(2,.1,n-1)
    elif case == 3:
        slopes=np.r_[np.linspace(.1,2,10),np.linspace(2,.1,10)]
    else:
        slopes=np.r_[np.linspace(2,.1,10),np.linspace(.1,2,10)]
    y=np.r_[0,np.cumsum(slopes)];y=.1+.7*y/y[-1]
    c=fit_simple_tone_curve(y)
    np.testing.assert_array_equal(c.samples,y)
    assert c.squared_error == 0 and c.candidates_solved == 0
    assert c.case == case


def test_known_complex_projection():
    c=fit_simple_tone_curve([0,.1,.5,.6,1])
    np.testing.assert_allclose(c.samples,[0,.15,.4,.65,1],rtol=0,atol=1e-8)
    assert c.squared_error == pytest.approx(.015,abs=1e-10)
    assert c.candidates_solved == 6


def _active_face_oracle(t):
    """Small-n independent Euclidean projection by enumerating polytope faces.

    Does not use scipy.optimize or the production constraint builder.
    """
    n=len(t); target=np.array(t[1:-1]); best=np.inf
    signs=[np.ones(n-2),-np.ones(n-2)]
    for cut in range(1,n-2):
        a=np.r_[np.ones(cut),-np.ones(n-2-cut)]; signs.extend([a,-a])
    for sign in signs:
        rows=[]
        for i in range(n-1):
            row=np.zeros(n); row[i:i+2]=[-1,1];rows.append(row)
        for i,s in enumerate(sign):
            row=np.zeros(n);row[i:i+3]=s*np.array([1,-2,1]);rows.append(row)
        M=np.array(rows); A=M[:,1:-1];b=M[:,0]*t[0]+M[:,-1]*t[-1]
        for k in range(n-1):
            for ids in combinations(range(len(rows)),k):
                if not ids:
                    z=target.copy()
                else:
                    active=A[list(ids)]
                    if np.linalg.matrix_rank(active) < k: continue
                    z=target-active.T@np.linalg.solve(active@active.T,active@target+b[list(ids)])
                if np.min(A@z+b) >= -1e-8:
                    best=min(best,float(np.sum((z-target)**2)))
    return best


@pytest.mark.parametrize("seed",range(5))
def test_qp_against_independent_active_face_oracle(seed):
    t=np.r_[0,np.sort(np.random.default_rng(seed).uniform(0,1,3)),1]
    c=fit_simple_tone_curve(t)
    assert c.squared_error == pytest.approx(_active_face_oracle(t),abs=1e-8)


@pytest.mark.parametrize("n",[8,17,40,100])
def test_complex_constraints_and_search_coverage(n):
    x=np.linspace(0,1,n);t=x+.035*np.sin(4*np.pi*x)
    c=fit_simple_tone_curve(t)
    assert np.min(constraint_matrix(n,c.case,c.inflection)@c.samples) >= -1e-7
    assert c.samples[0] == t[0] and c.samples[-1] == t[-1]
    assert c.candidates_solved == 2*n-4
    y=c.apply(np.linspace(0,1,2000))
    assert np.min(np.diff(y)) >= -1e-7


@pytest.mark.parametrize("target",[[0,.3,1], [0,.4,.2,1], [0,.3,np.nan,1],
                                  [-.1,.3,.5,1], [0,.3,.5,1.1], [True]*5])
def test_bad_tone_target(target):
    with pytest.raises(ValueError):fit_simple_tone_curve(target)


def test_solver_failure_not_silently_ignored():
    with pytest.raises(RuntimeError,match="QP failed"):
        fit_simple_tone_curve([0,.1,.5,.6,1],maxiter=1)


@pytest.mark.parametrize("interpolation",["pchip","linear"])
def test_curve_serialization_and_application(tmp_path,interpolation):
    c=fit_simple_tone_curve([.1,.15,.5,.55,.7],interpolation=interpolation)
    p=tmp_path/'curve.json';c.save(p); other=SimpleToneCurve.load(p)
    x=np.linspace(0,1,101).reshape(1,-1)
    np.testing.assert_array_equal(c.apply(x),other.apply(x))
    assert other.apply(np.array([0,1])).tolist() == pytest.approx([.1,.7])
    with pytest.raises(FileExistsError):c.save(p)
    d=c.to_dict();d['squared_error']=10
    with pytest.raises(ValueError):SimpleToneCurve.from_dict(d)
    d=c.to_dict();d['samples'][0]=.2
    with pytest.raises(ValueError):SimpleToneCurve.from_dict(d)
    with pytest.raises(ValueError):c.apply(np.array([2.]))
    with pytest.raises(ValueError):c.apply(np.array([np.nan]))


@pytest.mark.parametrize("masked",[False,True])
@pytest.mark.parametrize("redistribute",["leftmost","left","full"])
@pytest.mark.parametrize("ratio",[0,1.2])
def test_hifiem_contrast_against_pinned_excerpt(masked,redistribute,ratio):
    x=np.random.default_rng(3).uniform(.1,.8,(71,81)).astype(np.float32)
    m=np.ones(x.shape,bool)
    if masked:m[20:32,17:38]=False
    c=HiFiEMContrastConfig(smoothing=9,minmax_size=5,ratio=ratio,redistribute=redistribute)
    original=x.copy()
    got=enhance_contrast_af(x,m,c)
    ref=adjust_histogram_v_3_02(x.copy()*255,drange=255,dtype=np.float32,
         bwmask=m.copy(),smoothing=c.smoothing,method='image',sigma=c.sigma*255,
         offset=c.offset*255,ratio=c.ratio,minmax_size=c.minmax_size,
         clip=[-c.clip,c.clip],redistribute=c.redistribute,hp_sigma=c.hp_sigma*255,
         mean=c.mean*255,save_memory=False)/255
    np.testing.assert_allclose(got[m],ref[m],rtol=0,atol=5e-7)
    np.testing.assert_array_equal(got[~m],x[~m])
    np.testing.assert_array_equal(x,original)
    assert np.isfinite(got).all() and got.dtype == np.float32


def test_hifiem_global_adjustment_formula_without_local_path():
    x=np.linspace(0,1,100).reshape(10,10).astype(np.float32)
    c=HiFiEMContrastConfig(ratio=0,redistribute="full")
    z=(x.astype(np.float64)-c.mean)/c.hp_sigma
    a,b=-c.clip,c.clip; pa,pb=gaussian.pdf([a,b]);ca,cb=gaussian.cdf([a,b])
    q=np.clip(z-a,0,b-a)
    integral=pa*q + .5*(pb-pa)*q*q/(b-a)
    delta=cb-ca-.5*(pa+pb)*(b-a)
    expected=np.clip((np.minimum(gaussian.cdf(z),ca)+integral+
                      np.maximum(gaussian.cdf(z)-cb,0))/(1-delta)*256/255,0,1)
    np.testing.assert_allclose(enhance_contrast_af(x,np.ones(x.shape,bool),c),expected,atol=2e-7,rtol=0)
    assert c.halo == 0


@pytest.mark.parametrize("kwargs",[{"smoothing":4},{"minmax_size":True},{"sigma":0},
                                   {"hp_sigma":0},{"mean":np.nan},{"ratio":-1},
                                   {"variant":"destripe"},{"redistribute":"guess"},{"clip":0}])
def test_hifiem_bad_parameters(kwargs):
    with pytest.raises((ValueError,TypeError)):HiFiEMContrastConfig(**kwargs)


def test_hifiem_empty_mask_and_input_validation():
    x=np.ones((5,7),np.float32)*.3;m=np.zeros(x.shape,bool)
    np.testing.assert_array_equal(enhance_contrast_af(x,m,HiFiEMContrastConfig()),x)
    with pytest.raises(ValueError):enhance_contrast_af(x,m.astype(float),HiFiEMContrastConfig())
    with pytest.raises(ValueError):enhance_contrast_af(x*10,m,HiFiEMContrastConfig())


@pytest.fixture
def tile_source(tmp_path):
    return TileManifestSource(make_demo(tmp_path/'input'))


@pytest.fixture(params=["simple_tone_curves","hifiem"])
def prepared(request):
    if request.param == "simple_tone_curves":
        curve=fit_simple_tone_curve([0,.1,.5,.6,1])
        return prepare_method(request.param,tone_curve=curve)
    return prepare_method(request.param,method_config={"variant":"contrast_af","smoothing":15})


@pytest.mark.parametrize("region",[Region(0,0,31,37),Region(63,75,147,112),Region(350,230,39,27)])
def test_shifted_region_equals_full_reference(tile_source,prepared,region):
    full=Region(0,0,tile_source.shape[1],tile_source.shape[0]); n=Normalization(0,16000,'fixed')
    baseline,ref,_,_=render_region(tile_source,n,prepared,full)
    b,y,_,_=render_region(tile_source,n,prepared,region)
    s=np.s_[region.y:region.y+region.height,region.x:region.x+region.width]
    np.testing.assert_allclose(y,ref[s],rtol=0,atol=2e-7)
    np.testing.assert_array_equal(b,baseline[s])


@pytest.mark.parametrize("chunks",[(51,67),(128,160)])
def test_new_methods_end_to_end(tile_source,prepared,tmp_path,chunks):
    kwargs=({"tone_curve":prepared.operator} if prepared.method=='simple_tone_curves'
            else {"method_config":prepared.operator.to_dict()})
    output=tmp_path/'output'
    report=run_manifest(tile_source.path,output,method=prepared.method,limits=(0,16000),
                        chunk_shape=chunks,**kwargs)
    full=Region(0,0,tile_source.shape[1],tile_source.shape[0]); n=Normalization(0,16000,'fixed')
    _,ref,p,_=render_region(tile_source,n,prepared,full)
    got=TileManifestSource(output/'enhanced/manifest.json').read_region(full)
    np.testing.assert_allclose(got.pixels,ref,rtol=0,atol=2e-7)
    np.testing.assert_array_equal(got.tissue,p.tissue)
    assert report['status']=='complete' and (output/'fitted_state.json').exists()
    with pytest.raises(FileExistsError):run_manifest(tile_source.path,output,method=prepared.method,limits=(0,16000),**kwargs)


def test_gap_context_bypass_and_chunk_consistency(tile_source,tmp_path):
    m=tile_source.manifest;m['tiles'].pop(0);tile_source.path.write_text(json.dumps(m))
    src=TileManifestSource(tile_source.path); n=Normalization(0,16000,'fixed')
    p=prepare_method('hifiem',method_config={"variant":"contrast_af","smoothing":15})
    full=Region(0,0,src.shape[1],src.shape[0]);b,ref,part,count=render_region(src,n,p,full)
    assert count>0
    for region in [Region(110,60,80,70),Region(70,70,80,90),Region(0,0,31,37)]:
        _,y,_,_=render_region(src,n,p,region)
        np.testing.assert_allclose(y,ref[region.y:region.y+region.height,region.x:region.x+region.width],atol=2e-7,rtol=0)
    assert (ref[~part.valid]==0).all()


def test_missing_options_do_not_write(tile_source,tmp_path):
    with pytest.raises(ValueError,match='tone_curve'):
        run_manifest(tile_source.path,tmp_path/'stc',method='simple_tone_curves',limits=(0,16000))
    with pytest.raises(ValueError,match='variant'):
        run_manifest(tile_source.path,tmp_path/'hf',method='hifiem',limits=(0,16000))
    assert not (tmp_path/'stc').exists() and not (tmp_path/'hf').exists()


def test_large_geometry_small_request(prepared):
    class SyntheticSource:
        shape=(100_000,100_000)
        def read_region(self,r):
            assert r.width*r.height < 100_000
            y,x=np.ogrid[r.y:r.y+r.height,r.x:r.x+r.width]
            a=(.4+.05*np.sin(x/11)*np.cos(y/13)).astype(np.float32)
            return RegionPixels(a,np.ones(r.shape,bool),np.ones(r.shape,bool))
    _,out,_,_=render_region(SyntheticSource(),Normalization(0,1,'fixed'),prepared,Region(80_000,90_000,31,37))
    assert out.shape==(37,31)


def test_tone_cli_and_loaded_execution(tmp_path,tile_source,capsys):
    target=tmp_path/'target.json'; target.write_text(json.dumps({'samples':[0,.1,.5,.6,1]}))
    curve=tmp_path/'curve.json'
    assert main(['fit-tone-curve',str(target),str(curve)]) == 0
    assert main(['run',str(tile_source.path),str(tmp_path/'output'),'--method','simple_tone_curves',
                 '--tone-curve',str(curve),'--limits','0','16000']) == 0
    with pytest.raises(SystemExit):main(['fit-tone-curve',str(target),str(curve)])


def test_hifiem_cli(tmp_path,tile_source):
    config=tmp_path/'hf.json';config.write_text(json.dumps({'variant':'contrast_af','smoothing':9}))
    assert main(['run',str(tile_source.path),str(tmp_path/'out'),'--method','hifiem',
                 '--method-config',str(config),'--limits','0','16000'])==0


@pytest.mark.parametrize("n,seed",[(17,5),(17,11),(40,0),(40,14)])
def test_flat_endpoint_sections_stay_in_range(n,seed):
    t=np.sort(np.random.default_rng(seed).choice(np.linspace(0,1,7),size=n))
    t[0]=0;t[-1]=1
    c=fit_simple_tone_curve(t)
    assert min(c.samples)>=0 and max(c.samples)<=1
    assert np.min(constraint_matrix(n,c.case,c.inflection)@c.samples)>=-1e-7
