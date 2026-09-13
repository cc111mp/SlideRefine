from dataclasses import replace
import json
import csv
import numpy as np
import pytest
import tifffile

torch=pytest.importorskip('torch')
from tsclahe import Config,TissueSNRCLAHE,Normalization
from tsclahe.core import TileGrid
from tsclahe.torch_backend import (TileController,TorchPredictor,prepare_tensors,forward_prepared,
    save_checkpoint,load_predictor,clipped_luts_torch,apply_luts_torch)
from tsclahe.train import Record,load_sample,main
from tsclahe.version import PREPROCESS_VERSION
from tsclahe.streaming import fit_streaming,apply_streaming


def policy(limits):
    return {'preprocessing':{'version':PREPROCESS_VERSION,'limits':limits,
            'mask_policy':'explicit_binary','calibration_policy':'precorrected_or_none'}}


def test_torch_rejected_hole_exact_and_cpu_parity(config):
    y,x=np.mgrid[:96,:96]
    a=(.3+.02*np.sin(x/4)*np.cos(y/6)).astype(np.float32)
    a[32:64,32:64]=.3
    mask=np.ones_like(a,bool)
    initial=TissueSNRCLAHE(config)(a,mask=mask,limits=(0,1))
    model=TileController(config)
    prepared=prepare_tensors(initial.normalized,mask,initial.transform.statistics,config)
    out,_,_=forward_prepared(model,prepared,initial.transform.grid)
    got=out.detach().numpy()[0,0]
    pred=TorchPredictor(model,metadata=policy([0,1]))
    expected=TissueSNRCLAHE(config,predictor=pred)(a,mask=mask)
    np.testing.assert_array_equal(got[32:64,32:64],a[32:64,32:64])
    np.testing.assert_allclose(got,expected.enhanced,atol=6e-7)


def test_renderer_gradcheck_away_from_switching_points():
    cfg=Config(tile_size=(4,4),bins=16,reliability_feather=2)
    grid=TileGrid((8,12),cfg.tile_size)
    gen=torch.Generator().manual_seed(123)
    hist=torch.rand(1,2,3,16,generator=gen,dtype=torch.float64)**4
    x=torch.rand(1,1,8,12,generator=gen,dtype=torch.float64)*.6+.2
    mask=torch.ones_like(x)
    c=torch.full((1,2,3),2.137,dtype=torch.float64,requires_grad=True)
    strengths=torch.full_like(c,.1)
    assert torch.autograd.gradcheck(lambda clips: apply_luts_torch(x,mask,clipped_luts_torch(hist,clips),
                         strengths,grid,cfg),(c,),eps=1e-6,atol=1e-5)


def test_fixed_checkpoint_contract_enforced_in_python_and_streaming(tmp_path,scene,config):
    a,mask=scene
    model=TileController(config)
    path=tmp_path/'checkpoint.pt';save_checkpoint(path,model,metadata=policy([0,1]))
    pred=load_predictor(path)
    infer=TissueSNRCLAHE(config,predictor=pred)
    auto=infer(a,mask=mask)
    fixed=infer(a,mask=mask,limits=(0,1))
    np.testing.assert_array_equal(auto.enhanced,fixed.enhanced)
    assert len(pred.checkpoint_sha256)==64
    with pytest.raises(ValueError,match='limits differ'):
        infer(a,mask=mask,limits=(0,2))
    with pytest.raises(ValueError,match='explicit tissue'):
        infer(a)
    with pytest.raises(ValueError,match='pre-corrected'):
        infer(a,mask=mask,dark=0)
    reader=lambda y0,y1,x0,x1:a[y0:y1,x0:x1]
    masks=lambda y0,y1,x0,x1:mask[y0:y1,x0:x1]
    with pytest.raises(ValueError,match='normalization differs'):
        fit_streaming(a.shape,reader,masks,Normalization(0,2,'fixed'),config,predictor=pred)
    t=fit_streaming(a.shape,reader,masks,Normalization(0,1,'fixed'),config,predictor=pred)
    with pytest.raises(ValueError,match='normalization differs'):
        list(apply_streaming(t,reader,masks,Normalization(0,2,'fixed')))


def test_percentile_checkpoint_rejects_fixed_override(scene,config):
    a,mask=scene
    pred=TorchPredictor(TileController(config),metadata=policy(None))
    with pytest.raises(ValueError,match='percentile'):
        TissueSNRCLAHE(config,predictor=pred)(a,mask=mask,limits=(0,1))


def test_training_deployment_inputs_identical_and_raw_targets_shared(tmp_path,scene,config):
    a,mask=scene
    raw=np.rint(a*10000).astype(np.uint16)
    tifffile.imwrite(tmp_path/'input.tif',raw)
    tifffile.imwrite(tmp_path/'target.tif',raw)
    tifffile.imwrite(tmp_path/'mask.tif',mask) # Boolean training mask regression.
    rec=Record(tmp_path/'input.tif',tmp_path/'target.tif',tmp_path/'mask.tif','slide1')
    tensors,grid=load_sample(rec,config,limits=(0,10000),target_space='raw')
    pred=TorchPredictor(TileController(config),metadata=policy([0,10000]))
    deployed=TissueSNRCLAHE(config,predictor=pred)(raw,mask=mask)
    np.testing.assert_array_equal(tensors['image'].numpy()[0,0],deployed.normalized)
    np.testing.assert_array_equal(tensors['target'].numpy()[0,0],deployed.normalized)
    with pytest.raises(ValueError,match='shared'):
        load_sample(rec,config,target_space='raw')


def test_all_inactive_training_stops_without_checkpoint(tmp_path,config):
    # Constant images -> every tile invalid. A finite loss must not imply training.
    cfg=tmp_path/'config.json';cfg.write_text(json.dumps(config.to_dict()))
    for split,value in [('train',.2),('val',.25)]:
        for name,array in [('input',np.full((64,64),value,np.float32)),
                           ('target',np.full((64,64),value+.1,np.float32)),
                           ('mask',np.ones((64,64),bool))]:
            tifffile.imwrite(tmp_path/f'{split}_{name}.tif',array)
        with (tmp_path/f'{split}.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['input','target','mask','group_id'])
            writer.writerow([f'{split}_input.tif',f'{split}_target.tif',f'{split}_mask.tif',split])
    out=tmp_path/'run'
    result=main(['--train',str(tmp_path/'train.csv'),'--val',str(tmp_path/'val.csv'),
                 '--config',str(cfg),'--limits','0','1','--epochs','1','--output-dir',str(out)])
    assert result==2
    assert not (out/'best.pt').exists() and not (out/'last.pt').exists()


@pytest.mark.parametrize('defect',['old_schema','nan_weights','feature_names'])
def test_checkpoint_integrity_rejection(tmp_path,config,defect):
    path=tmp_path/'model.pt';save_checkpoint(path,TileController(config),metadata=policy([0,1]))
    data=torch.load(path,weights_only=True)
    if defect=='old_schema': data['schema_version']=1
    elif defect=='feature_names': data['feature_names']=['wrong']
    else:
        key=next(iter(data['state_dict']))
        data['state_dict'][key].view(-1)[0]=float('nan')
    torch.save(data,path)
    with pytest.raises(ValueError): load_predictor(path)


def test_sensor_saturation_streaming_matches_full(config):
    cfg=replace(config,sensor_max=4000)
    y,x=np.mgrid[:96,:96]
    raw=(1000+100*np.sin(x/4)*np.cos(y/5)).astype(np.uint16);raw[32:64,32:64]=4095
    mask=np.ones(raw.shape,bool);norm=Normalization(0,4095,'fixed')
    read=lambda y0,y1,x0,x1:raw[y0:y1,x0:x1]
    masks=lambda y0,y1,x0,x1:mask[y0:y1,x0:x1]
    saturated=lambda y0,y1,x0,x1:raw[y0:y1,x0:x1]>=4000
    with pytest.raises(ValueError,match='saturation_reader'):
        fit_streaming(raw.shape,read,masks,norm,cfg)
    fitted=fit_streaming(raw.shape,read,masks,norm,cfg,saturation_reader=saturated)
    output=np.empty(raw.shape,np.float32)
    for y0,x0,block in apply_streaming(fitted,read,masks,norm,chunk_size=(29,31)):
        output[y0:y0+block.shape[0],x0:x0+block.shape[1]]=block
    expected=TissueSNRCLAHE(cfg)(raw,mask=mask,limits=(0,4095))
    np.testing.assert_array_equal(output,expected.enhanced)
