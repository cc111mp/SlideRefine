"""Run in an isolated interpreter with PYTHONPATH selecting one source tree."""
import json, numpy as np, tempfile
from pathlib import Path
import tifffile
from tsclahe import Config,TissueSNRCLAHE,Normalization
from tsclahe.preprocess import flat_field_correct
from tsclahe.io import read_image

out={}
y,x=np.mgrid[:192,:192]
a=(.3+.02*np.sin(x/4)*np.cos(y/6)).astype(np.float32);a[64:128,64:128]=.3
r=TissueSNRCLAHE(Config(tile_size=(64,64),bins=64))(a,mask_mode='all',limits=(0,1))
d=r.enhanced[64:128,64:128]-a[64:128,64:128]
out['constant_rejected_tile']={'strength':float(r.transform.strengths[1,1]),'output_range':float(np.ptp(r.enhanced[64:128,64:128])), 'max_absolute_change':float(np.abs(d).max()),'changed_fraction':float((d!=0).mean())}
y,x=np.mgrid[:256,:256]
a=(.3+.03*np.sin(x/6)+np.random.default_rng(23).normal(0,.001,x.shape)).astype(np.float32)
r=TissueSNRCLAHE(Config(tile_size=(64,64)))(a,mask_mode='all',limits=(0,1))
out['synthetic_stripe']={'input_std':float(a.std()),'output_std':float(r.enhanced.std()),'output_input_std_ratio':float(r.enhanced.std()/a.std()),'max_absolute_change':float(np.abs(r.enhanced-a).max()),'mean_strength':float(r.transform.strengths.mean()),'median_snr_proxy':float(np.median(r.transform.statistics.snr_proxy))}
a=(2**30+np.arange(32)).astype(np.uint32).reshape(4,8)
out['high_offset_normalization']={'input_unique':int(np.unique(a).size),'output_unique':int(np.unique(Normalization(2**30,2**30+31,'fixed').apply(flat_field_correct(a))).size)}
with tempfile.TemporaryDirectory() as td:
 p=Path(td)/'mask.tif';tifffile.imwrite(p,np.ones((16,16),bool))
 try:
  from tsclahe.io import read_mask
  read_mask(p);out['boolean_mask_reader']='supported'
 except ImportError:
  try: read_image(p);out['boolean_mask_reader']='intensity_reader_supported'
  except ValueError:out['boolean_mask_reader']='rejected'
print(json.dumps(out,indent=2))
