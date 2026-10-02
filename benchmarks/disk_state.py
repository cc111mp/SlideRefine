"""Synthetic RSS measurement in a fresh process per mode/size; no full image."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from tsclahe import Config
from tsclahe.preprocess import Normalization
from tsclahe.streaming import fit_streaming
from sliderefine.wsi.state_store import fit_disk


def image(y0,y1,x0,x1):
    y=np.arange(y0,y1)[:,None]; x=np.arange(x0,x1)[None,:]
    return (1000+250*np.sin(x*.6)*np.cos(y*.4)+100*np.sin(x*.07+y*.03)).astype(np.uint16)


def mask(y0,y1,x0,x1):
    y=np.arange(y0,y1)[:,None]; x=np.arange(x0,x1)[None,:]
    return ((x//64+y//48)%7!=0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['memory','disk'],required=True)
    p.add_argument('--size',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    cfg=Config(tile_size=(16,16),bins=256,min_tissue_pixels=8,min_noise_coefficients=2,
               snr_low=0,snr_high=1,contrast_target=.9,artifact_policy='warn',apply_block_rows=64)
    norm=Normalization(0,2400,'fixed');shape=(a.size,a.size)
    start=time.monotonic(); initial=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if a.mode=='memory': t=fit_streaming(shape,image,mask,norm,cfg)
    else: t=fit_disk(shape,image,mask,norm,cfg,a.output/'state',
                    identity={'procedural_source':'disk_state_benchmark_v1','mask':'mod7','model':None},
                    block_shape=(8,8),cache_bytes=1024**2)
    fit_seconds=time.monotonic()-start
    h=hashlib.sha256();active=0;count=0; maxchange=0.
    for y in range(0,a.size,128):
        for x in range(0,a.size,128):
            b=(y,min(y+128,a.size),x,min(x+128,a.size))
            baseline=norm.apply(image(*b)); v=t.apply_region(baseline,mask(*b),origin=(y,x))
            h.update(v.tobytes());active+=int(np.count_nonzero(v!=baseline));count+=v.size
            maxchange=max(maxchange,float(np.max(np.abs(v-baseline))))
    result=dict(mode=a.mode,shape=list(shape),analysis_grid=list(t.grid.grid_shape),bins=cfg.bins,
                fit_seconds=fit_seconds,total_seconds=time.monotonic()-start,
                process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                initial_peak_rss_mib=initial/1024,output_sha256=h.hexdigest(),
                pixels=count,changed_pixels=active,max_absolute_change=maxchange,
                state_cache_peak_bytes=getattr(t,'peak_cache_bytes',None),
                state_disk_bytes=sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file()),
                scope='Linux fresh-process synthetic procedural image; no native I/O; not gigapixel')
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
