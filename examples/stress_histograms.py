"""Numerical stress check; synthetic distributions, not an image-quality benchmark."""
import argparse
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
from tsclahe import Config
from tsclahe.core import clipped_luts, limit_strengths


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    rng=np.random.default_rng(1701)
    rows=[]
    for bins in (32,64,128,256,512,1024,4096):
        hist=rng.dirichlet(np.full(bins,.15),size=(4,8))
        clips=rng.uniform(1,8,(4,8))
        luts=clipped_luts(hist,clips)
        mass=np.diff(luts.astype(np.float64),axis=-1)
        cfg=Config(bins=bins,clip_max=8)
        strengths=limit_strengths(luts,np.full((4,8),.65),cfg)
        gain=1+strengths*(bins*mass.max(-1)-1)
        row={'bins':bins,'histograms':32,'maximum_mass_error':float(np.abs(mass.sum(-1)-1).max()),
             'maximum_cap_excess':float(np.maximum(mass-clips[...,None]/bins,0).max()),
             'minimum_bin_mass':float(mass.min()),'maximum_fixed_context_gain':float(gain.max())}
        assert row['minimum_bin_mass']>=0
        assert row['maximum_mass_error']<1e-6 and row['maximum_cap_excess']<2e-7
        assert row['maximum_fixed_context_gain']<=2.00001
        rows.append(row)
    result={'synthetic_only':True,'total_histograms':sum(r['histograms'] for r in rows),'results':rows}
    text=json.dumps(result,indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text+'\n',encoding='utf-8')
    print(text)

if __name__=='__main__':
    main()
