"""Generate a small, explicitly synthetic tile-coordinate fixture; no downloads."""
from pathlib import Path
import json
import numpy as np
from .contracts import regions
from .wsi.tiles import SCHEMA

def make_demo(directory):
    out = Path(directory)
    if out.exists():
        raise FileExistsError("Demo output must not exist")
    out.mkdir(parents=True)
    y,x = np.mgrid[:257,:389]
    rng = np.random.default_rng(42)
    mask = ((x-194)/180)**2 + ((y-128)/114)**2 < 1
    image = np.clip(1000+mask*(8000+1400*np.sin(x/9)*np.cos(y/11)) +
                    rng.normal(0,40,x.shape),0,65535).astype(np.uint16)
    m = dict(schema=SCHEMA,slide_id="synthetic_slide",shape=list(image.shape),dtype="uint16",
             channel_id="synthetic_AF_0",axes="YX",level=0,intensity_domain="raw",
             mask_policy="supplied",pixel_size_um=[0.5,0.5],tiles=[])
    for i,r in enumerate(regions(image.shape,(96,128))):
        ys,xs = slice(r.y,r.y+r.height),slice(r.x,r.x+r.width)
        name,maskname = f"tile_{i:03d}.npy",f"mask_{i:03d}.npy"
        np.save(out/name,image[ys,xs],allow_pickle=False)
        np.save(out/maskname,mask[ys,xs],allow_pickle=False)
        m["tiles"].append(dict(x=r.x,y=r.y,width=r.width,height=r.height,path=name,mask_path=maskname))
    path = out/"manifest.json"
    path.write_text(json.dumps(m,indent=2),encoding="utf-8")
    return path
