"""Recover sketch sidecars from immutable response caches and run logs.

An earlier basename containing 'eps0.5' was passed to Path.with_suffix, so
decoded sidecars overwrote one another. Raw response tensors are intact. This
script archives legacy sidecars, links the unchanged raw tensors under safe
names, and recomputes the original diagnostic arrays without any model calls.
"""
import hashlib,json,os,shutil
from pathlib import Path
import numpy as np,torch
from safetensors.torch import load_file
from research_artifacts import response_stem
from research_decode_response import queries,metrics
from research_response_sketch import decode

def main():
    root=Path('/share/yty/heft');device='cuda:3';torch.set_num_threads(3)
    archive=root/'results/legacy_decimal_sketch_sidecars';archive.mkdir(exist_ok=True)
    names=['gold-fish','horsejump-high','bike-packing','blackswan','bmx-trees','breakdance']
    logs={'gold-fish':root/'logs/response_sketch_goldfish.log','horsejump-high':root/'logs/response_sketch_horse.log',
          **{name:root/f'logs/sketch_validation_gpu{4+i}_extract.log' for i,name in enumerate(names[2:])}}
    audit=[]
    for name in names:
        legacy=root/f'results/response_sketch_wan/{name}_k32_d2_eps0'
        target=response_stem(root,'rademacher',name,32,.5)
        raw=legacy.with_suffix('.safetensors');fixed=target.with_suffix('.safetensors')
        if not fixed.exists():os.link(raw,fixed)
        for suffix in ['.json','.npz']:
            old=legacy.with_suffix(suffix)
            if old.exists():shutil.copy2(old,archive/old.name)
        metadata=None
        for line in logs[name].read_text().splitlines():
            try:value=json.loads(line)
            except (ValueError,TypeError):continue
            if value.get('video')==name and 'total_model_inputs' in value:metadata=value
        assert metadata is not None,name
        data=load_file(str(fixed),device=device);points,valid=queries(root,name,8)
        xy=np.floor(points[0]*[52,30]).astype(int);xy[:,0]=np.clip(xy[:,0],0,51);xy[:,1]=np.clip(xy[:,1],0,29)
        index=torch.tensor(xy[:,1]*52+xy[:,0],device=device);saved={};pcks={}
        for k in [16,32]:
            for correction in [False,True]:
                energy=torch.stack([decode(data[f'response_{i}'][:k],data[f'codes_{i}'][:k],index,correction) for i in [0,1]]).mean(0)
                label=f'k{k}_'+('debiased' if correction else 'raw')
                pcks[label],prediction=metrics(energy,points,valid,30,52)
                saved[label+'_energy']=energy.cpu().numpy();saved[label+'_prediction']=prediction
        assert all(abs(pcks[k]-metadata['pck'][k])<1e-5 for k in pcks),(name,pcks,metadata['pck'])
        with np.load(root/f'data/davis_records/{name}.annotations.npz') as annotation:
            eligible=np.flatnonzero(~annotation['occluded'][:,0])
            rng=np.random.default_rng(int(hashlib.sha256(name.encode()).hexdigest()[:8],16))
            selected=np.sort(rng.choice(eligible,min(8,len(eligible)),replace=False))
        np.savez_compressed(target.with_suffix('.npz'),**saved,points=points,valid=valid,query_indices=selected)
        metadata.update(source_grid=[30,52],stored_output_grid=[30,52],stored_output_channels=64,
            artifact_recovery='raw response tensors unchanged; original readout arrays reconstructed and exactly match logged PCK',
            legacy_basename=str(legacy))
        target.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
        audit.append({'video':name,'raw_tensor_same_inode':raw.stat().st_ino==fixed.stat().st_ino,
                      'logged_pck_exactly_reproduced':True,'model_forward_calls':0,'new_stem':str(target)})
        print(name,'raw PCK reproduced exactly',flush=True)
    for name in names[:2]:
        legacy=root/f'results/gaussian_response_sketch_wan/{name}_k64_eps0'
        target=response_stem(root,'gaussian',name,64,.0625)
        if not target.with_suffix('.safetensors').exists():os.link(legacy.with_suffix('.safetensors'),target.with_suffix('.safetensors'))
        for suffix in ['.json','.npz']:
            if legacy.with_suffix(suffix).exists():shutil.copy2(legacy.with_suffix(suffix),archive/legacy.with_suffix(suffix).name)
    (root/'results/sketch_artifact_recovery_audit.json').write_text(json.dumps(audit,indent=2))

if __name__=='__main__':main()
