"""Direct matching without EMA/search: same encoder-only delay across four heads."""
import hashlib,json,time
from pathlib import Path
import numpy as np,torch
from safetensors import safe_open
from torch.nn import functional as F
from research_rope import frequency_pairs,pair_channels

def ci(values):
    a=np.asarray(values,float);r=np.random.default_rng(42)
    return [float(a.mean()),*np.quantile(a[r.integers(len(a),size=(10000,len(a)))].mean(1),[.025,.975]).tolist()]

@torch.inference_mode()
def main():
    root=Path('/share/yty/heft');torch.set_num_threads(3);device='cuda:2';torch.cuda.set_device(device);torch.backends.cuda.matmul.allow_tf32=True
    heads=[(15,2),(18,1),(18,5),(20,8)];results=[];began=time.monotonic();source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    destination=root/'results/head_time_probe';destination.mkdir(exist_ok=True)
    for path in sorted((root/'features/wan_native_polyphase_chunk25').glob('*_c000.safetensors')):
        name=path.stem.rsplit('_c',1)[0]
        with np.load(root/f'data/davis_records/{name}.annotations.npz') as a:
            selected=~a['occluded'][:,0];points=a['points'][selected,:25].transpose(1,0,2);valid=(~a['occluded'][selected,:25].T)[1:]
        q=torch.tensor(points[0],device=device)[None,None]*2-1;scores={};arrays={}
        with safe_open(path,framework='pt',device=device) as file:
            for layer,head in heads:
                feature=file.get_tensor(f'l{layer:02d}_h{head:02d}_dense_key').float();n=len(feature)
                pos=(torch.arange(n,device=device)+1.25).clamp(max=n-1);left=pos.floor().long();right=pos.ceil().long();a=(pos-left)[:,None,None,None]
                corrected=feature[left]*(1-a)+feature[right]*a;corrected[0]=feature[0]
                for corrected_flag,x in [(False,feature),(True,corrected)]:
                    for variant,channels in [('full',torch.arange(128,device=device)),('low15',pair_channels(frequency_pairs(.15,1)).to(device))]:
                        maps=x[...,channels].permute(0,3,1,2)
                        source=F.grid_sample(maps[:1],q,align_corners=False,padding_mode='border')[0,:,0].T
                        target=F.interpolate(maps[1:],size=(60,104),mode='bilinear',align_corners=False)
                        sim=torch.einsum('qc,tchw->qthw',F.normalize(source,dim=-1),F.normalize(target,dim=1)).flatten(2)
                        idx=sim.argmax(-1).cpu().numpy();pred=np.stack(((idx%104+.5)/104,(idx//104+.5)/60),-1).transpose(1,0,2)
                        error=np.linalg.norm((pred-points[1:])*256,axis=-1);score=float(((error[...,None]<[1,2,4,8,16]).mean(-1)*valid).sum()/valid.sum()*100)
                        label=f'l{layer}h{head}_{variant}_'+('calibrated' if corrected_flag else 'original');scores[label]=score;arrays[label]=pred
        payload={'video':name,'pck':scores,'source_sha256':source_hash,'delay':1.25,'queries':len(points[0]),
            'protocol':'all24targetframes of first25, all initially-visible queries, direct matching on native8pixel output grid; no search orEMA'}
        (destination/f'{name}.json').write_text(json.dumps(payload,indent=2));np.savez_compressed(destination/f'{name}.npz',**arrays,points=points,valid=valid)
        results.append(payload);print(name,flush=True)
    summary={'videos':len(results),'delay':1.25,'heads':{},'seconds':time.monotonic()-began}
    for layer,head in heads:
        for variant in ['full','low15']:
            key=f'l{layer}h{head}_{variant}';before=[r['pck'][key+'_original'] for r in results];after=[r['pck'][key+'_calibrated'] for r in results]
            summary['heads'][key]={'original':float(np.mean(before)),'calibrated':float(np.mean(after)),'gain_video_ci95':ci(np.array(after)-before)}
    (root/'results/head_time_probe_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
