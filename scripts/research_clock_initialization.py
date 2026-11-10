"""Separate query and target clock correction, with a fixed eight-frame horizon.

No EMA, search mask, or visibility estimator. Every target in the horizon must
be visible. Groups differ in query initialization: frame zero, first appearance,
or a deliberately later query on an already continuously visible trajectory.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np,torch
from safetensors import safe_open
from torch.nn import functional as F

def ci(x):
    a=np.asarray(x,float);r=np.random.default_rng(42)
    return [float(a.mean()),*np.quantile(a[r.integers(len(a),size=(10000,len(a)))].mean(1),[.025,.975]).tolist()]

@torch.inference_mode()
def main():
    root=Path('/share/yty/heft');device='cuda:0';torch.cuda.set_device(device);torch.set_num_threads(3);torch.backends.cuda.matmul.allow_tf32=True
    source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();rows=[];began=time.monotonic()
    design={'head':[18,5],'delay':1.25,'horizon':8,'direct_matching':'native8pixel output grid, no EMA, search or occlusion inference',
        'groups':{'frame0':'first query at0 and visible0..8','first_appearance':'official first-visible query after0 and continuously visible for8 subsequent frames',
                  'continuous_later':'query at8, visible6..16; no first-appearance event'},
        'factorial':'uncorrected/corrected query descriptor crossed with uncorrected/corrected target features',
        'hypothesis':'a pure common delay can cancel for continuous constant motion, but firstframe and firstappearance initialization violate that simplification'}
    (root/'results/clock_initialization_design.json').write_text(json.dumps(design,indent=2))
    dest=root/'results/clock_initialization';dest.mkdir(exist_ok=True)
    for path in sorted((root/'features/wan_native_polyphase_chunk25').glob('*_c000.safetensors')):
        name=path.stem.rsplit('_c',1)[0]
        with np.load(root/f'data/davis_records/{name}.annotations.npz') as ann:
            points=ann['points'];visible=~ann['occluded'];first=visible.argmax(1);nt=points.shape[1]
        original=[];calibrated=[]
        for chunk in sorted(path.parent.glob(name+'_c*.safetensors')):
            meta=json.loads(chunk.with_suffix('.json').read_text());count=meta['valid_frames']
            with safe_open(chunk,framework='pt',device=device) as f:feature=f.get_tensor('l18_h05_dense_key')[:count].float()
            pos=(torch.arange(count,device=device)+1.25).clamp(max=count-1);lo=pos.floor().long();hi=pos.ceil().long();alpha=(pos-lo)[:,None,None,None]
            corrected=feature[lo]*(1-alpha)+feature[hi]*alpha;corrected[0]=feature[0]
            original.append(feature);calibrated.append(corrected)
        coarse=[torch.cat(m).permute(0,3,1,2) for m in [original,calibrated]]
        target=[F.normalize(F.interpolate(m,size=(60,104),mode='bilinear',align_corners=False),dim=1) for m in coarse]
        for group in ['frame0','first_appearance','continuous_later']:
            queries=[]
            for point in range(len(points)):
                q=8 if group=='continuous_later' else int(first[point])
                if group=='frame0' and q!=0:continue
                if group=='first_appearance' and q==0:continue
                if q+8>=nt or not visible[point,q:q+9].all():continue
                if group=='continuous_later' and not visible[point,6:17].all():continue
                queries.append((point,q))
            if not queries:continue
            scores={};lags={}
            for source_clock in [0,1]:
                for target_clock in [0,1]:
                    errors=[];temporal=[]
                    for point,q in queries:
                        grid=torch.tensor(points[point,q],device=device)[None,None,None]*2-1
                        source=F.grid_sample(coarse[source_clock][q:q+1],grid,align_corners=False,padding_mode='border')[0,:,0,0]
                        sim=torch.einsum('c,tchw->thw',F.normalize(source,dim=0),target[target_clock][q+1:q+9]).flatten(1)
                        index=sim.argmax(-1).cpu().numpy();prediction=np.stack(((index%104+.5)/104,(index//104+.5)/60),-1)
                        gt=points[point,q+1:q+9];error=(prediction-gt)*256;errors.extend(np.linalg.norm(error,axis=-1).tolist())
                        t=np.arange(q+1,min(q+9,nt-1));velocity=(points[point,t+1]-points[point,t-1])*128;power=(velocity**2).sum(-1)
                        valid=visible[point,t-1]&visible[point,t]&visible[point,t+1]&(power>.25)
                        projected=-(error[:len(t)]*velocity).sum(-1)/np.maximum(power,1e-20);temporal.extend(projected[valid].tolist())
                    key=f'query{source_clock}_target{target_clock}';scores[key]=float((np.array(errors)[:,None]<[1,2,4,8,16]).mean()*100)
                    lags[key]=float(np.median(temporal)) if temporal else None
            row={'video':name,'group':group,'queries':len(queries),'pck':scores,'median_lag':lags};rows.append(row)
        print(name,flush=True)
    summary={'design':design,'groups':{},'seconds':time.monotonic()-began,'source_sha256':source_hash}
    for group in design['groups']:
        selected=[r for r in rows if r['group']==group]
        summary['groups'][group]={'videos':len(selected),'queries':sum(r['queries'] for r in selected),'pck':{},'median_lag':{},'records':selected}
        for key in ['query0_target0','query1_target0','query0_target1','query1_target1']:
            summary['groups'][group]['pck'][key]=ci([r['pck'][key] for r in selected])
            values=[r['median_lag'][key] for r in selected if r['median_lag'][key] is not None]
            summary['groups'][group]['median_lag'][key]=ci(values) if values else None
    (root/'results/clock_initialization_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:{x:y for x,y in v.items() if x!='records'} for k,v in summary['groups'].items()},indent=2))

if __name__=='__main__':main()
