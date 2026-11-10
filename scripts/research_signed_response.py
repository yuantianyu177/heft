"""Test signed covariance-like routing versus unsigned response power.

For rectified-flow v and distinct frames, J_m=-sigma J_v. If a
correspondence transports the same latent material channels, -u^T J_v u
should be positive. This is a restrictive, falsifiable assumption; general
off-diagonal posterior covariance blocks need not have positive trace.
"""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np,torch
from safetensors.torch import load_file
from torch.nn import functional as F
from research_artifacts import response_stem
from research_decode_response import queries,metrics

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/share/yty/heft'));p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--kind',choices=['rademacher','pixel','fourier'],default='pixel')
    p.add_argument('--names',default='gold-fish,horsejump-high');a=p.parse_args();torch.set_num_threads(3)
    device=f'cuda:{a.gpu}';torch.cuda.set_device(device)
    count={'rademacher':32,'pixel':64,'fourier':21}[a.kind];eps=.5 if a.kind=='rademacher' else .25
    for name in a.names.split(','):
        stem=response_stem(a.root,a.kind,name,count,eps);data=load_file(str(stem.with_suffix('.safetensors')),device=device)
        points,valid=queries(a.root,name,0);q=torch.tensor(points[0],device=device);results={};predictions={}
        for prefix in ([16,32,64] if a.kind=='pixel' else ([16,32] if a.kind=='rademacher' else [5,9,13,17,21])):
            ybanks=[];zbanks=[];ubanks=[]
            for bank in [0,1]:
                y=data[f'response_{bank}'][:prefix]
                if a.kind=='rademacher':
                    y=y.reshape(prefix,6,30,52,16,2,2).permute(0,1,2,5,3,6,4).reshape(prefix,6,6240,16)
                    z=data[f'codes_{bank}'][:prefix].reshape(prefix,1,30,52)
                    u=(torch.randint(0,2,(16,2,2),generator=torch.Generator().manual_seed(233+bank))*2-1).to(device,dtype=torch.float32)
                    xy=(q*torch.tensor([104,60],device=device)).floor().long();u=u[:,xy[:,1]%2,xy[:,0]%2].T/4
                else:
                    z=(data[f'codes_{bank}'] if a.kind=='pixel' else data['codes'])[:prefix].reshape(prefix,1,60,104)
                    u=(torch.randint(0,2,(16,),generator=torch.Generator().manual_seed(233+bank))*2-1).to(device,dtype=torch.float32)[None].expand(len(q),-1)/4
                code=F.grid_sample(z,q[None,None].expand(prefix,-1,-1,-1)*2-1,align_corners=False,padding_mode='border')[:,0,0]
                code=F.normalize(code,dim=0);zpower=y.square().sum((0,3))
                ybanks.append(y);zbanks.append((code,zpower));ubanks.append(u)
            fields={label:[] for label in ['signed','signed_normalized','scalar_correlation','unsigned_scalar']}
            for start in range(0,len(q),8):
                values=[];power=[];scalar_power=[]
                for y,(z,zp),u in zip(ybanks,zbanks,ubanks,strict=True):
                    scalar=torch.einsum('ktsc,qc->qkts',y,u[start:start+8])
                    values.append(torch.einsum('kq,qkts->qts',z[:,start:start+8],scalar))
                    power.append(zp);scalar_power.append(scalar.square().sum(1))
                signal=torch.stack(values).sum(0);energy=torch.stack(power).sum(0).sqrt()[None].clamp_min(1e-20)
                fields['signed'].append(-signal);fields['signed_normalized'].append(-signal/energy)
                fields['scalar_correlation'].append(-signal/torch.stack(scalar_power).sum(0).sqrt().clamp_min(1e-20))
                fields['unsigned_scalar'].append(torch.stack(values).square().sum(0)/torch.stack(power).sum(0)[None].clamp_min(1e-20))
            for label,parts in fields.items():
                key=f'k{prefix}_{label}';results[key],predictions[key]=metrics(torch.cat(parts),points,valid,60,104)
        target=stem.with_name(stem.name+'_signed')
        np.savez_compressed(target.with_suffix('.npz'),**predictions,points=points,valid=valid)
        payload={'video':name,'kind':a.kind,'queries':len(q),'pck':results,'model_forward_calls':0,
            'status':'exploratory latent-channel-identity assumption; signed cross-frame covariance is not necessarily point identity',
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        target.with_suffix('.json').write_text(json.dumps(payload,indent=2));print(json.dumps(payload),flush=True)

if __name__=='__main__':main()
