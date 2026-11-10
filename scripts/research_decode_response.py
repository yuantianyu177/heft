"""Decode cached universal response sketches without any model forward calls."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np,torch
from safetensors.torch import load_file
from torch.nn import functional as F
from research_gaussian_sketch import decode as gaussian_decode
from research_artifacts import response_stem


def queries(root,name,count):
    with np.load(root/f'data/davis_records/{name}.annotations.npz') as a:
        eligible=np.flatnonzero(~a['occluded'][:,0])
        if count:
            rng=np.random.default_rng(int(hashlib.sha256(name.encode()).hexdigest()[:8],16))
            eligible=np.sort(rng.choice(eligible,min(count,len(eligible)),replace=False))
        points=a['points'][eligible][:,[0,4,8,12,16,20,24]].transpose(1,0,2)
        valid=(~a['occluded'][eligible][:,[0,4,8,12,16,20,24]].T)[1:]
    return points,valid


def metrics(field,points,valid,h,w):
    index=field.argmax(-1).cpu().numpy()
    prediction=np.stack(((index%w+.5)/w,(index//w+.5)/h),-1).transpose(1,0,2)
    error=np.linalg.norm((prediction-points[1:])*256,axis=-1)
    pck=float(((error[...,None]<[1,2,4,8,16]).mean(-1)*valid).sum()/valid.sum()*100)
    return pck,prediction


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('/share/yty/heft'))
    parser.add_argument('--gpu',type=int,default=2);parser.add_argument('--names',default='gold-fish,horsejump-high')
    parser.add_argument('--kind',choices=['rademacher','pixel','gaussian'],default='rademacher')
    parser.add_argument('--codes',type=int,default=None)
    parser.add_argument('--epsilon',type=float,default=None)
    parser.add_argument('--sigma',type=float,default=.5)
    parser.add_argument('--directions',type=int,default=2)
    parser.add_argument('--precision',choices=['bf16','fp32'],default='bf16')
    parser.add_argument('--queries',type=int,default=8,help='0 evaluates all initially visible queries')
    args=parser.parse_args();torch.set_num_threads(3);device=f'cuda:{args.gpu}';torch.cuda.set_device(device)
    root=args.root
    for name in args.names.split(','):
        points,valid=queries(root,name,args.queries);started=time.monotonic();scores={};predictions={}
        if args.kind in ['rademacher','pixel']:
            count=args.codes or (64 if args.kind=='pixel' else 32)
            epsilon=args.epsilon if args.epsilon is not None else (.25 if args.kind=='pixel' else .5)
            stem=response_stem(root,args.kind,name,count,epsilon,args.sigma,args.directions)
            if args.precision=='fp32':stem=stem.with_name(stem.name+'_fp32')
            metadata=json.loads(stem.with_suffix('.json').read_text())
            source_h,source_w=metadata['source_grid']
            saved=load_file(str(stem.with_suffix('.safetensors')),device=device)
            for prefix in sorted(set([min(16,count),min(32,count),count])):
                banks=[]
                for bank in range(args.directions):
                    y=saved[f'response_{bank}'][:prefix]
                    if metadata['stored_output_channels']==64:
                        y=y.reshape(prefix,6,30,52,16,2,2).permute(0,1,2,5,3,6,4).reshape(prefix,6,6240,16)
                    else:
                        assert y.shape[-2:]==(6240,16),y.shape
                    z=saved[f'codes_{bank}'][:prefix].reshape(prefix,1,source_h,source_w)
                    banks.append((y,z))
                for mode in ['nearest','bilinear']:
                    fields=[]
                    for start in range(0,len(points[0]),8):
                        q=torch.tensor(points[0,start:start+8],device=device)
                        numerator=[];denominator=[]
                        for y,z in banks:
                            if mode=='nearest':
                                xy=(q*torch.tensor([source_w,source_h],device=device)).floor().long()
                                xy[:,0].clamp_(0,source_w-1);xy[:,1].clamp_(0,source_h-1)
                                code=z[:,0,xy[:,1],xy[:,0]]
                            else:
                                grid=q[None,None].expand(prefix,-1,-1,-1)*2-1
                                code=F.grid_sample(z,grid,align_corners=False,padding_mode='border')[:,0,0]
                            code=F.normalize(code,dim=0)
                            response=torch.einsum('kq,ktsc->qtsc',code,y)
                            numerator.append(response.square().sum(-1));denominator.append(y.square().sum((0,3)))
                        fields.append(torch.stack(numerator).sum(0)/torch.stack(denominator).sum(0)[None].clamp_min(1e-30))
                    field=torch.cat(fields);label=f'k{prefix}_{mode}_explained_fine'
                    scores[label],predictions[label]=metrics(field,points,valid,60,104)
        else:
            stem=response_stem(root,'gaussian',name,64,.0625)
            saved=load_file(str(stem.with_suffix('.safetensors')),device=device)
            codes=saved['codes'].permute(0,2,3,1).reshape(64,6240,16);response=saved['response']
            xy=np.floor(points[0]*[104,60]).astype(int);xy[:,0]=np.clip(xy[:,0],0,103);xy[:,1]=np.clip(xy[:,1],0,59)
            index=torch.tensor(xy[:,1]*104+xy[:,0],device=device)
            # Small pilot query set; larger runs should cache output whitening
            # outside query batches to avoid repeated eigendecompositions.
            for prefix in [32,64]:
                fields=gaussian_decode(response[:prefix],codes[:prefix,index])
                for method,field in fields.items():
                    label=f'k{prefix}_{method}'
                    scores[label],predictions[label]=metrics(field,points,valid,60,104)
        output=stem.with_name(stem.name+f'_decoded_q{len(points[0])}')
        np.savez_compressed(output.with_suffix('.npz'),**predictions,points=points,valid=valid)
        payload={'video':name,'kind':args.kind,'queries':len(points[0]),'pck':scores,'model_forward_calls':0,
                 'seconds':time.monotonic()-started,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        output.with_suffix('.json').write_text(json.dumps(payload,indent=2));print(json.dumps(payload),flush=True)

if __name__=='__main__':main()
