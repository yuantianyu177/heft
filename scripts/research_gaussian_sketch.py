"""All-channel Gaussian response codes with query-independent model calls.

Compare raw response power, an unbiased Gaussian-sketch block-energy estimate,
explained response variance, and response subspace overlap. Counterfactual flow
is established prior work; the proposed test concerns the coded estimators.
"""
import argparse,hashlib,json,time
from pathlib import Path
import cv2,numpy as np,torch
from safetensors.torch import load_file,save_file
from torch.nn import functional as F
from research_causal_denoising import native_vae_class
from research_artifacts import response_stem


def decode(response,query_codes):
    # Y[K,T,S,Cout], Zq[K,Q,Cin]. E[Y^T Zq/K] is the corresponding block.
    k,t,s,c=response.shape;cin=query_codes.shape[-1]
    estimate=torch.einsum('kqi,ktsj->qtsij',query_codes,response)/k
    raw=estimate.square().sum((-1,-2))
    total=response.square().sum((0,3))
    corrected=(raw-cin*total[None]/(k*k))/(1+1/k)
    uq=torch.linalg.qr(query_codes.permute(1,0,2),mode='reduced').Q
    projection=torch.einsum('qki,ktsj->qtsij',uq,response)
    explained=projection.square().sum((-1,-2))/total[None].clamp_min(1e-30)
    # Whiten the output response coordinates once per target patch. Threshold
    # 2^-10 is a fixed numerical rank cutoff, not selected using track labels.
    matrix=response.permute(1,2,0,3)
    covariance=matrix.transpose(-1,-2)@matrix
    flat=covariance.reshape(-1,c,c)
    # cuSOLVER's batched syev rejects very large batches on this CUDA build.
    decompositions=[torch.linalg.eigh(chunk.contiguous()) for chunk in flat.split(2048)]
    values=torch.cat([pair[0] for pair in decompositions]).reshape(t,s,c)
    vectors=torch.cat([pair[1] for pair in decompositions]).reshape(t,s,c,c)
    active=values>values.amax(-1,keepdim=True)*(2.**-10)
    factors=vectors*torch.where(active,values.clamp_min(1e-30).rsqrt(),0)[...,None,:]
    whitened=matrix@factors
    overlap=torch.einsum('qki,tskj->qtsij',uq,whitened).square().sum((-1,-2))/active.sum(-1)[None].clamp_min(1)
    return {'raw':raw,'unbiased':corrected,'explained':explained,'subspace':overlap}


def sanity():
    torch.manual_seed(8);n,cin,k=12,3,16
    operator=torch.randn(n,cin,2);truth=operator.square().sum((1,2));estimates=[]
    for _ in range(1000):
        z=torch.randn(k,n,cin);y=torch.einsum('kqi,qij->kj',z,operator)[:,None,None]
        cross=torch.einsum('kqi,ktsj->qtsij',z,y)/k
        estimate=(cross.square().sum((-1,-2))-cin*y.square().sum((0,3))[None]/k**2)/(1+1/k)
        estimates.append(estimate[:,0,0])
    relative=float((torch.stack(estimates).mean(0)-truth).abs().mean()/truth.mean())
    assert relative<.1,relative
    return {'relative_mean_absolute_bias':relative,'draws':1000,'status':'Gaussian block-energy correction verified'}


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('/share/yty/heft'))
    parser.add_argument('--gpu',type=int,default=0);parser.add_argument('--names',default='gold-fish')
    parser.add_argument('--codes',type=int,default=64);parser.add_argument('--epsilon',type=float,default=.0625)
    parser.add_argument('--batch',type=int,default=8);parser.add_argument('--queries',type=int,default=8)
    parser.add_argument('--sigma',type=float,default=.5);parser.add_argument('--sanity',action='store_true')
    args=parser.parse_args();torch.set_num_threads(3);cv2.setNumThreads(1)
    if args.sanity:print(json.dumps(sanity(),indent=2));return
    device=f'cuda:{args.gpu}';torch.cuda.set_device(device)
    from diffusers import WanTransformer3DModel
    path=json.loads((args.root/'models/Wan2.1-T2V-1.3B-Diffusers.json').read_text())['path']
    cls,vae_hash=native_vae_class()
    vae=cls.from_pretrained(path,subfolder='vae',torch_dtype=torch.bfloat16,local_files_only=True).to(device).eval()
    model=WanTransformer3DModel.from_pretrained(path,subfolder='transformer',torch_dtype=torch.bfloat16,local_files_only=True).to(device).eval()
    prompt=load_file(str(args.root/'models/wan_empty_prompt.safetensors'))['prompt_embeds'].to(device)
    mean=torch.tensor(vae.config.latents_mean,device=device).view(1,16,1,1,1)
    std=torch.tensor(vae.config.latents_std,device=device).view(1,16,1,1,1)
    destination=args.root/'results/gaussian_response_sketch_wan';destination.mkdir(exist_ok=True,parents=True)
    for name in args.names.split(','):
        output=response_stem(args.root,'gaussian',name,args.codes,args.epsilon,args.sigma)
        if output.with_suffix('.json').exists():continue
        started=time.monotonic();frames=np.load(args.root/f'data/davis_records/{name}.video.npy',mmap_mode='r')[:25]
        frames=np.stack([cv2.resize(f,(832,480),interpolation=cv2.INTER_LANCZOS4) for f in frames])
        tensor=torch.from_numpy(frames).permute(3,0,1,2)[None].to(device,dtype=torch.bfloat16)/127.5-1
        latent=vae.encode(tensor).latent_dist.mode().float();clean=(latent-mean)/std
        noise=torch.randn(clean.shape,generator=torch.Generator().manual_seed(1273)).to(device)
        noisy=((1-args.sigma)*clean+args.sigma*noise).to(torch.bfloat16)
        def forward(batch):return model(batch,timestep=torch.full((len(batch),),args.sigma*1000,device=device),
            encoder_hidden_states=prompt.expand(len(batch),-1,-1).contiguous(),return_dict=False)[0].float()
        codes=torch.randn((args.codes,16,60,104),generator=torch.Generator().manual_seed(8764)).to(device)
        responses=[]
        for offset in range(0,args.codes,args.batch//2):
            group=codes[offset:offset+args.batch//2];inputs=noisy.expand(2*len(group),-1,-1,-1,-1).clone()
            inputs[:len(group),:,0]=(inputs[:len(group),:,0].float()+args.epsilon*group).to(torch.bfloat16)
            inputs[len(group):,:,0]=(inputs[len(group):,:,0].float()-args.epsilon*group).to(torch.bfloat16)
            prediction=forward(inputs);response=(prediction[:len(group),:,1:]-prediction[len(group):,:,1:])/(2*args.epsilon)
            responses.append(response.permute(0,2,3,4,1).reshape(len(group),6,6240,16))
            print(json.dumps({'video':name,'codes_done':offset+len(group),'codes':args.codes}),flush=True)
        response=torch.cat(responses);code_grid=codes.permute(0,2,3,1).reshape(args.codes,6240,16)
        save_file({'response':response.cpu().contiguous(),'codes':codes.cpu().contiguous()},str(output.with_suffix('.safetensors')))
        # Annotation queries affect decoding only, after every model call is done.
        with np.load(args.root/f'data/davis_records/{name}.annotations.npz') as annotation:
            eligible=np.flatnonzero(~annotation['occluded'][:,0]);rng=np.random.default_rng(int(hashlib.sha256(name.encode()).hexdigest()[:8],16))
            selected=np.sort(rng.choice(eligible,min(args.queries,len(eligible)),replace=False))
            points=annotation['points'][selected][:,[0,4,8,12,16,20,24]].transpose(1,0,2)
            valid=(~annotation['occluded'][selected][:,[0,4,8,12,16,20,24]].T)[1:]
        xy=np.floor(points[0]*[104,60]).astype(int);xy[:,0]=np.clip(xy[:,0],0,103);xy[:,1]=np.clip(xy[:,1],0,59)
        indices=torch.tensor(xy[:,1]*104+xy[:,0],device=device)
        pcks={};saved={}
        for prefix in sorted(set([min(32,args.codes),args.codes])):
            fields=decode(response[:prefix],code_grid[:prefix,indices])
            for label,field in fields.items():
                index=field.argmax(-1).cpu().numpy();prediction=np.stack(((index%104+.5)/104,(index//104+.5)/60),-1).transpose(1,0,2)
                error=np.linalg.norm((prediction-points[1:])*256,axis=-1)
                key=f'k{prefix}_{label}';pcks[key]=float(((error[...,None]<[1,2,4,8,16]).mean(-1)*valid).sum()/valid.sum()*100)
                saved[key+'_energy']=field.cpu().numpy();saved[key+'_prediction']=prediction
        np.savez_compressed(output.with_suffix('.npz'),**saved,points=points,valid=valid,query_indices=selected)
        payload={'video':name,'codes':args.codes,'epsilon':args.epsilon,'sigma':args.sigma,'precision':'bf16',
                 'pck':pcks,'seconds':time.monotonic()-started,'query_dependent_model_calls':0,
                 'source_grid':[60,104],'output_grid':[60,104],'native_vae_hash':vae_hash,
                 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 'novelty_boundary':'Jacobian flow is established prior work; tests query amortization, Gaussian cross-talk correction and response normalization'}
        output.with_suffix('.json').write_text(json.dumps(payload,indent=2));print(json.dumps(payload),flush=True)

if __name__=='__main__':main()
