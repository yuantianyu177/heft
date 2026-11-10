"""Query-independent coded finite-response sketch of a video denoiser.

Counterfactual/Jacobian flow is prior work. This test concerns multiplexing:
fixed random spatial codes probe all source patches together. Code correlation
recovers each requested block response, with an explicit cross-talk bias control.
The codes and model calls never depend on the annotation queries.
"""
import argparse,hashlib,json,time
from contextlib import nullcontext
from pathlib import Path
import cv2,numpy as np,torch
from safetensors.torch import load_file,save_file
from torch.nn import functional as F
from research_causal_denoising import native_vae_class
from research_artifacts import response_stem


def decode(response,codes,query_indices,debias):
    # response[K,T,S,C], codes[K,S]. Each input patch receives a unit-norm
    # channel/spatial direction times its independent Rademacher code.
    k=len(response)
    selected=codes[:,query_indices].float()
    estimate=torch.einsum('kq,ktsc->qtsc',selected,response)/k
    power=estimate.square().sum(-1)
    if debias:
        # E ||Y r_q/K||² = ||v_q||² + sum_{r!=q}||v_r||²/K.
        # Subtract E||Y||_F²/K², then divide by 1-1/K.
        background=response.square().sum((0,3))/(k*k)
        power=(power-background[None])/(1-1/k)
    return power


def sanity():
    rng=torch.Generator().manual_seed(3)
    n,k=32,16;transport=torch.randn(n,3,generator=rng)
    samples=[]
    for _ in range(1000):
        codes=torch.randint(0,2,(k,n),generator=rng).float()*2-1
        output=(codes@transport)[:,None,None,:]
        samples.append(decode(output,codes,torch.arange(n),True)[:,0,0])
    average=torch.stack(samples).mean(0);truth=transport.square().sum(-1)
    error=float((average-truth).abs().mean()/truth.mean())
    assert error<.1,error
    return {'relative_mean_absolute_bias':error,'draws':1000,'status':'Monte Carlo bias correction verified'}


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('/share/yty/heft'))
    parser.add_argument('--gpu',type=int,default=0);parser.add_argument('--names',default='gold-fish')
    parser.add_argument('--codes',type=int,default=32);parser.add_argument('--directions',type=int,default=2)
    parser.add_argument('--epsilon',type=float,default=.5);parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--queries',type=int,default=8);parser.add_argument('--sigma',type=float,default=.5)
    parser.add_argument('--sanity',action='store_true')
    parser.add_argument('--precision',choices=['bf16','fp32'],default='bf16')
    args=parser.parse_args();torch.set_num_threads(3);cv2.setNumThreads(1)
    if args.sanity:print(json.dumps(sanity(),indent=2));return
    device=f'cuda:{args.gpu}';torch.cuda.set_device(device)
    from diffusers import WanTransformer3DModel
    path=json.loads((args.root/'models/Wan2.1-T2V-1.3B-Diffusers.json').read_text())['path']
    cls,vae_hash=native_vae_class()
    vae=cls.from_pretrained(path,subfolder='vae',torch_dtype=torch.bfloat16,local_files_only=True).to(device).eval()
    dtype=torch.float32 if args.precision=='fp32' else torch.bfloat16
    model=WanTransformer3DModel.from_pretrained(path,subfolder='transformer',torch_dtype=dtype,local_files_only=True).to(device).eval()
    state={'capture':False,'baseline_keys':None}
    def capture(*,key,**kwargs):
        if state['capture']:state['baseline_keys']=key[0,[1,5]].reshape(2,7,30,52,128).cpu().contiguous()
    model.blocks[18].attn1._heft_capture=capture
    prompt=load_file(str(args.root/'models/wan_empty_prompt.safetensors'))['prompt_embeds'].to(device,dtype=dtype)
    mean=torch.tensor(vae.config.latents_mean,device=device).view(1,16,1,1,1)
    std=torch.tensor(vae.config.latents_std,device=device).view(1,16,1,1,1)
    destination=args.root/'results/response_sketch_wan';destination.mkdir(exist_ok=True,parents=True)
    for name in args.names.split(','):
        output=response_stem(args.root,'rademacher',name,args.codes,args.epsilon,args.sigma,args.directions)
        if args.precision=='fp32':output=output.with_name(output.name+'_fp32')
        if output.with_suffix('.json').exists():continue
        started=time.monotonic();frames=np.load(args.root/f'data/davis_records/{name}.video.npy',mmap_mode='r')[:25]
        frames=np.stack([cv2.resize(f,(832,480),interpolation=cv2.INTER_LANCZOS4) for f in frames])
        tensor=torch.from_numpy(frames).permute(3,0,1,2)[None].to(device,dtype=torch.bfloat16)/127.5-1
        latent=vae.encode(tensor).latent_dist.mode().float();clean=(latent-mean)/std
        noise=torch.randn(clean.shape,generator=torch.Generator().manual_seed(1273)).to(device)
        noisy=((1-args.sigma)*clean+args.sigma*noise).to(torch.bfloat16).to(dtype)
        def forward(batch):
            from torch.nn.attention import sdpa_kernel,SDPBackend
            context=sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION) if dtype==torch.float32 else nullcontext()
            with context:
                return model(batch,timestep=torch.full((len(batch),),args.sigma*1000,device=device),
                    encoder_hidden_states=prompt.expand(len(batch),-1,-1).contiguous(),return_dict=False)[0].float()
        state['capture']=True;forward(noisy);state['capture']=False
        responses=[];all_codes=[]
        for direction in range(args.directions):
            generator=torch.Generator().manual_seed(9347+direction)
            codes=(torch.randint(0,2,(args.codes,30,52),generator=generator)*2-1).to(device,dtype=torch.float32)
            channel=(torch.randint(0,2,(16,2,2),generator=torch.Generator().manual_seed(233+direction))*2-1).to(device,dtype=torch.float32)/8
            patches=torch.einsum('khw,cij->kchiwj',codes,channel).reshape(args.codes,16,60,104)
            rows=[]
            for offset in range(0,args.codes,args.batch//2):
                group=patches[offset:offset+args.batch//2]
                inputs=noisy.expand(2*len(group),-1,-1,-1,-1).clone()
                inputs[:len(group),:,0]=(inputs[:len(group),:,0].float()+args.epsilon*group).to(dtype)
                inputs[len(group):,:,0]=(inputs[len(group):,:,0].float()-args.epsilon*group).to(dtype)
                prediction=forward(inputs)
                response=(prediction[:len(group),:,1:]-prediction[len(group):,:,1:])/(2*args.epsilon)
                response=response.reshape(len(group),16,6,30,2,52,2).permute(0,2,3,5,1,4,6).reshape(len(group),6,1560,64)
                rows.append(response)
            responses.append(torch.cat(rows));all_codes.append(codes.flatten(1))
            print(json.dumps({'video':name,'direction_complete':direction+1,'seconds':time.monotonic()-started}),flush=True)
        # Evaluation annotations are opened only after all universal probes finish.
        with np.load(args.root/f'data/davis_records/{name}.annotations.npz') as annotation:
            eligible=np.flatnonzero(~annotation['occluded'][:,0]);rng=np.random.default_rng(int(hashlib.sha256(name.encode()).hexdigest()[:8],16))
            selected=np.sort(rng.choice(eligible,min(args.queries,len(eligible)),replace=False))
            points=annotation['points'][selected][:,[0,4,8,12,16,20,24]].transpose(1,0,2)
            valid=(~annotation['occluded'][selected][:,[0,4,8,12,16,20,24]].T)[1:]
        xy=np.floor(points[0]*[52,30]).astype(int);xy[:,0]=np.clip(xy[:,0],0,51);xy[:,1]=np.clip(xy[:,1],0,29)
        indices=torch.tensor(xy[:,1]*52+xy[:,0],device=device)
        result={};saved={}
        for prefix in sorted(set([min(16,args.codes),args.codes])):
            for debias in [False,True]:
                energy=torch.stack([decode(response[:prefix],codes[:prefix],indices,debias) for response,codes in zip(responses,all_codes,strict=True)]).mean(0)
                index=energy.argmax(-1).cpu().numpy();prediction=np.stack(((index%52+.5)/52,(index//52+.5)/30),-1).transpose(1,0,2)
                errors=np.linalg.norm((prediction-points[1:])*256,axis=-1)
                pck=float(((errors[...,None]<[1,2,4,8,16]).mean(-1)*valid).sum()/valid.sum()*100)
                label=f'k{prefix}_'+('debiased' if debias else 'raw')
                result[label]=pck;saved[label+'_energy']=energy.cpu().numpy();saved[label+'_prediction']=prediction
        save_file({f'response_{i}':r.cpu().contiguous() for i,r in enumerate(responses)}|
                  {f'codes_{i}':r.cpu().contiguous() for i,r in enumerate(all_codes)}|
                  {'baseline_l18_key':state['baseline_keys']},str(output.with_suffix('.safetensors')))
        np.savez_compressed(output.with_suffix('.npz'),**saved,points=points,valid=valid,query_indices=selected)
        payload={'video':name,'sigma':args.sigma,'epsilon':args.epsilon,'codes':args.codes,'directions':args.directions,
                 'precision':args.precision,'base_noisy_precision':'bf16, then converted to forward dtype','pck':result,'seconds':time.monotonic()-started,
                 'query_dependent_model_calls':0,'total_model_inputs':args.codes*args.directions*2,
                 'extra_control_model_inputs':1,
                 'source_grid':[30,52],'stored_output_grid':[30,52],'stored_output_channels':64,
                 'native_vae_hash':vae_hash,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 'novelty_boundary':'counterfactual/Jacobian flow is prior work; this experiment tests coded query amortization and cross-talk correction'}
        output.with_suffix('.json').write_text(json.dumps(payload,indent=2));print(json.dumps(payload),flush=True)

if __name__=='__main__':main()
