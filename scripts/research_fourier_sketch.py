"""Probe a denoiser with a continuous coordinate code instead of point markers.

This is an exploratory structured-probing hypothesis, not an established new
tracker. Known Jacobian flow is prior work. Its possible increment is decoding
all source locations from a small, query-independent spatial basis.
"""
import argparse,hashlib,json,math,time
from pathlib import Path
import cv2,numpy as np,torch
from safetensors.torch import load_file,save_file
from research_artifacts import response_stem
from research_causal_denoising import native_vae_class
from research_decode_response import queries,metrics

def code_at(points,frequencies):
    rows=[torch.ones(points.shape[:-1],device=points.device)]
    for f in frequencies:
        for axis in [0,1]:
            angle=points[...,axis]*(2*math.pi*f)
            rows.extend([angle.cos()*math.sqrt(2),angle.sin()*math.sqrt(2)])
    return torch.stack(rows)

@torch.inference_mode()
def readout(root,name,stem,device,directions,frequencies):
    points,valid=queries(root,name,0);saved=load_file(str(stem.with_suffix('.safetensors')),device=device)
    q=torch.tensor(points[0],device=device);c=code_at(q,frequencies)
    result={};predictions={}
    for n in range(1,len(frequencies)+1):
        k=1+4*n;num=[];den=[];phases=[];phase_weights=[]
        for bank in range(directions):
            y=saved[f'response_{bank}'][:k]
            cq=torch.nn.functional.normalize(c[:k],dim=0)
            num.append(torch.einsum('kq,ktsc->qtsc',cq,y).square().sum(-1))
            den.append(y.square().sum((0,3)))
            # Constant-code response fixes the sign of each phase pair.
            ref=y[0];r=(y[1:]*ref[None]).sum(-1).reshape(2*n,2,6,6240)
            amplitude=r.square().sum(1).sqrt().clamp_min(1e-20)
            phase=r/amplitude[:,None]
            paircode=c[1:k].reshape(2*n,2,-1)/math.sqrt(2)
            correlation=torch.einsum('apq,apts->qats',paircode,phase)
            coherence=amplitude/(y[1:].reshape(2*n,2,6,6240,16).square().sum((1,4))*ref.square().sum(-1)[None]).sqrt().clamp_min(1e-20)
            phases.append(correlation);phase_weights.append(coherence)
        fields={'projection':torch.stack(num).sum(0)/torch.stack(den).sum(0)[None].clamp_min(1e-20),
                'phase':torch.stack(phases).mean((0,2)),
                'phase_coherence':sum(p*w[None] for p,w in zip(phases,phase_weights,strict=True))/sum(phase_weights).sum(0)[None].clamp_min(1e-20)}
        # Weighted phase sums over basis pairs after combining independent directions.
        fields['phase_coherence']=sum((p*w[None]).sum(1) for p,w in zip(phases,phase_weights,strict=True))/sum(w.sum(0) for w in phase_weights)[None].clamp_min(1e-20)
        for label,field in fields.items():
            key=f'fmax{frequencies[n-1]}_{label}';result[key],predictions[key]=metrics(field,points,valid,60,104)
    out=stem.with_name(stem.name+f'_decoded_q{len(points[0])}')
    np.savez_compressed(out.with_suffix('.npz'),**predictions,points=points,valid=valid)
    payload={'video':name,'queries':len(points[0]),'pck':result,'frequencies':frequencies,
        'code_count':1+4*len(frequencies),'model_inputs_for_full_basis':directions*2*(1+4*len(frequencies)),
        'query_dependent_model_calls':0,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    out.with_suffix('.json').write_text(json.dumps(payload,indent=2));print(json.dumps(payload),flush=True)

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/share/yty/heft'))
    p.add_argument('--gpu',type=int,default=0);p.add_argument('--names',default='gold-fish')
    p.add_argument('--frequencies',default='1,2,4,8,16');p.add_argument('--directions',type=int,default=2)
    p.add_argument('--epsilon',type=float,default=.25);p.add_argument('--sigma',type=float,default=.5)
    p.add_argument('--batch',type=int,default=8);p.add_argument('--decode-only',action='store_true')
    a=p.parse_args();torch.set_num_threads(3);cv2.setNumThreads(1);device=f'cuda:{a.gpu}';torch.cuda.set_device(device)
    frequencies=list(map(int,a.frequencies.split(',')));count=1+4*len(frequencies)
    if a.decode_only:
        for name in a.names.split(','):readout(a.root,name,response_stem(a.root,'fourier',name,count,a.epsilon,a.sigma,a.directions),device,a.directions,frequencies)
        return
    from diffusers import WanTransformer3DModel
    path=json.loads((a.root/'models/Wan2.1-T2V-1.3B-Diffusers.json').read_text())['path'];cls,vae_hash=native_vae_class()
    vae=cls.from_pretrained(path,subfolder='vae',torch_dtype=torch.bfloat16,local_files_only=True).to(device).eval()
    model=WanTransformer3DModel.from_pretrained(path,subfolder='transformer',torch_dtype=torch.bfloat16,local_files_only=True).to(device).eval()
    state={'capture':False,'keys':None}
    def capture(*,key,**kwargs):
        if state['capture']:state['keys']=key[0,[1,5]].reshape(2,7,30,52,128).cpu().contiguous()
    model.blocks[18].attn1._heft_capture=capture
    prompt=load_file(str(a.root/'models/wan_empty_prompt.safetensors'))['prompt_embeds'].to(device)
    mean=torch.tensor(vae.config.latents_mean,device=device).view(1,16,1,1,1);std=torch.tensor(vae.config.latents_std,device=device).view(1,16,1,1,1)
    yy,xx=torch.meshgrid((torch.arange(60,device=device)+.5)/60,(torch.arange(104,device=device)+.5)/104,indexing='ij')
    codes=code_at(torch.stack([xx,yy],-1),frequencies)
    gram=codes.flatten(1)@codes.flatten(1).T/6240
    assert (gram-torch.eye(count,device=device)).abs().max()<1e-5
    source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for name in a.names.split(','):
        out=response_stem(a.root,'fourier',name,count,a.epsilon,a.sigma,a.directions);out.parent.mkdir(exist_ok=True,parents=True)
        if out.with_suffix('.json').exists():
            readout(a.root,name,out,device,a.directions,frequencies);continue
        start=time.monotonic();frames=np.load(a.root/f'data/davis_records/{name}.video.npy',mmap_mode='r')[:25]
        frames=np.stack([cv2.resize(f,(832,480),interpolation=cv2.INTER_LANCZOS4) for f in frames])
        video=torch.from_numpy(frames).permute(3,0,1,2)[None].to(device,dtype=torch.bfloat16)/127.5-1
        clean=(vae.encode(video).latent_dist.mode().float()-mean)/std
        noise=torch.randn(clean.shape,generator=torch.Generator().manual_seed(1273)).to(device)
        noisy=((1-a.sigma)*clean+a.sigma*noise).to(torch.bfloat16)
        def forward(x):return model(x,timestep=torch.full((len(x),),a.sigma*1000,device=device),encoder_hidden_states=prompt.expand(len(x),-1,-1).contiguous(),return_dict=False)[0].float()
        state['capture']=True;forward(noisy);state['capture']=False
        saved={'baseline_l18_key':state['keys'],'codes':codes.flatten(1).cpu().contiguous()}
        for d in range(a.directions):
            channel=(torch.randint(0,2,(16,),generator=torch.Generator().manual_seed(233+d))*2-1).to(device,dtype=torch.float32)/4
            patches=codes[:,None]*channel[None,:,None,None];rows=[]
            for offset in range(0,count,a.batch//2):
                group=patches[offset:offset+a.batch//2];inputs=noisy.expand(2*len(group),-1,-1,-1,-1).clone()
                inputs[:len(group),:,0]=(inputs[:len(group),:,0].float()+a.epsilon*group).to(torch.bfloat16)
                inputs[len(group):,:,0]=(inputs[len(group):,:,0].float()-a.epsilon*group).to(torch.bfloat16)
                prediction=forward(inputs);r=(prediction[:len(group),:,1:]-prediction[len(group):,:,1:])/(2*a.epsilon)
                rows.append(r.permute(0,2,3,4,1).flatten(2,3))
            saved[f'response_{d}']=torch.cat(rows).cpu().contiguous()
        save_file(saved,str(out.with_suffix('.safetensors')))
        meta={'video':name,'sigma':a.sigma,'epsilon':a.epsilon,'codes':count,'directions':a.directions,'frequencies':frequencies,
            'source_grid':[60,104],'stored_output_grid':[60,104],'stored_output_channels':16,'annotations_opened_during_generation':False,
            'total_model_inputs':2*count*a.directions,'extra_control_model_inputs':1,'seconds':time.monotonic()-start,
            'precision':'bf16','native_vae_hash':vae_hash,'source_sha256':source_hash}
        out.with_suffix('.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
        readout(a.root,name,out,device,a.directions,frequencies)

if __name__=='__main__':main()
