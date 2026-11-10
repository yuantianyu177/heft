"""Compare readable correspondence with causal contribution to denoising.

Uses native temporal VAE compression (4), normalized training-space latents,
and known rectified-flow noise targets. Each ablation mini-batch includes an
untreated copy of exactly the same noisy input. Only a self-attention head's
output is zeroed; all projection weights and all other heads are unchanged.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import types

import cv2
import numpy as np
import torch
from safetensors.torch import load_file
from torch.nn import functional as F


def native_vae_class():
    """Restore the constructor's temporal flags in an isolated in-memory module.

    The repository hardcodes all temporal downsampling off. We only encode;
    the repository's modified decoding routine is never called here.
    """
    import diffusers.models.autoencoders.autoencoder_kl_wan as module
    source=Path(module.__file__).read_text()
    line='        self.temperal_downsample = [False, False, False]\n'
    assert source.count(line)==1
    native_source=source.replace(line,'')
    name='diffusers.models.autoencoders._heft_native_wan_encoder'
    clone=types.ModuleType(name)
    clone.__package__='diffusers.models.autoencoders'
    clone.__file__=module.__file__
    sys.modules[name]=clone
    exec(compile(native_source,module.__file__,'exec'),clone.__dict__)
    return clone.AutoencoderKLWan, hashlib.sha256(native_source.encode()).hexdigest()


def sample_errors(volume,points):
    b,c,t,h,w=volume.shape
    grid=(points*2-1)[None].expand(b,-1,-1,-1).reshape(b*t,-1,1,2)
    sampled=F.grid_sample(volume.permute(0,2,1,3,4).reshape(b*t,c,h,w),grid,
                          align_corners=False,padding_mode='border')
    return sampled.reshape(b,t,c,-1).permute(0,1,3,2)


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('/share/yty/heft'))
    parser.add_argument('--gpu',type=int,default=7)
    parser.add_argument('--names',default='gold-fish')
    parser.add_argument('--noise-levels',default='0.05,0.5,0.9')
    parser.add_argument('--batch-ablations',type=int,default=4)
    parser.add_argument('--heads',help='Subset such as 18:1,18:5,20:8; omitted evaluates all 360')
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args();torch.set_num_threads(4);cv2.setNumThreads(1)
    device=f'cuda:{args.gpu}';torch.cuda.set_device(device)
    from diffusers import WanTransformer3DModel
    path=json.loads((args.root/'models/Wan2.1-T2V-1.3B-Diffusers.json').read_text())['path']
    vae_class,vae_hash=native_vae_class()
    vae=vae_class.from_pretrained(path,subfolder='vae',torch_dtype=torch.bfloat16,local_files_only=True).to(device).eval()
    assert sum(vae.temperal_downsample)==2
    model=WanTransformer3DModel.from_pretrained(path,subfolder='transformer',torch_dtype=torch.bfloat16,
                                               local_files_only=True).to(device).eval()
    prompt=load_file(str(args.root/'models/wan_empty_prompt.safetensors'))['prompt_embeds'].to(device)
    targets=[(l,h) for l in range(30) for h in range(12)] if not args.heads else [tuple(map(int,x.split(':'))) for x in args.heads.split(',')]
    state={'targets':[],'diagnose':False}
    def make_hook(layer):
        def hook(*,query,key,hidden_states,**kwargs):
            if state['diagnose']:
                heads,t,h,w=12,7,30,52
                feature=key[0].reshape(heads,t,h,w,128).float()
                query_feature=query[0].reshape(heads,t,h,w,128).float()
                grid=(state['points'][0]*2-1)[None,None,:,None,:].expand(heads,1,-1,-1,-1).reshape(heads,-1,1,2)
                descriptors=F.grid_sample(feature[:,0].permute(0,3,1,2),grid,align_corners=False,
                                           padding_mode='border')[:,:, :,0].transpose(1,2)
                query_descriptors=F.grid_sample(query_feature[:,0].permute(0,3,1,2),grid,align_corners=False,
                                                padding_mode='border')[:,:,:,0].transpose(1,2)
                target=feature[:,1:].flatten(2,3)
                valid=state['visible'][None,1:]
                thresholds=torch.tensor([1,2,4,8,16],device=device)
                for label,descriptor,normalize in [('readability',descriptors,True),
                                                   ('readability_qk_cosine',query_descriptors,True),
                                                   ('readability_qk_dot',query_descriptors,False),
                                                   ('readability_kk_dot',descriptors,False)]:
                    source=F.normalize(descriptor,dim=-1) if normalize else descriptor
                    destination=F.normalize(target,dim=-1) if normalize else target
                    score=torch.einsum('hpd,htsd->htps',source,destination)
                    index=score.argmax(-1)
                    prediction=torch.stack(((index%w+.5)/w,(index//w+.5)/h),dim=-1)
                    error=((prediction-state['points'][None,1:])*256).norm(dim=-1)
                    within=(error[...,None]<thresholds).float().mean(-1)
                    score=(within*valid).sum((1,2))/valid.sum()
                    state[label][layer]=score.cpu().numpy()
                state['head_output_energy'][layer]=hidden_states[0].float().square().mean((1,2)).cpu().numpy()
            for batch,(target_layer,head) in enumerate(state['targets'],start=1):
                if target_layer==layer:
                    hidden_states[batch,head].zero_()
        return hook
    for l,block in enumerate(model.blocks):block.attn1._heft_capture=make_hook(l)
    destination=args.root/'results/causal_wan_native'
    destination.mkdir(exist_ok=True,parents=True)
    for name in args.names.split(','):
        raw=np.load(args.root/f'data/davis_records/{name}.video.npy',mmap_mode='r')[:25]
        resized=np.stack([cv2.resize(f,(832,480),interpolation=cv2.INTER_LANCZOS4) for f in raw])
        inputs=torch.from_numpy(resized).permute(3,0,1,2)[None].to(device,dtype=torch.bfloat16)/127.5-1
        latent=vae.encode(inputs).latent_dist.mode().float()
        mean=torch.tensor(vae.config.latents_mean,device=device).view(1,16,1,1,1)
        std=torch.tensor(vae.config.latents_std,device=device).view(1,16,1,1,1)
        clean=(latent-mean)/std
        assert clean.shape==(1,16,7,60,104),clean.shape
        indices=[0,4,8,12,16,20,24]
        with np.load(args.root/f'data/davis_records/{name}.annotations.npz') as ann:
            selected=~ann['occluded'][:,0]
            points=torch.tensor(ann['points'][selected][:,indices].transpose(1,0,2).copy(),device=device)
            visible=torch.tensor((~ann['occluded'][selected][:,indices].T).copy(),device=device)
        state.update(points=points,visible=visible)
        noise=torch.randn(clean.shape,generator=torch.Generator().manual_seed(1273)).to(device)
        for level in map(float,args.noise_levels.split(',')):
            output=destination/f'{name}_sigma{int(round(level*1000)):04d}_n{len(targets):03d}'
            if output.with_suffix('.json').exists() and not args.overwrite:continue
            start=time.monotonic()
            noisy=((1-level)*clean+level*noise).to(torch.bfloat16)
            flow=noise-clean
            state.update(targets=[],diagnose=True,readability=np.zeros((30,12)),head_output_energy=np.zeros((30,12)),
                         readability_qk_cosine=np.zeros((30,12)),readability_qk_dot=np.zeros((30,12)),
                         readability_kk_dot=np.zeros((30,12)))
            baseline=model(noisy,timestep=torch.tensor([level*1000],device=device),encoder_hidden_states=prompt,return_dict=False)[0].float()
            state['diagnose']=False
            base_flow_mse=float((baseline-flow).square().double().mean())
            rows=[]
            for offset in range(0,len(targets),args.batch_ablations):
                group=targets[offset:offset+args.batch_ablations]
                state['targets']=group
                count=len(group)+1
                prediction=model(noisy.expand(count,-1,-1,-1,-1).contiguous(),
                                 timestep=torch.full((count,),level*1000,device=device),
                                 encoder_hidden_states=prompt.expand(count,-1,-1).contiguous(),return_dict=False)[0].float()
                error=noisy.float().expand_as(prediction)-level*prediction-clean
                losses=(prediction-flow).square().double().mean((1,2,3,4))
                samples=sample_errors(error,points)
                roi=(samples.square().mean(-1)*visible).sum((1,2))/visible.sum()
                temporal=(samples[:,1:]-samples[:,:1]).square().mean(-1)
                temporal=(temporal*visible[1:]).sum((1,2))/visible[1:].sum()
                for i,(layer,head) in enumerate(group,start=1):
                    rows.append({'layer':layer,'head':head,
                                 'delta_flow_mse':float(losses[i]-losses[0]),'control_flow_mse':float(losses[0]),
                                 'delta_roi_mse':float(roi[i]-roi[0]),'control_roi_mse':float(roi[0]),
                                 'delta_transport_error':float(temporal[i]-temporal[0]),'control_transport_error':float(temporal[0]),
                                 'readability':float(state['readability'][layer,head]) if layer>=0 else None,
                                 'readability_qk_dot':float(state['readability_qk_dot'][layer,head]) if layer>=0 else None,
                                 'head_output_energy':float(state['head_output_energy'][layer,head]) if layer>=0 else None})
                print(json.dumps({'video':name,'sigma':level,'done':offset+len(group),'total':len(targets)}),flush=True)
            state['targets']=[]
            payload={'video':name,'sigma':level,'ablations':rows,'baseline_flow_mse':base_flow_mse,
                     'readability':state['readability'].tolist(),'head_output_energy':state['head_output_energy'].tolist(),
                     'readability_qk_cosine':state['readability_qk_cosine'].tolist(),
                     'readability_qk_dot':state['readability_qk_dot'].tolist(),
                     'readability_kk_dot':state['readability_kk_dot'].tolist(),
                     'native_vae_source_hash':vae_hash,'latent_shape':list(clean.shape),
                     'protocol':'native VAE temporal compression 4; normalized latents; correct rectified-flow target; batch-matched untreated controls',
                     'seconds':time.monotonic()-start,'peak_gpu_bytes':torch.cuda.max_memory_allocated(device)}
            output.with_suffix('.json').write_text(json.dumps(payload,indent=2))
            print(json.dumps({k:payload[k] for k in ['video','sigma','baseline_flow_mse','seconds','peak_gpu_bytes']}),flush=True)


if __name__=='__main__':main()
