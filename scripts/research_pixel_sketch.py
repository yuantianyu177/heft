"""Universal low-channel-rank probes at native latent-pixel resolution.

No annotation is opened by this extractor. A fixed code bank is reused for
every possible query. Prefixes of the saved bank permit cost/accuracy controls.
"""
import argparse,hashlib,json,time
from pathlib import Path
import cv2,numpy as np,torch
from safetensors.torch import load_file,save_file
from research_causal_denoising import native_vae_class
from research_artifacts import response_stem

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/share/yty/heft'))
    p.add_argument('--gpu',type=int,default=0);p.add_argument('--names',default='gold-fish')
    p.add_argument('--codes',type=int,default=64);p.add_argument('--directions',type=int,default=2)
    p.add_argument('--epsilon',type=float,default=.25);p.add_argument('--sigma',type=float,default=.5)
    p.add_argument('--batch',type=int,default=8)
    a=p.parse_args();torch.set_num_threads(3);cv2.setNumThreads(1)
    device=f'cuda:{a.gpu}';torch.cuda.set_device(device)
    from diffusers import WanTransformer3DModel
    path=json.loads((a.root/'models/Wan2.1-T2V-1.3B-Diffusers.json').read_text())['path']
    cls,vae_hash=native_vae_class()
    vae=cls.from_pretrained(path,subfolder='vae',torch_dtype=torch.bfloat16,local_files_only=True).to(device).eval()
    model=WanTransformer3DModel.from_pretrained(path,subfolder='transformer',torch_dtype=torch.bfloat16,local_files_only=True).to(device).eval()
    state={'capture':False,'keys':None}
    def capture(*,key,**kwargs):
        if state['capture']:state['keys']=key[0,[1,5]].reshape(2,7,30,52,128).cpu().contiguous()
    model.blocks[18].attn1._heft_capture=capture
    prompt=load_file(str(a.root/'models/wan_empty_prompt.safetensors'))['prompt_embeds'].to(device)
    mean=torch.tensor(vae.config.latents_mean,device=device).view(1,16,1,1,1)
    std=torch.tensor(vae.config.latents_std,device=device).view(1,16,1,1,1)
    source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for name in a.names.split(','):
        output=response_stem(a.root,'pixel',name,a.codes,a.epsilon,a.sigma,a.directions)
        output.parent.mkdir(exist_ok=True,parents=True)
        if output.with_suffix('.json').exists():continue
        started=time.monotonic();frames=np.load(a.root/f'data/davis_records/{name}.video.npy',mmap_mode='r')[:25]
        frames=np.stack([cv2.resize(f,(832,480),interpolation=cv2.INTER_LANCZOS4) for f in frames])
        video=torch.from_numpy(frames).permute(3,0,1,2)[None].to(device,dtype=torch.bfloat16)/127.5-1
        clean=(vae.encode(video).latent_dist.mode().float()-mean)/std
        noise=torch.randn(clean.shape,generator=torch.Generator().manual_seed(1273)).to(device)
        noisy=((1-a.sigma)*clean+a.sigma*noise).to(torch.bfloat16)
        def forward(x):
            return model(x,timestep=torch.full((len(x),),a.sigma*1000,device=device),
                encoder_hidden_states=prompt.expand(len(x),-1,-1).contiguous(),return_dict=False)[0].float()
        state['capture']=True;forward(noisy);state['capture']=False
        saved={'baseline_l18_key':state['keys']}
        for direction in range(a.directions):
            codes=(torch.randint(0,2,(a.codes,60,104),generator=torch.Generator().manual_seed(9347+direction))*2-1).to(device,dtype=torch.float32)
            channel=(torch.randint(0,2,(16,),generator=torch.Generator().manual_seed(233+direction))*2-1).to(device,dtype=torch.float32)/4
            patches=codes[:,None]*channel[None,:,None,None];rows=[]
            for offset in range(0,a.codes,a.batch//2):
                group=patches[offset:offset+a.batch//2]
                inputs=noisy.expand(2*len(group),-1,-1,-1,-1).clone()
                inputs[:len(group),:,0]=(inputs[:len(group),:,0].float()+a.epsilon*group).to(torch.bfloat16)
                inputs[len(group):,:,0]=(inputs[len(group):,:,0].float()-a.epsilon*group).to(torch.bfloat16)
                prediction=forward(inputs)
                response=(prediction[:len(group),:,1:]-prediction[len(group):,:,1:])/(2*a.epsilon)
                rows.append(response.permute(0,2,3,4,1).flatten(2,3))
            saved[f'response_{direction}']=torch.cat(rows).cpu().contiguous()
            saved[f'codes_{direction}']=codes.flatten(1).cpu().contiguous()
            print(json.dumps({'video':name,'direction_complete':direction+1,'seconds':time.monotonic()-started}),flush=True)
        save_file(saved,str(output.with_suffix('.safetensors')))
        metadata={'video':name,'sigma':a.sigma,'epsilon':a.epsilon,'codes':a.codes,'directions':a.directions,
            'precision':'bf16','seconds':time.monotonic()-started,'annotations_opened':False,
            'query_dependent_model_calls':0,'total_model_inputs':2*a.codes*a.directions,'extra_control_model_inputs':1,
            'source_grid':[60,104],'stored_output_grid':[60,104],'stored_output_channels':16,
            'native_vae_hash':vae_hash,'source_sha256':source_hash,
            'noise_seed':1273,'code_seed_base':9347,'channel_seed_base':233,
            'novelty_boundary':'prior work establishes Jacobian flow; this tests query-independent coding and normalized readout'}
        output.with_suffix('.json').write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata),flush=True)

if __name__=='__main__':main()
