"""Transfer the frozen encoder clock to a different diffusion transformer.

Uses the same byte-identical Wan VAE checkpoint with native compression4.
Cosmos rotary phases retain their own spatial NTK and fps conventions.
No annotations are opened, no new delay or head is selected.
"""
import argparse,hashlib,json,time
from pathlib import Path
import cv2,numpy as np,torch
from safetensors.torch import load_file,save_file
from torch.nn import functional as F
from research_causal_denoising import native_vae_class
from research_cosmos_rope import cosmos_angles,adjacent_storage

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/share/yty/heft'));p.add_argument('--gpu',type=int,default=2)
    p.add_argument('--names',default='gold-fish');p.add_argument('--chunked',action='store_true');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args();torch.set_num_threads(3);cv2.setNumThreads(1);device=f'cuda:{a.gpu}';torch.cuda.set_device(device)
    from diffusers import CosmosTransformer3DModel,FlowMatchEulerDiscreteScheduler
    path=json.loads((a.root/'models/Cosmos-Predict2-2B-Video2World.json').read_text())['path'];cls,vae_hash=native_vae_class()
    vae=cls.from_pretrained(path,subfolder='vae',torch_dtype=torch.bfloat16,local_files_only=True).to(device).eval()
    model=CosmosTransformer3DModel.from_pretrained(path,subfolder='transformer',torch_dtype=torch.bfloat16,local_files_only=True).to(device).eval()
    assert model.config.extra_pos_embed_type is None
    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(path,subfolder='scheduler',local_files_only=True)
    scheduler.set_timesteps(sigmas=torch.linspace(0,1,35,dtype=torch.float64),device=device)
    sigma=float(scheduler.sigmas[34]);step=scheduler.timesteps[34];timestep_value=sigma/(sigma+1)
    prompt=load_file(str(a.root/'models/cosmos_empty_prompt.safetensors'))['prompt_embeds'].to(device)
    mean=torch.tensor(vae.config.latents_mean,device=device).view(1,16,1,1,1);std=torch.tensor(vae.config.latents_std,device=device).view(1,16,1,1,1)
    source_indices=[[0,4,8,12,16,20,24],[0,1,5,9,13,17,21],[0,2,6,10,14,18,22],[0,3,7,11,15,19,23]]
    native_angles=cosmos_angles(7,44,80,device=device);frequency=native_angles[1,0,0,:22]/1.5
    head_ids=[1,5,6];state={}
    class Captured(Exception):pass
    def capture(*,key,**kwargs):
        state['key']=adjacent_storage(key[0,head_ids].reshape(3,7,44,80,128)).cpu().contiguous();raise Captured
    model.transformer_blocks[18].attn1._heft_capture=capture
    source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();suffix='chunk25' if a.chunked else 't25'
    manifest=json.loads((a.root/'data/davis_records/manifest.json').read_text())
    if a.names:manifest=[r for r in manifest if r['name'] in a.names.split(',')]
    if a.chunked:manifest=[dict(r,start_frame=s,valid_frames=min(25,r['frames']-s),storage_name=f'{r["name"]}_c{s//25:03d}') for r in manifest for s in range(0,r['frames'],25)]
    for record in manifest[a.shard::a.shards]:
        name=record['name'];storage=record.get('storage_name',name);start=record.get('start_frame',0)
        paths={mode:a.root/f'features/cosmos_{mode}_{suffix}/{storage}.safetensors' for mode in ['native_polyphase','native_interpolate']}
        if all(p.with_suffix('.json').exists() for p in paths.values()):continue
        began=time.monotonic();raw=np.load(a.root/f'data/davis_records/{name}.video.npy',mmap_mode='r');frames=raw[start:start+25]
        if len(frames)<25:frames=np.concatenate((frames,np.repeat(frames[-1:],25-len(frames),axis=0)))
        x=torch.from_numpy(np.stack([cv2.resize(f,(1280,704),interpolation=cv2.INTER_LANCZOS4) for f in frames])).permute(3,0,1,2)[None].to(device,dtype=torch.bfloat16)/127.5-1
        noise=torch.randn((1,16,25,88,160),generator=torch.Generator().manual_seed(42)).to(device)
        keys=[];anchor=None;anchor_difference=0.
        for phase,indices in enumerate(source_indices):
            before=0 if phase==0 else 4-phase;padded=torch.cat((x[:,:,:1].expand(-1,-1,before,-1,-1),x),2)
            padded=torch.cat((padded,padded[:,:,-1:].expand(-1,-1,29-padded.shape[2],-1,-1)),2)
            clean=(vae.encode(padded).latent_dist.mode()[:,:,:7].float()-mean)/std
            if anchor is None:anchor=clean[:,:,:1].clone()
            else:anchor_difference=max(anchor_difference,float((anchor-clean[:,:,:1]).abs().max()))
            noisy=scheduler.scale_noise(clean,step[None],noise[:,:,indices]).to(torch.bfloat16)
            angles=native_angles.clone();angles[...,:22]=(torch.tensor(indices,device=device,dtype=torch.float32)/4*1.5)[:,None,None,None]*frequency
            whole=torch.cat((angles,angles),-1).reshape(-1,128);rotary=(whole.cos(),whole.sin())
            model.rope.forward=lambda hidden_states,fps=None:rotary
            try:
                model(noisy,timestep=torch.full((1,1,7,1,1),timestep_value,device=device,dtype=torch.bfloat16),encoder_hidden_states=prompt,
                    fps=16,condition_mask=torch.zeros((1,1,7,88,160),device=device,dtype=torch.bfloat16),
                    padding_mask=torch.zeros((1,1,704,1280),device=device,dtype=torch.bfloat16),return_dict=False)
            except Captured:pass
            keys.append(state.pop('key'))
        assert anchor_difference==0.,anchor_difference
        dense=torch.empty((3,25,44,80,128),dtype=torch.bfloat16)
        for phase,indices in enumerate(source_indices):dense[:,indices]=keys[phase]
        dense[:,0]=torch.stack([k[:,0].float() for k in keys]).mean(0).to(torch.bfloat16)
        interpolated=F.interpolate(keys[0].float().permute(0,4,1,2,3),size=(25,44,80),mode='trilinear',align_corners=True).permute(0,2,3,4,1).to(torch.bfloat16)
        meta={'video':name,'storage_name':storage,'start_frame':start,'valid_frames':record.get('valid_frames',25),'frames':25,
            'backbone':'cosmos2-native','input_size':[704,1280],'feature_size':[44,80],'source_size':list(raw.shape[1:3]),
            'dense_heads':{'18':head_ids},'source_frame_indices':source_indices,'rope_temporal_scale':.25,
            'storage_pairing':'adjacent','model_pairing':'split_half','fps':16,'noise_sigma':sigma,'noise_timestep':timestep_value,
            'posterior':'mode','latent_normalization':'pretrained mean/std, FP32 noise, same original-frame noise across phases',
            'native_vae_source_hash':vae_hash,'anchor_latent_max_difference':anchor_difference,'extractor_source_sha256':source_hash,
            'total_seconds':time.monotonic()-began,'head_selection':'18/6 fixed from pre-existing extractor profile;1/5 secondary probes; no native result tuning'}
        for mode,value in [('native_polyphase',dense),('native_interpolate',interpolated)]:
            target=paths[mode];target.parent.mkdir(parents=True,exist_ok=True)
            save_file({f'l18_h{h:02d}_dense_key':value[i].contiguous() for i,h in enumerate(head_ids)},str(target))
            target.with_suffix('.json').write_text(json.dumps(meta|{'native_mode':mode},indent=2))
        print(json.dumps({'video':name,'storage':storage,'seconds':time.monotonic()-began,'noise_sigma':sigma}),flush=True)

if __name__=='__main__':main()
