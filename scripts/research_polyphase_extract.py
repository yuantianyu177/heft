"""Frame-dense readout using the native, temporally compressed Wan encoder.

Four causal padding phases expose different source-frame residues modulo four.
Each transformer context keeps a common frame-zero anchor and seven native
latents. RoPE timestamps follow source-frame/4, including fractional offsets.
No temporal convolution or downsampling weights are removed or re-trained.
"""
import argparse,hashlib,json,time
from pathlib import Path
import cv2,numpy as np,torch
from safetensors.torch import load_file,save_file
from torch.nn import functional as F
from research_causal_denoising import native_vae_class
from research_rope import wan_angles,phase_readout


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('/share/yty/heft'))
    parser.add_argument('--gpu',type=int,default=6);parser.add_argument('--names',default='gold-fish')
    parser.add_argument('--dense-only',action='store_true');parser.add_argument('--chunked',action='store_true')
    parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=1)
    args=parser.parse_args();torch.set_num_threads(3);cv2.setNumThreads(1)
    device=f'cuda:{args.gpu}';torch.cuda.set_device(device)
    from diffusers import WanTransformer3DModel,UniPCMultistepScheduler
    path=json.loads((args.root/'models/Wan2.1-T2V-1.3B-Diffusers.json').read_text())['path']
    vae_class,vae_hash=native_vae_class()
    vae=vae_class.from_pretrained(path,subfolder='vae',torch_dtype=torch.bfloat16,local_files_only=True).to(device).eval()
    model=WanTransformer3DModel.from_pretrained(path,subfolder='transformer',torch_dtype=torch.bfloat16,local_files_only=True).to(device).eval()
    prompt=load_file(str(args.root/'models/wan_empty_prompt.safetensors'))['prompt_embeds'].to(device)
    scheduler=UniPCMultistepScheduler.from_pretrained(path,subfolder='scheduler',local_files_only=True)
    scheduler.set_timesteps(50,device=device);sigma=float(scheduler.sigmas[49]);step=scheduler.timesteps[49]
    mean=torch.tensor(vae.config.latents_mean,device=device).view(1,16,1,1,1)
    std=torch.tensor(vae.config.latents_std,device=device).view(1,16,1,1,1)
    source_indices=[[0,4,8,12,16,20,24],*[([0]+list(range(k,25,4))) for k in [1,2,3]]]
    assert all(len(indices)==7 for indices in source_indices)
    native_angles=wan_angles(7,30,52,device=device)
    dense_angles=wan_angles(25,30,52,device=device);dense_angles[...,:22]*=.25
    frequencies=10000.**(-torch.arange(22,device=device,dtype=torch.float64)/22)
    times=torch.tensor(source_indices,device=device,dtype=torch.float64)/4
    phase_angles=native_angles[None].expand(4,-1,-1,-1,-1).clone()
    phase_angles[...,:22]=times[:,:,None,None,None]*frequencies[None,None,None,None]
    phase_rotations=torch.polar(torch.ones_like(phase_angles),phase_angles).reshape(4,1,-1,64)
    model.rope.forward=lambda hidden_states:phase_rotations
    modes=['native_polyphase','native_interpolate','native_phase_interpolate']
    dense_heads={15:[2],18:[1,5],20:[8]};sparse=[0,1,4,8,12,24]
    state={}
    def densify(feature):
        # feature [phase,head,native_t,H,W,D]
        full=torch.empty((12,25,30,52,128),device=device,dtype=feature.dtype)
        for phase,indices in enumerate(source_indices):full[:,indices]=feature[phase]
        full[:,0]=feature[:,:,0].float().mean(0).to(feature.dtype)
        coarse=feature[0].float()
        def interpolate(value):
            return F.interpolate(value.permute(0,4,1,2,3),size=(25,30,52),mode='trilinear',align_corners=True).permute(0,2,3,4,1)
        raw_interpolate=interpolate(coarse)
        demodulated=phase_readout(coarse,native_angles,temporal=0,spatial=1)
        phase_interpolate=phase_readout(interpolate(demodulated),dense_angles,temporal=2,spatial=1)
        return [full,raw_interpolate,phase_interpolate]
    class Captured(Exception):pass
    def capture(layer):
        def hook(*,query,key,**kwargs):
            if args.dense_only and layer not in dense_heads:return
            keys=densify(key.reshape(4,12,7,30,52,128))
            queries=densify(query.reshape(4,12,7,30,52,128)) if not args.dense_only else None
            for index,mode in enumerate(modes):
                if not args.dense_only:
                    state[mode][f'l{layer:02d}_key']=keys[index][:,sparse].to(torch.bfloat16).cpu().contiguous()
                    state[mode][f'l{layer:02d}_query']=queries[index][:,sparse].to(torch.bfloat16).cpu().contiguous()
                if layer in dense_heads:
                    for head in dense_heads[layer]:state[mode][f'l{layer:02d}_h{head:02d}_dense_key']=keys[index][head].to(torch.bfloat16).cpu().contiguous()
            if args.dense_only and layer==max(dense_heads):raise Captured
        return hook
    for layer,block in enumerate(model.blocks):block.attn1._heft_capture=capture(layer)
    manifest=json.loads((args.root/'data/davis_records/manifest.json').read_text())
    if args.names:manifest=[r for r in manifest if r['name'] in args.names.split(',')]
    if args.chunked:
        manifest=[dict(r,start_frame=s,valid_frames=min(25,r['frames']-s),storage_name=f'{r["name"]}_c{s//25:03d}')
                  for r in manifest for s in range(0,r['frames'],25)]
    manifest=manifest[args.shard::args.shards]
    suffix='chunk25' if args.chunked else 't25'
    for record in manifest:
        name=record['name'];storage=record.get('storage_name',name)
        if all((args.root/f'features/wan_{mode}_{suffix}/{storage}.json').exists() for mode in modes):continue
        started=time.monotonic();start=record.get('start_frame',0)
        video=np.load(args.root/f'data/davis_records/{name}.video.npy',mmap_mode='r')[start:start+25]
        if len(video)<25:video=np.concatenate((video,np.repeat(video[-1:],25-len(video),axis=0)))
        resized=np.stack([cv2.resize(f,(832,480),interpolation=cv2.INTER_LANCZOS4) for f in video])
        inputs=torch.from_numpy(resized).permute(3,0,1,2)[None].to(device,dtype=torch.bfloat16)/127.5-1
        clean=[]
        for phase in range(4):
            before=0 if phase==0 else 4-phase
            padded=torch.cat((inputs[:,:,:1].expand(-1,-1,before,-1,-1),inputs),dim=2)
            padded=torch.cat((padded,padded[:,:,-1:].expand(-1,-1,29-padded.shape[2],-1,-1)),dim=2)
            latent=vae.encode(padded).latent_dist.mode()[:,:,:7].float()
            assert latent.shape==(1,16,7,60,104)
            clean.append((latent-mean)/std)
        clean=torch.cat(clean)
        anchor_difference=float((clean[:,:,:1]-clean[:1,:,:1]).abs().max())
        assert anchor_difference==0.,anchor_difference
        noise=torch.randn((1,16,25,60,104),generator=torch.Generator().manual_seed(42)).to(device)
        phase_noise=torch.cat([noise[:,:,indices] for indices in source_indices])
        noisy=((1-sigma)*clean+sigma*phase_noise).to(torch.bfloat16)
        for mode in modes:state[mode]={}
        try:model(noisy,timestep=step.expand(4),encoder_hidden_states=prompt.expand(4,-1,-1).contiguous(),return_dict=False)
        except Captured:pass
        for mode in modes:
            directory=args.root/f'features/wan_{mode}_{suffix}';directory.mkdir(exist_ok=True,parents=True)
            save_file(state[mode],str(directory/f'{storage}.safetensors'))
            metadata={'video':name,'split':record['split'],'transform':'real','frames':25,'sparse_frames':sparse,
                'storage_name':storage,'start_frame':start,'valid_frames':record.get('valid_frames',25),
                'source_size':list(video.shape[1:3]),'input_size':[480,832],'feature_size':[30,52],
                'layers':30,'heads':12,'dense_heads':dense_heads,'noise_timestep':float(step),'noise_sigma':sigma,
                'latent_normalization':'pretrained','posterior':'mode','seed':42,'mode':mode,
                'rope_temporal_scale':.25,'phase_source_indices':source_indices,'phase_anchor':'mean of four context-specific descriptors',
                'vae_temporal_compression':'native 4; all pretrained encoder temporal operations retained',
                'native_vae_source_hash':vae_hash,'anchor_latent_max_difference':anchor_difference,
                'forward_seconds':time.monotonic()-started,'peak_gpu_bytes':torch.cuda.max_memory_allocated(device),
                'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
            (directory/f'{storage}.json').write_text(json.dumps(metadata,indent=2))
        print(json.dumps({'video':name,'storage':storage,'seconds':time.monotonic()-started,
                          'anchor_latent_max_difference':anchor_difference,'peak_gpu_bytes':torch.cuda.max_memory_allocated(device)}),flush=True)


if __name__=='__main__':main()
