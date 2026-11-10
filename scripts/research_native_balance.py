"""Canonical pair energy calibration on the exact unlabelled student inputs."""
import json,time
from pathlib import Path
import cv2,numpy as np,torch
from safetensors.torch import load_file
from torch.nn.attention import sdpa_kernel,SDPBackend
from research_causal_denoising import native_vae_class

@torch.inference_mode()
def main():
    root=Path('/share/yty/heft');device='cuda:7';torch.set_num_threads(3);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    design=json.loads((root/'results/functional_teacher_design.json').read_text())
    from diffusers import WanTransformer3DModel
    path=json.loads((root/'models/Wan2.1-T2V-1.3B-Diffusers.json').read_text())['path']
    cls,vae_hash=native_vae_class()
    vae=cls.from_pretrained(path,subfolder='vae',torch_dtype=torch.bfloat16,local_files_only=True).to(device).eval()
    model=WanTransformer3DModel.from_pretrained(path,subfolder='transformer',torch_dtype=torch.float32,local_files_only=True).to(device).eval()
    prompt=load_file(str(root/'models/wan_empty_prompt.safetensors'))['prompt_embeds'].to(device,dtype=torch.float32)
    layers=design['student_layers'];state={}
    def capture(layer):
        def hook(*,query,key,**kwargs):
            state[layer]=[(value[0].reshape(12,-1,64,2).square().sum(-1).mean(1)).cpu().numpy() for value in [query,key]]
        return hook
    for layer in layers:model.blocks[layer].attn1._heft_capture=capture(layer)
    records=[]
    for name in design['videos']:
        frames=np.load(root/f'data/davis_records/{name}.video.npy',mmap_mode='r')[:25]
        frames=np.stack([cv2.resize(f,(832,480),interpolation=cv2.INTER_LANCZOS4) for f in frames])
        tensor=torch.from_numpy(frames).permute(3,0,1,2)[None].to(device,dtype=torch.bfloat16)/127.5-1
        latent=vae.encode(tensor).latent_dist.mode().float()
        mean=torch.tensor(vae.config.latents_mean,device=device).view(1,16,1,1,1)
        std=torch.tensor(vae.config.latents_std,device=device).view(1,16,1,1,1)
        clean=(latent-mean)/std;noise=torch.randn(clean.shape,generator=torch.Generator().manual_seed(1273)).to(device)
        noisy=(1-design['student_sigma'])*clean+design['student_sigma']*noise
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            model(noisy,timestep=torch.tensor([design['student_timestep']],device=device),encoder_hidden_states=prompt,return_dict=False)
        records.append(np.stack([state[layer] for layer in layers]));print(name,flush=True)
    energies=np.mean(records,axis=0)
    scales=(energies[:,0]/np.maximum(energies[:,1],1e-30))**.25
    output={'head_ids':[[l,h] for l in layers for h in range(12)],'scales':scales.reshape(-1,64).tolist(),
            'q_energy':energies[:,0].tolist(),'k_energy':energies[:,1].tolist(),'design':design,
            'vae_hash':vae_hash,'human_annotations_opened':False}
    (root/'results/functional_student_balance.json').write_text(json.dumps(output,indent=2))

if __name__=='__main__':main()
