"""Numerically verify the native encoder against unmodified upstream source."""
import hashlib,json,sys,types
from pathlib import Path
import cv2,numpy as np,torch
from research_causal_denoising import native_vae_class

@torch.inference_mode()
def main():
    root=Path('/share/yty/heft');torch.set_num_threads(3);cv2.setNumThreads(1);device='cuda:1';torch.cuda.set_device(device)
    source=root/'provenance/official_autoencoder_kl_wan_v0p35p0.py'
    name='diffusers.models.autoencoders._heft_official_wan_v035';module=types.ModuleType(name);module.__package__='diffusers.models.autoencoders';module.__file__=str(source);sys.modules[name]=module
    exec(compile(source.read_text(),str(source),'exec'),module.__dict__)
    native_cls,native_hash=native_vae_class();path=json.loads((root/'models/Wan2.1-T2V-1.3B-Diffusers.json').read_text())['path']
    native=native_cls.from_pretrained(path,subfolder='vae',torch_dtype=torch.bfloat16,local_files_only=True).to(device).eval()
    upstream=module.AutoencoderKLWan.from_pretrained(path,subfolder='vae',torch_dtype=torch.bfloat16,local_files_only=True).to(device).eval()
    rows=[]
    for video in ['gold-fish','bike-packing','horsejump-high']:
        frames=np.load(root/f'data/davis_records/{video}.video.npy',mmap_mode='r')[:25]
        x=torch.from_numpy(np.stack([cv2.resize(f,(832,480),interpolation=cv2.INTER_LANCZOS4) for f in frames])).permute(3,0,1,2)[None].to(device,dtype=torch.bfloat16)/127.5-1
        a=native.encode(x).latent_dist.parameters;b=upstream.encode(x).latent_dist.parameters
        row={'video':video,'shape':list(a.shape),'max_absolute_difference':float((a-b).abs().max()),'all_values_bitwise_equal':bool(torch.equal(a,b))}
        assert row['all_values_bitwise_equal'],row;rows.append(row)
    payload={'official_source':str(source),'official_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'research_native_source_sha256':native_hash,'checkpoint_source':'existing ModelScope Wan2.1-T2V-1.3B Diffusers snapshot',
        'encoder_only':True,'comparisons':rows}
    (root/'provenance/native_vae_numerical_verification.json').write_text(json.dumps(payload,indent=2));print(json.dumps(payload,indent=2))

if __name__=='__main__':main()
