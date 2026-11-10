"""Compare the encoder clock with the complete pretrained codec's RGB clock.

Hypothesis: the decoder can recover frame-aligned RGB even when the encoder's
latent content has a lag. This is a test, not an assumed explanation.
Uses unmodified upstream VAE implementation and existing ModelScope weights.
"""
import hashlib,json,sys,time,types
from pathlib import Path
import cv2,numpy as np,torch
from torch.nn import functional as F

@torch.inference_mode()
def main():
    root=Path('/share/yty/heft');device='cuda:1';torch.cuda.set_device(device);torch.set_num_threads(3);cv2.setNumThreads(1)
    source=root/'provenance/official_autoencoder_kl_wan_v0p35p0.py';module_name='diffusers.models.autoencoders._heft_official_codec_clock'
    module=types.ModuleType(module_name);module.__package__='diffusers.models.autoencoders';module.__file__=str(source);sys.modules[module_name]=module
    exec(compile(source.read_text(),str(source),'exec'),module.__dict__)
    path=json.loads((root/'models/Wan2.1-T2V-1.3B-Diffusers.json').read_text())['path']
    vae=module.AutoencoderKLWan.from_pretrained(path,subfolder='vae',torch_dtype=torch.bfloat16,local_files_only=True).to(device).eval()
    mean=torch.tensor(vae.config.latents_mean,device=device).view(1,16,1,1,1);std=torch.tensor(vae.config.latents_std,device=device).view(1,16,1,1,1)
    yy,xx=torch.meshgrid(torch.linspace(.3,.7,16,device=device),torch.linspace(.3,.7,16,device=device),indexing='ij')
    query=torch.stack((xx,yy),-1).reshape(-1,2);times=torch.arange(-8,32.01,.25,device=device)
    names=['gold-fish','horsejump-high','india'];rows=[];began=time.monotonic()
    design={'images':names,'speeds':4,'directions':4,'source':'unmodified upstream VAE, same ModelScope checkpoint',
        'expected_relation_before_measurement':'full-codec RGB timing may differ from the latent content timing; measure both',
        'calibration_metric':'same static/moving registration objective; same video index controls startup context','human_annotations_opened':False}
    (root/'results/codec_clock_design.json').write_text(json.dumps(design,indent=2))
    def run(frames):
        x=torch.from_numpy(np.stack(frames)).permute(3,0,1,2)[None].to(device,dtype=torch.bfloat16)/127.5-1
        z=vae.encode(x).latent_dist.mode();rgb=vae.decode(z).sample.float();assert rgb.shape==x.shape,(rgb.shape,x.shape)
        return ((z.float()-mean)/std)[0].permute(1,0,2,3),rgb[0].permute(1,0,2,3),float((rgb-x).square().mean())
    for name in names:
        image=cv2.resize(np.load(root/f'data/davis_records/{name}.video.npy',mmap_mode='r')[0],(832,480),interpolation=cv2.INTER_LANCZOS4)
        static_z,static_rgb,_=run([image]*25)
        for vx,vy in [(4,0),(-4,0),(0,4),(0,-4)]:
            frames=[cv2.warpAffine(image,np.float32([[1,0,vx*t],[0,1,vy*t]]),(832,480),borderMode=cv2.BORDER_REFLECT_101) for t in range(25)]
            moving_z,moving_rgb,recon_mse=run(frames);grid=(query[None]+times[:,None,None]*torch.tensor([vx/832,vy/480],device=device)[None,None])*2-1
            record={'video':name,'velocity':[vx,vy],'reconstruction_mse':recon_mse}
            for label,reference,moving,nominal in [('latent',static_z,moving_z,np.arange(0,25,4)),('decoded_rgb',static_rgb,moving_rgb,np.arange(25))]:
                n=len(reference);ref=F.grid_sample(reference,query[None,None].expand(n,-1,-1,-1)*2-1,align_corners=False)[:,:,0].permute(0,2,1)
                sampled=F.grid_sample(moving,grid[None].expand(n,-1,-1,-1),align_corners=False).permute(0,2,3,1)
                mse=(sampled-ref[:,None]).square().mean((2,3));effective=times[mse.argmin(1)].cpu().numpy()
                record[label]={'nominal_time':nominal.tolist(),'effective_time':effective.tolist(),'delay':(nominal-effective).tolist()}
            rows.append(record)
        print(json.dumps({'video':name,'seconds':time.monotonic()-began}),flush=True)
    payload={'design':design,'records':rows,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'official_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'seconds':time.monotonic()-began}
    for key in ['latent','decoded_rgb']:
        d=np.array([r[key]['delay'][1:] for r in rows]);payload[key+'_summary']={'median_delay':float(np.median(d)),'mean_delay':float(np.mean(d)),'mean_delay_by_time':d.mean(0).tolist()}
    (root/'results/codec_clock_summary.json').write_text(json.dumps(payload,indent=2));print(json.dumps({k:v for k,v in payload.items() if k.endswith('_summary')},indent=2))

if __name__=='__main__':main()
