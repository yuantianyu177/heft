"""Measure effective temporal alignment of pretrained causal VAE content.

Known constant translations of seven calibration images, no motion annotations.
At each latent time, compare moving-frame content with an equally encoded
static video, removing the encoder's frame-zero/startup context as a confound.
"""
import argparse,hashlib,json,time
from pathlib import Path
import cv2,numpy as np,torch
from torch.nn import functional as F
from research_causal_denoising import native_vae_class

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/share/yty/heft'));p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--names',default='drift-straight,gold-fish,horsejump-high,india,kite-surf,lab-coat,paragliding-launch')
    p.add_argument('--velocities',default='4');p.add_argument('--native-only',action='store_true');p.add_argument('--output-tag',default='vae_temporal_alignment')
    p.add_argument('--temporal-ablation',choices=['early','late'],help='Retain exactly one of the two pretrained temporal downsampling modules')
    p.add_argument('--encoder',choices=['wan','cogvideox'],default='wan')
    a=p.parse_args()
    torch.set_num_threads(3);cv2.setNumThreads(1);device=f'cuda:{a.gpu}';torch.cuda.set_device(device)
    from diffusers import AutoencoderKLWan,AutoencoderKLCogVideoX
    cls,vae_hash=native_vae_class();path=json.loads((a.root/'models/Wan2.1-T2V-1.3B-Diffusers.json').read_text())['path']
    width,height=832,480
    if a.encoder=='cogvideox':
        import inspect
        assert not a.temporal_ablation
        cls=AutoencoderKLCogVideoX;AutoencoderKLWan=AutoencoderKLCogVideoX
        path=json.loads((a.root/'models/CogVideoX-2b-vae.json').read_text())['path'];width=720
        vae_hash=hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest()
    names=a.names.split(',')
    yy,xx=torch.meshgrid(torch.linspace(.3,.7,16,device=device),torch.linspace(.3,.7,16,device=device),indexing='ij')
    query=torch.stack((xx,yy),-1).reshape(-1,2);times=torch.arange(-8,32.01,.25,device=device)
    results={};began=time.monotonic()
    modes=[('native',cls,list(range(0,25,4))),('frame_aligned',AutoencoderKLWan,list(range(25)))]
    if a.temporal_ablation:modes=[('native_'+a.temporal_ablation,cls,list(range(0,25,2)))]
    for label,klass,indices in modes:
        if a.native_only and not label.startswith('native'):continue
        overrides={'temperal_downsample':[False,a.temporal_ablation=='early',a.temporal_ablation=='late']} if a.temporal_ablation else {}
        if a.encoder=='cogvideox' and label=='frame_aligned':overrides={'temporal_compression_ratio':1}
        vae=klass.from_pretrained(path,subfolder='vae',torch_dtype=torch.bfloat16,local_files_only=True,**overrides).to(device).eval()
        mean=torch.tensor(vae.config.latents_mean,device=device).view(1,16,1,1,1) if a.encoder=='wan' else 0.
        std=torch.tensor(vae.config.latents_std,device=device).view(1,16,1,1,1) if a.encoder=='wan' else 1.
        def encode(frames):
            x=torch.from_numpy(np.stack(frames)).permute(3,0,1,2)[None].to(device,dtype=torch.bfloat16)/127.5-1
            return ((vae.encode(x).latent_dist.mode().float()-mean)/std)[0].permute(1,0,2,3)
        rows=[]
        for name in names:
            image=cv2.resize(np.load(a.root/f'data/davis_records/{name}.video.npy',mmap_mode='r')[0],(width,height),interpolation=cv2.INTER_LANCZOS4)
            static=encode([image]*25)
            reference=F.grid_sample(static,query[None,None].expand(len(indices),-1,-1,-1)*2-1,align_corners=False)[:,:,0].permute(0,2,1)
            velocities=[v for speed in map(float,a.velocities.split(',')) for v in [(speed,0),(-speed,0),(0,speed),(0,-speed)]]
            for vx,vy in velocities:
                frames=[cv2.warpAffine(image,np.float32([[1,0,vx*t],[0,1,vy*t]]),(width,height),borderMode=cv2.BORDER_REFLECT_101) for t in range(25)]
                moving=encode(frames);offset=times[:,None,None]*torch.tensor([vx/width,vy/height],device=device)[None,None]
                locations=query[None]+offset
                safe=((locations>.02)&(locations<.98)).all((0,2))
                assert safe.sum()>16,(vx,vy,int(safe.sum()))
                grid=locations[:,safe]*2-1
                samples=F.grid_sample(moving,grid[None].expand(len(indices),-1,-1,-1),align_corners=False)
                error=(samples.permute(0,2,3,1)-reference[:,None,safe]).square().mean((2,3))
                best=times[error.argmin(1)];nominal=torch.tensor(indices,device=device)
                row={'video':name,'velocity_pixels_per_frame':[vx,vy],'nominal_times':indices,'effective_times':best.cpu().tolist(),
                     'delay_nominal_minus_effective':(nominal-best).cpu().tolist(),'minimum_mse':error.min(1).values.cpu().tolist(),'safe_query_count':int(safe.sum())}
                rows.append(row)
            print(json.dumps({'mode':label,'video':name,'seconds':time.monotonic()-began}),flush=True)
        results[label]=rows;del vae;torch.cuda.empty_cache()
    payload={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'native_vae_hash':vae_hash,
        'protocol':'known constant translations, static-video context matched at every latent time; latent-content diagnostic only',
        'images':names,'speeds_input_pixels_per_frame':list(map(float,a.velocities.split(','))),
        'temporal_module_ablation':a.temporal_ablation,
        'encoder':a.encoder,'input_size':[height,width],
        'human_annotations_opened':False,'results':results,'seconds':time.monotonic()-began}
    out=a.root/f'results/{a.output_tag}.json';out.write_text(json.dumps(payload,indent=2));print(str(out),flush=True)

if __name__=='__main__':main()
