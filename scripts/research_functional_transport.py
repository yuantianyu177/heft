"""Probe cross-frame model influence without selecting an internal head/basis.

Antithetic input perturbations estimate block-Jacobian response energy. This is
model intervention, not evidence of real-world physical causality. Native VAE
compression/normalization is restored. Several epsilons test numerical stability.
"""
import argparse,hashlib,json,time
from pathlib import Path
import cv2,numpy as np,torch
from safetensors.torch import load_file,save_file
from torch.nn import functional as F
from research_causal_denoising import native_vae_class
from research_probe import probe,grid_points


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('/share/yty/heft'))
    parser.add_argument('--gpu',type=int,default=2);parser.add_argument('--names',default='gold-fish')
    parser.add_argument('--noise-levels',default='0.5');parser.add_argument('--epsilons',default='0.5,1.0')
    parser.add_argument('--queries',type=int,default=8);parser.add_argument('--probes',type=int,default=4)
    parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--precision',choices=['bf16','fp32'],default='bf16')
    parser.add_argument('--tag',default='')
    parser.add_argument('--uniform-grid',type=int,default=0,help='Use an unlabelled grid, never open trajectory annotations')
    parser.add_argument('--save-student-features',action='store_true')
    parser.add_argument('--student-sigma',type=float,default=.05763683095574379)
    parser.add_argument('--student-timestep',type=float,default=57.)
    parser.add_argument('--ablate-heads',help='Zero selected self-attention outputs for model-function controls, e.g.18:1')
    args=parser.parse_args();torch.set_num_threads(3);cv2.setNumThreads(1)
    program_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    device=f'cuda:{args.gpu}';torch.cuda.set_device(device)
    from diffusers import WanTransformer3DModel
    model_path=json.loads((args.root/'models/Wan2.1-T2V-1.3B-Diffusers.json').read_text())['path']
    vae_class,vae_hash=native_vae_class()
    vae=vae_class.from_pretrained(model_path,subfolder='vae',torch_dtype=torch.bfloat16,local_files_only=True).to(device).eval()
    model_dtype=torch.float32 if args.precision=='fp32' else torch.bfloat16
    model=WanTransformer3DModel.from_pretrained(model_path,subfolder='transformer',torch_dtype=model_dtype,local_files_only=True).to(device).eval()
    prompt=load_file(str(args.root/'models/wan_empty_prompt.safetensors'))['prompt_embeds'].to(device,dtype=model_dtype)
    torch.backends.cuda.matmul.allow_tf32=False
    ablations=[tuple(map(int,s.split(':'))) for s in args.ablate_heads.split(',')] if args.ablate_heads else []
    captured={};state={'capture':False,'ablate':bool(ablations)}
    def capture(layer,heads):
        def hook(*,key,hidden_states,**kwargs):
            if state['capture'] and layer in student_layers:captured[layer]=key[0,heads].reshape(len(heads),7,30,52,128).float()
            if state['ablate']:
                for target_layer,head in ablations:
                    if target_layer==layer:hidden_states[:,head].zero_()
        return hook
    student_layers=[15,18,20] if args.save_student_features else [18]
    for layer in sorted(set(student_layers+[l for l,h in ablations])):
        model.blocks[layer].attn1._heft_capture=capture(layer,list(range(12)) if args.save_student_features else [1,5])
    destination=args.root/'results/functional_transport_wan';destination.mkdir(exist_ok=True,parents=True)
    for name in args.names.split(','):
        raw=np.load(args.root/f'data/davis_records/{name}.video.npy',mmap_mode='r')[:25]
        frames=np.stack([cv2.resize(f,(832,480),interpolation=cv2.INTER_LANCZOS4) for f in raw])
        inputs=torch.from_numpy(frames).permute(3,0,1,2)[None].to(device,dtype=torch.bfloat16)/127.5-1
        latent=vae.encode(inputs).latent_dist.mode().float()
        mean=torch.tensor(vae.config.latents_mean,device=device).view(1,16,1,1,1)
        std=torch.tensor(vae.config.latents_std,device=device).view(1,16,1,1,1)
        clean=(latent-mean)/std
        if args.uniform_grid:
            initial=grid_points(device,args.uniform_grid)
            selected=np.arange(len(initial))
            points=initial[None].expand(7,-1,-1).contiguous()
            visible=torch.ones(points.shape[:2],device=device,dtype=torch.bool)
        else:
            with np.load(args.root/f'data/davis_records/{name}.annotations.npz') as ann:
                eligible=np.flatnonzero(~ann['occluded'][:,0])
                rng=np.random.default_rng(int(hashlib.sha256(name.encode()).hexdigest()[:8],16))
                selected=np.sort(rng.choice(eligible,min(args.queries,len(eligible)),replace=False))
                points=torch.tensor(ann['points'][selected][:,[0,4,8,12,16,20,24]].transpose(1,0,2).copy(),device=device)
                visible=torch.tensor((~ann['occluded'][selected][:,[0,4,8,12,16,20,24]].T).copy(),device=device)
        noise=torch.randn(clean.shape,generator=torch.Generator().manual_seed(1273)).to(device)
        ntarget=len(selected);epsilons=list(map(float,args.epsilons.split(',')))
        for sigma in map(float,args.noise_levels.split(',')):
            output=destination/f'{name}_sigma{int(round(sigma*1000)):04d}_p{args.probes}_q{ntarget}'
            if args.precision=='fp32':output=output.with_name(output.name+'_fp32')
            if args.tag:output=output.with_name(output.name+'_'+args.tag)
            if args.uniform_grid:output=output.with_name(output.name+f'_grid{args.uniform_grid}')
            if ablations:
                mask_tag='_'.join(f'{l}h{h}' for l,h in ablations)
                if len(mask_tag)>80:mask_tag='group'+hashlib.sha256(mask_tag.encode()).hexdigest()[:12]
                output=output.with_name(output.name+'_ablate'+mask_tag)
            if output.with_suffix('.json').exists():continue
            start=time.monotonic();noisy=((1-sigma)*clean+sigma*noise).to(model_dtype)
            def forward(batch,timestep_value=None):
                from torch.nn.attention import sdpa_kernel,SDPBackend
                backends=[SDPBackend.EFFICIENT_ATTENTION] if args.precision=='fp32' else [SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION]
                with sdpa_kernel(backends):
                    return model(batch,timestep=torch.full((len(batch),),sigma*1000 if timestep_value is None else timestep_value,device=device),
                        encoder_hidden_states=prompt.expand(len(batch),-1,-1).contiguous(),return_dict=False)[0].float()
            if ablations:
                state['ablate']=False;original_prediction=forward(noisy);state['ablate']=True
                original_flow_mse=float((original_prediction-(noise-clean)).square().double().mean())
            else:original_flow_mse=None
            state['capture']=True;baseline_prediction=forward(noisy);state['capture']=False
            flow_mse=float((baseline_prediction-(noise-clean)).square().double().mean())
            if not args.uniform_grid:
                keys=captured[18][[1,5]] if args.save_student_features else captured[18]
                key_results=probe(keys,points[0],points[1:],visible[1:],grid_points(device))
            if args.save_student_features:
                state['capture']=True
                student_noisy=((1-args.student_sigma)*clean+args.student_sigma*noise).to(model_dtype)
                forward(student_noisy,args.student_timestep);state['capture']=False
                save_file({f'l{layer:02d}_key':captured[layer].to(torch.bfloat16).cpu().contiguous() for layer in student_layers},
                          str(output.with_name(output.name+'_student.safetensors')))
            control=forward(noisy.expand(2,-1,-1,-1,-1).contiguous())
            control_error=float((control[0]-control[1]).abs().max())
            directions=[];identities=[]
            for point in range(ntarget):
                x=int((points[0,point,0]*52).floor().clamp(0,51));y=int((points[0,point,1]*30).floor().clamp(0,29))
                for seed in range(args.probes):
                    direction=torch.zeros_like(noisy[0],dtype=torch.float32)
                    signs=torch.randint(0,2,(16,2,2),generator=torch.Generator().manual_seed(233+seed)).to(device).float()*2-1
                    direction[:,0,2*y:2*y+2,2*x:2*x+2]=signs/8 # unit Euclidean norm
                    directions.append(direction);identities.append(point)
            all_energy=[]
            for epsilon in epsilons:
                energies=torch.zeros((ntarget,6,30,52),device=device)
                for offset in range(0,len(directions),args.batch//2):
                    group=torch.stack(directions[offset:offset+args.batch//2])
                    plus=(noisy.float()+epsilon*group).to(model_dtype)
                    minus=(noisy.float()-epsilon*group).to(model_dtype)
                    result=forward(torch.cat((plus,minus)))
                    response=(result[:len(group)]-result[len(group):])/(2*epsilon)
                    energy=response[:,:,1:].square().sum(1)
                    energy=F.avg_pool2d(energy.flatten(0,1)[:,None],2).reshape(len(group),6,30,52)
                    for j,value in enumerate(energy):energies[identities[offset+j]]+=value/args.probes
                all_energy.append(energies.cpu().numpy())
                print(json.dumps({'video':name,'sigma':sigma,'epsilon_complete':epsilon,'precision':args.precision}),flush=True)
            energy=np.stack(all_energy)
            index=energy.reshape(len(epsilons),ntarget,6,-1).argmax(-1)
            prediction=np.stack(((index%52+.5)/52,(index//52+.5)/30),axis=-1).transpose(0,2,1,3)
            valid=visible[1:].cpu().numpy()
            if not args.uniform_grid:
                errors=np.linalg.norm((prediction-points[None,1:].cpu().numpy())*256,axis=-1)
                pck=((errors[...,None]<[1,2,4,8,16]).mean(-1)*valid).sum((-1,-2))/valid.sum()*100
                stationary_error=np.linalg.norm((points[:1].cpu().numpy()-points[1:].cpu().numpy())*256,axis=-1)
                stationary_pck=float(((stationary_error[...,None]<[1,2,4,8,16]).mean(-1)*valid).sum()/valid.sum()*100)
                kk_pck=((key_results['errors'][...,None]<[1,2,4,8,16]).mean(-1)*valid).sum((-1,-2))/valid.sum()*100
            else:errors=np.full((len(epsilons),6,ntarget),np.nan);pck=np.array([]);stationary_pck=None;kk_pck=np.array([])
            np.savez_compressed(output.with_suffix('.npz'),energy=energy,errors=errors,prediction=prediction,
                                points=points.cpu().numpy() if not args.uniform_grid else np.full(points.shape,np.nan),
                                query_points=points[0].cpu().numpy(),ground_truth_available=not bool(args.uniform_grid),
                                valid=valid,query_indices=selected,epsilons=epsilons)
            summary={'video':name,'sigma':sigma,'epsilons':epsilons,'response_pck':pck.tolist() if not args.uniform_grid else None,
                     'stationary_pck':stationary_pck if not args.uniform_grid else None,
                     'native_kk_pck_L18H1_H5':kk_pck.tolist() if not args.uniform_grid else None,
                     'duplicate_input_max_output_error':control_error,'probes':args.probes,'queries':ntarget,
                     'native_vae_source_hash':vae_hash,'seconds':time.monotonic()-start,
                     'precision':args.precision,'epsilon_map_cosine':float(F.cosine_similarity(torch.tensor(energy[0]).flatten(),torch.tensor(energy[-1]).flatten(),dim=0)) if len(energy)>1 else None,
                     'ablated_heads':ablations,'baseline_flow_mse':flow_mse,'original_flow_mse':original_flow_mse,
                     'human_annotation_access':not bool(args.uniform_grid),'uniform_grid':args.uniform_grid,
                     'student_sigma':args.student_sigma if args.save_student_features else None,
                     'student_timestep':args.student_timestep if args.save_student_features else None,
                     'student_layers':student_layers if args.save_student_features else None,
                     'source_sha256':program_hash,
                     'limitation':'finite antithetic input-response probe; tests influence inside the model, not physical world causality'}
            output.with_suffix('.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
