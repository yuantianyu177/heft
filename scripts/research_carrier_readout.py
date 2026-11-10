"""Parameter-defined affine carrier removal at readout, without changing attention.

Exact decomposition before QK RMSNorm:
  K_raw = W_K [LN(h) * (1+scale)] + (W_K shift + b_K).
The network always uses its original keys. Alternate descriptors subtract only
the chosen fixed vector before the original normalization and rotation. Random
controls preserve each head's removed-vector norm. No labels determine vectors.
"""
import argparse, hashlib, json, time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.nn import functional as F

from research_probe import annotations, grid_points, probe
from research_rope import wan_angles, phase_readout

VARIANTS=['post_full','minus_bias','minus_adaln','minus_shift','plus_bias',
          'random_bias_0','random_bias_1','minus_bias_pre','minus_adaln_pre',
          'bias_only','adaln_only']


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('/share/yty/heft'))
    parser.add_argument('--gpu',type=int,default=7)
    parser.add_argument('--names',default='gold-fish')
    parser.add_argument('--layers',default='all')
    parser.add_argument('--features',default='wan_real_t25')
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args();torch.set_num_threads(3)
    device=f'cuda:{args.gpu}';torch.cuda.set_device(device)
    from diffusers import WanTransformer3DModel
    path=json.loads((args.root/'models/Wan2.1-T2V-1.3B-Diffusers.json').read_text())['path']
    model=WanTransformer3DModel.from_pretrained(path,subfolder='transformer',torch_dtype=torch.bfloat16,
                                               local_files_only=True).to(device).eval()
    prompt=load_file(str(args.root/'models/wan_empty_prompt.safetensors'))['prompt_embeds'].to(device)
    layers=list(range(30)) if args.layers=='all' else list(map(int,args.layers.split(',')))
    state={};angles=wan_angles(25,30,52,device=device)
    sparse=[0,1,4,8,12,24];sparse_angles=angles[sparse]
    output_root=args.root/f'results/carrier_readout_{args.features}'
    output_root.mkdir(exist_ok=True,parents=True)
    dense_root=args.root/f'features/carrier_readout_{args.features}'
    dense_root.mkdir(exist_ok=True,parents=True)
    def raw_hook(layer):
        def hook(module,inputs,output):state['raw'][layer]=output
        return hook
    def capture(layer):
        def hook(*,key,**kwargs):
            block=model.blocks[layer];attention=block.attn1
            raw=state['raw'].pop(layer)[0].reshape(25,30,52,1536)
            bias=attention.to_k.bias.float()
            shift=(block.scale_shift_table+state['proj'].float())[:,0:1].to(torch.bfloat16)
            carrier=F.linear(shift.float(),attention.to_k.weight.float(),bias).flatten()
            raw_sparse=raw[sparse].float()
            by_variant={}
            random_vectors=[]
            for seed in [0,1]:
                vector=torch.randn((12,128),generator=torch.Generator().manual_seed(8713+seed)).to(device)
                vector=F.normalize(vector,dim=-1)*bias.reshape(12,128).norm(dim=-1,keepdim=True)
                random_vectors.append(vector.flatten())
            for variant in VARIANTS:
                if variant=='post_full':feature=key[0].reshape(12,25,30,52,128)[:,sparse].float()
                else:
                    if variant in ['minus_bias','minus_bias_pre']:altered=raw_sparse-bias
                    elif variant in ['minus_adaln','minus_adaln_pre']:altered=raw_sparse-carrier
                    elif variant=='minus_shift':altered=raw_sparse-(carrier-bias)
                    elif variant=='plus_bias':altered=raw_sparse+bias
                    elif variant.startswith('random_bias'):altered=raw_sparse-random_vectors[int(variant[-1])]
                    elif variant=='bias_only':altered=bias.expand_as(raw_sparse)
                    elif variant=='adaln_only':altered=carrier.expand_as(raw_sparse)
                    else:raise ValueError(variant)
                    normalized=attention.norm_k(altered.to(torch.bfloat16)[None])[0].float()
                    feature=normalized.reshape(6,30,52,12,128).permute(3,0,1,2,4).contiguous()
                    if not variant.endswith('_pre'):
                        feature=phase_readout(feature,sparse_angles,temporal=2,spatial=2)
                result=probe(feature,state['points'],state['truth'],state['valid'],state['cycle'])
                by_variant[variant]=result
            state['results'].append(by_variant)
            state['carrier_norm'].append(carrier.reshape(12,128).norm(dim=-1).cpu().numpy())
            state['raw_norm'].append(raw_sparse.reshape(6,30,52,12,128).square().mean((0,1,2,4)).sqrt().cpu().numpy())
            # Dense descriptors for a fixed small set of heads, enabling tracker tests.
            chosen={15:[2],18:[1,5,8],20:[8]}.get(layer,[])
            if chosen:
                for variant,vector in [('minus_bias',bias),('minus_adaln',carrier)]:
                    normalized=attention.norm_k((raw.float()-vector).to(torch.bfloat16)[None])[0].float()
                    feature=normalized.reshape(25,30,52,12,128).permute(3,0,1,2,4).contiguous()
                    feature=phase_readout(feature,angles,temporal=2,spatial=2)
                    for head in chosen:state['dense'][f'{variant}_l{layer:02d}_h{head:02d}']=feature[head].cpu().to(torch.bfloat16).contiguous()
        return hook
    for layer in layers:
        block=model.blocks[layer]
        block.attn1.to_k.register_forward_hook(raw_hook(layer))
        block.attn1._heft_capture=capture(layer)
    for name in args.names.split(','):
        output=output_root/name
        if output.with_suffix('.npz').exists() and not args.overwrite:continue
        started=time.monotonic()
        metadata=json.loads((args.root/f'features/{args.features}/{name}.json').read_text())
        points,truth,valid=annotations(args.root,name,metadata,device)
        timestep=float(metadata['noise_timestep'])
        _,proj,*_=model.condition_embedder(torch.tensor([timestep],device=device),prompt)
        state.update(raw={},results=[],dense={},proj=proj.unflatten(1,(6,-1)),points=points,truth=truth,valid=valid,
                     cycle=grid_points(device),carrier_norm=[],raw_norm=[])
        noisy=load_file(str(args.root/f'latents/{args.features}/{name}.safetensors'))['noisy'].to(device)
        model(noisy,timestep=torch.tensor([timestep],device=device),encoder_hidden_states=prompt,return_dict=False)
        errors=np.stack([[r[v]['errors'] for v in VARIANTS] for r in state['results']],axis=0).transpose(0,2,1,3,4)
        prediction=np.stack([[r[v]['prediction'] for v in VARIANTS] for r in state['results']],axis=0).transpose(0,2,1,3,4,5)
        np.savez_compressed(output.with_suffix('.npz'),errors=errors,prediction=prediction,valid=valid.cpu().numpy(),
                            variant_names=VARIANTS,layers=layers,carrier_norm=state['carrier_norm'],raw_rms=state['raw_norm'])
        save_file(state['dense'],str(dense_root/f'{name}.safetensors'))
        info={'video':name,'feature_metadata':metadata,'seconds':time.monotonic()-started,
              'selection_annotation_access':False,'evaluation_annotation_access':True,
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'intervention':'readout only; all original model keys and all attention computations preserved',
              'random_control':'subtract independent random direction with the exact same per-head Euclidean norm as projection bias'}
        output.with_suffix('.json').write_text(json.dumps(info,indent=2))
        scores=((errors[...,None]<np.array([1,2,4,8,16])).mean(-1)*valid.cpu().numpy()).sum((-1,-2))/valid.sum().item()
        print(json.dumps({'video':name,'seconds':info['seconds'],'population':dict(zip(VARIANTS,(scores.mean((0,1))*100).tolist()))}),flush=True)


if __name__=='__main__':main()
