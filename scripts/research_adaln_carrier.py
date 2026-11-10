"""Test whether deterministic AdaLN shifts predict headwise key anisotropy."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('/share/yty/heft'))
    parser.add_argument('--gpu',type=int,default=6)
    args=parser.parse_args(); torch.set_num_threads(3)
    from diffusers import WanTransformer3DModel
    device=f'cuda:{args.gpu}'
    path=json.loads((args.root/'models/Wan2.1-T2V-1.3B-Diffusers.json').read_text())['path']
    model=WanTransformer3DModel.from_pretrained(path,subfolder='transformer',torch_dtype=torch.bfloat16,
                                               local_files_only=True).to(device).eval()
    observed=np.mean([np.load(p)['content_means'] for p in sorted((args.root/'results/probe_wan_real_t25').glob('*.npz'))],0)
    energy=np.mean([np.load(p)['pair_energies'] for p in sorted((args.root/'results/probe_wan_real_t25').glob('*.npz'))],0).sum(-1)
    prompt=load_file(str(args.root/'models/wan_empty_prompt.safetensors'))['prompt_embeds'][:,:1].to(device)
    steps=[0.,20.,57.,100.,250.,500.,750.,950.]
    carriers=[];biases=[];query_carriers=[]
    for step in steps:
        temb,proj,*_=model.condition_embedder(torch.tensor([step],device=device),prompt)
        proj=proj.unflatten(1,(6,-1))
        kc=[];qc=[];bc=[]
        for block in model.blocks:
            shift=(block.scale_shift_table+proj.float())[:,0:1].to(torch.bfloat16)
            attention=block.attn1
            k=attention.norm_k(attention.to_k(shift)).float().reshape(12,128)
            q=attention.norm_q(attention.to_q(shift)).float().reshape(12,128)
            b=attention.norm_k(attention.to_k.bias[None,None]).float().reshape(12,128)
            kc.append(k.cpu().numpy());qc.append(q.cpu().numpy());bc.append(b.cpu().numpy())
        carriers.append(kc);query_carriers.append(qc);biases.append(bc)
    carriers=np.asarray(carriers);biases=np.asarray(biases);query_carriers=np.asarray(query_carriers)
    predicted=carriers[steps.index(57.)]
    def cosine(a,b):return (a*b).sum(-1)/np.maximum(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1),1e-8)
    alignment=cosine(observed,predicted);bias_alignment=cosine(observed,biases[0])
    summary={'hypothesis':'The content-independent W_K*AdaLN_shift + b_K carrier predicts the direction of shared pre-RoPE key means',
             'mean_head_cosine':float(alignment.mean()),'median_head_cosine':float(np.median(alignment)),
             'fraction_above_09':float(np.mean(alignment>.9)),
             'bias_only_mean_cosine':float(bias_alignment.mean()),'selected_heads':{}}
    for l,h in [(15,2),(18,1),(18,5),(18,8),(20,8),(14,6)]:
        summary['selected_heads'][f'L{l}H{h}']={'adaln_alignment':float(alignment[l,h]),
             'bias_alignment':float(bias_alignment[l,h]),'observed_shared_energy_fraction':float((observed[l,h]**2).sum()/energy[l,h])}
    np.savez_compressed(args.root/'results/adaln_carriers_wan.npz',steps=steps,key_carriers=carriers,
                        query_carriers=query_carriers,bias_only=biases,observed=observed,
                        alignment=alignment,bias_alignment=bias_alignment)
    (args.root/'results/adaln_carriers_wan.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
