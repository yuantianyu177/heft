"""Verify actual model-parameter QK gauge transformations preserve denoising."""
import argparse, json
from pathlib import Path
import torch
from safetensors.torch import load_file
from research_rope import frequency_pairs, pair_channels


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,default=7)
    parser.add_argument('--root',type=Path,default=Path('/share/yty/heft'))
    parser.add_argument('--names',default='gold-fish,drift-straight,kite-surf')
    parser.add_argument('--gauge-file',type=Path)
    args=parser.parse_args();torch.set_num_threads(3)
    device=f'cuda:{args.gpu}';torch.cuda.set_device(device)
    from diffusers import WanTransformer3DModel
    path=json.loads((args.root/'models/Wan2.1-T2V-1.3B-Diffusers.json').read_text())['path']
    model=WanTransformer3DModel.from_pretrained(path,subfolder='transformer',torch_dtype=torch.bfloat16,
                                               local_files_only=True).to(device).eval()
    prompt=load_file(str(args.root/'models/wan_empty_prompt.safetensors'))['prompt_embeds'].to(device)
    original=[(block.attn1.norm_q.weight.detach().clone(),block.attn1.norm_k.weight.detach().clone()) for block in model.blocks]
    high=pair_channels(frequency_pairs(0,.15)).to(device)
    low=pair_channels(frequency_pairs(.15,1)).to(device)
    gauges={}
    for name,indices in [('fast16',high),('slow16',low)]:
        gauge=torch.ones(128,device=device);gauge[indices]=16
        gauges[name]=gauge.repeat(30,12)
    if args.gauge_file:
        learned=json.loads(args.gauge_file.read_text())
        gauge=torch.ones((30,12,128),device=device)
        for (layer,head),scale in zip(learned['head_ids'],learned['scales_pow2'],strict=True):
            gauge[layer,head]=torch.tensor(scale,device=device).repeat_interleave(2)
        gauges={'calibrated_pow2':gauge.flatten(1)}
    results=[]
    for name in args.names.split(','):
        noisy=load_file(str(args.root/f'latents/wan_real_t25/{name}.safetensors'))['noisy'].to(device)
        meta=json.loads((args.root/f'features/wan_real_t25/{name}.json').read_text())
        timestep=torch.tensor([meta['noise_timestep']],device=device)
        def forward():return model(noisy,timestep=timestep,encoder_hidden_states=prompt,return_dict=False)[0]
        for block,(q,k) in zip(model.blocks,original,strict=True):
            block.attn1.norm_q.weight.copy_(q);block.attn1.norm_k.weight.copy_(k)
        baseline=forward()
        for label,gauge in gauges.items():
            for layer,(block,(q,k)) in enumerate(zip(model.blocks,original,strict=True)):
                block.attn1.norm_q.weight.copy_(q/gauge[layer])
                block.attn1.norm_k.weight.copy_(k*gauge[layer])
            changed=forward();difference=changed.float()-baseline.float()
            row={'video':name,'gauge':label,'maximum_absolute_output_error':float(difference.abs().max()),
                 'relative_l2_output_error':float(difference.norm()/baseline.float().norm()),
                 'fraction_output_entries_bitwise_equal':float((changed==baseline).float().mean()),
                 'nonidentity_layers':int((gauge!=1).any(-1).sum()),
                 'all_30_layers_transformed':bool((gauge!=1).any(-1).all()),
                 'precision':'bfloat16; powers of two prevent multiplier quantization',
                 'saved_model_files_modified':False}
            print(json.dumps(row),flush=True);results.append(row)
    for block,(q,k) in zip(model.blocks,original,strict=True):
        block.attn1.norm_q.weight.copy_(q);block.attn1.norm_k.weight.copy_(k)
    output='gauge_model_preservation_learned_wan.json' if args.gauge_file else 'gauge_model_preservation_wan.json'
    (args.root/'results'/output).write_text(json.dumps(results,indent=2))


if __name__=='__main__':main()
