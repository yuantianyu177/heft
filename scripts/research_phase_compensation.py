"""Test learned phase compensation with position-matched cross-video controls.

No improvement is assumed. A pre-RoPE vector still contains all preceding
layers' positional computations. True and unrelated cross-video pairs have
identical token coordinates, time gaps, and explicit RoPE phase differences.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from research_rope import wan_angles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/share/yty/heft"))
    parser.add_argument("--features", default="wan_real_t25")
    parser.add_argument("--layers", default="13,15,18,20")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(3)
    source = args.root/"features"/args.features
    paths = sorted(source.glob("*.safetensors"))
    layers = list(map(int, args.layers.split(",")))
    metadata = [json.loads(p.with_suffix(".json").read_text()) for p in paths]
    h, w = metadata[0]["feature_size"]
    frames = metadata[0]["sparse_frames"]
    angles = wan_angles(metadata[0]["frames"], h, w)[frames].numpy()
    rotation = np.exp(1j*angles).astype(np.complex64)
    names = [p.stem for p in paths]
    betas = np.linspace(-1, 2, 61)
    modes = ["raw", "global_mean_removed", "position_template_removed"]
    results = []
    for layer in layers:
        # A position-dependent, entirely annotation-free template.
        features = []
        for path in paths:
            with safe_open(path, framework="pt", device="cpu") as file:
                value = file.get_tensor(f"l{layer:02d}_key").float().numpy()
            z = (value[..., 0::2] + 1j*value[..., 1::2]).astype(np.complex64)
            features.append(z / rotation[None])
        template = np.mean(features, axis=0)
        global_mean = template.mean((1, 2, 3))
        layer_records = []
        for video_index, (path, m, pre) in enumerate(zip(paths, metadata, features, strict=True)):
            name = path.stem
            unrelated = features[(video_index+max(1,len(paths)//2)) % len(paths)]
            if args.synthetic:
                yy, xx = np.meshgrid(np.linspace(.2,.8,5), np.linspace(.2,.8,5), indexing="ij")
                initial = np.stack((xx.ravel(), yy.ravel()), -1).astype(np.float32)
                sh, sw = m["source_size"]
                shift = np.array(m["shifts"], np.float32)[frames] / np.array([sw, sh])
                points = initial[None] + shift[:,None]
                valid = ((points>0)&(points<1)).all(-1)[1:]
            else:
                with np.load(args.root/f"data/davis_records/{name}.annotations.npz") as ann:
                    selected = ~ann["occluded"][:,0]
                    points = ann["points"][selected][:,frames].transpose(1,0,2)
                    valid = (~ann["occluded"][selected][:,frames].T)[1:]
            xy = np.floor(points*np.array([w,h])).astype(int)
            xy[...,0] = np.clip(xy[...,0],0,w-1); xy[...,1] = np.clip(xy[...,1],0,h-1)
            tx, ty = xy[...,0], xy[...,1]
            reference = pre[:,0,ty[0],tx[0]]
            target = np.stack([pre[:,t,ty[t],tx[t]] for t in range(1,len(frames))],axis=1)
            null = np.stack([unrelated[:,t,ty[t],tx[t]] for t in range(1,len(frames))],axis=1)
            template_reference = template[:,0,ty[0],tx[0]]
            template_target = np.stack([template[:,t,ty[t],tx[t]] for t in range(1,len(frames))],axis=1)
            delta = np.stack([angles[t,ty[t],tx[t]]-angles[0,ty[0],tx[0]] for t in range(1,len(frames))])
            records = []
            for mode in modes:
                q, k, control = reference.copy(), target.copy(), null.copy()
                if mode == "global_mean_removed":
                    q -= global_mean[:,None]; k -= global_mean[:,None,None]; control -= global_mean[:,None,None]
                if mode == "position_template_removed":
                    q -= template_reference; k -= template_target; control -= template_target
                group = []
                for key in (k, control):
                    cross = np.conj(q[:,None])*key
                    amplitude = np.abs(q[:,None])*np.abs(key)
                    unit = cross / np.maximum(amplitude,1e-8)
                    mask = valid[None,:,:,None]
                    count = mask.sum((1,2))
                    pre_coherence = (unit.real*mask).sum((1,2))/count
                    post_coherence = ((unit*np.exp(1j*delta)[None]).real*mask).sum((1,2))/count
                    # Weighted by pair amplitudes, separately report exact dot contributions.
                    weighted_pre = (cross.real*mask).sum((1,2))/np.maximum((amplitude*mask).sum((1,2)),1e-8)
                    weighted_post = ((cross*np.exp(1j*delta)[None]).real*mask).sum((1,2))/np.maximum((amplitude*mask).sum((1,2)),1e-8)
                    curves = []
                    for axis in (slice(0,22),slice(22,43),slice(43,64)):
                        # Curves identify whether 0 (content stable) or 1 (RoPE compensated) aligns phases.
                        curves.append(np.stack([
                            ((unit[...,axis]*np.exp(1j*b*delta[...,axis])[None]).real*mask).mean(-1).sum((1,2))/count[:,0]
                            for b in betas
                        ],axis=-1))
                    group.append(dict(pre=pre_coherence, post=post_coherence,
                                      weighted_pre=weighted_pre, weighted_post=weighted_post,
                                      curves=np.stack(curves,axis=1)))
                records.append(group)
            layer_records.append(records)
        # [video, mode, true/control, head, ...]
        keys = ["pre","post","weighted_pre","weighted_post","curves"]
        arrays = {key: np.array([[[g[key] for g in groups] for groups in record] for record in layer_records]) for key in keys}
        output = args.root/"results"/f"phase_compensation_{args.features}_l{layer:02d}.npz"
        np.savez_compressed(output, **arrays, names=np.array(names), modes=np.array(modes), betas=betas)
        summary = {"layer":layer,"videos":len(paths),"modes":{},"control":"unrelated video, identical source/target token coordinates and time gap"}
        for mi, mode in enumerate(modes):
            summary["modes"][mode] = {}
            for head in range(template.shape[0]):
                entry = {}
                for ci, label in enumerate(("true","unrelated")):
                    pre_c = arrays["pre"][:,mi,ci,head].mean(0)
                    post_c = arrays["post"][:,mi,ci,head].mean(0)
                    curve = arrays["curves"][:,mi,ci,head].mean(0)
                    entry[label] = {"pre_axis": [float(pre_c[s].mean()) for s in (slice(0,22),slice(22,43),slice(43,64))],
                                    "post_axis": [float(post_c[s].mean()) for s in (slice(0,22),slice(22,43),slice(43,64))],
                                    "best_beta": [float(betas[i]) for i in curve.argmax(-1)]}
                summary["modes"][mode][f"H{head}"] = entry
        output.with_suffix(".json").write_text(json.dumps(summary,indent=2))
        for head in ([1,5,8] if layer==18 else [2,8]):
            if head < template.shape[0]:
                print(json.dumps({"layer":layer,"head":head,"results":{mode:summary["modes"][mode][f"H{head}"] for mode in modes}}),flush=True)


if __name__ == "__main__":
    main()
