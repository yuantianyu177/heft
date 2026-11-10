"""Extract reproducible Wan descriptor populations for causal readout probes.

Uses the repository's frame-aligned VAE and final-noise-step convention.
The conditional forward is identical to the pipeline's conditional forward;
the unconditional CFG pass and unused denoising output are not calculated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors.torch import load_file, save_file


def resolve_model() -> str:
    meta = Path("/share/yty/heft/models/Wan2.1-T2V-1.3B-Diffusers.json")
    return json.loads(meta.read_text())["path"]


@torch.inference_mode()
def prepare_prompt(model_path: str, output: Path, device: str):
    from transformers import AutoTokenizer, UMT5EncoderModel
    tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer", local_files_only=True)
    encoder = UMT5EncoderModel.from_pretrained(
        model_path, subfolder="text_encoder", torch_dtype=torch.bfloat16, local_files_only=True,
    ).to(device).eval()
    text = tokenizer([""], padding="max_length", max_length=512, truncation=True,
                     add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
    embeds = encoder(text.input_ids.to(device), text.attention_mask.to(device)).last_hidden_state
    length = int(text.attention_mask.sum())
    embeds[:, length:] = 0
    save_file({"prompt_embeds": embeds.cpu().contiguous()}, str(output))
    print(json.dumps({"prompt_file": str(output), "length": length, "shape": list(embeds.shape)}), flush=True)


def transformed_video(video: np.ndarray, frames: int, transform: str, motion_seed: int = 0):
    if transform == "real":
        return video[:frames], None
    base = video[0]
    h, w = base.shape[:2]
    if transform in ("translate", "translate_appearance"):
        dx = np.round(np.linspace(0, w / 8, frames)).astype(int)
        dy = np.round(h / 16 * np.sin(np.linspace(0, np.pi, frames))).astype(int)
    elif transform == "static":
        dx = np.zeros(frames, dtype=int)
        dy = np.zeros(frames, dtype=int)
    elif transform == "reverse_translate":
        dx = -np.round(np.linspace(0, w / 8, frames)).astype(int)
        dy = -np.round(h / 16 * np.sin(np.linspace(0, np.pi, frames))).astype(int)
    elif transform == 'diverse_motion':
        rng=np.random.default_rng(motion_seed)
        phase=np.linspace(0,1,frames)
        components=[]
        for size in [w,h]:
            drift=rng.uniform(-1,1);amplitude=rng.uniform(-.8,.8);frequency=rng.uniform(.7,2.2)
            curve=drift*phase+amplitude*np.sin(np.pi*frequency*phase)
            curve=curve/max(1.,np.abs(curve).max())
            components.append((curve*size*rng.uniform(.07,.15)).astype(np.float32))
        dx,dy=components
    else:
        raise ValueError(transform)
    shifts = np.stack((dx, dy), axis=-1)
    result = np.stack([
        cv2.warpAffine(base, np.array([[1, 0, x], [0, 1, y]], dtype=np.float32), (w, h),
                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        for x, y in shifts
    ])
    if transform in ("translate_appearance",'diverse_motion'):
        generator = np.random.default_rng(2398)
        for t in range(1, frames):
            phase = t / (frames - 1)
            gain = np.array([1 + .2*np.sin(phase*4), 1 - .15*np.sin(phase*3), 1 + .2*np.cos(phase*5)])
            altered = result[t].astype(np.float32) * gain + 12*np.sin(phase*5)
            sigma = 1.25 * (1 - np.cos(phase*5)) / 2
            altered = cv2.GaussianBlur(altered, (5, 5), sigmaX=max(.05, sigma))
            altered += generator.normal(0, 4, altered.shape)
            result[t] = np.clip(altered, 0, 255).astype(np.uint8)
    return result, shifts.tolist()


@torch.inference_mode()
def extract(args):
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanTransformer3DModel
    from diffusers.utils.torch_utils import randn_tensor

    torch.set_num_threads(4)
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cv2.setNumThreads(1)
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)
    model_path = args.model_path or resolve_model()
    dtype = torch.bfloat16
    vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype,
                                          local_files_only=True).to(device).eval()
    transformer = WanTransformer3DModel.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=dtype, local_files_only=True,
    ).to(device).eval()
    if args.rope_temporal_scale!=1.:
        from research_rope import wan_angles
        def scaled_rope(hidden_states):
            _,_,frames,height,width=hidden_states.shape
            pt,ph,pw=transformer.config.patch_size
            angles=wan_angles(frames//pt,height//ph,width//pw,device=hidden_states.device)
            angles[...,:22]*=args.rope_temporal_scale
            return torch.polar(torch.ones_like(angles),angles).reshape(1,1,-1,64)
        transformer.rope.forward=scaled_rope
    scheduler = UniPCMultistepScheduler.from_pretrained(model_path, subfolder="scheduler", local_files_only=True)
    scheduler.set_timesteps(50, device=device)
    timestep = scheduler.timesteps[49]
    prompt = load_file(str(args.root / "models/wan_empty_prompt.safetensors"))["prompt_embeds"].to(device)
    manifest = json.loads((args.root / "data/davis_records/manifest.json").read_text())
    if args.names:
        manifest = [r for r in manifest if r["name"] in args.names.split(",")]
    if args.split != "all":
        manifest = [r for r in manifest if r["split"] == args.split]
    if args.chunked:
        if args.transform != "real":
            raise ValueError("chunked extraction is only defined for real videos")
        manifest = [dict(r, start_frame=start, valid_frames=min(args.frames, r["frames"]-start),
                         storage_name=f"{r['name']}_c{start//args.frames:03d}")
                    for r in manifest for start in range(0, r["frames"], args.frames)]
    manifest = manifest[args.shard::args.shards]
    output_tag = args.output_tag or f"wan_{args.transform}_t{args.frames}"
    if args.output_tag is None and args.latent_normalization == "pretrained":
        output_tag += "_normalized"
    if args.output_tag is None and args.seed != 42:
        output_tag += f"_seed{args.seed}"
    destination = args.root / "features" / output_tag
    destination.mkdir(exist_ok=True, parents=True)
    height, width = args.height, args.width
    fh, fw = height // 16, width // 16
    dense_heads = {15: [2], 18: [1, 5], 20: [8]}
    if args.dense_heads:
        dense_heads = {}
        for item in args.dense_heads.split(","):
            layer, head = map(int, item.split(":"))
            dense_heads.setdefault(layer, []).append(head)
    sparse = sorted(set(t for t in (0, 1, 4, 8, 12, 24) if t < args.frames))
    state = {}

    class ReadoutComplete(Exception):
        pass

    def make_capture(layer):
        def capture(*, query, key, hidden_states, **kwargs):
            if not state.get("active"):
                return
            if args.dense_only and layer not in dense_heads:
                return
            q = query[0].reshape(query.shape[1], args.frames, fh, fw, -1)
            k = key[0].reshape(key.shape[1], args.frames, fh, fw, -1)
            if not args.dense_only:
                state["tensors"][f"l{layer:02d}_key"] = k[:, sparse].cpu().contiguous()
                state["tensors"][f"l{layer:02d}_query"] = q[:, sparse].cpu().contiguous()
            if layer in dense_heads:
                for head in dense_heads[layer]:
                    state["tensors"][f"l{layer:02d}_h{head:02d}_dense_key"] = k[head].cpu().contiguous()
            if args.dense_only and layer == max(dense_heads):
                raise ReadoutComplete
        return capture

    for layer, block in enumerate(transformer.blocks):
        block.attn1._heft_capture = make_capture(layer)

    for record in manifest:
        name = record["name"]
        storage_name = record.get("storage_name", name)
        output = destination / f"{storage_name}.safetensors"
        metadata_path = output.with_suffix(".json")
        if output.exists() and metadata_path.exists() and not args.overwrite:
            print(f"SKIP {name}", flush=True)
            continue
        start = time.monotonic()
        video = np.load(args.root / f"data/davis_records/{name}.video.npy", mmap_mode="r")
        if len(video) < args.frames and args.transform == "real" and not args.chunked:
            print(f"SKIP_SHORT {name}", flush=True)
            continue
        start_frame = record.get("start_frame", 0)
        input_video = video[start_frame:start_frame+args.frames] if args.chunked else video
        motion_seed=args.motion_seed+int(hashlib.sha256(name.encode()).hexdigest()[:8],16)
        transformed, shifts = transformed_video(input_video, args.frames, args.transform,motion_seed)
        if len(transformed) < args.frames:
            transformed = np.concatenate((transformed, np.repeat(transformed[-1:], args.frames-len(transformed), axis=0)))
        resized = np.stack([cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
                            for frame in transformed])
        tensor = torch.from_numpy(resized).permute(3, 0, 1, 2).unsqueeze(0).to(device, dtype=dtype)
        tensor = tensor.div(127.5).sub(1)
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        posterior=vae.encode(tensor).latent_dist
        latent=(posterior.mode() if args.posterior_mode else posterior.sample(generator=generator)).to(dtype=dtype)
        assert latent.shape[2:] == (args.frames, height // 8, width // 8), latent.shape
        if args.latent_normalization == "pretrained":
            mean = torch.tensor(vae.config.latents_mean, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
            std = torch.tensor(vae.config.latents_std, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
            latent = (latent - mean) / std
        noise = randn_tensor(latent.shape, generator=generator, device=device,
                             dtype=torch.float32 if args.noise_float32 else dtype).to(dtype)
        noisy = scheduler.add_noise(latent, noise, timestep.unsqueeze(0))
        # Preserve exact inputs for noise/normalization robustness experiments.
        latent_output = args.root / "latents" / output_tag
        latent_output.mkdir(exist_ok=True, parents=True)
        save_file({"latent": latent.cpu().contiguous(), "noisy": noisy.cpu().contiguous()},
                  str(latent_output / f"{storage_name}.safetensors"))
        state["tensors"] = {}
        state["active"] = True
        try:
            transformer(noisy, timestep=timestep.expand(1), encoder_hidden_states=prompt, return_dict=False)
        except ReadoutComplete:
            pass
        state["active"] = False
        torch.cuda.synchronize()
        elapsed_forward = time.monotonic() - start
        save_file(state["tensors"], str(output), metadata={"kind": "post_rope_conditional"})
        metadata = dict(
            video=name, split=record["split"], transform=args.transform, frames=args.frames,
            storage_name=storage_name, start_frame=start_frame, valid_frames=record.get("valid_frames", args.frames),
            source_size=list(video.shape[1:3]), input_size=[height, width], feature_size=[fh, fw],
            sparse_frames=sparse, dense_heads=dense_heads, layers=len(transformer.blocks),
            heads=transformer.blocks[0].attn1.heads, seed=args.seed, noise_timestep=float(timestep),
            noise_sigma=float(scheduler.sigmas[49]), model_path=model_path, dtype=str(dtype),
            rope_temporal_scale=args.rope_temporal_scale,
            latent_normalization=args.latent_normalization,
            posterior='mode' if args.posterior_mode else 'sample',
            noise_sampling_dtype='float32' if args.noise_float32 else str(dtype),
            vae_temporal_compression="repository patched VAE: 1", shifts=shifts,
            motion_seed=motion_seed if args.transform=='diverse_motion' else None,
            forward_seconds=elapsed_forward, total_seconds=time.monotonic()-start,
            peak_gpu_bytes=torch.cuda.max_memory_allocated(device),
            extractor_source_sha256=source_hash,
        )
        metadata_path.write_text(json.dumps(metadata, indent=2))
        state["tensors"] = {}
        print(json.dumps(metadata), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/share/yty/heft"))
    parser.add_argument("--model-path")
    parser.add_argument("--output-tag")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--frames", type=int, default=25)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--motion-seed',type=int,default=0)
    parser.add_argument('--rope-temporal-scale',type=float,default=1.)
    parser.add_argument('--posterior-mode',action='store_true')
    parser.add_argument('--noise-float32',action='store_true')
    parser.add_argument("--latent-normalization", choices=["repository", "pretrained"], default="repository")
    parser.add_argument("--split", choices=["all", "calibration", "evaluation"], default="all")
    parser.add_argument("--names")
    parser.add_argument("--dense-heads")
    parser.add_argument("--dense-only", action="store_true")
    parser.add_argument("--chunked", action="store_true")
    parser.add_argument("--transform", choices=["real", "translate", "translate_appearance", "reverse_translate", "static",'diverse_motion'], default="real")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prepare-prompt", action="store_true")
    args = parser.parse_args()
    if args.prepare_prompt:
        prepare_prompt(args.model_path or resolve_model(), args.root / "models/wan_empty_prompt.safetensors", f"cuda:{args.gpu}")
    else:
        extract(args)


if __name__ == "__main__":
    main()
