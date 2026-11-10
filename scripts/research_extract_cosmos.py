"""Cosmos-Predict2 feature probes using the repository's conditional pipeline."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors.torch import load_file, save_file

from research_cosmos_rope import adjacent_storage
from research_extract_wan import transformed_video


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/share/yty/heft"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--names")
    parser.add_argument("--split", default="all", choices=["all", "calibration", "evaluation"])
    parser.add_argument("--transform", default="real", choices=["real", "translate", "translate_appearance", "static"])
    parser.add_argument("--prepare-prompt", action="store_true")
    parser.add_argument("--dense-only", action="store_true")
    parser.add_argument("--chunked", action="store_true")
    parser.add_argument("--dense-heads", default="18:6")
    parser.add_argument("--output-tag")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)
    model = json.loads((args.root/"models/Cosmos-Predict2-2B-Video2World.json").read_text())["path"]
    dtype = torch.bfloat16
    prompt_file = args.root/"models/cosmos_empty_prompt.safetensors"
    if args.prepare_prompt:
        from transformers import T5EncoderModel, T5TokenizerFast
        tokenizer = T5TokenizerFast.from_pretrained(model, subfolder="tokenizer", local_files_only=True)
        encoder = T5EncoderModel.from_pretrained(model, subfolder="text_encoder", torch_dtype=dtype,
                                                  local_files_only=True).to(device).eval()
        text = tokenizer([""], padding="max_length", max_length=512, truncation=True, return_tensors="pt")
        embedding = encoder(text.input_ids.to(device), attention_mask=text.attention_mask.to(device)).last_hidden_state
        embedding[:, int(text.attention_mask.sum()):] = 0
        save_file({"prompt_embeds": embedding.cpu().contiguous()}, str(prompt_file))
        print(json.dumps({"prompt_file": str(prompt_file), "shape": list(embedding.shape)}), flush=True)
        return
    from diffusers import Cosmos2VideoToWorldPipeline
    pipeline = Cosmos2VideoToWorldPipeline.from_pretrained(
        model, text_encoder=None, tokenizer=None, torch_dtype=dtype, local_files_only=True,
    ).to(device)
    pipeline.set_progress_bar_config(disable=True)
    prompt = load_file(str(prompt_file))["prompt_embeds"].to(device)
    manifest = json.loads((args.root/"data/davis_records/manifest.json").read_text())
    if args.names:
        manifest = [r for r in manifest if r["name"] in args.names.split(",")]
    if args.split != "all":
        manifest = [r for r in manifest if r["split"] == args.split]
    nt, height, width = 25, 704, 1280
    h, w = height//16, width//16
    if args.chunked:
        assert args.transform == "real"
        manifest = [dict(r, start_frame=start, valid_frames=min(nt, r["frames"]-start),
                         storage_name=f"{r['name']}_c{start//nt:03d}")
                    for r in manifest for start in range(0, r["frames"], nt)]
    manifest = manifest[args.shard::args.shards]
    tag = args.output_tag or f"cosmos_{args.transform}_t25"
    destination = args.root/"features"/tag
    destination.mkdir(exist_ok=True, parents=True)
    dense_heads = {}
    for item in args.dense_heads.split(","):
        layer, head = map(int, item.split(":"))
        dense_heads.setdefault(layer, []).append(head)
    state = {"active": False}
    sparse = [0, 1, 4, 8, 12, 24]
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    class ReadoutComplete(Exception):
        pass

    def callback(layer):
        def capture(*, query, key, hidden_states, **kwargs):
            if not state["active"] or args.dense_only and layer not in dense_heads:
                return
            q = query[0].reshape(query.shape[1], nt, h, w, 128)
            k = key[0].reshape(key.shape[1], nt, h, w, 128)
            if not args.dense_only:
                state["tensors"][f"l{layer:02d}_query"] = adjacent_storage(q[:, sparse]).cpu().contiguous()
                state["tensors"][f"l{layer:02d}_key"] = adjacent_storage(k[:, sparse]).cpu().contiguous()
            for head in dense_heads.get(layer, []):
                state["tensors"][f"l{layer:02d}_h{head:02d}_dense_key"] = adjacent_storage(k[head]).cpu().contiguous()
            if args.dense_only and layer == max(dense_heads):
                raise ReadoutComplete
        return capture

    for layer, block in enumerate(pipeline.transformer.transformer_blocks):
        block.attn1._heft_capture = callback(layer)
    for record in manifest:
        name = record["name"]
        storage_name = record.get("storage_name", name)
        path = destination/f"{storage_name}.safetensors"
        if path.exists() and path.with_suffix(".json").exists() and not args.overwrite:
            continue
        start = time.monotonic()
        video = np.load(args.root/f"data/davis_records/{name}.video.npy", mmap_mode="r")
        start_frame = record.get("start_frame", 0)
        input_video = video[start_frame:start_frame+nt] if args.chunked else video
        frames, shifts = transformed_video(input_video, nt, args.transform)
        if len(frames) < nt:
            frames = np.concatenate((frames, np.repeat(frames[-1:], nt-len(frames), axis=0)))
        frames = np.stack([cv2.resize(f, (width, height), interpolation=cv2.INTER_LANCZOS4) for f in frames])
        prepared = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div(255)
        state["tensors"] = {}
        state["active"] = True
        try:
            pipeline(prompt_embeds=prompt, height=height, width=width, num_frames=nt,
                     num_inference_steps=35, start_step=34, guidance_scale=1.0, fps=16,
                     generator=torch.Generator(device="cpu").manual_seed(42), input_video=prepared,
                     output_type="latent")
        except ReadoutComplete:
            pass
        finally:
            state["active"] = False
        torch.cuda.synchronize()
        forward_seconds = time.monotonic()-start
        save_file(state["tensors"], str(path))
        metadata = dict(video=name, split=record["split"], transform=args.transform, backbone="cosmos2",
                        storage_name=storage_name, start_frame=start_frame,
                        valid_frames=record.get("valid_frames", nt), frames=nt,
                        input_size=[height, width], feature_size=[h, w], source_size=list(video.shape[1:3]),
                        sparse_frames=sparse, dense_heads=dense_heads, layers=28, heads=16,
                        seed=42, fps=16, storage_pairing="adjacent", model_pairing="split_half",
                        model_path=model, latent_normalization="repository Cosmos pipeline: VAE mean/std normalized",
                        extractor_source_sha256=source_hash, shifts=shifts, dtype=str(dtype),
                        forward_seconds=forward_seconds, total_seconds=time.monotonic()-start,
                        peak_gpu_bytes=torch.cuda.max_memory_allocated(device))
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
        state["tensors"] = {}
        print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
