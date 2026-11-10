"""Apply the measured readout clock to CogVideoX-2B's independent codec."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import load_file, save_file

import research_cvpr_readout as common


ROOT = Path('/share/yty/heft')
CHECKPOINT = Path('/share/yty/cache/modelscope-cache/heft-cvpr27/AI-ModelScope--CogVideoX-2b')
REVISION = ROOT / 'cvpr27'
PROMPT = REVISION / 'cogvideo_empty_prompt.safetensors'
HEADS = 30
LAYER = 18
OFFSET = .75


class CogExtractor(common.Extractor):
    def __init__(self, root, name, device, head, variants, seed):
        from diffusers import AutoencoderKLCogVideoX, CogVideoXDDIMScheduler, CogVideoXTransformer3DModel

        self.name = 'cogvideox-2b'
        self.device, self.head, self.seed = device, head, seed
        self.height, self.width = 480, 720
        self.h, self.w = 30, 45
        self.dtype = torch.bfloat16
        self.modes = {v[0] for v in variants}
        self.capture_all = any(v[1] == 'layer' for v in variants)
        if not self.modes <= {'native1', 'native4', 'framewise'} or any(v[1] not in {'head', 'layer'} for v in variants):
            raise ValueError('CogVideoX-2B has no rotary frequency bands in this checkpoint')
        weights = dict(torch_dtype=self.dtype, local_files_only=True)
        self.native = AutoencoderKLCogVideoX.from_pretrained(CHECKPOINT, subfolder='vae', **weights).to(device).eval()
        assert self.native.config.temporal_compression_ratio == 4
        self.encoder_hash = hashlib.sha256(Path(sys.modules[AutoencoderKLCogVideoX.__module__].__file__).read_bytes()).hexdigest()
        self.model = CogVideoXTransformer3DModel.from_pretrained(CHECKPOINT, subfolder='transformer', **weights).to(device).eval()
        assert not self.model.config.use_rotary_positional_embeddings
        self.scheduler = CogVideoXDDIMScheduler.from_pretrained(CHECKPOINT, subfolder='scheduler', timestep_spacing='trailing', local_files_only=True)
        self.scheduler.set_timesteps(50, device=device)
        self.step = self.scheduler.timesteps[-1:]
        self.sigma = float((1 - self.scheduler.alphas_cumprod[int(self.step.item())]).sqrt())
        self.prompt = load_file(str(PROMPT))['embedding'].to(device)
        self.model.transformer_blocks[LAYER].attn1._heft_capture = self.capture
        self.captured = None

    def capture(self, *, key, **kwargs):
        keys = key[0] if self.capture_all else key[0, self.head:self.head+1]
        self.captured = keys.reshape(keys.shape[0], -1, self.h, self.w, 64).contiguous()
        raise common.CaptureDone

    def denoise(self, clean, noise, times):
        noisy = self.scheduler.add_noise(clean, noise[:, :, times], self.step).to(self.dtype)
        try:
            self.model(hidden_states=noisy.permute(0, 2, 1, 3, 4), encoder_hidden_states=self.prompt,
                       timestep=self.step, return_dict=False)
        except common.CaptureDone:
            feature, self.captured = self.captured, None
            return feature
        raise RuntimeError('Failed to capture CogVideoX attention keys')

    def encode(self, vae, frames):
        return vae.encode(frames.contiguous()).latent_dist.mode().float() * vae.config.scaling_factor

    def preprocess(self, raw):
        resized = np.stack([cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LANCZOS4) for frame in raw])
        if len(resized) < common.CHUNK:
            resized = np.concatenate((resized, np.repeat(resized[-1:], common.CHUNK-len(resized), axis=0)))
        inputs = torch.from_numpy(resized).permute(3, 0, 1, 2)[None].to(self.device, dtype=self.dtype)/127.5 - 1
        noise = torch.randn((1, 16, common.CHUNK, self.h*2, self.w*2),
                            generator=torch.Generator().manual_seed(self.seed)).to(self.device)
        return inputs, noise

    def get(self, raw):
        inputs, noise = self.preprocess(raw)
        maps, profile = {}, {}
        if self.modes & {'native1', 'native4'}:
            phase_features = []
            phases = range(4) if 'native4' in self.modes else range(1)
            for phase in phases:
                torch.cuda.reset_peak_memory_stats(self.device)
                began = time.monotonic()
                prefix = 4 - phase if phase else 0
                clip = torch.cat((inputs[:, :, :1].expand(-1, -1, prefix, -1, -1), inputs), dim=2)
                clip = torch.cat((clip, clip[:, :, -1:].expand(-1, -1, 33-clip.shape[2], -1, -1)), dim=2)
                clean = self.encode(self.native, clip)[:, :, :7]
                phase_features.append(self.denoise(clean, noise, common.PHASES[phase]))
                torch.cuda.synchronize(self.device)
                profile[f'phase{phase}_s'] = time.monotonic() - began
                profile[f'phase{phase}_peak_bytes'] = torch.cuda.max_memory_allocated(self.device)
            if 'native1' in self.modes:
                maps['native1'] = common.interpolate(phase_features[0], common.PHASES[0], range(common.CHUNK)).to(self.dtype)
            if 'native4' in self.modes:
                heads = phase_features[0].shape[0]
                dense = torch.empty((heads, common.CHUNK, self.h, self.w, 64), device=self.device, dtype=self.dtype)
                for indices, values in zip(common.PHASES, phase_features, strict=True):
                    dense[:, indices] = values
                # This codec's within-chunk normalization can change the anchor.
                dense[:, 0] = torch.stack([values[:, 0] for values in phase_features]).float().mean(0)
                maps['native4'] = dense
        if 'framewise' in self.modes:
            torch.cuda.reset_peak_memory_stats(self.device)
            began = time.monotonic()
            clean = self.framewise(inputs)
            maps['framewise'] = self.denoise(clean, noise, list(range(common.CHUNK)))
            torch.cuda.synchronize(self.device)
            profile['framewise_s'] = time.monotonic() - began
            profile['framewise_peak_bytes'] = torch.cuda.max_memory_allocated(self.device)
        return maps, profile


@torch.inference_mode()
def prepare_prompt(device):
    from transformers import T5EncoderModel, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(CHECKPOINT, subfolder='tokenizer', local_files_only=True)
    text = T5EncoderModel.from_pretrained(CHECKPOINT, subfolder='text_encoder', torch_dtype=torch.bfloat16,
                                          local_files_only=True).to(device).eval()
    ids = tokenizer([''], padding='max_length', max_length=226, truncation=True, return_tensors='pt').input_ids
    embedding = text(ids.to(device))[0].cpu().contiguous()
    assert embedding.shape == (1, 226, 4096)
    save_file({'embedding': embedding}, str(PROMPT))


@torch.inference_mode()
def calibrate_head(device):
    cv2.setNumThreads(1)
    torch.set_num_threads(3)
    obj = CogExtractor(ROOT, 'cog', device, 0, [('native1', 'layer', 0.)], 42)
    bank = REVISION/'independent_clock'
    inputs = json.loads((bank/'images.json').read_text())
    ys, xs = torch.meshgrid(torch.linspace(.3, .7, 12, device=device), torch.linspace(.3, .7, 12, device=device), indexing='ij')
    points = torch.stack((xs, ys), dim=-1).reshape(-1, 2)
    thresholds = torch.tensor([1, 2, 4, 8, 16], device=device)
    rows = []
    for row in inputs:
        image = np.load(bank/'data/davis_records'/f'{row["name"]}.video.npy', mmap_mode='r')[0]
        image = cv2.resize(image, (obj.width, obj.height), interpolation=cv2.INTER_LANCZOS4)
        for vx, vy in ((4, 0), (-4, 0), (0, 4), (0, -4)):
            raw = [cv2.warpAffine(image, np.float32([[1, 0, vx*t], [0, 1, vy*t]]), (obj.width, obj.height),
                                  borderMode=cv2.BORDER_REFLECT_101) for t in range(common.CHUNK)]
            frames, noise = obj.preprocess(raw)
            clean = obj.encode(obj.native, frames)
            features = obj.denoise(clean, noise, common.PHASES[0]).float()
            query = F.grid_sample(features[:, 0].permute(0, 3, 1, 2),
                                  (points[None, :, None]*2-1).expand(HEADS, -1, -1, -1), align_corners=False)
            target = F.normalize(features[:, 1:].flatten(2, 3), dim=-1)
            similarity = torch.einsum('hdp,htsd->htps', F.normalize(query[..., 0], dim=1), target)
            ids = similarity.argmax(-1)
            pred = torch.stack(((ids % obj.w + .5)/obj.w, (ids // obj.w + .5)/obj.h), dim=-1)
            nominal = torch.tensor(common.PHASES[0][1:], device=device)[:, None, None]
            step = torch.tensor([vx/obj.width, vy/obj.height], device=device)
            gt = points[None] + (nominal - OFFSET) * step
            error = ((pred - gt[None]) * 256).norm(dim=-1)
            scores = (error[..., None] < thresholds).float().mean((1, 2, 3)).cpu().tolist()
            rows.append(dict(name=row['name'], velocity=[vx, vy], per_head_pck=scores))
        print(row['name'], flush=True)
    ranks = np.array([row['per_head_pck'] for row in rows]).mean(0)
    path = REVISION/'protocols/cog_head_calibration.json'
    path.write_text(json.dumps(dict(head=int(ranks.argmax()), layer=LAYER, tau=OFFSET, scores=ranks.tolist(),
                                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), rows=rows), indent=2))
    print('Frozen head', int(ranks.argmax()), flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in {'prepare-prompt', 'calibrate-head'}:
        cmd = sys.argv.pop(1)
        args = argparse.ArgumentParser()
        args.add_argument('--gpu', type=int, required=True)
        gpu = args.parse_args().gpu
        torch.cuda.set_device(gpu)
        (prepare_prompt if cmd == 'prepare-prompt' else calibrate_head)(torch.device(f'cuda:{gpu}'))
    else:
        common.CHECKPOINTS['cog'] = CHECKPOINT.name
        common.PRIMARY_HEADS['cog'] = json.loads((REVISION/'protocols/cog_head_calibration.json').read_text())['head']
        common.Extractor = CogExtractor
        common.main()
