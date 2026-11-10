"""Matched encoding and descriptor controls for the CVPR HeFT revision.

The VAE, denoiser, noise realization, query list, and tracker are shared. Only
the temporal encoding contract, feature coordinates, or channel selection
changes. Results are written per video and can be resumed by shard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
from safetensors.torch import load_file

from heft.attn_hook import FeatureKind
from heft.tracking import FeatureSelection, FeatureTracker, FeatureVideo, FrameRange, RopePairing, RopeSpec, TrackingConfig
from research_causal_denoising import native_vae_class
from research_cosmos_rope import adjacent_storage, cosmos_angles
from research_probe import random_kept
from research_rope import frequency_pairs, pair_channels, phase_readout, wan_angles
from research_track import official_metrics


CHUNK = 25
PHASES = ([0, 4, 8, 12, 16, 20, 24], *([0] + list(range(i, CHUNK, 4)) for i in (1, 2, 3)))
NATIVE_MODES = {'native1', 'native2', 'native4'}
JOINT_MODES = {'joint_sparse', 'joint_dense'}
CHECKPOINTS = {'wan': 'Wan2.1-T2V-1.3B-Diffusers', 'cosmos': 'Cosmos-Predict2-2B-Video2World'}
PRIMARY_HEADS = {'wan': 5, 'cosmos': 6}
DEFAULT_VARIANTS = 'native4:head:0,native4:head:0.25,native4:head:1.25,native4:low:0,native4:low:1.25,framewise:low:0,stride1:low:0,native1:head:0,native1:head:1.25,native2:head:1.25,native4:layer:0,native4:layer:1.25'


class CaptureDone(Exception):
    pass


def interpolate(maps, source_times, target_times):
    """Interpolate [heads,time,height,width,channels] on a specified time grid."""
    source = torch.as_tensor(source_times, device=maps.device, dtype=torch.float32)
    target = torch.as_tensor(target_times, device=maps.device, dtype=torch.float32)
    target = target.clamp(source[0], source[-1])
    right = torch.searchsorted(source, target).clamp(1, len(source) - 1)
    left = right - 1
    weight = ((target - source[left]) / (source[right] - source[left]))[None, :, None, None, None]
    return torch.lerp(maps[:, left].float(), maps[:, right].float(), weight)


class Extractor:
    def __init__(self, root, name, device, head, variants, seed):
        from diffusers import (AutoencoderKLWan, CosmosTransformer3DModel, FlowMatchEulerDiscreteScheduler,
                               UniPCMultistepScheduler, WanTransformer3DModel)

        self.name = name
        self.device = device
        self.dtype = torch.bfloat16
        self.seed = seed
        self.head = head
        self.modes = {variant[0] for variant in variants}
        self.capture_all = any(variant[1] in {'layer', 'layerlow'} for variant in variants)
        self.height, self.width = (480, 832) if name == 'wan' else (704, 1280)
        self.h, self.w = self.height // 16, self.width // 16
        self.path = json.loads((root / 'models' / (CHECKPOINTS[name] + '.json')).read_text())['path']
        weights = dict(torch_dtype=self.dtype, local_files_only=True)
        native_class, self.encoder_hash = native_vae_class()
        self.native = native_class.from_pretrained(self.path, subfolder='vae', **weights).to(device).eval()
        self.stride_one = None
        if 'stride1' in self.modes:
            self.stride_one = AutoencoderKLWan.from_pretrained(self.path, subfolder='vae', **weights).to(device).eval()
        self.mean = torch.tensor(self.native.config.latents_mean, device=device).view(1, 16, 1, 1, 1)
        self.std = torch.tensor(self.native.config.latents_std, device=device).view(1, 16, 1, 1, 1)
        prompt_path = root / 'models' / f'{name}_empty_prompt.safetensors'
        self.prompt = load_file(str(prompt_path))['prompt_embeds'].to(device)
        if name == 'wan':
            self.model = WanTransformer3DModel.from_pretrained(self.path, subfolder='transformer', **weights).to(device).eval()
            scheduler = UniPCMultistepScheduler.from_pretrained(self.path, subfolder='scheduler', local_files_only=True)
            scheduler.set_timesteps(50, device=device)
            self.step = scheduler.timesteps[49:50]
            self.sigma = float(scheduler.sigmas[49])
            block = self.model.blocks[18]
            self.angles = wan_angles(CHUNK, self.h, self.w, device=device)
        else:
            self.model = CosmosTransformer3DModel.from_pretrained(self.path, subfolder='transformer', **weights).to(device).eval()
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(self.path, subfolder='scheduler', local_files_only=True)
            scheduler.set_timesteps(sigmas=torch.linspace(0, 1, 35, dtype=torch.float64), device=device)
            self.step = scheduler.timesteps[34:35]
            self.sigma = float(scheduler.sigmas[34])
            block = self.model.transformer_blocks[18]
            self.angles = cosmos_angles(CHUNK, self.h, self.w, device=device)
        self.angles[..., :22] /= 4
        self.count = block.attn1.heads if self.capture_all else 1
        self.captured = None
        block.attn1._heft_capture = self.capture

    def capture(self, *, key, **kwargs):
        keys = key[0] if self.capture_all else key[0, self.head:self.head + 1]
        if self.name == 'cosmos':
            keys = adjacent_storage(keys)
        self.captured = keys.reshape(self.count, -1, self.h, self.w, 128).contiguous()
        raise CaptureDone

    def denoise(self, clean, noise, times):
        frames = len(times)
        phase = self.angles[times]
        noisy = ((1 - self.sigma) * clean + self.sigma * noise[:, :, times]).to(self.dtype)
        kwargs = dict(encoder_hidden_states=self.prompt, return_dict=False)
        if self.name == 'wan':
            rotary = torch.polar(torch.ones_like(phase), phase).reshape(1, 1, -1, 64)
            self.model.rope.forward = lambda hidden_states: rotary
            kwargs['timestep'] = self.step
        else:
            full_phase = torch.cat((phase, phase), dim=-1).reshape(-1, 128)
            rotary = (full_phase.cos(), full_phase.sin())
            self.model.rope.forward = lambda hidden_states, fps=None: rotary
            kwargs.update(timestep=torch.full((1, 1, frames, 1, 1), self.sigma / (1 + self.sigma),
                                              device=self.device, dtype=self.dtype), fps=16,
                          condition_mask=torch.zeros((1, 1, frames, self.h * 2, self.w * 2),
                                                     device=self.device, dtype=self.dtype),
                          padding_mask=torch.zeros((1, 1, self.height, self.width),
                                                   device=self.device, dtype=self.dtype))
        try:
            self.model(noisy, **kwargs)
        except CaptureDone:
            feature, self.captured = self.captured, None
            return feature
        raise RuntimeError('No self-attention descriptor was captured')

    def encode(self, vae, frames):
        return (vae.encode(frames).latent_dist.mode().float() - self.mean) / self.std

    def framewise(self, inputs):
        images = inputs[0].permute(1, 0, 2, 3).unsqueeze(2)
        latents = [self.encode(self.native, part) for part in images.split(5)]
        return torch.cat(latents)[:, :, 0].permute(1, 0, 2, 3).unsqueeze(0)

    def get(self, raw):
        began = time.monotonic()
        resized = np.stack([cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LANCZOS4)
                            for frame in raw])
        if len(resized) < CHUNK:
            resized = np.concatenate((resized, np.repeat(resized[-1:], CHUNK - len(resized), axis=0)))
        inputs = torch.from_numpy(resized).permute(3, 0, 1, 2)[None].to(self.device, dtype=self.dtype) / 127.5 - 1
        noise = torch.randn((1, 16, CHUNK, self.h * 2, self.w * 2),
                            generator=torch.Generator().manual_seed(self.seed)).to(self.device)
        prep = time.monotonic() - began
        features, profile = {}, {'prepare_s': prep}
        native_modes = self.modes & NATIVE_MODES
        joint_modes = self.modes & JOINT_MODES
        if native_modes or joint_modes:
            phase_ids = ((0, 1, 2, 3) if self.modes & {'native4', 'joint_dense'}
                         else (0, 2) if 'native2' in native_modes else (0,))
            phase_features, phase_latents = {}, {}
            anchor = None
            for i in phase_ids:
                torch.cuda.reset_peak_memory_stats(self.device)
                began = time.monotonic()
                prefix = 4 - i if i else 0
                padded = torch.cat((inputs[:, :, :1].expand(-1, -1, prefix, -1, -1), inputs), dim=2)
                padded = torch.cat((padded, padded[:, :, -1:].expand(-1, -1, 29 - padded.shape[2], -1, -1)), dim=2)
                clean = self.encode(self.native, padded)[:, :, :7]
                if anchor is None:
                    anchor = clean[:, :, :1].clone()
                elif not torch.equal(anchor, clean[:, :, :1]):
                    raise RuntimeError('Padding changed the first latent')
                if joint_modes:
                    phase_latents[i] = clean
                if native_modes:
                    phase_features[i] = self.denoise(clean, noise, PHASES[i])
                torch.cuda.synchronize(self.device)
                profile[f'phase{i}_s'] = time.monotonic() - began
                profile[f'phase{i}_peak_bytes'] = torch.cuda.max_memory_allocated(self.device)
            if 'native1' in native_modes:
                features['native1'] = interpolate(phase_features[0], PHASES[0], range(CHUNK)).to(self.dtype)
            if 'native2' in native_modes:
                half = torch.empty_like(phase_features[0][:, :1]).expand(-1, 13, -1, -1, -1).clone()
                half[:, ::2] = phase_features[0]
                half[:, 1::2] = phase_features[2][:, 1:]
                half[:, 0] = torch.stack((phase_features[0][:, 0], phase_features[2][:, 0])).float().mean(0)
                features['native2'] = interpolate(half, range(0, CHUNK, 2), range(CHUNK)).to(self.dtype)
            if 'native4' in native_modes:
                dense = torch.empty((self.count, CHUNK, self.h, self.w, 128), device=self.device, dtype=self.dtype)
                for i, phase in phase_features.items():
                    dense[:, PHASES[i]] = phase
                dense[:, 0] = torch.stack([phase[:, 0] for phase in phase_features.values()]).float().mean(0)
                features['native4'] = dense

            times = list(range(CHUNK))
            for mode in sorted(joint_modes):
                torch.cuda.reset_peak_memory_stats(self.device)
                began = time.monotonic()
                if mode == 'joint_sparse':
                    sparse = phase_latents[0].permute(0, 2, 3, 4, 1)
                    clean = interpolate(sparse, PHASES[0], times).permute(0, 4, 1, 2, 3).contiguous()
                else:
                    source = phase_latents[0]
                    clean = source.new_empty((1, source.shape[1], CHUNK, *source.shape[-2:]))
                    for i, latent in phase_latents.items():
                        clean[:, :, PHASES[i]] = latent
                features[mode] = self.denoise(clean, noise, times)
                torch.cuda.synchronize(self.device)
                profile[f'{mode}_s'] = time.monotonic() - began
                profile[f'{mode}_peak_bytes'] = torch.cuda.max_memory_allocated(self.device)

        for mode in ('framewise', 'stride1'):
            if mode not in self.modes:
                continue
            torch.cuda.reset_peak_memory_stats(self.device)
            began = time.monotonic()
            clean = self.framewise(inputs) if mode == 'framewise' else self.encode(self.stride_one, inputs)
            assert clean.shape[2] == CHUNK, (mode, clean.shape)
            features[mode] = self.denoise(clean, noise, list(range(CHUNK)))
            torch.cuda.synchronize(self.device)
            profile[f'{mode}_s'] = time.monotonic() - began
            profile[f'{mode}_peak_bytes'] = torch.cuda.max_memory_allocated(self.device)
        return features, profile

    def readout(self, maps, descriptor, delay, valid_frames):
        if descriptor not in {'layer', 'layerlow'}:
            head = self.head if self.capture_all else 0
            maps = maps[head:head + 1]
        maps = maps[:, :valid_frames]
        if descriptor in {'low', 'layerlow', 'low10'}:
            fraction = 0.10 if descriptor == 'low10' else 0.15
            channels = pair_channels(frequency_pairs(fraction)).to(self.device)
            maps = maps[..., channels]
        if descriptor.startswith('random'):
            maps = maps[..., random_kept(int(descriptor.removeprefix('random')), self.device)]
        if delay != 0 and valid_frames > 1:
            target = torch.arange(valid_frames, device=self.device, dtype=torch.float32)
            target[1:] += delay
            if descriptor == 'neutral':
                angles = self.angles[:valid_frames]
                maps = phase_readout(maps, angles, temporal=0, spatial=1)
                maps = interpolate(maps, range(valid_frames), target)
                maps = phase_readout(maps, angles, temporal=2, spatial=1)
            else:
                maps = interpolate(maps, range(valid_frames), target)
        return maps.permute(1, 0, 4, 2, 3).flatten(1, 2).cpu().contiguous()


def run_tracker(volumes, positions, query_frames, extractor, ema, upsample):
    """Adapt dense volumes to the unchanged tracker, with exact chunk handoff."""
    height, width = extractor.height, extractor.width
    size = np.array([width - 1, height - 1], dtype=np.float32)
    frames = sum(map(len, volumes))
    predictions = torch.zeros((len(positions), frames, 2))
    visibility = torch.zeros((len(positions), frames), dtype=torch.bool)
    config = TrackingConfig(query_feature=FeatureKind.KEY, target_feature=FeatureKind.KEY,
                            update_feature=FeatureKind.KEY, argmax_radius=16, search_radius=48,
                            visibility_threshold=24, feature_ema_alpha=ema, upsample_features=upsample)
    tracker = FeatureTracker(config=config, device=extractor.device)
    for query_frame in np.unique(query_frames):
        indices = np.flatnonzero(query_frames == query_frame)
        local, ranges = [], []
        start = 0
        for volume in volumes:
            stop = start + len(volume)
            if stop > query_frame:
                begin = max(start, query_frame)
                ranges.append(FrameRange(len(ranges), int(begin - query_frame), int(stop - query_frame), int(stop - begin)))
                local.append(volume[begin - start:])
            start = stop

        class Reader:
            def __init__(self, video, *, device):
                self.device = device

            def prefetch_chunks(self, chunks, **kwargs):
                for chunk in chunks:
                    volume = local[chunk.index]
                    if chunk.index:
                        volume = torch.cat((local[chunk.index - 1][-1:], volume))
                    yield chunk, {FeatureKind.KEY: volume.to(self.device, dtype=torch.float32)}

        video = FeatureVideo(root=Path('.'), name='paired', model=extractor.name,
                             num_frames=frames - int(query_frame), frame_size=(height, width),
                             feature_size=(extractor.h, extractor.w), chunks=tuple(ranges), step=0,
                             layers=(18,), heads=(extractor.head,), feature_kinds=('key',),
                             rope=RopeSpec(RopePairing.ADJACENT, 22, 21, 21))
        queries = torch.from_numpy(positions[indices] * size)
        queries = torch.cat((queries, torch.zeros((len(queries), 1))), dim=-1)
        with patch('heft.tracking.tracker.FeatureReader', Reader):
            prediction = tracker.track(video, queries, FeatureSelection(layer=18, head=extractor.head))
        predictions[indices, query_frame:] = prediction.tracks
        visibility[indices, query_frame:] = prediction.visibility
    return predictions.numpy() / size, visibility.numpy()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/share/yty/heft'))
    parser.add_argument('--data', required=True, type=Path, help='Directory of *.video.npy and *.annotations.npz')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model', choices=CHECKPOINTS, default='wan')
    parser.add_argument('--head', type=int)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--names', help='Comma-separated subset of manifest names')
    parser.add_argument('--variants', default=DEFAULT_VARIANTS)
    parser.add_argument('--ema', type=float, default=.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rope-time-scale', type=float, choices=(.25, 1.), default=.25,
                        help='Use unit temporal coordinates for the frame-aligned baseline')
    parser.add_argument('--upsample', action='store_true', help='Match with spatially upsampled rather than token-grid maps')
    parser.add_argument('--save-tracks', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    variants = [(mode, readout, float(delay)) for mode, readout, delay in
                (spec.split(':') for spec in args.variants.split(','))]
    if len(set(variants)) != len(variants):
        raise ValueError('Duplicate variants')
    if not {v[0] for v in variants} <= NATIVE_MODES | JOINT_MODES | {'framewise', 'stride1'}:
        raise ValueError('Unknown extraction mode')
    if not all(v[1] in {'head', 'layer', 'low', 'low10', 'layerlow', 'neutral', 'random0', 'random1', 'random2'} for v in variants):
        raise ValueError('Unknown descriptor mode')
    records = json.loads((args.data / 'manifest.json').read_text())
    if args.names:
        wanted = set(args.names.split(','))
        records = [record for record in records if record['name'] in wanted]
        if len(records) != len(wanted):
            raise ValueError('Names absent from manifest')
    records = records[args.shard::args.shards]
    evaluate, evaluator_hash = official_metrics(args.root)
    extractor = Extractor(args.root, args.model, device, PRIMARY_HEADS[args.model] if args.head is None else args.head,
                          variants, args.seed)
    if args.rope_time_scale == 1.:
        if not extractor.modes <= {'framewise', 'stride1'} or args.model not in {'wan', 'cosmos'}:
            raise ValueError('Unit time coordinates apply only to rotary frame-aligned controls')
        extractor.angles[..., :22] *= 4
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    protocol = dict(model=args.model, head=extractor.head, sigma=extractor.sigma, ema=args.ema, seed=args.seed,
                    variants=[list(v) for v in variants], source_sha256=script_hash,
                    native_vae_sha256=extractor.encoder_hash, evaluator_sha256=evaluator_hash,
                    chunk=CHUNK, rope_time_scale=args.rope_time_scale, query_policy='first visible', upsample_features=args.upsample,
                    radius_argmax_search_consistency=[16, 48, 24], dimensions=[extractor.height, extractor.width])
    for record in records:
        name = record['name']
        output = args.output / f'{name}.json'
        if output.exists():
            old = json.loads(output.read_text())
            if old['protocol'] != protocol:
                raise RuntimeError(f'Protocol differs in existing result {output}')
            continue
        raw = np.load(args.data / f'{name}.video.npy', mmap_mode='r')
        with np.load(args.data / f'{name}.annotations.npz') as annotation:
            points = annotation['points']
            occluded = annotation['occluded']
            valid = annotation['valid'] if 'valid' in annotation else np.ones_like(occluded)
        assert len(raw) == record['frames'] == points.shape[1]
        available = (~occluded) & valid
        eligible = available.any(axis=1) & (available.argmax(axis=1) < len(raw) - 1)
        points, occluded, valid, available = (value[eligible] for value in (points, occluded, valid, available))
        query_frames = available.argmax(axis=1)
        if not len(query_frames):
            raise ValueError(f'No evaluable queries in {name}')
        positions = points[np.arange(len(points)), query_frames]
        queries = np.column_stack((query_frames.astype(np.float32), positions[:, ::-1] * 256))
        volumes = {spec: [] for spec in variants}
        profiles = []
        began = time.monotonic()
        for start in range(0, len(raw), CHUNK):
            chunk = raw[start:start + CHUNK]
            features, profile = extractor.get(chunk)
            for spec, outputs in volumes.items():
                mode, descriptor, delay = spec
                outputs.append(extractor.readout(features[mode], descriptor, delay, len(chunk)))
            profiles.append(profile)
            del features
        result = {}
        for spec, outputs in volumes.items():
            tag = ':'.join(map(str, spec))
            start = time.monotonic()
            prediction, visible = run_tracker(outputs, positions, query_frames, extractor, args.ema, args.upsample)
            metrics = evaluate(queries[None], occluded[None], points[None] * 256,
                               (~visible)[None], prediction[None] * 256, 'first', valid_mask=valid[None])
            result[tag] = dict(metrics={k: float(v[0]) * 100 for k, v in metrics.items()},
                               track_s=time.monotonic() - start)
            if args.save_tracks:
                np.savez_compressed(args.output / f'{name}_{tag.replace(":", "_")}.npz',
                                    tracks=prediction, visibility=visible, gt=points, gt_occluded=occluded,
                                    gt_valid=valid, query_points=queries)
        row = dict(video=name, frames=len(raw), queries=len(positions), protocol=protocol,
                   profiles=profiles, results=result, seconds=time.monotonic() - began)
        staged = output.with_suffix('.tmp')
        staged.write_text(json.dumps(row, indent=2))
        staged.replace(output)
        print(json.dumps({k: row[k] for k in ('video', 'frames', 'queries', 'seconds', 'results')}), flush=True)


if __name__ == '__main__':
    main()
