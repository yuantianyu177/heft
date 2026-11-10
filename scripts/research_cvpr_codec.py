"""Measure latent and reconstructed RGB clocks with upstream Wan codecs."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F


def central_grid(count, device):
    coordinates = torch.linspace(.3, .7, count, device=device)
    y, x = torch.meshgrid(coordinates, coordinates, indexing='ij')
    return torch.stack((x, y), -1).reshape(-1, 2)


def register(moving, reference, grid, times, velocity, indices, cosine=False):
    query = grid[None] + times[:, None, None] * velocity[None, None]
    safe = ((query > .02) & (query < .98)).all((0, 2))
    candidates = query[:, safe] * 2 - 1
    num = len(reference)
    samples = F.grid_sample(reference, (grid[safe] * 2 - 1)[None, None].expand(num, -1, -1, -1), align_corners=False)
    values = F.grid_sample(moving, candidates[None].expand(num, -1, -1, -1), align_corners=False)
    errors = {'l2': (values - samples).square().mean((1, 3))}
    if cosine:
        errors['cosine'] = 1 - F.cosine_similarity(values, samples, dim=1).mean(-1)
    results = {}
    for name, error in errors.items():
        effective = times[error.argmin(dim=-1)]
        results[name] = {'effective_times': effective.cpu().tolist(), 'nominal_times': indices.cpu().tolist(),
                         'lag': (indices - effective).cpu().tolist()}
    return results


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--images', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--size', type=int, nargs=2, default=(480, 832))
    parser.add_argument('--speeds', type=float, nargs='+', default=[4.])
    parser.add_argument('--decode', action='store_true')
    parser.add_argument('--sensitivity', action='store_true',
                        help='Repeat the latent registration with cosine distance and 8, 16, or 24 grid samples per axis')
    args = parser.parse_args()
    from diffusers import AutoencoderKLWan

    cv2.setNumThreads(1)
    torch.set_num_threads(3)
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    dtype = torch.bfloat16
    codec = AutoencoderKLWan.from_pretrained(args.checkpoint, subfolder='vae', torch_dtype=dtype,
                                            local_files_only=True).to(device).eval()
    height, width = args.size
    mean = torch.tensor(codec.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(codec.config.latents_std, device=device).view(1, -1, 1, 1, 1)
    grid = central_grid(16, device)
    alternative_grids = {n: central_grid(n, device) for n in (8, 24)} if args.sensitivity else {}
    times = torch.arange(-8, 32.01, .25, device=device)

    def encode(images):
        pixels = torch.from_numpy(np.stack(images)).permute(3, 0, 1, 2)[None].to(device, dtype=dtype) / 127.5 - 1
        raw = codec.encode(pixels).latent_dist.mode()
        latents = ((raw.float() - mean) / std)[0].permute(1, 0, 2, 3)
        assert len(latents) == 7, latents.shape
        if not args.decode:
            return {'latent': latents}
        decoded = codec.decode(raw).sample
        assert decoded.shape == pixels.shape, (decoded.shape, pixels.shape)
        return {'latent': latents, 'decoded_rgb': decoded[0].permute(1, 0, 2, 3).float()}

    rows = []
    inputs = json.loads((args.images/'images.json').read_text())
    for record in inputs:
        name = record['name']
        path = args.images/'data/davis_records'/f'{name}.video.npy'
        image = cv2.resize(np.load(path, mmap_mode='r')[0], (width, height), interpolation=cv2.INTER_LANCZOS4)
        reference = encode([image] * 25)
        directions = [(vx, vy) for speed in args.speeds for vx, vy in ((speed, 0), (-speed, 0), (0, speed), (0, -speed))]
        for vx, vy in directions:
            video = [cv2.warpAffine(image, np.float32([[1, 0, vx*t], [0, 1, vy*t]]), (width, height),
                                   borderMode=cv2.BORDER_REFLECT_101) for t in range(25)]
            moving = encode(video)
            velocity = torch.tensor([vx / width, vy / height], device=device)
            row = {'image': name, 'velocity': [vx, vy]}
            for key, ref in reference.items():
                indices = torch.arange(0, 25, 4 if key == 'latent' else 1, device=device)
                sensitivity = args.sensitivity and key == 'latent'
                matches = register(moving[key], ref, grid, times, velocity, indices, cosine=sensitivity)
                row[key] = matches['l2']
                if sensitivity:
                    row['latent_cosine_16'] = matches['cosine']
                    for count, other in alternative_grids.items():
                        alternative = register(moving[key], ref, other, times, velocity, indices, cosine=True)
                        row.update({f'latent_{metric}_{count}': stats for metric, stats in alternative.items()})
            rows.append(row)
        print(name, len(rows), flush=True)
    summary, by_speed = {}, {}
    for key in rows[0].keys() - {'image', 'velocity'}:
        lag = np.array([row[key]['lag'][1:] for row in rows])
        summary[key] = {'mean': float(lag.mean()), 'median': float(np.median(lag)), 'mean_by_time': lag.mean(0).tolist()}
        by_speed[key] = {}
        for speed in args.speeds:
            a = np.array([r[key]['lag'][1:] for r in rows if max(map(abs, r['velocity'])) == speed])
            by_speed[key][str(speed)] = {'median': float(np.median(a)), 'mean': float(a.mean())}
    source = Path(inspect.getfile(AutoencoderKLWan))
    result = dict(checkpoint=str(args.checkpoint), source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  image_bank=inputs, dtype=str(dtype), size=[height, width], records=rows, summary=summary, by_speed=by_speed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(summary, flush=True)


if __name__ == '__main__':
    main()
