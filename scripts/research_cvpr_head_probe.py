"""Probe the clock and synthetic correspondence of all heads in one block."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from research_cvpr_readout import Extractor, PHASES, PRIMARY_HEADS
from research_rope import phase_readout


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/share/yty/heft'))
    parser.add_argument('--model', choices=PRIMARY_HEADS, required=True)
    parser.add_argument('--layer', type=int, default=18)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    torch.set_num_threads(3)
    cv2.setNumThreads(1)
    extractor = Extractor(args.root, args.model, device, PRIMARY_HEADS[args.model], [('native1', 'layer', 0.)], 42)
    blocks = extractor.model.blocks if args.model == 'wan' else extractor.model.transformer_blocks
    if args.layer != 18:
        del blocks[18].attn1._heft_capture
        blocks[args.layer].attn1._heft_capture = extractor.capture
    bank = args.root/'cvpr27/independent_clock'
    images = json.loads((bank/'images.json').read_text())
    nominal = PHASES[0]
    step = torch.tensor(nominal[1:], device=device, dtype=torch.float32)
    advances = torch.arange(-3, 3.01, .25, device=device)
    xy = torch.linspace(.32, .68, 10, device=device)
    ys, xs = torch.meshgrid(xy, xy, indexing='ij')
    points = torch.stack((xs, ys), dim=-1).reshape(-1, 2)
    thresholds = torch.tensor([1, 2, 4, 8, 16], device=device)
    heads = extractor.count

    def keys(frames):
        return extractor.get(frames)[0]['native1'][:, nominal]

    def canonical(feature):
        phase = extractor.angles[nominal]
        return F.normalize(phase_readout(feature, phase, temporal=0, spatial=0), dim=-1)

    def sample(volume, locations):
        # locations [time, candidates, points, xy]
        n = volume.shape[1]
        x = volume.permute(0, 1, 4, 2, 3).reshape(heads*n, 128, extractor.h, extractor.w)
        grid = locations[None].expand(heads, -1, -1, -1, -1).reshape(heads*n, *locations.shape[1:])*2-1
        out = F.grid_sample(x, grid, align_corners=False, padding_mode='border')
        return out.reshape(heads, n, 128, *locations.shape[1:3])

    rows = []
    for record in images:
        image = np.load(bank/'data/davis_records'/f'{record["name"]}.video.npy', mmap_mode='r')[0]
        width, height = extractor.width, extractor.height
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
        static = canonical(keys([image]*25))[:, 1:]
        ref = sample(static, points[None, None].expand(6, -1, -1, -1))
        for vx, vy in ((4, 0), (-4, 0), (0, 4), (0, -4)):
            video = [cv2.warpAffine(image, np.float32([[1, 0, vx*t], [0, 1, vy*t]]), (width, height),
                                    borderMode=cv2.BORDER_REFLECT_101) for t in range(25)]
            moving = keys(video)
            velocity = torch.tensor([vx/width, vy/height], device=device)
            loc = points[None, None] + (step[:, None] + advances[None])[:, :, None, None] * velocity
            assert ((loc >= 0) & (loc <= 1)).all()
            motion = sample(canonical(moving)[:, 1:], loc)
            mismatch = (motion - ref).square().mean((2, 4))
            lags = -advances[mismatch.argmin(dim=-1)]
            query = sample(moving[:, :1].float(), points[None, None])[:, 0, :, 0]
            target = F.normalize(moving[:, 1:].float().flatten(2, 3), dim=-1)
            similarity = torch.einsum('hdp,htsd->htps', F.normalize(query, dim=1), target)
            best = similarity.argmax(-1)
            pred = torch.stack(((best % extractor.w + .5)/extractor.w,
                                (best // extractor.w + .5)/extractor.h), dim=-1)
            expected = points[None] + (step[:, None, None] - 1.25) * velocity
            distances = ((pred - expected[None])*256).norm(dim=-1)
            pck = (distances[..., None] < thresholds).float().mean((1, 2, 3))
            rows.append(dict(image=record['name'], velocity=[vx, vy], lag=lags.cpu().tolist(), pck=pck.cpu().tolist()))
        print(record['name'], flush=True)
    lag = np.array([r['lag'] for r in rows])
    pck = np.array([r['pck'] for r in rows])
    selected = int(pck.mean(0).argmax())
    leave_one_out = {name: int(pck[[r['image'] != name for r in rows]].mean(0).argmax())
                     for name in [r['name'] for r in images]}
    report = dict(model=args.model, layer=args.layer, selected_head=selected, leave_one_out=leave_one_out,
                  calibrated_synthetic_pck=(pck.mean(0)*100).tolist(), median_lag=np.median(lag, axis=(0, 2)).tolist(),
                  lag_boundary_fraction=(np.abs(lag)==3).mean((0, 2)).tolist(), rows=rows)
    path = args.root/'cvpr27/summary'/f'head_probe_{args.model}_l{args.layer:02d}.json'
    path.write_text(json.dumps(report, indent=2))
    print('selected', selected, 'lag', report['median_lag'][selected], flush=True)


if __name__ == '__main__':
    main()
