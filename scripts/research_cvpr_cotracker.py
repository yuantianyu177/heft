"""Run the released CoTracker3 predictor on the same query lists and masks."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from research_track import official_metrics


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/share/yty/heft'))
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--names')
    parser.add_argument('--joint', action='store_true', help='Use the released joint evaluation mode')
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    torch.set_num_threads(3)
    torch.manual_seed(0)
    source = args.root/'cvpr27/external/co-tracker'
    sys.path.insert(0, str(source))
    from cotracker.models.build_cotracker import build_cotracker
    from cotracker.models.evaluation_predictor import EvaluationPredictor

    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    checkpoint = Path(json.loads((args.root/'models/CoTracker3-offline.json').read_text())['path'])/'scaled_offline.pth'
    with checkpoint.open('rb') as stream:
        checkpoint_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    model = build_cotracker(str(checkpoint), offline=True, window_len=60).to(device).eval()
    tracker = EvaluationPredictor(model, single_point=not args.joint).to(device).eval()
    evaluate, evaluator_hash = official_metrics(args.root)
    args.output.mkdir(parents=True, exist_ok=True)
    records = json.loads((args.data/'manifest.json').read_text())
    if args.names:
        names = set(args.names.split(','))
        records = [r for r in records if r['name'] in names]
        assert len(records) == len(names)
    protocol = dict(checkpoint_sha256=checkpoint_hash, code_revision=revision, evaluator_sha256=evaluator_hash,
                    single_point=not args.joint, visibility_threshold=.6, window=60, iterations=6,
                    resize=list(tracker.interp_shape), query_policy='first visible')
    for record in records[args.shard::args.shards]:
        name = record['name']
        output = args.output/f'{name}.json'
        if output.exists():
            assert json.loads(output.read_text())['protocol'] == protocol
            continue
        start = time.monotonic()
        with np.load(args.data/f'{name}.annotations.npz') as ann:
            gt, occ = ann['points'], ann['occluded']
            valid = ann['valid'] if 'valid' in ann else np.ones_like(occ)
        visible = (~occ) & valid
        eligible = visible.any(1) & (visible.argmax(1) < gt.shape[1] - 1)
        gt, occ, valid, visible = (a[eligible] for a in (gt, occ, valid, visible))
        frames = visible.argmax(1)
        points = gt[np.arange(len(gt)), frames]
        height, width = tracker.interp_shape
        size = np.array([width - 1, height - 1], dtype=np.float32)
        queries = torch.tensor(np.column_stack((frames, points*size)), device=device, dtype=torch.float32)[None]
        rgb = np.load(args.data/f'{name}.video.npy', mmap_mode='r')
        video = torch.cat([F.interpolate(torch.from_numpy(rgb[t:t+32].copy()).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32),
                                         size=(height, width), mode='bilinear', align_corners=True)
                           for t in range(0, len(rgb), 32)])[None]
        pred, score = tracker(video, queries)
        pred = pred[0].permute(1, 0, 2).cpu().numpy()/size
        pred_occ = score[0].T.cpu().numpy() <= .6
        metric_queries = np.column_stack((frames.astype(np.float32), points[:, ::-1]*256))
        metrics = evaluate(metric_queries[None], occ[None], gt[None]*256, pred_occ[None], pred[None]*256,
                           'first', valid_mask=valid[None])
        row = dict(video=name, frames=len(rgb), queries=len(frames), protocol=protocol,
                   metrics={k: float(v[0])*100 for k,v in metrics.items()}, seconds=time.monotonic()-start)
        temporary = output.with_suffix('.tmp')
        temporary.write_text(json.dumps(row, indent=2, allow_nan=False))
        temporary.replace(output)
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
