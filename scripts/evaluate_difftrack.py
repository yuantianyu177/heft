"""Run the DiffTrack reimplementation on the shared TAP-Vid record format."""

from argparse import ArgumentParser
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from heft.baselines import difftrack, difftrack_extractor
from heft.baselines.difftrack_extractor import DiffTrackExtractor
from heft.extraction import COGVIDEOX, WAN_2_1
from research_track import official_metrics

MODELS = {'wan': (WAN_2_1, 15), 'cog': (COGVIDEOX, 17)}
METRICS = ('average_jaccard', 'average_pts_within_thresh', 'occlusion_accuracy')


def annotations(root: Path, name: str):
    with np.load(root / f'{name}.annotations.npz') as file:
        points, occluded = file['points'], file['occluded']
        valid = file['valid'] if 'valid' in file else np.ones_like(occluded)
    visible = (~occluded) & valid
    first = visible.argmax(axis=1)
    select = visible.any(axis=1) & (first < points.shape[1] - 1)
    return points[select], occluded[select], valid[select], first[select]


def save(path: Path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def main():
    parser = ArgumentParser()
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--metric-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--layer', type=int)
    parser.add_argument('--names', help='Comma-separated record IDs for a restricted pilot')
    args = parser.parse_args()
    if not (args.checkpoint / 'model_index.json').exists():
        parser.error('--checkpoint must be an existing local model directory')
    args.output.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(args.gpu)
    torch.backends.cuda.matmul.allow_tf32 = False
    evaluate, evaluator_hash = official_metrics(args.metric_root)
    base, default_layer = MODELS[args.model]
    layer = default_layer if args.layer is None else args.layer
    model = replace(base, model_id=str(args.checkpoint))
    manifest = json.loads((args.records / 'manifest.json').read_text())
    selected = set(args.names.split(',')) if args.names else {row['name'] for row in manifest}
    records = [row for row in manifest if row['name'] in selected]
    if len(records) != len(selected):
        raise ValueError('Requested names do not match the record manifest')
    plan = dict(model=args.model, checkpoint=str(args.checkpoint), records=str(args.records), layer=layer,
                frames={row['name']: row['frames'] for row in records}, evaluator=evaluator_hash,
                chunk_size=13, visibility_policy='always visible after each query',
                sources={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in
                         map(Path, [__file__, difftrack.__file__, difftrack_extractor.__file__])})
    plan_path = args.output / 'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError(f'Existing experiment differs from {plan_path}')
    save(plan_path, plan)
    missing = [row for row in records if not (args.output / f'{row["name"]}.json').exists()]
    if not missing:
        return
    model_tracker = DiffTrackExtractor(model, layer, args.gpu)
    for record in missing:
        name = record['name']
        raw = np.load(args.records / f'{name}.video.npy', allow_pickle=False)
        rgb = torch.from_numpy(raw).permute(0, 3, 1, 2)
        gt, occ, valid, q = annotations(args.records, name)
        if len(rgb) != record['frames'] or gt.shape[1] != len(rgb):
            raise ValueError(f'Unexpected dataset frame coverage for {name}')
        anchors = gt[np.arange(len(q)), q]
        query_xy_t = np.column_stack((anchors, q)).astype(np.float32)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        prediction = model_tracker.track(rgb, torch.from_numpy(query_xy_t)).numpy()
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        query_yx_256 = np.column_stack((q.astype(np.float32), anchors[:, ::-1] * 256))
        pred_occ = np.arange(len(rgb))[None] < q[:, None]
        raw_metrics = evaluate(query_yx_256[None], occ[None], gt[None] * 256,
                               pred_occ[None], prediction[None] * 256, 'first', valid_mask=valid[None])
        metrics = {key: float(raw_metrics[key][0]) * 100 for key in METRICS}
        row = dict(video=name, frames=len(rgb), queries=len(q), metrics=metrics,
                   inversion=model_tracker.inverse_method, timestep=float(model_tracker.timestep),
                   seconds=elapsed, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
        save(args.output / f'{name}.json', row)
        print('DONE', json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
