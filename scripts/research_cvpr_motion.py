"""Separate signed motion error, localization, and band effects on saved tracks."""
import json
from pathlib import Path

import numpy as np

from research_cvpr_summarize import paired_interval


ROOT = Path('/share/yty/heft')
THRESHOLDS = np.array([1, 2, 4, 8, 16])
SPEED_BINS = [0, .5, 2, 4, np.inf]
DISPLACEMENT_BINS = [0, 4, 16, 32, np.inf]


def compare(file_a, file_b):
    with np.load(file_a) as a, np.load(file_b) as b:
        gt = a['gt'] * 256
        for field in ('gt', 'gt_occluded', 'gt_valid', 'query_points'):
            assert np.array_equal(a[field], b[field])
        query_frames = a['query_points'][:, 0].astype(np.int64)
        visible = ~a['gt_occluded'] & a['gt_valid']
        allowed = (visible[:, :-2] & visible[:, 1:-1] & visible[:, 2:]
                   & (np.arange(1, gt.shape[1] - 1)[None] > query_frames[:, None]))
        velocity = (gt[:, 2:] - gt[:, :-2]) * .5
        speed = np.linalg.norm(velocity, axis=-1)
        unit = velocity / np.maximum(speed[:, :, None], 1e-8)
        normal = np.stack((-unit[..., 1], unit[..., 0]), axis=-1)
        e0 = a['tracks'][:, 1:-1] * 256 - gt[:, 1:-1]
        e1 = b['tracks'][:, 1:-1] * 256 - gt[:, 1:-1]
    dist = np.linalg.norm(gt[:, 1:-1] - gt[np.arange(len(gt)), query_frames][:, None], axis=-1)
    pck0 = (np.linalg.norm(e0, axis=-1)[..., None] < THRESHOLDS).mean(-1) * 100
    pck1 = (np.linalg.norm(e1, axis=-1)[..., None] < THRESHOLDS).mean(-1) * 100
    proj0, proj1 = (e0 * unit).sum(-1), (e1 * unit).sum(-1)
    per0, per1 = (e0 * normal).sum(-1), (e1 * normal).sum(-1)
    change = {'pck': pck1 - pck0, 'abs_parallel': np.abs(proj0) - np.abs(proj1),
              'abs_perpendicular': np.abs(per0) - np.abs(per1), 'lag_removed': (proj1 - proj0) / np.maximum(speed, 1e-8)}
    row = dict(fast=int((allowed & (speed >= 2)).sum()), valid=int(allowed.sum()))
    for label, value, edges in [('speed', speed, SPEED_BINS), ('displacement', dist, DISPLACEMENT_BINS)]:
        row[label] = []
        for low, high in zip(edges[:-1], edges[1:], strict=True):
            mask = allowed & (value >= low) & (value < high)
            result = {'count': int(mask.sum())}
            if result['count']:
                result.update({name: float(metric[mask].mean()) for name, metric in change.items() if name != 'lag_removed'})
                moving = mask & (speed >= .5)
                if moving.any():
                    result['lag_removed'] = float(np.median(change['lag_removed'][moving]))
            row[label].append(result)
    return row


def summarize(rows):
    counts = sum(r['valid'] for r in rows.values())
    payload = {'n': len(rows), 'point_frames': counts,
               'share_speed_2': sum(r['fast'] for r in rows.values()) / counts, 'rows': rows}
    for label, edges in [('speed', SPEED_BINS), ('displacement', DISPLACEMENT_BINS)]:
        stats = []
        for i in range(len(edges) - 1):
            samples = [r[label][i] for r in rows.values() if r[label][i]['count']]
            stats.append({'videos': len(samples), 'point_frames': sum(s['count'] for s in samples),
                          **{k: paired_interval(values) for k in ('pck', 'abs_parallel', 'abs_perpendicular', 'lag_removed')
                             if (values := [r[k] for r in samples if k in r])}})
        payload[label] = {'bins': [v if np.isfinite(v) else 'inf' for v in edges], 'stats': stats}
    return payload


def main():
    comparisons = {
        'davis_time': (ROOT/'results/benchmark_davis_polyphase', ROOT/'results/benchmark_davis_polyphase_corrected', 30),
        'rgb_time': (ROOT/'rgb_stacking/results/benchmark_native_polyphase', ROOT/'rgb_stacking/results/latency_transfer_polyphase', 50),
        'po_time': (ROOT/'pointodyssey/results/native_transfer_polyphase', ROOT/'pointodyssey/results/latency_transfer_polyphase', 13),
    }
    output = {}
    for label, (before, after, count) in comparisons.items():
        files = sorted(before.glob('*_l18h05_post_full.npz'))
        assert len(files) == count
        time_rows = {f.stem: compare(f, after/f.name) for f in files}
        band_rows = {f.stem: compare(after/f.name, after/f.name.replace('_post_full.npz', '_post_low15.npz')) for f in files}
        output[label] = summarize(time_rows)
        output[label.replace('_time', '_band')] = summarize(band_rows)
    folder = ROOT/'cvpr27/robotap_split0/wan'
    rows = {}
    for done in sorted(folder.glob('*.json')):
        name = done.stem
        rows[name] = compare(folder/f'{name}_native4_head_0.0.npz', folder/f'{name}_native4_head_1.25.npz')
    if rows:
        output['robotap_time'] = summarize(rows)
    path = ROOT/'cvpr27/summary/motion_strata.json'
    path.write_text(json.dumps(output, indent=2, allow_nan=False))
    for label, report in output.items():
        bins = report['displacement' if label.endswith('_band') else 'speed']['stats']
        print(label, report['n'], round(report['share_speed_2'], 3),
              [(s['videos'], [round(x, 2) for x in s.get('pck', [])]) for s in bins])


if __name__ == '__main__':
    main()
