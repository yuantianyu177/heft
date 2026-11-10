"""Measure localization around reappearance and temporal chunk boundaries."""
import json
from pathlib import Path

import numpy as np

from research_cvpr_summarize import paired_interval


ROOT = Path('/share/yty/heft')
THRESHOLDS = np.array([1, 2, 4, 8, 16])


def measure(before, after):
    with np.load(before) as raw, np.load(after) as calibrated:
        gt = raw['gt']
        gt_visible = ~raw['gt_occluded'] & raw['gt_valid']
        queries = raw['query_points'][:, 0].astype(np.int64)
        assert all(np.array_equal(raw[field], calibrated[field]) for field in ('gt', 'gt_occluded', 'gt_valid', 'query_points'))
        times = np.arange(gt.shape[1])
        eligible = gt_visible & (times[None] > queries[:, None])
        known_occluded = raw['gt_occluded'] & raw['gt_valid']
        last_seen = np.maximum.accumulate(np.where(gt_visible, times, -1), axis=1)
        previous_seen = np.pad(last_seen[:, :-1], ((0, 0), (1, 0)), constant_values=-1)
        consecutive_occlusion = np.pad(known_occluded[:, :-3] & known_occluded[:, 1:-2] & known_occluded[:, 2:-1],
                                       ((0, 0), (3, 0)), constant_values=False)
        returns = eligible & (previous_seen >= queries[:, None]) & consecutive_occlusion
        last_return = np.maximum.accumulate(np.where(returns, times, -100), axis=1)
        time_since_return = times[None] - last_return
        last_chunk_frame = np.minimum((times//25+1)*25-1, len(times)-1)
        masks = dict(first_return=returns, following_four=eligible & (time_since_return >= 1) & (time_since_return <= 4),
                     chunk_anchor=eligible & (times[None] % 25 == 0),
                     chunk_tail=eligible & (times[None] >= last_chunk_frame[None]-1) & (times[None] % 25 != 0),
                     interior=eligible & (times[None] % 25 != 0) & (times[None] < last_chunk_frame[None]-1))
        result = {'frames': len(times), 'queries': len(queries)}
        for name, mask in masks.items():
            record = {'count': int(mask.sum())}
            if record['count']:
                for label, data in [('nominal', raw), ('calibrated', calibrated)]:
                    distance = np.linalg.norm((data['tracks'][mask] - gt[mask])*256, axis=-1)
                    pck = (distance[:, None] < THRESHOLDS).mean(axis=1)*100
                    record[label] = float(pck.mean())
                    record[label+'_visible'] = float((pck*data['visibility'][mask]).mean())
            result[name] = record
    return result


def summarize(rows, expected):
    aggregate = {'videos': len(rows), 'complete': len(rows) == expected, 'expected': expected,
                 'frames': sum(r['frames'] for r in rows.values()), 'queries': sum(r['queries'] for r in rows.values())}
    for key in ('first_return', 'following_four', 'chunk_anchor', 'chunk_tail', 'interior'):
        values = [r[key] for r in rows.values() if r[key]['count']]
        if not values:
            continue
        aggregate[key] = dict(videos=len(values), point_frames=sum(r['count'] for r in values),
                              pck=[float(np.mean([r[k] for r in values])) for k in ('nominal', 'calibrated')],
                              pck_gain_ci95=paired_interval([r['calibrated']-r['nominal'] for r in values]),
                              pck_visible_gain_ci95=paired_interval([r['calibrated_visible']-r['nominal_visible'] for r in values]))
    return aggregate | {'per_video': rows}


def main():
    sets = {}
    for label, base, names, count in [
            ('davis', ROOT/'results', ('benchmark_davis_polyphase', 'benchmark_davis_polyphase_corrected'), 30),
            ('rgb', ROOT/'rgb_stacking/results', ('benchmark_native_polyphase', 'latency_transfer_polyphase'), 50),
            ('po', ROOT/'pointodyssey/results', ('native_transfer_polyphase', 'latency_transfer_polyphase'), 13)]:
        before, after = (base/f for f in names)
        paths = sorted(before.glob('*_l18h05_post_full.npz'))
        sets[label] = summarize({p.stem: measure(p, after/p.name) for p in paths}, count)
    for label, directory, count in [('robotap', ROOT/'cvpr27/robotap_split0/wan', 53),
                                    ('itto_mose', ROOT/'cvpr27/itto_mose/wan', 44)]:
        names = [p.stem for p in sorted(directory.glob('*.json'))]
        if names:
            sets[label] = summarize({n: measure(directory/f'{n}_native4_head_0.0.npz',
                                               directory/f'{n}_native4_head_1.25.npz') for n in names}, count)
    output = ROOT/'cvpr27/summary/visibility_boundaries.json'
    output.write_text(json.dumps(sets, indent=2, allow_nan=False))
    print(json.dumps({k: {name: x for name, x in v.items() if name != 'per_video'} for k, v in sets.items()}, indent=2))


if __name__ == '__main__':
    main()
