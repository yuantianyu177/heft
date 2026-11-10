"""Summarize native sampling and attention-context controls with matched video pairs."""
import json
from pathlib import Path

import numpy as np

from research_cvpr_summarize import paired_interval


ROOT = Path('/share/yty/heft')
REVISION = ROOT / 'cvpr27'
METRICS = {'AJ': 'average_jaccard', 'PCK': 'average_pts_within_thresh', 'OA': 'occlusion_accuracy'}
MODES = {'joint_sparse': 'joint_sparse', 'joint_dense': 'joint_dense', 'separate': 'native4'}
MATCHED_SETTINGS = ('model', 'head', 'sigma', 'ema', 'seed', 'native_vae_sha256', 'evaluator_sha256',
                    'chunk', 'rope_time_scale', 'query_policy', 'upsample_features',
                    'radius_argmax_search_consistency', 'dimensions')


def summarize(model, names):
    folder = REVISION / 'polish/joint_context' / model
    paths = sorted(folder.glob('*.json'))
    completed = {path.stem for path in paths}
    assert completed <= names
    if not completed:
        return {'videos': 0, 'expected': len(names), 'complete': False}

    paired = {}
    for path in paths:
        before = json.loads((REVISION / f'davis_matched_{model}' / path.name).read_text())
        joint = json.loads(path.read_text())
        assert (before['video'], before['frames'], before['queries']) == (joint['video'], joint['frames'], joint['queries'])
        for key in MATCHED_SETTINGS:
            assert before['protocol'].get(key) == joint['protocol'].get(key), (key, path)
        values = {}
        for label, mode in MODES.items():
            report = before if label == 'separate' else joint
            for advance in ('0.0', '1.25'):
                raw = report['results'][f'{mode}:head:{advance}']['metrics']
                values[f'{label}:{advance}'] = {short: raw[key] for short, key in METRICS.items()}
        paired[path.stem] = values

    per_video = {
        label: {metric: np.asarray([row[label][metric] for row in paired.values()]) for metric in METRICS}
        for label in next(iter(paired.values()))
    }
    pairs = [('joint_dense:1.25', 'joint_sparse:1.25'), ('joint_dense:0.0', 'joint_sparse:0.0'),
             ('separate:1.25', 'joint_dense:1.25'), ('separate:0.0', 'joint_dense:0.0')]
    pairs.extend((f'{mode}:1.25', f'{mode}:0.0') for mode in MODES)
    return dict(
        complete=completed == names, videos=len(paths), expected=len(names),
        scores={label: {m: float(vals.mean()) for m, vals in metrics.items()} for label, metrics in per_video.items()},
        differences={f'{a} minus {b}': {m: paired_interval(per_video[a][m] - per_video[b][m]) for m in METRICS}
                     for a, b in pairs},
        rows=paired)


def main():
    manifest = json.loads((ROOT / 'data/davis_records/manifest.json').read_text())
    names = {row['name'] for row in manifest}
    assert len(names) == len(manifest) == 30
    summaries = {name: summarize(name, names) for name in ('wan', 'cosmos')}
    output = REVISION / 'polish/audit/joint_context.json'
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(summaries, indent=2, allow_nan=False) + '\n')
    temporary.replace(output)
    for name, report in summaries.items():
        print(name, report['videos'], report['complete'])
        if 'scores' in report:
            print('AJ', {k: round(v['AJ'], 3) for k, v in report['scores'].items()})
            print('paired AJ', {k: [round(x, 3) for x in v['AJ']] for k, v in report['differences'].items()})


if __name__ == '__main__':
    main()
