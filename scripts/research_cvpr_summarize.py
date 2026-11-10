"""Validate paired CVPR experiment coverage and aggregate official metrics."""
import argparse
import json
from pathlib import Path

import numpy as np


METRICS = {'AJ': 'average_jaccard', 'PCK': 'average_pts_within_thresh', 'OA': 'occlusion_accuracy'}


def paired_interval(differences):
    differences = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(42)
    samples = rng.choice(differences, size=(10000, len(differences)), replace=True).mean(axis=1)
    return [float(differences.mean()), *np.quantile(samples, [.025, .975]).tolist()]


def load_matched(path, manifest):
    expected = {row['name']: row for row in json.loads(manifest.read_text())}
    rows = {p.stem: json.loads(p.read_text()) for p in sorted(path.glob('*.json'))}
    assert rows.keys() <= expected.keys()
    if not rows:
        return {'complete': False, 'expected': len(expected), 'n': 0}
    protocol = next(iter(rows.values()))['protocol']
    assert all(row['protocol'] == protocol for row in rows.values())
    variants = [':'.join(map(str, variant)) for variant in protocol['variants']]
    for name, row in rows.items():
        assert row['video'] == name and row['frames'] == expected[name]['frames']
        assert row['results'].keys() == set(variants)
        if 'eligible' in expected[name]:
            assert row['queries'] == expected[name]['eligible']
        assert all(np.isfinite([row['results'][v]['metrics'][m] for m in METRICS.values()]).all() for v in variants)
    arrays = {v: {m: np.array([row['results'][v]['metrics'][metric] for row in rows.values()])
                  for m, metric in METRICS.items()} for v in variants}
    comparisons = {}
    for lhs, rhs in [('native4:head:1.25', 'native4:head:0.0'), ('native4:head:1.25', 'native4:head:0.25'),
                     ('native4:head:1.25', 'native4:head:-1.25'), ('native4:head:1.25', 'framewise:head:0.0'),
                     ('native4:head:1.25', 'stride1:head:0.0'), ('native4:head:1.25', 'native2:head:1.25'),
                     ('native4:head:1.25', 'native1:head:1.25'), ('native4:head:1.25', 'native4:layer:1.25'),
                     ('native4:neutral:1.25', 'native4:head:1.25'), ('native4:low:1.25', 'native4:low:0.0'),
                     ('native4:low:1.25', 'native4:head:1.25'), ('native4:layer:1.25', 'native4:layer:0.0'),
                     ('native1:head:1.25', 'native1:head:0.0'),
                     ('native4:head:0.75', 'native4:head:0.0'), ('native4:head:0.75', 'native4:head:-0.25'),
                     ('native4:head:0.75', 'native4:head:-0.75'), ('native4:head:0.75', 'framewise:head:0.0'),
                     ('native1:head:0.75', 'native1:head:0.0')]:
        if lhs in arrays and rhs in arrays:
            comparisons[f'{lhs} minus {rhs}'] = {m: paired_interval(arrays[lhs][m] - arrays[rhs][m]) for m in METRICS}
    profiles = [profile for row in rows.values() for profile in row['profiles']]
    times = {}
    for mode, names in [('native1', ['phase0']), ('native2', ['phase0', 'phase2']),
                        ('native4', [f'phase{i}' for i in range(4)]), ('framewise', ['framewise']), ('stride1', ['stride1'])]:
        if all(f'{name}_s' in profiles[0] for name in names):
            times[mode] = dict(seconds_median=float(np.median([sum(p[f'{name}_s'] for name in names) for p in profiles])),
                               peak_GiB_median=float(np.median([max(p[f'{name}_peak_bytes'] for name in names) for p in profiles])) / 2**30)
    return dict(complete=rows.keys() == expected.keys(), expected=len(expected), n=len(rows),
                videos=list(rows), frames=sum(r['frames'] for r in rows.values()), queries=sum(r['queries'] for r in rows.values()),
                protocol=protocol, scores={v: {m: float(a.mean()) for m, a in metrics.items()} for v, metrics in arrays.items()},
                paired_ci95=comparisons, profile=times, chunks=len(profiles))


def load_cached(path, pattern, expected):
    rows = [json.loads(p.read_text()) for p in path.glob(pattern)]
    scores = {row['video']: {m: row['metrics'][key] for m, key in METRICS.items()} for row in rows}
    assert len(scores) == len(rows) <= expected
    return dict(n=len(scores), complete=len(scores) == expected, videos=scores,
                scores={m: float(np.mean([row[m] for row in scores.values()])) for m in METRICS} if scores else {})


def unit_coordinate_comparisons(revision, model):
    nominal = revision/f'davis_matched_{model}'
    unit = revision/f'davis_framewise_unit_{model}'
    names = sorted(p.stem for p in unit.glob('*.json'))
    if not names:
        return {}
    shifts, rows = {}, []
    for name in names:
        a = json.loads((nominal/f'{name}.json').read_text())
        b = json.loads((unit/f'{name}.json').read_text())
        assert (a['frames'], a['queries']) == (b['frames'], b['queries'])
        rows.append((a['results']['native4:head:1.25']['metrics'], b['results']))
    for name in rows[0][1]:
        shifts[name] = {key: paired_interval([a[metric] - b[name]['metrics'][metric] for a, b in rows])
                        for key, metric in METRICS.items()}
    return shifts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/share/yty/heft'))
    args = parser.parse_args()
    root = args.root
    revision = root / 'cvpr27'
    summary = {'matched': {}, 'cached': {}, 'trained_reference': {}, 'unit_coordinate_comparisons': {}}
    davis = root/'data/davis_records/manifest.json'
    for name, folder, manifest in [('davis_wan', 'davis_matched_wan', root/'data/davis_records/manifest.json'),
                                   ('davis_cosmos', 'davis_matched_cosmos', davis),
                                   ('davis_wan_unit', 'davis_framewise_unit_wan', davis),
                                   ('davis_cosmos_unit', 'davis_framewise_unit_cosmos', davis),
                                   ('davis_cog', 'davis_cog', davis),
                                   ('davis_cosmos_synthetic_head', 'davis_cosmos_synthetic_head', davis),
                                   ('robotap_wan', 'robotap_split0/wan', revision/'robotap_split0/data/manifest.json'),
                                   ('itto_mose_wan', 'itto_mose/wan', revision/'itto_mose/data/manifest.json')]:
        if manifest.exists():
            summary['matched'][name] = load_matched(revision/folder, manifest)
    for model in ('wan', 'cosmos'):
        summary['unit_coordinate_comparisons'][model] = unit_coordinate_comparisons(revision, model)
    for name, base, count in [('davis', root, 30), ('rgb', root/'rgb_stacking', 50), ('po', root/'pointodyssey', 13)]:
        for tag in ['0', '1.25']:
            path = base/'results'/f'cvpr27_wan_legacy_head_{tag}'
            for variant in ['post_full', 'post_low15']:
                summary['cached'][f'{name}/wan_18_1/{tag}/{variant}'] = load_cached(path, f'*_{variant}.json', count)
    for tag in ['0', '1.25']:
        path = root/'results'/f'cvpr27_cosmos_bands_{tag}'
        for variant in ['post_low15', 'norm_matched', 'random_0', 'random_1', 'random_2']:
            summary['cached'][f'davis/cosmos/{tag}/{variant}'] = load_cached(path, f'*_{variant}.json', 30)
    for folder, count in [('davis_single', 30), ('rgb_single', 50), ('po_single', 13),
                          ('robotap_joint', 53), ('robotap_single', 53)]:
        summary['trained_reference'][folder] = load_cached(revision/'cotracker3'/folder, '*.json', count)
    destination = revision/'summary'/'paired_experiments.json'
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False))
    temporary.replace(destination)
    print(json.dumps({name: {k: data[k] for k in ('n', 'complete', 'expected')} for name, data in summary['matched'].items()}))


if __name__ == '__main__':
    main()
